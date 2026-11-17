#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/arm_web_jog.py - web slider jog: browser <-HTTP-> this gateway <-TCP-> bridge.

Serves a page with one slider per joint and forwards the targets to the bridge (bridge.py on the robot computer, or
tools/fake_bridge.py in simulation) through tools/arm_jog.BridgeClient. Standard library only (http.server): the
page polls /state at 10 Hz, and that poll doubles as the heartbeat of the browser.

Safety chain (three layers; each one covers a failure of the layer before it):
  1. The gateway forwards targets to the bridge only while the page has polled /state within the last 0.5 s.
     Page closed, tab in the background or browser frozen -> the gateway stops sending.
  2. The bridge goes to damping and then disables the motors when it receives no target for 1.5 s.
  3. Targets are always clamped to soft_range of the config (range where there is none), and their rate of change
     is limited to vel_max of the --safety profile.
  STOP button (or the space key) = set the target to the measured position, then disconnect from the bridge, which
  goes to damping and disables the motors. Connect enables them again.

Usage:
  python tools/arm_web_jog.py --config humanoid_arm/humanoid_arm.yaml --gripper                            # bridge on this machine (port 9879)
  python tools/arm_web_jog.py --config humanoid_arm/humanoid_arm.yaml --gripper --bridge-host <robot-ip>   # bridge on the robot computer
  python tools/arm_web_jog.py --config Arm/Arm.yaml --gripper --bridge-host <robot-ip>                     # single Arm (port 9880)
  python tools/arm_web_jog.py --config Arm/Arm.yaml --gripper --listen 0.0.0.0:8780                        # page reachable from other machines
Then open http://127.0.0.1:8780/ in a browser (with --listen 0.0.0.0:8780: http://<address of this machine>:8780/).
The page has no access control: use --listen 0.0.0.0 on a trusted network only.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from arm_jog import BridgeClient   # noqa: E402

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_web_jog.html")
BROWSER_TIMEOUT = 0.5      # s without a poll from the browser before the gateway stops forwarding targets
SEND_HZ = 20.0


class Kin:
    """MuJoCo forward kinematics: link origins plus the end effector, for the skeleton drawn on the page (not drawn
    without mujoco)."""

    def __init__(self, cfg):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, cfg["model"]["scene"]))
        self.d = mujoco.MjData(self.m)
        self.adr = {self.m.joint(i).name: self.m.jnt_qposadr[i] for i in range(self.m.njnt)}
        self.bodies = [i for i in range(1, self.m.nbody)]
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, cfg["end_effector"]["site"])

    def fk(self, q):
        for n, a in self.adr.items():
            self.d.qpos[a] = q.get(n, 0.0)
        self.mj.mj_kinematics(self.m, self.d)
        pts = [self.d.xpos[b].round(4).tolist() for b in self.bodies]
        pts.append(self.d.site_xpos[self.site].round(4).tolist())
        return pts


class Gateway:
    def __init__(self, cfg, bridge_host, bridge_port, safety, with_gripper, use_fk=True):
        self.cfg = cfg
        self.bridge = (bridge_host, bridge_port)
        self.log, self.log_meta = None, None                 # --log: set by main()
        rows = [j for j in cfg["joints"] if j.get("enabled", True) and (with_gripper or j.get("kind") != "gripper")]
        soft = cfg.get("soft_range") or {}
        self.names = [j["name"] for j in rows]
        self.alias = {j["name"]: j.get("alias", j["name"]) for j in rows}
        self.limits = {j["name"]: tuple(float(v) for v in soft.get(j["name"], j["range"])) for j in rows}
        self.vel_max = float(cfg["safety_profiles"][safety]["vel_max"])
        self.safety = safety
        self.poses = {}
        for k in ("ready_pose", "home_pose"):
            pose = cfg.get(k) or {}
            self.poses[k.replace("_pose", "")] = {n: self._clamp(n, float(v)) for n, v in pose.items() if n in self.limits}
        self.kin = None
        if use_fk:
            try:
                self.kin = Kin(cfg)
            except Exception:
                self.kin = None
        self.lock = threading.Lock()
        self.cli = None
        self.goal, self.target = {}, {}
        self.last_browser = 0.0
        self.msg = "disconnected"
        self.stats = {"sent": 0, "dropped_silent": 0}
        threading.Thread(target=self._forward, daemon=True).start()

    def _clamp(self, n, v):
        lo, hi = self.limits[n]
        return min(max(v, lo), hi)

    # ---- browser actions ----
    def touch(self):
        self.last_browser = time.monotonic()

    def connect(self):
        with self.lock:
            if self.cli and not self.cli.closed:
                return "connected"
            try:
                cli = BridgeClient(self.bridge[0], self.bridge[1], self.names, self.limits, mode="send", log=self.log, log_meta=self.log_meta)
            except Exception as e:
                self.msg = f"could not connect to the bridge: {e}"
                return self.msg
            self.cli = cli
            self.target = {n: cli.q0[n] for n in cli.matched}
            self.goal = dict(self.target)
            self.msg = f"connected, {len(cli.matched)} joints enabled (unknown={cli.unknown})"
            return self.msg

    def stop(self):
        """STOP: set the target to the measured q, then disconnect; the bridge goes to damping and disables."""
        with self.lock:
            cli, self.cli = self.cli, None
            if cli and not cli.closed:
                try:
                    q = cli.latest()
                    cli.send_target({n: q.get(n, self.target.get(n, 0.0)) for n in self.target})
                except Exception:
                    pass
                cli.close()
            self.goal, self.target = {}, {}
            self.msg = "STOP: disconnected, the bridge goes to damping and disables the motors"
            return self.msg

    def set_goal(self, updates):
        with self.lock:
            if not self.cli:
                return "disconnected"
            for n, v in updates.items():
                if n in self.goal:
                    self.goal[n] = self._clamp(n, float(v))
            return "ok"

    def set_pose(self, key):
        pose = self.poses.get(key)
        if not pose:
            return f"unknown pose: {key}"
        return self.set_goal(pose)

    def nudge(self, n, delta):
        with self.lock:
            if not self.cli or n not in self.goal:
                return "disconnected"
            self.goal[n] = self._clamp(n, self.goal[n] + float(delta))
            return "ok"

    # ---- forwarding thread: 20 Hz, only while the browser is alive ----
    def _forward(self):
        dt = 1.0 / SEND_HZ
        t_prev = time.monotonic()
        while True:
            time.sleep(dt)
            now = time.monotonic()
            step_dt, t_prev = now - t_prev, now
            with self.lock:
                cli = self.cli
                if not cli:
                    continue
                if cli.closed:
                    self.cli = None
                    self.msg = "bridge disconnected (heartbeat timeout or bridge stopped) -> press Connect again"
                    continue
                if now - self.last_browser > BROWSER_TIMEOUT:
                    self.stats["dropped_silent"] += 1          # browser silent: send nothing, let the bridge time out into damping
                    continue
                lim = self.vel_max * step_dt
                for n in self.target:
                    d = self.goal[n] - self.target[n]
                    self.target[n] += min(max(d, -lim), lim)
                try:
                    cli.send_target(self.target)
                    self.stats["sent"] += 1
                except Exception as e:
                    self.msg = f"send failed: {e}"
                    self.cli = None

    def state(self):
        with self.lock:
            cli = self.cli
            q = cli.latest() if cli else {}
            connected = bool(cli and not cli.closed)
            joints = [{"name": n, "alias": self.alias[n], "lo": self.limits[n][0], "hi": self.limits[n][1],
                       "q": q.get(n), "target": self.target.get(n), "goal": self.goal.get(n)} for n in self.names]
            out = {"connected": connected, "msg": self.msg, "safety": self.safety, "vel_max": self.vel_max,
                   "title": self.cfg.get("title") or "Arm",
                   "bridge": f"{self.bridge[0]}:{self.bridge[1]}", "joints": joints, "stats": dict(self.stats),
                   "poses": list(self.poses)}
        if self.kin and q:
            out["skeleton"] = self.kin.fk(q)
        return out


def make_handler(gw):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = open(HTML, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/state"):
                gw.touch()
                self._json(gw.state())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}") if n else {}
            gw.touch()
            if self.path == "/connect":
                self._json({"msg": gw.connect()})
            elif self.path == "/stop":
                self._json({"msg": gw.stop()})
            elif self.path == "/target":
                self._json({"msg": gw.set_goal(data)})
            elif self.path == "/pose":
                self._json({"msg": gw.set_pose(data.get("name", ""))})
            elif self.path == "/nudge":
                self._json({"msg": gw.nudge(data.get("name", ""), data.get("delta", 0.0))})
            else:
                self._json({"error": "not found"}, 404)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, metavar="YAML",
                    help="humanoid_arm/humanoid_arm.yaml | Arm/Arm.yaml | Arm/Arm_pair.yaml")
    ap.add_argument("--bridge-host", default="127.0.0.1", help="host the bridge runs on")
    ap.add_argument("--log", nargs="?", const="", default=None, metavar="CSV",
                    help="record every target sent and the measured joint angles to a CSV file (no name: "
                         "logs/<tool>_<time>.csv); recording does not change what is sent. "
                         "tools/arm_log_report.py reads the file")
    ap.add_argument("--bridge-port", type=int, default=None, help="bridge port (default: bridge.port of the config)")
    ap.add_argument("--listen", default="127.0.0.1:8780",
                    help="address the page is served on (0.0.0.0:8780 makes it reachable from other machines)")
    ap.add_argument("--safety", default="slow", help="safety profile from the robot config: safe / slow / normal (joint speed limit)")
    ap.add_argument("--gripper", action="store_true", help="show the gripper(s) on the page as well")
    ap.add_argument("--no-fk", action="store_true", help="do not show the end-effector position (no MuJoCo needed)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    bport = args.bridge_port or int((cfg.get("bridge") or {}).get("port", 9877))
    gw = Gateway(cfg, args.bridge_host, bport, args.safety, args.gripper, use_fk=not args.no_fk)
    gw.log, gw.log_meta = args.log, {"config": args.config}
    host, port = args.listen.rsplit(":", 1)
    srv = ThreadingHTTPServer((host, int(port)), make_handler(gw))
    print(f"[web] http://{host}:{port}/  -> bridge {args.bridge_host}:{bport}  safety profile {args.safety} "
          f"(vel_max {gw.vel_max} rad/s)  joints {gw.names}  skeleton view {'on' if gw.kin else 'off'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        gw.stop()


if __name__ == "__main__":
    main()
