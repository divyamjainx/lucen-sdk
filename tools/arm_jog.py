#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/arm_jog.py - jog arm joints from the keyboard; also home of the reusable BridgeClient class.

Joint-space jog client for the bridge: bridge.py on the robot computer, or tools/fake_bridge.py in simulation
(same port, same protocol). There is no Cartesian / inverse-kinematics mode. The keyboard is read from the
terminal (termios cbreak mode), so the tool also works over SSH. Ctrl+C always works as the emergency stop:
the bridge sees the connection drop, switches the motors to damping and disables them.

Keys:
  1..9 / [ ]   select a joint (joints in config order; grippers only with --gripper)
  w / s        selected joint +/- one step (--step, default 0.02 rad)        W / S   +/- 5 steps
  r            go to ready_pose of the config                                h       go to home_pose
  space        stop: the target becomes the measured position
  x            quit (the target is set to the measured position first, then the connection is closed)
Go home (h) before quitting: once the client is gone the bridge disables the motors, and an arm that is not in a
rest pose sinks.

Safety, in this client, on top of the range clamp and the 1 rad/s slew limit of the bridge:
  - the target is always clamped to the joint range of the config (soft_range where given), and each joint's
    target moves no faster than vel_max of the --safety profile
  - a target is sent every 0.1 s and doubles as the heartbeat: if this process hangs or is killed, the bridge
    goes to damping after 1.5 s
  - after connecting, the target starts at the measured pose (q0), never at zero

Usage:
  python tools/arm_jog.py --config Arm/Arm.yaml --gripper                           # bridge or fake bridge on this machine
  python tools/arm_jog.py --config Arm/Arm.yaml --gripper --host <robot-ip>         # bridge on the robot computer
  python tools/arm_jog.py --config humanoid_arm/humanoid_arm.yaml --gripper --host <robot-ip> --safety safe

BridgeClient is a minimal client of the bridge protocol (hello / target / q). The other arm tools import it, and
it can be used directly in your own programs (see docs/connect.md).
"""
import argparse
import json
import math
import os
import select
import shutil
import socket
import sys
import threading
import time
import unicodedata

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BridgeClient:
    """Minimal client of the bridge protocol (newline-delimited JSON over TCP).

    joints: names of the joints to use. limits: {joint: (lo, hi)} in rad; the bridge clamps every target to them.
    mode "send" enables the motors and accepts targets; mode "watch" only reports joint angles (motors not enabled).
    After construction: matched / unknown (requested joints the bridge knows / does not know) and, in send mode,
    q0, the measured pose right after enabling - start your target there.
    Thread-safe: a reader thread keeps the latest measured angles (latest()); the caller sends targets with
    send_target(). Keep sending: 1.5 s without a message and the bridge goes to damping and disables the motors.
    """

    def __init__(self, host, port, joints, limits, mode="send", timeout=6.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(None)
        self.lock = threading.Lock()
        self.q = {}
        self.q0 = None
        self.matched, self.unknown = [], []
        self.closed = False
        self._buf = b""
        self._send({"hello": {"mode": mode, "joints": list(joints),
                              "limits": {n: list(v) for n, v in limits.items()}}})
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout and (not self.matched or (mode == "send" and self.q0 is None)):
            for msg in self._read(0.2):
                if "error" in msg:
                    self.close()
                    raise RuntimeError(f"bridge refused the request: {msg['error']}")
                if "matched" in msg:
                    self.matched, self.unknown = msg["matched"], msg.get("unknown", [])
                elif "q0" in msg:
                    self.q0 = dict(msg["q0"])
                    with self.lock:
                        self.q.update(self.q0)
                elif "q" in msg:
                    with self.lock:
                        self.q.update(msg["q"])
        if not self.matched:
            raise RuntimeError(f"the bridge matched none of the requested joints (unknown={self.unknown})")
        if mode == "send" and self.q0 is None:
            raise RuntimeError("no q0 received within 6 s in send mode (did enabling fail? check the bridge output)")
        missing = [n for n in self.matched if n not in (self.q0 or {})] if mode == "send" else []
        if missing:
            raise RuntimeError(f"q0 is missing joints {missing} (no feedback from these motors after enabling)")
        threading.Thread(target=self._reader, daemon=True).start()

    def _send(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    def _read(self, timeout):
        r, _, _ = select.select([self.sock], [], [], timeout)
        if not r:
            return []
        chunk = self.sock.recv(65536)
        if not chunk:
            self.closed = True
            return []
        self._buf += chunk
        out = []
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            if line.strip():
                out.append(json.loads(line))
        return out

    def _reader(self):
        try:
            while not self.closed:
                for msg in self._read(0.5):
                    if "q" in msg:
                        with self.lock:
                            self.q.update(msg["q"])
        except (OSError, ValueError):
            self.closed = True

    def send_target(self, target):
        """target: {joint: rad}. An empty dict is a pure keep-alive: it resets the heartbeat but does not count as the
        first target, so the bridge stays in its start-up grace period."""
        if self.closed:
            raise ConnectionError("bridge disconnected")
        try:
            self._send({"target": {n: float(v) for n, v in target.items()}})
        except OSError:
            self.closed = True
            raise ConnectionError("bridge disconnected")

    def latest(self):
        with self.lock:
            return dict(self.q)

    def close(self):
        self.closed = True
        try:
            self.sock.close()
        except OSError:
            pass


class Fk:
    """Optional: end-effector position from the MuJoCo scene, shown in the status line only (omitted without mujoco)."""

    def __init__(self, cfg):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, cfg["model"]["scene"]))
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, cfg["end_effector"]["site"])
        self.adr = {self.m.joint(i).name: self.m.jnt_qposadr[i] for i in range(self.m.njnt)}

    def ee(self, q):
        for n, a in self.adr.items():
            self.d.qpos[a] = q.get(n, 0.0)
        self.mj.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy()


class Keys:
    """Keyboard in termios cbreak mode. Signals stay enabled, so Ctrl+C still works as the emergency stop."""

    def __enter__(self):
        import termios
        import tty
        if not sys.stdin.isatty():
            raise SystemExit("a real terminal (tty) is required: do not run under nohup or with stdin piped")
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self):
        # Read the fd with os.read, not sys.stdin.read(1): the text wrapper buffers bytes that arrive together,
        # select() then no longer sees them, and repeated keys would lag one poll behind.
        out = []
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 64)
            if not chunk:
                break
            out.extend(chunk.decode(errors="ignore"))
        return out


def fit_width(text, cols):
    """Trim text to at most cols display columns (wide / full-width characters count as two)."""
    w = 0
    for i, ch in enumerate(text):
        w += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if w > cols:
            return text[:i]
    return text


def status_line(text):
    """Status line refreshed in place. A line wider than the terminal wraps, and \\r then only returns to the start of
    the last wrapped row, so every refresh would scroll the screen by one line. The text is therefore trimmed to the
    terminal width, and "erase to end of line" clears what is left of a longer previous line."""
    if sys.stdout.isatty():
        text = fit_width(text, shutil.get_terminal_size((80, 24)).columns - 1) + "\x1b[K"
    sys.stdout.write("\r" + text)
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, metavar="YAML",
                    help="humanoid_arm/humanoid_arm.yaml | Arm/Arm.yaml | Arm/Arm_pair.yaml")
    ap.add_argument("--host", default="127.0.0.1", help="host the bridge runs on")
    ap.add_argument("--port", type=int, default=None, help="bridge port (default: bridge.port of the config)")
    ap.add_argument("--step", type=float, default=0.02, help="target increment per key press (rad)")
    ap.add_argument("--safety", default="slow", help="safety profile (safety_profiles in the config); its vel_max limits how fast the target moves")
    ap.add_argument("--gripper", action="store_true", help="include the gripper(s) in the joint list")
    ap.add_argument("--no-fk", action="store_true", help="do not show the end-effector position (no MuJoCo needed)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    port = args.port or int((cfg.get("bridge") or {}).get("port", 9877))
    rows = [j for j in cfg["joints"] if j.get("enabled", True) and (args.gripper or j.get("kind") != "gripper")]
    names = [j["name"] for j in rows]
    # clamp range: soft_range of the config where given, otherwise the joint range
    soft = cfg.get("soft_range") or {}
    limits = {j["name"]: tuple(float(v) for v in soft.get(j["name"], j["range"])) for j in rows}
    prof = cfg["safety_profiles"][args.safety]
    vel_max = float(prof["vel_max"])
    poses = {"r": cfg.get("ready_pose") or {}, "h": cfg.get("home_pose") or {}}
    for k, pose in poses.items():                              # preset poses are clamped to the same limits
        poses[k] = {n: min(max(float(v), limits[n][0]), limits[n][1]) for n, v in pose.items() if n in limits}
    fk = None
    if not args.no_fk:
        try:
            fk = Fk(cfg)
        except Exception:
            fk = None

    print(f"[jog] connecting to {args.host}:{port} ({os.path.basename(args.config)}), safety profile {args.safety}: "
          f"vel_max {vel_max} rad/s, step {args.step} rad; joints {names}")
    cli = BridgeClient(args.host, port, names, limits, mode="send")
    if cli.unknown:
        print(f"[jog] !! joints unknown to the bridge: {cli.unknown}")
    target = {n: cli.q0[n] for n in cli.matched}        # the target starts at the measured pose, not at zero
    goal = dict(target)                                  # what the keys ask for; target follows goal at vel_max
    sel = 0
    active = [n for n in names if n in cli.matched]
    print("[jog] q0 = " + " ".join(f"{n.split('_')[0]}={cli.q0[n]:+.3f}" for n in active))
    print("[jog] keys: 1-9 / [ ] select joint  w/s ±step  W/S ±5 steps  r ready  h home  space stop  x quit  (Ctrl+C = emergency stop)")

    last_send, last_print, t_prev = 0.0, 0.0, time.monotonic()
    try:
        with Keys() as keys:
            while True:
                now = time.monotonic()
                dt, t_prev = now - t_prev, now
                for k in keys.poll():
                    if k in "123456789":
                        i = int(k) - 1
                        if i < len(active):
                            sel = i
                    elif k == "[":
                        sel = (sel - 1) % len(active)
                    elif k == "]":
                        sel = (sel + 1) % len(active)
                    elif k in "wsWS":
                        n = active[sel]
                        d = args.step * (5 if k.isupper() else 1) * (1 if k.lower() == "w" else -1)
                        goal[n] = min(max(goal[n] + d, limits[n][0]), limits[n][1])
                    elif k in poses and poses[k]:
                        for n in active:
                            if n in poses[k]:
                                goal[n] = min(max(float(poses[k][n]), limits[n][0]), limits[n][1])
                    elif k == " ":
                        q = cli.latest()
                        for n in active:
                            goal[n] = target[n] = q.get(n, target[n])
                    elif k == "x":
                        raise KeyboardInterrupt
                # client-side rate limit: target follows goal at vel_max (the bridge adds its own 1 rad/s slew limit and range clamp)
                lim = vel_max * dt
                for n in active:
                    step = goal[n] - target[n]
                    target[n] += min(max(step, -lim), lim)
                if now - last_send >= 0.1:
                    cli.send_target(target)
                    last_send = now
                if now - last_print >= 0.2:
                    q = cli.latest()
                    cur = active[sel]
                    ee = f"  ee={fk.ee(q).round(3).tolist()}" if fk else ""
                    line = " ".join(f"{'>' if n == cur else ' '}{n.split('_')[0]}:{q.get(n, float('nan')):+.3f}"
                                    f"→{target[n]:+.3f}" for n in active)
                    sys.stdout.write("\r" + line + ee + "   ")
                    sys.stdout.flush()
                    last_print = now
                if cli.closed:
                    print("\n[jog] bridge disconnected")
                    break
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            q = cli.latest()
            cli.send_target({n: q.get(n, target[n]) for n in active})     # hold the measured pose before leaving
        except Exception:
            pass
        cli.close()
        print("\n[jog] exit (the bridge goes to damping, then disables, after its 1.5 s heartbeat timeout)")


if __name__ == "__main__":
    main()
