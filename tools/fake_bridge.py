#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/fake_bridge.py - the real bridge code driving simulated motors in MuJoCo: try everything without hardware.

This runs bridge.py itself - same server, same protocol, same port, same slew limit, same 1.5 s heartbeat ->
damping - with only the CAN layer replaced: the motor frames go to a MuJoCo simulation of the robot instead of a
real bus. A client that works here (arm_jog, arm_web_jog, arm_drag_teleop, arm_lead_follow, your own BridgeClient
program) works unchanged with the bridge on the robot computer.

What is simulated:
  - motion command frames: tau = kp (p_des - p) + kd (v_des - v) + t_ff, clipped to the torque limit and evaluated
    at 500 Hz, as the motor firmware does
  - feedback frames: q / qd / tau / temperature of the simulation, scaled per motor model as given in the config
    (a wrong motor model in the config gives wrong values here, exactly as on the real bus)
  - id query, enable, stop, parameter read / write (position, torque limit, speed limit, run mode) and save are
    answered
  - a joint that has no body in the MJCF is simulated as a simple virtual motor; it can be enabled, commanded and
    read like the others
  - a hand pushing the arm: with --view, double-click a link in the window and Ctrl + right-drag it; the force
    acts on the simulated arm (this is how to try tools/arm_lead_follow.py). --push BODY,FX,FY,FZ,T0,T1 applies a
    world-frame force (N) to BODY from T0 to T1 seconds after the first enable; it may be given several times
  - --lost-turn ID: that motor powers up with its raw angle shifted by +2π, as after a turn lost while powered off,
    to exercise the bridge's handling of it

Gravity feed-forward: pass --grav-scale 1.0. In simulation the arm is the model, so the gravity torque computed
from the model is exactly right. The scale stored in the config is tuned on the real arm (docs/arm_guide.md) and is
lower; with it the simulated arm sags.

Usage:
  python tools/fake_bridge.py --config humanoid_arm/humanoid_arm.yaml --grav-scale 1.0     # 16 motors on two buses, port 9879
  python tools/fake_bridge.py --config Arm/Arm.yaml --grav-scale 1.0                       # 7 motors on one bus, port 9880
  python tools/fake_bridge.py --config Arm/Arm_pair.yaml --grav-scale 1.0                  # leader + follower, port 9881
  mjpython tools/fake_bridge.py --config Arm/Arm.yaml --grav-scale 1.0 --view              # with a window (macOS: mjpython)
Then start any client with the same --config in another terminal, for example:
  python tools/arm_web_jog.py --config Arm/Arm.yaml --gripper

A config with `calibrated: false` is accepted here: the simulated motors are built from the sign / offset values of
the config, so the mapping is consistent. The real bridge refuses send mode for such a config.

This is a development stand-in, not a validation tool: CAN timing, frame loss and bus errors are not modelled.
"""
import argparse
import collections
import math
import os
import struct
import sys
import threading
import time
import types

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Stand-in for python-can: must be in sys.modules before robstride_mit / bridge are imported
# ---------------------------------------------------------------------------


class Message:
    def __init__(self, arbitration_id, is_extended_id=True, data=b"\x00" * 8, **_):
        self.arbitration_id, self.is_extended_id, self.data = arbitration_id, is_extended_id, bytes(data)


_SIM = None          # the single MotorSim, shared by every FakeCanBus


class FakeCanBus:
    """Stand-in for one `can.Bus` = one bus. All buses talk to the same MotorSim; only the motors on this bus answer."""

    def __init__(self, *a, channel=None, **k):
        self.channel = channel
        self.q = collections.deque()
        self.lock = threading.Lock()

    def send(self, msg):
        reply = _SIM.handle(msg, self.channel)
        if reply:
            with self.lock:
                self.q.extend(reply)

    def recv(self, timeout=0.0):
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            with self.lock:
                if self.q:
                    return self.q.popleft()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.0005)

    def shutdown(self):
        pass


fake_can = types.ModuleType("can")
fake_can.Bus = lambda *a, **k: FakeCanBus(*a, **k)
fake_can.Message = Message
sys.modules["can"] = fake_can

import robstride_mit as rm          # noqa: E402  (its constants and scaling are used, so the encoding matches the real driver)

# mc02:N channels do not go through can.Bus (the driver opens the MC02 port itself): its hook swaps in the same stand-in
rm.OPEN_BUS_HOOK = lambda ch: FakeCanBus(channel=ch) if str(ch).startswith("mc02:") else None

IDX_LIMIT_SPD = 0x7017
IDX_ZERO_STA = 0x7029


class Motor:
    """State of the "firmware" of one motor. Quantities are in motor coordinates (raw = sign*q + offset [+ k·2π])."""

    def __init__(self, mid, model, name, sign, offset, tau_limit, vel_limit, qposadr, dofadr, lost_turn=False, bus=None):
        self.mid, self.model, self.name, self.bus = mid, model, name, bus
        self.sign, self.offset = sign, offset
        self.lim = rm.MODEL_LIMITS[model]
        self.params = {rm.IDX_RUN_MODE: 0.0, rm.IDX_LIMIT_TORQUE: float(tau_limit),
                       IDX_LIMIT_SPD: float(vel_limit), IDX_ZERO_STA: 0.0}
        self.qposadr, self.dofadr = qposadr, dofadr          # None = virtual motor (no body in the MJCF)
        self.turn = 2.0 * math.pi if lost_turn else 0.0       # turn lost at power-up: raw is 2π larger
        self.enabled = False
        self.cradle = True            # until the first enable a stiff PD holds the power-up pose (a real arm rests on its structure)
        self.cmd = (0.0, 0.0, 0.0, 0.0, 0.0)                  # p v kp kd tau (motor coordinates)
        self.tau_out = 0.0
        self.temp = 32.0
        self.vq, self.vqd = 0.0, 0.0                          # state of a virtual motor (joint coordinates)

    def raw(self, q):
        return self.sign * q + self.offset + self.turn


class MotorSim:
    """The MuJoCo world plus all motors; a 500 Hz thread runs the "firmware PD" and integrates."""

    def __init__(self, cfg, lost_turn=(), realtime=True):
        import mujoco
        self.mj = mujoco
        self.cfg = cfg
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, cfg["model"]["scene"]))
        self.d = mujoco.MjData(self.m)
        kid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, kid)
        self.lock = threading.Lock()
        self.motors = {}
        for j in cfg["joints"]:
            if not j.get("enabled", True):
                continue
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
            qa, da = (self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]) if jid >= 0 else (None, None)
            mo = Motor(
                int(j["id"]), j["motor"], j["name"], float(j["sign"]), float(j["offset"]),
                cfg["tau_limit"][j["motor"]], cfg["vel_limit"][j["motor"]], qa, da,
                lost_turn=int(j["id"]) in lost_turn, bus=str(j["bus"]))
            mo.raw0 = mo.raw(float(self.d.qpos[qa]) if qa is not None else 0.0)
            self.motors[(str(j["bus"]), int(j["id"]))] = mo
        self.realtime = realtime
        self.dt = float(self.m.opt.timestep)               # 0.002 -> 500Hz
        self.t_sim = 0.0
        self.t_enable = None                               # simulation time of the first enable (time zero of --push)
        self.pushes = []                                   # [(body id, force (3,), t0, t1)]
        self.pert = None                                   # --view: the MjvPerturb of the window (mouse drag = a hand pushing)
        self.alive = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[fake] MuJoCo {mujoco.__version__} {os.path.basename(cfg['model']['scene'])}: "
              f"{len(self.motors)} motors " + " ".join(f"{m.mid}:{m.model}" for m in self.motors.values())
              + (f"  lost turn injected on {sorted(lost_turn)}" if lost_turn else ""))

    # ---- physics ----
    def _loop(self):
        t0 = time.monotonic()
        n = 0
        while self.alive:
            with self.lock:
                self.d.qfrc_applied[:] = 0.0
                self.d.xfrc_applied[:] = 0.0
                if self.t_enable is not None:
                    t = self.t_sim - self.t_enable
                    for bid, f, ta, tb in self.pushes:       # not named t0: the outer t0 is the origin of the real-time pacing
                        if ta <= t < tb:
                            self.d.xfrc_applied[bid, :3] += f
                if self.pert is not None:
                    self.mj.mjv_applyPerturbForce(self.m, self.d, self.pert)
                for mo in self.motors.values():
                    if mo.qposadr is None:                   # virtual motor
                        self._virtual_step(mo)
                        continue
                    q, qd = float(self.d.qpos[mo.qposadr]), float(self.d.qvel[mo.dofadr])
                    tau_m = self._firmware(mo, mo.raw(q), mo.sign * qd)
                    self.d.qfrc_applied[mo.dofadr] = mo.sign * tau_m
                self.mj.mj_step(self.m, self.d)
                self.t_sim += self.dt
            n += 1
            if self.realtime:
                lag = (t0 + n * self.dt) - time.monotonic()
                if lag > 0:
                    time.sleep(lag)

    def _firmware(self, mo, p, v):
        """MIT control law with the firmware torque limit and a toy temperature model. Returns the torque in motor
        coordinates."""
        if not mo.enabled:
            mo.tau_out = 0.0
            if mo.cradle:                                    # cradle: not a motor torque; it only holds the joint until the first enable
                return 200.0 * (mo.raw0 - p) - 5.0 * v
            return 0.0
        p_des, v_des, kp, kd, t_ff = mo.cmd
        tau = kp * (p_des - p) + kd * (v_des - v) + t_ff
        lim = min(mo.params[rm.IDX_LIMIT_TORQUE], mo.lim.t)
        tau = min(max(tau, -lim), lim)
        mo.tau_out = tau
        mo.temp += (0.02 * tau * tau - 0.5 * (mo.temp - 32.0)) * self.dt     # heating ∝ τ², cooling toward 32 °C
        return tau

    def _virtual_step(self, mo):
        """Motor without a body in the MJCF: inertia plus viscous friction, no gravity."""
        J, b = 0.002, 0.01
        tau = self._firmware(mo, mo.raw(mo.vq), mo.sign * mo.vqd)
        qdd = (mo.sign * tau - b * mo.vqd) / J
        mo.vqd += qdd * self.dt
        mo.vq += mo.vqd * self.dt

    def state(self, mo):
        """-> (raw, v, tau, temp) in motor coordinates."""
        if mo.qposadr is None:
            q, qd = mo.vq, mo.vqd
        else:
            q, qd = float(self.d.qpos[mo.qposadr]), float(self.d.qvel[mo.dofadr])
        return mo.raw(q), mo.sign * qd, mo.tau_out, mo.temp

    # ---- protocol (byte-for-byte counterpart of the encoding in robstride_mit) ----
    def _feedback(self, mo):
        with self.lock:
            raw, v, tau, temp = self.state(mo)
        lm = mo.lim
        p_u, v_u, t_u = rm.f2u(raw, -lm.p, lm.p), rm.f2u(v, -lm.v, lm.v), rm.f2u(tau, -lm.t, lm.t)
        tmp = int(temp * 10)
        mode = 2 if mo.enabled else 0
        ext = rm.TYPE_FEEDBACK << 24 | mode << 22 | mo.mid << 8 | rm.HOST_ID
        return Message(ext, True, bytes([p_u >> 8, p_u & 0xFF, v_u >> 8, v_u & 0xFF,
                                         t_u >> 8, t_u & 0xFF, tmp >> 8, tmp & 0xFF]))

    def handle(self, msg, channel=None):
        ext, d = msg.arbitration_id, msg.data
        typ, mid = (ext >> 24) & 0x3F, ext & 0xFF
        if channel is None:                                  # bus not given: find the motor by id
            mo = next((m for m in self.motors.values() if m.mid == mid), None)
        else:
            mo = self.motors.get((str(channel), mid))
        if mo is None:
            return []
        if typ == rm.TYPE_GET_ID:                        # ping: low byte 0xFE, id in bits 8-15, data = MCU UID
            return [Message(mid << 8 | 0xFE, True, struct.pack(">Q", 0xFA4E0000 + mid))]
        if typ == rm.TYPE_ENABLE:
            if self.t_enable is None:
                self.t_enable = self.t_sim
            mo.enabled = True
            mo.cradle = False
            with self.lock:
                raw, v, _, _ = self.state(mo)
            mo.cmd = (raw, 0.0, 0.0, 0.0, 0.0)           # at enable: zero gains, hold in place
            return [self._feedback(mo)]
        if typ == rm.TYPE_STOP:
            mo.enabled = False
            return [self._feedback(mo)]
        if typ == rm.TYPE_SET_PARAM:
            idx = d[0] | d[1] << 8
            if idx == rm.IDX_RUN_MODE:
                mo.params[idx] = float(d[4])
            else:
                mo.params[idx] = struct.unpack("<f", bytes(d[4:8]))[0]
                if idx == IDX_ZERO_STA and mo.params[idx] == 1.0:      # set zero: the present position becomes 0
                    with self.lock:
                        raw, *_ = self.state(mo)
                    mo.offset -= raw - mo.turn
            return [self._feedback(mo)]
        if typ == rm.TYPE_GET_PARAM:
            idx = d[0] | d[1] << 8
            if idx == rm.IDX_MECH_POS:
                with self.lock:
                    val = self.state(mo)[0]
            else:
                val = mo.params.get(idx, 0.0)
            return [Message(rm.TYPE_GET_PARAM << 24 | mid << 8 | rm.HOST_ID, True,
                            bytes([d[0], d[1], 0, 0]) + struct.pack("<f", val))]
        if typ == rm.TYPE_SAVE:
            return []
        if typ == rm.TYPE_MOTION:
            lm = mo.lim
            p = rm.u2f(d[0] << 8 | d[1], -lm.p, lm.p)
            v = rm.u2f(d[2] << 8 | d[3], -lm.v, lm.v)
            kp = rm.u2f(d[4] << 8 | d[5], 0, lm.kp)
            kd = rm.u2f(d[6] << 8 | d[7], 0, lm.kd)
            tau = rm.u2f((ext >> 8) & 0xFFFF, -lm.t, lm.t)
            mo.cmd = (p, v, kp, kd, tau)
            return [self._feedback(mo)]
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, metavar="YAML",
                    help="humanoid_arm/humanoid_arm.yaml | Arm/Arm.yaml | Arm/Arm_pair.yaml")
    ap.add_argument("--port", type=int, default=None, help="listening port (default: bridge.port of the config)")
    ap.add_argument("--profile", default=None, help="gain profile (default: mirror)")
    ap.add_argument("--lost-turn", default="",
                    help="comma-separated ids of motors that power up with raw +2π (reproduces a lost turn)")
    ap.add_argument("--view", action="store_true", help="open a MuJoCo window (needs mjpython on macOS)")
    ap.add_argument("--no-realtime", action="store_true",
                    help="run the physics as fast as possible instead of in real time (for automated tests)")
    ap.add_argument("--grav-scale", type=float, default=None,
                    help="override gravity_ff.scale of the config; use 1.0 in simulation")
    ap.add_argument("--push", action="append", default=[], metavar="BODY,FX,FY,FZ,T0,T1",
                    help="apply a world-frame force (N) to BODY from T0 to T1 seconds after the first enable, to simulate "
                         "a push; may be given several times")
    args = ap.parse_args()

    global _SIM
    cfg = yaml.safe_load(open(args.config))
    if args.grav_scale is not None and cfg.get("gravity_ff"):
        cfg["gravity_ff"]["scale"] = args.grav_scale
    lost = {int(x) for x in args.lost_turn.split(",") if x.strip()}
    _SIM = MotorSim(cfg, lost_turn=lost, realtime=not args.no_realtime)
    for spec in args.push:
        body, fx, fy, fz, t0, t1 = spec.split(",")
        bid = _SIM.mj.mj_name2id(_SIM.m, _SIM.mj.mjtObj.mjOBJ_BODY, body.strip())
        if bid < 0:
            sys.exit(f"--push: the model has no body {body!r}")
        _SIM.pushes.append((bid, np.array([float(fx), float(fy), float(fz)]), float(t0), float(t1)))

    import bridge as lb                              # imported only now: the can module it gets is the stand-in
    lb.load_config(args.config)
    if not lb.CALIBRATED:
        # The simulated motors are built from the sign / offset of this config, so the mapping is consistent even with
        # placeholder values. The real bridge refuses send mode.
        print(f"[fake] {os.path.basename(args.config)} says calibrated: false - send mode is allowed in the fake bridge "
              "(the real bridge refuses it)")
        lb.CALIBRATED = True
    profile = args.profile or ("mirror" if "mirror" in lb.GAIN_PROFILES else next(iter(lb.GAIN_PROFILES)))
    lb.MIRROR_KP = {m: g[0] for m, g in lb.GAIN_PROFILES[profile].items()}
    lb.MIRROR_KD = {m: g[1] for m, g in lb.GAIN_PROFILES[profile].items()}
    lb.JOINT_KP = {j: g[0] for j, g in lb.GAIN_OVERRIDES[profile].items()}    # per-joint overrides of the profile
    lb.JOINT_KD = {j: g[1] for j, g in lb.GAIN_OVERRIDES[profile].items()}
    port = args.port or int((cfg.get("bridge") or {}).get("port", 9877))
    grav = None
    if cfg.get("gravity_ff"):
        grav = lb.GravityFF(cfg, lb.JOINT_MAP)
    print(f"[fake] running the real bridge code: config {os.path.basename(args.config)}, gain profile {profile} "
          f"kp={lb.MIRROR_KP} kd={lb.MIRROR_KD}")
    buses = {}
    for e in lb.JOINT_MAP:
        buses.setdefault(e[2], []).append(e)
    for ch, entries in buses.items():
        lb.BusWorker(ch, entries, grav=grav).start()

    if args.view:
        import mujoco.viewer
        srv = threading.Thread(target=lb.serve, args=(port,), daemon=True)
        srv.start()
        with mujoco.viewer.launch_passive(_SIM.m, _SIM.d) as v:
            _SIM.pert = v.perturb                          # Ctrl + drag in the window = a hand pushing the simulated arm
            while v.is_running():
                with _SIM.lock:
                    v.sync()
                time.sleep(1 / 30)
        lb.safe_idle("viewer closed")
        return
    try:
        lb.serve(port)
    except KeyboardInterrupt:
        lb.safe_idle("Ctrl+C")
        time.sleep(0.8)


if __name__ == "__main__":
    main()
