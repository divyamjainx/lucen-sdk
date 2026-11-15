#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bridge.py - bridge process on the robot computer: owns the motor buses, serves joint targets and angles over TCP

The bridge opens the MC02 board, runs one 50 Hz real-time thread per CAN bus that keeps the RobStride motors
commanded, and exposes a newline-delimited JSON protocol on a TCP port. A client (mirror_viewer.py, the arm
tools under tools/, or your own program) only sends joint targets and reads joint angles; the safety logic
stays in the bridge. Use it to drive the arms and to watch / calibrate the biped. tools/deploy_policy.py
does not need a running bridge: it reuses the same bus workers in its own process to run a policy.

Usage (on the robot computer, from the SDK root):
  python bridge.py                                            # biped_Laika/biped_Laika.yaml, port 9877
  python bridge.py --config humanoid_arm/humanoid_arm.yaml    # dual 7-DoF arms,              port 9879
  python bridge.py --config Arm/Arm.yaml                      # 6+1-DoF arm,                  port 9880
  python bridge.py --config Arm/Arm_pair.yaml                 # leader + follower arms,       port 9881
The bridge has no built-in notion of "leg" or "arm": joints, buses, motor models, calibration (sign / offset),
gains and the port (bridge.port, default 9877) all come from the --config yaml. If the yaml has a gravity_ff
section (the arms), gravity feed-forward is added to the motor commands. Restart the bridge after editing
the yaml.

Protocol: TCP, one JSON object per line, one client at a time.
  client -> bridge: {"hello": {"mode": "watch"|"send", "joints": [...], "limits": {...}}}
                    {"target": {joint_name: rad, ...}}     # send mode: targets, and the heartbeat
  bridge -> client: {"matched": [...], "unknown": [...]}   # reply to hello
                    {"q0": {...}}                          # send mode: joint angles right after enabling
                    {"q": {...}}                           # live joint angles (URDF coordinates)
                    {"matched": [], "unknown": [], "error": "..."}   # hello refused
  hello   mode "watch" (the default): the motors stay disabled and the bridge only reads the joint angles,
          so the joints can be moved by hand. mode "send": the motors are enabled with the gains of the
          bridge's gain profile (--profile) and follow the targets.
          joints: names of the joints to use (omit for all joints of the config). The reply lists the names
          found in the config (matched) and the others (unknown).
          limits: {joint_name: [lo, hi]} in rad; targets are clamped to them.
  q0      sent once in send mode, within 4 s after hello: the measured joint angles right after enabling.
          A joint that could not be enabled may be missing from it.
  q       measured joint angles, streamed in both modes about every 33 ms; same coordinates as the targets.
  target  send it periodically. A joint without a target holds its position; {"target": {}} is a pure
          keep-alive.

Safety behaviour:
  - Send mode needs a line from the client at least every 1.5 s (3 s until the first non-empty target). When
    the client goes silent or disconnects, all motors go to damping and are then disabled.
  - Targets are clamped to the limits sent in hello and rate-limited to 1 rad/s (slew.py). Non-finite values
    are dropped.
  - A config with `calibrated: false` is watch-only: a hello with mode "send" is answered with "error" and
    nothing is enabled. Calibrate first (tools/calib_arm.py).
  - MC02 watchdog: right after enabling, the bridge registers a damping frame for every motor on the board and
    arms the board's link watchdog (mc02.watchdog_ms in the yaml). If the bridge stalls or crashes, the board
    keeps the motors in damping by itself. After a watchdog event the bridge re-homes: it restarts from the
    measured joint angles and approaches the targets at 1 rad/s again. The board is powered over USB, so a
    pulled USB cable is not covered by this watchdog; for that case set the motor-side CAN timeout
    (tools/set_can_timeout.py).
  - A warning is printed when a motor reports more than 75 °C or stops sending feedback for more than 0.5 s.
See docs/safety.md and docs/connect.md.
"""

import argparse
import json
import math
import os
import select
import socket
import threading
import time
import traceback

import yaml

import robstride_mit as rm


def wrap_pi(a):
    """Wrap to (-pi, pi] (same formula as rm.wrap_pi): a motor loses its turn count at power-off, so the raw
    angle after power-on may be off by a multiple of 2*pi."""
    return -((-a + math.pi) % (2.0 * math.pi) - math.pi)


# The robot config yaml is the single source of joints, buses, motor models, calibration and gains; nothing is
# hard-coded here. Default: biped_Laika/biped_Laika.yaml; the other robots use the same schema (--config).
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CFG_PATH = os.path.join(_ROOT, "biped_Laika/biped_Laika.yaml")


def load_config(path=None):
    """(Re)load a robot config into the module globals: JOINT_MAP / MODEL_OF / GAIN_PROFILES / MIRROR_KP / MIRROR_KD.
    Called once at import with biped_Laika/biped_Laika.yaml, and again by main() when --config is given.
    Returns the config dict."""
    global _CFG_PATH, _CFG, JOINT_MAP, MODEL_OF, GAIN_PROFILES, GAIN_OVERRIDES, \
        MIRROR_KP, MIRROR_KD, JOINT_KP, JOINT_KD, WATCHDOG_MS, CALIBRATED
    _CFG_PATH = os.path.abspath(path) if path else _DEFAULT_CFG_PATH
    with open(_CFG_PATH) as _f:
        _CFG = yaml.safe_load(_f)
    # MC02 board (docs/mc02.md): serial port and link-watchdog timeout come from the `mc02` section of the yaml.
    # The default of 100 ms is 5 control periods of 20 ms: worker jitter does not trigger it, and a real stall
    # costs at most 5 control steps.
    _mc02 = _CFG.get("mc02") or {}
    rm.MC02_PORT = _mc02.get("port")
    WATCHDOG_MS = int(_mc02.get("watchdog_ms", 100))
    # `calibrated: false` in the yaml means that sign / offset are still placeholders: only watch is allowed
    # and send is refused (see handle_line). A config without this key counts as calibrated.
    CALIBRATED = bool(_CFG.get("calibrated", True))
    joints = [j for j in _CFG["joints"] if j.get("enabled", True)]
    # (urdf_joint_name, motor_id, bus, sign, offset_rad): tools/deploy_policy.py relies on this 5-tuple layout,
    # do not add fields
    JOINT_MAP = [(j["name"], j["id"], j["bus"], j["sign"], j["offset"]) for j in joints]
    # (bus, id) -> motor model: the same id may be a different model on another bus
    MODEL_OF = {}
    for j in joints:
        key = (j["bus"], j["id"])
        assert j["motor"] in rm.MODEL_LIMITS, \
            f"{_CFG_PATH}: {j['name']} motor={j['motor']} is not in robstride_mit.MODEL_LIMITS"
        assert key not in MODEL_OF, f"{_CFG_PATH}: bus={j['bus']} id={j['id']} is listed twice"
        MODEL_OF[key] = j["motor"]
        rng = j.get("range")
        if rng and max(abs(rng[0]), abs(rng[1])) > math.pi - 0.2:
            print(f"[cfg] !! {j['name']} range {rng} is close to ±pi, where wrap_pi wraps around: power on with"
                  f" the joint near its home position, otherwise the first target may wrap to the other side")
    if _CFG_PATH == _DEFAULT_CFG_PATH:
        # Biped only: the yaml must agree with the driver's built-in model table rm.MOTOR_MODEL, which the
        # command-line tools use to encode / decode biped frames when no --config is given.
        for _j in joints:
            assert rm.MOTOR_MODEL[_j["id"]] == _j["motor"], \
                f"biped_Laika/biped_Laika.yaml id={_j['id']} motor={_j['motor']} does not match robstride_mit.MOTOR_MODEL"
    # joint_overrides: (kp, kd) overrides per joint name within a gain profile; all other joints use the
    # per-model values.
    GAIN_PROFILES = {p: {m: tuple(v) for m, v in d.items() if m != "joint_overrides"}
                     for p, d in _CFG["gain_profiles"].items()}
    GAIN_OVERRIDES = {p: {j: tuple(v) for j, v in (d.get("joint_overrides") or {}).items()}
                      for p, d in _CFG["gain_profiles"].items()}
    _base = "mirror" if "mirror" in GAIN_PROFILES else next(iter(GAIN_PROFILES))
    MIRROR_KP = {m: g[0] for m, g in GAIN_PROFILES[_base].items()}
    MIRROR_KD = {m: g[1] for m, g in GAIN_PROFILES[_base].items()}
    JOINT_KP = {j: g[0] for j, g in GAIN_OVERRIDES[_base].items()}
    JOINT_KD = {j: g[1] for j, g in GAIN_OVERRIDES[_base].items()}
    return _CFG


load_config()
# The rate limiter lives in slew.py in the SDK root and is shared with tools/sim2sim.py, so that the simulated
# command path is exactly the one of the real robot (see the header of slew.py).
from slew import FAST_SPD, HOME_TOL, SOFT_SPD, slew_step   # noqa: E402,F401
RATE = 50.0
ENABLE_TRIES = 4      # enable attempts per motor; one missed reply must not cost the joint for the whole session
ENABLE_WAIT = 0.05    # s to wait for feedback after each enable (the driver's 4 ms default suits the batch exchange)
# The two timeouts below apply to the network (serve) path only; tools/deploy_policy.py runs the workers
# in-process with its own worker / IMU / tilt protections and no client heartbeat. On the network path
# fast_slew is always False, so the 1 rad/s limit holds; CLIENT_TIMEOUT is sized for that limit and must be
# reconsidered if FAST_SPD is ever allowed here.
CLIENT_TIMEOUT = 1.5  # s without a line from the client in send mode before the bridge goes to damping and
                      # disables; long enough to ride out network jitter and stalls of the viewer UI
BOOT_GRACE = 3.0      # s allowed before the first non-empty target (enabling, opening the viewer window);
                      # meanwhile the motors hold the current pose with the soft gains


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "idle"          # idle | watch | send
        self.active = set()         # joint names used in this session
        self.limits = {}            # name -> (lo, hi)
        self.target = {}            # name -> rad (URDF coordinates, latest target from the client)
        self.q = {}                 # name -> rad (URDF coordinates, measured)
        self.qd = {}                # name -> rad/s (URDF coordinates, measured; a policy observation)
        self.q_stamp = {}           # name -> time.monotonic() at which this joint's feedback frame arrived
                                    # (= Feedback.stamp, not the time it was copied into STATE).
                                    # The motors of a bus are commanded back to back and reply in turn,
                                    # so the joint angles of one control step are not sampled at the
                                    # same instant; this field gives the age of each of them.
        self.tau = {}               # name -> Nm (torque reported by the motor)
        self.temp = {}              # name -> °C (temperature reported by the motor)
        self.ready = set()          # send mode: joints that are enabled and have an initial angle
        self.fast_slew = False      # True: once a joint is homed, rate-limit it at FAST_SPD (policy deployment
                                    # only). Must stay False for teleoperation and the viewer: a slider
                                    # jump would otherwise jerk the joint


STATE = State()


class GravityFF:
    """Gravity feed-forward for the arms (built only when the yaml has a gravity_ff section; the biped has none).

    On the biped the torque field of the MIT frame is always 0 (standing and walking rely on PD alone). A
    fixed-base arm is different: with low kp it sags under gravity, and high kp is not safe, so it is driven
    with PD + gravity feed-forward. MuJoCo evaluates qfrc_bias at the measured joint angles with qvel = 0,
    which is the pure gravity torque (MuJoCo's convention is M*qacc + qfrc_bias = applied force, so holding
    still needs applied = qfrc_bias). The result is multiplied by `scale` and clamped to ±clip_frac * tau_limit.
    Sign: URDF coordinates -> motor coordinates by multiplying with sign, the same convention as
    STATE.tau = sign * fb.torque.
    scale defaults to 0 (off). When bringing up an arm, raise it in steps (0.3 -> 0.6 -> 1.0, --grav-scale)
    and check the direction at each step."""

    def __init__(self, cfg, joint_map):
        g = cfg["gravity_ff"]
        self.scale = float(g.get("scale", 0.0))
        self.frac = float(g.get("clip_frac", 0.6))
        import mujoco                                   # needed only with gravity_ff; if missing, main() runs without feed-forward
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(os.path.join(_ROOT, g["model"]))
        self.d = mujoco.MjData(self.m)
        # One GravityFF is shared by the worker threads of all buses (two arms = two buses). Concurrent
        # mj_forward calls on the same MjData crash the process, so the whole computation is serialized.
        self.lock = threading.Lock()
        motor_of = {j["name"]: j["motor"] for j in cfg["joints"]}
        self.sign = {e[0]: e[3] for e in joint_map}
        self.adr, self.clip = {}, {}
        for name in self.sign:
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue                                # joints that are not in the MJCF (e.g. a gripper): no feed-forward
            self.adr[name] = (self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid])
            self.clip[name] = self.frac * float(cfg["tau_limit"][motor_of[name]])
        print(f"[grav] gravity feed-forward: {g['model']} scale={self.scale} clip_frac={self.frac} "
              f"joints {sorted(self.adr)}")

    def tau_motor(self, q):
        """q: {name: rad (URDF coordinates, measured)} -> {name: tau_ff (motor coordinates, Nm)}"""
        if self.scale == 0.0:
            return {}
        with self.lock:
            for name, (qa, _) in self.adr.items():
                self.d.qpos[qa] = q.get(name, 0.0)
            self.d.qvel[:] = 0.0
            self.mj.mj_forward(self.m, self.d)
            out = {}
            for name, (_, da) in self.adr.items():
                t = self.scale * float(self.d.qfrc_bias[da])
                t = min(max(t, -self.clip[name]), self.clip[name])
                out[name] = self.sign[name] * t
            return out


class BusWorker(threading.Thread):
    def __init__(self, channel, entries, models=None, grav=None):
        super().__init__(daemon=True)
        self.channel = channel
        self.entries = entries      # JOINT_MAP rows of this bus
        self.ids = [e[1] for e in entries]
        # id -> motor model: from the (bus, id) table built by load_config, falling back to the driver's
        # built-in biped table
        self.models = models or {e[1]: MODEL_OF.get((e[2], e[1]), rm.MOTOR_MODEL.get(e[1]))
                                 for e in entries}
        self.grav = grav            # GravityFF or None (always None for the biped)
        self.alive = True           # False once run() has exited; callers use it to detect a dead bus

    def to_motor(self, name, q, raw_now):
        # Command within the motor's current turn: the target is wrapped into the turn of raw_now, so
        # |target - raw_now| < pi. After the turn count was lost at power-off, the joint is therefore never
        # driven through a full turn to reach an absolute angle.
        _, _, _, sign, off = self._row[name]
        return raw_now + wrap_pi(sign * q + off - raw_now)

    def to_sim(self, name, a):
        _, _, _, sign, off = self._row[name]
        return sign * wrap_pi(a - off)

    def run(self):
        self._row = {e[0]: e for e in self.entries}
        leg = rm.LegBus(self.channel, self.ids, models=self.models)
        mode_prev = "idle"
        cur = {}                                    # name -> commanded target (URDF coordinates)
        raw_now = {}                                # name -> latest fb.raw (may be off by k*2*pi)
        lost_logged = set()                         # a lost turn count is logged once per joint
        homed = set()                               # joints that were homed gently and may use the faster limit
        fb_seen = {}                                # mid -> time of the latest feedback
        health_t = 0.0                              # throttle for the feedback health report
        rehome = set()                              # joints whose cur must be reset from fresh feedback after a watchdog event
        fired_seen = 0                              # board failsafe count seen so far
        t_fire = 0.0                                # time of the latest watchdog event (only later feedback counts)
        last_tick = time.monotonic()                # start of the previous control step (detects a stall of this thread)
        dt = 1.0 / RATE
        print(f"[bus] {self.channel} worker ready: {self.ids}")
        try:
            while True:
                tick = time.monotonic()
                with STATE.lock:
                    mode = STATE.mode
                    active = [e for e in self.entries if e[0] in STATE.active]
                    tgt_snapshot = dict(STATE.target)
                    limits = dict(STATE.limits)
                    fast = STATE.fast_slew
                    q_snapshot = dict(STATE.q) if self.grav else None

                # ---- mode change ----
                if mode != mode_prev:
                    homed.clear()                   # every entry into send mode homes gently again
                    if mode_prev == "send":
                        print(f"[bus] {self.channel} leaving send: damping -> disable")
                        leg.clear_failsafe()        # the host does damping -> disable itself; clear the board's failsafe frames first
                        leg.arm_watchdog(0)
                        for _ in range(int(0.4 / dt)):
                            leg.kd_only(1.0)
                            time.sleep(dt)
                        leg.stop_all()
                        cur.clear()
                        raw_now.clear()
                        fb_seen.clear()
                        rehome.clear()
                    if mode == "send" and active:
                        names = [e[0] for e in active]
                        print(f"[bus] {self.channel} enabling (bridge gain profile): {names}")
                        for name, mid, *_ in active:
                            leg.set_run_mode_motion(mid)
                            # Enable with retries: the driver's default wait of 4 ms is sized for the
                            # 50 Hz batch exchange and is too short for a one-off enable; a single late
                            # reply would otherwise cost the joint for the whole session.
                            fb = None
                            for _try in range(ENABLE_TRIES):
                                leg.enable(mid, timeout=ENABLE_WAIT)
                                fb = leg.last_fb.get(mid)
                                # last_fb survives across sessions: only fresh feedback from this enable counts
                                if fb is not None and time.monotonic() - fb.stamp <= 0.5:
                                    if _try:
                                        print(f"[bus] id={mid} {name} enabled on attempt {_try+1}")
                                    break
                                time.sleep(0.02)
                            if fb is None or time.monotonic() - fb.stamp > 0.5:
                                print(f"[bus] !! id={mid} {name} no fresh feedback after enabling"
                                      f" ({ENABLE_TRIES} attempts), joint skipped")
                                continue
                            fb_seen[mid] = fb.stamp
                            if abs(fb.raw - fb.angle) > 0.01:
                                print(f"[bus] id={mid} {name} turn count lost: raw={fb.raw:+.3f}"
                                      f" -> wrapped to {fb.angle:+.3f} (targets are sent within the current turn)")
                            raw_now[name] = fb.raw
                            q = self.to_sim(name, fb.raw)
                            cur[name] = q
                            with STATE.lock:
                                STATE.q[name] = q
                                STATE.qd[name] = self._row[name][3] * fb.speed
                                STATE.tau[name] = self._row[name][3] * fb.torque
                                STATE.q_stamp[name] = fb.stamp
                                STATE.temp[name] = fb.temp
                                STATE.ready.add(name)
                        # MC02 hardware watchdog (both calls are no-ops on other backends): right after
                        # enabling, register a kd-only (damping) frame per motor on the board and arm the
                        # watchdog. From then on every command_all feeds it; when the host goes silent or
                        # the process crashes, the board switches the motors to damping by itself.
                        if leg.register_failsafe(kd=1.0):
                            leg.arm_watchdog(WATCHDOG_MS)
                            print(f"[bus] {self.channel} board failsafe registered (kd-only x{len(leg.ids)}),"
                                  f" watchdog {WATCHDOG_MS} ms")
                        fired_seen = leg.failsafe_fired()
                        last_tick = time.monotonic()
                    mode_prev = mode

                # ---- periodic work ----
                if mode == "watch" and active:
                    for name, mid, *_ in active:
                        v = leg.read_float_param(mid, rm.IDX_MECH_POS, timeout=0.01)
                        if v is not None:
                            if name not in lost_logged and abs(v - wrap_pi(v)) > 0.01:
                                lost_logged.add(name)
                                print(f"[bus] id={mid} {name} turn count lost: raw={v:+.3f}"
                                      f" -> wrapped to {wrap_pi(v):+.3f}")
                            with STATE.lock:
                                STATE.q[name] = self.to_sim(name, v)
                                # Watch mode polls the motors one by one (read_float_param), so
                                # there is no Feedback stamp; the time of the read is used instead.
                                # This is not the batch path of send mode / policy deployment: do
                                # not use watch mode to measure feedback timing.
                                STATE.q_stamp[name] = time.monotonic()
                elif mode == "send" and cur:
                    # ---- watchdog fired / this thread stalled -> cur is stale ----
                    # While the board replays kd-only frames the joints sag away from the last target;
                    # resuming with the old cur and kp would snap them back within one control step.
                    # Two triggers: the board reports that the failsafe fired (the count changed), or two
                    # control steps of this thread are further apart than the watchdog timeout (the
                    # board's report may not have been processed yet, so waiting for it is not enough).
                    # Handling: mark the joints for re-homing, reset cur only from feedback that arrives
                    # after the event, and send kd-only (kp = 0) frames until it does.
                    fired_now = leg.failsafe_fired()
                    gap = tick - last_tick
                    if fired_now != fired_seen or gap > WATCHDOG_MS / 1000.0:
                        if fired_now != fired_seen:
                            print(f"[bus] !! {self.channel} board watchdog fired (total {fired_now}),"
                                  f" re-homing once fresh feedback arrives")
                        else:
                            print(f"[bus] !! {self.channel} worker stalled for {gap*1000:.0f} ms"
                                  f" (> watchdog {WATCHDOG_MS} ms), handled as a watchdog event, re-homing")
                        fired_seen = fired_now
                        t_fire = tick
                        rehome |= {e[0] for e in active}
                        homed.clear()
                    last_tick = tick
                    cmd = {}
                    ff = self.grav.tau_motor(q_snapshot) if self.grav else {}
                    for name, mid, *_ in active:
                        if name not in cur:
                            continue
                        model = self.models[mid]
                        if name in rehome:
                            fb = leg.last_fb.get(mid)
                            if fb is not None and fb.stamp > t_fire:
                                raw_now[name] = fb.raw
                                cur[name] = self.to_sim(name, fb.raw)
                                rehome.discard(name)
                            else:
                                # no feedback since the event yet: send only a damping frame in this
                                # control step (the same frame the board replays)
                                cmd[mid] = (raw_now[name], 0.0, 0.0, 1.0, 0.0)
                                continue
                        des = tgt_snapshot.get(name, cur[name])
                        if name in limits:
                            lo, hi = limits[name]
                            des = min(max(des, lo), hi)
                        cur[name], _h = slew_step(cur[name], des, model,
                                                  name in homed, fast, dt)
                        if _h:
                            homed.add(name)
                        cmd[mid] = (self.to_motor(name, cur[name], raw_now[name]), 0.0,
                                    JOINT_KP.get(name, MIRROR_KP[model]), JOINT_KD.get(name, MIRROR_KD[model]), ff.get(name, 0.0))
                    if cmd:
                        got = leg.command_all(cmd)
                        now = time.monotonic()
                        for mid in got:
                            fb_seen[mid] = now
                        with STATE.lock:
                            for name, mid, *_ in active:
                                fb = leg.last_fb.get(mid)
                                if fb is not None and now - fb.stamp < 0.5:
                                    raw_now[name] = fb.raw
                                    STATE.q[name] = self.to_sim(name, fb.raw)
                                    STATE.qd[name] = self._row[name][3] * fb.speed
                                    STATE.tau[name] = self._row[name][3] * fb.torque
                                    STATE.q_stamp[name] = fb.stamp
                                    STATE.temp[name] = fb.temp
                        if now - health_t > 1.0:
                            health_t = now
                            silent = [m for m in cmd if now - fb_seen.get(m, 0.0) > 0.5]
                            if silent:
                                print(f"[bus] !! {self.channel} no motor feedback for >0.5s:"
                                      f" id={silent} (commands may not be reaching the motors)")
                            hot = [(mid, leg.last_fb[mid].temp)
                                   for _, mid, *_ in active
                                   if mid in leg.last_fb and leg.last_fb[mid].temp > 75.0]
                            if hot:
                                print(f"[bus] !! {self.channel} motor over temperature: "
                                      + ", ".join(f"id={m} {t:.0f}C" for m, t in hot))
                else:
                    time.sleep(0.05)

                # Each worker sleeps a full dt from its own tick, so the buses run with a random phase offset
                # between them. This is intentional load balancing, do not "fix" it: workers locked to a
                # common time grid burst at the same instant, compete for USB bandwidth and CPU time, and
                # the feedback gets older and less regular.
                time.sleep(max(0.0, dt - (time.monotonic() - tick)))
        except Exception as e:
            # A dead worker means that its bus is no longer controlled, and the main loop cannot notice:
            # put all buses into the safe state.
            traceback.print_exc()
            print(f"[bus] !! {self.channel} worker crashed: {e}")
            safe_idle(f"{self.channel} worker crashed")
        finally:
            self.alive = False
            leg.shutdown()


def safe_idle(reason):
    with STATE.lock:
        if STATE.mode != "idle":
            print(f"[bridge] -> idle ({reason})")
        STATE.mode = "idle"
        STATE.active.clear()
        STATE.target.clear()
        STATE.ready.clear()


def handle_line(line, conn):
    msg = json.loads(line)
    if "hello" in msg:
        h = msg["hello"]
        if h.get("mode", "watch") == "send" and not CALIBRATED:
            # With a wrong sign the joint moves opposite to the target and the limit clamp is computed in the
            # wrong coordinates, which can drive the joint into its mechanical stop.
            err = (f"{os.path.basename(_CFG_PATH)} has calibrated: false (sign / offset are still placeholders), send refused;"
                   f" calibrate in watch mode first (tools/calib_arm.py)")
            conn.sendall((json.dumps({"matched": [], "unknown": [], "error": err}) + "\n").encode())
            print(f"[bridge] !! {err}")
            return False
        known = {e[0] for e in JOINT_MAP}
        req = h.get("joints") or list(known)
        matched = [n for n in req if n in known]
        unknown = [n for n in req if n not in known]
        with STATE.lock:
            STATE.active = set(matched)
            STATE.limits = {n: tuple(v) for n, v in (h.get("limits") or {}).items()}
            STATE.target.clear()
            STATE.ready.clear()
            STATE.mode = h.get("mode", "watch")
            mode = STATE.mode
        conn.sendall((json.dumps({"matched": matched, "unknown": unknown})
                      + "\n").encode())
        print(f"[bridge] hello mode={mode} matched={len(matched)} unknown={unknown}")
        if mode == "send":
            t0 = time.monotonic()
            while time.monotonic() - t0 < 4.0:
                with STATE.lock:
                    if STATE.ready >= STATE.active:
                        break
                time.sleep(0.05)
            with STATE.lock:
                q0 = {n: STATE.q[n] for n in STATE.active if n in STATE.q}
            conn.sendall((json.dumps({"q0": q0}) + "\n").encode())
    elif "target" in msg:
        bad = []
        with STATE.lock:
            for n, v in msg["target"].items():
                if n in STATE.active:
                    v = float(v)
                    if math.isfinite(v):
                        STATE.target[n] = v
                    else:
                        bad.append(n)        # JSON accepts NaN / Infinity: rejected, the joint keeps its previous target
        if bad:
            print(f"[bridge] !! non-finite target values dropped: {bad}")
        return bool(msg["target"])   # an empty target is only a keep-alive and does not end BOOT_GRACE
    return False


def serve(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"[bridge] listening on :{port}, waiting for a client (Ctrl+C to quit)")
    while True:
        conn, addr = srv.accept()
        conn.setblocking(False)
        print(f"[bridge] client connected {addr}")
        buf, last_rx, got_target = b"", time.monotonic(), False
        try:
            while True:
                r, _, _ = select.select([conn], [], [], 0.033)
                if r:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            if handle_line(line.decode(), conn):
                                got_target = True
                    # handle_line blocks while the motors are being enabled after hello (up to 4 s);
                    # last_rx is taken afterwards so that this wait does not count against the client's
                    # heartbeat timeout
                    last_rx = time.monotonic()
                with STATE.lock:
                    mode = STATE.mode
                    snap = {n: STATE.q[n] for n in STATE.active if n in STATE.q}
                allow = CLIENT_TIMEOUT if got_target else BOOT_GRACE
                if mode == "send" and time.monotonic() - last_rx > allow:
                    print(f"[bridge] client heartbeat timeout (>{allow:.1f}s)")
                    break
                if mode != "idle" and snap:
                    try:
                        conn.sendall((json.dumps({"q": snap}) + "\n").encode())
                    except (BlockingIOError, BrokenPipeError):
                        pass
        except (ConnectionResetError, json.JSONDecodeError) as e:
            print(f"[bridge] connection error: {e}")
        finally:
            safe_idle("client disconnected")
            conn.close()
            print("[bridge] waiting for the next client...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", metavar="YAML", default=None,
                    help="robot config with joints / buses / motor models (default biped_Laika/biped_Laika.yaml; "
                         "arms: humanoid_arm/humanoid_arm.yaml, Arm/Arm.yaml, Arm/Arm_pair.yaml)")
    ap.add_argument("--port", type=int, default=None,
                    help="TCP port to listen on (default: bridge.port of the yaml, else 9877)")
    ap.add_argument("--profile", default=None,
                    help="gain profile for send mode (a key of gain_profiles in the yaml; default mirror; use rl_default to bench-test the policy gains)")
    ap.add_argument("--grav-scale", type=float, default=None,
                    help="override gravity_ff.scale of the yaml (to raise an arm's gravity feed-forward in steps, 0.3 -> 0.6 -> 1.0, "
                         "without editing the yaml; ignored for a config without gravity_ff)")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else _CFG
    if args.grav_scale is not None and cfg.get("gravity_ff"):
        cfg["gravity_ff"]["scale"] = args.grav_scale
    profile = args.profile or ("mirror" if "mirror" in GAIN_PROFILES else next(iter(GAIN_PROFILES)))
    if profile not in GAIN_PROFILES:
        ap.error(f"--profile {profile} is not in {os.path.basename(_CFG_PATH)}; its gain_profiles are {list(GAIN_PROFILES)}")
    port = args.port or int((cfg.get("bridge") or {}).get("port", 9877))
    MIRROR_KP = {m: g[0] for m, g in GAIN_PROFILES[profile].items()}
    MIRROR_KD = {m: g[1] for m, g in GAIN_PROFILES[profile].items()}
    JOINT_KP = {j: g[0] for j, g in GAIN_OVERRIDES[profile].items()}
    JOINT_KD = {j: g[1] for j, g in GAIN_OVERRIDES[profile].items()}
    print(f"[bridge] config {os.path.basename(_CFG_PATH)}: {len(JOINT_MAP)} joints / "
          f"{len({e[2] for e in JOINT_MAP})} buses; gain profile {profile}: kp={MIRROR_KP} kd={MIRROR_KD}")
    if not CALIBRATED:
        print(f"[bridge] !! {os.path.basename(_CFG_PATH)} has calibrated: false: only watch is allowed, send (motor torque) is refused")
    grav = None
    if cfg.get("gravity_ff"):
        try:
            grav = GravityFF(cfg, JOINT_MAP)
        except Exception as e:                   # mujoco missing / wrong model path: say so and run without feed-forward (the arm only sags)
            print(f"[grav] !! gravity feed-forward unavailable, running without it: {type(e).__name__}: {e}")
    buses = {}
    for e in JOINT_MAP:
        buses.setdefault(e[2], []).append(e)
    for ch, entries in buses.items():
        BusWorker(ch, entries, grav=grav).start()
    try:
        serve(port)
    except KeyboardInterrupt:
        safe_idle("Ctrl+C")
        time.sleep(0.8)   # give the workers time to finish damping -> disable
