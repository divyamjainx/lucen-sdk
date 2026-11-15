#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/arm_drag_teleop.py - mouse-drag teleoperation: drag a "puppet" arm in a MuJoCo window, the real arm follows.

  double-click a link to select it (usually the hand; the elbow or the forearm work too)
  Ctrl + right-drag = translate          Ctrl + left-drag = rotate
  [ / ]   left gripper (single arm: its only gripper) close / open          - / =   right gripper close / open
  close the window or press Ctrl+C = stop (the target is set to the measured pose; the bridge goes to damping and
  disables the motors 1.5 s later)

No inverse kinematics is involved. The puppet is the MuJoCo model of the robot (model.scene in the config) with
gravity removed and damping added to every joint. The pull of the mouse is distributed over the joints by the
physics step, so joint limits, contact between links and the redundancy of a 7-axis arm are handled by the model
itself; release the mouse and the puppet stays where it is. The joint angles of the puppet are sent to the bridge
at 20 Hz with the same protocol as arm_jog / arm_web_jog (tools/arm_jog.BridgeClient). Works with bridge.py on the
robot computer and with tools/fake_bridge.py.

Safety, on top of the range clamp, the 1 rad/s slew limit and the 1.5 s heartbeat of the bridge:
  - after connecting, the puppet starts at the measured pose of the real arm, never at the zero pose
  - every puppet joint is limited to vel_max of the safety profile (safety_profiles[--safety] in the config) and
    clamped to soft_range / range
  - leash: a puppet joint may lead the real arm by at most --leash (default 0.35 rad). Beyond that it cannot be
    dragged further until the real arm catches up, so the target sent is never far from the real arm; dragging back
    is always possible. Without enough gravity feed-forward the real arm sags and keeps lagging, which shows as a
    direction in which the puppet cannot be dragged: set the feed-forward scale correctly first (docs/arm_guide.md)
    instead of raising --leash
  - the translucent "ghost" is the measured pose of the real arm; where it does not coincide with the puppet, the
    real arm has not caught up
The status line shows the largest lag of the real arm behind the puppet; "leash N" means that N joints are held
back by the leash at the moment.

Usage (on macOS the MuJoCo window needs mjpython; on Linux use python):
  python tools/fake_bridge.py --config humanoid_arm/humanoid_arm.yaml --grav-scale 1.0          # terminal 1: simulation
  mjpython tools/arm_drag_teleop.py --config humanoid_arm/humanoid_arm.yaml                     # terminal 2: drag
  mjpython tools/arm_drag_teleop.py --config humanoid_arm/humanoid_arm.yaml --host <robot-ip> --safety safe   # real robot
  mjpython tools/arm_drag_teleop.py --config Arm/Arm.yaml [--host <robot-ip>]                   # single Arm (bridge port 9880)
On the real robot, use it only after the arm is calibrated and the gravity feed-forward is set.
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from arm_jog import BridgeClient   # noqa: E402

TAU = 0.10            # s: deceleration time constant of the puppet joints (damping = joint inertia / TAU); they coast about vel_max * TAU
SEND_HZ = 20.0
LOOP_HZ = 100.0
GRIP_STEP = 0.15      # rad: gripper step per key press


class Puppet:
    """The model that the mouse drags. Needs only mujoco and the config and opens no window, so it also runs headless."""

    def __init__(self, cfg, vel_max, leash=0.35):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, cfg["model"]["scene"]))
        self.d = mujoco.MjData(self.m)
        self.vel_max, self.leash = float(vel_max), float(leash)
        soft = cfg.get("soft_range") or {}
        self.names, self.qadr, self.dadr, self.limits = [], {}, {}, {}
        for j in cfg["joints"]:
            if not j.get("enabled", True):
                continue
            jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
            if jid < 0:
                continue                                     # a joint that is not in the model cannot be dragged
            n = j["name"]
            self.names.append(n)
            self.qadr[n], self.dadr[n] = int(self.m.jnt_qposadr[jid]), int(self.m.jnt_dofadr[jid])
            self.limits[n] = tuple(float(v) for v in soft.get(n, j["range"]))
            self.m.jnt_range[jid] = self.limits[n]           # the puppet itself stops at the soft limits
        self.grippers = [j["name"] for j in cfg["joints"] if j.get("kind") == "gripper" and j["name"] in self.qadr]
        self.m.opt.gravity[:] = 0.0                          # no gravity on the puppet: it stays where released, the whole pull goes into motion
        mujoco.mj_forward(self.m, self.d)
        M = np.zeros((self.m.nv, self.m.nv))
        try:
            mujoco.mj_fullM(self.m, self.d, M)               # signature of MuJoCo >= 3.3
        except TypeError:
            mujoco.mj_fullM(self.m, M, self.d.qM)
        self.m.dof_frictionloss[:] = 0.0
        for n in self.names:                                 # damping scaled to the inertia each joint sees, for a uniform feel
            self.m.dof_damping[self.dadr[n]] = max(M[self.dadr[n], self.dadr[n]] / TAU, 0.05)
        for n in self.grippers:                              # grippers move by keys only: locked here, because dragging the hand
            self.m.dof_damping[self.dadr[n]] = 1e4           # sideways would let finger inertia open them (nudge still works)
        self._dofs = np.array([self.dadr[n] for n in self.names])
        self.held = []                                       # names of the joints held back by the leash in this step

    def _follow_passive(self):
        """Place the passive joints (gripper fingers) directly from the coefficients of their equality constraints, so the
        constraint does not jerk them at the start."""
        mj, m, d = self.mj, self.m, self.d
        for e in range(m.neq):
            if m.eq_type[e] == mj.mjtEq.mjEQ_JOINT and m.eq_obj2id[e] >= 0:
                j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
                c = m.eq_data[e]
                x = d.qpos[m.jnt_qposadr[j2]]
                d.qpos[m.jnt_qposadr[j1]] = c[0] + c[1] * x + c[2] * x * x

    def set_q(self, q):
        for n in self.names:
            if n in q:
                self.d.qpos[self.qadr[n]] = min(max(float(q[n]), self.limits[n][0]), self.limits[n][1])
        self._follow_passive()
        self.d.qvel[:] = 0.0
        self.mj.mj_forward(self.m, self.d)

    def q(self):
        return {n: float(self.d.qpos[self.qadr[n]]) for n in self.names}

    def nudge(self, name, delta):
        lo, hi = self.limits[name]
        self.d.qpos[self.qadr[name]] = min(max(self.d.qpos[self.qadr[name]] + delta, lo), hi)
        self._follow_passive()

    def lag(self, q_real):
        """Joint in which the real arm lags the puppet the most -> (rad, joint name)."""
        worst, who = 0.0, ""
        for n in self.names:
            if n in q_real:
                e = abs(self.d.qpos[self.qadr[n]] - q_real[n])
                if e > worst:
                    worst, who = e, n
        return worst, who

    def step(self, pert, dt, q_real=None):
        """Advance by dt seconds. pert = the MjvPerturb of the viewer (mouse drag); no force is applied while nothing is
        dragged. With q_real given, the leash is active."""
        mj, m, d = self.mj, self.m, self.d
        for _ in range(max(1, int(round(dt / m.opt.timestep)))):
            d.xfrc_applied[:] = 0.0
            if pert is not None:
                mj.mjv_applyPerturbForce(m, d, pert)
            mj.mj_step(m, d)
            d.qvel[self._dofs] = np.clip(d.qvel[self._dofs], -self.vel_max, self.vel_max)
        self.held = []
        for n in self.names if q_real else ():               # leash: clamp to real arm ± leash, block only outward velocity
            if n not in q_real:
                continue
            qa, da = self.qadr[n], self.dadr[n]
            lo, hi = q_real[n] - self.leash, q_real[n] + self.leash
            if d.qpos[qa] > hi:
                d.qpos[qa], d.qvel[da] = hi, min(d.qvel[da], 0.0)
                self.held.append(n)
            elif d.qpos[qa] < lo:
                d.qpos[qa], d.qvel[da] = lo, max(d.qvel[da], 0.0)
                self.held.append(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "humanoid_arm/humanoid_arm.yaml"), help="robot config (default: humanoid_arm/humanoid_arm.yaml)")
    ap.add_argument("--host", default="127.0.0.1", help="host the bridge runs on")
    ap.add_argument("--port", type=int, default=None, help="bridge port (default: bridge.port of the config)")
    ap.add_argument("--safety", default="slow",
                    help="safety profile (safety_profiles in the config); its vel_max limits the joint speed of the puppet")
    ap.add_argument("--leash", type=float, default=0.35,
                    help="rad: when the real arm lags the puppet by more than this, the puppet waits for it")
    ap.add_argument("--no-ghost", action="store_true", help="do not draw the translucent ghost of the real arm")
    ap.add_argument("--seconds", type=float, default=None, help="exit by itself after this many seconds (for automated tests)")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer
    cfg = yaml.safe_load(open(args.config))
    port = args.port or int((cfg.get("bridge") or {}).get("port", 9877))
    vel_max = float(cfg["safety_profiles"][args.safety]["vel_max"])
    pup = Puppet(cfg, vel_max, leash=args.leash)
    print(f"[drag] connecting to {args.host}:{port} ({os.path.basename(args.config)}), safety profile {args.safety}: "
          f"joint speed ≤ {vel_max} rad/s, leash {args.leash} rad; {len(pup.names)} joints")
    cli = BridgeClient(args.host, port, pup.names, pup.limits, mode="send")
    if cli.unknown:
        print(f"[drag] !! joints unknown to the bridge: {cli.unknown}")
    pup.set_q(cli.q0)                                        # the puppet starts at the measured pose of the real arm
    ghost = None if args.no_ghost else mujoco.MjData(pup.m)
    left, right = (pup.grippers + [None, None])[:2]
    keys = {ord("["): (left, +GRIP_STEP), ord("]"): (left, -GRIP_STEP),
            ord("-"): (right, +GRIP_STEP), ord("="): (right, -GRIP_STEP)}
    pending = []

    def on_key(key):
        if key in keys and keys[key][0]:
            pending.append(keys[key])

    print("[drag] double-click the hand (or any link) to select it -> Ctrl + right-drag translates / "
          "Ctrl + left-drag rotates;  "
          + ("[ / ] left gripper close / open   - / = right gripper close / open" if right
             else "[ / ] gripper close / open") + ";  close the window = stop")
    dt = 1.0 / LOOP_HZ
    last_send = last_print = 0.0
    t_start = time.monotonic()
    try:
        with mujoco.viewer.launch_passive(pup.m, pup.d, key_callback=on_key) as viewer:
            while viewer.is_running() and not cli.closed:
                tick = time.monotonic()
                if args.seconds is not None and tick - t_start > args.seconds:
                    break
                q_real = cli.latest()
                with viewer.lock():
                    while pending:
                        name, delta = pending.pop()
                        pup.nudge(name, delta)
                    pup.step(viewer.perturb, dt, q_real)
                    if ghost is not None:                    # ghost = measured pose of the real arm, drawn translucent over the puppet
                        ghost.qpos[:] = pup.d.qpos
                        for n, v in q_real.items():
                            if n in pup.qadr:
                                ghost.qpos[pup.qadr[n]] = v
                        mujoco.mj_kinematics(pup.m, ghost)
                        scn = viewer.user_scn
                        scn.ngeom = 0
                        mujoco.mjv_addGeoms(pup.m, ghost, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                        for i in range(scn.ngeom):
                            scn.geoms[i].rgba[:] = (0.25, 0.85, 0.75, 0.25)
                viewer.sync()
                if tick - last_send >= 1.0 / SEND_HZ:
                    cli.send_target(pup.q())
                    last_send = tick
                if tick - last_print >= 0.5:
                    lag, who = pup.lag(q_real)
                    sys.stdout.write(f"\r[drag] largest lag of the real arm {lag:.3f} rad ({who or '-'})"
                                     + (f"  leash {len(pup.held)}" if pup.held else "") + "          ")
                    sys.stdout.flush()
                    last_print = tick
                time.sleep(max(0.0, dt - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            q = cli.latest()
            cli.send_target({n: q.get(n, v) for n, v in pup.q().items()})     # hold the measured pose before leaving
        except Exception:
            pass
        cli.close()
        print("\n[drag] exit (the bridge goes to damping, then disables, after its 1.5 s heartbeat timeout)")


if __name__ == "__main__":
    main()
