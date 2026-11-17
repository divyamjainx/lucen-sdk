#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/arm_lead_follow.py - lead one arm by hand, the other arm follows (leader-follower teleoperation).

Grab one real arm (the leader) and move it: it goes where the hand takes it and stays where it is released. The
other arm (the follower) reproduces the motion: mirrored on humanoid_arm, joint for joint on the Arm pair. No
torque sensor is used, only joint positions.

Safety:
  - Both arms are enabled and hold position with torque all the time, the leader included. The leader is not limp:
    it gives way once a hand pushes a joint past a small threshold.
  - About 5 s after start-up both arms MOVE BY THEMSELVES for the gravity calibration: they visit a few poses,
    pause a few seconds in each and return to home_pose (about 20 s in total). Keep people and objects out of
    reach. Press n during those 5 s, or start with --no-calib, to skip it.
  - Press h (home) before quitting. After x or Ctrl+C the bridge switches to damping and disables the motors
    1.5 s later; an arm that is not in a rest pose then sinks.
  - At start-up the tool refuses to take over if a joint measures more than 0.3 rad outside its range (this
    points to a wrong zero offset or sign): recalibrate with tools/calib_arm.py.

Keys:
  l / r        humanoid_arm: lead the left / right arm by hand, the other one follows
  l / f        Arm pair: lead the arm named leader / follower (the key is the first letter of the arm name)
  h            both arms go to home_pose                    y   both arms go to ready_pose
  g g          gravity calibration (press g twice within 3 s): both arms visit the calibration poses, then go home
  o / c        gripper open / close, one step per key press (while leading: the leader's gripper moves and the
               follower's gripper follows it)
  space        lock / unlock (locked: both arms hold rigidly and cannot be led)
  x or Ctrl+C  quit (the target is set to the measured pose first)

Grippers follow as well: squeeze or spread the leader's gripper by hand and the follower's gripper goes to the
same opening; release it and it stays there. Home, ready and the gravity calibration do not move the grippers, so
an object held in a gripper is not dropped. When the follower's gripper holds an object or reaches its end stop,
its target stays at most GRIP_LEAD ahead of the measured angle, so the grip force does not keep rising.

How it works:
  The bridge holds every leader joint in position (kp plus gravity feed-forward). A hand moving the joint makes
  the measured angle deviate from the target. Once the deviation exceeds a threshold the joint counts as led and
  its target tracks the hand; when the hand lets go or moves back (the joint retreats by --retreat, less than
  0.5°) the target stops at the current angle. Slow motion counts too: the test is the distance from the target,
  not the speed.
  The follower's target is the measured angle of the leader, rate-limited by the safety profile. On mirrored arms,
  joints rotating about the X axis keep their sign and joints about Y / Z change it. If the next step would bring
  the follower into contact with another link (checked on the model), the follower does not advance in that
  control step.

Gravity calibration and the "feed-forward suggestion":
  How light the leader feels depends on the accuracy of the bridge's gravity feed-forward: the threshold has to
  cover the feed-forward error, otherwise the arm would drift with gravity by itself. The tool therefore learns
  that error. With a joint at rest, "commanded angle - measured angle" is the share of gravity the feed-forward
  does not carry. Per joint a coefficient a is fitted (sag = a * gravity torque of the model), and a * gravity
  torque is added to the command in every pose from then on. A joint that has not been learned keeps a margin
  for a feed-forward error of 30 percent (it feels stiff when raised and springs back a little on release); a
  learned joint keeps 6 percent, and about 1 N·m at the shoulder is enough to lead it. Samples are taken at rest
  only; there is no continuous feedback that could make the arm oscillate.
  Status line: "learned n/m" counts the arm joints learned so far. "feed-forward suggestion +0.07" (for example)
  is the learned error expressed as a correction of the bridge's gravity feed-forward scale: add it to the
  --grav-scale the bridge runs with (or to gravity_ff.scale in the config) and restart the bridge. The arm is
  then light even without the calibration.

Usage (run in a terminal, SSH is fine; the bridge accepts one client at a time):
  python tools/fake_bridge.py --config humanoid_arm/humanoid_arm.yaml --grav-scale 1.0 --view     # simulation
  python tools/arm_lead_follow.py --config humanoid_arm/humanoid_arm.yaml                         # second terminal
  python tools/arm_lead_follow.py --config humanoid_arm/humanoid_arm.yaml --host <robot-ip> --leader left --safety safe   # real robot

  python tools/fake_bridge.py --config Arm/Arm_pair.yaml --grav-scale 1.0 --view
  python tools/arm_lead_follow.py --config Arm/Arm_pair.yaml [--host <robot-ip>]
In the simulation window, double-click a link of an arm and Ctrl + right-drag it: that is the hand leading the arm.

The pairing is described by the lead_follow section of the config:
  arms         arm name -> joint-name prefix; the arm listed first is the leader at start-up
  map          same = the follower copies the leader's joint angles; mirror = left / right mirror image
  calib_poses  poses visited by the gravity calibration
  ff_joints    joints whose learned error is used for the feed-forward suggestion
Without that section the defaults are those of humanoid_arm: arms left / right, prefixes l_ / r_, mirrored.
"""
import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from arm_jog import BridgeClient, Keys, status_line   # noqa: E402

LOOP_HZ = 50.0
FOLLOW_K = 0.5         # learned joint: while led, the target trails the hand by only FOLLOW_K * threshold (halves the force needed)
U_NEW = 0.30           # joint not learned yet: threshold margin for a feed-forward error of up to 30 %
U_TRUST = 0.06         # learned joint: 6 % margin
TRUST_SGG = 9.0        # (N·m)²: learned = squared gravity torques of the samples sum above this (one settled sample in a pose above 3 N·m)
AFTER = 0.6            # s: for this long after release the threshold is at least AFTER_THR (the spring-back on release is not a hand)
AFTER_THR = 0.12       # rad
STILL = 0.006          # rad: a joint is at rest while its measured angle stays within this of its own 0.15 s average
STILL_FOR = 0.8        # s: at rest for this long -> take one sample
TAU_OBS = 0.3          # s: time constant of the average of "command - measured"
FIT_LAMBDA = 1.0       # (N·m)²: regularization of the fit "sag = a * gravity torque" (keeps a steady in poses with little gravity torque)
FIT_KEEP = 0.9         # each new sample multiplies the weight of the older samples by this
A_MAX = 0.04           # rad/(N·m): upper bound of a (at kp 25 it corresponds to a 100 % feed-forward error)
COMP_MAX = 0.6         # rad: upper bound of the compensation
GRIP_STEP = 0.15       # rad per key press
GRIP_LEAD = 0.25       # rad: max lead of the gripper target over the measured angle; grip torque stays near kp * this (2 N·m at kp 8)
GRIP_DRIVE = 1.5       # s: for this long after a key moved the gripper target (or until it arrives), deviation is not read as a hand on it
RANGE_SLACK = 0.30     # rad: measured angle this far outside its range at take-over = wrong zero offset or sign: do not take over
POSE_TOL = 0.05        # rad: a home / ready move has arrived when the measured angle is this close to the goal
CALIB_HOLD = 3.5       # s: longest stay in each calibration pose
STOP_EPS = 0.05        # rad: this close to a range end with gravity pushing outward = resting on the end stop: no gravity sample
HOLD_SKIP = 0.5        # s: wait in a calibration pose before averaging "command - measured" (used for joints that do not settle)
# Default gravity-calibration poses (values for the left arm, the right arm is mirrored); the arms return to home afterwards.
# Contact-free in simulation. In the second pose the hands are about 0.4 m forward and outward, 0.6 m high (about 13 N·m at the shoulder).
CALIB_POSES = [
    {"hip_pitch_joint": 0.2, "knee_joint": -1.2},
    {"hip_pitch_joint": -0.4, "hip_roll_joint": -0.4, "hip_yaw_joint": 0.3, "knee_joint": -0.9, "link4_joint": 0.4, "link5_joint": -0.3},
]


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def out_of_range(q, limits, slack=RANGE_SLACK):
    """Joints whose measured angle is outside its range by more than slack -> [(name, measured, lo, hi)]."""
    return [(n, float(q[n]), lo, hi) for n, (lo, hi) in limits.items() if n in q and not lo - slack <= float(q[n]) <= hi + slack]


class LeadFollow:
    """Pure logic: tick(measured q, dt) -> commanded angle of every joint. No network and no keyboard, so it can be
    driven from your own program or from tests."""

    def __init__(self, cfg, leader=None, band=0.03, retreat=0.008, vel_max=1.0, collide=True, profile="mirror"):
        self.cfg = cfg
        self.band, self.retreat, self.vel_max = float(band), float(retreat), float(vel_max)
        soft = cfg.get("soft_range") or {}
        rows = [j for j in cfg["joints"] if j.get("enabled", True)]
        self.names = [j["name"] for j in rows]
        self.limits = {j["name"]: tuple(float(v) for v in soft.get(j["name"], j["range"])) for j in rows}
        self.grippers = [j["name"] for j in rows if j.get("kind") == "gripper"]
        lfc = cfg.get("lead_follow") or {}
        self.pre = dict(lfc.get("arms") or {"left": "l_", "right": "r_"})     # arm name -> joint-name prefix (default: left / right)
        if len(self.pre) != 2:
            raise SystemExit(f"lead_follow.arms must name exactly two arms, got {list(self.pre)}")
        first, second = self.pre
        self.arm = {a: [n for n in self.names if n.startswith(p) and n not in self.grippers] for a, p in self.pre.items()}
        self._pre_of = {n: p for p in self.pre.values() for n in self.names if n.startswith(p)}
        if not self.arm[first] or [self.base(n) for n in self.arm[first]] != [self.base(n) for n in self.arm[second]]:
            raise SystemExit(f"two matching arms are required (joint names = {' / '.join(self.pre.values())} + the same "
                             "list of names); this config does not provide them")
        self.all_arm = self.arm[first] + self.arm[second]
        self._mates = {n: self.arm[a] for a in self.pre for n in self.arm[a]}     # joints of the same arm
        gains = (cfg.get("gain_profiles") or {}).get(profile) or {}
        self.kp = {j["name"]: float(gains[j["motor"]][0]) if j["motor"] in gains else 25.0 for j in rows}
        self.same = (lfc.get("map") or "mirror") == "same"
        if self.same:                                        # two identical arms: the follower copies the leader's joint angles
            self.msign = {self.base(n): 1.0 for n in self.arm[first]}
        else:
            # mirror signs: the arms are symmetric about the sagittal plane (x -> -x); rotations about X keep their sign, about Y / Z flip
            axis = {j.get("name"): [abs(float(v)) for v in j.get("axis").split()]
                    for j in ET.parse(os.path.join(ROOT, cfg["model"]["mjcf"])).getroot().iter("joint") if j.get("axis")}
            self.msign = {self.base(n): (1.0 if axis[n][0] > 0.9 else -1.0) for n in self.arm[first]}
        self.calib_poses = lfc.get("calib_poses") or CALIB_POSES            # values for the first arm; the second arm uses msign
        self.ff_joints = tuple(lfc.get("ff_joints") or ("hip_pitch_joint", "hip_roll_joint", "knee_joint"))
        # Preset poses cover the arm joints only: home / ready in the config list the gripper at 0 (fully open), and applying
        # that would open the gripper on every home move and every calibration.
        self.poses = {k: {n: clamp(float(v), *self.limits[n]) for n, v in (cfg.get(f"{k}_pose") or {}).items()
                          if n in self.limits and n not in self.grippers}
                      for k in ("home", "ready")}
        self.kin = None
        if collide:
            try:
                import mujoco
                self.mj = mujoco
                self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, cfg["model"]["scene"]))
                self.d = mujoco.MjData(self.m)
                self.qadr = {n: int(self.m.joint(n).qposadr[0]) for n in self.names}
                self.kin = True
            except Exception as e:                           # no mujoco: no collision check and no gravity learning, everything else works
                print(f"[lead] follower collision check and gravity learning unavailable ({type(e).__name__}: {e}); "
                      "continuing without them")
        self.grip_of = {side: next((g for g in self.grippers if g.startswith(pre)), None) for side, pre in self.pre.items()}
        leader = leader or first                             # default: the arm listed first in the config (left on humanoid_arm)
        if leader not in self.pre:
            raise SystemExit(f"leader = {leader!r} is not an arm of this config {list(self.pre)}")
        self.leader = leader
        self.mode = "lead"                                   # lead | pose | lock
        self.tgt, self.push, self.peak = {}, {}, {}
        self.driven = {n: 0.0 for n in self.names}           # gripper moved by a key: seconds left in which deviation is not read as a hand
        self.after = {n: 0.0 for n in self.names}            # timer after release
        self.g = {n: 0.0 for n in self.names}                # gravity torque of the model at the measured pose (N·m)
        self.a = {n: 0.0 for n in self.names}                # learned: sag not carried by the feed-forward ≈ a * gravity torque
        self.obs = {n: 0.0 for n in self.names}              # average of "command - measured"
        self.qf = {}
        self._sgg = {n: 0.0 for n in self.names}
        self._sgs = {n: 0.0 for n in self.names}
        self._still = {n: 0.0 for n in self.names}
        self.pose_goal, self.queue, self._pose_wait = None, [], 0.0
        self._acc, self._acc_t = {}, 0.0                     # calibration: "command - measured" integrated in the current pose / for how long
        self.blocked, self._was_hit = False, False

    # ---- state ----
    def base(self, n):
        """Joint name without its arm prefix (the part both arms share)."""
        return n[len(self._pre_of.get(n, "")):]

    def short(self, n):
        """Short name for the status line: no prefix, no trailing _joint."""
        b = self.base(n)
        return b[:-6] if b.endswith("_joint") else b

    @property
    def follower(self):
        return next(a for a in self.pre if a != self.leader)

    def start(self, q):
        """Take over at the measured pose: every joint holds where it is, any "being led" state is cleared."""
        self.tgt = {n: clamp(float(q[n]), *self.limits[n]) for n in self.names if n in q}
        self.push = {n: 0 for n in self.names}
        self.peak = {n: 0.0 for n in self.names}
        self.qf = {n: float(q[n]) for n in self.names if n in q}
        self.after = {n: AFTER for n in self.names}
        self.driven = {n: 0.0 for n in self.names}
        self._still = {n: 0.0 for n in self.names}
        self.blocked = False
        self._was_hit = bool(self.kin) and self._hits(q, {})  # already in contact at the start: allow it to move out

    def set_leader(self, side, q):
        self.leader = side
        self.mode = "lead"
        self.start(q)

    def go(self, pose_name):
        if self.poses.get(pose_name):
            self.queue = []
            self.pose_goal = dict(self.poses[pose_name])
            self.mode, self._pose_wait = "pose", 0.0

    def calibrate(self):
        """Gravity calibration: visit the calibration poses in turn, pausing a few seconds in each (samples are taken at
        rest and a is fitted), then return to home."""
        home = self.poses.get("home") or {n: 0.0 for n in self.all_arm}
        pa, pb = self.pre.values()
        seq = []
        for lp in self.calib_poses:
            pose = dict(home)
            for key, v in lp.items():
                pose[pa + key] = clamp(v, *self.limits[pa + key])
                pose[pb + key] = clamp(self.msign[key] * v, *self.limits[pb + key])
            seq.append(pose)
        seq.append(dict(home))
        self.queue = seq[1:]
        self.pose_goal = seq[0]
        self.mode, self._pose_wait = "pose", 0.0
        self._acc, self._acc_t = {}, 0.0

    def toggle_lock(self, q):
        if self.mode == "lock":
            self.mode = "lead"
            self.start(q)
        else:
            self.mode = "lock"

    def grip(self, delta, q=None):
        """Key press: move the gripper target by one step. While leading, only the leader's gripper moves (the follower's
        gripper tracks its measured opening); when locked or moving to a preset pose, both grippers move.
        With the measured q given, the target leads the measured angle by at most GRIP_LEAD (no further squeeze once
        an object is held or the end stop is reached)."""
        lead = self.grip_of.get(self.leader)
        for g in ([lead] if self.mode == "lead" and lead else self.grippers):
            if g in self.tgt:
                v = self.tgt[g] + delta
                if q is not None and g in q:
                    v = clamp(v, q[g] - GRIP_LEAD, q[g] + GRIP_LEAD)
                self.tgt[g] = clamp(v, *self.limits[g])
                self.push[g], self.driven[g] = 0, GRIP_DRIVE  # its deviation now comes from the key press, not from a hand

    def pushing(self):
        return [n for n in self.arm[self.leader] + [self.grip_of.get(self.leader)] if n and self.push.get(n)]

    def trusted(self, n):
        return self._sgg[n] > TRUST_SGG

    def thr(self, n):
        """Deviation (rad) beyond which this joint counts as led by a hand: band plus a margin for the possible error of
        the gravity feed-forward."""
        return self.band + (U_TRUST if self.trusted(n) else U_NEW) * abs(self.g[n]) / self.kp[n]

    def learned(self):
        return sum(1 for n in self.all_arm if self.trusted(n))

    def ff_hint(self):
        """Learned coefficients converted into a correction of the gravity feed-forward scale (negative = reduce it):
        the bridge's --grav-scale plus this value is the scale to use.
        At rest kp * sag = real gravity torque - feed-forward = delta * model gravity torque, hence delta = kp * a.
        Returns the median over the learned ff_joints (large gravity torque, well learned), or None if there are none."""
        vals = [self.kp[n] * self.a[n] for n in self.all_arm if self.trusted(n) and self.base(n) in self.ff_joints]
        return sorted(vals)[len(vals) // 2] if vals else None

    def comp(self, n):
        """Compensation added to the command (rad): the sag expected in this pose from the learned coefficient."""
        return clamp(self.a[n] * self.g[n], -COMP_MAX, COMP_MAX)

    def _cmd(self):
        return {n: clamp(v + self.comp(n), *self.limits[n]) for n, v in self.tgt.items()}

    def _gravity(self, q):
        """Gravity torque of every joint from the model at the measured pose -> self.g (stays 0 without mujoco: nothing
        is learned and the threshold is band alone)."""
        if not self.kin:
            return
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = 0.0
        for n in self.names:
            d.qpos[self.qadr[n]] = q.get(n, 0.0)
        d.qvel[:] = 0.0
        mj.mj_forward(m, d)
        for n in self.names:
            self.g[n] = float(d.qfrc_bias[m.joint(n).dofadr[0]])

    def _watch(self, n, q, dt, moving=False):
        """For a joint whose target is not moving and that nobody is leading: after STILL_FOR seconds at rest, take one
        sample (gravity torque, command - measured) for the fit of a. Observes only; never changes the command."""
        if moving:
            self.qf[n], self._still[n], self.obs[n] = q, 0.0, self.comp(n)
            return
        self.qf[n] += (q - self.qf[n]) * min(1.0, dt / 0.15)
        before = self._still[n]
        self._still[n] = before + dt if abs(q - self.qf[n]) < STILL else 0.0
        self.obs[n] += ((self.tgt[n] + self.comp(n) - q) - self.obs[n]) * min(1.0, dt / TAU_OBS)
        if self._still[n] >= STILL_FOR and int(before / STILL_FOR) != int(self._still[n] / STILL_FOR):
            self._sample(n)

    def _on_stop(self, n):
        """True if the joint rests on the end of its range while gravity pushes it further out (stowed pose of the Arm:
        upper arm and forearm on their lower end stops). The measured angle then comes from the end stop, not from
        the feed-forward."""
        lo, hi = self.limits[n]
        q = self.qf.get(n)
        return q is not None and ((q <= lo + STOP_EPS and self.g[n] > 0) or (q >= hi - STOP_EPS and self.g[n] < 0))

    def _sample(self, n):
        if abs(self.g[n]) < 1.0:                             # hardly any gravity in this pose: no information about a, skip
            return
        if any(self._on_stop(k) for k in self._mates.get(n, (n,))):
            # A joint of this arm rests on an end stop (stowed pose): the structure carries the arm, not the feed-forward,
            # so "command - measured" would wrongly teach "no error". Skip the whole arm.
            return
        self._sgg[n] = FIT_KEEP * self._sgg[n] + self.g[n] * self.g[n]
        self._sgs[n] = FIT_KEEP * self._sgs[n] + self.g[n] * self.obs[n]
        self.a[n] = clamp(self._sgs[n] / (self._sgg[n] + FIT_LAMBDA), -A_MAX, A_MAX)

    # ---- follower collision check ----
    def _hits(self, q, tgt_f):
        """True if a follower link touches anything with the leader at its measured pose and the follower at tgt_f."""
        return bool(self._contacts(q, tgt_f))

    def _contacts(self, q, tgt_f):
        """Same check, returning the set of (link, link) pairs in contact."""
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = 0.0
        for n in self.names:
            d.qpos[self.qadr[n]] = tgt_f.get(n, q.get(n, self.tgt.get(n, 0.0)))
        for e in range(m.neq):                               # gripper fingers follow the gripper joint
            if m.eq_type[e] == mj.mjtEq.mjEQ_JOINT and m.eq_obj2id[e] >= 0:
                c, x = m.eq_data[e], d.qpos[m.jnt_qposadr[m.eq_obj2id[e]]]
                d.qpos[m.jnt_qposadr[m.eq_obj1id[e]]] = c[0] + c[1] * x
        mj.mj_fwdPosition(m, d)
        pre = self.pre[self.follower]
        found = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
            if b1.startswith(pre) or b2.startswith(pre):
                found.add(frozenset((b1, b2)))
        return found

    def _lead(self, n, qn, dt):
        """One leader joint (measured angle qn). Further than the threshold from its target: a hand is leading it and the
        target tracks the hand. Retreating by retreat: released, the target stops there."""
        lo, hi = self.limits[n]
        thr = self.thr(n)
        if self.driven[n] > 0:                               # the gripper is moving to a new target set by a key
            self.driven[n] = 0.0 if abs(qn - self.tgt[n]) < 0.5 * thr else self.driven[n] - dt
            self._watch(n, qn, dt, moving=True)
            return
        if not self.push[n]:
            recent = self.after[n] > 0
            self.after[n] = max(0.0, self.after[n] - dt)
            dev = qn - self.tgt[n]
            if abs(dev) > (max(thr, AFTER_THR) if recent else thr):
                self.push[n] = 1 if dev > 0 else -1
                self.peak[n] = self.push[n] * qn
            else:
                self._watch(n, qn, dt)
        s = self.push[n]
        if s:
            self.peak[n] = max(self.peak[n], s * qn)
            if s * qn < self.peak[n] - self.retreat:         # released or moved back: stop here
                self.tgt[n], self.push[n], self.after[n] = clamp(qn, lo, hi), 0, AFTER
            else:
                off = (FOLLOW_K if self.trusted(n) else 1.0) * thr
                self.tgt[n] = clamp(qn - s * max(off, 1.5 * self.retreat), lo, hi)
            self._watch(n, qn, dt, moving=True)

    # ---- every control step ----
    def tick(self, q, dt):
        if not self.tgt:
            self.start(q)
        if self.mode == "lock":
            return self._cmd()
        self._gravity(q)
        step = self.vel_max * dt
        arms = self.all_arm
        if self.mode == "pose":
            arrived = all(abs(self.tgt[n] - g) < 1e-6 for n, g in self.pose_goal.items())
            for n, g in self.pose_goal.items():
                self.tgt[n] += clamp(g - self.tgt[n], -step, step)
            for n in arms:
                if n in q:
                    self._watch(n, q[n], dt, moving=not arrived or not self.queue)
            if arrived:
                self._pose_wait += dt
                if self.queue:                               # calibration: sample while holding this pose, then go to the next one
                    if self._pose_wait > HOLD_SKIP:
                        for n in arms:
                            if n in q:
                                self._acc[n] = self._acc.get(n, 0.0) + (self.tgt[n] + self.comp(n) - q[n]) * dt
                        self._acc_t += dt
                    done = all(self._still[n] >= STILL_FOR for n in arms)
                    if self._pose_wait > CALIB_HOLD or (self._pose_wait > 1.5 and done):
                        for n in arms:
                            if self._still[n] < STILL_FOR:
                                # Still swinging at the deadline (a heavy arm is underdamped, and a fresh sample changes the
                                # compensation and excites it again): use the average over the stay. An instantaneous
                                # value taken mid-swing would bias the fit.
                                if self._acc_t >= 1.0 and n in self._acc:
                                    self.obs[n] = self._acc[n] / self._acc_t
                                self._sample(n)
                        self.pose_goal, self._pose_wait = self.queue.pop(0), 0.0
                        self._acc, self._acc_t = {}, 0.0
                elif self._pose_wait > 2.0 or all(abs(q.get(n, g) - g) < POSE_TOL for n, g in self.pose_goal.items()):
                    self.mode = "lead"                       # arrived (or 2 s have passed without reaching the tolerance)
                    self.start(dict(self.tgt))
            return self._cmd()
        # leader (arm joints and its gripper): track the hand beyond the threshold, stop where the joint retreats
        lead_g, fol_g = self.grip_of.get(self.leader), self.grip_of.get(self.follower)
        for n in self.arm[self.leader] + ([lead_g] if lead_g else []):
            if n in q:
                self._lead(n, q[n], dt)
        # follower gripper: tracks the measured opening of the leader's gripper, rate-limited; when it holds an object or is
        # blocked, its target leads its own measured angle by at most GRIP_LEAD
        if lead_g and fol_g and lead_g in q and fol_g in q and fol_g in self.tgt:
            des = clamp(clamp(q[lead_g], q[fol_g] - GRIP_LEAD, q[fol_g] + GRIP_LEAD), *self.limits[fol_g])
            self.tgt[fol_g] += clamp(des - self.tgt[fol_g], -step, step)
        # follower arm: the leader's measured angles (mirrored or copied), rate-limited
        cand = {}
        for ln, fn in zip(self.arm[self.leader], self.arm[self.follower]):
            if ln in q and fn in self.tgt:
                des = clamp(self.msign[self.base(ln)] * q[ln], *self.limits[fn])
                cand[fn] = self.tgt[fn] + clamp(des - self.tgt[fn], -step, step)
        if self.kin:
            found = self._contacts(q, cand)
            if self.same and found:
                # Two identical arms: the leader's present pose exists physically, so the follower can take it too. Contacts
                # the model reports in that very pose (wrist resting on the upper arm when stowed, convex-hull
                # approximation) are not blocked; only contacts that appear on the follower's way there are.
                found -= self._contacts(q, {fn: clamp(q[ln], *self.limits[fn])
                                            for ln, fn in zip(self.arm[self.leader], self.arm[self.follower]) if ln in q})
            hit = bool(found)
            self.blocked = hit and not self._was_hit          # block only moves into a collision; a pose already in contact may move out
            self._was_hit = hit if not self.blocked else self._was_hit
        if not self.blocked:
            for fn, v in cand.items():
                if fn in q:
                    self._watch(fn, q[fn], dt, moving=abs(v - self.tgt[fn]) >= 1e-5)
            self.tgt.update(cand)
        return self._cmd()


def status_text(lf, hint):
    """Text of the status line. status_line trims from the end when the terminal is too narrow, so the important part
    comes first."""
    state = {"lead": f"leading {lf.leader} arm (learned {lf.learned()}/{len(lf.all_arm)})", "pose": "calibrating / moving to a pose",
             "lock": "locked"}[lf.mode]
    moving = [lf.short(n) for n in lf.pushing()]
    return (f"[lead] {state}" + ("  !! follower would collide, holding" if lf.blocked else "")
            + (f"  led joints: {' '.join(moving)}" if moving else "")
            + (f"  feed-forward suggestion {hint:+.2f} (add to bridge --grav-scale)" if hint is not None and abs(hint) > 0.03 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "humanoid_arm/humanoid_arm.yaml"), help="robot config (default: humanoid_arm/humanoid_arm.yaml); for the Arm pair use Arm/Arm_pair.yaml")
    ap.add_argument("--host", default="127.0.0.1", help="host the bridge runs on")
    ap.add_argument("--log", nargs="?", const="", default=None, metavar="CSV",
                    help="record every target sent and the measured joint angles to a CSV file (no name: "
                         "logs/<tool>_<time>.csv); recording does not change what is sent. "
                         "tools/arm_log_report.py reads the file")
    ap.add_argument("--port", type=int, default=None, help="bridge port (default: bridge.port of the config)")
    ap.add_argument("--leader", default=None,
                    help="arm led by hand at start-up, by name (humanoid_arm: left / right; default: the arm listed first "
                         "in the config); switch while running with the first letter of the name")
    ap.add_argument("--band", type=float, default=0.03,
                    help="rad: deviation from the held position beyond which a joint counts as led by a hand (smaller = lighter)")
    ap.add_argument("--retreat", type=float, default=0.008,
                    help="rad: a led joint that moves back by this much is released and stops there")
    ap.add_argument("--safety", default="slow",
                    help="safety profile (safety_profiles in the config); its vel_max limits the follower and the moves "
                         "to home / ready")
    ap.add_argument("--no-collide", action="store_true", help="skip the collision check of the follower")
    ap.add_argument("--profile", default="mirror",
                    help="gain profile the bridge runs with; used here to compute thresholds and the feed-forward suggestion, "
                         "it does not change the bridge (pass jog if the bridge was started with --profile jog)")
    ap.add_argument("--no-calib", action="store_true",
                    help="skip the gravity calibration at start-up (joints that have not been learned feel stiff when raised)")
    ap.add_argument("--no-range-check", action="store_true",
                    help="take over even if a measured angle is outside its range (default: refuse beyond 0.3 rad, which "
                         "usually means a wrong zero offset or sign)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    port = args.port or int((cfg.get("bridge") or {}).get("port", 9877))
    vel_max = float(cfg["safety_profiles"][args.safety]["vel_max"])
    arm_names = list((cfg.get("lead_follow") or {}).get("arms") or ("left", "right"))
    args.leader = args.leader or arm_names[0]
    lf = LeadFollow(cfg, leader=args.leader, band=args.band, retreat=args.retreat, vel_max=vel_max,
                    collide=not args.no_collide, profile=args.profile)
    switch = {a[0]: a for a in lf.pre}                       # leader keys = first letters of the arm names (l / r)
    same = all(v > 0 for v in lf.msign.values())
    how = "copies the motion" if same else "mirrors it"
    own_poses = bool((cfg.get("lead_follow") or {}).get("calib_poses"))
    calib_note = ("both arms leave home by themselves, hold %d poses for a few seconds each, then return to home"
                  % len(lf.calib_poses)) if own_poses \
        else "both arms raise themselves forward and outward to about 0.6 m height, hold for a few seconds, then return to home"
    print(f"[lead] connecting to {args.host}:{port} ({os.path.basename(args.config)}), leader = {args.leader}, band {args.band} rad,"
          f" safety profile {args.safety} (follower ≤ {vel_max} rad/s)")
    cli = BridgeClient(args.host, port, lf.names, lf.limits, mode="send", log=args.log, log_meta={"config": args.config})
    bad = out_of_range(cli.q0, lf.limits)
    if bad and not args.no_range_check:
        cli.close()                                          # no target sent: the bridge disconnects (damping -> disable), no joint moves
        print(f"[lead] not taking over: these joints measure more than {RANGE_SLACK} rad outside their range - wrong zero "
              "offset or sign (taken over, they would move back into range by themselves):")
        for n, v, lo, hi in bad:
            print(f"    {n:<20} measured {v:+.2f}   range [{lo:+.2f}, {hi:+.2f}]")
        cfg_name = os.path.basename(args.config)
        print(f"  Stop the bridge and check these joints again: tools/calib_arm.py --signs with {cfg_name} (press s to "
              f"skip the other joints); if the direction is right but the zero is not, use --zero-here <joint>.")
        print("  If the arm really is in this position and must be taken over as it is: add --no-range-check")
        sys.exit(1)
    lf.start(cli.q0)
    hint, g_armed = None, 0.0
    t_calib = None if (args.no_calib or not lf.kin) else time.monotonic() + 5.0
    print(f"[lead] keys: {' / '.join(switch)} lead the {' / '.join(switch.values())} arm   h home   y ready   g g gravity calibration"
          "   o / c gripper open / close   space lock / unlock   x quit (Ctrl+C = emergency stop)")
    print("[lead] grippers follow too: squeeze or spread the leader's gripper by hand and the other one goes to the same "
          "opening (right after take-over it first moves to the leader gripper's present opening)")
    if t_calib:
        print(f"[lead] !! automatic gravity calibration in 5 s: {calib_note} (about 20 s). "
              "Clear the area around the arms; press n to skip")
    else:
        print(f"[lead] grab the {lf.leader} arm and lead it; it stays where it is released, and the other arm {how}. "
              "Joints that have not learned gravity feel stiff when raised (press g twice to calibrate)")
    dt = 1.0 / LOOP_HZ
    last_print = 0.0
    try:
        with Keys() as keys:
            while not cli.closed:
                tick = time.monotonic()
                q = cli.latest()
                if t_calib and tick >= t_calib:
                    t_calib = None
                    lf.calibrate()
                for k in keys.poll():
                    if t_calib and k == "n":
                        t_calib = None
                        print("\n[lead] gravity calibration skipped")
                    elif k in switch:
                        lf.set_leader(switch[k], q)
                    elif k == "h":
                        lf.go("home")
                    elif k == "y":
                        lf.go("ready")
                    elif k == "g":
                        if tick - g_armed < 3.0:
                            lf.calibrate()
                            g_armed = 0.0
                        else:
                            g_armed = tick
                            print(f"\n[lead] gravity calibration: {calib_note}. "
                                  "Clear the area around the arms, then press g again within 3 s")
                    elif k == "o":
                        lf.grip(-GRIP_STEP, q)
                    elif k == "c":
                        lf.grip(+GRIP_STEP, q)
                    elif k == " ":
                        lf.toggle_lock(q)
                    elif k == "x":
                        raise KeyboardInterrupt
                cli.send_target(lf.tick(q, dt))
                if tick - last_print >= 0.2:
                    if lf.mode == "lead" and not lf.pushing():
                        hint = lf.ff_hint()
                    status_line(status_text(lf, hint))
                    last_print = tick
                time.sleep(max(0.0, dt - (time.monotonic() - tick)))
        print("\n[lead] bridge disconnected")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            q = cli.latest()
            cli.send_target({n: q.get(n, v) for n, v in lf.tgt.items()})     # hold the measured pose before leaving
        except Exception:
            pass
        cli.close()
        print("\n[lead] exit (the bridge goes to damping, then disables, after its 1.5 s heartbeat timeout)")


if __name__ == "__main__":
    main()
