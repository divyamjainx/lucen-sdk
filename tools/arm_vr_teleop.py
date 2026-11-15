#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/arm_vr_teleop.py - teleoperate the humanoid_arm with Meta Quest controllers through a WebXR page.

Hold the grip button and the robot's hand goes where your hand goes. Nothing is installed on the headset: its
browser opens a page served by this tool, and the page sends the pose and buttons of both controllers every frame.
This tool adds the displacement and rotation of a controller since its grip was pressed to the pose of that hand
of the robot (l_ee / r_ee), solves the inverse kinematics for the 7 joints of the arm and sends the joint targets
to the bridge. Works with Quest 3 / 3S and any other headset whose browser supports WebXR.

Requirements:
  - The headset is on the same LAN as the computer running this tool.
  - The page is served over https (WebXR requires it) with a self-signed certificate, created with openssl on
    first start. The browser warns about it: choose "Advanced" -> "Proceed".
  - mujoco is installed on the computer running this tool (kinematics and collision pre-check use the model).

Run it against the simulated bridge first:
  python tools/fake_bridge.py --config humanoid_arm/humanoid_arm.yaml --grav-scale 1.0 --view    # mjpython on macOS
  python tools/arm_vr_teleop.py --config humanoid_arm/humanoid_arm.yaml
Then on the robot:
  python bridge.py --config humanoid_arm/humanoid_arm.yaml                                  # on the robot computer
  python tools/arm_vr_teleop.py --config humanoid_arm/humanoid_arm.yaml --host <robot computer>
  (--host is not needed when this tool runs on the same machine as the bridge.)
In the headset browser open the address printed in the terminal, https://<this computer's IP>:8443/?k=<token>,
and press "Enter AR". Without a headset, open the same address in any browser: the page has a simulated
controller that exercises the whole chain.
About 5 s after start both arms move to ready_pose by themselves (forearms raised in front of the body): keep the
area clear, press n to cancel, or start with --no-ready.

Controls:
  Grip (middle finger)     Hold = this controller takes over its arm; release = that arm stops in place.
                           Out of reach? Release, bring your hand back and grip again (like lifting a mouse).
  Trigger (index finger)   Drives the gripper while the grip is held: fully pressed = closed, released = open.
                           With the grip released the gripper keeps its position (to rest while holding an object,
                           release the grip first, then the trigger). After a take-over the gripper waits until the
                           trigger comes close to its current opening, so taking over does not drop the object.
  Terminal keys            y  both arms to ready_pose (elbows bent; start teleoperating from here)
                           h  both arms to home_pose (hanging)        x or Ctrl+C  quit

Hand tracking (optional, off by default): start with --hands, enable hand tracking in the headset settings and
put the controllers down; the headset switches to hands by itself. The rules are the same, without buttons:
  Fist       Middle, ring and little finger curled into the palm and held for 0.25 s = grip held; opening these
             three fingers = grip released. The arm follows the displacement and rotation of your wrist.
  Pinch      The distance between thumb tip and index fingertip is the trigger: together = closed, 8 cm apart
             or more = fully open, proportional in between. Thumb and index finger stay free while the fist is held.
  A controller on one side and a bare hand on the other is fine; switching while engaged releases the arm first.
  Keep your hands where the headset cameras see them (in front of you): out of view = tracking lost = released.
  Do not pinch with the palm facing your face: that is the headset's system gesture (the left hand leaves the
  immersive session, the right hand opens the system menu).
  Hand tracking is noticeably less precise than controllers: validate against the simulated bridge before using
  it on the robot.

Stance (--stance):
  behind (default)  You face the same way as the robot (stand behind or beside it): the left controller drives
                    the left arm, the right controller the right arm; your forward = the robot's forward.
  facing            You face the robot, as in a mirror: the right controller drives the arm on your right (the
                    robot's left arm); pushing your hand forward moves the robot's hand toward its back.
  In both stances the robot's hand moves in the same direction in the room as your hand. Directions are fixed by
  the heading of the headset at the moment you enter AR / VR: take your position and look straight ahead before
  entering. If the directions are off, long-press the Meta button to recenter.

Safety:
  - An arm moves only while its grip is held (and during the ready / home moves).
  - Joint speed is limited by the safety profile (--safety, default slow = 1 rad/s).
  - Leash: a commanded angle never leads the measured angle by more than LEASH.
  - Collision pre-check: a control step that would bring the arm into contact with the other arm or with itself
    is not taken.
  - Auto-release: if the controller (or wrist) position jumps by more than 15 cm or its orientation by more than
    1 rad between two frames (tracking lost), or no new pose arrives for 0.25 s, the arm is released. Release
    the grip (open the hand) before taking over again.
  - The joint limits, watchdog and heartbeat of the bridge stay in force.
  To stop: release the grip. If that does not help, press Ctrl+C (the bridge goes to damping, then disables).
  The last resort is the 48 V emergency stop.
  Before quitting, press h to lower the arms: once the tool has exited, the bridge goes to damping and then
  disables after its 1.5 s heartbeat timeout, and a raised arm drops.

Access control. This page can move a real arm, so being able to connect is not enough to send commands:
  - Token: the ?k=xxxxxxxx at the end of the address (8 characters, generated randomly on first start, stored in
    ~/.cache/lucen_vr/token and reused afterwards, so a bookmark in the headset keeps working; --new-token
    replaces it). Requests that send poses, press buttons or read state must carry it; without the right
    token the answer is always 403.
  - One interface: by default the server listens only on this computer's LAN address, not on 0.0.0.0 (--listen
    changes that). --http (unencrypted) is accepted on 127.0.0.1 only.
  - Requests made by other web pages through your browser (cross-site) and names resolved to this machine (DNS
    rebinding) are rejected: Origin and Host are checked, and only application/json bodies are accepted.
  - Controller data is validated first (3 / 4 finite numbers, a sane quaternion norm, a plausible position); an
    invalid sample is dropped whole. If the inverse kinematics returns a non-finite value, the arm is released
    and that value never reaches the bridge.
  The bridge's own port (9879) has no token, and this tool does not change that: the robot must be on a
  trusted network.
"""
import argparse
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from arm_jog import BridgeClient, Keys                       # noqa: E402
from arm_lead_follow import out_of_range, RANGE_SLACK        # noqa: E402

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_vr_teleop.html")
LOOP_HZ = 50.0
ENGAGE, RELEASE = 0.6, 0.3   # grip value above ENGAGE = held, below RELEASE = released
STALE = 0.25           # s: no new pose from this controller for this long (tracking lost / page stopped) -> auto-release
JUMP = 0.15            # m: position change between two frames that counts as a tracking jump -> auto-release
JUMP_ROT = 1.0         # rad: orientation change between two frames that counts as a tracking flip (more than a wrist turns in one frame) -> auto-release
# Hand tracking (no buttons): the page sends curl and pinch; they are converted here to a 0..1 grip / trigger value
#   curl  = mean over middle / ring / little finger of (fingertip-to-wrist distance / bone length from wrist to fingertip):
#           straight ≈ 0.95, relaxed ≈ 0.9, fist ≈ 0.4
#   pinch = distance between thumb tip and index fingertip (m)
HAND_OPEN, HAND_FIST = 0.85, 0.50   # curl >= HAND_OPEN -> grip 0; <= HAND_FIST -> grip 1 (ENGAGE 0.6 = curl 0.64, RELEASE 0.3 = curl 0.745)
PINCH_OPEN, PINCH_SHUT = 0.08, 0.02  # m: pinch >= PINCH_OPEN -> trigger 0 (gripper fully open); <= PINCH_SHUT -> trigger 1 (closed)
HAND_HOLD = 0.25       # s: a fist must persist this long to take over (filters tracking flicker and the wrist wobble while it closes); a controller grip engages at once
K_POS = 10.0           # 1/s: hand position error -> velocity
K_ROT = 8.0            # 1/s: hand orientation error -> angular velocity
E_POS = 0.08           # m: position error followed per control step is capped at this (a target out of reach is approached slowly, and the arm comes back with your hand)
E_ROT = 0.40           # rad
W_ROT = 0.20           # m/rad: in the inverse kinematics, 1 rad of orientation error weighs as much as this many metres of position error
LAMBDA = 0.05          # damping of the damped least-squares solve (keeps joint speeds bounded near singular configurations)
K_NULL = 0.3           # 1/s: pulls the redundant degree of freedom of the 7-joint arm toward the ready_pose angles (keeps the elbow in place without moving the hand)
LEASH = 0.35           # rad: leash - a commanded angle leads the measured angle by at most this much
GRIP_VEL = 2.5         # rad/s: speed limit of the gripper target
GRIP_LEAD = 0.25       # rad: the gripper target leads the measured angle by at most this much (about 2 N·m on a held object; the grasp does not keep tightening)
GRIP_CATCH = 0.20      # rad: after a take-over the gripper starts following once the trigger is this close to its current opening
POSE_TOL = 1e-6
P_MAX = 20.0           # m: a controller coordinate beyond this is not valid data; the sample is dropped
BODY_MAX = 8192        # bytes: request body limit (one frame with both controllers is under 400 bytes)
TOKEN_ABC = "abcdefghjkmnpqrstuvwxyz23456789"                # token alphabet, without the look-alikes 0 / o / 1 / l / i
FF_G_MIN = 3.0         # N·m: joints with less gravity torque than this are not used for the feed-forward estimate (too little information)
FF_STILL = 0.10        # rad/s: a joint counts as at rest below this measured speed (while it moves, the error also contains friction and inertia)
FF_TAU = 2.0           # s: forgetting time of the estimate
FF_MIN_SGG = 150.0     # (N·m)²: weight to accumulate before a value is reported (about one 9 N·m joint at rest for 0.4 s)
FF_WARN = 0.25         # |Δ| above this = the feed-forward is clearly wrong; the status line shows a warning

# WebXR local-floor frame (x right, y up, z toward your back) -> robot frame (x robot's left, y robot's back, z up; the robot faces -y)
M_STANCE = {
    "behind": np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),   # your right = robot's right, your forward = robot's forward
    "facing": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),   # face to face: your right = robot's left, your forward = robot's back
}
ARM_OF = {"behind": {"left": "left", "right": "right"}, "facing": {"left": "right", "right": "left"}}   # controller -> arm


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def quat_to_mat(q):
    """WebXR quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_vec(R):
    """Rotation matrix -> rotation vector (axis × angle, rad; angle in [0, π]). Goes through a quaternion, so the
    result stays finite near 180°."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12))
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12))
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12))
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    v = np.array([x, y, z], dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    if w < 0.0:
        v, w = -v, -w
    return v / n * (2.0 * float(np.arctan2(n, w)))


def clean_sample(c):
    """Validate one controller (or hand) sample sent by the page -> {"p", "q", "grip", "trigger"}, or None if it is
    malformed (missing field, not a number, non-finite, implausible).
    Hand-tracking samples carry curl / pinch (see the constants above); they are converted to grip / trigger here
    and marked "hand": True. JSON accepts NaN and Infinity: without this check they would pass through the inverse
    kinematics and reach the motors as non-finite targets."""
    try:
        if not isinstance(c, dict) or not isinstance(c.get("p"), (list, tuple)) or not isinstance(c.get("q"), (list, tuple)):
            return None
        p, q = [float(v) for v in c["p"]], [float(v) for v in c["q"]]
        by_hand = "curl" in c or "pinch" in c
        if by_hand:
            curl, pinch = float(c["curl"]), float(c["pinch"])
            if not (0.0 <= curl <= 2.0 and 0.0 <= pinch <= 0.5):         # NaN fails both comparisons too
                return None
            grip, trig = (HAND_OPEN - curl) / (HAND_OPEN - HAND_FIST), (PINCH_OPEN - pinch) / (PINCH_OPEN - PINCH_SHUT)
        else:
            grip, trig = float(c.get("grip", 0.0)), float(c.get("trigger", 0.0))
    except (TypeError, ValueError, KeyError):
        return None
    if len(p) != 3 or len(q) != 4 or not all(math.isfinite(v) for v in p + q + [grip, trig]):
        return None
    if max(abs(v) for v in p) > P_MAX or not 0.5 < sum(v * v for v in q) ** 0.5 < 2.0:
        return None
    out = {"p": p, "q": q, "grip": clamp(grip, 0.0, 1.0), "trigger": clamp(trig, 0.0, 1.0)}
    if by_hand:
        out["hand"] = True
    return out


class FfProbe:
    """Estimate how far the bridge's gravity feed-forward is off, from (commanded - measured) angle. This is the
    same quantity as the "feed-forward suggestion" of arm_lead_follow:
      at rest  kp·(commanded - measured) = true gravity torque - feed-forward = Δ·model gravity torque
      ->  Δ = Σ kp·e·g / Σ g²   (joints weighted by their gravity torque, with forgetting)
    Δ > 0: too little feed-forward, the arm sits below the command (it sags).
    Δ < 0: too much, the arm rides above the command (floats up, rises faster than you, will not come down).
    The value to use is the bridge's current --grav-scale plus Δ. While the command is held back by the leash
    (|e| at LEASH) the result is only a lower bound.
    kp is taken from the gain profile the bridge runs (yaml gain_profiles; --profile, default mirror)."""

    def __init__(self, mj, m, cfg, names, profile="mirror"):
        gains = (cfg.get("gain_profiles") or {}).get(profile) or {}
        self.mj, self.m, self.d = mj, m, mj.MjData(m)
        self.kp = {j["name"]: float(gains[j["motor"]][0]) for j in cfg["joints"]
                   if j.get("enabled", True) and j["name"] in names and j.get("kind") != "gripper" and j.get("motor") in gains}
        self.qadr = {n: int(m.joint(n).qposadr[0]) for n in names}
        self.dadr = {n: int(m.joint(n).dofadr[0]) for n in self.kp}
        self.seg = self.sgg = self.ssat = 0.0
        self.prev, self.acc = None, 0.0

    def gravity(self, q):
        """Torque each joint needs to hold the measured pose against gravity (N·m): the same model and the same
        quantity (qfrc_bias) the bridge uses for its feed-forward."""
        d = self.d
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        for n, a in self.qadr.items():
            d.qpos[a] = q.get(n, 0.0)
        self.mj.mj_forward(self.m, d)
        return {n: float(d.qfrc_bias[a]) for n, a in self.dadr.items()}

    def update(self, q, cmd, dt):
        self.acc += dt
        if self.acc < 0.1 - 1e-9:                             # 10 Hz is enough
            return
        T, self.acc = self.acc, 0.0
        k = math.exp(-T / FF_TAU)
        self.seg, self.sgg, self.ssat = self.seg * k, self.sgg * k, self.ssat * k
        prev, self.prev = self.prev, dict(q)
        if prev is None or not all(math.isfinite(q.get(n, float("nan"))) and n in cmd for n in self.kp):
            return
        g = self.gravity(q)
        for n, kp in self.kp.items():
            if abs(g[n]) < FF_G_MIN or n not in prev or abs(q[n] - prev[n]) / T > FF_STILL:
                continue
            e = cmd[n] - q[n]
            self.seg += kp * e * g[n]
            self.sgg += g[n] * g[n]
            if abs(e) > LEASH - 0.03:
                self.ssat += g[n] * g[n]

    def value(self):
        """-> (Δ, True if it is only a lower bound), or (None, False) while there is not enough at-rest data."""
        if self.sgg < FF_MIN_SGG:
            return None, False
        return self.seg / self.sgg, self.ssat > 0.5 * self.sgg


def ff_text(ff, floor):
    """Feed-forward note for the status line (empty when there is no estimate or the error is small)."""
    if ff is None or abs(ff) < 0.08:
        return ""
    amt = f"{'at least ' if floor else ''}{ff:+.2f}"
    if abs(ff) < FF_WARN:
        return f"  feed-forward suggestion {amt}"
    return f"  !! gravity feed-forward too {'strong' if ff < 0 else 'weak'} (arm {'floats up' if ff < 0 else 'sags'}): change the bridge's --grav-scale by {amt} and restart it"


class Hand:
    """Take-over state of one arm."""
    def __init__(self):
        self.engaged = False
        self.armed = True                                    # after a fault (jump / timeout) the grip must be released before it can take over again
        self.note = ""
        self.blocked = False
        self.was_hit = False
        self.caught = False                                  # True once the gripper has started following the trigger
        self.c_p0 = self.c_R0 = self.ee_p0 = self.ee_R0 = None
        self.last_p = self.last_R = None
        self.by_hand = False                                 # the latest sample came from hand tracking (not a controller)
        self.since = None                                    # time since which the grip / fist has stayed above ENGAGE
        self.tgt_p = self.tgt_R = None


class VrTeleop:
    """Teleoperation logic without any I/O: tick(measured q, controller samples, current time, dt) -> commanded
    angles of the 16 joints."""

    def __init__(self, cfg, stance="behind", scale=1.0, vel_max=1.0, collide=True, profile="mirror"):
        import mujoco
        self.mj = mujoco
        self.cfg, self.stance, self.scale, self.vel_max, self.collide = cfg, stance, float(scale), float(vel_max), collide
        self.M = M_STANCE[stance]
        soft = cfg.get("soft_range") or {}
        rows = [j for j in cfg["joints"] if j.get("enabled", True)]
        self.names = [j["name"] for j in rows]
        self.limits = {j["name"]: tuple(float(v) for v in soft.get(j["name"], j["range"])) for j in rows}
        self.grippers = [j["name"] for j in rows if j.get("kind") == "gripper"]
        pre = {"left": "l_", "right": "r_"}
        self.arm = {s: [n for n in self.names if n.startswith(p) and n not in self.grippers] for s, p in pre.items()}
        self.grip_of = {s: next((g for g in self.grippers if g.startswith(p)), None) for s, p in pre.items()}
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, cfg["model"]["scene"]))
        self.d = mujoco.MjData(self.m)
        self.qadr = {n: int(self.m.joint(n).qposadr[0]) for n in self.names}
        self.dofs = {s: [int(self.m.joint(n).dofadr[0]) for n in self.arm[s]] for s in pre}
        ee = cfg["end_effector"] if isinstance(cfg["end_effector"], list) else [cfg["end_effector"]]
        site = {e["site"][:2]: e["site"] for e in ee}
        self.site = {s: int(self.m.site(site[p]).id) for s, p in pre.items()}
        self.prefix = pre
        self.poses = {k: {n: clamp(float(v), *self.limits[n]) for n, v in (cfg.get(f"{k}_pose") or {}).items()
                          if n in self.limits and n not in self.grippers} for k in ("home", "ready")}
        self.rest = {n: self.poses["ready"].get(n, 0.0) for s in pre for n in self.arm[s]}
        self.hand = {s: Hand() for s in pre}
        self.q_cmd, self._good = {}, {}
        self.mode, self.pose_goal = "hold", None
        self.ff = FfProbe(mujoco, self.m, cfg, self.names, profile)

    # ---- kinematics ----
    def _fk(self, q):
        mj, m, d = self.mj, self.m, self.d
        d.qpos[:] = 0.0
        for n in self.names:
            d.qpos[self.qadr[n]] = q.get(n, 0.0)
        for e in range(m.neq):                               # the gripper fingers follow the gripper joint
            if m.eq_type[e] == mj.mjtEq.mjEQ_JOINT and m.eq_obj2id[e] >= 0:
                c, x = m.eq_data[e], d.qpos[m.jnt_qposadr[m.eq_obj2id[e]]]
                d.qpos[m.jnt_qposadr[m.eq_obj1id[e]]] = c[0] + c[1] * x
        mj.mj_fwdPosition(m, d)

    def ee_pose(self, side, q=None):
        """Position and orientation of this arm's hand (ee site) in the robot frame."""
        self._fk(self.q_cmd if q is None else q)
        return self.d.site_xpos[self.site[side]].copy(), self.d.site_xmat[self.site[side]].reshape(3, 3).copy()

    def _hits(self, side):
        """True if a link of this arm is in contact in the pose last passed to _fk."""
        m, d, pre = self.m, self.d, self.prefix[side]
        for i in range(d.ncon):
            c = d.contact[i]
            if m.body(m.geom_bodyid[c.geom1]).name.startswith(pre) or m.body(m.geom_bodyid[c.geom2]).name.startswith(pre):
                return True
        return False

    def _ik_step(self, side, p_tgt, R_tgt, dt):
        """One control step toward the target pose -> new commanded angles of this arm's 7 joints (speed-limited,
        clamped to the joint range)."""
        mj, m, d = self.mj, self.m, self.d
        self._fk(self.q_cmd)
        sid, dofs, names = self.site[side], self.dofs[side], self.arm[side]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mj.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, dofs], W_ROT * jr[:, dofs]])
        e_p = p_tgt - d.site_xpos[sid]
        n_p = float(np.linalg.norm(e_p))
        if n_p > E_POS:
            e_p *= E_POS / n_p
        e_r = rot_vec(R_tgt @ d.site_xmat[sid].reshape(3, 3).T)
        n_r = float(np.linalg.norm(e_r))
        if n_r > E_ROT:
            e_r *= E_ROT / n_r
        v = np.concatenate([K_POS * e_p, W_ROT * K_ROT * e_r])
        A = J @ J.T + LAMBDA ** 2 * np.eye(6)
        q = np.array([self.q_cmd[n] for n in names])
        qd = J.T @ np.linalg.solve(A, v)
        null = np.eye(len(names)) - J.T @ np.linalg.solve(A, J)
        qd += null @ (K_NULL * (np.array([self.rest[n] for n in names]) - q))
        top = float(np.max(np.abs(qd)))
        if top > self.vel_max:                               # scale all joints together to keep the direction
            qd *= self.vel_max / top
        return {n: clamp(float(q[i] + qd[i] * dt), *self.limits[n]) for i, n in enumerate(names)}

    # ---- state ----
    def start(self, q):
        """Start from the measured pose: every joint holds its position and neither arm is engaged."""
        self.q_cmd = {n: clamp(float(q[n]), *self.limits[n]) for n in self.names if n in q and math.isfinite(float(q[n]))}
        self._good = dict(self.q_cmd)
        for h in self.hand.values():
            h.engaged, h.armed, h.note, h.blocked, h.since = False, True, "", False, None
        self._fk(self.q_cmd)
        for s, h in self.hand.items():
            h.was_hit = self._hits(s)                        # already in contact at start: allow it to move out

    def go(self, pose_name):
        if self.poses.get(pose_name):
            self.pose_goal, self.mode = dict(self.poses[pose_name]), "pose"
            for h in self.hand.values():
                h.engaged, h.note = False, ""

    def _drop(self, h, why):
        h.engaged, h.armed, h.note = False, False, why

    def tick(self, q, ctl, now, dt):
        """q: measured angles; ctl: {"left" / "right": {"p": [x, y, z], "q": [x, y, z, w], "grip": 0..1, "trigger": 0..1, "t": time received}}"""
        if not self.q_cmd:
            self.start(q)
        step = self.vel_max * dt
        if self.mode == "pose":
            for n, g in self.pose_goal.items():
                self.q_cmd[n] += clamp(g - self.q_cmd[n], -step, step)
            if all(abs(self.q_cmd[n] - g) < POSE_TOL for n, g in self.pose_goal.items()):
                self.mode = "hold"
                self._fk(self.q_cmd)
                for s, h in self.hand.items():
                    h.was_hit = self._hits(s)
            for h in self.hand.values():
                h.armed = False                              # a grip held during a preset-pose move does not count: release it and grip again after arrival
        else:
            for ctl_side, arm_side in ARM_OF[self.stance].items():
                self._hand_tick(arm_side, (ctl or {}).get(ctl_side), q, now, dt)
        for n in self.q_cmd:                                 # leash
            if n in q and math.isfinite(q[n]):
                self.q_cmd[n] = clamp(clamp(self.q_cmd[n], q[n] - LEASH, q[n] + LEASH), *self.limits[n])
        if all(math.isfinite(v) for v in self.q_cmd.values()):
            self._good = dict(self.q_cmd)
        else:                                                # last guard: no non-finite value leaves this function; restore the previous command and release both arms
            self.q_cmd = dict(self._good)
            for h in self.hand.values():
                self._drop(h, "non-finite value in the command, released")
        self.ff.update(q, self.q_cmd, dt)
        return dict(self.q_cmd)

    def _hand_tick(self, side, c, q, now, dt):
        h = self.hand[side]
        fresh = bool(c) and now - float(c.get("t", -1e9)) < STALE
        grip = float(c.get("grip", 0.0)) if fresh else 0.0
        p = np.array(c["p"], dtype=float) if fresh else None
        R = quat_to_mat(c["q"]) if fresh else None
        by_hand = bool(fresh and c.get("hand"))
        again = "open the hand and make a fist again" if h.by_hand else "release and grip again"
        if h.engaged:
            if not fresh:
                self._drop(h, ("hand not tracked" if h.by_hand else "no controller signal") + f", released ({again})")
            elif by_hand != h.by_hand:                       # controller put down / picked up while engaged: the two track different points, so the reference is void
                self._drop(h, "switched between controller and hand, released (" + ("open the hand and make a fist again" if by_hand else "release and grip again") + ")")
            elif grip < RELEASE:
                h.engaged, h.note = False, ""
            elif float(np.linalg.norm(p - h.last_p)) > JUMP:
                self._drop(h, ("hand" if h.by_hand else "controller") + f" position jumped, released ({again})")
            elif float(np.linalg.norm(rot_vec(R @ h.last_R.T))) > JUMP_ROT:
                self._drop(h, ("hand" if h.by_hand else "controller") + f" orientation jumped, released ({again})")
        elif fresh and grip < RELEASE:
            h.armed, h.note = True, ""                       # re-arm only after the grip has been seen released
        elif fresh and h.armed and grip > ENGAGE and self.mode == "hold":
            h.since = now if h.since is None else h.since
            if now - h.since >= (HAND_HOLD if by_hand else 0.0) - 1e-9:
                h.engaged, h.caught, h.blocked, h.note = True, False, False, ""
                h.c_p0, h.c_R0 = p.copy(), R                 # the reference is taken at the take-over step (wrist motion during the 0.25 s fist hold is ignored)
                h.ee_p0, h.ee_R0 = self.ee_pose(side)
        if h.engaged or not (fresh and grip > ENGAGE):
            h.since = None
        if fresh:
            h.last_p, h.last_R, h.by_hand = p, R, by_hand
        if not h.engaged:
            h.blocked = False
            return
        # controller displacement / rotation since the grip was pressed (world frame), mapped to the robot frame and added to the hand pose at that moment
        M = self.M
        h.tgt_p = h.ee_p0 + self.scale * (M @ (p - h.c_p0))
        h.tgt_R = (M @ (R @ h.c_R0.T) @ M.T) @ h.ee_R0
        cand = self._ik_step(side, h.tgt_p, h.tgt_R, dt)
        if not all(math.isfinite(v) for v in cand.values()):  # should not happen; if it does, release and keep the previous command
            self._drop(h, "inverse kinematics returned a non-finite value, released")
            return
        if self.collide:
            self._fk({**self.q_cmd, **cand})
            hit = self._hits(side)
            h.blocked = hit and not h.was_hit                # only moving into contact is blocked; an arm already in contact may move out
            if not h.blocked:
                h.was_hit = hit
        if not h.blocked:
            self.q_cmd.update(cand)
        g = self.grip_of.get(side)
        if g and g in self.q_cmd:
            lo, hi = self.limits[g]
            des = lo + clamp(float(c.get("trigger", 0.0)), 0.0, 1.0) * (hi - lo)
            h.caught = h.caught or abs(des - self.q_cmd[g]) < GRIP_CATCH
            if h.caught:
                v = self.q_cmd[g] + clamp(des - self.q_cmd[g], -GRIP_VEL * dt, GRIP_VEL * dt)
                if g in q:
                    v = clamp(v, q[g] - GRIP_LEAD, q[g] + GRIP_LEAD)
                self.q_cmd[g] = clamp(v, lo, hi)

    def status(self):
        """Status for the page, keyed by controller / hand side (not by arm)."""
        ff, floor = self.ff.value()
        return {"mode": self.mode, "stance": self.stance, "ff": ff, "ff_floor": floor,
                "hands": {cs: {"arm": a, "engaged": self.hand[a].engaged, "blocked": self.hand[a].blocked, "note": self.hand[a].note,
                               "by_hand": self.hand[a].by_hand, "caught": self.hand[a].engaged and self.hand[a].caught}
                          for cs, a in ARM_OF[self.stance].items()}}


# ---- web page / HTTPS ----
class Link:
    """Latest controller samples received from the page, and the status sent back to it."""
    def __init__(self, allow_hands=False):
        self.allow_hands = allow_hands                       # whether hand-tracking samples are accepted (--hands)
        self.lock = threading.Lock()
        self.ctl, self.state, self.frames, self.last_rx = {}, {}, 0, 0.0
        self.cmds = []                                       # ready / home requests from the buttons on the page, picked up by the main loop
        self.joints = {}                                     # {"q": measured angles, "cmd": commanded angles}, served at GET /joints (for debugging)
        self.bad = 0                                         # number of invalid samples dropped

    def put(self, ctl):
        now = time.monotonic()
        with self.lock:
            for side in ("left", "right"):
                c = clean_sample(ctl.get(side)) if isinstance(ctl, dict) else None
                if c and c.get("hand") and not self.allow_hands:
                    c = None                                 # without --hands, hand-tracking samples are ignored
                if c:
                    self.ctl[side] = {**c, "t": now}
                elif isinstance(ctl, dict) and ctl.get(side) is not None:
                    self.bad += 1                            # an invalid sample is dropped whole: that side times out after 0.25 s and releases by itself
            self.frames += 1
            self.last_rx = now
            return dict(self.state)

    def get(self):
        with self.lock:
            return {k: dict(v) for k, v in self.ctl.items()}

    def push_cmd(self, name):
        if name in ("ready", "home"):
            with self.lock:
                self.cmds.append(name)

    def pop_cmds(self):
        with self.lock:
            out, self.cmds = self.cmds, []
            return out


def host_ok(host, port):
    """True if the Host header addresses this machine: an IP literal, localhost or this machine's hostname, with the
    right port. Any other name is rejected (DNS rebinding)."""
    name, _, hp = (host or "").rpartition(":")
    if not name or hp != str(port):
        return False
    if name.lower() in ("localhost", socket.gethostname().lower(), socket.gethostname().lower() + ".local"):
        return True
    try:
        ipaddress.ip_address(name.strip("[]"))
        return True
    except ValueError:
        return False


def make_handler(link, token, port, scheme):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"                        # keep-alive: there is one POST per frame, which must not pay for a new handshake every time

        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _deny(self, code, why):
            self._send(json.dumps({"error": why}).encode(), "application/json", code)
            return False

        def _from_here(self):
            """Host / Origin check: the request addresses this machine and was sent by the page served from here."""
            host = self.headers.get("Host") or ""
            if not host_ok(host, port):
                return self._deny(403, "host")
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"{scheme}://{host}":
                return self._deny(403, "origin")
            return True

        def _authed(self):
            if not self._from_here():
                return False
            if not hmac.compare_digest((self.headers.get("X-Lucen-Token") or "").encode(), token.encode()):
                time.sleep(0.5)                              # every wrong token costs half a second (slows down guessing)
                return self._deny(403, "token")
            return True

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):                 # the page holds no token and anyone may fetch it; it reads ?k= from its address and sends it with its requests
                if self._from_here():
                    self._send(open(HTML, "rb").read(), "text/html; charset=utf-8")
            elif path in ("/state", "/joints"):
                if self._authed():
                    with link.lock:
                        body = json.dumps(link.state if path == "/state" else link.joints).encode()
                    self._send(body, "application/json")
            else:
                self._deny(404, "not found")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > BODY_MAX:
                self.close_connection = True                 # the unread body is still on the connection, so the connection cannot be reused
                self._deny(413, "too large")
                return
            raw = self.rfile.read(n) if n else b""
            if not self._authed():
                return
            if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                self._deny(415, "content-type")
                return
            try:
                data = json.loads(raw or b"{}")
            except ValueError:
                data = None
            if not isinstance(data, dict):
                self._deny(400, "json")
                return
            path = self.path.split("?", 1)[0]
            if path == "/pose":
                self._send(json.dumps(link.put(data.get("ctl"))).encode(), "application/json")
            elif path == "/cmd":
                link.push_cmd(data.get("name"))
                self._send(b'{"ok": true}', "application/json")
            else:
                self._deny(404, "not found")
    return H


def load_token(path, new=False):
    """Access token: generated randomly and saved (mode 0600) on first use, reused afterwards so that a bookmark in
    the headset keeps working."""
    if not new and os.path.exists(path):
        tok = open(path, encoding="utf-8").read().strip()
        if len(tok) >= 8 and all(ch in TOKEN_ABC for ch in tok):
            return tok
    tok = "".join(secrets.choice(TOKEN_ABC) for _ in range(8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(tok + "\n")
    return tok


class NoKeys:
    """Keyboard stand-in when there is no terminal (running in the background): it never returns a key; use the
    buttons on the page for the preset poses."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def poll(self):
        return []


def ensure_cert(cert_dir):
    """Self-signed certificate (WebXR is only available over https or on localhost). An existing one is reused. -> (cert, key)"""
    cert, key = os.path.join(cert_dir, "cert.pem"), os.path.join(cert_dir, "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        return cert, key
    os.makedirs(cert_dir, exist_ok=True)
    try:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert, "-days", "825",
                        "-subj", "/CN=lucen-vr-teleop"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as e:
        detail = e.stderr.decode(errors="replace")[-300:] if getattr(e, "stderr", None) else str(e)
        raise SystemExit(f"could not create the self-signed certificate (openssl is required): {detail}\n  Or pass your own --cert / --key; --http is enough to try it in a browser on this computer")
    os.chmod(key, 0o600)
    return cert, key


def local_ips():
    """IPv4 addresses of this machine's interfaces, the default-route interface first. The hostname is deliberately
    not resolved: the DNS of a LAN may return the addresses of every host that has the same name."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))                     # sends nothing; it only makes the kernel pick the address of the default route
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    for cmd in (["ip", "-4", "-o", "addr"], ["ifconfig"]):   # Linux / macOS
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        ips += re.findall(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", out)
        if out:
            break
    return [ip for ip in dict.fromkeys(ips) if not ip.startswith("127.")]


def serve(link, listen, port, token, cert=None, key=None):
    srv = ThreadingHTTPServer((listen, port), make_handler(link, token, port, "https" if cert else "http"))
    srv.daemon_threads = True
    if cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)   # the handshake runs in each connection's own thread
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "humanoid_arm/humanoid_arm.yaml"), help="robot config (default: humanoid_arm/humanoid_arm.yaml)")
    ap.add_argument("--host", default="127.0.0.1", help="host the bridge runs on (default: this computer)")
    ap.add_argument("--port", type=int, default=None, help="bridge port (default: bridge.port in the yaml)")
    ap.add_argument("--listen", default=None, help="address the web page is served on (default: this computer's LAN address; 127.0.0.1 with --http)")
    ap.add_argument("--web-port", type=int, default=8443, help="port of the web page served to the headset")
    ap.add_argument("--http", action="store_true", help="serve plain http (only usable from a browser on this computer at http://127.0.0.1; a headset cannot connect)")
    ap.add_argument("--cert", help="your own certificate (default: a self-signed one is created in ~/.cache/lucen_vr)")
    ap.add_argument("--key", help="private key file that belongs to --cert")
    ap.add_argument("--token-file", default=os.path.expanduser("~/.cache/lucen_vr/token"), help="file the access token is stored in (the token is the ?k= part of the address)")
    ap.add_argument("--new-token", action="store_true", help="generate a new access token (old bookmarks stop working)")
    ap.add_argument("--stance", choices=list(M_STANCE), default="behind", help="behind: you face the same way as the robot; facing: you face the robot (see the top of this file)")
    ap.add_argument("--hands", action="store_true", help="also accept bare hands (headset hand tracking; default: controllers only)")
    ap.add_argument("--scale", type=float, default=1.0, help="metres the robot's hand moves per metre of your hand")
    ap.add_argument("--safety", default="slow", help="safety profile from the yaml safety_profiles (joint speed limit)")
    ap.add_argument("--profile", default="mirror", help="gain profile the bridge runs, e.g. jog (used only for the feed-forward suggestion in the status line; does not change the bridge)")
    ap.add_argument("--no-collide", action="store_true", help="disable the collision pre-check")
    ap.add_argument("--no-ready", action="store_true", help="do not move to ready_pose automatically after start")
    ap.add_argument("--ready-delay", type=float, default=5.0, help="seconds after start before the arms move to ready_pose automatically")
    ap.add_argument("--no-range-check", action="store_true", help="take over even if a measured angle is outside its joint range")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bport = args.port or int((cfg.get("bridge") or {}).get("port", 9877))
    vel_max = float(cfg["safety_profiles"][args.safety]["vel_max"])
    vt = VrTeleop(cfg, stance=args.stance, scale=args.scale, vel_max=vel_max, collide=not args.no_collide, profile=args.profile)
    link = Link(allow_hands=args.hands)
    listen = args.listen or ("127.0.0.1" if args.http else next(iter(local_ips()), None))
    if listen is None:
        sys.exit("[vr] no LAN address found on this computer (not connected?). Pass --listen <this computer's IP on the headset's network>")
    if args.http and listen not in ("127.0.0.1", "localhost"):
        sys.exit("[vr] --http is unencrypted and only allowed on 127.0.0.1 (the token would cross the network in clear text); do not use --http with a headset")
    token = load_token(args.token_file, args.new_token)
    cert, key = (None, None) if args.http else ((args.cert, args.key) if args.cert else ensure_cert(os.path.expanduser("~/.cache/lucen_vr")))
    serve(link, listen, args.web_port, token, cert, key)
    scheme = "http" if args.http else "https"
    shown = local_ips() or ["<this computer's IP>"] if listen == "0.0.0.0" else [listen]
    print("[vr] web page (full address including the token; bookmark it in the headset): " + "  ".join(f"{scheme}://{ip}:{args.web_port}/?k={token}" for ip in shown))
    if not args.http:
        print(f"[vr] listening on {listen} only (headset cannot connect? pass --listen <this computer's address on its network>); the certificate is self-signed: Advanced -> Proceed")
    print(f"[vr] connecting to bridge {args.host}:{bport} ({os.path.basename(args.config)}), stance {args.stance}, scale {args.scale}, safety profile {args.safety} (≤ {vel_max} rad/s)")
    cli = BridgeClient(args.host, bport, vt.names, vt.limits, mode="send")
    bad = out_of_range(cli.q0, vt.limits)
    if bad and not args.no_range_check:
        cli.close()
        print(f"[vr] not taking over: these joints measure more than {RANGE_SLACK} rad outside their range - the zero offset or the direction is wrong:")
        for n, v, lo, hi in bad:
            print(f"    {n:<20} measured {v:+.2f}   range [{lo:+.2f}, {hi:+.2f}]")
        sys.exit(1)
    vt.start(cli.q0)
    print("[vr] keys: y  go to ready (elbows bent; start from here)   h  back to home (arms hanging)   x  quit (Ctrl+C = emergency stop)")
    t_ready = None if args.no_ready else time.monotonic() + args.ready_delay
    if t_ready:
        print(f"[vr] !! in {args.ready_delay:g} s both arms move to the ready pose by themselves (forearms raised in front of the body). Clear the area; press n to cancel")
    dt = 1.0 / LOOP_HZ
    last_print = 0.0
    try:
        with (Keys() if sys.stdin.isatty() else NoKeys()) as keys:
            while not cli.closed:
                tick = time.monotonic()
                if t_ready and tick >= t_ready:
                    t_ready = None
                    vt.go("ready")
                for name in link.pop_cmds():                  # buttons on the page
                    if name in ("ready", "home"):
                        vt.go(name)
                for k in keys.poll():
                    if t_ready and k == "n":
                        t_ready = None
                        print("\n[vr] not moving to ready, staying here (with the arms hanging the inverse kinematics is singular, so the hands respond poorly at first)")
                    elif k == "y":
                        vt.go("ready")
                    elif k == "h":
                        vt.go("home")
                    elif k == "x":
                        raise KeyboardInterrupt
                q = cli.latest()
                cmd = vt.tick(q, link.get(), tick, dt)
                cli.send_target(cmd)
                st = {**vt.status(), "allow_hands": args.hands}
                with link.lock:
                    link.state, link.joints = st, {"q": q, "cmd": cmd}
                    frames, age = link.frames, tick - link.last_rx
                if tick - last_print >= 0.2:
                    web = "page not connected" if not frames else (f"no data from the page for {age:.1f} s" if age > 1.0 else "page online")
                    hands = "  ".join(f"{cs} {'hand' if v['by_hand'] else 'controller'} -> {v['arm']} arm: "
                                      + ("!! would collide, holding" if v["blocked"] else "engaged" if v["engaged"] else v["note"] or ("no fist" if v["by_hand"] else "idle"))
                                      for cs, v in st["hands"].items())
                    sys.stdout.write(f"\r[vr] {'moving to preset pose' if st['mode'] == 'pose' else web}  {hands}{ff_text(st['ff'], st['ff_floor'])}" + " " * 20)
                    sys.stdout.flush()
                    last_print = tick
                time.sleep(max(0.0, dt - (time.monotonic() - tick)))
        print("\n[vr] bridge disconnected")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            q = cli.latest()
            cli.send_target({n: q.get(n, v) for n, v in vt.q_cmd.items()})      # pin the target to the measured pose before leaving
        except Exception:
            pass
        cli.close()
        print("\n[vr] exiting (after the 1.5 s heartbeat timeout the bridge goes to damping, then disables)")


if __name__ == "__main__":
    main()
