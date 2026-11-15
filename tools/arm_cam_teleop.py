#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/arm_cam_teleop.py - teleoperate the humanoid_arm with a webcam: the robot's arms copy the pose of your arms.

No headset and no controllers. A camera (the one built into a laptop is enough) looks at you, and MediaPipe finds
your shoulders, elbows, wrists and hands:
  upper arm direction  -> the two shoulder angles (hip_pitch: swing forward / back, hip_roll: raise sideways)
  forearm direction    -> the elbow angle (knee) and the rotation of the upper arm about its own axis (hip_yaw)
  hand                 -> fist = gripper closed, open hand = gripper open (two states only; a change must be held
                          for 0.2 s; while no hand is recognised the gripper keeps its position)
  wrist (link4 / 5 / 6)   stays at 0 by default (gripper in line with the forearm). --wrist makes the three joints
                          follow the orientation of your palm. This is experimental, a single camera sees palm
                          orientation poorly: validate against the simulated bridge before using it on the robot.
This is pose copying (joint angles are mapped directly), not hand-position tracking: your arms and the robot's
arms differ in length, so the gripper does not end up exactly where your hand is.

Requirements:
  pip install -r requirements-extra.txt
  This installs MediaPipe, which brings its own OpenCV build: use a separate virtual environment if an existing
  OpenCV installation must stay untouched. The first run downloads two models (pose 9 MB + hand 8 MB) from
  storage.googleapis.com/mediapipe-models to ~/.cache/lucen_cam/. macOS asks once whether the terminal may use
  the camera: allow it (System Settings -> Privacy & Security -> Camera).

Run it against the simulated bridge first:
  python tools/arm_cam_teleop.py --dry      # recognition only, no bridge: skeleton, fist and computed joint angles
  python tools/fake_bridge.py --config humanoid_arm/humanoid_arm.yaml --grav-scale 1.0 --view    # mjpython on macOS
  python tools/arm_cam_teleop.py            # in a second terminal: drives the simulated bridge
  python tools/arm_cam_teleop.py --demo     # the same without a camera: a synthetic arm motion exercises the whole chain
Then on the robot:
  python bridge.py --config humanoid_arm/humanoid_arm.yaml                    # on the robot computer
  python tools/arm_cam_teleop.py --host <robot computer> --safety safe        # first time: safe profile (0.5 rad/s)

Where to stand: face the camera, 1.5 to 2.5 m away, with your upper body (shoulders to hips) and both arms in the
picture and nobody else in the background.
  During the countdown stand still with both arms hanging naturally. The last second of the countdown measures
  the direction in which the camera sees a hanging arm (the camera is tilted and the model has a bias of its own;
  uncorrected, the robot's arms would point somewhat forward or backward while yours hang). After this
  calibration, your arms hanging = the robot's arms hanging. If your arms are not hanging or are still moving in
  that second, no calibration is made (the terminal and the preview say so): do not be walking when it ends.
  default (same side)   your left arm = the robot's left arm, reaching forward = the robot reaches forward
                        (use when you and the robot face the same way)
  --mirror              as in a mirror: your right arm = the robot's left arm (use when the robot faces you)

Keys (in the preview window or in the terminal):
  space          start / stop following. Starting begins a countdown of --countdown seconds, time to walk back
                 in front of the camera. The tool starts in the stopped state.
  y              both arms to ready_pose        h   both arms to home_pose      (both stop following first)
  x / q / Esc    quit (Ctrl+C = emergency stop)

Safety:
  - The arms do not move until you press space.
  - Joint speed is limited by the safety profile (--safety, default slow = 1 rad/s), so when following starts
    the arms swing to your pose slowly instead of jumping.
  - Leash: a commanded angle never leads the measured angle by more than LEASH.
  - Collision pre-check: a control step that would bring the arm into contact with the other arm or with itself
    is not taken.
  - If you leave the picture, a shoulder / elbow / wrist is not seen reliably, or the camera stalls, that arm
    stops in place and resumes when it is seen again.
  - An arm raised above the shoulder is out of reach for the robot's shoulder: it stops at the edge of the range.
  To stop: space. If that does not help, Ctrl+C. The last resort is the 48 V emergency stop.
  To lower the robot's arms, lower your own. Do this (or press h) before quitting: once the tool has exited, the
  bridge goes to damping and then disables after its 1.5 s heartbeat timeout, and a raised arm drops.

Jitter: every joint angle passes a One-Euro filter and then a hysteresis band of 0.02 rad, so the command does
  not move while you stand still. Still shaky: --smooth 2 (steadier, a little slower). Too slow: --smooth 0.5.

If the real arm does not track the command (it rises faster than you, does not come down, or cannot lift), the
  camera is not the cause: the gravity feed-forward of the bridge (--grav-scale) is off. Too much makes the arm
  float up, too little makes it sag. The status line shows a "feed-forward suggestion" and the preview a warning:
  add that number to the bridge's --grav-scale and restart the bridge.

Log: every run records key points, target angles, commanded angles and measured angles in
  ~/.cache/lucen_cam/last_run.jsonl (numbers only, no images; overwritten by the next run; --no-log turns it off).
  Use it to analyse a problem afterwards.

Limitations: a single camera has no true depth. Directions left / right and up / down are accurate to a few
  degrees; the direction toward the camera is estimated by the model and is often off by more than ten degrees,
  worst when an arm points straight at the lens. Good for large gestures and posing, not for precise grasping:
  use the VR controllers (arm_vr_teleop.py) or lead by hand (arm_lead_follow.py) for that.
"""
import argparse
import contextlib
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.request

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import arm_vr_teleop as avt                                  # noqa: E402
from arm_jog import BridgeClient, Keys                       # noqa: E402
from arm_lead_follow import out_of_range, RANGE_SLACK        # noqa: E402

LOOP_HZ = 50.0
CAM_STALE = 0.35       # s: no new data for an arm for this long (camera stalled / person left the picture) -> that arm stops in place
VIS_MIN = 0.5          # shoulder, elbow and wrist visibility (0..1, from MediaPipe) must all exceed this for the arm to count as seen
BONE = (0.10, 0.55)    # m: an upper arm / forearm length outside this range is a misdetection; the frame is not used
EURO_MIN = 0.8         # Hz: minimum cutoff frequency of the One-Euro filter (suppresses jitter while the arm is still)
EURO_BETA = 0.4        # the cutoff rises with speed (larger = more responsive, more jitter). --smooth 2 halves both values: steadier and slower
DEADBAND = 0.02        # rad: filtered-target motion inside a band this wide does not move the command (no jitter while you stand still; costs 1° of hysteresis)
FIST_ON, FIST_OFF = 0.62, 0.76   # curl (fingertip-to-wrist distance / finger length, four fingers): < FIST_ON = fist, > FIST_OFF = open, unchanged in between.
#                                  On MediaPipe's sample images: open hand 0.88, fist / thumbs-up 0.46, index finger only 0.58, V sign 0.73
FIST_HOLD = 0.2        # s: a fist / open hand must persist this long to count (one misread frame does not make the gripper drop what it holds)
GRIP_VEL, GRIP_LEAD, LEASH = avt.GRIP_VEL, avt.GRIP_LEAD, avt.LEASH
P_MAX = 5.0            # m: a key-point coordinate beyond this is not valid data
CAL_MAX = math.radians(30.0)     # a measured "down" further off than this means the arms were not hanging (or the camera is badly tilted): no calibration
CAL_WIN = 1.0          # s: only this last stretch before following starts is used (the seconds spent walking back from the keyboard do not count)
CAL_MIN_N = 12         # minimum number of frames in that window
CAL_STEADY = math.radians(6.0)   # standing still = every frame's direction is within this of the mean over the window
CAL_PARALLEL = 0.96    # agreement of the segment directions within one frame (upper arms / forearms of both arms; 1 = parallel); below this the arms are bent, not hanging
LOG_MAX = 30_000_000   # bytes: the log file stops growing at this size

# MediaPipe world frame (x image right, y down, z away from the camera; facing the camera, image right = your left) -> robot frame (x left, y back, z up)
M_CAM = {
    False: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),    # same side: your left = robot's left, your forward = robot's forward
    True: np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),    # mirror: your right = robot's left (left and right swapped: a reflection)
}
ARM_JOINTS = ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint")
WRIST_JOINTS = ("link4_joint", "link5_joint", "link6_joint")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {"lite": "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
          "full": "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
          "heavy": "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
          "hand": "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"}

clamp = avt.clamp


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot_between(a, b):
    """Smallest rotation that takes unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    s2 = float(v @ v)
    if s2 < 1e-12:
        return np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / s2)


class OneEuro:
    """One-Euro filter (Casiez 2012): low cutoff frequency at rest (suppresses jitter), high cutoff in fast motion (little lag)."""
    def __init__(self, min_cutoff=EURO_MIN, beta=EURO_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.t = None
        self.dx = 0.0

    @staticmethod
    def _alpha(dt, fc):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * fc * dt))

    def __call__(self, x, t):
        if self.x is None or t <= self.t:
            self.x, self.t, self.dx = float(x), t, 0.0
            return self.x
        dt = t - self.t
        self.t = t
        self.dx += self._alpha(dt, self.d_cutoff) * ((x - self.x) / dt - self.dx)
        self.x += self._alpha(dt, self.min_cutoff + self.beta * abs(self.dx)) * (x - self.x)
        return self.x


def arm_angles(sh, el, wr, prev=None):
    """Shoulder / elbow / wrist points (robot frame, m) -> target angles of the first four joints of the arm,
    {"pitch", "roll", "yaw", "knee"} (rad, not yet clamped to the joint range).
    The robot's shoulder is pitch (about x) -> roll (about y) -> yaw (about z, along the upper arm); the elbow (knee)
    turns about x; all zeros = arm hanging straight down.
      upper arm direction u = Rx(pitch)·Ry(roll)·(0, 0, -1) = (-sin roll, sin pitch·cos roll, -cos pitch·cos roll)
      forearm direction   f = Rx(pitch)·Ry(roll)·Rz(yaw)·(0, sin knee, -cos knee)
    prev is the result of the previous frame. It fills in where the solution is undetermined: pitch when the arm is
    raised sideways to horizontal, yaw when the elbow is straight; the closer to these poses, the more prev counts.
    Returns None when the bone lengths are implausible (misdetection)."""
    prev = prev or {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "knee": 0.0}
    u, f = np.asarray(el, dtype=float) - np.asarray(sh, dtype=float), np.asarray(wr, dtype=float) - np.asarray(el, dtype=float)
    lu, lf = float(np.linalg.norm(u)), float(np.linalg.norm(f))
    if not (BONE[0] < lu < BONE[1] and BONE[0] < lf < BONE[1]):
        return None
    u, f = u / lu, f / lf
    uc = np.array([u[0], u[1], min(u[2], 0.0)])              # upper arm above horizontal: out of reach for the robot's shoulder, treated as horizontal (also avoids the ±180° jump)
    n = float(np.linalg.norm(uc))
    if n < 1e-6:                                             # straight up: the direction is undetermined, keep the previous frame's
        roll, pitch = prev["roll"], prev["pitch"]
    else:
        uc /= n
        roll = -math.asin(clamp(float(uc[0]), -1.0, 1.0))
        c = math.hypot(uc[1], uc[2])                         # = cos(roll); goes to 0 when the arm is raised sideways, where pitch is undetermined
        w = clamp((c - 0.15) / 0.30, 0.0, 1.0)
        pitch = prev["pitch"] + w * (math.atan2(uc[1], -uc[2]) - prev["pitch"])
    knee = -math.acos(clamp(float(u @ f), -1.0, 1.0))
    g = (rot_x(pitch) @ rot_y(roll)).T @ f                   # forearm direction in the upper-arm frame = (-sin yaw·sin knee, cos yaw·sin knee, -cos knee)
    s = math.hypot(g[0], g[1])                               # = |sin knee|; goes to 0 when the elbow is straight, where yaw is undetermined
    w = clamp((s - 0.20) / 0.30, 0.0, 1.0)
    yaw = clamp(prev["yaw"] + w * wrap(math.atan2(g[0], -g[1]) - prev["yaw"]), -2.0, 2.0)
    return {"pitch": pitch, "roll": roll, "yaw": yaw, "knee": knee}


def forearm_frame(a):
    """Four angles -> orientation of the forearm frame in the robot frame (its -z axis points along the forearm)."""
    return rot_x(a["pitch"]) @ rot_y(a["roll"]) @ rot_z(a["yaw"]) @ rot_x(a["knee"])


def hand_frame(h):
    """21 hand points (MediaPipe order, robot frame) -> orientation the gripper should have, in the convention of the
    model's ee frame: -z is the pointing direction, x the direction in which the two fingers open.
    Pointing direction = wrist -> base of the middle finger; opening direction = palm normal (the direction in which
    the thumb opposes the fingers). Arm hanging, palm inward, thumb forward = identity (the zero pose of the model)."""
    h = np.asarray(h, dtype=float)
    d, s = h[9] - h[0], h[5] - h[17]
    d = d / (np.linalg.norm(d) or 1.0)
    n = np.cross(d, s)
    if np.linalg.norm(n) < 1e-9:
        return None
    x = -n / np.linalg.norm(n)
    z = -d
    return np.column_stack([x, np.cross(z, x), z])


def wrist_angles(R_fore, R_hand):
    """Forearm orientation + desired gripper orientation -> (link4, link5, link6). The three wrist axes are -x, +y, -z
    in this order: R_fore^T·R_hand = Rx(-a4)·Ry(a5)·Rz(-a6).
    A two-finger gripper looks the same after a 180° turn, so link6 takes the solution nearer to 0."""
    R = R_fore.T @ R_hand
    a5 = math.asin(clamp(float(R[0, 2]), -1.0, 1.0))
    a4 = -math.atan2(-R[1, 2], R[2, 2])
    a6 = -math.atan2(-R[0, 1], R[0, 0])
    if abs(a6) > math.pi / 2:
        a6 -= math.copysign(math.pi, a6)
    return a4, a5, a6


def fist_curl(h):
    """21 hand points -> fist curl: mean over index / middle / ring / little finger of (fingertip-to-wrist distance /
    length along the bones from wrist to fingertip). Straight ≈ 1, fist ≈ 0.45."""
    h = np.asarray(h, dtype=float)
    out = []
    for b in (5, 9, 13, 17):
        chain = np.linalg.norm(h[b] - h[0]) + sum(np.linalg.norm(h[b + k + 1] - h[b + k]) for k in range(3))
        out.append(float(np.linalg.norm(h[b + 3] - h[0]) / max(chain, 1e-9)))
    return sum(out) / len(out)


def clean_obs(o):
    """Validate the data of one arm from the camera side; returns None if it is malformed (missing field, non-finite, implausible)."""
    try:
        if not isinstance(o, dict):
            return None
        pts = {k: [float(v) for v in o[k]] for k in ("sh", "el", "wr")}
        vis, t = float(o["vis"]), float(o["t"])
        hand = o.get("hand")
        if hand is not None:
            hand = [[float(v) for v in p] for p in hand]
            if len(hand) != 21 or any(len(p) != 3 for p in hand) or not all(math.isfinite(v) and abs(v) < P_MAX for p in hand for v in p):
                hand = None
        hand_t = float(o.get("hand_t", t))
    except (TypeError, ValueError, KeyError):
        return None
    flat = [v for p in pts.values() for v in p]
    if any(len(p) != 3 for p in pts.values()) or not all(math.isfinite(v) and abs(v) < P_MAX for v in flat):
        return None
    if not all(math.isfinite(v) for v in (vis, t, hand_t)):
        return None
    return {**pts, "vis": vis, "t": t, "hand": hand, "hand_t": hand_t}


class CamArm:
    """Following state of one arm."""
    def __init__(self):
        self.state = "paused"                                # paused / following / lost / blocked
        self.raw = None                                      # arm_angles result of the previous frame
        self.euro = {}                                       # joint name -> OneEuro
        self.tgt = {}                                        # joint name -> filtered target angle, clamped to the joint range
        self.last_t = self.hand_t = None
        self.fist = None                                     # True fist / False open / None not recognised yet
        self.want, self.want_t = None, 0.0
        self.curl = None
        self.cal_t = None

    def forget(self):
        """Stopped or lost: clear the filters. When the arm is seen again, filtering restarts from the pose at that
        moment (the command is speed-limited, so it does not jump)."""
        self.raw, self.euro, self.tgt, self.last_t = None, {}, {}, None


class CamTeleop(avt.VrTeleop):
    """Teleoperation logic without camera or network I/O: tick(measured q, key points of both arms, current time, dt)
    -> commanded angles of the 16 joints.
    Kinematics, collision pre-check, preset poses, leash and the non-finite guard come from VrTeleop; only the
    per-arm control step is replaced here."""

    def __init__(self, cfg, vel_max=1.0, collide=True, wrist=False, smooth=1.0, profile="mirror"):
        super().__init__(cfg, stance="behind", vel_max=vel_max, collide=collide, profile=profile)
        k = max(float(smooth), 0.05)
        self.euro_kw = {"min_cutoff": EURO_MIN / k, "beta": EURO_BETA / k}
        need = [p + j for p in self.prefix.values() for j in ARM_JOINTS + WRIST_JOINTS]
        missing = [n for n in need if n not in self.limits]
        if missing:
            raise SystemExit(f"these joints are missing from the config; camera teleoperation supports the humanoid_arm only: {missing}")
        self.wrist = wrist
        self.follow = False
        self.cam = {s: CamArm() for s in self.prefix}
        self.R_cal, self.cal_deg, self.cal_note = np.eye(3), None, ""    # camera-tilt correction (measured during the countdown, with both arms hanging)
        self._cal = None

    def cal_begin(self):
        """Countdown started: from now until following starts, record the upper-arm / forearm directions seen in every
        frame (the person should end up standing with both arms hanging)."""
        self._cal = {}                                       # frame time -> unit directions of the segments in that frame

    def _cal_add(self, c):
        segs = self._cal.setdefault(c["t"], [])
        for a, b in ((c["sh"], c["el"]), (c["el"], c["wr"])):
            v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
            n = float(np.linalg.norm(v))
            if BONE[0] < n < BONE[1]:
                segs.append(v / n)

    def _cal_end(self):
        """Countdown finished: the mean arm direction over the last CAL_WIN seconds is the direction in which the camera
        sees a hanging arm; the rotation that takes it to true down (0, 0, -1) is the correction.
        During that window the arms must be hanging (segments parallel) and still (directions agree across frames),
        otherwise no calibration is made."""
        frames, self._cal = self._cal, None
        keep = "" if self.cal_deg is None else f", keeping the previous {self.cal_deg:.0f}°"
        t_end = max(frames) if frames else 0.0
        dirs = []
        for t, segs in frames.items():
            if t_end - t <= CAL_WIN and segs:
                m = np.mean(segs, axis=0)
                dirs.append((m / (np.linalg.norm(m) or 1.0), float(np.linalg.norm(m))))
        if len(dirs) < CAL_MIN_N:
            self.cal_note = "arms not detected in the last second of the countdown, not calibrated" + keep
            return
        d = np.mean([u for u, _ in dirs], axis=0)
        d /= np.linalg.norm(d) or 1.0
        down = np.array([0.0, 0.0, -1.0])
        ang = math.acos(clamp(float(d @ down), -1.0, 1.0))
        if min(L for _, L in dirs) < CAL_PARALLEL or ang > CAL_MAX:
            self.cal_note = "arms not hanging in the last second of the countdown, not calibrated" + keep
            return
        if max(math.acos(clamp(float(u @ d), -1.0, 1.0)) for u, _ in dirs) > CAL_STEADY:
            self.cal_note = "arms still moving in the last second of the countdown, not calibrated" + keep
            return
        self.R_cal, self.cal_deg = rot_between(d, down), math.degrees(ang)
        self.cal_note = f"arms-down direction corrected by {self.cal_deg:.0f}°"

    def set_follow(self, on):
        self.follow = bool(on) and self.mode == "hold"
        if self.follow and self._cal is not None:
            self._cal_end()
        if not self.follow:
            self._cal = None
            for a in self.cam.values():
                a.forget()
                a.state = "paused"

    def go(self, pose_name):
        self.set_follow(False)
        super().go(pose_name)

    def _aim(self, a, n, v, t):
        """New target of one joint: One-Euro filter, then a hysteresis band DEADBAND wide."""
        if n not in a.euro:
            a.euro[n] = OneEuro(**self.euro_kw)
            a.tgt[n] = a.euro[n](v, t)
            return
        f = a.euro[n](v, t)
        a.tgt[n] += max(f - a.tgt[n] - DEADBAND, 0.0) + min(f - a.tgt[n] + DEADBAND, 0.0)

    def _hand_tick(self, side, c, q, now, dt):
        a, h, pre = self.cam[side], self.hand[side], self.prefix[side]
        c = clean_obs(c)
        seen = c is not None and now - c["t"] <= CAM_STALE and c["vis"] >= VIS_MIN
        if not self.follow or self.mode != "hold":
            if self._cal is not None and seen and c["t"] != a.cal_t:      # during the countdown: collect the directions for the calibration
                a.cal_t = c["t"]
                self._cal_add(c)
            a.state, h.blocked = "paused", False
            return
        if not seen:
            a.state, h.blocked = "lost", False
            a.forget()
            return
        Rc = self.R_cal
        c = {**c, "sh": Rc @ np.asarray(c["sh"]), "el": Rc @ np.asarray(c["el"]), "wr": Rc @ np.asarray(c["wr"]),
             "hand": None if c["hand"] is None else (np.asarray(c["hand"]) @ Rc.T)}
        if c["t"] != a.last_t:                               # a new camera frame has arrived
            raw = arm_angles(c["sh"], c["el"], c["wr"], a.raw)
            if raw is None:
                a.state, h.blocked = "lost", False
                a.forget()
                return
            a.raw, a.last_t = raw, c["t"]
            want = {pre + j: raw[k] for j, k in zip(ARM_JOINTS, ("pitch", "roll", "yaw", "knee"))}
            for n, v in want.items():                        # clamp to the joint range before filtering, so the filter does not wind up when your arm goes where the robot cannot
                self._aim(a, n, clamp(v, *self.limits[n]), c["t"])
            for j in WRIST_JOINTS:
                a.tgt.setdefault(pre + j, self.rest.get(pre + j, 0.0))
        if c["hand"] is not None and c["hand_t"] != a.hand_t and now - c["hand_t"] < CAM_STALE:    # new hand result
            a.hand_t = c["hand_t"]
            a.curl = fist_curl(c["hand"])
            want = True if a.curl < FIST_ON else (False if a.curl > FIST_OFF else a.fist)
            if want == a.fist:
                a.want = None
            elif want != a.want:
                a.want, a.want_t = want, c["hand_t"]
            elif c["hand_t"] - a.want_t >= FIST_HOLD - 1e-9:
                a.fist, a.want = want, None
            if self.wrist and a.raw is not None:
                Rh = hand_frame(c["hand"])
                if Rh is not None:
                    now_a = {k: clamp(a.tgt[pre + j], *self.limits[pre + j]) for j, k in zip(ARM_JOINTS, ("pitch", "roll", "yaw", "knee"))}
                    for j, v in zip(WRIST_JOINTS, wrist_angles(forearm_frame(now_a), Rh)):
                        self._aim(a, pre + j, clamp(v, *self.limits[pre + j]), c["hand_t"])
        if not a.tgt:
            return
        step = self.vel_max * dt
        cand = {n: clamp(self.q_cmd[n] + clamp(v - self.q_cmd[n], -step, step), *self.limits[n]) for n, v in a.tgt.items()}
        if not all(math.isfinite(v) for v in cand.values()):
            a.state = "lost"
            a.forget()
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
        if g and g in self.q_cmd and a.fist is not None:
            lo, hi = self.limits[g]
            des = hi if a.fist else lo
            v = self.q_cmd[g] + clamp(des - self.q_cmd[g], -GRIP_VEL * dt, GRIP_VEL * dt)
            if g in q:
                v = clamp(v, q[g] - GRIP_LEAD, q[g] + GRIP_LEAD)
            self.q_cmd[g] = clamp(v, lo, hi)
        a.state = "blocked" if h.blocked else "following"

    def status(self):
        ff, floor = self.ff.value()
        return {"mode": self.mode, "follow": self.follow, "cal_deg": self.cal_deg, "cal_note": self.cal_note, "ff": ff, "ff_floor": floor,
                "arms": {s: {"state": a.state, "fist": a.fist, "curl": a.curl, "raw": a.raw, "tgt": dict(a.tgt)} for s, a in self.cam.items()}}


# ---- synthetic person (for --demo): shoulder / elbow / wrist points and a hand built from the four angles ----
def arm_points(a, upper=0.30, fore=0.26, sh=(0.0, 0.0, 0.0)):
    """Four angles -> shoulder / elbow / wrist points (robot frame); the inverse of arm_angles."""
    sh = np.asarray(sh, dtype=float)
    el = sh + upper * (rot_x(a["pitch"]) @ rot_y(a["roll"]) @ np.array([0.0, 0.0, -1.0]))
    wr = el + fore * (forearm_frame(a) @ np.array([0.0, 0.0, -1.0]))
    return sh, el, wr


def synth_hand(R=None, fist=False, origin=(0.0, 0.0, 0.0)):
    """A synthetic hand (21 points, MediaPipe order) with orientation R in the gripper convention (-z pointing
    direction, x opening direction); fist=True curls the four fingers into the palm."""
    R = np.eye(3) if R is None else R
    d, s = -R[:, 2], -R[:, 1]                                # pointing direction; direction little finger -> index finger
    n = -R[:, 0]                                             # palm normal
    pts = [np.zeros(3)]
    pts += [d * 0.03 + s * 0.03, d * 0.06 + s * 0.05, d * 0.08 + s * 0.07, d * 0.10 + s * 0.08]      # thumb 1-4
    for k, off in enumerate((0.022, 0.0, -0.020, -0.038)):   # index 5-8, middle 9-12 (along the pointing direction), ring 13-16, little finger 17-20
        base = d * (0.09 - 0.004 * (k - 1) ** 2) + s * off
        seg, ang, p = (0.045, 0.028, 0.022), ((80, 100, 60) if fist else (5, 5, 5)), base
        pts.append(base)
        tot = 0.0
        for L, a in zip(seg, ang):
            tot += math.radians(a)
            p = p + L * (math.cos(tot) * d + math.sin(tot) * n)
            pts.append(p)
    return (np.array(pts) + np.asarray(origin, dtype=float)).tolist()


class DemoSource:
    """No camera: a synthetic person whose left arm swings slowly while the right arm stays raised; the left hand
    alternates between open and fist every 3 s. read() -> (None, obs)"""
    def __init__(self):
        self.t0 = time.monotonic()

    def read(self):
        time.sleep(1 / 30)
        now = time.monotonic()
        t = now - self.t0
        pose = {"left": {"pitch": -0.6 - 0.3 * math.sin(0.8 * t), "roll": -0.3, "yaw": 0.2, "knee": -1.0 + 0.4 * math.sin(0.5 * t)},
                "right": {"pitch": -0.3, "roll": 0.2, "yaw": 0.0, "knee": -1.2}}
        obs = {}
        for side, a in pose.items():
            sh, el, wr = arm_points(a)
            obs[side] = {"sh": sh.tolist(), "el": el.tolist(), "wr": wr.tolist(), "vis": 1.0, "t": now,
                         "hand": synth_hand(fist=(side == "left" and int(t / 3.0) % 2 == 1)), "hand_t": now}
        return None, obs

    def close(self):
        pass


# ---- camera + MediaPipe ----
@contextlib.contextmanager
def quiet_stderr():
    """Send stderr to /dev/null for the duration of the block. The C++ side of MediaPipe prints dozens of INFO /
    WARNING lines while loading a model and during the first inferences, which would push the status line away.
    (Python errors are not lost: an exception is printed after the block has exited.)"""
    sys.stderr.flush()
    saved, null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(null)


def ensure_models(model_dir, names):
    """Download the model files that are missing (from Google's official MediaPipe model storage) -> {name: path}"""
    os.makedirs(model_dir, exist_ok=True)
    out = {}
    for name in names:
        path = os.path.join(model_dir, os.path.basename(MODELS[name]))
        if not os.path.exists(path) or os.path.getsize(path) < 100_000:
            url = MODEL_URL + MODELS[name]
            print(f"[cam] downloading model {os.path.basename(path)} <- {url}")
            try:
                urllib.request.urlretrieve(url, path + ".part")
            except OSError as e:
                raise SystemExit(f"[cam] could not download the model ({e}). You can also download this file by hand and put it in {model_dir}/:\n  {url}")
            os.replace(path + ".part", path)
            print(f"[cam]   {os.path.getsize(path) / 1e6:.1f} MB -> {path}")
        out[name] = path
    return out


class Tracker:
    """One image -> key points of both arms (in the robot frame, already assigned to the robot's left / right arm)
    plus the pixel coordinates needed to draw the preview."""
    IDX = {"left": (11, 13, 15, 17, 19, 21), "right": (12, 14, 16, 18, 20, 22)}      # person's left / right: shoulder, elbow, wrist, little-finger base, index-finger base, thumb

    def __init__(self, model_dir, pose_model="full", mirror=False, hands=True):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise SystemExit(f"[cam] missing package {e.name}. Install the extra requirements (see the top of this file):\n"
                             '  pip install -r requirements-extra.txt\n'
                             "  MediaPipe brings its own OpenCV build; use a separate virtual environment to keep an existing one")
        self.cv2, self.mp = cv2, mp
        paths = ensure_models(model_dir, [pose_model] + (["hand"] if hands else []))
        with quiet_stderr():
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=paths[pose_model]), running_mode=vision.RunningMode.VIDEO, num_poses=1))
            self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=paths["hand"]), running_mode=vision.RunningMode.IMAGE, num_hands=1,
                min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4)) if hands else None
        self.warm = [3, 3]                                   # the first few pose / hand inferences still print warnings: silence those too
        self.mirror, self.M = mirror, M_CAM[mirror]
        self.turn = 0                                        # only one hand is processed per frame (the two take turns), so pose tracking keeps the camera frame rate
        self.last_hand = {}                                  # robot arm -> (21 points, time, pixel coordinates)
        self.last_ms = -1

    def _image(self, rgb):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def process(self, bgr, t):
        """-> (obs, draw). obs: {"left" / "right" (robot arm): {"sh", "el", "wr", "vis", "t", "hand", "hand_t"}}"""
        cv2 = self.cv2
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ms = max(int(t * 1000), self.last_ms + 1)            # VIDEO mode requires strictly increasing timestamps
        self.last_ms = ms
        with (quiet_stderr() if self.warm[0] > 0 else contextlib.nullcontext()):
            res = self.pose.detect_for_video(self._image(rgb), ms)
        self.warm[0] -= 1
        obs, draw = {}, {"arms": {}, "boxes": {}, "hands": {}}
        if not res.pose_landmarks:
            return obs, draw
        lm, wl = res.pose_landmarks[0], res.pose_world_landmarks[0]
        px = lambda i: np.array([lm[i].x * W, lm[i].y * H])   # noqa: E731
        sw = float(np.linalg.norm(px(11) - px(12)))          # shoulder width in pixels: the size of the hand crop follows it
        seen = []
        for hside, idx in self.IDX.items():
            rside = ({"left": "right", "right": "left"}[hside]) if self.mirror else hside
            s, e, w = idx[:3]
            vis = min(float(lm[i].visibility if lm[i].visibility is not None else 0.0) for i in (s, e, w))
            pts = [self.M @ np.array([wl[i].x, wl[i].y, wl[i].z]) for i in (s, e, w)]
            obs[rside] = {"sh": pts[0].tolist(), "el": pts[1].tolist(), "wr": pts[2].tolist(), "vis": vis, "t": t, "hand": None, "hand_t": t}
            draw["arms"][rside] = {"px": [px(i).tolist() for i in (s, e, w)], "vis": vis}
            if vis >= VIS_MIN:
                seen.append((hside, rside, idx))
        if self.hand is not None and seen:
            hside, rside, idx = seen[self.turn % len(seen)]
            self.turn += 1
            ctr = (px(idx[3]) + px(idx[4])) / 2.0                                                # midpoint between the little-finger and index-finger bases ≈ centre of the palm
            size = int(max(4.5 * float(np.linalg.norm(ctr - px(idx[2]))), 1.1 * sw, 96.0))       # box ≈ two hand lengths (from the shoulder width when the hand points at the camera)
            x0, y0 = int(ctr[0] - size / 2), int(ctr[1] - size / 2)
            crop = np.zeros((size, size, 3), dtype=np.uint8)                                      # the part outside the image is padded with black
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, W), min(y0 + size, H)
            if xb > xa and yb > ya:
                crop[ya - y0:yb - y0, xa - x0:xb - x0] = rgb[ya:yb, xa:xb]
                with (quiet_stderr() if self.warm[1] > 0 else contextlib.nullcontext()):
                    hres = self.hand.detect(self._image(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)))
                self.warm[1] -= 1
                draw["boxes"][rside] = [x0, y0, size, bool(hres.hand_world_landmarks)]
                if hres.hand_world_landmarks:
                    pts = [(self.M @ np.array([p.x, p.y, p.z])).tolist() for p in hres.hand_world_landmarks[0]]
                    self.last_hand[rside] = (pts, t, [[x0 + p.x * size, y0 + p.y * size] for p in hres.hand_landmarks[0]])
        for rside, (pts, th, hpx) in self.last_hand.items():
            if rside in obs and t - th < CAM_STALE:
                obs[rside]["hand"], obs[rside]["hand_t"] = pts, th
                draw["hands"][rside] = hpx
        return obs, draw


class CameraSource:
    """Camera (or a video file) + Tracker. read() -> (frame, obs, draw), or None when no frame can be read."""
    def __init__(self, tracker, camera=0, video=None, width=1280, height=720):
        cv2 = tracker.cv2
        self.tracker, self.cv2, self.video = tracker, cv2, video
        self.cap = cv2.VideoCapture(video) if video else cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise SystemExit(f"[cam] cannot open {'video ' + video if video else f'camera {camera}'}." + ("" if video else
                             " macOS: allow this terminal under System Settings -> Privacy & Security -> Camera, then reopen the terminal; with several cameras try --camera 1"))
        if not video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.dt = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if video else 0.0
        self.next_t = time.monotonic()

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if self.video:                                       # a video file is played at its own frame rate
            time.sleep(max(0.0, self.next_t - time.monotonic()))
            self.next_t = max(self.next_t + self.dt, time.monotonic())
        t = time.monotonic()
        obs, draw = self.tracker.process(frame, t)
        return frame, obs, draw

    def close(self):
        self.cap.release()


COLORS = {"following": (90, 220, 110), "blocked": (60, 60, 240), "lost": (40, 160, 250), "paused": (170, 170, 170)}   # BGR


def draw_overlay(cv2, frame, draw, st, info, flip=True):
    """Draw the skeletons of both arms, the hand boxes and the status text on the frame -> image to display.
    flip: show it mirrored left-right (like a mirror, which is easier to follow)."""
    img = cv2.flip(frame, 1) if flip else frame.copy()
    W = img.shape[1]
    fx = (lambda x: W - 1 - x) if flip else (lambda x: x)
    arms = (st or {}).get("arms", {})
    for side, a in draw.get("arms", {}).items():
        col = COLORS.get(arms.get(side, {}).get("state", "paused"), COLORS["paused"]) if a["vis"] >= VIS_MIN else (90, 90, 90)
        p = [(int(fx(x)), int(y)) for x, y in a["px"]]
        cv2.line(img, p[0], p[1], col, 6, cv2.LINE_AA)
        cv2.line(img, p[1], p[2], col, 6, cv2.LINE_AA)
        for q in p:
            cv2.circle(img, q, 9, (255, 255, 255), -1, cv2.LINE_AA)
        fist = arms.get(side, {}).get("fist")
        label = f"robot {'L' if side == 'left' else 'R'}" + ("" if fist is None else (" FIST" if fist else " OPEN"))
        cv2.putText(img, label, (p[2][0] + 14, p[2][1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    for side, (x0, y0, size, found) in draw.get("boxes", {}).items():
        xa, xb = sorted((int(fx(x0)), int(fx(x0 + size))))
        cv2.rectangle(img, (xa, int(y0)), (xb, int(y0 + size)), (255, 255, 255) if found else (90, 90, 90), 1)
    for side, pts in draw.get("hands", {}).items():
        for x, y in pts:
            cv2.circle(img, (int(fx(x)), int(y)), 3, (255, 220, 60), -1, cv2.LINE_AA)
    head, col = info.get("head", ""), info.get("color", (255, 255, 255))
    cv2.rectangle(img, (0, 0), (W, 86), (0, 0, 0), -1)
    cv2.putText(img, head, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
    cv2.putText(img, info.get("sub", ""), (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    if info.get("warn"):
        cv2.rectangle(img, (0, 86), (W, 126), (40, 40, 200), -1)
        cv2.putText(img, info["warn"], (16, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


class RunLog:
    """Log of this run (JSON lines): the key points of every frame, and measured angles / commanded angles / status
    every 0.1 s. Numbers only. Use it to analyse a problem afterwards."""
    def __init__(self, path, meta):
        self.lock, self.size, self.fh = threading.Lock(), 0, None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.fh = open(path, "w", encoding="utf-8")
            self.write({"k": "meta", **meta})

    @staticmethod
    def _round(v):
        if isinstance(v, float):
            return round(v, 4) if math.isfinite(v) else None
        if isinstance(v, dict):
            return {k: RunLog._round(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [RunLog._round(x) for x in v]
        if isinstance(v, np.ndarray):
            return RunLog._round(v.tolist())
        return v

    def write(self, obj):
        if self.fh is None or self.size > LOG_MAX:
            return
        line = json.dumps(self._round(obj), ensure_ascii=False) + "\n"
        with self.lock:
            self.fh.write(line)
            self.size += len(line)

    def close(self):
        if self.fh is not None:
            with self.lock:
                self.fh.close()
                self.fh = None


class DryClient:
    """--dry: no bridge. Pretends that the measured angles equal the previous command, only so that the logic runs
    and the computed joint angles can be inspected."""
    def __init__(self, names, start):
        self.q0 = {n: float(start.get(n, 0.0)) for n in names}
        self.q, self.closed = dict(self.q0), False

    def latest(self):
        return dict(self.q)

    def send_target(self, cmd):
        self.q.update(cmd)

    def close(self):
        self.closed = True


def main():
    ap = argparse.ArgumentParser(description="Webcam teleoperation of the humanoid_arm (see the top of this file)")
    ap.add_argument("--config", default=os.path.join(ROOT, "humanoid_arm/humanoid_arm.yaml"), help="robot config (default: humanoid_arm/humanoid_arm.yaml)")
    ap.add_argument("--host", default="127.0.0.1", help="host the bridge runs on (default: this computer)")
    ap.add_argument("--port", type=int, default=None, help="bridge port (default: bridge.port in the yaml)")
    ap.add_argument("--camera", type=int, default=0, help="camera index (with a phone or an external camera attached, 0 is not necessarily the built-in one)")
    ap.add_argument("--video", help="read a video file instead of the camera")
    ap.add_argument("--demo", action="store_true", help="no camera: use a synthetic arm motion (does not need mediapipe)")
    ap.add_argument("--dry", action="store_true", help="do not connect to a bridge; only show the recognition and the computed joint angles")
    ap.add_argument("--mirror", action="store_true", help="as in a mirror: your right arm = the robot's left arm (use when the robot faces you)")
    ap.add_argument("--wrist", action="store_true", help="the three wrist joints follow the orientation of your palm (experimental, inaccurate)")
    ap.add_argument("--no-hands", action="store_true", help="do not track the hands (the grippers do not move)")
    ap.add_argument("--pose-model", choices=("lite", "full", "heavy"), default="full", help="pose model: heavy is more accurate and slower")
    ap.add_argument("--model-dir", default=os.path.expanduser("~/.cache/lucen_cam"), help="directory where the MediaPipe model files are stored (downloaded on first use)")
    ap.add_argument("--smooth", type=float, default=1.0, help="filter strength: 2 = steadier but slower, 0.5 = more responsive but shakier")
    ap.add_argument("--safety", default="slow", help="safety profile from the yaml safety_profiles (joint speed limit)")
    ap.add_argument("--profile", default="mirror", help="gain profile the bridge runs, e.g. jog (used only for the feed-forward suggestion in the status line; does not change the bridge)")
    ap.add_argument("--no-collide", action="store_true", help="disable the collision pre-check")
    ap.add_argument("--countdown", type=float, default=5.0, help="seconds from pressing space to the start of following (time to walk back to the camera and stand still, arms hanging)")
    ap.add_argument("--log", default=os.path.expanduser("~/.cache/lucen_cam/last_run.jsonl"), help="file the log of this run is written to (numbers only, no images)")
    ap.add_argument("--no-log", action="store_true", help="do not write a log")
    ap.add_argument("--auto-start", action="store_true", help="start following without waiting for the space key (the countdown still applies)")
    ap.add_argument("--no-view", action="store_true", help="do not open the preview window")
    ap.add_argument("--no-flip", action="store_true", help="do not mirror the preview left-right")
    ap.add_argument("--no-range-check", action="store_true", help="take over even if a measured angle is outside its joint range")
    ap.add_argument("--state-file", help=argparse.SUPPRESS)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bport = args.port or int((cfg.get("bridge") or {}).get("port", 9877))
    vel_max = float(cfg["safety_profiles"][args.safety]["vel_max"])
    ct = CamTeleop(cfg, vel_max=vel_max, collide=not args.no_collide, wrist=args.wrist, smooth=args.smooth, profile=args.profile)
    if args.demo:                                            # set up the camera / models first; if that fails, do not connect to the bridge at all
        src, cv2 = DemoSource(), None
    else:
        tracker = Tracker(args.model_dir, args.pose_model, mirror=args.mirror, hands=not args.no_hands)
        src, cv2 = CameraSource(tracker, args.camera, args.video), tracker.cv2
    view = cv2 is not None and not args.no_view
    if args.dry:
        cli = DryClient(ct.names, cfg.get("home_pose") or {})
        print("[cam] --dry: no bridge, recognition only")
    else:
        print(f"[cam] connecting to bridge {args.host}:{bport} ({os.path.basename(args.config)}), safety profile {args.safety} (≤ {vel_max} rad/s), "
              + ("mirror (your right arm = the robot's left arm)" if args.mirror else "same side (your left arm = the robot's left arm)"))
        cli = BridgeClient(args.host, bport, ct.names, ct.limits, mode="send")
        bad = out_of_range(cli.q0, ct.limits)
        if bad and not args.no_range_check:
            cli.close()
            print(f"[cam] not taking over: these joints measure more than {RANGE_SLACK} rad outside their range - the zero offset or the direction is wrong:")
            for n, v, lo, hi in bad:
                print(f"    {n:<20} measured {v:+.2f}   range [{lo:+.2f}, {hi:+.2f}]")
            sys.exit(1)
    ct.start(cli.q0)
    log = RunLog(None if args.no_log else args.log, {"argv": sys.argv[1:], "t0": time.monotonic(), "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                      "vel_max": vel_max, "q0": cli.q0})
    print("[cam] keys (in the preview window or the terminal): space  start / stop following   y  go to ready   h  back to home   x  quit (Ctrl+C = emergency stop)")
    print("[cam] stopped for now. Press space, then stand facing the camera (shoulders to hips and both arms in the picture) and let both arms hang during the countdown")

    lock, keys_q = threading.Lock(), queue.Queue()
    shared = {"obs": {}, "st": ct.status(), "start_at": None, "run": True, "fps": 0.0}
    if args.auto_start:
        shared["start_at"] = time.monotonic() + args.countdown
        ct.cal_begin()

    def control():
        dt, last_print, last_log = 1.0 / LOOP_HZ, 0.0, 0.0
        while shared["run"] and not cli.closed:
            tick = time.monotonic()
            while not keys_q.empty():
                k = keys_q.get()
                if k == " ":
                    if ct.follow or shared["start_at"]:
                        ct.set_follow(False)
                        shared["start_at"] = None
                    elif ct.mode == "hold":
                        shared["start_at"] = tick + args.countdown
                        ct.cal_begin()                       # the camera tilt is measured during the countdown (the person stands with both arms hanging)
                elif k in ("y", "h"):
                    shared["start_at"] = None
                    ct.go("ready" if k == "y" else "home")
                elif k in ("x", "q", "\x1b"):
                    shared["run"] = False
            if shared["start_at"] and tick >= shared["start_at"]:
                shared["start_at"] = None
                ct.set_follow(True)
                sys.stdout.write(f"\r[cam] following started. {ct.cal_note}" + " " * 60 + "\n")
            with lock:
                obs = shared["obs"]
            q = cli.latest()
            cmd = ct.tick(q, obs, tick, dt)
            cli.send_target(cmd)
            st = ct.status()
            with lock:
                shared["st"], shared["cmd"] = st, cmd
            if tick - last_log >= 0.1:
                log.write({"k": "c", "t": tick, "q": q, "cmd": cmd, "st": st})
                last_log = tick
            if tick - last_print >= 0.2:
                if args.state_file:
                    with open(args.state_file + ".tmp", "w", encoding="utf-8") as fh:
                        json.dump({"q": q, "cmd": cmd, "st": st, "t": tick}, fh)
                    os.replace(args.state_file + ".tmp", args.state_file)
                word = {"paused": "stopped", "following": "following", "lost": "not detected, holding", "blocked": "!! would collide, holding"}
                arms = "  ".join(f"{s} arm: {word[a['state']]}" + ("" if a["fist"] is None else (" (fist)" if a["fist"] else " (open)")) for s, a in st["arms"].items())
                head = "moving to preset pose" if st["mode"] == "pose" else (f"starting in {math.ceil(shared['start_at'] - tick)} s: stand still, arms hanging" if shared["start_at"] else ("following" if st["follow"] else "stopped (space to start)"))
                sys.stdout.write(f"\r[cam] {head}  {arms}  {shared['fps']:.0f} fps{avt.ff_text(st['ff'], st['ff_floor'])}" + " " * 16)
                sys.stdout.flush()
                last_print = tick
            time.sleep(max(0.0, dt - (time.monotonic() - tick)))
        shared["run"] = False

    th = threading.Thread(target=control, daemon=True)
    try:
        with (Keys() if sys.stdin.isatty() else avt.NoKeys()) as keys:
            th.start()
            t_fps, n_fps = time.monotonic(), 0
            while shared["run"]:
                for k in keys.poll():
                    keys_q.put(k)
                got = src.read()
                if got is None:
                    print("\n[cam] " + ("end of the video" if args.video else "no image from the camera (camera permission not granted to the terminal? System Settings -> Privacy & Security -> Camera; or try --camera 1)"))
                    break
                frame, obs = got[0], got[1]
                with lock:
                    shared["obs"] = obs
                    st, start_at = shared["st"], shared["start_at"]
                log.write({"k": "f", "t": time.monotonic(), "obs": {s_: {k_: v_ for k_, v_ in o_.items() if k_ != "hand" or args.wrist} for s_, o_ in obs.items()}})
                n_fps += 1
                if time.monotonic() - t_fps >= 1.0:
                    shared["fps"], t_fps, n_fps = n_fps / (time.monotonic() - t_fps), time.monotonic(), 0
                if view:
                    if st["mode"] == "pose":
                        info = {"head": "MOVING TO PRESET POSE", "color": (255, 160, 60)}
                    elif start_at:
                        info = {"head": f"ARMS DOWN  -  START IN {max(math.ceil(start_at - time.monotonic()), 0)}", "color": (60, 200, 255)}
                    elif st["follow"]:
                        info = {"head": "FOLLOWING   [space] stop", "color": (90, 220, 110)}
                    else:
                        info = {"head": "PAUSED   [space] start", "color": (200, 200, 200)}
                    cal = "" if not st["follow"] else ("   NO ARMS-DOWN CAL (arms were not down / still)" if st["cal_deg"] is None else f"   arms-down cal {st['cal_deg']:.0f} deg")
                    info["sub"] = (f"{'MIRROR' if args.mirror else 'SAME SIDE'}   {shared['fps']:.0f} fps   [y] ready  [h] home  [x] quit"
                                   + cal + ("   DRY RUN (no robot)" if args.dry else ""))
                    if st["ff"] is not None and abs(st["ff"]) >= avt.FF_WARN and not args.dry:      # real arm not tracking the command: the bridge's gravity feed-forward is off
                        info["warn"] = (f"ROBOT {'FLOATS UP' if st['ff'] < 0 else 'SAGS'}: bridge gravity FF too {'STRONG' if st['ff'] < 0 else 'WEAK'}"
                                        f" -> restart bridge, --grav-scale {'at least ' if st['ff_floor'] else ''}{st['ff']:+.2f}")
                    cv2.imshow("Lucen camera teleop", draw_overlay(cv2, frame, got[2], st, info, flip=not args.no_flip))
                    k = cv2.waitKey(1) & 0xFF
                    if k != 255:
                        keys_q.put(chr(k))
        print("\n[cam] bridge disconnected" if cli.closed else "")
    except KeyboardInterrupt:
        pass
    finally:
        shared["run"] = False
        th.join(timeout=1.0)
        try:
            q = cli.latest()
            cli.send_target({n: q.get(n, v) for n, v in ct.q_cmd.items()})      # pin the target to the measured pose before leaving
        except Exception:
            pass
        cli.close()
        src.close()
        log.close()
        if view:
            cv2.destroyAllWindows()
        print("\n[cam] exiting" + ("" if args.dry else " (after the 1.5 s heartbeat timeout the bridge goes to damping, then disables)") + ("" if args.no_log else f"   log: {args.log}"))


if __name__ == "__main__":
    main()
