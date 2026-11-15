#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/policy_io.py - the biped policy I/O contract: observation layout and action -> joint-target mapping.

This module is the single implementation of the contract between a policy network and the robot. The MuJoCo
simulators (tools/sim2sim.py, tools/sim_fsm.py) and the real-robot deployment (tools/deploy_policy.py) all
build observations and joint targets through it, so simulation and the real robot use identical numbers.
Training code must reproduce it exactly. The values (terms, dimensions, scales, poses, timing) are read from
biped_Laika/biped_Laika.yaml, sections `policy`, `standing_pose` and `policy_profiles`.

Conventions
  Joint vectors (12 values) follow the order of the `joints` list in the configuration file.
  Joint angles are in rad and joint velocities in rad/s, in the URDF convention. Torso quantities are
  expressed in the torso frame.
  The policy runs at control_rate_hz (50 Hz). One evaluation = one control step.

Observation: the terms below, each multiplied by its scale and concatenated in this order (values as shipped;
the configuration file is authoritative).

  term               dim  scale              raw quantity
  base_ang_vel         3  0.25               torso angular velocity (rad/s)
  projected_gravity    3  1.0                unit gravity vector in the torso frame, [0, 0, -1] when upright
  commands             3  [2.0, 2.0, 0.25]   velocity command [vx (m/s), vy (m/s), yaw rate (rad/s)]
  joint_pos_rel       12  1.0                joint positions minus the nominal pose
  joint_vel           12  0.05               joint velocities
  last_action         12  1.0                action of the previous control step: the network output after
                                             clipping (and after the power scale, if one is used), before the
                                             action scale; zeros on the first step

These 45 values are the base observation; their order never changes. The default contract (standing policy)
uses exactly these. A profile can only append terms after them:

  gait_phase           2  1.0                [cos(2*pi*phase), sin(2*pi*phase)], see PolicyIO.phase_of
  frame_hist         168  1.0                the 4 previous frames of 42 values each, see PolicyIO.build_obs

Action: 12 values, one per joint, mapped to joint position targets for the joint PD loop (motor firmware on
the real robot, physics sub-steps in simulation):
  target = nominal_pose + action_scale * clip(action, -action_clip, +action_clip)
with the per-profile options described in PolicyIO.action_to_target. In particular, the profile of the shipped
walking policy adds a lateral feed-forward to the target (PolicyIO.lateral_ff).
"""
import numpy as np

# A frame_hist frame holds the five base terms other than commands and gait_phase, in contract order (42 values).
# commands are left out because they are an external input whose history carries no information; gait_phase is
# left out because it is an open-loop clock that follows from the phase. Each term is stored already multiplied
# by its scale from the base contract.
_HIST_FRAME_TERMS = ("base_ang_vel", "projected_gravity",
                     "joint_pos_rel", "joint_vel", "last_action")


class PolicyIO:
    """The contract of one policy profile.

    profile=None: the default contract (`policy` + `standing_pose` in biped_Laika/biped_Laika.yaml), used by the
    standing policy. profile='walk' etc.: the default contract with the overrides of `policy_profiles.<profile>`:
      control_rate_hz, action_scale, action_clip, cycle_time_s    replace the default values
      nominal_pose         replaces standing_pose: the pose that a zero action maps to and that joint_pos_rel
                           is measured from
      obs_extra            observation terms appended after the base 45 (gait_phase, frame_hist, ...)
      action_scale_joint   per-joint action scale by joint name (overrides the scalar action_scale)
      target_bounds        per-joint [lo, hi] target bounds, tighter than the joint limits
      lat_ff_gain          gain of the lateral feed-forward added to the target (absent or 0 = off)
      init_base_height     base height in the nominal pose (m)
    One ONNX file belongs to exactly one profile. The manifest stamped into the file names it
    (tools/onnx_manifest.py), and the tools pick the profile from there."""

    def __init__(self, cfg, profile=None):
        self.names = [j["name"] for j in cfg["joints"]]
        self.jlo = np.array([j["range"][0] for j in cfg["joints"]])
        self.jhi = np.array([j["range"][1] for j in cfg["joints"]])
        pol = dict(cfg["policy"])
        pose = cfg["standing_pose"]
        self.profile = profile
        # lateral (side-step) feed-forward gain, part of the contract (0 = off)
        self.lat_ff_gain = float((cfg.get("policy_profiles") or {})
                                 .get(profile, {}).get("lat_ff_gain") or 0.0)
        # Name of the gain profile the policy was trained with (None = rl_default). It is recorded in the
        # manifest so that a consumer can refuse to run the policy with other gains.
        self.gain_profile = (cfg.get("policy_profiles") or {}).get(profile, {}).get("gain_profile")
        self.init_base_height = (cfg.get("policy_profiles") or {}).get(profile, {}).get("init_base_height")
        if profile is not None:
            avail = cfg.get("policy_profiles") or {}
            if profile not in avail:
                raise KeyError(f"biped_Laika/biped_Laika.yaml has no policy_profiles.{profile}; available: {sorted(avail)}")
            p = avail[profile]
            pol.update({k: v for k, v in p.items()
                        if k in ("control_rate_hz", "action_scale", "action_clip", "cycle_time_s")})
            pose = p.get("nominal_pose", pose)
            extra = p.get("obs_extra") or []
            asj = p.get("action_scale_joint") or {}
            tbnd = p.get("target_bounds") or {}
        else:
            extra = []
            asj, tbnd = {}, {}
        self.default = np.array([pose[n] for n in self.names])
        self.rate = pol["control_rate_hz"]
        self.action_scale = pol["action_scale"]
        self.action_clip = pol["action_clip"]
        # Per-joint action scale and target bounds.
        # action_scale_joint overrides the scalar action_scale by joint name; a scale of 0 pins that joint at
        # its nominal position (the action stays 12-dim). target_bounds are tighter than the URDF joint limits
        # (e.g. to keep a margin before full knee extension). A profile that declares neither uses the scalar
        # scale and the joint limits.
        self.action_scale_vec = np.array(
            [float(asj.get(n, self.action_scale)) for n in self.names])
        self.tlo = np.array([max(self.jlo[i], tbnd[n][0]) if n in tbnd else self.jlo[i]
                             for i, n in enumerate(self.names)])
        self.thi = np.array([min(self.jhi[i], tbnd[n][1]) if n in tbnd else self.jhi[i]
                             for i, n in enumerate(self.names)])
        # period of the gait clock (s); only profiles with a gait_phase term have one
        self.cycle_time = pol.get("cycle_time_s")
        # a profile can only append observation terms at the END; the base 45 values keep their order
        self.spec = list(pol["obs"]) + list(extra)
        self.scales = {o["name"]: np.asarray(o["scale"], dtype=float) for o in self.spec}
        self.obs_dim = sum(o["dim"] for o in self.spec)
        # frame_hist: the block of past frames at the end of the observation; semantics in build_obs
        hist = [o for o in self.spec if o["name"] == "frame_hist"]
        if hist:
            self._frame_dim = sum(o["dim"] for o in self.spec
                                  if o["name"] in _HIST_FRAME_TERMS)
            assert hist[0]["dim"] % self._frame_dim == 0, \
                f"frame_hist dim {hist[0]['dim']} is not a multiple of the frame length {self._frame_dim}"
            assert float(self.scales["frame_hist"]) == 1.0, \
                "frame_hist frames are stored pre-scaled: the scale of the term itself must be 1.0 (else it is applied twice)"
            self.hist_len = hist[0]["dim"] // self._frame_dim
        else:
            self.hist_len = 0
        self._hist = None          # None = empty; the first call fills it as if the robot had been at rest in that state

    def reset_history(self):
        """Clear the frame history. Call it whenever the policy is enabled or an episode is reset
        (creating a new PolicyIO has the same effect)."""
        self._hist = None

    def phase_of(self, t):
        """Gait phase in [0, 1) at time t (s): phase = (t / cycle_time_s) mod 1. None if the profile has no gait clock.

        The clock is open-loop: it depends on absolute time only, never on the robot state. t = 0 is the
        moment the policy is enabled (or the episode is reset). The real robot and the simulation must use
        this same origin, otherwise the phase does not line up with what the policy was trained on.
        """
        if self.cycle_time is None:
            return None
        return (float(t) / self.cycle_time) % 1.0

    def build_obs(self, ang_vel, gravity, cmd, q, qd, last_action, phase=None):
        """Assemble the observation vector of the current control step.

        All inputs are raw (unscaled) quantities, torso frame / URDF joint convention:
          ang_vel      (3,)   torso angular velocity (rad/s)
          gravity      (3,)   unit gravity vector in the torso frame
          cmd          (3,)   velocity command [vx, vy, yaw rate]
          q, qd        (12,)  joint positions and velocities, contract joint order
          last_action  (12,)  action of the previous control step (zeros on the first step)
          phase               gait phase from phase_of(t); required if the contract has a gait_phase term,
                              which is observed as [cos(2*pi*phase), sin(2*pi*phase)]
        The terms are scaled and concatenated in contract order (see the module docstring).

        frame_hist term: this method is the authoritative definition of its timing.
          - Call build_obs exactly once per control step. Each call advances the history, so an extra or a
            missing call shifts it by one step and simulation and the real robot no longer agree.
          - If the history is empty (first call after enabling or reset_history), it is filled as if the robot
            had always been at rest in its current state: all 4 slots = the current frame.
          - The block placed in the observation is [t-1, t-2, t-3, t-4], newest first. It does NOT contain
            the current frame: the current frame is the base 45 values at the head of the observation.
          - The current frame is pushed into the history after the observation has been assembled.
        A frame is the 42 scaled values [base_ang_vel, projected_gravity, joint_pos_rel, joint_vel,
        last_action] (no commands, no gait_phase). Training code must reproduce these four rules exactly.
        """
        src = {
            "base_ang_vel": ang_vel,
            "projected_gravity": gravity,
            "commands": cmd,
            "joint_pos_rel": np.asarray(q) - self.default,
            "joint_vel": qd,
            "last_action": last_action,
        }
        if any(o["name"] == "gait_phase" for o in self.spec):
            assert phase is not None, "the contract has gait_phase: build_obs needs phase (use phase_of(t))"
            src["gait_phase"] = np.array([np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)])
        scaled = {}
        for o in self.spec:
            if o["name"] == "frame_hist":
                continue
            v = np.asarray(src[o["name"]], dtype=float)
            assert v.shape == (o["dim"],), f"{o['name']}: shape {v.shape} != {o['dim']}"
            scaled[o["name"]] = v * self.scales[o["name"]]
        if self.hist_len:
            frame = np.concatenate([scaled[n] for n in _HIST_FRAME_TERMS])
            if self._hist is None:
                self._hist = [frame.copy() for _ in range(self.hist_len)]
            scaled["frame_hist"] = np.concatenate(self._hist)
            self._hist = [frame] + self._hist[:-1]
        return np.concatenate([scaled[o["name"]] for o in self.spec])

    def action_to_target(self, action, cmd=None, phase=None):
        """Map a network output to joint position targets (rad, URDF convention, within limits).

            a      = clip(action, -action_clip, +action_clip)
            target = nominal_pose + scale_j * a_j         scale_j = action_scale, or the joint's entry in
                                                          action_scale_joint if the profile has one
            target = target + lateral_ff(cmd_vy, phase)   only for profiles with lat_ff_gain != 0
            target = clip(target, lo_j, hi_j)             joint limits, tightened by target_bounds if declared

        Profiles with `lat_ff_gain != 0` add the lateral feed-forward (see lateral_ff), so `cmd` and `phase`
        are mandatory for them: if one is missing this raises ValueError instead of silently dropping the
        feed-forward. Pass the same phase value that build_obs received in this control step.
        """
        a = np.clip(np.asarray(action, dtype=float), -self.action_clip, self.action_clip)
        target = self.default + self.action_scale_vec * a
        if self.lat_ff_gain:
            if cmd is None or phase is None:
                raise ValueError(
                    f"this profile has lat_ff_gain={self.lat_ff_gain}: action_to_target needs "
                    f"both cmd and phase (the feed-forward is part of the contract)")
            target = target + self.lateral_ff(float(np.asarray(cmd)[1]), float(phase))
        return np.clip(target, self.tlo, self.thi)

    def lateral_ff(self, cmd_vy, phase):
        """Lateral (side-step) feed-forward (12,) in rad. All zeros if cmd_vy == 0 or lat_ff_gain == 0.

        A triangle wave on each hip roll joint, with the opposite sign on the ankle roll joint of the same leg:
            amp  = cmd_vy * cycle_time_s / HIP_ROLL_Y_PER_RAD * lat_ff_gain
            u    = (phase - swing_lo) mod 1               swing_lo = 0.5 for the left leg, 0.0 for the right leg
            saw  = 4*u - 1  if u < 0.5  else  1 - 4*(u - 0.5)        runs -1 -> +1 -> -1 over one cycle
            hip_roll offset = saw * amp,  ankle_roll offset = -saw * amp
        The feed-forward is part of the contract: training code must apply exactly the same function
        (agreement within 1e-5).
        """
        off = np.zeros(len(self.names))
        if not self.lat_ff_gain or cmd_vy == 0.0:
            return off
        amp = cmd_vy * self.cycle_time / HIP_ROLL_Y_PER_RAD * self.lat_ff_gain
        for swing_lo, hip, ank in ((0.5, "l_hip_roll_joint", "l_ankle_roll_joint"),
                                   (0.0, "r_hip_roll_joint", "r_ankle_roll_joint")):
            u = (phase - swing_lo) % 1.0
            saw = 4.0 * u - 1.0 if u < 0.5 else 1.0 - 4.0 * (u - 0.5)
            off[self.names.index(hip)] = saw * amp
            off[self.names.index(ank)] = -saw * amp
        return off


#: Lateral displacement of a foot (m, along the torso y axis) per rad of hip roll on the same side, from the
#: forward kinematics of the model. Training code that reproduces lateral_ff must use the same value.
HIP_ROLL_Y_PER_RAD = 0.325

MIRROR = [("l_hip_pitch_joint", "r_hip_pitch_joint"), ("l_hip_roll_joint", "r_hip_roll_joint"),
          ("l_hip_yaw_joint", "r_hip_yaw_joint"), ("l_knee_joint", "r_knee_joint"),
          ("l_ankle_pitch_joint", "r_ankle_pitch_joint"),
          ("l_ankle_roll_joint", "r_ankle_roll_joint")]


def summarize(rec, io, csv_path):
    """Summarize a logged run joint by joint and write the per-step data to csv_path. The printed tables help to
    tell apart calibration offsets, tracking errors and deliberate policy action."""
    t = np.array([r[0] for r in rec])
    grav = np.array([r[1] for r in rec])
    q = np.array([r[2] for r in rec])
    tgt = np.array([r[3] for r in rec])
    act = np.array([r[4] for r in rec])
    half = len(t) // 2                      # second half = steady state

    with open(csv_path, "w") as f:
        f.write("t," + ",".join(f"gx,gy,gz".split(",")) + ","
                + ",".join(f"q_{n}" for n in io.names) + ","
                + ",".join(f"tgt_{n}" for n in io.names) + ","
                + ",".join(f"act_{n}" for n in io.names) + "\n")
        for i in range(len(t)):
            f.write(",".join(f"{v:.5f}" for v in
                             [t[i], *grav[i], *q[i], *tgt[i], *act[i]]) + "\n")
    print(f"[log] per-step data -> {csv_path} ({len(t)} steps)")

    print()
    print("Per-joint steady state (mean over the second half, rad):")
    print(f"{'joint':<22}{'default':>9}{'target':>9}{'measured':>9}{'track err':>10}{'action':>10}")
    for i, n in enumerate(io.names):
        print(f"{n:<22}{io.default[i]:>9.4f}{tgt[half:, i].mean():>9.4f}"
              f"{q[half:, i].mean():>9.4f}{(tgt[half:, i]-q[half:, i]).mean():>10.4f}"
              f"{act[half:, i].mean():>10.3f}")

    print()
    print("Left / right symmetry (mirror convention l ≈ -r; a large |asym| = calibration or mechanical asymmetry of that joint):")
    idx = {n: i for i, n in enumerate(io.names)}
    for ln, rn in MIRROR:
        li, ri = idx[ln], idx[rn]
        for label, arr in (("measured", q), ("target", tgt)):
            asym = arr[half:, li].mean() + arr[half:, ri].mean()   # ≈ 0 when symmetric
            if label == "measured":
                print(f"  {ln[2:]:<20} {label} l={arr[half:, li].mean():+.4f} "
                      f"r={arr[half:, ri].mean():+.4f}  asym={asym:+.4f} "
                      f"({np.degrees(abs(asym)):.1f}°)")

    # Step detection: when the robot steps, the two knees swing in anti-phase. The signal
    # l_knee - (-r_knee) gives the dominant frequency and the amplitude.
    idx = {n: i for i, n in enumerate(io.names)}
    if "l_knee_joint" in idx and len(t) > 40:
        rate = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 50.0
        sig = q[:, idx["l_knee_joint"]] + q[:, idx["r_knee_joint"]]   # this sum oscillates when the knees move in anti-phase
        sig = sig - sig.mean()
        n = len(sig)
        sp = np.abs(np.fft.rfft(sig * np.hanning(n)))
        fr = np.fft.rfftfreq(n, d=1.0 / rate)
        band = (fr > 0.3) & (fr < 5.0)              # plausible step-frequency band
        if band.any():
            f0 = fr[band][np.argmax(sp[band])]
            amp = np.percentile(sig, 95) - np.percentile(sig, 5)
            print()
            if amp > 0.05:
                print(f"Gait: step frequency {f0:.2f} Hz (period {1/f0:.2f}s), "
                      f"anti-phase knee swing {amp:.3f} rad ({np.degrees(amp):.1f}°) -> stepping")
            else:
                print(f"Gait: anti-phase knee swing of only {amp:.3f} rad ({np.degrees(amp):.1f}°) "
                      f"-> NOT stepping, just holding position / jittering")

    print()
    print(f"Torso projected gravity, steady state: {np.round(grav[half:].mean(axis=0), 4)} (ideal [0,0,-1])")
    tilt = np.degrees(np.arccos(np.clip(-grav[half:, 2], -1, 1)))
    print(f"Torso tilt: mean {tilt.mean():.2f}°, max {tilt.max():.2f}°")
