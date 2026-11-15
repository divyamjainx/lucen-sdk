#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/wheeled_base_io.py - reference implementation (numpy) of the wheeled_base policy I/O contract.

Implements the `policy` section of wheeled_base/wheeled_base.yaml: how an observation is built from sensor
readings, and how a network output becomes leg position targets and wheel speed targets. The MuJoCo
validation (tools/wheeled_base_sim2sim.py) uses it, and the same code is intended to be shared by a future
real-robot deployment. A training environment has to implement exactly the same contract: use this file as
the reference when aligning training code. Depends only on numpy and yaml.

Conventions:
  Joint vector (10) = the order of `joints` in wheeled_base/wheeled_base.yaml.
  Leg vector (8)    = the joints with `drive: position`, in the same order. The remaining 2 joints are the
                      wheels (`drive: velocity`).
  Observation = history_len frames of 39 values each (layout: see build_obs); with the shipped config that
                is 10 frames = 390 inputs. The last 2 values of a frame are the position-hold error
                (see update_hold).
  Action a (10) = clip(network output, ±action_clip). This clipped a is also the last_action observation.
    legs:   q_des = nominal + leg_action_scale * a, then clipped to the joint limits
    wheels: v_des = wheel_action_scale * a
  Motors: tau = clip(kp * (q_des - q) + kd * (v_des - v), ±tau_limit); legs use v_des = 0, wheels use kp = 0.
"""
import os

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAML_PATH = os.path.join(ROOT, "wheeled_base/wheeled_base.yaml")


def load_cfg(path=YAML_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


# Observation builder and action decoder for one running policy. An instance keeps the observation history
# and the position-hold state: call reset() whenever the policy is (re)started.
class WheelbaseIO:
    def __init__(self, cfg=None, history_len=None, hold=None):
        """history_len / hold (whether a frame carries the hold_error observation): None = the values of the
        contract in wheeled_base/wheeled_base.yaml. To run an ONNX trained with a different history length or
        without hold_error, pass the values recorded in its manifest."""
        cfg = cfg or load_cfg()
        self.cfg = cfg
        joints = cfg["joints"]
        self.names = [j["name"] for j in joints]
        self.n = len(joints)
        self.leg_idx = np.array([i for i, j in enumerate(joints) if j["drive"] == "position"])
        self.wheel_idx = np.array([i for i, j in enumerate(joints) if j["drive"] == "velocity"])
        self.leg_names = [self.names[i] for i in self.leg_idx]
        self.wheel_names = [self.names[i] for i in self.wheel_idx]
        assert set(cfg["nominal_pose"]) == set(self.leg_names), "nominal_pose must list exactly the leg joints"
        self.nominal = np.array([cfg["nominal_pose"][n] for n in self.leg_names], dtype=np.float64)
        self.leg_lo = np.array([joints[i]["range"][0] for i in self.leg_idx], dtype=np.float64)
        self.leg_hi = np.array([joints[i]["range"][1] for i in self.leg_idx], dtype=np.float64)

        p = cfg["policy"]
        self.rate_hz = float(p["control_rate_hz"])
        self.action_clip = float(p["action_clip"])
        self.leg_scale = float(p["leg_action_scale"])
        self.wheel_scale = float(p["wheel_action_scale"])
        self.obs_spec = list(p["obs"])
        if hold is False:                              # policy trained without hold_error in its frame
            self.obs_spec = [o for o in self.obs_spec if o["name"] != "hold_error"]
        self.hold = self.obs_spec[-1]["name"] == "hold_error"
        self.hold_thresh = float(p.get("hold_cmd_thresh", 0.05))
        self.hold_clip = np.array(p.get("hold_clip", [0.3, 0.5]), dtype=np.float64)
        self.hold_delay_steps = int(round(float(p.get("hold_delay_s", 0.0)) * float(p["control_rate_hz"])))
        self._stand_steps = 0
        self._wheel_q_prev = None
        self.wheel_radius = float(cfg["model"]["wheel_radius_m"])
        self._hold = np.zeros(2)
        self.frame_dim = sum(o["dim"] for o in self.obs_spec)       # values per frame: 39 (37 without hold_error)
        self.hist_len = int(history_len if history_len is not None else p.get("history_len", 1))
        assert self.hist_len >= 1
        self.obs_dim = self.frame_dim * self.hist_len
        self.act_dim = self.n
        self._hist = None
        want = {"base_ang_vel": 3, "projected_gravity": 3, "commands": 3,
                "leg_pos_rel": len(self.leg_idx), "joint_vel": self.n, "last_action": self.n}
        if self.hold:
            want["hold_error"] = 2
        assert [(o["name"], o["dim"]) for o in self.obs_spec] == list(want.items()), \
            f"policy.obs in wheeled_base/wheeled_base.yaml does not match this implementation: {self.obs_spec}"
        self._scale = {o["name"]: np.asarray(o["scale"], dtype=np.float64) for o in self.obs_spec}

        # per-joint motor parameters, in joint-vector order
        motor = [j["motor"] for j in joints]
        self.kp = np.array([cfg["gains"][m]["kp"] for m in motor], dtype=np.float64)
        self.kd = np.array([cfg["gains"][m]["kd"] for m in motor], dtype=np.float64)
        self.tau_limit = np.array([cfg["tau_limit"][m] for m in motor], dtype=np.float64)
        self.vel_limit = np.array([cfg["vel_limit"][m] for m in motor], dtype=np.float64)
        self.armature = np.array([cfg["armature"][m] for m in motor], dtype=np.float64)
        self.friction = np.array([cfg["joint_friction"][m] for m in motor], dtype=np.float64)
        self.damping = np.full(self.n, float(cfg["joint_damping"]))
        assert np.all(self.kp[self.wheel_idx] == 0.0), "the wheels are speed-controlled: their kp must be 0"

        # left-right mirror (y -> -y): x_mirror = sign * x[perm]
        self.mirror_perm, self.mirror_sign = self._mirror(self.names, cfg["mirror_sign"])
        self.leg_mirror_perm, self.leg_mirror_sign = self._mirror(self.leg_names, cfg["mirror_sign"])
        assert np.allclose(self.leg_mirror_sign * self.nominal[self.leg_mirror_perm], self.nominal), \
            "nominal_pose is not left-right symmetric, so mirroring the leg_pos_rel observation is not valid"

    @staticmethod
    def _mirror(names, sign_by_kind):
        perm, sign = [], []
        for n in names:
            side, kind = n[0], n[2:-len("_joint")]
            other = ("r" if side == "l" else "l") + n[1:]
            perm.append(names.index(other))
            sign.append(float(sign_by_kind[kind]))
        return np.array(perm), np.array(sign)

    # ---- observation ----
    def reset(self):
        """Call when the policy is enabled or a new episode starts: clears the observation history (the next
        build_obs fills every history slot with its own frame) and zeroes the position-hold error."""
        self._hist = None
        self._hold = np.zeros(2)
        self._stand_steps = 0
        self._wheel_q_prev = None

    def update_hold(self, cmd, q, ang_vel):
        """Advance the position-hold error [e_x (m), e_yaw (rad)] by one control step and return it.

        build_obs calls this; do not call it separately. Definition (hold_cmd_thresh, hold_delay_s and
        hold_clip are set in wheeled_base/wheeled_base.yaml): once the command has been zero (|vx| and |wz|
        both below hold_cmd_thresh) for longer than hold_delay_s, every control step
            e_x   <- clip(e_x   - wheel_radius * (mean wheel-angle increment of this step), ±hold_clip[0])
            e_yaw <- clip(e_yaw - gyro_z / control_rate_hz,                                 ±hold_clip[1])
        so e_x is how far the robot has rolled away from the point it is holding (positive = it should move
        forward) and e_yaw is how far it has turned away. While the command is non-zero, or has been zero
        for less than hold_delay_s, both are 0.

        cmd: (vx, vy, wz); q: joint vector (10), of which only the two wheel angles are used; ang_vel: body
        angular velocity, of which only z is used.

        Distance travelled = wheel radius * wheel-angle increment (wrapped to ±π, so encoder wrap-around is
        harmless), never wheel speed * dt. Angle increments add up exactly to the difference between the last
        and the first angle, so no error accumulates. A sampled wheel speed does not integrate to the actual
        change of the wheel angle: in PhysX the joint speed reading is not consistent with the angle, and a
        speed sampled at 50 Hz on hardware would be biased as well.
        """
        qw = np.asarray(q, dtype=np.float64)[self.wheel_idx]
        if self._wheel_q_prev is None:
            dq = np.zeros(len(self.wheel_idx))             # first step after reset(): no previous angle, increment 0
        else:
            dq = np.remainder(qw - self._wheel_q_prev + np.pi, 2.0 * np.pi) - np.pi
        self._wheel_q_prev = qw.copy()
        if abs(cmd[0]) < self.hold_thresh and abs(cmd[2]) < self.hold_thresh:
            self._stand_steps += 1
        else:
            self._stand_steps = 0
        if self._stand_steps > self.hold_delay_steps:          # the command has been zero for hold_delay_s
            step = np.array([-self.wheel_radius * float(dq.mean()), -float(ang_vel[2]) / self.rate_hz])
            self._hold = np.clip(self._hold + step, -self.hold_clip, self.hold_clip)
        else:
            self._hold = np.zeros(2)
        return self._hold

    def build_obs(self, ang_vel, gravity, cmd, q, qd, last_action):
        """Build the policy input for the current control step.

        ang_vel / gravity: base angular velocity and gravity direction (unit vector, (0, 0, -1) when upright),
        both in the body frame; cmd: (vx, vy, wz), where vy is always 0 because the wheels cannot move
        sideways; q / qd: joint vector (10) of positions / velocities; last_action: the clipped action a of
        the previous control step.

        One frame, in this order, each term multiplied by its `scale` from policy.obs:
            base_ang_vel (3), projected_gravity (3), commands (3), leg_pos_rel (8) = leg q - nominal,
            joint_vel (10, wheels included), last_action (10), hold_error (2) = update_hold()
        Wheel angles are unbounded and are not observed.

        Call this exactly once per control step: every call advances the observation history. The history
        layout is that of Isaac Lab's ObservationManager: each term keeps its own most recent hist_len frames,
        ordered oldest -> newest, and the terms are then concatenated in the order of the obs table:
            [base_ang_vel(t-H+1 .. t), projected_gravity(t-H+1 .. t), commands(...), leg_pos_rel(...),
             joint_vel(...), last_action(...), hold_error(...)]
        The first frame after reset() fills every history slot with itself. With hist_len = 1 the result is
        a single frame.
        """
        q = np.asarray(q, dtype=np.float64)
        parts = [
            np.asarray(ang_vel, dtype=np.float64) * self._scale["base_ang_vel"],
            np.asarray(gravity, dtype=np.float64) * self._scale["projected_gravity"],
            np.asarray(cmd, dtype=np.float64) * self._scale["commands"],
            (q[self.leg_idx] - self.nominal) * self._scale["leg_pos_rel"],
            np.asarray(qd, dtype=np.float64) * self._scale["joint_vel"],
            np.asarray(last_action, dtype=np.float64) * self._scale["last_action"],
        ]
        if self.hold:
            parts.append(self.update_hold(cmd, q, ang_vel) * self._scale["hold_error"])
        if self._hist is None:
            self._hist = [np.repeat(p[None], self.hist_len, axis=0) for p in parts]
        else:
            for h, p in zip(self._hist, parts):
                h[:-1] = h[1:]
                h[-1] = p
        obs = np.concatenate([h.reshape(-1) for h in self._hist])
        assert obs.shape == (self.obs_dim,)
        return obs.astype(np.float32)

    # ---- action ----
    def clip_action(self, raw):
        return np.clip(np.asarray(raw, dtype=np.float64), -self.action_clip, self.action_clip)

    def action_to_target(self, a):
        """a: clipped action (10) -> (q_des, v_des), both joint vectors (10).

        Legs:   q_des = nominal + leg_action_scale * a, clipped to the joint limits; v_des = 0.
        Wheels: v_des = wheel_action_scale * a (rad/s); q_des is left at 0 and has no effect (wheel kp = 0).
        """
        a = np.asarray(a, dtype=np.float64)
        q_des = np.zeros(self.n)
        v_des = np.zeros(self.n)
        q_des[self.leg_idx] = np.clip(self.nominal + self.leg_scale * a[self.leg_idx], self.leg_lo, self.leg_hi)
        v_des[self.wheel_idx] = self.wheel_scale * a[self.wheel_idx]
        return q_des, v_des

    # Motor law of the contract (see the module docstring). kp_scale / kd_scale model gain errors in simulation.
    def torque(self, q_des, v_des, q, qd, kp_scale=1.0, kd_scale=1.0):
        tau = kp_scale * self.kp * (q_des - q) + kd_scale * self.kd * (v_des - qd)
        return np.clip(tau, -self.tau_limit, self.tau_limit)

    # ---- left-right mirror ----
    def mirror_joints(self, x):
        return self.mirror_sign * np.asarray(x)[..., self.mirror_perm]

    def mirror_legs(self, x):
        return self.leg_mirror_sign * np.asarray(x)[..., self.leg_mirror_perm]

    def mirror_obs(self, obs):
        """Left-right mirror of an observation. Each term is a stack of hist_len frames (oldest -> newest);
        the same rule is applied to every frame."""
        obs = np.asarray(obs)
        out, i = [], 0
        H = self.hist_len
        for o in self.obs_spec:
            d = o["dim"]
            part = obs[..., i:i + d * H].reshape(*obs.shape[:-1], H, d)
            i += d * H
            name = o["name"]
            if name == "base_ang_vel":
                m = part * np.array([-1.0, 1.0, -1.0])      # pseudovector
            elif name == "projected_gravity":
                m = part * np.array([1.0, -1.0, 1.0])       # polar vector
            elif name == "commands":
                m = part * np.array([1.0, -1.0, -1.0])      # vx, vy, wz
            elif name == "leg_pos_rel":
                m = self.mirror_legs(part)
            elif name == "hold_error":
                m = part * np.array([1.0, -1.0])            # e_x unchanged, e_yaw changes sign
            else:                                           # joint_vel / last_action
                m = self.mirror_joints(part)
            out.append(m.reshape(*obs.shape[:-1], H * d))
        return np.concatenate(out, axis=-1)
