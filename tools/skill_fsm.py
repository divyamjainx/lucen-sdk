# -*- coding: utf-8 -*-
"""tools/skill_fsm.py - state machine that switches between the standing and the walking policy.

Pure logic with no hardware dependency. tools/deploy_policy.py (real robot) and tools/sim_fsm.py (MuJoCo) drive
the same class: once per control step they call update() and then act().

States
  STAND   the standing policy runs
  LOCO    the walking policy runs
  BLEND   transition between two different nominal poses; no policy runs

Hand-over rules, applied on every switch
  1. On entering a policy state, last_action is set to zero and the observation history of that policy is
     reset (io.reset_history()), so the policy never sees history left over from an earlier run.
  2. The gait clock of LOCO starts when LOCO is entered: phase 0, as at the start of a training episode.
  3. If the two policies have different nominal poses (standing: straight legs; walking: knees bent), the
     switch goes through BLEND: a minimum-jerk interpolation from the measured joint positions to the nominal
     pose of the next policy over BLEND_S, held by the joint PD loop with both feet on the ground. The next
     policy then starts exactly in its own nominal pose, as in training.

Home state and automatic walking
  `home` is the state the machine starts in and returns to.
  - Standing policy mounted: home = STAND with auto_walk. Any non-zero velocity command (W/A/S/D/Q/E)
    switches to LOCO; once the command has been zero for walk_release_s, the machine returns to STAND.
    With auto_walk off the switch is explicit: request("stand") / request("loco") (keys 1 / 2).
  - Only a walking policy mounted: home = LOCO, and the robot follows the velocity command directly.

Use
  slots = {"STAND": Slot("STAND", io, policy), "LOCO": Slot("LOCO", io, policy, power)}
  fsm = SkillFSM(slots, "STAND", home="STAND", auto_walk=True)
  each control step:  fsm.update(now, q, cmd)
                      target, raw = fsm.act(now, ang_vel, gravity, cmd, q, qd)
  fsm.events records every transition as (t, from, to, reason).
"""
import numpy as np

REQ_STATE = {"stand": "STAND", "loco": "LOCO"}


class LocoAssist:
    """Walking assistance: adjusts the velocity command that is fed to the LOCO policy (deployment side only).

      1. Heading hold: while translating (vx or vy non-zero) without a turn command, the yaw-rate command
         becomes wz = clip(k * (yaw_ref - yaw), ±wz_max). yaw_ref is latched when the translation starts or
         when the turn key is released.
      2. Lateral trim: while walking forward (vx > 0, no turn command), trim_vy is added to vy to cancel a
         steady sideways drift of the walking policy.
    Only the command given to the policy is changed. The automatic-walking logic of the state machine keeps
    using the raw operator command; otherwise the trim would be mistaken for a request to walk.
    """

    def __init__(self, k=0.5, wz_max=0.3, trim_vy=0.0, hold=True):
        self.k, self.wz_max, self.trim_vy, self.hold = float(k), float(wz_max), float(trim_vy), bool(hold)
        self.ref = None

    def __call__(self, state, cmd, yaw):
        out = np.array(cmd, dtype=float)
        no_turn = abs(out[2]) < 1e-6
        translating = state == "LOCO" and no_turn and (abs(out[0]) > 1e-6 or abs(out[1]) > 1e-6)
        if self.hold and translating and yaw is not None:
            if self.ref is None:
                self.ref = float(yaw)
            e = float(np.arctan2(np.sin(self.ref - yaw), np.cos(self.ref - yaw)))
            out[2] = float(np.clip(self.k * e, -self.wz_max, self.wz_max))
        else:
            self.ref = None
        if state == "LOCO" and no_turn and cmd[0] > 1e-6:
            out[1] += self.trim_vy
        return out


class Slot:
    """The policy mounted on one state: its contract (PolicyIO), the inference function and the action power
    scale (only LOCO uses --power-scale)."""

    def __init__(self, name, io, policy, power=1.0, path=""):
        self.name, self.io, self.policy, self.power, self.path = name, io, policy, float(power), path


class SkillFSM:
    BLEND_S = 0.8          # duration of a nominal-pose transition (s)
    NOM_TOL = 0.05         # BLEND is needed if two nominal poses differ by more than this on any joint (rad)

    def __init__(self, slots, start, log=print, home=None, auto_walk=False, walk_release_s=1.0):
        assert start in slots, f"start state {start} has no policy mounted"
        self.slots, self.log = slots, log
        self.home = home or start
        assert self.home in slots, f"home state {self.home} has no policy mounted"
        self.auto_walk = bool(auto_walk) and "LOCO" in slots and self.home != "LOCO"
        self.events = []                 # [(t, from, to, why)], read by offline tools
        self.state = None
        self.last_act = np.zeros(12)
        self.t_enter = 0.0
        self.blend = None                # dict(q0, q1, t0, dur, then)
        self.last_move_t = -1e9
        # Time with a zero command before returning home. The default of 1.0 s keeps a tap on W from switching
        # back and forth through BLEND.
        self.walk_release_s = float(walk_release_s)
        self._enter(start, 0.0, "start")

    # ------------------------------------------------------------------ internals
    def _nominal(self, state):
        return self.slots[state].io.default

    def _enter(self, state, now, why):
        prev = self.state
        self.state, self.t_enter = state, now
        self.last_act = np.zeros(12)
        if state == "LOCO" and self.auto_walk:
            self.last_move_t = now           # the release timer starts on entering LOCO, so LOCO always gets the full walk_release_s
        if state in self.slots:
            self.slots[state].io.reset_history()
        self.events.append((now, prev, state, why))
        if prev is not None:
            self.log(f"[fsm] {prev} -> {state} ({why})")

    def _goto(self, dst, now, q, why):
        """Go to dst. If the nominal poses of the current state and dst differ, BLEND first (it starts from the
        measured q, so there is no jump)."""
        cur = self.state if self.state in self.slots else None
        need = cur is None or \
            float(np.max(np.abs(self._nominal(cur) - self._nominal(dst)))) > self.NOM_TOL
        if need:
            self.blend = dict(q0=np.asarray(q, float).copy(), q1=self._nominal(dst).copy(),
                              t0=now, dur=self.BLEND_S, then=dst)
            self._enter("BLEND", now, f"{why}; different nominal pose, blending {self.BLEND_S:.1f}s -> {dst}")
        else:
            self._enter(dst, now, why)

    def busy(self):
        return self.state == "BLEND"

    # ------------------------------------------------------------------ interface
    def request(self, req, now, q):
        """Handle a key request ("stand" or "loco"). Returns whether it was accepted; a rejection logs the reason."""
        dst = REQ_STATE.get(req)
        if dst is None:
            return False
        if req == "stand" and "STAND" not in self.slots:
            dst = self.home                      # no standing policy: "stand" means "go home"
        if dst not in self.slots:
            self.log(f"[fsm] no {dst} policy mounted, ignored ({req})")
            return False
        if self.busy():
            self.log(f"[fsm] {self.state} in progress, {req} ignored")
            return False
        if dst == self.state:
            return False
        self._goto(dst, now, q, f"request {req}")
        return True

    def update(self, now, q, cmd):
        """Advance the state; call once per control step. Handles automatic walking and the end of a transition."""
        if self.state == "LOCO" and np.any(np.abs(np.asarray(cmd)) > 1e-6):
            self.last_move_t = now
        if self.auto_walk:
            moving = bool(np.any(np.abs(np.asarray(cmd)) > 1e-6))
            if moving and self.state == self.home:
                self._goto("LOCO", now, q, "walk command")
                self.last_move_t = now
                return
            if self.state == "LOCO" and not moving and now - self.last_move_t >= self.walk_release_s:
                self._goto(self.home, now, q, f"zero command for {self.walk_release_s:.1f}s, back to standing")
                return
        if self.state == "BLEND" and now - self.blend["t0"] >= self.blend["dur"]:
            then = self.blend["then"]
            self.blend = None
            self._enter(then, now, "blend done")

    def act(self, now, ang_vel, gravity, cmd, q, qd):
        """-> (target (12,), raw network output | None). In BLEND no policy runs. In every other state the
        mounted policy is evaluated and its output is mapped to joint targets through the contract of that state."""
        if self.state == "BLEND":
            b = self.blend
            u = float(np.clip((now - b["t0"]) / b["dur"], 0.0, 1.0))
            s = u ** 3 * (10 - 15 * u + 6 * u * u)
            return b["q0"] + (b["q1"] - b["q0"]) * s, None
        slot = self.slots[self.state]
        io = slot.io
        ccmd = np.asarray(cmd, float) if self.state == "LOCO" else np.zeros(3)
        ph = io.phase_of(now - self.t_enter) if io.cycle_time else None
        obs = io.build_obs(ang_vel, gravity, ccmd, q, qd, self.last_act, phase=ph)
        raw = slot.policy(obs)
        a = np.clip(raw, -io.action_clip, io.action_clip) * slot.power
        self.last_act = np.asarray(a, dtype=float)
        target = io.action_to_target(a, cmd=ccmd, phase=ph)
        return target, raw
