# -*- coding: utf-8 -*-
"""tools/bal_push.py - push disturbances for the MuJoCo simulators (shared by sim2sim.py and sim_fsm.py).

Two kinds of disturbance, both applied to base_link as a horizontal force in the world frame:
  torso force pulse (N_S, DUR_S, AT_S, DEG): a constant force N_S / DUR_S during [AT_S, AT_S + DUR_S), in the
    horizontal direction DEG; the total impulse is N_S (N·s).
  sustained hand push (V, AT_S, DUR_S, DEG, FMAX): a "hand" advances at constant speed V (m/s) along DEG and
    presses on the torso through a spring (200 N/m) and a damper (20 N·s/m). It only pushes, never pulls (if
    the robot moves ahead of the hand, the force is 0), and the force is capped at FMAX (N). While the robot
    resists, the hand stops 5 cm beyond the displacement that gives FMAX instead of winding up further.
The force is computed from the state at the start of a control step and held constant for that step.
"""
import numpy as np

HAND_K, HAND_C = 200.0, 20.0


def _unit(deg):
    a = np.radians(deg)
    return np.array([np.cos(a), np.sin(a)])


class BalPush:
    def __init__(self, rate, torso=None, hand=None):
        self.rate = float(rate)
        self.torso = [tuple(map(float, p)) for p in (torso or [])]
        self.hand = None
        if hand is not None:
            v, at, dur, deg, fmax = map(float, hand)
            self.hand = dict(v=v, at=at, dur=dur, deg=deg, dir=_unit(deg), fmax=fmax, s=None)

    def windows(self):
        """-> [(t0, t1, kind, direction in deg, magnitude)]: the disturbance windows, for the metrics."""
        w = [(p[2], p[2] + p[1], "pulse", p[3], p[0]) for p in self.torso]
        if self.hand:
            h = self.hand
            w.append((h["at"], h["at"] + h["dur"], "hand", h["deg"], h["v"]))
        return w

    def force(self, t0, pos_xy, vel_xy):
        """Horizontal external force (N, world frame) for the control step that starts at time t0."""
        F = np.zeros(2)
        for n_s, dur, at, deg in self.torso:
            if at - 1e-9 <= t0 < at + dur - 1e-9:
                F += n_s / dur * _unit(deg)
        h = self.hand
        if h is not None and h["at"] - 1e-9 <= t0 < h["at"] + h["dur"] - 1e-9:
            s_b = float(np.asarray(pos_xy) @ h["dir"])
            v_b = float(np.asarray(vel_xy) @ h["dir"])
            if h["s"] is None:
                h["s"] = s_b
            h["s"] += h["v"] / self.rate
            cap = h["fmax"] / HAND_K + 0.05
            gap = h["s"] - s_b
            if gap > cap:
                h["s"], gap = s_b + cap, cap
            f = 0.0 if gap < 0.0 else float(np.clip(HAND_K * gap + HAND_C * (h["v"] - v_b), 0.0, h["fmax"]))
            F += f * h["dir"]
        return F
