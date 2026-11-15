#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""slew.py - target rate limiter of the bridge (single implementation, shared by bridge.py and tools/sim2sim.py).

On the real robot every joint target passes through this limiter:
    policy -> target -> [this limiter] -> motor PD -> torque
The simulator imports the same code, so that it models this part of the command path exactly. A stage that
exists only on the real robot invalidates simulation results: a 1 rad/s limit, for example, cuts a 2.5 Hz knee
swing from 45.8° to 11.6° and shifts its phase by about 90°. Whatever is added to the command path of the real
robot must be added to the simulation as well. The module depends only on biped_Laika/biped_Laika.yaml.

Analytic behaviour (speed limit S, sine of frequency f):
  - peak-to-peak amplitude <= S/(pi*f): passes undistorted
  - larger: saturates to a triangle wave with peak-to-peak S/(2*f), lagging by about 1/(4f)
"""
import os

import yaml

_CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "biped_Laika/biped_Laika.yaml")))

SOFT_SPD = 1.0        # rad/s; limit for teleoperation, and for the first homing move in every mode
# Policy deployment: once a joint is homed, its limit is the vel_limit of its motor model. A higher value would
# be pointless: the motor firmware clamps to its own speed limit as well.
FAST_SPD = {m: float(v) for m, v in _CFG["vel_limit"].items()}
HOME_TOL = 0.05       # rad; a joint counts as homed once the commanded value first gets this close to the target


def slew_step(cur, des, model, homed, fast, dt):
    """Advance the limiter by one control step. Returns (new commanded target, whether the joint is homed).

    Homing latch: until the commanded value first converges to the target, SOFT_SPD always applies. Right after
    enabling, the target can be far from the current pose (e.g. standing pose -> crouched walking pose), and
    sending it directly with stiff gains would jerk the joint. Once homed, the latch stays released: while
    walking the target legitimately jumps by up to about 0.9 rad per control step, so a large jump must not be
    taken as a reason to home again. In the step where the latch releases, the command moves by at most the
    remaining HOME_TOL (0.05 rad = 2.9°, i.e. 1.5 Nm at kp = 30).
    """
    step = des - cur
    homed = homed or abs(step) < HOME_TOL
    lim = (FAST_SPD[model] if (fast and homed) else SOFT_SPD) * dt
    return cur + min(max(step, -lim), lim), homed
