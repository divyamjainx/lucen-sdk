#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gains.py - gain-profile loader for simulation: kp / kd / torque limits from biped_Laika/biped_Laika.yaml.

biped_Laika/biped_Laika.yaml is the single source of these values; the real robot and the simulation tools
read the same file. Run `python gains.py` to print the rl_default profile expanded per joint.

Principles:
  - The same PD law everywhere: tau = clip(kp*(q_des - q) - kd*qvel, ±tau_max)
  - In simulation the PD is evaluated at every physics substep (on the real robot it runs in the motor firmware
    at kHz rate); the policy only updates q_des at 50 Hz
  - The values are not tuned separately for simulation; robustness to the sim-to-real gap comes from DOMAIN_RAND
  - The torque clip is tau_limit (the limit written to the motor firmware, see tools/set_torque.py), not
    tau_max (the rated peak torque of the motor): simulation must clip at the value the real robot uses,
    otherwise the two diverge silently

To check on the real robot that a motor applies the commanded kp (hoist the robot first):
  python robstride_mit.py --bus <bus> --id N --scale-check
"""
import os

import yaml

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biped_Laika/biped_Laika.yaml")
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

MOTOR_OF = {j["name"]: j["motor"] for j in _CFG["joints"]}
TAU_RATED = dict(_CFG["tau_max"])                       # rated peak torque per motor model
TAU_LIMIT = dict(_CFG.get("tau_limit") or _CFG["tau_max"])   # deployed torque limit = torque clip in simulation
TAU_MAX = TAU_LIMIT                                     # alias for existing callers; means the deployed torque limit
for _m, _v in TAU_LIMIT.items():
    assert _v <= TAU_RATED[_m] + 1e-9, f"tau_limit[{_m}]={_v} exceeds the rated torque {TAU_RATED[_m]}"
# joint_overrides: within a gain profile, (kp, kd) can be overridden per joint name; all other joints use the
# per-model values. They are kept in a separate table so that the per-model table holds only (kp, kd) pairs.
PROFILES = {p: {m: tuple(v) for m, v in d.items() if m != "joint_overrides"}
            for p, d in _CFG["gain_profiles"].items()}
OVERRIDES = {p: {j: tuple(v) for j, v in (d.get("joint_overrides") or {}).items()}
             for p, d in _CFG["gain_profiles"].items()}
for _p, _ov in OVERRIDES.items():
    for _j in _ov:
        assert _j in MOTOR_OF, f"gain_profiles.{_p}.joint_overrides: unknown joint {_j}"
DOMAIN_RAND = dict(_CFG["domain_rand"])


def profile(name="rl_default"):
    """-> {joint: {"kp", "kd", "tau_max", "motor"}} for all 12 joints, with the per-joint overrides applied."""
    p, ov = PROFILES[name], OVERRIDES[name]
    out = {}
    for j, m in MOTOR_OF.items():
        kp, kd = ov.get(j, p[m])
        out[j] = {"kp": kp, "kd": kd, "tau_max": TAU_MAX[m], "motor": m}
    return out


if __name__ == "__main__":
    for j, g in profile().items():
        print(f"{j:<22} {g['motor']}  kp={g['kp']:<5} kd={g['kd']:<4} "
              f"tau_max={g['tau_max']}")
