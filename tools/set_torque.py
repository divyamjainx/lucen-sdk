#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/set_torque.py - write the torque limits from the robot config into the motor firmware.

Every motor stores one torque limit (`limit_torque`). This tool sets it, motor by motor, to the `tau_limit`
value that the robot config gives for that motor model, and reads it back.

The limit in the motor must equal `tau_limit` in the config, because the simulation clips torque at that
same value (gains.py). If only one side is limited, the real robot and the simulation diverge without any
error message. tools/deploy_policy.py reads every motor back at startup and refuses to run on a mismatch.

Usage (on the robot computer; motors powered, bridge / deploy process stopped):
  python tools/set_torque.py --check          # read back and compare only, writes nothing (run this first)
  python tools/set_torque.py --write          # write to RAM (lost at power-off)
  python tools/set_torque.py --write --save   # write and save in the motor (kept across power cycles)
  python tools/set_torque.py --config humanoid_arm/humanoid_arm.yaml --check   # another robot: same procedure,
  python tools/set_torque.py --config Arm/Arm.yaml --check                     # limits taken from that config

`tau_limit` is a deployment safety value of the robot, not a property of the motor, and a motor stores only
one value. Run --write again whenever a motor is moved to a different robot or `tau_limit` is edited.
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import robstride_mit as rm                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="read back and compare only; writes nothing")
    ap.add_argument("--write", action="store_true",
                    help="write limit_torque to the motors (RAM only unless --save is given)")
    ap.add_argument("--save", action="store_true",
                    help="with --write: also save in the motor, so the value survives a power cycle")
    ap.add_argument("--config", default=os.path.join(ROOT, "biped_Laika/biped_Laika.yaml"),
                    help="robot config (default: biped_Laika/biped_Laika.yaml); e.g. "
                         "humanoid_arm/humanoid_arm.yaml, Arm/Arm.yaml, Arm/Arm_pair.yaml")
    args = ap.parse_args()
    if not (args.check or args.write):
        ap.error("give --check or --write")

    cfg = yaml.safe_load(open(args.config))
    cfg_name = os.path.basename(args.config)
    want = cfg.get("tau_limit") or cfg["tau_max"]
    rated = cfg["tau_max"]
    print(f"{cfg_name} tau_limit: " + "  ".join(
        f"{m}={want[m]} ({want[m]/rated[m]*100:.0f}% of rated)" for m in want))

    buses = {}
    for j in cfg["joints"]:
        if j.get("enabled", True):
            buses.setdefault(j["bus"], []).append(j)

    bad = []
    for ch, joints in sorted(buses.items()):
        # motor models come from the config: the same id can be a different model on another robot
        leg = rm.LegBus(ch, [j["id"] for j in joints], models=rm.models_from_cfg(cfg, ch))
        print(f"\n--- {ch} ---")
        for j in joints:
            mid, model, v = j["id"], j["motor"], float(want[j["motor"]])
            if args.write:
                ok = leg.set_torque_limit(mid, v, save=args.save)
            else:
                rb = leg.read_float_param(mid, rm.IDX_LIMIT_TORQUE)
                ok = rb is not None and abs(rb - v) < 0.05
                print(f"  id={mid:<2} ({model}) {j['name']:<22} "
                      f"read back {rb}  expected {v}  {'OK' if ok else '!! MISMATCH'}")
            if not ok:
                bad.append(j["name"])
        leg.shutdown()

    if bad:
        print(f"\n!! limit_torque differs from {cfg_name} on these joints: {bad}")
        print("   Run --write, then check again. Do not run a policy while they differ: "
              "the simulation clips torque at tau_limit and the real motors would not")
        return 1
    print(f"\nlimit_torque of all {sum(len(v) for v in buses.values())} motors matches {cfg_name} ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
