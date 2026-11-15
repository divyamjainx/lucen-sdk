#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/set_can_timeout.py - write the CAN timeout from the robot config into every motor (canTimeout, 0x7028).

Motor-side CAN timeout (RobStride parameter 0x7028 `canTimeout`, uint32): when it is non-zero, a motor that
receives no CAN command for this long resets itself, i.e. it disables and the joint is released. The unit is
50 µs (20000 = 1 s); the factory default is 0 (off).

Why it is needed: the MC02 board has its own link watchdog (after 100 ms of host silence the board itself
sends damping frames to the motors), but the board is powered from the USB port of the robot computer.
Unplugging USB powers the board off, so its watchdog cannot act and the motors keep holding the last MIT
command they received. canTimeout does not depend on the board: it is the last-resort deadman. While the
board is alive it never fires, because the damping frames (one every 10 ms) keep feeding it; it releases the
joints only after the link is lost completely. The value is therefore "how long the joints hold their
position after the link is lost".

`can_timeout` in the robot config is the reference value. tools/deploy_policy.py reads every motor back at
startup and refuses to run on a mismatch (the same rule as for set_torque.py).

Usage (on the robot computer; stop the bridge / deploy process first):
  python tools/set_can_timeout.py --check          # read back and compare only (run this first)
  python tools/set_can_timeout.py --write          # write to RAM (back to the factory 0 after power-off)
  python tools/set_can_timeout.py --write --save   # write and save in the motor <- the normal choice
  python tools/set_can_timeout.py --config humanoid_arm/humanoid_arm.yaml --check   # another robot
A config without a `can_timeout` key is skipped.
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import robstride_mit as rm                                   # noqa: E402


def _buses(cfg):
    buses = {}
    for j in cfg.get("joints", []):
        if j.get("enabled", True):
            buses.setdefault(j["bus"], []).append(j)
    return buses


def _want(cfg):
    v = cfg.get("can_timeout")
    return None if v is None else int(v)


def check_all(cfg, verbose=True):
    """Read every motor back and compare -> list of mismatch descriptions (empty = all match).
    A config without a can_timeout key is not checked and returns []."""
    want = _want(cfg)
    if want is None:
        return []
    bad = []
    for ch, joints in sorted(_buses(cfg).items()):
        leg = rm.LegBus(ch, [j["id"] for j in joints], models=rm.models_from_cfg(cfg, ch))
        try:
            if verbose:
                print(f"\n--- {ch} ---")
            for j in joints:
                rb = leg.read_u32_param(j["id"], rm.IDX_CAN_TIMEOUT)
                ok = rb == want
                if verbose:
                    print(f"  id={j['id']:<2} ({j['motor']}) {j['name']:<22} read back {rb}  expected {want}"
                          f"  {'OK' if ok else '!! MISMATCH'}")
                if not ok:
                    bad.append(f"{j['name']}: read back {rb}, expected {want}")
        finally:
            leg.shutdown()
    return bad


def write_all(cfg, save=False, verbose=True):
    """Write the value (and optionally save it), then read every motor back -> number of mismatches."""
    want = _want(cfg)
    if want is None:
        raise SystemExit("the config has no can_timeout key: nothing to write")
    n_bad = 0
    for ch, joints in sorted(_buses(cfg).items()):
        leg = rm.LegBus(ch, [j["id"] for j in joints], models=rm.models_from_cfg(cfg, ch))
        try:
            if verbose:
                print(f"\n--- {ch} ---")
            for j in joints:
                leg.write_u32_param(j["id"], rm.IDX_CAN_TIMEOUT, want)
                rb = leg.read_u32_param(j["id"], rm.IDX_CAN_TIMEOUT)
                ok = rb == want
                if ok and save:
                    leg.save_params(j["id"])
                if verbose:
                    print(f"  id={j['id']:<2} ({j['motor']}) {j['name']:<22} wrote {want}, read back {rb}"
                          f"  {'OK' if ok else '!! MISMATCH, do not continue'}{'  saved' if ok and save else ''}")
                if not ok:
                    n_bad += 1
        finally:
            leg.shutdown()
    return n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="read back and compare only; writes nothing")
    ap.add_argument("--write", action="store_true",
                    help="write canTimeout to the motors (RAM only unless --save is given)")
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
    want = _want(cfg)
    if want is None:
        print(f"{cfg_name} has no can_timeout key: skipped (the motor-side timeout is left as it is)")
        return 0
    print(f"{cfg_name} can_timeout = {want} ({want * 50 / 1000:.0f} ms; 0 = off)")
    rm.MC02_PORT = (cfg.get("mc02") or {}).get("port")

    if args.write:
        n_bad = write_all(cfg, save=args.save)
        if n_bad:
            print(f"\n!! {n_bad} motor(s) read back a different value after the write: do not run a policy")
            return 1
        print(f"\ncanTimeout = {want} on all motors ✓"
              + ("  (saved)" if args.save else "  (RAM only: back to the factory 0 after power-off; add --save)"))
        return 0
    bad = check_all(cfg)
    if bad:
        print(f"\n!! canTimeout differs from {cfg_name} on these joints:\n  " + "\n  ".join(bad))
        print("   Run --write --save, then check again. With a mismatch there is no motor-side deadman "
              "when the link is lost")
        return 1
    print(f"\ncanTimeout matches {cfg_name} on all motors ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
