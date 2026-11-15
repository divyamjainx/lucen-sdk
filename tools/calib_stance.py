#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/calib_stance.py - zero-pose calibration of biped_Laika on the ground: 12 joint offsets and the IMU level.

Principle: the zero pose of the URDF is legs straight, soles flat on the ground, torso vertical. Put the real
robot in exactly that pose: every joint angle is then 0 by definition, so offset = the raw motor angle read
at that moment. In the same pose the IMU must read gravity = [0, 0, -1], which also gives mount_rpy_deg.

Calibrate with the robot standing on the ground, not hanging from the hoist. A hanging leg does not settle at
the mechanical zero; offsets taken that way are a few degrees off and the soles do not sit flat afterwards.

Posing requirements (posing by hand repeats to about +/-1.5 deg; points a-c decide the calibration quality):
  a. Both feet completely flat on level ground (fixes the sum of hip, knee and ankle pitch).
  b. 214.5 mm between the two sole centres, left-right symmetric (fixes the split between hip roll and ankle
     roll: both rotate about the x axis, so "feet flat" cannot tell them apart; the lateral foot position can).
  c. Both toes pointing straight ahead and parallel to each other (fixes hip yaw, which "feet flat" does not
     constrain at all).
  Also: legs straight (knees near 0), torso vertical (check with a spirit level on the torso), motors not enabled.
  Sole size for reference: 170 mm long (x from -65 to +105 mm), 80 mm wide.

Usage (on the robot computer; stop the bridge / deploy process first):
  python tools/calib_stance.py --repeat 3            # average of 3 independent poses (recommended), read-only
  python tools/calib_stance.py                       # single pose, read-only
  python tools/calib_stance.py --repeat 3 --write    # write the result to the config (a .bak copy is made first)
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import robstride_mit as rm                    # noqa: E402


def read_raw(cfg, n_samples=20):
    """Read the raw angle of the 12 joints without enabling the motors (median of several samples)."""
    buses = {}
    for j in cfg["joints"]:
        buses.setdefault(j["bus"], []).append(j)
    out = {}
    for ch, joints in buses.items():
        leg = rm.LegBus(ch, [j["id"] for j in joints])
        try:
            for j in joints:
                vals = []
                for _ in range(n_samples):
                    v = leg.read_float_param(j["id"], rm.IDX_MECH_POS, timeout=0.05)
                    if v is not None:
                        vals.append(v)
                    time.sleep(0.005)
                if len(vals) < n_samples // 2:
                    sys.exit(f"[err] id={j['id']} {j['name']}: too few readings ({len(vals)}/{n_samples})")
                out[j["name"]] = float(np.median(vals))
        finally:
            leg.bus.shutdown()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the result to biped_Laika/biped_Laika.yaml (a .bak copy is made first)")
    ap.add_argument("--skip-imu", action="store_true", help="calibrate the joint offsets only, leave the IMU mounting angles unchanged")
    ap.add_argument("--repeat", type=int, default=1,
                    help="number of independent poses to average, to reduce hand-posing noise (3 recommended)")
    args = ap.parse_args()

    path = os.path.join(ROOT, "biped_Laika/biped_Laika.yaml")
    cfg = yaml.safe_load(open(path))
    print("Pose: 1) both feet completely flat  2) 214.5 mm between the sole centres, left-right symmetric"
          "  3) toes pointing straight ahead and parallel\n"
          "      4) legs straight  5) torso vertical (spirit level)  6) motors not enabled")

    samples = []
    for k in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- pose {k+1}/{args.repeat} ---")
            if k > 0:
                print("Pose the robot again from scratch (lift it and set it down): only independent poses average out the noise")
        input(">>> Hold the pose still and press Enter to start sampling...")
        samples.append(read_raw(cfg))

    # circular mean over the poses, joint by joint (angles cannot be averaged arithmetically)
    raw, spread = {}, {}
    for j in cfg["joints"]:
        n = j["name"]
        vs = np.array([s[n] for s in samples])
        ref = vs[0]
        centered = np.array([rm.wrap_pi(v - ref) for v in vs])
        raw[n] = float(ref + centered.mean())
        spread[n] = float(np.degrees(centered.max() - centered.min()))

    print("\n=== Joint offsets (rad) ===")
    hdr = f"{'joint':<22}{'raw':>10}{'old':>10}{'new':>10}{'diff(°)':>10}"
    print(hdr + (f"{'spread(°)':>12}" if args.repeat > 1 else ""))
    new_off = {}
    for j in cfg["joints"]:
        n, old = j["name"], j["offset"]
        # in this pose the true joint angle is 0: q = sign * wrap(raw - offset) = 0, hence offset = raw
        new = float(rm.wrap_pi(raw[n]))
        new_off[n] = new
        delta = np.degrees(rm.wrap_pi(new - old))
        flag = "  <<" if abs(delta) > 2.0 else ""
        line = f"{n:<22}{raw[n]:>10.4f}{old:>10.5f}{new:>10.5f}{delta:>10.2f}"
        if args.repeat > 1:
            line += f"{spread[n]:>12.2f}"
        print(line + flag)
    if args.repeat > 1:
        worst = max(spread.values())
        print(f"\nPose repeatability: largest spread {worst:.2f}° "
              f"({'good' if worst < 1.0 else 'large; check posing points 2) and 3)'})")

    imu_rpy = None
    if not args.skip_imu:
        try:
            from tools.imu_reader import ImuReader, solve_mount_rpy, tilt_deg, mount_matrix
            print("\n=== IMU level (same pose) ===")
            r = ImuReader(cfg["imu"])
            time.sleep(2.5)
            gs = []
            for _ in range(60):
                _, g, age = r.latest()
                if age < 0.1:
                    gs.append(g)
                time.sleep(0.05)
            g = np.mean(gs, axis=0)
            g /= np.linalg.norm(g)
            print(f"measured gravity_base = {np.round(g, 4)}  residual tilt {tilt_deg(g):.3f}°")
            imu_rpy = solve_mount_rpy(g, cfg["imu"])
            g_fix = mount_matrix({**cfg["imu"], "mount_rpy_deg": imu_rpy}) @ (
                mount_matrix(cfg["imu"]).T @ g)
            if tilt_deg(g_fix) > 0.1:
                print("!! IMU self-check failed; the IMU mount angles are left unchanged")
                imu_rpy = None
            else:
                print(f"new mount_rpy_deg = [{imu_rpy[0]:.2f}, {imu_rpy[1]:.2f}, 0.0] "
                      f"(old {cfg['imu']['mount_rpy_deg']}), residual tilt after correction {tilt_deg(g_fix):.3f}°")
        except Exception as e:
            print(f"IMU read failed, skipping the IMU part: {e}")

    if not args.write:
        print("\n(read-only; if the values look right, run again with --write to save them to biped_Laika/biped_Laika.yaml)")
        return

    shutil.copy(path, path + ".bak")
    txt = open(path).read()
    for j in cfg["joints"]:
        n = j["name"]
        old_s = f"offset: {j['offset']!r}"
        # joints are written one per line in the yaml: find the line by joint name, then replace its offset field
        lines = txt.splitlines()
        for i, ln in enumerate(lines):
            if f"name: {n}," in ln:
                import re
                lines[i] = re.sub(r"offset:\s*-?[\d.eE+-]+",
                                  f"offset: {new_off[n]:+.5f}", ln)
        txt = "\n".join(lines) + ("\n" if txt.endswith("\n") else "")
    if imu_rpy is not None:
        import re
        txt = re.sub(r"mount_rpy_deg:\s*\[[^\]]*\]",
                     f"mount_rpy_deg: [{imu_rpy[0]:.2f}, {imu_rpy[1]:.2f}, 0.0]", txt)
    open(path, "w").write(txt)
    yaml.safe_load(open(path))               # make sure the file still parses
    print(f"\nWritten to {path} (backup: {path}.bak)")
    print("Next: compare the file with its .bak copy, then run the standing policy and check that both soles sit flat")


if __name__ == "__main__":
    main()
