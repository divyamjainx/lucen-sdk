#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/wheeled_base_accept.py - MuJoCo acceptance matrix for a wheeled_base policy (stand / drive / turn + disturbances).

Runs every ONNX through the same set of cases (tools/wheeled_base_sim2sim.run_episode), judges each case
pass / fail and prints a summary. A policy that looks perfect in the training simulator does not necessarily
transfer, so choose a checkpoint by this result, not by the training curves. Exit code 0 = every counted
case passed.

  python tools/wheeled_base_accept.py --policy wheeled_base/policies/drive.onnx
  python tools/wheeled_base_accept.py --policy a.onnx b.onnx c.onnx        # compare several checkpoints
  python tools/wheeled_base_accept.py --policy a.onnx --verbose            # print every case (default: failed ones only)

Pass criteria, by group:
  all groups:  no fall (anything but a wheel touching the ground, or a body tilt > 0.8 rad, counts as a fall)
  nominal      (observation noise at the training amplitude): steady-state vx error ≤ max(0.08, 12%)  m/s,
               wz error ≤ max(0.15, 15%) rad/s; standing (20 s and 90 s) drift ≤ 0.20 m, yaw drift ≤ 20°;
               in steady state (each command segment without its first 1.5 s) wheel torque jitter (rms after
               removing the mean) ≤ 1.0 N·m and longitudinal wheel slip rms ≤ 0.05 m/s
  disturbance  (action delay / gains / friction / added mass / centre of mass / noise / IMU bias / initial
               pitch): error thresholds doubled
  slope        (3° and 5°; the training range domain_rand.slope_max_deg in wheeled_base.yaml is 6°): standing
               (20 s and 90 s) drift ≤ 30 cm; driving / turning thresholds as in the disturbance group
  push:        no fall, nothing else
  extreme      (8 N·s sideways push): reported only, not counted in the total. This case is at the physical
               tip-over limit of the robot.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.wheeled_base_sim2sim import load_policy, parse_segments, run_episode  # noqa: E402

# The standing-on-a-slope cases enable MuJoCo's noslip solver. The default soft contact creeps under a sustained
# tangential force (0.32 cm/s on a 5° slope while the wheels do not turn), so without it the result would measure
# the simulator's contact creep rather than the policy. All other cases (driving on a slope included) use the
# default contact.
NOSLIP = 20

SEQ = "0:0:0, 3:0.8:0, 8:0:0, 11:-0.8:0, 16:0:0, 19:0:1.5, 24:0:-1.5, 29:0.6:1.0, 34:0:0"   # 38 s mixed route


def cases():
    """-> [(group, name, segments, duration, kwargs for run_episode, threshold multiplier; None = only "no fall")]"""
    out = []
    nom = dict(obs_noise=True)
    out.append(("nominal", "stand 20 s", "0:0:0", 20.0, nom, 1.0))
    # long stand: shows slow drift that a 20 s case cannot (a policy can hold its place for 30 s and still creep away)
    out.append(("nominal", "stand 90 s", "0:0:0", 90.0, nom, 1.0))
    for vx in (0.3, 0.6, 1.0, -0.3, -0.6, -1.0):
        out.append(("nominal", f"straight vx {vx:+.1f}", f"0:{vx}:0", 10.0, nom, 1.0))
    for wz in (1.0, 2.0, -1.0, -2.0):
        out.append(("nominal", f"turn in place wz {wz:+.1f}", f"0:0:{wz}", 10.0, nom, 1.0))
    for vx, wz in ((0.6, 1.0), (0.6, -1.0), (-0.6, 1.0), (-0.6, -1.0), (1.0, 0.5), (0.3, 2.0)):
        out.append(("nominal", f"drive and turn vx {vx:+.1f} wz {wz:+.1f}", f"0:{vx}:{wz}", 10.0, nom, 1.0))
    out.append(("nominal", "mixed route (start / stop / reverse / turn both ways)", SEQ, 38.0, nom, 1.0))
    out.append(("nominal", "hard reversals 1.0 <-> -1.0", "0:1.0:0, 4:-1.0:0, 8:1.0:0, 12:0:0", 16.0, nom, 1.0))

    for n in (1, 2, 3):
        out.append(("disturbance", f"action delay {n} step(s)", SEQ, 38.0, dict(delay=n, obs_noise=True), 2.0))
    for kp, kd in ((0.8, 1.0), (1.2, 1.0), (1.0, 0.7), (1.0, 1.3), (0.8, 0.7), (1.2, 1.3)):
        out.append(("disturbance", f"gains kp×{kp} kd×{kd}", SEQ, 38.0, dict(kp_scale=kp, kd_scale=kd, obs_noise=True), 2.0))
    for mu in (0.4, 0.6, 1.2):
        out.append(("disturbance", f"friction {mu}", SEQ, 38.0, dict(friction=mu, obs_noise=True), 2.0))
    for add in (-0.3, 1.0):
        out.append(("disturbance", f"base mass {add:+.1f} kg", SEQ, 38.0, dict(base_mass_add=add, obs_noise=True), 2.0))
    for com in ((0.03, 0, 0), (-0.03, 0, 0), (0, 0.02, 0), (0, 0, 0.03), (0.03, 0.02, 0.03), (-0.03, -0.02, -0.03)):
        out.append(("disturbance", f"base centre of mass offset {tuple(int(c * 1000) for c in com)} mm", SEQ, 38.0,
                    dict(base_com=com, obs_noise=True), 2.0))
    out.append(("disturbance", "all combined (delay 2 / kp×0.8 kd×0.7 / friction 0.6 / +1 kg / centre of mass 30 mm forward)", SEQ, 38.0,
                dict(delay=2, kp_scale=0.8, kd_scale=0.7, friction=0.6, base_mass_add=1.0,
                     base_com=(0.03, 0, 0), obs_noise=True), 2.0))
    out.append(("disturbance", "observation noise ×2.5", SEQ, 38.0, dict(obs_noise=2.5), 2.0))
    out.append(("disturbance", "observation noise at amplitudes common for legged robots (gyro 0.2 / gravity 0.05 / joint speed 1.5)", SEQ, 38.0,
                dict(obs_noise=dict(base_ang_vel=0.2, projected_gravity=0.05, leg_pos=0.01, leg_vel=1.5, wheel_vel=1.5)), 2.0))
    for gx, gy in ((0.03, 0.0), (-0.03, 0.0), (0.0, 0.03), (0.0, -0.03)):
        out.append(("disturbance", f"IMU bias on projected gravity x {gx:+.2f} y {gy:+.2f} (about 1.7°)", SEQ, 38.0,
                    dict(imu_bias=(gx, gy), obs_noise=True), 2.0))
    for pitch in (0.15, -0.15, 0.3, -0.3):
        out.append(("disturbance", f"initial pitch {pitch:+.2f} rad", "0:0:0", 8.0, dict(init_pitch=pitch, obs_noise=True), 2.0))

    # slopes: uphill direction 0° = robot faces uphill, 180° = faces downhill, 90° = stands across the slope (left side higher)
    for deg in (3.0, 5.0):
        for d, name in ((0, "facing uphill"), (180, "facing downhill"), (90, "across the slope")):
            out.append(("slope", f"{deg:.0f}° slope, stand 20 s ({name})", "0:0:0", 20.0,
                        dict(slope_deg=deg, slope_dir_deg=d, obs_noise=True, noslip=NOSLIP), 1.0))
    for d, name in ((0, "facing uphill"), (180, "facing downhill")):
        out.append(("slope", f"5° slope, stand 90 s ({name})", "0:0:0", 90.0,
                    dict(slope_deg=5.0, slope_dir_deg=d, obs_noise=True, noslip=NOSLIP), 1.0))
    for d, name in ((0, "uphill"), (180, "downhill"), (90, "across")):
        out.append(("slope", f"5° slope, drive 0.5 m/s ({name})", "0:0.5:0", 10.0,
                    dict(slope_deg=5.0, slope_dir_deg=d, obs_noise=True), 2.0))
    out.append(("slope", "5° slope, turn in place 1.0 rad/s", "0:0:1.0", 12.0, dict(slope_deg=5.0, obs_noise=True), 2.0))
    out.append(("slope", "5° slope, mixed route", SEQ, 38.0, dict(slope_deg=5.0, slope_dir_deg=45.0, obs_noise=True), 2.0))

    for imp in (3.0, 5.0, 8.0):
        for deg, name in ((0, "forward"), (180, "backward"), (90, "left"), (270, "right")):
            # The 8 N·s sideways push is not counted: the wheels cannot move sideways, so the robot can only resist
            # it. 8 N·s is a lateral speed of 0.77 m/s, and the critical speed for tipping over one wheel is
            # 0.81 m/s (centre of mass 0.31 m high, half track 0.148 m): this is the physical limit.
            group = "extreme" if imp >= 8.0 and deg in (90, 270) else "push"
            out.append((group, f"standing, pushed {name} with {imp:.0f} N·s", "0:0:0", 8.0,
                        dict(push=(imp, 3.0, np.radians(deg)), obs_noise=True), None))
    for imp in (3.0, 5.0):
        for deg, name in ((0, "forward"), (180, "backward"), (90, "left")):
            out.append(("push", f"driving 0.6 m/s, pushed {name} with {imp:.0f} N·s", "0:0.6:0", 8.0,
                        dict(push=(imp, 4.0, np.radians(deg)), obs_noise=True), None))
    return out


def judge(res, tol, slope=False):
    """-> (passed?, reason). slope: the case is on a slope, where the standing-drift threshold is relaxed to
    30 cm and the steady-state wheel torque / slip checks are skipped (the wheels have to push against the
    slope all the time)."""
    if not res["survived"]:
        return False, f"fell @ {res['fell_at']:.1f} s"
    if tol is None:
        return True, ""
    why = []
    # in steady state the wheels should neither jitter nor slip (checked for nominal cases only: tol == 1)
    if tol == 1.0 and not slope:
        if res.get("wheel_tau_jitter_steady", 0.0) > 1.0:
            why.append(f"steady-state wheel torque jitter {res['wheel_tau_jitter_steady']:.2f} N·m")
        if res.get("slip_rms_steady", 0.0) > 0.05:
            why.append(f"steady-state slip rms {res['slip_rms_steady']:.3f} m/s")
    for s in res["segments"]:
        ex, ew = abs(s["vx"] - s["cmd_vx"]), abs(s["wz"] - s["cmd_wz"])
        if ex > tol * max(0.08, 0.12 * abs(s["cmd_vx"])):
            why.append(f"vx {s['cmd_vx']:+.1f}->{s['vx']:+.2f}")
        if ew > tol * max(0.15, 0.15 * abs(s["cmd_wz"])):
            why.append(f"wz {s['cmd_wz']:+.1f}->{s['wz']:+.2f}")
        if "drift_m" in s and s["t0"] == 0.0 and res["duration"] >= 20.0 and len(res["segments"]) == 1:
            if s["drift_m"] > 0.20 * tol * (1.5 if slope else 1.0):
                why.append(f"drift {s['drift_m'] * 100:.0f} cm")
            if abs(np.degrees(s["yaw_drift_rad"])) > 20.0 * tol:
                why.append(f"yaw drift {np.degrees(s['yaw_drift_rad']):+.0f}°")
    return not why, "; ".join(why)


_POLICY = {}


def _run(job):
    path, idx, seed, allow = job
    if path not in _POLICY:
        _POLICY[path] = load_policy(path, allow)
    pol, io = _POLICY[path]
    group, name, segs, dur, kw, tol = cases()[idx]
    res = run_episode(pol, io, parse_segments(segs), dur, seed=seed, **kw)
    ok, why = judge(res, tol, slope=(group == "slope"))
    return path, idx, seed, ok, why, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", nargs="+", required=True, help="one or more ONNX files to evaluate")
    ap.add_argument("--seeds", type=int, default=2, help="seeds per case (each seed draws different observation noise)")
    ap.add_argument("--workers", type=int, default=8, help="number of parallel worker processes")
    ap.add_argument("--verbose", action="store_true", help="print every case, not only the failed ones")
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="accept an ONNX without a manifest (e.g. a checkpoint from the middle of training)")
    args = ap.parse_args()
    for p in args.policy:                          # reject a bad manifest here in the main process, before the pool starts
        load_policy(p, args.allow_unstamped)

    cs = cases()
    jobs = [(p, i, s, args.allow_unstamped) for p in args.policy for i in range(len(cs)) for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(_run, jobs, chunksize=1)

    worst = 0
    for path in args.policy:
        mine = [r for r in results if r[0] == path]
        print(f"\n===== {path} =====")
        total = fails = 0
        for group in ("nominal", "disturbance", "slope", "push", "extreme"):
            rows = [r for r in mine if cs[r[1]][0] == group]
            bad = [r for r in rows if not r[3]]
            if group != "extreme":                  # the "extreme" group is reported only: not in the total, not in the exit code
                total += len(rows)
                fails += len(bad)
            print(f"[{group}] {len(rows) - len(bad)}/{len(rows)} passed" + ("  (reported only, not counted in the total)" if group == "extreme" else ""))
            for r in (rows if args.verbose else bad):
                res = r[5]
                seg = "  ".join(f"({s['cmd_vx']:+.1f},{s['cmd_wz']:+.1f})->({s['vx']:+.2f},{s['wz']:+.2f})"
                                for s in res["segments"][:4])
                print(f"    {'PASS' if r[3] else 'FAIL'}  {cs[r[1]][1]}  [seed {r[2]}]  {r[4]}   {seg}")
        allr = [r[5] for r in mine]
        nomr = [r[5] for r in mine if cs[r[1]][0] == "nominal"]
        stand = [r[5]["segments"][0] for r in mine if cs[r[1]][1] == "stand 20 s" and r[5]["segments"]]
        print(f"  total {total - fails}/{total} passed")
        if stand:
            print(f"  stand 20 s: drift {max(s['drift_m'] for s in stand) * 100:.1f} cm, "
                  f"yaw drift {max(abs(np.degrees(s['yaw_drift_rad'])) for s in stand):.1f}° (worst seed)")
        print(f"  nominal cases: pitch max {max(r['pitch_max_deg'] for r in nomr):.1f}°  roll max {max(r['roll_max_deg'] for r in nomr):.1f}°  "
              f"leg torque max {max(r['leg_tau_max'] for r in nomr):.1f}  wheel torque max {max(r['wheel_tau_max'] for r in nomr):.1f} N·m  "
              f"wheel speed max {max(r['wheel_speed_max'] for r in nomr):.1f} rad/s")
        print(f"                 actions at the clip {np.mean([r['a_sat_frac'] for r in nomr]) * 100:.1f}%  step-to-step change rms {np.mean([r['a_rate_rms'] for r in nomr]):.3f}  "
              f"hip roll deviation max {max(r['hip_roll_dev_max'] for r in nomr):.3f}  hip yaw deviation max {max(r['hip_yaw_dev_max'] for r in nomr):.3f} rad  "
              f"wheel off the ground {np.mean([r['wheel_air_frac'] for r in nomr]) * 100:.1f}%")
        print(f"                 wheels in steady state: torque jitter {np.mean([r['wheel_tau_jitter_steady'] for r in nomr]):.2f} (max "
              f"{max(r['wheel_tau_jitter_steady'] for r in nomr):.2f}) N·m  torque rms {np.mean([r['wheel_tau_rms_steady'] for r in nomr]):.2f} N·m  "
              f"slip rms {np.mean([r['slip_rms_steady'] for r in nomr]):.3f} m/s   slip max incl. transients {max(r['slip_max'] for r in nomr):.2f} m/s")
        print(f"  all cases: leg torque max {max(r['leg_tau_max'] for r in allr):.1f}  wheel torque max {max(r['wheel_tau_max'] for r in allr):.1f} N·m")
        worst = max(worst, fails)
    sys.exit(1 if worst else 0)


if __name__ == "__main__":
    main()
