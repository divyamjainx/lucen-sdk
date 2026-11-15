#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/calib_arm.py - arm calibration with the motors disabled: zero offsets, direction check, mark calibrated.

Every joint row of the robot config maps the angle reported by the motor to the joint angle of the model:
    q = sign * wrap_pi(raw - offset)
sign and offset belong to one physical robot. This tool measures them and writes them into the config. It only
reads motor angles and never enables a motor; the arm is moved by hand.
While the config says `calibrated: false`, the bridge refuses send mode (no enable, no torque). Set it to true with
--mark-calibrated once zero offsets and directions are verified. Calibrate again after replacing a motor or
re-assembling a joint (set calibrated back to false in the config first).

Run on the robot computer, with the arm powered and the bridge stopped.

humanoid_arm - both arms are handled together. Zero pose: both arms hanging naturally, grippers pointing down and
fully open, the two fingers of each gripper side by side along the line through the two shoulders.
  python tools/calib_arm.py --config humanoid_arm/humanoid_arm.yaml --zero-here all      # 1. in the zero pose: sets the zero of all 16 joints
  python tools/calib_arm.py --config humanoid_arm/humanoid_arm.yaml --signs              # 2. direction check; a reversed sign is flipped
  python tools/calib_arm.py --config humanoid_arm/humanoid_arm.yaml --live               # 3. optional: print q, one line per arm
  python tools/calib_arm.py --config humanoid_arm/humanoid_arm.yaml --mark-calibrated    # 4. all verified: calibrated -> true
  python tools/calib_arm.py --config humanoid_arm/humanoid_arm.yaml --flip l_knee_joint  # flip the sign of one joint by hand

Arm (6+1 DoF) - zero pose = stowed pose: upper arm lying flat, forearm folded back on top of the upper arm, wrist
held straight, gripper fully open.
  python tools/calib_arm.py --config Arm/Arm.yaml --zero-here all          # 1. in the stowed pose: sets the zero of all 7 joints
  python tools/calib_arm.py --config Arm/Arm.yaml --signs                  # 2. direction check; a reversed sign is flipped
  python tools/calib_arm.py --config Arm/Arm.yaml --mark-calibrated        # 3. calibrated -> true
Leader + follower pair (--config Arm/Arm_pair.yaml, joint names carry the prefix leader_ / follower_): same
commands, both buses are handled together; for one arm only add --bus mc02:1 (leader) or --bus mc02:0 (follower).

Other options:
  --zero-here J1[,J2...]   make the current pose the zero of the listed joints only
  --flip J1[,J2...]        flip the sign of the listed joints; the offset, and so the zero pose, stays as it is
  --live                   print q with the current sign / offset; "!" marks a joint outside its range
Every change to the config keeps a backup next to it (<config>.bak). The bridge reads the new values when it starts.

Two-stop procedure (no action option): for joints that have a mechanical end stop at both ends of their range.
Each joint is pushed by hand to both end stops and the motor angle is read at each one.
  python tools/calib_arm.py --config <config>              # guided; prints the suggested sign / offset of every joint
  python tools/calib_arm.py --config <config> --write      # same, and writes them into the config
  - range that starts or ends at 0: the zero pose is that end stop, the offset is the reading there, and the sign
    follows from whether the reading rises or falls when the joint is pushed into its range
  - symmetric range: the offset is the midpoint of the two readings. Push to the negative end stop first, then to
    the positive one; a rising reading gives sign = +1. The tool describes the positive direction of each joint.
    If unsure, push either way, check afterwards with --live and correct the sign with --flip
  - gripper: the two stops are fully open and fully closed; the offset is taken at fully open and the range is
    set to the measured travel
  - the measured travel is compared with the range in the config; a difference above 0.15 rad means that an end
    stop was not reached, the wrong joint was moved, or the range in the model is wrong
humanoid_arm has no mechanical end stops: use --zero-here for it.
"""
import argparse
import math
import os
import re
import shutil
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import robstride_mit as rm   # noqa: E402

# Positive joint direction in words, shown by the two-stop procedure (zero pose; X forward, Y left, Z up).
# Joints that are not listed use the motion from the direction-check tables below.
HINT = {
    "Link1_joint": "shoulder yaw, axis -Z: positive = **clockwise** seen from above; "
                   "negative end stop = fully counter-clockwise seen from above",
    "Link2_joint": "shoulder pitch, axis +Y, range [-180°, 0]: zero = the fully folded end stop; "
                   "every value in the range is negative",
    "Link3_joint": "elbow, axis +Y, range [0, 150°]: zero = the fully folded end stop; every value in the range is positive",
    "Link4_joint": "wrist pitch, axis +Y: positive = right-hand rule about +Y (tool end pitches down / forward); "
                   "negative end stop = fully the other way",
    "Link5_joint": "wrist yaw, axis -Z: positive = same direction as Link1 (clockwise seen from above); "
                   "negative end stop = fully the other way",
    "Link6_joint": "wrist roll, axis -X: positive = right-hand rule about -X; negative end stop = fully the other way",
    "gripper_joint": "gripper: stop 1 = fully open, stop 2 = fully closed "
                     "(the offset is taken at fully open, range = measured travel)",
}


def read_raw(leg, mid, n=5, wrap=True):
    """Median of n readings of the motor angle. wrap=True folds it into (-π, π] (used for offsets); wrap=False returns
    the raw multi-turn angle (used for travel: on a joint that travels more than 180° the difference of two folded
    readings is ambiguous by 2π, the difference of multi-turn angles read in one session is not)."""
    vals = []
    for _ in range(n):
        v = leg.read_float_param(mid, rm.IDX_MECH_POS)
        if v is not None:
            vals.append(rm.wrap_pi(v) if wrap else v)
        time.sleep(0.02)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def ask(prompt):
    try:
        input(prompt)
    except EOFError:
        sys.exit("\naborted")


def short(name):
    return name[:-6] if name.endswith("_joint") else name


# Direction check for humanoid_arm: joint name without the l_ / r_ prefix -> (start pose, motion to perform, sign of q
# for this motion on the left arm, sign on the right arm).
# "Forward" = the front of the robot = the side the elbows bend toward; "outward" = away from the column the arms are
# mounted on. The signs follow from the forward kinematics of the model; the motions are worded so that they do not
# depend on where the operator stands.
SIGN_CHECK = {
    "hip_pitch_joint": ("arm hanging naturally", "swing the whole arm forward (like the forward arm swing when walking)", -1, -1),
    "hip_roll_joint": ("arm hanging naturally", "raise the whole arm sideways, away from the column (like a lateral raise)", -1, +1),
    "hip_yaw_joint": ("elbow bent to about 90°, forearm pointing forward, supported by hand",
                      "keeping the elbow bent, swing the forearm outward (the upper arm turns about its own axis)", +1, -1),
    "knee_joint": ("arm hanging naturally", "bend the elbow so that the forearm rises forward", -1, -1),
    "link4_joint": ("arm hanging naturally", "bend the wrist forward (the gripper tilts up toward the front)", +1, +1),
    "link5_joint": ("arm hanging naturally", "bend the wrist outward (the gripper tilts up, away from the column)", -1, +1),
    "link6_joint": ("arm hanging naturally",
                    "turn the gripper about the forearm axis so that its outer finger (the one away from the column) moves forward", +1, -1),
    "gripper_joint": ("gripper fully open", "close the gripper (bring the two fingers together)", +1, +1),
}


# Direction check for the Arm (6+1 DoF): joint name -> (start pose, motion to perform, sign of q for this motion).
# Every step starts from the stowed pose (= zero pose: upper arm lying flat, forearm folded back on top of the upper arm).
# The motions are worded so that they do not depend on where the operator stands ("seen from directly above", "raise",
# "unfold"); the signs follow from the forward kinematics of the model.
SIGN_CHECK_6DOF = {
    "joint1": ("stowed pose", "seen from directly above, turn the whole arm clockwise about the base", +1),
    "joint2": ("stowed pose", "raise the upper arm from its flat position (the elbow moves up; let the forearm come along, "
                              "do not unfold it)", +1),
    "joint3": ("stowed pose", "unfold the forearm away from the upper arm (like opening a folding knife; keep the upper arm still "
                              "and do not let the wrist swing)", +1),
    "joint4": ("stowed pose, wrist held straight", "tilt the wrist up so that the gripper rises toward the ceiling (same turning "
                                                   "direction as unfolding the forearm)", +1),
    "joint5": ("stowed pose, wrist held straight", "seen from directly above, swing the gripper clockwise about the wrist "
                                                   "(keep the forearm still)", +1),
    "joint6": ("stowed pose, wrist held straight", "looking from behind the wrist along the direction the gripper points, "
                                                   "twist the gripper clockwise", +1),
    "gripper_joint": ("gripper fully open", "close the gripper (bring the two fingers together)", +1),
}


def base_6dof(name):
    """Joint name of a 6+1 DoF Arm -> its key in SIGN_CHECK_6DOF (the name itself on a single arm; on a pair, the name
    without its leader_ / follower_ prefix). Any other joint -> None."""
    if name in SIGN_CHECK_6DOF:
        return name
    if name[:2] in ("l_", "r_") and name[2:] in SIGN_CHECK:  # l_gripper_joint / r_gripper_joint belong to the humanoid_arm table
        return None
    return next((b for b in SIGN_CHECK_6DOF if name.endswith("_" + b)), None)


def sign_check_row(name):
    """-> (arm label, start pose, motion, sign of q for this motion), or None if the joint is in neither table."""
    b = base_6dof(name)
    if b is not None:
        return ((name[:-len(b) - 1] + " arm ") if name != b else "",) + SIGN_CHECK_6DOF[b]
    side = name[:2]
    if side in ("l_", "r_") and name[2:] in SIGN_CHECK:
        prep, move, want_l, want_r = SIGN_CHECK[name[2:]]
        return ("left arm " if side == "l_" else "right arm ", prep, move, want_l if side == "l_" else want_r)
    return None


def hint(name):
    """Positive joint direction in words for the two-stop procedure; joints missing from HINT use the motion of the
    direction-check tables."""
    if name in HINT:
        return HINT[name]
    row = sign_check_row(name)
    if row:
        return f"{'positive' if row[3] > 0 else 'negative'} direction = start pose: {row[1]}; motion: {row[2]}"
    return ("(no description available: find the positive direction with mirror_viewer.py in watch mode; "
            "a wrong sign can be corrected afterwards with --flip)")


def check_signs(cfg, path, legs, ask=input, log=print):
    """Direction check, joint by joint, with the motors disabled: record the start angle, let the operator perform the
    described motion and see which way the motor angle changed. If that disagrees with the positive direction of the
    model, the sign in the yaml is flipped; the offset is left alone, so the zero pose does not change. The check also
    verifies that the joint that moved is the one being checked, which catches a motor id assigned to the wrong joint.
    Returns (flipped, skipped)."""
    rows = [j for j in cfg["joints"] if j.get("enabled", True) and j["bus"] in legs]

    def snap():
        return {j["name"]: read_raw(legs[j["bus"]], int(j["id"]), n=3, wrap=False) for j in rows}

    flipped, skipped = [], []
    for k, j in enumerate(rows):
        name, row = j["name"], sign_check_row(j["name"])
        if row is None:
            log(f"[{k + 1}/{len(rows)}] {name}: no motion description for this joint, skipped "
                "(check it with mirror_viewer.py in watch mode)")
            skipped.append(name)
            continue
        arm, prep, move, want = row
        while True:
            a = ask(f"[{k + 1}/{len(rows)}] {arm}{name} (id {j['id']}) - start pose: {prep}. "
                    "Press Enter when in this pose (s = skip) ").strip().lower()
            if a == "s":
                skipped.append(name)
                break
            r0 = snap()
            ask(f"      Motion (this joint only): {move}. Move it by more than 20°, hold it there, then press Enter ")
            r1 = snap()
            d = {n: (r1[n] - r0[n]) if r0[n] is not None and r1[n] is not None else 0.0 for n in r0}
            mover = max(d, key=lambda n: abs(d[n]))
            if r0[name] is None or r1[name] is None:
                log("      !! no reading from this motor, skipped")
                skipped.append(name)
                break
            if abs(d[name]) < 0.15 or (mover != name and abs(d[mover]) > 1.5 * abs(d[name])):
                log(f"      {name} moved only {d[name]:+.2f} rad"
                    + (f", the joint that moved most is {mover} ({d[mover]:+.2f} rad)" if mover != name else "")
                    + " - move this joint only and move it further; starting this joint again")
                continue
            if int(j["sign"]) * d[name] * want > 0:
                log(f"      direction correct (sign {int(j['sign']):+d})")
            else:
                patch_yaml(path, name, sign=-int(j["sign"]))
                j["sign"] = -int(j["sign"])
                flipped.append(name)
                log(f"      direction reversed -> flipped to sign {int(j['sign']):+d}")
            break
    return flipped, skipped


def calibrate(cfg, leg, bus):
    rows = []
    for j in cfg["joints"]:
        if j["bus"] != bus or not j.get("enabled", True):
            continue
        name, mid = j["name"], int(j["id"])
        lo, hi = float(j["range"][0]), float(j["range"][1])
        one_sided_zero = None                                 # end of the range that is the zero pose: "lo" / "hi" / None
        if abs(hi) < 1e-6:
            one_sided_zero = "hi"
        elif abs(lo) < 1e-6:
            one_sided_zero = "lo"
        print(f"\n=== {name} (id {mid}, {j['motor']}) range [{lo:+.3f}, {hi:+.3f}] ===")
        print("   " + hint(name))
        if j.get("kind") == "gripper":
            ask("   Move the gripper to **fully open**, then press Enter > ")
            r_open = read_raw(leg, mid, wrap=False)
            ask("   Move the gripper to **fully closed**, then press Enter > ")
            r_close = read_raw(leg, mid, wrap=False)
            if r_open is None or r_close is None:
                print("   !! no reading, skipped"); continue
            travel = r_close - r_open
            r_open = rm.wrap_pi(r_open)
            sign = 1 if travel > 0 else -1
            rows.append(dict(j, sign=sign, offset=round(r_open, 5), range=[0.0, round(abs(travel), 4)],
                             _note=f"fully open raw {r_open:+.4f}, fully closed {r_close:+.4f}, travel {abs(travel):.3f} rad"))
            print(f"   -> sign {sign:+d} offset {r_open:+.4f} range [0, {abs(travel):.3f}]")
            continue
        if one_sided_zero:
            ask("   Push the joint to its **zero end stop** (the end of its range where q = 0), then press Enter > ")
            r_zero = read_raw(leg, mid, wrap=False)
            ask("   Push it into its range, all the way to the **other end stop**, then press Enter > ")
            r_far = read_raw(leg, mid, wrap=False)
            if r_zero is None or r_far is None:
                print("   !! no reading, skipped"); continue
            travel = r_far - r_zero                           # difference of multi-turn angles: no 2π ambiguity
            r_zero, r_far = rm.wrap_pi(r_zero), rm.wrap_pi(r_far)
            span = hi - lo
            # zero at the hi end: q goes negative into the range, so a rising raw angle means sign = -1;
            # zero at the lo end: q goes positive, so a rising raw angle means sign = +1
            sign = (-1 if travel > 0 else 1) if one_sided_zero == "hi" else (1 if travel > 0 else -1)
            offset = r_zero
            note = f"zero raw {r_zero:+.4f}, far end {r_far:+.4f}, mechanical travel {abs(travel):.3f} vs range {span:.3f}"
        else:
            ask("   Push the joint to its **negative** end stop (see the description above), then press Enter > ")
            r_lo = read_raw(leg, mid, wrap=False)
            ask("   Push the joint to its **positive** end stop, then press Enter > ")
            r_hi = read_raw(leg, mid, wrap=False)
            if r_lo is None or r_hi is None:
                print("   !! no reading, skipped"); continue
            travel = r_hi - r_lo                               # difference of multi-turn angles: no 2π ambiguity
            span = hi - lo
            sign = 1 if travel > 0 else -1
            offset = rm.wrap_pi(r_lo + travel / 2.0)         # midpoint (symmetric range)
            r_lo, r_hi = rm.wrap_pi(r_lo), rm.wrap_pi(r_hi)
            note = f"negative end raw {r_lo:+.4f}, positive end {r_hi:+.4f}, mechanical travel {abs(travel):.3f} vs range {span:.3f}"
        flag = ""
        if abs(abs(travel) - span) > 0.15:
            flag = (f"  !! travel differs by {abs(travel) - span:+.3f} rad: "
                    "end stop not reached / wrong joint moved / wrong range in the model")
        print(f"   -> sign {sign:+d} offset {offset:+.4f}   ({note}){flag}")
        rows.append(dict(j, sign=sign, offset=round(offset, 5), _note=note + flag))
    return rows


def live(cfg, legs):
    """legs: {bus: LegBus}. One line per bus, refreshed in place (one line for a single arm, two lines for two arms)."""
    rows = [(b, [j for j in cfg["joints"] if j["bus"] == b and j.get("enabled", True)]) for b in legs]
    print("Printing q continuously with the sign / offset of the current config (model coordinates, rad; ! = outside the joint range). "
          "Every joint reads about 0 in the zero pose; move a joint by hand to check its direction. Ctrl+C to quit")
    first = True
    try:
        while True:
            out = []
            for b, js in rows:
                parts = []
                for j in js:
                    r = read_raw(legs[b], int(j["id"]), n=1)
                    q = j["sign"] * rm.wrap_pi(r - j["offset"]) if r is not None else float("nan")
                    lo, hi = j["range"]
                    mark = "" if (r is None or lo - 0.05 <= q <= hi + 0.05) else "!"
                    parts.append(f"{short(j['name'])}={q:+.3f}{mark}")
                out.append("  ".join(parts) + "   ")
            if len(out) == 1:
                sys.stdout.write("\r" + out[0])
            else:
                sys.stdout.write(("" if first else f"\033[{len(out)}A") + "".join("\r\033[K" + ln + "\n" for ln in out))
            first = False
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


def patch_yaml(path, name, sign=None, offset=None):
    """Change the sign and / or the offset of one joint row of the yaml (None = leave as it is). Returns (old_sign, old_offset)."""
    shutil.copy(path, path + ".bak")
    txt = open(path, encoding="utf-8").read()
    pat = re.compile(r"(\{name: %s,.*?sign: )([+-]?\d+)(, offset: )([+-]?[\d.]+)" % re.escape(name))
    m = pat.search(txt)
    if not m:
        sys.exit(f"{name} not found in {os.path.basename(path)}")
    old = (int(m.group(2)), float(m.group(4)))
    new_sign = old[0] if sign is None else sign
    new_off = old[1] if offset is None else offset
    txt = txt[:m.start()] + f"{m.group(1)}{new_sign:+d}{m.group(3)}{new_off:+.5f}" + txt[m.end():]
    open(path, "w", encoding="utf-8").write(txt)
    return old


def write_yaml(path, rows):
    shutil.copy(path, path + ".bak")
    txt = open(path, encoding="utf-8").read()
    for r in rows:
        pat = re.compile(r"(\{name: %s,.*?sign: )([+-]?\d+)(, offset: )([+-]?[\d.]+)(.*?range: \[)([^\]]*)(\])" % re.escape(r["name"]))
        m = pat.search(txt)
        if not m:
            print(f"  !! no row for {r['name']} found in {os.path.basename(path)}: edit it by hand"); continue
        rng = m.group(6) if r.get("kind") != "gripper" else f"{r['range'][0]}, {r['range'][1]}"
        txt = txt[:m.start()] + f"{m.group(1)}{r['sign']:+d}{m.group(3)}{r['offset']:+.5f}{m.group(5)}{rng}{m.group(7)}" + txt[m.end():]
    open(path, "w", encoding="utf-8").write(txt)


def mark_calibrated(path):
    """Change the top-level `calibrated: false` of the yaml to true (the bridge then accepts send mode). A config without
    this key already counts as calibrated."""
    txt = open(path, encoding="utf-8").read()
    new, n = re.subn(r"(?m)^calibrated:[ \t]*false\b", "calibrated: true", txt)
    if not n:
        print(f"{os.path.basename(path)} has no line `calibrated: false` (it is already true, or this config does not use the switch)")
        return False
    shutil.copy(path, path + ".bak")
    open(path, "w", encoding="utf-8").write(new)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, metavar="YAML",
                    help="humanoid_arm/humanoid_arm.yaml | Arm/Arm.yaml | Arm/Arm_pair.yaml")
    ap.add_argument("--bus", default=None, help="handle this bus only (default: every bus in the config)")
    ap.add_argument("--write", action="store_true", help="two-stop procedure: write the result into the config (a .bak backup is kept)")
    ap.add_argument("--live", action="store_true", help="print q continuously with the current sign / offset, to check directions")
    ap.add_argument("--zero-here", metavar="JOINTS",
                    help="make the current pose the zero of these joints: comma-separated names, or all (edits the config)")
    ap.add_argument("--flip", metavar="JOINTS",
                    help="flip the sign of these joints, comma-separated (edits the config, does not use the bus)")
    ap.add_argument("--signs", action="store_true",
                    help="direction check: move each joint by hand as prompted; a reversed sign is flipped automatically "
                         "(edits the config; motors stay disabled, the bridge must be stopped)")
    ap.add_argument("--mark-calibrated", action="store_true",
                    help="zero offsets and directions are verified: change calibrated: false to true in the config "
                         "(does not use the bus)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg_name = os.path.basename(args.config)
    rm.MC02_PORT = (cfg.get("mc02") or {}).get("port")
    enabled = [j for j in cfg["joints"] if j.get("enabled", True)]
    buses = [args.bus] if args.bus else list(dict.fromkeys(j["bus"] for j in enabled))
    by_name = {j["name"]: j for j in cfg["joints"]}

    def pick(spec):
        names = [j["name"] for j in enabled if j["bus"] in buses] if spec.strip() == "all" \
            else [n.strip() for n in spec.split(",") if n.strip()]
        unknown = [n for n in names if n not in by_name]
        if unknown:
            sys.exit(f"no such joints in {cfg_name}: {unknown}")
        return [by_name[n] for n in names]

    if args.mark_calibrated:
        if mark_calibrated(args.config):
            print(f"{cfg_name} updated: calibrated: true (backup with the suffix .bak). Restart the bridge to load it, and use "
                  "the same file on every computer that runs the arm tools")
        return
    if args.flip:
        for j in pick(args.flip):
            old = patch_yaml(args.config, j["name"], sign=-int(j["sign"]))
            print(f"  {j['name']}: sign {old[0]:+d} -> {-old[0]:+d} (offset {old[1]:+.5f} unchanged)")
        print(f"{cfg_name} updated (backup with the suffix .bak); verify with --live"); return
    legs = {b: rm.LegBus(b, rm.ids_from_cfg(cfg, b), models=rm.models_from_cfg(cfg, b)) for b in buses}
    rows = []
    try:
        if args.zero_here:
            for j in pick(args.zero_here):
                if j["bus"] not in legs:
                    print(f"  {j['name']}: is on {j['bus']}, outside the --bus selection; skipped"); continue
                r = read_raw(legs[j["bus"]], int(j["id"]))
                if r is None:
                    print(f"  {j['name']}: no reading, skipped"); continue
                old = patch_yaml(args.config, j["name"], offset=r)
                print(f"  {j['name']}: offset {old[1]:+.5f} -> {r:+.5f} (current pose = 0; sign {old[0]:+d} unchanged)")
            print(f"{cfg_name} updated (backup with the suffix .bak); verify with --live"); return
        if args.live:
            live(cfg, legs); return
        if args.signs:
            print("Direction check: the motors are disabled (the bridge must be stopped), so the arm can be moved by hand. "
                  + ("Every step starts from the stowed pose (zero pose): upper arm lying flat, forearm folded back on top of it."
                     if any(base_6dof(j["name"]) for j in enabled) else
                     "\"Forward\" = the front of the robot = the side the elbows bend toward; \"outward\" = away from the column."))
            flipped, skipped = check_signs(cfg, args.config, legs)
            print(f"\nFlipped {len(flipped)}: {flipped or 'none'}" + (f"; skipped {skipped}" if skipped else ""))
            print(f"Sign changes are written to {cfg_name} (backup: {cfg_name}.bak). When every joint is checked and none was skipped, "
                  "run --mark-calibrated")
            return
        print("Arm powered, motors disabled (the bridge must be stopped). Push every joint by hand to both of its mechanical end stops "
              "and press Enter at each one to take the reading. A joint that does not reach an end stop is flagged by the travel check.")
        for b in buses:
            rows += calibrate(cfg, legs[b], b)
    finally:
        for leg in legs.values():
            leg.bus.shutdown()
    print("\n=== suggested joints table ===")
    for r in rows:
        print(f"  {r['name']:<14} sign {r['sign']:+d}  offset {r['offset']:+.5f}   # {r['_note']}")
    if args.write:
        write_yaml(args.config, rows)
        print(f"\nWritten to {args.config} (backup: {cfg_name}.bak). Next: check the directions with --live")
    else:
        print(f"\nAdd --write to store these values in {cfg_name}, or copy the sign / offset values above by hand")


if __name__ == "__main__":
    main()
