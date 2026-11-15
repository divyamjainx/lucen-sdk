#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/set_motor_id.py - assign, check and verify RobStride motor CAN ids without enabling the motors.

No motor is enabled and nothing moves. The tool only sends RobStride communication type 0 (get id: the
motor answers with its MCU UID), type 7 (set CAN id, effective immediately), type 17 (read a parameter) and
type 22 (save parameters).

IMPORTANT: type 7 addresses a motor by its current id, so ALL motors on the bus that currently have that id
are changed together. New motors all leave the factory with the same id. Hence the one rule: at the moment
an id is changed, only one motor on the bus may have it - connect new motors one at a time.
The tool enforces this. Before every change it scans the bus and identifies each motor by its MCU UID: the
source id must belong to exactly one motor and the target id must be free, otherwise the change is refused.
After the change it scans again and accepts only that one motor having a new id.

Usage (stop the bridge / deploy process first - only one process can open an MC02 board; motors powered
with 48 V, not enabled):
  python tools/set_motor_id.py --bus mc02:0 --scan                    # query ids 0..127: id / UID / limit_torque
  python tools/set_motor_id.py --bus mc02:0 --to 3                    # only one motor on the bus: make it id 3
  python tools/set_motor_id.py --bus mc02:0 --from 127 --to 3         # several motors on the bus: name the id to change
  python tools/set_motor_id.py --config humanoid_arm/humanoid_arm.yaml --assign   # see below
  python tools/set_motor_id.py --config humanoid_arm/humanoid_arm.yaml --check
  python tools/set_motor_id.py --config humanoid_arm/humanoid_arm.yaml --verify
  python tools/set_motor_id.py --config Arm/Arm.yaml --assign         # 6+1-DoF arm: base to gripper = 1-7, all on CAN1
  python tools/set_motor_id.py --config Arm/Arm_pair.yaml --check     # leader + follower arms
  python tools/set_motor_id.py --config biped_Laika/biped_Laika.yaml --check
    --assign   guided assignment, one motor at a time in the order of the config
               (humanoid_arm: left arm 1-8 first, then right arm 9-16)
    --check    the ids answering on each bus must equal the config, with no duplicate id
    --verify   you turn each joint by hand and the tool checks that the expected id moves
    --port     MC02 serial port; default: mc02.port of the config, else /dev/mc02. Give it on macOS
               (/dev/cu.usbmodemXXXX) or to pick one of two boards on the same host (/dev/serial/by-id/...)

--assign finds the motor of each step in one of two ways (chosen automatically):
  - One more motor is connected per step (the only way while new motors still share the factory id): the
    UID that newly appears belongs to the joint of this step.
  - The whole chain is connected and all ids are distinct (e.g. motors that were numbered before): the tool
    asks you to turn the joint back and forth by hand and reads the mechanical angles to see which motor moves.
A new id is saved in the motor by default (type 22, kept across power cycles); --no-save changes RAM only.
When all ids are assigned, power the motors off and on, then run --check once more to confirm the ids were kept.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import robstride_mit as rm                                   # noqa: E402

SCAN_IDS = range(0, 128)        # value range of the CAN_ID parameter: 0..127
PARK_FROM = 100                 # temporary ids are taken from here upwards: clear of target ids and of the usual defaults 1 / 127
PER_ID, TAIL = 0.004, 0.15      # scan timing [s]: wait per id / extra wait at the end
WIGGLE_THRESH = 0.15            # rad: a joint turned by hand counts as "moved" beyond this
WIGGLE_STILL = 0.05             # rad: the other joints may move along by at most this
WIGGLE_TIMEOUT = 30.0
WIGGLE_PERIOD = 0.1
# limit_torque read back from a factory-new motor: the peak torque of its model. Any other value means that
# the motor is a different model, or that its torque limit has been written before.
FACTORY_LT = {"RS06": 36.0, "RS02": 17.0, "RS00": 14.0, "EL05": 6.0, "RS05": 5.5}
LT_TOL = 0.25                   # EL05 (6) vs RS05 (5.5): the tolerance must stay below their difference of 0.5
_sleep = time.sleep             # test hook


class Refuse(RuntimeError):
    """A precondition is not met; no motor has been touched."""


def scan(leg):
    """Query every id on the bus -> [(id, uid)] sorted by id. An id listed twice = two motors share it."""
    return sorted(leg.whois(SCAN_IDS, per_id=PER_ID, tail=TAIL))


def by_id(entries):
    out = {}
    for i, u in entries:
        out.setdefault(i, []).append(u)
    return out


def _short(uid):
    return uid[-8:]


def free_park_id(in_use, targets):
    for i in range(PARK_FROM, 127):
        if i not in in_use and i not in targets:
            return i
    raise Refuse(f"no free temporary id in {PARK_FROM}..126")


def renumber(leg, cur, new, uid=None, save=True, log=print):
    """Change the one motor with id=cur to id=new -> its UID. Raises Refuse when a precondition is not met (no
    motor touched) and RuntimeError when the bus does not look as expected after the change."""
    cur, new = int(cur), int(new)
    if not 1 <= new <= 127:
        raise Refuse(f"target id {new} is out of range (1..127)")
    before = scan(leg)
    ids = by_id(before)
    if cur not in ids:
        raise Refuse(f"no motor answers on id={cur} (ids answering: {sorted(ids) or 'none'})")
    if len(ids[cur]) != 1:
        raise Refuse(f"id={cur} is used by {len(ids[cur])} motors (UID …{' …'.join(_short(u) for u in ids[cur])}): "
                     f"type 7 would change all of them to {new}. Disconnect motors until only one has this id, then retry")
    if uid is not None and ids[cur][0] != uid:
        raise Refuse(f"id={cur} is now a different motor (UID …{_short(ids[cur][0])}; "
                     f"the one to change is …{_short(uid)})")
    uid = ids[cur][0]
    if cur == new:
        return uid
    # Query the target id three more times. If the full scan missed a motor on it, the change would put two
    # motors on one id, and that can only be undone by disconnecting one of them.
    if new in ids or leg.whois([new] * 3, per_id=0.01, tail=TAIL):
        raise Refuse(f"target id {new} is already used by a motor")
    leg.set_can_id(cur, new)
    want = sorted((new if (i, u) == (cur, uid) else i, u) for i, u in before)
    after = scan(leg)
    if after != want:
        after = scan(leg)                       # an answer gets lost now and then: scan once more before deciding
    if after != want:
        raise RuntimeError(f"the bus does not match after the id change: expected {want}, found {after}. "
                           f"Run --scan to see the actual state before doing anything else")
    if save:
        leg.save_params(new)
    log(f"  id {cur} -> {new}  UID …{_short(uid)}" + ("  saved" if save else "  (RAM only, lost at power-off)"))
    return uid


def find_mover(leg, ids, log=print):
    """Find the motor that is turned by hand: read the mechanical angle of every motor (0x7019, the motors stay
    disabled) and track each one's largest deviation from its start value. Exactly one moved -> its id, else None."""
    ids = list(ids)

    def read_all():
        out = {}
        for i in ids:
            for _ in range(3):
                out[i] = leg.read_float_param(i, rm.IDX_MECH_POS)
                if out[i] is not None:
                    break
        return out

    base = read_all()
    dead = [i for i in ids if base[i] is None]
    if dead:
        log(f"  cannot read the mechanical angle of id {dead}: finding the motor by hand motion is not possible")
        return None
    peak = {i: 0.0 for i in ids}
    last, streak = None, 0
    t_end = time.monotonic() + WIGGLE_TIMEOUT
    while time.monotonic() < t_end:
        _sleep(WIGGLE_PERIOD)
        for i, v in read_all().items():
            if v is not None:
                peak[i] = max(peak[i], abs(v - base[i]))
        rank = sorted(ids, key=lambda i: -peak[i])
        top = peak[rank[0]]
        second = peak[rank[1]] if len(rank) > 1 else 0.0
        if top > WIGGLE_THRESH and second < max(WIGGLE_STILL, 0.3 * top):
            streak = streak + 1 if rank[0] == last else 1
            last = rank[0]
            if streak >= 2:                     # two readings in a row: a second joint dragged along has time to show up
                return last
        else:
            last, streak = None, 0
    log("  not identified: " + ", ".join(f"id {i} moved {peak[i]:.2f} rad" for i in sorted(ids))
        + f" (one motor must move by more than {WIGGLE_THRESH} rad while the others stay almost still)")
    return None


def plan_from_cfg(cfg, bus):
    """Config joints -> [(name, id, motor)] for this bus, in the order of the joint table."""
    return [(j["name"], int(j["id"]), j["motor"]) for j in cfg["joints"] if j["bus"] == bus]


def assign(leg, plan, ask=input, log=print, save=True):
    """Guided id assignment, one motor at a time -> {name: (id, uid)} (only the joints confirmed in this run).
    The two ways of finding the motor are described under "--assign" in the module docstring."""
    targets = {t for _, t, _ in plan}
    done = {}                   # uid -> final id: confirmed in this run
    parked = []                 # (uid, temporary id, final id, name): moved back at the end
    result = {}

    def settled():
        return set(done) | {u for u, *_ in parked}

    def step(k, name, tgt, motor):
        entries = scan(leg)
        ids = by_id(entries)
        fresh = [(i, u) for i, u in entries if u not in settled()]
        if not fresh:
            log("  no new motor answers: is this motor connected? Is 48 V on? "
                "(tools/mc02_selftest.py reports LEC=3 when nothing answers on the bus)")
            return False
        lone = len(fresh) == 1
        if lone:
            cur, uid = fresh[0]
        else:
            clash = sorted({i for i, _ in fresh if len(ids[i]) > 1})
            if clash:
                log(f"  {len(fresh)} motors have no final id yet and id {clash} is a duplicate id (new motors with the same "
                    f"factory id are connected together): disconnect them until {name} is the only new motor")
                return False
            log(f"  {len(fresh)} motors have no final id yet (id {sorted(i for i, _ in fresh)}): "
                f"now turn {name} back and forth by hand; do not touch the other joints ...")
            cur = find_mover(leg, [i for i, _ in fresh], log=log)
            if cur is None:
                return False
            uid = ids[cur][0]
        if len(ids[cur]) > 1:
            log(f"  the newly connected motor has id={cur}, a duplicate id of a motor that is already numbered: type 7 "
                f"would change both. Connect it to this bus alone, change it with --to <free id>, reconnect and continue")
            return False
        if cur == tgt and lone and k + 1 < len(plan):
            # The new motor already has the target id. That id is probably the factory default, and every later new
            # motor will arrive with it too -> move this one out of the way first.
            park = free_park_id(ids, targets)
            log(f"  id {cur} is already the target id, but later new motors may arrive with the same id: "
                f"parking it on temporary id {park}; it is moved back to {tgt} at the end")
            renumber(leg, cur, park, uid=uid, save=save, log=log)
            parked.append((uid, park, tgt, name))
            return True
        if cur != tgt and tgt in ids:
            occ = ids[tgt]
            if len(occ) != 1 or occ[0] in settled():
                raise Refuse(f"target id {tgt} is used by a motor that is already numbered (UID …{_short(occ[0])}): "
                             f"is the id listed twice in the config?")
            park = free_park_id(ids, targets)
            log(f"  target id {tgt} is used by a motor whose turn has not come yet: moving that one to temporary id {park}")
            renumber(leg, tgt, park, uid=occ[0], save=save, log=log)
        renumber(leg, cur, tgt, uid=uid, save=save, log=log)
        if cur == tgt:
            log(f"  id {tgt} was already correct  UID …{_short(uid)}")
        done[uid] = tgt
        result[name] = (tgt, uid)
        lt = leg.read_float_param(tgt, rm.IDX_LIMIT_TORQUE)
        if lt is not None:
            want = FACTORY_LT.get(motor)
            odd = want is not None and abs(lt - want) > LT_TOL
            log(f"  limit_torque {lt:g}; the config lists {motor} for this joint"
                + (f"  !! a factory-new {motor} reads {want:g}: this motor is not a {motor}, or its torque limit"
                   f" has been written before - check the nameplate" if odd else ""))
        return True

    try:
        for k, (name, tgt, motor) in enumerate(plan):
            while True:
                ans = ask(f"[{k + 1}/{len(plan)}] {name} -> id {tgt} ({motor}): "
                          f"connect it, then press Enter (s = skip / q = quit) ").strip().lower()
                if ans == "q":
                    return result
                if ans == "s":
                    log("  skipped")
                    break
                try:
                    if step(k, name, tgt, motor):
                        break
                except Refuse as e:
                    log(f"  refused: {e}")
    finally:
        for uid, park, tgt, name in parked:
            try:
                renumber(leg, park, tgt, uid=uid, save=save, log=log)
                done[uid] = tgt
                result[name] = (tgt, uid)
            except Exception as e:                   # noqa: BLE001
                log(f"  !! {name} is still parked on temporary id {park}, not moved back to {tgt}: {e}\n"
                    f"     once the cause is fixed: --from {park} --to {tgt}")
    return result


def check(found, plans):
    """found: {bus: [(id, uid)]}, plans: {bus: [(name, id, motor)]} -> list of problems (empty = all correct)"""
    bad = []
    want = {b: {t for _, t, _ in p} for b, p in plans.items()}
    for b, p in plans.items():
        ids = by_id(found.get(b, []))
        got = set(ids)
        for i in sorted(i for i in ids if len(ids[i]) > 1):
            bad.append(f"{b}: {len(ids[i])} motors on id {i} (duplicate id)")
        if got == want[b]:
            continue
        other = next((o for o in plans if o != b and got == want[o]), None)
        if other is not None:
            bad.append(f"{b} answers with the id set of {other} {sorted(got)}: "
                       f"the two buses are swapped, exchange the CAN1 / CAN2 connectors")
        else:
            bad.append(f"{b}: answering {sorted(got)} != expected {sorted(want[b])} "
                       f"(missing {sorted(want[b] - got)}, unexpected {sorted(got - want[b])})")
    return bad


def verify(leg, plan, ask=input, log=print):
    """Check every joint by turning it by hand -> names of the joints whose id is wrong."""
    ids = [t for _, t, _ in plan]
    wrong = []
    for name, tgt, _motor in plan:
        if ask(f"{name} should be id {tgt}: press Enter, then turn it back and forth by hand "
               f"(s = skip) ").strip().lower() == "s":
            continue
        got = find_mover(leg, ids, log=log)
        if got == tgt:
            log(f"  ✓ id {tgt} moved")
        else:
            log(f"  ✗ id {got} moved, not {tgt}")
            wrong.append(name)
    return wrong


def print_scan(leg, entries, plan=None, log=print):
    ids = by_id(entries)
    names = {t: (n, m) for n, t, m in (plan or [])}
    if not entries:
        log("  no motor answers (not connected / 48 V off / CAN L-H swapped / no termination resistor)")
    for i in sorted(ids):
        tag = f"  {names[i][0]} ({names[i][1]})" if i in names else ("  (this id is not in the config)" if plan else "")
        if len(ids[i]) > 1:
            log(f"  id {i:<3} !! duplicate id: {len(ids[i])} motors, UID …{' …'.join(_short(u) for u in ids[i])}{tag}")
            continue
        lt = leg.read_float_param(i, rm.IDX_LIMIT_TORQUE)
        log(f"  id {i:<3} UID {ids[i][0]}  limit_torque {'?' if lt is None else f'{lt:g}'}{tag}")
    stray = getattr(leg, "unsolicited", None)
    if stray:
        log(f"  (also received {len(stray)} unsolicited type-0 broadcast frame(s), id field {sorted({i for i, _ in stray})}:"
            f" motors send them by themselves at power-up / reset; they do not indicate a motor on that id and are ignored)")
    if entries:
        log("  (limit_torque is the torque limit stored in the motor; use it as a hint for the motor model: factory-new = "
            "peak torque, RS06 36 / RS02 17 / RS00 14 / EL05 6 / RS05 5.5; any other value has been written)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", metavar="YAML",
                    help="robot config with the joint -> id table (e.g. humanoid_arm/humanoid_arm.yaml, Arm/Arm.yaml, "
                         "Arm/Arm_pair.yaml, biped_Laika/biped_Laika.yaml); required for --assign / --check / --verify")
    ap.add_argument("--bus", help="CAN bus: mc02:0 (CAN1) or mc02:1 (CAN2); with --config and no --bus: every bus in the config")
    ap.add_argument("--port",
                    help="MC02 serial port (default: mc02.port of the config, else /dev/mc02; on macOS /dev/cu.usbmodem*)")
    ap.add_argument("--scan", action="store_true", help="query ids 0..127 and list id / UID / limit_torque")
    ap.add_argument("--from", dest="src", type=int, metavar="ID",
                    help="current id of the motor to change (may be omitted when only one motor is on the bus)")
    ap.add_argument("--to", dest="dst", type=int, metavar="ID", help="new id (1..127)")
    ap.add_argument("--assign", action="store_true", help="guided assignment, one motor at a time in config order")
    ap.add_argument("--check", action="store_true", help="compare the ids answering on each bus with the config")
    ap.add_argument("--verify", action="store_true", help="turn each joint by hand and check that the expected id moves")
    ap.add_argument("--no-save", action="store_true", help="do not save the new id in the motor (it is lost at power-off)")
    ap.add_argument("--yes", action="store_true", help="with --to: do not ask for confirmation")
    args = ap.parse_args()
    if sum(map(bool, (args.scan, args.dst is not None, args.assign, args.check, args.verify))) != 1:
        ap.error("choose exactly one of --scan / --to / --assign / --check / --verify")

    cfg = None
    if args.config:
        import yaml
        cfg = yaml.safe_load(open(args.config))
        rm.MC02_PORT = (cfg.get("mc02") or {}).get("port")
    elif args.assign or args.check or args.verify:
        ap.error("--assign / --check / --verify need --config")
    if args.port:
        rm.MC02_PORT = args.port
    if cfg and args.bus:
        buses = [args.bus]
        if not plan_from_cfg(cfg, args.bus):
            ap.error(f"{args.config} has no joint on bus={args.bus}")
    elif cfg:
        buses = list(dict.fromkeys(j["bus"] for j in cfg["joints"]))
    elif args.bus:
        buses = [args.bus]
    else:
        ap.error("give --bus or --config")
    if args.dst is not None and len(buses) != 1:
        ap.error("--to needs --bus to name the bus")
    plans = {b: plan_from_cfg(cfg, b) for b in buses} if cfg else {}
    save = not args.no_save

    rc, found = 0, {}
    for b in buses:
        plan = plans.get(b)
        leg = rm.LegBus(b, [])
        try:
            if args.scan:
                print(f"--- {b} ---")
                print_scan(leg, scan(leg), plan)
            elif args.check:
                found[b] = scan(leg)
                print(f"--- {b} ---")
                print_scan(leg, found[b], plan)
            elif args.dst is not None:
                entries = scan(leg)
                src = args.src
                if src is None:
                    if len(entries) != 1:
                        print(f"{len(entries)} motors on {b}: use --from to name the id to change:")
                        print_scan(leg, entries, plan)
                        return 1
                    src = entries[0][0]
                if not args.yes and input(f"{b}: id {src} -> {args.dst}{'' if save else ' (not saved)'}: "
                                          f"press Enter to confirm / q to cancel ").strip().lower() == "q":
                    return 1
                try:
                    renumber(leg, src, args.dst, save=save)
                except Refuse as e:
                    print(f"refused (no motor touched): {e}")
                    return 1
            elif args.assign:
                print(f"=== {b}: {plan[0][0]} … {plan[-1][0]} -> id {plan[0][1]}..{plan[-1][1]} ===")
                got = assign(leg, plan, save=save)
                missing = [n for n, _, _ in plan if n not in got]
                found[b] = scan(leg)
                print_scan(leg, found[b], plan)
                if missing:
                    print(f"  not assigned yet: {missing}")
                    rc = 1
            elif args.verify:
                print(f"=== {b} ===")
                wrong = verify(leg, plan)
                if wrong:
                    print(f"  !! id and joint do not match: {wrong}")
                    rc = 1
        finally:
            leg.shutdown()
    if args.check or args.assign:
        bad = check(found, plans)
        if bad:
            print("\n!! differs from " + os.path.basename(args.config) + " as follows:\n  " + "\n  ".join(bad))
            return 1
        if rc == 0:
            print(f"\neverything matches {os.path.basename(args.config)} ✓"
                  + ("  - power the motors off and on, then run --check again to confirm the ids were kept" if args.assign else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
