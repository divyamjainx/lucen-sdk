#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/mc02_selftest.py - MC02 board self-test: link, boot flags, CAN bus scan against the config, round-trip time.

Run this first after plugging the robot in. By default it NEVER enables a motor: it only sends PING and
STATUS requests and RobStride type-0 (get id) queries. Steps, in this order:
  [1] PING    firmware version, protocol version, BMI088 (IMU) present, MCU UID of the board
  [2] STATUS  boot flags (bit0 = FDCAN kernel clock does not match the bit timing -> wrong CAN bit rate;
              bit1 = IMU initialization failed) and the last error code (LEC) of both CAN buses
              (LEC 3 = ACK error: no other node on the bus / no termination)
  [3] Scan    query ids 1-14 on both buses (further if the config uses higher ids) and compare the answers
              with the bus -> id sets of the config:
                biped_Laika/biped_Laika.yaml      CAN1 = left leg 1-6,  CAN2 = right leg 7-12
                humanoid_arm/humanoid_arm.yaml    CAN1 = left arm 1-8,  CAN2 = right arm 9-16
                Arm/Arm.yaml                      CAN1 = the arm 1-7,   CAN2 unused
                Arm/Arm_pair.yaml                 CAN1 = follower 1-7,  CAN2 = leader 8-14
  [4] RTT     round-trip time of 100 PINGs (USB full-speed CDC: expect a p95 of 2-4 ms)
  [5] --wd-test: registers type-4 stop frames as failsafe frames (the motors are not enabled, so replaying
      them does nothing) and checks the chain "host goes silent -> the board sends frames by itself".
      The robot must be on the hoist.
  [6] --imu: streams IMU samples at 500 Hz for 1 s and prints the rate and the mean values at rest

Usage (on the robot computer; stop the bridge / deploy process first - only one process can open an MC02 board):
  python tools/mc02_selftest.py                 # steps 1-4 for biped_Laika/biped_Laika.yaml
  python tools/mc02_selftest.py --wd-test       # + watchdog chain
  python tools/mc02_selftest.py --imu           # + IMU stream
  python tools/mc02_selftest.py --dfu           # reboot the board into the USB bootloader to flash new firmware
  python tools/mc02_selftest.py --config humanoid_arm/humanoid_arm.yaml   # expect the id sets of the dual arms
  python tools/mc02_selftest.py --config Arm/Arm.yaml                     # 6+1-DoF arm
  python tools/mc02_selftest.py --config Arm/Arm_pair.yaml                # leader + follower arms
  python tools/mc02_selftest.py --port /dev/serial/by-id/<board>          # one specific board

With two MC02 boards plugged into one host, /dev/mc02 is ambiguous: pass the board's /dev/serial/by-id/...
path with --port, or put it into `mc02.port` of the config. The exit code is 0 on SELFTEST PASS.
Board, wiring and flashing: docs/mc02.md.
"""
import argparse
import os
import struct
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.mc02_link import (BOOTF_FDCAN_CLK_MISMATCH, BOOTF_IMU_INIT_FAIL, BUS_COUNT,   # noqa: E402
                             MC02Link)

RED, GRN, YEL, RST = "\033[91m", "\033[92m", "\033[93m", "\033[0m"
HOST_ID = 0xFD
T_GET_ID, T_STOP = 0x00, 0x04


def ok(s):
    print(f"  {GRN}✓{RST} {s}")


def bad(s):
    print(f"  {RED}✗ {s}{RST}")


def warn(s):
    print(f"  {YEL}! {s}{RST}")


def scan(link, ids=range(1, 15), settle=0.3):
    """Send type-0 (get id) queries on both buses - no motion, nothing is enabled - and collect the UIDs
    -> {bus: {id: uid_hex}}"""
    ids = list(ids)
    link.drain_can()
    link.send_can_batch([(b, T_GET_ID << 24 | HOST_ID << 8 | mid, bytes(8)) for b in range(BUS_COUNT) for mid in ids])
    found = {b: {} for b in range(BUS_COUNT)}
    t_end = time.monotonic() + settle
    while time.monotonic() < t_end:
        for b in range(BUS_COUNT):
            f = link.recv_can_bus(b, timeout=0.005)
            if f is None or not f.ext:
                continue
            # A reply has 0xFE in the low 8 bits and the motor id in bits 15..8. Only ids that were queried
            # count: the frame a motor broadcasts by itself at power-up carries id 127, whatever its real id.
            if (f.can_id & 0xFF) == 0xFE and ((f.can_id >> 8) & 0xFF) in ids:
                found[b][(f.can_id >> 8) & 0xFF] = f.data.hex()
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="MC02 serial port (default: mc02.port of the config, normally /dev/mc02)")
    ap.add_argument("--config", default=os.path.join(ROOT, "biped_Laika/biped_Laika.yaml"),
                    help="robot config that gives the expected bus -> id sets (default: biped_Laika/biped_Laika.yaml); "
                         "e.g. humanoid_arm/humanoid_arm.yaml, Arm/Arm.yaml, Arm/Arm_pair.yaml")
    ap.add_argument("--wd-test", action="store_true",
                    help="also test the link watchdog (motors stay disabled; the robot must be on the hoist)")
    ap.add_argument("--imu", action="store_true", help="also stream IMU samples for 1 s and print rate and mean values")
    ap.add_argument("--dfu", action="store_true",
                    help="reboot the board into the STM32 system bootloader (USB DFU), ready for flashing with dfu-util")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    cfg_name = os.path.basename(args.config)
    port = args.port or (cfg.get("mc02") or {}).get("port", "/dev/mc02")
    want = {}
    for j in cfg["joints"]:
        b = str(j["bus"])
        if b.startswith("mc02:"):
            want.setdefault(int(b.split(":")[1]), set()).add(int(j["id"]))

    fail = 0
    link = MC02Link(port)

    if args.dfu:
        print(f"[dfu] asking the firmware on {port} to reset into the system bootloader ...")
        link.dfu_jump()
        time.sleep(0.5)
        print("  Next: `lsusb | grep -i 0483:df11` should show 'STM Device in DFU Mode'. Then flash the image"
              " supplied by Lucen:")
        print("  sudo dfu-util -a 0 -s 0x08000000:leave -D <firmware image>")
        return

    print("[1] PING")
    try:
        p = link.ping()
    except TimeoutError as e:
        bad(str(e))
        sys.exit(2)
    ok(f"fw {p.fw[0]}.{p.fw[1]}  proto v{p.proto_ver}  MCU UID {p.uid}")
    if p.proto_ver != 1:
        bad(f"protocol version mismatch (host=1, fw={p.proto_ver})")
        fail += 1
    (ok if p.imu_ok else bad)(f"BMI088 {'present' if p.imu_ok else 'initialization failed'}"
                              f" (range codes acc={p.acc_range_code} gyr={p.gyr_range_code})"
                              + ("" if p.imu_ok else " - the IMU does not answer on SPI2 (mode 3, chip selects PC0 / PC3)"))
    fail += 0 if p.imu_ok else 1

    print("[2] STATUS")
    st = link.status()
    if st.boot_flags & BOOTF_FDCAN_CLK_MISMATCH:
        bad("FDCAN kernel clock differs from the clock the bit timing was built for -> the CAN bit rate is wrong. "
            "Do not connect motors; the firmware clock setup must be fixed first")
        fail += 1
    else:
        ok("FDCAN kernel clock matches the bit timing (1 Mbps)")
    for b in range(BUS_COUNT):
        s = st.bus[b]
        tag = "bus0 CAN1 (connectors at the board's top right)" if b == 0 else "bus1 CAN2 (connectors at the board's top left)"
        extra = []
        if s.bus_off:
            extra.append("BUS-OFF")
        if s.err_passive:
            extra.append("err-passive")
        line = (f"{tag}: rx={s.rx_cnt} tx={s.tx_cnt} drop={s.rx_drop}/{s.tx_drop} LEC={s.last_err}"
                + ((" " + ",".join(extra)) if extra else ""))
        (warn if extra else ok)(line)
    ok(f"USB: crc_err={st.usb_crc_err} cobs_err={st.usb_cobs_err} rx_ovf={st.usb_rx_ovf} tx_ovf={st.usb_tx_ovf}"
       f"  watchdog={('disarmed', 'armed', 'fired')[min(st.wd_state, 2)]}")

    print("[3] Scan (type-0 queries, nothing is enabled)")
    found = scan(link, ids=range(1, max([14] + [i for e in want.values() for i in e]) + 1))
    # Read STATUS again after the scan: LEC is the most recent error, so the value read before the scan is stale.
    # When a bus does not answer, this names the cause.
    LEC = {0: "no error", 1: "stuff error (wrong bit rate / noise on the line)", 2: "form error",
           3: "ACK error (frame sent, nobody acknowledged: motors not powered / L-H swapped / no termination)",
           4: "bit1 error (sent recessive, read dominant: bus held dominant / L-H shorted)",
           5: "bit0 error (sent dominant, read recessive: transceiver / broken wire / no common ground)",
           6: "CRC error", 7: "no change (no error since the last read)"}
    try:
        st2 = link.status()
        for b in range(BUS_COUNT):
            s = st2.bus[b]
            flags = ("BUS-OFF " if s.bus_off else "") + ("err-passive" if s.err_passive else "")
            (warn if (s.last_err not in (0, 7) or flags) else ok)(
                f"bus{b} after the scan: tx={s.tx_cnt} rx={s.rx_cnt} LEC={s.last_err} {LEC.get(s.last_err, '?')} {flags}")
    except TimeoutError:
        warn("no STATUS reply after the scan")
    for b in range(BUS_COUNT):
        ids = sorted(found[b])
        tag = f"bus{b}"
        if ids:
            ok(f"{tag}: ID {ids}")
        else:
            warn(f"{tag}: no answer (not connected / motors not powered / termination -> see LEC above, 3 = ACK error)")
    for b, exp in sorted(want.items()):
        got = set(found.get(b, {}))
        if got == exp:
            ok(f"bus{b} matches {cfg_name}: {sorted(exp)}")
        else:
            other = next((o for o, e in want.items() if o != b and got == e), None)
            if other is not None:
                bad(f"bus{b} answers {sorted(got)}, which is the id set of bus{other} - the two buses are swapped: "
                    f"exchange the CAN1 / CAN2 L/H connectors")
            else:
                bad(f"bus{b} answers {sorted(got)}, expected {sorted(exp)} "
                    f"(missing {sorted(exp - got)}, unexpected {sorted(got - exp)})")
            fail += 1

    print("[4] Link round-trip time (100 PINGs)")
    ts = []
    for _ in range(100):
        t0 = time.monotonic()
        link.ping(timeout=0.5)
        ts.append((time.monotonic() - t0) * 1000)
    ts.sort()
    ok(f"RTT p50={ts[49]:.2f}ms p95={ts[94]:.2f}ms max={ts[-1]:.2f}ms (USB full-speed CDC: expect a p95 of 2-4 ms)")

    if args.imu:
        print("[6] IMU stream, 1 s @ 500 Hz")
        link.imu_stream(2)
        n, acc, gyr = 0, [0.0] * 3, [0.0] * 3
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            s = link.recv_imu(timeout=0.1)
            if s is None:
                continue
            n += 1
            acc = [a + v for a, v in zip(acc, s.acc_mss)]
            gyr = [g + v for g, v in zip(gyr, s.gyr_rads)]
        link.imu_stream(0)
        if n:
            ok(f"{n} samples/s  acc mean {[round(a / n, 2) for a in acc]} m/s²  gyr mean {[round(g / n, 4) for g in gyr]} rad/s"
               f"  (at rest expect |acc| ≈ 9.8, gyr ≈ 0)")
        else:
            bad("no IMU sample within 1 s")
            fail += 1

    if args.wd_test:
        print("[5] Watchdog: the host stops sending -> the board should report FIRED within ~100 ms")
        wd_ms = int((cfg.get("mc02") or {}).get("watchdog_ms", 100))
        all_ids = [(b, m) for b in range(BUS_COUNT) for m in sorted(found[b])]
        if not all_ids:
            warn("the scan found no motor: only the event chain is tested")
        for i, (b, m) in enumerate(all_ids[:16]):
            data = bytes(8)
            link.set_failsafe(i, b, T_STOP << 24 | HOST_ID << 8 | m, data, repeat_ms=10)
        link.set_watchdog(wd_ms, fire_on_suspend=True)
        for _ in range(50):                      # feed the watchdog normally for 0.5 s
            link.ping(timeout=0.5)
            time.sleep(0.01)
        while not link.wd_events.empty():
            link.wd_events.get_nowait()
        t_stop = time.monotonic()
        try:
            ev, _t_us, t_host = link.wd_events.get(timeout=1.0)
            dt = (t_host - t_stop) * 1000
            (ok if ev == 1 else bad)(f"FIRED event {dt:.0f} ms after the host went silent (expected {wd_ms}-{wd_ms + 40})")
            fail += 0 if (ev == 1 and dt < wd_ms + 80) else 1
        except Exception:
            bad("no FIRED event within 1 s of silence")
            fail += 1
        link.ping()
        try:
            ev, *_ = link.wd_events.get(timeout=0.5)
            (ok if ev == 0 else bad)("host is back -> RELEASED")
        except Exception:
            bad("no RELEASED event")
            fail += 1
        link.set_watchdog(0)
        link.clear_failsafe()

    link.close()
    print()
    if fail:
        print(f"{RED}SELFTEST FAIL ({fail}){RST}")
        sys.exit(1)
    print(f"{GRN}SELFTEST PASS{RST}")


if __name__ == "__main__":
    main()
