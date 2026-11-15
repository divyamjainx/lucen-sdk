#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/check_ports.py - check that the serial devices named in a robot config exist and do not collide.

USB serial devices get /dev/ttyACM* names in enumeration order, and several boards share the generic ST CDC
identity (VID 0x0483 / PID 0x5740, "STM32 Virtual ComPort"). Two config entries can therefore end up on the
same physical device, or two bus adapters can be swapped. The symptoms are hard to read (garbled CAN frames,
one leg following the other leg's targets), so this check resolves every port to its real device and USB
serial number and fails early with a clear message.

Usage (on the robot computer):
  python tools/check_ports.py                        # biped_Laika/biped_Laika.yaml: list every port, check for conflicts
  python tools/check_ports.py --config Arm/Arm.yaml  # any other robot config

tools/deploy_policy.py runs the same check (assert_no_conflict) before it starts.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def usb_info(dev):
    """-> (product, serial, usb_path); empty strings when the information is not available."""
    try:
        out = subprocess.run(["udevadm", "info", "-q", "property", "-n", dev],
                             capture_output=True, text=True, timeout=5).stdout
        d = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
        return (d.get("ID_MODEL", "") or d.get("ID_MODEL_ENC", ""),
                d.get("ID_SERIAL_SHORT", ""), d.get("DEVPATH", "").split("/tty")[0])
    except Exception:
        return ("", "", "")


def ports_of(cfg):
    """-> {logical name: device path from the config}. A config without an imu section has no IMU port.
    A bus written as mc02:N is a channel of the MC02 board, not a device path: it maps to mc02.port in the
    config (default /dev/mc02). Both CAN channels and the IMU sharing that one device is by design;
    assert_no_conflict only reports different config paths that resolve to the same device."""
    mc02_port = (cfg.get("mc02") or {}).get("port", "/dev/mc02")
    p = {}
    for j in cfg["joints"]:
        if not j.get("enabled", True):
            continue
        b = str(j["bus"])
        p[b] = mc02_port if b.startswith("mc02:") else b
    if cfg.get("imu"):
        p["imu"] = cfg["imu"]["port"]
    return p


def resolve(cfg):
    """-> [(logical name, config path, real device, product, serial)]"""
    rows = []
    for name, path in ports_of(cfg).items():
        real = os.path.realpath(path) if os.path.exists(path) else "(missing)"
        prod, ser, _ = usb_info(path) if real != "(missing)" else ("", "", "")
        rows.append((name, path, real, prod, ser))
    return rows


def assert_no_conflict(cfg, verbose=True):
    """Exit with a clear error if a device is missing, two config paths resolve to the same physical device,
    or a USB serial number does not match bus_serials in the config."""
    rows = resolve(cfg)
    if verbose:
        print("Port check:")
        for name, path, real, prod, ser in rows:
            tag = "" if path == real else f" -> {real}"
            print(f"  {name:<16} {path}{tag}   [{prod} {ser}]")
    seen, bad = {}, []            # real device -> (logical name, config path)
    for name, path, real, _, _ in rows:
        if real == "(missing)":
            bad.append(f"{name}: device {path} does not exist")
            continue
        if real in seen and seen[real][1] != path:
            # Two different config paths on one physical device: the enumeration order changed or a udev rule
            # matched the wrong device. One config path used under several logical names (mc02:0 / mc02:1 /
            # imu are all /dev/mc02) is by design.
            bad.append(f"{name} and {seen[real][0]} point to the same physical device {real}")
        seen.setdefault(real, (name, path))

    # Swapped bus adapters are more dangerous than a port collision: one leg would receive the other leg's targets.
    want = cfg.get("bus_serials") or {}
    got = {name: ser for name, _, _, _, ser in rows}
    for bus, want_ser in want.items():
        have = got.get(bus, "")
        allowed = [x for x in (want_ser if isinstance(want_ser, list) else [want_ser]) if x]   # list = any of them is accepted
        if have and allowed and have not in allowed:
            other = [b for b, s in want.items() if have in (s if isinstance(s, list) else [s])]
            hint = f" - that adapter belongs to {other[0]}: the two are swapped!" if other else ""
            bad.append(f"{bus} has USB serial number {have}, but the config lists {allowed}{hint}")
    if bad:
        print()
        for b in bad:
            print(f"[err] {b}")
        print("How to fix:")
        print("  Device missing or shared -> give every device a fixed udev name (for example /dev/mc02);")
        print("    do not put /dev/ttyACM* in the config, those names change with the enumeration order.")
        print("    The udev rule for the MC02 board is described in docs/mc02.md.")
        print("  Serial number mismatch -> adapters swapped or replaced: swap the USB plugs or update bus_serials in the config.")
        raise SystemExit(1)
    if verbose:
        print("  -> no conflicts")
    return True


if __name__ == "__main__":
    import argparse
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "biped_Laika/biped_Laika.yaml"),
                    help="robot config to check (default: biped_Laika/biped_Laika.yaml)")
    a = ap.parse_args()
    assert_no_conflict(yaml.safe_load(open(a.config)))
