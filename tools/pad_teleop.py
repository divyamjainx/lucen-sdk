#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/pad_teleop.py - PlayStation gamepad (DualShock 4 / DualSense, Bluetooth or USB) reader for deploy_policy --pad.

No dependencies: it reads the kernel joydev interface /dev/input/js0 directly (struct js_event, 8 bytes);
evdev and pygame are not needed. With the hid-sony (DS4) or hid-playstation (DualSense) kernel driver the
joydev numbering is (Xbox-style pads use the same order):
  axes:    0 left stick X   1 left stick Y   2 L2   3 right stick X   4 right stick Y   5 R2   6/7 D-pad
  buttons: 0 ×(Cross)  1 ○(Circle)  2 △  3 □  4 L1  5 R1  6 L2  7 R2  8 Share/Create  9 Options  10 PS  11 L3  12 R3
Kernels without those drivers use hid-generic, which numbers differently; see MAPS below (selected automatically).

Controls (the same commands as the keyboard of deploy_policy.KeyTeleop):
  left stick up / down      forward / back  (vx, full deflection = --teleop-speed)
  left stick left / right   side-step       (vy, full deflection = --teleop-side)
  right stick left / right  turn            (wz, full deflection = --teleop-turn)
  Options = stand    △ = walk    Share = stop (zero the command)    PS = quit
  R1 = dead-man button: unless R1 is held, the stick command is zero (--pad-no-deadman turns this off).
  Releasing the sticks gives a zero command; with a stand policy loaded the robot then returns to standing.

Safety: joydev does not repeat events while a stick is held, and when Bluetooth drops, the kernel only removes
js0 after its supervision timeout (several seconds); until then the last stick value would stay in effect.
The dead-man button narrows that window to "R1 was still held". When the device disappears (read fails with
ENODEV) the command is zeroed at once and the program quits.

Run `python tools/pad_teleop.py` on its own to print the raw events and the commands they map to; use it to
check that the gamepad is connected and that the button numbering is right.
"""
from __future__ import annotations

import os
import struct
import sys
import time

import numpy as np

JS_EVENT = struct.Struct("IhBB")          # time(ms) value type number
JS_BUTTON, JS_AXIS, JS_INIT = 0x01, 0x02, 0x80
# Two joydev numberings. Kernels that have the hid-playstation / hid-sony driver use "ds". Kernels without it
# (for example the stock Jetson kernel) drive the pad with hid-generic, which numbers in HID descriptor order:
# □ is button 0 and right stick X is axis 2.
#   ds      (hid-playstation / hid-sony): axes LX0 LY1 L2 RX3 RY4 R2 | buttons ×0 ○1 △2 □3 L1 R1 L2 R2 Share Options PS L3 R3  (13 buttons)
#   generic (hid-generic, USB or basic Bluetooth mode): axes LX0 LY1 RX2 RY3 L2 R2 | buttons □0 ×1 ○2 △3 L1 R1 L2 R2 Create Options L3 R3 PS touchpad mute (15 buttons)
MAPS = {
    "ds": dict(axes={"lx": 0, "ly": 1, "rx": 3},
               btn={"cross": 0, "circle": 1, "triangle": 2, "square": 3, "l1": 4, "r1": 5, "l2": 6, "r2": 7,
                    "share": 8, "options": 9, "ps": 10, "l3": 11, "r3": 12}),
    "generic": dict(axes={"lx": 0, "ly": 1, "rx": 2},
                    btn={"square": 0, "cross": 1, "circle": 2, "triangle": 3, "l1": 4, "r1": 5, "l2": 6, "r2": 7,
                         "share": 8, "options": 9, "l3": 10, "r3": 11, "ps": 12, "touchpad": 13, "mute": 14}),
}
# module-level names for the default "ds" numbering
AXIS_LX, AXIS_LY, AXIS_RX = 0, 1, 3
BTN = MAPS["ds"]["btn"]
BTN_NAME = {v: k for k, v in BTN.items()}


class PadTeleop:
    """Same interface as deploy_policy.KeyTeleop: __enter__/__exit__ / poll(now)->quit / command(now)->(vx,vy,wz),
    attributes vec / quit / reqs / speed / side / turn / hold."""

    def __init__(self, speed, hold, turn=0.20, side=0.10, dev="/dev/input/js0", deadman=True,
                 deadzone=0.15, fd=None, pad_map="auto"):
        self.speed, self.hold, self.turn, self.side = float(speed), float(hold), float(turn), float(side)
        self.dev, self.deadman, self.deadzone = dev, bool(deadman), float(deadzone)
        self.pad_map = pad_map              # "auto" = detect from the kernel driver / button count, or force ds / generic
        self._set_map("ds" if pad_map == "auto" else pad_map)
        self.vec = np.zeros(3)
        self.until = float("inf")       # sticks are continuous: no hold window
        self.quit = False
        self.lost = False               # device gone -> the command stays 0
        self.reqs = []
        self._btn = {}
        self._fd = fd                   # tests can inject the read end of a pipe
        self._buf = b""
        self.name = ""

    def _set_map(self, name):
        if name not in MAPS:
            raise ValueError(f"unknown gamepad numbering {name!r}; choose from {sorted(MAPS)} / auto")
        self.map_name = name
        self._ax = dict(MAPS[name]["axes"])
        self._btnmap = dict(MAPS[name]["btn"])
        self._btn_name = {v: k for k, v in self._btnmap.items()}
        self._axes = {self._ax["lx"]: 0.0, self._ax["ly"]: 0.0, self._ax["rx"]: 0.0}

    # -- device --
    def __enter__(self):
        if self._fd is None:
            try:
                self._fd = os.open(self.dev, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as e:
                sys.exit(f"[pad] cannot open {self.dev}: {e}\n"
                         f"  The gamepad is not connected (Bluetooth pairing / USB cable), or it is not js0: ls /dev/input/js*")
            self.name = self._query_name()
            self.n_buttons = self._query_u8(0x80016a12)       # JSIOCGBUTTONS
            self.n_axes = self._query_u8(0x80016a11)          # JSIOCGAXES
            self.driver = self._query_driver()
            if self.pad_map == "auto":
                # The kernel driver is the most reliable indicator: hid-generic -> "generic", playstation / sony -> "ds";
                # if the driver cannot be read, decide by the button count.
                if self.driver == "hid-generic":
                    self._set_map("generic")
                elif self.driver in ("playstation", "sony"):
                    self._set_map("ds")
                else:
                    self._set_map("generic" if (self.n_buttons or 0) >= 15 else "ds")
        return self

    def _query_driver(self):
        """js node -> input device -> HID device -> name of the bound driver (sysfs); '' on failure."""
        try:
            base = os.path.basename(os.path.realpath(self.dev))
            return os.path.basename(os.path.realpath(f"/sys/class/input/{base}/device/device/driver"))
        except OSError:
            return ""

    def __exit__(self, *exc):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _query_name(self):
        try:
            import fcntl
            buf = bytearray(128)
            fcntl.ioctl(self._fd, 0x80006a13 | (len(buf) << 16), buf)    # JSIOCGNAME(len)
            return buf.split(b"\0", 1)[0].decode(errors="replace")
        except Exception:
            return ""

    def _query_u8(self, req):
        try:
            import fcntl
            buf = bytearray(1)
            fcntl.ioctl(self._fd, req, buf)
            return int(buf[0])
        except Exception:
            return None

    # -- events --
    def _norm(self, v):
        x = float(v) / 32767.0
        if abs(x) < self.deadzone:
            return 0.0
        s = 1.0 if x > 0 else -1.0
        return s * (abs(x) - self.deadzone) / (1.0 - self.deadzone)     # linear from the dead-zone edge to 1.0 at full deflection

    def _handle(self, value, etype, number, now):
        init = bool(etype & JS_INIT)
        etype &= ~JS_INIT
        if etype == JS_AXIS:
            if number in self._axes:
                self._axes[number] = self._norm(value)
                self.vec = np.array([-self._axes[self._ax["ly"]] * self.speed,
                                     -self._axes[self._ax["lx"]] * self.side,
                                     -self._axes[self._ax["rx"]] * self.turn])
        elif etype == JS_BUTTON:
            pressed = bool(value)
            self._btn[number] = pressed
            if init or not pressed:
                return
            name = self._btn_name.get(number)
            if name == "options":
                self.reqs.append("stand")
            elif name == "triangle":
                self.reqs.append("loco")
            elif name == "share":
                self.vec = np.zeros(3)
                print("\r[pad] stop (Share)      ", end="", flush=True)
            elif name == "ps":
                self.quit = True

    def poll(self, now):
        """Consume all pending events; return True when the program should quit.
        If the device disappears, lost = True (the command stays 0) and quit is requested."""
        if self._fd is None:
            return self.quit
        while True:
            try:
                chunk = os.read(self._fd, 64 * JS_EVENT.size)
            except BlockingIOError:
                break
            except OSError as e:
                self.lost = True
                self.vec = np.zeros(3)
                print(f"\n[pad] gamepad disconnected ({e}): command zeroed, quitting")
                self.quit = True
                break
            if not chunk:
                self.lost = True
                self.vec = np.zeros(3)
                print("\n[pad] gamepad device returned EOF: command zeroed, quitting")
                self.quit = True
                break
            self._buf += chunk
            n = len(self._buf) // JS_EVENT.size
            for i in range(n):
                _t, value, etype, number = JS_EVENT.unpack_from(self._buf, i * JS_EVENT.size)
                self._handle(value, etype, number, now)
            self._buf = self._buf[n * JS_EVENT.size:]
        return self.quit

    def deadman_ok(self):
        return (not self.deadman) or bool(self._btn.get(self._btnmap["r1"], False))

    def command(self, now):
        """-> (vx, vy, wz). Zero when the device is lost or the dead-man button is not held."""
        if self.lost or not self.deadman_ok():
            return np.zeros(3)
        return self.vec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Gamepad mapping test: print the raw events and the mapped command (vx, vy, wz) / requests")
    ap.add_argument("--dev", default="/dev/input/js0")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--no-deadman", action="store_true")
    ap.add_argument("--map", default="auto", choices=["auto", "ds", "generic"], help="joydev numbering (default: detected automatically)")
    args = ap.parse_args(argv)
    with PadTeleop(0.20, 1.0, 0.20, 0.10, dev=args.dev, deadman=not args.no_deadman, pad_map=args.map) as pad:
        print(f"[pad] {args.dev}  name '{pad.name}'  driver '{pad.driver}'  "
              f"joydev reports {pad.n_axes} axes, {pad.n_buttons} buttons -> numbering '{pad.map_name}'  "
              f"dead-man R1 {'off' if args.no_deadman else 'on'}; move the sticks / press buttons to see the mapping, "
              f"PS quits, ends after {args.duration:.0f} s")
        t0 = time.monotonic()
        last = None
        raw_seen = []
        _orig = pad._handle

        def _spy(value, etype, number, now):     # print the raw event numbers, to verify the numbering
            if not (etype & JS_INIT):
                raw_seen.append(f"{'btn' if etype & JS_BUTTON else 'axis'}{number}={value}")
            _orig(value, etype, number, now)
        pad._handle = _spy
        while time.monotonic() - t0 < args.duration:
            now = time.monotonic()
            q = pad.poll(now)
            if raw_seen:
                print("  raw events: " + " ".join(raw_seen[-8:])); raw_seen.clear()
            cmd = pad.command(now)
            reqs, pad.reqs = list(pad.reqs), []
            state = (tuple(np.round(cmd, 3)), pad.deadman_ok())
            if reqs or state != last:
                print(f"  t={now - t0:5.1f}s  cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}  "
                      f"R1 {'held' if pad.deadman_ok() else 'released'}  requests {reqs}  "
                      f"sticks raw {dict((k, round(v, 2)) for k, v in pad._axes.items())}")
                last = state
            if q:
                print("[pad] quit (PS button / disconnected)")
                break
            time.sleep(0.02)


if __name__ == "__main__":
    main()
