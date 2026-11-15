#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robstride_mit.py - RobStride motor driver in MIT (motion-control) mode, bus backends and single-motor self-tests

Implements the RobStride private CAN protocol (the motors' default protocol, 29-bit extended frames) as
described in the motor manuals and the vendor's sample code. bridge.py and the tools import this module and
use LegBus.command_all() in their 50 Hz loop; run as a script it offers bus scans and self-tests.

Encoding ranges per motor model (P in rad, V in rad/s, T in Nm). They differ between models; in particular
the Kp / Kd scale of the RS06 is 10x / 20x that of the others:
  RS00: P ±4pi, V ±33,  T ±14,  Kp 0~500,  Kd 0~5
  RS02: P ±4pi, V ±44,  T ±17,  Kp 0~500,  Kd 0~5
  RS05: P ±4pi, V ±50,  T ±5.5, Kp 0~500,  Kd 0~5    (RobStride 05, the gripper motor of the arms)
  EL05: P ±4pi, V ±50,  T ±6,   Kp 0~500,  Kd 0~5    (EduLite 05)
  RS06: P ±4pi, V ±50,  T ±36,  Kp 0~5000, Kd 0~100

The id -> model table comes from one of two places:
  - without --config: the built-in MOTOR_MODEL table of the biped (12 joints, left leg ids 1-6, right leg 7-12)
  - with --config <robot yaml>: the joints table of that yaml (id -> motor on the given bus). Always pass
    --config for the arms: their motor sequence differs from the legs, and frames encoded with another
    model's scale (kp off by 10x) are a silent and dangerous error.

Usage (on the robot computer, with bridge.py / tools/deploy_policy.py stopped; hoist the biped first):
  python robstride_mit.py --bus mc02:0 --ping                                           # bus scan, no motion
  python robstride_mit.py --bus mc02:0 --read                                           # joint angles, nothing enabled
  python robstride_mit.py --bus mc02:0 --id 4                                           # self-test of one motor (left knee, RS06)
  python robstride_mit.py --bus mc02:0 --config humanoid_arm/humanoid_arm.yaml --ping   # arms: ids / models from the yaml
  Self-test sequence: motion-control mode -> enable -> 2 s pure damping (feel it by hand) -> 3 s sine
  tracking of ±0.08 rad -> disable. See --help for the other checks.

Safety:
  - kd_only() is the global safe state: kp = 0, damping only. Put all motors into it on a communication
    timeout or an error.
  - Every command value is saturated to the range of the motor model when it is encoded.
"""

import argparse
import math
import time

import struct

import can

FOUR_PI = 4.0 * math.pi
HOST_ID = 0xFD          # host CAN id (Master_CAN_ID in the vendor's sample code)

# communication types (ExtID bits 28-24)
TYPE_GET_ID   = 0x00    # get device id (used by ping; no motion)
TYPE_MOTION   = 0x01    # motion-control command (torque in ID bits 23-8)
TYPE_FEEDBACK = 0x02    # motor feedback
TYPE_ENABLE   = 0x03
TYPE_STOP     = 0x04
TYPE_SET_ID   = 0x07    # change the motor's CAN id, effective immediately (new id in ExtID bits 23-16); the reply is a type-0 broadcast frame
TYPE_GET_PARAM = 0x11   # read one parameter
TYPE_SET_PARAM = 0x12   # write one parameter (run_mode = 0x7005)
TYPE_SAVE      = 0x16   # save the parameters to flash

IDX_RUN_MODE = 0x7005
IDX_LIMIT_TORQUE = 0x700B
IDX_MECH_POS = 0x7019
IDX_CAN_TIMEOUT = 0x7028  # uint32 in units of 50 µs (20000 = 1 s), 0 = off; after this long without a command the motor resets (disables) itself
RUN_MODE_MOTION = 0     # motion-control mode


class Limits:
    def __init__(self, v_max, t_max, kp_max, kd_max):
        self.p = FOUR_PI
        self.v, self.t, self.kp, self.kd = v_max, t_max, kp_max, kd_max


MODEL_LIMITS = {
    "RS00": Limits(33.0, 14.0, 500.0, 5.0),
    "RS02": Limits(44.0, 17.0, 500.0, 5.0),
    # RS05 (RobStride 05, the gripper motor of the arms): rated 1.6 / peak 5.5 Nm, gear ratio 7.75:1.
    "RS05": Limits(50.0, 5.5, 500.0, 5.0),
    # EL05 (EduLite 05): rated 1.8 / peak 6 Nm, gear ratio 9:1, same 29-bit private protocol. The frame
    # tables of its manual give torque ±6 Nm (limit_torque 0~6), while the C sample in the same manual still
    # uses T_MAX 5.5; the frame table is what is used here. Before enabling an EL05 for the first time, run
    # --scale-check on it and read back limit_torque to confirm the range.
    "EL05": Limits(50.0, 6.0, 500.0, 5.0),
    "RS06": Limits(50.0, 36.0, 5000.0, 100.0),
}

# Built-in motor model table of the biped's 12 joints, used when no --config is given (bridge.py cross-checks
# it against biped_Laika/biped_Laika.yaml): hip_pitch / knee = RS06, hip_roll / ankle_pitch = RS02,
# hip_yaw / ankle_roll = RS00; left leg ids 1-6, right leg ids 7-12.
# The other robots do not use this table: the same id can be a different model there (see the module docstring).
MOTOR_MODEL = {
    1: "RS06", 2: "RS02", 3: "RS00", 4: "RS06", 5: "RS02", 6: "RS00",
    7: "RS06", 8: "RS02", 9: "RS00", 10: "RS06", 11: "RS02", 12: "RS00",
}


def models_from_cfg(cfg, bus=None):
    """joints table of a robot yaml -> {motor_id: model}. With `bus`, only the joints on that bus are taken (the
    same id can be a different model on another bus, which is why models are keyed by (bus, id)); without it,
    an id must not map to two models anywhere in the table."""
    out = {}
    for j in cfg["joints"]:
        if bus is not None and j["bus"] != bus:
            continue
        if not j.get("enabled", True):
            continue
        mid, model = int(j["id"]), j["motor"]
        if model not in MODEL_LIMITS:
            raise KeyError(f"{j['name']}: motor={model!r} is not in MODEL_LIMITS {sorted(MODEL_LIMITS)}")
        if mid in out and out[mid] != model:
            raise ValueError(f"id={mid} on {bus or 'all buses'} is listed as both {out[mid]} and {model}")
        out[mid] = model
    return out


def ids_from_cfg(cfg, bus=None):
    """joints table of a robot yaml -> list of the ids on that bus (in table order)"""
    return [int(j["id"]) for j in cfg["joints"]
            if (bus is None or j["bus"] == bus) and j.get("enabled", True)]


# Test hook: when set, it is asked first (e.g. to substitute a simulated bus for can.Bus / the MC02 link, as
# tools/fake_bridge.py does); returning None falls through to the normal logic.
OPEN_BUS_HOOK = None
# Serial port of the mc02 channels (None = tools/mc02_link.DEFAULT_PORT = /dev/mc02); bridge.load_config and
# the tools set it from mc02.port of the robot yaml.
MC02_PORT = None


def open_bus(channel):
    """Open a CAN bus. The backend is chosen by the channel prefix, callers do not need to care:
      mc02:N   -> MC02 board (tools/mc02_link; one USB link carrying two CAN buses), used by the robot configs
      /dev/... -> serial CAN adapter speaking slcan (an alternative that needs no kernel module)
      other    -> SocketCAN network interface (e.g. can0, brought up with ip link)"""
    if OPEN_BUS_HOOK is not None:
        hooked = OPEN_BUS_HOOK(channel)
        if hooked is not None:
            return hooked
    ch = str(channel)
    if ch.startswith("mc02:"):
        import os
        import sys
        _root = os.path.dirname(os.path.abspath(__file__))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from tools.mc02_link import open_mc02_bus
        return open_mc02_bus(ch, port=MC02_PORT)
    if not ch.startswith("/dev/"):
        return can.Bus(interface="socketcan", channel=channel)
    bus = can.Bus(interface="slcan", channel=channel, bitrate=1000000)
    # The adapter may still have been sending when the previous process exited, which leaves a partial frame
    # in the serial buffer; the parser would then start out of sync ("invalid literal for int() with base 16").
    # Discard what is left so that parsing starts at the next complete frame.
    ser = getattr(bus, "serialPortOrig", None) or getattr(bus, "_ser", None)
    if ser is not None:
        try:
            time.sleep(0.05)              # let bytes in transit arrive before flushing
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception as e:            # the attribute names differ between python-can versions; not fatal
            print(f"[bus] {channel} serial buffer flush skipped: {e}")
    return bus


def f2u(x, lo, hi, bits=16):
    """float -> uint with saturation (same as float_to_uint in the vendor's sample code)"""
    x = min(max(x, lo), hi)
    span = hi - lo
    return int((x - lo) * ((1 << bits) - 1) / span)


def u2f(u, lo, hi, bits=16):
    span = hi - lo
    return u * span / ((1 << bits) - 1) + lo


def wrap_pi(a):
    """Wrap to (-pi, pi]. A motor loses its turn count at power-off, but the position within one turn stays
    absolute, and the range of every joint is less than one turn."""
    return -((-a + math.pi) % (2.0 * math.pi) - math.pi)


class Feedback:
    __slots__ = ("motor_id", "raw", "angle", "speed", "torque", "temp", "fault", "mode", "stamp", "t_dev")

    def __repr__(self):
        return (f"<id={self.motor_id} q={self.angle:+.3f} dq={self.speed:+.2f} "
                f"tau={self.torque:+.2f} {self.temp:.1f}C fault=0x{self.fault:02x}>")


class LegBus:
    """Batched motion-control exchange with the motors on one CAN bus (one leg of the biped, or one arm
    including its gripper).

    models: {motor_id: model}. If omitted, the built-in biped table MOTOR_MODEL is used. Every other robot
    must pass it explicitly (bridge.py takes it from the yaml); otherwise the frames are silently encoded
    with the wrong scale."""

    def __init__(self, channel, motor_ids, models=None):
        self.bus = open_bus(channel)
        self.channel = channel
        self.ids = list(motor_ids)
        self.model = (dict(models) if models is not None
                      else {i: MOTOR_MODEL[i] for i in self.ids})
        missing = [i for i in self.ids if i not in self.model]
        if missing:
            raise KeyError(f"{channel}: id {missing} is not in the model table (models={self.model})")
        self.lim = {i: MODEL_LIMITS[self.model[i]] for i in self.ids}
        self.last_fb = {}
        self.raw = {}      # raw multi-turn angle of each motor, used to command within its current turn
        self.rx_errors = 0  # number of malformed serial (slcan) frames, see _recv

    # ---------- low-level frames ----------
    def _send(self, ext_id, data=b"\x00" * 8):
        self.bus.send(can.Message(arbitration_id=ext_id,
                                  is_extended_id=True, data=data))

    def _parse(self, msg):
        if msg is None or not msg.is_extended_id:
            return None
        ext = msg.arbitration_id
        if (ext >> 24) & 0x3F != TYPE_FEEDBACK:
            return None
        mid = (ext >> 8) & 0xFF
        if mid not in self.lim:
            return None
        lm, d = self.lim[mid], msg.data
        fb = Feedback()
        fb.motor_id = mid
        fb.raw    = u2f(d[0] << 8 | d[1], -lm.p, lm.p)
        fb.angle  = wrap_pi(fb.raw)   # the turn count is lost at power-off (raw may be off by k*2pi); within one turn the angle is absolute
        fb.speed  = u2f(d[2] << 8 | d[3], -lm.v, lm.v)
        fb.torque = u2f(d[4] << 8 | d[5], -lm.t, lm.t)
        fb.temp   = (d[6] << 8 | d[7]) * 0.1
        fb.fault  = (ext >> 16) & 0x3F
        fb.mode   = (ext >> 22) & 0x3
        fb.stamp  = time.monotonic()
        # mc02 backend: µs timestamp taken on the board when the frame was received (same clock as the IMU).
        # It is recorded only; `stamp` stays on the host clock.
        fb.t_dev  = getattr(msg, "t_us", None)
        self.raw[mid] = fb.raw
        self.last_fb[mid] = fb
        return fb

    def _recv(self, timeout):
        """Fault-tolerant wrapper around bus.recv. On a malformed serial (slcan) frame python-can raises
        ValueError (invalid literal for int() with base 16). One damaged frame must not kill the caller's
        thread: the frame is dropped and the parser resynchronizes on the next complete frame."""
        try:
            return self.bus.recv(timeout)
        except (ValueError, IndexError) as e:
            # ValueError: fromhex / int parsing failed; IndexError: truncated frame (dlc = string[9])
            self.rx_errors += 1
            if self.rx_errors in (1, 10, 100) or self.rx_errors % 1000 == 0:
                print(f"[bus] {self.channel} malformed serial frame x{self.rx_errors} (dropped): {e}")
            return None

    def _collect(self, want, timeout=0.004):
        got, deadline = set(), time.monotonic() + timeout
        while len(got) < len(want):
            left = deadline - time.monotonic()
            if left <= 0:
                break
            fb = self._parse(self._recv(left))
            if fb and fb.motor_id in want:
                got.add(fb.motor_id)
        return got

    # ---------- management commands ----------
    def set_run_mode_motion(self, mid):
        data = bytes([IDX_RUN_MODE & 0xFF, IDX_RUN_MODE >> 8, 0, 0,
                      RUN_MODE_MOTION, 0, 0, 0])
        self._send(TYPE_SET_PARAM << 24 | HOST_ID << 8 | mid, data)
        self._collect({mid})

    def enable(self, mid, timeout=None):
        """Enable one motor. By default the wait is the 4 ms of _collect, which is sized for the 50 Hz batch
        exchange. Pass a much larger timeout for a one-off enable (see the enable retries in bridge.py): a
        caller that sees no feedback in time treats the joint as failed."""
        self._send(TYPE_ENABLE << 24 | HOST_ID << 8 | mid)
        self._collect({mid}) if timeout is None else self._collect({mid}, timeout)

    def stop(self, mid, clear_fault=False):
        data = bytes([1] + [0] * 7) if clear_fault else b"\x00" * 8
        self._send(TYPE_STOP << 24 | HOST_ID << 8 | mid, data)
        self._collect({mid})

    def _write_param(self, mid, index, payload4):
        """Type-18 single-parameter write (RAM only, lost at power-off; persist with save_params):
        data[0:2] = index (little-endian), [2:4] = 0, [4:8] = value"""
        data = bytes([index & 0xFF, index >> 8, 0, 0]) + bytes(payload4)
        self._send(TYPE_SET_PARAM << 24 | HOST_ID << 8 | mid, data)
        self._collect({mid})

    def _read_param_raw(self, mid, index, timeout=0.05):
        """Type-17 single-parameter read -> the 4 raw bytes data[4:8] of the reply, or None on timeout"""
        self._send(TYPE_GET_PARAM << 24 | HOST_ID << 8 | mid,
                   bytes([index & 0xFF, index >> 8, 0, 0, 0, 0, 0, 0]))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            m = self._recv(deadline - time.monotonic())
            if (m and m.is_extended_id
                    and (m.arbitration_id >> 24) & 0x3F == TYPE_GET_PARAM
                    and (m.arbitration_id >> 8) & 0xFF == mid
                    and (m.data[1] << 8 | m.data[0]) == index):
                return bytes(m.data[4:8])
        return None

    def write_float_param(self, mid, index, value):
        self._write_param(mid, index, struct.pack("<f", value))

    def read_float_param(self, mid, index, timeout=0.05):
        raw = self._read_param_raw(mid, index, timeout)
        return None if raw is None else struct.unpack("<f", raw)[0]

    def write_u32_param(self, mid, index, value):
        """uint32 parameter (e.g. IDX_CAN_TIMEOUT), little-endian"""
        self._write_param(mid, index, struct.pack("<I", int(value)))

    def read_u32_param(self, mid, index, timeout=0.05):
        raw = self._read_param_raw(mid, index, timeout)
        return None if raw is None else struct.unpack("<I", raw)[0]

    def save_params(self, mid):
        """Save the parameters to flash so that they survive power-off (magic bytes 01~08 of the vendor's DataSave)"""
        self._send(TYPE_SAVE << 24 | HOST_ID << 8 | mid,
                   bytes([1, 2, 3, 4, 5, 6, 7, 8]))
        time.sleep(0.1)

    def set_torque_limit(self, mid, value, save=False):
        self.write_float_param(mid, IDX_LIMIT_TORQUE, value)
        rb = self.read_float_param(mid, IDX_LIMIT_TORQUE)
        ok = rb is not None and abs(rb - value) < 0.05
        print(f"  id={mid:<2} ({self.model.get(mid, '?')}) limit_torque written {value} "
              f"read back {rb}  {'OK' if ok else '!! mismatch, do not continue'}")
        if ok and save:
            self.save_params(mid)
            print(f"  id={mid:<2} saved to flash (kept after power-off)")
        return ok

    # ---------- ids: type-0 roll call / type-7 CAN id change (used by tools/set_motor_id.py; nothing is enabled or moved) ----------
    def whois(self, ids, per_id=0.004, tail=0.1):
        """Type-0 roll call, one id at a time -> [(motor_id, mcu_uid_hex)] in order of reply, without duplicates.
        The model table is not consulted, so any id can be asked.
        Two different UIDs under the same id mean that two motors share this id (e.g. several new motors with
        the same factory default id on the bus at once).
        The ids are asked one at a time with a wait in between rather than back to back: two motors sharing an
        id that both wait for an idle bus start their replies together, with the same header but different
        data (UID), which causes bit errors and retransmissions. On an idle bus each one answers after its
        own processing delay and both replies arrive.
        Only replies that arrive after their id was asked are accepted. On power-up or reset a motor
        broadcasts one type-0 frame by itself, and its id field is the factory default 127, not the stored id.
        Such frames wait in the receive queue and would look like id collisions; they are collected in
        self.unsolicited and left out of the result."""
        seen, out, asked = set(), [], set()
        self.unsolicited = []

        def take(window):
            deadline = time.monotonic() + window
            while True:
                left = deadline - time.monotonic()
                m = self._recv(max(0.0, left))
                if m is None:
                    if left <= 0:
                        return
                    continue
                ext = m.arbitration_id
                if m.is_extended_id and (ext >> 24) & 0x3F == TYPE_GET_ID and (ext & 0xFF) == 0xFE:
                    key = ((ext >> 8) & 0xFF, bytes(m.data).hex())
                    if key[0] not in asked:
                        if key not in self.unsolicited:
                            self.unsolicited.append(key)
                    elif key not in seen:
                        seen.add(key)
                        out.append(key)

        take(0.0)                                   # drain the frames already queued (no id has been asked yet)
        for mid in ids:
            asked.add(int(mid))
            self._send(TYPE_GET_ID << 24 | HOST_ID << 8 | mid)
            take(per_id)
        take(tail)
        return out

    def set_can_id(self, mid, new_id):
        """Type 7 (communication type 7 of the motor manual): change the motor whose id is `mid` to `new_id`,
        effective immediately.
        Frame: bits 28-24 = 7, bits 23-16 = new CAN id, bits 15-8 = host id, bits 7-0 = current id of the
        target motor; data all 0.
        Every motor on the bus with id == mid is changed. Use whois first to make sure that only one motor
        has this id and that new_id is free."""
        if not (0 <= int(mid) <= 127 and 1 <= int(new_id) <= 127):
            raise ValueError(f"CAN_ID out of range: {mid} -> {new_id} (motor id 0..127, new id 1..127)")
        self._send(TYPE_SET_ID << 24 | int(new_id) << 16 | HOST_ID << 8 | int(mid))

    def enable_all(self):
        for i in self.ids:
            self.set_run_mode_motion(i)
            self.enable(i)

    def stop_all(self):
        for i in self.ids:
            try:
                self.stop(i)
            except Exception:
                pass

    # ---------- motion control ----------
    def mit_frame(self, mid, pos, vel, kp, kd, tau):
        """pos is the joint angle wrapped to one turn; it is converted to the raw coordinate of the turn the
        motor is currently in"""
        lm = self.lim[mid]
        if mid in self.raw:
            pos = self.raw[mid] + wrap_pi(pos - wrap_pi(self.raw[mid]))
        ext = (TYPE_MOTION << 24 |
               f2u(tau, -lm.t, lm.t) << 8 | mid)
        data = bytes([
            f2u(pos, -lm.p, lm.p) >> 8, f2u(pos, -lm.p, lm.p) & 0xFF,
            f2u(vel, -lm.v, lm.v) >> 8, f2u(vel, -lm.v, lm.v) & 0xFF,
            f2u(kp, 0, lm.kp) >> 8,     f2u(kp, 0, lm.kp) & 0xFF,
            f2u(kd, 0, lm.kd) >> 8,     f2u(kd, 0, lm.kd) & 0xFF,
        ])
        return ext, data

    def command_all(self, cmd):
        """cmd: {motor_id: (pos, vel, kp, kd, tau)}. All frames are sent back to back, then the feedback is collected.
        Returns the set of ids that replied in this cycle; the caller decides whether a missing one calls
        for the safe state.
        The mc02 backend provides send_batch: the frames of one control step go out in a single USB packet."""
        frames = [self.mit_frame(mid, *c) for mid, c in cmd.items()]
        if hasattr(self.bus, "send_batch"):
            self.bus.send_batch([can.Message(arbitration_id=ext, is_extended_id=True, data=data)
                                 for ext, data in frames])
        else:
            for ext, data in frames:
                self._send(ext, data)
        return self._collect(set(cmd.keys()))

    def kd_only(self, kd=1.0):
        """Safe state: pure damping. Call it on a timeout, on an error and on exit."""
        self.command_all({i: (0.0, 0.0, 0.0, kd, 0.0) for i in self.ids})

    # ---------- MC02 hardware watchdog / failsafe (no-ops on the other backends) ----------
    def register_failsafe(self, kd=1.0, repeat_ms=10):
        """Register one kd-only frame per motor on the board (the same frame kd_only() sends). When the host
        goes silent, the board replays these frames every repeat_ms. Always register them when motors are
        enabled on an MC02 bus. Returns False if the backend has no hardware watchdog."""
        if not hasattr(self.bus, "register_failsafe"):
            return False
        frames = [self.mit_frame(mid, 0.0, 0.0, 0.0, kd, 0.0) for mid in self.ids]
        self.bus.register_failsafe(frames, repeat_ms=repeat_ms)
        return True

    def clear_failsafe(self):
        if hasattr(self.bus, "clear_failsafe"):
            self.bus.clear_failsafe()

    def arm_watchdog(self, timeout_ms):
        """Arm the link watchdog; from then on every command_all feeds it. Returns False if the backend does
        not support it."""
        if not hasattr(self.bus, "arm_watchdog"):
            return False
        self.bus.arm_watchdog(int(timeout_ms))
        return True

    def failsafe_fired(self):
        """Cumulative number of watchdog events. When it changes, the board has replayed the failsafe frames:
        the motors are in damping and the caller must re-home."""
        return int(getattr(self.bus, "fired_count", 0))

    def shutdown(self):
        try:
            self.stop_all()
        finally:
            self.bus.shutdown()


# ============================ self-tests ============================

def selftest(channel, mid, rate=50.0, models=None):
    model = (models or MOTOR_MODEL)[mid]
    print(f"[selftest] {channel} id={mid} ({model})  - make sure the robot is hoisted!")
    leg = LegBus(channel, [mid], models=models)
    dt = 1.0 / rate
    try:
        leg.set_run_mode_motion(mid)
        leg.enable(mid)
        fb = leg.last_fb.get(mid)
        if fb is None:
            print("[err] no feedback after enabling: check the id / bus / termination resistor")
            return
        print(f"[enable] {fb}")
        p0 = fb.angle

        print("[phase A] 2s pure damping kd=1.0 - moved by hand the joint should feel evenly viscous, without oscillation")
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            leg.command_all({mid: (0, 0, 0.0, 1.0, 0)})
            time.sleep(dt)

        print("[phase B] 3s sine ±0.08 rad, kp=10 kd=0.5 - the joint should track smoothly")
        n, miss, err_max = 0, 0, 0.0
        t0 = time.monotonic()
        while (t := time.monotonic() - t0) < 3.0:
            ref = p0 + 0.08 * math.sin(2 * math.pi * 0.5 * t)
            got = leg.command_all({mid: (ref, 0.0, 10.0, 0.5, 0.0)})
            n += 1
            if mid not in got:
                miss += 1
            else:
                err_max = max(err_max, abs(leg.last_fb[mid].angle - ref))
            time.sleep(max(0.0, dt - ((time.monotonic() - t0) % dt)))
        print(f"[result] cycles {n}, missed feedback {miss}, max tracking error {err_max:.3f} rad")
        print("         common causes of a large error: kp=10 is soft / the joint is held or touches something / "
              "the motor saturates at its torque limit - no oscillation means pass")
    finally:
        print("[exit] disabling")
        leg.shutdown()


def scale_check(channel, mid, kp_cmd=20.0, kd_cmd=1.0, models=None):
    """Automatic check of the firmware's kp scale (hands off): the motor performs symmetric steps of
    ±0.12 rad; at steady state kp_actual = d_tau / d_e, and the symmetric difference cancels the constant
    terms of gravity and friction.
    Precondition: no hard stop and no collision within ±0.2 rad of the joint's current position. Takes about 8 s.
    Run it on every RS06 of an arm (with --config): a motor that is encoded with another model's scale shows
    up here as FAIL / ???? instead of going unnoticed."""
    model = (models or MOTOR_MODEL)[mid]
    print(f"[scale-check] {channel} id={mid} ({model}) commanded kp={kp_cmd} - hoist the robot, do not touch!")
    leg = LegBus(channel, [mid], models=models)
    dt = 0.02

    def run(duration, ref_fn, collect_after=None):
        es, taus, t0 = [], [], time.monotonic()
        while (t := time.monotonic() - t0) < duration:
            ref = ref_fn(t)
            leg.command_all({mid: (ref, 0.0, kp_cmd, kd_cmd, 0.0)})
            fb = leg.last_fb.get(mid)
            if collect_after is not None and t > collect_after and fb:
                es.append(ref - fb.angle)
                taus.append(fb.torque)
            time.sleep(dt)
        return es, taus

    try:
        leg.set_run_mode_motion(mid)
        leg.enable(mid)
        fb = leg.last_fb.get(mid)
        if fb is None:
            print("[err] no feedback after enabling: check the id / bus / termination resistor")
            return
        p0 = fb.angle
        print(f"[auto] holding {p0:+.3f} rad, starting the symmetric ±0.12 rad step test...")
        run(0.6, lambda t: p0)                                   # settle
        res = {}
        for s in (+1, -1):
            tgt = p0 + s * 0.12
            run(0.5, lambda t: p0 + s * 0.12 * (t / 0.5))        # 0.5 s ramp
            es, taus = run(1.6, lambda t: tgt, collect_after=0.8)  # sample at steady state
            if not es:
                print("[err] no feedback samples at steady state, test invalid")
                return
            res[s] = (sum(es) / len(es), sum(taus) / len(taus))
            print(f"  step {s:+d}: e={res[s][0]:+.4f} rad  tau={res[s][1]:+.3f} Nm")
        run(0.5, lambda t: (p0 - 0.12) + 0.12 * (t / 0.5))       # back to the start
        run(0.3, lambda t: p0)
        de = res[+1][0] - res[-1][0]
        dtau = res[+1][1] - res[-1][1]
        if abs(de) < 2e-3:
            print("[err] displacement difference too small (joint blocked or at a limit?), check invalid")
            return
        k = dtau / de
        print(f"[measured] implied_kp = {k:.1f}  (commanded kp = {kp_cmd:.0f})")
        if 0.5 * kp_cmd <= k <= 2.0 * kp_cmd:
            print(f"[PASS] {model} firmware decodes with the scale given in the manual, MODEL_LIMITS is correct")
        elif k <= 0.25 * kp_cmd:
            print(f"[FAIL] the firmware decodes with the small-model scale (0~500 / 0~5) - "
                  f"change MODEL_LIMITS['{model}'] to Limits(v, t, 500.0, 5.0) and test again")
        else:
            print("[????] unexpected reading - keep the complete output and contact support")
    finally:
        print("[exit] disabling")
        leg.shutdown()


def group_test(channel, rate=50.0, damp_s=3.0, hold_s=5.0, ids=None, models=None):
    """Whole-bus test: all motors that respond on the bus do 3 s of pure damping, then hold their current pose
    for 5 s. Prints the measured loop rate, the missed feedback count per motor and the final temperatures;
    use it to judge whether the bus link carries the deployment load.
    Hoist the robot first: all motors of the bus are enabled at the same time."""
    hold_kp = {"RS06": 20.0, "RS02": 15.0, "RS00": 12.0, "RS05": 8.0, "EL05": 8.0}
    leg = LegBus(channel, list(ids) if ids else list(range(1, 13)), models=models)
    dt = 1.0 / rate
    try:
        active = []
        for mid in leg.ids:
            leg.set_run_mode_motion(mid)
            leg.enable(mid)
            if mid in leg.last_fb:
                active.append(mid)
        if not active:
            print("[err] no motor responded")
            return
        print(f"[group] {channel}: {len(active)} motors online: {active}")
        pose = {m: leg.last_fb[m].angle for m in active}
        miss = {m: 0 for m in active}

        def loop(dur, cmd_of, label):
            n, t0 = 0, time.monotonic()
            while time.monotonic() - t0 < dur:
                tick = time.monotonic()
                got = leg.command_all({m: cmd_of(m) for m in active})
                for m in active:
                    if m not in got:
                        miss[m] += 1
                n += 1
                time.sleep(max(0.0, dt - (time.monotonic() - tick)))
            hz = n / (time.monotonic() - t0)
            print(f"[{label}] measured {hz:.1f} Hz (target {rate:.0f})")
            return hz

        print(f"[phase A] {damp_s:.0f}s pure damping on all motors, kd=1.0")
        loop(damp_s, lambda m: (0.0, 0.0, 0.0, 1.0, 0.0), "A")
        print(f"[phase B] {hold_s:.0f}s all motors hold their current pose (kp per model)")
        hz = loop(hold_s, lambda m: (pose[m], 0.0,
                                     hold_kp[leg.model[m]], 1.0, 0.0), "B")
        print("[result] missed feedback:", {m: miss[m] for m in active})
        for m in active:
            fb = leg.last_fb[m]
            print(f"  id={m:<2} q={fb.angle:+.3f} tau={fb.torque:+.2f} {fb.temp:.0f}C")
        if hz >= rate * 0.9 and max(miss.values()) == 0:
            print("[verdict] this bus link carries the deployment load")
        else:
            print("[verdict] loop rate or missed feedback out of bounds -> this bus link does not carry the deployment load")
    finally:
        print("[exit] disabling all motors")
        leg.shutdown()


def read_angles(channel, ids=range(1, 13), models=None):
    """Power-on check: read all joint angles without enabling anything. Run it after every power cycle and
    confirm that the wrapped angles are physically plausible (the turn count is lost at power-off, so the
    raw value may be off by k*2pi)."""
    leg = LegBus(channel, list(ids), models=models)
    try:
        for mid in ids:
            v = leg.read_float_param(mid, IDX_MECH_POS)
            if v is None:
                print(f"  id={mid:<2} ({leg.model.get(mid, '?')})  ---- no response")
            else:
                note = "" if abs(v - wrap_pi(v)) < 0.01 else f"  (raw {v:+.3f}, turn count lost, wrapped)"
                print(f"  id={mid:<2} ({leg.model.get(mid, '?')})  "
                      f"q={wrap_pi(v):+.3f} rad{note}")
    finally:
        leg.bus.shutdown()


def ping(channel, ids=range(1, 13), models=None, expect_note=None):
    """Bus scan with communication type 0 (get device id): nothing is enabled and nothing moves. Reports the
    motors that reply. Returns {id: mcu_uid_hex}."""
    models = models if models is not None else MOTOR_MODEL
    bus = open_bus(channel)
    alive = {}
    try:
        for mid in ids:
            bus.send(can.Message(arbitration_id=TYPE_GET_ID << 24 | HOST_ID << 8 | mid,
                                 is_extended_id=True, data=b"\x00" * 8))
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            m = bus.recv(deadline - time.monotonic())
            if m and m.is_extended_id and (m.arbitration_id & 0xFF) == 0xFE:
                alive[(m.arbitration_id >> 8) & 0xFF] = bytes(m.data).hex()
        for mid in ids:
            mark = f"online  MCU={alive[mid]}" if mid in alive else "---- no response"
            print(f"  id={mid:<2} ({models.get(mid, '?')})  {mark}")
        note = expect_note if expect_note is not None else "(expected: 1-6 on the left-leg bus, 7-12 on the right-leg bus)"
        print(f"[ping] {channel}: {len(alive)}/{len(list(ids))} online {note}")
        extra = sorted(set(alive) - set(ids))
        if extra:
            print(f"[ping] !! ids outside the table replied as well: {extra} - does the config match the hardware?")
    finally:
        bus.shutdown()
    return alive


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", required=True, help="CAN bus: mc02:0 (CAN1) or mc02:1 (CAN2) of the MC02 board")
    ap.add_argument("--id", type=int, help="motor id (required for the self-test, --scale-check and --set-torque-limit)")
    ap.add_argument("--config", metavar="YAML",
                    help="robot yaml that provides the id -> motor model table (e.g. humanoid_arm/humanoid_arm.yaml, Arm/Arm.yaml). "
                         "Default: the built-in table of the biped's 12 joints. Required for the arms, whose frames would otherwise be encoded with the biped's scales")
    ap.add_argument("--rate", type=float, default=50.0, help="control loop rate in Hz for the self-test")
    ap.add_argument("--ping", action="store_true",
                    help="bus scan: list the motors that reply (no motion)")
    ap.add_argument("--read", action="store_true",
                    help="power-on check: read all joint angles without enabling (wrapped to one turn)")
    ap.add_argument("--group-test", action="store_true",
                    help="whole-bus test: all motors damp, then hold their pose; measures the loop rate and missed feedback")
    ap.add_argument("--scale-check", action="store_true",
                    help="firmware scale check: the motor steps by +/-0.12 rad on its own and the actual kp is derived from the torque feedback")
    ap.add_argument("--set-torque-limit", type=float, metavar="NM",
                    help="write limit_torque and verify it by reading it back (RAM only)")
    ap.add_argument("--save", action="store_true",
                    help="with --set-torque-limit: save the parameter to flash so that it survives power-off")
    args = ap.parse_args()

    # id -> model and the id list of this bus: from --config, otherwise the built-in biped table
    if args.config:
        import yaml
        _cfg = yaml.safe_load(open(args.config))
        _ids = ids_from_cfg(_cfg, args.bus)
        if not _ids:                       # no joint on a bus with this name: fall back to the whole table and say so
            _ids = ids_from_cfg(_cfg)
            print(f"[cfg] {args.config} has no joints on bus={args.bus}; using the ids of the whole table {_ids}")
        _models = models_from_cfg(_cfg, args.bus if ids_from_cfg(_cfg, args.bus) else None)
        _note = f"(expected by {args.config}: {_ids})"
        print(f"[cfg] model table from {args.config}: " +
              ", ".join(f"{i}:{_models[i]}" for i in _ids))
    else:
        _ids, _models, _note = list(range(1, 13)), None, None
        print("[cfg] using the built-in model table of the biped's 12 joints (left leg 1-6 / right leg 7-12); for the arms add --config <robot yaml>")
    if args.id is not None and _models is not None and args.id not in _models:
        ap.error(f"--id {args.id} is not in the joints table of {args.config} for bus={args.bus}: {sorted(_models)}")

    if args.ping:
        ping(args.bus, _ids, models=_models, expect_note=_note)
    elif args.read:
        read_angles(args.bus, _ids, models=_models)
    elif args.group_test:
        group_test(args.bus, ids=_ids, models=_models)
    elif args.set_torque_limit is not None:
        if args.id is None:
            ap.error("--set-torque-limit requires --id")
        leg = LegBus(args.bus, [args.id], models=_models)
        try:
            leg.set_torque_limit(args.id, args.set_torque_limit, save=args.save)
        finally:
            leg.bus.shutdown()
    elif args.scale_check:
        if args.id is None:
            ap.error("--scale-check requires --id")
        scale_check(args.bus, args.id, models=_models)
    else:
        if args.id is None:
            ap.error("the self-test requires --id")
        selftest(args.bus, args.id, args.rate, models=_models)
