#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/mc02_link.py - host side of the MC02 USB link: framing, two CAN buses, IMU stream, watchdog / failsafe.

The MC02 board is one USB CDC serial device that carries both CAN buses and the IMU samples. The board
forwards CAN frames as they are; building and parsing motor frames stays on the host (robstride_mit.py).
Packets are COBS-framed with a CRC16-CCITT-FALSE, all fields little-endian. The constants below have the
same values as in the firmware, and PROTO_VER must be equal on both sides. Protocol reference:
docs/mc02_protocol.md; board, wiring and flashing: docs/mc02.md.

Needs pyserial. python-can is not needed: MC02CanBus duck-types the three methods that
robstride_mit.LegBus uses (send / recv / shutdown).

One process per MC02 board. The serial port is opened exclusively, so a second process cannot open it.
Inside a process the link is a singleton per port with reference counting (get_link / release_link): the
bridge or the deploy process owns both CAN buses and the IMU, and `imu_reader.py --daemon` can run only
while neither of them is running.

Two boards on one host: /dev/mc02 is then ambiguous and is refused. Put each board's own
/dev/serial/by-id/... path into `mc02.port` of its robot config.

Unplugging USB powers the board off. The board's watchdog (arm_watchdog / register_failsafe) covers a host
that hangs, crashes or goes silent, not a pulled cable: in that case only the motor-side CAN timeout
(tools/set_can_timeout.py) releases the motors.

Usage:
    from tools.mc02_link import open_mc02_bus
    bus_l = open_mc02_bus("mc02:0")          # CAN1 (biped_Laika: left leg, ids 1-6)
    bus_r = open_mc02_bus("mc02:1")          # CAN2 (biped_Laika: right leg, ids 7-12); shares the same link
    bus_l.send(msg); m = bus_l.recv(0.004); bus_l.shutdown()
"""
from __future__ import annotations

import glob
import struct
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue

PROTO_VER = 1
DEFAULT_PORT = "/dev/mc02"
# Stable per-board path created by Linux; the USB serial number is the MCU UID, different for every board:
# usb-Lucen_Robotics_Lucen_MC02_v1_<serial>-if00
BY_ID_GLOB = "/dev/serial/by-id/*Lucen_MC02_v1*"

# ---- message types (same values as in the firmware; packet layouts: docs/mc02_protocol.md) ----
# host -> board
MSG_CAN_TX, MSG_PING, MSG_WD_CFG, MSG_FAILSAFE_SET = 0x01, 0x02, 0x03, 0x04
MSG_TIME_SYNC, MSG_IMU_CFG, MSG_STATUS_REQ, MSG_DFU_JUMP = 0x05, 0x06, 0x07, 0x08
# board -> host
MSG_CAN_RX, MSG_IMU, MSG_STATUS, MSG_PONG = 0x81, 0x82, 0x83, 0x84
MSG_WD_EVENT, MSG_LOG, MSG_TIME_SYNC_RESP = 0x85, 0x86, 0x87
DFU_MAGIC = 0x44465521              # payload of MSG_DFU_JUMP

CANF_EXT, CANF_RTR = 0x01, 0x02     # CAN frame flags: 29-bit extended id (every RobStride frame) / remote frame
BUS_COUNT = 2
# bus 0 = CAN1 (connectors at the top right of the board), bus 1 = CAN2 (top left).
# biped_Laika: CAN1 = left leg, ids 1-6; CAN2 = right leg, ids 7-12.
BUS_LEFT, BUS_RIGHT = 0, 1
FAILSAFE_SLOTS = 16                 # failsafe frame slots on the board
SLOTS_PER_BUS = 8                   # slot = bus * 8 + i: at most 8 motors per bus

# bits of Status.boot_flags
BOOTF_FDCAN_CLK_MISMATCH = 0x01     # FDCAN kernel clock does not match the bit timing -> wrong CAN bit rate
BOOTF_IMU_INIT_FAIL = 0x02          # BMI088 initialization failed


# ---- COBS / CRC --------------------------------------------------------------
def cobs_encode(data: bytes) -> bytes:
    out = bytearray()
    idx = 0
    while True:
        nz = data.find(b"\x00", idx)
        block = data[idx:] if nz < 0 else data[idx:nz]
        while len(block) >= 254:
            out.append(0xFF)
            out += block[:254]
            block = block[254:]
        out.append(len(block) + 1)
        out += block
        if nz < 0:
            return bytes(out)
        idx = nz + 1


def cobs_decode(data: bytes):
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        code = data[i]
        if code == 0 or i + code > n:
            return None
        out += data[i + 1:i + code]
        i += code
        if code != 0xFF and i < n:
            out.append(0)
    return bytes(out)


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def frame(msg_type: int, seq: int, payload: bytes) -> bytes:
    """Complete packet on the wire: COBS(type, seq, payload, crc16) followed by a 0x00 delimiter.
    crc16 = CRC16-CCITT-FALSE over type, seq and payload, sent little-endian."""
    raw = bytes([msg_type, seq & 0xFF]) + payload
    raw += struct.pack("<H", crc16_ccitt(raw))
    return cobs_encode(raw) + b"\x00"


# ---- data objects ------------------------------------------------------------
@dataclass
class CanFrame:
    t_us: int
    bus: int
    can_id: int
    data: bytes
    ext: bool = True
    rtr: bool = False


class Mc02Msg:
    """Stand-in for a python-can Message: robstride_mit.LegBus._parse reads only these attributes."""
    __slots__ = ("arbitration_id", "data", "is_extended_id", "is_remote_frame",
                 "timestamp", "channel", "dlc", "t_us")

    def __init__(self, f: CanFrame, channel: str):
        self.arbitration_id = f.can_id
        self.data = f.data
        self.dlc = len(f.data)
        self.is_extended_id = f.ext
        self.is_remote_frame = f.rtr
        self.timestamp = f.t_us / 1e6          # board time base [s]; wraps after ~71.6 min
        self.t_us = f.t_us
        self.channel = channel

    def __repr__(self):
        return (f"Mc02Msg({self.channel} id=0x{self.arbitration_id:08X} "
                f"[{self.dlc}] {self.data.hex(' ')} t={self.t_us}us)")


@dataclass
class ImuSample:
    t_us: int
    acc_raw: tuple
    gyr_raw: tuple
    temp_raw: int
    acc_range_code: int = 0x01
    gyr_range_code: int = 0x01

    @property
    def acc_mss(self):
        # BMI088 ACC_RANGE: 0x00 ±3g, 0x01 ±6g, 0x02 ±12g, 0x03 ±24g -> fs_g = 1.5 * 2**(code+1)
        fs_g = 1.5 * (2 ** (self.acc_range_code + 1))
        k = fs_g * 9.80665 / 32768.0
        return tuple(v * k for v in self.acc_raw)

    @property
    def gyr_rads(self):
        # BMI088 GYRO_RANGE: 0x00 ±2000, 0x01 ±1000, 0x02 ±500 ... dps
        fs_dps = 2000.0 / (2 ** self.gyr_range_code)
        k = fs_dps / 32767.0 * 3.141592653589793 / 180.0
        return tuple(v * k for v in self.gyr_raw)

    @property
    def temp_c(self):
        return self.temp_raw * 0.125 + 23.0


@dataclass
class BusStat:
    rx_cnt: int
    tx_cnt: int
    rx_drop: int
    tx_drop: int
    last_err: int
    flags: int

    @property
    def bus_off(self):
        return bool(self.flags & 1)

    @property
    def err_passive(self):
        return bool(self.flags & 2)


@dataclass
class Status:
    uptime_ms: int
    bus: list
    wd_state: int
    boot_flags: int
    usb_crc_err: int
    usb_cobs_err: int
    usb_rx_ovf: int
    usb_tx_ovf: int


@dataclass
class Pong:
    nonce: int
    proto_ver: int
    fw: tuple
    imu_ok: bool
    acc_range_code: int
    gyr_range_code: int
    uid: str


# ---- link --------------------------------------------------------------------
class MC02Link:
    """One MC02 board = one USB link: a receive queue per CAN bus, an IMU queue and the watchdog events.
    The reader thread puts every received CAN frame straight into the queue of its bus."""

    def __init__(self, port: str = None, ser=None, can_q_size: int = 4096, imu_q_size: int = 2048):
        self.port = port or DEFAULT_PORT
        self.ser = ser if ser is not None else self._open(self.port)
        self.can_q = [Queue(can_q_size) for _ in range(BUS_COUNT)]
        self.imu_q: Queue = Queue(imu_q_size)
        self.wd_events: Queue = Queue(64)
        self.fired_count = 0                   # watchdog FIRED events so far; the bridge worker homes again when it changes
        self.released_count = 0
        self.last_status = None
        self.last_pong = None
        self.dropped_can = 0
        self.dropped_imu = 0
        self.rx_bad = 0                        # packets dropped for a CRC / COBS error
        self.dead = False                      # set when a serial read / write raises (device gone): from then on every
        self.dead_reason = ""                  # send / receive raises RuntimeError instead of idling silently
        self._acc_rc, self._gyr_rc = 0x01, 0x01   # firmware defaults; replaced by the codes reported in PONG
        self._seq = 0
        self._wlock = threading.Lock()
        self._pong_evt = threading.Event()
        self._status_evt = threading.Event()
        self._tsync_evt = threading.Event()
        self._tsync_resp = None
        self._stop = False
        self._rx_buf = bytearray()
        self._t = threading.Thread(target=self._reader, daemon=True, name="mc02-reader")
        self._t.start()

    @staticmethod
    def _open(port: str):
        import serial
        if port == DEFAULT_PORT:
            # /dev/mc02 is a udev link made from the USB product name, so every MC02 board gets this name. With two
            # boards on one host (e.g. legs + arms, or two arms) it is undefined which board it points to, and
            # frames meant for one robot would reach the other, which has the same motor ids and would enable.
            # Refuse to open the port in that case.
            boards = sorted(glob.glob(BY_ID_GLOB))
            if len(boards) > 1:
                raise RuntimeError(
                    f"[mc02] {len(boards)} MC02 boards are plugged into this host, so {DEFAULT_PORT} is ambiguous:\n"
                    + "".join(f"    {b}\n" for b in boards)
                    + "  Unplug the board that is not needed, or set mc02.port in this robot's config to its full path above\n"
                    "  (which board is which: python tools/mc02_selftest.py --port <path> prints the MCU UID and the motor ids)")
        try:
            # The baud rate means nothing on USB CDC. exclusive=True: a second process on the same port would
            # silently steal bytes, so opening fails instead.
            return serial.Serial(port, 921600, timeout=0.02, exclusive=True)
        except (serial.SerialException, OSError) as e:
            raise RuntimeError(
                f"[mc02] cannot open {port}: {e}\n"
                f"  Only one process can open an MC02 board; that process owns both CAN buses and the IMU.\n"
                f"  Stop the bridge / deploy process / `imu_reader.py --daemon` that still holds it.\n"
                f"  If the device name does not exist, check the udev rule\n"
                f"  (ATTRS{{product}}==\"Lucen MC02 v1\" -> /dev/mc02, see docs/mc02.md)") from e

    # -- low-level send --------------------------------------------------------
    def _check_alive(self):
        if self.dead:
            raise RuntimeError(f"[mc02] the link on {self.port} is down ({self.dead_reason}) - after re-plugging, "
                               f"restart the bridge / deploy process (there is no automatic reconnect)")

    def _mark_dead(self, where: str, e: Exception):
        if not self.dead:
            self.dead = True
            self.dead_reason = f"{where}: {type(e).__name__}: {e}"
            print(f"[mc02] !! link on {self.port} lost ({self.dead_reason}). If the board still has power, its watchdog"
                  f" replays the failsafe frames once the host has been silent for the timeout. If USB was unplugged,"
                  f" the board is off and the motors hold their last command until the motor-side CAN timeout"
                  f" releases them (see docs/mc02.md)")

    def _send(self, msg_type: int, payload: bytes = b""):
        self._check_alive()
        with self._wlock:
            try:
                # Send a 0x00 delimiter first. The board collects bytes until it sees 0x00, so stray bytes that the
                # OS writes to a newly opened port (e.g. the AT probe of ModemManager) would otherwise be glued to
                # the first packet and both dropped as one bad packet. The firmware does not count empty packets;
                # the cost is 1 byte per packet.
                self.ser.write(b"\x00" + frame(msg_type, self._seq, payload))
            except Exception as e:
                self._mark_dead("write", e)
                raise RuntimeError(f"[mc02] write to {self.port} failed: {e}") from e
            self._seq = (self._seq + 1) & 0xFF

    # -- API: CAN --------------------------------------------------------------
    def send_can(self, bus: int, can_id: int, data: bytes, ext: bool = True, rtr: bool = False):
        self.send_can_batch([(bus, can_id, data, ext, rtr)])

    def send_can_batch(self, frames):
        """frames: iterable of (bus, can_id, data[, ext, rtr]). Sent as CAN_TX packets of at most 16 frames
        each (firmware limit)."""
        frames = list(frames)
        for i in range(0, len(frames), 16):
            chunk = frames[i:i + 16]
            # CAN_TX payload: u8 count, then count x 15 bytes: u8 bus, u8 flags, u32 id, u8 dlc, u8 data[8]
            payload = bytes([len(chunk)])
            for f in chunk:
                bus, can_id, data = int(f[0]), int(f[1]), bytes(f[2])
                ext = f[3] if len(f) > 3 else True
                rtr = f[4] if len(f) > 4 else False
                if len(data) > 8:
                    raise ValueError(f"CAN data longer than 8 bytes: {data.hex()}")
                flags = (CANF_EXT if ext else 0) | (CANF_RTR if rtr else 0)
                payload += struct.pack("<BBIB", bus, flags, can_id, len(data)) + data.ljust(8, b"\x00")
            self._send(MSG_CAN_TX, payload)

    def recv_can_bus(self, bus: int, timeout: float = 0.1):
        self._check_alive()
        try:
            return self.can_q[bus].get(timeout=timeout)
        except Empty:
            self._check_alive()              # also report a link that died while waiting (not just "no frame")
            return None

    def drain_can(self, bus: int = None):
        for b in ([bus] if bus is not None else range(BUS_COUNT)):
            while True:
                try:
                    self.can_q[b].get_nowait()
                except Empty:
                    break

    # -- API: control / status -------------------------------------------------
    def ping(self, timeout: float = 0.5) -> Pong:
        self._pong_evt.clear()
        self._send(MSG_PING, struct.pack("<I", 0xC0FFEE01))
        if not self._pong_evt.wait(timeout):
            raise TimeoutError(f"no PONG from the MC02 on {self.port} - firmware not running / wrong port / "
                               f"USB cable that only carries power")
        return self.last_pong

    def status(self, timeout: float = 0.5) -> Status:
        self._status_evt.clear()
        self._send(MSG_STATUS_REQ)
        if not self._status_evt.wait(timeout):
            raise TimeoutError(f"no STATUS from the MC02 on {self.port}")
        return self.last_status

    def set_watchdog(self, timeout_ms: int, fire_on_suspend: bool = True):
        """Arm the board's link watchdog; timeout_ms=0 disarms it. Once armed, every valid packet from the
        host (CAN_TX included) feeds it. After timeout_ms of host silence the watchdog fires and the board
        sends the registered failsafe frames (see set_failsafe). fire_on_suspend: also fire on USB suspend.
        Not covered: a pulled USB cable, because that powers the board off."""
        self._send(MSG_WD_CFG, struct.pack("<HB", int(timeout_ms), 1 if fire_on_suspend else 0))

    def set_failsafe(self, slot: int, bus: int, can_id: int, data: bytes, repeat_ms: int = 0, ext: bool = True):
        """Register one failsafe frame. Payload, 18 bytes, exactly as the firmware parses it:
        u8 slot, u8 bus, u8 flags, u32 id, u8 dlc, u8 data[8], u16 repeat_ms.
        repeat_ms=0: sent once when the watchdog fires; otherwise repeated at that period until the host is back."""
        data = bytes(data)
        flags = CANF_EXT if ext else 0
        pl = struct.pack("<BBBIB", slot, bus, flags, can_id, len(data))
        pl += data.ljust(8, b"\x00") + struct.pack("<H", int(repeat_ms))
        assert len(pl) == 18
        self._send(MSG_FAILSAFE_SET, pl)

    def clear_failsafe(self, slot: int = None):
        """slot=None clears all slots (0xFF); otherwise only that slot."""
        self._send(MSG_FAILSAFE_SET, bytes([0xFF if slot is None else int(slot)]))

    def imu_stream(self, div: int = 2):
        """IMU stream divider: 0 = off, 1 = 1 kHz, 2 = 500 Hz, n = 1000/n Hz."""
        self._send(MSG_IMU_CFG, bytes([int(div)]))

    def recv_imu(self, timeout: float = 0.1):
        self._check_alive()
        try:
            return self.imu_q.get(timeout=timeout)
        except Empty:
            self._check_alive()
            return None

    def imu_qsize(self) -> int:
        return self.imu_q.qsize()

    def time_sync(self, timeout: float = 0.5) -> dict:
        """-> {'offset_us': host_mono_us - dev_us, 'rtt_us': ...}"""
        self._tsync_evt.clear()
        t0 = time.monotonic_ns()
        self._send(MSG_TIME_SYNC, struct.pack("<Q", t0))
        if not self._tsync_evt.wait(timeout):
            raise TimeoutError("no TIME_SYNC_RESP from the MC02")
        t1 = time.monotonic_ns()
        _echo, t_dev_us = self._tsync_resp
        rtt_us = (t1 - t0) / 1000.0
        host_mid_us = (t0 / 1000.0) + rtt_us / 2.0
        return {"offset_us": host_mid_us - t_dev_us, "rtt_us": rtt_us}

    def dfu_jump(self):
        """Soft-reset the board into the STM32 system bootloader (USB DFU); flash with dfu-util afterwards
        (see docs/mc02.md)."""
        self._send(MSG_DFU_JUMP, struct.pack("<I", DFU_MAGIC))

    def close(self):
        if self._stop:
            return
        self._stop = True
        self._t.join(timeout=0.5)
        try:
            self.ser.close()
        except Exception:
            pass

    # -- reader thread ---------------------------------------------------------
    def _reader(self):
        while not self._stop:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
            except Exception as e:
                if self._stop:
                    break
                self._mark_dead("read", e)   # device gone: log once, mark the link dead, end the reader thread
                break
            if not chunk:
                continue
            self._rx_buf += chunk
            while True:
                idx = self._rx_buf.find(b"\x00")
                if idx < 0:
                    break
                pkt = bytes(self._rx_buf[:idx])
                del self._rx_buf[:idx + 1]
                if len(pkt) < 4:
                    if pkt:
                        self.rx_bad += 1
                    continue
                raw = cobs_decode(pkt)
                if raw is None or len(raw) < 4 or struct.unpack("<H", raw[-2:])[0] != crc16_ccitt(raw[:-2]):
                    self.rx_bad += 1
                    continue
                try:
                    self._dispatch(raw[0], raw[2:-2])
                except Exception as e:       # one bad packet must not kill the reader thread
                    self.rx_bad += 1
                    if self.rx_bad in (1, 10, 100) or self.rx_bad % 1000 == 0:
                        print(f"[mc02] packet parse error x{self.rx_bad} (dropped): {type(e).__name__}: {e}")

    def _dispatch(self, mtype: int, pl: bytes):
        if mtype == MSG_CAN_RX and len(pl) >= 1:
            # u8 count, then count x 19 bytes: u32 t_us, u8 bus, u8 flags, u32 id, u8 dlc, u8 data[8]
            n = pl[0]
            if len(pl) != 1 + n * 19:
                self.rx_bad += 1
                return
            for i in range(n):
                o = 1 + i * 19
                t_us, bus, flags, cid, dlc = struct.unpack_from("<IBBIB", pl, o)
                data = pl[o + 11:o + 11 + min(dlc, 8)]
                f = CanFrame(t_us, bus, cid, bytes(data), bool(flags & CANF_EXT), bool(flags & CANF_RTR))
                if bus >= BUS_COUNT:
                    self.rx_bad += 1
                    continue
                try:
                    self.can_q[bus].put_nowait(f)           # one queue per bus: each consumer takes only its own frames
                except Full:
                    self.dropped_can += 1
        elif mtype == MSG_IMU and len(pl) == 18:
            # u32 t_us, i16 acc[3], i16 gyr[3], i16 temp_raw: raw LSB, same time base as CAN_RX
            t_us, ax, ay, az, gx, gy, gz, tr = struct.unpack("<I7h", pl)
            s = ImuSample(t_us, (ax, ay, az), (gx, gy, gz), tr, self._acc_rc, self._gyr_rc)
            try:
                self.imu_q.put_nowait(s)
            except Full:
                self.dropped_imu += 1
        elif mtype == MSG_STATUS and len(pl) == 50:
            # u32 uptime_ms; per bus (18 bytes): u32 rx_cnt, tx_cnt, rx_drop, tx_drop, u8 last_err (LEC), u8 flags;
            # from offset 40: u8 wd_state, u8 boot_flags, u16 usb_crc_err, usb_cobs_err, usb_rx_ovf, usb_tx_ovf
            up = struct.unpack_from("<I", pl, 0)[0]
            buses = []
            for b in range(BUS_COUNT):
                o = 4 + b * 18
                buses.append(BusStat(*struct.unpack_from("<IIIIBB", pl, o)))
            wd, bf, ce, co, ro, to = struct.unpack_from("<BBHHHH", pl, 40)
            self.last_status = Status(up, buses, wd, bf, ce, co, ro, to)
            self._status_evt.set()
        elif mtype == MSG_PONG and len(pl) == 22:
            # u32 nonce, u8 proto_ver, u8 fw_major, u8 fw_minor, u8 imu_ok, u8 acc_range, u8 gyr_range, u8 uid[12]
            nonce, pv, fmaj, fmin, iok, arc, grc = struct.unpack_from("<IBBBBBB", pl, 0)
            self._acc_rc, self._gyr_rc = arc, grc
            self.last_pong = Pong(nonce, pv, (fmaj, fmin), bool(iok), arc, grc, pl[10:22].hex())
            self._pong_evt.set()
        elif mtype == MSG_WD_EVENT and len(pl) == 5:
            # u8 event (1 = FIRED, 0 = RELEASED: the host is back), u32 t_us
            ev, t_us = struct.unpack("<BI", pl)
            if ev == 1:
                self.fired_count += 1
            else:
                self.released_count += 1
            try:
                self.wd_events.put_nowait((ev, t_us, time.monotonic()))
            except Full:
                pass
        elif mtype == MSG_TIME_SYNC_RESP and len(pl) == 12:
            # u64 t_host_ns (echo of the request), u32 t_dev_us
            self._tsync_resp = struct.unpack("<QI", pl)
            self._tsync_evt.set()
        elif mtype == MSG_LOG:
            print(f"[mc02] {pl.decode(errors='replace')}")


# ---- per-port singleton with reference counting -------------------------------
_LINKS = {}          # port -> [link, refcount]
_LINKS_LOCK = threading.Lock()


def get_link(port: str = None, ser=None) -> MC02Link:
    """Open each port only once per process. Every user (each CAN bus, the IMU) takes one reference and
    returns it with release_link()."""
    port = port or DEFAULT_PORT
    with _LINKS_LOCK:
        ent = _LINKS.get(port)
        if ent is None:
            ent = [MC02Link(port=port, ser=ser), 0]
            _LINKS[port] = ent
        ent[1] += 1
        return ent[0]


def release_link(link: MC02Link):
    with _LINKS_LOCK:
        ent = _LINKS.get(link.port)
        if ent is None:
            return
        ent[1] -= 1
        if ent[1] > 0:
            return
        del _LINKS[link.port]
    try:
        link.set_watchdog(0)             # last user gone: disarm, so the board does not replay failsafe frames unattended
        link.clear_failsafe()
    except Exception:
        pass
    link.close()


def parse_channel(channel: str) -> int:
    """'mc02:0' -> 0"""
    ch = str(channel)
    if not ch.startswith("mc02:"):
        raise ValueError(f"not an mc02 channel: {channel!r} (expected mc02:0 / mc02:1)")
    bus = int(ch.split(":", 1)[1])
    if not 0 <= bus < BUS_COUNT:
        raise ValueError(f"mc02 bus out of range: {channel!r} (0..{BUS_COUNT - 1})")
    return bus


class MC02CanBus:
    """One CAN bus of the board as a stand-in for a python-can Bus (duck-typed): LegBus uses send / recv /
    shutdown; the bridge also uses register_failsafe / clear_failsafe / arm_watchdog / fired_count."""

    def __init__(self, link: MC02Link, bus: int):
        self.link = link
        self.bus = bus
        self.channel = f"mc02:{bus}"
        self.channel_info = self.channel
        self._slots = []
        self._closed = False

    def send(self, msg, timeout=None):
        self.link.send_can(self.bus, msg.arbitration_id, bytes(msg.data),
                           ext=getattr(msg, "is_extended_id", True),
                           rtr=getattr(msg, "is_remote_frame", False))

    def send_batch(self, msgs):
        """Send the N frames of one control step as a single USB packet (used by LegBus.command_all)."""
        self.link.send_can_batch([(self.bus, m.arbitration_id, bytes(m.data),
                                   getattr(m, "is_extended_id", True), getattr(m, "is_remote_frame", False))
                                  for m in msgs])

    def recv(self, timeout=None):
        f = self.link.recv_can_bus(self.bus, 1e9 if timeout is None else max(0.0, timeout))
        return Mc02Msg(f, self.channel) if f is not None else None

    def register_failsafe(self, frames, repeat_ms: int = 10):
        """frames: [(can_id, data)], at most 8 per bus (slots bus*8+i). Register them before enabling the
        motors: without registered frames a fired watchdog only reports the event and sends nothing."""
        frames = list(frames)
        if len(frames) > SLOTS_PER_BUS:
            raise ValueError(f"mc02:{self.bus}: {len(frames)} failsafe frames, but a bus has only {SLOTS_PER_BUS} slots")
        self.clear_failsafe()
        for i, (cid, data) in enumerate(frames):
            slot = self.bus * SLOTS_PER_BUS + i
            self.link.set_failsafe(slot, self.bus, cid, data, repeat_ms=repeat_ms)
            self._slots.append(slot)

    def clear_failsafe(self):
        for slot in self._slots:
            self.link.clear_failsafe(slot)
        self._slots = []

    def arm_watchdog(self, timeout_ms: int):
        self.link.set_watchdog(timeout_ms, fire_on_suspend=True)

    @property
    def fired_count(self) -> int:
        return self.link.fired_count

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.clear_failsafe()
        except Exception:
            pass
        release_link(self.link)


def open_mc02_bus(channel: str, port: str = None, ser=None) -> MC02CanBus:
    """mc02 backend of robstride_mit.open_bus: 'mc02:0' / 'mc02:1' -> MC02CanBus (channels on the same port
    share one link)."""
    bus = parse_channel(channel)
    return MC02CanBus(get_link(port, ser=ser), bus)
