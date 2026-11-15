#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/imu_reader.py - torso IMU of biped_Laika: raw BMI088 samples from the MC02 board -> fused attitude.

The MC02 firmware sends raw gyro / accelerometer samples only. The attitude fusion (a Mahony filter) and the
mount calibration run here, on the host.

Outputs, expressed in the torso (base) frame - x forward, y left, z up:
  ang_vel_base   angular velocity [rad/s]
  gravity_base   projected gravity as a unit vector: [0, 0, -1] when the torso is upright
In a program:
  from tools.imu_reader import ImuReader
  imu = ImuReader(cfg["imu"])                               # cfg = the loaded robot config
  ang_vel_base, gravity_base, age_s = imu.latest()          # age_s = age of the data in seconds

Command line (on the robot computer; needs pyserial):
  python tools/imu_reader.py --view          # print angular velocity / gravity / data age continuously
  python tools/imu_reader.py --calib-level   # robot standing level and still: samples 10 s, prints mount_rpy_deg
  python tools/imu_reader.py --verify-axes   # nod the torso down, then tilt it to the right: checks mount_axes
  python tools/imu_reader.py --daemon        # standalone fusion process publishing to a shared state file

Mount calibration (`imu` section of biped_Laika/biped_Laika.yaml), v_base = R @ v_board:
  mount_axes     which board axis points to the torso's front / left / up (given by how the board is mounted)
  mount_rpy_deg  fine roll / pitch correction applied after the axis mapping; --calib-level solves it from
                 the gravity reading of the level robot, which must become [0, 0, -1]
Gravity cannot show a rotation about the vertical axis, so a board mounted 90 or 180 degrees off still passes
--calib-level. After any change to the IMU mounting run --verify-axes first, then --calib-level.

Only one process can open the MC02 board. With the default IMU source (`dm_mc02_can_v1`: the samples arrive
over the MC02 link, tools/mc02_link.py) stop the bridge / deploy process before running this tool; the deploy
process runs the same fusion itself. While --daemon is running, --view / --calib-level / --verify-axes attach
to its state file instead of opening the device.
"""
import argparse
import math
import mmap
import os
import signal
import struct
import sys
import tempfile
import threading
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_AX = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0],
       "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}


def mount_matrix(imu_cfg):
    """v_base = R @ v_board: the axis mapping (mount_axes) first, then the fine rpy correction (mount_rpy_deg)."""
    ax = imu_cfg["mount_axes"]
    R_axes = np.array([_AX[ax["x"]], _AX[ax["y"]], _AX[ax["z"]]], dtype=float)
    r, p, y = [math.radians(v) for v in imu_cfg["mount_rpy_deg"]]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx @ R_axes


def tilt_deg(g):
    """Angle in degrees between a gravity vector and the ideal [0, 0, -1]."""
    g = np.asarray(g, dtype=float)
    g = g / np.linalg.norm(g)
    return math.degrees(math.acos(np.clip(g @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0)))


def solve_mount_rpy(g_base, cfg):
    """Solve the mount_rpy_deg to put into the config from a gravity_base measured with the current mount.

    mount_matrix = R_rpy @ R_axes: the rpy rotation acts on the vector after the axis mapping. So only the
    current R_rpy is removed (not the whole R), which gives the axis-mapped, uncorrected gravity u. Then:
      Rx removes the y component -> roll  = atan2(-uy, -uz)
      Ry removes the x component -> pitch = atan2(ux, hypot(uy, uz))
    so that R_rpy_new @ u == [0, 0, -1].
    Yaw cannot be observed from gravity (a rotation about the gravity axis leaves gravity unchanged). It is
    always 0 here; the heading of the board is set by mount_axes.
    """
    R_full = mount_matrix(cfg)
    R_axes = mount_matrix({**cfg, "mount_rpy_deg": [0.0, 0.0, 0.0]})
    R_rpy_old = R_full @ R_axes.T
    u = R_rpy_old.T @ np.asarray(g_base, dtype=float)
    u = u / np.linalg.norm(u)
    gx, gy, gz = u
    roll = math.degrees(math.atan2(-gy, -gz))
    pitch = math.degrees(math.atan2(gx, math.hypot(gy, gz)))
    return [roll, pitch, 0.0]


class Mahony:
    """Attitude fusion in the board frame; gravity_board() returns the unit gravity direction in that frame."""

    def __init__(self, kp=2.0, ki=0.05):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.kp, self.ki = kp, ki
        self.integ = np.zeros(3)

    def _rot(self):
        w, x, y, z = self.q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def update(self, gyro, accel, dt):
        a = np.asarray(accel, dtype=float)
        w = np.asarray(gyro, dtype=float)
        n = np.linalg.norm(a)
        if 2.0 < n < 30.0:                       # use gravity only when the accelerometer norm is plausible
            v_meas = -a / n                      # measured gravity direction, board frame
            v_est = self._rot().T @ np.array([0.0, 0.0, -1.0])
            err = np.cross(v_meas, v_est)
            self.integ += self.ki * err * dt
            w = w + self.kp * err + self.integ
        q = self.q
        dq = 0.5 * np.array([
            -q[1] * w[0] - q[2] * w[1] - q[3] * w[2],
            q[0] * w[0] + q[2] * w[2] - q[3] * w[1],
            q[0] * w[1] - q[1] * w[2] + q[3] * w[0],
            q[0] * w[2] + q[1] * w[1] - q[2] * w[0]])
        self.q = self.q + dq * dt
        self.q /= np.linalg.norm(self.q)

    def gravity_board(self):
        return self._rot().T @ np.array([0.0, 0.0, -1.0])


# ---------- standalone fusion process (--daemon) ----------
# One fusion step takes only ~50 µs; what hurts is contention. When the fusion thread shares a Python process
# with the policy loop, anything that holds the GIL for tens of ms makes the 500 Hz reader fall behind. In a
# process of its own the reader keeps up, and a consumer reads one 128-byte snapshot per control step.
#
# Division of work: the daemon owns the device and runs the Mahony filter. At 500 Hz it publishes the
# board-frame quaternion / gyro / accelerometer. The mount_axes / mount_rpy_deg correction is applied by each
# consumer, so a new calibration needs only the consumer restarted. --view, --calib-level and --verify-axes
# attach to the daemon in the same way. If the daemon dies, the timestamp stops advancing, the data age grows,
# and the deploy process trips its stale-IMU limit (age > 0.1 s -> damping, then disable).
#
# A plain file + mmap is used instead of multiprocessing.shared_memory: before Python 3.13 the resource
# tracker of an exiting consumer removes the segment, which would take the daemon's data away with it.
class ShmState:
    """Single-writer, multi-reader IMU state file: [seq | pid, stamp, q4, gyro3, accel3, backlog].

    Two-phase seqlock. The writer makes seq odd (write in progress), writes the payload, then makes seq even.
    A reader reads seq (it must be even), copies the payload and reads seq again; if the two values differ or
    seq is odd, it retries. On weakly ordered CPUs (aarch64) a torn read can still slip through a window of a
    few ns; the |q| ≈ 1 check catches it, and if all retries fail the reader keeps the last good frame (one
    sample, 2 ms, of stale data: below the sensor noise).
    """
    _HEAD = struct.Struct("<Q")
    _PAY = struct.Struct("<Qd4d3d3dQ")   # pid,stamp,q4,gyro3,accel3,backlog
    SIZE = 128

    def __init__(self, mm, writable, path):
        self.mm, self.writable, self.path = mm, writable, path
        self._seq = 0
        self._last = None                # reader side: the last good frame

    @staticmethod
    def default_path():
        base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        return os.path.join(base, "lucen_imu_state")

    @classmethod
    def create(cls, path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(b"\x00" * cls.SIZE)
        os.replace(tmp, path)            # appears atomically: a reader never sees a half-written file
        f = open(path, "r+b")
        return cls(mmap.mmap(f.fileno(), cls.SIZE), True, path)

    @classmethod
    def attach(cls, path):
        f = open(path, "rb")
        return cls(mmap.mmap(f.fileno(), cls.SIZE, access=mmap.ACCESS_READ), False, path)

    def write(self, stamp, q, gyro, accel, backlog):
        pay = self._PAY.pack(os.getpid(), stamp, *q, *gyro, *accel, int(backlog))
        self._seq += 1                                   # odd = write in progress
        self.mm[:8] = self._HEAD.pack(self._seq)
        self.mm[8:8 + len(pay)] = pay
        self._seq += 1                                   # even = stable
        self.mm[:8] = self._HEAD.pack(self._seq)

    def read(self):
        """-> (stamp, q, gyro, accel, backlog, pid), or None if nothing has ever been written."""
        for _ in range(4):
            s1 = self._HEAD.unpack(bytes(self.mm[:8]))[0]
            if s1 == 0 or s1 % 2:
                continue
            v = self._PAY.unpack(bytes(self.mm[8:8 + self._PAY.size]))
            if self._HEAD.unpack(bytes(self.mm[:8]))[0] != s1:
                continue
            q = np.array(v[2:6])
            if abs(np.linalg.norm(q) - 1.0) < 0.01:
                self._last = (v[1], q, np.array(v[6:9]), np.array(v[9:12]), v[12], v[0])
                return self._last
        return self._last


class ImuReader:
    """IMU state source. Three modes:
      auto    default. If the state file exists and its data is fresh, attach to the standalone daemon: the
              device is not opened, and latest() / yaw() read one 128-byte snapshot per call. Otherwise run
              the fusion in a thread of this process.
      serve   the daemon itself: no thread; serve() owns the device, fuses and publishes in the foreground.
      inproc  always use the in-process thread (for tests).
    latest() / yaw() / backlog_events mean the same in all three modes."""

    def __init__(self, cfg=None, port=None, mode="auto"):
        if cfg is None:
            cfg = yaml.safe_load(open(os.path.join(ROOT, "biped_Laika/biped_Laika.yaml")))["imu"]
        self.cfg = cfg
        self.port = port or cfg["port"]
        # Data source: dm_mc02_can_v1 = binary samples over tools/mc02_link, the USB link shared with the two
        # CAN buses; dm_mc02_usb = ASCII lines from an IMU-only firmware on a serial port of its own.
        self.source = cfg.get("source", "dm_mc02_usb")
        self.mc02_ranges = None                  # (acc_range_code, gyr_range_code) reported by the firmware in PONG
        self.R = mount_matrix(cfg)
        self.fuse = Mahony()
        self.lock = threading.Lock()
        self._gyro_b = np.zeros(3)
        self._stamp = 0.0
        self._last_t = None
        self.raw_accel = np.zeros(3)             # used by the calibration
        self._backlog = 0                        # inproc / serve: backlog discards counted in this process
        self._backlog_seen = 0                   # attached: daemon backlog count already reported here
        self._warn_t = 0.0
        self.backlog_bytes = 0
        self.shm = None
        path = cfg.get("state_file") or ShmState.default_path()
        if mode == "auto" and os.path.exists(path):
            self.shm = ShmState.attach(path)
            rec = self.shm.read()
            age = None if rec is None else time.monotonic() - rec[0]
            if rec is None or age > 1.0:
                # No silent fallback: the daemon may still hold the device (two readers steal each other's bytes).
                raise RuntimeError(
                    f"[imu] {path} exists but "
                    f"{'has never been written' if rec is None else f'its data is {age:.1f}s old'}"
                    f" - is the daemon dead? Restart `python tools/imu_reader.py --daemon`,"
                    f" or make sure the daemon has exited and delete the file to use in-process fusion")
            print(f"[imu] attitude comes from the standalone daemon (pid {rec[5]}, data age {age*1000:.0f} ms);"
                  f" the mount correction is applied in this process")
            return                               # no thread, the device is not opened
        if mode == "serve":
            return                               # serve() runs the loop in the foreground
        if mode == "auto":
            if self.source == "dm_mc02_can_v1":
                print(f"[imu] no daemon state file ({path}): fusing the MC02 samples in this process"
                      f" (normal for this source: one process owns the MC02 link)")
            else:
                print(f"[imu] no daemon state file ({path}): falling back to the in-process fusion thread."
                      f" Sharing a process with the policy can make the IMU reader fall behind;"
                      f" consider starting `python tools/imu_reader.py --daemon` first")
        threading.Thread(target=self._run_loop, daemon=True).start()

    @property
    def backlog_events(self):
        """Number of times a sample backlog was discarded (direct evidence of attitude lag); the safety check
        of the deploy process reads it."""
        if self.shm is not None and not self.shm.writable:
            rec = self.shm.read()
            return rec[4] if rec else 0
        return self._backlog

    def _refresh(self):
        """Attached mode: copy the daemon snapshot into the same internal state the in-process mode uses."""
        rec = self.shm.read()
        if rec is None:
            return
        stamp, q, gyro, accel, backlog, _pid = rec
        with self.lock:
            self.fuse.q = q
            self._gyro_b = gyro
            self.raw_accel = accel
            self._stamp = stamp
        if backlog > self._backlog_seen:         # repeat the daemon's backlog warning in this process
            self._backlog_seen = backlog
            now = time.monotonic()
            if now - self._warn_t > 2.0:
                self._warn_t = now
                print(f"[imu] !! the daemon reports a sample backlog ({backlog} time(s) in total)")

    def serve(self):
        """Daemon main loop, in the foreground: own the device, fuse, publish at 500 Hz. SIGINT / SIGTERM
        remove the state file."""
        path = self.cfg.get("state_file") or ShmState.default_path()
        if os.path.exists(path):
            try:
                rec = ShmState.attach(path).read()
            except (OSError, ValueError):
                rec = None
            if rec is not None and time.monotonic() - rec[0] < 1.0:
                sys.exit(f"[imu] a daemon is already running (pid {rec[5]}, {path})"
                         f" - two processes cannot share one device")
            print(f"[imu] taking over the stale state file {path}")
        # Open the device before creating the state file: if it cannot be opened, no empty file is left behind
        # for consumers to attach to. (MC02 source: the link is opened in _mc02_loop, which raises RuntimeError.)
        ser = self._open_serial() if self.source != "dm_mc02_can_v1" else None
        pub = ShmState.create(path)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # a plain kill also runs the clean-up in finally
        print(f"[imu] daemon: {self.port} ({self.source}) @ {self.cfg['rate_hz']}Hz -> {path}\n"
              f"      publishing the board-frame state; consumers apply mount_axes / mount_rpy_deg, so a new"
              f" calibration needs no daemon restart")
        if self.source == "dm_mc02_can_v1":
            print("      WARNING: only one process can open the MC02 board. Do NOT run this daemon together with the"
                  " bridge or the deploy process (it would hold the link); the deploy process fuses the IMU itself."
                  " Use this daemon only for standalone calibration / viewing")
        try:
            self._run_loop(pub, ser)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _open_serial(self):
        import serial
        # The baud rate means nothing on USB CDC. imu.baud overrides it (a macOS pty rejects non-standard rates).
        return serial.Serial(self.port, int(self.cfg.get("baud", 921600)), timeout=1)

    def _run_loop(self, pub=None, ser=None):
        """Dispatch on the data source: binary samples over the MC02 link, or ASCII lines from a serial port.
        Fusion and publishing are the same for both."""
        if self.source == "dm_mc02_can_v1":
            self._mc02_loop(pub)
        else:
            self._serial_loop(pub, ser)

    def _fuse_one(self, gyro, accel, nominal_dt, pub):
        now = time.monotonic()
        dt = nominal_dt if self._last_t is None else min(max(now - self._last_t, 1e-4), 0.05)
        self._last_t = now
        self.fuse.update(gyro, accel, dt)
        with self.lock:
            self._gyro_b = gyro
            self.raw_accel = accel
            self._stamp = now
        if pub is not None:
            pub.write(now, self.fuse.q, gyro, accel, self._backlog)

    def _mc02_loop(self, pub=None):
        """MC02 link source: raw int16 samples from tools/mc02_link (the USB link shared with the two CAN
        buses). The range codes come from the firmware's PONG, not from constants; IMU_CFG div = 1000 / rate_hz."""
        from tools.mc02_link import get_link
        link = get_link(self.port)
        try:
            pong = link.ping()
        except TimeoutError as e:
            raise RuntimeError(f"[imu] no answer from the MC02 on {self.port} - firmware not running or wrong port: {e}")
        if not pong.imu_ok:
            raise RuntimeError("[imu] the MC02 firmware reports that the BMI088 failed to initialize (boot_flags bit1): "
                               "the on-board IMU does not answer (SPI2, chip selects PC0 / PC3)")
        self.mc02_ranges = (pong.acc_range_code, pong.gyr_range_code)
        div = max(1, int(round(1000.0 / float(self.cfg["rate_hz"]))))
        link.imu_stream(div)
        nominal_dt = div / 1000.0
        print(f"[imu] MC02 source: fw {pong.fw[0]}.{pong.fw[1]}, range codes acc={pong.acc_range_code}"
              f" gyr={pong.gyr_range_code}, IMU stream {1000 // div} Hz")
        # Backlog guard (same idea as for the serial source): samples waiting in the queue are old, and the data
        # age does not show it. At about 50 ms of backlog, discard all but the newest few samples.
        BACKLOG_N = 25
        warned_t = 0.0
        while True:
            n = link.imu_qsize()
            if n > BACKLOG_N:
                for _ in range(n - 5):
                    link.recv_imu(timeout=0)
                self._backlog += 1
                self.backlog_bytes = n * 18
                now = time.monotonic()
                if now - warned_t > 2.0:
                    warned_t = now
                    print(f"[imu] !! IMU backlog of {n} samples (~{n*nominal_dt*1000:.0f} ms of attitude lag), discarded;"
                          f" {self._backlog} time(s) in total - the fusion thread cannot keep up with {1000 // div} Hz")
            s = link.recv_imu(timeout=1.0)
            if s is None:
                continue
            self._fuse_one(np.array(s.gyr_rads), np.array(s.acc_mss), nominal_dt, pub)

    def _serial_loop(self, pub=None, ser=None):
        if ser is None:
            ser = self._open_serial()
        # Bytes left over from a previous process would be parsed as fresh data: drop them.
        ser.reset_input_buffer()
        nominal_dt = 1.0 / self.cfg["rate_hz"]
        # Backlog guard: `age` is the time since a line was last *parsed*, not since the firmware *sampled* it.
        # A thread that cannot keep up with 500 Hz still parses lines all the time, so age stays small while
        # every line is old, and the consumer's age > 0.1 s check cannot see it. The stream is about 30 KB/s
        # (500 Hz x ~60 bytes), so 3 KB of backlog means 100 ms of attitude lag.
        BACKLOG_WARN = 1500          # bytes, about 50 ms
        warned_t = 0.0
        while True:
            n = ser.in_waiting
            if n > BACKLOG_WARN:
                self.backlog_bytes = n
                # Discard all but the newest few lines: a jump in attitude is better than a lagging one.
                ser.read(n - BACKLOG_WARN // 4)
                self._backlog += 1
                now = time.monotonic()
                if now - warned_t > 2.0:
                    warned_t = now
                    print(f"[imu] !! serial backlog of {n} B (~{n/30000*1000:.0f} ms of attitude lag), "
                          f"discarded; {self._backlog} time(s) in total - the fusion thread cannot keep up with 500Hz")
            line = ser.readline().decode(errors="ignore").strip()
            if not line.startswith("I,"):
                continue
            try:
                f = [float(x) for x in line.split(",")[2:8]]
            except ValueError:
                continue
            self._fuse_one(np.array(f[:3]), np.array(f[3:6]), nominal_dt, pub)

    def yaw(self):
        """Torso yaw [rad], obtained by integrating the gyro: a 6-axis IMU cannot derive yaw from gravity.

        It drifts slowly (a few degrees per minute once the BMI088 is warm). Use it only as a relative heading
        over tens of seconds, never as an absolute bearing. tools/deploy_policy.py uses it for the heading
        hold (--heading), a proportional loop added on the deployment side.
        """
        if self.shm is not None and not self.shm.writable:
            self._refresh()
        R = self.R @ self.fuse._rot()          # full attitude, board -> base frame (mount correction included)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def latest(self):
        """-> (ang_vel_base, gravity_base, age_s) in the torso frame (x forward, y left, z up):
        angular velocity [rad/s], unit projected gravity ([0, 0, -1] when upright), age of the data [s]."""
        if self.shm is not None and not self.shm.writable:
            self._refresh()
        with self.lock:
            gyro, stamp = self._gyro_b.copy(), self._stamp
        g_board = self.fuse.gravity_board()
        return self.R @ gyro, self.R @ g_board, time.monotonic() - stamp


def verify_axes(reader, hold_s=6.0, min_deg=15.0):
    """Check the orientation of the IMU about the vertical axis. The gravity calibration cannot see it (a
    rotation about the gravity axis leaves the gravity reading unchanged), but a board mounted 90 or 180
    degrees off makes the policy correct pitch / roll in the wrong direction: the robot falls as soon as it
    is put on the ground.

    Torso frame: x forward, y left, z up. Therefore:
      nod down          (rotation about +y): ang_vel[1] > 0, gravity[0] > 0
      tilt to the right (rotation about +x): ang_vel[0] > 0, gravity[1] < 0
    """
    # (prompt, gyro axis, expected gyro sign, gravity component, expected sign of its change)
    tests = [
        ("nod the torso DOWN (lean the robot forward), then bring it back upright", 1, +1, 0, +1),
        ("tilt the torso to the RIGHT (the right side goes down), then bring it back upright", 0, +1, 1, -1),
    ]
    ok_all = True
    for prompt, w_ax, w_sign, g_ax, g_sign in tests:
        input(f"\n>>> Slowly {prompt} (tilt by more than {min_deg:.0f}°). Press Enter when ready to start...")
        print(f"    Recording for {hold_s:.0f}s - move now...")
        t0 = time.monotonic()
        w_peak = np.zeros(3)
        g_peak = np.zeros(3)
        g0 = reader.latest()[1]
        while time.monotonic() - t0 < hold_s:
            w, g, _ = reader.latest()
            if abs(w[w_ax]) > abs(w_peak[w_ax]):
                w_peak = w.copy()
            if abs(g[g_ax] - g0[g_ax]) > abs(g_peak[g_ax]):
                g_peak = g - g0
            time.sleep(0.02)
        tilt = math.degrees(math.asin(min(abs(g_peak[g_ax]), 1.0)))
        w_ok = w_peak[w_ax] * w_sign > 0
        g_ok = g_peak[g_ax] * g_sign > 0
        # the dominant gyro axis must be the expected one; otherwise the board is mounted 90° off
        dominant = int(np.argmax(np.abs(w_peak)))
        ax_ok = dominant == w_ax
        print(f"    peak ang_vel={np.round(w_peak, 3)} (dominant axis {'xyz'[dominant]}), "
              f"gravity change={np.round(g_peak, 3)}, tilt ≈{tilt:.0f}°")
        if tilt < min_deg * 0.5:
            print(f"    ?? The motion was too small ({tilt:.0f}°): the result is not reliable, repeat this step")
            ok_all = False
            continue
        for label, good in (("dominant gyro axis", ax_ok), ("gyro sign", w_ok), ("sign of the gravity change", g_ok)):
            print(f"    {'OK  ' if good else 'FAIL'} {label}")
            ok_all &= good
    print()
    if ok_all:
        print("[PASS] IMU axes match the torso frame (x forward / y left / z up); the IMU orientation is safe for ground tests")
    else:
        print("[FAIL] IMU axes are wrong - on the ground the policy would correct in the wrong direction and the robot would fall.")
        print("       Edit imu.mount_axes in biped_Laika/biped_Laika.yaml to match how the board is actually mounted:")
        print("       x / y / z of mount_axes name the board axis that points to the torso's front / left / up,")
        print("       e.g. a board mounted turned by 180° -> {x: '-x', y: '-y', z: '+z'}")
        print("       Then run this command again, followed by --calib-level.")
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="serial port of the MC02 board (default: the port in the robot config)")
    ap.add_argument("--view", action="store_true", help="print the fused attitude continuously")
    ap.add_argument("--calib-level", action="store_true",
                    help="with the robot standing level and still, sample for 10 s and print the mount_rpy_deg "
                         "to put into biped_Laika/biped_Laika.yaml")
    ap.add_argument("--verify-axes", action="store_true",
                    help="follow the prompts (nod the torso down, tilt it to the right) to check the IMU axis mapping; "
                         "the gravity calibration cannot detect a board that is turned about the vertical axis")
    ap.add_argument("--daemon", action="store_true",
                    help="run as a standalone fusion process: owns the IMU device and publishes snapshots to a shared"
                         " state file at 500Hz; other IMU consumers attach to it automatically")
    args = ap.parse_args()
    if args.daemon:
        ImuReader(port=args.port, mode="serve").serve()
        return
    r = ImuReader(port=args.port)
    time.sleep(2.0)                              # let the fusion converge
    if args.calib_level:
        print("Sampling for 10 s: keep the robot standing level and still...")
        gs = []
        for _ in range(100):
            _, g, age = r.latest()
            if age < 0.1:
                gs.append(g)
            time.sleep(0.1)
        g = np.mean(gs, axis=0)
        g /= np.linalg.norm(g)
        print(f"measured gravity_base = {np.round(g, 4)} (ideal [0,0,-1])")
        print(f"residual tilt now = {tilt_deg(g):.3f}°")
        rpy = solve_mount_rpy(g, r.cfg)
        # Self-check: apply the solution and confirm that the residual really shrinks (catches a sign error).
        cfg2 = {**r.cfg, "mount_rpy_deg": rpy}
        g_fix = mount_matrix(cfg2) @ (r.R.T @ g)      # back to the board frame, then correct with the new R
        err = tilt_deg(g_fix)
        print(f"residual after the correction = {err:.3f}°  ({np.round(g_fix, 5)})")
        if err > 0.1:
            print("!! Self-check failed: the solved correction does not level the measurement. Do not put it into the config; "
                  "check imu.mount_axes with --verify-axes and repeat")
        else:
            print(f"put into biped_Laika/biped_Laika.yaml -> mount_rpy_deg: [{rpy[0]:.2f}, {rpy[1]:.2f}, 0.0]"
                  f"  (yaw comes from the mechanical orientation in mount_axes; gravity cannot measure it)")
        return
    if args.verify_axes:
        verify_axes(r)
        return
    while True:
        w, g, age = r.latest()
        print(f"\rang_vel={np.round(w, 3)}  gravity={np.round(g, 3)}  "
              f"age={age*1000:.0f}ms   ", end="", flush=True)
        time.sleep(0.05)


if __name__ == "__main__":
    main()
