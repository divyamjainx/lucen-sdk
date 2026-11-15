# MC02 USB protocol

This page describes the protocol between a host and the MC02 board, protocol version 1, for writing your own
host software. For wiring, self-test and firmware update see [`mc02.md`](mc02.md).

The reference implementation is `tools/mc02_link.py` (class `MC02Link`). It needs only `pyserial`, and every
layout on this page can be checked against it.

## 1. Overview

- Transport: USB Full-Speed CDC-ACM, one serial device (`/dev/mc02` on Linux). The baud rate setting has no
  effect. Only one process can have the port open.
- The board is a transparent bridge. The host builds and parses the motor CAN frames (the SDK does this in
  `robstride_mit.py`); the board forwards them and knows nothing about the motor protocol.
- One link carries: CAN frames to and from two buses, raw IMU samples, the link watchdog with its failsafe
  frames, status counters, time synchronisation, and a command that restarts the board into its USB
  bootloader.
- All multi-byte fields are little-endian.

## 2. Packet framing

A packet, before encoding:

| Offset | Field | Description |
|---|---|---|
| 0 | `u8 type` | Message type (section 3) |
| 1 | `u8 seq` | Rolling counter, separate for each direction. For diagnostics only |
| 2 | `u8 payload[N-4]` | Payload, at most 250 bytes |
| N-2 | `u16 crc16` | CRC of everything before it (`type`, `seq`, `payload`) |

- **CRC**: CRC16-CCITT-FALSE. Polynomial 0x1021, initial value 0xFFFF, bits processed most significant
  first, no final XOR. Check value: the CRC of the ASCII string `123456789` is 0x29B1.
- **Encoding**: the whole packet, CRC included, is encoded with
  [COBS](https://en.wikipedia.org/wiki/Consistent_Overhead_Byte_Stuffing), and a single **0x00** byte
  terminates it on the wire.
- **Resynchronisation**: after lost or corrupted bytes, either side is back in step at the next 0x00.
- **Errors**: a packet that fails COBS decoding or the CRC is dropped as a whole and counted (`usb_cobs_err`,
  `usb_crc_err` in STATUS). The stream continues.
- **Empty packets** (two consecutive 0x00 bytes) are ignored by the board and not counted. The reference
  implementation sends a 0x00 before every packet, so that stray bytes written to a freshly opened port by
  the operating system (modem probing, for example) cannot merge with the first real packet.

Worked example, a PING with `seq` = 0 and `nonce` = 0x12345678:

```
packet without CRC   02 00 78 56 34 12
CRC16 = 0x55AA       02 00 78 56 34 12 AA 55
COBS + delimiter     02 02 07 78 56 34 12 AA 55 00
```

## 3. Message types

### Host to board

| Type | Name | Payload | Notes |
|---|---|---|---|
| 0x01 | CAN_TX | `u8 count`, then `count` records of 15 bytes | `count` is at most 16. Record layout below |
| 0x02 | PING | `u32 nonce` | Answered with PONG |
| 0x03 | WD_CFG | `u16 timeout_ms; u8 flags` | Arms the watchdog; `timeout_ms` = 0 disarms it. `flags` bit 0: also fire on USB suspend. Section 7 |
| 0x04 | FAILSAFE_SET | 18 bytes to register a slot, 1 byte to clear | Registers or clears a failsafe frame. Layout in section 7 |
| 0x05 | TIME_SYNC | `u64 t_host_ns` | Answered with TIME_SYNC_RESP. Section 5 |
| 0x06 | IMU_CFG | `u8 div` | IMU stream rate: 0 = off (the state after power-on), 1 = 1 kHz, 2 = 500 Hz, n = 1000/n Hz. Section 6 |
| 0x07 | STATUS_REQ | empty | Answered with STATUS at once |
| 0x08 | DFU_JUMP | `u32 magic` = 0x44465521 | Restarts the board into the USB bootloader. Section 8 |

### Board to host

| Type | Name | Payload | Notes |
|---|---|---|---|
| 0x81 | CAN_RX | `u8 count`, then `count` records of 19 bytes | Frames received on both buses, in one stream, each with a device time stamp |
| 0x82 | IMU | 18 bytes | One raw IMU sample. Section 6 |
| 0x83 | STATUS | 50 bytes | Sent twice per second, and on STATUS_REQ. Section 4 |
| 0x84 | PONG | 22 bytes | Identity. Section 4 |
| 0x85 | WD_EVENT | `u8 event; u32 t_us` | `event` 1 = FIRED, 0 = RELEASED (host returned). Section 7 |
| 0x86 | LOG | ASCII text | Debug text from the firmware |
| 0x87 | TIME_SYNC_RESP | `u64 t_host_ns; u32 t_dev_us` | Echo of the host time, and the device time. Section 5 |

### CAN frame records

CAN_TX record, 15 bytes:

| Offset | Field | Description |
|---|---|---|
| 0 | `u8 bus` | 0 = CAN1, 1 = CAN2 |
| 1 | `u8 flags` | bit 0 = EXT (29-bit identifier), bit 1 = RTR |
| 2 | `u32 id` | CAN identifier |
| 6 | `u8 dlc` | Number of data bytes, 0-8 |
| 7 | `u8 data[8]` | Data, padded with zeros to 8 bytes |

CAN_RX record, 19 bytes:

| Offset | Field | Description |
|---|---|---|
| 0 | `u32 t_us` | Device time stamp, taken in the receive interrupt (section 5) |
| 4 | `u8 bus` | 0 = CAN1, 1 = CAN2 |
| 5 | `u8 flags` | bit 0 = EXT, bit 1 = RTR |
| 6 | `u32 id` | CAN identifier |
| 10 | `u8 dlc` | Number of valid data bytes |
| 11 | `u8 data[8]` | Data; only the first `dlc` bytes are valid |

Both buses run classic CAN at 1 Mbps without filtering. All RobStride motor frames are extended frames, so
bit 0 of `flags` is set for them. Bus 0 is CAN1, the port group at the top right of the board; bus 1 is CAN2,
at the top left. Which robot joints are on which bus is listed in [`mc02.md`](mc02.md), section 2.

A CAN_TX packet with several records is the efficient way to send one control step: the SDK sends all the
frames of one bus for one step in a single USB packet.

## 4. Identity and status

### PONG (0x84), 22 bytes

| Offset | Field | Description |
|---|---|---|
| 0 | `u32 nonce` | The nonce of the PING being answered |
| 4 | `u8 proto` | Protocol version: 1 |
| 5 | `u8 fw_maj` | Firmware version, major |
| 6 | `u8 fw_min` | Firmware version, minor |
| 7 | `u8 imu_ok` | Not 0 = the BMI088 initialised |
| 8 | `u8 acc_range` | Accelerometer full-scale code (section 6) |
| 9 | `u8 gyr_range` | Gyroscope full-scale code (section 6) |
| 10 | `u8 uid[12]` | The MCU's 96-bit unique id |

Check `proto` before anything else; the host and the firmware must agree on it.

### STATUS (0x83), 50 bytes

| Offset | Field | Description |
|---|---|---|
| 0 | `u32 uptime_ms` | Time since the board started |
| 4 | bus 0 block, 18 bytes | Counters of CAN1, layout below |
| 22 | bus 1 block, 18 bytes | Counters of CAN2 |
| 40 | `u8 wd_state` | Watchdog state: 0 = not configured, 1 = armed, 2 = fired |
| 41 | `u8 boot_flags` | Start-up checks. Bit 0 (0x01) = the CAN clock does not match the bit timing, so the CAN bit rate is wrong. Bit 1 (0x02) = the IMU failed to initialise. 0 = healthy |
| 42 | `u16 usb_crc_err` | Packets from the host dropped for a bad CRC |
| 44 | `u16 usb_cobs_err` | Packets from the host dropped for a COBS error |
| 46 | `u16 usb_rx_ovf` | USB receive overflow count |
| 48 | `u16 usb_tx_ovf` | USB transmit overflow count |

Per-bus block, 18 bytes:

| Offset | Field | Description |
|---|---|---|
| 0 | `u32 rx_cnt` | CAN frames received |
| 4 | `u32 tx_cnt` | CAN frames sent |
| 8 | `u32 rx_drop` | Received frames dropped by the board |
| 12 | `u32 tx_drop` | Frames to send dropped by the board |
| 16 | `u8 last_err` | LEC, the CAN controller's last error code: 0 none, 1 stuff, 2 form, 3 ACK, 4 bit-1, 5 bit-0, 6 CRC, 7 no change since the last read |
| 17 | `u8 flags` | bit 0 = bus-off, bit 1 = error-passive |

On a healthy link the drop counters and the USB error counters stay at 0. Do not drive motors while
`boot_flags` is not 0. `last_err` describes the most recent error, so read STATUS again after sending frames
to judge a bus. LEC = 3 on a silent bus means that frames were sent and no node acknowledged them.

## 5. Time stamps

- `t_us` is a free-running 32-bit counter at 1 MHz on the board. It wraps about every 71.6 minutes; work
  with differences.
- CAN frames are stamped in the receive interrupt, IMU samples when the sensor read completes. Joint feedback
  and IMU samples therefore share one time base.
- TIME_SYNC gives the offset between the host clock and the device clock. Send the host time, measure the
  round-trip time RTT, and compute `offset = t_host + RTT/2 - t_dev`. The accuracy is about ±RTT/2.
  `MC02Link.time_sync()` does this and returns the offset and the RTT in microseconds.
- In the SDK's motor driver, `Feedback.stamp` is the host receive time (`time.monotonic()`); the device time
  stamp of the same frame is kept in `Feedback.t_dev`.

## 6. IMU samples

The stream is off after power-on. Start it with IMU_CFG (`div` = 2 for 500 Hz, the rate the SDK uses) and
stop it with `div` = 0.

IMU (0x82), 18 bytes:

| Offset | Field | Description |
|---|---|---|
| 0 | `u32 t_us` | Device time stamp, same time base as CAN_RX |
| 4 | `i16 acc[3]` | Raw accelerometer x, y, z |
| 10 | `i16 gyr[3]` | Raw gyroscope x, y, z |
| 16 | `i16 temp_raw` | Raw temperature |

Values are raw sensor counts in the BMI088's own axes. The full-scale ranges are reported in PONG as the
sensor's range codes; read them from PONG instead of hard-coding them. The shipped firmware reports
`acc_range` = 0x01 (±6 g) and `gyr_range` = 0x01 (±1000 dps).

```
fs_g   = 1.5 * 2**(acc_range + 1)          # 0x00 ±3 g, 0x01 ±6 g, 0x02 ±12 g, 0x03 ±24 g
acc    = raw * fs_g * 9.80665 / 32768      # m/s^2

fs_dps = 2000 / 2**gyr_range               # 0x00 ±2000 dps, 0x01 ±1000 dps, 0x02 ±500 dps, ...
gyr    = raw * fs_dps / 32767 * pi / 180   # rad/s

temp   = temp_raw * 0.125 + 23             # degrees C
```

Mapping the sensor axes to the robot body, and the mounting correction, are done on the host from the robot
config (`imu.mount_axes`, `imu.mount_rpy_deg`). The board does no sensor fusion.

## 7. Watchdog and failsafe frames

The board can send CAN frames on its own when the host goes silent. The host registers the frames in
advance; the board only replays them and does not need to understand them.

```
DISARMED --(WD_CFG, timeout_ms > 0)--> ARMED --(host silent longer than timeout_ms, or USB suspend)--> FIRED
FIRED --(any valid packet from the host)--> ARMED
```

- **Feeding.** Every packet from the host that passes the CRC restarts the timer: CAN_TX, PING, any
  configuration message. A periodic CAN_TX stream is sufficient.
- **Firing.** The board sends WD_EVENT with `event` = 1 and transmits the frame of every registered slot.
  Slots with `repeat_ms` > 0 are sent again at that period until the host returns; slots with
  `repeat_ms` = 0 are sent once.
- **Release.** The first valid packet from the host stops the replay. The board sends WD_EVENT with
  `event` = 0 and is armed again.
- **No slots registered.** A firing watchdog only reports the event and sends no frames. Register the
  failsafe frames before relying on the watchdog.
- **USB suspend.** With `flags` bit 0 set in WD_CFG, a USB suspend also fires the watchdog. The board is
  powered from USB, so this only helps when the host suspends the bus while USB power stays on. An unplugged
  cable switches the board off; nothing is sent in that case (see [`mc02.md`](mc02.md), section 7).

FAILSAFE_SET (0x04), 18 bytes, registers a slot:

| Offset | Field | Description |
|---|---|---|
| 0 | `u8 slot` | Slot number, 0-15 |
| 1 | `u8 bus` | 0 = CAN1, 1 = CAN2 |
| 2 | `u8 flags` | bit 0 = EXT, bit 1 = RTR |
| 3 | `u32 id` | CAN identifier |
| 7 | `u8 dlc` | Number of data bytes |
| 8 | `u8 data[8]` | Data, padded with zeros |
| 16 | `u16 repeat_ms` | 0 = send once when the watchdog fires; otherwise repeat at this period until the host returns |

FAILSAFE_SET with a **1-byte payload** clears: `slot` = 0xFF clears all slots, `slot` < 16 clears that slot.

How the SDK uses it, as a pattern for your own host:

1. After enabling the motors, register one slot per motor: that motor's damping-only command (kp = 0,
   kd = 1.0, no position, velocity or torque target) with `repeat_ms` = 10. The SDK uses slot
   `bus * 8 + i` for the i-th motor of a bus, so 8 slots per bus.
2. Arm the watchdog with WD_CFG. The SDK uses 100 ms for a 50 Hz control loop. A timeout close to the control
   period fires on ordinary host scheduling jitter.
3. Run the control loop. The CAN_TX packets feed the watchdog.
4. On a normal exit, clear the slots and disarm the watchdog (`timeout_ms` = 0) before you stop sending.

After a FIRED event the joints have moved under the failsafe frames. Do not resume with the targets from
before the event: wait for joint feedback that arrived after it, restart from the measured positions, and
approach the target slowly. Detect the event by counting WD_EVENT packets, and also by checking your own loop
timing; if your loop was stalled for longer than the timeout, treat it as fired even if the event packet has
not been processed yet.

## 8. Restart into the USB bootloader

DFU_JUMP (0x08) with `magic` = 0x44465521 makes the firmware reset and start the MCU's built-in USB
bootloader (STM32 system bootloader, USB DFU). The board then appears as USB device `0483:df11`, and new
firmware can be written with `dfu-util`. The commands are in [`mc02.md`](mc02.md), section 8. In the
reference implementation: `MC02Link.dfu_jump()`, or `python tools/mc02_selftest.py --dfu`.

## 9. Throughput

Classic CAN at 1 Mbps; the USB Full-Speed CDC link carries roughly 0.8-1 MB/s.

| Load | Board to host | Host to board |
|---|---|---|
| 12 motors at 50 Hz | 600 frames/s, about 21 bytes each: about 13 KB/s | about 13 KB/s (the frames of one bus and one step in one packet) |
| plus the IMU at 500 Hz | about 11 KB/s more | - |
| 12 motors at 500 Hz | about 125 KB/s | about 125 KB/s |

Bandwidth is not the limit. The limit of a pass-through design is the roughly 1 ms granularity of USB CDC
transfers: expect a round-trip time with a p95 of 2-4 ms.

## 10. Compatibility

Within protocol version 1, message types may be added, but the payload layout of an existing type does not
change. Ignore types you do not know. Types 0x10-0x1F and 0x90-0x9F are reserved.

## 11. Reference implementation

`tools/mc02_link.py`, run from the repository root:

```python
from tools.mc02_link import MC02Link

link = MC02Link("/dev/mc02")                 # opens the port and starts the reader thread

pong = link.ping()                           # Pong: proto_ver, fw, imu_ok, acc_range_code, gyr_range_code, uid
st = link.status()                           # Status: uptime_ms, bus[0..1], wd_state, boot_flags, usb_* counters

link.send_can(0, can_id, data)               # one frame on CAN1 (extended identifier by default)
link.send_can_batch([(0, id_a, data_a), (0, id_b, data_b)])     # several frames in one USB packet
frame = link.recv_can_bus(0, timeout=0.01)   # CanFrame: t_us, bus, can_id, data, ext, rtr - or None

link.imu_stream(2)                           # 500 Hz
sample = link.recv_imu(timeout=0.1)          # ImuSample: t_us, acc_mss, gyr_rads, temp_c - or None
link.imu_stream(0)

link.set_failsafe(0, 0, can_id, data, repeat_ms=10)             # slot 0, CAN1
link.set_watchdog(100)                       # arm, 100 ms
# ... control loop ...
link.set_watchdog(0)                         # disarm
link.clear_failsafe()                        # clear all slots
link.close()
```

| Function | Message |
|---|---|
| `ping()` | PING / PONG |
| `status()` | STATUS_REQ / STATUS |
| `send_can()`, `send_can_batch()` | CAN_TX |
| `recv_can_bus(bus)` | CAN_RX, split into one queue per bus by the reader thread |
| `imu_stream(div)`, `recv_imu()` | IMU_CFG / IMU |
| `set_watchdog(timeout_ms)` | WD_CFG |
| `set_failsafe(...)`, `clear_failsafe(slot)` | FAILSAFE_SET |
| `wd_events`, `fired_count`, `released_count` | WD_EVENT |
| `time_sync()` | TIME_SYNC / TIME_SYNC_RESP |
| `dfu_jump()` | DFU_JUMP |

`open_mc02_bus("mc02:0")` in the same file wraps one bus in an object with `send`, `recv` and `shutdown`, in
the style of a python-can bus; the SDK's motor driver uses it. If the serial port fails, the link is marked
dead and every later call raises `RuntimeError`; it does not reconnect.
