# Connecting to a robot

Every Lucen robot has the same hardware chain:

```
your computer ──network──> robot computer (Jetson) ──USB──> MC02 board ──CAN1 / CAN2──> motors
                                                              └─ on-board IMU (used by biped_Laika)
```

- The **MC02 board** is a USB device with two CAN buses and an IMU. It is the only path between the SDK and
  the motors, and appears as one serial port (`/dev/mc02`). See [`mc02.md`](mc02.md).
- The **robot computer** is whichever machine the MC02 is plugged into. On biped_Laika it is the Jetson inside
  the body. For the arms it is a Jetson next to the robot — or simply your own computer.
- The motors' 48 V supply does not pass through the MC02. The emergency stop cuts that supply.

## 1. Choose how to connect

| Method | Control loop runs on | Good for | Not for |
|---|---|---|---|
| **A. Log in to the robot computer** | the robot computer | biped_Laika and wheeled_base policies (required: A or C); arms too | — |
| **B. Bridge** | the bridge runs on the robot computer; your computer sends joint targets | arm teleoperation, your own arm control program | walking policies — the bridge reports joint angles only, no IMU and no joint velocities |
| **C. Plug the MC02 into your computer** | your computer | arms; bench work with biped_Laika on the hoist | walking on the ground (trailing USB cable) |

Recommended per robot:

| Robot | Method | Note |
|---|---|---|
| biped_Laika | A | The 50 Hz policy loop must run on the machine the MC02 is plugged into |
| wheeled_base | A | Simulation only in this release |
| humanoid_arm | B | Bridge on the robot computer; front ends (web sliders, mouse drag, VR, camera) on yours |
| Arm | B or A | The leader–follower program is simplest on the same machine as the bridge |

## 2. Method A — log in to the robot computer

The robot computer's address, user name and password are supplied with your robot.

**Ways to reach it**

1. **Same local network.** The robot computer is on Wi-Fi or Ethernet and so is your computer:
   `ssh <user>@<robot-ip>`.
2. **USB cable to the Jetson.** Connect your computer to the Jetson's USB-C device port. The Jetson creates a
   network on that cable at the fixed address `192.168.55.1`: `ssh <user>@192.168.55.1`. No configuration is
   needed; use it when there is no network on site. One robot at a time.
3. **From another network.** Use your own VPN to the robot's network. The SDK does not set one up.

**First login** (once):

```bash
git clone https://github.com/ICE-CUBA/lucen-sdk.git ~/lucen-sdk
cd ~/lucen-sdk && bash setup.sh        # virtual env ~/lucen_env, dependencies, udev rule for /dev/mc02
# log out and in again so the serial-port group membership takes effect
```

**Every session:**

```bash
source ~/lucen_env/bin/activate && cd ~/lucen-sdk
python tools/mc02_selftest.py          # board answers; the expected motor ids are on both buses
```

**Copy your policy over:** `scp my_policy.onnx <user>@<robot-ip>:~/lucen-sdk/biped_Laika/policies/`

**Run deployment commands inside `tmux`.** In a plain SSH session a network drop kills the program; the motors
go to damping and the robot goes limp. Inside `tmux` the program keeps running: a keyboard velocity command
expires after one second and the robot returns to standing; log in again and `tmux attach`.
Keyboard teleoperation (`--teleop`) uses this SSH terminal. On an unreliable network use the gamepad instead
(`--pad`): the gamepad connects to the robot computer directly, not through your computer.

Continue with [`biped_Laika/README.md`](../biped_Laika/README.md), section 5.

## 3. Method B — the bridge

The bridge runs on the robot computer. It owns the MC02, keeps the 50 Hz exchange with the motors alive, and
accepts joint targets from one client on a TCP port.

```bash
# robot computer
python bridge.py --config humanoid_arm/humanoid_arm.yaml

# your computer (example: web sliders)
python tools/arm_web_jog.py --config humanoid_arm/humanoid_arm.yaml --gripper --bridge-host <robot-ip>
```

| Config | Robot | Bridge port |
|---|---|---|
| `biped_Laika/biped_Laika.yaml` (default) | biped_Laika — for watching joint angles and calibration, not for policies | 9877 |
| `humanoid_arm/humanoid_arm.yaml` | humanoid_arm | 9879 |
| `Arm/Arm.yaml` | Arm, one arm on the board | 9880 |
| `Arm/Arm_pair.yaml` | Arm, leader + follower | 9881 |

What to know:

- **The bridge port has no authentication.** Anyone who can reach it can move the motors. Use it only on a
  trusted network, or tunnel it through SSH: on your computer run
  `ssh -L 9879:127.0.0.1:9879 <user>@<robot-ip>` and point the front end at `127.0.0.1`.
- **One client at a time.** While the web sliders are connected, another front end cannot connect.
- **Heartbeat.** If the client sends no target for 1.5 s, the bridge switches all motors to damping and then
  disables them. An arm sinks slowly; it does not hold its position.
- **A config with `calibrated: false` is watch-only.** The bridge refuses to enable the motors.
  See [`arm_guide.md`](arm_guide.md), section 5.
- Use the **same config file** (same SDK version) on both machines: joint names, ranges and ports come from it.

### Writing your own client

The protocol is one JSON object per line over TCP.

```
client -> bridge:  {"hello": {"mode": "watch" | "send", "joints": [names...], "limits": {name: [lo, hi]}}}
                   {"target": {name: radians, ...}}      # send mode: target and heartbeat; 50 Hz recommended
bridge -> client:  {"matched": [...], "unknown": [...]}   # which joints the bridge recognised
                   {"q0": {name: radians}}                # send mode: pose right after enabling
                   {"q": {name: radians}}                 # live joint angles
                   {"error": "..."}                       # why the request was refused
```

A ready-made Python client, `BridgeClient`, lives in `tools/arm_jog.py`:

```python
import sys, time, yaml
sys.path.insert(0, "tools")
from arm_jog import BridgeClient

cfg = yaml.safe_load(open("Arm/Arm.yaml"))
names = [j["name"] for j in cfg["joints"]]
limits = {j["name"]: j["range"] for j in cfg["joints"]}
c = BridgeClient("127.0.0.1", cfg["bridge"]["port"], names, limits, mode="send")   # connecting enables the motors
target = dict(c.q0)                    # start from the current pose, never from a distant target
for _ in range(250):                   # 5 seconds
    target["joint1"] += 0.001          # 0.001 rad per step = 0.05 rad/s
    c.send_target(target)
    time.sleep(0.02)
c.close()                              # the bridge goes to damping, then disables, within 1.5 s
```

Try it against simulation first: `python tools/fake_bridge.py --config Arm/Arm.yaml --grav-scale 1.0` runs the
same bridge code in front of simulated motors, on the same port with the same protocol.

To record what your program sent and what the arm did, add `log="my_run.csv"` (and, so that the report finds
the model, `log_meta={"config": "Arm/Arm.yaml"}`) to the `BridgeClient(...)` call. Every `send_target` then
writes one row with the targets and the latest measured angles; `python tools/arm_log_report.py my_run.csv`
turns the file into tracking errors and hand poses. See [`arm_guide.md`](arm_guide.md), section 8.

The bridge does two things for you: it clamps each target to the `limits` you sent in `hello` (send the
`range` / `soft_range` from the config, not something wider), and it limits how fast the commanded position
moves to 1 rad/s per joint. It does **not** check for self-collision and cannot tell whether your target makes
sense. The SDK front ends add their own speed limit from the config's `safety_profiles`.

## 4. Method C — plug the MC02 into your computer

The MC02 is powered from USB; the machine it is plugged into is the robot computer. The SDK runs on Linux and
macOS.

**Linux:** `bash setup.sh` installs the udev rule; the board appears as `/dev/mc02`.

**macOS:** there is no udev. Find the port with `ls /dev/cu.usbmodem*` and set `mc02.port` in the robot's
YAML to that name. For the self-test alone you can pass it on the command line:
`python tools/mc02_selftest.py --port /dev/cu.usbmodemXXXX`.

```bash
python tools/mc02_selftest.py --config humanoid_arm/humanoid_arm.yaml
python bridge.py --config humanoid_arm/humanoid_arm.yaml     # bridge and front ends on the same machine (host 127.0.0.1)
```

- **Only one process can open an MC02 at a time.** While the bridge or a deployment is running, the
  self-test, calibration and motor-parameter tools cannot connect. Stop the bridge first.
- **Two MC02 boards on one computer:** `/dev/mc02` becomes ambiguous and the tools refuse to open it; they
  list each board's `/dev/serial/by-id/...` path. Put the right path in each robot's `mc02.port`.
- **Unplugging USB powers the board off.** The board's watchdog cannot help in that case; protection comes
  from the motors' own timeout (they release after 0.5 s without commands). Use a short, good cable and
  secure it.

## 5. First checks after connecting

Run these with the motors disabled, whichever method you use:

```bash
python tools/mc02_selftest.py --config <robot yaml>           # board answers, boot flags clear, motor ids on both buses match the config
python tools/set_torque.py --config <robot yaml> --check       # torque limit stored in each motor == tau_limit in the config
python tools/set_can_timeout.py --config <robot yaml> --check  # each motor's communication timeout == can_timeout (0.5 s)
```

For biped_Laika omit `--config` (its config is the default) and add
`python tools/imu_reader.py --verify-axes`, which asks you to tilt the torso and checks the IMU directions.

If the second or third check reports a difference, write the config values to the motors with
`--write --save` — after confirming the config is the one for this robot.

## 6. Troubleshooting

| Symptom | Likely cause |
|---|---|
| No `/dev/mc02` | Not plugged in; udev rule not installed (`bash setup.sh`); installed but the board was not re-plugged. `lsusb` should list "Lucen MC02 v1" |
| Permission denied on the port | Added to the `dialout` group but not logged in again |
| Self-test gets no answer | Another process holds the board (a bridge or deployment still running) |
| Self-test: motor ids missing | That motor is unpowered or its CAN cable is open |
| Self-test: the two buses' ids are swapped | CAN1 / CAN2 harnesses are exchanged — swap the connectors; do not edit the config |
| A whole bus is silent | CAN L / H reversed, missing ground wire between the board and motor power ground, or missing 120 Ω termination. See [`mc02.md`](mc02.md) |
| Bridge answers `error` and does not enable | The config says `calibrated: false`, or the requested joint names are unknown (different config versions on the two machines) |
| Front end cannot reach the bridge | Another client is connected; wrong host or port; firewall |
| Deployment exits at start-up reporting a torque-limit or timeout mismatch | Checks 2 and 3 in section 5 |
