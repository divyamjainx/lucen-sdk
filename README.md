# Lucen SDK

Models, parameters, simulation tools and real-robot software for Lucen robots.

| Robot | Folder | Description | Controlled by | Real-robot support |
|---|---|---|---|---|
| **biped_Laika** | [`biped_Laika/`](biped_Laika/) | 12-DoF biped | RL policy (ONNX) | Yes — reference stand and walk policies included |
| **wheeled_base** | [`wheeled_base/`](wheeled_base/) | Wheeled biped: 8 leg joints + 2 wheels | RL policy (ONNX) | Simulation only in this release |
| **humanoid_arm** | [`humanoid_arm/`](humanoid_arm/) | Two 7-DoF arms with grippers | Teleoperation / joint position targets | Yes |
| **Arm** | [`Arm/`](Arm/) | 6+1-DoF arm, single or leader–follower pair | Teleoperation / joint position targets | Yes |

All four use the same RobStride actuators, the same motor driver, and the same MC02 interface board
(USB to two CAN buses, with an on-board IMU).

## Where to start

| You want to | Read |
|---|---|
| Connect your computer to a robot | [`docs/connect.md`](docs/connect.md) |
| Bring your own walking policy to biped_Laika | [`biped_Laika/README.md`](biped_Laika/README.md) |
| Train and validate a wheeled_base policy | [`wheeled_base/README.md`](wheeled_base/README.md) |
| Operate humanoid_arm or Arm | [`docs/arm_guide.md`](docs/arm_guide.md), then [`humanoid_arm/README.md`](humanoid_arm/README.md) or [`Arm/README.md`](Arm/README.md) |
| Know what protects you and the robot | [`docs/safety.md`](docs/safety.md) — read before powering anything |
| Let an AI coding agent work with the SDK for you | [`AGENTS.md`](AGENTS.md) — agents read it automatically; it tells them what they may run on their own and what needs your confirmation |
| Wire, check or re-flash the MC02 board | [`docs/mc02.md`](docs/mc02.md), [`docs/mc02_protocol.md`](docs/mc02_protocol.md) |

## Supplied with your robot

A few things are specific to each robot and are not in this repository. Have them before you power it on:
the robot computer's address and login, the power-up sequence and the location of the 48 V emergency stop,
the hoist attachment (biped_Laika), and the configuration file calibrated for your unit.

## Install

Python 3.10 or newer. Linux (including Jetson) and macOS are supported; use Linux on the machine that drives
hardware.

```bash
git clone https://github.com/ICE-CUBA/lucen-sdk.git && cd lucen-sdk
bash setup.sh                 # virtual env (~/lucen_env), dependencies, udev rule for /dev/mc02 (needs sudo)
bash setup.sh --no-udev       # simulation only, no hardware access (use this on macOS)
source ~/lucen_env/bin/activate
```

Run every command from the repository root (`python tools/<name>.py ...`).

## Try it without hardware

```bash
# biped_Laika: run the reference policies in MuJoCo (add --view for a window; on macOS use mjpython)
python tools/sim2sim.py --policy biped_Laika/policies/stand.onnx --duration 5
python tools/sim2sim.py --policy biped_Laika/policies/walk.onnx --power-scale 0.8 --cmd 0.2 0 0

# wheeled_base
python tools/wheeled_base_sim2sim.py --policy wheeled_base/policies/drive.onnx --cmd 0.5 0

# humanoid_arm: simulated motors behind the real bridge code, driven from a web page
python tools/fake_bridge.py --config humanoid_arm/humanoid_arm.yaml --grav-scale 1.0     # terminal 1
python tools/arm_web_jog.py --config humanoid_arm/humanoid_arm.yaml --gripper            # terminal 2 -> http://127.0.0.1:8780
```

## Repository layout

```
biped_Laika/    wheeled_base/    humanoid_arm/    Arm/
    <robot>.yaml      the robot's configuration: joint table, motor parameters, gains, policy interface
    urdf, mjcf, meshes, usd      models (URDF, MuJoCo, Isaac Lab USD where applicable)
    policies/         reference policies (biped_Laika, wheeled_base)
    README.md         the guide for that robot
bridge.py           bridge process: owns the MC02 board, talks to the motors at 50 Hz, serves a TCP port
robstride_mit.py    RobStride motor driver
tools/              command-line tools (see below)
docs/               connection, safety, arm operation, MC02 board
LICENSE, NOTICE     Apache License 2.0
```

Tools by purpose:

| Purpose | Tools |
|---|---|
| biped_Laika — simulation | `sim2sim.py` (one policy) · `sim_fsm.py` (stand + walk together) · `policy_gain_probe.py` |
| biped_Laika — policy interface | `policy_io.py` (the observation / action code shared by simulation and robot) · `onnx_manifest.py` · `contract_check.py` · `export_isaac.py` · `export_ckpt.py` |
| biped_Laika — real robot | `deploy_policy.py` · `imu_reader.py` · `calib_stance.py` |
| wheeled_base | `wheeled_base_io.py` · `wheeled_base_sim2sim.py` · `wheeled_base_accept.py` · `wheeled_base_export.py` |
| Arms — front ends | `arm_web_jog.py` · `arm_jog.py` · `arm_drag_teleop.py` · `arm_lead_follow.py` · `arm_vr_teleop.py` · `arm_cam_teleop.py` |
| Arms — calibration, model, simulation | `calib_arm.py` · `build_arm_model.py` · `fake_bridge.py` |
| Arms — recording | `--log` on the front ends, `log=` in your own client · `arm_log_report.py` (targets vs. measurements, hand poses) |
| Hardware | `mc02_selftest.py` · `set_motor_id.py` · `set_torque.py` · `set_can_timeout.py` · `check_ports.py` |

Every tool has `--help`, and a usage section at the top of its source file.

## Ground rules

- **The robot's YAML file is the single source of truth.** Simulation, training export and the real-robot
  tools all read it. Change parameters there, nowhere else.
- **Calibration is per unit.** Each joint's `sign` and `offset` (and the biped's IMU mounting angles) belong
  to one physical robot. Verify them on your robot before the first enable, and recalibrate after servicing
  a joint. Each robot guide has the procedure.
- **Motor firmware limits must match the config.** The torque limit stored in each motor must equal
  `tau_limit` in the YAML — the same value simulation clips at. Check with `tools/set_torque.py --check`; the
  biped deployment program also reads the limits back at start-up and refuses to run on a mismatch.
- **A biped policy must be stamped before it runs.** `tools/onnx_manifest.py --stamp` embeds the policy's
  interface in the ONNX file; simulation and deployment verify it at load time and refuse a mismatch.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Lucen Robotics.

The license covers the software, models, configuration files and reference policies in this repository. It
does not grant rights to the Lucen names and logos.
