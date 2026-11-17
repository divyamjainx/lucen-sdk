# Instructions for AI coding agents

You are helping a person who owns or works with a Lucen robot. This file tells you how to use this
repository on their behalf. Read it fully before running anything.

## What this repository is

| Robot | Folder | Controlled by | Real robot |
|---|---|---|---|
| biped_Laika | `biped_Laika/` | RL policy (ONNX) | supported |
| wheeled_base | `wheeled_base/` | RL policy (ONNX) | simulation only |
| humanoid_arm | `humanoid_arm/` | teleoperation / joint targets through the bridge | supported |
| Arm | `Arm/` | teleoperation / joint targets through the bridge | supported |

Each robot folder holds its configuration (`<robot>.yaml`), its models and a `README.md` that is the guide for
that robot. Cross-cutting guides are in `docs/`: `connect.md`, `safety.md`, `arm_guide.md`, `mc02.md`.
Tools are in `tools/`; every tool has `--help` and a usage section at the top of its source.

Run everything from the repository root, with the SDK environment active
(`bash setup.sh --no-udev` creates `~/lucen_env`; `source ~/lucen_env/bin/activate`).

## Rule 1: you may simulate freely, you may not move hardware on your own

These motors can injure a person and damage the robot. You cannot see the robot, the hoist or the people
around it. Therefore:

**Never run a command that can enable motors or write to motors unless the person has, in this conversation
and for this specific command, confirmed the physical state of the robot** (biped: hoisted with the feet off
the ground, or held on the ground by a second person; arms: nobody within reach) **and that the 48 V
emergency stop is within their reach.** Prefer to give the person the exact command and let them run it.
A confirmation given earlier for another command does not carry over.

| Class | Commands |
|---|---|
| **Simulation only — run freely** | `tools/sim2sim.py`, `tools/sim_fsm.py`, `tools/policy_gain_probe.py`, `tools/wheeled_base_sim2sim.py`, `tools/wheeled_base_accept.py`, `tools/fake_bridge.py` and any arm front end pointed at it (`127.0.0.1`), `tools/onnx_manifest.py`, `tools/contract_check.py`, `tools/export_isaac.py`, `tools/export_ckpt.py`, `tools/build_arm_model.py`, `tools/arm_log_report.py` (reads a recorded file, connects to nothing) |
| **Hardware, read-only — ask first, low risk** | `tools/mc02_selftest.py` (without `--wd-test` / `--dfu`), `tools/check_ports.py`, `tools/set_torque.py --check`, `tools/set_can_timeout.py --check`, `tools/set_motor_id.py --scan` / `--check`, `tools/imu_reader.py --view` / `--verify-axes`, `tools/calib_arm.py --live`, `tools/calib_stance.py` without `--write`, `bridge.py` with a client in watch mode, `mirror_viewer.py --mode watch` |
| **Enables motors — only with the confirmation above** | `tools/deploy_policy.py` (every mode), any client that connects to a real `bridge.py` in send mode (`tools/arm_web_jog.py`, `arm_jog.py`, `arm_drag_teleop.py`, `arm_lead_follow.py`, `arm_vr_teleop.py`, `arm_cam_teleop.py`, `mirror_viewer.py --mode send`, your own `BridgeClient(..., mode="send")`), `robstride_mit.py` self-tests, `tools/mc02_selftest.py --wd-test` |
| **Writes to motors, the board or the calibration — only when the person asks for exactly that** | `tools/set_torque.py --write`, `tools/set_can_timeout.py --write`, `tools/set_motor_id.py --assign` / `--to`, `tools/calib_stance.py --write`, `tools/calib_arm.py --zero-here` / `--signs` / `--flip` / `--write` / `--mark-calibrated`, `tools/mc02_selftest.py --dfu` |

Before running an arm front end, check where it points: with `tools/fake_bridge.py` on `127.0.0.1` it is a
simulation; with `--host` / `--bridge-host` set to a robot computer it moves a real arm.

Do not use these to get past a refusal on the real robot: `--allow-unstamped`, `--allow-plant-drift`,
`--skip-torque-check`, `--pad-no-deadman`, `--no-range-check`, `--no-collide`. A refusal means a check
failed; find out why and tell the person.

Never claim that something works on the robot. You can only report what a simulation showed or what the
person told you they observed.

## Rule 2: the robot's YAML file is the single source of truth

Simulation, training export and the real-robot tools all read `<robot>/<robot>.yaml`.

- Do not edit `sign`, `offset`, `imu.mount_rpy_deg` or `calibrated` by hand. They are per-unit calibration;
  only the calibration tools change them, on the person's request. Never set `calibrated: true` to get past
  the bridge's refusal.
- Do not change `tau_limit`, `vel_limit`, `can_timeout` or the gains to make something work. They must match
  what is stored in the motors and what the policy was trained with.
- Adding a policy profile to `policy_profiles` in `biped_Laika/biped_Laika.yaml` is fine (see below).
- Do not edit `tools/policy_io.py` or `tools/wheeled_base_io.py` to adapt an interface. They are the
  contract shared by simulation and the robot; a policy with a different interface must be retrained.

## Task: train a walking policy for biped_Laika

Read `biped_Laika/README.md` sections 1 to 3, `biped_Laika/isaac_values.py` and the docstrings of
`tools/policy_io.py`. Then build the training environment so that all of the following hold; these are the
mistakes that make a policy fail on this robot:

1. **Joint order** is left leg then right leg (`JOINT_ORDER` in `isaac_values.py`), in observations and in
   actions. Isaac Lab's default order interleaves the legs: select joints explicitly and preserve the order.
2. **Actuator parameters** come from `isaac_values.ACTUATORS` (kp, kd, torque limit, speed limit, armature,
   friction), not from the values embedded in the URDF / USD, which are placeholders.
3. **Observation**: exactly the terms, order and scales of the chosen profile (`walk` is recommended: the 45
   base values plus a 2-value gait phase). No base linear velocity, contact or height in the actor's
   observation. Reproduce the gait clock and, for history profiles, the frame layout as `build_obs` defines
   them; Isaac Lab's built-in observation history has a different layout.
4. **Action**: `target = nominal pose + action_scale * clip(action, -1, 1)`, clamped to the joint limits. Use
   the nominal pose and scale of the profile (`POLICY_PROFILES["walk"]`), not the top-level standing defaults.
5. **Timing**: 50 Hz control; randomise an action delay of 0 to 0.06 s; note the target rate limiter
   described in the README.
6. **Export**: one input `[1, obs_dim]`, first 12 outputs = action, observation normalisation inside the
   ONNX. `tools/export_ckpt.py` only handles an un-normalised ELU MLP.

Use the reference simulation settings in the README (step size, noise, terminations) unless the person wants
otherwise. Reward design is the person's choice; do not present your reward choices as vendor recommendations.

If the interface must differ in nominal pose, action scale or clock period, add a profile to the config, then
run `python tools/export_isaac.py && python tools/contract_check.py --onehot` and expect `ALL PASS`.

## Task: validate a biped policy before it goes on the robot

Follow `biped_Laika/README.md` section 4, in this order, and report the numbers next to the reference
numbers in that section:

```bash
python tools/onnx_manifest.py --stamp POLICY.onnx --profile walk
python tools/sim2sim.py --policy POLICY.onnx --cmd 0.2 0 0 --delay 2 --jitter 0.05     # also 0 0 0, 0 0 0.3, and the sweeps
python tools/sim_fsm.py --policy POLICY.onnx --stand-policy biped_Laika/policies/stand.onnx \
    --script "2:w3,8:w2" --duration 13 --delay 2 --jitter 0.05
python tools/policy_gain_probe.py --policy POLICY.onnx --stand-policy biped_Laika/policies/stand.onnx \
    --delay 1 --jitter 0.05 --script 3:w6 --duration 12 --quiet
```

"Survived" alone is not a pass: a policy that outputs zeros survives most runs. Report speed tracking, tilt,
torque margin, touchdown force, the stand-to-walk hand-over, and the gain-probe result. If any of them is
clearly worse than the reference, say so and recommend more training rather than a hoist test.

## Task: put a biped policy on the robot

Your job is to prepare, not to drive. Give the person the sequence from `biped_Laika/README.md` section 5
(checks, calibration check, hoist steps 1 to 5, then ground), one step at a time, with the exact command and
what they should see. Wait for their report before the next step. If a step fails or looks wrong, stop and
diagnose; do not skip ahead, and do not suggest going to the ground before every hoist step has passed.
Remind them that every way of ending the program leaves the robot limp.

Robot-specific facts (address and login of the robot computer, where the emergency stop is, how the hoist
attaches, whether the config matches their unit) are not in this repository. Ask the person; never guess.

## Task: control humanoid_arm or Arm

Read `docs/arm_guide.md` and the robot's `README.md`. Develop against the simulated bridge first:

```bash
python tools/fake_bridge.py --config Arm/Arm.yaml --grav-scale 1.0        # same code path, port and protocol as the real bridge
```

For your own program use `BridgeClient` from `tools/arm_jog.py` (example in `docs/connect.md`, section 3):
start from the pose the bridge returns (`q0`), move targets gradually, send at 50 Hz, pass the config's
`range` / `soft_range` as limits, and remember that the bridge disables the motors if you stop sending for
1.5 s. The bridge does not check for self-collision. Only after the program behaves in simulation, and with
the confirmation of Rule 1, may it be pointed at a real bridge, first with slow motions.

Record what the program does: pass `log="run.csv", log_meta={"config": "<yaml>"}` to `BridgeClient` (or `--log`
to a front end), then run `python tools/arm_log_report.py run.csv`. It reports how far the measured arm was
from its targets and, with `--poses`, the target and reached pose of the hand for every row
(`docs/arm_guide.md`, section 8). Report these numbers rather than "it moved". The reached pose comes from the
encoders: it cannot reveal a calibration error.

## Task: wheeled_base

Simulation only: `wheeled_base/README.md` describes the interface and the validation tools. There is no
real-robot path in this release; do not improvise one with the biped tools.

## When something is unclear

Read the tool's docstring and `--help` before guessing an option. Do not invent parameter values, file
paths or procedures; if the repository does not say, tell the person what is missing and ask.
