# wheeled_base

A wheeled biped: two legs with four joints each and a driven wheel at the end of each leg. It balances on
its two wheels and is controlled by a reinforcement-learning policy exported to ONNX.

**This release supports simulation only.** The joint table in [`wheeled_base.yaml`](wheeled_base.yaml) has
no CAN ids, bus assignment or calibration yet, the motor parameters are per-model values rather than
measurements on this robot, and the SDK contains no real-robot deployment program for it. What you can do
today: train on the model, export a policy that follows the interface below, and validate it in MuJoCo.

The authoritative configuration is `wheeled_base.yaml`; the numbers below come from it.

## 1. Model

| File | Use |
|---|---|
| `urdf/wheeled_base.urdf` | URDF |
| `mjcf/wheeled_base.xml` | MuJoCo model |
| `usd/wheeled_base.usd` | Isaac Lab asset |
| `meshes/` | STL meshes, simplified for display and collision. Mass, centre of mass and inertia come from the URDF / MJCF, not from the meshes |

Wheel radius 0.085 m; distance between wheel centres 0.28 m.

In Isaac the wheel collider is a sphere (single contact point), which by itself offers no resistance to
spinning in place. The supplied USD adds torsional friction (8 mm contact patch radius) to approximate a real
tyre. If you build your own asset, add it — otherwise turning in place is easier in simulation than on a
real tyre.

## 2. Joints and actuators

Ten joints, in this order everywhere:

```
l_hip_pitch, l_hip_roll, l_hip_yaw, l_knee, l_wheel,
r_hip_pitch, r_hip_roll, r_hip_yaw, r_knee, r_wheel            (each with the suffix _joint)
```

| Joints | Motor | Control | kp | kd | Torque limit N·m | Speed limit rad/s |
|---|---|---|---|---|---|---|
| hip_pitch, hip_roll, knee | RS06 | position | 30 | 1.5 | 12 | 10 |
| hip_yaw | RS00 | position | 12 | 0.6 | 11 | 33 |
| wheel | RS02 | velocity | 0 | 1.0 | 17 | 20 |

Motor law: `tau = clip(kp·(q_des − q) + kd·(v_des − v), ±torque limit)`. Legs: `v_des = 0` and `q_des` comes
from the policy. Wheels: `kp = 0` and `v_des` comes from the policy.

Armature, friction and the randomisation ranges used for the reference policy are in the config.

Nominal pose (leg targets when the action is zero): hip pitch −0.595, knee 1.0, hip roll and yaw 0, the same
on both sides (the two legs' joint axes are parallel, not mirrored). In this pose the centre of mass is
directly above the wheel axle, so a level body is the balance point. Base height 0.383 m.

## 3. Policy interface

Control rate 50 Hz. The observation is the **last 10 frames** of 39 values each — **390 values**, oldest
first. The exact layout is `build_obs` in `tools/wheeled_base_io.py`.

| Term | Size | Scale | Meaning |
|---|---|---|---|
| `base_ang_vel` | 3 | 0.25 | Angular velocity in the body frame |
| `projected_gravity` | 3 | 1.0 | Unit gravity vector in the body frame |
| `commands` | 3 | [2.0, 2.0, 0.25] | vx, vy, wz; vy is always 0 (the wheels cannot move sideways) |
| `leg_pos_rel` | 8 | 1.0 | Leg joint angles − nominal (wheel angles are unbounded and are not observed) |
| `joint_vel` | 10 | 0.05 | Velocities of all ten joints, wheels included |
| `last_action` | 10 | 1.0 | Previous action, after clipping to ±1 |
| `hold_error` | 2 | [4.0, 2.0] | Position-hold error: once the command has been zero for 1 s, how far the robot has moved (m) and turned (rad) from where it stopped |

Action, 10 values, clipped to ±1:

- legs: `q_des = nominal + 0.25 × a`, then clamped to the joint limits;
- wheels: `v_des = 20 × a` (rad/s).

`hold_error` is computed from wheel-encoder angle increments and the gyro only, so it is available on real
hardware. It is what lets the robot stay on one spot under a zero command and hold its place on a slope; the
definition and the reasoning are in the comments of the config. Keep it in your own training.

## 4. Simulation

```bash
python tools/wheeled_base_sim2sim.py --policy wheeled_base/policies/drive.onnx --cmd 0.5 0      # vx 0.5 m/s
python tools/wheeled_base_sim2sim.py --policy wheeled_base/policies/drive.onnx --cmd 0 1.0      # turn in place, 1 rad/s
python tools/wheeled_base_sim2sim.py --policy <yours>.onnx --allow-unstamped --cmd 0.5 0
python tools/wheeled_base_sim2sim.py ... --view                                                 # window (mjpython on macOS)
python tools/wheeled_base_sim2sim.py ... --slope 5 --delay 2 --obs-noise --push 5 3 90          # slope, delay, noise, push
```

The acceptance matrix runs standing, driving, turning, disturbances, slopes and pushes, and reports pass or
fail per case:

```bash
python tools/wheeled_base_accept.py --policy <yours>.onnx --allow-unstamped
python tools/wheeled_base_accept.py --policy a.onnx b.onnx c.onnx        # compare checkpoints
```

**Reference policy** — `policies/drive.onnx`: balances, holds its position (including on slopes), drives
forward / back and turns. It passes all 160 cases of the acceptance matrix with the default seeds. It has not
been run on hardware; use it as a baseline in simulation.

**Stamps.** A wheeled_base ONNX carries a stamp — a snapshot of the parts of the config that define the
meaning of its inputs and outputs — which the tools verify when loading. For your own file either pass
`--allow-unstamped` (the tool infers the history length from the input size), or export and stamp directly
from an rsl_rl checkpoint (requires PyTorch; the actor must be an ELU MLP trained without observation
normalisation — a checkpoint with a normaliser is refused, export that one yourself with the normaliser
included and use `--allow-unstamped`):

```bash
python tools/wheeled_base_export.py <model_N.pt> wheeled_base/policies/<name>.onnx
```
