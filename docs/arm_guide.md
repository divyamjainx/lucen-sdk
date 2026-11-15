# Operating the arms

This guide applies to both arm robots. They share the bridge (`bridge.py`), the front ends and the safety
chain; only `--config` differs.

| | humanoid_arm | Arm |
|---|---|---|
| Config | `humanoid_arm/humanoid_arm.yaml` | `Arm/Arm_pair.yaml` (leader + follower) · `Arm/Arm.yaml` (one arm on the board) |
| Actuators | 2 × (7 joints + gripper) = 16 | 6 joints + gripper per arm |
| Bridge port | 9879 | 9881 (pair) · 9880 (single) |
| Details | [`humanoid_arm/README.md`](../humanoid_arm/README.md) | [`Arm/README.md`](../Arm/README.md) |

The arms are driven by teleoperation or by joint position targets from your own program. To record
demonstrations for imitation learning, log joint angles and targets through the bridge protocol
([`connect.md`](connect.md), section 3).

Below, `<yaml>` stands for the config of your robot.

## 1. How it fits together

```
front end (your computer) ──TCP──> bridge.py --config <yaml> (robot computer) ──USB──> MC02 ──CAN1 / CAN2──> motors
```

At 50 Hz the bridge sends each motor a position target plus a gravity feed-forward torque. A front end only
supplies joint targets. Front end and bridge may run on the same machine (the default host is `127.0.0.1`).

## 2. Front ends

All connect to the same bridge. **The bridge accepts one client at a time.**

| Front end | Command | Use | Robots |
|---|---|---|---|
| Web sliders | `python tools/arm_web_jog.py --config <yaml> --gripper`, then open `http://127.0.0.1:8780` | Precise per-joint positioning; Ready / Home / STOP. **Use this for the first enable** | both |
| Keyboard | `python tools/arm_jog.py --config <yaml> --gripper` | Select a joint, step with `w` / `s`; works over SSH | both |
| Mouse drag | `mjpython tools/arm_drag_teleop.py --config <yaml>` | In the MuJoCo window, double-click the hand, Ctrl+right-drag to move, Ctrl+left-drag to rotate | both |
| Lead by hand | `python tools/arm_lead_follow.py --config <yaml>` | Move one arm by hand, the other follows (humanoid_arm: mirrored; Arm pair: copied) | both |
| VR controllers *(experimental)* | `python tools/arm_vr_teleop.py --config humanoid_arm/humanoid_arm.yaml` | Open the page it serves in a Meta Quest browser; hold the grip button to take over an arm | humanoid_arm |
| Camera *(experimental)* | `python tools/arm_cam_teleop.py` | A webcam watches you; the arms copy your arm pose, a fist closes the gripper. Needs `requirements-extra.txt` | humanoid_arm |

To reach a bridge on another machine use `--bridge-host <ip>` for the web sliders and `--host <ip>` for the
others. On Linux run `python` instead of `mjpython`.

Every front end takes `--safety safe | slow | normal` (joint speed limit 0.5 / 1.0 / 2.0 rad/s; default
`slow`). Use `safe` the first time you try a front end.

Keys, behaviour and limits of each front end are described at the top of its source file and in `--help`.
Two things to know in advance:

- **Lead by hand moves the arms by itself about 5 seconds after start**: it visits two poses to measure the
  gravity feed-forward error. Clear the area. Before quitting, press `h` to return home — once the program
  exits the bridge disables the motors within 1.5 s and an unsupported arm drops.
- **Web sliders: keep the browser tab in the foreground.** The page's polling is the heartbeat; if the tab is
  closed, backgrounded or the computer sleeps, the gateway stops sending within half a second and the bridge
  disables the motors about 1.5 s later — an arm that is held up sinks.
- **VR and camera teleoperation are experimental.** A single camera has no true depth, so camera
  teleoperation reproduces poses, not precise grasps. Try both against the simulated bridge first.

## 3. Simulation first

`fake_bridge.py` runs the real bridge code in front of simulated motors in MuJoCo. Port, protocol and front
ends behave exactly as on the robot.

```bash
# terminal 1
python tools/fake_bridge.py --config humanoid_arm/humanoid_arm.yaml --grav-scale 1.0     # add --view for a window (mjpython on macOS)
# terminal 2 (any front end)
python tools/arm_web_jog.py --config humanoid_arm/humanoid_arm.yaml --gripper            # http://127.0.0.1:8780
```

In simulation a gravity scale of 1.0 is correct because the simulated arm *is* the model. On the robot it is
not — see step 3 below.

## 4. On the robot

For how to connect see [`connect.md`](connect.md).

**Step 1 — checks, motors disabled, bridge stopped**

```bash
python tools/mc02_selftest.py --config <yaml>             # board answers; motor ids on both buses match the config
python tools/set_torque.py --config <yaml> --check         # torque limits match the config
python tools/set_can_timeout.py --config <yaml> --check    # communication timeout 0.5 s
python tools/calib_arm.py --config <yaml> --live           # live joint angles: all near 0 in the zero pose
```

The last command confirms that the calibration in the config matches the robot in front of you. Put the arm
in its zero pose (described in the robot's README) and check that every joint reads close to zero. If not,
calibrate first (section 5).

**Step 2 — first enable** (stand outside the arm's reach, 48 V emergency stop at hand)

```bash
# robot computer
python bridge.py --config <yaml> --profile jog            # jog = softer gains
# your computer
python tools/arm_web_jog.py --config <yaml> --gripper --safety safe --bridge-host <robot-ip>
```

Press Connect and do nothing at first: each joint's measured angle should equal its target and the arm should
stay where it is. Then move one joint at a time, a few degrees each. **Use the sliders for this, not mouse
drag or lead by hand** — those move several joints at once, which hides a joint whose direction is wrong.

**Step 3 — confirm the gravity feed-forward**

The bridge compensates the arm's weight from the model. The model is heavier than the real arm, so the model
torque is multiplied by `gravity_ff.scale` in the config. The shipped value was measured on the robot; confirm
it on yours:

- Raise the forearm to horizontal with the sliders and compare measured angle with target. Clearly below the
  target: too little feed-forward. **The arm rises by itself and will not come down: too much.**
- To adjust, start the bridge with `--grav-scale <value>` (overrides the config). Work upward from a low
  value. **Never start at 1.0**: too much makes the arm run upward, too little only makes it sag.
- **If a larger scale makes a joint lift *worse*, that joint's `sign` is wrong.** Go back to 0 and redo the
  direction check in section 5.
- Lead by hand measures the error at start-up and shows "feed-forward suggestion ±0.xx" in its status line —
  the amount to add to the current scale. Write the settled value into the config.

**Step 4 — daily use**

```bash
python bridge.py --config <yaml>                                          # robot computer; default gain profile, scale from the config
python tools/arm_lead_follow.py --config <yaml> --host <robot-ip>         # or any other front end
```

To finish: send the arm home from the front end (a pose that needs no torque), quit the front end, stop the
bridge with Ctrl+C, switch off 48 V.

## 5. Calibration

Each joint's `sign` (direction) and `offset` (zero position) in the config belong to one physical robot.
Recalibrate after replacing a motor or re-assembling a joint, or whenever step 1 does not read zero in the
zero pose.

```bash
# bridge stopped; motors are never enabled by these commands
python tools/calib_arm.py --config <yaml> --zero-here all       # arm in its zero pose: writes every offset into the config
python tools/calib_arm.py --config <yaml> --zero-here <joint>    # one joint only
python tools/calib_arm.py --config <yaml> --signs               # direction check: move each joint as prompted; wrong signs are flipped
python tools/calib_arm.py --config <yaml> --mark-calibrated     # sets calibrated: true once zero and directions are verified
python tools/build_arm_model.py --config <yaml> --check         # config and simulation scene are in sync
```

- With `calibrated: false` the bridge refuses to enable the motors (watching still works). The flag exists so
  that an uncalibrated arm is never powered; set it back to `false` yourself after servicing a joint.
- Check directions with `--signs`, not by comparing with a simulation window: the default camera looks at the
  robot from the front, so left and right are mirrored relative to standing behind it.

## 6. Assigning motor ids

Needed only after replacing a motor. New motors all leave the factory with the same id, and an id change
addresses every motor on the bus that currently has that id — so connect new motors one at a time.

```bash
python tools/set_motor_id.py --bus mc02:0 --scan                 # what is on the bus: id, serial, torque limit
python tools/set_motor_id.py --config <yaml> --assign            # guided: plug in one more connector, press Enter
python tools/set_motor_id.py --config <yaml> --check             # ids on each bus match the config, no duplicates
python tools/set_motor_id.py --config <yaml> --verify            # turn each joint by hand to confirm which id moves
```

Motors stay disabled throughout. Power-cycle and run `--check` once more when done.

## 7. Limits to keep in mind

- **The bridge does not check for self-collision**; it only clamps to the joint range and limits speed.
  Mouse drag, lead by hand, VR and camera do a model-based collision pre-check; sliders and keyboard do not.
- **Motors heat up when an arm is held against gravity.** Rest the arm in its ready or home pose rather than
  holding it stretched out. The bridge prints an over-temperature warning above 75 °C.
- **Grippers** are limited to 5.5 N·m. When the gripper closes on an object, the front ends keep its target
  at most 0.25 rad ahead of the measured position, so it does not keep squeezing harder.
