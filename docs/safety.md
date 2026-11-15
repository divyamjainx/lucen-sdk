# Safety

These actuators are strong enough to injure a hand and to damage the robot. This page lists what the system
does to protect you, and what remains your responsibility.

## Protection layers

Each layer covers a failure the one above cannot.

| Layer | Triggered by | Action | Does not cover |
|---|---|---|---|
| Deployment program (biped_Laika) | Ctrl+C; an exception in the policy; torso tilt > 45°; no IMU data for 0.1 s | All motors to damping, then disabled | The program itself hanging |
| Bridge (arms) | No target from the client for 1.5 s; client disconnects | All motors to damping, then disabled | The bridge process hanging |
| Command path (bridge and deployment) | Every control step | Targets clamped to the joint range; target rate limited; torque capped at the limit | Collisions inside the joint range |
| MC02 watchdog | Host silent for more than 100 ms (process hung or killed, host frozen) | The board itself sends a damping frame to every motor every 10 ms until the host returns | USB unplugged — the board is USB-powered and loses power too |
| Motor timeout | No CAN command for 0.5 s | The motor releases by itself | — |
| Motor firmware limits | Always | Torque and speed never exceed the limits stored in the motor | — |
| Start-up checks, deployment program (biped_Laika) | Torque limit or timeout stored in a motor differs from the config; ONNX not stamped or stamp mismatch | Refuses to start | The bridge does not read motor parameters back: for the arms run `set_torque.py --check` and `set_can_timeout.py --check` yourself |
| Start-up check, bridge | Config marked `calibrated: false` | Refuses to enable the motors (watching still works) | — |
| **48 V emergency stop** | You | Cuts motor power | — |

*Damping* means the motor stops tracking a target and only resists motion. *Disabled* means it is fully
released. Either way **the robot goes limp**: a biped falls, an arm drops. These layers prevent uncontrolled
motion; they do not prevent a fall.

## Your responsibilities

1. **Keep the 48 V emergency stop within reach**, where you can hit it without thinking. It is what remains
   when everything else fails.
2. **biped_Laika: hoist first, ground second.** Start on the hoist again for every new policy, every new way
   of connecting, and after any mechanical work. On the ground, keep the hoist line attached (slack) or have
   a person guarding the robot.
3. **Arms: stand outside the arm's reach.** Use the slowest safety profile (`--safety safe`) for the first
   enable and move one joint at a time.
4. **Secure the USB cable.** Unplugging it is the one disconnection the MC02 watchdog cannot handle; the
   motors hold their last target for 0.5 s before releasing.
5. **Do not raise the torque limit in the motors to "try more force".** The limit is the same number the
   simulation clips at. Changing it only on the robot runs the policy in conditions it has never seen.
   Change the config, the motors and the training setup together.
6. **Do not set `calibrated: true` by hand to get past the check.** Enabling an uncalibrated joint drives it
   at full torque into its mechanical stop.
7. **Stop when motors get hot.** An arm held up, or a biped standing for a long time, is producing torque
   continuously. The bridge prints a warning above 75 °C; the front ends do not show temperature.
8. **Change one thing at a time.** With a new policy, new gains and a new speed all at once, you cannot tell
   which one caused a problem.

## Ways to stop, mildest first

| Action | Effect |
|---|---|
| biped_Laika: space bar on the keyboard / release R1 on the gamepad | Velocity command goes to zero; the robot returns to standing, still under control |
| Arms: STOP or "home" in the front end | The arm stops where it is / returns to a pose that needs no torque, still under control |
| Ctrl+C in the deployment program or the bridge | Damping, then disabled; the robot goes limp |
| 48 V emergency stop | Power is cut at once; the motors produce no torque and the robot goes limp |
