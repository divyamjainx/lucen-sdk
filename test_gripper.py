import time

import mujoco
import mujoco.viewer
import numpy as np

from reach_env import ReachEnv


def main():
    env = ReachEnv()

    motor_names = [
        joint["name"] for joint in env.config["joints"]
    ]
    gripper_index = motor_names.index("r_gripper_joint")

    gripper_address = int(
        env.model.joint("r_gripper_joint").qposadr[0]
    )
    right_finger_address = int(
        env.model.joint("r_jaw_right_joint").qposadr[0]
    )
    left_finger_address = int(
        env.model.joint("r_jaw_left_joint").qposadr[0]
    )

    action = np.zeros(7)
    next_report_time = 0.0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Focus the camera near the right hand.
        viewer.cam.lookat[:] = env.data.site("r_ee").xpos
        viewer.cam.distance = 0.5

        while viewer.is_running():
            step_start = time.monotonic()

            with viewer.lock():
                # Smoothly open and close over an eight-second cycle.
                phase = 2.0 * np.pi * env.data.time / 8.0
                requested_angle = 0.75 * (1.0 - np.cos(phase))

                env.ready_motor_targets[gripper_index] = requested_angle
                env.step(action)

                if env.data.time >= next_report_time:
                    print(
                        f"Target {requested_angle:.2f} rad | "
                        f"Actual {env.data.qpos[gripper_address]:.2f} rad | "
                        f"Finger positions "
                        f"{env.data.qpos[right_finger_address] * 1000:.1f}, "
                        f"{env.data.qpos[left_finger_address] * 1000:.1f} mm"
                    )
                    next_report_time = env.data.time + 1.0

            viewer.sync()

            remaining = env.control_dt - (time.monotonic() - step_start)
            time.sleep(max(0.0, remaining))


if __name__ == "__main__":
    main()
