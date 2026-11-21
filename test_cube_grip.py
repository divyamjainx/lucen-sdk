"""Measure cube contact using the modified grasp scene's flat collision pads."""

import argparse
from contextlib import nullcontext
import time

import mujoco
import mujoco.viewer
import numpy as np

from reach_env import ReachEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without a viewer or wall-clock pacing.")
    args = parser.parse_args()
    env = ReachEnv(
        scene_path="humanoid_arm/mjcf/humanoid_arm_grasp_scene.xml"
    )

    motor_names = [
        joint["name"] for joint in env.config["joints"]
    ]
    gripper_index = motor_names.index("r_gripper_joint")

    cube_id = env.model.geom("task_cube_geom").id
    cube_body_id = env.model.body("task_cube").id
    finger_ids = {
        env.model.geom("r_jaw_right_pad").id,
        env.model.geom("r_jaw_left_pad").id,
    }

    initial_height = float(env.data.xpos[cube_body_id, 2])
    action = np.zeros(7)
    next_report_time = 0.0

    viewer_context = (
        nullcontext(None) if args.headless
        else mujoco.viewer.launch_passive(env.model, env.data)
    )
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = env.data.site("r_ee").xpos
            viewer.cam.distance = 0.5

        # Run an eight-second experiment, then close automatically.
        while (viewer is None or viewer.is_running()) and env.data.time < 8.0:
            step_start = time.monotonic()

            with (viewer.lock() if viewer is not None else nullcontext()):
                # Wait one second, then close gradually over two seconds.
                progress = np.clip((env.data.time - 1.0) / 2.0, 0.0, 1.0)
                env.ready_motor_targets[gripper_index] = 1.5 * progress
                env.step(action)

                touching_fingers = set()

                for index in range(env.data.ncon):
                    contact = env.data.contact[index]
                    pair = {int(contact.geom1), int(contact.geom2)}

                    if cube_id in pair:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(
                            env.model, env.data, index, force
                        )

                        if force[0] > 0.01:
                            touching_fingers.update(pair & finger_ids)

                if env.data.time >= next_report_time:
                    height_change = (
                        env.data.xpos[cube_body_id, 2] - initial_height
                    )

                    print(
                        f"Time {env.data.time:.1f} s | "
                        f"Fingers contacting cube {len(touching_fingers)}/2 | "
                        f"Cube height change {height_change * 1000:.1f} mm"
                    )
                    next_report_time = env.data.time + 1.0

            if viewer is not None:
                viewer.sync()
                remaining = env.control_dt - (time.monotonic() - step_start)
                time.sleep(max(0.0, remaining))


if __name__ == "__main__":
    main()
