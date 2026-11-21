"""Measure cube contact using the modified grasp scene's flat collision pads."""

import numpy as np
import mujoco

from reach_env import ReachEnv


def main():
    for closing_target in [0.8, 1.0, 1.2, 1.5]:
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

        initial_position = env.data.xpos[cube_body_id].copy()
        action = np.zeros(7)
        bilateral_samples = 0
        hold_samples = 0

        while env.data.time < 6.0:
            progress = np.clip(
                (env.data.time - 1.0) / 2.0, 0.0, 1.0
            )
            env.ready_motor_targets[gripper_index] = (
                closing_target * progress
            )
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

            # Measure sustained two-finger contact over the final second.
            if env.data.time > 5.0:
                hold_samples += 1
                bilateral_samples += int(len(touching_fingers) == 2)

        displacement = env.data.xpos[cube_body_id] - initial_position
        lateral_shift = np.linalg.norm(displacement[:2])
        contact_fraction = bilateral_samples / hold_samples

        print(
            f"Target {closing_target:.1f} rad | "
            f"Vertical change {displacement[2] * 1000:+.1f} mm | "
            f"Sideways shift {lateral_shift * 1000:.1f} mm | "
            f"Two-finger contact {contact_fraction:.0%}"
        )


if __name__ == "__main__":
    main()
