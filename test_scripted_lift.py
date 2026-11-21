import mujoco
import numpy as np

from reach_env import ReachEnv


def main():
    env = ReachEnv(
        scene_path="humanoid_arm/mjcf/humanoid_arm_grasp_scene.xml"
    )

    motor_names = [
        joint["name"] for joint in env.config["joints"]
    ]
    gripper_index = motor_names.index("r_gripper_joint")
    elbow_index = env.joint_names.index("r_knee_joint")

    cube_id = env.model.geom("task_cube_geom").id
    pedestal_id = env.model.geom("cube_pedestal").id
    cube_body_id = env.model.body("task_cube").id

    finger_ids = {
        env.model.geom("r_jaw_right_pad").id,
        env.model.geom("r_jaw_left_pad").id,
    }

    initial_height = float(env.data.xpos[cube_body_id, 2])
    next_report_time = 0.0
    hold_samples = 0
    successful_samples = 0

    while env.data.time < 9.0:
        close_progress = np.clip(
            (env.data.time - 1.0) / 2.0, 0.0, 1.0
        )
        env.ready_motor_targets[gripper_index] = 1.1 * close_progress

        # After gripping, bend the elbow gradually to raise the hand.
        lift_progress = np.clip(
            (env.data.time - 4.0) / 2.0, 0.0, 1.0
        )
        action = np.zeros(7)
        action[elbow_index] = -0.75 * lift_progress
        env.step(action)

        touching_fingers = set()
        pedestal_contact = False

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
                    pedestal_contact |= pedestal_id in pair

                    # Inspect finger forces immediately before and during lift.
                    report_contact = (
                        env.data.time >= next_report_time
                        and 3.0 <= env.data.time <= 5.0
                        and bool(pair & finger_ids)
                    )

                    if report_contact:
                        # Contact normal points from geom1 toward geom2.
                        normal_on_cube = contact.frame[:3].copy()

                        if int(contact.geom1) == cube_id:
                            normal_on_cube *= -1.0

                        finger_id = next(iter(pair & finger_ids))
                        finger_name = env.model.geom(finger_id).name

                        print(
                            f"  {finger_name} | "
                            f"Normal force {force[0]:.2f} N | "
                            f"Direction on cube "
                            f"{np.round(normal_on_cube, 3)}"
                        )

        height_gain = (
            float(env.data.xpos[cube_body_id, 2]) - initial_height
        )

        if env.data.time > 8.0:
            hold_samples += 1
            successful_samples += int(
                height_gain > 0.02
                and len(touching_fingers) == 2
                and not pedestal_contact
            )

        if env.data.time >= next_report_time:
            print(
                f"Time {env.data.time:.1f} s | "
                f"Lift {height_gain * 1000:+.1f} mm | "
                f"Finger contacts {len(touching_fingers)}/2 | "
                f"Pedestal contact {pedestal_contact}"
            )
            next_report_time = env.data.time + 1.0

    print(
        "Successful lift-and-hold samples during final second:",
        f"{successful_samples}/{hold_samples}",
    )


if __name__ == "__main__":
    main()
