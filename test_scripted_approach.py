import numpy as np

from approach_env import ApproachEnv


def main():
    env = ApproachEnv()
    elbow_index = env.sim.joint_names.index("r_knee_joint")

    for episode in range(5):
        env.reset(seed=100 + episode)

        # Actions are offsets from the controller's ready-pose targets.
        reference = env.sim.ready_motor_targets[
            env.sim.right_motor_indices
        ].copy()

        start_action = (
            env.data.qpos[env.sim.position_indices] - reference
        ) / env.sim.action_scale

        goal_action = (
            env.goal_joint_positions - reference
        ) / env.sim.action_scale

        initial_distance = np.linalg.norm(
            env.get_observation()[16:19]
        )

        done = False
        longest_hold = 0.0

        while not done:
            elapsed = env.elapsed_steps * env.control_dt

            approach_progress = np.clip(
                (elapsed - 0.5) / 2.0, 0.0, 1.0
            )
            close_progress = np.clip(
                (elapsed - 2.5) / 1.5, 0.0, 1.0
            )
            lift_progress = np.clip(
                (elapsed - 4.5) / 2.0, 0.0, 1.0
            )

            action = np.zeros(8)

            # Move from the initial arm configuration to the grasp pose.
            action[:7] = (
                (1.0 - approach_progress) * start_action
                + approach_progress * goal_action
            )

            # Close the fingers, then bend the elbow to raise the cube.
            action[7] = 2.0 * close_progress - 1.0
            action[elbow_index] -= 0.75 * lift_progress

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            longest_hold = max(longest_hold, info["hold_time"])
            done = terminated or truncated

        success = (
            truncated
            and not terminated
            and info["success"]
        )

        print(
            f"Episode {episode + 1} | "
            f"Initial distance {initial_distance * 100:.2f} cm | "
            f"Final lift {info['height_gain'] * 1000:.1f} mm | "
            f"Longest hold {longest_hold:.2f} s | "
            f"Final hold {info['hold_time']:.2f} s | "
            f"Success {success}"
        )


if __name__ == "__main__":
    main()
