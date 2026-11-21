from pathlib import Path

import numpy as np

from approach_env import ApproachEnv


def scripted_action(env, start_action, goal_action):
    elapsed = env.elapsed_steps * env.control_dt

    approach = np.clip((elapsed - 0.5) / 2.0, 0.0, 1.0)
    close = np.clip((elapsed - 2.5) / 1.5, 0.0, 1.0)
    lift = np.clip((elapsed - 4.5) / 2.0, 0.0, 1.0)

    action = np.zeros(8)
    action[:7] = (
        (1.0 - approach) * start_action
        + approach * goal_action
    )
    action[7] = 2.0 * close - 1.0

    elbow_index = env.sim.joint_names.index("r_knee_joint")
    action[elbow_index] -= 0.75 * lift

    return np.clip(action, -1.0, 1.0)


def main():
    env = ApproachEnv(start_elbow_offset=0.18)

    observations = []
    actions = []
    successful_episodes = 0

    for episode in range(20):
        observation = env.reset(seed=1000 + episode)

        reference = env.sim.ready_motor_targets[
            env.sim.right_motor_indices
        ].copy()

        start_action = (
            env.data.qpos[env.sim.position_indices] - reference
        ) / env.sim.action_scale

        goal_action = (
            env.goal_joint_positions - reference
        ) / env.sim.action_scale

        episode_observations = []
        episode_actions = []
        done = False

        while not done:
            action = scripted_action(env, start_action, goal_action)

            # Pair each observation with the action chosen from that state.
            episode_observations.append(observation.copy())
            episode_actions.append(action.copy())

            observation, reward, terminated, truncated, info = env.step(
                action
            )
            done = terminated or truncated

        success = truncated and not terminated and info["success"]

        if success:
            observations.extend(episode_observations)
            actions.extend(episode_actions)
            successful_episodes += 1

        print(f"Demonstration {episode + 1}/20 | Success {success}")

    if not observations:
        raise RuntimeError("No successful demonstrations were collected.")

    destination = (
        Path(__file__).resolve().parent
        / "data"
        / "approach_demos.npz"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        destination,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        start_elbow_offset=np.float32(0.18),
    )

    print(f"Successful demonstrations: {successful_episodes}/20")
    print(f"Observation shape: {np.asarray(observations).shape}")
    print(f"Action shape: {np.asarray(actions).shape}")
    print(f"Saved to: {destination}")


if __name__ == "__main__":
    main()
