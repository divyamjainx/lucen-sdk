import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safe_runner import SafeOnPolicyRunner

from approach_env import ApproachEnv
from collect_approach_demos import scripted_action
from rsl_env import RslReachEnv


def make_environment():
    return ApproachEnv(start_elbow_offset=0.18)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    sdk_root = Path(__file__).resolve().parent

    with (args.checkpoint.parent / "config.json").open() as file:
        config = json.load(file)

    wrapper = RslReachEnv(
        num_envs=1,
        env_class=make_environment,
        num_actions=8,
    )
    runner = SafeOnPolicyRunner(wrapper, config, device="cpu")
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cpu")
    env = wrapper.envs[0]

    with np.load(sdk_root / "data" / "approach_demos.npz") as dataset:
        observations = list(dataset["observations"].copy())
        actions = list(dataset["actions"].copy())

    successful_episodes = 0

    for episode in range(20):
        observation = env.reset(seed=2000 + episode)

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
            teacher_action = scripted_action(
                env, start_action, goal_action
            )

            with torch.inference_mode():
                learner_action = policy(
                    wrapper.get_observations()
                )[0].cpu().numpy()

            # Let the learner perturb the teacher's trajectory.
            executed_action = np.clip(
                0.8 * teacher_action + 0.2 * learner_action,
                -1.0,
                1.0,
            )

            episode_observations.append(observation.copy())
            episode_actions.append(teacher_action.copy())

            observation, reward, terminated, truncated, info = env.step(
                executed_action
            )
            done = terminated or truncated

        success = truncated and not terminated and info["success"]

        if success:
            observations.extend(episode_observations)
            actions.extend(episode_actions)
            successful_episodes += 1

        print(
            f"Guided episode {episode + 1}/20 | "
            f"Success {success}"
        )

    destination = sdk_root / "data" / "approach_guided.npz"

    np.savez_compressed(
        destination,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        start_elbow_offset=np.float32(0.18),
    )

    print(f"Successful guided episodes: {successful_episodes}/20")
    print(f"Total training pairs: {len(observations)}")
    print(f"Saved to: {destination}")


if __name__ == "__main__":
    main()
