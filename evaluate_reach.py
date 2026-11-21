import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safe_runner import SafeOnPolicyRunner

from rsl_env import RslReachEnv


def evaluate(wrapper, policy=None, episodes=20):
    env = wrapper.envs[0]

    # Restart the target sequence so both methods get identical targets.
    env.rng = np.random.default_rng(10000)

    final_distances = []
    hold_distances = []
    successful_holds = 0
    hold_steps = max(1, round(1.0 / env.control_dt))

    for _ in range(episodes):
        env.reset()
        distances = []
        done = False

        while not done:
            if policy is None:
                action = np.zeros(7)
            else:
                with torch.inference_mode():
                    action = policy(
                        wrapper.get_observations()
                    )[0].cpu().numpy()

            observation, reward, terminated, truncated, info = env.step(
                action
            )
            distances.append(info["distance"])
            done = terminated or truncated

        last_second = np.array(distances[-hold_steps:])

        final_distances.append(distances[-1])
        hold_distances.append(last_second.mean())

        # Require the gripper to stay within 1 cm for the final second.
        successful_holds += int(np.all(last_second < 0.01))

    print(
        f"Mean final distance: "
        f"{np.mean(final_distances) * 100:.2f} cm"
    )
    print(
        f"Mean distance during final second: "
        f"{np.mean(hold_distances) * 100:.2f} cm"
    )
    print(f"Successful holds: {successful_holds}/{episodes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)

    config_path = args.checkpoint.parent / "config.json"
    with config_path.open() as file:
        config = json.load(file)

    wrapper = RslReachEnv(num_envs=1, seed=10000)
    runner = SafeOnPolicyRunner(wrapper, config, device="cpu")
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cpu")

    print("\nHolding the ready pose")
    evaluate(wrapper)

    print("\nTrained policy")
    evaluate(wrapper, policy=policy)


if __name__ == "__main__":
    main()
