import argparse
import json
from pathlib import Path

import torch
from safe_runner import SafeOnPolicyRunner

from grasp_env import GraspEnv
from rsl_env import RslReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)

    with (args.checkpoint.parent / "config.json").open() as file:
        config = json.load(file)

    wrapper = RslReachEnv(
        num_envs=1,
        env_class=GraspEnv,
        num_actions=8,
    )
    runner = SafeOnPolicyRunner(wrapper, config, device="cpu")
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cpu")

    env = wrapper.envs[0]
    env.reset(seed=10000)

    peak_height = 0.0
    longest_hold = 0.0
    next_report_time = 0.0
    done = False

    while not done:
        with torch.inference_mode():
            action = policy(
                wrapper.get_observations()
            )[0].cpu().numpy()

        observation, reward, terminated, truncated, info = env.step(action)

        peak_height = max(peak_height, info["height_gain"])
        longest_hold = max(longest_hold, info["hold_time"])
        elapsed = env.elapsed_steps * env.control_dt

        if elapsed >= next_report_time or terminated or truncated:
            print(
                f"Time {elapsed:.2f} s | "
                f"Lift {info['height_gain'] * 1000:+.1f} mm | "
                f"Fingers {info['finger_contacts']}/2 | "
                f"Pedestal {info['pedestal_contact']} | "
                f"Hold {info['hold_time']:.2f} s"
            )
            next_report_time = elapsed + 1.0

        done = terminated or truncated

    success = (
        not terminated
        and truncated
        and info["success"]
    )

    print(f"\nPeak lift: {peak_height * 1000:.1f} mm")
    print(f"Longest continuous hold: {longest_hold:.2f} s")
    print(f"Final continuous hold: {info['hold_time']:.2f} s")
    print(f"Cube-drop termination: {terminated}")
    print(f"Successful hold at episode end: {success}")


if __name__ == "__main__":
    main()
