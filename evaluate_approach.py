import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safe_runner import SafeOnPolicyRunner

from approach_env import ApproachEnv
from rsl_env import RslReachEnv


def evaluate_episode(wrapper, policy, seed, report_interval=None):
    env = wrapper.envs[0]
    observation = env.reset(seed=seed)
    initial_distance = float(np.linalg.norm(observation[16:19]))
    peak_height = 0.0
    longest_hold = 0.0
    next_report_time = 0.0
    done = False

    if report_interval is not None:
        print(f"Before first action: {initial_distance * 100:.2f} cm")

    while not done:
        with torch.inference_mode():
            action = policy(wrapper.get_observations())[0].cpu().numpy()

        observation, reward, terminated, truncated, info = env.step(action)
        peak_height = max(peak_height, info["height_gain"])
        longest_hold = max(longest_hold, info["hold_time"])
        elapsed = env.elapsed_steps * env.control_dt
        done = terminated or truncated

        if report_interval is not None and (
            elapsed >= next_report_time or done
        ):
            requested_gripper = 0.55 * (float(np.clip(action[7], -1, 1)) + 1)
            actual_gripper = float(
                env.data.qpos[env.gripper_position_index]
            )
            print(
                f"Time {elapsed:.2f} s | "
                f"Hand distance {info['distance'] * 100:.2f} cm | "
                f"Gripper target {requested_gripper:.2f} rad | "
                f"Actual {actual_gripper:.2f} rad | "
                f"Fingers {info['finger_contacts']}/2 | "
                f"Lift {info['height_gain'] * 1000:+.1f} mm"
            )
            next_report_time = elapsed + report_interval

    return {
        "seed": seed,
        "initial_distance_cm": initial_distance * 100,
        "peak_lift_mm": peak_height * 1000,
        "final_lift_mm": info["height_gain"] * 1000,
        "longest_hold_s": longest_hold,
        "final_hold_s": info["hold_time"],
        "dropped": bool(terminated),
        "success": bool(truncated and not terminated and info["success"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--report-interval", type=float, default=1.0)
    args = parser.parse_args()

    if args.episodes < 1 or args.report_interval <= 0:
        parser.error("Episodes and report interval must be positive.")

    torch.set_num_threads(1)
    with (args.checkpoint.parent / "config.json").open() as file:
        config = json.load(file)

    start_offset = config.get("task_settings", {}).get(
        "start_elbow_offset", 0.18
    )

    def make_evaluation_env():
        return ApproachEnv(start_elbow_offset=start_offset)

    print(f"Evaluating starting elbow offset: {start_offset:.3f} rad")
    wrapper = RslReachEnv(
        num_envs=1,
        env_class=make_evaluation_env,
        num_actions=8,
    )
    runner = SafeOnPolicyRunner(wrapper, config, device="cpu")
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cpu")

    results = []
    for episode in range(args.episodes):
        result = evaluate_episode(
            wrapper,
            policy,
            seed=args.seed + episode,
            report_interval=args.report_interval if args.episodes == 1 else None,
        )
        results.append(result)
        print(
            f"Episode {episode + 1}/{args.episodes} | "
            f"Seed {result['seed']} | "
            f"Final lift {result['final_lift_mm']:.1f} mm | "
            f"Final hold {result['final_hold_s']:.2f} s | "
            f"Success {result['success']}"
        )

    if args.episodes == 1:
        result = results[0]
        print(f"\nPeak lift: {result['peak_lift_mm']:.1f} mm")
        print(f"Longest continuous hold: {result['longest_hold_s']:.2f} s")
        print(f"Final continuous hold: {result['final_hold_s']:.2f} s")
        print(f"Cube-drop termination: {result['dropped']}")
        print(f"Successful hold at episode end: {result['success']}")

    successes = sum(result["success"] for result in results)
    print(f"\nSuccessful episodes: {successes}/{len(results)}")
    print(
        "Mean final lift: "
        f"{np.mean([result['final_lift_mm'] for result in results]):.1f} mm"
    )
    print(
        "Minimum final hold: "
        f"{min(result['final_hold_s'] for result in results):.2f} s"
    )


if __name__ == "__main__":
    main()
