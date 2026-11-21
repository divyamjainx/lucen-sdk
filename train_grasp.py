import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from safe_runner import SafeOnPolicyRunner

from grasp_env import GraspEnv
from rsl_env import RslReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "arm_ppo.json")
    parser.add_argument("--checkpoint", type=Path, help="Optional PPO checkpoint to resume, including optimizer state.")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("--checkpoint must name an existing file")

    torch.manual_seed(42)
    torch.set_num_threads(1)

    sdk_root = Path(__file__).resolve().parent

    with args.config.open() as file:
        config = json.load(file)

    env = RslReachEnv(
        num_envs=1,
        seed=42,
        env_class=GraspEnv,
        num_actions=8,
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = sdk_root / "runs" / f"grasp_{run_name}"
    log_dir.mkdir(parents=True)

    with (log_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    runner = SafeOnPolicyRunner(
        env,
        config,
        log_dir=str(log_dir),
        device="cpu",
    )

    if args.checkpoint is not None:
        runner.load(str(args.checkpoint), load_optimizer=True)
        print(f"Continuing from: {args.checkpoint}")

    print(f"Training a grasping policy for {args.iterations} iterations.")
    print(f"Saving results to: {log_dir}")

    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=False,
    )

    print(f"Training finished. Checkpoints are in: {log_dir}")


if __name__ == "__main__":
    main()
