import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from safe_runner import SafeOnPolicyRunner

from approach_env import ApproachEnv
from rsl_env import RslReachEnv

def main():
    parser = argparse.ArgumentParser(description="Experimental PPO approach-and-grasp training; the published final policy uses behavior cloning.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "arm_ppo.json")
    parser.add_argument("--checkpoint", type=Path, help="Optional PPO checkpoint to resume, including optimizer state.")
    parser.add_argument("--start-elbow-offset", type=float, default=0.06)
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

    config["task_settings"] = {
        "start_elbow_offset": args.start_elbow_offset,
    }

    def make_training_env():
        return ApproachEnv(start_elbow_offset=args.start_elbow_offset)

    env = RslReachEnv(
        num_envs=1,
        seed=42,
        env_class=make_training_env,
        num_actions=8,
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = sdk_root / "runs" / f"approach_{run_name}"
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

    print(f"Training approach-and-grasp for {args.iterations} iterations.")
    print(f"Saving results to: {log_dir}")

    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=False,
    )

    print(f"Training finished. Checkpoints are in: {log_dir}")


if __name__ == "__main__":
    main()
