import argparse
from datetime import datetime
from pathlib import Path
import json

import torch
from safe_runner import SafeOnPolicyRunner

from rsl_env import RslReachEnv


def main():
    parser = argparse.ArgumentParser(description="Train the reaching policy with PPO.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "arm_ppo.json")
    parser.add_argument("--checkpoint", type=Path, help="Optional PPO checkpoint to resume, including optimizer state.")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error("--checkpoint must name an existing file")
    # Reproduce network initialization and action sampling.
    torch.manual_seed(42)

    # Small networks often run efficiently with one CPU thread.
    torch.set_num_threads(1)

    env = RslReachEnv(num_envs=1, seed=42)

    with args.config.open() as file:
        config = json.load(file)

    # Give every run a separate folder for checkpoints and training curves.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = Path(__file__).resolve().parent / "runs" / run_name
    log_dir.mkdir(parents=True)

    # Save before constructing the runner, which modifies its configuration.
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

    print(f"Saving this run to: {log_dir}")

    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=False,
    )
    print(f"Training finished. Checkpoints are in: {log_dir}")


if __name__ == "__main__":
    main()
