import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from safe_runner import SafeOnPolicyRunner
from tensordict import TensorDict

from approach_env import ApproachEnv
from rsl_env import RslReachEnv


def main():
    sdk_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the approach policy by behavior cloning.")
    parser.add_argument("--config", type=Path, default=sdk_root / "configs" / "approach_bc.json")
    parser.add_argument("--dataset", type=Path, default=sdk_root / "data" / "approach_guided.npz")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if not args.dataset.is_file():
        parser.error("Dataset missing. See docs/arm_training.md for dataset preparation.")
    torch.manual_seed(42)
    torch.set_num_threads(1)

    sdk_root = Path(__file__).resolve().parent

    with np.load(args.dataset, allow_pickle=False) as dataset:
        observations = torch.from_numpy(dataset["observations"].copy())
        actions = torch.from_numpy(dataset["actions"].copy())

    if (
        observations.ndim != 2 or observations.shape[1] != 31
        or actions.shape != (len(observations), 8)
        or len(observations) < 2
        or not torch.isfinite(observations).all()
        or not torch.isfinite(actions).all()
    ):
        parser.error("Dataset must contain finite observations [N, 31] and actions [N, 8], with N >= 2.")

    split = int(0.9 * len(observations))
    train_observations = observations[:split]
    train_actions = actions[:split]
    validation_observations = observations[split:]
    validation_actions = actions[split:]

    with args.config.open() as file:
        config = json.load(file)
    if args.epochs is not None:
        config["behavior_cloning"]["epochs"] = args.epochs

    def make_environment():
        return ApproachEnv(start_elbow_offset=config["task_settings"]["start_elbow_offset"])

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = sdk_root / "runs" / f"approach_bc_robust_{run_name}"
    log_dir.mkdir(parents=True)

    with (log_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    wrapper = RslReachEnv(
        num_envs=1,
        env_class=make_environment,
        num_actions=8,
    )
    runner = SafeOnPolicyRunner(wrapper, config, device="cpu")
    policy = runner.alg.policy

    # Estimate observation scaling from training demonstrations only.
    training_batch = TensorDict(
        {"policy": train_observations},
        batch_size=[len(train_observations)],
    )

    with torch.no_grad():
        policy.update_normalization(training_batch)

    # Nearly constant demonstration features must not amplify small deviations.
    # These buffers are part of RSL-RL 3.1's saved normalization state.
    with torch.no_grad():
        for normalizer in [
            policy.actor_obs_normalizer,
            policy.critic_obs_normalizer,
        ]:
            normalizer._std.clamp_(
                min=config["behavior_cloning"]["observation_std_floor"]
            )
            normalizer._var.copy_(normalizer._std.square())

    noise_scale = torch.tensor(
        config["behavior_cloning"]["observation_noise_std"],
        dtype=torch.float32,
    )

    # Freeze normalization statistics while fitting the actor.
    policy.eval()
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=0.001)

    validation_batch = TensorDict(
        {"policy": validation_observations},
        batch_size=[len(validation_observations)],
    )

    best_validation_loss = float("inf")
    checkpoint = log_dir / "model_bc.pt"
    batch_size = 256

    for epoch in range(1, config["behavior_cloning"]["epochs"] + 1):
        order = torch.randperm(len(train_observations))
        total_loss = 0.0

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            # Small observation perturbations discourage brittle feedback.
            noisy_observations = (
                train_observations[indices]
                + torch.randn_like(train_observations[indices]) * noise_scale
            )
            batch = TensorDict(
                {"policy": noisy_observations},
                batch_size=[len(indices)],
            )

            predicted_actions = policy.act_inference(batch)
            loss = torch.nn.functional.mse_loss(
                predicted_actions, train_actions[indices]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(indices)

        if epoch % 10 == 0 or epoch == config["behavior_cloning"]["epochs"]:
            with torch.no_grad():
                predicted = policy.act_inference(validation_batch)
                validation_loss = torch.nn.functional.mse_loss(
                    predicted, validation_actions
                ).item()

            print(
                f"Epoch {epoch:3d} | "
                f"Training MSE {total_loss / split:.6f} | "
                f"Validation MSE {validation_loss:.6f}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                # Save an RSL-RL compatible checkpoint.
                torch.save({
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                    "iter": 0,
                    "infos": {
                        "method": "behavior_cloning",
                        "validation_mse": validation_loss,
                    },
                }, checkpoint)

    print(f"Best validation MSE: {best_validation_loss:.6f}")
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
