"""Restricted runner for the arm experiments (RSL-RL 3.1 checkpoint format)."""

from copy import deepcopy

import torch
from rsl_rl.runners import OnPolicyRunner


def validate_config(config):
    """Reject executable class expressions and optional callable/plugin hooks."""
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a dictionary.")
    expected = {"policy": "ActorCritic", "algorithm": "PPO"}
    allowed = {
        "policy": {
            "class_name", "actor_hidden_dims", "critic_hidden_dims", "activation",
            "actor_obs_normalization", "critic_obs_normalization", "init_noise_std",
            "noise_std_type",
        },
        "algorithm": {
            "class_name", "num_learning_epochs", "num_mini_batches", "clip_param",
            "gamma", "lam", "value_loss_coef", "entropy_coef", "learning_rate",
            "max_grad_norm", "use_clipped_value_loss", "schedule", "desired_kl",
        },
    }
    for section, class_name in expected.items():
        values = config.get(section)
        if not isinstance(values, dict) or values.get("class_name") != class_name:
            raise ValueError(f"{section}.class_name must be exactly {class_name!r}.")
        unexpected = set(values) - allowed[section]
        if unexpected:
            raise ValueError(f"Unsupported {section} options: {sorted(unexpected)}")
    if config.get("logger", "tensorboard") != "tensorboard":
        raise ValueError("Only the local tensorboard logger is supported.")


class SafeOnPolicyRunner(OnPolicyRunner):
    """Validate before RSL-RL construction and deserialize tensors restrictively."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        validate_config(train_cfg)
        super().__init__(env, deepcopy(train_cfg), log_dir=log_dir, device=device)

    def load(self, path, load_optimizer=True, map_location=None):
        # Never fall back to unrestricted pickle loading or allowlist globals.
        checkpoint = torch.load(
            path, weights_only=True,
            map_location=map_location if map_location is not None else self.device,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary of model state and metadata.")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError("model_state_dict must map parameter names to tensors.")
        iteration = checkpoint.get("iter", 0)
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Checkpoint iteration must be a nonnegative integer.")
        if load_optimizer and not isinstance(checkpoint.get("optimizer_state_dict"), dict):
            raise ValueError("Resuming training requires optimizer_state_dict.")
        resumed = self.alg.policy.load_state_dict(state)
        if resumed:
            if load_optimizer:
                self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_learning_iteration = iteration
        return checkpoint.get("infos")
