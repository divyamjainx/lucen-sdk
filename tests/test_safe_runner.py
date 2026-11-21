"""Regression checks for malicious configuration and checkpoint rejection."""
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from safe_runner import SafeOnPolicyRunner, validate_config


class ForbiddenPayload:
    def __reduce__(self):
        return (eval, ("1 + 1",))


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/arm_ppo.json").read_text()
        )

    def test_published_configuration_is_allowed(self):
        validate_config(self.config)

    def test_expression_rejected_before_runner_initialization(self):
        self.config["policy"]["class_name"] = "__import__('os').system('false')"
        with patch("safe_runner.OnPolicyRunner.__init__") as constructor:
            with self.assertRaises(ValueError):
                SafeOnPolicyRunner(None, self.config)
            constructor.assert_not_called()

    def test_callable_hooks_and_remote_logger_rejected(self):
        for key in ["symmetry_cfg", "rnd_cfg"]:
            config = json.loads(json.dumps(self.config))
            config["algorithm"][key] = {"function": "os:system"}
            with self.assertRaises(ValueError):
                validate_config(config)
        self.config["logger"] = "wandb"
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_checkpoint_loading(self):
        runner = SafeOnPolicyRunner.__new__(SafeOnPolicyRunner)
        runner.device = "cpu"
        runner.alg = Mock()
        runner.alg.policy.load_state_dict.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_state_dict": {"weight": torch.ones(1)},
                        "optimizer_state_dict": {}, "iter": 7}, path)
            runner.load(path)
            self.assertEqual(runner.current_learning_iteration, 7)
            runner.alg.optimizer.load_state_dict.assert_called_once_with({})
            torch.save({"payload": ForbiddenPayload()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                runner.load(path)
            torch.save({"model_state_dict": {"weight": "not a tensor"}}, path)
            with self.assertRaises(ValueError):
                runner.load(path, load_optimizer=False)


if __name__ == "__main__":
    unittest.main()
