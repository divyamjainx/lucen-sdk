import math

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from reach_env import ReachEnv


class RslReachEnv(VecEnv):
    def __init__(
        self,
        num_envs=1,
        seed=42,
        env_class=ReachEnv,
        num_actions=7,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be at least one.")

        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = "cpu"
        self.cfg = {"task": env_class.__name__}

        self.envs = []

        for index in range(num_envs):
            env = env_class()
            env.reset(seed=seed + index)
            self.envs.append(env)

        first_env = self.envs[0]
        self.max_episode_length = math.ceil(
            first_env.episode_duration / first_env.control_dt
        )
        self.episode_length_buf = torch.zeros(
            num_envs, dtype=torch.long
        )

    def get_observations(self):
        observations = np.stack([
            env.get_observation() for env in self.envs
        ])
        observations = torch.from_numpy(observations)

        return TensorDict(
            {"policy": observations},
            batch_size=[self.num_envs],
        )

    def step(self, actions):
        actions = actions.detach().cpu().numpy()

        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(
                f"Expected actions with shape "
                f"({self.num_envs}, {self.num_actions})."
            )

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)
        distances = np.zeros(self.num_envs, dtype=np.float32)

        for index, env in enumerate(self.envs):
            observation, reward, terminated, truncated, info = env.step(
                actions[index]
            )

            rewards[index] = reward
            dones[index] = terminated or truncated
            time_outs[index] = truncated and not terminated
            distances[index] = info["distance"]
            self.episode_length_buf[index] += 1

            if dones[index]:
                env.reset()
                self.episode_length_buf[index] = 0

        extras = {
            "time_outs": torch.from_numpy(time_outs),
            "log": {
                "Reach/distance_cm": float(distances.mean() * 100),
            },
        }

        return (
            self.get_observations(),
            torch.from_numpy(rewards),
            torch.from_numpy(dones),
            extras,
        )


if __name__ == "__main__":
    env = RslReachEnv(num_envs=1)
    observations = env.get_observations()

    print("Observation shape:", observations["policy"].shape)

    actions = torch.zeros((env.num_envs, env.num_actions))
    completed_episodes = 0

    for _ in range(env.max_episode_length + 1):
        observations, rewards, dones, extras = env.step(actions)
        completed_episodes += int(dones.sum().item())

    print("Reward shape:", rewards.shape)
    print("Completed episodes:", completed_episodes)
    print("Current episode lengths:", env.episode_length_buf.tolist())
