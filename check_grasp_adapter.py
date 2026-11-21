import torch

from grasp_env import GraspEnv
from rsl_env import RslReachEnv


env = RslReachEnv(
    num_envs=1,
    env_class=GraspEnv,
    num_actions=8,
)

# Hold the arm's ready pose and request an open gripper.
actions = torch.zeros((1, 8))
actions[:, 7] = -1.0

observations, rewards, dones, extras = env.step(actions)

print("Observation shape:", observations["policy"].shape)
print("Action shape:", actions.shape)
print("Reward shape:", rewards.shape)
