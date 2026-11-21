import argparse
import json
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from safe_runner import SafeOnPolicyRunner

from rsl_env import RslReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)

    with (args.checkpoint.parent / "config.json").open() as file:
        config = json.load(file)

    wrapper = RslReachEnv(num_envs=1, seed=10000)
    runner = SafeOnPolicyRunner(wrapper, config, device="cpu")
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cpu")

    # Access the single MuJoCo environment inside the adapter.
    env = wrapper.envs[0]

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 1
        print(f"Episode {episode}")

        while viewer.is_running():
            step_start = time.monotonic()

            with viewer.lock():
                # Ask the trained actor for seven joint-target actions.
                with torch.inference_mode():
                    action = policy(
                        wrapper.get_observations()
                    )[0].cpu().numpy()

                observation, reward, terminated, truncated, info = env.step(
                    action
                )

                if terminated or truncated:
                    print(
                        f"Final distance: {info['distance'] * 100:.2f} cm"
                    )
                    env.reset()
                    episode += 1
                    print(f"Episode {episode}")

                # Refresh the marker because reset can change the target.
                viewer.user_scn.ngeom = 1
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.015, 0.015, 0.015]),
                    pos=env.target_position,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.0, 1.0, 0.0, 0.7]),
                )

            viewer.sync()

            remaining = env.control_dt - (time.monotonic() - step_start)
            time.sleep(max(0.0, remaining))


if __name__ == "__main__":
    main()
