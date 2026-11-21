import argparse
import json
from pathlib import Path
import time

import mujoco.viewer
import torch
from safe_runner import SafeOnPolicyRunner

from approach_env import ApproachEnv
from grasp_env import GraspEnv
from rsl_env import RslReachEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()

    torch.set_num_threads(1)

    with (args.checkpoint.parent / "config.json").open() as file:
        config = json.load(file)

    start_offset = config.get("task_settings", {}).get("start_elbow_offset")

    def make_environment():
        if start_offset is None:
            return GraspEnv()
        return ApproachEnv(start_elbow_offset=start_offset)

    wrapper = RslReachEnv(
        num_envs=1,
        env_class=make_environment,
        num_actions=8,
    )
    runner = SafeOnPolicyRunner(wrapper, config, device="cpu")
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cpu")

    env = wrapper.envs[0]
    env.reset(seed=args.seed)

    done = False
    next_report_time = 0.0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = env.data.site("r_ee").xpos
        viewer.cam.distance = 0.6

        while viewer.is_running():
            step_start = time.monotonic()

            if not done:
                with viewer.lock():
                    with torch.inference_mode():
                        action = policy(
                            wrapper.get_observations()
                        )[0].cpu().numpy()

                    observation, reward, terminated, truncated, info = (
                        env.step(action)
                    )
                    done = terminated or truncated
                    elapsed = env.elapsed_steps * env.control_dt

                    if elapsed >= next_report_time or done:
                        print(
                            f"Time {elapsed:.2f} s | "
                            f"Lift {info['height_gain'] * 1000:.1f} mm | "
                            f"Hold {info['hold_time']:.2f} s"
                        )
                        next_report_time = elapsed + 1.0

                    if done:
                        print(
                            "Successful final hold:",
                            info["success"] and not terminated,
                        )
                        print("Episode finished. Close the viewer to exit.")

            viewer.sync()

            remaining = env.control_dt - (time.monotonic() - step_start)
            time.sleep(max(0.0, remaining))


if __name__ == "__main__":
    main()
