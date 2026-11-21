from time import perf_counter

import numpy as np

from reach_env import ReachEnv


def main():
    env = ReachEnv()
    action = np.zeros(7)

    # Warm up the simulation before measuring its speed.
    for _ in range(20):
        env.step(action)

    env.reset()
    number_of_steps = 1000
    episode_count = 0

    start = perf_counter()

    for _ in range(number_of_steps):
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            episode_count += 1
            env.reset()

    elapsed = perf_counter() - start
    steps_per_second = number_of_steps / elapsed
    simulated_seconds = number_of_steps * env.control_dt

    print(f"Wall-clock time: {elapsed:.2f} seconds")
    print(f"Environment steps per second: {steps_per_second:.0f}")
    print(f"Simulated time: {simulated_seconds:.2f} seconds")
    print(f"Simulation speed: {simulated_seconds / elapsed:.1f}x real time")
    print(f"Completed episodes: {episode_count}")


if __name__ == "__main__":
    main()
