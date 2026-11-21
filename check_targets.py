import mujoco

from reach_env import ReachEnv


def main():
    env = ReachEnv()
    number_of_targets = 100
    penetrating_poses = 0
    deepest_penetration = 0.0

    for sample in range(number_of_targets):
        env.reset()

        # reset() already calculates this separate target pose.
        target_state = env.target_data
        has_penetration = False

        for contact_index in range(target_state.ncon):
            contact = target_state.contact[contact_index]

            # Negative contact distance means overlapping collision surfaces.
            if contact.dist < -0.0001:
                has_penetration = True
                depth = -float(contact.dist)
                deepest_penetration = max(deepest_penetration, depth)

                if penetrating_poses < 3:
                    first_geom = mujoco.mj_id2name(
                        env.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom1),
                    )
                    second_geom = mujoco.mj_id2name(
                        env.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom2),
                    )

                    print(
                        f"Sample {sample + 1}: "
                        f"{first_geom} / {second_geom}, "
                        f"overlap {depth * 1000:.2f} mm"
                    )

        penetrating_poses += int(has_penetration)

    print(f"Checked target poses: {number_of_targets}")
    print(f"Poses with overlap above 0.1 mm: {penetrating_poses}")
    print(f"Deepest overlap: {deepest_penetration * 1000:.2f} mm")


if __name__ == "__main__":
    main()
