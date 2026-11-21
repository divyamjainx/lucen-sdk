"""Experimental local inverse-kinematics search, not a reachability certificate.

Search bounds are centered on the episode's starting pose, unlike the controller's
ready-pose-centered action bounds. The solver also preserves the starting hand
orientation. A failed local search does not establish that a target is unreachable;
a found pose does not establish collision-free or dynamically feasible motion.
"""

import mujoco
import numpy as np

from approach_env import ApproachEnv


def main():
    print("Experimental local IK search: starting-pose bounds and orientation.")
    print("Results do not certify controller reachability, collision clearance, or dynamic feasibility.")
    env = ApproachEnv()
    model = env.model

    position_indices = env.sim.position_indices
    velocity_indices = env.sim.velocity_indices
    joint_ids = [
        model.joint(name).id for name in env.sim.joint_names
    ]
    joint_limits = model.jnt_range[joint_ids]
    site_id = model.site("r_ee").id

    for sample in range(5):
        env.reset()

        # Use a separate candidate state to preserve the episode state.
        candidate = mujoco.MjData(model)
        candidate.qpos[:] = env.data.qpos
        mujoco.mj_forward(model, candidate)

        reference_angles = candidate.qpos[position_indices].copy()
        lower = np.maximum(
            reference_angles - env.sim.action_scale,
            joint_limits[:, 0],
        )
        upper = np.minimum(
            reference_angles + env.sim.action_scale,
            joint_limits[:, 1],
        )

        cube_start = env.cube_position_index
        target_position = env.data.qpos[cube_start:cube_start + 3].copy()
        target_rotation = candidate.site("r_ee").xmat.reshape(3, 3).copy()

        jacobian_position = np.zeros((3, model.nv))
        jacobian_rotation = np.zeros((3, model.nv))

        for _ in range(200):
            hand = candidate.site("r_ee")
            rotation = hand.xmat.reshape(3, 3)

            position_error = target_position - hand.xpos

            # Small-angle orientation error expressed in world coordinates.
            rotation_error = 0.5 * sum(
                np.cross(rotation[:, axis], target_rotation[:, axis])
                for axis in range(3)
            )
            error = np.concatenate([position_error, rotation_error])

            mujoco.mj_jacSite(
                model,
                candidate,
                jacobian_position,
                jacobian_rotation,
                site_id,
            )
            jacobian = np.vstack([
                jacobian_position[:, velocity_indices],
                jacobian_rotation[:, velocity_indices],
            ])

            # Damping keeps the calculation stable near difficult poses.
            change = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.0001 * np.eye(6),
                error,
            )
            change = np.clip(0.5 * change, -0.02, 0.02)

            candidate.qpos[position_indices] = np.clip(
                candidate.qpos[position_indices] + change,
                lower,
                upper,
            )
            mujoco.mj_forward(model, candidate)

        final_hand = candidate.site("r_ee")
        position_error = np.linalg.norm(
            target_position - final_hand.xpos
        )
        final_rotation = final_hand.xmat.reshape(3, 3)
        angle_error = np.arccos(np.clip(
            (np.trace(target_rotation.T @ final_rotation) - 1.0) / 2.0,
            -1.0,
            1.0,
        ))

        pose_found = (
            position_error < 0.001
            and angle_error < np.deg2rad(2.0)
        )

        print(
            f"Sample {sample + 1} | "
            f"Position error {position_error * 1000:.2f} mm | "
            f"Orientation error {np.rad2deg(angle_error):.2f} degrees | "
            f"Local pose found {pose_found}"
        )


if __name__ == "__main__":
    main()
