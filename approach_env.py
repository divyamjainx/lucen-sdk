import mujoco
import numpy as np

from grasp_env import GraspEnv


class ApproachEnv(GraspEnv):
    def __init__(self, start_elbow_offset=0.18):
        self.start_elbow_offset = float(start_elbow_offset)
        self.rng = np.random.default_rng(42)
        super().__init__()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        super().reset(seed=seed)

        # Choose a grasp configuration close to the ready pose.
        goal_data = mujoco.MjData(self.model)
        goal_data.qpos[:] = self.data.qpos

        goal_offsets = np.zeros(7)

        for joint_name in [
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_knee_joint",
        ]:
            index = self.sim.joint_names.index(joint_name)
            goal_offsets[index] = self.rng.uniform(-0.02, 0.02)

        goal_data.qpos[self.sim.position_indices] += goal_offsets
        mujoco.mj_forward(self.model, goal_data)

        cube_position = goal_data.site("r_ee").xpos.copy()

        # Keep these known targets for a scripted approach check.
        self.goal_joint_positions = goal_data.qpos[
            self.sim.position_indices
        ].copy()

        cube_start = self.cube_position_index
        self.data.qpos[cube_start:cube_start + 3] = cube_position
        self.data.qpos[cube_start + 3:cube_start + 7] = [1, 0, 0, 0]

        # Start with the elbow bent farther, moving the hand away.
        elbow_index = self.sim.joint_names.index("r_knee_joint")
        elbow_address = self.sim.position_indices[elbow_index]
        self.data.qpos[elbow_address] -= self.start_elbow_offset

        self.data.qvel[:] = 0.0

        pedestal_id = self.model.geom("cube_pedestal").id
        pedestal_half_height = self.model.geom_size[pedestal_id, 2]

        self.model.geom_pos[pedestal_id] = [
            cube_position[0],
            cube_position[1],
            cube_position[2] - 0.015 - pedestal_half_height,
        ]

        self.initial_cube_height = float(cube_position[2])
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation()


if __name__ == "__main__":
    env = ApproachEnv()

    for episode in range(5):
        observation = env.reset()
        distance = np.linalg.norm(observation[16:19])

        penetrating_pairs = []

        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}

            if env.cube_geom_id in pair and contact.dist < -0.0001:
                penetrating_pairs.append((
                    env.model.geom(int(contact.geom1)).name,
                    env.model.geom(int(contact.geom2)).name,
                ))

        print(
            f"Episode {episode + 1} | "
            f"Hand-to-cube distance {distance * 100:.2f} cm | "
            f"Initial cube overlaps {penetrating_pairs}"
        )
