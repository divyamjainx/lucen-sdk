from pathlib import Path

import mujoco
import numpy as np
import yaml


class ReachEnv:
    def __init__(self, scene_path=None):
        sdk_root = Path(__file__).resolve().parent
        config_path = sdk_root / "humanoid_arm" / "humanoid_arm.yaml"

        with config_path.open() as file:
            self.config = yaml.safe_load(file)

        scene = sdk_root / (
            scene_path or self.config["model"]["scene"]
        )
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)

        # Preserve the configuration's order for the seven right-arm joints.
        right_joints = [
            joint for joint in self.config["joints"]
            if joint["name"].startswith("r_")
            and joint.get("kind") != "gripper"
        ]
        self.joint_names = [joint["name"] for joint in right_joints]
        joint_ids = [
            self.model.joint(name).id for name in self.joint_names
        ]

        self.position_indices = self.model.jnt_qposadr[joint_ids]
        self.velocity_indices = self.model.jnt_dofadr[joint_ids]
        self.target_data = mujoco.MjData(self.model)

        # Controller settings for all 16 motors, including the left arm.
        motors = self.config["joints"]
        motor_ids = [
            self.model.joint(joint["name"]).id for joint in motors
        ]
        self.motor_position_indices = self.model.jnt_qposadr[motor_ids]
        self.motor_velocity_indices = self.model.jnt_dofadr[motor_ids]

        motor_names = [joint["name"] for joint in motors]
        self.right_motor_indices = [
            motor_names.index(name) for name in self.joint_names
        ]

        gains = np.array([
            self.config["gain_profiles"]["hold"][joint["motor"]]
            for joint in motors
        ])
        self.kp = gains[:, 0]
        self.kd = gains[:, 1]
        self.torque_limits = np.array([
            self.config["tau_limit"][joint["motor"]]
            for joint in motors
        ])

        self.gravity_data = mujoco.MjData(self.model)
        self.action_scale = 0.2

        # Hold each policy action for approximately 20 ms.
        self.physics_steps = max(
            1, round(0.02 / self.model.opt.timestep)
        )
        self.control_dt = self.physics_steps * self.model.opt.timestep
        self.episode_duration = 5.0
        self.rng = np.random.default_rng(42)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Start every attempt from the ready pose.
        ready_id = self.model.key("ready").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, ready_id)
        mujoco.mj_forward(self.model, self.data)

        # Generate a target by varying three joints around the ready pose.
        self.target_data.qpos[:] = self.data.qpos

        target_ranges = {
            "r_hip_pitch_joint": 0.10,
            "r_hip_roll_joint": 0.10,
            "r_knee_joint": 0.15,
        }

        for joint_name, offset_limit in target_ranges.items():
            joint_index = self.joint_names.index(joint_name)
            position_address = self.position_indices[joint_index]

            offset = self.rng.uniform(-offset_limit, offset_limit)
            self.target_data.qpos[position_address] += offset

        mujoco.mj_forward(self.model, self.target_data)
        self.target_position = self.target_data.site("r_ee").xpos.copy()
        self.ready_motor_targets = self.data.qpos[
            self.motor_position_indices
        ].copy()
        self.elapsed_steps = 0
        return self.get_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain seven finite numbers.")

        action = np.clip(action, -1.0, 1.0)

        targets = self.ready_motor_targets.copy()
        targets[self.right_motor_indices] += self.action_scale * action

        joint_ids = [
            self.model.joint(joint["name"]).id
            for joint in self.config["joints"]
        ]
        limits = self.model.jnt_range[joint_ids]
        targets = np.clip(targets, limits[:, 0], limits[:, 1])

        for _ in range(self.physics_steps):
            positions = self.data.qpos[self.motor_position_indices]
            velocities = self.data.qvel[self.motor_velocity_indices]

            self.gravity_data.qpos[:] = self.data.qpos
            self.gravity_data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.gravity_data)

            gravity_torque = self.gravity_data.qfrc_bias[
                self.motor_velocity_indices
            ]
            gravity_limit = (
                self.config["gravity_ff"]["clip_frac"] * self.torque_limits
            )
            gravity_torque = np.clip(
                gravity_torque, -gravity_limit, gravity_limit
            )

            torque = (
                self.kp * (targets - positions)
                - self.kd * velocities
                + gravity_torque
            )

            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.motor_velocity_indices] = np.clip(
                torque, -self.torque_limits, self.torque_limits
            )
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps += 1

        observation = self.get_observation()
        distance = float(np.linalg.norm(observation[-3:]))
        reward = float(np.exp(-np.square(distance / 0.05)))
        truncated = (
            self.elapsed_steps * self.control_dt >= self.episode_duration
        )

        info = {"distance": distance}
        return observation, reward, False, truncated, info

    def get_observation(self):
        positions = self.data.qpos[self.position_indices]
        velocities = self.data.qvel[self.velocity_indices]
        gripper_position = self.data.site("r_ee").xpos
        target_error = self.target_position - gripper_position

        return np.concatenate(
            [positions, velocities, target_error]
        ).astype(np.float32)


if __name__ == "__main__":
    env = ReachEnv()
    target_positions = []

    for episode in range(10):
        observation = env.reset()
        target_positions.append(env.target_position.copy())

        print(
            f"Episode {episode + 1:2d} | "
            f"Target {np.round(env.target_position, 4)} | "
            f"Distance {np.linalg.norm(observation[-3:]) * 100:.2f} cm"
        )

    target_positions = np.array(target_positions)
    coordinate_spread = np.ptp(target_positions, axis=0)

    print(
        "Sampled spread along x, y, z (cm):",
        np.round(coordinate_spread * 100, 2),
    )
