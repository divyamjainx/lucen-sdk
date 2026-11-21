import numpy as np
import mujoco
from reach_env import ReachEnv


class GraspEnv:
    def __init__(self):
        # Reuse the mujoco simluation and joint controller.
        self.sim = ReachEnv(
            scene_path="humanoid_arm/mjcf/humanoid_arm_grasp_scene.xml"
        )

        self.model = self.sim.model
        self.data = self.sim.data
        self.control_dt = self.sim.control_dt
        self.episode_duration = 8.0

        motor_names = [
            joint["name"] for joint in self.sim.config["joints"]
        ]
        self.gripper_motor_index = motor_names.index("r_gripper_joint")

        gripper = self.model.joint("r_gripper_joint")
        self.gripper_position_index = int(gripper.qposadr[0])
        self.gripper_velocity_index = int(gripper.dofadr[0])

        cube_joint = self.model.joint("task_cube_free")
        self.cube_position_index = int(cube_joint.qposadr[0])
        self.cube_velocity_index = int(cube_joint.dofadr[0])
        self.cube_geom_id = self.model.geom("task_cube_geom").id
        self.pedestal_geom_id = self.model.geom("cube_pedestal").id
        self.finger_geom_ids = {
            self.model.geom("r_jaw_right_pad").id,
            self.model.geom("r_jaw_left_pad").id,
        }
        self.reset()

    def reset(self, seed=None):
        self.sim.reset(seed=seed)
        cube_start = self.cube_position_index
        self.initial_cube_height = float(self.data.qpos[cube_start + 2])
        self.elapsed_steps = 0
        self.hold_time = 0.0

        return self.get_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain eight finite numbers.")

        action = np.clip(action, -1.0, 1.0)

        # Map the eighth action from [-1, 1] to [0, 1.1] radians.
        gripper_target = 0.55 * (action[7] + 1.0)
        self.sim.ready_motor_targets[self.gripper_motor_index] = (
            gripper_target
        )

        # Reuse the controller for the seven arm targets.
        self.sim.step(action[:7])
        self.elapsed_steps += 1

        touching_fingers = set()
        pedestal_contact = False

        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}

            if self.cube_geom_id in pair:
                force = np.zeros(6)
                mujoco.mj_contactForce(
                    self.model, self.data, index, force
                )

                if force[0] > 0.01:
                    touching_fingers.update(pair & self.finger_geom_ids)
                    pedestal_contact |= self.pedestal_geom_id in pair

        observation = self.get_observation()

        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("Simulation produced invalid values.")

        height_gain = float(observation[-2])
        hand_distance = float(np.linalg.norm(observation[16:19]))
        bilateral_contact = len(touching_fingers) == 2

        lifted = (
            height_gain > 0.02
            and bilateral_contact
            and not pedestal_contact
        )

        if lifted:
            self.hold_time += self.control_dt
        else:
            self.hold_time = 0.0

        # Combine proximity, grasping, and supported lifting.
        reach_reward = np.exp(-np.square(hand_distance / 0.05))
        lift_progress = np.clip(height_gain / 0.04, 0.0, 1.0)
        supported_lift = bilateral_contact and not pedestal_contact

        reward = float(
            0.25 * reach_reward
            + 0.5 * bilateral_contact
            + 3.0 * lift_progress * supported_lift
            + 1.0 * lifted
        )

        terminated = height_gain < -0.10
        truncated = (
            self.elapsed_steps * self.control_dt >= self.episode_duration
        )

        info = {
            "distance": hand_distance,
            "height_gain": height_gain,
            "finger_contacts": len(touching_fingers),
            "pedestal_contact": pedestal_contact,
            "hold_time": self.hold_time,
            "success": self.hold_time >= 1.0,
        }

        return observation, reward, terminated, truncated, info

    def get_observation(self):
        arm_positions = self.data.qpos[self.sim.position_indices]
        arm_velocities = self.data.qvel[self.sim.velocity_indices]

        gripper_state = np.array([
            self.data.qpos[self.gripper_position_index],
            self.data.qvel[self.gripper_velocity_index],
        ])

        cube_start = self.cube_position_index
        cube_position = self.data.qpos[cube_start:cube_start + 3]
        cube_orientation = self.data.qpos[cube_start + 3:cube_start + 7]

        velocity_start = self.cube_velocity_index
        cube_velocity = self.data.qvel[velocity_start:velocity_start + 6]

        hand_position = self.data.site("r_ee").xpos
        relative_position = cube_position - hand_position
        height_gain = cube_position[2] - self.initial_cube_height

        remaining_time = max(
            0.0,
            1.0 - self.elapsed_steps * self.control_dt / self.episode_duration,
        )

        return np.concatenate([
            arm_positions,
            arm_velocities,
            gripper_state,
            relative_position,
            cube_orientation,
            cube_velocity,
            [height_gain, remaining_time],
        ]).astype(np.float32)


if __name__ == "__main__":
    env = GraspEnv()
    elbow_index = env.sim.joint_names.index("r_knee_joint")
    done = False

    while not done:
        simulation_time = env.elapsed_steps * env.control_dt

        close_progress = np.clip(
            (simulation_time - 1.0) / 2.0, 0.0, 1.0
        )
        lift_progress = np.clip(
            (simulation_time - 4.0) / 2.0, 0.0, 1.0
        )

        action = np.zeros(8)
        action[elbow_index] = -0.75 * lift_progress
        action[7] = 2.0 * close_progress - 1.0

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print(f"Final lift: {info['height_gain'] * 1000:.1f} mm")
    print(f"Finger contacts: {info['finger_contacts']}/2")
    print(f"Consecutive hold: {info['hold_time']:.2f} seconds")
    print(f"Success: {info['success']}")
    print(f"Final reward: {reward:.3f}")
