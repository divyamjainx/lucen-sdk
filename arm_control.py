from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml


def main():
    sdk_root = Path(__file__).resolve().parent
    config_path = sdk_root / "humanoid_arm" / "humanoid_arm.yaml"

    with config_path.open() as file:
        config = yaml.safe_load(file)

    scene = sdk_root / config["model"]["scene"]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    # A separate state lets us calculate gravity torque at zero velocity.
    gravity_data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)

    # Only control the 16 motor joints listed in the configuration.
    # The finger sliders follow their existing equality constraints.
    joints = config["joints"]
    joint_ids = [model.joint(joint["name"]).id for joint in joints]

    position_indices = model.jnt_qposadr[joint_ids]
    velocity_indices = model.jnt_dofadr[joint_ids]

    gains = np.array([
        config["gain_profiles"]["hold"][joint["motor"]]
        for joint in joints
    ])
    kp = gains[:, 0]
    kd = gains[:, 1]

    torque_limits = np.array([
        config["tau_limit"][joint["motor"]]
        for joint in joints
    ])

    ready_targets = data.qpos[position_indices].copy()
    joint_names = [joint["name"] for joint in joints]
    elbow_index = joint_names.index("r_knee_joint")

    # Calculate a reachable target using a separate robot state.
    target_data = mujoco.MjData(model)
    target_data.qpos[:] = data.qpos
    target_data.qpos[position_indices[elbow_index]] -= 0.1
    mujoco.mj_forward(model, target_data)

    target_position = target_data.site("r_ee").xpos.copy()
    print(f"Gripper target in metres: {np.round(target_position, 4)}")

    print("Holding both arms at the target pose.")
    print("Close the viewer to exit.")

    next_report_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Draw a green sphere at the target position.
        with viewer.lock():
            viewer.user_scn.ngeom = 1
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.015, 0.015, 0.015]),
                pos=target_position,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.0, 1.0, 0.0, 0.7]),
            )

        while viewer.is_running():
            step_start = time.monotonic()

            with viewer.lock():
                targets = ready_targets.copy()

                # Request the pose used to calculate the green target.
                targets[elbow_index] -= 0.1

                positions = data.qpos[position_indices]
                velocities = data.qvel[velocity_indices]

                # At zero velocity, qfrc_bias gives the gravity load.
                gravity_data.qpos[:] = data.qpos
                gravity_data.qvel[:] = 0.0
                mujoco.mj_forward(model, gravity_data)

                gravity_torque = gravity_data.qfrc_bias[velocity_indices]
                gravity_limit = (
                    config["gravity_ff"]["clip_frac"] * torque_limits
                )
                gravity_torque = np.clip(
                    gravity_torque, -gravity_limit, gravity_limit
                )

                torque = (
                    kp * (targets - positions)
                    - kd * velocities
                    + gravity_torque
                )

                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[velocity_indices] = np.clip(
                    torque, -torque_limits, torque_limits
                )

                mujoco.mj_step(model, data)

                # Refresh positions for the state produced by the physics step.
                mujoco.mj_forward(model, data)

                gripper_position = data.site("r_ee").xpos
                position_error = target_position - gripper_position
                distance = np.linalg.norm(position_error)

                # Give higher scores when the gripper is closer.
                reward = float(np.exp(-np.square(distance / 0.05)))

                # Print once per simulated second.
                if data.time >= next_report_time:
                    print(
                        f"Time {data.time:5.1f} s | "
                        f"Distance {distance * 100:6.2f} cm | "
                        f"Reward {reward:.3f}"
                    )
                    next_report_time = data.time + 1.0

            viewer.sync()

            remaining = model.opt.timestep - (time.monotonic() - step_start)
            time.sleep(max(0.0, remaining))


if __name__ == "__main__":
    main()
