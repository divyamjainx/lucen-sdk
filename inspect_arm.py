from pathlib import Path

import mujoco
import numpy as np
import time
import mujoco.viewer

def main():
    sdk_root = Path(__file__).resolve().parent
    scene = sdk_root / "humanoid_arm" / "mjcf" / "humanoid_arm_scene.xml"

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    # Start with both arms in Lucen's ready pose.
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)

    elbow = model.joint("r_knee_joint")
    qpos_index = int(model.jnt_qposadr[elbow.id])
    end_effector = data.site("r_ee")

    before = end_effector.xpos.copy()
    original_angle = float(data.qpos[qpos_index])

    # Bend the right elbow by another 0.2 radians.
    data.qpos[qpos_index] = original_angle - 0.2
    mujoco.mj_forward(model, data)

    after = end_effector.xpos.copy()
    displacement = np.linalg.norm(after - before)

    print(f"Elbow angle (rad)  {original_angle:.2f} -> {data.qpos[qpos_index]:.2f}")
    print(f"Gripper before (m) {np.round(before, 4)}")
    print(f"Gripper after  (m) {np.round(after, 4)}")
    print(f"Distance moved (m) {displacement:.4f}")

    # Alternate between the original and bent elbow poses.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.monotonic()

        while viewer.is_running():
            elapsed = time.monotonic() - start_time
            bent_pose = int(elapsed / 2.0) % 2 == 1

            angle_change = -0.2 if bent_pose else 0.0
            data.qpos[qpos_index] = original_angle + angle_change

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)

if __name__ == "__main__":
    main()
