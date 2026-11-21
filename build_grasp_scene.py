from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from reach_env import ReachEnv


def main():
    env = ReachEnv()
    sdk_root = Path(__file__).resolve().parent

    source = sdk_root / env.config["model"]["scene"]
    destination = source.with_name("humanoid_arm_grasp_scene.xml")

    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find("worldbody")

    # Use flat contact pads in this experimental grasp scene.
    hand_position = env.data.site("r_ee").xpos.copy()
    hand_rotation = env.data.site("r_ee").xmat.reshape(3, 3).copy()

    finger_settings = [
        ("r_jaw_right", "r_jaw_right_col", -1.0),
        ("r_jaw_left", "r_jaw_left_col", 1.0),
    ]

    for body_name, collision_name, side in finger_settings:
        body_id = env.model.body(body_name).id
        body_position = env.data.xpos[body_id]
        body_rotation = env.data.xmat[body_id].reshape(3, 3)

        # Disable the original finger collision mesh.
        original_geom = root.find(
            f".//geom[@name='{collision_name}']"
        )
        original_geom.set("contype", "0")
        original_geom.set("conaffinity", "0")

        # Place a pad 4 cm from the gripper centre along its closing axis.
        world_position = (
            hand_position
            + hand_rotation @ np.array([side * 0.04, 0.0, 0.0])
        )

        # Convert the pad pose into the moving finger body's frame.
        local_position = body_rotation.T @ (
            world_position - body_position
        )
        local_rotation = body_rotation.T @ hand_rotation

        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, local_rotation.flatten())

        finger_body = root.find(f".//body[@name='{body_name}']")

        ET.SubElement(
            finger_body,
            "geom",
            name=f"{body_name}_pad",
            type="box",
            pos=" ".join(map(str, local_position)),
            quat=" ".join(map(str, quaternion)),
            size="0.004 0.018 0.020",
            rgba="0.2 0.7 0.3 1",
            contype="1",
            conaffinity="1",
            friction="1.0 0.005 0.0001",
            condim="4",
        )
    cube_half_size = 0.015
    cube_position = env.data.site("r_ee").xpos.copy()
    position_text = " ".join(str(value) for value in cube_position)

    # A narrow pedestal supports the cube while leaving room for the fingers.
    pedestal_top = cube_position[2] - cube_half_size

    ET.SubElement(
        world,
        "geom",
        name="cube_pedestal",
        type="box",
        pos=f"{cube_position[0]} {cube_position[1]} {pedestal_top / 2}",
        size=f"0.01 0.01 {pedestal_top / 2}",
        rgba="0.4 0.4 0.5 1",
        contype="1",
        conaffinity="1",
        friction="1.0 0.005 0.0001",
    )

    cube_body = ET.SubElement(
        world,
        "body",
        name="task_cube",
        pos=position_text,
    )
    ET.SubElement(cube_body, "freejoint", name="task_cube_free")

    # Explicit mass and inertia for a 30 g cube with 3 cm sides.
    ET.SubElement(
        cube_body,
        "inertial",
        pos="0 0 0",
        mass="0.03",
        diaginertia="0.0000045 0.0000045 0.0000045",
    )

    ET.SubElement(
        cube_body,
        "geom",
        name="task_cube_geom",
        type="box",
        size="0.015 0.015 0.015",
        mass="0.03",
        rgba="0.95 0.55 0.1 1",
        contype="1",
        conaffinity="1",
        friction="1.0 0.005 0.0001",
        condim="4",
    )

    # A free joint adds seven position values and six velocity values.
    for key in root.findall("./keyframe/key"):
        old_positions = key.get("qpos", "")
        key.set("qpos", f"{old_positions} {position_text} 1 0 0 0")

        if "qvel" in key.attrib:
            key.set("qvel", key.get("qvel") + " 0 0 0 0 0 0")

    # Keep the generated scene beside its source so mesh paths still resolve.
    tree.write(destination, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(destination))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)

    cube_geom_id = model.geom("task_cube_geom").id

    print(f"Generated scene: {destination}")
    print("Cube centre (m):", np.round(cube_position, 4))

    for index in range(data.ncon):
        contact = data.contact[index]

        if cube_geom_id in (contact.geom1, contact.geom2):
            first = model.geom(int(contact.geom1)).name
            second = model.geom(int(contact.geom2)).name
            print(
                f"Cube contact: {first} / {second}, "
                f"distance {contact.dist * 1000:.3f} mm"
            )

    # Preview the geometry with the arm frozen in its ready pose.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = cube_position
        viewer.cam.distance = 0.5

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
