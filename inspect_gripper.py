import numpy as np

from reach_env import ReachEnv


def main():
    env = ReachEnv()
    model = env.model
    data = env.data

    gripper = model.joint("r_gripper_joint")
    print("Gripper motor range (rad):", gripper.range)

    for joint_name in [
        "r_jaw_right_joint",
        "r_jaw_left_joint",
    ]:
        joint = model.joint(joint_name)
        body_id = int(model.jnt_bodyid[joint.id])
        body = model.body(body_id)

        print(f"\nFinger joint: {joint_name}")
        print("Travel range (m):", joint.range)
        print("Movement axis in local frame:", joint.axis)
        print(
            "Body origin in world frame (m):",
            np.round(data.xpos[body_id], 4),
        )

        collision_names = []

        for geom_id in range(model.ngeom):
            belongs_to_finger = model.geom_bodyid[geom_id] == body_id
            can_collide = (
                model.geom_contype[geom_id] != 0
                or model.geom_conaffinity[geom_id] != 0
            )

            if belongs_to_finger and can_collide:
                collision_names.append(model.geom(geom_id).name)

        print("Collision geometries:", collision_names)

    end_effector = data.site("r_ee")

    print(
        "\nEnd-effector position (m):",
        np.round(end_effector.xpos, 4),
    )
    print(
        "End-effector orientation matrix:\n",
        np.round(end_effector.xmat.reshape(3, 3), 3),
    )


if __name__ == "__main__":
    main()
