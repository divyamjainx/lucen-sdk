from pathlib import Path
import time

import mujoco
import mujoco.viewer


def main():
    sdk_root = Path(__file__).resolve().parent
    scene = sdk_root / "humanoid_arm" / "mjcf" / "humanoid_arm_scene.xml"

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    # Initialize the robot in the ready pose
    mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
    mujoco.mj_forward(model, data)

    print(f"Physics timestep: {model.opt.timestep:.4f} seconds")
    print("Watch how the arms move under gravity.")
    print("Press Backspace to reset the experiment. Close the window to exit.")

    def handle_key(key):
        if key == 259:  # GLFW's key code for Backspace
            mujoco.mj_resetDataKeyframe(model, data, model.key("ready").id)
            mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=handle_key
    ) as viewer:
        while viewer.is_running():
            step_start = time.monotonic()

            # Advance the robot's state by one physics timestep.
            with viewer.lock():
                mujoco.mj_step(model, data)

            viewer.sync()

            # Pace the display approximately to real time.
            remaining = model.opt.timestep - (time.monotonic() - step_start)
            time.sleep(max(0.0, remaining))


if __name__ == "__main__":
    main()
