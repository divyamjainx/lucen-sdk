#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mirror_viewer.py - MuJoCo "digital twin" viewer: mirror the real robot's joint angles, or drive it with sliders

Connects to bridge.py running on the robot computer. Start the bridge first, then run the viewer on your
computer (on macOS the MuJoCo viewer requires mjpython instead of python):
  python mirror_viewer.py --model biped_Laika/biped_Laika.xml --list-joints
  python mirror_viewer.py --model biped_Laika/biped_Laika.xml --host <robot computer IP> --mode watch
  python mirror_viewer.py --model biped_Laika/biped_Laika.xml --host <robot computer IP> --mode send --joints l_knee_joint
  python mirror_viewer.py --model biped_Laika/biped_Laika.xml --host <robot computer IP> --mode send     # all 12 joints
When the viewer runs on the robot computer itself (display attached), use --host 127.0.0.1.
For another robot, pass its MuJoCo model and the bridge port of its config with --port (default 9877).

watch: move the joints of the real robot by hand and the model follows. Use it to check the sign / offset
       calibration in the robot config.
send : drag the sliders of the Control panel on the right side of the viewer and the real robot follows. The
       motors are enabled, so hoist the robot first. The bridge rate-limits the targets, clamps them to the
       joint limits and disables the motors when the connection drops.
       The Joint panel stays greyed out in the passive viewer (MuJoCo only allows editing it while paused);
       do not use it.
"""

import argparse
import json
import select
import socket
import sys
import threading
import time

import mujoco
import mujoco.viewer


def load(model_path, ctrl_sliders=False):
    # MuJoCo only allows editing the Joint sliders while paused, and the passive viewer cannot pause, so they
    # are always greyed out. Send mode uses the Control panel instead: one position actuator is added per
    # 1-DoF joint. mj_step is never called, so the actuators only carry the slider values and apply no force.
    if ctrl_sliders:
        spec = mujoco.MjSpec.from_file(model_path)
        for j in spec.joints:
            if j.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                a = spec.add_actuator()
                a.name = j.name
                a.target = j.name
                a.trntype = mujoco.mjtTrn.mjTRN_JOINT
                a.ctrlrange = j.range          # autolimits -> slider range = joint limits
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(model.njnt)]
    return model, data, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="MuJoCo model of the robot, e.g. biped_Laika/biped_Laika.xml or humanoid_arm/mjcf/humanoid_arm_scene.xml")
    ap.add_argument("--host", default="127.0.0.1", help="host the bridge runs on")
    ap.add_argument("--port", type=int, default=9877, help="bridge port (9877 biped_Laika, 9879 humanoid_arm, 9880 Arm, 9881 Arm pair)")
    ap.add_argument("--mode", choices=["watch", "send"], default="watch", help="watch = display the robot's joint angles; send = drive the robot from the viewer's sliders")
    ap.add_argument("--joints", default="", help="comma-separated joint names: use only these joints (default: all 1-DoF joints)")
    ap.add_argument("--rate", type=float, default=30.0, help="viewer update rate in Hz")
    ap.add_argument("--list-joints", action="store_true", help="print the model's joint names and exit")
    args = ap.parse_args()

    model, data, all_names = load(args.model, ctrl_sliders=(args.mode == "send"))
    if args.list_joints:
        print("Model joints (use these names in the joints table of the robot config):")
        for j, n in enumerate(all_names):
            print(f"  {n:<26} range={model.jnt_range[j]}")
        return

    if args.joints:
        names = [s.strip() for s in args.joints.split(",") if s.strip()]
    else:                                   # default: all 1-DoF joints (the root free joint is excluded)
        names = [n for j, n in enumerate(all_names)
                 if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                          mujoco.mjtJoint.mjJNT_SLIDE)]
    adr, limits, cadr = {}, {}, {}
    for n in names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0:
            sys.exit(f"[err] the model has no joint '{n}'")
        adr[n] = int(model.jnt_qposadr[j])
        if model.jnt_limited[j]:
            limits[n] = [float(model.jnt_range[j][0]), float(model.jnt_range[j][1])]
        if args.mode == "send":
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if aid < 0:
                sys.exit(f"[err] joint '{n}' has no slider actuator")
            cadr[n] = aid

    sock = socket.create_connection((args.host, args.port), timeout=5)
    sock.sendall((json.dumps({"hello": {
        "mode": args.mode, "joints": names, "limits": limits}}) + "\n").encode())
    sock.setblocking(False)
    buf = b""

    # Heartbeat thread for send mode, alive for the whole session. An empty target is a pure heartbeat: the
    # bridge takes no action on it and does not end its start-up grace period. It keeps the bridge's client
    # timeout from firing while the window opens, rendering stalls or a slider is being dragged. When this
    # process exits or the link drops, the heartbeat stops and the bridge goes to damping and disables.
    send_lock = threading.Lock()

    def keepalive():
        while True:
            try:
                with send_lock:
                    sock.sendall(b'{"target": {}}\n')
            except OSError:
                return
            time.sleep(0.2)

    if args.mode == "send":
        threading.Thread(target=keepalive, daemon=True).start()

    def lines(wait=0.0):
        nonlocal buf
        out = []
        r, _, _ = select.select([sock], [], [], wait)
        if r:
            try:
                chunk = sock.recv(1 << 20)
            except (ConnectionResetError, OSError):
                sys.exit("[err] bridge disconnected")
            if not chunk:
                sys.exit("[err] bridge disconnected")
            buf += chunk
        while b"\n" in buf:
            ln, buf2 = buf.split(b"\n", 1)
            buf = buf2
            if ln.strip():
                out.append(json.loads(ln))
        return out

    # handshake replies
    t0 = time.monotonic()
    q0 = None
    while time.monotonic() - t0 < 6.0:
        for msg in lines(0.1):
            if "error" in msg:
                sys.exit(f"[err] refused by the bridge: {msg['error']}")
            if "matched" in msg:
                print(f"[hello] matched={len(msg['matched'])}", end="")
                if msg.get("unknown"):
                    print(f"  !! the bridge config has no joints with these names: {msg['unknown']}")
                else:
                    print()
            if "q0" in msg:
                q0 = msg["q0"]
        if args.mode == "watch" or q0 is not None:
            break
    if args.mode == "send":
        if not q0:                       # None and an empty dict both mean that enabling failed
            sys.exit("[err] no q0 received / enabling failed - check the bridge output on the robot computer")
        missing = [n for n in names if n not in q0]
        if missing:
            print(f"[warn] these joints failed to enable; their sliders have no effect: {missing}")
        for n, v in q0.items():
            if n in adr:
                data.qpos[adr[n]] = v
            if n in cadr:
                data.ctrl[cadr[n]] = v
        print("[send] initial pose = current pose of the real robot; drag the Control panel sliders on the right and the robot follows")
        print("       (the Joint panel cannot be edited in the passive viewer; this is how MuJoCo works)")
        print("       keyboard fallback: [ ] select a joint, - / = jog by ±0.05 rad")
    else:
        print("[watch] move the real robot's joints by hand; the model should follow in the same direction (checks sign / offset)")

    mujoco.mj_forward(model, data)
    dt = 1.0 / args.rate

    sel = {"i": 0}                    # joint selected for keyboard jogging (fallback for the Control sliders)
    order = list(cadr)

    def on_key(key):
        if args.mode != "send" or not order:
            return
        if key in (ord("["), ord("]")):
            sel["i"] = (sel["i"] + (1 if key == ord("]") else -1)) % len(order)
            n = order[sel["i"]]
            print(f"[jog] selected joint: {n}  ctrl={data.ctrl[cadr[n]]:+.3f}")
        elif key in (ord("-"), ord("=")):
            n = order[sel["i"]]
            step = 0.05 if key == ord("=") else -0.05
            lo, hi = model.actuator_ctrlrange[cadr[n]]
            data.ctrl[cadr[n]] = min(max(data.ctrl[cadr[n]] + step, lo), hi)
            print(f"[jog] {n} -> {data.ctrl[cadr[n]]:+.3f}")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        if args.mode == "send":
            print("[send] window ready, 30 Hz target stream started")
        while viewer.is_running():
            tick = time.monotonic()
            msgs = lines(0.0)
            if args.mode == "watch":
                for msg in msgs:
                    for n, v in msg.get("q", {}).items():
                        if n in adr:
                            data.qpos[adr[n]] = v
            else:
                for msg in msgs:          # send: the model shows the measured joint angles of the real robot
                    for n, v in msg.get("q", {}).items():
                        if n in adr:
                            data.qpos[adr[n]] = v
                tgt = {n: float(data.ctrl[a]) for n, a in cadr.items()}
                try:
                    with send_lock:
                        sock.sendall((json.dumps({"target": tgt}) + "\n").encode())
                except (BrokenPipeError, OSError):
                    sys.exit("[err] bridge disconnected")
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.monotonic() - tick)))


if __name__ == "__main__":
    main()
