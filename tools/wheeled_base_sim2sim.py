#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/wheeled_base_sim2sim.py - run a wheeled_base policy in MuJoCo (the check between training and a real robot).

The model is wheeled_base/mjcf/wheeled_base.xml with its plant parameters overridden from
wheeled_base/wheeled_base.yaml after loading (the file itself is not touched): joint armature, Coulomb friction
and viscous damping. The actuators become pure torque sources, and every physics step (2 ms) applies the
motor law of the contract
  tau = clip(kp * (q_des - q) + kd * (v_des - v), ±tau_limit)
just as the PD loop in a motor's firmware does, also at kHz rate. Observations and actions go through
tools/wheeled_base_io.py, the same contract code that a future real-robot deployment is intended to share.

Usage:
  python tools/wheeled_base_sim2sim.py --policy wheeled_base/policies/drive.onnx --cmd 0.5 0      # vx m/s, wz rad/s
  python tools/wheeled_base_sim2sim.py --policy P.onnx --segments "0:0:0, 3:0.8:0, 8:0:0, 11:0:1.5, 16:0:0"
  python tools/wheeled_base_sim2sim.py --policy P.onnx --cmd 0.5 0 --view     # live viewer window
  python tools/wheeled_base_sim2sim.py --policy zero --cmd 0 0                 # no policy: PD only, the robot falls over
  python tools/wheeled_base_sim2sim.py --policy P.onnx --cmd 0.5 0 --video out/demo   # no window; records out/demo_{oblique,side}.mp4
Each --segments entry is "start_s:vx:wz". Disturbance options: see --help. The exit code is 0 if the robot
survived and 1 if it fell. Full acceptance matrix: tools/wheeled_base_accept.py.
"""
import argparse
import json
import os
import sys
import time

if "--video" in sys.argv:                          # video is rendered offscreen; the GL backend must be chosen before mujoco is imported
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.wheeled_base_io import WheelbaseIO  # noqa: E402

FALL_TILT_RAD = 0.8          # body tilt that counts as a fall; same value as the training termination


def load_policy(path, allow_unstamped=False):
    """ONNX path -> (function obs -> raw network output (10), the WheelbaseIO that goes with the policy).

    Two properties belong to the policy and are taken from the manifest in the ONNX: the number of history
    frames, and whether a frame contains the position-hold error. Policies trained with a single frame or
    without hold_error (37 values per frame) therefore run as well. The rest of the I/O contract must match
    the current wheeled_base/wheeled_base.yaml, otherwise the program exits; a plant mismatch only prints a
    warning. An ONNX without a manifest (e.g. a checkpoint from the middle of training) is accepted only
    with allow_unstamped; the two properties are then inferred from the input size, trying the frame layout
    of the current contract first. path "zero" gives a policy that always outputs zeros.
    """
    if path == "zero":
        io = WheelbaseIO()
        return (lambda obs: np.zeros(io.act_dim, dtype=np.float32)), io
    import onnxruntime as ort

    from tools.wheeled_base_export import read, verify
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1                   # a small MLP runs fastest on one thread, and parallel acceptance workers do not compete for cores
    sess = ort.InferenceSession(path, opt, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    man = read(path)
    if man:
        io = WheelbaseIO(history_len=man["io"]["history_len"],
                         hold=any(o["name"] == "hold_error" for o in man["io"]["obs"]))
    else:
        io = None
        for hold in (True, False):
            frame = WheelbaseIO(history_len=1, hold=hold).frame_dim
            if inp.shape[-1] % frame == 0:
                io = WheelbaseIO(history_len=inp.shape[-1] // frame, hold=hold)
                break
        assert io is not None, f"ONNX input {inp.shape} is not a whole number of contract frames"
    assert inp.shape[-1] == io.obs_dim, f"ONNX input {inp.shape} does not match the manifest: {io.frame_dim} × {io.hist_len}"
    assert sess.get_outputs()[0].shape[-1] == io.act_dim, "the ONNX output does not have the 10 values of the contract"
    fatal, warn = verify(path, io)
    if allow_unstamped:
        fatal = [f for f in fatal if not f.startswith("not stamped")]
    for w in warn:
        print(f"[manifest] warning: {w}", file=sys.stderr)
    if fatal:
        sys.exit(f"{path}: " + "; ".join(fatal))
    return (lambda obs: sess.run(None, {inp.name: obs[None].astype(np.float32)})[0][0]), io


def build_model(io, friction=None, base_mass_add=0.0, base_com=(0.0, 0.0, 0.0), link_mass_scale=1.0):
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, io.cfg["model"]["mjcf"]))
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in io.names]
    assert all(j >= 0 for j in jid)
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    vadr = np.array([m.jnt_dofadr[j] for j in jid])
    m.dof_armature[vadr] = io.armature
    m.dof_frictionloss[vadr] = io.friction
    m.dof_damping[vadr] = io.damping
    # actuators -> pure torque sources (ctrl = N·m), indexed in the joint order of the contract
    aid = []
    for j in jid:
        hit = [a for a in range(m.nu) if m.actuator_trnid[a, 0] == j]
        assert len(hit) == 1, f"joint {j} must have exactly one actuator"
        aid.append(hit[0])
    aid = np.array(aid)
    m.actuator_gainprm[:] = 0.0
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_biasprm[:] = 0.0
    m.actuator_ctrllimited[:] = 0
    m.actuator_forcelimited[:] = 0
    # disturbances
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if link_mass_scale != 1.0:
        m.body_mass[1:] *= link_mass_scale
        m.body_inertia[1:] *= link_mass_scale
    if base_mass_add:
        k = (m.body_mass[base] + base_mass_add) / m.body_mass[base]
        m.body_mass[base] *= k
        m.body_inertia[base] *= k
    m.body_ipos[base] += np.asarray(base_com, dtype=np.float64)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    wheels = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_wheel_link_col") for s in "lr"]
    if friction is not None:                       # MuJoCo uses the larger value of the two geoms
        m.geom_friction[floor, 0] = friction
        for g in wheels:
            m.geom_friction[g, 0] = friction
    return m, dict(qadr=qadr, vadr=vadr, aid=aid, base=base, floor=floor, wheels=wheels)


VIDEO_VIEWS = {"oblique": (140.0, -18.0, 1.7, 0.22), "side": (90.0, -10.0, 1.7, 0.22)}   # azimuth, elevation, distance, look-at height


class VideoRecorder:
    """Offscreen recorder: one frame per control step (50 fps), piped to ffmpeg while rendering (frames are
    not kept in memory). Writes PFX_oblique.mp4 and PFX_side.mp4.

    The camera follows the robot (fixed=True keeps it at the start position). With a slope the ground stays
    level in the simulation (gravity is tilted instead), so the camera is rotated until true gravity points
    down and the picture shows an actual slope. Overlays: time / command / actual speed, the hold point while
    standing (a red dot on the ground) with the distance to it, and an arrow at the moment of a push.
    """

    def __init__(self, m, pfx, fps, g_dir, fixed=False, label=None, size=(960, 720)):
        import subprocess
        from PIL import ImageFont
        self.m, self.fps, self.fixed, self.label = m, fps, fixed, label
        self.w, self.h = size
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, self.w)
        m.vis.global_.offheight = max(m.vis.global_.offheight, self.h)
        # display-only changes (no effect on physics): draw the floor as infinite so the robot never drives
        # off its edge, remove reflections, brighten the light
        m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"), :2] = 0.0
        m.mat_reflectance[:] = 0.0
        m.vis.headlight.ambient[:] = 0.35
        m.vis.headlight.diffuse[:] = 0.7
        self.rend = mujoco.Renderer(m, height=self.h, width=self.w)
        # smallest rotation that takes the world z axis to the true "up" (-g_dir); identity on level ground
        up = -np.asarray(g_dir, dtype=np.float64)
        axis = np.cross([0.0, 0.0, 1.0], up)
        sn, cs = np.linalg.norm(axis), up[2]
        if sn < 1e-9:
            self.tilt = np.eye(3)
        else:
            k = axis / sn
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            self.tilt = np.eye(3) + sn * K + (1 - cs) * K @ K
        self.cams = {}
        for name, (az, el, dist, lz) in VIDEO_VIEWS.items():
            c = mujoco.MjvCamera()
            c.azimuth, c.elevation, c.distance = az, el, (dist * 2.0 if fixed else dist)
            c.lookat[:] = [0.0, 0.0, lz]
            self.cams[name] = c
        os.makedirs(os.path.dirname(os.path.abspath(pfx)) or ".", exist_ok=True)
        import shutil
        ff = shutil.which("ffmpeg")
        if not ff:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        self.paths = {n: f"{pfx}_{n}.mp4" for n in self.cams}
        self.procs = {n: subprocess.Popen(
            [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.w}x{self.h}",
             "-r", f"{fps:g}", "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", out],
            stdin=subprocess.PIPE) for n, out in self.paths.items()}
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
        except OSError:
            self.font = ImageFont.load_default()
        self.n = 0

    def _add_geom(self, scene, gtype, size, pos, mat, rgba):
        if scene.ngeom >= scene.maxgeom:
            return None
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
                            np.asarray(mat, dtype=np.float64).reshape(9), np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1
        return g

    def grab(self, d, lines, anchor=None, arrow=None):
        """lines: text lines drawn in the top-left corner; anchor: hold point (x, y) or None; arrow: (start xyz, end xyz) or None."""
        from PIL import Image, ImageDraw
        for name, cam in self.cams.items():
            if not self.fixed:
                cam.lookat[0], cam.lookat[1] = d.qpos[0], d.qpos[1]
            self.rend.update_scene(d, camera=cam)
            sc = self.rend.scene
            if anchor is not None:
                self._add_geom(sc, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.03, 0.002, 0.0], [anchor[0], anchor[1], 0.002],
                               np.eye(3), [0.95, 0.15, 0.15, 1.0])
            if arrow is not None:
                g = self._add_geom(sc, mujoco.mjtGeom.mjGEOM_ARROW, [0.01, 0.01, 0.01], [0, 0, 0], np.eye(3),
                                   [1.0, 0.75, 0.1, 1.0])
                if g is not None:
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, np.asarray(arrow[0], dtype=np.float64),
                                         np.asarray(arrow[1], dtype=np.float64))
            look = np.array(cam.lookat)
            for gc in sc.camera:                       # rotate both "eyes" together (mono rendering uses their average)
                gc.pos[:] = look + self.tilt @ (np.array(gc.pos) - look)
                gc.forward[:] = self.tilt @ np.array(gc.forward)
                gc.up[:] = self.tilt @ np.array(gc.up)
            img = Image.fromarray(self.rend.render())
            dr = ImageDraw.Draw(img)
            y = 12
            for ln in ([self.label] if self.label else []) + list(lines):
                dr.text((15, y), ln, font=self.font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                y += 28
            self.procs[name].stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        for n, pr in self.procs.items():
            pr.stdin.close()
            pr.wait()
            print(f"[video] {self.paths[n]}  ({self.n} frames @{self.fps:g}fps)")
        self.rend.close()


def parse_segments(text):
    segs = []
    for part in text.split(","):
        t, vx, wz = (float(x) for x in part.strip().split(":"))
        segs.append((t, vx, wz))
    assert segs and segs[0][0] == 0.0, "the first --segments entry must start at 0 s"
    return segs


def cmd_at(segs, t):
    cur = segs[0]
    for s in segs:
        if t >= s[0]:
            cur = s
    return cur[1], cur[2]


def run_episode(policy, io, segs, duration, *, seed=0, delay=0, kp_scale=1.0, kd_scale=1.0, friction=None,
                base_mass_add=0.0, base_com=(0.0, 0.0, 0.0), link_mass_scale=1.0, push=None, init_pitch=0.0,
                obs_noise=False, imu_bias=(0.0, 0.0), slope_deg=0.0, slope_dir_deg=0.0, noslip=0, view=False,
                log=None, video=None, video_fixed=False, video_label=None):
    """Run one episode and return a dict of metrics (see summarize).

    segs: command segments [(start_s, vx, wz), ...]; duration: seconds.
    push = (impulse N·s, time s, direction rad: 0 = towards the front of the robot, π/2 = towards its left).
    slope_deg: ground slope; slope_dir_deg: uphill direction in the world frame (0 = +x, which is the robot's
    initial heading; 90 = its initial left). The slope is produced by tilting the gravity vector while the
    ground stays where it is. The projected-gravity observation and the tilt angle use the true gravity
    direction, which is what an IMU reads on a real slope.
    noslip: number of MuJoCo noslip iterations. The default soft contact creeps slowly under a sustained
    tangential force (0.32 cm/s at the contact point on a 5° slope while the wheels do not turn). This is a
    property of the simulator that a real tyre does not have: enable it (20) to test whether the robot can
    hold its place on a slope, otherwise the result measures contact creep rather than the policy.
    video: file-name prefix of a recording (see VideoRecorder); video_fixed: the camera does not follow the
    robot; video_label: title drawn in the top-left corner of the picture.
    """
    io.reset()                                     # clear the observation history (io may have been used by a previous episode)
    rng = np.random.default_rng(seed)
    m, ix = build_model(io, friction, base_mass_add, base_com, link_mass_scale)
    d = mujoco.MjData(m)
    th, psi = np.radians(slope_deg), np.radians(slope_dir_deg)
    g_dir = np.array([-np.sin(th) * np.cos(psi), -np.sin(th) * np.sin(psi), -np.cos(th)])   # gravity direction (unit vector, world frame)
    m.opt.gravity[:] = 9.81 * g_dir
    m.opt.noslip_iterations = int(noslip)
    qadr, vadr, aid = ix["qadr"], ix["vadr"], ix["aid"]
    n_sub = int(round(1.0 / io.rate_hz / m.opt.timestep))
    assert abs(n_sub * m.opt.timestep * io.rate_hz - 1.0) < 1e-9, "the control period is not a whole number of physics steps"

    d.qpos[:] = 0.0
    d.qpos[2] = io.cfg["init_base_height"] + 0.002
    d.qpos[3:7] = [np.cos(init_pitch / 2), 0.0, np.sin(init_pitch / 2), 0.0]
    d.qpos[qadr[io.leg_idx]] = io.nominal
    mujoco.mj_forward(m, d)

    viewer = None
    if view:
        from mujoco import viewer as mj_viewer      # not "import mujoco.viewer": that would make mujoco a local name in this function
        viewer = mj_viewer.launch_passive(m, d)

    vid = VideoRecorder(m, video, io.rate_hz, g_dir, video_fixed, video_label) if video else None
    stand_steps, anchor, arrow_until, arrow = 0, None, -1.0, None    # used by the video only: hold point / push arrow

    last_a = np.zeros(io.act_dim)
    fifo = [np.zeros(io.act_dim) for _ in range(delay + 1)]      # fifo[0] = newest
    rot = np.zeros(9)
    rec = {k: [] for k in ("t", "cmd_vx", "cmd_wz", "vx", "vy", "wz", "pitch", "roll", "x", "y", "yaw",
                           "tau", "tau_avg", "qd", "a", "leg_dev", "wheel_contact", "slip")}
    wheel_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n.replace("_joint", "_link")) for n in io.wheel_names]
    r_wheel = io.cfg["model"]["wheel_radius_m"]
    vel6 = np.zeros(6)
    fell_at = None
    steps = int(round(duration * io.rate_hz))
    for i in range(steps):
        t = i / io.rate_hz
        cvx, cwz = cmd_at(segs, t)
        mujoco.mju_quat2Mat(rot, d.qpos[3:7])
        R = rot.reshape(3, 3)
        ang_b = d.qvel[3:6].copy()                 # the angular velocity of a free joint is in the body frame
        grav_b = R.T @ g_dir
        q, qd = d.qpos[qadr].copy(), d.qvel[vadr].copy()
        if obs_noise:
            # obs_noise: True / a number = the training amplitudes (domain_rand.obs_noise in wheeled_base/wheeled_base.yaml)
            # times that factor; a dict = the half-width of each channel, given directly
            n, k = (obs_noise, 1.0) if isinstance(obs_noise, dict) else (io.cfg["domain_rand"]["obs_noise"], float(obs_noise))
            ang_b = ang_b + rng.uniform(-1, 1, 3) * n["base_ang_vel"] * k
            grav_b = grav_b + rng.uniform(-1, 1, 3) * n["projected_gravity"] * k
            q = q + rng.uniform(-1, 1, q.shape) * n["leg_pos"] * k
            half = np.full(io.n, n["leg_vel"])
            half[io.wheel_idx] = n["wheel_vel"]
            qd = qd + rng.uniform(-1, 1, qd.shape) * half * k
        grav_b = grav_b + np.array([imu_bias[0], imu_bias[1], 0.0])     # IMU mounting / calibration bias (x / y components of projected gravity, ≈ rad)
        obs = io.build_obs(ang_b, grav_b, (cvx, 0.0, cwz), q, qd, last_a)
        a = io.clip_action(policy(obs))
        last_a = a
        fifo.insert(0, a)
        fifo.pop()
        q_des, v_des = io.action_to_target(fifo[-1])

        if push is not None and abs(t - push[1]) < 0.5 / io.rate_hz:
            mass = float(m.body_subtreemass[ix["base"]])
            yaw = np.arctan2(R[1, 0], R[0, 0])
            d.qvel[0] += push[0] / mass * np.cos(yaw + push[2])
            d.qvel[1] += push[0] / mass * np.sin(yaw + push[2])
            arrow_until = t + 0.4
            arrow = (push[0] * 0.05 * np.array([np.cos(yaw + push[2]), np.sin(yaw + push[2]), 0.0]), None)

        tau_max = np.zeros(io.n)
        tau_sum = np.zeros(io.n)
        for _ in range(n_sub):
            tau = io.torque(q_des, v_des, d.qpos[qadr], d.qvel[vadr], kp_scale, kd_scale)
            d.ctrl[aid] = tau
            mujoco.mj_step(m, d)
            tau_max = np.maximum(tau_max, np.abs(tau))
            tau_sum += tau
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(1.0 / io.rate_hz)

        mujoco.mju_quat2Mat(rot, d.qpos[3:7])
        R = rot.reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        v_w = d.qvel[0:3]
        g_b = R.T @ g_dir
        touching = set()
        bad_contact = False
        for c in d.contact[:d.ncon]:
            if ix["floor"] in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == ix["floor"] else c.geom1
                if other in ix["wheels"]:
                    touching.add(other)
                else:
                    bad_contact = True
        tilt = np.arccos(np.clip(-g_b[2], -1.0, 1.0))
        rec["t"].append(t)
        rec["cmd_vx"].append(cvx)
        rec["cmd_wz"].append(cwz)
        rec["vx"].append(cy * v_w[0] + sy * v_w[1])
        rec["vy"].append(-sy * v_w[0] + cy * v_w[1])
        rec["wz"].append((R @ d.qvel[3:6])[2])
        rec["pitch"].append(np.arctan2(-g_b[0], -g_b[2]))
        rec["roll"].append(np.arctan2(g_b[1], -g_b[2]))
        rec["x"].append(d.qpos[0])
        rec["y"].append(d.qpos[1])
        rec["yaw"].append(yaw)
        rec["tau"].append(tau_max)                 # peak |torque| within this control step
        rec["tau_avg"].append(tau_sum / n_sub)     # signed mean torque within this control step
        rec["qd"].append(d.qvel[vadr].copy())
        rec["a"].append(a)
        rec["leg_dev"].append(d.qpos[qadr[io.leg_idx]] - io.nominal)
        rec["wheel_contact"].append(len(touching))
        # longitudinal slip speed at each wheel's contact point (0 = pure rolling): velocity of the wheel centre
        # + ω × (centre -> contact point), projected on the direction the wheel is heading
        slip = []
        for b, side in zip(wheel_bodies, (1.0, -1.0)):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel6, 0)       # [angular, linear] velocity, world orientation
            axle = d.xmat[b].reshape(3, 3)[:, 1]
            rc = axle * (0.0075 * side) - np.array([0.0, 0.0, r_wheel])
            head = np.cross(axle, [0.0, 0.0, 1.0])
            head /= max(np.linalg.norm(head), 1e-9)
            slip.append(float((vel6[3:] + np.cross(vel6[:3], rc)) @ head))
        rec["slip"].append(slip)
        if vid is not None:
            mid = d.xpos[wheel_bodies].mean(axis=0)[:2]            # midpoint between the two wheels
            standing = abs(cvx) < io.hold_thresh and abs(cwz) < io.hold_thresh
            stand_steps = stand_steps + 1 if standing else 0
            if not standing:
                anchor = None
            elif anchor is None and stand_steps > io.hold_delay_steps:   # set on the same control step as the hold point in wheeled_base_io
                anchor = mid.copy()
            lines = [f"t {t:5.1f} s   cmd vx {cvx:+.2f} wz {cwz:+.2f}",
                     f"          act vx {rec['vx'][-1]:+.2f} wz {rec['wz'][-1]:+.2f}"]
            if slope_deg:
                lines.append(f"slope {slope_deg:g} deg")
            if anchor is not None:
                lines.append(f"hold point: {np.linalg.norm(mid - anchor) * 100:5.1f} cm away")
            arr = None
            if t < arrow_until:
                top = d.xpos[ix["base"]] + np.array([0.0, 0.0, 0.05])
                arr = (top - arrow[0], top)
                lines.append(f"push {push[0]:g} N.s")
            vid.grab(d, lines, anchor, arr)
        if bad_contact or tilt > FALL_TILT_RAD:
            fell_at = t
            break

    if viewer is not None:
        viewer.close()
    if vid is not None:
        vid.close()
    r = {k: np.asarray(v) for k, v in rec.items()}
    if log:
        np.savez(log, names=np.array(io.names), **r)
    return summarize(r, io, segs, fell_at, duration)


def summarize(r, io, segs, fell_at, duration):
    """Steady-state tracking for each command segment (its first 1.5 s of transient removed); attitude /
    torque / action statistics over the whole episode."""
    t = r["t"]
    out = {"survived": fell_at is None, "fell_at": fell_at, "duration": duration, "segments": []}
    bounds = [s[0] for s in segs] + [duration]
    for (t0, vx, wz), t1 in zip(segs, bounds[1:]):
        sel = (t >= min(t0 + 1.5, 0.5 * (t0 + t1))) & (t < t1)
        if not sel.any():
            continue
        seg = {"t0": t0, "cmd_vx": vx, "cmd_wz": wz,
               "vx": float(r["vx"][sel].mean()), "wz": float(r["wz"][sel].mean()),
               "vx_std": float(r["vx"][sel].std()), "wz_std": float(r["wz"][sel].std())}
        if vx == 0.0 and wz == 0.0:                # standing segment: drift = distance between its first and last position
            xy = np.stack([r["x"][sel], r["y"][sel]], axis=1)
            seg["drift_m"] = float(np.linalg.norm(xy[-1] - xy[0]))
            seg["yaw_drift_rad"] = float(np.unwrap(r["yaw"][sel])[-1] - np.unwrap(r["yaw"][sel])[0])
        out["segments"].append(seg)
    leg, wheel = io.leg_idx, io.wheel_idx
    kind = [n[2:-len("_joint")] for n in io.leg_names]
    dev = np.abs(r["leg_dev"])
    out.update({
        "pitch_max_deg": float(np.degrees(np.abs(r["pitch"]).max())),
        "roll_max_deg": float(np.degrees(np.abs(r["roll"]).max())),
        "vy_max": float(np.abs(r["vy"]).max()),
        "leg_tau_max": float(r["tau"][:, leg].max()), "wheel_tau_max": float(r["tau"][:, wheel].max()),
        "wheel_tau_rms": float(np.sqrt((r["tau"][:, wheel] ** 2).mean())),
        "wheel_speed_max": float(np.abs(r["qd"][:, wheel]).max()),
        "leg_speed_max": float(np.abs(r["qd"][:, leg]).max()),
        "a_abs_mean": float(np.abs(r["a"]).mean()),
        "a_sat_frac": float((np.abs(r["a"]) >= io.action_clip - 1e-6).mean()),
        "a_rate_rms": float(np.sqrt((np.diff(r["a"], axis=0) ** 2).mean())) if len(t) > 1 else 0.0,
        "hip_roll_dev_max": float(dev[:, [i for i, k in enumerate(kind) if k == "hip_roll"]].max()),
        "hip_yaw_dev_max": float(dev[:, [i for i, k in enumerate(kind) if k == "hip_yaw"]].max()),
        "pitch_knee_dev_max": float(dev[:, [i for i, k in enumerate(kind) if k in ("hip_pitch", "knee")]].max()),
        "wheel_air_frac": float((r["wheel_contact"] < 2).mean()),
        "slip_rms": float(np.sqrt((r["slip"] ** 2).mean())), "slip_max": float(np.abs(r["slip"]).max()),
    })
    # Wheel torque / slip in steady state (each command segment without its first 1.5 s). Two torque numbers:
    #   rms    = rms of the signed torque (includes the constant part that overcomes friction / scrubbing)
    #   jitter = rms after removing the mean of that segment and wheel: pure jitter, the part that the policy
    #            amplifies out of the observation noise
    steady = np.zeros(len(t), dtype=bool)
    dev2 = []
    for (t0, _, _), t1 in zip(segs, bounds[1:]):
        sel = (t >= min(t0 + 1.5, 0.5 * (t0 + t1))) & (t < t1)
        steady |= sel
        if sel.sum() > 1:
            tw = r["tau_avg"][sel][:, wheel]
            dev2.append((tw - tw.mean(axis=0)) ** 2)
    if steady.any():
        out["wheel_tau_rms_steady"] = float(np.sqrt((r["tau_avg"][steady][:, wheel] ** 2).mean()))
        out["wheel_tau_jitter_steady"] = float(np.sqrt(np.concatenate(dev2).mean())) if dev2 else 0.0
        out["slip_rms_steady"] = float(np.sqrt((r["slip"][steady] ** 2).mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="ONNX path, or 'zero' (no policy: all actions 0)")
    ap.add_argument("--cmd", nargs=2, type=float, default=[0.0, 0.0], metavar=("VX", "WZ"), help="velocity command: vx (m/s) wz (rad/s)")
    ap.add_argument("--segments", default=None, help='command segments "start_s:vx:wz, ..." (--cmd is ignored when given)')
    ap.add_argument("--duration", type=float, default=10.0, help="simulated time in seconds")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--delay", type=int, default=0, metavar="N",
                    help="delay the action by N control steps (domain_rand in wheeled_base.yaml trains 0 to 3)")
    ap.add_argument("--kp-scale", type=float, default=1.0, help="scale the motor kp gains")
    ap.add_argument("--kd-scale", type=float, default=1.0, help="scale the motor kd gains (including the wheel velocity loop)")
    ap.add_argument("--friction", type=float, default=None, metavar="MU", help="tyre-ground friction (default: 1.2, from the MJCF)")
    ap.add_argument("--base-mass-add", type=float, default=0.0, metavar="KG", help="extra mass on the base, kg")
    ap.add_argument("--base-com", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"), help="shift of the base centre of mass, m")
    ap.add_argument("--push", nargs=3, type=float, default=None, metavar=("N_S", "AT_S", "DIR_DEG"),
                    help="horizontal impulse (N·s), time (s), direction (deg: 0 = pushed forward, 90 = pushed to the left)")
    ap.add_argument("--init-pitch", type=float, default=0.0, metavar="RAD", help="initial body pitch (positive = nose down)")
    ap.add_argument("--obs-noise", nargs="?", type=float, const=1.0, default=0.0, metavar="K",
                    help="add observation noise at K times the training amplitude (K = 1 when omitted)")
    ap.add_argument("--slope", type=float, default=0.0, metavar="DEG", help="ground slope (uphill direction: see --slope-dir)")
    ap.add_argument("--slope-dir", type=float, default=0.0, metavar="DEG",
                    help="uphill direction: 0 = the robot's initial heading, 90 = its left")
    ap.add_argument("--noslip", type=int, default=0, metavar="N",
                    help="MuJoCo noslip iterations; use 20 to test standing on a slope (removes the creep of the soft contact)")
    ap.add_argument("--imu-bias", nargs=2, type=float, default=[0.0, 0.0], metavar=("GX", "GY"),
                    help="IMU bias: constants added to the x / y components of projected gravity (≈ pitch / roll reading error in rad)")
    ap.add_argument("--view", action="store_true", help="open a viewer window (on macOS run with mjpython)")
    ap.add_argument("--video", metavar="PFX",
                    help="record without a window: writes PFX_oblique.mp4 (front quarter view) and PFX_side.mp4 (side view), 50 fps")
    ap.add_argument("--video-fixed", action="store_true",
                    help="keep the video camera at the start position instead of following the robot (shows how far it drifts while standing)")
    ap.add_argument("--video-label", metavar="TEXT", help="title drawn in the top-left corner of the video (ASCII only)")
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="accept an ONNX without a manifest (e.g. a checkpoint from the middle of training)")
    ap.add_argument("--log", metavar="NPZ", help="save the per-step log to an .npz file")
    ap.add_argument("--json", metavar="PATH", help="write the metrics to a JSON file")
    args = ap.parse_args()

    policy, io = load_policy(args.policy, args.allow_unstamped)
    segs = parse_segments(args.segments) if args.segments else [(0.0, args.cmd[0], args.cmd[1])]
    push = (args.push[0], args.push[1], np.radians(args.push[2])) if args.push else None
    kw = dict(seed=args.seed, delay=args.delay, kp_scale=args.kp_scale, kd_scale=args.kd_scale,
              friction=args.friction, base_mass_add=args.base_mass_add, base_com=args.base_com, push=push,
              init_pitch=args.init_pitch, obs_noise=args.obs_noise, imu_bias=args.imu_bias, slope_deg=args.slope,
              slope_dir_deg=args.slope_dir, noslip=args.noslip,
              view=args.view, log=args.log, video=args.video, video_fixed=args.video_fixed,
              video_label=args.video_label)
    res = run_episode(policy, io, segs, args.duration, **kw)

    print(f"{'survived' if res['survived'] else 'fell @ %.2f s' % res['fell_at']}  ({args.duration:.0f} s)")
    for s in res["segments"]:
        line = (f"  t>={s['t0']:4.1f}  command vx {s['cmd_vx']:+.2f} wz {s['cmd_wz']:+.2f}  ->  "
                f"actual vx {s['vx']:+.3f} (±{s['vx_std']:.3f})  wz {s['wz']:+.3f} (±{s['wz_std']:.3f})")
        if "drift_m" in s:
            line += f"  drift {s['drift_m'] * 100:.1f} cm / yaw {np.degrees(s['yaw_drift_rad']):+.1f}°"
        print(line)
    print(f"  pitch max {res['pitch_max_deg']:.1f}°  roll max {res['roll_max_deg']:.1f}°  "
          f"sideways slip max {res['vy_max']:.3f} m/s  wheel off the ground {res['wheel_air_frac'] * 100:.1f}%")
    print(f"  torque max: legs {res['leg_tau_max']:.1f}  wheels {res['wheel_tau_max']:.1f} (steady-state rms "
          f"{res.get('wheel_tau_rms_steady', float('nan')):.2f}, of which jitter {res.get('wheel_tau_jitter_steady', float('nan')):.2f}) N·m   "
          f"wheel speed max {res['wheel_speed_max']:.1f} rad/s  leg speed max {res['leg_speed_max']:.1f} rad/s")
    print(f"  longitudinal wheel slip: rms {res['slip_rms']:.3f}  steady-state rms {res.get('slip_rms_steady', float('nan')):.3f}  "
          f"max {res['slip_max']:.2f} m/s")
    print(f"  action: mean |a| {res['a_abs_mean']:.3f}  at the clip {res['a_sat_frac'] * 100:.1f}%  step-to-step change rms {res['a_rate_rms']:.3f}   "
          f"leg deviation from nominal, max: hip roll {res['hip_roll_dev_max']:.3f} hip yaw {res['hip_yaw_dev_max']:.3f} "
          f"hip pitch / knee {res['pitch_knee_dev_max']:.3f} rad")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    sys.exit(0 if res["survived"] else 1)


if __name__ == "__main__":
    main()
