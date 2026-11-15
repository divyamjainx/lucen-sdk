#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/sim2sim.py - run a biped_Laika policy (ONNX) in MuJoCo and report how it behaves, before it goes on the real robot.

The observation / action contract, the gains and the nominal pose all come from biped_Laika/biped_Laika.yaml,
the same file the real-robot tools read. The control chain is modelled as on the robot: the policy produces
targets at control_rate_hz, the targets pass through the bridge speed limit (slew.py), and the joint PD is
evaluated at every physics substep, as the motor firmware does.

Usage (run from the SDK root):
  python tools/sim2sim.py --policy biped_Laika/policies/stand.onnx               # stand for 10 s, headless, print metrics
  python tools/sim2sim.py --policy biped_Laika/policies/walk.onnx --power-scale 0.8 --cmd 0.2 0 0   # velocity command: vx vy (m/s) wz (rad/s)
  python tools/sim2sim.py --policy zero                                          # zero action = PD hold at the nominal pose (pipeline check)
  mjpython tools/sim2sim.py --policy ... --view                                  # watch in the viewer (macOS: use mjpython)

Options that bring the run closer to the real robot or probe robustness (see --help for the full list):
  --delay N                     execution delay of N control steps (1-2 on the real robot)
  --jitter RAD                  random offset on the initial joint angles (0.05 is realistic)
  --kp-scale K / --kd-scale K   scale the PD stiffness / damping
  --friction MU                 sliding friction of the ground contact
  --push N_S AT_S               horizontal impulse of N_S (N*s) at time AT_S (s)
  --power-scale K               scale the policy output, as deploy_policy --power-scale does

Reading the output: the first line says whether the robot survived (upright at the end of the run) or fell; the
exit code is 0 or 1 accordingly. The lines below it report base height and tilt, actual speed against the
command, world displacement and net yaw, foot clearance per swing, foot spacing and leg-leg contact, stance-foot
yaw slip, touchdown force relative to body weight, action saturation, and the peak joint torque relative to the
torque limit. Most gait statistics skip the start-up transient (the first second or the first third of the run).
--json PATH writes the same numbers to a file, --log CSV records every control step, --video PFX renders the run.
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_policy(path, obs_dim, act_dim):
    if path == "zero":
        return lambda obs: np.zeros(act_dim, dtype=np.float32)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    assert ishape[-1] == obs_dim, f"policy input shape {ishape} != contract obs_dim {obs_dim}"

    def run(obs):
        out = sess.run(None, {iname: obs.reshape(1, -1).astype(np.float32)})[0]
        return np.asarray(out).reshape(-1)[:act_dim]
    return run


# Off-screen camera views for --video: (azimuth, elevation, distance, look-at height). The video is
# rendered from the same rollout that produces the metrics, so it includes --delay / --push.
VIDEO_VIEWS = {"side": (90.0, -5.0, 0.95, 0.20),
               "front": (180.0, -5.0, 0.95, 0.20),
               "feet": (75.0, -12.0, 0.55, 0.06)}
LEG_TINT = {"r_": (0.85, 0.25, 0.20, 1.0), "l_": (0.20, 0.45, 0.85, 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="path to the policy ONNX, or 'zero' (zero action = PD hold at the nominal pose)")
    ap.add_argument("--model", default=os.path.join(ROOT, "biped_Laika", "biped_Laika.xml"), help="MuJoCo model file (default: the biped_Laika model)")
    ap.add_argument("--profile", default=None,
                    help="PD gain profile (gain_profiles in the robot config). Default: the gain profile declared "
                         "by the policy profile, otherwise rl_default")
    ap.add_argument("--policy-profile", default=None,
                    help="policy I/O contract profile (policy_profiles in biped_Laika/biped_Laika.yaml), e.g. walk. "
                         "A stamped ONNX carries its profile, so this can be omitted (if given, it must match the "
                         "manifest); an ONNX without a manifest needs this flag together with --allow-unstamped")
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="accept an ONNX without a manifest (the contract then rests entirely on the flags you "
                         "pass; at your own risk)")
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="velocity command: vx vy (m/s) wz (rad/s)")
    ap.add_argument("--duration", type=float, default=10.0, help="simulated time in seconds")
    ap.add_argument("--view", action="store_true", help="open a viewer window (on macOS run with mjpython)")
    ap.add_argument("--jitter", type=float, default=0.0, metavar="RAD",
                    help="random offset on the initial joint angles (rad). The default 0 starts exactly at the "
                         "nominal pose, a perfectly symmetric state that the real robot never starts from. "
                         "Use 0.05 to reproduce real-robot conditions")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --jitter and other randomised options")
    ap.add_argument("--friction", type=float, default=None, metavar="MU",
                    help="override the sliding friction coefficient of all geoms (model default 1.0). "
                         "Use it to sweep the sensitivity to ground friction")
    ap.add_argument("--torsion", type=float, default=None, metavar="MU_T",
                    help="override the torsional friction coefficient (model default 0.005; unit m: maximum torque = "
                         "MU_T x normal force). Only effective with --condim 4 or 6. With condim 3 a foot resting on "
                         "a single contact point has no torsional resistance and can spin freely; a 170 x 80 mm sole "
                         "in flat contact at a friction of about 0.8 corresponds to roughly 0.03")
    ap.add_argument("--stiff-contact", action="store_true",
                    help="override all contacts with a stiff ground (o_solref 0.006/1 = 3 x timestep, o_solimp 0.95/0.99). "
                         "With the model's default contact the standing robot sinks 9-12 mm into the floor; with this "
                         "option about 0.3 mm")
    ap.add_argument("--action-ema", type=float, default=None, metavar="ALPHA",
                    help="diagnostic: first-order low-pass on the raw policy action, a_f = ALPHA*a + (1-ALPHA)*a_f "
                         "(ALPHA = 1 disables it). Shows how much the behaviour depends on high-frequency action content")
    ap.add_argument("--foot-box", action="store_true",
                    help="replace the foot collision meshes with a box (foot_geometry.FOOT_BOX: about 170 x 80 mm, sole at "
                         "z = -59 mm in the foot frame); visual meshes and inertia are unchanged. On flat ground the box "
                         "gives four stable corner contacts, whereas the mesh hull often yields a single contact point, "
                         "which makes the normal force noisy when the robot stands on one foot")
    ap.add_argument("--hard-friction", action="store_true",
                    help="harder friction model: elliptic friction cone with impratio 10. MuJoCo's default (pyramidal "
                         "cone, impratio 1) treats friction as a soft constraint that creeps under sustained tangential "
                         "or torsional load. Independent of --stiff-contact")
    ap.add_argument("--noslip", type=int, default=None, metavar="N",
                    help="number of MuJoCo noslip iterations. Soft contacts creep under sustained tangential force; "
                         "enable this to tell real foot slip from solver creep")
    ap.add_argument("--condim", type=int, default=None, choices=[1, 3, 4, 6],
                    help="contact dimension. The model uses 3 everywhere (sliding friction only); "
                         "4 adds torsional friction, 6 adds rolling friction as well")
    ap.add_argument("--heading", action="store_true",
                    help="hold the initial heading: the yaw-rate command is computed from the heading error, "
                         "wz = clip(k * err, ±0.6). This outer P loop is added by the tool and is not part of the "
                         "policy; it replaces the wz given with --cmd")
    ap.add_argument("--heading-k", type=float, default=0.5,
                    help="gain k of the --heading loop")
    ap.add_argument("--cycle-time", type=float, default=None, metavar="S",
                    help="override the gait clock period in seconds (cycle_time_s of the policy profile in "
                         "biped_Laika/biped_Laika.yaml). Larger = slower stepping. A policy is trained with the "
                         "configured period and any other value is out of distribution: sweep it in simulation "
                         "before trying it on the real robot")
    ap.add_argument("--kp-scale", type=float, default=1.0, metavar="K",
                    help="multiply the PD stiffness by K. On the real robot kp / kd are sent with every motor "
                         "command, so a change takes effect without retraining (softer = lighter touchdown, "
                         "smaller reaction torque)")
    ap.add_argument("--kd-scale", type=float, default=1.0, metavar="K",
                    help="multiply the PD damping by K")
    ap.add_argument("--push", nargs=2, type=float, default=None, metavar=("N_S", "AT_S"),
                    help="apply a horizontal impulse of N_S (N*s) to the robot at time AT_S (s) to test disturbance "
                         "rejection. Recovering from a push, not steady walking, is what decides how much torque "
                         "margin a policy needs")
    ap.add_argument("--push-dir", type=float, default=90.0, metavar="DEG",
                    help="direction of --push in the world horizontal plane (degrees). Default 90 = +y (sideways)")
    ap.add_argument("--push-torso", nargs=4, type=float, action="append", default=None,
                    metavar=("N_S", "DUR_S", "AT_S", "DEG"),
                    help="force pulse on the torso: from AT_S, a constant force N_S / DUR_S acts on base_link for "
                         "DUR_S seconds in direction DEG (world horizontal plane). May be given several times. Closer "
                         "to a person pushing the torso than --push, which changes the velocity of the whole robot")
    ap.add_argument("--hand", nargs=5, type=float, default=None, metavar=("V", "AT_S", "DUR_S", "DEG", "FMAX"),
                    help="sustained push by a hand: from AT_S for DUR_S seconds the hand advances at V m/s along DEG "
                         "and presses on the torso through a 200 N/m spring and a 20 N*s/m damper; it pushes but never "
                         "pulls, and the force is capped at FMAX N. If the robot walks along, the force drops to "
                         "about zero. Any push option adds the [balance] summary line and the 'balance' group in --json")
    ap.add_argument("--suspend", action="store_true",
                    help="hoisted: pin the base pose to its initial value at every step (rigid hoist), feet off the "
                         "ground. Reproduces watching the legs of a hoisted robot; without ground contact the "
                         "policy effectively runs open-loop")
    ap.add_argument("--suspend-pivot", action="store_true",
                    help="hoisted from a single point: the base position is pinned but the torso is free to rotate "
                         "about the hoist point. Closer to a real hoist than --suspend: leg reaction torques swing "
                         "the torso, the IMU measures it and the policy reacts")
    ap.add_argument("--power-scale", type=float, default=1.0, metavar="K",
                    help="multiply the clipped policy output by K before it becomes a joint target, in the same "
                         "order and with the same meaning as deploy_policy --power-scale (clip -> x K -> fed back as "
                         "the last action). K changes the motion amplitude, not the step frequency, which is fixed "
                         "by the gait clock")
    ap.add_argument("--tau-scale", type=float, default=1.0, metavar="K",
                    help="multiply the torque limits by K. Use it to test whether actuators weaker than the "
                         "simulated ones would change the gait: if the real robot saturates where the simulation "
                         "does not, the two diverge")
    ap.add_argument("--delay", type=int, default=0, metavar="N",
                    help="execution delay of N control steps. On the real robot a target written by the deploy "
                         "tool is picked up by the bridge on its next cycle and then travels over CAN, which adds "
                         "up to 1-2 control steps. The default is 0; use --delay 1 or 2 when you reproduce "
                         "real-robot behaviour")
    ap.add_argument("--delay-jitter", nargs=2, type=int, default=None, metavar=("LO", "HI"),
                    help="time-varying execution delay: every control step draws a whole number of steps uniformly "
                         "from {LO..HI} (reproducible with --seed). The delay of the real chain varies between 1 and "
                         "2 steps, and a constant --delay N does not reveal sensitivity to that variation. "
                         "Mutually exclusive with --delay")
    ap.add_argument("--video", metavar="PFX", default=None,
                    help="off-screen video of this rollout (the same simulation that produces the metrics, including "
                         "--delay / --push): PFX_{side,front,feet,side_r,oblique}.mp4 at normal speed (--video-fps) "
                         "plus PFX_side_r_slow4x.mp4 in 4x slow motion. Right leg red, left leg blue, the camera "
                         "follows the base. Uses ffmpeg from PATH and falls back to imageio_ffmpeg")
    ap.add_argument("--no-slew", action="store_true",
                    help="disable the model of the bridge speed limit. It is ON by default: on the real robot every "
                         "command passes through this limiter, so switching it off simulates a robot that does not "
                         "exist (see slew.py)")
    ap.add_argument("--slew-spd", default=None, metavar="M=V,...",
                    help="override the bridge speed limits used under policy control (rad/s) per motor model, e.g. "
                         "RS06=4,RS02=3,RS00=2. A probe option like --tau-scale; permanent values belong in "
                         "biped_Laika/biped_Laika.yaml")
    ap.add_argument("--video-fps", type=int, default=50,
                    help="frame rate of the normal-speed --video files (rendered at 100 Hz and decimated; the "
                         "slow-motion file is always 4x slower)")
    ap.add_argument("--log", metavar="CSV",
                    help="record every control step to CSV and print a per-joint summary, in the same format as "
                         "deploy_policy --log (for comparison with the real robot)")
    ap.add_argument("--grf-dump", metavar="NPZ",
                    help="record the normal force, vertical foot velocity and sole height of both feet at every "
                         "physics substep (2 ms) to an npz file. For touchdown-force diagnosis: sampling at the "
                         "control step (20 ms) cannot separate impact spikes from load transfer")
    ap.add_argument("--json", metavar="PATH",
                    help="write the metrics of this rollout to a JSON file (the same numbers as the terminal "
                         "output), e.g. for multi-seed evaluation scripts")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "biped_Laika/biped_Laika.yaml")))
    import gains
    from tools.onnx_manifest import enforce
    from tools.policy_io import PolicyIO
    # Manifest check: a stamped ONNX carries its own profile; a conflict with the flags is refused.
    pprof = enforce(args.policy, cfg, args.policy_profile,
                    allow_unstamped=args.allow_unstamped)
    io = PolicyIO(cfg, pprof)                           # the single implementation of the observation / action contract
    # Gain profile: without --profile, follow the gain profile declared by the policy profile, else rl_default.
    # An explicit --profile that differs only warns here (this tool allows sweeps); deploy_policy refuses it.
    if args.profile is None:
        args.profile = io.gain_profile or "rl_default"
        if io.gain_profile:
            print(f"[sim] policy profile {pprof!r} declares gain profile {io.gain_profile} -> used as --profile")
    elif io.gain_profile and args.profile != io.gain_profile:
        print(f"[warn] policy profile {pprof!r} declares gain profile {io.gain_profile}, but --profile is {args.profile}: "
              f"this run does not match the policy contract")
    g = gains.profile(args.profile)
    if (not np.allclose(io.action_scale_vec, io.action_scale)
            and abs(args.power_scale - 1.0) > 1e-6):
        # This tool allows amplitude sweeps and only warns; deploy_policy refuses this combination.
        print(f"[warn] profile {pprof!r} already declares per-joint action scales; --power-scale "
              f"{args.power_scale} attenuates the action a second time (deploy with 1.0)")
    if args.cycle_time:
        print(f"[sim] gait clock period "
              f"{io.cycle_time:.3f}s -> {args.cycle_time:.3f}s "
              f"({1/io.cycle_time:.2f} -> {1/args.cycle_time:.2f} Hz)")
        io.cycle_time = args.cycle_time
    names = io.names

    if args.video or args.foot_box:
        _sp = mujoco.MjSpec.from_file(args.model)
        if args.video:
            _sp.visual.global_.offwidth = 960
            _sp.visual.global_.offheight = 720
    if args.foot_box:
        from tools import foot_geometry as _KRfb
        _fb = np.array(_KRfb.FOOT_BOX)
        for _fn in ("l_ankle_roll_link", "r_ankle_roll_link"):
            _fbody = _sp.body(_fn)
            for _g in _fbody.geoms:
                if _g.contype != 0 or _g.conaffinity != 0:
                    _g.contype, _g.conaffinity = 0, 0
            _fbody.add_geom(name=f"{_fn}_solebox", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=list((_fb[:, 1] - _fb[:, 0]) / 2), pos=list((_fb[:, 1] + _fb[:, 0]) / 2),
                            contype=1, conaffinity=1, density=0.0, group=3, rgba=[0, 0, 0, 0])
        print("[sim] foot collision = box (foot_geometry.FOOT_BOX), mesh collision off")
    if args.video or args.foot_box:
        m = _sp.compile()
    else:
        m = mujoco.MjModel.from_xml_path(args.model)
    if args.video:
        # right leg red / left leg blue
        for _gi in range(m.ngeom):
            _bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[_gi])) or ""
            if _bn.startswith("r_"):
                m.geom_rgba[_gi] = (0.85, 0.25, 0.20, 1.0)
            elif _bn.startswith("l_"):
                m.geom_rgba[_gi] = (0.20, 0.45, 0.85, 1.0)
    if args.friction is not None:
        m.geom_friction[:, 0] = args.friction      # sliding friction
    if args.torsion is not None:
        m.geom_friction[:, 1] = args.torsion       # torsional friction (model default 0.005, very low)
    if args.noslip is not None:
        m.opt.noslip_iterations = args.noslip
    if args.condim is not None:
        m.geom_condim[:] = args.condim
    if args.hard_friction:
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        m.opt.impratio = 10.0
    if args.stiff_contact:
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
        m.opt.o_solref[:] = [0.006, 1.0]
        m.opt.o_solimp[:] = [0.95, 0.99, 0.001, 0.5, 2.0]
        # The override flag also replaces the friction of every contact with o_friction (default torsion
        # 0.005), so --torsion and --friction must be copied into it or they are silently ignored.
        if args.torsion is not None:
            m.opt.o_friction[2] = args.torsion
        if args.friction is not None:
            m.opt.o_friction[:2] = args.friction
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "standing")
    mujoco.mj_resetDataKeyframe(m, d, key)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    jadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    vadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    if pprof:
        # The keyframe is the standing pose; a policy profile starts from its own nominal pose
        # (the robot first drops a few millimetres onto the ground).
        d.qpos[jadr] = io.default
        d.qvel[vadr] = 0.0
        mujoco.mj_forward(m, d)
    if args.jitter > 0.0:
        rng = np.random.default_rng(args.seed)
        lo = np.array([j["range"][0] for j in cfg["joints"]])
        hi = np.array([j["range"][1] for j in cfg["joints"]])
        d.qpos[jadr] = np.clip(d.qpos[jadr] + rng.uniform(-args.jitter, args.jitter, 12), lo, hi)
        mujoco.mj_forward(m, d)
    kp = np.array([g[n]["kp"] for n in names]) * args.kp_scale
    kd = np.array([g[n]["kd"] for n in names]) * args.kd_scale
    tmax = np.array([g[n]["tau_max"] for n in names]) * args.tau_scale
    decim = int(round(1.0 / (io.rate * m.opt.timestep)))
    policy = load_policy(args.policy, io.obs_dim, 12)
    cmd = np.array(args.cmd)
    last_act = np.zeros(12)

    def build_obs(step_i=0):
        # Base angular velocity: for a free joint, qvel[3:6] already is the body-frame angular velocity,
        # the quantity the real IMU measures. (mj_objectVelocity with flg_local=1 returns it in the body's
        # inertial frame instead, which need not be aligned with the body frame, so it is not used here.)
        R = d.xmat[base].reshape(3, 3)
        ang_b = d.qvel[3:6].copy()
        # Gait clock: t = 0 is the moment the policy starts.
        return io.build_obs(ang_vel=ang_b,
                            gravity=R.T @ np.array([0.0, 0.0, -1.0]),
                            cmd=cmd, q=d.qpos[jadr], qd=d.qvel[vadr],
                            last_action=last_act,
                            phase=io.phase_of(step_i / io.rate))

    steps = int(args.duration * io.rate)
    tilts, heights, acts, vels, yaws = [], [], [], [], []
    xy0 = d.qpos[:2].copy()
    # Foot clearance: the body position is the origin of ankle_roll_link, which is SOLE_OFFSET above the
    # ground when the foot is flat on it. Subtract it to get the real clearance of the sole.
    SOLE_OFFSET = 0.0585
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("l_ankle_roll_link", "r_ankle_roll_link")]
    clear = []
    grf = []
    hz = []        # whole-robot angular momentum about the vertical axis; d/dt = yaw moment the ground must supply
    yslip = [0.0, 0.0]                 # accumulated yaw slip of each stance foot (deg)
    _fyaw_prev = [None, None]
    landv = []                         # downward foot velocity at each touchdown (m/s)
    _land_prev = [None, None]
    hipyaw_dev = []                    # hip_yaw deviation from the nominal pose, per control step
    landwin = []                       # (sole height, vertical foot velocity) of both feet, per control step
    _yaw_idx = [names.index(n) for n in ("l_hip_yaw_joint", "r_hip_yaw_joint")]
    act_hist = []                      # action vector per control step: the raw network output, before clipping
    # geoms that belong to each foot (looked up by body, no hard-coded geom names)
    foot_geoms = [{gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == b} for b in feet]
    gdump = [] if args.grf_dump else None   # --grf-dump: (t, F_L, F_R, vz_L, vz_R, h_L, h_R)
    # Foot spacing and leg-leg contact. Each geom is assigned to the left / right leg by the prefix of its
    # body name; the ground (world) and the torso (base_link) map to None and never count as leg-leg.
    # Self-collision is enabled in the model, so contacts between the left and right leg are real contacts.
    def _leg_of(gi):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
        return "L" if bn.startswith("l_") else ("R" if bn.startswith("r_") else None)
    geom_leg = [_leg_of(gi) for gi in range(m.ngeom)]
    fdist = []                         # per control step: (horizontal distance, lateral distance in the base yaw frame) [m]
    legleg = 0                         # number of control steps with left-right leg contact

    def yaw_now():
        w, x, y, z = d.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    yaws.append(yaw_now())
    yaw0 = yaw_now()          # target heading of the --heading loop
    # Bridge speed limit: start from the current joint angles (as the bridge does when it starts
    # sending), so that the initial homing phase also takes place in simulation.
    from slew import slew_step
    if args.slew_spd:
        import slew as _slew_mod
        _ov = {k: float(v) for k, v in (kv.split("=") for kv in args.slew_spd.split(","))}
        assert set(_ov) <= set(_slew_mod.FAST_SPD), f"unknown motor model: {_ov}"
        _slew_mod.FAST_SPD.update(_ov)
        print(f"[slew] policy-control speed limits overridden: {_ov} rad/s")
    slew_on = not args.no_slew
    slew_cur = d.qpos[jadr].copy()
    slew_homed = [False] * 12
    _mot = {j["name"]: j["motor"] for j in cfg["joints"]}
    slew_model = [_mot[n] for n in names]        # in io.names order; does not assume the yaml order
    if args.suspend or args.suspend_pivot:
        d.qpos[2] += 0.35                   # lift until the feet are clear of the ground
        mujoco.mj_forward(m, d)
        print(f"[sim] hoisted ({'rigid' if args.suspend else 'single point, free to rotate'}, "
              f"base pinned at {d.qpos[2]:.2f} m, feet off the ground)")
    base_pin = d.qpos[:7].copy()
    delay_buf = [d.qpos[jadr].copy() for _ in range(args.delay)]
    jit_rng = jit_buf = None
    if args.delay_jitter is not None:
        jlo, jhi = args.delay_jitter
        assert args.delay == 0, "--delay-jitter and --delay are mutually exclusive (both model the execution delay)"
        assert 0 <= jlo <= jhi, f"--delay-jitter needs 0 <= LO <= HI, got {jlo},{jhi}"
        jit_rng = np.random.RandomState(args.seed)   # independent stream; leaves the other random draws unchanged
        jit_buf = [d.qpos[jadr].copy() for _ in range(jhi + 1)]
        print(f"[sim] time-varying execution delay modelled: {jlo}-{jhi} control steps (uniform draw per step, seed {args.seed})")
    if slew_on:
        print("[sim] bridge speed limit modelled (after homing: each motor's vel_limit)")
    if args.delay:
        print(f"[sim] execution delay modelled: {args.delay} control steps ({args.delay * 1000 / io.rate:.0f} ms)")
    tau_ratio = []          # per control step and joint: max |applied torque| / torque limit
    rec = [] if args.log else None
    viewer = None
    if args.view:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)
    vid = None
    if args.video:
        # Render at 100 Hz (one frame every decim/2 physics substeps): the normal-speed files use every
        # other frame, the slow-motion file uses all frames at a quarter of the rate.
        # View = (azimuth, elevation, distance, look-at height).
        _views = dict(VIDEO_VIEWS)
        _views.update({"side_r": (-90.0, -8.0, 1.0, 0.20), "oblique": (-135.0, -15.0, 1.1, 0.18)})
        _rend = mujoco.Renderer(m, height=720, width=960)
        _cams = {}
        for _n, (_az, _el, _di, _lz) in _views.items():
            _c = mujoco.MjvCamera()
            _c.azimuth, _c.elevation, _c.distance = _az, _el, _di
            _c.lookat[:] = [0.0, 0.0, _lz]
            _cams[_n] = _c
        vid = dict(rend=_rend, cams=_cams, frames={n: [] for n in _views},
                   every=max(1, decim // 2))

    def _grab():
        for _n, _c in vid["cams"].items():
            _c.lookat[0], _c.lookat[1] = d.qpos[0], d.qpos[1]
            vid["rend"].update_scene(d, camera=_c)
            vid["frames"][_n].append(vid["rend"].render().copy())

    # -- Push disturbances: torso force pulse and sustained hand push (tools/bal_push.py) --
    bal_on = bool(args.push_torso or args.hand or args.push)     # adds external force / metrics only; --push keeps its own path
    bal_rec = []
    from tools.bal_push import BalPush
    _bp = BalPush(io.rate, torso=args.push_torso, hand=args.hand)
    bal_win = ([(args.push[1], args.push[1], "vel", args.push_dir, args.push[0])] if args.push else []) + _bp.windows()
    bal_on = bal_on or bool(_bp.torso or _bp.hand)

    def _bal_force(step_i):
        return _bp.force(step_i / io.rate, d.qpos[:2], d.qvel[:2])

    try:
        for i in range(steps):
            if args.heading:
                # Heading loop: wz_cmd = clip(k * wrap(heading_target - yaw), -0.6, 0.6)
                _e = np.arctan2(np.sin(yaw0 - yaw_now()), np.cos(yaw0 - yaw_now()))
                cmd[2] = float(np.clip(args.heading_k * _e, -0.6, 0.6))
            a_raw = policy(build_obs(i))
            if args.action_ema is not None:
                _ema = a_raw if i == 0 or "_ema" not in dir() else args.action_ema * a_raw + (1 - args.action_ema) * _ema
                a_raw = _ema
            a = np.clip(a_raw, -io.action_clip, io.action_clip) * args.power_scale
            last_act = a.copy()
            act_hist.append(np.asarray(a_raw, dtype=float).copy())
            target = io.action_to_target(a, cmd=cmd, phase=io.phase_of(i / io.rate))
            if slew_on:
                # Model the real chain: policy -> target -> [bridge speed limit] -> motor PD.
                # Without this stage the simulation misses the amplitude loss and the lag that the
                # limiter causes on the real robot (see slew.py).
                for k in range(12):
                    slew_cur[k], slew_homed[k] = slew_step(
                        slew_cur[k], target[k], slew_model[k],
                        slew_homed[k], True, 1.0 / io.rate)
                target = slew_cur.copy()
            if args.delay:
                delay_buf.append(target)
                target = delay_buf.pop(0)
            elif jit_buf is not None:
                # Ring buffer: buf[-1] = this step, buf[-(k+1)] = k steps ago; k is drawn from {LO..HI} every step
                jit_buf.append(target)
                jit_buf.pop(0)
                k = int(jit_rng.randint(args.delay_jitter[0], args.delay_jitter[1] + 1))
                target = jit_buf[-(k + 1)]
            if args.push and abs(i / io.rate - args.push[1]) < 0.5 / io.rate:
                if args.push_dir == 90.0:
                    # sideways impulse: change the base linear velocity directly (impulse / total mass)
                    d.qvel[1] += args.push[0] / m.body_subtreemass[0]
                    print(f"[sim] t={i/io.rate:.2f}s sideways push {args.push[0]:.1f} N·s "
                          f"-> dvy {args.push[0]/m.body_subtreemass[0]:+.3f} m/s")
                else:
                    _pd = np.radians(args.push_dir)
                    _dv = args.push[0] / m.body_subtreemass[0]
                    d.qvel[0] += _dv * np.cos(_pd)
                    d.qvel[1] += _dv * np.sin(_pd)
                    print(f"[sim] t={i/io.rate:.2f}s push {args.push[0]:.1f} N·s @ {args.push_dir:.0f}° -> dv {_dv:.3f} m/s")
            if bal_on:
                d.xfrc_applied[base, :] = 0.0
                d.xfrc_applied[base, :2] = _bal_force(i)
            for _sub in range(decim):
                tau = kp * (target - d.qpos[jadr]) - kd * d.qvel[vadr]
                d.qfrc_applied[vadr] = np.clip(tau, -tmax, tmax)
                if _sub == 0:
                    tau_ratio.append(np.zeros(12))
                np.maximum(tau_ratio[-1], np.abs(d.qfrc_applied[vadr]) / tmax, out=tau_ratio[-1])
                mujoco.mj_step(m, d)
                if gdump is not None:
                    _gf = [0.0, 0.0]
                    for _ci in range(d.ncon):
                        _c = d.contact[_ci]
                        for _k, _bs in enumerate(foot_geoms):
                            if _c.geom1 in _bs or _c.geom2 in _bs:
                                _f6 = np.zeros(6)
                                mujoco.mj_contactForce(m, d, _ci, _f6)
                                _gf[_k] += abs(_f6[0])
                    _gv = []
                    for _b in feet:   # mj_objectVelocity = [angular 3; linear 3], the vertical linear velocity is [5]
                        _v6 = np.zeros(6)
                        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, _b, _v6, 0)
                        _gv.append(float(_v6[5]))
                    gdump.append((float(d.time), _gf[0], _gf[1], _gv[0], _gv[1],
                                  float(d.xpos[feet[0]][2] - SOLE_OFFSET), float(d.xpos[feet[1]][2] - SOLE_OFFSET)))
                if vid is not None and _sub % vid["every"] == 0:
                    _grab()
                if args.suspend:            # rigid hoist: base pinned, legs swing freely
                    d.qpos[:7], d.qvel[:6] = base_pin, 0.0
                elif args.suspend_pivot:    # single-point hoist: position pinned, torso free to rotate about the hoist point
                    d.qpos[:3], d.qvel[:3] = base_pin[:3], 0.0
            R = d.xmat[base].reshape(3, 3)
            tilts.append(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
            heights.append(float(d.qpos[2]))
            if bal_on:
                _w, _x, _y, _z = d.qpos[3:7]
                bal_rec.append([(i + 1) / io.rate, d.qpos[0], d.qpos[1], d.qvel[0], d.qvel[1],
                                d.xpos[feet[0]][2] - SOLE_OFFSET, d.xpos[feet[1]][2] - SOLE_OFFSET,
                                d.xpos[feet[0]][0], d.xpos[feet[0]][1], d.xpos[feet[1]][0], d.xpos[feet[1]][1],
                                float(np.arctan2(2 * (_w * _z + _x * _y), 1 - 2 * (_y * _y + _z * _z))),
                                tilts[-1], d.xfrc_applied[base, 0], d.xfrc_applied[base, 1]])
            acts.append(float(np.abs(a).mean()))
            # Body-frame linear velocity = xmat^T * world linear velocity; the yaw rate is the z component
            # of the free joint's angular velocity.
            _lin_b = d.xmat[base].reshape(3, 3).T @ d.qvel[0:3]
            vels.append((_lin_b[0], _lin_b[1], d.qvel[5]))  # body-frame vx, vy, yaw rate
            yaws.append(yaw_now())
            clear.append([d.xpos[b][2] - SOLE_OFFSET for b in feet])
            # Foot spacing: the horizontal distance is smallest at the moment the swing leg passes the
            # stance leg, a direct indicator of how close the legs come to hitting each other. The lateral
            # component is taken in the base yaw frame so that it can be compared with the nominal
            # standing spacing of 214.5 mm.
            _dxy = d.xpos[feet[0]][:2] - d.xpos[feet[1]][:2]
            _yw = yaws[-1]
            fdist.append((float(np.hypot(*_dxy)),
                          float(abs(np.cos(_yw) * _dxy[1] - np.sin(_yw) * _dxy[0]))))
            # Touchdown vertical speed: the downward foot velocity in the control step in which the foot
            # first touches the ground. This is not the same quantity as the touchdown force, whose peak can
            # be dominated by load transfer from torso sway instead of the descent speed of the foot.
            for _fi, _b in enumerate(feet):
                # Keep bool(): the comparison yields numpy.bool_, and `np.bool_(False) is False` is false,
                # so the rising-edge test below would never fire.
                _on = bool(d.xpos[_b][2] - SOLE_OFFSET < 0.004)
                if _on and _land_prev[_fi] is False and i > steps // 3:
                    _v6 = np.zeros(6)
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, _b, _v6, 0)
                    landv.append(max(0.0, -float(_v6[5])))   # [angular 3; linear 3]: the vertical linear velocity is [5]
                _land_prev[_fi] = _on
            # Raw gait quantities for the summary: hip_yaw deviation from the nominal pose (meaningful when
            # walking straight), and sole height / vertical velocity of both feet.
            if i > steps // 3:
                _q = d.qpos[jadr]
                hipyaw_dev.append(float(sum(abs(_q[k] - io.default[k]) for k in _yaw_idx)))
                _row = []
                for _b in feet:
                    _v = np.zeros(6)
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, _b, _v, 0)
                    _row.append((float(d.xpos[_b][2] - SOLE_OFFSET), float(_v[5])))   # [5] = vertical linear velocity
                landwin.append(_row)
            mujoco.mj_subtreeVel(m, d)
            hz.append(float(d.subtree_angmom[0][2]))
            # Stance-foot yaw slip: a foot on the ground should not rotate about the vertical axis; if it
            # does, it is slipping. This is measured directly from the foot orientation.
            for _fi, _b in enumerate(feet):
                _on = d.xpos[_b][2] - SOLE_OFFSET < 0.004
                _R = d.xmat[_b].reshape(3, 3)
                _y = float(np.arctan2(_R[1, 0], _R[0, 0]))
                if _on and _fyaw_prev[_fi] is not None and i > steps // 3:
                    _dy = np.arctan2(np.sin(_y - _fyaw_prev[_fi]),
                                     np.cos(_y - _fyaw_prev[_fi]))
                    yslip[_fi] += abs(np.degrees(_dy))
                _fyaw_prev[_fi] = _y if _on else None
            # Touchdown force: the sum of the normal forces of all contacts on each foot. It is sampled after
            # the decim substeps, so spikes within a control step are missed, but it is good enough for
            # relative comparison.
            fpk = [0.0, 0.0]
            _ll = False
            for ci in range(d.ncon):
                c = d.contact[ci]
                # Left-right leg contact, sampled like the touchdown force: a graze shorter than one
                # control step can be missed, but a real leg collision lasts several steps.
                if (geom_leg[c.geom1] is not None and geom_leg[c.geom2] is not None
                        and geom_leg[c.geom1] != geom_leg[c.geom2]):
                    _ll = True
                for k, bset in enumerate(foot_geoms):
                    if c.geom1 in bset or c.geom2 in bset:
                        f6 = np.zeros(6)
                        mujoco.mj_contactForce(m, d, ci, f6)
                        fpk[k] += abs(f6[0])
            if _ll:
                legleg += 1
            grf.append(fpk)
            if rec is not None:
                rec.append((i / io.rate, R.T @ np.array([0.0, 0.0, -1.0]),
                            d.qpos[jadr].copy(), target.copy(), a.copy()))
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break
            if d.qpos[2] < 0.25:
                print(f"[fall] t={i/io.rate:.2f}s base height {d.qpos[2]:.3f} m")
                break
    finally:
        if viewer is not None:
            viewer.close()

    if vid is not None:
        write_videos(vid["frames"], args.video, fps_full=int(round(io.rate * decim / vid["every"])),
                     fps_out=args.video_fps)
    t = len(heights) / io.rate
    alive = heights and heights[-1] > 0.3 and tilts[-1] < 25
    # Metrics for --json: filled block by block from the same arrays as the terminal output below.
    met = {"policy": args.policy, "cmd": [float(x) for x in cmd], "t": float(t),
           "alive": bool(alive), "seed": args.seed, "jitter": args.jitter,
           "yaw_slip_deg": float(sum(yslip)), "legleg_frames": int(legleg),
           "tilt_max_deg": float(max(tilts)) if tilts else None}
    if args.video:
        met["video"] = args.video
    print(f"policy={args.policy}  cmd={list(cmd)}  {t:.1f}s: "
          f"{'survived ✓' if alive else 'fell / aborted ✗'}")
    print(f"  base height  end {heights[-1]:.3f} m (min {min(heights):.3f})  "
          f"tilt  max {max(tilts):.1f}°  steady mean {np.mean(tilts[len(tilts)//3:]):.1f}°  "
          f"end {tilts[-1]:.1f}°  mean |action| {np.mean(acts):.3f}")
    if vels:
        v = np.array(vels[int(io.rate):])                   # skip the 1 s start-up transient
        v = v if len(v) else np.array(vels)
        dxy = d.qpos[:2] - xy0
        # The yaw rate is taken from the world-frame yaw angle: the torso sways during the gait, and
        # averaging the body-frame z rate over a swaying axis gives a biased result.
        dyaw = np.unwrap(np.array(yaws))[-1] - yaws[0]
        print(f"  actual speed  vx {v[:, 0].mean():+.3f} (cmd {cmd[0]:+.2f})  "
              f"vy {v[:, 1].mean():+.3f} ({cmd[1]:+.2f}) m/s  "
              f"wz {dyaw / t:+.3f} ({cmd[2]:+.2f}) rad/s")
        print(f"  world displacement {np.hypot(*dxy):.2f} m (dx {dxy[0]:+.2f} dy {dxy[1]:+.2f})  "
              f"net yaw {np.degrees(dyaw):+.1f}°")
        met.update(vx_mean=float(v[:, 0].mean()), vy_mean=float(v[:, 1].mean()),
                   dyaw_deg=float(np.degrees(dyaw)), dx=float(dxy[0]), dy=float(dxy[1]))
    if clear:
        # Peak of every swing: a sole more than 2 mm above the ground is in swing; take the peak of each
        # continuous swing segment. The minimum over all swings is the number to watch.
        c = np.array(clear[int(io.rate):]) if len(clear) > io.rate else np.array(clear)
        # A real swing lasts about 0.2 s. Lift-offs shorter than 100 ms are bounces, not steps, and are
        # filtered out so that they do not spoil the minimum.
        min_len = max(3, int(0.1 * io.rate))
        peaks = []
        for k in range(2):
            seg, cur = [], []
            for z in c[:, k]:
                if z > 0.002:
                    cur.append(z)
                else:
                    if len(cur) >= min_len:
                        seg.append(max(cur))
                    cur = []
            if len(cur) >= min_len:
                seg.append(max(cur))
            peaks.append(seg)
        allp = peaks[0] + peaks[1]
        if allp:
            print(f"  foot clearance  swing peak min {min(allp)*1000:.0f} mm  mean {np.mean(allp)*1000:.0f} mm  "
                  f"(left {len(peaks[0])} steps / right {len(peaks[1])} steps, highest {c.max()*1000:.0f} mm)")
        else:
            print(f"  foot clearance  neither foot lifted more than 2 mm (highest {c.max()*1000:.0f} mm): the feet drag")
        met.update(swing_min_mm=float(min(allp) * 1000) if allp else 0.0,
                   swing_mean_mm=float(np.mean(allp) * 1000) if allp else 0.0,
                   swing_steps=[len(peaks[0]), len(peaks[1])])
    if fdist:
        # Foot spacing: the feet are 80 mm wide, so a centre distance of 120 mm leaves only 40 mm between
        # their edges; keep the minimum above that. Any step with leg-leg contact means the legs hit each other.
        fd = np.array(fdist[len(fdist) // 3:])
        print(f"  foot spacing  lateral mean {fd[:, 1].mean()*1000:.0f} mm (nominal 214.5)  "
              f"closest horizontal distance {fd[:, 0].min()*1000:.0f} mm  "
              f"{'⚠️ ' if legleg else ''}leg-leg contact {legleg} steps")
        met.update(fdist_min_mm=float(fd[:, 0].min() * 1000),
                   fdist_lat_mean_mm=float(fd[:, 1].mean() * 1000))
        # How far the lateral spacing falls below a minimum d_min, averaged per step, for d_min = 0.14 / 0.16 / 0.18 m.
        _cal = "  ".join(f"@{t:.2f} {np.mean(np.clip(t - fd[:, 1], 0.0, None)):.4f}"
                         for t in (0.14, 0.16, 0.18))
        print(f"  foot spacing shortfall  mean(relu(d_min - lateral)) per step [m] at d_min: {_cal}")
        # -- The next two readouts are meaningful for straight walking only --
        if hipyaw_dev:
            _hy = np.array(hipyaw_dev)
            print(f"  hip_yaw deviation  sum|q - nominal| per step: mean {_hy.mean():.4f} rad "
                  f"({np.degrees(_hy.mean()):.2f}°)  p95 {np.percentile(_hy, 95):.4f}")
        if landwin:
            _lw = np.array(landwin)                       # (steps, 2 feet, [h, vz])
            _cal5 = "  ".join(
                f"@{g:.2f} {np.mean(np.sum(np.clip(-_lw[:, :, 1] - 0.30, 0.0, None) * (_lw[:, :, 0] < g), axis=1)):.4f}"
                for g in (0.015, 0.02, 0.03))
            print(f"  fast foot descent near the ground  mean(sum relu(-vz - 0.30) * [h < h_gate]) per step [m/s] at h_gate: {_cal5}")
    if sum(yslip) > 0:
        # The slip occurs in the lightly loaded phases just after touchdown and just before lift-off,
        # where the instantaneous normal force, and with it the available yaw friction, is small. Producing
        # less yaw momentum and keeping the foot flat help more than a higher ground friction.
        print(f"  stance-foot yaw slip  total {sum(yslip):.0f}° (left {yslip[0]:.0f} right {yslip[1]:.0f})"
              f"  (a foot that turns while on the ground is slipping; this causes heading drift)")
    if len(hz) > 4:
        # Legs swinging in antiphase feed yaw angular momentum into the robot, and the contributions of
        # the two legs add up instead of cancelling. Without arms to compensate, only the yaw friction of the
        # stance foot holds it. For the 170 x 80 mm sole the limit of one foot is about 2.8 (mu 0.6) /
        # 3.3 (mu 0.7) / 4.7 (mu 1.0) N*m; beyond it the foot slips and the torso is twisted.
        _h = np.array(hz); _n = len(_h) // 3
        _tau = np.abs(np.gradient(_h, 1.0 / io.rate))[_n:]
        _hh = _h[_n:]
        print(f"  yaw angular momentum H_z  amplitude ±{np.percentile(np.abs(_hh - _hh.mean()), 95):.4f} kg·m²/s"
              f"  dominant frequency {np.fft.rfftfreq(len(_hh), 1.0/io.rate)[np.argmax(np.abs(np.fft.rfft((_hh-_hh.mean())*np.hanning(len(_hh))))[1:])+1]:.2f} Hz")
        print(f"  yaw moment the ground must supply {np.percentile(_tau, 95):.2f} N·m (p95, peak "
              f"{_tau.max():.2f})  (one foot can supply about 2.8-3.3 N·m at friction 0.6-0.7)")
    if grf:
        # Normalised by body weight: touchdown force / mg. Human walking peaks at about 1.2-1.5 x body
        # weight; above 2 the foot is slamming into the ground.
        mg = m.body_subtreemass[0] * 9.81 if m.body_subtreemass[0] > 0 else \
            float(np.sum(m.body_mass)) * 9.81
        gf = np.array(grf)
        half = len(gf) // 3
        pk = gf[half:].max()
        # Per-touchdown peaks: the peak within 0.1 s after each rising edge of the force, / body weight
        _pks = []
        for _j in range(gf.shape[1]):
            _f = gf[half:, _j]
            for _i in np.flatnonzero((_f[1:] > 5.0) & (_f[:-1] <= 5.0)) + 1:
                _pks.append(_f[_i:_i + max(int(0.1 * io.rate), 1)].max() / mg)
        print(f"  touchdown force  peak {pk:.0f} N ({pk/mg:.2f}× body weight)  "
              f"single-foot p95 {np.percentile(gf[half:], 95):.0f} N  (body weight {mg:.0f} N)"
              + (f"  per-touchdown peak p95 {np.percentile(_pks, 95):.2f}× median {np.median(_pks):.2f}× ({len(_pks)} touchdowns)" if _pks else ""))
        # Flight phase: both feet off the ground, judged by contact force (< 5 N on each foot) and not by
        # sole height: at toe-off and heel-strike the foot pitches, so its origin rises by a few millimetres
        # while the toe still touches. A walking policy should show about zero here. A gap of a single
        # control step is contact chatter; only segments of 2 or more steps are counted.
        fly = (gf[half:, 0] < 5.0) & (gf[half:, 1] < 5.0)
        fl_segs, _n = [], 0
        for _b in fly:
            if _b:
                _n += 1
            elif _n:
                fl_segs.append(_n); _n = 0
        if _n:
            fl_segs.append(_n)
        fl_segs = [s for s in fl_segs if s >= 2]
        fl_frac = float(np.sum(fl_segs)) / max(len(fly), 1)
        met.update(flight_frac=fl_frac, flight_segs=len(fl_segs),
                   flight_seg_ms=float(np.mean(fl_segs) / io.rate * 1000) if fl_segs else 0.0)
        if fl_segs:
            print(f"  flight phase  both feet off the ground {fl_frac*100:.1f}% of the time  "
                  f"segments {len(fl_segs)}  mean length {np.mean(fl_segs)/io.rate*1000:.0f} ms")
    if landv:
        lv = np.array(landv)
        # A p95 below 0.3 m/s is a reasonable target: human touchdown speeds of 0.2-0.4 m/s, scaled with
        # the square root of the leg length, correspond to about 0.19 m/s for this robot.
        print(f"  touchdown vertical speed  mean {lv.mean():.3f} m/s  p95 {np.percentile(lv, 95):.3f}  "
              f"max {lv.max():.3f}  ({len(lv)} touchdowns)")
        met.update(land_vz_p95=float(np.percentile(lv, 95)))
    if act_hist:
        # Action saturation: a joint whose action sits on the clip no longer responds to the policy; its
        # target stays frozen at the edge of the action range. act_hist holds the raw network output
        # before clipping, so "just at the clip" and "far beyond it" can be told apart.
        A = np.abs(np.array(act_hist)[len(act_hist) // 3:])
        sat = (A >= io.action_clip - 1e-6).mean(axis=0) * 100
        worst = int(np.argmax(sat))
        names = list(io.names)
        over = np.clip(A - 0.9, 0.0, None).sum(axis=1)     # sum over the joints of the excess of |a| above 0.9
        print(f"  action saturation  worst joint {sat.max():.0f}% of steps ({names[worst]})  "
              f"joints saturated in ≥50% of steps: {int((sat >= 50).sum())}  (a saturated joint stays pinned at the action clip)")
        print(f"  raw |a|  max {A.max():.3f} ({names[int(np.unravel_index(A.argmax(), A.shape)[1])]})  "
              f"excess above 0.9, sum(relu(|a|-0.9)): mean {over.mean():.3f} per step")
        met.update(sat_max_pct=float(sat.max()), sat_worst_joint=names[worst],
                   raw_a_max=float(A.max()))
    if tau_ratio:
        TR = np.array(tau_ratio)
        hot = TR >= 0.9
        run_mx = np.zeros(12, int)
        for _k in range(12):
            _r = 0
            for _h in hot[:, _k]:
                _r = _r + 1 if _h else 0
                run_mx[_k] = max(run_mx[_k], _r)
        _w = np.argsort(-TR.max(0))[:3]
        print("  joint torque  peak / torque limit (time at ≥90%: longest run / total): " + "  ".join(
            f"{names[k].replace('_joint', '')} {TR[:, k].max()*100:.0f}% ({run_mx[k]*1000/io.rate:.0f}/{hot[:, k].sum()*1000/io.rate:.0f} ms)"
            for k in _w))
        met.update(tau_ratio_peak={names[k]: float(TR[:, k].max()) for k in range(12)},
                   tau90_run_ms={names[k]: float(run_mx[k] * 1000 / io.rate) for k in range(12)})
    if rec:
        from tools.policy_io import summarize
        summarize(rec, io, args.log)
    if gdump is not None:
        np.savez(args.grf_dump, rec=np.array(gdump), mass=float(m.body_subtreemass[0]), dt=float(m.opt.timestep),
                 rate=float(io.rate))
        print(f"[grf-dump] {len(gdump)} substeps -> {args.grf_dump}")
    if bal_on and bal_rec:
        met["balance"] = balance_metrics(np.array(bal_rec), io.rate, bal_win, alive)
        _bm = met["balance"]
        print(f"  [balance] push response: steps taken {_bm['steps']} (disturbance window) / idle steps {_bm['idle_steps']}  displacement {_bm['disp_m']:.2f} m  "
              f"peak tilt {_bm['tilt_max_deg']:.1f}°  settle time {_bm['settle_s'] if _bm['settle_s'] is not None else 'n/a'} s  "
              f"final stance Δx {_bm['final_dx_m']*1000:+.0f} mm |Δy| {_bm['final_dy_m']*1000:.0f} mm"
              + (f"  hand force median {_bm['hand_f_med']:.1f} / peak {_bm['hand_f_max']:.1f} N follow ratio {_bm['hand_follow']:.2f}"
                 if _bm.get("hand_f_med") is not None else ""))
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(met, f, ensure_ascii=False, indent=1)
        print(f"[json] metrics -> {args.json}")
    # Exit code: 0 if the robot survived, else 1.
    sys.exit(0 if alive else 1)


def balance_metrics(rec, rate, wins, alive, h_on=0.008, settle_v=0.05, settle_hold=0.5):
    """Push-response metrics. rec columns: t x y vx vy hL hR fLx fLy fRx fRy yaw tilt Fx Fy.

    A step = one foot off the ground (sole height > h_on) for at least 60 ms with a peak height of at least 10 mm.
    The disturbance window runs from the start of the first disturbance to 3 s after the end of the last one;
    steps outside it are idle steps (the robot steps although nothing pushes it). Settled = start of the first
    0.5 s span after the last disturbance in which the horizontal speed stays below 5 cm/s with both feet on the
    ground, in seconds after the end of the last disturbance.
    """
    t = rec[:, 0]
    t_first = min(w[0] for w in wins)
    t_last = max(w[1] for w in wins)
    steps, idle_steps = 0, 0
    for col in (5, 6):
        air = rec[:, col] > h_on
        i = 0
        while i < len(air):
            if air[i]:
                j = i
                while j < len(air) and air[j]:
                    j += 1
                if (j - i) / rate >= 0.06 and rec[i:j, col].max() >= 0.010:
                    if t_first <= t[i] <= t_last + 3.0:
                        steps += 1
                    else:
                        idle_steps += 1
                i = j
            else:
                i += 1
    k0 = int(np.searchsorted(t, t_first))
    disp = float(np.hypot(rec[-1, 1] - rec[k0, 1], rec[-1, 2] - rec[k0, 2]))
    spd = np.hypot(rec[:, 3], rec[:, 4])
    both = (rec[:, 5] < h_on) & (rec[:, 6] < h_on)
    ok = (spd < settle_v) & both
    need = int(round(settle_hold * rate))
    settle = None
    k1 = int(np.searchsorted(t, t_last))
    run = 0
    for k in range(k1, len(t)):
        run = run + 1 if ok[k] else 0
        if run >= need:
            settle = round(float(t[k - need + 1] - t_last), 2)
            break
    yaw = rec[-1, 11]
    dlr = np.array([rec[-1, 9] - rec[-1, 7], rec[-1, 10] - rec[-1, 8]])
    dx = float(np.cos(yaw) * dlr[0] + np.sin(yaw) * dlr[1])
    dy = float(abs(-np.sin(yaw) * dlr[0] + np.cos(yaw) * dlr[1]))
    out = dict(alive=bool(alive), steps=int(steps), idle_steps=int(idle_steps), disp_m=disp,
               tilt_max_deg=float(rec[k0:, 12].max()), settle_s=settle, final_dx_m=dx, final_dy_m=dy,
               final_both_down=bool(both[-1]), wins=[list(w) for w in wins])
    for w in wins:
        if w[2] != "hand":
            continue
        sel = (t > w[0] + 0.3) & (t <= w[1])
        if sel.any():
            dirv = np.array([np.cos(np.radians(w[3])), np.sin(np.radians(w[3]))])
            f = np.hypot(rec[sel, 13], rec[sel, 14])
            v_al = rec[sel][:, 3:5] @ dirv
            out.update(hand_f_med=float(np.median(f)), hand_f_max=float(f.max()),
                       hand_follow=float(v_al.mean() / max(w[4], 1e-6)))
    return out


def _ffmpeg():
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_videos(frames, pfx, fps_full, fps_out=50):
    """frames: {view: [HxWx3 uint8]} at fps_full. Writes one normal-speed file per view (decimated to fps_out)
    and a 4x slow-motion file of the side_r view."""
    import subprocess
    os.makedirs(os.path.dirname(os.path.abspath(pfx)) or ".", exist_ok=True)
    ff = _ffmpeg()
    step = max(1, int(round(fps_full / fps_out)))
    jobs = [(n, fs[::step], fps_full / step, f"{pfx}_{n}.mp4") for n, fs in frames.items()]
    if "side_r" in frames:
        jobs.append(("side_r", frames["side_r"], fps_full / 4.0, f"{pfx}_side_r_slow4x.mp4"))
    for n, fs, fps, out in jobs:
        if not fs:
            continue
        h, w = fs[0].shape[:2]
        pr = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-pix_fmt", "yuv420p",
                               "-vcodec", "libx264", "-crf", "20", out], stdin=subprocess.PIPE)
        for fr in fs:
            pr.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        pr.stdin.close()
        pr.wait()
        print(f"[video] {out}  ({len(fs)} frames @{fps:g}fps)")


if __name__ == "__main__":
    main()
