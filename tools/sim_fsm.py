#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/sim_fsm.py - run the stand / walk state machine in MuJoCo, driven by scripted key presses.

The simulation uses the same SkillFSM (tools/skill_fsm.py), PolicyIO and manifest check as
tools/deploy_policy.py on the real robot, and the same plant model as tools/sim2sim.py: joint PD at 500 Hz,
the torque limits of biped_Laika/biped_Laika.yaml, the speed limiting of the bridge (slew.py) and an execution
delay of N control steps. Use it to check that the robot stays on its feet through a policy switch, for
example the blend from straight-leg standing into the walking stance and back, before trying the switch on
the real robot.

Usage (from the SDK root):
  python tools/sim_fsm.py --policy biped_Laika/policies/walk.onnx --power-scale 0.8 \\
      --stand-policy biped_Laika/policies/stand.onnx --script "2:w3,8:w2" --duration 13 --delay 2 --jitter 0.05
  (script: at 2 s hold W for 3 s, at 8 s hold W for 2 s)

Options:
  --policy ONNX          walking policy (LOCO state); --power-scale K scales its action (default 1.0)
  --stand-policy ONNX    standing policy (STAND state). With it the robot starts standing and walks
                         automatically while a velocity command is present. --no-auto-walk turns that off,
                         --walk-release S sets how long the command must be zero before the robot stands
                         again, --home stand|loco selects the start state explicitly.
  --script "T:KEY,..."   key presses, each at time T (s):
                           wN  hold W (walk forward at 0.2 m/s) for N seconds (default 1 s)
                           1   request standing          2   request walking
  --duration S           simulated time (default 20 s)
  --delay N              execution delay of the joint targets, in control steps (default 2)
  --jitter R, --seed N   random offset of up to ±R rad on the initial joint positions
  --friction MU          sliding friction coefficient of all geoms
  --stiff-contact, --foot-box       contact model options, as in tools/sim2sim.py
  --heading-hold, --trim-vy V       walking assistance, as in tools/deploy_policy.py
  --push, --push-dir, --push-torso, --hand      push disturbances (see tools/bal_push.py)
  --tau-log NPZ, --tilt-log NPZ     save per-step data
  --video PFX            record a video to PFX_fsm.mp4 (needs ffmpeg on the PATH or the imageio-ffmpeg package)
  --quiet                do not print the state machine log

Output: the sequence of states, the largest tilt in each state segment, for each state the peak |tau| / torque
limit (with its joint) and the longest continuous time at or above 90% of the torque limit, and for each
walking segment the distance walked, the lateral drift and the heading change. The run stops when the base
drops below 0.22 m, which is reported as a fall. The exit code is 0 if the robot did not fall, 1 otherwise.
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gains  # noqa: E402
from slew import slew_step  # noqa: E402
from tools.onnx_manifest import enforce  # noqa: E402
from tools.policy_io import PolicyIO  # noqa: E402
from tools.skill_fsm import LocoAssist, SkillFSM, Slot  # noqa: E402


def load_policy(path, obs_dim):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    assert sess.get_inputs()[0].shape[-1] == obs_dim, f"{path}: input dimension != {obs_dim}"
    return lambda obs: np.asarray(sess.run(None, {iname: obs.reshape(1, -1).astype(np.float32)})[0]).reshape(-1)[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="walking policy (ONNX), mounted on the LOCO state")
    ap.add_argument("--power-scale", type=float, default=1.0, help="multiply the walking policy's output by K (use the value you deploy with, e.g. 0.8 for the reference walking policy)")
    ap.add_argument("--stand-policy", help="standing policy for the stand + walk state machine (--policy is the walking policy)")
    ap.add_argument("--heading-hold", action="store_true", help="as in deploy_policy: hold the heading while walking without a turn command")
    ap.add_argument("--trim-vy", type=float, default=0.0, help="as in deploy_policy: lateral trim (m/s) added to vy while walking forward")
    ap.add_argument("--foot-box", action="store_true",
                    help="as in sim2sim: use a box as the foot collision shape (the flat sole mesh often yields a single "
                         "contact point in single-foot support, which lets the robot spin about it)")
    ap.add_argument("--stiff-contact", action="store_true",
                    help="as in sim2sim: stiff ground contact (with the default contact of the model the feet sink 9-12 mm into the ground)")
    ap.add_argument("--home", "--start-state", dest="home", default="auto",
                    choices=["auto", "stand", "loco"], help="home state of the state machine: auto / stand = standing (a movement key starts walking), loco = walking")
    ap.add_argument("--no-auto-walk", action="store_true", help="do not start walking automatically on a movement key; switch with key 1 (stand) / 2 (walk)")
    ap.add_argument("--walk-release", type=float, default=1.0, help="return to standing S seconds after the movement command ends (default 1.0)")
    ap.add_argument("--script", default="", help="scripted key presses 'T:KEY,...': wN = walk forward at 0.2 m/s for N seconds (w = 1 s), 1 = stand, 2 = walk; e.g. '2:w3,8:w2' = at 2 s walk for 3 s, at 8 s for 2 s")
    ap.add_argument("--duration", type=float, default=20.0, help="simulated time in seconds")
    ap.add_argument("--delay", type=int, default=2, help="action delay in control steps (the real robot has 1-2)")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--jitter", type=float, default=0.0, help="random perturbation of the initial joint angles, rad")
    ap.add_argument("--friction", type=float, default=None, help="ground friction coefficient (default: the model's value)")
    ap.add_argument("--push", nargs=2, type=float, default=None, metavar=("N_S", "AT_S"),
                    help="at AT_S seconds, give the robot a horizontal impulse of N_S (N·s): a disturbance test")
    ap.add_argument("--push-dir", type=float, default=90.0, metavar="DEG",
                    help="direction of --push (horizontal angle in the world frame, deg; default 90 = +y, sideways)")
    ap.add_argument("--push-torso", nargs=4, type=float, action="append", default=None,
                    metavar=("N_S", "DUR_S", "AT_S", "DEG"), help="torso force pulse, as in sim2sim (see tools/bal_push.py); may be repeated")
    ap.add_argument("--hand", nargs=5, type=float, default=None, metavar=("V", "AT_S", "DUR_S", "DEG", "FMAX"),
                    help="sustained hand push, as in sim2sim (see tools/bal_push.py)")
    ap.add_argument("--tau-log", default=None, metavar="NPZ", help="save per control step: t, state, per-joint peak |tau| / torque limit, "
                                                                   "measured q, FSM target")
    ap.add_argument("--tilt-log", default=None, metavar="NPZ", help="save per control step: t, state, tilt, |omega|, |omega_xy|, projected gravity")
    ap.add_argument("--video", default=None, metavar="PFX", help="record a video; PFX is the output file prefix (needs requirements-extra.txt)")
    ap.add_argument("--quiet", action="store_true", help="print only the final summary")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "biped_Laika/biped_Laika.yaml")))
    log = (lambda m: None) if args.quiet else print

    def mount(path, name, power=1.0):
        prof = enforce(path, cfg, None, strict_plant=True)
        io = PolicyIO(cfg, prof)
        return Slot(name, io, load_policy(path, io.obs_dim), power, path)

    slots = {"LOCO": mount(args.policy, "LOCO", args.power_scale)}
    if args.stand_policy:
        slots["STAND"] = mount(args.stand_policy, "STAND")
    names = slots["LOCO"].io.names
    rate = slots["LOCO"].io.rate
    start = ("STAND" if "STAND" in slots else "LOCO") if args.home == "auto" else args.home.upper()
    auto_walk = start == "STAND" and not args.no_auto_walk
    fsm = SkillFSM(slots, start, home=start, auto_walk=auto_walk, walk_release_s=args.walk_release, log=log)

    # -- scene (+ optional video) --
    sp = mujoco.MjSpec.from_file(os.path.join(ROOT, "biped_Laika", "biped_Laika.xml"))
    if args.video:
        sp.visual.global_.offwidth, sp.visual.global_.offheight = 960, 720
    if args.foot_box:
        from tools import foot_geometry
        fb = np.array(foot_geometry.FOOT_BOX)
        for fn in ("l_ankle_roll_link", "r_ankle_roll_link"):
            fbody = sp.body(fn)
            for g_ in fbody.geoms:
                g_.contype, g_.conaffinity = 0, 0
            fbody.add_geom(name=f"{fn}_solebox", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=list((fb[:, 1] - fb[:, 0]) / 2), pos=list((fb[:, 1] + fb[:, 0]) / 2),
                           contype=1, conaffinity=1, density=0.0, group=3, rgba=[0, 0, 0, 0])
    m = sp.compile()
    if args.stiff_contact:
        m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
        m.opt.o_solref[:] = [0.006, 1.0]
        m.opt.o_solimp[:] = [0.95, 0.99, 0.001, 0.5, 2.0]
    if args.friction is not None:
        m.geom_friction[:, 0] = args.friction
        if args.stiff_contact:              # with the contact override, o_friction replaces the per-geom friction
            m.opt.o_friction[:2] = args.friction
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("standing").id)
    jadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
    vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in names])
    d.qpos[jadr] = slots[start].io.default
    if args.jitter > 0:
        rng = np.random.default_rng(args.seed)
        lo = np.array([j["range"][0] for j in cfg["joints"]])
        hi = np.array([j["range"][1] for j in cfg["joints"]])
        d.qpos[jadr] = np.clip(d.qpos[jadr] + rng.uniform(-args.jitter, args.jitter, 12), lo, hi)
    mujoco.mj_forward(m, d)
    g = gains.profile("rl_default")
    kp = np.array([g[n]["kp"] for n in names])
    kd = np.array([g[n]["kd"] for n in names])
    tmax = np.array([g[n]["tau_max"] for n in names])
    mot = {j["name"]: j["motor"] for j in cfg["joints"]}
    decim = int(round(1.0 / (rate * m.opt.timestep)))
    base = m.body("base_link").id
    slew_cur, homed = d.qpos[jadr].copy(), [False] * 12
    buf = [d.qpos[jadr].copy() for _ in range(args.delay)]
    keymap = {"1": "stand", "2": "loco"}
    tau_seg = {}                          # state -> (peak ratio, peak joint, longest run of steps at ≥90%, current run)
    tau_log = []                          # --tau-log: (t, state, |τ|/limit (12,), q, target)
    assist = LocoAssist(trim_vy=args.trim_vy, hold=args.heading_hold) if (args.heading_hold or args.trim_vy) else None
    loco_seg, loco0 = [], None            # LOCO segments: [(forward m, lateral m, heading change °)], in the heading frame at segment start
    sched = sorted((float(t), k.strip()) for t, k in (tok.split(":") for tok in args.script.split(",") if tok.strip()))
    rend = cam = None
    frames = []
    if args.video:
        rend = mujoco.Renderer(m, height=720, width=960)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[2] = -120.0, -12.0, 1.3, 0.2
        for gi in range(m.ngeom):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
            if bn.startswith("r_"):
                m.geom_rgba[gi] = (0.85, 0.25, 0.20, 1.0)
            elif bn.startswith("l_"):
                m.geom_rgba[gi] = (0.20, 0.45, 0.85, 1.0)

    cmd, move_until = np.zeros(3), -1.0
    tilt_log = [] if args.tilt_log else None
    bpush = None
    if args.push_torso or args.hand:
        from tools.bal_push import BalPush
        bpush = BalPush(rate, torso=args.push_torso, hand=args.hand)
    seg_tilt = {}                         # (segment index, state) -> max tilt
    seg_i, seg_state = 0, fsm.state
    fell = None
    for i in range(int(args.duration * rate)):
        t = i / rate
        while sched and t >= sched[0][0]:
            key = sched.pop(0)[1]
            if key[0] == "w":
                move_until = t + (float(key[1:]) if len(key) > 1 else 1.0)
            else:
                fsm.request(keymap[key], t, d.qpos[jadr].copy())
        # the raw command goes to the FSM (automatic walking switches on it; outside LOCO the FSM zeroes it)
        cmd = np.array([0.2, 0.0, 0.0]) if t < move_until else np.zeros(3)
        if args.push and abs(t - args.push[1]) < 0.5 / rate:
            if args.push_dir == 90.0:
                d.qvel[1] += args.push[0] / m.body_subtreemass[0]
            else:
                d.qvel[:2] += args.push[0] / m.body_subtreemass[0] * np.array(
                    [np.cos(np.radians(args.push_dir)), np.sin(np.radians(args.push_dir))])
        if bpush is not None:
            d.xfrc_applied[base, :] = 0.0
            d.xfrc_applied[base, :2] = bpush.force(t, d.qpos[:2], d.qvel[:2])
        R = d.xmat[base].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        ang = d.qvel[3:6].copy()
        q, qd = d.qpos[jadr].copy(), d.qvel[vadr].copy()
        fsm.update(t, q, cmd)
        if tilt_log is not None:
            tilt_log.append((t, fsm.state, tilt, float(np.linalg.norm(ang)), float(np.linalg.norm(ang[:2])), *(R.T @ np.array([0.0, 0.0, -1.0]))))
        if fsm.state != seg_state:
            seg_i, seg_state = seg_i + 1, fsm.state
        seg_tilt[(seg_i, seg_state)] = max(seg_tilt.get((seg_i, seg_state), 0.0), tilt)
        yaw_now = float(np.arctan2(R[1, 0], R[0, 0]))
        if fsm.state == "LOCO" and loco0 is None:
            loco0 = (d.qpos[:2].copy(), yaw_now)
        elif fsm.state != "LOCO" and loco0 is not None:
            dp = d.qpos[:2] - loco0[0]; c0, s0 = np.cos(loco0[1]), np.sin(loco0[1])
            dyaw = np.degrees(np.arctan2(np.sin(yaw_now - loco0[1]), np.cos(yaw_now - loco0[1])))
            loco_seg.append((c0 * dp[0] + s0 * dp[1], -s0 * dp[0] + c0 * dp[1], dyaw))
            loco0 = None
        act_cmd = assist(fsm.state, cmd, yaw_now) if assist is not None else cmd
        target, _ = fsm.act(t, ang, R.T @ np.array([0.0, 0.0, -1.0]), act_cmd, q, qd)
        for k in range(12):
            slew_cur[k], homed[k] = slew_step(slew_cur[k], target[k], mot[names[k]], homed[k], True, 1.0 / rate)
        tt = slew_cur.copy()
        if args.delay:
            buf.append(tt)
            tt = buf.pop(0)
        tr = np.zeros(12)
        for sub in range(decim):
            tau = np.clip(kp * (tt - d.qpos[jadr]) - kd * d.qvel[vadr], -tmax, tmax)
            tr = np.maximum(tr, np.abs(tau) / tmax)
            d.qfrc_applied[vadr] = tau
            mujoco.mj_step(m, d)
            if rend is not None and sub % max(1, decim // 2) == 0:
                cam.lookat[0], cam.lookat[1] = d.qpos[0], d.qpos[1]
                rend.update_scene(d, camera=cam)
                frames.append(rend.render().copy())
        pk, pj, run_mx, run = tau_seg.get(fsm.state, (0.0, "", 0, 0))
        run = run + 1 if tr.max() >= 0.9 else 0
        if tr.max() > pk:
            pk, pj = float(tr.max()), names[int(tr.argmax())]
        tau_seg[fsm.state] = (pk, pj, max(run_mx, run), run)
        if args.tau_log:
            tau_log.append((t, fsm.state, tr.copy(), q.copy(), target.copy()))
        if d.qpos[2] < 0.22 and fell is None:
            fell = t
            break
    print(f"== sim_fsm seed {args.seed}  states: " + " -> ".join(e[2] for e in fsm.events))
    worst = max(seg_tilt.values())
    print("   max tilt per segment: " + "  ".join(f"{s}{v:.0f}°" for (_, s), v in sorted(seg_tilt.items())))
    print(f"   max tilt {worst:.1f}°  final state {fsm.state}"
          + (f"  ✗ fell @{fell:.2f}s" if fell else ""))
    print("   torque (peak |τ|/limit, longest run at ≥90%): " + "  ".join(
        f"{st} {pk*100:.0f}% {pj.replace('_joint', '')} {rm*1000/rate:.0f}ms" for st, (pk, pj, rm, _) in sorted(tau_seg.items())))
    if loco_seg:
        print("   walking segments (heading frame at segment start): " + "  ".join(f"forward {fx:.2f} m lateral {fy:+.2f} m heading {dy:+.0f}°" for fx, fy, dy in loco_seg))
    if tilt_log is not None:
        np.savez(args.tilt_log, t=np.array([x[0] for x in tilt_log]), state=np.array([x[1] for x in tilt_log]),
                 tilt=np.array([x[2] for x in tilt_log]), omega=np.array([x[3] for x in tilt_log]),
                 omega_xy=np.array([x[4] for x in tilt_log]), g_b=np.array([x[5:8] for x in tilt_log]))
    if args.tau_log:
        np.savez(args.tau_log, t=np.array([x[0] for x in tau_log]), state=np.array([x[1] for x in tau_log]),
                 tau_ratio=np.array([x[2] for x in tau_log]), q=np.array([x[3] for x in tau_log]),
                 target=np.array([x[4] for x in tau_log]), names=np.array(names))
    if frames:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from sim2sim import write_videos
        write_videos({"fsm": frames}, args.video, fps_full=int(round(rate * decim / max(1, decim // 2))))
    return 0 if not fell else 1


if __name__ == "__main__":
    sys.exit(main())
