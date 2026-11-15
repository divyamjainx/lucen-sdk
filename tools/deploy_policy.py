#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/deploy_policy.py - run an ONNX policy on the real biped_Laika robot (runs on the robot computer).

Data flow: IMU (tools/imu_reader.py) + motor feedback -> observation -> ONNX policy -> joint targets
           -> bus workers of bridge.py (speed limit, joint limits, damping and disable on stop).
Observations and joint targets are built by tools/policy_io.py, the same code tools/sim2sim.py uses, so the
real robot and the simulation share one policy I/O contract. Before any motor is enabled the script checks
the serial ports, the policy manifest, and the torque limit / CAN timeout stored in the motors.

Usage (run from the SDK root; hoist the robot for first runs: a new policy, new gains, a new calibration):
  # smoke test without a network: PD hold at the nominal pose, IMU not needed
  python tools/deploy_policy.py --policy zero --no-imu --duration 5
  # joint check, one joint at a time: direction / index / amplitude against the contract
  python tools/deploy_policy.py --policy zero --no-imu --onehot-real
  # a stamped policy (here: standing, velocity command 0 0 0); add --power-scale 0.1 for a first look at 10% amplitude
  python tools/deploy_policy.py --policy biped_Laika/policies/stand.onnx
  # stand + walk with keyboard teleop (use --pad instead of --teleop for a gamepad)
  python tools/deploy_policy.py --policy biped_Laika/policies/walk.onnx --stand-policy biped_Laika/policies/stand.onnx --teleop

Stand + walk: with --stand-policy the robot stands by default, walks while a movement command is active and
returns to standing shortly after the command goes back to zero.
Keyboard (--teleop): W/S forward / back, A/D side-step, Q/E turn, space = zero the command, x = quit.
Gamepad (--pad): left stick walk / side-step, right stick turn, hold R1 to move (dead-man button),
Options = stand, Share = stop, PS = quit.
Ctrl+C is the safe stop in every mode.

Safety chain: Ctrl+C / an exception in the control loop (policy, bus worker) / IMU data stalled for more than
0.1 s / tilt above 45 deg (--max-tilt-deg) -> all joints go to damping, then disable.
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bridge as lb                     # noqa: E402  (BusWorker / STATE / safety logic)
from tools.policy_io import PolicyIO, summarize   # noqa: E402


class StaticImu:
    """--no-imu: stand-in IMU for hoisted smoke tests; reports a level, motionless torso."""

    def latest(self):
        return np.zeros(3), np.array([0.0, 0.0, -1.0]), 0.0


def onehot_real(io, workers, hold):
    """Joint check on the hoisted real robot: command +0.1 on one action at a time and verify
    direction / index / amplitude from the measured joint motion.

    The test is differential (q_after - q_before). Gravity cannot be switched off on the real robot, and a
    hanging leg sags by up to about 0.05 rad, so absolute convergence means little. A one-hot step moves a
    joint by about 3 deg only, so pose-dependent terms nearly cancel in the difference; what remains is the
    static friction band (0.08-0.25 N·m per motor, at most ±0.02 rad) and backlash.
    The thresholds are deliberately loose: direction and index are what this check is for, and the amplitude
    band [0.3, 1.7]× only catches gross errors.
    tools/contract_check.py --onehot-mujoco applies the same criterion in simulation."""
    names = io.names

    def snap():
        with lb.STATE.lock:
            return np.array([lb.STATE.q[n] for n in names])

    def hold_target(vec, secs):
        end = time.monotonic() + secs
        while time.monotonic() < end:
            dead = [w.channel for w in workers if not w.alive]
            if dead:
                raise RuntimeError(f"bus worker died: {dead} (that leg is no longer under control)")
            with lb.STATE.lock:
                for n, v in zip(names, vec):
                    lb.STATE.target[n] = float(v)
            time.sleep(0.02)

    print(f"[joint-check] one-hot sequence: 12 joints × +0.1, {hold:.1f}s per step "
          f"(make sure the robot is hoisted! each joint moves ~{0.1 * float(np.max(io.action_scale_vec)) * 57.3:.1f}°)")
    hold_target(io.default, max(2.0, hold))          # move to the nominal pose (slowly: bridge speed limit)
    rows, ok_all = [], True
    for i, n in enumerate(names):
        a = np.zeros(12)
        a[i] = 0.1
        # Zero command -> zero feed-forward, so profiles with feed-forward can be checked as well.
        target = io.action_to_target(a, cmd=np.zeros(3), phase=0.0)
        delta = target[i] - io.default[i]
        if abs(delta) < 0.01:
            rows.append(f"  SKIP  {n:<22} target clipped by the joint limit (nominal pose sits on the limit): nothing to measure")
            continue
        hold_target(io.default, hold)
        q0 = snap()
        hold_target(target, hold)
        dq = snap() - q0
        moved = int(np.argmax(np.abs(dq)))
        ratio = float(dq[i] / delta)
        others = float(np.abs(np.delete(dq, i)).max())
        ok = (moved == i and np.sign(dq[i]) == np.sign(delta)
              and 0.3 < ratio < 1.7 and others < 0.03)
        ok_all &= ok
        rows.append(f"  {'PASS' if ok else 'FAIL'}  {n:<22} commanded Δ {delta:+.3f} "
                    f"measured {dq[i]:+.3f} (×{ratio:.2f})  other joints max {others:.3f}"
                    f"{'' if moved == i else f'  ⚠ argmax={names[moved]}'}")
    hold_target(io.default, max(2.0, hold))          # back to the nominal pose before leaving
    print("\n".join(rows))
    print(f"[joint-check] {'ALL PASS - direction / index / amplitude on the real robot match the contract' if ok_all else 'FAILED'}")
    return ok_all


def load_policy(path, obs_dim):
    if path == "zero":
        return lambda obs: np.zeros(12, dtype=np.float32)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    assert sess.get_inputs()[0].shape[-1] == obs_dim
    return lambda obs: np.asarray(
        sess.run(None, {iname: obs.reshape(1, -1).astype(np.float32)})[0]).reshape(-1)[:12]


FSM_LAST = None      # state machine of the most recent main() call (offline tests read its events)


class KeyTeleop:
    """Non-blocking W/A/S/D keyboard teleop (works over SSH).

    Each key press commands a velocity for `hold` seconds, then the command returns to zero; with no key
    pressed the command is zero. The terminal is switched with `tty.setcbreak`, not to raw mode: line
    buffering and echo are off but signals stay on, so Ctrl+C still raises KeyboardInterrupt and triggers the
    safe stop of the deploy loop. Ctrl+C is the last-resort stop during teleop and must stay available.

    Keys: W forward / S back / A side-step left / D side-step right / Q turn left / E turn right /
    space = zero the command / x = quit. A/D command a lateral velocity vy, Q/E a yaw rate wz.
    With the stand + walk state machine, 1 = stand and 2 = walk.
    """

    def __init__(self, speed, hold, turn=0.20, side=0.10):
        self.speed, self.hold, self.turn = float(speed), float(hold), float(turn)
        # Side-step speed has its own (lower) setting, separate from --teleop-speed: at higher lateral speed
        # the feet come close to touching.
        self.side = float(side)
        self.vec = np.zeros(3)          # (vx, vy, wz)
        self.until = 0.0
        self.quit = False
        self.reqs = []                  # request queue for the state machine (consumed by the main loop): stand / loco
        self._fd = None
        self._saved = None

    def __enter__(self):
        import termios
        import tty
        if not sys.stdin.isatty():
            sys.exit("[err] --teleop needs a terminal (stdin is not a tty). "
                     "Do not use a pipe or nohup; run it directly in an SSH session.")
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)         # ISIG stays on -> Ctrl+C still stops the robot
        return self

    def __exit__(self, *exc):
        import termios
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def poll(self, now):
        """Consume all pending key presses and update the command. Return True when the quit key was pressed."""
        import select
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1).lower()
            v = None
            if ch == "w":
                v = (self.speed, 0.0, 0.0)          # forward
            elif ch == "s":
                v = (-self.speed, 0.0, 0.0)         # back
            elif ch == "a":
                v = (0.0, self.side, 0.0)           # side-step left
            elif ch == "d":
                v = (0.0, -self.side, 0.0)          # side-step right
            elif ch == "q":
                v = (0.0, 0.0, self.turn)           # turn left
            elif ch == "e":
                v = (0.0, 0.0, -self.turn)          # turn right
            elif ch == " ":
                self.vec, self.until = np.zeros(3), 0.0      # stop: zero the command at once
                print("\r[teleop] stop            ", end="", flush=True)
            elif ch == "x":
                self.quit = True
            elif ch == "1":
                self.reqs.append("stand")
            elif ch == "2":
                self.reqs.append("loco")
            if v is not None:
                self.vec, self.until = np.array(v), now + self.hold
        return self.quit

    def command(self, now):
        """-> (vx, vy, wz); zero once the hold window has passed."""
        if now >= self.until:
            return np.zeros(3)
        return self.vec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    help="ONNX policy to run, or 'zero' (no network: PD hold at the nominal pose). "
                         "With --stand-policy this is the walking policy")
    ap.add_argument("--profile", default="rl_default",
                    help="PD gain profile (gain_profiles in biped_Laika/biped_Laika.yaml)")
    ap.add_argument("--policy-profile", default=None,
                    help="policy I/O contract profile (policy_profiles in biped_Laika/biped_Laika.yaml), e.g. walk. "
                         "A stamped ONNX carries its profile, so this can be omitted (if given, it must match the manifest)")
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="accept an ONNX without a manifest (the contract then rests on the command-line flags alone; "
                         "at your own risk on the real robot)")
    ap.add_argument("--allow-power-override", action="store_true",
                    help="allow --power-scale != 1.0 with a profile that declares per-joint action scales "
                         "(refused by default: the amplitude would be reduced twice, see tools/policy_io.py)")
    ap.add_argument("--allow-plant-drift", action="store_true",
                    help="accept a policy whose manifest records plant parameters that differ from the current config "
                         "(for experiments only)")
    ap.add_argument("--dump-frames", type=int, default=0, metavar="K",
                    help="contract test: save the first K real observations and the network outputs computed on the "
                         "robot computer to --dump-path; replay and compare them on your computer with "
                         "tools/contract_check.py --replay")
    ap.add_argument("--dump-path", default="obs_dump.npz", help="output file for --dump-frames")
    ap.add_argument("--onehot-real", action="store_true",
                    help="joint check on the hoisted robot: command +0.1 on one action at a time and verify "
                         "direction / index / amplitude from the measured motion (no policy runs; use with --policy zero --no-imu)")
    ap.add_argument("--onehot-hold", type=float, default=1.2, metavar="S",
                    help="hold time of each one-hot step in seconds (default 1.2)")
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="constant velocity command: vx vy (m/s) wz (rad/s); without teleop")
    ap.add_argument("--teleop", action="store_true",
                    help="keyboard teleop: W/S forward / back, A/D side-step left / right, Q/E turn left / right, "
                         "space = zero the command, x = quit. Each key press commands --teleop-hold seconds, then the "
                         "command returns to zero; no key = zero command. Replaces --cmd (do not combine with --heading)")
    ap.add_argument("--teleop-speed", type=float, default=0.20, metavar="V",
                    help="teleop forward / back speed in m/s (default 0.20)")
    ap.add_argument("--teleop-hold", type=float, default=1.0, metavar="S",
                    help="how long one key press lasts, in seconds (default 1.0)")
    ap.add_argument("--teleop-turn", type=float, default=0.20, metavar="W",
                    help="teleop turn rate in rad/s (default 0.20)")
    ap.add_argument("--teleop-side", type=float, default=0.10, metavar="V",
                    help="teleop side-step speed in m/s (default 0.10). It has its own, lower setting: the clearance "
                         "between the two feet shrinks quickly as the lateral speed grows, and at higher values the "
                         "feet come close to touching. Do not simply set it to the forward speed")
    ap.add_argument("--pad", nargs="?", const="/dev/input/js0", default=None, metavar="DEV",
                    help="gamepad teleop (DualShock 4 / DualSense, Bluetooth or USB; default device /dev/input/js0) "
                         "instead of the --teleop keyboard: left stick walk / side-step, right stick turn, Options = stand, "
                         "Share = stop, PS = quit; hold R1 to move (dead-man button). Full stick deflection = "
                         "--teleop-speed / --teleop-side / --teleop-turn. Needs no terminal, so it also runs under "
                         "nohup / tmux. See tools/pad_teleop.py")
    ap.add_argument("--pad-no-deadman", action="store_true",
                    help="turn off the R1 dead-man button of --pad (the sticks command directly). Not recommended on the real robot")
    ap.add_argument("--pad-map", default="auto", choices=["auto", "ds", "generic"],
                    help="gamepad joydev numbering: ds = hid-playstation / hid-sony driver; generic = hid-generic (kernels built "
                         "without the PlayStation drivers, e.g. the stock Jetson kernel, over USB and Bluetooth alike: Square is "
                         "button 0, right stick X is axis 2). auto = detect from the driver, else from the button count (>= 15 -> generic)")
    ap.add_argument("--duration", type=float, default=0.0, help="run time in seconds; 0 = run until Ctrl+C")
    ap.add_argument("--no-imu", action="store_true",
                    help="run without the IMU (hoisted smoke test): angular velocity = 0, gravity = [0, 0, -1]")
    ap.add_argument("--cycle-time", type=float, default=None, metavar="S",
                    help="override the gait clock period in seconds (cycle_time_s of the policy profile in "
                         "biped_Laika/biped_Laika.yaml). Larger = slower steps. The policy is trained at the configured "
                         "period and any other value is out of distribution: sweep it in simulation before using it "
                         "on the real robot")
    ap.add_argument("--heading", action="store_true",
                    help="heading hold for a single walking policy: the yaw-rate command is computed from the heading "
                         "error (wz = clip(k * err, ±0.6)), with the target locked to the heading at start. "
                         "This P loop is added on the deploy side; the policy is not trained with it. "
                         "Yaw comes from gyro integration and drifts slowly, "
                         "so use it for runs of a few tens of seconds only")
    ap.add_argument("--heading-k", type=float, default=0.5,
                    help="heading gain k of --heading / --heading-hold (default 0.5)")
    ap.add_argument("--heading-hold", action="store_true",
                    help="heading hold while walking in stand + walk mode: while a W/A/S/D command is active and Q/E is "
                         "not, a yaw-rate correction from the IMU yaw is added (gain --heading-k, limited to ±0.3 rad/s); "
                         "releasing a turn key locks the new heading. Unlike --heading it leaves the raw command alone, "
                         "so standing is not mistaken for a walk request")
    ap.add_argument("--trim-vy", type=float, default=0.0, metavar="V",
                    help="lateral trim while walking in stand + walk mode, m/s: added to vy when walking forward "
                         "(negative = to the right). Use it to cancel a steady sideways drift, e.g. --heading-hold with "
                         "--trim-vy -0.03 for a drift of about 0.03 m/s to the left; tune it on the real robot")
    ap.add_argument("--power-scale", type=float, default=1.0, metavar="K",
                    help="multiply the policy output by K before converting it to joint targets (0.1 = 10%%). For a first run "
                         "or a new policy, start at 0.1 to see the motion pattern and raise it step by step once it looks safe. "
                         "With K < 1 the robot mostly stays at the nominal pose and does not walk; "
                         "that is intended")
    ap.add_argument("--max-tilt-deg", type=float, default=45.0, help="torso tilt in degrees above which the robot is stopped (damping, then disabled)")
    # -- state machine: enabled by --stand-policy; --policy is then the walking (LOCO) policy --
    ap.add_argument("--stand-policy", default=None, metavar="ONNX",
                    help="standing policy; enables the stand + walk state machine, with --policy as the walking policy. "
                         "The robot stands by default, walks while a movement command (W/A/S/D/Q/E or the sticks) is active, "
                         "and returns to standing --walk-release seconds after the command goes back to zero")
    ap.add_argument("--home", "--start-state", dest="home", default="auto", choices=["auto", "stand", "loco"],
                    help="home state of the stand + walk state machine: where it starts and where it returns to. "
                         "auto / stand = standing: a movement command starts walking (after a short transition between "
                         "the two nominal poses) and the robot returns to standing --walk-release seconds after the "
                         "command goes back to zero. loco = walking: switch with key 1 (stand) / 2 (walk)")
    ap.add_argument("--no-auto-walk", action="store_true",
                    help="turn off automatic walking: switch explicitly with key 1 = stand / 2 = walk (gamepad: Options / △); "
                         "in the walking state a zero command steps in place instead of returning to standing")
    ap.add_argument("--walk-release", type=float, default=1.0, metavar="S",
                    help="automatic walking: return to standing S seconds after the command goes back to zero (default 1.0; "
                         "a teleop key press itself lasts --teleop-hold)")
    ap.add_argument("--fsm-script", default=None, metavar="T:KEY,...",
                    help="scripted key presses for the state machine (hoisted / unattended tests), e.g. '3:w,8:1' = "
                         "press w at 3 s and 1 at 8 s. KEY is 1 (stand), 2 (walk) or one of w/a/s/d/q/e; "
                         "times count from the start of the control loop")
    ap.add_argument("--skip-torque-check", action="store_true",
                    help="skip the startup check of the torque limit and CAN timeout stored in the motors "
                         "(only for debugging the bus itself)")
    ap.add_argument("--log", metavar="CSV",
                    help="record target / measured / action for every control step to a CSV file and print a per-joint "
                         "summary on exit (for diagnosis)")
    args = ap.parse_args(argv)
    if args.log:
        # create the log directory now: a missing directory would only fail when the CSV is written at exit
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)

    cfg = yaml.safe_load(open(os.path.join(ROOT, "biped_Laika/biped_Laika.yaml")))
    if not args.no_imu and sys.platform.startswith("linux"):
        from tools.check_ports import assert_no_conflict
        assert_no_conflict(cfg)          # report a serial-port conflict (IMU vs CAN) before anything is opened
    # Manifest check. On the real robot a plant-parameter mismatch is fatal as well (--allow-plant-drift overrides).
    from tools.onnx_manifest import enforce
    pprof = enforce(args.policy, cfg, args.policy_profile,
                    strict_plant=not args.allow_plant_drift,
                    allow_unstamped=args.allow_unstamped)
    io = PolicyIO(cfg, pprof)
    # The action amplitude may be reduced in one place only. A profile with per-joint action scales
    # (action_scale_joint) already carries the reduction in its contract; a global --power-scale on top would
    # reduce it twice and leave the robot visibly weak, and the manifest cannot catch a run-time option.
    if (not np.allclose(io.action_scale_vec, io.action_scale)
            and abs(args.power_scale - 1.0) > 1e-6 and not args.allow_power_override):
        sys.exit(f"[err] profile {pprof!r} already declares per-joint action scales "
                 f"({sorted(set(round(float(x), 3) for x in io.action_scale_vec))}) in its contract, "
                 f"so --power-scale must be 1.0 (got "
                 f"{args.power_scale}); scaling again would reduce the amplitude twice.\n"
                 f"    To sweep the amplitude anyway, add --allow-power-override (at your own risk on the real robot).")
    # Gain profile: if the policy contract declares a gain_profile, --profile must match it (the gains are part of the plant).
    if io.gain_profile and args.profile != io.gain_profile:
        sys.exit(f"[err] policy profile {pprof!r} declares gain profile {io.gain_profile}, but --profile is {args.profile}; "
                 f"add --profile {io.gain_profile}")
    if (args.heading_hold or args.trim_vy) and not args.stand_policy:
        sys.exit("[err] --heading-hold / --trim-vy need the stand + walk state machine (--stand-policy); with a single walking policy use --heading")
    if args.cycle_time:
        print(f"[deploy] gait clock period "
              f"{io.cycle_time:.3f}s -> {args.cycle_time:.3f}s "
              f"({1/io.cycle_time:.2f} -> {1/args.cycle_time:.2f} Hz)")
        io.cycle_time = args.cycle_time
    policy = load_policy(args.policy, io.obs_dim)
    # -- State machine: every loaded policy has its own manifest and PolicyIO (contracts are not shared).
    # --policy is always the walking (LOCO) policy. --
    global FSM_LAST
    fsm = assist = None
    if args.stand_policy:
        from tools.skill_fsm import Slot, SkillFSM
        slots = {"LOCO": Slot("LOCO", io, policy, args.power_scale, args.policy)}

        def _mount(path, name):
            sprof = enforce(path, cfg, None, strict_plant=not args.allow_plant_drift,
                            allow_unstamped=args.allow_unstamped)
            sio = PolicyIO(cfg, sprof)
            if sio.names != io.names or sio.rate != io.rate:
                sys.exit(f"[err] {name} policy: joint order / control rate differ from the walking (LOCO) policy: {path}")
            if sio.gain_profile and sio.gain_profile != args.profile:
                sys.exit(f"[err] {name} policy declares gain profile {sio.gain_profile}, which differs from --profile {args.profile} "
                         f"(the state machine runs one set of gains throughout): {path}")
            slots[name] = Slot(name, sio, load_policy(path, sio.obs_dim), 1.0, path)
            print(f"[fsm] {name:<9} loaded {os.path.basename(path)} (profile={sprof or 'default (standing)'})")

        _mount(args.stand_policy, "STAND")
        start = "STAND" if args.home == "auto" else args.home.upper()
        auto_walk = start == "STAND" and not args.no_auto_walk
        fsm = SkillFSM(slots, start, home=start, auto_walk=auto_walk, walk_release_s=args.walk_release,
                       log=lambda m: print("\r" + m + "        ", flush=True))
        if args.heading:
            sys.exit("[err] --heading cannot be used with the state machine (it rewrites the raw command, "
                     "so standing would be taken for a walk request); use --heading-hold")
        from tools.skill_fsm import LocoAssist
        assist = (LocoAssist(k=args.heading_k, trim_vy=args.trim_vy, hold=args.heading_hold)
                  if (args.heading_hold or args.trim_vy) else None)
        if assist is not None:
            print(f"[fsm] walking assist: heading hold {'on (k=%.2f)' % args.heading_k if args.heading_hold else 'off'}, "
                  f"lateral trim when walking forward vy {args.trim_vy:+.3f} m/s")
        FSM_LAST = fsm
        print(f"[fsm] slots {sorted(slots)} home {start}"
              f"{' (automatic walking: a movement command walks, back to standing %.1fs after release)' % fsm.walk_release_s if auto_walk else ''}; "
              f"--power-scale applies to LOCO only")
    elif args.fsm_script:
        sys.exit("[err] --fsm-script needs the stand + walk state machine (add --stand-policy)")
    if pprof:
        # The nominal pose of this profile is not the straight standing pose (walking uses a crouch): once enabled,
        # the robot lowers itself into it. Take up the slack in the hoist first.
        print(f"[deploy] policy contract profile={pprof}: "
              f"action_scale={io.action_scale}, the nominal pose is not the straight standing pose")
    lb.MIRROR_KP = {m: g[0] for m, g in lb.GAIN_PROFILES[args.profile].items()}
    lb.MIRROR_KD = {m: g[1] for m, g in lb.GAIN_PROFILES[args.profile].items()}
    lb.JOINT_KP = {j: g[0] for j, g in lb.GAIN_OVERRIDES[args.profile].items()}    # per-joint gain overrides of the profile
    lb.JOINT_KD = {j: g[1] for j, g in lb.GAIN_OVERRIDES[args.profile].items()}
    print(f"[deploy] gains {args.profile}: kp={lb.MIRROR_KP} kd={lb.MIRROR_KD}"
          + (f" per-joint overrides kp={lb.JOINT_KP} kd={lb.JOINT_KD}" if lb.JOINT_KP else ""))

    imu = StaticImu() if args.no_imu else None
    if imu is None:
        from tools.imu_reader import ImuReader
        imu = ImuReader(cfg["imu"])
        time.sleep(2.0)                      # let the attitude filter converge

    # Torque-limit self-check: limit_torque in the motor firmware must equal tau_limit in
    # biped_Laika/biped_Laika.yaml, because the simulation clips with the same value; a limit on one side only
    # makes the two diverge silently. This must run before the bus workers take over the serial port.
    if not args.skip_torque_check:
        import robstride_mit as rm
        want = cfg.get("tau_limit") or cfg["tau_max"]
        mismatch = []
        for ch in sorted({j["bus"] for j in cfg["joints"]}):
            js = [j for j in cfg["joints"] if j["bus"] == ch]
            leg = rm.LegBus(ch, [j["id"] for j in js])
            try:
                for j in js:
                    rb = leg.read_float_param(j["id"], rm.IDX_LIMIT_TORQUE)
                    if rb is None or abs(rb - float(want[j["motor"]])) > 0.05:
                        mismatch.append(f"{j['name']}: read back {rb}, expected {want[j['motor']]}")
            finally:
                leg.shutdown()
        if mismatch:
            sys.exit("[err] motor limit_torque does not match tau_limit in biped_Laika/biped_Laika.yaml:\n  "
                     + "\n  ".join(mismatch)
                     + "\n  Run python tools/set_torque.py --write --save first")
        print(f"[deploy] torque-limit check ✓ limit_torque of all 12 motors == tau_limit {dict(want)}")
        # Motor-side CAN timeout (can_timeout in the config): the only dead-man left when the MC02 board loses
        # power or its USB cable is pulled. The value stored in each motor must equal the config, otherwise the
        # legs keep holding the last target after the link is lost. Skipped when the config has no can_timeout.
        from tools.set_can_timeout import check_all as _ct_check
        ct_bad = _ct_check(cfg, verbose=False)
        if ct_bad:
            sys.exit("[err] motor canTimeout does not match can_timeout in biped_Laika/biped_Laika.yaml:\n  "
                     + "\n  ".join(ct_bad)
                     + "\n  Run python tools/set_can_timeout.py --write --save first")
        if cfg.get("can_timeout") is not None:
            print(f"[deploy] motor CAN-timeout check ✓ canTimeout == {cfg['can_timeout']} ({cfg['can_timeout'] * 50 / 1000:.0f} ms)")

    buses = {}
    for e in lb.JOINT_MAP:
        buses.setdefault(e[2], []).append(e)
    workers = []
    for ch, entries in buses.items():
        w = lb.BusWorker(ch, entries)
        w.start()
        workers.append(w)
    time.sleep(0.3)

    with lb.STATE.lock:
        lb.STATE.active = set(io.names)
        lb.STATE.limits = {n: (float(lo), float(hi))
                           for n, lo, hi in zip(io.names, io.jlo, io.jhi)}
        lb.STATE.mode = "send"
        # Raise the bridge speed limit to each motor's vel_limit (takes effect once homing is done, see
        # bridge.FAST_SPD). The 1 rad/s default meant for manual teleoperation would cut a walking gait down to
        # a small, phase-lagged shuffle.
        lb.STATE.fast_slew = True
    print(f"[deploy] bridge speed limit: raised to {lb.FAST_SPD} rad/s after homing (teleoperation default {lb.SOFT_SPD})")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        with lb.STATE.lock:
            if lb.STATE.ready >= lb.STATE.active:
                break
        time.sleep(0.05)
    with lb.STATE.lock:
        missing = lb.STATE.active - lb.STATE.ready
    if missing:
        lb.safe_idle(f"enable failed: {sorted(missing)}")
        time.sleep(1.0)
        sys.exit(f"[err] these joints did not enable: {sorted(missing)}")
    print(f"[deploy] 12 joints enabled, policy running at {io.rate}Hz (Ctrl+C = safe stop)")

    if args.onehot_real:
        try:
            ok = onehot_real(io, workers, args.onehot_hold)
        finally:
            lb.safe_idle("one-hot sequence finished")
            time.sleep(1.0)
        sys.exit(0 if ok else 1)

    cmd = np.array(args.cmd)
    last_act = np.zeros(12)
    dt = 1.0 / io.rate
    n_steps = 0
    _keymap = {"1": "stand", "2": "loco"}
    # scripted movement keys (without teleop: emulate one W/A/S/D/Q/E press = a command for --teleop-hold seconds)
    _movemap = {"w": (args.teleop_speed, 0, 0), "s": (-args.teleop_speed, 0, 0),
                "a": (0, args.teleop_side, 0), "d": (0, -args.teleop_side, 0),
                "q": (0, 0, args.teleop_turn), "e": (0, 0, -args.teleop_turn)}
    script_cmd, script_until = np.zeros(3), -1.0
    fsm_sched = sorted((float(t), _keymap.get(k.strip(), k.strip())) for t, k in
                       (tok.split(":") for tok in (args.fsm_script or "").split(",") if tok.strip()))
    stand_hint = [False]
    t_loop = time.monotonic()
    imu_backlog0 = getattr(imu, 'backlog_events', 0)
    yaw0 = imu.yaw() if (args.heading and hasattr(imu, "yaw")) else 0.0
    if args.heading:
        print(f"[deploy] heading hold on (k={args.heading_k}); the target heading is locked to the current one")
    # The policy is trained at exactly the control rate; running slower silently changes the dynamics it sees.
    # The achieved rate is therefore checked once per second.
    rate_t, rate_n, rate_warned = t_loop, 0, False
    rec = [] if args.log else None
    dump = {"obs": [], "out": []} if args.dump_frames > 0 else None
    tele = None
    if args.teleop or args.pad:
        if args.pad:
            from tools.pad_teleop import PadTeleop
            tele = PadTeleop(args.teleop_speed, args.teleop_hold, args.teleop_turn,
                             args.teleop_side, dev=args.pad, deadman=not args.pad_no_deadman, pad_map=args.pad_map)
            tele.__enter__()
            print(f"[pad] gamepad {args.pad} '{tele.name}' (driver {tele.driver or '?'}, {tele.n_axes} axes, "
                  f"{tele.n_buttons} buttons, numbering {tele.map_name}): "
                  f"left stick forward / back {args.teleop_speed:.2f} m/s, side-step {args.teleop_side:.2f} m/s; "
                  f"right stick turn {args.teleop_turn:.2f} rad/s (full deflection); Options = stand / △ = walk / "
                  f"Share = stop / PS = quit; dead-man R1 {'off' if args.pad_no_deadman else 'on (hold R1 to move)'}; Ctrl+C = safe stop")
        else:
            tele = KeyTeleop(args.teleop_speed, args.teleop_hold, args.teleop_turn,
                             args.teleop_side)
            tele.__enter__()
        if not args.pad:
            print(f"[teleop] W forward / S back / A side-step left / D side-step right / Q turn left / E turn right - "
                  f"each press commands "
                  f"{args.teleop_hold:.1f}s × {args.teleop_speed:.2f} m/s "
                  f"(side-step {tele.side:.2f} m/s, turn {args.teleop_turn:.2f} rad/s); "
                  f"space = zero the command, x = quit, Ctrl+C = safe stop")
            print("[teleop] note: A/D side-step (lateral velocity); "
                  "use Q/E to turn")
        print("[teleop] no input = zero command. A walking policy on its own keeps stepping in place at zero command "
              "and drifts slowly: keep clear space around the robot")
        if fsm is not None:
            if fsm.auto_walk:
                print(f"[teleop] stand + walk: the robot stands by default; a movement command (W/A/S/D/Q/E or the sticks) "
                      f"walks, after a {fsm.BLEND_S:.1f}s transition to the walking policy; "
                      f"back to standing {fsm.walk_release_s:.1f}s after release"
                      f"; 1 = stand now, 2 = walk (back to standing after {fsm.walk_release_s:.1f}s of zero command)")
            elif fsm.home == "LOCO":
                print("[teleop] walking home: W/A/S/D/Q/E or the sticks walk")
            else:
                print("[teleop] 1 = stand / 2 = walk; "
                      "W/A/S/D/Q/E and the sticks only act in the walking state")
    try:
        while args.duration <= 0 or time.monotonic() - t_loop < args.duration:
            tick = time.monotonic()
            if tele is not None:
                if tele.poll(tick):
                    print("\n[teleop] quit key received, exiting")
                    break
                cmd[0], cmd[1], cmd[2] = tele.command(tick)
            dead = [w.channel for w in workers if not w.alive]
            if dead:
                raise RuntimeError(f"bus worker died: {dead} (that leg is no longer under control)")
            ang_vel, gravity, age = imu.latest()
            if age > 0.1:
                raise RuntimeError(f"IMU data stalled for {age*1000:.0f}ms")
            # age only tells when the last frame was parsed. It cannot see a serial backlog, where frames keep
            # arriving but all of them are old; the backlog counter of imu_reader covers that case.
            if getattr(imu, "backlog_events", 0) > imu_backlog0 + 20:
                raise RuntimeError(f"IMU serial backlog: {imu.backlog_events} events "
                                   f"(the attitude is badly delayed; the fusion thread cannot keep up with 500Hz)")
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
            if tilt > args.max_tilt_deg:
                raise RuntimeError(f"tilt {tilt:.0f}° is over the limit (fall protection)")
            if args.heading and hasattr(imu, "yaw"):
                _e = imu.yaw() - yaw0
                _e = float(np.arctan2(np.sin(-_e), np.cos(-_e)))
                cmd[2] = float(np.clip(args.heading_k * _e, -0.6, 0.6))
            with lb.STATE.lock:
                q = np.array([lb.STATE.q[n] for n in io.names])
                qd = np.array([lb.STATE.qd.get(n, 0.0) for n in io.names])
            # Gait clock: t counts from the start of the control loop, matching the episode start in simulation.
            # It uses the monotonic clock, not the step count: if the loop drops steps, the clock must not slow down.
            if fsm is not None:
                # ---- state machine (tools/skill_fsm.py) ----
                # key / script requests -> state update (transitions) -> inference with the policy of the current state
                t_now = tick - t_loop
                reqs = []
                if tele is not None:
                    reqs, tele.reqs = list(tele.reqs), []
                while fsm_sched and t_now >= fsm_sched[0][0]:
                    _k = fsm_sched.pop(0)[1]
                    if _k in _movemap:
                        script_cmd, script_until = np.array(_movemap[_k], float), t_now + args.teleop_hold
                    else:
                        reqs.append(_k)
                if tele is None and args.fsm_script:
                    cmd[:] = script_cmd if t_now < script_until else 0.0
                for r in reqs:
                    fsm.request(r, t_now, q)
                fsm.update(t_now, q, cmd)
                if fsm.state == "STAND" and not fsm.auto_walk and np.any(np.abs(cmd) > 1e-6) \
                        and not stand_hint[0]:
                    print("\n[fsm] W/A/S/D/Q/E have no effect in STAND - press 2 to walk")
                    stand_hint[0] = True
                act_cmd = cmd
                if assist is not None:
                    act_cmd = assist(fsm.state, cmd, imu.yaw() if (assist.hold and hasattr(imu, "yaw")) else None)
                target, raw_out = fsm.act(t_now, ang_vel, gravity, act_cmd, q, qd)
                a = fsm.last_act.copy()
                last_act = a
            else:
                obs = io.build_obs(ang_vel, gravity, cmd, q, qd, last_act, phase=io.phase_of(tick - t_loop))
                raw_out = policy(obs)
                if dump is not None and len(dump["obs"]) < args.dump_frames:
                    # Contract test: save the raw network output (before clip / power scale) and the float32
                    # observation (bit-identical to what onnxruntime received), for replay on another machine.
                    dump["obs"].append(np.asarray(obs, dtype=np.float32))
                    dump["out"].append(np.asarray(raw_out, dtype=np.float32))
                    if len(dump["obs"]) == args.dump_frames:
                        from tools.onnx_manifest import _sha16
                        np.savez(args.dump_path, obs=np.stack(dump["obs"]),
                                 out=np.stack(dump["out"]),
                                 policy=os.path.basename(args.policy),
                                 model_sha16=_sha16(args.policy), profile=pprof or "")
                        print(f"[contract] saved {args.dump_frames} real observations + network outputs -> "
                              f"{args.dump_path}; compare on your computer with tools/contract_check.py --replay")
                a = np.clip(raw_out, -io.action_clip, io.action_clip) * args.power_scale
                last_act = np.asarray(a, dtype=float)
                # Profiles with side-step feed-forward need cmd and the gait phase: pass the very phase value
                # that build_obs received (same tick); arguments a profile does not use have no effect.
                target = io.action_to_target(a, cmd=cmd, phase=io.phase_of(tick - t_loop))
            with lb.STATE.lock:
                for n, v in zip(io.names, target):
                    lb.STATE.target[n] = float(v)
            if rec is not None:
                rec.append((tick - t_loop, gravity.copy(), q.copy(), target.copy(), a.copy()))
            n_steps += 1
            rate_n += 1
            now = time.monotonic()
            if now - rate_t >= 1.0:
                hz = rate_n / (now - rate_t)
                if hz < io.rate * 0.9:
                    print(f"[deploy] !! measured {hz:.1f}Hz < target {io.rate}Hz "
                          f"- the policy is trained at {io.rate}Hz; a lower rate changes the dynamics it sees")
                    rate_warned = True
                rate_t, rate_n = now, 0
            time.sleep(max(0.0, dt - (now - tick)))
    except KeyboardInterrupt:
        print("\n[deploy] Ctrl+C")
    except Exception as e:
        print(f"\n[deploy] !! {e}")
    finally:
        if tele is not None:
            tele.__exit__()              # restore the terminal first, otherwise the prints below come out garbled
        # Stop the clock before shutting down, otherwise the sleep(1.0) below would count into the average rate.
        wall = time.monotonic() - t_loop
        # Motor temperatures at exit. The bridge only prints a warning above 75 °C; this report shows all values.
        # The two hip_roll motors run warmest by design: in single support they carry the lateral moment of the
        # body weight (about 9 N·m peak, 4 N·m on average), which is most of the steady-state copper loss of the
        # whole robot. Lower gains do not change that; manage it with duty cycle (cool-down pauses between runs).
        try:
            with lb.STATE.lock:
                temps = dict(lb.STATE.temp)
            if temps:
                hot = sorted(temps.items(), key=lambda kv: -kv[1])
                print("\nMotor temperatures at exit (°C):")
                for n, t in hot:
                    flag = "  <-- high" if t >= 60.0 else ("  !! overheated" if t >= 75.0 else "")
                    print(f"  {n:<24}{t:>6.1f}{flag}")
                mx = hot[0]
                print(f"  highest {mx[0]} {mx[1]:.1f}°C"
                      + ("  - hip_roll motors running warmer is expected (lateral moment in single support); "
                         "allow cool-down pauses between runs" if "hip_roll" in mx[0] else ""))
        except Exception as e:
            print(f"[temp] readings not available: {e}")
        lb.safe_idle("deploy exit")
        time.sleep(1.0)                      # let the workers finish damping -> disable
        print(f"[deploy] stopped safely; {n_steps} steps in {wall:.1f}s wall time, "
              f"average {n_steps/max(wall,1e-6):.1f}Hz (target {io.rate}Hz)"
              + ("  !! the rate dropped below the target during the run" if rate_warned else ""))
        if rec:
            summarize(rec, io, args.log)


if __name__ == "__main__":
    main()
