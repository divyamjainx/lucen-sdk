#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/policy_gain_probe.py - measure a biped policy's own feedback gains to see how close it is to self-excited oscillation.

A policy can pass every simulation check and still shake audibly on the real robot. MuJoCo has none of the
velocity noise, structural compliance and actuator dynamics of the hardware, so it cannot show how far from
the edge a policy is. This tool measures the policy itself, at the operating points it visits during a
tools/sim_fsm.py run:
  - last_action self-loop: eigenvalues of the mean of d a / d a_prev (action units). The closer the most
    negative real part is to -1, the less a sign flip on every control step decays in the software loop
    alone (at the 50 Hz control rate that is a 25 Hz oscillation);
  - equivalent dq damping: kp * d target / d dq (N·m/(rad/s)), i.e. the derivative control the policy does
    through its joint-velocity observation. It acts one control step late, so a large gain amplifies noise;
  - mean Lipschitz: mean squared change of the action when the proprioceptive observations (gyro, gravity,
    q, dq, last_action) are perturbed by the amounts in SIGMA;
  - raw action: sign-flip rate per control step and energy of the second difference.
How to read the result: a last_action self-loop eigenvalue close to -1 together with a high per-step
sign-flip rate means the policy is near a 25 Hz oscillation on real hardware, even if the simulation looks
fine. Every policy loaded during the run is reported on its own line.

Usage: takes the same arguments as tools/sim_fsm.py (it wraps that tool's policy loader to record what each
policy sees and outputs), for example:
  python tools/policy_gain_probe.py --policy my_policy.onnx --stand-policy biped_Laika/policies/stand.onnx \\
      --delay 1 --jitter 0.05 --script 3:w6 --duration 12 --quiet
"""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gains  # noqa: E402
import tools.sim_fsm as S  # noqa: E402
from tools.onnx_manifest import enforce  # noqa: E402
from tools.policy_io import PolicyIO  # noqa: E402

#: standard deviation of the perturbation added to each proprioceptive observation (physical units)
SIGMA = {"base_ang_vel": 0.3, "projected_gravity": 0.02, "joint_pos_rel": 0.01, "joint_vel": 1.0, "last_action": 0.05}


def probe(path, obs, act, cfg, kp, stride=4, seed=0):
    io = PolicyIO(cfg, enforce(path, cfg, None))
    off, o = {}, 0
    for s in io.spec:
        sc = np.broadcast_to(np.asarray(s.get("scale", 1.0), float), (s["dim"],))
        off[s["name"]] = (slice(o, o + s["dim"]), sc)
        o += s["dim"]
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def f(x):                                     # the exported graph has a fixed batch size of 1: run row by row
        x = np.atleast_2d(x).astype(np.float32)
        return np.concatenate([sess.run(None, {iname: r[None]})[0][:, :12] for r in x])

    asc = np.asarray(io.action_scale, float) * np.ones(12)
    X = obs[20:][::stride]
    la, lsc = off["last_action"]
    dq, dsc = off["joint_vel"]
    rng = np.random.default_rng(seed)
    L, D, lip = [], [], []
    for x in X:
        h = 0.02
        P = np.repeat(x[None], 48, 0)
        for j in range(12):
            P[j, la.start + j] += h
            P[12 + j, la.start + j] -= h
            P[24 + j, dq.start + j] += h
            P[36 + j, dq.start + j] -= h
        y = f(P)
        L.append(((y[:12] - y[12:24]) / (2 * h)).T)                        # [out, in], action units
        D.append(np.diagonal((y[24:36] - y[36:48]) / (2 * h)) * dsc * asc)  # rad of target per rad/s
        xn = x.copy()
        for name, sig in SIGMA.items():
            if name in off:
                sl, sc = off[name]
                xn[sl] += rng.normal(0.0, sig, sl.stop - sl.start) * sc
        lip.append(float(((f(xn) - f(x)) ** 2).mean()))
    ev = np.linalg.eigvals(np.mean(L, 0))
    damp = -np.mean(D, 0) * kp                     # positive = the policy adds damping (the target moves against the velocity)
    a = np.clip(act[20:], -io.action_clip, io.action_clip)
    d1, d2 = np.diff(a, axis=0), np.diff(a, 2, axis=0)
    return dict(names=io.names, n=len(X), eig_min=float(ev.real.min()), eig_rad=float(np.abs(ev).max()), damp=damp,
                lip=float(np.mean(lip)), flip=float((d1[1:] * d1[:-1] < 0).mean()),
                d2e=float((d2 ** 2).sum(1).mean()))


def main():
    rec = {}
    load = S.load_policy

    def recording(path, obs_dim):
        fn = load(path, obs_dim)

        def g(obs):
            out = fn(obs)
            rec.setdefault(path, []).append((np.array(obs, float), np.array(out, float)))
            return out
        return g
    S.load_policy = recording
    try:
        S.main()
    except SystemExit:
        pass
    cfg = yaml.safe_load(open(os.path.join(ROOT, "biped_Laika/biped_Laika.yaml")))
    g = gains.profile("rl_default")
    print("\n== policy gain probe (operating points of the sim_fsm run) ==")
    for path, v in rec.items():
        if len(v) < 60:
            print(f"  {os.path.basename(path)}: only {len(v)} control steps, skipped")
            continue
        obs = np.stack([o for o, _ in v])
        act = np.stack([a for _, a in v])
        io_names = PolicyIO(cfg, enforce(path, cfg, None)).names
        r = probe(path, obs, act, cfg, np.array([g[n]["kp"] for n in io_names]))
        j = int(np.argmax(np.abs(r["damp"])))
        print(f"  {os.path.basename(path):34s} {r['n']:4d} points | last_action self-loop: most negative real part {r['eig_min']:+.2f} (spectral radius {r['eig_rad']:.2f}) "
              f"| equivalent dq damping, peak {r['damp'][j]:+.2f} N·m/(rad/s) @{r['names'][j].replace('_joint', '')} "
              f"| mean Lipschitz {r['lip']:.4f} | raw action: sign flips {r['flip'] * 100:.0f}%, Δ² energy {r['d2e']:.3f}")
        print("    equivalent dq damping (N·m/(rad/s)): " + " ".join(f"{n.replace('_joint', '')[:9]} {d:+.2f}"
                                                   for n, d in zip(r["names"], r["damp"])))


if __name__ == "__main__":
    main()
