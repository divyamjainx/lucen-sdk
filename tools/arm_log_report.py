#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/arm_log_report.py - read a session log and report how closely the arm followed its targets.

Reads the CSV written by --log of the arm front ends, or by BridgeClient(log=...) in your own program
(tools/arm_jog.py). It only reads files: it never connects to a bridge and runs on any computer.

For every joint it compares the target that was sent with the angle the bridge reported. Through the robot model
it also compares where the targets put the hand (the target pose) with where the measured angles put it (the
reached pose). Three numbers are given for each:
  while moving    the error over the whole run. Most of it is delay: the bridge moves the commanded position at no
                  more than 1 rad/s, and the position loop follows its command with a lag
  delay removed   the same after shifting each joint's targets by the delay estimated from the data
  at rest         the error once all targets have stood still for a second: what the gains and the gravity
                  feed-forward leave. It grows with the load on the joint

--poses writes the target pose and the reached pose of every row to a CSV (position in metres, orientation as a
quaternion w x y z, in the world frame of the scene file named by model.scene in the config), for example to
compare where a model aimed with where your perception saw the object. With t0_unix from the first line of the
log, t + t0_unix is the wall-clock time of a row.

What a log cannot show: an error in a joint's zero position. The reached pose is computed from the encoder
angles; if the calibration is off, the real hand is somewhere else, and only a measurement from outside shows it.
Rows are assumed to be evenly spaced (they are, when targets are sent at a fixed rate).

Usage:
  python tools/arm_log_report.py logs/arm_lead_follow_20261116_101500.csv
  python tools/arm_log_report.py run.csv --config Arm/Arm_pair.yaml --poses run_poses.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REST_HOLD = 1.0       # s: the targets must have stood still this long before a row counts as "at rest"
REST_TOL = 0.003      # rad: "still" = no target moved more than this within REST_HOLD
MOVED_MIN = 0.05      # rad: a joint's delay is estimated only if its target moved at least this much
DELAY_MAX = 0.6       # s: longest delay searched


def load(path):
    """-> (meta, joint names, t (N,), cmd (N, J) with nan where no target was sent, q (N, J))"""
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        meta = json.loads(first[1:]) if first.startswith("#") else {}
        head = (f.readline() if first.startswith("#") else first).strip().split(",")
        rows = [r for r in (line.rstrip("\n").split(",") for line in f) if len(r) == len(head)]   # a run that was cut off may end with half a row
    names = [h[4:] for h in head if h.startswith("cmd:")]
    if not rows or not names or head[0] != "t":
        sys.exit(f"{path}: no data rows, or not a log written by --log / BridgeClient(log=...)")
    col = {h: i for i, h in enumerate(head)}
    num = lambda s: float(s) if s else np.nan                                 # noqa: E731
    mat = lambda k: np.array([[num(r[col[f"{k}:{n}"]]) for n in names] for r in rows])     # noqa: E731
    return meta, names, np.array([float(r[0]) for r in rows]), mat("cmd"), mat("q")


def hold_last(x):
    """Fill the gaps of every column with the last value before them: a joint without a new target keeps its
    previous one. Rows before the first value stay nan."""
    out = x.copy()
    for j in range(x.shape[1]):
        idx = np.where(np.isfinite(x[:, j]), np.arange(len(x)), -1)
        idx = np.maximum.accumulate(idx)
        out[:, j] = np.where(idx >= 0, x[np.maximum(idx, 0), j], np.nan)
    return out


def shifted(x, n):
    """x delayed by n rows (row k takes x[k - n]; the first n rows repeat the first value)"""
    return x if n <= 0 else np.concatenate([np.repeat(x[:1], n), x[:-n]])


def at_rest(cmd, rows_hold):
    """Rows at which no target moved more than REST_TOL during the last rows_hold rows."""
    n = len(cmd)
    if n <= rows_hold:
        return np.zeros(n, bool)
    win = np.lib.stride_tricks.sliding_window_view(np.nan_to_num(cmd), rows_hold + 1, axis=0)   # (n - rows_hold, J, rows_hold + 1)
    still = ((win.max(axis=2) - win.min(axis=2)) < REST_TOL).all(axis=1)
    return np.concatenate([np.zeros(rows_hold, bool), still]) & np.isfinite(cmd).any(axis=1)


class Hands:
    """End-effector poses from the MuJoCo scene of the config."""

    def __init__(self, cfg):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, cfg["model"]["scene"]))
        self.d = mujoco.MjData(self.m)
        ee = cfg["end_effector"]
        self.sites = [e["site"] for e in (ee if isinstance(ee, list) else [ee])]
        self.sid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, s) for s in self.sites]
        self.adr = {self.m.joint(i).name: int(self.m.jnt_qposadr[i]) for i in range(self.m.njnt)}
        skip = set((cfg.get("model") or {}).get("passive_joints") or [])
        self.driven = [n for n in self.adr if n not in skip]

    def poses(self, names, rows):
        """rows: (N, len(names)) joint angles -> positions (N, S, 3), rotation matrices (N, S, 3, 3). A joint that
        is not in names, or is nan in a row, is left at the model's default angle."""
        cols = [(k, self.adr[n]) for k, n in enumerate(names) if n in self.adr]
        p, R = np.zeros((len(rows), len(self.sid), 3)), np.zeros((len(rows), len(self.sid), 3, 3))
        for i, row in enumerate(rows):
            self.d.qpos[:] = self.m.qpos0
            for k, a in cols:
                if np.isfinite(row[k]):
                    self.d.qpos[a] = row[k]
            self.mj.mj_kinematics(self.m, self.d)
            for s, sid in enumerate(self.sid):
                p[i, s], R[i, s] = self.d.site_xpos[sid], self.d.site_xmat[sid].reshape(3, 3)
        return p, R

    def quats(self, R):
        out, q = np.zeros(R.shape[:2] + (4,)), np.zeros(4)
        for i in range(R.shape[0]):
            for s in range(R.shape[1]):
                self.mj.mju_mat2Quat(q, R[i, s].reshape(9))
                out[i, s] = q
        return out


def angle_between(Ra, Rb):
    """Angle (rad) of the rotation that takes Ra to Rb, per row"""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def peak(x):
    x = x[np.isfinite(x)]
    return float(np.max(np.abs(x))) if len(x) else float("nan")


def fmt(v, spec):
    return format(v, spec) if np.isfinite(v) else "-".rjust(len(format(0.0, spec)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", metavar="CSV", help="file written by --log of an arm front end, or by BridgeClient(log=...)")
    ap.add_argument("--config", metavar="YAML",
                    help="robot config the session used (default: the one recorded in the log); needed for the hand poses")
    ap.add_argument("--poses", metavar="OUT.csv", help="also write the target pose and the reached pose of every row to this file")
    ap.add_argument("--no-hand", action="store_true", help="joint errors only (no MuJoCo needed)")
    args = ap.parse_args()

    meta, names, t, cmd_raw, q = load(args.log)
    cmd = hold_last(cmd_raw)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    if dt <= 0:
        sys.exit(f"{args.log}: fewer than two rows")
    err = cmd - q
    rest = at_rest(cmd, max(1, int(round(REST_HOLD / dt))))
    moved = np.array([np.ptp(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in cmd.T])
    # per-joint delay: the shift of the targets that makes "target - measured" vary least (a constant offset does not matter)
    delay = np.zeros(len(names), int)
    for j in range(len(names)):
        ok = np.isfinite(err[:, j])
        if moved[j] >= MOVED_MIN and ok.sum() > 10:
            spread = [np.std((shifted(cmd[:, j], n) - q[:, j])[ok]) for n in range(int(DELAY_MAX / dt) + 1)]
            delay[j] = int(np.argmin(spread))
    cmd_late = np.stack([shifted(cmd[:, j], delay[j]) for j in range(len(names))], axis=1)
    err_late = cmd_late - q

    cfg_path = args.config or meta.get("config")
    print(f"[report] {args.log}")
    print(f"         {t[-1] - t[0]:.1f} s, {len(t)} rows at {1.0 / dt:.0f} Hz, {len(names)} joints"
          + (f", config {cfg_path}" if cfg_path else "") + (f", started {meta['started']}" if meta.get("started") else ""))
    print(f"         at rest (no target moved for {REST_HOLD:.0f} s): {int(rest.sum())} rows")
    w = max(len(n) for n in names)
    print(f"\n{'joint'.ljust(w)}   moved   while moving      delay   delay removed   at rest")
    print(f"{''.ljust(w)}   (rad)   rms      max      (s)     rms             mean      max      (rad)")
    for j, n in enumerate(names):
        r = err[rest, j]
        r = r[np.isfinite(r)]
        print(f"{n.ljust(w)}   {moved[j]:5.3f}   {fmt(rms(err[:, j]), '6.4f')}   {fmt(peak(err[:, j]), '6.4f')}   "
              f"{(format(delay[j] * dt, '4.2f') if moved[j] >= MOVED_MIN else '   -')}    {fmt(rms(err_late[:, j]), '6.4f')}          "
              f"{fmt(float(np.mean(r)) if len(r) else float('nan'), '+7.4f')}   {fmt(peak(r), '6.4f')}")

    if args.no_hand:
        return
    if not cfg_path:
        sys.exit("\n[report] the log does not name a config: pass --config to get the hand poses (or --no-hand)")
    if not os.path.exists(cfg_path) and os.path.exists(os.path.join(ROOT, cfg_path)):
        cfg_path = os.path.join(ROOT, cfg_path)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    try:
        hands = Hands(cfg)
    except ImportError:
        sys.exit("\n[report] the hand poses need the mujoco package (pip install mujoco), or pass --no-hand")
    grip = {jt["name"] for jt in cfg["joints"] if jt.get("kind") == "gripper"}
    absent = [n for n in hands.driven if n not in names and n not in grip]
    if absent:
        print(f"\n[report] !! not in the log, taken at the model's default angle for both poses: {' '.join(absent)}")
    full = np.isfinite(cmd).any(axis=1)                       # rows from the first target on
    p_cmd, R_cmd = hands.poses(names, np.where(np.isfinite(cmd), cmd, q))
    p_late, _ = hands.poses(names, np.where(np.isfinite(cmd_late), cmd_late, q))
    p_q, R_q = hands.poses(names, q)
    d_mm = np.linalg.norm(p_cmd - p_q, axis=2) * 1e3
    d_late = np.linalg.norm(p_late - p_q, axis=2) * 1e3
    ang = np.degrees(angle_between(R_cmd, R_q))
    for s, site in enumerate(hands.sites):
        r, a = d_mm[rest & full, s], ang[rest & full, s]
        print(f"\nhand (site {site}): distance between the target pose and the reached pose")
        print(f"    while moving    rms {fmt(rms(d_mm[full, s]), '.1f')} mm   max {fmt(peak(d_mm[full, s]), '.1f')} mm")
        print(f"    delay removed   rms {fmt(rms(d_late[full, s]), '.1f')} mm")
        print(f"    at rest         mean {fmt(float(np.mean(r)) if len(r) else float('nan'), '.1f')} mm   max {fmt(peak(r), '.1f')} mm"
              f"   orientation mean {fmt(float(np.mean(a)) if len(a) else float('nan'), '.2f')} deg")

    if args.poses:
        qc, qq = hands.quats(R_cmd), hands.quats(R_q)
        cols = ["t"] + [f"{site}:{kind}:{c}" for site in hands.sites for kind in ("target", "reached")
                        for c in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        os.makedirs(os.path.dirname(os.path.abspath(args.poses)), exist_ok=True)
        with open(args.poses, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for i in np.flatnonzero(full):
                cells = [f"{t[i]:.4f}"]
                for s in range(len(hands.sites)):
                    for p, quat in ((p_cmd[i, s], qc[i, s]), (p_q[i, s], qq[i, s])):
                        cells += [f"{v:.5f}" for v in p] + [f"{v:.5f}" for v in quat]
                f.write(",".join(cells) + "\n")
        print(f"\n[report] poses of {int(full.sum())} rows written to {args.poses}")


if __name__ == "__main__":
    main()
