#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/wheeled_base_export.py - export a wheeled_base rsl_rl checkpoint to a stamped ONNX policy.

  python tools/wheeled_base_export.py <run_dir>/model_N.pt wheeled_base/policies/<name>.onnx   # export and stamp
  python tools/wheeled_base_export.py --show wheeled_base/policies/<name>.onnx                 # print the manifest

Stamping writes a manifest into the ONNX metadata: a snapshot of the policy I/O contract (the part of
wheeled_base/wheeled_base.yaml that defines what the observations and actions mean), the plant parameters
(gains, torque / speed limits, armature, joint friction and damping), the checkpoint path and the git
revision of the working tree. tools/wheeled_base_sim2sim.py compares the manifest with the current config
when it loads a policy: an I/O contract mismatch is fatal (the observations or actions no longer mean the
same thing), a plant mismatch only prints a warning. To try a checkpoint in the middle of training, an
unstamped export from tools/export_ckpt.py is enough (run the simulation tools with --allow-unstamped).
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.wheeled_base_io import WheelbaseIO  # noqa: E402

KEY = "lucen_wheelbase_manifest"


def contract(io):
    """-> (I/O contract, plant parameters) of the current config. A difference in the former means the
    policy's inputs / outputs have a different meaning."""
    p = io.cfg["policy"]
    io_part = {
        "joints": io.names,
        "drive": [j["drive"] for j in io.cfg["joints"]],
        "leg_range": [[float(a), float(b)] for a, b in zip(io.leg_lo, io.leg_hi)],
        "nominal": [float(v) for v in io.nominal],
        "history_len": io.hist_len, "hold_cmd_thresh": io.hold_thresh if io.hold else None,
        "hold_clip": [float(v) for v in io.hold_clip] if io.hold else None,
        "hold_delay_steps": io.hold_delay_steps if io.hold else None,
        "control_rate_hz": p["control_rate_hz"], "action_clip": p["action_clip"],
        "leg_action_scale": p["leg_action_scale"], "wheel_action_scale": p["wheel_action_scale"],
        "obs": io.obs_spec,
    }
    plant = {k: io.cfg[k] for k in ("gains", "tau_limit", "vel_limit", "armature", "joint_friction", "joint_damping")}
    return io_part, plant


def read(path):
    import onnx
    meta = {p.key: p.value for p in onnx.load(path).metadata_props}
    if KEY not in meta:
        return None
    man = json.loads(meta[KEY])
    man["io"].setdefault("history_len", 1)         # manifests without these fields: single frame, no position-hold error
    man["io"].setdefault("hold_cmd_thresh", None)
    man["io"].setdefault("hold_clip", None)
    man["io"].setdefault("hold_delay_steps", None)
    return man


def verify(path, io):
    """-> (fatal problems, warnings), two lists of messages. An ONNX without a manifest counts as fatal."""
    man = read(path)
    if man is None:
        return ["not stamped (export it with tools/wheeled_base_export.py, or pass --allow-unstamped)"], []
    io_part, plant = contract(io)
    fatal = [f"I/O contract {k}: ONNX {man['io'].get(k)!r} != wheeled_base/wheeled_base.yaml {v!r}"
             for k, v in io_part.items() if man["io"].get(k) != v]
    warn = [f"plant {k}: ONNX {man['plant'].get(k)!r} != wheeled_base/wheeled_base.yaml {v!r}"
            for k, v in plant.items() if man["plant"].get(k) != v]
    return fatal, warn


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--show":
        print(json.dumps(read(sys.argv[2]), indent=1, ensure_ascii=False))
        return
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    ckpt, out = sys.argv[1], sys.argv[2]
    io = WheelbaseIO()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "export_ckpt.py"), ckpt, out], check=True)
    import onnx
    model = onnx.load(out)
    dims = (model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value,
            model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value)
    assert dims == (io.obs_dim, io.act_dim), f"the checkpoint maps {dims[0]} -> {dims[1]}, the contract is {io.obs_dim} -> {io.act_dim}"
    io_part, plant = contract(io)
    git = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    man = {"io": io_part, "plant": plant, "checkpoint": os.path.relpath(os.path.abspath(ckpt), ROOT),
           "git": git + ("+dirty" if dirty else "")}
    del model.metadata_props[:]
    entry = model.metadata_props.add()
    entry.key, entry.value = KEY, json.dumps(man, ensure_ascii=False)
    onnx.save(model, out)
    print(f"stamped {out}  (git {man['git']}, {man['checkpoint']})")


if __name__ == "__main__":
    main()
