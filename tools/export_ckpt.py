#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/export_ckpt.py - export an rsl_rl checkpoint (.pt) to ONNX without launching the simulator.

Only for an actor that is a plain MLP (Linear + ELU) trained WITHOUT observation normalisation; the layer
sizes are read from the state dict. A checkpoint that contains a normaliser is refused, because the bare MLP
would be exported without it; a network with other activations is exported wrongly without any warning. In
those cases use Isaac Lab's export_policy_as_onnx(..., normalizer=...). For a matching actor the result equals
the standard exporter's ONNX on random inputs (--check compares it with such a reference). Use it to try a
checkpoint in MuJoCo during or after training, without waiting for the simulator to start.

The exported ONNX is not stamped. Before it goes on the real robot, stamp it:
  biped_Laika:   python tools/onnx_manifest.py --stamp <out.onnx> --profile <profile>
  wheeled_base:  use tools/wheeled_base_export.py instead (it exports and stamps in one step)
For a quick evaluation, run the simulation tools with --allow-unstamped (with tools/sim2sim.py, also name
the contract: --policy-profile <profile>).

Usage:
  python tools/export_ckpt.py <model_N.pt> <out.onnx> [--check <ref.onnx>]
"""
import sys

import numpy as np
import torch
import torch.nn as nn


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    ref = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--check" else None

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model_state_dict"]
    norm = sorted(k for k in sd if "normalizer" in k)
    if norm:
        sys.exit(f"[export_ckpt] this checkpoint has an observation normalizer ({norm[0]}, ...). This tool exports the bare "
                 "actor MLP and would silently drop it. Export with Isaac Lab's export_policy_as_onnx(..., normalizer=...) instead.")
    # actor.{0,2,4,...}.weight, sorted by layer index; the sizes are read from the weight shapes
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for n, i in enumerate(idx):
        w = sd[f"actor.{i}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(sd[f"actor.{i}.bias"])
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers).eval()
    obs_dim = layers[0].in_features

    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(actor, dummy, out_path, input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    print(f"exported {out_path}  ({obs_dim} -> {layers[-1].out_features})")

    if ref:
        import onnxruntime as ort
        a = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        b = ort.InferenceSession(ref, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(20):
            x = rng.standard_normal((1, obs_dim)).astype(np.float32)
            worst = max(worst, float(np.abs(a.run(None, {"obs": x})[0] - b.run(None, {"obs": x})[0]).max()))
        print(f"vs ref max|Δ| = {worst:.3e}  {'PASS' if worst < 1e-5 else 'FAIL'}")
        sys.exit(0 if worst < 1e-5 else 1)


if __name__ == "__main__":
    main()
