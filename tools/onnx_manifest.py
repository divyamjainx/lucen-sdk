#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/onnx_manifest.py - stamp, read and verify the policy I/O contract manifest embedded in an ONNX file.

One ONNX file belongs to exactly one policy profile. Running it with another profile feeds the policy a
misaligned gait phase and nominal pose, and nothing fails visibly. The contract therefore travels with the
file: when a policy is exported, the complete policy I/O contract is written ("stamped") into the ONNX
metadata_props under the key "lucen_manifest". Every consumer (tools/sim2sim.py, tools/sim_fsm.py,
tools/deploy_policy.py) reads the manifest when it loads the policy, compares it item by item with
biped_Laika/biped_Laika.yaml, and refuses to run on a mismatch.

Two verification levels:
  FATAL (I/O contract): joint order, observation terms with their dimensions and scales, action clip and
    per-joint action scale, target bounds, nominal pose, gait cycle time, control rate, profile name.
    If any of these differs, the meaning of the policy inputs or outputs is wrong: the policy is refused.
  WARN (plant parameters): gains, torque limits, speed limits, armature, joint friction and damping, URDF and
    USD hashes. A difference is reported and the policy still runs, unless the consumer asks for strict plant
    checking (enforce(strict_plant=True)); then it is a refusal as well. tools/deploy_policy.py is strict on
    the real robot unless --allow-plant-drift is given, and tools/sim_fsm.py is always strict.

Reading a manifest needs only onnxruntime. Stamping needs the `onnx` package.

Usage (from the SDK root):
  python tools/onnx_manifest.py --show biped_Laika/policies/walk.onnx
  python tools/onnx_manifest.py --verify biped_Laika/policies/walk.onnx
  python tools/onnx_manifest.py --stamp X.onnx --profile P [--policy-name N]
--show prints the manifest as JSON. --verify compares it with the current configuration file, prints one FATAL
line per contract difference and one WARN line per plant-parameter difference, then PASS or FAIL; only FATAL
lines make it FAIL (exit code 1). --stamp writes the contract of profile P
(omit --profile for the default standing contract) as biped_Laika/biped_Laika.yaml defines it at that moment,
so stamp a policy with the configuration it was trained with. --stamp refuses a file whose input size does not
match the profile.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

KEY = "lucen_manifest"
MANIFEST_VERSION = 1


def _sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _round(x):
    """Round floats to 6 decimals before storing. Comparison is then exact equality (both sides come from the yaml)."""
    if isinstance(x, dict):
        return {k: _round(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, float):
        return round(x, 6)
    return x


def build(cfg, profile, policy_name):
    """Build the complete manifest for `profile` from the current biped_Laika/biped_Laika.yaml (PolicyIO resolves the contract)."""
    from tools.policy_io import PolicyIO
    io = PolicyIO(cfg, profile)
    prof = (cfg.get("policy_profiles") or {}).get(profile, {}) if profile else {}
    contract = {
        "profile": profile or "",
        "joint_order": list(io.names),
        "obs_spec": [{"name": o["name"], "dim": int(o["dim"]),
                      "scale": [float(s) for s in (o["scale"] if isinstance(o["scale"], (list, tuple))
                                                   else [o["scale"]] * o["dim"])]}
                     for o in io.spec],
        "obs_dim": int(io.obs_dim),
        "action_dim": len(io.names),
        "action_clip": float(io.action_clip),
        "action_scale_vec": [float(x) for x in io.action_scale_vec],
        "target_lo": [float(x) for x in io.tlo],
        "target_hi": [float(x) for x in io.thi],
        "nominal_pose": {n: float(v) for n, v in zip(io.names, io.default)},
        "cycle_time_s": io.cycle_time,
        # Lateral feed-forward gain (0 = none). It changes the action -> target mapping, so it is part of
        # the contract: a consumer refuses an ONNX / profile combination that does not match.
        "lat_ff_gain": float(io.lat_ff_gain),
        "control_rate_hz": float(io.rate),
        "init_base_height": prof.get("init_base_height"),
    }
    if io.gain_profile:
        # Gain profile the policy was trained with (only for profiles that declare one): other gains = another plant.
        contract["gain_profile"] = str(io.gain_profile)
    plant = {
        "gains_rl_default": cfg["gain_profiles"]["rl_default"],
        "tau_limit": cfg.get("tau_limit") or cfg["tau_max"],
        "vel_limit": cfg["vel_limit"],
        "armature": cfg.get("armature"),
        "joint_friction": cfg.get("joint_friction"),
        "joint_damping": cfg.get("joint_damping"),
        "urdf_sha16": _sha16(os.path.join(ROOT, "biped_Laika", "biped_Laika.urdf")),
        "usd_sha16": _sha16(os.path.join(ROOT, "biped_Laika", "usd", "laika_v3.usd")),
    }
    if io.gain_profile:
        plant[f"gains_{io.gain_profile}"] = cfg["gain_profiles"][io.gain_profile]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    return {"manifest_version": MANIFEST_VERSION, "policy": policy_name,
            "git_commit": commit, "created": datetime.date.today().isoformat(),
            "contract": _round(contract), "plant": _round(plant)}


def stamp(path, manifest):
    """Write the manifest into metadata_props (idempotent: an existing entry is replaced).
    Only the metadata changes; the weights stay bit-identical."""
    import onnx
    m = onnx.load(path)
    keep = [p for p in m.metadata_props if p.key != KEY]
    del m.metadata_props[:]
    m.metadata_props.extend(keep)
    e = m.metadata_props.add()
    e.key, e.value = KEY, json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    onnx.save(m, path)


def read(path=None, session=None):
    """-> manifest dict, or None if the file is not stamped.

    With an existing onnxruntime session the manifest is read from it; otherwise from the file. Without the
    `onnx` package this falls back to onnxruntime, so a deployment machine does not need onnx / protobuf. The
    temporary session loads the whole model, which is negligible for these small policies."""
    if session is None:
        try:
            import onnx
            raw = next((p.value for p in onnx.load(path, load_external_data=False).metadata_props
                        if p.key == KEY), None)
            return json.loads(raw) if raw else None
        except ImportError:
            import onnxruntime as ort
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    raw = session.get_modelmeta().custom_metadata_map.get(KEY)
    return json.loads(raw) if raw else None


def _diff(want, got, prefix, out):
    """Recursively collect the leaves that differ, as readable messages."""
    if isinstance(want, dict) and isinstance(got, dict):
        for k in sorted(set(want) | set(got)):
            _diff(want.get(k), got.get(k), f"{prefix}.{k}", out)
    elif _round(want) != _round(got):
        out.append(f"{prefix}: stamped {want!r} != current {got!r}")


def verify(manifest, cfg, requested_profile=None):
    """-> (fatal messages, warnings). requested_profile=None lets the manifest decide the profile."""
    fatal, warn = [], []
    mc = manifest["contract"]
    prof = mc["profile"] or None
    if requested_profile is not None and (requested_profile or "") != mc["profile"]:
        fatal.append(f"profile: the command line asks for {requested_profile!r} but the ONNX is stamped {mc['profile']!r}"
                     f"; an ONNX belongs to exactly one profile and cannot run under another")
        return fatal, warn
    try:
        live = build(cfg, prof, manifest.get("policy", ""))
    except KeyError as e:
        return [f"biped_Laika/biped_Laika.yaml has no such profile: {e}"], warn
    # Backward compatibility: a manifest without the lat_ff_gain key means "no feed-forward", which is exactly
    # 0.0. Fill in that equivalent default instead of failing; a real difference is still reported.
    mc = dict(mc)
    mc.setdefault("lat_ff_gain", 0.0)
    _diff(mc, live["contract"], "contract", fatal)
    _diff(manifest.get("plant", {}), live["plant"], "plant", warn)
    return fatal, warn


def enforce(path, cfg, cli_profile=None, strict_plant=False, allow_unstamped=False,
            session=None, log=print):
    """Single entry point for consumers: resolve the profile, verify, and sys.exit on a mismatch.

    Returns the profile name to use: the one in the manifest, or cli_profile if the file is not stamped and
    allow_unstamped is set. path == 'zero' (no ONNX) passes straight through.
    strict_plant=True turns plant-parameter differences into a refusal as well.
    """
    if path == "zero":
        return cli_profile
    manifest = read(path, session=session)
    if manifest is None:
        msg = (f"[manifest] {os.path.basename(path)} is not stamped: nothing checks that it runs with the "
               f"right profile. Stamp it: python tools/onnx_manifest.py --stamp ...")
        if allow_unstamped:
            log(msg + " (allowed by --allow-unstamped)")
            return cli_profile
        sys.exit(msg + "\nRefusing to run (tools that offer --allow-unstamped run it anyway, for experiments).")
    fatal, warn = verify(manifest, cfg, cli_profile)
    prof = manifest["contract"]["profile"] or None
    for w in warn:
        log(f"[manifest] ⚠ plant parameters differ: {w}")
    if warn and strict_plant:
        fatal.append(f"strict_plant: plant parameters must match exactly ({len(warn)} difference(s), listed above)"
                     f"; tools/deploy_policy.py runs with different plant parameters only with --allow-plant-drift")
    if fatal:
        sys.exit("[manifest] ✗ contract check failed, refusing to run:\n  " + "\n  ".join(fatal))
    log(f"[manifest] ✓ {manifest.get('policy') or os.path.basename(path)}"
        f" profile={manifest['contract']['profile'] or '(default)'}"
        f" obs={manifest['contract']['obs_dim']}"
        f" (stamped {manifest.get('git_commit', '?')} {manifest.get('created', '')})"
        + (f"; {len(warn)} plant-parameter difference(s)" if warn else ""))
    return prof


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stamp", metavar="ONNX", help="embed the manifest for --profile into this ONNX file (metadata only)")
    ap.add_argument("--profile", default=None, help="with --stamp: profile to stamp; omit for the default contract (standing)")
    ap.add_argument("--policy-name", default=None, help="with --stamp: policy name stored in the manifest (default: the file name)")
    ap.add_argument("--show", metavar="ONNX", help="print the manifest of this ONNX file")
    ap.add_argument("--verify", metavar="ONNX", help="check this ONNX file's manifest against the robot config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "biped_Laika/biped_Laika.yaml")))

    if args.show:
        m = read(args.show)
        print(json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) if m else "(not stamped)")
    elif args.verify:
        m = read(args.verify)
        if m is None:
            sys.exit("(not stamped)")
        fatal, warn = verify(m, cfg)
        for w in warn:
            print(f"WARN  {w}")
        for f_ in fatal:
            print(f"FATAL {f_}")
        print("PASS" if not fatal else f"FAIL ({len(fatal)} fatal)")
        sys.exit(1 if fatal else 0)
    elif args.stamp:
        name = args.policy_name or os.path.splitext(os.path.basename(args.stamp))[0]
        man = build(cfg, args.profile, name)
        import onnx
        dim = onnx.load(args.stamp, load_external_data=False).graph.input[0].type.tensor_type.shape.dim[-1].dim_value
        if dim and dim != man["contract"]["obs_dim"]:        # 0 = dynamic size, cannot be checked here
            sys.exit(f"[manifest] {args.stamp} takes {dim} input values, but profile {args.profile or '(default)'} "
                     f"has {man['contract']['obs_dim']} observation values. Not stamped.")
        stamp(args.stamp, man)
        print(f"stamped {args.stamp} (profile={args.profile or '(default)'})")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
