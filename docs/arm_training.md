# Humanoid-arm policy experiments

These simulation experiments extend the Lucen Robotics SDK. Run commands from the
repository root in the Python environment used for training. They do not connect
to hardware. The grasp scene replaces the right jaw mesh collisions with flat
collision pads; results describe this modified simulation, not hardware validation.

## Dependencies and scene

The working environment used Python 3.11, MuJoCo 4.0.1, NumPy 2.4.6,
PyYAML 6.0.3, Torch 2.15.0+cpu, rsl-rl-lib 3.1.0, TensorDict 0.15.6,
and TensorBoard 2.21.0. RSL-RL 3.1.0 is important because the BC trainer uses
its normalization buffers and checkpoint format. Install a compatible CPU Torch
build, then install the simulation/training dependencies:

```bash
python -m pip install -r requirements-training.txt
```

The generated grasp scene is included. To regenerate it, run
`python build_grasp_scene.py`; this also opens its preview viewer. Close the viewer
to finish. Keep the SDK humanoid-arm YAML, MJCF source scene, and mesh assets.

## Replay the published approach policy

```bash
python evaluate_approach.py models/approach_bc/model_bc.pt --episodes 20
python watch_grasp.py models/approach_bc/model_bc.pt
```

Keep `model_bc.pt` and `config.json` together. The configuration preserves the
0.18-radian starting elbow offset and normalization/training settings. The policy
uses 31 simulator-state observations and outputs eight joint-target actions.
It uses no camera input or language commands. Evaluation samples a small workspace
near the ready pose; success does not establish performance across the full workspace.

## Train reaching and grasping with PPO

```bash
python train_reach.py --iterations 500
python train_grasp.py --iterations 500
```

Both commands start from random network initialization by default and load
`configs/arm_ppo.json`. Outputs go into new folders under `runs/`.
To resume a compatible PPO checkpoint, add `--checkpoint PATH_TO_CHECKPOINT`.
The optimizer state is restored too. Iterations specify additional training work.
Use a checkpoint from the same task and compatible observation/action interface.
A custom configuration can be selected with `--config PATH_TO_JSON`.

`train_approach.py` is an experimental PPO approach-and-grasp trainer, not the
method used for the published final approach policy:

```bash
python train_approach.py --iterations 500 --start-elbow-offset 0.06
```

It also supports `--config` and optional `--checkpoint`. Starting this experiment
from scratch is supported, but successful learning is not guaranteed.

## Train the final approach architecture by behavior cloning

The included `data/approach_guided.npz` contains 16,000 numerical observation/action
pairs: scripted demonstrations plus teacher-labeled states from guided rollouts.
`data/approach_demos.npz` contains the original 8,000 scripted pairs. No images or
human demonstrations are present. Both datasets use a 0.18-radian start offset.

```bash
python train_approach_bc.py
```

The trainer reads `configs/approach_bc.json`, uses the first 90% of samples for
training and the remainder for validation, adds configured observation noise,
and applies a normalization standard-deviation floor. It saves the best checkpoint
among validation checks, plus its configuration, in a new `runs/approach_bc_robust_*`
folder. Use `--dataset PATH`, `--config PATH`, or `--epochs N` to override defaults.
Evaluate the resulting checkpoint in simulation; a low validation loss alone does
not establish successful closed-loop control.

To regenerate the data without relying on an old timestamped experiment:

1. Run `python collect_approach_demos.py` to regenerate the scripted dataset.
2. Run `python train_approach_bc.py --dataset data/approach_demos.npz`.
3. Pass the checkpoint printed by step 2 to
   `python collect_approach_guided.py PATH_TO_CHECKPOINT`.
4. Run `python train_approach_bc.py` on the regenerated guided dataset.

Collection scripts overwrite their corresponding dataset files. Guided collection
mixes 80% teacher action with 20% learner action and retains successful episodes;
the number of added pairs can vary with the learner. This bootstrap uses the current
robust trainer, so it need not recreate the historical dataset or model bit for bit.
The included guided dataset allows training without running this bootstrap.

## Publication layout

Publish `configs/`, both selected datasets, and `models/approach_bc/` alongside the
source and required SDK assets. Local `runs/`, `work/`, editor files, and the old
elbow-only backup are ignored. Existing experiments remain on disk. Retain the
upstream LICENSE and NOTICE, and attribute the SDK and models to Lucen Robotics.

## Diagnostic scripts

`python test_cube_grip.py` previews cube contact with the active flat collision
pads. Add `--headless` for an unpaced numerical run without a viewer.
`python check_grip_strength.py` sweeps gripper targets and reports displacement
and the fraction of final-second samples with two-pad contact. These contact
checks do not establish successful lifting; use the lift tests or task evaluators.

`python check_approach_reachability.py` is an experimental local IK search.
Its bounds are centered on the starting pose, rather than the controller's ready
pose, and it preserves the starting hand orientation. Failure is not proof of
unreachability, and a found pose does not certify collision clearance or dynamic
feasibility. Use `test_scripted_approach.py` to check a simulated approach trajectory.

## Security

The arm experiment entry points use `safe_runner.py`: checkpoint deserialization
uses `weights_only=True`, policy/algorithm classes are restricted to `ActorCritic`
and `PPO`, and unsupported options (including callable symmetry/RND hooks) are
rejected before RSL-RL constructs the algorithm. The SDK checkpoint exporter also
uses restricted loading. There is no fallback to unrestricted pickle loading.
The scratch scripts under ignored `work/` are historical and are not hardened.
Restricted loading is not a sandbox: use model/configuration files from trusted
sources; huge or malformed inputs may still exhaust resources.

The upstream `bridge.py` listens on all network interfaces and has no client
authentication. A reachable client can submit hardware-control messages, subject
to the bridge's existing checks. `tools/arm_web_jog.py` also has no authentication;
it defaults to localhost, but `--listen 0.0.0.0:8780` exposes its control endpoints.
Do not expose either service to the Internet or untrusted networks. Hardware use
requires network isolation or an authenticated access layer, in addition to the
SDK's physical safety procedures. The simulation training/evaluation commands in
this guide do not start these network services. Their upstream behavior has not
been changed by the arm experiment hardening.
