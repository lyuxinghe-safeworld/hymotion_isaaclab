# HY-Motion IsaacLab

Convert HY-Motion generated motion (NPZ files) to ProtoMotions format and track in Isaac Lab.

---

## Prerequisites

- GCP VM with an NVIDIA GPU (A100 / T4 / V100 recommended)
- TurboVNC installed and configured (required for Isaac Lab headless display)
- Python 3.10+
- The following repos cloned under `~/code/`:

| Repo | Path |
|------|------|
| ProtoMotions | `~/code/ProtoMotions` |
| HY-Motion | `~/code/HY-Motion` (for NPZ generation only; not needed at conversion time) |
| hymotion_isaaclab (this repo) | `~/code/hymotion_isaaclab` |

---

## Setup

### 1. Download the ProtoMotions motion tracker checkpoint

```bash
cd ~/code/ProtoMotions
git lfs pull
```

Verify the checkpoint exists:

```bash
ls -lh ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt
```

### 2. Install this package

Activate the `env_isaaclab` environment and install in editable mode:

```bash
source ~/code/env_isaaclab/bin/activate
pip install -e "~/code/hymotion_isaaclab[dev]"
```

### 3. Set environment variables

```bash
export PYTHONPATH="$HOME/code/ProtoMotions:$HOME/code/ProtoMotions/data/scripts:$PYTHONPATH"
export DISPLAY=:1   # adjust to match your active VNC display
```

---

## TurboVNC Notes

Isaac Lab requires a display. On a headless GCP VM, use TurboVNC:

```bash
# Start the VNC server on display :1
vncserver :1 -geometry 1920x1080 -depth 24

# Export the display in your shell
export DISPLAY=:1
```

To stop the server:

```bash
vncserver -kill :1
```

---

## Usage

### Step 1 — Convert HY-Motion NPZ to .motion format (no simulator needed)

Single file:

```bash
python scripts/convert_npz.py \
    --npz-file ~/code/HY-Motion/output/generated/a_person_walks_forward_000.npz \
    --output-dir output/
```

Batch (all NPZ files in a directory):

```bash
python scripts/convert_npz.py \
    --npz-dir ~/code/HY-Motion/output/local_infer/test_prompts_subset/ \
    --output-dir output/
```

### Step 2 — Track in Isaac Lab (needs DISPLAY)

```bash
# Required env vars for GCP VMs
export DISPLAY=:1
export LD_LIBRARY_PATH="$HOME/code/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/extscore/omni.client.lib/bin:${LD_LIBRARY_PATH:-}"
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

python scripts/run_tracking.py \
    --checkpoint ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
    --motion-file output/00000001_000.motion \
    --num-envs 1
```

---

## Architecture

```
Text prompt
    │
    ▼
HY-Motion generate_motion.py (hymotion conda env)
    │  SMPL-H NPZ (angle-axis poses + trans, 30fps) + skeleton MP4
    ▼
convert_npz.py (env_isaaclab)
    │  1. Extract 22 body joints + 2 zero hands -> 24 joints
    │  2. Angle-axis -> quaternions -> local rotation matrices
    │  3. ProtoMotions FK (global rotations from MJCF skeleton)
    │  4. Coordinate rotation (canonical ProtoMotions quat right-multiply)
    │  5. FK with velocities (positions, rotations, velocities)
    │  6. DOF extraction (exp_map) + angular velocities
    │  7. Height fix + contact detection
    ▼
.motion file (ProtoMotions RobotState format)
    │
    ▼
run_tracking.py -> ProtoMotions inference_agent.py
    │  Open-loop tracking in Isaac Lab
    ▼
Physics simulation of humanoid following the motion
```

---

## Running Tests

```bash
source ~/code/env_isaaclab/bin/activate
cd ~/code/hymotion_isaaclab
PYTHONPATH="$HOME/code/ProtoMotions:$HOME/code/ProtoMotions/data/scripts:$PYTHONPATH" \
    pytest tests/ -v
```
