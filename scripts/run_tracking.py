#!/usr/bin/env python3
"""Run ProtoMotions tracker on a converted .motion file in Isaac Lab.

Usage:
    python scripts/run_tracking.py \
        --checkpoint ~/code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
        --motion-file output/00000001_000.motion \
        --num-envs 1
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CHECKPOINT = (
    Path.home() / "code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt"
)
INFERENCE_AGENT = Path.home() / "code/ProtoMotions/protomotions/inference_agent.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ProtoMotions tracker on a .motion file in Isaac Lab."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to tracker checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="Path to .motion or .pt file from convert_npz.py",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments (default: %(default)s)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        default="isaaclab",
        choices=["isaacgym", "isaaclab", "newton"],
        help="Simulator backend (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate paths
    checkpoint = Path(args.checkpoint)
    motion_file = Path(args.motion_file).resolve()
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)
    if not motion_file.exists():
        print(f"ERROR: Motion file not found: {motion_file}")
        sys.exit(1)
    if not INFERENCE_AGENT.exists():
        print(f"ERROR: inference_agent.py not found: {INFERENCE_AGENT}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(INFERENCE_AGENT),
        "--checkpoint", str(checkpoint),
        "--motion-file", str(motion_file),
        "--num-envs", str(args.num_envs),
        "--simulator", args.simulator,
    ]

    if args.headless:
        cmd.append("--headless")

    # ProtoMotions uses relative paths for USD assets — must run from its root
    protomotions_root = Path.home() / "code" / "ProtoMotions"

    env = os.environ.copy()

    # Isaac Sim needs libomniclient.so on LD_LIBRARY_PATH
    isaacsim_lib = (
        Path.home() / "code" / "env_isaaclab" / "lib" / "python3.11"
        / "site-packages" / "isaacsim"
    )
    omniclient_dir = isaacsim_lib / "kit" / "extscore" / "omni.client.lib" / "bin"
    if omniclient_dir.exists():
        env["LD_LIBRARY_PATH"] = (
            str(omniclient_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
        )

    # NCCL fix for GCP VMs without InfiniBand
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_NET"] = "Socket"
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29500")

    print("Running command:")
    print(f"  cwd: {protomotions_root}")
    print(f"  cmd: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(protomotions_root), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
