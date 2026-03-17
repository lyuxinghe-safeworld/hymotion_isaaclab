#!/usr/bin/env python3
"""Generate human motion from a text prompt, save as NPZ + skeleton visualization.

Produces NPZ files identical to HY-Motion's local_infer.py output, consumable
by hymotion_isaaclab's convert_npz.py. Also renders a skeleton animation MP4
using the same visualization as visualize_skeleton.py.

Requirements:
    - Must be run in the hymotion conda environment (not env_isaaclab)
    - HY-Motion repo at ~/code/HY-Motion

Usage:
    conda activate hymotion
    HY_MOTION_LLM_4BIT=1 python scripts/generate_motion.py \
        --prompt "A person walks forward" \
        --output-dir output/generated/

    # Then convert to .motion for Isaac Lab tracking:
    source ~/code/env_isaaclab/bin/activate
    PYTHONPATH="$HOME/code/ProtoMotions:$HOME/code/ProtoMotions/data/scripts:$PYTHONPATH" \
        python scripts/convert_npz.py \
            --npz-file output/generated/a_person_walks_forward_000.npz \
            --output-dir output/generated/
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

# HY-Motion must be importable
HYMOTION_ROOT = Path.home() / "code" / "HY-Motion"
if str(HYMOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(HYMOTION_ROOT))


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate motion from a text prompt, save NPZ + skeleton MP4.",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt describing the desired motion.",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Motion duration in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--model-path", type=str,
        default=str(HYMOTION_ROOT / "ckpts/tencent/HY-Motion-1.0-Lite"),
        help="Path to HY-Motion model directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/generated",
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=5.0,
        help="Classifier-free guidance scale (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: %(default)s).",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output FPS (default: %(default)s, matches HY-Motion).",
    )
    parser.add_argument(
        "--skip-video", action="store_true",
        help="Skip skeleton MP4 rendering.",
    )
    return parser.parse_args()


def generate_motion(prompt, duration, model_path, cfg_scale, seed):
    """Run HY-Motion inference, return model_output dict."""
    import os.path as osp
    from hymotion.utils.t2m_runtime import T2MRuntime

    cfg = osp.join(model_path, "config.yml")
    ckpt = osp.join(model_path, "latest.ckpt")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"Config not found: {cfg}")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading model from {model_path} ...")
    runtime = T2MRuntime(
        config_path=cfg,
        ckpt_name=ckpt,
        disable_prompt_engineering=True,
    )

    print(f"Generating motion for: '{prompt}' ({duration:.1f}s, seed={seed})")
    _, _, model_output = runtime.generate_motion(
        text=prompt,
        seeds_csv=str(seed),
        duration=duration,
        cfg_scale=cfg_scale,
        output_format="dict",
    )
    return model_output


def save_npz(model_output, output_dir, base_filename):
    """Save NPZ in the same format as HY-Motion's save_visualization_data.

    Uses construct_smpl_data_dict to ensure identical output to local_infer.py.
    """
    from hymotion.pipeline.body_model import construct_smpl_data_dict

    rot6d = model_output["rot6d"]    # (B, T, J, 6)
    transl = model_output["transl"]  # (B, T, 3)
    batch_size = rot6d.shape[0]

    npz_paths = []
    for bb in range(batch_size):
        smpl_data = construct_smpl_data_dict(rot6d[bb].clone(), transl[bb].clone())

        npz_dict = {}
        npz_dict["gender"] = np.array([smpl_data.get("gender", "neutral")], dtype=str)
        for key in ["Rh", "trans", "poses", "betas"]:
            if key in smpl_data:
                val = smpl_data[key]
                if isinstance(val, (list, tuple)):
                    val = np.array(val)
                elif isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                npz_dict[key] = val

        sample_filename = f"{base_filename}_{bb:03d}.npz"
        sample_path = os.path.join(output_dir, sample_filename)
        np.savez_compressed(sample_path, **npz_dict)
        npz_paths.append(sample_path)
        print(f"  Saved NPZ: {sample_path}")

    return npz_paths


def extract_keypoints(model_output):
    """Extract 3D joint positions from model output (world-space, ground-aligned)."""
    kp = model_output["keypoints3d"]  # (B, T, J, 3)
    transl = model_output["transl"]   # (B, T, 3)

    if kp.dim() == 4:
        kp = kp[0]
        transl = transl[0]

    # keypoints3d is pelvis-local; add root translation for world-space
    kp = kp + transl.unsqueeze(1)

    # Ground alignment: shift so minimum y = 0
    min_y = kp[..., 1].min()
    kp[..., 1] -= min_y

    return kp.cpu().numpy()


def render_skeleton_mp4(xyz, mp4_path, title, fps=30):
    """Render skeleton animation MP4 (same style as visualize_skeleton.py)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    KINEMATIC_CHAINS = [
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
        [0, 3, 6, 9, 12, 15],
        [9, 13, 16, 18, 20],
        [9, 14, 17, 19, 21],
    ]
    CHAIN_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#e67e22"]
    BODY_JOINT_INDICES = list(range(22))

    body_xyz = xyz[:, BODY_JOINT_INDICES, :]
    T = body_xyz.shape[0]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    mins = body_xyz.min(axis=(0, 1))
    maxs = body_xyz.max(axis=(0, 1))
    center = (mins + maxs) / 2
    span = (maxs - mins).max() * 0.6

    def update(frame):
        ax.clear()
        ax.set_title(f"{title}\nFrame {frame}/{T}", fontsize=10)
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        pts = body_xyz[frame]
        for chain, color in zip(KINEMATIC_CHAINS, CHAIN_COLORS):
            chain_pts = pts[chain]
            ax.plot3D(
                chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2],
                color=color, linewidth=2,
            )
        ax.scatter3D(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=10, c="black", zorder=5,
        )

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(str(mp4_path), writer=writer)
    plt.close(fig)
    print(f"  Saved MP4: {mp4_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(args.prompt)

    # Step 1: Generate motion
    model_output = generate_motion(
        prompt=args.prompt,
        duration=args.duration,
        model_path=args.model_path,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )

    # Step 2: Save NPZ (identical to local_infer.py output)
    print("Saving NPZ files...")
    npz_paths = save_npz(model_output, str(output_dir), slug)

    # Step 3: Save prompt text alongside NPZ
    for npz_path in npz_paths:
        txt_path = npz_path.replace(".npz", ".txt")
        with open(txt_path, "w") as f:
            f.write(args.prompt)

    # Step 4: Render skeleton visualization
    if not args.skip_video:
        print("Rendering skeleton visualization...")
        kp = extract_keypoints(model_output)
        T = kp.shape[0]
        mp4_path = output_dir / f"{slug}_skeleton.mp4"
        render_skeleton_mp4(kp, str(mp4_path), args.prompt, fps=args.fps)
    else:
        T = model_output["rot6d"].shape[1]

    # Summary
    print()
    print("=" * 60)
    print("Done")
    print(f"  Prompt   : {args.prompt}")
    print(f"  Frames   : {T}")
    print(f"  Duration : {T / args.fps:.2f}s @ {args.fps} FPS")
    for p in npz_paths:
        print(f"  NPZ      : {p}")
    if not args.skip_video:
        print(f"  Video    : {mp4_path}")
    print()
    print("Next step — convert to .motion for Isaac Lab:")
    print(f"  python scripts/convert_npz.py --npz-file {npz_paths[0]} --output-dir {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
