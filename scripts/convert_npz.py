#!/usr/bin/env python3
"""Convert HY-Motion NPZ files to ProtoMotions .motion format.

Usage:
    # Single file
    python scripts/convert_npz.py \
        --npz-file ~/code/HY-Motion/output/local_infer/test_prompts_subset/00000001_000.npz \
        --output-dir output/

    # Directory (batch)
    python scripts/convert_npz.py \
        --npz-dir ~/code/HY-Motion/output/local_infer/test_prompts_subset/ \
        --output-dir output/
"""

import argparse
import sys
from pathlib import Path

from hymotion_isaaclab.conversion.npz_to_motion import convert_hymotion_npz

DEFAULT_PROTOMOTIONS_ROOT = Path.home() / "code" / "ProtoMotions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HY-Motion NPZ files to ProtoMotions .motion format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--npz-file",
        type=str,
        help="Path to a single HY-Motion .npz file",
    )
    group.add_argument(
        "--npz-dir",
        type=str,
        help="Path to a directory of HY-Motion .npz files (batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for .motion files (default: output/)",
    )
    parser.add_argument(
        "--protomotions-root",
        type=str,
        default=str(DEFAULT_PROTOMOTIONS_ROOT),
        help="Path to ProtoMotions repo root (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protomotions_root = Path(args.protomotions_root)

    if args.npz_file:
        npz_files = [Path(args.npz_file)]
    else:
        npz_files = sorted(Path(args.npz_dir).glob("*.npz"))
        if not npz_files:
            print(f"No .npz files found in {args.npz_dir}")
            sys.exit(1)

    print(f"Converting {len(npz_files)} file(s) to {output_dir}/")

    for npz_path in npz_files:
        out_name = npz_path.stem + ".motion"
        out_path = output_dir / out_name
        print(f"  {npz_path.name} -> {out_name}")
        try:
            convert_hymotion_npz(
                npz_path=npz_path,
                output_path=out_path,
                protomotions_root=protomotions_root,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()
