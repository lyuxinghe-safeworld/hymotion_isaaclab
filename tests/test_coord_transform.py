"""Verify the canonical ProtoMotions coordinate rotation works for HY-Motion data."""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

SAMPLE_NPZ = Path.home() / "code/HY-Motion/output/local_infer/test_prompts_subset/00000001_000.npz"


def _get_coord_rotation():
    """Return the canonical ProtoMotions coordinate rotation as a 3x3 matrix."""
    rot1 = sRot.from_euler("xyz", [-np.pi / 2, -np.pi / 2, 0])
    return torch.from_numpy(rot1.as_matrix()).to(torch.float64)


def test_rotation_is_proper():
    """The canonical coordinate rotation should be a proper rotation (det=+1)."""
    R = _get_coord_rotation()
    det = torch.det(R)
    assert torch.allclose(det, torch.tensor(1.0, dtype=torch.float64), atol=1e-12), \
        f"Rotation matrix det != +1: {det}"
    # Verify orthogonality: R^T @ R = I
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(R.T @ R, eye, atol=1e-12), "R^T @ R != I"


def test_round_trip():
    """Rotate then inverse-rotate recovers original vectors."""
    R = _get_coord_rotation()
    R_inv = R.T
    points = torch.randn(10, 3, dtype=torch.float64)
    recovered = (R_inv @ (R @ points.T)).T
    assert torch.allclose(points, recovered, atol=1e-12), "Round-trip failed"


def test_hymotion_convention_feet_near_ground():
    """After full conversion, feet should be near the ground plane (Z ~ 0)."""
    assert SAMPLE_NPZ.exists(), f"Sample NPZ not found: {SAMPLE_NPZ}"
    from hymotion_isaaclab.conversion.npz_to_motion import convert_hymotion_npz
    import tempfile

    protomotions_root = Path.home() / "code/ProtoMotions"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.motion"
        convert_hymotion_npz(SAMPLE_NPZ, out_path, protomotions_root)
        data = torch.load(out_path, weights_only=False)

        positions = data["rigid_body_pos"]  # (T, 24, 3)

        # No NaN or Inf
        assert torch.isfinite(positions).all(), "Non-finite positions after conversion"

        # Feet (MuJoCo indices 4=L_Toe, 8=R_Toe) should be near ground
        feet_z = torch.cat([positions[:, 4, 2], positions[:, 8, 2]])
        min_feet_z = feet_z.min()
        assert min_feet_z >= -0.05, f"Feet below ground: min_z={min_feet_z}"
        assert min_feet_z < 0.5, f"Feet too high: min_z={min_feet_z}"

        # Character upright: head (13) Z > pelvis (0) Z on average
        head_z_mean = positions[:, 13, 2].mean()
        pelvis_z_mean = positions[:, 0, 2].mean()
        assert head_z_mean > pelvis_z_mean, \
            f"Not upright: head_z={head_z_mean}, pelvis_z={pelvis_z_mean}"
