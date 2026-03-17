"""Verify angle-axis -> quaternion -> rotation matrix pipeline."""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from protomotions.utils.rotations import matrix_to_quaternion, quaternion_to_matrix

SAMPLE_NPZ = Path.home() / "code/HY-Motion/output/local_infer/test_prompts_subset/00000001_000.npz"


def test_angle_axis_quat_matrix_round_trip():
    """angle-axis -> quat -> matrix -> quat -> angle-axis recovers original."""
    # Generate random angle-axis vectors
    rng = np.random.default_rng(42)
    aa = rng.standard_normal((50, 3)).astype(np.float64)
    # Clamp magnitudes to avoid singularity at 0 and 2*pi
    norms = np.linalg.norm(aa, axis=-1, keepdims=True)
    aa = aa / (norms + 1e-8) * np.clip(norms, 0.1, np.pi - 0.1)

    quats = sRot.from_rotvec(aa).as_quat()  # (50, 4) xyzw
    quats_t = torch.from_numpy(quats).float()
    mats = quaternion_to_matrix(quats_t, w_last=True)  # (50, 3, 3)
    quats_back = matrix_to_quaternion(mats, w_last=True)  # (50, 4)

    # Quaternions may differ by sign (q and -q represent same rotation)
    dot = (quats_t * quats_back).sum(dim=-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-4), \
        f"Quaternion round-trip failed, max deviation: {(dot - 1).abs().max()}"


def test_rotation_matrices_orthogonal():
    """All rotation matrices should have det=+1 and R^T @ R = I."""
    rng = np.random.default_rng(123)
    aa = rng.standard_normal((100, 3)).astype(np.float64)
    quats = sRot.from_rotvec(aa).as_quat()
    quats_t = torch.from_numpy(quats).float()
    mats = quaternion_to_matrix(quats_t, w_last=True)

    # det = +1
    dets = torch.det(mats)
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-4), \
        f"Determinants not +1: {dets}"

    # R^T @ R = I
    eye = torch.eye(3).unsqueeze(0).expand_as(mats)
    rtrs = torch.bmm(mats.transpose(-1, -2), mats)
    assert torch.allclose(rtrs, eye, atol=1e-4), "R^T @ R != I"


def test_fk_positions_physically_plausible():
    """After full pipeline, FK positions should have reasonable bone lengths."""
    assert SAMPLE_NPZ.exists()
    from hymotion_isaaclab.conversion.npz_to_motion import convert_hymotion_npz
    import tempfile

    protomotions_root = Path.home() / "code/ProtoMotions"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.motion"
        convert_hymotion_npz(SAMPLE_NPZ, out_path, protomotions_root)
        data = torch.load(out_path, weights_only=False)

        pos = data["rigid_body_pos"]  # (T, 24, 3)
        T = pos.shape[0]

        # Check bone lengths between parent-child pairs are in [0.01, 1.0] m
        # Pelvis(0)->L_Hip(1), Pelvis(0)->R_Hip(5) as examples
        parent_child_pairs = [(0, 1), (0, 5), (1, 2), (5, 6)]
        for pi, ci in parent_child_pairs:
            bone_len = (pos[:, ci] - pos[:, pi]).norm(dim=-1)
            assert bone_len.min() > 0.01, f"Bone {pi}->{ci} too short: {bone_len.min()}"
            assert bone_len.max() < 1.0, f"Bone {pi}->{ci} too long: {bone_len.max()}"


def test_rotated_fk_bone_directions_consistent():
    """Bone directions from FK positions should match directions from rotated global rotations."""
    assert SAMPLE_NPZ.exists()
    from hymotion_isaaclab.conversion.npz_to_motion import convert_hymotion_npz
    from protomotions.components.pose_lib import extract_kinematic_info
    import tempfile

    protomotions_root = Path.home() / "code/ProtoMotions"
    ki = extract_kinematic_info(
        str(protomotions_root / "protomotions/data/assets/mjcf/smpl_humanoid.xml")
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.motion"
        convert_hymotion_npz(SAMPLE_NPZ, out_path, protomotions_root)
        data = torch.load(out_path, weights_only=False)

        pos = data["rigid_body_pos"]  # (T, 24, 3)
        rot = data["rigid_body_rot"]  # (T, 24, 4) xyzw
        rot_mats = quaternion_to_matrix(rot, w_last=True)  # (T, 24, 3, 3)

        # For a non-root body, check: pos[child] - pos[parent] direction
        # should roughly match R_parent @ rest_bone_offset direction
        for body_idx in [1, 2, 5, 6]:  # L_Hip, L_Knee, R_Hip, R_Knee
            pi = ki.parent_indices[body_idx]
            if pi < 0:
                continue
            rest_offset = ki.local_pos[body_idx].float()
            if rest_offset.norm() < 1e-6:
                continue

            # Direction from FK positions
            actual_bone = pos[:, body_idx] - pos[:, pi]  # (T, 3)
            actual_dir = actual_bone / (actual_bone.norm(dim=-1, keepdim=True) + 1e-8)

            # Direction from parent rotation * rest offset
            predicted_bone = torch.bmm(
                rot_mats[:, pi],
                rest_offset.unsqueeze(0).expand(pos.shape[0], -1).unsqueeze(-1),
            ).squeeze(-1)
            predicted_dir = predicted_bone / (predicted_bone.norm(dim=-1, keepdim=True) + 1e-8)

            # Cosine similarity should be high (> 0.9)
            cos_sim = (actual_dir * predicted_dir).sum(dim=-1)
            mean_cos = cos_sim.mean()
            assert mean_cos > 0.9, \
                f"Body {body_idx}: bone direction inconsistency, mean cos_sim={mean_cos}"
