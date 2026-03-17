"""End-to-end test: HY-Motion NPZ -> .motion file."""

import tempfile
from pathlib import Path

import torch

from hymotion_isaaclab.conversion.npz_to_motion import convert_hymotion_npz

# Path to a real HY-Motion NPZ for testing
SAMPLE_NPZ = Path.home() / "code/HY-Motion/output/local_infer/test_prompts_subset/00000001_000.npz"
PROTOMOTIONS_ROOT = Path.home() / "code/ProtoMotions"


def test_convert_produces_valid_motion_file():
    """Convert a sample NPZ and verify the .motion output has all required fields."""
    assert SAMPLE_NPZ.exists(), f"Sample NPZ not found: {SAMPLE_NPZ}"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.motion"
        convert_hymotion_npz(
            npz_path=SAMPLE_NPZ,
            output_path=out_path,
            protomotions_root=PROTOMOTIONS_ROOT,
        )

        assert out_path.exists(), "Output .motion file was not created"
        data = torch.load(out_path, weights_only=False)

        # --- Shape checks ---
        T = data["rigid_body_pos"].shape[0]
        assert T > 1, "Motion must have more than 1 frame"
        assert data["rigid_body_pos"].shape == (T, 24, 3)
        assert data["rigid_body_rot"].shape == (T, 24, 4)
        assert data["rigid_body_vel"].shape == (T, 24, 3)
        assert data["rigid_body_ang_vel"].shape == (T, 24, 3)
        assert data["dof_pos"].shape == (T, 69)
        assert data["dof_vel"].shape == (T, 69)
        assert data["rigid_body_contacts"].shape == (T, 24)

        # --- Quaternions are unit-norm ---
        quat_norms = data["rigid_body_rot"].norm(dim=-1)
        assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-4), \
            f"Quaternions not unit-norm, max deviation: {(quat_norms - 1).abs().max()}"

        # --- No NaN or Inf ---
        for key in ["rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "dof_pos", "dof_vel"]:
            assert torch.isfinite(data[key]).all(), f"Non-finite values in {key}"

        # --- Velocities are reasonable ---
        max_vel = data["rigid_body_vel"].abs().max()
        assert max_vel < 50.0, f"Unreasonable velocity: {max_vel}"

        # --- Contacts are boolean ---
        assert data["rigid_body_contacts"].dtype == torch.bool

        # --- fix_height applied: min Z >= 0 ---
        min_z = data["rigid_body_pos"][:, :, 2].min()
        assert min_z >= -0.01, f"fix_height not applied, min Z = {min_z}"

        # --- Metadata ---
        # Note: motion_dt and motion_num_frames are @property on RobotState,
        # NOT stored in the dict. Only fps is a dataclass field.
        assert data["fps"] == 30

        # --- local_rigid_body_rot present ---
        assert "local_rigid_body_rot" in data
        assert data["local_rigid_body_rot"].shape == (T, 24, 4)


def test_character_is_upright():
    """After conversion, the character should be upright (head above pelvis)."""
    assert SAMPLE_NPZ.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.motion"
        convert_hymotion_npz(
            npz_path=SAMPLE_NPZ,
            output_path=out_path,
            protomotions_root=PROTOMOTIONS_ROOT,
        )
        data = torch.load(out_path, weights_only=False)

        # MuJoCo body indices: Pelvis=0, Head=13
        pelvis_z = data["rigid_body_pos"][:, 0, 2].mean()
        head_z = data["rigid_body_pos"][:, 13, 2].mean()
        assert head_z > pelvis_z, f"Character not upright: head_z={head_z}, pelvis_z={pelvis_z}"
