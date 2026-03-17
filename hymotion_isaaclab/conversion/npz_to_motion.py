"""Convert HY-Motion NPZ files to ProtoMotions .motion format.

Follows the canonical ProtoMotions convert_amass_to_proto.py pipeline exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

# These imports require ProtoMotions on PYTHONPATH
from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
from protomotions.components.pose_lib import (
    compute_angular_velocity,
    compute_forward_kinematics_from_transforms,
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    fk_from_transforms_with_velocities,
)
from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quat_mul,
    quaternion_to_matrix,
)

_FOOT_HEIGHT_OFFSET = 0.015  # metres, same as convert_amass_to_proto.py for SMPL
_N_BODY_JOINTS = 22
_N_HAND_JOINTS = 2  # synthetic zero-rotation hands
_N_TOTAL_JOINTS = _N_BODY_JOINTS + _N_HAND_JOINTS  # 24
_N_NON_ROOT_JOINTS = 23


def convert_hymotion_npz(
    npz_path: str | Path,
    output_path: str | Path,
    protomotions_root: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> None:
    """Convert a single HY-Motion NPZ file to a ProtoMotions .motion file.

    Parameters
    ----------
    npz_path : path
        Path to HY-Motion .npz file (contains poses, trans, Rh, betas, mocap_framerate).
    output_path : path
        Where to save the .motion file.
    protomotions_root : path
        Root directory of the ProtoMotions repo (needed for MJCF asset path).
    device : str
        Torch device (default "cpu").
    dtype : torch.dtype
        Torch dtype (default float32).
    """
    npz_path = Path(npz_path)
    output_path = Path(output_path)
    protomotions_root = Path(protomotions_root)

    # --- Step 1: Load NPZ and prepare 24-joint angle-axis ---
    npz_data = np.load(npz_path)
    poses = npz_data["poses"]  # (T, 156) — 52 joints x 3 angle-axis
    amass_trans = npz_data["trans"]  # (T, 3)
    if "mocap_framerate" in npz_data:
        output_fps = int(np.round(npz_data["mocap_framerate"]))
    else:
        output_fps = 30  # HY-Motion default

    T = poses.shape[0]
    assert T >= 2, f"Motion too short: {T} frames, need >= 2"

    # Extract 22 body joints, append 2 zero-rotation hand joints
    pose_aa = np.concatenate(
        [poses[:, :66], np.zeros((T, 6))],
        axis=1,
    )  # (T, 72)

    # --- Step 2: Reorder to MuJoCo joint order and convert to quaternions ---
    smpl_2_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(q)
        for q in SMPL_MUJOCO_NAMES
        if q in SMPL_BONE_ORDER_NAMES
    ]
    pose_aa_mj = pose_aa.reshape(T, _N_TOTAL_JOINTS, 3)[:, smpl_2_mujoco]
    pose_quat = (
        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
        .as_quat()
        .reshape(T, _N_TOTAL_JOINTS, 4)
    )

    amass_trans = torch.from_numpy(amass_trans).to(device, dtype)
    pose_quat = torch.from_numpy(pose_quat).to(device, dtype)
    local_rot_mats = quaternion_to_matrix(pose_quat, w_last=True)

    # --- Step 3: Forward kinematics (local -> global rotations) ---
    mjcf_path = str(protomotions_root / "protomotions/data/assets/mjcf/smpl_humanoid.xml")
    kinematic_info = extract_kinematic_info(mjcf_path)

    _, world_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, amass_trans, local_rot_mats
    )
    global_quat = matrix_to_quaternion(world_rot_mat, w_last=True)

    # --- Step 4: Apply canonical ProtoMotions coordinate rotation ---
    rot1 = sRot.from_euler(
        "xyz", np.array([-np.pi / 2, -np.pi / 2, 0]), degrees=False
    )
    rot1_quat = (
        torch.from_numpy(rot1.as_quat()).to(device, dtype).expand(T, -1)
    )

    for i in range(_N_TOTAL_JOINTS):
        global_quat[:, i, :] = quat_mul(
            global_quat[:, i, :], rot1_quat, w_last=True
        )

    # --- Step 5: Recompute local rotations and run FK with velocities ---
    local_rot_mats_rotated = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=quaternion_to_matrix(global_quat, w_last=True),
    )

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=amass_trans,
        joint_rot_mats=local_rot_mats_rotated,
        fps=output_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    # Cache local rotations for MotionLib interpolation
    pose_quat_rotated = matrix_to_quaternion(local_rot_mats_rotated, w_last=True)
    motion.local_rigid_body_rot = pose_quat_rotated.clone()

    # --- Step 6: Extract DOF positions and velocities ---
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=amass_trans,
        joint_rot_mats=local_rot_mats_rotated,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]  # skip root 7-DOF (3 pos + 4 quat)

    local_angular_vels = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats_rotated[:, 1:, :, :],
        fps=output_fps,
    )
    assert local_angular_vels.shape[1] == _N_NON_ROOT_JOINTS
    motion.dof_vel = local_angular_vels.reshape(-1, _N_NON_ROOT_JOINTS * 3)

    # --- Step 7: Fix height and compute contacts ---
    motion.fix_height(height_offset=_FOOT_HEIGHT_OFFSET)

    # contact_detection.py is in ProtoMotions/data/scripts/ — must be on PYTHONPATH
    from contact_detection import compute_contact_labels_from_pos_and_vel

    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    # --- Step 8: Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion.to_dict(), str(output_path))
