from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MotionNPZ:
    bvh_local_transforms: np.ndarray
    bvh_parent_indices: np.ndarray
    bvh_joint_names: list[str]
    fps: float
    robot_data: np.ndarray
    robot_dim_names: list[str]
    robot_name: str


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[:3]
    qw = quat[3]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return vec + 2.0 * (qw * uv + uuv)


def quat_rotate_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[..., :3]
    qw = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return (vec + 2.0 * (qw * uv + uuv)).astype(np.float32, copy=False)


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float32)


def quat_conjugate_batch(quat: np.ndarray) -> np.ndarray:
    result = np.array(quat, dtype=np.float32, copy=True)
    result[..., :3] *= -1.0
    return result


def compute_global_joint_transforms(
    local_transforms: np.ndarray, parent_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    num_frames, num_joints = local_transforms.shape[:2]
    global_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    global_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)

    local_positions = local_transforms[..., :3]
    local_rotations = local_transforms[..., 3:7]

    for joint_idx in range(num_joints):
        parent_idx = parent_indices[joint_idx]
        if parent_idx < 0:
            global_positions[:, joint_idx] = local_positions[:, joint_idx]
            global_rotations[:, joint_idx] = local_rotations[:, joint_idx]
            continue

        parent_rot = global_rotations[:, parent_idx]
        parent_pos = global_positions[:, parent_idx]
        global_positions[:, joint_idx] = parent_pos + quat_rotate_batch(
            parent_rot, local_positions[:, joint_idx]
        )
        global_rotations[:, joint_idx] = quat_mul_batch(
            parent_rot, local_rotations[:, joint_idx]
        )

    return global_positions, global_rotations


def compute_global_joint_positions(
    local_transforms: np.ndarray, parent_indices: np.ndarray
) -> np.ndarray:
    positions, _ = compute_global_joint_transforms(local_transforms, parent_indices)
    return positions


def qpos_from_robot_frame(robot_frame: np.ndarray, expected_nq: int) -> np.ndarray:
    qpos = np.asarray(robot_frame[1:], dtype=np.float32).copy()
    if qpos.shape[0] >= 7:
        qpos[3:7] = np.array([qpos[6], qpos[3], qpos[4], qpos[5]], dtype=np.float32)
    if qpos.shape[0] != expected_nq:
        raise ValueError(
            f"Robot frame qpos width {qpos.shape[0]} does not match model.nq {expected_nq}."
        )
    return qpos


def load_motion_npz(npz_path: str | Path) -> MotionNPZ:
    payload = np.load(npz_path, allow_pickle=False)
    if "bvh_local_transforms" in payload.files:
        return MotionNPZ(
            bvh_local_transforms=np.asarray(
                payload["bvh_local_transforms"], dtype=np.float32
            ),
            bvh_parent_indices=np.asarray(payload["bvh_parent_indices"], dtype=np.int32),
            bvh_joint_names=payload["bvh_joint_names"].tolist(),
            fps=float(payload["bvh_sample_rate"]),
            robot_data=np.asarray(payload["robot_data"], dtype=np.float32),
            robot_dim_names=payload["robot_dim_names"].tolist(),
            robot_name=str(payload["robot_name"].tolist()),
        )

    robot_root_pos = np.asarray(payload["robot_root_pos"], dtype=np.float32)
    robot_root_quat = np.asarray(payload["robot_root_quat"], dtype=np.float32)
    scalar_first = bool(payload["scalar_first"].item()) if "scalar_first" in payload.files else True
    if scalar_first:
        robot_root_quat = robot_root_quat[:, [1, 2, 3, 0]]
    robot_joint_pos = np.asarray(payload["robot_joint_pos"], dtype=np.float32)
    frame_ids = np.arange(robot_root_pos.shape[0], dtype=np.float32).reshape(-1, 1)
    robot_data = np.concatenate(
        (frame_ids, robot_root_pos, robot_root_quat, robot_joint_pos), axis=1
    )
    return MotionNPZ(
        bvh_local_transforms=np.asarray(
            payload["human_local_transforms"], dtype=np.float32
        ),
        bvh_parent_indices=np.asarray(payload["human_parent_indices"], dtype=np.int32),
        bvh_joint_names=payload["human_joint_names"].tolist(),
        fps=float(payload["fps"]),
        robot_data=robot_data,
        robot_dim_names=[
            "Frame",
            "root_translateX",
            "root_translateY",
            "root_translateZ",
            "root_quatX",
            "root_quatY",
            "root_quatZ",
            "root_quatW",
            *payload["robot_joint_names"].tolist(),
        ],
        robot_name=str(payload["robot_name"].tolist()),
    )


def apply_visualization_frame(
    positions: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y_up_to_z_up = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    expanded = np.broadcast_to(y_up_to_z_up, rotations.shape)
    corrected_positions = quat_rotate_batch(expanded, positions)
    corrected_rotations = quat_mul_batch(
        quat_mul_batch(expanded, rotations),
        quat_conjugate_batch(expanded),
    )

    return corrected_positions.astype(np.float32, copy=False), corrected_rotations.astype(
        np.float32, copy=False
    )
