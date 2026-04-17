from pathlib import Path

import numpy as np


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def _quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    conjugate = np.array(quat, dtype=np.float32, copy=True)
    conjugate[..., :3] *= -1.0
    return conjugate


def _quat_rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[..., :3]
    qw = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return (vec + 2.0 * (qw * uv + uuv)).astype(np.float32, copy=False)

def _split_robot_motion(robot_motion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot_motion = np.asarray(robot_motion, dtype=np.float32)
    robot_root_pos = robot_motion[:, :3]
    robot_root_quat = robot_motion[:, 3:7]
    robot_joint_pos = robot_motion[:, 7:]
    return robot_root_pos, robot_root_quat, robot_joint_pos


def _forward_difference(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == 0:
        return np.zeros_like(values, dtype=np.float32)
    if values.shape[0] == 1 or dt <= 0.0:
        return np.zeros_like(values, dtype=np.float32)

    diff = (values[1:] - values[:-1]) / dt
    velocities = np.zeros_like(values, dtype=np.float32)
    velocities[:-1] = diff
    velocities[-1] = diff[-1]
    return velocities


def _quat_to_angular_velocity_xyzw(quat: np.ndarray, dt: float) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[0] == 0:
        return np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)
    if quat.shape[0] == 1 or dt <= 0.0:
        return np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)

    q_curr = quat[:-1]
    q_next = quat[1:]
    q_rel = _quat_mul_xyzw(q_next, _quat_conjugate_xyzw(q_curr))

    negative_w = q_rel[..., 3] < 0.0
    q_rel[negative_w] *= -1.0

    xyz = q_rel[..., :3]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, q_rel[..., 3:4])

    angular_step = np.zeros_like(xyz, dtype=np.float32)
    small_angle = xyz_norm[..., 0] < 1e-8
    if np.any(~small_angle):
        angular_step[~small_angle] = (
            xyz[~small_angle] / xyz_norm[~small_angle]
        ) * angle[~small_angle]
    if np.any(small_angle):
        angular_step[small_angle] = 2.0 * xyz[small_angle]

    ang_vel = angular_step / dt
    velocities = np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)
    velocities[:-1] = ang_vel
    velocities[-1] = ang_vel[-1]
    return velocities.astype(np.float32, copy=False)

def quat_rotate_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[..., :3]
    qw = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return (vec + 2.0 * (qw * uv + uuv)).astype(np.float32, copy=False)

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

def _build_retarget_payload(
    *,
    fps,
    skeleton,
    human_local_transforms,
    robot_motion,
    robot_name,
    robot_joint_names,
    robot_body_names,
    robot_body_pos,
    robot_body_quat,
    source_fps,
    source_robot_motion,
    source_human_local_transforms,
) -> dict:
    human_local_transforms = np.asarray(human_local_transforms, dtype=np.float32)
    global_positions, global_rotations = compute_global_joint_transforms(
        human_local_transforms,
        np.asarray(skeleton.parent_indices, dtype=np.int32),
    )
    human_global_pos, human_global_quat = apply_visualization_frame(
        global_positions, global_rotations
    )
    robot_motion = np.asarray(robot_motion, dtype=np.float32)
    robot_root_pos, robot_root_quat, robot_joint_pos = _split_robot_motion(robot_motion)
    dt = 1.0 / float(fps) if float(fps) > 0.0 else 0.0
    robot_root_lin_vel = _forward_difference(robot_root_pos, dt)
    robot_root_ang_vel = _quat_to_angular_velocity_xyzw(robot_root_quat, dt)
    robot_joint_vel = _forward_difference(robot_joint_pos, dt)

    payload = {
        # Target retargeted sequence frame rate in Hz.
        "fps": np.asarray(fps, dtype=np.int32),
        # Number of frames in the exported target sequence.
        "num_frames": np.asarray(human_local_transforms.shape[0], dtype=np.int32),
        # Quaternion storage convention flag; False means xyzw instead of wxyz.
        "scalar_first": np.asarray(False),
        # Name / identifier of the target robot model.
        "robot_name": np.asarray(robot_name),
        # Ordered joint names of the target robot, aligned with robot_joint_pos / vel.
        "robot_joint_names": np.asarray(robot_joint_names),
        # Target robot root translation in world frame, shape [T, 3].
        "robot_root_pos": robot_root_pos,
        # Target robot root orientation quaternion in xyzw order, shape [T, 4].
        "robot_root_quat": robot_root_quat,
        # Target robot joint positions, shape [T, num_robot_joints].
        "robot_joint_pos": robot_joint_pos,
        # Finite-difference linear velocity of the target robot root, shape [T, 3].
        "robot_root_lin_vel": robot_root_lin_vel,
        # Finite-difference angular velocity of the target robot root, shape [T, 3].
        "robot_root_ang_vel": robot_root_ang_vel,
        # Finite-difference joint velocities of the target robot, shape [T, num_robot_joints].
        "robot_joint_vel": robot_joint_vel,
        # Ordered human skeleton joint names, aligned with human transform arrays.
        "human_joint_names": np.asarray(skeleton.joint_names),
        # Parent index of each human joint in the kinematic tree; root uses -1.
        "human_parent_indices": np.asarray(skeleton.parent_indices, dtype=np.int32),
        # Human skeleton up-axis expressed in the source coordinate convention.
        "human_up_axis": np.asarray(skeleton.up_axis, dtype=np.float32),
        # Human skeleton forward-axis expressed in the source coordinate convention.
        "human_forward_axis": np.asarray(skeleton.forward_axis, dtype=np.float32),
        # Rest/reference local transforms of the human skeleton, shape [J, 7].骨架的变换描述
        "human_reference_local_transforms": np.asarray(skeleton.reference_local_transforms, dtype=np.float32),
        # Per-frame human local joint transforms, shape [T, J, 7].每一帧里，每个 human joint 相对于其父节点局部坐标系的变换
        "human_local_transforms": human_local_transforms,
        # Per-frame human global body positions computed from local transforms, shape [T, J, 3].
        "human_global_pos": human_global_pos,
        # Per-frame human global body orientations in xyzw order, shape [T, J, 4].
        "human_global_quat": human_global_quat,
    }

    if robot_body_names is not None and robot_body_pos is not None and robot_body_quat is not None:
        payload.update(
            {
                # Ordered target robot body/link names, aligned with robot_body_* arrays.
                "robot_body_names": np.asarray(robot_body_names),
                # Per-frame target robot body positions in world frame, shape [T, B, 3].
                "robot_body_pos": np.asarray(robot_body_pos, dtype=np.float32),
                # Per-frame target robot body orientations in xyzw order, shape [T, B, 4].
                "robot_body_quat": np.asarray(robot_body_quat, dtype=np.float32),
                # Finite-difference linear velocities of target robot bodies, shape [T, B, 3].
                "robot_body_lin_vel": _forward_difference(robot_body_pos, dt),
                # Finite-difference angular velocities of target robot bodies, shape [T, B, 3].
                "robot_body_ang_vel": _quat_to_angular_velocity_xyzw(robot_body_quat, dt),
            }
        )

    if source_fps is not None and source_robot_motion is not None and source_human_local_transforms is not None:
        source_robot_motion = np.asarray(source_robot_motion, dtype=np.float32)
        source_robot_root_pos, source_robot_root_quat, source_robot_joint_pos = _split_robot_motion(source_robot_motion)
        payload.update(
            {
                # Frame rate of the original source sequence before retargeting.
                "source_fps": np.asarray(source_fps, dtype=np.int32),
                # Number of frames in the original source sequence.
                "source_num_frames": np.asarray(source_human_local_transforms.shape[0], dtype=np.int32),
                # Source robot root translation in world frame, shape [T, 3].
                "source_robot_root_pos": source_robot_root_pos,
                # Source robot root orientation quaternion in xyzw order, shape [T, 4].
                "source_robot_root_quat": source_robot_root_quat,
                # Source robot joint positions before retargeting, shape [T, num_robot_joints].
                "source_robot_joint_pos": source_robot_joint_pos,
                # Source human local joint transforms paired with the source motion, shape [T, J, 7].
                "source_human_local_transforms": np.asarray(source_human_local_transforms, dtype=np.float32),
            }
        )

    return payload


def save_retarget_npz(
    file_path,
    *,
    fps,
    skeleton,
    human_local_transforms,
    robot_motion,
    robot_name,
    robot_joint_names,
    robot_body_names=None,
    robot_body_pos=None,
    robot_body_quat=None,
    source_fps=None,
    source_robot_motion=None,
    source_human_local_transforms=None,
) -> None:
    path = Path(file_path)
    payload = _build_retarget_payload(
        fps=fps,
        skeleton=skeleton,
        human_local_transforms=human_local_transforms,
        robot_motion=robot_motion,
        robot_name=robot_name,
        robot_joint_names=robot_joint_names,
        robot_body_names=robot_body_names,
        robot_body_pos=robot_body_pos,
        robot_body_quat=robot_body_quat,
        source_fps=source_fps,
        source_robot_motion=source_robot_motion,
        source_human_local_transforms=source_human_local_transforms,
    )
    np.savez(path, **payload)
