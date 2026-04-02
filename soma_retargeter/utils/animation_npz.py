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


def _quat_rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[..., :3]
    qw = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return (vec + 2.0 * (qw * uv + uuv)).astype(np.float32, copy=False)


def _quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    return quat[..., [3, 0, 1, 2]]


def _compute_sample_times(sample_rate: float, num_frames: int, output_fps: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    if num_frames == 1:
        return np.zeros((1,), dtype=np.float32)

    duration = (num_frames - 1) / float(sample_rate)
    times = np.arange(0.0, duration, 1.0 / float(output_fps), dtype=np.float32)
    if times.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return times


def _sample_human_local_transforms(animation, output_fps: int) -> np.ndarray:
    times = _compute_sample_times(animation.sample_rate, animation.num_frames, output_fps)
    return np.asarray([animation.sample(float(t)) for t in times], dtype=np.float32)


def _compute_human_global_transforms(local_transforms: np.ndarray, parent_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_frames, num_joints = local_transforms.shape[:2]
    global_pos = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    global_quat_xyzw = np.zeros((num_frames, num_joints, 4), dtype=np.float32)

    local_pos = local_transforms[..., :3]
    local_quat_xyzw = local_transforms[..., 3:7]

    for joint_idx in range(num_joints):
        parent_idx = parent_indices[joint_idx]
        if parent_idx < 0:
            global_pos[:, joint_idx] = local_pos[:, joint_idx]
            global_quat_xyzw[:, joint_idx] = local_quat_xyzw[:, joint_idx]
            continue

        parent_pos = global_pos[:, parent_idx]
        parent_quat = global_quat_xyzw[:, parent_idx]
        global_pos[:, joint_idx] = parent_pos + _quat_rotate_xyzw(parent_quat, local_pos[:, joint_idx])
        global_quat_xyzw[:, joint_idx] = _quat_mul_xyzw(parent_quat, local_quat_xyzw[:, joint_idx])

    return global_pos, _quat_xyzw_to_wxyz(global_quat_xyzw)


def _split_robot_motion(robot_motion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot_motion = np.asarray(robot_motion, dtype=np.float32)
    robot_root_pos = robot_motion[:, :3]
    robot_root_quat = _quat_xyzw_to_wxyz(robot_motion[:, 3:7])
    robot_joint_pos = robot_motion[:, 7:]
    return robot_root_pos, robot_root_quat, robot_joint_pos


def _extract_source_robot_motion(csv_buffer) -> np.ndarray:
    if hasattr(csv_buffer, "data"):
        return np.asarray(csv_buffer.data, dtype=np.float32)
    return np.asarray(
        [csv_buffer.get_data(frame_idx) for frame_idx in range(csv_buffer.num_frames)],
        dtype=np.float32,
    )


def _resample_robot_motion(csv_buffer, output_fps: int) -> np.ndarray:
    times = _compute_sample_times(csv_buffer.sample_rate, csv_buffer.num_frames, output_fps)
    return np.asarray([csv_buffer.sample(float(t)) for t in times], dtype=np.float32)


def _build_retarget_payload(animation, csv_buffer, robot_name, robot_joint_names, output_fps, include_source_data) -> dict:
    skeleton = animation.skeleton

    human_local_transforms = _sample_human_local_transforms(animation, output_fps)
    human_global_pos, human_global_quat = _compute_human_global_transforms(
        human_local_transforms, np.asarray(skeleton.parent_indices, dtype=np.int32)
    )

    robot_motion = _resample_robot_motion(csv_buffer, output_fps)
    robot_root_pos, robot_root_quat, robot_joint_pos = _split_robot_motion(robot_motion)

    payload = {
        "fps": np.asarray(output_fps, dtype=np.int32),
        "num_frames": np.asarray(human_local_transforms.shape[0], dtype=np.int32),
        "robot_name": np.asarray(robot_name),
        "robot_joint_names": np.asarray(robot_joint_names),
        "robot_root_pos": robot_root_pos,
        "robot_root_quat": robot_root_quat,
        "robot_joint_pos": robot_joint_pos,
        "human_joint_names": np.asarray(skeleton.joint_names),
        "human_parent_indices": np.asarray(skeleton.parent_indices, dtype=np.int32),
        "human_up_axis": np.asarray(skeleton.up_axis, dtype=np.float32),
        "human_forward_axis": np.asarray(skeleton.forward_axis, dtype=np.float32),
        "human_reference_local_transforms": np.asarray(skeleton.reference_local_transforms, dtype=np.float32),
        "human_local_transforms": human_local_transforms,
        "human_global_pos": human_global_pos,
        "human_global_quat": human_global_quat,
    }

    if include_source_data:
        source_robot_motion = _extract_source_robot_motion(csv_buffer)
        source_robot_root_pos, source_robot_root_quat, source_robot_joint_pos = _split_robot_motion(source_robot_motion)
        payload.update(
            {
                "source_fps": np.asarray(animation.sample_rate, dtype=np.int32),
                "source_num_frames": np.asarray(animation.num_frames, dtype=np.int32),
                "source_robot_root_pos": source_robot_root_pos,
                "source_robot_root_quat": source_robot_root_quat,
                "source_robot_joint_pos": source_robot_joint_pos,
                "source_human_local_transforms": np.asarray(animation.local_transforms, dtype=np.float32),
            }
        )

    return payload


def save_retarget_npz(
    file_path,
    animation,
    csv_buffer,
    robot_name,
    robot_joint_names,
    output_fps: int = 50,
    include_source_data: bool = False,
) -> None:
    path = Path(file_path)
    payload = _build_retarget_payload(
        animation,
        csv_buffer,
        robot_name=robot_name,
        robot_joint_names=robot_joint_names,
        output_fps=output_fps,
        include_source_data=include_source_data,
    )
    np.savez(path, **payload)
