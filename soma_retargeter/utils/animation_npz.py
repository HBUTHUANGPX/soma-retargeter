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

    return global_pos, np.asarray(global_quat_xyzw, dtype=np.float32)


def _split_robot_motion(robot_motion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot_motion = np.asarray(robot_motion, dtype=np.float32)
    robot_root_pos = robot_motion[:, :3]
    robot_root_quat = robot_motion[:, 3:7]
    robot_joint_pos = robot_motion[:, 7:]
    return robot_root_pos, robot_root_quat, robot_joint_pos


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
    human_global_pos, human_global_quat = _compute_human_global_transforms(
        human_local_transforms, np.asarray(skeleton.parent_indices, dtype=np.int32)
    )

    robot_motion = np.asarray(robot_motion, dtype=np.float32)
    robot_root_pos, robot_root_quat, robot_joint_pos = _split_robot_motion(robot_motion)

    payload = {
        "fps": np.asarray(fps, dtype=np.int32),
        "num_frames": np.asarray(human_local_transforms.shape[0], dtype=np.int32),
        "scalar_first": np.asarray(False),
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

    if robot_body_names is not None and robot_body_pos is not None and robot_body_quat is not None:
        payload.update(
            {
                "robot_body_names": np.asarray(robot_body_names),
                "robot_body_pos": np.asarray(robot_body_pos, dtype=np.float32),
                "robot_body_quat": np.asarray(robot_body_quat, dtype=np.float32),
            }
        )

    if source_fps is not None and source_robot_motion is not None and source_human_local_transforms is not None:
        source_robot_motion = np.asarray(source_robot_motion, dtype=np.float32)
        source_robot_root_pos, source_robot_root_quat, source_robot_joint_pos = _split_robot_motion(source_robot_motion)
        payload.update(
            {
                "source_fps": np.asarray(source_fps, dtype=np.int32),
                "source_num_frames": np.asarray(source_human_local_transforms.shape[0], dtype=np.int32),
                "source_robot_root_pos": source_robot_root_pos,
                "source_robot_root_quat": source_robot_root_quat,
                "source_robot_joint_pos": source_robot_joint_pos,
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
