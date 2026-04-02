from pathlib import Path

import numpy as np


def build_robot_data(csv_buffer, csv_header):
    robot_rows = []
    for frame_idx in range(csv_buffer.num_frames):
        frame_data = csv_buffer.get_data(frame_idx)
        robot_rows.append([frame_idx, *np.asarray(frame_data, dtype=np.float32).tolist()])

    return np.asarray(robot_rows, dtype=np.float32), np.asarray(csv_header)


def save_retarget_npz(file_path, animation, csv_buffer, robot_name, csv_header) -> None:
    path = Path(file_path)
    skeleton = animation.skeleton
    robot_data, robot_dim_names = build_robot_data(csv_buffer, csv_header)

    np.savez(
        path,
        bvh_local_transforms=np.asarray(animation.local_transforms, dtype=np.float32),
        bvh_num_frames=np.asarray(animation.num_frames, dtype=np.int32),
        bvh_sample_rate=np.asarray(animation.sample_rate, dtype=np.float32),
        bvh_forward_axis=np.asarray(skeleton.forward_axis, dtype=np.float32),
        bvh_joint_names=np.asarray(skeleton.joint_names),
        bvh_num_joints=np.asarray(skeleton.num_joints, dtype=np.int32),
        bvh_parent_indices=np.asarray(skeleton.parent_indices, dtype=np.int32),
        bvh_reference_local_transforms=np.asarray(skeleton.reference_local_transforms, dtype=np.float32),
        bvh_up_axis=np.asarray(skeleton.up_axis, dtype=np.float32),
        robot_num_frames=np.asarray(csv_buffer.num_frames, dtype=np.int32),
        robot_dim_names=robot_dim_names,
        robot_name=np.asarray(robot_name),
        robot_data=robot_data,
    )
