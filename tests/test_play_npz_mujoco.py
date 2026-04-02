from pathlib import Path

import numpy as np

from app.play_npz_mujoco import (
    apply_visualization_frame,
    compute_global_joint_positions,
    load_motion_npz,
    qpos_from_robot_frame,
)


def test_qpos_from_robot_frame_skips_frame_index():
    robot_frame = np.array([12.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 9.0], dtype=np.float32)

    qpos = qpos_from_robot_frame(robot_frame, expected_nq=8)

    np.testing.assert_allclose(qpos, np.array([1.0, 2.0, 3.0, 0.4, 0.1, 0.2, 0.3, 9.0], dtype=np.float32))


def test_compute_global_joint_positions_accumulates_parent_transforms():
    local_transforms = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    parent_indices = np.array([-1, 0, 1], dtype=np.int32)

    positions = compute_global_joint_positions(local_transforms, parent_indices)

    np.testing.assert_allclose(
        positions[0],
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def test_load_motion_npz_reads_expected_payload(tmp_path: Path):
    npz_path = tmp_path / "demo.npz"
    np.savez(
        npz_path,
        bvh_local_transforms=np.zeros((2, 3, 7), dtype=np.float32),
        bvh_parent_indices=np.array([-1, 0, 1], dtype=np.int32),
        bvh_joint_names=np.array(["root", "joint1", "joint2"]),
        bvh_sample_rate=np.array(120.0, dtype=np.float32),
        robot_data=np.zeros((2, 37), dtype=np.float32),
        robot_dim_names=np.array(["Frame", "a"]),
        robot_name=np.array("Q1"),
    )

    motion = load_motion_npz(npz_path)

    assert motion.bvh_local_transforms.shape == (2, 3, 7)
    np.testing.assert_array_equal(motion.bvh_parent_indices, np.array([-1, 0, 1], dtype=np.int32))
    assert motion.bvh_joint_names == ["root", "joint1", "joint2"]
    assert motion.fps == 120.0
    assert motion.robot_name == "Q1"


def test_apply_visualization_frame_rotates_y_up_positions_to_z_up():
    positions = np.array([[[0.0, 1.5, 0.0]]], dtype=np.float32)
    rotations = np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)

    transformed_positions, transformed_rotations = apply_visualization_frame(positions, rotations)

    np.testing.assert_allclose(transformed_positions[0, 0], np.array([0.0, 0.0, 1.5], dtype=np.float32), atol=1e-6)
    assert transformed_rotations.shape == rotations.shape
