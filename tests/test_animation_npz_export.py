from pathlib import Path

import numpy as np

from soma_retargeter.utils.animation_npz import save_retarget_npz


def quat_xyzw_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    half_angle = angle_rad * 0.5
    sin_half = np.sin(half_angle)
    return np.array(
        [
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half_angle),
        ],
        dtype=np.float32,
    )


class SkeletonStub:
    def __init__(self):
        self.forward_axis = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self.joint_names = ["root", "hand"]
        self.num_joints = 2
        self.parent_indices = np.array([-1, 0], dtype=np.int32)
        self.up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.reference_local_transforms = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


class AnimationStub:
    def __init__(self):
        self.sample_rate = 100.0
        self.skeleton = SkeletonStub()
        self.local_transforms = np.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ],
            dtype=np.float32,
        )
        self.num_frames = self.local_transforms.shape[0]

    def sample(self, time_s: float):
        frame_float = min(time_s * self.sample_rate, self.num_frames - 1)
        frame0 = int(np.floor(frame_float))
        frame1 = min(frame0 + 1, self.num_frames - 1)
        blend = frame_float - frame0
        return (
            self.local_transforms[frame0] * (1.0 - blend)
            + self.local_transforms[frame1] * blend
        ).astype(np.float32)


class CSVBufferStub:
    def __init__(self):
        self.sample_rate = 100.0
        self.data = np.array(
            [
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 10.0, 20.0],
                [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 11.0, 21.0],
                [2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 12.0, 22.0],
                [3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 13.0, 23.0],
                [4.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 14.0, 24.0],
            ],
            dtype=np.float32,
        )
        self.num_frames = self.data.shape[0]

    def get_data(self, frame_idx: int):
        return self.data[frame_idx]

    def sample(self, time_s: float):
        frame_float = min(time_s * self.sample_rate, self.num_frames - 1)
        frame0 = int(np.floor(frame_float))
        frame1 = min(frame0 + 1, self.num_frames - 1)
        blend = frame_float - frame0
        return (
            self.data[frame0] * (1.0 - blend)
            + self.data[frame1] * blend
        ).astype(np.float32)


def test_save_retarget_npz_writes_resampled_isaaclab_ready_payload(tmp_path: Path):
    output_path = tmp_path / "example_motion.npz"
    skeleton = AnimationStub().skeleton
    human_local_transforms = np.array(
        [
            AnimationStub().local_transforms[0],
            AnimationStub().local_transforms[2],
        ],
        dtype=np.float32,
    )
    robot_motion = np.array(
        [
            CSVBufferStub().data[0],
            CSVBufferStub().data[2],
        ],
        dtype=np.float32,
    )
    save_retarget_npz(
        output_path,
        fps=50,
        skeleton=skeleton,
        human_local_transforms=human_local_transforms,
        robot_motion=robot_motion,
        robot_name="Q1",
        robot_joint_names=["joint_a", "joint_b"],
        robot_body_names=["joint_a", "joint_b"],
        robot_body_pos=np.array(
            [
                [[0.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
                [[2.0, 0.0, 0.5], [2.0, 1.0, 0.5]],
            ],
            dtype=np.float32,
        ),
        robot_body_quat=np.array(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    )

    exported = np.load(output_path, allow_pickle=False)

    assert exported["fps"].item() == 50
    assert exported["num_frames"].item() == 2
    assert exported["scalar_first"].item() is False
    assert exported["robot_name"].tolist() == "Q1"
    assert exported["robot_joint_names"].tolist() == ["joint_a", "joint_b"]
    assert exported["robot_body_names"].tolist() == ["joint_a", "joint_b"]
    np.testing.assert_allclose(exported["robot_root_pos"], np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=np.float32))
    np.testing.assert_allclose(exported["robot_root_quat"], np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(exported["robot_joint_pos"], np.array([[10.0, 20.0], [12.0, 22.0]], dtype=np.float32))
    np.testing.assert_allclose(
        exported["robot_body_pos"],
        np.array(
            [
                [[0.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
                [[2.0, 0.0, 0.5], [2.0, 1.0, 0.5]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        exported["robot_body_quat"],
        np.array(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert exported["human_joint_names"].tolist() == ["root", "hand"]
    np.testing.assert_array_equal(exported["human_parent_indices"], np.array([-1, 0], dtype=np.int32))
    np.testing.assert_allclose(exported["human_up_axis"], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(exported["human_forward_axis"], np.array([0.0, -1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(exported["human_reference_local_transforms"], AnimationStub().skeleton.reference_local_transforms)
    assert exported["human_local_transforms"].shape == (2, 2, 7)
    np.testing.assert_allclose(exported["human_global_pos"][0], np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(exported["human_global_pos"][1], np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(exported["human_global_quat"][0, 0], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert "source_fps" not in exported.files


def test_save_retarget_npz_can_include_minimal_source_payload(tmp_path: Path):
    output_path = tmp_path / "example_motion_with_source.npz"
    skeleton = AnimationStub().skeleton
    save_retarget_npz(
        output_path,
        fps=50,
        skeleton=skeleton,
        human_local_transforms=np.array(
            [
                AnimationStub().local_transforms[0],
                AnimationStub().local_transforms[2],
            ],
            dtype=np.float32,
        ),
        robot_motion=np.array(
            [
                CSVBufferStub().data[0],
                CSVBufferStub().data[2],
            ],
            dtype=np.float32,
        ),
        robot_name="unitree_g1",
        robot_joint_names=["joint_a", "joint_b"],
        source_fps=100,
        source_robot_motion=CSVBufferStub().data,
        source_human_local_transforms=AnimationStub().local_transforms,
    )

    exported = np.load(output_path, allow_pickle=False)

    assert exported["source_fps"].item() == 100
    assert exported["source_num_frames"].item() == 5
    np.testing.assert_allclose(exported["source_robot_root_pos"][0], np.array([0.0, 0.0, 0.5], dtype=np.float32))
    np.testing.assert_allclose(exported["source_robot_root_quat"][0], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(exported["source_robot_joint_pos"][0], np.array([10.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(exported["source_human_local_transforms"][0], AnimationStub().local_transforms[0])


def test_save_retarget_npz_exports_forward_difference_velocities(tmp_path: Path):
    output_path = tmp_path / "example_motion_with_velocities.npz"
    skeleton = AnimationStub().skeleton

    robot_motion = np.array(
        [
            [0.0, 0.0, 0.0, *quat_xyzw_from_axis_angle([0.0, 0.0, 1.0], 0.0), 0.0, 1.0],
            [1.0, 2.0, 3.0, *quat_xyzw_from_axis_angle([0.0, 0.0, 1.0], 0.1), 0.5, 1.5],
            [3.0, 6.0, 9.0, *quat_xyzw_from_axis_angle([0.0, 0.0, 1.0], 0.3), 1.5, 2.5],
        ],
        dtype=np.float32,
    )
    robot_body_pos = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.5, 1.0, 1.5], [1.5, 1.0, 0.0]],
            [[1.5, 3.0, 4.5], [2.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    robot_body_quat = np.array(
        [
            [
                quat_xyzw_from_axis_angle([1.0, 0.0, 0.0], 0.0),
                quat_xyzw_from_axis_angle([0.0, 1.0, 0.0], 0.0),
            ],
            [
                quat_xyzw_from_axis_angle([1.0, 0.0, 0.0], 0.2),
                quat_xyzw_from_axis_angle([0.0, 1.0, 0.0], 0.1),
            ],
            [
                quat_xyzw_from_axis_angle([1.0, 0.0, 0.0], 0.5),
                quat_xyzw_from_axis_angle([0.0, 1.0, 0.0], 0.3),
            ],
        ],
        dtype=np.float32,
    )

    save_retarget_npz(
        output_path,
        fps=10,
        skeleton=skeleton,
        human_local_transforms=np.array(
            [
                AnimationStub().local_transforms[0],
                AnimationStub().local_transforms[1],
                AnimationStub().local_transforms[2],
            ],
            dtype=np.float32,
        ),
        robot_motion=robot_motion,
        robot_name="Q1",
        robot_joint_names=["joint_a", "joint_b"],
        robot_body_names=["body_a", "body_b"],
        robot_body_pos=robot_body_pos,
        robot_body_quat=robot_body_quat,
    )

    exported = np.load(output_path, allow_pickle=False)

    np.testing.assert_allclose(
        exported["robot_root_lin_vel"],
        np.array(
            [
                [10.0, 20.0, 30.0],
                [20.0, 40.0, 60.0],
                [20.0, 40.0, 60.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        exported["robot_joint_vel"],
        np.array(
            [
                [5.0, 5.0],
                [10.0, 10.0],
                [10.0, 10.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        exported["robot_root_ang_vel"],
        np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        exported["robot_body_lin_vel"],
        np.array(
            [
                [[5.0, 10.0, 15.0], [5.0, 10.0, 0.0]],
                [[10.0, 20.0, 30.0], [5.0, 10.0, 0.0]],
                [[10.0, 20.0, 30.0], [5.0, 10.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        exported["robot_body_ang_vel"],
        np.array(
            [
                [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )
