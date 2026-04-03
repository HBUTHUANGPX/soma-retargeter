from pathlib import Path

import numpy as np

from soma_retargeter.utils.animation_npz import save_retarget_npz


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
    save_retarget_npz(
        output_path,
        AnimationStub(),
        CSVBufferStub(),
        robot_name="Q1",
        robot_joint_names=["joint_a", "joint_b"],
        output_fps=50,
        include_source_data=False,
    )

    exported = np.load(output_path, allow_pickle=False)

    assert exported["fps"].item() == 50
    assert exported["num_frames"].item() == 2
    assert exported["scalar_first"].item() is False
    assert exported["robot_name"].tolist() == "Q1"
    assert exported["robot_joint_names"].tolist() == ["joint_a", "joint_b"]
    np.testing.assert_allclose(exported["robot_root_pos"], np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=np.float32))
    np.testing.assert_allclose(exported["robot_root_quat"], np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(exported["robot_joint_pos"], np.array([[10.0, 20.0], [12.0, 22.0]], dtype=np.float32))
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
    save_retarget_npz(
        output_path,
        AnimationStub(),
        CSVBufferStub(),
        robot_name="unitree_g1",
        robot_joint_names=["joint_a", "joint_b"],
        output_fps=50,
        include_source_data=True,
    )

    exported = np.load(output_path, allow_pickle=False)

    assert exported["source_fps"].item() == 100
    assert exported["source_num_frames"].item() == 5
    np.testing.assert_allclose(exported["source_robot_root_pos"][0], np.array([0.0, 0.0, 0.5], dtype=np.float32))
    np.testing.assert_allclose(exported["source_robot_root_quat"][0], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(exported["source_robot_joint_pos"][0], np.array([10.0, 20.0], dtype=np.float32))
    np.testing.assert_allclose(exported["source_human_local_transforms"][0], AnimationStub().local_transforms[0])
