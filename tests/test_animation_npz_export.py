from pathlib import Path

import numpy as np

from soma_retargeter.utils.animation_npz import save_retarget_npz


def test_save_retarget_npz_writes_bvh_and_robot_payloads(tmp_path: Path):
    bvh_local_transforms = np.asarray(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [1.1, 1.2, 1.3, 0.0, 0.0, 0.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )

    class SkeletonStub:
        def __init__(self):
            self.forward_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            self.joint_names = ["root", "hand"]
            self.num_joints = 2
            self.parent_indices = np.array([-1, 0], dtype=np.int32)
            self.up_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.reference_local_transforms = bvh_local_transforms[0]

    class AnimationStub:
        def __init__(self):
            self.num_frames = 2
            self.sample_rate = 120.0
            self.local_transforms = bvh_local_transforms
            self.skeleton = SkeletonStub()

    class CSVBufferStub:
        def __init__(self):
            self.num_frames = 2
            self._frames = [
                np.array([10.0, 11.0, 12.0], dtype=np.float32),
                np.array([20.0, 21.0, 22.0], dtype=np.float32),
            ]

        def get_data(self, frame_idx):
            return self._frames[frame_idx]

    output_path = tmp_path / "example_motion.npz"
    save_retarget_npz(
        output_path,
        AnimationStub(),
        CSVBufferStub(),
        robot_name="Q1",
        csv_header=["Frame", "root_x", "root_y", "root_z"],
    )

    exported = np.load(output_path, allow_pickle=False)

    np.testing.assert_allclose(exported["bvh_local_transforms"], bvh_local_transforms)
    assert exported["bvh_num_frames"].item() == 2
    assert exported["bvh_sample_rate"].item() == 120.0
    np.testing.assert_allclose(exported["bvh_forward_axis"], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert exported["bvh_joint_names"].tolist() == ["root", "hand"]
    assert exported["bvh_num_joints"].item() == 2
    np.testing.assert_array_equal(exported["bvh_parent_indices"], np.array([-1, 0], dtype=np.int32))
    np.testing.assert_allclose(exported["bvh_reference_local_transforms"], bvh_local_transforms[0])
    np.testing.assert_allclose(exported["bvh_up_axis"], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert exported["robot_num_frames"].item() == 2
    assert exported["robot_dim_names"].tolist() == ["Frame", "root_x", "root_y", "root_z"]
    assert exported["robot_name"].tolist() == "Q1"
    np.testing.assert_allclose(
        exported["robot_data"],
        np.array(
            [
                [0.0, 10.0, 11.0, 12.0],
                [1.0, 20.0, 21.0, 22.0],
            ],
            dtype=np.float32,
        ),
    )
