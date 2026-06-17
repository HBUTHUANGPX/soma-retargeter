from pathlib import Path

import numpy as np

from app.load_npz_qpos_example import load_qpos_from_npz


def test_load_qpos_from_npz_returns_newton_ordered_qpos(tmp_path: Path):
    npz_path = tmp_path / "motion.npz"
    np.savez(
        npz_path,
        fps=np.array(50, dtype=np.int32),
        num_frames=np.array(2, dtype=np.int32),
        scalar_first=np.array(True),
        robot_name=np.array("unitree_g1"),
        robot_joint_names=np.array(["hip", "knee"]),
        robot_root_pos=np.array(
            [[0.0, 0.0, 0.5], [1.0, 2.0, 0.75]],
            dtype=np.float32,
        ),
        robot_root_quat=np.array(
            [[1.0, 0.1, 0.2, 0.3], [0.9, 0.4, 0.5, 0.6]],
            dtype=np.float32,
        ),
        robot_joint_pos=np.array(
            [[0.11, 0.22], [0.33, 0.44]],
            dtype=np.float32,
        ),
        human_local_transforms=np.zeros((2, 1, 7), dtype=np.float32),
        human_parent_indices=np.array([-1], dtype=np.int32),
        human_joint_names=np.array(["root"]),
    )

    qpos = load_qpos_from_npz(npz_path)

    assert qpos.shape == (2, 9)
    np.testing.assert_allclose(
        qpos,
        np.array(
            [
                [0.0, 0.0, 0.5, 0.1, 0.2, 0.3, 1.0, 0.11, 0.22],
                [1.0, 2.0, 0.75, 0.4, 0.5, 0.6, 0.9, 0.33, 0.44],
            ],
            dtype=np.float32,
        ),
    )
