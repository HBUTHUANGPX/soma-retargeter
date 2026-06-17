import argparse
from pathlib import Path

import numpy as np

try:
    from app.motion_npz_player_common import load_motion_npz, qpos_from_robot_frame
except ModuleNotFoundError:
    from motion_npz_player_common import load_motion_npz, qpos_from_robot_frame


def load_qpos_from_npz(
    npz_path: str | Path,
    *,
    quat_order: str = "newton",
    expected_nq: int | None = None,
) -> np.ndarray:
    motion = load_motion_npz(npz_path)
    nq = expected_nq if expected_nq is not None else motion.robot_data.shape[1] - 1
    return np.stack(
        [
            qpos_from_robot_frame(
                robot_frame,
                expected_nq=nq,
                scalar_first=motion.scalar_first,
                quat_order=quat_order,
            )
            for robot_frame in motion.robot_data
        ],
        axis=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a motion npz file and print robot qpos frames."
    )
    parser.add_argument("npz", type=Path, help="Path to the motion .npz file.")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to print. Defaults to 0.",
    )
    parser.add_argument(
        "--quat-order",
        choices=("newton", "mujoco"),
        default="newton",
        help="Target root quaternion order for qpos. Defaults to newton.",
    )
    parser.add_argument(
        "--expected-nq",
        type=int,
        default=None,
        help="Optional expected qpos width. Defaults to the npz qpos width.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qpos = load_qpos_from_npz(
        args.npz,
        quat_order=args.quat_order,
        expected_nq=args.expected_nq,
    )
    if args.frame < 0 or args.frame >= qpos.shape[0]:
        raise ValueError(f"--frame must be in [0, {qpos.shape[0] - 1}], got {args.frame}.")

    np.set_printoptions(precision=6, suppress=True)
    print(f"qpos shape: {qpos.shape}")
    print(f"qpos[{args.frame}]:")
    print(qpos[args.frame])


if __name__ == "__main__":
    main()
