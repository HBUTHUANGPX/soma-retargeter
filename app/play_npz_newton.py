import argparse
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp

from motion_npz_player_common import (
    load_motion_npz,
    qpos_from_robot_frame,
)
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.utils.newton_asset_utils import as_newton_usd_source


def default_asset_path(robot_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    normalized = robot_name.lower()
    if "q1" in normalized:
        return repo_root / "assets/Q1/mjcf/Q1_wo_hand.xml"
    if "g1" in normalized or "unitree" in normalized:
        return repo_root / "assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
    raise ValueError(
        f"Unable to infer Newton asset path for robot {robot_name!r}. Please provide --asset-path."
    )


def build_robot_model(asset_path: Path) -> tuple[newton.Model, object]:
    builder = newton.ModelBuilder()
    builder.add_usd(as_newton_usd_source(asset_path))
    model = builder.finalize()
    return model, model.state()


def set_default_camera(viewer) -> None:
    viewer.camera.pos = wp.vec3(3.0, 3.0, 1.8)
    viewer.camera.yaw = 135.0
    viewer.camera.pitch = -20.0


class HumanOverlay:
    def __init__(self, motion):
        skeleton = Skeleton(
            num_joints=len(motion.bvh_joint_names),
            joint_names=motion.bvh_joint_names,
            parent_indices=motion.bvh_parent_indices,
            local_transforms=motion.bvh_local_transforms[0],
        )
        self.instance = SkeletonInstance(
            skeleton,
            wp.vec3(1.0, 0.8, 0.1),
            wp.transform(
                wp.vec3(0.0, 0.0, 0.0),
                wp.quat(np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
            ),
        )
        self.skeleton_renderer = SkeletonRenderer(skeleton, [0])
        self.coordinate_renderer = CoordinateRenderer()

    def draw(self, viewer, local_transforms: np.ndarray, show_axes: bool) -> None:
        self.instance.set_local_transforms(local_transforms)
        self.skeleton_renderer.draw(viewer, self.instance, 0)
        if show_axes:
            self.coordinate_renderer.draw(viewer, self.instance.compute_global_transforms(), 0.06, 0)
        else:
            self.coordinate_renderer.clear(viewer)

    def clear(self, viewer) -> None:
        self.skeleton_renderer.clear(viewer)
        self.coordinate_renderer.clear(viewer)


def play_motion(npz_path: Path, asset_path: Path, loop: bool, show_axes: bool, viewer) -> None:
    motion = load_motion_npz(npz_path)
    model, state = build_robot_model(asset_path)
    viewer.set_model(model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))
    set_default_camera(viewer)

    overlay = HumanOverlay(motion)
    dt = 1.0 / max(motion.fps, 1e-6)

    frame_idx = 0
    sim_time = 0.0
    while viewer.is_running():
        qpos = qpos_from_robot_frame(
            motion.robot_data[frame_idx],
            expected_nq=model.joint_coord_count,
            scalar_first=motion.scalar_first,
            quat_order="newton",
        )
        wp.copy(model.joint_q, wp.array(qpos, dtype=wp.float32), 0, 0, model.joint_coord_count)
        model.joint_qd.zero_()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state, None)

        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        overlay.draw(viewer, motion.bvh_local_transforms[frame_idx], show_axes=show_axes)
        viewer.end_frame()

        time.sleep(dt)
        sim_time += dt
        frame_idx += 1
        if frame_idx >= motion.robot_data.shape[0]:
            if not loop:
                break
            frame_idx = 0
    print("Playback finished.")
    overlay.clear(viewer)
def main() -> None:
    import newton.examples

    parser = newton.examples.create_parser()
    parser.add_argument(
        "--npz",
        type=Path,
        default="soma-retargeter/assets/motions/test-export/dance_hiphop_shuffle_square_R_fast_002__A318.npz",
        help="Path to a retargeted motion npz file.",
    )
    parser.add_argument(
        "--asset-path",
        type=Path,
        default=None,
        help="Path to robot Newton-importable asset (USD/MJCF/XML).",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--show-axes", action="store_true", help="Draw BVH joint coordinate axes."
    )

    viewer, args = newton.examples.init(parser)
    motion = load_motion_npz(args.npz)
    asset_path = args.asset_path if args.asset_path is not None else default_asset_path(motion.robot_name)
    print(f"Using asset path: {asset_path}")
    play_motion(args.npz, asset_path, loop=args.loop, show_axes=args.show_axes, viewer=viewer)
    viewer.close()


if __name__ == "__main__":
    main()
