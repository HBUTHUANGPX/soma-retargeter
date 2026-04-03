import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

try:
    from app.motion_npz_player_common import (
        apply_visualization_frame,
        compute_global_joint_transforms,
        load_motion_npz,
        qpos_from_robot_frame,
        quat_mul_batch,
        quat_rotate_batch,
        quat_to_mat,
    )
except ModuleNotFoundError:
    from motion_npz_player_common import (
        apply_visualization_frame,
        compute_global_joint_transforms,
        load_motion_npz,
        qpos_from_robot_frame,
        quat_mul_batch,
        quat_rotate_batch,
        quat_to_mat,
    )


def draw_sphere(scene, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def draw_line(
    scene, start: np.ndarray, end: np.ndarray, width: float, rgba: np.ndarray
) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=np.asarray(start, dtype=np.float64),
        to=np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def draw_axes(
    scene,
    position: np.ndarray,
    rotation: np.ndarray,
    axis_length: float,
    axis_width: float,
) -> None:
    rot_mat = quat_to_mat(rotation)
    colors = (
        np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    for axis_idx in range(3):
        if scene.ngeom >= scene.maxgeom:
            return
        draw_line(
            scene,
            position,
            position + rot_mat[:, axis_idx] * axis_length,
            axis_width,
            colors[axis_idx],
        )


def draw_animation_frame(
    viewer,
    positions: np.ndarray,
    rotations: np.ndarray,
    parent_indices: np.ndarray,
    show_axes: bool,
    joint_radius: float,
    bone_width: float,
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    joint_rgba = np.array([1.0, 0.8, 0.1, 0.9], dtype=np.float32)
    bone_rgba = np.array([0.3, 0.9, 1.0, 0.7], dtype=np.float32)

    for joint_idx, position in enumerate(positions):
        if scene.ngeom >= scene.maxgeom:
            break
        draw_sphere(scene, position, joint_radius, joint_rgba)
        parent_idx = parent_indices[joint_idx]
        if parent_idx >= 0 and scene.ngeom < scene.maxgeom:
            draw_line(scene, positions[parent_idx], position, bone_width, bone_rgba)
        if show_axes and scene.ngeom + 3 < scene.maxgeom:
            draw_axes(
                scene,
                position,
                rotations[joint_idx],
                axis_length=0.06,
                axis_width=0.003,
            )


def set_default_camera(viewer) -> None:
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -15.0
    viewer.cam.fixedcamid = -1
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE


def default_xml_path(robot_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    normalized = robot_name.lower()
    if "q1" in normalized:
        return repo_root / "assets/Q1/mjcf/Q1_wo_hand.xml"
    if "g1" in normalized or "unitree" in normalized:
        return repo_root / "assets/unitree_g1/g1_29dof_rev_1_0.xml"
    raise ValueError(
        f"Unable to infer xml path for robot {robot_name!r}. Please provide --xml-path."
    )


def play_motion(npz_path: Path, xml_path: Path, loop: bool, show_axes: bool) -> None:
    motion = load_motion_npz(npz_path)
    global_positions, global_rotations = compute_global_joint_transforms(
        motion.bvh_local_transforms,
        motion.bvh_parent_indices,
    )
    global_positions, global_rotations = apply_visualization_frame(
        global_positions, global_rotations
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    dt = 1.0 / max(motion.fps, 1e-6)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_default_camera(viewer)
        frame_idx = 0
        while viewer.is_running():
            qpos = qpos_from_robot_frame(
                motion.robot_data[frame_idx], expected_nq=model.nq
            )
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            draw_animation_frame(
                viewer,
                global_positions[frame_idx],
                global_rotations[frame_idx],
                motion.bvh_parent_indices,
                show_axes=show_axes,
                joint_radius=0.025,
                bone_width=0.008,
            )
            viewer.sync()
            time.sleep(dt)

            frame_idx += 1
            if frame_idx >= motion.robot_data.shape[0]:
                if not loop:
                    break
                frame_idx = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play retargeted motion npz in MuJoCo."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default="soma-retargeter/assets/motions/test-export/dance_hiphop_shuffle_square_R_fast_002__A318.npz",
        help="Path to a retargeted motion npz file.",
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default="assets/unitree_g1/g1_29dof_rev_1_0.xml",
        help="Path to robot MuJoCo XML.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--show-axes", action="store_true", help="Draw BVH joint coordinate axes."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = (
        args.xml_path
        if args.xml_path is not None
        else default_xml_path(load_motion_npz(args.npz).robot_name)
    )
    play_motion(args.npz, xml_path, loop=args.loop, show_axes=args.show_axes)


if __name__ == "__main__":
    main()
