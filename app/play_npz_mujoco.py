import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class MotionNPZ:
    bvh_local_transforms: np.ndarray
    bvh_parent_indices: np.ndarray
    bvh_joint_names: list[str]
    fps: float
    robot_data: np.ndarray
    robot_dim_names: list[str]
    robot_name: str


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[:3]
    qw = quat[3]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return vec + 2.0 * (qw * uv + uuv)


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float32)


def compute_global_joint_transforms(
    local_transforms: np.ndarray, parent_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    num_frames, num_joints = local_transforms.shape[:2]
    global_positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    global_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)

    for frame_idx in range(num_frames):
        for joint_idx in range(num_joints):
            local_pos = local_transforms[frame_idx, joint_idx, :3]
            local_rot = local_transforms[frame_idx, joint_idx, 3:7]
            parent_idx = parent_indices[joint_idx]
            if parent_idx < 0:
                global_positions[frame_idx, joint_idx] = local_pos
                global_rotations[frame_idx, joint_idx] = local_rot
                continue

            parent_rot = global_rotations[frame_idx, parent_idx]
            parent_pos = global_positions[frame_idx, parent_idx]
            global_positions[frame_idx, joint_idx] = parent_pos + quat_rotate(
                parent_rot, local_pos
            )
            global_rotations[frame_idx, joint_idx] = quat_mul(parent_rot, local_rot)

    return global_positions, global_rotations


def compute_global_joint_positions(
    local_transforms: np.ndarray, parent_indices: np.ndarray
) -> np.ndarray:
    positions, _ = compute_global_joint_transforms(local_transforms, parent_indices)
    return positions


def qpos_from_robot_frame(robot_frame: np.ndarray, expected_nq: int) -> np.ndarray:
    qpos = np.asarray(robot_frame[1:], dtype=np.float32).copy()
    if qpos.shape[0] >= 7:
        qpos[3:7] = np.array([qpos[6], qpos[3], qpos[4], qpos[5]], dtype=np.float32)
    if qpos.shape[0] != expected_nq:
        raise ValueError(
            f"Robot frame qpos width {qpos.shape[0]} does not match model.nq {expected_nq}."
        )
    return qpos


def load_motion_npz(npz_path: str | Path) -> MotionNPZ:
    payload = np.load(npz_path, allow_pickle=False)
    return MotionNPZ(
        bvh_local_transforms=np.asarray(
            payload["bvh_local_transforms"], dtype=np.float32
        ),
        bvh_parent_indices=np.asarray(payload["bvh_parent_indices"], dtype=np.int32),
        bvh_joint_names=payload["bvh_joint_names"].tolist(),
        fps=float(payload["bvh_sample_rate"]),
        robot_data=np.asarray(payload["robot_data"], dtype=np.float32),
        robot_dim_names=payload["robot_dim_names"].tolist(),
        robot_name=str(payload["robot_name"].tolist()),
    )


def apply_visualization_frame(
    positions: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # BVH global pose data is effectively Y-up; rotate it into MuJoCo's Z-up world.
    y_up_to_z_up = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    corrected_positions = np.empty_like(positions)
    corrected_rotations = np.empty_like(rotations)

    for frame_idx in range(positions.shape[0]):
        for joint_idx in range(positions.shape[1]):
            corrected_positions[frame_idx, joint_idx] = quat_rotate(
                y_up_to_z_up, positions[frame_idx, joint_idx]
            )
            corrected_rotations[frame_idx, joint_idx] = quat_mul(
                quat_mul(y_up_to_z_up, rotations[frame_idx, joint_idx]),
                quat_conjugate(y_up_to_z_up),
            )

    return corrected_positions, corrected_rotations


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
