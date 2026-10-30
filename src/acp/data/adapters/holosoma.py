"""Normalize Holosoma retargeting output and extract geometric contact labels."""

import argparse
import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from acp.data.contacts import next_onset_targets
from acp.data.motions import fixed_rate_times
from acp.data.scene import BODY_NAMES, MotionScene


def sha256(path):
    return __import__('acp.management.fingerprints', fromlist=['digest']).digest(path)


def holosoma_to_canonical(qpos):
    """MuJoCo free qpos [x,y,z,w,x,y,z,joints] -> canonical [wxyz,xyz,joints]."""
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] < 8 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError("Expected finite Holosoma qpos [T, 7+DoF], at least two frames")
    quaternion_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.any(np.abs(quaternion_norm - 1.0) > 0.02):
        raise ValueError("Expected MuJoCo free-joint quaternion in qpos columns 3:7 (wxyz)")
    canonical = np.empty_like(qpos)
    canonical[:, :4] = qpos[:, [3, 4, 5, 6]]
    canonical[:, 4:7] = qpos[:, :3]
    canonical[:, 7:] = qpos[:, 7:]
    return canonical


def resample_holosoma(qpos, fps, output_fps=50.0):
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("fps values must be finite and positive")
    canonical = holosoma_to_canonical(qpos)
    if fps == output_fps:
        return fixed_rate_times(len(canonical), fps, output_fps), canonical
    source_times = np.arange(len(canonical), dtype=float) / fps
    times = fixed_rate_times(len(canonical), fps, output_fps)
    result = np.empty((len(times), canonical.shape[1]))
    rotations = Rotation.from_quat(canonical[:, [1, 2, 3, 0]])
    interpolated = Slerp(source_times, rotations)(np.clip(times, source_times[0], source_times[-1])).as_quat()
    result[:, :4] = interpolated[:, [3, 0, 1, 2]]
    for column in range(4, canonical.shape[1]):
        result[:, column] = np.interp(times, source_times, canonical[:, column])
    return times, result


def label_holosoma_clip(input_path, target_urdf, terrain_urdf, output_dir,
                        source_clip=None,
                        output_fps=50.0, contact_distance=0.015,
                        penetration_tolerance=0.003, minimum_contact_steps=None,
                        horizon_steps=None, input_method="holosoma"):
    # The paper's controller is 50 Hz; source storage does not have to be.
    # Express durations in seconds so a 30 Hz export keeps the same semantics.
    if minimum_contact_steps is None:
        minimum_contact_steps = max(1, int(np.ceil(0.2 * output_fps)))
    if horizon_steps is None:
        horizon_steps = max(1, int(round(1.5 * output_fps)))
    input_path, target_urdf, terrain_urdf = map(Path, (input_path, target_urdf, terrain_urdf))
    output_dir = Path(output_dir)
    with np.load(input_path, allow_pickle=False) as data:
        qpos, source_fps = data["qpos"].copy(), float(data["fps"])
    times, canonical = resample_holosoma(qpos, source_fps, output_fps)
    scene = MotionScene(target_urdf, terrain_urdf)
    xml_root = ET.parse(target_urdf.with_suffix(".xml")).getroot()
    xml_joint_names = [joint.get("name") for joint in xml_root.findall(".//joint")
                       if joint.get("type") != "free"]
    if xml_joint_names != scene.joint_names:
        raise ValueError("Target URDF and MJCF joint order differs; cannot interpret Holosoma qpos safely")
    if canonical.shape[1] != 7 + len(scene.joint_names):
        raise ValueError(f"qpos has {canonical.shape[1] - 7} joints; target model has {len(scene.joint_names)}")
    distances, witnesses, minimum_distances, landmarks = [], [], [], []
    geometry_valid = []
    for q in canonical:
        scene.set_positions(q)
        frame_distances, frame_witnesses, minimum = scene.contact_geometry(contact_distance)
        distances.append(frame_distances)
        witnesses.append(frame_witnesses)
        minimum_distances.append(minimum)
        landmarks.append(scene.landmarks())
        geometry_valid.append(minimum >= -penetration_tolerance)
    distances = np.asarray(distances)
    witnesses = np.asarray(witnesses)
    minimum_distances = np.asarray(minimum_distances)
    geometry_valid = np.asarray(geometry_valid, dtype=bool)
    landmarks = np.asarray(landmarks)
    lower, upper = scene.plant.GetPositionLowerLimits(), scene.plant.GetPositionUpperLimits()
    joint_limits_valid = ((canonical[:, 7:] >= lower[7:] - 1e-6)
                          & (canonical[:, 7:] <= upper[7:] + 1e-6)).all(axis=1)
    kinematic_valid = geometry_valid & joint_limits_valid
    raw_flags = (distances <= contact_distance) & (distances >= -penetration_tolerance)
    raw_flags &= kinematic_valid[:, None]
    labels = next_onset_targets(raw_flags, witnesses, canonical[:, 4:7], canonical[:, :4],
                                horizon_steps=horizon_steps, minimum_steps=minimum_contact_steps)
    geometry_window_valid = np.array([
        labels["future_complete"][t]
        and kinematic_valid[t:min(t + horizon_steps + 1, len(geometry_valid))].all()
        for t in range(len(geometry_valid))
    ])
    joint_velocities = np.gradient(canonical[:, 7:], times, axis=0)
    urdf_root = ET.parse(target_urdf).getroot()
    velocity_limits = {joint.get("name"): float(joint.find("limit").get("velocity"))
                       for joint in urdf_root.findall("joint") if joint.get("type") == "revolute"}
    ordered_velocity_limits = np.asarray([velocity_limits[name] for name in scene.joint_names])
    velocity_limits_valid = (np.abs(joint_velocities) <= ordered_velocity_limits[None] + 1e-6).all(axis=1)
    base_linear_velocity = np.gradient(canonical[:, 4:7], times, axis=0)
    rotations = Rotation.from_quat(canonical[:, [1, 2, 3, 0]])
    angular_midpoints = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / np.diff(times)[:, None]
    angular_velocity = np.vstack((angular_midpoints[0],
                                   (angular_midpoints[:-1] + angular_midpoints[1:]) / 2,
                                   angular_midpoints[-1]))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "motion.npz", timestamps=times, fps=output_fps,
        base_position_w=canonical[:, 4:7], base_quaternion_wxyz=canonical[:, :4],
        base_linear_velocity_w=base_linear_velocity, base_angular_velocity_w=angular_velocity,
        joint_positions=canonical[:, 7:], joint_velocities=joint_velocities,
        joint_names=np.asarray(scene.joint_names), body_names=np.asarray(BODY_NAMES),
        body_landmarks_w=landmarks, contact_surface_positions_w=witnesses,
        geometric_signed_distances=distances, minimum_environment_distances=minimum_distances,
        geometry_valid=geometry_valid, joint_limits_valid=joint_limits_valid,
        kinematic_constraints_valid=kinematic_valid, geometry_window_valid=geometry_window_valid,
        velocity_limits_valid=velocity_limits_valid,
        **labels,
    )
    report = {
        "status": f"provisional_{input_method}_geometric_contact_labels",
        "input_method": input_method,
        "source_qpos": str(input_path.resolve()), "source_qpos_sha256": sha256(input_path),
        "target_urdf": str(target_urdf.resolve()), "target_urdf_sha256": sha256(target_urdf),
        "target_xml": str(target_urdf.with_suffix(".xml").resolve()),
        "target_xml_sha256": sha256(target_urdf.with_suffix(".xml")),
        "terrain_urdf": str(terrain_urdf.resolve()), "terrain_urdf_sha256": sha256(terrain_urdf),
        "source_clip": str(Path(source_clip).resolve()) if source_clip else None,
        "source_clip_sha256": sha256(source_clip) if source_clip else None,
        "source_fps": source_fps, "output_fps": output_fps, "source_frames": len(qpos),
        "output_frames": len(times), "joint_names": scene.joint_names,
        "qpos_input_order": "xyz,wxyz,joints_in_target_MJCF_order",
        "qpos_output_order": "wxyz,xyz,joints_in_target_URDF_order",
        "contact_body_names": list(BODY_NAMES),
        "geometry_valid_fraction": float(geometry_valid.mean()),
        "joint_limits_valid_fraction": float(joint_limits_valid.mean()),
        "kinematic_valid_fraction": float(kinematic_valid.mean()),
        "invalid_geometry_frames": np.flatnonzero(~geometry_valid).tolist(),
        "minimum_signed_distance_m": float(np.min(minimum_distances)),
        "contact_steps_by_body": dict(zip(BODY_NAMES, labels["contact_flags"].sum(axis=0).tolist())),
        "future_targets": int(labels["target_occupancy"].sum()),
        "geometry_window_valid_fraction": float(geometry_window_valid.mean()),
        "max_joint_velocity_rad_s": float(np.abs(joint_velocities).max()),
        "velocity_limits_valid_fraction": float(velocity_limits_valid.mean()),
        "max_joint_velocity_limit_ratio": float((np.abs(joint_velocities) / ordered_velocity_limits[None]).max()),
        "landmark_frames": scene.landmark_frame_links,
        "landmark_offsets": scene.offsets.tolist(),
        "settings": {"contact_distance_m": contact_distance,
                     "penetration_tolerance_m": penetration_tolerance,
                     "minimum_contact_steps": minimum_contact_steps,
                     "horizon_steps": horizon_steps},
        "versions": {package: importlib.metadata.version(package)
                     for package in ("numpy", "scipy", "drake", "trimesh")},
        "limitations": (["Holosoma does not expose per-frame solver convergence status"]
                        if input_method == "holosoma" else
                        ["Joint-subset baseline preserves retained joints; wrist contact is not optimized"]
                        if input_method == "asap_joint_subset" else
                        ["Local shoulder/elbow repair preserves root, waist and legs; consult arm_repair_report for geometry failures"]) + [
                        "contacts come from static collision geometry, not measured contact forces",
                        "Drake convex hull distance queries may differ from MuJoCo collision contacts",
                        "joint velocity is a finite-difference diagnostic; no dynamics or actuator validation",
                        "self-collision has not been validated"],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Holosoma retargeting result NPZ")
    parser.add_argument("--target-urdf", type=Path, required=True)
    parser.add_argument("--terrain-urdf", type=Path, required=True)
    parser.add_argument("--source-clip", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--input-method", choices=("holosoma", "asap_joint_subset", "asap_local_arm_repair"), default="holosoma")
    args = parser.parse_args()
    label_holosoma_clip(args.input, args.target_urdf, args.terrain_urdf, args.output,
                        source_clip=args.source_clip, output_fps=args.fps, input_method=args.input_method)


if __name__ == "__main__":
    main()
