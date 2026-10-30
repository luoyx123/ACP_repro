"""Cross-check normalized ACP reference states against their native MuJoCo scene."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from holosoma_retargeting.config_types.data_type import MOCAP_DEMO_JOINTS
from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity


def check(scene_path, motion_path, output, retargeted_path=None):
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    with np.load(motion_path) as motion:
        qpos = np.c_[motion["base_position_w"], motion["base_quaternion_wxyz"], motion["joint_positions"]]
        drake_minimum = motion["minimum_environment_distances"].copy()
        drake_flags = motion["contact_flags"].copy()
        drake_raw_flags = ((motion["geometric_signed_distances"] <= 0.015)
                          & (motion["geometric_signed_distances"] >= -0.003)
                          & motion["kinematic_constraints_valid"][:, None])
        body_names = motion["body_names"].tolist()
        joint_names = motion["joint_names"].tolist()
    actual_names = [model.joint(i).name for i in range(model.njnt)
                    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if actual_names != joint_names or model.nq != qpos.shape[1]:
        raise ValueError("Scene joint ordering does not match the labeled motion")
    hands = ([f"{side}_sphere_hand_link" for side in ("left", "right")]
             if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_sphere_hand_link") >= 0
             else [f"{side}_wrist_roll_rubber_hand" for side in ("left", "right")])
    physical = ["left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link", *hands,
                "left_elbow_link", "right_elbow_link", "pelvis"]
    body_groups = {model.body(name).id: index for index, name in enumerate(physical)}
    # URDF models retain sole spheres and other fixed collision descendants.
    for body_id in range(1, model.nbody):
        parent = int(model.body_parentid[body_id])
        if body_id not in body_groups and not model.body_jntnum[body_id] and parent in body_groups:
            body_groups[body_id] = body_groups[parent]
    robots, terrain = [], []
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body_id = int(model.geom_bodyid[geom])
        name = model.geom(geom).name or ""
        if body_id == 0 or name.startswith("multi_boxes_"):
            terrain.append(geom)
        else:
            robots.append(geom)
    minimum = np.full(len(qpos), 0.1)
    distances = np.full((len(qpos), 9), 0.1)
    for index, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for robot in robots:
            group = body_groups.get(int(model.geom_bodyid[robot]))
            for env in terrain:
                distance = mujoco.mj_geomDistance(model, data, robot, env, 0.1, None)
                minimum[index] = min(minimum[index], distance)
                if group is not None:
                    distances[index, group] = min(distances[index, group], distance)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    flags = distances <= 0.015
    stance_metrics = {}
    if retargeted_path is not None:
        with np.load(retargeted_path) as source:
            if "source_stance_flags" in source:
                supplied_stance = source["source_stance_flags"].copy()
                if supplied_stance.shape != (len(qpos), 2):
                    raise ValueError("Source stance flags must match labeled frame count")
                sequence = [{"L_Toe": left, "R_Toe": right} for left, right in supplied_stance]
            else:
                human, source_fps = source["human_joints"].copy(), float(source["fps"])
                if len(human) != len(qpos):
                    raise ValueError("Stance checks require IK output and labels at the same frame rate")
                sequence = extract_foot_sticking_sequence_velocity(
                    human, MOCAP_DEMO_JOINTS, ["LeftToeBase", "RightToeBase"],
                    velocity_threshold=0.01 * 30.0 / source_fps,
                )
        for side, key, index in (("left", "L_Toe", 0), ("right", "R_Toe", 1)):
            stance = np.asarray([entry[key] for entry in sequence], dtype=bool)
            stance_metrics[side] = {
                "stance_steps": int(stance.sum()),
                "stance_near_surface_fraction": float((distances[stance, index] <= 0.015).mean()) if stance.any() else None,
                "stance_gap_p95_m": float(np.percentile(distances[stance, index], 95)) if stance.any() else None,
            }
    summary = {"scene": str(Path(scene_path).resolve()), "mujoco_version": mujoco.__version__,
               "frames": len(qpos), "robot_collision_geoms": len(robots), "terrain_collision_geoms": len(terrain),
               "native_minimum_distance_m": float(minimum.min()),
               "native_invalid_geometry_frames": np.flatnonzero(minimum < -0.003).tolist(),
               "drake_invalid_geometry_frames": np.flatnonzero(drake_minimum < -0.003).tolist(),
               "native_near_contact_steps_unfiltered": dict(zip(body_names, flags.sum(axis=0).tolist())),
               "native_vs_drake_raw_contact_flag_agreement": float((flags == drake_raw_flags).mean()),
               "native_raw_vs_drake_filtered_flag_agreement": float((flags == drake_flags).mean()),
               "source_stance_diagnostics": stance_metrics,
               "note": "Raw flags are distance-only before segment filtering; this is not force validation."}
    np.savez_compressed(output / "mujoco_geometry.npz", distances=distances,
                        minimum_distances=minimum, near_contact_flags=flags)
    (output / "mujoco_geometry_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retargeted", type=Path)
    args = parser.parse_args()
    check(args.scene, args.motion, args.output, args.retargeted)
