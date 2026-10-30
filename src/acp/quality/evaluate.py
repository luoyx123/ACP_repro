"""Evaluate all 145 robot-terrain inputs, including failed or missing outputs.

Run with the Holosoma Python environment. Native self-collision checks use
MuJoCo collision filtering, including its adjacency exclusions.
"""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from acp.data.assets import read_record, resolve_path, run_path

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))

import mujoco
import numpy as np


def digest(path):
    return __import__('acp.management.fingerprints', fromlist=['digest']).digest(path)


def scene_from_manifest(manifest):
    if "scene" in manifest:
        return Path(manifest["scene"])
    for command in manifest.get("commands", []):
        argv = command["argv"]
        if "--scene" in argv:
            return Path(argv[argv.index("--scene") + 1])
    raise ValueError("No native scene recorded in manifest")


def assess(row, root, criteria):
    clip = Path(root) / "clips" / row["clip"]
    manifest_path = clip / "manifest.json"
    result = {"clip": row["clip"], "scene_id": row.get("motion_group", row["clip"].split("_z_scale_")[0]),
              "z_scale": float(row["clip"].split("_z_scale_")[1]) if "_z_scale_" in row["clip"] else None, "route": row["route"],
              "source_frames": row["source_frames"], "source_fps": row["source_fps"]}
    if not manifest_path.exists():
        return {**result, "status": "not_processed", "reasons": ["no_output"]}
    manifest = read_record(manifest_path)
    if manifest.get('status', '').startswith('excluded_'):
        return {**result, 'status': 'excluded_input', 'reasons': [manifest['status']]}
    if manifest.get("status") == "failed":
        result.update(primary_route=row["route"], primary_attempt_status="failed", fallback_attempts={})
        for attempt in ("robot_fk_relaxed", "robot_fk"):
            fallback = clip / "attempts" / attempt / "manifest.json"
            if not fallback.exists():
                continue
            alternative = read_record(fallback)
            result["fallback_attempts"][attempt] = alternative.get("status")
            if alternative.get("status", "").startswith("end_to_end"):
                clip = fallback.parent
                manifest = alternative
                result["route"] = "robot29_to_robot23_" + attempt
                break
    if not manifest.get("status", "").startswith("end_to_end"):
        if "running" in result.get("fallback_attempts", {}).values():
            return {**result, "status": "running", "reasons": ["fallback_in_progress"]}
        return {**result, "status": "processing_failed" if manifest.get("status") == "failed" else "running",
                "reasons": [manifest.get("failed_stage", manifest.get("status", "unknown"))]}
    motion_path = clip / "labeled/motion.npz"
    label_path = clip / "labeled/report.json"
    native_path = clip / "validation/mujoco_geometry_report.json"
    if not all(p.exists() for p in (motion_path, label_path, native_path)):
        return {**result, "status": "processing_failed", "reasons": ["missing_stage_artifact"]}
    retargeted = clip / "retargeted" / f"{row['clip']}_original.npz"
    collision_policy_path=Path(__file__).resolve().parents[3]/'configs/g1_spherehand_collision_exemptions.json'
    collision_policy=read_record(collision_policy_path) if collision_policy_path.is_file() else {}
    use_collision_policy=(manifest.get('source_format')=='mixamo_fbx_world_bones'
                          and digest(manifest['target_urdf'])==collision_policy.get('target_urdf_sha256'))
    signature = {"motion_sha256": digest(motion_path), "evaluator_sha256": digest(__file__),
                 "manifest_sha256": digest(clip / "manifest.json"),
                 "native_geometry_sha256": digest(clip / "validation/mujoco_geometry.npz"),
                 "source_keypoints_sha256": digest(retargeted),
                 "criteria": criteria, "mujoco_version": mujoco.__version__}
    if use_collision_policy:signature['collision_policy_sha256']=digest(collision_policy_path)
    cache_path = clip / "quality/report.json"
    if cache_path.exists():
        previous = read_record(cache_path)
        if previous.get("signature") == signature:
            return previous
    from holosoma_retargeting.config_types.data_type import MotionDataConfig, MOCAP_DEMO_JOINTS

    report = read_record(label_path)
    native = read_record(native_path)
    scene_path = scene_from_manifest(manifest)
    result.update(artifact_directory=str(clip.resolve()),
                  labeled_motion=str(motion_path.resolve()),
                  labeled_motion_relative=str(motion_path.relative_to(Path(root))),
                  native_scene=str(scene_path.resolve()))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mapping = manifest.get("target_mapping") or MotionDataConfig(
        data_format="mocap", robot_type="g1_23dof_rev_1_0").resolved_joints_mapping
    body_ids = [model.body(name).id for name in mapping.values()]
    mapped_indices = [MOCAP_DEMO_JOINTS.index(key) for key in mapping]
    with np.load(motion_path, allow_pickle=False) as motion:
        times = motion["timestamps"].copy()
        qpos = np.c_[motion["base_position_w"], motion["base_quaternion_wxyz"], motion["joint_positions"]]
        flags = motion["contact_flags"].copy()
        body_names = motion['body_names'].tolist()
        feet = motion["body_landmarks_w"][:, :2].copy()
        fps = float(motion["fps"])
        position_invalid = int((~motion["joint_limits_valid"]).sum())
        velocity_invalid = int((~motion["velocity_limits_valid"]).sum())
        drake_invalid = int((~motion["geometry_valid"]).sum())
        quaternion_norm_error = float(np.max(np.abs(np.linalg.norm(motion["base_quaternion_wxyz"], axis=1) - 1)))
        velocity_ratio = float(report["max_joint_velocity_limit_ratio"])
        kinematic_valid = motion["kinematic_constraints_valid"].copy()
        drake_raw = ((motion["geometric_signed_distances"] <= 0.015)
                     & (motion["geometric_signed_distances"] >= -0.003)
                     & kinematic_valid[:, None])
    with np.load(clip / "validation/mujoco_geometry.npz") as native_arrays:
        native_distances = native_arrays["distances"]
        native_raw = ((native_distances <= 0.015) & (native_distances >= -0.003)
                      & kinematic_valid[:, None])
        # Compare the same definition and masks, including invalid frames.
        raw_agreement = float((native_raw == drake_raw).mean())
    with np.load(retargeted, allow_pickle=False) as source:
        points_key = "robot_fk_keypoints" if "robot_fk_keypoints" in source else "human_joints"
        desired = source[points_key][:, mapped_indices].copy()
        source_stance = source['source_stance_flags'].copy() if 'source_stance_flags' in source else None
        source_support = {name:mask for name,mask in zip(source['surface_contact_body_names'].tolist(),source['surface_contact_masks'].T)} if 'surface_contact_masks' in source else {}
    if desired.shape != (len(qpos), len(mapping), 3):
        raise ValueError(f"Landmark timestamps/shape do not match in {row['clip']}")
    errors = np.zeros((len(qpos), len(mapping)))
    self_min = np.zeros(len(qpos))
    nonlocal_self_min = np.zeros(len(qpos))
    self_pairs = Counter()
    unexpected_self_min = np.zeros(len(qpos))
    unexpected_self_pairs = Counter()
    structural_contact_count=0
    nonlocal_self_pairs = Counter()
    toe_positions = np.zeros((len(qpos), 2, 3))
    use_toe_stance = manifest.get('source_format') == 'mixamo_fbx_world_bones' and source_stance is not None
    toe_ids = [model.body(mapping[s+'ToeBase']).id for s in ('Left','Right')] if use_toe_stance else None
    neighbour_cache = {}

    def joint_distance(a, b):
        key = tuple(sorted((a, b)))
        if key not in neighbour_cache:
            ancestors = {}
            count, current = 0, a
            while True:
                ancestors[current] = count
                if current == 0:
                    break
                count += int(model.body_jntnum[current] > 0)
                current = int(model.body_parentid[current])
            count, current = 0, b
            while current not in ancestors:
                count += int(model.body_jntnum[current] > 0)
                current = int(model.body_parentid[current])
            neighbour_cache[key] = count + ancestors[current]
        return neighbour_cache[key]
    robot_geoms = set()
    for geom_id in range(model.ngeom):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        if model.geom_bodyid[geom_id] != 0 and not (model.geom(geom_id).name or "").startswith("multi_boxes_"):
            robot_geoms.add(geom_id)
    for index, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        errors[index] = np.linalg.norm(data.xpos[body_ids] - desired[index], axis=1)
        if use_toe_stance:
            toe_positions[index] = data.xpos[toe_ids]
        pairs_this_frame = set()
        nonlocal_pairs_this_frame = set()
        unexpected_pairs_this_frame = set()
        anchors={rule['joint']:data.xanchor[model.joint(rule['joint']).id].copy()
                 for rule in collision_policy.get('rules',[])} if use_collision_policy else {}
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if a in robot_geoms and b in robot_geoms:
                self_min[index] = min(self_min[index], float(contact.dist))
                if contact.dist < -criteria["self_penetration_tolerance_m"]:
                    body_a, body_b = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                    names = sorted((model.body(body_a).name, model.body(body_b).name))
                    pair_name = " / ".join(names)
                    pairs_this_frame.add(pair_name)
                    from acp.robot.collision import joint_housing_overlap
                    allowed=use_collision_policy and joint_housing_overlap(names[0],names[1],contact.pos,
                              float(contact.dist),anchors,collision_policy['rules'])
                    if allowed:structural_contact_count+=1
                    else:
                        unexpected_self_min[index]=min(unexpected_self_min[index],float(contact.dist))
                        unexpected_pairs_this_frame.add(pair_name)
                    # Supplementary diagnostic only. Keep the conservative raw
                    # gate, but identify distant-body collisions separately from
                    # possible joint-neighbour collision-asset overlaps.
                    if joint_distance(body_a, body_b) > 2:
                        nonlocal_self_min[index] = min(nonlocal_self_min[index], float(contact.dist))
                        nonlocal_pairs_this_frame.add(pair_name)
        self_pairs.update(pairs_this_frame)
        nonlocal_self_pairs.update(nonlocal_pairs_this_frame)
        unexpected_self_pairs.update(unexpected_pairs_this_frame)
    self_invalid = np.flatnonzero(self_min < -criteria["self_penetration_tolerance_m"])
    planar_speed = np.linalg.norm(np.gradient(feet[:, :, :2], times, axis=0), axis=-1)
    toe_speed = np.linalg.norm(np.gradient(toe_positions[:, :, :2], times, axis=0), axis=-1) if use_toe_stance else None
    foot_metrics = {}
    for index, side in enumerate(("left", "right")):
        # Exclude contact transitions: a sole frame can move during heel strike
        # or toe off. This remains a sole-landmark diagnostic, not contact velocity.
        interior = flags[:, index].copy()
        if use_toe_stance:
            interior &= source_stance[:, index]
        selected = interior.copy()
        interior[1:] &= selected[:-1]
        interior[:-1] &= selected[1:]
        interior[0] = interior[-1] = False
        speeds = (toe_speed if use_toe_stance else planar_speed)[interior, index]
        stance = native["source_stance_diagnostics"].get(side, {})
        foot_metrics[side] = {**stance, "contact_interior_steps": int(interior.sum()),
            'speed_landmark':mapping[side.title()+'ToeBase'] if use_toe_stance else 'semantic_sole',
            'speed_mask':'source stance and target contact, interior samples' if use_toe_stance else 'target contact interior samples',
            "contact_planar_speed_mean_m_s": float(speeds.mean()) if len(speeds) else None,
            "contact_planar_speed_p95_m_s": float(np.percentile(speeds, 95)) if len(speeds) else None}
    expected = int(np.floor((row["source_frames"] - 1) / row["source_fps"] * fps + 1e-9)) + 1
    reasons = []
    if len(qpos) != expected or len(times) != expected:
        reasons.append("incomplete_output")
    if not np.allclose(np.diff(times), 1 / fps, atol=1e-9):
        reasons.append("nonuniform_time_grid")
    if quaternion_norm_error > 1e-5:
        reasons.append("invalid_quaternion")
    if drake_invalid > criteria["maximum_invalid_environment_frames"] or native["native_invalid_geometry_frames"]:
        reasons.append("environment_penetration")
    if position_invalid > criteria["maximum_invalid_joint_position_frames"]:
        reasons.append("joint_position_limits")
    if velocity_invalid > criteria["maximum_invalid_joint_velocity_frames"]:
        reasons.append("joint_velocity_limits")
    unexpected_self_invalid=np.flatnonzero(unexpected_self_min < -criteria['self_penetration_tolerance_m'])
    if len(unexpected_self_invalid) > criteria["maximum_self_collision_frames"]:
        reasons.append("self_collision")
    if errors.mean() > criteria["maximum_mapped_keypoint_mean_error_m"] or np.percentile(errors, 95) > criteria["maximum_mapped_keypoint_p95_error_m"]:
        reasons.append("mapped_keypoint_error")
    for side, metrics in foot_metrics.items():
        near = metrics.get("stance_near_surface_fraction")
        slide = metrics.get("contact_planar_speed_p95_m_s")
        # No ground stance is a valid observation for a jump, seated clip or
        # ladder phase. Missing samples cannot establish either a gap or sliding.
        if metrics.get('stance_steps', 0) and (near is None or near < criteria["minimum_stance_near_surface_fraction_per_foot"]):
            reasons.append(f"{side}_stance_gap")
        if slide is not None and slide > criteria["maximum_contact_foot_planar_speed_p95_m_s"]:
            reasons.append(f"{side}_foot_sliding")
    semantic = manifest.get('semantic') or {}
    if semantic.get('support_kind') in ('seat', 'ladder') and not manifest.get('support_scene'):
        reasons.append('missing_'+semantic['support_kind']+'_geometry')
    support = manifest.get('support_scene')
    support_metrics = None
    if support:
        if support.get('review_status') != 'confirmed' or not support.get('approved_for_training', False):
            reasons.append('unconfirmed_support_scene')
        counts = flags.sum(axis=0)
        support_metrics = {'kind':support['support_kind'], 'contact_steps_by_body':dict(zip(body_names, counts.tolist()))}
        if support['support_kind'] == 'seat':
            terminal=flags[-max(5,len(flags)//5):,body_names.index('pelvis')]
            support_metrics['terminal_pelvis_contact_fraction'] = float(terminal.mean())
            if terminal.mean() < .8:reasons.append('seat_pelvis_contact_coverage')
        elif support['support_kind'] == 'ladder':
            from acp.data.adapters.mixamo import stable_segments
            expectations={}
            coverage={}
            for body in ('left_hand','right_hand','left_foot','right_foot'):
                if body.endswith('_foot'):
                    side=0 if body.startswith('left') else 1
                    mask=source_stance[:,side] if source_stance is not None else np.zeros(len(qpos),dtype=bool)
                else:
                    mask=source_support.get(body.replace('_hand','_sphere_hand_link'),np.zeros(len(qpos),dtype=bool))
                expected_contact=stable_segments(mask,max(1,int(np.ceil(.2*fps))))
                expectations[body]=int(expected_contact.sum())
                fraction=float(flags[expected_contact,body_names.index(body)].mean()) if expected_contact.any() else None
                coverage[body]=fraction
                if fraction is not None and fraction < .8:reasons.append('ladder_'+body+'_contact_coverage')
            support_metrics['expected_source_support_steps']=expectations
            support_metrics['source_support_contact_coverage']=coverage
    if raw_agreement < criteria["minimum_native_drake_raw_flag_agreement"]:
        reasons.append("native_drake_flags_disagree")
    result.update(status="offline_kinematic_candidate" if not reasons else "needs_repair",
                  reasons=reasons, stage1_offline_eligible=not reasons,
                  stage2_hindsight_eligible=False,
                  frames=len(qpos), fps=fps, expected_frames=expected,
                  support_metrics=support_metrics,
                  duration_s=float(times[-1] - times[0]),
                  geometry_invalid_frames=drake_invalid,
                  joint_position_invalid_frames=position_invalid,
                  joint_velocity_invalid_frames=velocity_invalid,
                  max_joint_velocity_limit_ratio=velocity_ratio,
                  minimum_environment_distance_m=native["native_minimum_distance_m"],
                  self_collision_frames=int(len(self_invalid)),
                  self_collision_fraction=float(len(self_invalid) / len(qpos)),
                  minimum_self_contact_distance_m=float(self_min.min()),
                  self_collision_pairs=dict(self_pairs.most_common()),
                  unexpected_self_collision_frames=int(len(unexpected_self_invalid)),
                  unexpected_self_collision_pairs=dict(unexpected_self_pairs.most_common()),
                  structural_joint_housing_contact_count=structural_contact_count,
                  collision_policy=collision_policy if use_collision_policy else None,
                  nonlocal_self_collision_frames=int(np.count_nonzero(nonlocal_self_min < -criteria["self_penetration_tolerance_m"])),
                  nonlocal_self_collision_pairs=dict(nonlocal_self_pairs.most_common()),
                  nonlocal_self_collision_definition="MuJoCo self contact beyond 3 mm between bodies separated by more than two moving joints; supplementary diagnostic, not a calibrated allowed-collision matrix",
                  mapped_keypoint_mean_error_m=float(errors.mean()),
                  mapped_keypoint_p95_error_m=float(np.percentile(errors, 95)),
                  keypoint_errors_by_name={name: float(errors[:, i].mean()) for i, name in enumerate(mapping)},
                  foot_metrics=foot_metrics,
                  native_drake_raw_flag_agreement=raw_agreement,
                  contact_steps_by_body=report["contact_steps_by_body"],
                  signature=signature,
                  note="Offline Stage-1 reference screening only. Stage-2 hindsight requires successful tracker rollouts and force events separately.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(cache_path.parent / "diagnostics.npz", self_contact_minimum_distances=self_min,
                        unexpected_self_contact_minimum_distances=unexpected_self_min,
                        nonlocal_self_contact_minimum_distances=nonlocal_self_min,
                        self_collision_frames=self_invalid, mapped_keypoint_errors=errors,
                        foot_planar_speeds=planar_speed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, default=Path(__file__).resolve().parents[3] / "configs/data_quality.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run = args.run.resolve()
    criteria = read_record(args.criteria)
    rows = read_record(run / "inventory.json")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(assess, row, str(run), criteria): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"clip": row["clip"], "route": row["route"], "status": "evaluation_failed",
                          "reasons": [f"{type(exc).__name__}: {exc}"]}
            results.append(result)
    results.sort(key=lambda row: row["clip"])
    statuses = Counter(row["status"] for row in results)
    reasons = Counter(reason for row in results for reason in row.get("reasons", []))
    evaluated = [row for row in results if "frames" in row]
    frames = sum(row["frames"] for row in evaluated)
    routes = {}
    for route in sorted({row["route"] for row in results}):
        selected = [row for row in results if row["route"] == route]
        routes[route] = {"inputs": len(selected), "statuses": dict(Counter(row["status"] for row in selected))}
    scope = rows[0].get("scope", "robot-terrain only") if rows else "empty inventory"
    summary = {"scope": scope, "expected_inputs": len(rows), "results": len(results),
               "base_motion_groups": len({row.get("motion_group", row["clip"].split("_z_scale_")[0]) for row in rows}),
               "has_human_joints": sum(row["has_human_joints"] for row in rows),
               "missing_human_joints": sum(not row["has_human_joints"] for row in rows),
               "statuses": dict(statuses), "reason_counts": dict(reasons.most_common()), "routes": routes,
               "evaluated_frames": frames, "evaluated_duration_s": sum(row["duration_s"] for row in evaluated),
               "frame_weighted_environment_invalid_fraction": sum(row["geometry_invalid_frames"] for row in evaluated) / frames if frames else None,
               "frame_weighted_self_collision_fraction": sum(row["self_collision_frames"] for row in evaluated) / frames if frames else None,
               "frame_weighted_unexpected_self_collision_fraction": sum(row.get('unexpected_self_collision_frames',row['self_collision_frames']) for row in evaluated) / frames if frames else None,
               "clips_with_nonlocal_self_collision": sum(row["nonlocal_self_collision_frames"] > 0 for row in evaluated),
               "frame_weighted_nonlocal_self_collision_fraction": sum(row["nonlocal_self_collision_frames"] for row in evaluated) / frames if frames else None,
               "frame_weighted_joint_velocity_invalid_fraction": sum(row["joint_velocity_invalid_frames"] for row in evaluated) / frames if frames else None,
               "stage1_offline_candidates": statuses.get("offline_kinematic_candidate", 0),
               "stage2_hindsight_admitted": 0, "criteria": criteria,
               "complete_inventory_handled": not any(row["status"] in ("not_processed", "running", "evaluation_failed") for row in results),
               "all_processing_successful": all(row["status"] in ("offline_kinematic_candidate", "needs_repair") for row in results),
               "note": "All-input denominator includes failed processing. Acceptance thresholds are engineering choices. Split future train/validation sets by source motion groups, not individual augmentations or windows."}
    (run / "quality_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (run / "quality_all_clips.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    index = [{"clip": row["clip"], "route": row["route"], "status": row["status"],
              "motion": row.get("labeled_motion_relative"),
              "stage1_offline_eligible": row.get("stage1_offline_eligible", False),
              "reasons": row.get("reasons", [])} for row in results]
    (run / "artifact_index.json").write_text(json.dumps(index, indent=2) + "\n")
    for filename, allowed in (("stage1_candidate_clips.txt", {"offline_kinematic_candidate"}),
                              ("needs_repair_clips.txt", {"needs_repair"}),
                              ("processing_failed_clips.txt", {"processing_failed", "evaluation_failed"})):
        (run / filename).write_text("".join(row["clip"] + "\n" for row in results if row["status"] in allowed))
    fields = ("clip", "route", "status", "frames", "self_collision_fraction", "geometry_invalid_frames",
              "joint_velocity_invalid_frames", "mapped_keypoint_mean_error_m", "mapped_keypoint_p95_error_m", "reasons")
    with (run / "quality_all_clips.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "reasons": ";".join(row.get("reasons", []))})
    lines = [f"# {scope} 全量离线评估", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
             "| 动作 | 路径 | 状态 | 不通过项 |", "| --- | --- | --- | --- |"]
    lines += [f"| {row['clip']} | {row['route']} | {row['status']} | {', '.join(row.get('reasons', []))} |" for row in results]
    (run / "QUALITY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
