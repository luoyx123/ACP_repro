"""Paired G1-29 / wrist-fixed-23 / previous IK diagnostics on the same clock."""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from acp.data.assets import read_record, resolve_path, run_path

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from acp.robot.joints import project_joint_subset
from acp.data.motions import resample_drake
from acp.data.playback import resolve_clip
from acp.robot.model import compile_robot, digest, make_scene, save_robot_xml


def contact_summary(model, data):
    environment = {i for i in range(model.ngeom) if model.geom_bodyid[i] == 0
                   or (model.geom(i).name or "").startswith("multi_boxes_")}
    env_min, self_min = 0., 0.
    pairs, env_pairs = set(), set()
    for contact in data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if a in environment and b in environment:
            continue
        if a in environment or b in environment:
            env_min = min(env_min, float(contact.dist))
            if contact.dist < -.003:
                env_pairs.add(" / ".join(sorted(model.body(int(model.geom_bodyid[g])).name or "world" for g in (a, b))))
        else:
            self_min = min(self_min, float(contact.dist))
            if contact.dist < -.003:
                pairs.add(" / ".join(sorted(model.body(int(model.geom_bodyid[g])).name for g in (a, b))))
    hands = np.full(2, .1)
    for index, side in enumerate(("left", "right")):
        body = model.body(f"{side}_sphere_hand_link").id
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
        for geom in geoms:
            for env in environment:
                if model.geom_contype[env] or model.geom_conaffinity[env]:
                    hands[index] = min(hands[index], mujoco.mj_geomDistance(model, data, geom, env, .1, None))
    return env_min, self_min, pairs, hands, env_pairs


def compare(row, run, old_run):
    run = Path(run)
    clip = run / "clips" / row["clip"]
    manifest = read_record(clip / "manifest.json")
    source_urdf = (Path(row["reference_urdf"]) if "reference_urdf" in row else
                   Path(row["source"]).parent.parent / "models/g1/g1_29dof_spherehand.urdf")
    signature = {"script_sha256": digest(__file__), "motion_sha256": digest(clip / "labeled/motion.npz"),
                 "scene_sha256": digest(manifest["scene"]),
                 "source_sha256": digest(row["source"]), "source_urdf_sha256": digest(source_urdf)}
    reference_motion = Path(row.get("reference_motion_npz", row["source"]))
    signature["reference_motion_sha256"] = digest(reference_motion)
    old_info = resolve_clip(old_run, row["clip"]) if old_run else None
    if old_info:
        signature["old_motion_sha256"] = digest(old_info["motion"])
        signature["old_scene_sha256"] = digest(old_info["scene"])
    report_path = clip / "comparison/report.json"
    if report_path.is_file():
        previous = read_record(report_path)
        if previous.get("signature") == signature:
            return previous
    with np.load(reference_motion, allow_pickle=False) as z:
        source_raw, source_fps = z["qpos"].copy(), float(z["fps"])
    with np.load(clip / "labeled/motion.npz") as z:
        fps, times = float(z["fps"]), z["timestamps"].copy()
        q = np.c_[z["base_quaternion_wxyz"], z["base_position_w"], z["joint_positions"]]
        names = z["joint_names"].tolist()
    source_times, source_q, _ = resample_drake(source_raw, source_fps, fps)
    np.testing.assert_array_equal(times, source_times)
    source_model = compile_robot(source_urdf)
    source_names = [source_model.joint(i).name for i in range(1, source_model.njnt)]
    expected = project_joint_subset(source_q, source_names, names)
    np.testing.assert_array_equal(q, expected)
    with np.load(clip / "source_rate_subset.npz") as raw_export:
        np.testing.assert_array_equal(raw_export["qpos"], project_joint_subset(source_raw, source_names, names))
    source_xml = clip / "comparison/source_robot.xml"
    source_xml.parent.mkdir(exist_ok=True)
    save_robot_xml(source_model, source_xml)
    if "terrain_urdf" in row:
        terrain = Path(row["terrain_urdf"])
    else:
        base, scale = row["clip"].split("_z_scale_")
        terrain = source_urdf.parent.parent / "terrain" / base / f"multi_boxes_z_scale_{scale}.urdf"
    source_model = make_scene(source_xml, terrain, clip / "comparison/source_scene.xml")
    target_model = mujoco.MjModel.from_xml_path(manifest["scene"])
    source_data, target_data = mujoco.MjData(source_model), mujoco.MjData(target_model)
    if old_info:
        old_model = mujoco.MjModel.from_xml_path(str(old_info["scene"]))
        old_data = mujoco.MjData(old_model)
        with np.load(old_info["motion"]) as z:
            old_times = z["timestamps"].copy()
            old_q = np.c_[z["base_position_w"], z["base_quaternion_wxyz"], z["joint_positions"]]
        np.testing.assert_allclose(times, old_times, atol=1e-12)
    protected = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
                 "left_elbow_link", "right_elbow_link"]
    source_ids = [source_model.body(name).id for name in protected]
    target_ids = [target_model.body(name).id for name in protected]
    hand_ids = [[model.body(side + "_sphere_hand_link").id for side in ("left", "right")]
                for model in (source_model, target_model)]
    values = {key: [] for key in ["source_env", "source_self", "target_env", "target_self", "source_hand_distances",
                                  "target_hand_distances", "protected_position_errors", "protected_rotation_errors",
                                  "hand_position_errors", "old_foot_errors", "old_root_errors", "old_root_rotation_errors"]}
    pairs_src, pairs_dst, introduced_self_pairs, introduced_env_pairs = Counter(), Counter(), Counter(), Counter()
    introduced_self_frames = 0
    for frame in range(len(q)):
        source_data.qpos[:] = np.r_[source_q[frame, 4:7], source_q[frame, :4], source_q[frame, 7:]]
        target_data.qpos[:] = np.r_[q[frame, 4:7], q[frame, :4], q[frame, 7:]]
        mujoco.mj_forward(source_model, source_data)
        mujoco.mj_forward(target_model, target_data)
        frame_pairs = {}
        for prefix, model, data, counter in [("source", source_model, source_data, pairs_src),
                                             ("target", target_model, target_data, pairs_dst)]:
            env, self_distance, pairs, hands, env_pairs = contact_summary(model, data)
            values[prefix + "_env"].append(env)
            values[prefix + "_self"].append(self_distance)
            values[prefix + "_hand_distances"].append(hands)
            counter.update(pairs)
            frame_pairs[prefix] = (pairs, env_pairs)
        added_self = frame_pairs["target"][0] - frame_pairs["source"][0]
        introduced_self_frames += bool(added_self)
        introduced_self_pairs.update(added_self)
        introduced_env_pairs.update(frame_pairs["target"][1] - frame_pairs["source"][1])
        values["protected_position_errors"].append(np.linalg.norm(
            source_data.xpos[source_ids] - target_data.xpos[target_ids], axis=1))
        values["protected_rotation_errors"].append(np.max(np.abs(
            source_data.xmat[source_ids] - target_data.xmat[target_ids]), axis=1))
        values["hand_position_errors"].append(np.linalg.norm(
            source_data.xpos[hand_ids[0]] - target_data.xpos[hand_ids[1]], axis=1))
        if old_info:
            old_data.qpos[:] = old_q[frame]
            mujoco.mj_forward(old_model, old_data)
            values["old_root_errors"].append(np.linalg.norm(old_q[frame, :3] - source_q[frame, 4:7]))
            values["old_root_rotation_errors"].append((Rotation.from_quat(old_q[frame, [4, 5, 6, 3]]) *
                Rotation.from_quat(source_q[frame, [1, 2, 3, 0]]).inv()).magnitude())
            values["old_foot_errors"].append([np.linalg.norm(old_data.xpos[old_model.body(side + "_ankle_roll_link").id]
                          - source_data.xpos[source_model.body(side + "_ankle_roll_link").id]) for side in ("left", "right")])
    arrays = {key: np.asarray(value) for key, value in values.items()}
    source_bad, target_bad = arrays["source_env"] < -.003, arrays["target_env"] < -.003
    source_hands = (arrays["source_hand_distances"] <= .015) & (arrays["source_hand_distances"] >= -.003)
    target_hands = (arrays["target_hand_distances"] <= .015) & (arrays["target_hand_distances"] >= -.003)
    result = {"clip": row["clip"], "frames": len(q), "signature": signature,
              "source_rate_subset_exact": True, "resampled_subset_exact": True,
              "protected_links": protected,
              "maximum_protected_position_error_m": float(arrays["protected_position_errors"].max()),
              "maximum_protected_rotation_matrix_error": float(arrays["protected_rotation_errors"].max()),
              "hand_position_error_mean_m": float(arrays["hand_position_errors"].mean()),
              "hand_position_error_p95_m": float(np.percentile(arrays["hand_position_errors"], 95)),
              "hand_position_error_max_m": float(arrays["hand_position_errors"].max()),
              "source_native_environment_invalid_frames": int(source_bad.sum()),
              "target_native_environment_invalid_frames": int(target_bad.sum()),
              "introduced_environment_invalid_frames": int((target_bad & ~source_bad).sum()),
              "inherited_environment_invalid_frames": int((target_bad & source_bad).sum()),
              "source_native_minimum_environment_m": float(arrays["source_env"].min()),
              "target_native_minimum_environment_m": float(arrays["target_env"].min()),
              "source_native_self_invalid_frames": int((arrays["source_self"] < -.003).sum()),
              "target_native_self_invalid_frames": int((arrays["target_self"] < -.003).sum()),
              "source_self_pairs": dict(pairs_src.most_common()), "target_self_pairs": dict(pairs_dst.most_common()),
              "introduced_native_self_pair_frames": introduced_self_frames,
              "introduced_native_self_pairs": dict(introduced_self_pairs.most_common()),
              "introduced_native_environment_pairs": dict(introduced_env_pairs.most_common()),
              "source_hand_near_surface_steps": int(source_hands.sum()),
              "retained_hand_near_surface_steps": int((source_hands & target_hands).sum()),
              "source_hand_contact_retention": float(target_hands[source_hands].mean()) if source_hands.any() else None,
              "reference_urdf": str(source_urdf), "reference_motion": str(reference_motion),
              "note": "Paired native geometry diagnostics, not dataset validity or physical contact truth. Fixed wrist welds can change native self-collision filtering."}
    if old_info:
        result.update(old_foot_position_error_mean_m=float(arrays["old_foot_errors"].mean()),
                      old_foot_position_error_max_m=float(arrays["old_foot_errors"].max()),
                      old_root_position_error_mean_m=float(arrays["old_root_errors"].mean()),
                      old_root_position_error_max_m=float(arrays["old_root_errors"].max()),
                      old_root_rotation_error_max_rad=float(arrays["old_root_rotation_errors"].max()))
    review = []
    if introduced_env_pairs:
        review.append("introduced_native_environment_pairs")
    if introduced_self_pairs:
        review.append("introduced_native_self_pairs")
    if result["retained_hand_near_surface_steps"] < result["source_hand_near_surface_steps"]:
        review.append("source_hand_near_surface_steps_lost")
    result["conversion_review_reasons"] = review
    result["conversion_review"] = "wrist_contact_review" if review else "no_added_issue_in_paired_checks"
    result["stage1_training_admitted"] = False
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(report_path.with_name("diagnostics.npz"), **arrays)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=run_path("omni_asap23"))
    parser.add_argument("--old-run", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clips", nargs="+")
    args = parser.parse_args()
    run = resolve_path(args.run).resolve()
    rows = read_record(run / "inventory.json")
    selected = [row for row in rows if args.clips is None or row["clip"] in args.clips]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(compare, row, str(run), str(resolve_path(args.old_run)) if args.old_run else None) for row in selected]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(selected)} {result['clip']}", flush=True)
    results.sort(key=lambda row: row["clip"])
    frames = sum(row["frames"] for row in results)
    source_contact_steps = sum(row["source_hand_near_surface_steps"] for row in results)
    summary = {"clips": len(results), "expected_clips": len(rows), "frames": frames,
               "all_raw_and_resampled_subsets_exact": all(row["source_rate_subset_exact"] and row["resampled_subset_exact"] for row in results),
               "maximum_protected_position_error_m": max(row["maximum_protected_position_error_m"] for row in results),
               "maximum_protected_rotation_matrix_error": max(row["maximum_protected_rotation_matrix_error"] for row in results),
               "source_native_environment_invalid_frames": sum(row["source_native_environment_invalid_frames"] for row in results),
               "target_native_environment_invalid_frames": sum(row["target_native_environment_invalid_frames"] for row in results),
               "introduced_environment_invalid_frames": sum(row["introduced_environment_invalid_frames"] for row in results),
               "inherited_environment_invalid_frames": sum(row["inherited_environment_invalid_frames"] for row in results),
               "source_native_self_invalid_frames": sum(row["source_native_self_invalid_frames"] for row in results),
               "target_native_self_invalid_frames": sum(row["target_native_self_invalid_frames"] for row in results),
               "introduced_native_self_pair_frames": sum(row["introduced_native_self_pair_frames"] for row in results),
               "introduced_native_self_pairs": dict(sum((Counter(row["introduced_native_self_pairs"]) for row in results), Counter()).most_common()),
               "introduced_native_environment_pairs": dict(sum((Counter(row["introduced_native_environment_pairs"]) for row in results), Counter()).most_common()),
               "source_hand_near_surface_steps": source_contact_steps,
               "source_hand_contact_retention": sum(row["retained_hand_near_surface_steps"] for row in results) / source_contact_steps if source_contact_steps else None,
               "hand_position_error_mean_m": sum(row["hand_position_error_mean_m"] * row["frames"] for row in results) / frames,
               "hand_position_error_max_m": max(row["hand_position_error_max_m"] for row in results),
               "old_foot_position_error_mean_m": sum(row["old_foot_position_error_mean_m"] * row["frames"] for row in results) / frames,
               "old_foot_position_error_max_m": max(row["old_foot_position_error_max_m"] for row in results),
               "old_root_position_error_max_m": max(row["old_root_position_error_max_m"] for row in results),
               "conversion_review_counts": dict(Counter(row["conversion_review"] for row in results)),
               "note": "Source native collision counts describe the chosen geometry; they do not invalidate the dataset. Protected links include feet, torso, pelvis and elbows. Wrist-fixed hands may need local contact correction."}
    suffix = "" if len(selected) == len(rows) else "_partial"
    (run / f"paired_comparison{suffix}.json").write_text(json.dumps(results, indent=2) + "\n")
    (run / f"paired_summary{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    queue = [{"clip": row["clip"], "reasons": row["conversion_review_reasons"],
              "introduced_environment_frames": row["introduced_environment_invalid_frames"],
              "lost_hand_near_surface_steps": row["source_hand_near_surface_steps"] - row["retained_hand_near_surface_steps"],
              "diagnostics": f"clips/{row['clip']}/comparison/diagnostics.npz"}
             for row in results if row["conversion_review_reasons"]]
    queue.sort(key=lambda row: (row["introduced_environment_frames"], row["lost_hand_near_surface_steps"]), reverse=True)
    (run / f"wrist_contact_repair_queue{suffix}.json").write_text(json.dumps(queue, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
