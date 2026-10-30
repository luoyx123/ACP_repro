"""Convert local LAFAN1 G1-29 CSVs to the same spherehand ASAP-23 references.

Keep source-rate subsets exact, produce optional 50-Hz labels on flat ground,
and separate robot-revision differences from the six-wrist reduction.
Run with the Holosoma Python; labeling uses the ACP Python environment.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src"))
from acp.robot.joints import moving_joint_names, project_joint_subset
from acp.data.motions import LAFAN_G1_JOINT_NAMES, read_lafan_csv, resample_drake
from acp.robot.model import compile_robot, digest, make_scene, prepare, save_robot_xml


def revision_reference(original_model, reference_model, canonical, mapping, demo_joints):
    """Original CSV robot vs unified 29-DoF robot, with identical joint values."""
    od, rd = mujoco.MjData(original_model), mujoco.MjData(reference_model)
    core = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
            "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_elbow_link", "right_elbow_link"]
    oi = [original_model.body(name).id for name in core]
    ri = [reference_model.body(name).id for name in core]
    point_indices = [demo_joints.index(key) for key in mapping]
    bodies = [reference_model.body(name).id for name in mapping.values()]
    points = np.zeros((len(canonical), len(demo_joints), 3))
    errors = np.zeros((len(canonical), len(core)))
    original_env = np.zeros(len(canonical))
    for frame, q in enumerate(canonical):
        pose = np.r_[q[4:7], q[:4], q[7:]]
        od.qpos[:], rd.qpos[:] = pose, pose
        mujoco.mj_forward(original_model, od)
        mujoco.mj_forward(reference_model, rd)
        points[frame, point_indices] = rd.xpos[bodies]
        errors[frame] = np.linalg.norm(od.xpos[oi] - rd.xpos[ri], axis=1)
        for contact in od.contact:
            if original_model.geom_bodyid[contact.geom1] == 0 or original_model.geom_bodyid[contact.geom2] == 0:
                original_env[frame] = min(original_env[frame], float(contact.dist))
    np.testing.assert_allclose(errors[:, 2:4], 0, atol=1e-12)
    return points, errors, original_env, core


def convert(row, run, label_python, fps, force=False):
    from holosoma_retargeting.config_types.data_type import MotionDataConfig, MOCAP_DEMO_JOINTS
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity
    from acp.quality.native import check
    from acp.quality.compare import compare

    run = Path(run)
    artifact = run / "clips" / row["clip"]
    artifact.mkdir(parents=True, exist_ok=True)
    path = artifact / "manifest.json"
    assets = run / "assets"
    urdf = assets / "g1_asap23_spherehand.urdf"
    signature = {"source_sha256": digest(row["source"]), "fps": fps,
                 "target_urdf_sha256": digest(urdf), "target_xml_sha256": digest(urdf.with_suffix(".xml")),
                 "original_urdf_sha256": digest(row["original_urdf"]),
                 "reference_urdf_sha256": digest(row["reference_urdf"]),
                 "source_scene_sha256": digest(assets / "original_scene.xml"),
                 "terrain_sha256": digest(row["terrain_urdf"]),
                 "code": {name: digest(PROJECT / name) for name in
                          ("scripts/batch_lafan_asap23.py", "scripts/prepare_asap23.py",
                           "src/acp/data/motions.py", "src/acp/data/asap.py", "src/acp/data/holosoma.py",
                           "scripts/check_holosoma_mujoco.py", "scripts/compare_asap23.py")}}
    if path.is_file() and not force:
        previous = json.loads(path.read_text())
        required = [artifact / name for name in ("labeled/motion.npz", "validation/mujoco_geometry.npz",
                                                  "comparison/report.json", "model_revision_report.json")]
        if previous.get("signature") == signature and previous.get("status", "").startswith("end_to_end") and all(p.is_file() for p in required):
            return {"clip": row["clip"], "status": "resumed"}
    manifest = {"status": "running", "route": row["route"], "source": row["source"],
                "source_format": "lafan_g1_csv_xyz_xyzw", "signature": signature,
                "source_motion_npz": row["reference_motion_npz"],
                "source_robot_xml": str(assets / "reference_robot.xml"),
                "source_urdf": row["original_urdf"], "reference_urdf": row["reference_urdf"],
                "target_urdf": str(urdf), "scene": str(assets / "scene.xml"),
                "terrain_urdf": row["terrain_urdf"], "stage1_training_admitted": False,
                "reference_semantics": "CSV joint values on unified spherehand29; original model differences reported separately"}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        original = read_lafan_csv(row["source"])
        times, sampled, _ = resample_drake(original, 30., fps)
        model = mujoco.MjModel.from_xml_path(str(assets / "reference_scene.xml"))
        original_model = mujoco.MjModel.from_xml_path(str(assets / "original_scene.xml"))
        target_model = mujoco.MjModel.from_xml_path(str(assets / "scene.xml"))
        source_names = [model.joint(i).name for i in range(1, model.njnt)]
        target_names = [target_model.joint(i).name for i in range(1, target_model.njnt)]
        if tuple(source_names) != LAFAN_G1_JOINT_NAMES:
            raise ValueError("Reference model and documented CSV joint order disagree")
        raw_subset = project_joint_subset(original, source_names, target_names)
        q = project_joint_subset(sampled, source_names, target_names)
        np.savez_compressed(row["reference_motion_npz"], qpos=original, fps=30.,
                            joint_names=np.asarray(source_names), qpos_order="wxyz,xyz,joints")
        np.savez_compressed(artifact / "source_rate_subset.npz", qpos=raw_subset, fps=30.,
                            joint_names=np.asarray(target_names), qpos_order="wxyz,xyz,joints")
        # Convenient CSV has the same root layout as the input, with 23 named joints.
        np.savetxt(artifact / "source_rate_subset.csv",
                   np.c_[raw_subset[:, 4:7], raw_subset[:, [1, 2, 3, 0]], raw_subset[:, 7:]],
                   delimiter=",", fmt="%.17g")
        mapping = MotionDataConfig(data_format="mocap", robot_type="g1").resolved_joints_mapping
        points, errors, original_env, core = revision_reference(original_model, model, sampled, mapping, MOCAP_DEMO_JOINTS)
        sequence = extract_foot_sticking_sequence_velocity(points, MOCAP_DEMO_JOINTS,
                       ["LeftToeBase", "RightToeBase"], velocity_threshold=.3 / fps)
        retargeted = artifact / "retargeted" / (row["clip"] + "_original.npz")
        retargeted.parent.mkdir(exist_ok=True)
        np.savez_compressed(retargeted, qpos=np.c_[q[:, 4:7], q[:, :4], q[:, 7:]], fps=fps,
                            joint_names=np.asarray(target_names), robot_fk_keypoints=points,
                            source_keypoints_kind="csv_joint_values_on_unified_spherehand29",
                            source_stance_flags=np.asarray([[s["L_Toe"], s["R_Toe"]] for s in sequence]))
        revision = {"clip": row["clip"], "frames": len(q), "links": core,
                    "original_urdf": row["original_urdf"], "unified_reference_urdf": row["reference_urdf"],
                    "mean_fk_position_error_m": dict(zip(core, errors.mean(axis=0).tolist())),
                    "max_fk_position_error_m": dict(zip(core, errors.max(axis=0).tolist())),
                    "original_native_environment_invalid_frames": int((original_env < -.003).sum()),
                    "note": "Robot revision and geometry differences, before fixing wrists; original flat palms and sphere hands are not equivalent contact surfaces"}
        (artifact / "model_revision_report.json").write_text(json.dumps(revision, indent=2) + "\n")
        np.savez_compressed(artifact / "model_revision_diagnostics.npz", core_position_errors=errors,
                            original_native_environment_minimum_distances=original_env)
        manifest.update(target_mapping=mapping, source_frames=len(original), source_fps=30., output_frames=len(times))
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        command = [label_python, "-m", "acp.data.adapters.holosoma", "--input", str(retargeted),
                   "--target-urdf", str(urdf), "--terrain-urdf", row["terrain_urdf"],
                   "--source-clip", row["source"], "--output", str(artifact / "labeled"),
                   "--fps", str(fps), "--input-method", "asap_joint_subset"]
        with (artifact / "labels.log").open("w") as log:
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (artifact / "native.log").open("w") as log, redirect_stdout(log):
            check(assets / "scene.xml", artifact / "labeled/motion.npz", artifact / "validation", retargeted)
        compare(row, run, None)
        manifest.update(status="end_to_end_data_pipeline_completed_not_training_admitted")
    except Exception as exc:
        manifest.update(status="failed", failed_stage="lafan_asap_joint_subset_pipeline", error=f"{type(exc).__name__}: {exc}")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"clip": row["clip"], "status": manifest["status"], "error": manifest.get("error")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/lyx/train/dataset/LAFAN1_Retargeting_Dataset"))
    parser.add_argument("--reference-urdf", type=Path, default=Path("/home/lyx/train/retarget/dataset/OmniRetarget_Dataset/models/g1/g1_29dof_spherehand.urdf"))
    parser.add_argument("--asap-root", type=Path, default=Path("/home/lyx/train/ASAP"))
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/lafan_asap23")
    parser.add_argument("--label-python", type=Path, default=PROJECT / ".venv/bin/python")
    parser.add_argument("--fps", type=float, default=50.)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0 or args.workers < 1:
        parser.error("fps and workers must be positive")
    run, dataset = args.output.resolve(), args.dataset_root.resolve()
    if run == dataset or dataset in run.parents:
        parser.error("Keep outputs outside the source dataset")
    assets = run / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    original_urdf = dataset / "robot_description/g1/g1_29dof_rev_1_0.urdf"
    reference = args.reference_urdf.resolve()
    for urdf in (original_urdf, reference):
        if tuple(moving_joint_names(urdf)) != LAFAN_G1_JOINT_NAMES:
            raise ValueError(f"Documented CSV order differs from {urdf}")
    target = prepare(reference, args.asap_root, assets)
    terrain = assets / "flat_ground.urdf"
    terrain.write_text('<robot name="flat_ground"><link name="world"/></robot>\n')
    make_scene(target.with_suffix(".xml"), terrain, assets / "scene.xml")
    for name, urdf in (("reference", reference), ("original", original_urdf)):
        save_robot_xml(compile_robot(urdf), assets / (name + "_robot.xml"))
        make_scene(assets / (name + "_robot.xml"), terrain, assets / (name + "_scene.xml"))
    originals = {j.get("name"): j for j in ET.parse(original_urdf).getroot().findall("joint")}
    unified = {j.get("name"): j for j in ET.parse(reference).getroot().findall("joint")}
    differences = []
    for name in LAFAN_G1_JOINT_NAMES:
        for tag in ("origin", "axis", "limit"):
            a, b = originals[name].find(tag), unified[name].find(tag)
            if a is not None and b is not None and a.attrib != b.attrib:
                differences.append({"joint": name, "field": tag, "original": a.attrib, "unified": b.attrib})
    commit = subprocess.check_output(["git", "-C", str(dataset), "rev-parse", "HEAD"], text=True).strip()
    (assets / "lafan_provenance.json").write_text(json.dumps({"dataset_root": str(dataset), "dataset_git_commit": commit,
        "readme_sha256": digest(dataset / "README.md"), "original_urdf_sha256": digest(original_urdf),
        "reference_urdf_sha256": digest(reference), "documented_csv_joint_names": LAFAN_G1_JOINT_NAMES,
        "source_fps": 30., "input_root_order": "xyz,xyzw", "source_to_unified_joint_differences": differences,
        "ground": "flat z=0; root height and world trajectory are not adjusted"}, indent=2) + "\n")
    rows = []
    for source in sorted((dataset / "g1").glob("*.csv")):
        if args.clips is not None and source.stem not in args.clips:
            continue
        q = read_lafan_csv(source)
        rows.append({"clip": source.stem, "source": str(source), "source_sha256": digest(source),
                     "source_frames": len(q), "source_fps": 30., "has_human_joints": False,
                     "scope": "LAFAN1 G1 on flat ground", "motion_group": source.stem.split("_subject")[0],
                     "route": "lafan_asap_joint_subset", "reference_urdf": str(reference),
                     "original_urdf": str(original_urdf), "terrain_urdf": str(terrain),
                     "reference_motion_npz": str(run / "clips" / source.stem / "source29.npz")})
    if not rows or (args.clips and set(args.clips) - {r["clip"] for r in rows}):
        parser.error("No G1 CSV files or unknown clip names")
    (run / "inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(convert, row, str(run), str(args.label_python.absolute()), args.fps, args.force) for row in rows]
        for future in as_completed(tasks):
            result = future.result()
            results.append(result)
            print(json.dumps({"done": len(results), "total": len(rows), **result}), flush=True)
    (run / "conversion_results.json").write_text(json.dumps(results, indent=2) + "\n")
    subprocess.run([sys.executable, str(PROJECT / "src/acp/quality/evaluate.py"),
                    "--run", str(run), "--workers", str(args.workers)], check=True)
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
