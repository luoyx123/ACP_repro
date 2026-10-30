"""Preserve dataset G1-29 motions using ASAP's six-wrist-fixed 23-DoF subset."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src"))
from acp.robot.joints import project_joint_subset
from acp.data.motions import resample_drake
from acp.robot.model import compile_robot, digest, make_scene, prepare
from acp.robot.transfer import fk_keypoints


def convert(row, run, dataset, urdf, label_python, fps, force):
    from holosoma_retargeting.config_types.data_type import MotionDataConfig, MOCAP_DEMO_JOINTS
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity
    from acp.quality.native import check

    run, dataset, urdf = Path(run), Path(dataset), Path(urdf)
    artifact = run / "clips" / row["clip"]
    artifact.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact / "manifest.json"
    source = Path(row["source"])
    signature = {"source_sha256": digest(source), "target_sha256": digest(urdf),
                 "target_xml_sha256": digest(urdf.with_suffix(".xml")), "fps": fps,
                 "asset_preparer_sha256": digest(PROJECT / "scripts/prepare_asap23.py"),
                 "converter_sha256": digest(__file__), "projection_sha256": digest(PROJECT / "src/acp/data/asap.py"),
                 "labeler_sha256": digest(PROJECT / "src/acp/data/holosoma.py"),
                 "native_checker_sha256": digest(PROJECT / "scripts/check_holosoma_mujoco.py")}
    if manifest_path.exists() and not force:
        previous = json.loads(manifest_path.read_text())
        if previous.get("signature") == signature and previous.get("status", "").startswith("end_to_end"):
            return {"clip": row["clip"], "status": "resumed"}
    manifest = {"status": "running", "route": "asap_joint_subset", "source": str(source),
                "signature": signature, "target_urdf": str(urdf), "target_mapping": {},
                "method": "ASAP wrist-fixed joint definition; direct subset, no human refitting"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        with np.load(source, allow_pickle=False) as raw:
            original, source_fps = raw["qpos"].copy(), float(raw["fps"])
        times, source_q, _ = resample_drake(original, source_fps, fps)
        source_model = compile_robot(dataset / "models/g1/g1_29dof_spherehand.urdf")
        target_model = mujoco.MjModel.from_xml_path(str(urdf.with_suffix(".xml")))
        source_names = [source_model.joint(i).name for i in range(1, source_model.njnt)]
        target_names = [target_model.joint(i).name for i in range(1, target_model.njnt)]
        target_q = project_joint_subset(source_q, source_names, target_names)
        # Preserve a source-rate export exactly, independently of optional 50-Hz labels.
        raw_subset = project_joint_subset(original, source_names, target_names)
        np.savez_compressed(artifact / "source_rate_subset.npz", qpos=raw_subset, fps=source_fps,
                            joint_names=np.asarray(target_names), qpos_order="wxyz,xyz,joints")
        mapping = MotionDataConfig(data_format="mocap", robot_type="g1").resolved_joints_mapping
        points = fk_keypoints(source_model, source_q, MOCAP_DEMO_JOINTS, mapping)
        sequence = extract_foot_sticking_sequence_velocity(points, MOCAP_DEMO_JOINTS,
                      ["LeftToeBase", "RightToeBase"], velocity_threshold=0.3 / fps)
        retargeted = artifact / "retargeted" / (row["clip"] + "_original.npz")
        retargeted.parent.mkdir(exist_ok=True)
        np.savez_compressed(retargeted, qpos=np.c_[target_q[:, 4:7], target_q[:, :4], target_q[:, 7:]], fps=fps,
                            joint_names=np.asarray(target_names), robot_fk_keypoints=points,
                            source_keypoints_kind="source_robot_fk",
                            source_stance_flags=np.asarray([[s["L_Toe"], s["R_Toe"]] for s in sequence]))
        base, scale = row["clip"].split("_z_scale_")
        terrain = dataset / "models/terrain" / base / f"multi_boxes_z_scale_{scale}.urdf"
        scene = artifact / "scene.xml"
        make_scene(urdf.with_suffix(".xml"), terrain, scene)
        manifest.update(scene=str(scene), target_mapping=mapping,
                        source_frames=len(original), source_fps=source_fps, output_frames=len(times))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        command = [label_python, "-m", "acp.data.adapters.holosoma", "--input", str(retargeted),
                   "--target-urdf", str(urdf), "--terrain-urdf", str(terrain), "--source-clip", str(source),
                   "--output", str(artifact / "labeled"), "--fps", str(fps), "--input-method", "asap_joint_subset"]
        with (artifact / "labels.log").open("w") as log:
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (artifact / "native.log").open("w") as log:
            from contextlib import redirect_stdout
            with redirect_stdout(log):
                check(scene, artifact / "labeled/motion.npz", artifact / "validation", retargeted)
        manifest.update(status="end_to_end_data_pipeline_completed_not_training_admitted")
    except Exception as exc:
        manifest.update(status="failed", failed_stage="asap_joint_subset_pipeline", error=f"{type(exc).__name__}: {exc}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"clip": row["clip"], "status": manifest["status"], "error": manifest.get("error")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/lyx/train/retarget/dataset/OmniRetarget_Dataset"))
    parser.add_argument("--asap-root", type=Path, default=Path("/home/lyx/train/ASAP"))
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/omni_asap23")
    parser.add_argument("--label-python", type=Path, default=PROJECT / ".venv/bin/python")
    parser.add_argument("--fps", type=float, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or not np.isfinite(args.fps) or args.workers < 1:
        parser.error("fps and workers must be positive")
    run, dataset = args.output.resolve(), args.dataset_root.resolve()
    urdf = prepare(dataset / "models/g1/g1_29dof_spherehand.urdf", args.asap_root, run / "assets")
    rows = []
    for source in sorted((dataset / "robot-terrain").glob("*.npz")):
        with np.load(source, allow_pickle=False) as raw:
            rows.append({"clip": source.stem, "source": str(source), "source_sha256": digest(source),
                         "source_frames": len(raw["qpos"]), "source_fps": float(raw["fps"]),
                         "has_human_joints": "human_joints" in raw,
                         "route": "asap_joint_subset"})
    (run / "inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    selected = [row for row in rows if args.clips is None or row["clip"] in args.clips]
    if args.clips and set(args.clips) - {row["clip"] for row in selected}:
        parser.error("Unknown clip names")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(convert, row, str(run), str(dataset), str(urdf),
                               str(args.label_python.absolute()), args.fps, args.force) for row in selected]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            print(json.dumps({"done": len(results), "total": len(selected), **result}), flush=True)
    (run / "conversion_results.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
