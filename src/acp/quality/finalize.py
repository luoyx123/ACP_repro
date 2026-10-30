"""Label and evaluate completed local arm repairs without admitting them to training."""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from acp.data.assets import read_record, resolve_path, run_path

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from acp.quality.native import check

PROJECT = Path(__file__).resolve().parents[3]


def finalize(row, run, label_python):
    from contextlib import redirect_stdout
    artifact = run / "clips" / row["clip"]
    manifest_path = artifact / "manifest.json"
    manifest = read_record(manifest_path)
    report = read_record(artifact / "arm_repair_report.json")
    # Prefix-only development exports are deliberately excluded from full-clip evaluation.
    with np.load(artifact / "retargeted" / (row["clip"] + "_original.npz")) as z:
        expected = int((row["source_frames"] - 1) / row["source_fps"] * float(z["fps"]) + 1e-8) + 1
        if len(z["qpos"]) != expected:
            raise ValueError("Cannot finalize a partial development trajectory as a full clip")
    source = Path(manifest["source"])
    if "terrain_urdf" in manifest:
        terrain = Path(manifest["terrain_urdf"])
    else:
        base, scale = row["clip"].split("_z_scale_")
        terrain = source.parent.parent / "models/terrain" / base / f"multi_boxes_z_scale_{scale}.urdf"
    retargeted = artifact / "retargeted" / (row["clip"] + "_original.npz")
    command = [label_python, "-m", "acp.data.adapters.holosoma", "--input", str(retargeted),
               "--target-urdf", manifest["target_urdf"], "--terrain-urdf", str(terrain),
               "--source-clip", str(source), "--output", str(artifact / "labeled"),
               "--fps", str(report["fps"]), "--input-method", "asap_local_arm_repair"]
    with (artifact / "labels.log").open("w") as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (artifact / "native.log").open("w") as log, redirect_stdout(log):
        check(Path(manifest["scene"]), artifact / "labeled/motion.npz", artifact / "validation", retargeted)
    manifest.update(status="end_to_end_arm_repair_pipeline_completed_not_training_admitted",
                    final_geometry_constraint_failed_frames=report["final_geometry_constraint_failed_frames"],
                    stage1_training_admitted=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return row["clip"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=run_path("omni_asap23_arm_repair"))
    parser.add_argument("--baseline", type=Path, default=run_path("omni_asap23"))
    parser.add_argument("--label-python", type=Path, default=PROJECT / ".venv/bin/python")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    clips = {p.parent.name for p in (args.run / "clips").glob("*/arm_repair_report.json")}
    rows = [row for row in read_record(args.baseline / "inventory.json") if row["clip"] in clips]
    rows = [{**row, "route": "asap23_local_arm_repair"} for row in rows]
    (args.run / "inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(finalize, row, args.run.resolve(), str(args.label_python.absolute())) for row in rows]
        for task in as_completed(pending):
            print(task.result(), flush=True)
    subprocess.run([sys.executable, str(PROJECT / "src/acp/quality/evaluate.py"),
                    "--run", str(args.run.resolve()), "--workers", str(args.workers)], check=True)
