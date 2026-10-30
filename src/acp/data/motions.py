"""Explicit motion format adapters. Canonical floating-base order is WXYZ, XYZ."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def fixed_rate_times(frame_count, source_fps, output_fps):
    """Fixed output clock, retaining an endpoint that is on-grid within roundoff."""
    if frame_count < 2 or not np.isfinite(source_fps) or source_fps <= 0 or not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("Expected at least two frames and finite positive rates")
    if source_fps == output_fps:
        return np.arange(frame_count, dtype=float) / output_fps
    duration = (frame_count - 1) / source_fps
    return np.arange(int(np.floor(duration * output_fps + 1e-9)) + 1, dtype=float) / output_fps


def resample_drake(q, fps, output_fps=50):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < 2:
        raise ValueError("Expected one G1 29-DoF episode [T,36], no object poses")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(output_fps) or output_fps <= 0 or not np.all(np.isfinite(q)):
        raise ValueError("Invalid fps or motion values")
    norm = np.linalg.norm(q[:, :4], axis=1)
    if np.any(abs(norm - 1) > 0.02):
        raise ValueError("Expected quaternion-first convention: qw,qx,qy,qz,x,y,z,joints")
    time_in = np.arange(len(q)) / fps
    time_out = fixed_rate_times(len(q), fps, output_fps)
    output = np.empty((len(time_out), 36))
    query_times = np.clip(time_out, time_in[0], time_in[-1])
    rotations = Slerp(time_in, Rotation.from_quat(q[:, [1, 2, 3, 0]]))(query_times).as_quat()
    output[:, :4] = rotations[:, [3, 0, 1, 2]]
    for column in range(4, 36):
        output[:, column] = np.interp(time_out, time_in, q[:, column])
    return time_out, output, fps


def load_omni(path, output_fps=50):
    with np.load(path, allow_pickle=False) as data:
        return resample_drake(data["qpos"], float(data["fps"]), output_fps)


LAFAN_G1_JOINT_NAMES = tuple(
    [f"{side}_{joint}_joint" for side in ("left", "right")
     for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [f"{side}_{joint}_joint" for side in ("left", "right")
       for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                     "wrist_roll", "wrist_pitch", "wrist_yaw")])


def read_lafan_csv(path):
    """Read 30-Hz G1 CSV as canonical qpos, preserving every source value."""
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("Expected finite lvhaidong G1 CSV [T,36], at least two frames, no header")
    if np.any(np.abs(np.linalg.norm(raw[:, 3:7], axis=1) - 1) > .02):
        raise ValueError("LAFAN G1 root must be XYZ followed by a unit XYZW quaternion")
    return np.c_[raw[:, [6, 3, 4, 5]], raw[:, :3], raw[:, 7:]]


def load_lafan_csv(path, output_fps=50):
    """lvhaidong dataset: 30Hz, root XYZ + quaternion XYZW + 29 named joints."""
    return resample_drake(read_lafan_csv(path), 30.0, output_fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["omni", "lafan_csv"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loader = load_omni if args.format == "omni" else load_lafan_csv
    times, qpos, source_fps = loader(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, timestamps=times, fps=50.0, qpos=qpos)
    report = {"source": str(args.input.resolve()), "source_format": args.format,
              "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "source_fps": source_fps, "output_fps": 50, "frames": len(times),
              "qpos_order": "qw,qx,qy,qz,x,y,z,29 joints in source dataset order",
              "status": "normalized_only_not_retargeted_not_contact_labeled"}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
