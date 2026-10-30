"""Repair wrist-fixed G1 references by moving only the eight shoulder/elbow DoFs.

Uses source surface distances as soft contact tasks, exact nonlinear geometry
constraints, and a time-coherent joint correction. Keeps the baseline intact.
"""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from acp.data.assets import read_record, resolve_path, run_path

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from acp.data.motions import resample_drake
from acp.data.playback import resolve_clip
from acp.robot.model import digest


class ArmProblem:
    def __init__(self, target, source, smooth_weight=3.0):
        self.model, self.source = target, source
        self.smooth_weight = smooth_weight
        self.data, self.source_data = mujoco.MjData(target), mujoco.MjData(source)
        names = [f"{side}_{joint}_joint" for side in ("left", "right")
                 for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
        self.qids = np.array([target.joint(name).qposadr[0] for name in names])
        self.vids = np.array([target.joint(name).dofadr[0] for name in names])
        self.joint_names = names
        self.lower = np.array([target.jnt_range[target.joint(name).id, 0] for name in names])
        self.upper = np.array([target.jnt_range[target.joint(name).id, 1] for name in names])
        self.hands = [target.body(side + "_sphere_hand_link").id for side in ("left", "right")]
        self.source_hands = [source.body(side + "_sphere_hand_link").id for side in ("left", "right")]
        collidable = [g for g in range(target.ngeom) if target.geom_contype[g] or target.geom_conaffinity[g]]
        self.env = [g for g in collidable if target.geom_bodyid[g] == 0
                    or (target.geom(g).name or "").startswith("multi_boxes_")]
        robot = [g for g in collidable if g not in self.env]
        arm_bodies = set()
        for body in range(target.nbody):
            ancestor = body
            while ancestor:
                if target.body(ancestor).name in ("left_shoulder_pitch_link", "right_shoulder_pitch_link"):
                    arm_bodies.add(body)
                    break
                ancestor = int(target.body_parentid[ancestor])
        arm = [g for g in robot if target.geom_bodyid[g] in arm_bodies]
        self.hand_geoms = [next(g for g in arm if target.geom_bodyid[g] == body) for body in self.hands]

        def separation(a, b):
            ancestors, count = {}, 0
            while True:
                ancestors[a] = count
                if a == 0:
                    break
                count += int(target.body_jntnum[a] > 0)
                a = int(target.body_parentid[a])
            count = 0
            while b not in ancestors:
                count += int(target.body_jntnum[b] > 0)
                b = int(target.body_parentid[b])
            return count + ancestors[b]

        pairs = {(a, b) for a in arm for b in self.env}
        for a in arm:
            for b in robot:
                if a == b:
                    continue
                ba, bb = int(target.geom_bodyid[a]), int(target.geom_bodyid[b])
                if target.body_weldid[ba] == target.body_weldid[bb] or separation(ba, bb) <= 2:
                    continue
                pairs.add(tuple(sorted((a, b))))
        self.pairs = sorted(pairs)
        self.source_pairs = [(source.geom(target.geom(a).name).id, source.geom(target.geom(b).name).id)
                             for a, b in self.pairs]
        self.x = None

    def set_source(self, source_q, nominal):
        self.source_data.qpos[:] = source_q
        mujoco.mj_forward(self.source, self.source_data)
        self.nominal = nominal.copy()
        self.desired_centers = self.source_data.xpos[self.source_hands].copy()
        # Per-pair source distances preserve inherited interpenetration explicitly.
        # No global exemption is inferred from a body-pair name.
        self.source_distances = np.array([mujoco.mj_geomDistance(
            self.source, self.source_data, a, b, .5, None) for a, b in self.source_pairs])
        self.bounds_distance = np.minimum(-.0025, self.source_distances - .0005)
        self.contacts = []
        self.source_hand_distances = []
        for hand in self.hand_geoms:
            source_hand = self.source.geom(self.model.geom(hand).name).id
            options = [(mujoco.mj_geomDistance(self.source, self.source_data, source_hand,
                       self.source.geom(self.model.geom(env).name).id, .5, None), env) for env in self.env]
            distance, env = min(options)
            self.source_hand_distances.append(distance)
            if distance < .030:
                # Smoothly introduce surface tracking before the labeling band.
                # Preserve positive source clearance rather than snapping a hand
                # at 15 mm to an artificial 6 mm target in one frame.
                blend = float(np.clip((.030 - distance) / .015, 0., 1.))
                weight = 150 * blend**2 * (3 - 2 * blend)
                self.contacts.append((hand, env, max(float(distance), 0.), weight))
        self.x = None

    def update(self, x):
        if self.x is None or not np.array_equal(x, self.x):
            self.data.qpos[:] = self.nominal
            self.data.qpos[self.qids] = x
            mujoco.mj_forward(self.model, self.data)
            self.x = x.copy()

    def point_jacobian(self, body, point):
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac, None, point, body)
        return jac[:, self.vids]

    def distance(self, a, b, jacobian=False):
        witness = np.zeros(6)
        phi = mujoco.mj_geomDistance(self.model, self.data, a, b, .5, witness)
        if not jacobian:
            return phi
        direction = witness[3:] - witness[:3]
        if abs(phi) > 1e-9 and phi < .5:
            normal = direction / phi
            normal /= max(np.linalg.norm(normal), 1e-12)
            gradient = normal @ (self.point_jacobian(int(self.model.geom_bodyid[b]), witness[3:])
                                 - self.point_jacobian(int(self.model.geom_bodyid[a]), witness[:3]))
        else:
            # Rare coincident witnesses: finite difference the actual query.
            original = self.x.copy()
            gradient = np.zeros(8)
            for index in range(8):
                delta = np.eye(8)[index] * 1e-6
                self.update(original + delta)
                upper = self.distance(a, b)
                self.update(original - delta)
                lower = self.distance(a, b)
                gradient[index] = (upper - lower) / 2e-6
            self.update(original)
        return phi, gradient

    def objective(self, x, correction_anchor):
        self.update(x)
        residuals, jacobians = [], []
        for index, body in enumerate(self.hands):
            residuals.extend(20 * (self.data.xpos[body] - self.desired_centers[index]))
            jacobians.extend(20 * self.point_jacobian(body, self.data.xpos[body]))
        for hand, env, desired, weight in self.contacts:
            distance, jac = self.distance(hand, env, True)
            residuals.append(weight * (distance - desired))
            jacobians.append(weight * jac)
        offset = x - self.nominal[self.qids]
        residuals.extend(.15 * offset)
        jacobians.extend(.15 * np.eye(8))
        residuals.extend(self.smooth_weight * (offset - correction_anchor))
        jacobians.extend(self.smooth_weight * np.eye(8))
        r, j = np.asarray(residuals), np.asarray(jacobians)
        return float(r @ r), 2 * j.T @ r

    def constraints(self, x, indices):
        self.update(x)
        distances, jacobians = [], []
        for index in indices:
            phi, jac = self.distance(*self.pairs[index], True)
            distances.append(phi - self.bounds_distance[index])
            jacobians.append(jac)
        return np.asarray(distances), np.asarray(jacobians).reshape(-1, 8)

    def all_distances(self, x):
        self.update(x)
        return np.array([self.distance(a, b) for a, b in self.pairs])

    def solve(self, initial, anchor, lower, upper):
        initial = np.clip(initial, lower, upper)
        distances = self.all_distances(initial)
        # Active pairs selected generously; every returned candidate is checked
        # against ALL eligible pairs, then re-solved if new constraints activate.
        indices = set(np.flatnonzero(np.minimum(distances, self.source_distances) < .12).tolist())
        result = None
        for _ in range(3):
            active = np.array(sorted(indices), dtype=int)
            constraint = {"type": "ineq", "fun": lambda x: self.constraints(x, active)[0],
                          "jac": lambda x: self.constraints(x, active)[1]}
            result = minimize(lambda x: self.objective(x, anchor), initial, jac=True, method="SLSQP",
                              bounds=list(zip(lower, upper)), constraints=constraint,
                              options={"maxiter": 35, "ftol": 1e-8})
            actual = self.all_distances(result.x)
            violated = set(np.flatnonzero(actual < self.bounds_distance - 2e-5).tolist())
            new = violated - indices
            if not new:
                break
            indices.update(new)
            initial = result.x
        feasible = np.all(actual >= self.bounds_distance - 2e-5)
        self.update(result.x)
        return result.x, bool(feasible), bool(result.success), str(result.message), float(np.min(actual - self.bounds_distance))


def repair(clip_name, baseline, output, passes=2, maximum_frames=None, smooth_weight=3.0):
    began = time.monotonic()
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    info = resolve_clip(baseline, clip_name)
    baseline_manifest = read_record(info["artifact"] / "manifest.json")
    reference_motion = Path(baseline_manifest.get("source_motion_npz", info["source"]))
    artifact = output / "clips" / clip_name
    artifact.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(info["scene"]))
    source = mujoco.MjModel.from_xml_path(str(info["artifact"] / "comparison/source_scene.xml"))
    problem = ArmProblem(model, source, smooth_weight=smooth_weight)
    with np.load(info["motion"]) as z:
        fps, times = float(z["fps"]), z["timestamps"].copy()
        nominal = np.c_[z["base_position_w"], z["base_quaternion_wxyz"], z["joint_positions"]]
    with np.load(reference_motion) as z:
        _, canonical, _ = resample_drake(z["qpos"], float(z["fps"]), fps)
    source_q = np.c_[canonical[:, 4:7], canonical[:, :4], canonical[:, 7:]]
    if maximum_frames is not None:
        nominal, source_q, times = nominal[:maximum_frames], source_q[:maximum_frames], times[:maximum_frames]
    import xml.etree.ElementTree as ET
    urdf = Path(baseline_manifest["target_urdf"])
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    speed_limits = np.array([float(joints[name].find("limit").get("velocity")) for name in problem.joint_names])
    corrected = nominal.copy()
    statuses = []
    for pass_index in range(passes):
        forward = pass_index % 2 == 0
        for frame in (range(len(corrected)) if forward else range(len(corrected)-1, -1, -1)):
            problem.set_source(source_q[frame], nominal[frame])
            x0 = corrected[frame, problem.qids].copy()
            neighbours = [f for f in (frame - 1, frame + 1) if 0 <= f < len(corrected)]
            anchor = np.mean([corrected[f, problem.qids] - nominal[f, problem.qids] for f in neighbours], axis=0) if neighbours else np.zeros(8)
            lower = np.maximum(problem.lower, nominal[frame, problem.qids] - .8)
            upper = np.minimum(problem.upper, nominal[frame, problem.qids] + .8)
            for neighbour in neighbours:
                change = speed_limits * abs(times[frame] - times[neighbour]) * .95
                lower = np.maximum(lower, corrected[neighbour, problem.qids] - change)
                upper = np.minimum(upper, corrected[neighbour, problem.qids] + change)
            if np.any(lower > upper):
                raise RuntimeError("Neighbour velocity bounds are inconsistent")
            candidate, feasible, converged, message, margin = problem.solve(x0, anchor, lower, upper)
            # Failed frames retain their prior baseline/candidate and remain
            # explicitly marked for review; they are never declared successful.
            if feasible:
                corrected[frame, problem.qids] = candidate
            statuses.append({"pass": pass_index, "frame": frame, "accepted": feasible,
                             "optimizer_success": converged, "message": message,
                             "minimum_constraint_margin_m": margin})
        print(f"{clip_name} pass {pass_index+1}/{passes} completed", flush=True)
    unchanged = [i for i in range(model.nq) if i not in set(problem.qids)]
    np.testing.assert_array_equal(corrected[:, unchanged], nominal[:, unchanged])
    # Validate the saved trajectory independently after both directional passes.
    final_ok, source_distances, before_distances, after_distances, frame_margins = [], [], [], [], []
    centers_before, centers_after = [], []
    for frame in range(len(corrected)):
        problem.set_source(source_q[frame], nominal[frame])
        problem.update(nominal[frame, problem.qids])
        before_distances.append([min(problem.distance(hand, env) for env in problem.env) for hand in problem.hand_geoms])
        centers_before.append(np.linalg.norm(problem.data.xpos[problem.hands] - problem.desired_centers, axis=1))
        distances = problem.all_distances(corrected[frame, problem.qids])
        margin = float(np.min(distances - problem.bounds_distance))
        final_ok.append(margin >= -2e-5)
        frame_margins.append(margin)
        after_distances.append([min(problem.distance(hand, env) for env in problem.env) for hand in problem.hand_geoms])
        centers_after.append(np.linalg.norm(problem.data.xpos[problem.hands] - problem.desired_centers, axis=1))
        source_distances.append(problem.source_hand_distances)
    src, before, after = map(np.asarray, (source_distances, before_distances, after_distances))
    src_contact = (src >= -.003) & (src <= .015)
    near_before = (before >= -.003) & (before <= .015)
    near_after = (after >= -.003) & (after <= .015)
    joint_speed = np.abs(np.diff(corrected[:, problem.qids], axis=0) / np.diff(times)[:, None])
    gradient_speed = np.abs(np.gradient(corrected[:, problem.qids], times, axis=0))
    offset = corrected[:, problem.qids] - nominal[:, problem.qids]
    correction_speed = np.diff(offset, axis=0) / np.diff(times)[:, None]
    report = {"clip": clip_name, "frames": len(times), "fps": fps, "passes": passes,
              "smooth_weight": smooth_weight, "surface_task_transition_m": [.015, .030],
              "method": "eight_arm_dofs_slsqp_source_surface_tasks_exact_geometry_acceptance",
              "signature": {"script_sha256": digest(__file__), "baseline_motion_sha256": digest(info["motion"]),
                            "scene_sha256": digest(info["scene"]), "source_sha256": digest(info["source"]),
                            "reference_motion_sha256": digest(reference_motion)},
              "baseline_motion": str(info["motion"]), "scene": str(info["scene"]),
              "source": str(info["source"]), "reference_motion_npz": str(reference_motion),
              "arm_joint_names": problem.joint_names,
              "root_waist_legs_exactly_preserved": True,
              "source_hand_near_surface_steps": int(src_contact.sum()),
              "contact_retention_before": float(near_before[src_contact].mean()) if src_contact.any() else None,
              "contact_retention_after": float(near_after[src_contact].mean()) if src_contact.any() else None,
              "hand_environment_invalid_steps_before": int((before < -.003).sum()),
              "hand_environment_invalid_steps_after": int((after < -.003).sum()),
              "hand_center_error_before_mean_m": float(np.mean(centers_before)),
              "hand_center_error_after_mean_m": float(np.mean(centers_after)),
              "maximum_arm_correction_rad": float(np.max(np.abs(corrected[:, problem.qids] - nominal[:, problem.qids]))),
              "maximum_arm_velocity_limit_ratio": float(np.max(joint_speed / speed_limits)),
              "maximum_gradient_arm_velocity_limit_ratio": float(np.max(gradient_speed / speed_limits)),
              "maximum_correction_step_rad": float(np.max(np.abs(np.diff(offset, axis=0)))),
              "rms_correction_velocity_rad_s": float(np.sqrt(np.mean(correction_speed**2))),
              "final_geometry_constraint_failed_frames": int(len(times) - sum(final_ok)),
              "final_geometry_failed_frame_indices": np.flatnonzero(~np.asarray(final_ok)).tolist(),
              "solver_nonconverged_attempts": sum(not s["optimizer_success"] for s in statuses),
              "rejected_attempts": sum(not s["accepted"] for s in statuses),
              "elapsed_s": time.monotonic() - began,
              "stage1_training_admitted": False,
              "limitations": ["Source distance contacts are not force labels", "Inherited per-pair penetrations are reported, not globally erased",
                              "Self pairs within two moving joints or the same welded body are excluded; calibrated allowed pairs still pending",
                              "Arm deviation bounded to 0.8 rad; not all source wrist poses are reachable", "No dynamics validation"]}
    retargeted = artifact / "retargeted" / (clip_name + "_original.npz")
    retargeted.parent.mkdir(exist_ok=True)
    original_retargeted = info["artifact"] / "retargeted" / (clip_name + "_original.npz")
    with np.load(original_retargeted) as z:
        metadata = {name: z[name][:len(times)]
                    for name in ("robot_fk_keypoints", "source_stance_flags")}
    np.savez_compressed(retargeted, qpos=corrected, fps=fps, **metadata)
    np.savez_compressed(artifact / "arm_repair_diagnostics.npz", timestamps=times, qpos=corrected,
                        baseline_qpos=nominal, source_hand_distances=src, before_hand_distances=before,
                        after_hand_distances=after, frame_constraint_margins=frame_margins,
                        hand_center_errors_before=centers_before, hand_center_errors_after=centers_after)
    (artifact / "arm_repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (artifact / "solver_status.json").write_text(json.dumps(statuses, indent=2) + "\n")
    manifest = {**baseline_manifest, "status": "arm_repair_completed_pending_labels_and_evaluation",
                "route": "asap23_local_arm_repair", "baseline_artifact": str(info["artifact"]),
                "arm_repair_report": str(artifact / "arm_repair_report.json"),
                "baseline_signature": baseline_manifest.get("signature"), "signature": report["signature"]}
    (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=run_path("omni_asap23"))
    parser.add_argument("--output", type=Path, default=run_path("omni_asap23_arm_repair"))
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument("--smooth-weight", type=float, default=3.0)
    args = parser.parse_args()
    if args.workers < 1 or args.passes < 1 or (args.maximum_frames is not None and args.maximum_frames < 2):
        parser.error("workers/passes positive; maximum-frames at least two")
    if args.output.resolve() == args.baseline.resolve():
        parser.error("Output must differ from the preserved baseline")
    if not np.isfinite(args.smooth_weight) or args.smooth_weight < 0:
        parser.error("smooth-weight must be finite and nonnegative")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(repair, name, args.baseline, args.output, args.passes, args.maximum_frames,
                             args.smooth_weight) for name in args.clips]
        results = [future.result() for future in as_completed(tasks)]
    (args.output / "arm_repair_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
