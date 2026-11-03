"""29DoF 原生碰撞几何约束修复；保留基线，重新标注并按原门槛评估。"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from acp.data.assets import PROJECT, atomic_json, read_record
from acp.robot.model import digest


class GeometryRepair:
    """Sequential constrained least squares with exact native collision rechecks.

    Root orientation stays fixed. Translation and all named hinge joints are
    optimized; every native active collision (not a hand-picked body blacklist)
    supplies a distance constraint. Local point Jacobians include free-base
    translation and the entire reversed waist chain.
    """
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.qids = np.r_[0:3, model.jnt_qposadr[joints]]
        self.vids = np.r_[0:3, model.jnt_dofadr[joints]]
        self.lower = np.r_[[-np.inf] * 3, model.jnt_range[joints, 0]]
        self.upper = np.r_[[np.inf] * 3, model.jnt_range[joints, 1]]
        self.robot = np.array([model.geom_bodyid[g] != 0 and
            not (model.geom(g).name or '').startswith('multi_boxes_') for g in range(model.ngeom)])
        self.feet = [model.body(s + '_ankle_roll_link').id for s in ('left', 'right')]
        self.bodies = [model.body(n).id for n in ('pelvis', 'left_knee_link', 'right_knee_link',
            'left_elbow_link', 'right_elbow_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link')]
        self.margin = -.0015
        self.jac = np.zeros((3, model.nv))

    def set(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def point_jac(self, body, point):
        mujoco.mj_jac(self.model, self.data, self.jac, None, point, body)
        return self.jac[:, self.vids].copy()

    def collisions(self):
        """Use MuJoCo's actual filtering, including authored exclusions/adjacency."""
        return [c for c in self.data.contact if self.robot[c.geom1] or self.robot[c.geom2]]

    def worst(self, q):
        self.set(q)
        contacts = self.collisions()
        return min([0.] + [float(c.dist) for c in contacts])

    def nonlinear(self, nominal, initial, desired, previous_delta):
        """Exact-distance fallback for changing box corners/capsule contact features."""
        q = initial.copy()
        pairs = set()
        for seed in (nominal, initial):
            self.set(seed)
            pairs.update(tuple(sorted((int(c.geom1), int(c.geom2)))) for c in self.collisions())
        weights = np.r_[[120.] * 3, [.3] * (len(self.qids)-3)]
        lower = np.maximum(self.lower, nominal[self.qids] - np.r_[[.06,.06,.12], [.8]*(len(self.qids)-3)])
        upper = np.minimum(self.upper, nominal[self.qids] + np.r_[[.06,.06,.12], [.8]*(len(self.qids)-3)])
        for _ in range(4):
            ordered = sorted(pairs)
            cache = {}
            def evaluate(x):
                if 'x' in cache and np.array_equal(cache['x'], x):
                    return cache
                candidate = q.copy();candidate[self.qids] = x;self.set(candidate)
                delta = x-nominal[self.qids]
                smooth = delta-previous_delta
                cost = .5*np.sum(weights*delta**2)+.5*.55**2*np.sum(smooth**2)
                grad = weights*delta+.55**2*smooth
                for body, target in zip(self.bodies, desired):
                    error = self.data.xpos[body]-target
                    cost += .5*64.*np.sum(error**2)
                    grad += 64.*error@self.point_jac(body,self.data.xpos[body])
                distances, jacobians = [], []
                for a,b in ordered:
                    points = np.zeros(6)
                    distance = mujoco.mj_geomDistance(self.model,self.data,a,b,.2,points)
                    normal = points[3:]-points[:3]
                    if np.linalg.norm(normal)>1e-10 and abs(distance)<.2:
                        normal *= np.sign(distance)/np.linalg.norm(normal)
                        jac = normal@(self.point_jac(self.model.geom_bodyid[b],points[3:])-
                                      self.point_jac(self.model.geom_bodyid[a],points[:3]))
                    else:
                        jac = np.zeros(len(self.qids))
                    distances.append(distance-self.margin);jacobians.append(jac)
                cache.update(x=x.copy(),cost=cost,grad=grad,distance=np.array(distances),jac=np.array(jacobians))
                return cache
            result=minimize(lambda x:evaluate(x)['cost'],np.clip(q[self.qids],lower,upper),
                jac=lambda x:evaluate(x)['grad'],bounds=list(zip(lower,upper)),
                constraints=({'type':'ineq','fun':lambda x:evaluate(x)['distance'],
                              'jac':lambda x:evaluate(x)['jac']},),
                method='SLSQP',options={'maxiter':100,'ftol':1e-9})
            candidate=q.copy();candidate[self.qids]=result.x
            if self.worst(candidate)>self.worst(q):
                q=candidate
            self.set(candidate)
            new={tuple(sorted((int(c.geom1),int(c.geom2)))) for c in self.collisions()}
            if self.worst(q)>=-.0025:
                break
            if new<=pairs:
                break
            pairs.update(new)
        return q

    def frame(self, nominal, previous_delta, max_iterations=10, stance_weights=(0., 0.)):
        self.set(nominal)
        desired = self.data.xpos[self.bodies].copy()
        # Keep original world XY. Only near-surface soles receive a soft height
        # task; a swinging foot is never forced onto a top-down heightmap.
        foot_tasks = []
        env_geoms = [g for g in range(self.model.ngeom) if not self.robot[g] and
                     (self.model.geom_contype[g] or self.model.geom_conaffinity[g])]
        for side, body in enumerate(self.feet):
            if stance_weights[side] < .001:
                continue
            geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body and
                     (self.model.geom_contype[g] or self.model.geom_conaffinity[g])]
            nearest = None
            for g in geoms:
                for env in env_geoms:
                    points = np.zeros(6)
                    dist = mujoco.mj_geomDistance(self.model, self.data, g, env, .07, points)
                    if -.07 < dist < .06 and (nearest is None or dist < nearest[0]):
                        nearest = (dist, points.copy())
            if nearest is not None:
                dist, points = nearest
                normal = points[3:] - points[:3]
                if np.linalg.norm(normal) > 1e-8:
                    # Signed distance reverses witness displacement in penetration.
                    normal *= np.sign(dist) / np.linalg.norm(normal)
                    local = self.data.xmat[body].reshape(3, 3).T @ (points[:3] - self.data.xpos[body])
                    weight = stance_weights[side] * np.clip((.06-max(dist,0.))/.02, 0., 1.)
                    foot_tasks.append((body, local, points[:3] + normal * (dist - .003), weight))
        q = nominal.copy()
        delta_previous = np.asarray(previous_delta)
        best, best_distance = q.copy(), self.worst(q)
        status = 'iteration_limit'
        for iteration in range(max_iterations):
            self.set(q)
            width = len(self.qids)
            regularization = np.r_[[120.] * 3, [.3] * (width - 3)]
            A = [np.diag(np.sqrt(regularization))]
            b = [np.sqrt(regularization) * (nominal[self.qids] - q[self.qids])]
            # Smooth the correction, not the original motion itself.
            A.append(np.eye(width) * .55)
            b.append(.55 * (delta_previous - (q[self.qids] - nominal[self.qids])))
            for body, target in zip(self.bodies, desired):
                A.append(8. * self.point_jac(body, self.data.xpos[body]))
                b.append(8. * (target - self.data.xpos[body]))
            for body, local, target, weight in foot_tasks:
                point = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ local
                scale = 25. * np.sqrt(weight)
                A.append(scale * self.point_jac(body, point))
                b.append(scale * (target - point))
            A, b = np.vstack(A), np.concatenate(b)
            H, f = A.T @ A, A.T @ b
            C, d = [], []
            for contact in self.collisions():
                if contact.dist > .012:
                    continue
                body1, body2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
                jac = self.point_jac(body2, contact.pos) - self.point_jac(body1, contact.pos)
                C.append(contact.frame.reshape(3, 3)[0] @ jac)
                d.append(self.margin - float(contact.dist))
            constraints = ()
            if C:
                C, d = np.asarray(C), np.asarray(d)
                constraints = ({'type':'ineq', 'fun':lambda x: C @ x - d, 'jac':lambda x: C},)
            lo = np.maximum(self.lower - q[self.qids], -.12)
            hi = np.minimum(self.upper - q[self.qids], .12)
            # Limit translation relative to the preserved source, not per SQP step.
            lo[:3] = np.maximum(lo[:3], nominal[:3] - q[:3] - [.06, .06, .12])
            hi[:3] = np.minimum(hi[:3], nominal[:3] - q[:3] + [.06, .06, .12])
            if (lo > hi).any():
                # Original joint-limit violations can exceed the trust region.
                lo = np.minimum(lo, hi)
            result = minimize(lambda x: .5 * x @ H @ x - f @ x, np.clip(np.zeros(width), lo, hi),
                jac=lambda x: H @ x - f, bounds=list(zip(lo, hi)), constraints=constraints,
                method='SLSQP', options={'maxiter':60, 'ftol':1e-9})
            candidate = q.copy()
            candidate[self.qids] += result.x
            distance = self.worst(candidate)
            if distance > best_distance + 1e-8 or distance >= self.margin - 1e-5:
                best, best_distance = candidate.copy(), distance
            if not np.isfinite(candidate).all():
                break
            q = candidate
            if distance >= self.margin - 1e-5 and np.linalg.norm(result.x) < .0005:
                status = 'converged'
                break
        if best_distance < -.0025:
            refined = self.nonlinear(nominal, best, desired, delta_previous)
            refined_distance = self.worst(refined)
            if refined_distance > best_distance:
                best, best_distance = refined, refined_distance
                status = 'exact_distance_fallback'
        return best, dict(status=status, iterations=iteration + 1, minimum_distance_m=best_distance)


def repair_clip(row, baseline, output, passes=2, maximum_frames=None, label_python=None):
    from acp.quality.evaluate import assess
    from acp.quality.native import check
    start = time.monotonic()
    baseline, output = Path(baseline), Path(output)
    src, dst = baseline/'clips'/row['clip'], output/'clips'/row['clip']
    manifest = read_record(src/'manifest.json')
    if manifest.get('dof') != 29:
        raise ValueError('fullbody repair currently requires the native 29DoF model')
    signature = dict(baseline=digest(src/'retargeted'/f"{row['clip']}_original.npz"),
        scene=digest(manifest['scene']), code=digest(__file__), passes=passes, maximum_frames=maximum_frames,
        target=digest(manifest['target_urdf']),
        processing={name:digest(PROJECT/name) for name in ('src/acp/data/adapters/holosoma.py',
            'src/acp/data/scene.py','src/acp/quality/native.py','src/acp/quality/evaluate.py','configs/data_quality.json')})
    if (dst/'manifest.json').exists():
        previous = read_record(dst/'manifest.json')
        if previous.get('signature') == signature and previous.get('status','').startswith('end_to_end'):
            return read_record(dst/'fullbody_repair_report.json')
        raise ValueError(f'Output exists with incomplete/different signature: {dst}; choose a fresh output')
    dst.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(manifest['scene'])
    problem = GeometryRepair(model)
    with np.load(src/'retargeted'/f"{row['clip']}_original.npz", allow_pickle=False) as z:
        values = {k:z[k].copy() for k in z.files}
    original_frames = len(values['qpos'])
    nominal = values['qpos'][:maximum_frames].copy()
    q = nominal.copy()
    stance = values.get('source_stance_flags', np.zeros((original_frames,2),dtype=bool))[:len(q)]
    stance_weights = gaussian_filter1d(stance.astype(float), 2., axis=0, mode='nearest')
    before = np.array([problem.worst(frame) for frame in q])
    statuses = []
    for turn in range(passes):
        previous_delta = np.zeros(len(problem.qids))
        for i in (range(len(q)) if turn % 2 == 0 else range(len(q)-1, -1, -1)):
            # Keep the original motion as the fidelity target in every pass.
            candidate, status = problem.frame(nominal[i], previous_delta, stance_weights=stance_weights[i])
            if problem.worst(candidate) >= problem.worst(q[i]) - 1e-6:
                q[i] = candidate
            previous_delta = q[i, problem.qids] - nominal[i, problem.qids]
            statuses.append(status)
        atomic_json(dst/'progress.json',dict(pass_completed=turn+1,frames=len(q),passes=passes))
    after = np.array([problem.worst(frame) for frame in q])
    for key, value in list(values.items()):
        if value.ndim and len(value) == original_frames:
            values[key] = value[:len(q)]
    values['qpos'] = q
    (dst/'retargeted').mkdir(exist_ok=True)
    retargeted = dst/'retargeted'/f"{row['clip']}_original.npz"
    np.savez_compressed(retargeted, **values)
    report = dict(clip=row['clip'], frames=len(q), full_clip=len(q)==original_frames,
        before_invalid_collision_frames=int((before < -.003).sum()),
        after_invalid_collision_frames=int((after < -.003).sum()),
        minimum_distance_before_m=float(before.min()), minimum_distance_after_m=float(after.min()),
        maximum_root_shift_m=float(np.linalg.norm(q[:,:3]-nominal[:,:3],axis=-1).max()),
        maximum_joint_correction_rad=float(np.abs(q[:,7:]-nominal[:,7:]).max()),
        signature=signature, duration_s=time.monotonic()-start,
        note='Geometry repair only; original offline thresholds and velocity/support gates remain active')
    atomic_json(dst/'fullbody_repair_report.json', report)
    atomic_json(dst/'solver_status.json', statuses)
    for name in ('source29.npz', 'model_conversion_report.json', 'source_rate_motion.npz'):
        if (src/name).exists():
            shutil.copy2(src/name, dst/name)
    manifest = {**manifest, 'signature':signature, 'baseline_artifact':str(src),
        'status':'fullbody_repair_pending_labels', 'route':'native29_fullbody_geometry_repair',
        'stage1_training_admitted':False}
    atomic_json(dst/'manifest.json', manifest)
    if maximum_frames is not None:
        return report  # Development prefix never masquerades as a complete clip.
    with (dst/'labels.log').open('w') as log:
        subprocess.run([label_python or str(PROJECT/'.venv/bin/python'), '-m', 'acp.data.adapters.holosoma',
            '--input',str(retargeted),'--target-urdf',manifest['target_urdf'],
            '--terrain-urdf',manifest['terrain_urdf'],'--source-clip',manifest['source'],
            '--output',str(dst/'labeled'),'--fps',str(float(values['fps'])),'--input-method','native_g1_29'],
            cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (dst/'native.log').open('w') as log, redirect_stdout(log):
        check(Path(manifest['scene']), dst/'labeled/motion.npz', dst/'validation', retargeted)
    manifest['status'] = 'end_to_end_fullbody_repair_pipeline_completed_not_training_admitted'
    atomic_json(dst/'manifest.json', manifest)
    quality = assess({**row,'route':manifest['route']}, output, read_record(PROJECT/'configs/data_quality.json'))
    report.update(quality_status=quality['status'], quality_reasons=quality['reasons'])
    # Supplement the original conservative screen; never clear its failure flags.
    with np.load(dst/'labeled/motion.npz',allow_pickle=False) as labeled:
        speed=np.linalg.norm(np.gradient(labeled['body_landmarks_w'][:,:2,:2],labeled['timestamps'],axis=0),axis=-1)
        mask=labeled['contact_flags'][:,:2]&stance
        original=mask.copy();mask[1:]&=original[:-1];mask[:-1]&=original[1:];mask[[0,-1]]=False
        report['source_stance_contact_interior_speed_p95_m_s']={side:float(np.percentile(speed[mask[:,k],k],95))
            if mask[:,k].any() else None for k,side in enumerate(('left','right'))}
    atomic_json(dst/'fullbody_repair_report.json',report)
    return report


def run(args):
    baseline, output = args.baseline.resolve(), args.output.resolve()
    if baseline == output or baseline in output.parents:
        raise ValueError('Preserve baseline; use an independent output directory')
    rows = [{**r,'route':'native29_fullbody_geometry_repair'} for r in read_record(baseline/'inventory.json')]
    if args.clips:
        unknown = set(args.clips)-{r['clip'] for r in rows}
        if unknown:
            raise ValueError(f'Unknown clips: {unknown}')
        rows = [r for r in rows if r['clip'] in args.clips]
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'inventory.json',rows)
    atomic_json(output/'dataset_version.json',dict(dof=29,root_body='torso_link',baseline=str(baseline),method='fullbody_geometry_repair'))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(repair_clip,r,baseline,output,args.passes,args.maximum_frames) for r in rows]
        for task in as_completed(tasks):
            result=task.result();results.append(result)
            atomic_json(output/'fullbody_repair_summary.json',results)
            print(json.dumps(dict(done=len(results),total=len(rows),**result)),flush=True)
    if args.maximum_frames is None:
        subprocess.run([__import__('sys').executable,'-m','acp.quality.evaluate','--run',str(output),
                        '--workers',str(args.workers)],check=True)
