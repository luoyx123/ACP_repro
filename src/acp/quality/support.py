"""用原模型几何核实支撑意图；保留已修复姿态与标签，独立复评。

几何意图仅用于离线筛查，不是物理受力或 Stage-2 hindsight 标签。
姿态不变时复用并核对 Drake/MuJoCo 几何；可选跨帧修正会重新标注和核验。
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from contextlib import redirect_stdout
import shutil
import subprocess
import sys
import time

import numpy as np

from acp.data.assets import PROJECT, atomic_json, read_record
from acp.robot.model import digest


def geometric_stance(velocity_stance, distances):
    """低速悬空脚不是支撑；原模型穿透仍保留意图，交给独立几何门槛处理。"""
    velocity_stance, distances = np.asarray(velocity_stance), np.asarray(distances)
    if velocity_stance.ndim != 2 or velocity_stance.shape[1] != 2 or distances.shape != velocity_stance.shape:
        raise ValueError('Need matching [frames,2] foot velocity flags and original surface distances')
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite original surface distances cannot establish support')
    return velocity_stance.astype(bool) & (distances <= .015)


def support_violation(quality, criteria):
    """Physical gate excess, used only to select between fully evaluated attempts."""
    value=max(0.,quality.get('max_joint_velocity_limit_ratio',1.)-1.)
    for metrics in quality['foot_metrics'].values():
        speed=metrics.get('contact_tangential_speed_p95_m_s')
        near=metrics.get('stance_near_surface_fraction')
        if speed is not None:value+=max(0.,speed/criteria['maximum_contact_foot_planar_speed_p95_m_s']-1.)
        if near is not None:value+=max(0.,criteria['minimum_stance_near_surface_fraction_per_foot']-near)
    return value


def prefer_temporal(candidate, baseline, criteria):
    """A lower sliding score cannot excuse any newly introduced failure."""
    safe=set(candidate['reasons'])<=set(baseline['reasons'])
    better=(candidate['status']=='offline_kinematic_candidate' or
            support_violation(candidate,criteria)<support_violation(baseline,criteria)-1e-6)
    return bool(safe and better)


def write_contact_audit(root):
    """Supplementary nine-body review; never equate geometric proximity with load.

    Keep the existing offline candidate list intact. A separate conservative
    subset excludes any body whose source-near samples lose more than 10%
    coverage, provided the aggregate source-near duration is at least 0.2s.
    This is a review heuristic, not ACP's training-success or contact-force gate.
    """
    root=Path(root);rows=read_record(root/'quality_all_clips.json')
    names=['left_foot','right_foot','left_knee','right_knee','left_hand','right_hand',
           'left_elbow','right_elbow','pelvis']
    counts=np.zeros((9,2),dtype=int);results=[];conservative=[]
    for row in rows:
        if 'frames' not in row:continue
        clip=root/'clips'/row['clip']
        with np.load(clip/'retargeted'/f"{row['clip']}_original.npz",allow_pickle=False) as z:
            if 'source_geometric_contact_intent' not in z:continue
            intent=z['source_geometric_contact_intent'].copy();fps=float(z['fps'])
            if z['source_geometric_contact_body_names'].tolist()!=names:
                raise ValueError('Source geometric contact body order mismatch')
        with np.load(clip/'validation/mujoco_geometry.npz',allow_pickle=False) as z:
            near=(z['distances']>=-.003)&(z['distances']<=.015)
        if intent.shape!=near.shape:raise ValueError('Contact audit time/body shape mismatch')
        source=intent.sum(axis=0);retained=(near&intent).sum(axis=0)
        counts[:,0]+=source;counts[:,1]+=retained
        fractions={n:float(b/a) if a else None for n,a,b in zip(names,source,retained)}
        review={n:fractions[n] for n,a in zip(names,source)
                if a>=int(np.ceil(.2*fps)) and fractions[n]<.9}
        results.append(dict(clip=row['clip'],source_intent_steps=dict(zip(names,source.tolist())),
            near_retention=fractions,review=review,offline_candidate=row.get('stage1_offline_eligible',False)))
        if row.get('stage1_offline_eligible') and not review:conservative.append(row['clip'])
    summary=dict(clips=len(results),expected_inputs=len(rows),review_clips=sum(bool(r['review']) for r in results),
        offline_candidates_with_contact_review=sum(bool(r['review']) and r['offline_candidate'] for r in results),
        conservative_candidates=len(conservative),minimum_aggregate_source_near_duration_s=.2,
        review_minimum_coverage=.9,
        bodies={n:dict(source_near_steps=int(a),target_near_steps=int(b),fraction=float(b/a) if a else None)
                for n,(a,b) in zip(names,counts)},
        note='Geometric-nearness retention only, not force/load or Stage-2 hindsight evidence. Extra review does not alter the versioned offline criteria.')
    atomic_json(root/'fullbody_contact_retention_by_clip.json',results)
    atomic_json(root/'fullbody_contact_retention_summary.json',summary)
    (root/'contact_review_clips.txt').write_text(''.join(r['clip']+'\n' for r in results if r['review']))
    (root/'conservative_candidate_clips.txt').write_text(''.join(c+'\n' for c in sorted(conservative)))
    return summary


def temporal_attempt(row, dst, original, values, manifest, criteria, smooth_weight):
    """Smooth the correction across frames, retaining the raw motion as fidelity target."""
    import mujoco
    from scipy.ndimage import gaussian_filter1d
    from acp.quality.fullbody import GeometryRepair
    from acp.quality.native import check
    from acp.quality.evaluate import assess
    start=time.monotonic()
    model=mujoco.MjModel.from_xml_path(manifest['scene'])
    problem=GeometryRepair(model,smooth_weight=smooth_weight)
    with np.load(original/'retargeted'/f"{row['clip']}_original.npz",allow_pickle=False) as z:
        nominal=z['qpos'].copy()
    q=values['qpos'].copy()
    if q.shape!=nominal.shape:raise ValueError('Temporal repair requires complete identical time grids')
    weights=gaussian_filter1d(values['source_stance_flags'].astype(float),2.,axis=0,mode='nearest')
    accepted=0
    for turn in range(2):
        previous_delta=np.zeros(len(problem.qids))
        for i in (range(len(q)) if turn%2==0 else range(len(q)-1,-1,-1)):
            candidate,_=problem.frame(nominal[i],previous_delta,stance_weights=weights[i])
            if problem.worst(candidate)>=-.003:
                q[i]=candidate;accepted+=1
            previous_delta=q[i,problem.qids]-nominal[i,problem.qids]
    root=dst/'attempts'/f'temporal_{smooth_weight:g}'
    artifact=root/'clips'/row['clip'];(artifact/'retargeted').mkdir(parents=True,exist_ok=True)
    motion=artifact/'retargeted'/f"{row['clip']}_original.npz"
    np.savez_compressed(motion,**{**values,'qpos':q})
    attempt_manifest={**manifest,'status':'temporal_pending_labels','temporal_smooth_weight':smooth_weight}
    atomic_json(artifact/'manifest.json',attempt_manifest)
    with (artifact/'labels.log').open('w') as log:
        subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','acp.data.adapters.holosoma',
            '--input',str(motion),'--target-urdf',manifest['target_urdf'],'--terrain-urdf',manifest['terrain_urdf'],
            '--source-clip',manifest['source'],'--output',str(artifact/'labeled'),'--fps',str(float(values['fps'])),
            '--input-method','native_g1_29'],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    with (artifact/'native.log').open('w') as log,redirect_stdout(log):
        check(Path(manifest['scene']),artifact/'labeled/motion.npz',artifact/'validation',motion)
    attempt_manifest['status']='end_to_end_temporal_attempt_not_training_admitted'
    atomic_json(artifact/'manifest.json',attempt_manifest)
    quality=assess(row,root,criteria)
    report=dict(smooth_weight=smooth_weight,passes=2,accepted_frame_updates=accepted,
        elapsed_s=time.monotonic()-start,status=quality['status'],reasons=quality['reasons'],
        score=support_violation(quality,criteria),artifact=str(artifact))
    atomic_json(artifact/'temporal_report.json',report)
    return report,quality,artifact,q


def refine_clip(row, baseline, source_baseline, output, criteria, temporal=False):
    from acp.quality.evaluate import assess
    src, original = baseline/'clips'/row['clip'], source_baseline/'clips'/row['clip']
    dst = output/'clips'/row['clip']
    manifest = read_record(src/'manifest.json')
    if manifest.get('dof') != 29 or not manifest.get('status','').startswith('end_to_end'):
        raise ValueError('Support refinement needs a completed native29 whole clip')
    retargeted = src/'retargeted'/f"{row['clip']}_original.npz"
    evidence = original/'native29_comparison/diagnostics.npz'
    signature = dict(code=digest(__file__), baseline_manifest=digest(src/'manifest.json'),
        baseline_motion=digest(retargeted), original_geometry=digest(evidence),
        original_target_motion=digest(original/'retargeted'/retargeted.name),
        target_scene=digest(manifest['scene']),target_urdf=digest(manifest['target_urdf']),
        original_comparison=digest(original/'native29_comparison/report.json'),
        labeled=digest(src/'labeled/motion.npz'), criteria=criteria,temporal=temporal,
        temporal_solver=digest(Path(__file__).with_name('fullbody.py')) if temporal else None)
    if (dst/'manifest.json').exists():
        previous = read_record(dst/'manifest.json')
        if previous.get('signature') == signature and previous.get('status','').startswith('end_to_end'):
            quality=assess(row,output,criteria)
            return dict(clip=row['clip'],quality_status=quality['status'],reasons=quality['reasons'])
        raise ValueError(f'Incomplete/different output already exists: {dst}')
    with np.load(retargeted,allow_pickle=False) as z:
        values={k:z[k].copy() for k in z.files}
    with np.load(evidence,allow_pickle=False) as z:
        times=z['timestamps'].copy()
        distances=z['source_body_environment_distances'].copy()
        witnesses=z['source_environment_witnesses'].copy()
    np.testing.assert_allclose(times,np.arange(len(values['qpos']))/float(values['fps']),atol=1e-10,rtol=0)
    if distances.shape != (len(times),9) or witnesses.shape != (len(times),9,3):
        raise ValueError('Original geometry evidence must cover all nine contact bodies')
    with np.load(src/'labeled/motion.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(times,z['timestamps'],atol=1e-10,rtol=0)
        q=np.c_[z['base_position_w'],z['base_quaternion_wxyz'],z['joint_positions']]
        np.testing.assert_allclose(q,values['qpos'],atol=1e-10,rtol=0)
    old=values.get('source_stance_flags_velocity_only',values['source_stance_flags']).copy()
    refined=geometric_stance(old,distances[:,:2])
    values.update(source_stance_flags=refined,source_stance_flags_velocity_only=old,
        source_body_environment_distances=distances,source_environment_witnesses=witnesses,
        source_geometric_contact_intent=distances<=.015,
        source_geometric_contact_body_names=np.array(['left_foot','right_foot','left_knee','right_knee',
            'left_hand','right_hand','left_elbow','right_elbow','pelvis']))
    dst.mkdir(parents=True,exist_ok=True)
    manifest={**manifest,'signature':signature,'baseline_artifact':str(src),'original_artifact':str(original),
        'route':row['route'],'status':'running','stage1_training_admitted':False,
        'quality_profile':criteria['version'],'source_support_is_force_evidence':False}
    atomic_json(dst/'manifest.json',manifest)
    (dst/'retargeted').mkdir(exist_ok=True)
    np.savez_compressed(dst/'retargeted'/retargeted.name,**values)
    for name in ('labeled','validation'):
        shutil.copytree(src/name,dst/name,dirs_exist_ok=True)
    for name in ('source29.npz','source_rate_motion.npz','model_conversion_report.json',
                 'fullbody_repair_report.json','solver_status.json'):
        if (src/name).exists():
            saved='baseline_'+name if name in ('fullbody_repair_report.json','solver_status.json') else name
            shutil.copy2(src/name,dst/saved)
    # Geometry and poses are identical. Recompute only the changed source mask's
    # diagnostics from the independently validated native distances.
    with np.load(dst/'validation/mujoco_geometry.npz',allow_pickle=False) as z:
        target_distances=z['distances'].copy()
    native_path=dst/'validation/mujoco_geometry_report.json'
    native=read_record(native_path)
    native['source_stance_diagnostics']={side:dict(stance_steps=int(mask.sum()),
        stance_near_surface_fraction=float((target_distances[mask,i]<=.015).mean()) if mask.any() else None,
        stance_gap_p95_m=float(np.percentile(target_distances[mask,i],95)) if mask.any() else None)
        for i,side in enumerate(('left','right')) for mask in [refined[:,i]]}
    native['support_refinement']=dict(original_geometry=str(evidence),pose_and_geometry_unchanged=True)
    atomic_json(native_path,native)
    report=dict(clip=row['clip'],frames=len(times),pose_exactly_preserved=True,
        velocity_only_stance_steps=old.sum(axis=0).tolist(),geometric_stance_steps=refined.sum(axis=0).tolist(),
        stationary_airborne_steps_removed=(old&~refined).sum(axis=0).tolist(),signature=signature,
        note='Source geometric intent, not force evidence. Keep all old flags for comparison; same numeric thresholds, explicit new contact-speed metric.')
    atomic_json(dst/'support_refinement_report.json',report)
    manifest['status']='end_to_end_native29_geometric_support_refined_not_training_admitted'
    atomic_json(dst/'manifest.json',manifest)
    quality=assess(row,output,criteria)
    if temporal:
        atomic_json(dst/'quality_before_temporal.json',quality)
        attempts=[];selected=None
        for smooth_weight in (3.,8.):
            if quality['status']=='offline_kinematic_candidate':break
            allowed={'left_foot_sliding','right_foot_sliding','left_stance_gap','right_stance_gap',
                     'joint_velocity_limits'}
            if not set(quality['reasons'])<=allowed:break
            result,candidate,artifact,q=temporal_attempt(row,dst,original,values,manifest,criteria,smooth_weight)
            result['selected']=prefer_temporal(candidate,quality,criteria)
            attempts.append(result)
            if result['selected']:
                for name in ('retargeted','labeled','validation'):
                    shutil.copytree(artifact/name,dst/name,dirs_exist_ok=True)
                values['qpos']=q;selected=smooth_weight
                manifest['temporal_selected_attempt']=str(artifact)
                manifest['status']='end_to_end_native29_temporal_support_repaired_not_training_admitted'
                atomic_json(dst/'manifest.json',manifest)
                quality=assess(row,output,criteria)
        report.update(pose_exactly_preserved=selected is None,temporal_attempts=attempts,
            selected_smooth_weight=selected)
        atomic_json(dst/'support_refinement_report.json',report)
    return dict(clip=row['clip'],quality_status=quality['status'],reasons=quality['reasons'])


def run(args):
    baseline,source,output=args.baseline.resolve(),args.source_baseline.resolve(),args.output.resolve()
    if output in (baseline,source) or baseline in output.parents or source in output.parents:
        raise ValueError('Use an independent output; preserve input trajectories and original geometry')
    rows=[{**r,'route':'native29_fullbody_geometric_support'} for r in read_record(baseline/'inventory.json')]
    if args.clips:
        unknown=set(args.clips)-{r['clip'] for r in rows}
        if unknown:raise ValueError(f'Unknown clips: {unknown}')
        rows=[r for r in rows if r['clip'] in args.clips]
    criteria=read_record(PROJECT/'configs/data_quality_native29.json')
    output.mkdir(parents=True,exist_ok=True)
    for path,value in [('inventory.json',rows),('quality_criteria.json',criteria),
        ('dataset_version.json',dict(dof=29,root_body='torso_link',baseline=str(baseline),
            original_baseline=str(source),method='native29_geometry_and_geometric_support'))]:
        atomic_json(output/path,value)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(refine_clip,r,baseline,source,output,criteria,args.temporal):r for r in rows}
        for task in as_completed(tasks):
            row=tasks[task]
            try:result=task.result()
            except Exception as exc:
                result=dict(clip=row['clip'],status='failed',error=f'{type(exc).__name__}: {exc}')
                dst=output/'clips'/row['clip'];dst.mkdir(parents=True,exist_ok=True)
                path=dst/'manifest.json'
                manifest=read_record(path) if path.exists() else {}
                atomic_json(path,{**manifest,**result,'stage1_training_admitted':False})
            results.append(result)
            atomic_json(output/'support_refinement_summary.json',results)
            atomic_json(output/'progress.json',dict(expected=len(rows),processed=len(results),
                failed=sum(r.get('status')=='failed' for r in results),
                candidates=sum(r.get('quality_status')=='offline_kinematic_candidate' for r in results)))
    subprocess.run([sys.executable,'-m','acp.quality.evaluate','--run',str(output),
        '--workers',str(args.workers)],check=True)
    if any(r.get('status')=='failed' for r in results):
        raise SystemExit('Some support refinements failed; see support_refinement_summary.json')
