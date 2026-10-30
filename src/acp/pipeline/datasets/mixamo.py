"""Raw Mixamo FBX -> user Holosoma interaction mesh -> spherehand G1-23 -> labels.

Resumable extraction and retargeting; source files are read only. No synthetic
stairs, chair, ladder or object geometry is inferred from an animation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / 'src'))
from acp.data.motions import fixed_rate_times
from acp.data.adapters.mixamo import target_mapping, transform_reference, ground_stance, initial_quaternion, stable_segments
from acp.robot.model import prepare, digest, make_scene


@contextmanager
def numerical_recovery(events):
    """Retry the SAME cone problem with relaxed numerical tolerance, no dropped constraints."""
    import cvxpy as cp
    original=cp.Problem.solve
    def solve(problem,*args,**kwargs):
        try:return original(problem,*args,**kwargs)
        except cp.error.SolverError as exc:
            if kwargs.get('solver')!=cp.CLARABEL:raise
            retry={**kwargs,'max_iter':300,'tol_gap_abs':1e-6,'tol_gap_rel':1e-6,'tol_feas':1e-6}
            backend='CLARABEL relaxed tolerance'
            try:value=original(problem,*args,**retry)
            except cp.error.SolverError:
                backend='SCS same cone constraints'
                value=original(problem,*args,solver=cp.SCS,eps=1e-5,max_iters=15000,verbose=False)
            residual=max((float(np.max(np.abs(c.violation()))) for c in problem.constraints),default=0.)
            if problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE) and residual>1e-4:
                backend='SCS same cone constraints refined'
                value=original(problem,*args,solver=cp.SCS,eps=1e-7,max_iters=60000,verbose=False)
                residual=max((float(np.max(np.abs(c.violation()))) for c in problem.constraints),default=0.)
            if problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE) and residual>1e-4:
                backend='SCS same cone constraints high precision'
                value=original(problem,*args,solver=cp.SCS,eps=1e-8,max_iters=200000,scale=1.,verbose=False)
                residual=max((float(np.max(np.abs(c.violation()))) for c in problem.constraints),default=0.)
            events.append({'trigger':str(exc),'backend':backend,'status':problem.status,'maximum_linearized_constraint_violation':residual})
            if problem.status not in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE) or not np.isfinite(residual) or residual>1e-4:
                raise cp.error.SolverError('Numerical retry failed residual acceptance')
            return value
    cp.Problem.solve=solve
    try:yield
    finally:cp.Problem.solve=original


def convert(row, run, label_python, fps):
    from holosoma_retargeting.config_types.data_type import MOCAP_DEMO_JOINTS
    from acp.pipeline.contact_solver import MixamoContactRetargeter
    from holosoma_retargeting.config_types.retargeter import SelfCollisionConfig
    from acp.quality.native import check
    run = Path(run)
    artifact = run / 'clips' / row['clip']
    artifact.mkdir(parents=True, exist_ok=True)
    assets = run / 'assets'
    urdf = assets / 'g1_asap23_spherehand.urdf'
    terrain = assets/'flat_ground.urdf'
    scene_path = urdf.with_suffix('.xml')
    support = row.get('support_scene')
    if support:
        terrain = artifact/'support.urdf'
        write_support_urdf(support,terrain)
        scene_path = artifact/'support_scene.xml'
        make_scene(assets/'robot_only.xml',terrain,scene_path)
    mapping = target_mapping()
    signature = {'source':digest(row['source']), 'extract':digest(row['extracted']),
                 'runner':digest(__file__), 'target':digest(urdf), 'scene':digest(urdf.with_suffix('.xml')),
                 'solver':digest(Path(__import__('holosoma_retargeting').__file__).parent/'src/interaction_mesh_retargeter.py'),
                 'labeler':digest(PROJECT/'src/acp/data/holosoma.py'),
                 'preprocessor':digest(PROJECT/'src/acp/data/mixamo.py'), 'fps':fps,
                 'contact_retargeter':digest(PROJECT/'scripts/mixamo_contact_retargeter.py'),
                 'scene_builder':digest(PROJECT/'scripts/prepare_asap23.py'),
                 'semantic':row.get('semantic'), 'support_scene':support}
    manifest_path = artifact/'manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get('signature')==signature and previous.get('status','').startswith('end_to_end'):
            return {'clip':row['clip'],'status':'resumed'}
    manifest = {'source':row['source'], 'signature':signature, 'route':row['route'], 'status':'running',
                'target_urdf':str(urdf), 'scene':str(scene_path),
                'terrain_urdf':str(terrain), 'target_mapping':mapping,
                'support_scene':support,
                'stage1_training_admitted':False, 'source_format':'mixamo_fbx_world_bones',
                'semantic':row.get('semantic'),
                'limitations':[('explicit test support geometry; original scene unverified' if support else 'flat-ground hypothesis; external support geometry unavailable'),
                               'geometric contacts, not force hindsight', 'no dynamics validation']}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        with np.load(row['extracted']) as z:
            raw, rest, names, source_fps = z['positions'],z['rest_positions'],z['joint_names'].tolist(),float(z['fps'])
        times = fixed_rate_times(len(raw),source_fps,fps)
        if len(times)<2:
            manifest.update(status='excluded_too_short',error='Fewer than two output samples; reference pose, not usable trajectory')
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
            return {'clip':row['clip'],'status':manifest['status']}
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        data.qpos[3]=1
        mujoco.mj_forward(model,data)
        leg_robot = sum(np.linalg.norm(data.xpos[model.body(a).id]-data.xpos[model.body(b).id]) for a,b in
                        [('left_hip_pitch_link','left_knee_link'),('left_knee_link','left_ankle_roll_link')])
        leg_human = sum(np.linalg.norm(rest[names.index(a)]-rest[names.index(b)]) for a,b in
                        [('LeftUpLeg','LeftLeg'),('LeftLeg','LeftFoot')])
        scale=leg_robot/leg_human
        if not .25<scale<2.0:
            raise ValueError(f'Unexpected human-to-robot limb scale {scale}')
        # Blender already converts FBX centimetres/Y-up to metres/Z-up.
        # Rotate Mixamo's forward -Y to robot +X; never apply an extra cm scale.
        raw=transform_reference(raw,names,scale)
        points=np.zeros((len(times),len(MOCAP_DEMO_JOINTS),3))
        for name in mapping:
            source_name=name
            for axis in range(3):
                points[:,MOCAP_DEMO_JOINTS.index(name),axis]=np.interp(times,np.arange(len(raw))/source_fps,raw[:,names.index(source_name),axis])
        # Only mapped entries are used; absent finger points are not fabricated.
        left,right=ground_stance(points,MOCAP_DEMO_JOINTS,times,(support or {}).get('boxes',())).T
        sequence=[{'L_Toe':bool(l),'R_Toe':bool(r)} for l,r in zip(left,right)]
        constants=SimpleNamespace(ROBOT_URDF_FILE=str(urdf),ROBOT_DOF=23,
           OBJECT_NAME='multi_boxes' if support else 'ground',SCENE_XML_FILE=str(scene_path),
           FOOT_STICKING_LINKS=[mapping[s+'ToeBase'] for s in ('Left','Right')],DEMO_JOINTS=MOCAP_DEMO_JOINTS,
           JOINTS_MAPPING=mapping,MANUAL_LB={},MANUAL_UB={},MANUAL_COST={},NOMINAL_TRACKING_INDICES=[])
        hips=points[:,MOCAP_DEMO_JOINTS.index('Hips'),:2]
        lo,hi=hips.min(axis=0)-1,hips.max(axis=0)+1
        ground=np.array([[x,y,0.] for x in np.linspace(lo[0],hi[0],6) for y in np.linspace(lo[1],hi[1],6)])
        if support:
            from scipy.spatial.transform import Rotation
            surfaces=[]
            for box in support['boxes']:
                half=np.asarray(box['size'])/2
                surface=np.array([[x,y,z] for x in (-half[0],0,half[0])
                                  for y in (-half[1],0,half[1]) for z in (-half[2],half[2])])
                surfaces.extend(Rotation.from_euler('xyz',box.get('rpy',[0,0,0])).apply(surface)+box['position'])
            ground=np.vstack([ground,surfaces])
        initial=np.zeros(30);initial[3]=1
        initial[:3]=points[0,MOCAP_DEMO_JOINTS.index('Hips')]
        for side in ('left','right'):
            for joint,value in [('hip_pitch',-.15),('knee',.3),('ankle_pitch',-.15),('elbow',.3)]:
                initial[model.joint(side+'_'+joint+'_joint').qposadr[0]]=value
        # Initialize yaw from the lateral hip axis, retaining animated heading.
        initial[3:7]=initial_quaternion(points,MOCAP_DEMO_JOINTS)
        if support and support.get('fit_from_manifest'):
            prior=Path(support['fit_from_manifest']).parent/'retargeted'/(row['clip']+'_original.npz')
            if prior.is_file():
                with np.load(prior) as z:
                    warm=z['qpos'][0].copy()
                if warm.shape==initial.shape and np.isfinite(warm).all():initial=warm
        poses=np.tile([0.,0,0,1,0,0,0],(len(points),1))
        output=artifact/'retargeted'/(row['clip']+'_original.npz');output.parent.mkdir(exist_ok=True)
        events=[]
        avoid_bodies=['torso_link','pelvis_contour_link']+[
            s+'_'+part+'_link' for s in ('left','right') for part in ('hip_pitch','hip_roll','knee')]
        avoid_pairs=[(s+'_sphere_hand_link',body) for s in ('left','right') for body in avoid_bodies]
        with (artifact/'retarget.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),numerical_recovery(events):
            solver=MixamoContactRetargeter(constants,None,activate_foot_support=True,
                       activate_foot_sticking=True,penetration_tolerance=.001,smooth_weight=.3,
                       step_size=(support or {}).get('step_size',.2),
                       self_collision=SelfCollisionConfig(enable=True,pairs=avoid_pairs,tolerance=.003))
            solver.surface_contact_masks={}
            solver.surface_contact_weight=(support or {}).get('contact_weight',200.)
            if support:
                targets = [('LeftHand','left_sphere_hand_link'),('RightHand','right_sphere_hand_link')] if support['support_kind']=='ladder' else [('Hips','pelvis_contour_link')]
                for source_name,body in targets:
                    p=points[:,MOCAP_DEMO_JOINTS.index(source_name)]
                    mask=stable_segments(np.linalg.norm(np.gradient(p,times,axis=0),axis=1)<.25,max(2,int(np.ceil(.12*fps))))
                    if support['support_kind']=='seat':mask[:len(mask)//2]=False
                    solver.surface_contact_masks[body]=mask
            # Actual evaluated source and final geometry are saved independently;
            # a solver return does not grant training eligibility.
            solver.retarget_motion(points,poses,poses.copy(),ground,ground,sequence,
                                   q_a_init=initial,dest_res_path=str(output),fps=fps)
        (artifact/'numerical_recovery.json').write_text(json.dumps(events,indent=2)+'\n')
        with np.load(output) as z:
            result=z['qpos'].copy()
        if result.shape!=(len(points),30) or not np.isfinite(result).all():
            raise ValueError('Nonfinite or incomplete retargeter output')
        # Conic numerical tolerances can leave microradian bound violations.
        # Remove only sub-0.1mrad residuals, then run geometry and velocity checks
        # on the exact exported values. Larger violations remain failures.
        maximum_bound_projection=0.
        for joint in range(model.njnt):
            if model.jnt_limited[joint]:
                index=model.jnt_qposadr[joint]
                bounded=np.clip(result[:,index],*model.jnt_range[joint])
                residual=float(np.max(np.abs(result[:,index]-bounded)))
                if residual>1e-4:raise ValueError(f'Joint bound residual {residual} exceeds export tolerance')
                maximum_bound_projection=max(maximum_bound_projection,residual)
                result[:,index]=bounded
        np.savez_compressed(output,qpos=result,fps=fps,human_joints=points,
                            keypoint_names=np.asarray(MOCAP_DEMO_JOINTS),source_stance_flags=np.c_[left,right],
                            surface_contact_body_names=np.asarray(list(solver.surface_contact_masks)),
                            surface_contact_masks=np.column_stack(list(solver.surface_contact_masks.values())) if solver.surface_contact_masks else np.zeros((len(points),0),dtype=bool))
        np.savez_compressed(artifact/'human_reference.npz',positions=points,joint_names=np.asarray(MOCAP_DEMO_JOINTS),
                            timestamps=times,fps=fps,mapped_joint_names=np.asarray(list(mapping)))
        manifest.update(source_frames=len(raw),source_fps=source_fps,output_frames=len(points),
                        scale=scale,global_floor_shift_m=0.,
                        maximum_joint_bound_projection_rad=maximum_bound_projection,
                        self_collision_avoidance={'body_pairs':avoid_pairs,'minimum_distance_m':.003},
                        stance_definition='ground toe speed <0.10m/s, abs(height)<0.045m, duration>=0.08s; previous sample also in stance; hard toe XY locking 1mm/frame',
                        ground_policy='preserve authored world Z=0; never ground elevated toes')
        with (artifact/'labels.log').open('w') as log:
            subprocess.run([label_python,'-m','acp.data.adapters.holosoma','--input',str(output),
               '--target-urdf',str(urdf),'--terrain-urdf',str(terrain),
               '--source-clip',row['source'],'--output',str(artifact/'labeled'),'--fps',str(fps)],
               cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (artifact/'native.log').open('w') as log,redirect_stdout(log):
            check(scene_path,artifact/'labeled/motion.npz',artifact/'validation',output)
        manifest['status']='end_to_end_data_pipeline_completed_not_training_admitted'
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if 'events' in locals():
            (artifact/'numerical_recovery.json').write_text(json.dumps(events,indent=2)+'\n')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if manifest['status'].startswith('end_to_end'):
        from acp.quality.evaluate import assess
        try:
            assess(row,run,json.loads((PROJECT/'configs/data_quality.json').read_text()))
        except Exception as exc:
            manifest['assessment_error']=f'{type(exc).__name__}: {exc}'
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
            return {'clip':row['clip'],'status':'assessment_failed','error':manifest['assessment_error']}
    return {'clip':row['clip'],'status':manifest['status'],'error':manifest.get('error')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=Path('/home/lyx/train/dataset/Mixamo_dataset'))
    p.add_argument('--output',type=Path,default=PROJECT/'outputs/mixamo_asap23_fixed')
    p.add_argument('--extracted-root',type=Path,help='Reuse evaluated FBX skeletons from a previous batch')
    p.add_argument('--semantic-map',type=Path,default=PROJECT/'configs/mixamo_semantics.json')
    p.add_argument('--support-scenes',type=Path,help='Explicit per-clip support box configurations')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--fps',type=float,default=50)
    p.add_argument('--clips',nargs='+')
    p.add_argument('--extract-only',action='store_true')
    args=p.parse_args()
    if args.workers<1 or not np.isfinite(args.fps) or args.fps<=0:p.error('Positive workers/fps required')
    run=args.output.resolve();assets=run/'assets';assets.mkdir(parents=True,exist_ok=True)
    if args.dataset_root.resolve() == run or args.dataset_root.resolve() in run.parents:
        p.error('Keep outputs outside the source dataset')
    urdf=prepare('/home/lyx/train/retarget/dataset/OmniRetarget_Dataset/models/g1/g1_29dof_spherehand.urdf',
                 '/home/lyx/train/ASAP',assets)
    (assets/'robot_only.xml').write_bytes(urdf.with_suffix('.xml').read_bytes())
    terrain=assets/'flat_ground.urdf';terrain.write_text('<robot name="ground"><link name="world"/></robot>\n')
    # The native retargeter reads the sibling XML when OBJECT_NAME=ground.
    scene=assets/'ground_scene.xml';make_scene(urdf.with_suffix('.xml'),terrain,scene)
    urdf.with_suffix('.xml').write_bytes(scene.read_bytes())
    sources=sorted((args.dataset_root/'animation').glob('*.fbx'))
    if args.clips:sources=[s for s in sources if s.stem in args.clips]
    if not sources or (args.clips and set(args.clips)-{s.stem for s in sources}):p.error('No matching animations')
    semantics=json.loads(args.semantic_map.read_text()) if args.semantic_map.is_file() else {}
    supports=json.loads(args.support_scenes.read_text()) if args.support_scenes else {}
    extracted=(args.extracted_root or run/'extracted').absolute()
    rows=[{'clip':s.stem,'source':str(s.resolve()),'extracted':str(extracted/(s.stem+'.npz')),
           'route':'mixamo_holosoma_direct23','has_human_joints':True,'scope':'Mixamo FBX flat-ground hypothesis',
           'motion_group':s.stem,'semantic':semantics.get(s.stem),'support_scene':supports.get(s.stem)} for s in sources]
    def priority(row):
        semantic=row['semantic'] or {}
        return (0 if semantic.get('name')=='Sitting_6' or semantic.get('support_kind')=='ladder'
                else 1 if semantic.get('priority') else 2,row['clip'])
    rows.sort(key=priority)
    jobs=[]
    for r in rows:
        out=Path(r['extracted']);meta=out.with_suffix('.json')
        if not out.is_file() or not meta.is_file() or json.loads(meta.read_text()).get('sha256')!=digest(r['source']):
            jobs.append({'source':r['source'],'output':r['extracted']})
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers,8)) as pool:
            tasks=[]
            for index in range(min(args.workers,8)):
                part=jobs[index::min(args.workers,8)]
                if part:
                    jobfile=run/f'extract_jobs_{index}.json';jobfile.write_text(json.dumps(part))
                    tasks.append(pool.submit(extract_batch,str(jobfile),str(run/f'extract_{index}.log')))
            for f in as_completed(tasks):f.result()
    for r in rows:
        meta=json.loads(Path(r['extracted']).with_suffix('.json').read_text())
        r.update(source_frames=meta.get('frames',0),source_fps=meta.get('fps',0),extraction_status=meta['status'])
    (run/'inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
    if args.extract_only:return
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    results=[]
    for r in rows:
        if r['extraction_status']!='extracted':
            results.append({'clip':r['clip'],'status':'extraction_failed'})
    (run/'conversion_results.json').write_text(json.dumps(results,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(convert_safe,r,str(run),str(PROJECT/'.venv/bin/python'),args.fps) for r in rows if r['extraction_status']=='extracted']
        for f in as_completed(futures):
            result=f.result();results.append(result)
            (run/'conversion_results.json').write_text(json.dumps(results,indent=2)+'\n')
            # Keep completed clips viewable while the full batch continues.
            index=[]
            for report in sorted((run/'clips').glob('*/quality/report.json')):
                q=json.loads(report.read_text())
                index.append({'clip':q['clip'],'route':q['route'],'status':q['status'],
                    'motion':q['labeled_motion_relative'],'stage1_offline_eligible':False,'reasons':q['reasons']})
            (run/'artifact_index.json').write_text(json.dumps(index,indent=2)+'\n')
            from acp.management.status import status
            status(run,args.dataset_root)
            print(json.dumps({'done':len(results),'total':len(rows),**result}),flush=True)
    (run/'conversion_results.json').write_text(json.dumps(results,indent=2)+'\n')
    subprocess.run([sys.executable,str(PROJECT/'src/acp/quality/evaluate.py'),'--run',str(run),'--workers',str(args.workers)],check=True)
    from acp.management.status import status
    status(run,args.dataset_root)


def extract_batch(jobfile, logpath):
    with open(logpath,'w') as log:
        subprocess.run(['blender','-b','-t','1','--python',str(PROJECT/'src/acp/pipeline/blender.py'),
                        '--','--jobs',jobfile],stdout=log,stderr=subprocess.STDOUT,check=True)


def convert_safe(row, run, label_python, fps):
    """A scene setup/assessment failure must not cancel unrelated animations."""
    try:
        return convert(row,run,label_python,fps)
    except Exception as exc:
        artifact=Path(run)/'clips'/row['clip'];artifact.mkdir(parents=True,exist_ok=True)
        path=artifact/'manifest.json'
        manifest=json.loads(path.read_text()) if path.is_file() else {'source':row['source'],'route':row['route']}
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        return {'clip':row['clip'],'status':'failed','error':manifest['error']}


def write_support_urdf(support, output):
    root=ET.Element('robot',name='mixamo_support')
    ET.SubElement(root,'link',name='world')
    boxes=support.get('boxes',[])
    if not boxes:raise ValueError('Support scene needs at least one box')
    for index,box in enumerate(boxes):
        size=np.asarray(box['size'],dtype=float)
        xyz=np.asarray(box['position'],dtype=float)
        rpy=np.asarray(box.get('rpy',[0.,0.,0.]),dtype=float)
        if (size.shape!=(3,) or xyz.shape!=(3,) or rpy.shape!=(3,)
                or not np.isfinite(np.r_[size,xyz,rpy]).all() or np.any(size<=0)):
            raise ValueError('Invalid support box')
        name=f'support_{index}'
        link=ET.SubElement(root,'link',name=name)
        geometry=ET.SubElement(ET.SubElement(link,'collision'),'geometry')
        ET.SubElement(geometry,'box',size=' '.join(map(str,size)))
        joint=ET.SubElement(root,'joint',name=name+'_fixed',type='fixed')
        ET.SubElement(joint,'parent',link='world');ET.SubElement(joint,'child',link=name)
        ET.SubElement(joint,'origin',xyz=' '.join(map(str,xyz)),rpy=' '.join(map(str,rpy)))
    ET.indent(root)
    ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)


if __name__=='__main__':main()
