"""PHP G1-29→23：名称投影、箱体场景、接触标签与成对质量检查。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / 'src'))
import mujoco
import numpy as np
from acp.data.assets import atomic_json, machine, read_record, run_path
from acp.robot.joints import project_joint_subset
from acp.data.motions import resample_drake
from acp.data.adapters.php import inventory, read_motion, terrain_variants, write_terrain
from acp.robot.model import compile_robot, digest, make_scene, prepare, save_robot_xml
from acp.robot.transfer import fk_keypoints


def convert(row, run, label_python, fps):
    from holosoma_retargeting.config_types.data_type import MotionDataConfig, MOCAP_DEMO_JOINTS
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity
    from acp.quality.native import check
    from acp.quality.compare import compare
    run = Path(run)
    clip = run / 'clips' / row['clip']
    clip.mkdir(parents=True, exist_ok=True)
    assets = run / 'assets'
    urdf = assets / 'g1_asap23_spherehand.urdf'
    signature = dict(source=digest(row['source']), terrain=digest(row['terrain_source']),
        variant=row['terrain_variant'], target=digest(urdf), source_model=digest(assets/'php_robot.xml'),
        reference=digest(row['reference_urdf']), fps=fps,
        code={p:digest(PROJECT/p) for p in ('scripts/batch_php_asap23.py','src/acp/data/php.py',
            'scripts/prepare_asap23.py','src/acp/data/asap.py','src/acp/data/holosoma.py')})
    path = clip / 'manifest.json'
    if path.is_file():
        previous = read_record(path)
        if (previous.get('signature') == signature and previous.get('status','').startswith('end_to_end')
                and all((clip/p).is_file() for p in ('labeled/motion.npz','validation/mujoco_geometry.npz','comparison/report.json'))):
            return dict(clip=row['clip'], status='resumed')
    scene = clip / 'scene.xml'
    terrain = Path(row['terrain_urdf'])
    manifest = dict(status='running', route=row['route'], signature=signature, source=row['source'],
        source_format='php_g1_training_npz_xyz_wxyz', source_motion_npz=row['reference_motion_npz'],
        source_robot_xml=str(assets/'reference_robot.xml'), original_robot_xml=str(assets/'php_robot.xml'),
        reference_urdf=row['reference_urdf'], target_urdf=str(urdf), scene=str(scene),
        terrain_urdf=str(terrain), terrain_variant=row['terrain_variant'], motion_group=row['motion_group'],
        stage1_training_admitted=False, terrain_note=row['terrain_note'])
    atomic_json(path, manifest)
    try:
        reference = mujoco.MjModel.from_xml_path(str(assets/'reference_robot.xml'))
        target = mujoco.MjModel.from_xml_path(str(urdf.with_suffix('.xml')))
        names = [reference.joint(i).name for i in range(1, reference.njnt)]
        target_names = [target.joint(i).name for i in range(1, target.njnt)]
        original, source_fps = read_motion(row['source'], names)
        times, sampled, _ = resample_drake(original, source_fps, fps)
        np.savez_compressed(row['reference_motion_npz'], qpos=original, fps=source_fps, joint_names=names,
            qpos_order='wxyz,xyz,joints')
        raw_subset = project_joint_subset(original, names, target_names)
        q = project_joint_subset(sampled, names, target_names)
        np.savez_compressed(clip/'source_rate_subset.npz', qpos=raw_subset, fps=source_fps,
            joint_names=target_names, qpos_order='wxyz,xyz,joints')
        boxes = terrain_variants(row['terrain_source'])[:,row['terrain_variant'],:]
        write_terrain(boxes, terrain)
        make_scene(urdf.with_suffix('.xml'), terrain, scene)
        mapping = MotionDataConfig(data_format='mocap', robot_type='g1').resolved_joints_mapping
        points = fk_keypoints(reference, sampled, MOCAP_DEMO_JOINTS, mapping)
        stance = extract_foot_sticking_sequence_velocity(points, MOCAP_DEMO_JOINTS,
            ['LeftToeBase','RightToeBase'], velocity_threshold=.3/fps)
        retargeted = clip/'retargeted'/f"{row['clip']}_original.npz"
        retargeted.parent.mkdir(exist_ok=True)
        np.savez_compressed(retargeted, qpos=np.c_[q[:,4:7],q[:,:4],q[:,7:]], fps=fps,
            joint_names=target_names, robot_fk_keypoints=points,
            source_keypoints_kind='php_joint_values_on_unified_spherehand29',
            source_stance_flags=np.array([[s['L_Toe'],s['R_Toe']] for s in stance]))
        # Validate the source format against the actual PHP MJCF, independently of target geometry.
        original_model = mujoco.MjModel.from_xml_path(str(assets/'php_robot.xml'))
        data = mujoco.MjData(original_model)
        with np.load(row['source'], allow_pickle=False) as z:
            source_names = z['joint_names'].tolist()
            body_names = z['body_names'].tolist()
            saved = z['body_pos_w']
            q_native = z['joint_pos']
            selected = [(i,original_model.body(n).id) for i,n in enumerate(body_names)
                        if mujoco.mj_name2id(original_model,mujoco.mjtObj.mjOBJ_BODY,n)>=0]
            model_names = [original_model.joint(i).name for i in range(1,original_model.njnt)]
            errors = []
            for f in sorted(set((0,len(q_native)//2,len(q_native)-1))):
                data.qpos[:7] = q_native[f,:7]
                data.qpos[7:] = q_native[f,[7+source_names.index(n) for n in model_names]]
                mujoco.mj_forward(original_model,data)
                errors.append(float(np.max(np.linalg.norm(data.xpos[[b for _,b in selected]]-
                    saved[f,[i for i,_ in selected]],axis=1))))
        if max(errors) > 1e-5:
            raise ValueError(f'PHP saved-body FK mismatch: {max(errors)} m')
        atomic_json(clip/'source_format_validation.json', dict(sampled_frames=3,common_bodies=len(selected),
            maximum_saved_body_position_error_m=max(errors), original_model_sha256=digest(assets/'php_robot.xml'),
            note='Checks input layout/model at first, middle and last frame; not a full-motion geometry admission'))
        manifest.update(target_mapping=mapping, source_frames=len(original), source_fps=source_fps, output_frames=len(times))
        atomic_json(path, manifest)
        with (clip/'labels.log').open('w') as log:
            subprocess.run([label_python,'-m','acp.data.adapters.holosoma','--input',str(retargeted),
                '--target-urdf',str(urdf),'--terrain-urdf',str(terrain),'--source-clip',row['source'],
                '--output',str(clip/'labeled'),'--fps',str(fps),'--input-method','asap_joint_subset'],
                cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (clip/'native.log').open('w') as log, redirect_stdout(log):
            check(scene,clip/'labeled/motion.npz',clip/'validation',retargeted)
        compare(row,run,None)
        manifest['status']='end_to_end_data_pipeline_completed_not_training_admitted'
    except Exception as exc:
        manifest.update(status='failed',failed_stage='php_asap_joint_subset_pipeline',error=f'{type(exc).__name__}: {exc}')
    atomic_json(path,manifest)
    return dict(clip=row['clip'],status=manifest['status'],error=manifest.get('error'))


def main():
    cfg=machine()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=Path(cfg['php_dataset']))
    p.add_argument('--php-root',type=Path,default=Path(cfg['php_root']))
    p.add_argument('--reference-urdf',type=Path,default=Path(cfg['omni_dataset'])/'models/g1/g1_29dof_spherehand.urdf')
    p.add_argument('--asap-root',type=Path,default=Path(cfg['asap_root']))
    p.add_argument('--output',type=Path,default=run_path('php_asap23'))
    p.add_argument('--label-python',type=Path,default=PROJECT/cfg['label_python'])
    p.add_argument('--fps',type=float,default=50.)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--clips',nargs='+')
    p.add_argument('--all-variants',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args()
    if args.workers<1 or not np.isfinite(args.fps) or args.fps<=0:p.error('Positive fps/workers required')
    run=args.output.resolve()
    if run==args.dataset_root.resolve() or args.dataset_root.resolve() in run.parents:p.error('Keep output outside raw dataset')
    assets=run/'assets';assets.mkdir(parents=True,exist_ok=True)
    reference=compile_robot(args.reference_urdf)
    names=[reference.joint(i).name for i in range(1,reference.njnt)]
    rows=inventory(args.dataset_root,names,None if args.all_variants else (0,))
    if args.clips:
        rows=[r for r in rows if r['clip'] in args.clips]
        if set(args.clips)-{r['clip'] for r in rows}:p.error('Unknown clip IDs; run --audit-only first')
    for row in rows:
        row.update(reference_urdf=str(args.reference_urdf.resolve()),
            reference_motion_npz=str(run/'clips'/row['clip']/'source29.npz'),
            terrain_urdf=str(run/'clips'/row['clip']/'terrain.urdf'))
    atomic_json(run/'inventory.json',rows)
    atomic_json(run/'source_inventory_summary.json',dict(inputs=len(rows),motion_groups=len({r['motion_group'] for r in rows}),
        variants='all' if args.all_variants else [0], source_frames=sum(r['source_frames'] for r in rows),
        source_fps=sorted({r['source_fps'] for r in rows}), category_counts={c:sum(c in r['clip'] for r in rows) for c in ('climb','locomotion')}))
    if args.audit_only:
        print((run/'source_inventory_summary.json').read_text());return
    prepare(args.reference_urdf,args.asap_root,assets)
    # mj_saveLastXML uses MuJoCo's last compiled XML structure. Compile this
    # model immediately before exporting, after prepare() compiled the target.
    save_robot_xml(compile_robot(args.reference_urdf),assets/'reference_robot.xml')
    source=args.php_root/'motion_matching/motion_matching/assets/robots/unitree_description/mjcf/g1.xml'
    xml=ET.parse(source)
    compiler=xml.getroot().find('compiler')
    compiler.set('meshdir',str((source.parent/compiler.get('meshdir','')).resolve()))
    xml.write(assets/'php_robot.xml',encoding='utf-8',xml_declaration=True)
    atomic_json(assets/'php_provenance.json',dict(original_mjcf=str(source),original_mjcf_sha256=digest(source),
        cached_mjcf_sha256=digest(assets/'php_robot.xml'),reference_urdf_sha256=digest(args.reference_urdf),
        target_definition='23DoF on unified Omni spherehand geometry; source PHP MJCF validated separately',
        terrain_encoding='[boxes,variants,XYZ+WXYZ+fullXYZsize]', variant_note='No supplied variant is certified original'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(convert,r,str(run),str(args.label_python.absolute()),args.fps) for r in rows]
        for f in as_completed(jobs):
            result=f.result();results.append(result)
            atomic_json(run/'conversion_results.json',results)
            print(json.dumps(dict(done=len(results),total=len(rows),**result)),flush=True)
    subprocess.run([sys.executable,str(PROJECT/'src/acp/quality/evaluate.py'),'--run',str(run),'--workers',str(args.workers)],check=True)
    if any(r['status']=='failed' for r in results):raise SystemExit(1)

if __name__=='__main__':main()
