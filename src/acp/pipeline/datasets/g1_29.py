"""三类现成 G1-29 数据 → InstinctMJ 躯干基座 29DoF → 独立接触标签与质量报告。"""
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

from acp.data.assets import atomic_json, machine, read_record, run_path
from acp.data.motions import read_lafan_csv, resample_drake
from acp.robot.instinct import STEM, joint_names, prepare, target_mapping, transfer
from acp.robot.model import compile_robot, digest, make_scene, save_robot_xml

PROJECT = Path(__file__).resolve().parents[4]


def source_points(model, q, mapping):
    from holosoma_retargeting.config_types.data_type import MOCAP_DEMO_JOINTS
    points = np.zeros((len(q), len(MOCAP_DEMO_JOINTS), 3))
    data = mujoco.MjData(model)
    for i, frame in enumerate(q):
        data.qpos[:] = np.r_[frame[4:7], frame[:4], frame[7:]]
        mujoco.mj_forward(model, data)
        for name, body in mapping.items():
            offset = np.zeros(3)
            if body.startswith('acp_') and body.endswith('_toe'):
                body = body.removeprefix('acp_').removesuffix('_toe') + '_ankle_roll_link'
                offset = np.array([.10, 0, -.035])
            elif body.endswith('_hand_target'):
                body = body.removesuffix('_hand_target') + '_wrist_yaw_link'
                offset = np.array([.075, 0, 0])
            # PHP's native compiler fuses this zero-offset fixed contour into pelvis.
            if body == 'pelvis_contour_link' and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
                body = 'pelvis'
            b = model.body(body).id
            points[i, MOCAP_DEMO_JOINTS.index(name)] = data.xpos[b] + data.xmat[b].reshape(3, 3) @ offset
    return points


def copy_terrain(source, target):
    source = Path(source)
    root = ET.parse(source).getroot()
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str((source.parent / mesh.get('filename')).resolve()))
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)


def convert(row, run, label_python, fps, force=False):
    from holosoma_retargeting.config_types.data_type import MOCAP_DEMO_JOINTS
    from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity
    from acp.quality.native import check
    from acp.quality.evaluate import assess
    run = Path(run)
    clip = run / 'clips' / row['clip']
    clip.mkdir(parents=True, exist_ok=True)
    urdf = run / 'assets' / (STEM + '.urdf')
    model_path = run / 'assets/source_robot.xml'
    signature = dict(source=digest(row['source']), target=digest(urdf), target_xml=digest(urdf.with_suffix('.xml')),
                     source_model=digest(model_path), terrain=digest(row['terrain_urdf']), fps=fps,
                     code={n: digest(PROJECT / n) for n in ('src/acp/pipeline/datasets/g1_29.py',
                         'src/acp/robot/instinct.py', 'src/acp/robot/joints.py', 'src/acp/data/motions.py',
                         'src/acp/data/scene.py', 'src/acp/data/adapters/holosoma.py', 'src/acp/quality/native.py')})
    path = clip / 'manifest.json'
    if path.exists() and not force:
        previous = read_record(path)
        if (previous.get('signature') == signature and previous.get('status', '').startswith('end_to_end')
                and all((clip / p).exists() for p in ('labeled/motion.npz', 'validation/mujoco_geometry.npz', 'quality/report.json'))):
            return dict(clip=row['clip'], status='resumed')
    manifest = dict(status='running', route=row['route'], dof=29, root_body='torso_link',
                    source=row['source'], source_format=row['source_format'], signature=signature,
                    source_motion_npz=str(clip / 'source29.npz'), source_robot_xml=str(model_path),
                    target_urdf=str(urdf), scene=str(clip / 'scene.xml'), terrain_urdf=str(clip / 'terrain.urdf'),
                    target_mapping=target_mapping(), stage1_training_admitted=False,
                    source_fk_aliases={'pelvis_contour_link': 'pelvis if fused by source compiler; zero-offset fixed link'},
                    lineage='Original 29DoF source; does not inherit 23DoF arm repairs or contact labels')
    atomic_json(path, manifest)
    try:
        source_model = mujoco.MjModel.from_xml_path(str(model_path))
        target_model = mujoco.MjModel.from_xml_path(str(urdf.with_suffix('.xml')))
        names, target_names = joint_names(source_model), joint_names(target_model)
        if row['dataset'] == 'lafan1':
            from acp.data.motions import LAFAN_G1_JOINT_NAMES
            original, source_fps = read_lafan_csv(row['source']), 30.
            original = original[:, [*range(7), *(7 + LAFAN_G1_JOINT_NAMES.index(n) for n in names)]]
        elif row['dataset'] == 'php':
            from acp.data.adapters.php import read_motion
            original, source_fps = read_motion(row['source'], names)
        else:
            with np.load(row['source'], allow_pickle=False) as raw:
                original, source_fps = raw['qpos'].copy(), float(raw['fps'])
        times, sampled, _ = resample_drake(original, source_fps, fps)
        np.savez_compressed(clip / 'source29.npz', qpos=original, fps=source_fps, joint_names=names,
                            qpos_order='wxyz,xyz,joints', root_body=source_model.body(1).name)
        native_raw, raw_report = transfer(original, source_model, target_model)
        q, report = transfer(sampled, source_model, target_model)
        atomic_json(clip / 'model_conversion_report.json', dict(source_rate=raw_report, output_rate=report,
            source_model_sha256=digest(model_path), target_model_sha256=digest(urdf.with_suffix('.xml'))))
        np.savez_compressed(clip / 'source_rate_motion.npz', qpos=native_raw, fps=source_fps,
                            joint_names=target_names, qpos_order='wxyz,xyz,joints', root_body='torso_link')
        copy_terrain(row['terrain_urdf'], clip / 'terrain.urdf')
        make_scene(urdf.with_suffix('.xml'), clip / 'terrain.urdf', clip / 'scene.xml')
        points = source_points(source_model, sampled, manifest['target_mapping'])
        stance = extract_foot_sticking_sequence_velocity(points, MOCAP_DEMO_JOINTS,
                     ['LeftToeBase', 'RightToeBase'], velocity_threshold=.3 / fps)
        retargeted = clip / 'retargeted' / (row['clip'] + '_original.npz')
        retargeted.parent.mkdir(exist_ok=True)
        np.savez_compressed(retargeted, qpos=np.c_[q[:, 4:7], q[:, :4], q[:, 7:]], fps=fps,
                            joint_names=target_names, root_body='torso_link', robot_fk_keypoints=points,
                            source_keypoints_kind='original29_fk_with_explicit_native_hand_toe_offsets',
                            source_stance_flags=np.array([[s['L_Toe'], s['R_Toe']] for s in stance]))
        manifest.update(source_frames=len(original), source_fps=source_fps, output_frames=len(q))
        atomic_json(path, manifest)
        with (clip / 'labels.log').open('w') as log:
            subprocess.run([label_python, '-m', 'acp.data.adapters.holosoma', '--input', str(retargeted),
                '--target-urdf', str(urdf), '--terrain-urdf', str(clip / 'terrain.urdf'), '--source-clip', row['source'],
                '--output', str(clip / 'labeled'), '--fps', str(fps), '--input-method', 'native_g1_29'],
                cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (clip / 'native.log').open('w') as log, redirect_stdout(log):
            check(clip / 'scene.xml', clip / 'labeled/motion.npz', clip / 'validation', retargeted)
        manifest['status'] = 'end_to_end_data_pipeline_completed_not_training_admitted'
        atomic_json(path, manifest)
        assess(row, run, read_record(PROJECT / 'configs/data_quality.json'))
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        atomic_json(path, manifest)
    return dict(clip=row['clip'], status=manifest['status'], error=manifest.get('error'))


def main():
    cfg = machine()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=('omni', 'lafan1', 'php'))
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--robot-resource', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--fps', type=float, default=50.)
    p.add_argument('--label-python', default=str(PROJECT / '.venv/bin/python'))
    p.add_argument('--clips', nargs='+')
    p.add_argument('--force', action='store_true')
    p.add_argument('--all-variants', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    if args.workers < 1 or not np.isfinite(args.fps) or args.fps <= 0:
        p.error('Positive workers/fps required')
    baseline = run_path(dict(omni='omni_asap23', lafan1='lafan_asap23', php='php_asap23')[args.dataset])
    run = args.output.resolve()
    if run == baseline.resolve() or baseline.resolve() in run.parents:
        p.error('29DoF outputs must be independent of the 23DoF dataset')
    dataset = (args.dataset_root or Path(cfg[{'omni': 'omni_dataset', 'lafan1': 'lafan_dataset',
                                            'php': 'php_dataset'}[args.dataset]])).resolve()
    if run == dataset or dataset in run.parents:
        p.error('Keep output outside the original dataset')
    assets = run / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    prepare(args.robot_resource, assets)
    if args.dataset == 'omni':
        source = dataset / 'models/g1/g1_29dof_spherehand.urdf'
        save_robot_xml(compile_robot(source), assets / 'source_robot.xml')
        rows = []
        for path in sorted((dataset / 'robot-terrain').glob('*.npz')):
            with np.load(path, allow_pickle=False) as z:
                rows.append(dict(clip=path.stem, source=str(path), source_frames=len(z['qpos']),
                                 source_fps=float(z['fps']), has_human_joints='human_joints' in z))
    elif args.dataset == 'lafan1':
        source = dataset / 'robot_description/g1/g1_29dof_rev_1_0.urdf'
        save_robot_xml(compile_robot(source), assets / 'source_robot.xml')
        terrain = assets / 'flat_ground.urdf'
        terrain.write_text('<robot name="flat_ground"><link name="world"/></robot>\n')
        rows = [dict(clip=f.stem, source=str(f), source_frames=len(read_lafan_csv(f)), source_fps=30.,
                     has_human_joints=False, terrain_urdf=str(terrain), scope='LAFAN1 G1 on flat ground',
                     motion_group=f.stem.split('_subject')[0]) for f in sorted((dataset / 'g1').glob('*.csv'))]
    else:
        from acp.data.adapters.php import inventory, terrain_variants, write_terrain
        source = Path(cfg['php_root']) / 'motion_matching/motion_matching/assets/robots/unitree_description/mjcf/g1.xml'
        tree = ET.parse(source)
        compiler = tree.getroot().find('compiler')
        compiler.set('meshdir', str((source.parent / compiler.get('meshdir', '')).resolve()))
        tree.write(assets / 'source_robot.xml', encoding='utf-8', xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(assets / 'source_robot.xml'))
        rows = inventory(dataset, joint_names(model), None if args.all_variants else (0,))
        for row in rows:
            terrain = run / 'clips' / row['clip'] / 'terrain.urdf'
            terrain.parent.mkdir(parents=True, exist_ok=True)
            write_terrain(terrain_variants(row['terrain_source'])[:, row['terrain_variant'], :], terrain)
            row['terrain_urdf'] = str(terrain)
    if args.clips:
        rows = [r for r in rows if r['clip'] in args.clips]
        if set(args.clips) - {r['clip'] for r in rows}:
            p.error('Unknown clip names')
    if not rows:
        p.error('No matching original 29DoF inputs')
    for row in rows:
        if args.dataset == 'omni':
            base, scale = row['clip'].split('_z_scale_')
            terrain = dataset / 'models/terrain' / base / f'multi_boxes_z_scale_{scale}.urdf'
        else:
            terrain = Path(row['terrain_urdf'])
        row.update(dataset=args.dataset, dof=29, root_body='torso_link', terrain_urdf=str(terrain),
                   route=f'{args.dataset}_original29_to_instinct29',
                   source_format={'omni': 'omni_g1_29_wxyz_xyz', 'lafan1': 'lafan_g1_csv_xyz_xyzw',
                                  'php': 'php_g1_training_npz_xyz_wxyz'}[args.dataset])
    atomic_json(run / 'inventory.json', rows)
    atomic_json(assets / 'source_provenance.json', dict(source=str(source), source_sha256=digest(source),
        cached_model_sha256=digest(assets / 'source_robot.xml'), dataset_root=str(dataset),
        note='Read directly from original datasets; no dependency on 23DoF trajectory artifacts'))
    if args.audit_only:
        print(json.dumps(dict(inputs=len(rows), frames=sum(r['source_frames'] for r in rows))))
        return
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert, row, str(run), args.label_python, args.fps, args.force) for row in rows]
        for f in as_completed(futures):
            result = f.result()
            results.append(result)
            atomic_json(run / 'conversion_results.json', results)
            print(json.dumps(dict(done=len(results), total=len(rows), **result)), flush=True)
    subprocess.run([sys.executable, '-m', 'acp.quality.evaluate', '--run', str(run), '--workers', str(args.workers)], check=True)
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
