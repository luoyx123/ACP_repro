"""Blender entry point: evaluate imported FBX bones in world metres, not Euler curves."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

NAMES = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head'] + [
    side + part for side in ('Left', 'Right') for part in
    ('UpLeg', 'Leg', 'Foot', 'ToeBase', 'Shoulder', 'Arm', 'ForeArm', 'Hand')]


def extract(source, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source))
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE' and o.animation_data and o.animation_data.action]
    if len(armatures) != 1:
        raise ValueError(f'Expected one animated armature, got {len(armatures)}')
    arm = armatures[0]
    by_name = {b.name.split(':')[-1]: b.name for b in arm.pose.bones}
    if set(NAMES) - by_name.keys():
        raise ValueError(f'Missing bones: {set(NAMES) - by_name.keys()}')
    start, end = arm.animation_data.action.frame_range
    frames = start + np.arange(int(np.floor(end-start+1e-4))+1)
    points, tails = [], []
    for frame in frames:
        integer=int(np.floor(frame))
        bpy.context.scene.frame_set(integer, subframe=float(frame-integer))
        points.append([tuple(arm.matrix_world @ arm.pose.bones[by_name[n]].head) for n in NAMES])
        tails.append([tuple(arm.matrix_world @ arm.pose.bones[by_name[n]].tail) for n in NAMES])
    rest = np.asarray([tuple(arm.matrix_world @ arm.data.bones[by_name[n]].head_local) for n in NAMES])
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    values = np.asarray(points)
    if len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Insufficient or nonfinite animation')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, positions=values, tails=np.asarray(tails), rest_positions=rest,
                        joint_names=np.asarray(NAMES), fps=fps, frame_start=start, frame_end=end)
    return {'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'frames': len(values), 'fps': fps, 'action': arm.animation_data.action.name,
            'blender_version': bpy.app.version_string, 'units': 'world metres after Blender FBX import',
            'joint_names': NAMES}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    results = []
    for job in json.loads(args.jobs.read_text()):
        output = Path(job['output'])
        try:
            report = extract(Path(job['source']), output)
            report['status'] = 'extracted'
        except Exception as exc:
            report = {'source': job['source'], 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        results.append(report)
        print('ACP_EXTRACT', json.dumps(report), flush=True)
