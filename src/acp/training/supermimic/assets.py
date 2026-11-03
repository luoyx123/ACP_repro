"""Small, hash-verified, portable Stage-1 data packages for MuJoCo/SuperMimic."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from acp.data.assets import PROJECT, atomic_json, registry, resolve_path, run_path
from acp.data.playback import resolve_clip


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'package.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(root/name) != expected:
            raise ValueError(f'Portable package checksum mismatch: {name}')
    return manifest


def prepare(output, config_path=PROJECT/'configs/stage1.json', task='walk_box'):
    """Copy selected motions and every mesh dependency; never edit source artifacts."""
    import mujoco
    config = json.loads(Path(config_path).read_text())
    robot = json.loads((PROJECT/'configs/m2.json').read_text())
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    names = config['tasks'][task].get('members', [task])
    clips = []
    for name in names:
        spec = config['tasks'][name]
        dataset = registry()['runs'][spec['run']]['dataset']
        if dataset in config['excluded_datasets']:
            raise ValueError(f'Excluded dataset: {dataset}')
        info = resolve_clip(run_path(spec['run']), spec['clip'])
        if not info.get('stage1_offline_eligible') and not spec.get('experimental_data_override'):
            raise ValueError(f'{name}: explicit experimental data override required')
        folder = output/name
        folder.mkdir()
        shutil.copy2(info['motion'], folder/'motion.npz')
        tree = ET.parse(info['scene'])
        root = tree.getroot()
        compiler = root.find('compiler')
        meshdir = Path(compiler.get('meshdir', '')) if compiler is not None else Path()
        texturedir = Path(compiler.get('texturedir', '')) if compiler is not None else Path()
        for tag, directory in [('mesh', meshdir), ('texture', texturedir)]:
            for node in root.findall(f'./asset/{tag}'):
                filename = node.get('file')
                if not filename:
                    continue
                path = Path(filename)
                source = resolve_path(path) if path.is_absolute() else Path(info['scene']).parent/directory/path
                target = output/'meshes'/(sha(source)+source.suffix.lower())
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
                node.set('file', '../meshes/'+target.name)
        if compiler is not None:
            compiler.attrib.pop('meshdir', None)
            compiler.attrib.pop('texturedir', None)
        tree.write(folder/'scene.xml', encoding='utf-8', xml_declaration=True)
        model = compile_model(folder/'scene.xml', robot)
        clips.append(dict(name=name, specification=deepcopy(spec),
                          motion=f'{name}/motion.npz', scene=f'{name}/scene.xml',
                          source_motion_sha256=sha(info['motion']), source_scene_sha256=sha(info['scene']),
                          nq=model.nq, mass_kg=float(model.body_mass.sum())))
    manifest = dict(schema_version=1, task=task, config=config, robot_config=robot, clips=clips,
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                    files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    atomic_json(output/'package.json', manifest)
    verify(output)
    return manifest


def compile_model(scene, cfg):
    """Identical named robot/terrain geometry, explicit PD and no robot self collision."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nq != 30 or model.nv != 29:
        raise ValueError('Expected a free base and exactly 23 hinge joints')
    model.opt.timestep = cfg['physics_dt']
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.iterations = 10
    model.opt.ls_iterations = 10
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.dof_damping[:] = 0
    model.dof_frictionloss[:] = 0
    for name, armature in zip(cfg['joint_names'], cfg['armature']):
        model.dof_armature[model.jnt_dofadr[model.joint(name).id]] = armature
    pelvis = model.body('pelvis').id
    robot_bodies = set([pelvis])
    for b in range(pelvis+1, model.nbody):
        if model.body_parentid[b] in robot_bodies:
            robot_bodies.add(b)
    for g, body in enumerate(model.geom_bodyid):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            model.geom_group[g] = 5
            continue
        is_robot = body in robot_bodies
        model.geom_contype[g] = 1 if is_robot else 2
        model.geom_conaffinity[g] = (3 if cfg['self_collisions'] else 2) if is_robot else 1
        model.geom_group[g] = 1 if is_robot else 0
        # Let the sampled robot coefficient govern robot/terrain contacts rather
        # than MuJoCo's default max() masking every coefficient below the floor's.
        model.geom_priority[g] = 1 if is_robot else 0
        model.geom_friction[g, 0] = cfg.get('friction', 1.)
        model.geom_margin[g] = cfg.get('contact_offset', .002)
    return model
