"""InstinctMJ 躯干基座 G1-29：原生 MJCF、等价 Drake URDF 与骨盆基座转换。"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import warnings

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from acp.data.assets import atomic_json
from acp.robot.joints import select_g1_joints
from acp.robot.model import digest

STEM = 'g1_29dof_torsobase_popsicle'


def numbers(values):
    return ' '.join(format(float(x), '.17g') for x in np.atleast_1d(values))


def rpy(quaternion):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_euler('xyz')


def prepare(resource, destination):
    """Keep native collision shapes; generate URDF from that SAME compiled MJCF.

    The repository's hand-authored URDF uses cylinders where the actual training
    MJCF uses capsules. It must not supply contact labels for the latter.
    """
    resource, destination = Path(resource).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = resource / 'xml' / (STEM + '.xml')
    root = ET.parse(source).getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((source.parent / compiler.get('meshdir', '')).resolve()))
    compiler.set('fusestatic', 'false')
    # Every geom gets an explicit stable name for scene/environment separation.
    for body in root.findall('.//body'):
        for i, geom in enumerate(body.findall('geom')):
            geom.set('name', geom.get('name', f"{body.get('name')}_geom_{i}"))
    for side in ('left', 'right'):
        for name, parent, offset in (
                (f'acp_{side}_foot', f'{side}_ankle_roll_link', [0, 0, -.035]),
                (f'acp_{side}_toe', f'{side}_ankle_roll_link', [.10, 0, -.035]),
                (f'{side}_hand_target', f'{side}_wrist_yaw_link', [.075, 0, 0])):
            parent_body = next(b for b in root.findall('.//body') if b.get('name') == parent)
            ET.SubElement(parent_body, 'body', name=name, pos=numbers(offset))
    xml = destination / (STEM + '.xml')
    ET.indent(root)
    ET.ElementTree(root).write(xml, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.nq != 36 or model.body(1).name != 'torso_link':
        raise ValueError('Expected the native InstinctMJ torso-root 29DoF robot')
    source_urdf = ET.parse(resource / 'urdf' / (STEM + '.urdf')).getroot()
    limits = {j.get('name'): j.find('limit') for j in source_urdf.findall('joint')}
    urdf = ET.Element('robot', name='instinct_g1_29')
    for b in range(1, model.nbody):
        name = model.body(b).name
        link = ET.SubElement(urdf, 'link', name=name)
        if model.body_mass[b] > 0:
            inertia = ET.SubElement(link, 'inertial')
            ET.SubElement(inertia, 'origin', xyz=numbers(model.body_ipos[b]), rpy=numbers(rpy(model.body_iquat[b])))
            ET.SubElement(inertia, 'mass', value=numbers(model.body_mass[b]))
            diagonal = model.body_inertia[b]
            ET.SubElement(inertia, 'inertia', ixx=numbers(diagonal[0]), iyy=numbers(diagonal[1]),
                          izz=numbers(diagonal[2]), ixy='0', ixz='0', iyz='0')
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            collision = ET.SubElement(link, 'collision', name=model.geom(g).name)
            ET.SubElement(collision, 'origin', xyz=numbers(model.geom_pos[g]), rpy=numbers(rpy(model.geom_quat[g])))
            geometry = ET.SubElement(collision, 'geometry')
            kind, size = model.geom_type[g], model.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ET.SubElement(geometry, 'capsule', radius=numbers(size[0]), length=numbers(2 * size[1]))
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                ET.SubElement(geometry, 'sphere', radius=numbers(size[0]))
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                ET.SubElement(geometry, 'cylinder', radius=numbers(size[0]), length=numbers(2 * size[1]))
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                ET.SubElement(geometry, 'box', size=numbers(2 * size))
            else:
                raise ValueError(f'Unsupported native collision shape {kind}; do not approximate it')
        parent = int(model.body_parentid[b])
        if parent == 0:
            continue  # Drake automatically supplies the floating torso base.
        j = int(model.body_jntadr[b])
        moving = bool(model.body_jntnum[b])
        if moving and (model.body_jntnum[b] != 1 or np.any(model.jnt_pos[j]) or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('URDF export requires one zero-anchor hinge per moving body')
        joint = ET.SubElement(urdf, 'joint', name=model.joint(j).name if moving else name + '_fixed',
                              type='revolute' if moving else 'fixed')
        ET.SubElement(joint, 'parent', link=model.body(parent).name)
        ET.SubElement(joint, 'child', link=name)
        ET.SubElement(joint, 'origin', xyz=numbers(model.body_pos[b]), rpy=numbers(rpy(model.body_quat[b])))
        if moving:
            ET.SubElement(joint, 'axis', xyz=numbers(model.jnt_axis[j]))
            limit = limits[model.joint(j).name]
            ET.SubElement(joint, 'limit', lower=numbers(model.jnt_range[j, 0]), upper=numbers(model.jnt_range[j, 1]),
                          effort=limit.get('effort'), velocity=limit.get('velocity'))
    path = xml.with_suffix('.urdf')
    ET.indent(urdf)
    ET.ElementTree(urdf).write(path, encoding='utf-8', xml_declaration=True)
    atomic_json(destination / 'provenance.json', dict(method='instinctmj_native_mjcf_29', dof=29,
        source=str(source), source_sha256=digest(source), target_xml_sha256=digest(xml),
        target_urdf_sha256=digest(path), target_joint_names=joint_names(model), root_body='torso_link',
        label_geometry='URDF exported from native MJCF; exact sphere/capsule/cylinder/box primitives',
        semantics='Existing hand-authored InstinctMJ URDF is used only for actuator velocity/effort limits'))
    return path


def joint_names(model):
    return [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]


def transfer(canonical, source_model, target_model):
    """Real source wrists, reversed waist signs, and world torso pose as new root."""
    source_names, target_names = joint_names(source_model), joint_names(target_model)
    q = select_g1_joints(canonical, source_names, target_names)
    root_body = source_model.body(1).name
    if root_body not in ('pelvis', 'torso_link'):
        raise ValueError(f'Unsupported source base {root_body}')
    if root_body == 'pelvis':
        for name in ('waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint'):
            q[:, 7 + target_names.index(name)] *= -1
    src, dst = mujoco.MjData(source_model), mujoco.MjData(target_model)
    protected = ['pelvis', 'torso_link', 'left_ankle_roll_link', 'right_ankle_roll_link',
                 'left_elbow_link', 'right_elbow_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']
    errors = []
    for i, frame in enumerate(canonical):
        src.qpos[:] = np.r_[frame[4:7], frame[:4], frame[7:]]
        mujoco.mj_forward(source_model, src)
        torso = source_model.body('torso_link').id
        q[i, :4], q[i, 4:7] = src.xquat[torso], src.xpos[torso]
        dst.qpos[:] = np.r_[q[i, 4:7], q[i, :4], q[i, 7:]]
        mujoco.mj_forward(target_model, dst)
        errors.append([np.linalg.norm(src.xpos[source_model.body(b).id] - dst.xpos[target_model.body(b).id]) for b in protected])
    errors = np.asarray(errors)
    report = dict(root_conversion=f'{root_body}_to_torso_link', joint_reordering='by_name',
                  waist_sign=-1 if root_body == 'pelvis' else 1, wrist_values_preserved_exactly=True,
                  links=protected, maximum_position_error_m=dict(zip(protected, errors.max(axis=0).tolist())),
                  mean_position_error_m=dict(zip(protected, errors.mean(axis=0).tolist())),
                  note='Remaining errors measure robot revision differences; no IK or root-height adjustment applied')
    return q, report


def target_mapping():
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    mapping = MotionDataConfig(data_format='mocap', robot_type='g1').resolved_joints_mapping.copy()
    for side in ('Left', 'Right'):
        mapping[side + 'Foot'] = f'{side.lower()}_ankle_roll_link'
        mapping[side + 'ToeBase'] = f'acp_{side.lower()}_toe'
        mapping[side + 'HandMiddle3'] = f'{side.lower()}_hand_target'
    return mapping
