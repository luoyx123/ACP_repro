"""Reproducible URDF cleanup and exact static scene triangles for PhysX/Warp."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def transform(origin):
    result = np.eye(4)
    if origin is not None:
        result[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        result[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
    return result


def prepare_urdf(source, destination):
    """Only remove massless fixed leaves; move their collision shapes to parent."""
    root = ET.parse(source).getroot()
    removed = []
    while True:
        links = {x.get('name'): x for x in root.findall('link')}
        parent_names = {x.find('parent').get('link') for x in root.findall('joint')}
        candidates = [j for j in root.findall('joint') if j.get('type') == 'fixed'
                      and j.find('child').get('link') not in parent_names
                      and links[j.find('child').get('link')].find('inertial') is None]
        if not candidates: break
        for joint in candidates:
            child = links[joint.find('child').get('link')]
            parent = links[joint.find('parent').get('link')]
            for kind in ('collision', 'visual'):
                for shape in child.findall(kind):
                    copied = deepcopy(shape)
                    t = transform(joint.find('origin'))@transform(shape.find('origin'))
                    old = copied.find('origin')
                    if old is not None: copied.remove(old)
                    ET.SubElement(copied, 'origin', xyz=' '.join(map(str, t[:3, 3])),
                                  rpy=' '.join(map(str, Rotation.from_matrix(t[:3, :3]).as_euler('xyz'))))
                    parent.append(copied)
            removed.append(child.get('name'));root.remove(child);root.remove(joint)
    extra = root.find('mujoco')
    if extra is not None: root.remove(extra)
    for mesh in root.findall('.//mesh'):
        p = Path(mesh.get('filename'))
        mesh.set('filename', str(p if p.is_absolute() else (Path(source).parent/p).resolve()))
    destination = Path(destination);destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root);ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    return removed


def body_groups(urdf, semantic_links):
    groups = {name: i for i, name in enumerate(semantic_links)}
    joints = ET.parse(urdf).getroot().findall('joint')
    while True:
        size = len(groups)
        for j in joints:
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if j.get('type') == 'fixed' and parent in groups and child not in groups:
                groups[child] = groups[parent]
        if len(groups) == size: return groups


def load_robot(urdf):
    import mujoco
    root = ET.parse(urdf).getroot()
    extension = root.find('mujoco')
    if extension is not None: root.remove(extension)
    extension = ET.SubElement(root, 'mujoco')
    ET.SubElement(extension, 'compiler', fusestatic='false', discardvisual='false',
                  balanceinertia='true', strippath='false')
    ET.SubElement(root, 'link', name='world')
    joint = ET.SubElement(root, 'joint', name='m2_floating_base', type='floating')
    ET.SubElement(joint, 'parent', link='world');ET.SubElement(joint, 'child', link='pelvis')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def geom_mesh(model, data, index):
    import mujoco
    import trimesh
    kind = model.geom_type[index]; size = model.geom_size[index]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[index]
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        vertices = model.mesh_vert[va:va+vn].astype(float)
        faces = model.mesh_face[fa:fa+fn].copy()
    else:
        if kind == mujoco.mjtGeom.mjGEOM_BOX: shape = trimesh.creation.box(extents=2*size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE: shape = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER: shape = trimesh.creation.cylinder(radius=size[0], height=2*size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE: shape = trimesh.creation.capsule(radius=size[0], height=2*size[1])
        else: raise ValueError(f'Unsupported geometry {kind}')
        vertices, faces = np.asarray(shape.vertices), np.asarray(shape.faces)
    return vertices@data.geom_xmat[index].reshape(3, 3).T+data.geom_xpos[index], faces


def terrain_triangles(scene):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene));data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    parts = []
    for i in range(model.ngeom):
        body = model.body(model.geom_bodyid[i]).name
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE: continue
        if model.geom_bodyid[i] == 0 or body.startswith('multi_boxes_'):
            parts.append(geom_mesh(model, data, i))
    return parts


def combine_meshes(parts):
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v);faces.append(f+offset);offset += len(v)
    return (np.concatenate(vertices), np.concatenate(faces)) if vertices else (np.empty((0, 3)), np.empty((0, 3), dtype=int))
