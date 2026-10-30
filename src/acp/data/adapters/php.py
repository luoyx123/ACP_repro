"""PHP training-package adapter: named joints and oriented box variants."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def read_motion(path, joint_names):
    with np.load(path, allow_pickle=False) as z:
        names = z['joint_names'].tolist()
        q = np.array(z['joint_pos'], dtype=float)
        fps = float(np.asarray(z['fps']).reshape(-1)[0])
    if (len(names) != len(set(names)) or set(names) != set(joint_names)
            or q.ndim != 2 or q.shape[1] != 7 + len(names) or len(q) < 2
            or not np.isfinite(q).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError(f'Invalid PHP motion or joint-name set: {path}')
    if np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1)) > .002:
        raise ValueError('PHP root quaternion must be unit WXYZ in columns 3:7')
    canonical = np.c_[q[:, 3:7], q[:, :3], q[:, [7 + names.index(n) for n in joint_names]]]
    return canonical, fps


def terrain_variants(path):
    boxes = np.load(path, allow_pickle=False)
    if boxes.ndim != 3 or boxes.shape[2] != 10 or boxes.shape[1] < 1 or not np.isfinite(boxes).all():
        raise ValueError(f'Expected terrain [boxes, variants, 10]: {path}')
    if len(boxes) and (np.any(boxes[:, :, 7:] <= 0)
                      or np.max(np.abs(np.linalg.norm(boxes[:, :, 3:7], axis=-1) - 1)) > .002):
        raise ValueError('Box sizes must be positive, quaternions WXYZ and normalized')
    return boxes


def write_terrain(boxes, output):
    """Rows are [XYZ, WXYZ, full size XYZ]; URDF sizes are full extents too."""
    root = ET.Element('robot', name='php_terrain')
    ET.SubElement(root, 'link', name='world')
    for index, row in enumerate(boxes):
        name = f'box_{index:03d}'
        link = ET.SubElement(root, 'link', name=name)
        for kind in ('collision', 'visual'):
            geom = ET.SubElement(ET.SubElement(link, kind), 'geometry')
            ET.SubElement(geom, 'box', size=' '.join(format(float(v), '.17g') for v in row[7:]))
        joint = ET.SubElement(root, 'joint', name=name + '_fixed', type='fixed')
        ET.SubElement(joint, 'parent', link='world')
        ET.SubElement(joint, 'child', link=name)
        rpy = Rotation.from_quat(row[[4, 5, 6, 3]]).as_euler('xyz')
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, row[:3])), rpy=' '.join(map(str, rpy)))
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)


def inventory(dataset, source_joint_names, variants=(0,)):
    rows = []
    for source in sorted(Path(dataset).glob('*/*_motion.npz')):
        terrain = source.with_name(source.name.removesuffix('_motion.npz') + '_terrain.npy')
        q, fps = read_motion(source, source_joint_names)
        boxes = terrain_variants(terrain)
        group = source.parent.name + '/' + source.name
        key = hashlib.sha256(group.encode()).hexdigest()[:12]
        category = 'climb' if 'climb' in source.parent.name else 'locomotion'
        for variant in (range(boxes.shape[1]) if variants is None else variants):
            if variant < 0 or variant >= boxes.shape[1]:
                raise ValueError(f'Unknown variant {variant} for {terrain}')
            rows.append(dict(clip=f'php_{category}_{key}_v{variant:02d}', source=str(source.resolve()),
                terrain_source=str(terrain.resolve()), terrain_variant=int(variant), terrain_variants=boxes.shape[1],
                terrain_boxes=len(boxes), source_frames=len(q), source_fps=fps, has_human_joints=False,
                motion_group=group, route='php_asap_joint_subset', scope='PHP robot-terrain training packages',
                terrain_note='Variant 0 is a supplied variant, not a certified original scene'))
    if not rows:
        raise ValueError(f'No paired PHP training packages in {dataset}')
    return rows
