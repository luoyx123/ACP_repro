"""Mixamo world-space references; preserve authored height and support semantics."""
import numpy as np
from scipy.spatial.transform import Rotation


def target_mapping():
    mapping = {'Hips': 'pelvis', 'Spine': 'torso_link'}
    for side in ('Left', 'Right'):
        s = side.lower()
        mapping.update({side+'UpLeg': s+'_hip_pitch_link', side+'Leg': s+'_knee_link',
                        side+'Foot': s+'_ankle_roll_link',
                        side+'ToeBase': s+'_ankle_roll_sphere_5_link',
                        side+'Arm': s+'_shoulder_roll_link', side+'ForeArm': s+'_elbow_link',
                        side+'Hand': s+'_sphere_hand_link'})
    return mapping


def transform_reference(positions, names, scale):
    """FBX import already gives metres/Z-up. Only rotate heading and centre XY.

    A high toe can mean a chair, ladder, jump, or inversion. It never defines
    the ground. In particular do not subtract a trajectory-wide minimum toe Z.
    """
    p = np.asarray(positions, dtype=float).copy()
    if p.ndim != 3 or p.shape[1:] != (len(names), 3) or not np.isfinite(p).all():
        raise ValueError('Invalid Mixamo joint positions')
    p = p @ np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]).T * scale
    p[:, :, :2] -= p[0, names.index('Hips'), :2].copy()
    return p


def stable_segments(mask, minimum_samples):
    """Discard brief stance detections, without bridging a real swing phase."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for begin, end in zip(boundaries[::2], boundaries[1::2]):
        if end-begin >= minimum_samples:
            out[begin:end] = True
    return out


def ground_stance(positions, names, times, support_boxes=()):
    """Ground support only; elevated ladder/seat support needs explicit geometry.

    Require both adjacent samples to be in stance before a previous-frame XY
    constraint is applied. This leaves touchdown free to find its new anchor.
    """
    fps = 1 / np.median(np.diff(times))
    flags = []
    for side in ('Left', 'Right'):
        toe = positions[:, names.index(side+'ToeBase')]
        speed = np.linalg.norm(np.gradient(toe, times, axis=0), axis=1)
        near = np.abs(toe[:, 2]) < .045
        for box in support_boxes:
            rpy = box.get('rpy', [0., 0., 0.])
            if not np.allclose(rpy[:2], 0):
                raise ValueError('Stance support boxes must have horizontal top surfaces')
            local = Rotation.from_euler('xyz', rpy).inv().apply(toe-np.asarray(box['position']))
            half = np.asarray(box['size'])/2
            near |= ((np.abs(local[:, :2]) <= half[:2]+.015).all(axis=1)
                     & (np.abs(local[:, 2]-half[2]) < .045))
        candidate = (speed < .10) & near
        stable = stable_segments(candidate, max(2, int(np.ceil(.08*fps))))
        flags.append(stable & np.r_[False, stable[:-1]])
    return np.column_stack(flags)


def initial_quaternion(positions, names):
    """Use the complete pelvis basis, including lying/inverted initial poses."""
    p = positions[0]
    y = p[names.index('LeftUpLeg')]-p[names.index('RightUpLeg')]
    y /= np.linalg.norm(y)
    up = p[names.index('Spine')]-p[names.index('Hips')]
    z = up-y*np.dot(up, y)
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    xyzw = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
    return xyzw[[3, 0, 1, 2]]
