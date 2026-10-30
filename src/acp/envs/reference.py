"""Simulator-independent named reference motion queries and action conventions."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def name_indices(source, target):
    source, target = list(source), list(target)
    if len(set(source)) != len(source) or len(set(target)) != len(target) or set(source) != set(target):
        raise ValueError('Joint/body name sets must be unique and identical')
    return [source.index(name) for name in target]


class ReferenceMotion:
    def __init__(self, path, joint_names, body_names):
        with np.load(path, allow_pickle=False) as z:
            self.times = z['timestamps'].copy()
            self.joint_names = list(joint_names)
            ji = name_indices(z['joint_names'].tolist(), joint_names)
            bi = name_indices(z['body_names'].tolist(), body_names)
            self.values = {k: z[k].copy() for k in (
                'base_position_w', 'base_quaternion_wxyz', 'base_linear_velocity_w',
                'base_angular_velocity_w', 'joint_positions', 'joint_velocities',
                'body_landmarks_w', 'contact_flags', 'target_positions_body_yaw', 'target_occupancy')}
        if len(self.times) < 2 or not np.all(np.diff(self.times) > 0):
            raise ValueError('Reference timestamps must be strictly increasing')
        for k, values in self.values.items():
            if len(values) != len(self.times) or not np.isfinite(values).all():
                raise ValueError(f'Invalid reference field: {k}')
            if k.startswith('joint_'): self.values[k] = values[:, ji]
            if k in ('body_landmarks_w', 'contact_flags', 'target_positions_body_yaw', 'target_occupancy'):
                self.values[k] = values[:, bi]
        quat = self.values['base_quaternion_wxyz']
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=0.002):
            raise ValueError('Reference quaternion must be WXYZ and normalized')
        self.rotations = Slerp(self.times, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))

    def sample(self, times):
        query = np.asarray(times, dtype=float).reshape(-1)
        if not np.isfinite(query).all(): raise ValueError('Nonfinite query time')
        query = np.clip(query, self.times[0], self.times[-1])
        lo = np.clip(np.searchsorted(self.times, query, side='right')-1, 0, len(self.times)-2)
        alpha = (query-self.times[lo])/(self.times[lo+1]-self.times[lo])
        out = {}
        for k, v in self.values.items():
            if k == 'base_quaternion_wxyz':
                out[k] = self.rotations(query).as_quat()[:, [3, 0, 1, 2]]
            elif v.dtype == bool or k == 'target_positions_body_yaw':
                # Future target positions are piecewise query labels, not continuous kinematics.
                out[k] = v[np.searchsorted(self.times, query, side='right').clip(1, len(self.times))-1]
            else:
                a = alpha.reshape((-1,)+(1,)*(v.ndim-1))
                out[k] = v[lo]*(1-a)+v[lo+1]*a
        return out


def action_to_target(action, default, scale=0.25):
    return np.asarray(default)+scale*np.asarray(action)


def target_to_action(target, default, scale=0.25):
    return (np.asarray(target)-np.asarray(default))/scale
