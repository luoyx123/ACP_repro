"""23-DoF SuperMimic dynamics and differentiable FK/terrain heads.

Adapted from InstinctMJ rl/supermimic/networks.py and kinematics.py, commit
7ca851e (local modified reference). Attribution/license: third_party/instinctmj.
ACP has height observations, so there is no image encoder or visual latent.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from acp.training.ppo import RunningNormalizer
from .rotations import axis_angle_to_mat, mat_to_6d, mm, mv, quat_to_mat, so3_exp, so3_log, yaw_mat

STATE_DIM = 64  # p3, R9, v3, omega3, q23, qd23; matrices avoid quaternion branch gradients
DELTA_DIM = 58


def pack(S):
    p, R, v, w, q, qd = S
    return torch.cat((p, R.flatten(-2), v, w, q, qd), -1)


def unpack(x):
    return x[..., :3], x[..., 3:12].reshape(*x.shape[:-1], 3, 3), x[..., 12:15], x[..., 15:18], x[..., 18:41], x[..., 41:64]


def invariant(S):
    _, R, v, w, q, qd = S
    return torch.cat((mat_to_6d(mm(yaw_mat(R).transpose(-1, -2), R)),
                      mv(R.transpose(-1, -2), v), mv(R.transpose(-1, -2), w), q, qd), -1)


class WorldModel(nn.Module):
    """Frame-local residual dynamics, plus 9 contact logits; fp32 integration."""
    def __init__(self, context_dim, hidden=(2048, 2048, 2048), amp=False):
        super().__init__()
        widths = [58+23+context_dim+187, *hidden]
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()])
        layers.append(nn.Linear(widths[-1], DELTA_DIM+9))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].bias.data[DELTA_DIM:] = -3
        self.input_norm = RunningNormalizer(widths[0])
        self.output_norm = RunningNormalizer(DELTA_DIM)
        self.amp = amp

    def inputs(self, S, action, context, scan):
        return torch.cat((invariant(S), action, context, scan), -1)

    @staticmethod
    def delta(S, S1):
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        return torch.cat((mv(yaw_mat(R).transpose(-1, -2), p1-p), so3_log(mm(R.transpose(-1, -2), R1)),
                          mv(R.transpose(-1, -2), v1-v), mv(R.transpose(-1, -2), w1-w), q1-q, qd1-qd), -1)

    def step(self, S, action, context, scan):
        p, R, v, w, q, qd = S
        x = self.input_norm(self.inputs(S, action, context, scan))
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            y = self.net(x).float()
        # Unlike observation normalization, the dynamics normalization must not clip targets.
        delta = y[..., :DELTA_DIM]*torch.sqrt(self.output_norm.variance+1e-8)+self.output_norm.mean
        dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 23, 23), -1)
        return (p+mv(yaw_mat(R), dp), mm(R, so3_exp(dr)), v+mv(R, dv), w+mv(R, dw), q+dq, qd+dqd), y[..., DELTA_DIM:]


class Kinematics:
    """Use the actual 23-DoF model, keeping fixed wrists and sphere-hand links."""
    def __init__(self, model, cfg, device):
        import mujoco
        root = model.body('pelvis').id
        bodies = [root]
        for b in range(root+1, model.nbody):
            if model.body_parentid[b] in bodies:
                bodies.append(b)
        parent = [-1]+[bodies.index(model.body_parentid[b]) for b in bodies[1:]]
        depth = [0]
        for par in parent[1:]:
            depth.append(depth[par]+1)
        order = sorted(range(len(bodies)), key=lambda i:depth[i])
        position = {b:i for i, b in enumerate(order)}
        tensor = lambda x, dtype=torch.float32:torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
        self.levels = []
        for level in range(1, max(depth)+1):
            ids = [i for i in order if depth[i] == level]
            joint, axes = [], []
            for i in ids:
                b = bodies[i]
                if model.body_jntnum[b]:
                    j = model.body_jntadr[b]
                    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not np.allclose(model.jnt_pos[j], 0):
                        raise ValueError('Only body-origin hinge joints supported')
                    joint.append(cfg['joint_names'].index(model.joint(j).name))
                    axes.append(model.jnt_axis[j])
                else:
                    joint.append(-1)
                    axes.append(np.zeros(3))
            self.levels.append(dict(parent=tensor([position[parent[i]] for i in ids], torch.long),
                                    joint=tensor(joint, torch.long), axis=tensor(axes),
                                    pos=tensor(model.body_pos[[bodies[i] for i in ids]]),
                                    rot=quat_to_mat(tensor(model.body_quat[[bodies[i] for i in ids]]))))
        self.tracked = [position[bodies.index(model.body(n).id)] for n in cfg['body_links']]
        self.offsets = tensor(cfg['landmark_offsets'])

    def __call__(self, S):
        p, R, _, _, q, _ = S
        P, Rs = p[:, None], R[:, None]
        for level in self.levels:
            par, j = level['parent'], level['joint']
            Pi = P[:, par]+mv(Rs[:, par], level['pos'])
            angle = torch.where(j >= 0, q[:, j.clamp(min=0)], 0.)
            Ri = mm(mm(Rs[:, par], level['rot']), axis_angle_to_mat(level['axis'], angle))
            P, Rs = torch.cat((P, Pi), 1), torch.cat((Rs, Ri), 1)
        return P[:, self.tracked]+mv(Rs[:, self.tracked], self.offsets)


class Terrain:
    """Exact top-down triangle intersection on static meshes, no smoothed step edges.

    Queries are chunked over triangles, keeping memory proportional to N*rays*32.
    This matches the old top-down scan; overhang undersides are not represented.
    """
    def __init__(self, scene, device):
        from acp.envs.conversion import terrain_triangles
        parts = terrain_triangles(scene)
        triangles = np.concatenate([v[f] for v, f in parts]) if parts else np.empty((0, 3, 3))
        # Vertical triangles have no top-down intersection area.
        det = ((triangles[:, 1, 0]-triangles[:, 0, 0])*(triangles[:, 2, 1]-triangles[:, 0, 1])-
               (triangles[:, 1, 1]-triangles[:, 0, 1])*(triangles[:, 2, 0]-triangles[:, 0, 0]))
        self.triangles = torch.as_tensor(triangles[np.abs(det)>1e-10], device=device, dtype=torch.float32)
        self.grids = {}
        for nx, ny in [(9, 7), (17, 11)]:
            x, y = torch.meshgrid(torch.linspace(-(nx-1)*.05, (nx-1)*.05, nx, device=device),
                                  torch.linspace(-(ny-1)*.05, (ny-1)*.05, ny, device=device), indexing='ij')
            self.grids[nx*ny] = torch.stack((x.flatten(), y.flatten()), -1)

    def scan(self, S, count=187):
        p, R = S[:2]
        xy = p[:, None, :2]+mv(yaw_mat(R)[:, None, :2, :2], self.grids[count])
        height = torch.zeros(xy.shape[:-1], device=p.device)  # all prepared scenes have a z=0 plane
        for triangles in self.triangles.split(32):
            if not len(triangles):
                continue
            a, b, c = triangles.unbind(1)
            u, v = b-a, c-a
            d = xy[..., None, :]-a[:, :2]
            det = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
            s = (d[..., 0]*v[:, 1]-d[..., 1]*v[:, 0])/det
            t = (u[:, 0]*d[..., 1]-u[:, 1]*d[..., 0])/det
            z = a[:, 2]+s*u[:, 2]+t*v[:, 2]
            valid = (s>=-1e-6) & (t>=-1e-6) & (s+t<=1+1e-6)
            height = torch.maximum(height, torch.where(valid, z, -torch.inf).amax(-1))
        return height-p[:, 2:3]
