"""GPU reference queries and Stage-1 task math; no Isaac imports."""
import numpy as np
import torch


def yaw(q):
    w,x,y,z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def wrap(angle):
    return (angle+torch.pi)%(2*torch.pi)-torch.pi


def yaw_local(points, root, quaternion):
    delta = points-root[:,None];angle = yaw(quaternion)
    c,s = torch.cos(angle)[:,None],torch.sin(angle)[:,None]
    return torch.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1],delta[...,2]),-1)


def quaternion_error(actual, reference):
    return 2*torch.acos((actual*reference).sum(-1).abs().clamp(0,1))


def inverse_rotate(q, vector):
    xyz = -q[:,1:];t = 2*torch.cross(xyz,vector,dim=-1)
    return vector+q[:,:1]*t+torch.cross(xyz,t,dim=-1)


def update_csi(phase, success, adjustment=.05):
    return (phase+torch.where(success,-adjustment,adjustment)).clamp(0,.99)


class TensorReference:
    def __init__(self, motion, start, end, dt, device):
        if start < motion.times[0]-1e-6 or end > motion.times[-1]+1e-6 or end <= start:
            raise ValueError('Invalid task reference time window')
        self.start,self.end,self.dt = start,end,dt
        times = start+np.arange(round((end-start)/dt)+1)*dt
        if abs(times[-1]-end) > 1e-6: raise ValueError('Reference window must align to control timestep')
        values = motion.sample(times)
        self.values = {k:torch.as_tensor(v,device=device,dtype=torch.bool if v.dtype==bool else torch.float32)
                       for k,v in values.items()}
        # Use the original clip's true onsets, including contacts already active
        # before a selected window. Do not invent an onset at a crop boundary.
        flags = motion.values['contact_flags']
        starts = flags & ~np.vstack((np.zeros((1,9),bool),flags[:-1]))
        self.onsets = []
        for b in range(9):
            indices = np.flatnonzero(starts[:,b])
            self.onsets.append((torch.as_tensor(motion.times[indices],device=device,dtype=torch.float32),
                torch.as_tensor(motion.values['body_landmarks_w'][indices,b],device=device,dtype=torch.float32)))

    def sample(self, times):
        phase = (times.clamp(self.start,self.end)-self.start)/self.dt
        lo = phase.floor().long().clamp(0,len(self.values['joint_positions'])-2)
        alpha = (phase-lo).clamp(0,1)
        result = {}
        for k,v in self.values.items():
            if v.dtype==torch.bool or k=='target_positions_body_yaw':
                result[k] = v[phase.floor().long().clamp(0,len(v)-1)]
            elif k=='base_quaternion_wxyz':
                a,b = v[lo],v[lo+1]
                b = torch.where((a*b).sum(-1,keepdim=True)<0,-b,b)
                dot = (a*b).sum(-1).clamp(0,1);theta = torch.acos(dot.clamp(max=.999999))
                denominator = torch.sin(theta)
                spherical = (torch.sin((1-alpha)*theta)[:,None]*a+torch.sin(alpha*theta)[:,None]*b)/denominator[:,None]
                linear = a+(b-a)*alpha[:,None]
                q = torch.where((dot>.9995)[:,None],linear,spherical)
                result[k] = q/q.norm(dim=-1,keepdim=True)
            else:
                scale = alpha.reshape((-1,)+(1,)*(v.ndim-1))
                result[k] = v[lo]+(v[lo+1]-v[lo])*scale
        return result

    def contacts(self, times, actual_root, actual_quaternion):
        targets = torch.zeros((len(times),9,3),device=times.device)
        occupied = torch.zeros((len(times),9),device=times.device,dtype=torch.bool)
        for body,(onsets,positions) in enumerate(self.onsets):
            if not len(onsets): continue
            index = torch.searchsorted(onsets,times.contiguous())
            safe = index.clamp(max=len(onsets)-1)
            valid = (index<len(onsets)) & (onsets[safe]<=times+1.5+1e-6) & (onsets[safe]<=self.end+1e-6)
            targets[:,body] = positions[safe];occupied[:,body] = valid
        local = yaw_local(targets,actual_root,actual_quaternion)
        occupied &= (local[...,0].abs()<=.8) & (local[...,1].abs()<=.5)
        return torch.where(occupied[...,None],local,0),occupied


class ReferenceBank:
    """Fixed motion/terrain assignment, sharing one policy without skill labels."""
    def __init__(self,references,assignment):
        self.references,self.assignment = references,assignment
        self.start = torch.tensor([r.start for r in references],device=assignment.device)[assignment]
        self.end = torch.tensor([r.end for r in references],device=assignment.device)[assignment]

    def sample(self,times,env_ids=None):
        env_ids = torch.arange(len(times),device=times.device) if env_ids is None else env_ids
        result = {}
        for index,reference in enumerate(self.references):
            mask = self.assignment[env_ids]==index
            if not mask.any(): continue
            values = reference.sample(times[mask])
            for key,value in values.items():
                if key not in result:result[key] = torch.empty((len(times),*value.shape[1:]),device=value.device,dtype=value.dtype)
                result[key][mask] = value
        return result

    def contacts(self,times,actual_root,actual_quaternion):
        targets = torch.zeros((len(times),9,3),device=times.device)
        occupied = torch.zeros((len(times),9),device=times.device,dtype=torch.bool)
        for index,reference in enumerate(self.references):
            mask = self.assignment==index
            if mask.any():targets[mask],occupied[mask] = reference.contacts(times[mask],actual_root[mask],actual_quaternion[mask])
        return targets,occupied


def rewards(state, reference, previous_action, previous_previous_action, body_acceleration):
    """Table 3 raw terms; caller applies weights and the 0.02s control dt."""
    move = (reference['base_linear_velocity_w'].norm(dim=-1)>.1).float()
    root_error = (state['root_position']-reference['base_position_w']).square().sum(-1)
    angle = quaternion_error(state['root_quaternion'],reference['base_quaternion_wxyz'])
    actual_local = yaw_local(state['body_positions'],state['root_position'],state['root_quaternion'])
    reference_local = yaw_local(reference['body_landmarks_w'],reference['base_position_w'],reference['base_quaternion_wxyz'])
    q = state['joint_positions'];bounds = state['joint_limits']
    result = {
        'track_base_pos': torch.exp(-move*root_error/.5**2-angle.square()),
        'track_base_vel': torch.exp(-(state['linear_velocity']-reference['base_linear_velocity_w']).square().sum(-1)
            -(state['angular_velocity']-reference['base_angular_velocity_w']).square().sum(-1)/3**2),
        'track_joint_pos': torch.exp(-wrap(q-reference['joint_positions']).square().sum(-1)/2**2),
        'track_joint_vel': torch.exp(-(state['joint_velocities']-reference['joint_velocities']).square().sum(-1)/10**2),
        'track_contact_label': (state['measured_contacts']*(2*reference['contact_flags'].float()-1)).mean(-1),
        'track_contact_pos_base': torch.exp(-(actual_local-reference_local).square().sum((1,2))),
        'track_contact_pos_global': torch.exp(-move*(state['body_positions']-reference['body_landmarks_w']).square().sum((1,2))),
        'action_rate': (state['action']-previous_action).square().sum(-1),
        'action_rate_2nd': (state['action']-2*previous_action+previous_previous_action).square().sum(-1),
        'joint_torque': state['applied_torque'].square().sum(-1),
        'torque_limits': (state['computed_torque']-state['applied_torque']).abs().sum(-1),
        'dof_pos_limits': ((bounds[...,0]-q).clamp(min=0)+(q-bounds[...,1]).clamp(min=0)).sum(-1).clamp(max=30),
        'dof_vel_limits': (state['joint_velocities'].abs()-.9*state['velocity_limits']).clamp(0,1).sum(-1),
        'contact_acc': body_acceleration.norm(dim=-1).sum(-1).clamp(max=1e4)
    }
    return result,dict(root_error=root_error.sqrt(),orientation_error=angle,
        joint_rmse=(q-reference['joint_positions']).square().mean(-1).sqrt())
