"""Scene-conditioned support costs for Mixamo's non-foot contact phases."""
import cvxpy as cp
import mujoco
import numpy as np
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


class MixamoContactRetargeter(InteractionMeshRetargeter):
    def support_error(self, q, frame_idx):
        masks=getattr(self,'surface_contact_masks',{})
        if not masks:return 0.
        model,data=self.robot_model,self.robot_data
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        surfaces=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('multi_boxes_')
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
        error=0.
        for name,mask in masks.items():
            if not mask[frame_idx]:continue
            body=model.body(name).id
            geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]==body
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
            distances=[mujoco.mj_geomDistance(model,data,g,s,.3,None) for g in geoms for s in surfaces]
            if distances:
                nearest=min(distances)
                if nearest < -.001:return float('inf')
                error+=max(0.,nearest-.005)**2
        return error

    def solve_single_iteration(self, *args, **kwargs):
        candidate,cost=super().solve_single_iteration(*args,**kwargs)
        frame=kwargs.get('frame_idx',0)
        if not getattr(self,'surface_contact_masks',{}):return candidate,cost
        current=kwargs['q_locked'].copy()
        current[self.q_a_indices]=kwargs['q_a_n_last']
        error=self.support_error(current,frame)
        # A thin rung can be crossed by a linearized SQP step, yielding a small
        # predicted cost but a large true gap on its other side. Backtrack on
        # the exact geometry. Keep up to 3mm support slack for other objectives;
        # the exported trajectory still undergoes all geometry/limit checks.
        tolerance=max(error,9e-6)
        if self.support_error(candidate,frame)<=tolerance:return candidate,cost
        for alpha in (.5,.25,.125,.0625):
            trial=current+alpha*(candidate-current)
            trial[3:7]/=np.linalg.norm(trial[3:7])
            if self.support_error(trial,frame)<=tolerance:return trial,cost
        return current,cost

    def additional_pose_costs(self, q, dqa, frame_idx):
        masks=getattr(self,'surface_contact_masks',{})
        if not masks:return []
        model,data=self.robot_model,self.robot_data
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        surfaces=[g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('multi_boxes_')
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
        terms=[]
        for body_name,mask in masks.items():
            if not mask[frame_idx]:continue
            body=model.body(body_name).id
            geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]==body
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
            nearest=None
            for geom in geoms:
                for surface in surfaces:
                    segment=np.zeros(6)
                    distance=mujoco.mj_geomDistance(model,data,geom,surface,.30,segment)
                    if not 0<distance<.30:continue
                    if nearest is None or distance<nearest[0]:nearest=(distance,segment.copy())
            if nearest is None:continue
            distance,segment=nearest
            normal=(segment[3:]-segment[:3])/distance
            jac=self._calc_contact_jacobian_from_point(body,segment[:3],input_world=True)
            terms.append(getattr(self,'surface_contact_weight',200.)*cp.square(cp.pos(distance-.005-normal@jac[:,self.q_a_indices]@dqa)))
        return terms
