"""Numerical backend recovery must preserve constraints and restore global state."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import cvxpy as cp
    import mujoco
except ImportError:
    cp=None


@unittest.skipUnless(cp is not None,'Needs Holosoma CVXPY/MuJoCo environment')
class MixamoRecoveryTest(unittest.TestCase):
    def test_active_hand_support_reduces_actual_surface_gap(self):
        import numpy as np
        import mujoco
        from acp.pipeline.contact_solver import MixamoContactRetargeter
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="right_sphere_hand_link" pos="0 0 .15"><freejoint/>
            <geom name="hand" type="sphere" size=".05" mass="1"/></body>
          <body name="support"><geom name="multi_boxes_rung" type="box" size=".1 .2 .05"/></body>
        </worldbody></mujoco>''')
        solver=MixamoContactRetargeter.__new__(MixamoContactRetargeter)
        solver.robot_model=model;solver.robot_data=mujoco.MjData(model)
        solver.has_dynamic_object=False;solver.q_a_indices=np.arange(model.nq)
        solver.surface_contact_masks={'right_sphere_hand_link':np.array([True,False])}
        q=model.qpos0.copy();dq=cp.Variable(model.nq)
        terms=solver.additional_pose_costs(q,dq,0)
        problem=cp.Problem(cp.Minimize(cp.sum(terms)+cp.sum_squares(dq)),[dq[3:]==0,cp.norm(dq)<=.1])
        problem.solve(solver=cp.CLARABEL)
        solver.robot_data.qpos[:]=q+dq.value;mujoco.mj_forward(model,solver.robot_data)
        gap=mujoco.mj_geomDistance(model,solver.robot_data,model.geom('hand').id,model.geom('multi_boxes_rung').id,.2,None)
        self.assertGreaterEqual(gap,0.)
        self.assertLess(gap,.006)
        self.assertEqual(solver.additional_pose_costs(q,dq,1),[])
        # A linearized step can cross a thin rung and emerge on the other
        # side. The true geometry check must reject its penetrating midpoint.
        from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
        overshoot=q.copy();overshoot[2]=-.25
        with patch.object(InteractionMeshRetargeter,'solve_single_iteration',return_value=(overshoot,0.)):
            accepted,_=solver.solve_single_iteration(q_locked=q,q_a_n_last=q.copy(),frame_idx=0)
        solver.robot_data.qpos[:]=accepted;mujoco.mj_forward(model,solver.robot_data)
        gap=mujoco.mj_geomDistance(model,solver.robot_data,model.geom('hand').id,model.geom('multi_boxes_rung').id,.2,None)
        self.assertGreaterEqual(gap,-.001)
        self.assertLess(gap,.05)

    def test_same_problem_recovers_with_scs_and_restores_solve(self):
        from acp.pipeline.datasets.mixamo import numerical_recovery
        real=cp.Problem.solve
        def fail_clarabel(problem,*args,**kwargs):
            if kwargs.get('solver')==cp.CLARABEL:raise cp.error.SolverError('synthetic numerical failure')
            return real(problem,*args,**kwargs)
        x=cp.Variable(2)
        problem=cp.Problem(cp.Minimize(cp.sum_squares(x)),[x[0]>=1,x[1]>=2,cp.norm(x)<=3])
        events=[]
        with patch.object(cp.Problem,'solve',fail_clarabel):
            with numerical_recovery(events):problem.solve(solver=cp.CLARABEL)
            self.assertIs(cp.Problem.solve,fail_clarabel)
        self.assertIs(cp.Problem.solve,real)
        self.assertEqual(len(problem.constraints),3)
        self.assertLessEqual(max(float(c.violation().max()) for c in problem.constraints),1e-4)
        self.assertEqual(events[0]['backend'],'SCS same cone constraints')


if __name__=='__main__':unittest.main()
