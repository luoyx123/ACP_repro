"""Physics-condition isolation, substep trace timing and alive-only statistics."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np
from acp.training.diagnostics import configuration,trace_metrics
try:
    import torch
except ImportError:
    torch=None


class DiagnosticConfigurationTest(unittest.TestCase):
    def test_trace_checks_pd_formula_and_torque_clipping(self):
        shape=(2,2,23);tau=np.full(shape,1.9)
        r=dict(substep_computed_torque=tau,substep_applied_torque=np.ones(shape),kp=np.full(23,2.),kd=np.full(23,.1),
            substep_target=np.ones(shape),substep_position=np.zeros(shape),substep_velocity=np.ones(shape),
            effort_limits=np.ones(23),time_before=np.array([0.,.02]),time_after=np.array([.02,.04]),physics_dt=.01,
            states=np.zeros((2,36)),joint_names=[str(i) for i in range(23)],normalized_actor_observation=np.zeros((2,468)))
        r['states'][:,3]=1
        m=trace_metrics(r)
        self.assertAlmostEqual(m['pd_formula_max_abs_error_nm'],0.)
        self.assertAlmostEqual(m['torque_clip_formula_max_abs_error_nm'],0.)
        self.assertEqual(m['first_torque_clipping_seconds'],0.)
        self.assertEqual(m['clipped_fraction_before_tilt_20'],1.)
        r['substep_computed_torque'][0,0,0]+=1
        self.assertAlmostEqual(trace_metrics(r)['pd_formula_max_abs_error_nm'],1.)

    def test_nominal_uses_frozen_config_and_correct_ground_without_mutation(self):
        p={'config':{'observation_noise':True,'dynamics_randomization':True,
                    'randomization':{'dynamics':{'ground_friction':1.}}},'robot_config':{'friction':.8,'kp':[400]}}
        original=json.loads(json.dumps(p))
        cfg,robot=configuration(p,'nominal')
        self.assertFalse(cfg['observation_noise']);self.assertFalse(cfg['dynamics_randomization'])
        self.assertEqual(robot['friction'],1.)
        robot['kp'][0]=1
        self.assertEqual(p,original)
        for mode in ('randomized','randomized-zero-delay'):
            cfg,_=configuration(p,mode)
            self.assertTrue(cfg['dynamics_randomization']);self.assertFalse(cfg['observation_noise'])
        with self.assertRaises(ValueError):configuration(p,'unknown')


@unittest.skipUnless(torch is not None,'Requires separate PyTorch environment; no GPU needed')
class DiagnosticCaptureTest(unittest.TestCase):
    def test_g1_trace_records_prephysics_pd_and_respects_delay(self):
        # Compile the actual method without importing Isaac; exercise its command,
        # delay and trace ordering against a deterministic explicit-PD mock.
        source=Path(__file__).parents[1]/'src/acp/envs/g1.py'
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='G1ReferenceEnv')
        step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        namespace={'torch':torch};exec(compile(ast.Module(body=[step],type_ignores=[]),str(source),'exec'),namespace)
        def make(enabled):
            d=NS(joint_pos=torch.zeros(1,23),joint_vel=torch.zeros(1,23),computed_torque=torch.zeros(1,23),applied_torque=torch.zeros(1,23))
            robot=NS(data=d,update=lambda dt:None)
            robot.set_joint_position_target=lambda target:setattr(robot,'target',target.clone())
            def write():
                d.computed_torque.copy_(100*(robot.target-d.joint_pos))
                d.applied_torque.copy_(d.computed_torque.clamp(-1,1))
            robot.write_data_to_sim=write
            env=NS(num_envs=1,decimation=4,dt=.005,joint_ids=torch.arange(23),default=torch.zeros(23),
                joint_zero_bias=torch.full((1,23),.01),action_delay_ticks=torch.tensor([2]),executed_action=torch.zeros(1,23),
                previous_action=torch.zeros(1,23),cfg={'action_scale':.25},history=torch.zeros(1,5,75),
                force_history=torch.zeros(1,4,9,3),robot=robot,diagnostics_enabled=enabled,diagnostic_trace=None,
                sensor=NS(update=lambda *a,**k:None),probe=NS(update=lambda dt:None))
            env.sim=NS(step=lambda **k:d.joint_pos.add_(.01));env.contact_forces=lambda:torch.zeros(1,9,3)
            env.proprio=lambda:torch.zeros(1,75)
            return env
        traced=make(True);plain=make(False)
        namespace['step'](traced,torch.ones(1,23));namespace['step'](plain,torch.ones(1,23))
        trace=traced.diagnostic_trace
        torch.testing.assert_close(trace['target'][0,:,0],torch.tensor([.01,.01,.26,.26]))
        torch.testing.assert_close(trace['position'][0,:,0],torch.tensor([0.,.01,.02,.03]))
        torch.testing.assert_close(trace['computed_torque'],100*(trace['target']-trace['position']))
        torch.testing.assert_close(trace['applied_torque'],trace['computed_torque'].clamp(-1,1))
        torch.testing.assert_close(traced.robot.data.joint_pos,plain.robot.data.joint_pos)
        self.assertIsNone(plain.diagnostic_trace)

    def test_evaluate_excludes_dead_envs_from_saturation_and_keeps_terminal_frame(self):
        from acp.training.runner import evaluate
        n=2;d=NS(root_link_state_w=torch.zeros(n,13),joint_pos=torch.zeros(n,23),joint_vel=torch.zeros(n,23))
        d.root_link_state_w[:,3]=1
        pd=NS(stiffness=torch.ones(n,23),damping=torch.ones(n,23),effort_limit=torch.ones(n,23))
        env=NS(robot=NS(data=d,actuators={'pd':pd}),origins=torch.zeros(n,3),joint_ids=torch.arange(23),
            force_history=torch.zeros(n,4,9,3),dt=.005,decimation=4,joint_names=[str(i) for i in range(23)],
            body_names=[str(i) for i in range(9)],action_delay_ticks=torch.zeros(n,dtype=torch.long),
            joint_zero_bias=torch.zeros(n,23),body_positions=lambda:torch.zeros(n,9,3),info={'scene':'scene','motion':'motion'})
        task=NS(env=env,n=n,device='cpu',time=torch.zeros(n),duration=torch.full((n,),.1),dt=.02,
            assignment=torch.tensor([0,1]),member_names=['walk','box'],infos=[env.info,env.info],name='walk_box',
            cfg={'dynamics_randomization':True})
        def sample(times,**kwargs):
            count=len(times)
            return dict(base_position_w=torch.zeros(count,3),joint_positions=torch.zeros(count,23),
                base_quaternion_wxyz=torch.tensor([[1.,0,0,0]]).repeat(count,1),joint_velocities=torch.zeros(count,23),
                body_landmarks_w=torch.zeros(count,9,3))
        task.reference=NS(sample=sample)
        def reset(**kwargs):task.time.zero_();task.count=0
        task.reset=reset;task.observations=lambda:(torch.zeros(n,468),torch.zeros(n,353))
        def step(action):
            task.count+=1;task.time.add_(.02)
            trace={k:torch.zeros(n,4,23) for k in ('target','position','velocity','computed_torque','applied_torque')}
            if task.count>1:trace['computed_torque'][0]=10  # already dead; must not count
            env.diagnostic_trace=trace
            done=torch.tensor([True,task.count>=3]);zero=torch.zeros(n)
            return zero,done,done,torch.zeros(n,dtype=torch.bool),{'term':zero},{'root_error':zero,'orientation_error':zero,'joint_rmse':zero}
        task.step=step;model=NS(actor_norm=lambda x:x,actor=lambda x:torch.zeros(n,23))
        with tempfile.TemporaryDirectory() as folder:
            evaluate(task,model,Path(folder),10000,phases=(0.,),diagnostics=True,reset_after=False)
            out=Path(folder)/'iteration_010000'
            r=json.loads((out/'report.json').read_text())
            self.assertFalse(r['action_delay'])
            self.assertEqual(r['results'][0]['per_task']['walk']['torque_clipped_fraction_by_joint'],[0.]*23)
            with np.load(out/'walk_from_start.npz') as z:
                self.assertEqual(z['states'].shape[0],1)
                self.assertEqual(z['substep_target'].shape,(1,4,23))
            with np.load(out/'box_from_start.npz') as z:self.assertEqual(z['states'].shape[0],3)
        self.assertFalse(env.diagnostics_enabled)


if __name__=='__main__':unittest.main()
