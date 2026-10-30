"""Run in the training Python: timeout GAE, oracle frame and actor isolation."""
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None,'Requires the separate PyTorch training environment')
class Stage1MathTest(unittest.TestCase):
    def test_timeout_bootstrap_stops_at_reset(self):
        from acp.training.ppo import generalized_advantages
        reward = torch.tensor([[1.],[100.]])
        value = torch.tensor([[2.],[3.]]);nxt = torch.tensor([[10.],[4.]])
        terminal = torch.tensor([[False],[True]]);done = torch.tensor([[True],[True]])
        a,r = generalized_advantages(reward,value,nxt,terminal,done,gamma=.9,lam=.95)
        torch.testing.assert_close(a,torch.tensor([[8.],[97.]]))
        torch.testing.assert_close(r,torch.tensor([[10.],[100.]]))

    def test_oracle_uses_actual_yaw_and_true_onsets(self):
        from acp.envs.reference import ReferenceMotion
        from acp.training.task import TensorReference
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'motion.npz';times=np.arange(101)/50
            flags=np.zeros((101,9),bool);flags[20:40,0]=True;flags[70:90,0]=True
            body=np.zeros((101,9,3));body[:,0]=[1.,2.2,0.]
            np.savez(p,timestamps=times,joint_names=[str(i) for i in range(23)],body_names=[str(i) for i in range(9)],
                base_position_w=np.zeros((101,3)),base_quaternion_wxyz=np.tile([1,0,0,0],(101,1)),
                base_linear_velocity_w=np.zeros((101,3)),base_angular_velocity_w=np.zeros((101,3)),
                joint_positions=np.zeros((101,23)),joint_velocities=np.zeros((101,23)),body_landmarks_w=body,
                contact_flags=flags,target_positions_body_yaw=body,target_occupancy=flags)
            ref=TensorReference(ReferenceMotion(p,[str(i) for i in range(23)],[str(i) for i in range(9)]),.5,2.,.02,'cpu')
            q=torch.tensor([[2**-.5,0,0,2**-.5]])
            targets,occupied=ref.contacts(torch.tensor([.5]),torch.tensor([[1.,2.,0.]]),q)
            self.assertTrue(occupied[0,0])
            torch.testing.assert_close(targets[0,0],torch.tensor([.2,0.,0.]),atol=1e-6,rtol=0)
            # At 0.5s the contact at 0.4s is already ongoing. Its next onset
            # is 1.4s; moving beyond the footprint must mask it, not clamp it.
            _,occupied=ref.contacts(torch.tensor([.5]),torch.zeros((1,3)),q)
            self.assertFalse(occupied[0,0])
            _,occupied=ref.contacts(torch.tensor([1.5]),torch.tensor([[1.,2.,0.]]),q)
            self.assertFalse(occupied.any())

    def test_csi_bounds_and_ppo_update_with_privileged_critic(self):
        from acp.training.task import update_csi
        from acp.training.ppo import ActorCritic,PPO
        import json
        from pathlib import Path
        torch.manual_seed(9);torch.set_num_threads(2)
        torch.testing.assert_close(update_csi(torch.tensor([0.,.99,.9]),torch.tensor([True,False,True])),torch.tensor([0.,.99,.85]))
        model=ActorCritic();config=json.loads((Path(__file__).parents[1]/'configs/stage1.json').read_text())['ppo']
        agent=PPO(model,config);rollout=[]
        for _ in range(2):
            with torch.no_grad():
                a=torch.randn(8,468);c=torch.randn(8,353)
                action,prob,value,mean,std=model.act(a,c)
                rollout.append(dict(actor=a,critic=c,action=action,log_prob=prob,value=value,mean=mean,std=std,
                    reward=torch.randn(8),next_value=torch.zeros(8),terminated=torch.zeros(8,dtype=torch.bool),done=torch.zeros(8,dtype=torch.bool)))
        before=[p.detach().clone() for p in model.actor.parameters()]
        result=agent.update(rollout)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.actor.parameters())))
        # Changing privileged critic inputs cannot change the actor distribution.
        a=torch.randn(8,468);mean=model.distribution(a).mean.detach().clone()
        model.value(torch.randn(8,353)*100)
        torch.testing.assert_close(mean,model.distribution(a).mean)

    def test_reward_world_and_individual_yaw_frames_are_distinct(self):
        from acp.training.task import rewards
        q=torch.zeros((1,23));root=torch.tensor([[0.,0.,1.]]);quat=torch.tensor([[1.,0.,0.,0.]])
        velocity=torch.tensor([[1.,0.,0.]]);body=root[:,None].expand(-1,9,-1).clone()
        reference=dict(base_position_w=root,base_quaternion_wxyz=quat,base_linear_velocity_w=velocity,
            base_angular_velocity_w=torch.zeros((1,3)),joint_positions=q,joint_velocities=q,
            body_landmarks_w=body,contact_flags=torch.ones((1,9),dtype=torch.bool))
        actual=root+torch.tensor([[1.,0.,0.]])
        state=dict(root_position=actual,root_quaternion=quat,linear_velocity=velocity,angular_velocity=torch.zeros((1,3)),
            joint_positions=q,joint_velocities=q,body_positions=body+torch.tensor([[[1.,0.,0.]]]),
            measured_contacts=torch.ones((1,9)),joint_limits=torch.stack((q-1,q+1),-1),velocity_limits=q+10,
            applied_torque=q,computed_torque=q,action=q)
        terms,_=rewards(state,reference,q,q,torch.zeros((1,9,3)))
        torch.testing.assert_close(terms['track_base_pos'],torch.tensor([np.exp(-4)],dtype=torch.float32))
        torch.testing.assert_close(terms['track_contact_pos_base'],torch.ones(1))
        torch.testing.assert_close(terms['track_contact_pos_global'],torch.tensor([np.exp(-9)],dtype=torch.float32))
        torch.testing.assert_close(terms['track_contact_label'],torch.ones(1))
        self.assertEqual(float(terms['contact_acc']),0.)


if __name__=='__main__':unittest.main()
