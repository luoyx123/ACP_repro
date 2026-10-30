"""Unified Stage-1 train/evaluate launcher with isolated GPU workers."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from acp.data.assets import PROJECT,atomic_json,machine,run_path,registry
from acp.management.simulation import select_gpu


def evaluate(task,model,folder,iteration):
    import numpy as np
    import torch
    folder = folder/f'iteration_{iteration:06d}';folder.mkdir(parents=True,exist_ok=True)
    task.evaluating = True;results = []
    # The simulator retains mutable history tensors after evaluation; no_grad
    # keeps them writable when normal training resumes.
    with torch.no_grad():
        for phase,label in ((0.,'from_start'),(.9,'from_tail')):
            task.reset(phase=phase)
            alive = torch.ones(task.n,device=task.device,dtype=torch.bool)
            successes = torch.zeros_like(alive);lengths = torch.zeros_like(task.time)
            error = torch.zeros_like(task.time);joint_error = torch.zeros_like(task.time)
            returns = torch.zeros_like(task.time)
            representatives = {name:int((task.assignment==i).nonzero()[0]) for i,name in enumerate(task.member_names)}
            records = {name:dict(states=[],contacts=[],reference=[]) for name in representatives}
            for _ in range(int(float(task.duration.max())/task.dt)+2):
                actor,_ = task.observations()
                action = model.actor(model.actor_norm(actor))
                reward,terminated,done,success,_,metrics = task.step(action)
                for name,index in representatives.items():
                    if alive[index]:
                        env = task.env;record = records[name]
                        state = torch.cat((env.robot.data.root_link_state_w[index],env.robot.data.joint_pos[index,env.joint_ids])).clone()
                        state[:3] -= env.origins[index];record['states'].append(state.cpu().numpy())
                        record['contacts'].append(env.force_history[index].cpu().numpy())
                        ref = task.reference.sample(task.time[index:index+1],env_ids=torch.tensor([index],device=task.device))
                        record['reference'].append(torch.cat((ref['base_position_w'][0],ref['joint_positions'][0])).cpu().numpy())
                lengths += alive;error += metrics['root_error']*alive;joint_error += metrics['joint_rmse']*alive
                returns += reward*alive;successes |= success & alive;alive &= ~done
                if not alive.any(): break
            result = dict(initial_phase=phase,success_rate=float(successes.float().mean()),
                episodes=task.n,mean_control_steps=float(lengths.mean()),mean_root_error_m=float((error/lengths.clamp(min=1)).mean()),
                mean_joint_rmse_rad=float((joint_error/lengths.clamp(min=1)).mean()),mean_return=float(returns.mean()))
            result['per_task'] = {name:dict(success_rate=float(successes[task.assignment==i].float().mean()),
                episodes=int((task.assignment==i).sum()),mean_root_error_m=float((error/lengths.clamp(min=1))[task.assignment==i].mean()))
                for i,name in enumerate(task.member_names)}
            results.append(result)
            for name,record in records.items():
                np.savez_compressed(folder/(name+'_'+label+'.npz'),states=np.asarray(record['states']),
                    contact_force_history=np.asarray(record['contacts']),reference=np.asarray(record['reference']),
                    dt=task.dt,joint_names=task.env.joint_names,body_names=task.env.body_names)
    task.evaluating = False;task.reset()
    atomic_json(folder/'report.json',dict(iteration=iteration,task=task.name,results=results,
        scene=str(task.env.info['scene']),motion=str(task.env.info['motion']),joint_names=task.env.joint_names,
        scenes={name:str(info['scene']) for name,info in zip(task.member_names,task.infos)},
        note='Deterministic policy evaluation; reference kinematics are withheld from actor.'))
    metrics = dict(evaluation_full_success=results[0]['success_rate'],evaluation_tail_success=results[1]['success_rate'],
                evaluation_root_error_m=results[0]['mean_root_error_m'],evaluation_joint_rmse_rad=results[0]['mean_joint_rmse_rad'])
    for name in task.member_names:
        metrics['evaluation_'+name+'_full_success'] = results[0]['per_task'][name]['success_rate']
        metrics['evaluation_'+name+'_tail_success'] = results[1]['per_task'][name]['success_rate']
    return metrics


def worker(args):
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    launcher = AppLauncher(headless=True,device='cuda:0',multi_gpu=False,limit_cpu_threads=4,
        kit_args=f'--/renderer/activeGpu={args._worker} --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1')
    app = launcher.app;task = None
    try:
        import numpy as np
        import torch
        from acp.data.playback import resolve_clip
        from acp.envs.stage1 import Stage1Task
        from acp.training.ppo import ActorCritic,PPO
        torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
        config = json.loads(args.training_config.read_text());selected = config['tasks'][args.task]
        specs = [config['tasks'][name] for name in selected.get('members',[args.task])]
        infos = []
        for spec in specs:
            dataset = registry()['runs'][spec['run']]['dataset']
            if dataset in config['excluded_datasets']: raise ValueError(f'Excluded dataset: {dataset}')
            info = resolve_clip(run_path(spec['run']),spec['clip'])
            if not info.get('stage1_offline_eligible',False) and not spec.get('experimental_data_override',False):
                raise ValueError('Selected motion is not admitted; explicit experimental override required')
            infos.append(info)
        if args.num_envs<len(infos):raise ValueError('Each task needs at least one environment')
        robot_cfg = json.loads((PROJECT/'configs/m2.json').read_text())
        task = Stage1Task(infos,robot_cfg,config,args.task,args.num_envs)
        model = ActorCritic(config['ppo']['initial_std'],config['ppo']['normalize_observations']).to(task.device)
        optimizer = PPO(model,config['ppo']);start_iteration = 0
        if args.resume:
            checkpoint = torch.load(args.resume,map_location=task.device,weights_only=False)
            if checkpoint['task_config']!=selected or checkpoint['robot_config']!=robot_cfg or checkpoint['config']!=config:
                raise ValueError('Resume task/model/training configuration differs from checkpoint')
            model.load_state_dict(checkpoint['model']);optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
            if len(checkpoint['curriculum_phase'])!=task.n: raise ValueError('Resume environment count differs')
            task.phase.copy_(checkpoint['curriculum_phase']);task.reset()
            start_iteration = checkpoint['iteration']
        source_files = [*Path(__file__).parent.glob('*.py'),PROJECT/'src/acp/envs/stage1.py',PROJECT/'src/acp/envs/g1.py',
                        args.training_config,PROJECT/'configs/m2.json',task.env.urdf,
                        *[item[key] for item in infos for key in ('motion','scene')]]
        provenance = dict(task=args.task,task_config=selected,config=config,robot_config=robot_cfg,
            seed=args.seed,num_envs=args.num_envs,torch=torch.__version__,
            actor_input=468,critic_input=353,actor_reference_kinematics=False,
            original_assets=[dict(clip=item['clip'],status=item.get('status'),reasons=item.get('reasons')) for item in infos],
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),resume=str(args.resume) if args.resume else None,
            files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
        atomic_json(args.output/'provenance.json',provenance)
        actor,critic = task.observations()
        if actor.shape[1]!=468 or critic.shape[1]!=353: raise ValueError('Invalid actor/critic dimensions')
        # Integration check: privileged reference joint streams must influence
        # only the critic, even through the task's observation assembler.
        task.evaluating = True
        before_actor,before_critic = task.observations()
        originals = [ref.values['joint_positions'] for ref in task.reference.references]
        for ref,original in zip(task.reference.references,originals):ref.values['joint_positions'] = original+.25
        after_actor,after_critic = task.observations()
        for ref,original in zip(task.reference.references,originals):ref.values['joint_positions'] = original
        task.evaluating = False
        if not torch.equal(before_actor,after_actor) or torch.equal(before_critic,after_critic):
            raise ValueError('Reference kinematics leaked into actor, or critic lost privileged reference')
        atomic_json(args.output/'observation_contract.json',dict(actor_dimension=468,critic_dimension=353,
            actor_unchanged_when_reference_joints_perturbed=True,critic_changed_when_reference_joints_perturbed=True))
        evaluation = evaluate(task,model,args.output/'evaluation',start_iteration)
        print(json.dumps(dict(iteration=start_iteration,baseline=True,**evaluation)),flush=True)
        total_steps = 0;recent_success = [];recent_start = []
        started = time.monotonic();last_evaluation = evaluation
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output/'tensorboard'))
        with (args.output/'metrics.jsonl').open('a',buffering=1) as log:
            log.write(json.dumps(dict(iteration=start_iteration,baseline=True,**evaluation))+'\n')
            for iteration in range(start_iteration+1,start_iteration+args.iterations+1):
                rollout = [];term_sums = {};reward_sum = 0.;root_error_sum = 0.
                for _ in range(config['ppo']['rollout_steps']):
                    with torch.no_grad():
                        actor,critic = task.observations()
                        if config['ppo']['normalize_observations']:
                            model.actor_norm.update(actor);model.critic_norm.update(critic)
                        a,c = model.actor_norm(actor),model.critic_norm(critic)
                        action,log_prob,value,mean,std = model.act(a,c)
                        reward,terminated,done,success,terms,errors = task.step(action)
                        _,next_critic = task.observations()
                        next_value = model.value(model.critic_norm(next_critic))
                        rollout.append(dict(actor=a,critic=c,action=action,log_prob=log_prob,value=value,mean=mean,std=std,
                            reward=reward,terminated=terminated,done=done,next_value=next_value))
                        if done.any():
                            recent_success.extend(success[done].cpu().tolist());recent_start.extend(task.start_phase[done].cpu().tolist())
                            recent_success = recent_success[-1000:];recent_start = recent_start[-1000:]
                        task.finish(done,success)
                        reward_sum += float(reward.mean());root_error_sum += float(errors['root_error'].mean())
                        for key,v in terms.items(): term_sums[key] = term_sums.get(key,0.)+float(v.mean())
                optimization = optimizer.update(rollout)
                total_steps += task.n*config['ppo']['rollout_steps']
                steps = config['ppo']['rollout_steps']
                row = dict(iteration=iteration,environment_steps=total_steps,
                    elapsed_seconds=time.monotonic()-started,environment_steps_per_second=total_steps/(time.monotonic()-started),
                    reward_per_control_step=reward_sum/steps,root_error_m=root_error_sum/steps,
                    curriculum_mean=float(task.phase.mean()),curriculum_minimum=float(task.phase.min()),
                    curriculum_from_start_fraction=float((task.phase<=.001).float().mean()),
                    recent_csi_episode_success=float(np.mean(recent_success)) if recent_success else None,
                    recent_episode_initial_phase=float(np.mean(recent_start)) if recent_start else None,
                    reward_terms={key:v/steps for key,v in term_sums.items()},**optimization)
                if iteration%args.eval_interval==0 or iteration==start_iteration+args.iterations:
                    last_evaluation = evaluate(task,model,args.output/'evaluation',iteration);row.update(last_evaluation)
                if iteration%args.save_interval==0 or iteration==start_iteration+args.iterations:
                    checkpoint = dict(iteration=iteration,model=model.state_dict(),optimizer=optimizer.optimizer.state_dict(),
                        curriculum_phase=task.phase.clone(),config=config,task_config=selected,robot_config=robot_cfg,
                        provenance=provenance,last_evaluation=last_evaluation)
                    temporary = args.output/'checkpoint.tmp';torch.save(checkpoint,temporary)
                    temporary.replace(args.output/'latest.pt')
                    if args.keep_checkpoints:
                        import shutil
                        shutil.copy2(args.output/'latest.pt',args.output/f'checkpoint_{iteration:06d}.pt')
                log.write(json.dumps(row,allow_nan=False)+'\n')
                for key,val in row.items():
                    if isinstance(val,(int,float)): writer.add_scalar(key,val,iteration)
                for key,val in row['reward_terms'].items():writer.add_scalar('reward_terms/'+key,val,iteration)
                if iteration%10==0 or iteration==start_iteration+args.iterations:
                    print(json.dumps({k:v for k,v in row.items() if k!='reward_terms'}),flush=True)
                atomic_json(args.output/'progress.json',row)
        writer.close()
        atomic_json(args.output/'worker_result.json',dict(passed=True,task=args.task,iterations=args.iterations,
            final_iteration=start_iteration+args.iterations,environment_steps=total_steps,last_evaluation=last_evaluation,
            checkpoint=str(args.output/'latest.pt'),note='Completed training run; convergence is judged separately from process success.'))
    except Exception as exc:
        import traceback
        traceback.print_exc();atomic_json(args.output/'worker_result.json',dict(passed=False,error=str(exc)))
    finally:
        from isaaclab.sim import SimulationContext
        if task is not None: task.close()
        elif SimulationContext.instance() is not None:
            SimulationContext.instance().clear_all_callbacks();SimulationContext.clear_instance()
        app.close()


def supervise(args):
    gpu = select_gpu(str(args.gpu or machine().get('simulation_gpu','')))
    python = str((PROJECT/Path(machine()['simulation_python'])).absolute())
    command = [python,'-u','-m','acp.training.runner','--_worker',gpu['index'],
        '--task',args.task,'--num-envs',str(args.num_envs),'--iterations',str(args.iterations),'--seed',str(args.seed),
        '--training-config',str(args.training_config),'--output',str(args.output),
        '--eval-interval',str(args.eval_interval),'--save-interval',str(args.save_interval)]
    if args.resume:command += ['--resume',str(args.resume)]
    if args.keep_checkpoints:command += ['--keep-checkpoints']
    env = {**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':gpu['uuid'],
        'PYTHONPATH':str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'4'}
    atomic_json(args.output/'launch.json',dict(command=command,gpu=gpu,supervisor_pid=os.getpid()))
    with (args.output/'run.log').open('w') as log:
        try:
            child = subprocess.run(command,env=env,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            code = child.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            atomic_json(args.output/'result.json',dict(passed=False,error=str(exc)));return 1
    record = args.output/'worker_result.json'
    result = json.loads(record.read_text()) if record.is_file() else dict(passed=False,error='Worker did not write result')
    result['exit_code'] = code;result['passed'] = result['passed'] and code==0
    atomic_json(args.output/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['passed'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['walk','box','walk_box'],required=True)
    p.add_argument('--gpu');p.add_argument('--num-envs',type=int,default=2048)
    p.add_argument('--iterations',type=int,default=10000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--training-config',type=Path,default=PROJECT/'configs/stage1.json')
    p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--eval-interval',type=int,default=100);p.add_argument('--save-interval',type=int,default=100)
    p.add_argument('--keep-checkpoints',action='store_true');p.add_argument('--background',action='store_true')
    p.add_argument('--timeout',type=float,default=86400)
    p.add_argument('--_worker',type=int,help=argparse.SUPPRESS);p.add_argument('--_supervise',action='store_true',help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.num_envs,args.iterations,args.eval_interval,args.save_interval)<1:p.error('Counts must be positive')
    args.training_config = args.training_config.absolute()
    if args.resume:args.resume = args.resume.absolute()
    if args._worker is not None:worker(args);return
    if args._supervise:raise SystemExit(supervise(args))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    args.output = (args.output or PROJECT/f'outputs/实验归档/stage1/{args.task}/{stamp}').absolute()
    select_gpu(str(args.gpu or machine().get('simulation_gpu','')))
    args.output.mkdir(parents=True,exist_ok=False)
    if args.background:
        command = [sys.executable,'-u','-m','acp.training.runner',*sys.argv[1:]]
        command.remove('--background');command += ['--_supervise']
        if '--output' not in command:command += ['--output',str(args.output)]
        env = {**os.environ,'PYTHONPATH':str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')}
        with (args.output/'supervisor.log').open('w') as log:
            child = subprocess.Popen(command,env=env,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(args.output/'background.json',dict(pid=child.pid,command=command));print(args.output);return
    raise SystemExit(supervise(args))


if __name__=='__main__':main()
