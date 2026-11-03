"""Stage 1 离线诊断、冻结 checkpoint 物理对照和诊断 HTML；不执行优化。"""
import argparse
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from acp.data.assets import PROJECT,atomic_json,machine,run_path
from acp.data.playback import resolve_clip


def configuration(provenance,physics):
    """Fresh environments: disabling noise alone does not undo startup physics."""
    if physics not in ('nominal','randomized','randomized-zero-delay'):raise ValueError('Unknown physics condition')
    cfg=deepcopy(provenance['config']);robot=deepcopy(provenance['robot_config'])
    cfg['observation_noise']=False
    cfg['dynamics_randomization']=physics!='nominal'
    # Do not accidentally fall back to m2.json's 0.8 ground friction.
    robot['friction']=cfg['randomization']['dynamics']['ground_friction']
    return cfg,robot


def first_time(mask,times):
    indices=np.flatnonzero(mask)
    return float(times[indices[0]]) if len(indices) else None


def trace_metrics(record):
    """Check PD arithmetic at captured pre-physics instants, not next-step states."""
    tau=record['substep_computed_torque'];applied=record['substep_applied_torque']
    predicted=record['kp']*(record['substep_target']-record['substep_position'])-record['kd']*record['substep_velocity']
    clipped=np.abs(tau-applied)>1e-4
    times=(record['time_before'][:,None]+np.arange(tau.shape[1])[None]*float(record['physics_dt']))
    tilt=np.rad2deg(np.arccos(np.clip(1-2*(record['states'][:,4]**2+record['states'][:,5]**2),-1,1)))
    lost=first_time(tilt>20,record['time_after']);early=times<(lost if lost is not None else np.inf)
    fractions=clipped.mean((0,1));order=np.argsort(fractions)[::-1][:5]
    return dict(pd_formula_max_abs_error_nm=float(np.max(np.abs(tau-predicted))),
        torque_clip_formula_max_abs_error_nm=float(np.max(np.abs(applied-np.clip(tau,-record['effort_limits'],record['effort_limits'])))),
        first_torque_clipping_seconds=first_time(clipped.any(-1).ravel(),times.ravel()),
        first_tilt_20_seconds=lost,
        clipped_fraction_before_tilt_20=float(clipped[early].mean()) if early.any() else None,
        most_clipped_joints={str(record['joint_names'][i]):float(fractions[i]) for i in order},
        normalized_actor_clip_fraction=float((np.abs(record['normalized_actor_observation'])>=9.999).mean()),
        mean_target_position_error_rad=float(np.abs(record['substep_target']-record['substep_position']).mean()))


def plot_settings():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10})
    return plt


def offline(run,output):
    """Use recorded states only; never invent missing old actions/torques."""
    from acp.envs.reference import ReferenceMotion
    from acp.training.contact_audit import posture_evidence
    p=json.loads((run/'provenance.json').read_text());iteration=json.loads((run/'result.json').read_text())['final_iteration']
    output.mkdir(parents=True,exist_ok=True);plt=plot_settings();fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    reports=[]
    for task,color in (('walk','#007f82'),('box','#c57031')):
        path=run/'evaluation'/f'iteration_{iteration:06d}'/f'{task}_from_start.npz'
        with np.load(path) as z:
            states=z['states'];ref=z['reference'];force=z['contact_force_history'];dt=float(z['dt'])
            names=z['joint_names'].tolist();bodies=z['body_names'].tolist()
        elapsed=(np.arange(len(states))+1)*dt;spec=p['config']['tasks'][task];times=spec['start_seconds']+elapsed
        info=resolve_clip(run_path(spec['run']),spec['clip'])
        motion=ReferenceMotion(info['motion'],names,bodies);target=motion.sample(times)
        tilt=np.rad2deg(np.arccos(np.clip(1-2*(states[:,4]**2+states[:,5]**2),-1,1)))
        refq=target['base_quaternion_wxyz'];ref_tilt=np.rad2deg(np.arccos(np.clip(1-2*(refq[:,1]**2+refq[:,2]**2),-1,1)))
        qerror=states[:,13:]-ref[:,3:]
        root_error=np.linalg.norm(states[:,:3]-ref[:,:3],axis=-1)
        axes[0,0].plot(elapsed,tilt,label=task+' 实际',color=color)
        axes[0,0].plot(elapsed,ref_tilt,'--',label=task+' 参考',color=color)
        axes[0,1].plot(elapsed,root_error,label=task,color=color)
        duration=spec['end_seconds']-spec['start_seconds']
        allref=motion.sample(spec['start_seconds']+np.arange(round(duration/dt)+1)*dt)
        velocities=allref['joint_velocities'];limit=np.asarray(p['robot_config']['velocity_limits'])
        accelerations=np.diff(velocities,axis=0)/dt
        byjoint=np.sqrt(np.mean(qerror*qerror,axis=0));order=np.argsort(byjoint)[::-1][:5]
        snapshots=[]
        for seconds in (0.5,1.,1.5,2.,2.26,2.98):
            if elapsed[-1]<seconds:continue
            i=int(np.abs(elapsed-seconds).argmin());worst=np.argsort(np.abs(qerror[i]))[::-1][:4]
            snapshots.append(dict(elapsed_seconds=float(elapsed[i]),tilt_degrees=float(tilt[i]),
                reference_tilt_degrees=float(ref_tilt[i]),root_error_m=float(root_error[i]),
                worst_joint_errors_rad={names[j]:float(qerror[i,j]) for j in worst}))
        reports.append(dict(task=task,clip=spec['clip'],seconds=float(elapsed[-1]),
            first_tilt_20_seconds=first_time(tilt>20,elapsed),first_tilt_30_seconds=first_time(tilt>30,elapsed),
            first_root_error_20cm_seconds=first_time(root_error>.2,elapsed),snapshots=snapshots,
            worst_joint_rmse_rad={names[j]:float(byjoint[j]) for j in order},
            reference_velocity_limit_max_ratio=float(np.max(np.abs(velocities)/limit)),
            reference_acceleration_abs_p99_rad_s2=float(np.quantile(np.abs(accelerations),.99)),
            reference_acceleration_abs_max_rad_s2=float(np.max(np.abs(accelerations))),
            reference_stats_window='entire selected task, not only recorded failure prefix',
            posture=posture_evidence(states,force,times,bodies),
            rollout_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if task=='box':
            for b,label in ((0,'左脚'),(1,'右脚')):
                axes[1,0].plot(elapsed,np.abs(force[:,:,b,2]).max(axis=1),label=label+' 法向Fz')
            image=axes[1,1].imshow(np.abs(qerror).T,origin='lower',aspect='auto',
                extent=(elapsed[0],elapsed[-1],-.5,22.5),cmap='magma',vmin=0,vmax=.7)
            axes[1,1].set_yticks(range(23),names,fontsize=7);fig.colorbar(image,ax=axes[1,1],label='关节角绝对误差 rad')
    for ax,title,unit in zip(axes.flat,('保存回放：基座倾斜','保存回放：根位置误差','爬箱：足部竖直法向力','爬箱：逐关节跟踪偏差'),('°','m','N','关节')):
        ax.set(title=title,xlabel='从片段开头经过时间 s',ylabel=unit)
        if ax is not axes[1,1]:ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'offline.png',dpi=160);plt.close(fig)
    result=dict(source_run=str(run),iteration=iteration,tasks=reports,
        limitations=['One recorded representative per task, not all environments',
            'Old rollout has no actions, PD targets or torques; no saturation inference made',
            'Reference velocity/acceleration checks alone do not establish dynamic feasibility',
            'Normal-force vectors exclude friction in the installed Isaac Lab v2.2.x backend'])
    atomic_json(output/'offline.json',result)
    from acp.training.report import inline
    text='<p>已有 PhysX 状态的离线诊断，不是新仿真。不能从旧文件恢复缺失的动作、PD目标或力矩。</p>'
    for r in reports:
        text+=f'<h2>{r["task"]}</h2><p>倾斜首次超过20°：{r["first_tilt_20_seconds"]}s；超过30°：{r["first_tilt_30_seconds"]}s；根误差首次超过0.2m：{r["first_root_error_20cm_seconds"]}s。</p>'
        text+='<p>全参考窗口关节速度／URDF上限最大比：'+f'{r["reference_velocity_limit_max_ratio"]:.3f}；加速度绝对值P99：{r["reference_acceleration_abs_p99_rad_s2"]:.2f} rad/s²。通过运动学限位不能证明动力学可执行。</p>'
    page(output/'index.html','Stage 1 失稳离线诊断',text+f'<img src="{inline(output/"offline.png","image/png")}" alt="失稳时序与关节误差">')
    print(output/'index.html')


def page(path,title,body):
    path.write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>body{{max-width:1200px;margin:30px auto;padding:20px;font:16px/1.7 system-ui;color:#183144;background:#eef2f5}}img,video{{width:100%}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:10px;border-bottom:1px solid #ccd7df;text-align:left}}code{{overflow-wrap:anywhere}}</style><h1>{html.escape(title)}</h1>{body}</html>''')


def rollout_worker(args):
    from isaaclab.app import AppLauncher
    sys.argv=[sys.argv[0]]
    launcher=AppLauncher(headless=True,device='cuda:0',multi_gpu=False,limit_cpu_threads=4,
        kit_args=f'--/renderer/activeGpu={args._worker} --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1')
    app=launcher.app;task=None
    try:
        import torch
        from acp.envs.stage1 import Stage1Task
        from acp.training.ppo import ActorCritic
        from acp.training.runner import evaluate
        torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
        checkpoint=args.run/'latest.pt';checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        provenance=json.loads((args.run/'provenance.json').read_text())
        if saved['config']!=provenance['config'] or saved['robot_config']!=provenance['robot_config']:
            raise ValueError('Checkpoint and training provenance disagree')
        cfg,robot=configuration(provenance,args.physics)
        specs=[cfg['tasks'][name] for name in cfg['tasks'][args.task].get('members',[args.task])]
        infos=[resolve_clip(run_path(spec['run']),spec['clip']) for spec in specs]
        for info in infos:
            for key in ('motion','scene'):
                expected=provenance['files'].get(str(info[key]))
                if expected is None:
                    suffix='/'+str(info[key].relative_to(PROJECT))
                    candidates=[digest for filename,digest in provenance['files'].items() if filename.endswith(suffix)]
                    if len(candidates)==1:expected=candidates[0]
                if expected is None or hashlib.sha256(info[key].read_bytes()).hexdigest()!=expected:
                    raise ValueError(f'{key} does not match frozen training provenance: {info[key]}')
        task=Stage1Task(infos,robot,cfg,args.task,args.num_envs)
        if args.physics=='randomized-zero-delay':task.env.action_delay_ticks.zero_()
        model=ActorCritic(cfg['ppo']['initial_std'],cfg['ppo']['normalize_observations']).to(task.device)
        model.load_state_dict(saved['model']);model.eval()
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        view=task.env.robot.root_physx_view;pd=task.env.robot.actuators['pd'];j=task.env.joint_ids
        actual=dict(mass=view.get_masses().cpu(),com=view.get_coms().cpu(),material=view.get_material_properties().cpu(),
            kp=pd.stiffness[:,j].cpu(),kd=pd.damping[:,j].cpu(),action_delay_ticks=task.env.action_delay_ticks.cpu(),
            joint_zero_bias=task.env.joint_zero_bias.cpu(),armature=view.get_dof_armatures()[:,j.cpu()].cpu())
        if args.physics=='nominal':
            torch.testing.assert_close(actual['kp'],torch.tensor(robot['kp'],dtype=torch.float32).expand(args.num_envs,-1))
            torch.testing.assert_close(actual['kd'],torch.tensor(robot['kd'],dtype=torch.float32).expand(args.num_envs,-1))
            torch.testing.assert_close(actual['material'][...,:2],torch.full_like(actual['material'][...,:2],robot['friction']))
            torch.testing.assert_close(actual['material'][...,2],torch.zeros_like(actual['material'][...,2]))
            nominal_mass=torch.tensor([task.env.model.body(name).mass[0] for name in task.env.robot.body_names],dtype=actual['mass'].dtype)
            torch.testing.assert_close(actual['mass'],nominal_mass.expand_as(actual['mass']),rtol=1e-4,atol=1e-6)
            assert not actual['action_delay_ticks'].any() and not actual['joint_zero_bias'].any()
        np.savez_compressed(args.output/'physics.npz',**{k:v.numpy() for k,v in actual.items()},body_names=task.env.robot.body_names)
        atomic_json(args.output/'provenance.json',dict(source_run=str(args.run),checkpoint_sha256=checkpoint_hash,
            checkpoint_iteration=saved['iteration'],physics=args.physics,controller=args.controller,seed=args.seed,
            num_envs=args.num_envs,phases=args.phases,config=cfg,robot_config=robot,
            note='Fresh physical environment; nominal robot friction=ground friction=1 is a recorded engineering nominal, not author-disclosed robot nominal; randomized conditions use new instances, not historical instances',
            files={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),PROJECT/'src/acp/training/runner.py',PROJECT/'src/acp/envs/g1.py',PROJECT/'src/acp/envs/stage1.py',PROJECT/'src/acp/envs/randomization.py')}))
        metrics=evaluate(task,model,args.output/'evaluation',saved['iteration'],phases=args.phases,
            diagnostics=True,controller=args.controller,reset_after=False)
        for key,value in model.state_dict().items():torch.testing.assert_close(value.cpu(),before[key],rtol=0,atol=0)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_hash
        atomic_json(args.output/'worker.json',dict(completed=True,optimizer_steps=0,model_unchanged=True,
            checkpoint_unchanged=True,physics=args.physics,controller=args.controller,metrics=metrics))
    except Exception as exc:
        import traceback
        traceback.print_exc();atomic_json(args.output/'worker.json',dict(completed=False,error=str(exc)))
    finally:
        from isaaclab.sim import SimulationContext
        if task is not None:task.close()
        elif SimulationContext.instance() is not None:
            SimulationContext.instance().clear_all_callbacks();SimulationContext.clear_instance()
        app.close()


def supervise(args):
    from acp.management.simulation import select_gpu
    gpu=select_gpu(str(args.gpu or machine().get('simulation_gpu','')))
    args.output.mkdir(parents=True,exist_ok=False)
    command=[str((PROJECT/Path(machine()['simulation_python'])).absolute()),'-u','-m','acp.training.diagnostics',
        'rollout','--_worker',gpu['index'],'--run',str(args.run),'--output',str(args.output),'--task',args.task,
        '--num-envs',str(args.num_envs),'--seed',str(args.seed),'--physics',args.physics,'--controller',args.controller,
        '--phases',*[str(p) for p in args.phases]]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu['uuid'],'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
        'PYTHONPATH':str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'4'}
    atomic_json(args.output/'launch.json',dict(command=command,gpu=gpu))
    with (args.output/'run.log').open('w') as log:
        try:
            result=subprocess.run(command,env=env,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            exit_code=result.returncode;error=None
        except subprocess.TimeoutExpired:
            exit_code=None;error='Simulation diagnostic timed out'
    record=args.output/'worker.json';data=json.loads(record.read_text()) if record.exists() else dict(completed=False)
    data.update(exit_code=exit_code,supervisor_error=error)
    data['passed']=exit_code==0 and data.get('completed',False)
    atomic_json(args.output/'result.json',data);print(json.dumps(data,ensure_ascii=False,indent=2))
    return 0 if data['passed'] else 1


def wait_idle(args,condition):
    from acp.management.simulation import select_gpu
    started=time.monotonic()
    while True:
        try:return select_gpu(str(args.gpu or machine().get('simulation_gpu','')))
        except ValueError as exc:
            if not args.wait_for_gpu or '已有计算任务' not in str(exc):raise
            atomic_json(args.output/'queue_progress.json',dict(state='waiting_for_gpu',condition=condition,
                reason=str(exc),pid=os.getpid(),elapsed_wait_seconds=time.monotonic()-started,
                checked_at=time.time(),wait_timeout_seconds=args.wait_timeout))
            if time.monotonic()-started>=args.wait_timeout:raise TimeoutError('GPU wait timed out; no existing task was stopped')
            time.sleep(min(30,args.wait_timeout-(time.monotonic()-started)))


def suite(args):
    """Sequential evaluation-only jobs; wait without creating a CUDA context."""
    args.output.mkdir(parents=True,exist_ok=True)
    conditions=[('nominal','policy'),('randomized','policy'),('randomized-zero-delay','policy'),('nominal','reference_pd')]
    try:
        for physics,controller in conditions:
            name=physics+'_'+controller;wait_idle(args,name)
            atomic_json(args.output/'queue_progress.json',dict(state='running',condition=name,pid=os.getpid(),checked_at=time.time()))
            child=deepcopy(args);child.physics=physics;child.controller=controller;child.output=args.output/name
            if supervise(child):raise RuntimeError('Diagnostic failed: '+name+'; inspect its run.log')
        gpu=wait_idle(args,'render_report')
        command=[str((PROJECT/Path(machine()['retarget_python'])).absolute()),'-m','acp.training.diagnostics','report',
            '--run',str(args.run),'--output',str(args.output)]
        env={**os.environ,'PYTHONPATH':str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''),
             'CUDA_VISIBLE_DEVICES':gpu['uuid'],'MUJOCO_GL':'egl','MUJOCO_EGL_DEVICE_ID':str(gpu['index']),
             'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4'}
        with (args.output/'report.log').open('w') as log:
            subprocess.run(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=args.timeout)
        atomic_json(args.output/'queue_progress.json',dict(state='completed',pid=os.getpid(),optimizer_steps=0,checked_at=time.time()))
        return 0
    except Exception as exc:
        atomic_json(args.output/'queue_progress.json',dict(state='failed',error=str(exc),pid=os.getpid(),checked_at=time.time()))
        raise


def report(run,output):
    """Summarize real diagnostic captures; leave missing conditions explicit."""
    from acp.training.report import inline,render
    from acp.envs.reference import ReferenceMotion
    provenance=json.loads((run/'provenance.json').read_text());plt=plot_settings()
    rows='';details='';found=[];controls=[]
    for path in sorted(output.glob('*/evaluation/iteration_*/report.json')):
        completed=path.parents[2]/'result.json'
        if not completed.exists() or not json.loads(completed.read_text()).get('passed',False):continue
        condition=path.parents[2].name;data=json.loads(path.read_text());meta=json.loads((path.parents[2]/'provenance.json').read_text())
        found.append(dict(condition=condition,physics=meta['physics'],controller=meta['controller'],checkpoint_sha256=meta['checkpoint_sha256']))
        for result in data['results']:
            for name,r in result['per_task'].items():
                rows+=f'<tr><td>{html.escape(condition)}</td><td>{name}</td><td>{result["initial_phase"]:.2f}</td><td>{r["success_rate"]*100:.1f}% / {r["episodes"]}</td><td>{r["mean_root_error_m"]:.3f}m</td></tr>'
        for path_npz in sorted(path.parent.glob('box_*.npz')):
            with np.load(path_npz) as z:record={k:z[k].copy() for k in z.files}
            control=trace_metrics(record);controls.append(dict(condition=condition,rollout=path_npz.name,**control))
            t=record['time_after'];sub_t=(record['time_before'][:,None]+np.arange(record['substep_target'].shape[1])[None]*float(record['physics_dt'])).ravel()
            tau=record['substep_computed_torque'].reshape(-1,23);applied=record['substep_applied_torque'].reshape(-1,23)
            clipped=np.abs(tau-applied)>1e-4
            fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
            tilt=np.rad2deg(np.arccos(np.clip(1-2*(record['states'][:,4]**2+record['states'][:,5]**2),-1,1)))
            axes[0,0].plot(t,tilt,label='基座倾斜 °');axes[0,0].plot(t,record['root_error']*100,label='根误差 cm');axes[0,0].legend()
            im=axes[0,1].imshow(clipped.T,aspect='auto',origin='lower',extent=(sub_t[0],sub_t[-1],-.5,22.5),cmap='Reds',vmin=0,vmax=1)
            axes[0,1].set_yticks(range(23),record['joint_names'],fontsize=6);axes[0,1].set_title('200Hz 力矩限幅（红色）')
            for j in np.argsort(clipped.mean(0))[-3:]:
                axes[1,0].plot(sub_t,tau[:,j],label=str(record['joint_names'][j]))
                axes[1,0].plot(sub_t,applied[:,j],'--',alpha=.6)
            axes[1,0].set_title('computed实线／applied虚线 Nm');axes[1,0].legend(fontsize=7)
            target_error=np.abs(record['substep_target']-record['substep_position']).mean((1,2))
            axes[1,1].plot(t,target_error,label='平均 |PD目标-q| rad')
            axes[1,1].plot(t,np.abs(record['action']).max(1),label='最大 |策略／重放动作|');axes[1,1].legend()
            for ax in axes.flat:ax.set_xlabel('参考片段时间 s')
            image=path_npz.with_suffix('.png');fig.savefig(image,dpi=140);plt.close(fig)
            spec=provenance['config']['tasks']['box'];info=resolve_clip(run_path(spec['run']),spec['clip'])
            render.reference=ReferenceMotion(info['motion'],record['joint_names'].tolist(),record['body_names'].tolist())
            video=path_npz.with_suffix('.mp4')
            render(path_npz,info['scene'],video,condition+' / '+path_npz.stem,float(record['time_before'][0]))
            details+=f'<h2>{html.escape(condition+" / "+path_npz.stem)}</h2><p>记录PD公式最大残差 {control["pd_formula_max_abs_error_nm"]:.5g} Nm；限幅公式残差 {control["torque_clip_formula_max_abs_error_nm"]:.5g} Nm。倾斜超过20°前的关节×子步限幅比例：{control["clipped_fraction_before_tilt_20"]}。</p><img src="{inline(image,"image/png")}"><video controls preload="metadata" src="{inline(video,"video/mp4")}"></video>'
    if len({r['checkpoint_sha256'] for r in found})>1:raise ValueError('Diagnostic conditions do not share one checkpoint')
    offline_path=output/'offline/offline.png'
    offline_html=f'<h2>已有回放离线诊断</h2><img src="{inline(offline_path,"image/png")}">' if offline_path.exists() else ''
    pending='<p>尚无新的物理诊断结果；下图来自旧 PhysX 回放，不能用于推断新条件的完成率或力矩。</p>' if not found else ''
    body='<p>同一个冻结checkpoint，不更新优化器和归一化统计。名义物理多个环境为重复确定性实例，不是独立样本；随机物理是新采样实例，不能与旧环境逐一配对。reference_pd是无学习的关节参考重放诊断，不是论文策略，也不能单凭其失败证明动作不可实现。</p>'+pending
    body+='<table><tr><th>条件</th><th>动作</th><th>初始相位</th><th>完成率／环境数</th><th>根误差</th></tr>'+rows+'</table>'+offline_html+details
    output.mkdir(parents=True,exist_ok=True);page(output/'index.html','Stage 1 控制与物理条件诊断',body)
    pairing=None
    a=output/'randomized_policy/physics.npz';b=output/'randomized-zero-delay_policy/physics.npz'
    if a.exists() and b.exists():
        with np.load(a) as za,np.load(b) as zb:
            pairing={key:bool(np.array_equal(za[key],zb[key])) for key in ('mass','com','material','kp','kd','joint_zero_bias','armature')}
        if not all(pairing.values()):raise ValueError('Randomized/zero-delay physical samples differ; not a paired delay comparison')
    atomic_json(output/'comparison.json',dict(conditions=found,control_metrics=controls,paired_physics_checks=pairing,physical_results_available=bool(found)))
    print(output/'index.html')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('offline','rollout','suite','report'));p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu');p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--task',choices=('walk','box','walk_box'),default='walk_box');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--physics',choices=('nominal','randomized','randomized-zero-delay'),default='nominal')
    p.add_argument('--controller',choices=('policy','reference_pd'),default='policy')
    p.add_argument('--phases',type=float,nargs='+',default=[0.,.24,.3,.9]);p.add_argument('--timeout',type=float,default=1200)
    p.add_argument('--wait-for-gpu',action='store_true',help='suite：只轮询空闲GPU，不终止已有任务、不预先创建CUDA上下文')
    p.add_argument('--wait-timeout',type=float,default=86400,help='每组开始前最多等待秒数；默认24小时')
    p.add_argument('--background',action='store_true',help='suite：后台执行，状态写queue_progress.json')
    p.add_argument('--_worker',type=int,help=argparse.SUPPRESS)
    args=p.parse_args();args.run=args.run.resolve();args.output=args.output.resolve()
    if args.num_envs<2 or args.phases[0]!=0 or any(not 0<=v<.99 for v in args.phases):p.error('num-envs>=2; phases must start at 0 and lie in [0,.99)')
    if not 0<args.timeout<float('inf') or not 0<args.wait_timeout<float('inf'):p.error('Timeouts must be finite and positive')
    if args.background and args.mode!='suite':p.error('--background is supported for suite only')
    if args.mode=='offline':offline(args.run,args.output)
    elif args.mode=='report':report(args.run,args.output)
    elif args.mode=='suite':
        if args.background:
            args.output.mkdir(parents=True,exist_ok=True)
            previous=args.output/'queue.json'
            if previous.exists():
                pid=json.loads(previous.read_text())['pid']
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:raise ValueError('A diagnostic queue is already running for this output')
            command=[sys.executable,'-u','-m','acp.training.diagnostics',*[x for x in sys.argv[1:] if x!='--background']]
            with (args.output/'queue.log').open('a') as log:
                child=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'PYTHONPATH':str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')})
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
            atomic_json(previous,dict(pid=child.pid,command=command,cwd=str(PROJECT),source_commit=revision,optimizer_steps=0));print(previous)
        else:raise SystemExit(suite(args))
    elif args._worker is not None:rollout_worker(args)
    else:raise SystemExit(supervise(args))


if __name__=='__main__':main()
