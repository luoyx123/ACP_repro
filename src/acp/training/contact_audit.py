"""离线审计参考接触与真实力，输出证据和仅改变oracle点源的成对实验配置。"""
import base64
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from acp.data.assets import atomic_json,run_path
from acp.data.playback import resolve_clip
from acp.data.contacts import measured_force_flags,filter_segments


def segments(flags,times,dt):
    edge=np.diff(np.r_[False,flags,False].astype(int))
    return [[float(times[a]),float(times[b-1]+dt)] for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1))]


def posture_evidence(states,force,times,names):
    """Normal-force impacts and base tilt; impacts do not establish useful support."""
    tilt=np.degrees(np.arccos(np.clip(1-2*(states[:,4]**2+states[:,5]**2),-1,1)))
    norm=np.linalg.norm(force,axis=-1).max(axis=1)
    hands={}
    for name in ('left_hand','right_hand'):
        b=names.index(name);hits=np.flatnonzero(norm[:,b]>1);peak=int(norm[:,b].argmax())
        hands[name]=dict(first_over_1n_seconds=float(times[hits[0]]) if len(hits) else None,
            peak_seconds=float(times[peak]),peak_normal_force_norm_n=float(norm[peak,b]),
            base_tilt_at_peak_degrees=float(tilt[peak]),base_height_at_peak_m=float(states[peak,2]))
    return dict(tilt_definition='angle between base local z and world z; not reference orientation error',
        force_definition='world vector of summed normal forces; excludes tangential friction',hands=hands,
        samples=[dict(seconds=float(times[i]),base_tilt_degrees=float(tilt[i]),base_height_m=float(states[i,2]))
                 for seconds in (1.,1.5,2.) if times[0]<=seconds<=times[-1]
                 for i in [int(np.abs(times-seconds).argmin())]])


def audit(run,output):
    output.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((run/'provenance.json').read_text());config=provenance['config']
    result=json.loads((run/'result.json').read_text());iteration=result['final_iteration'];report=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    for row_index,task in enumerate(('walk','box')):
        spec=config['tasks'][task];info=resolve_clip(run_path(spec['run']),spec['clip'])
        with np.load(info['motion']) as z:motion={k:z[k].copy() for k in z.files}
        times=motion['timestamps'];flags=motion['contact_flags'];names=motion['body_names'].tolist()
        window=(times>=spec['start_seconds'])&(times<=spec['end_seconds'])
        onsets=flags & ~np.vstack((np.zeros((1,9),bool),flags[:-1]))
        bodies=[]
        for b,name in enumerate(names):
            ix=np.flatnonzero(onsets[:,b]&window)
            delta=motion['body_landmarks_w'][ix,b]-motion['contact_surface_positions_w'][ix,b]
            offset=np.linalg.norm(delta,axis=-1)
            bodies.append(dict(body=name,onset_count=len(ix),onset_times_seconds=times[ix].tolist(),
                landmark_surface_offset_mm=(offset*1000).tolist(),
                offset_mean_mm=float(offset.mean()*1000) if len(ix) else None,
                offset_max_mm=float(offset.max()*1000) if len(ix) else None))
        path=run/'evaluation'/f'iteration_{iteration:06d}'/f'{task}_from_start.npz'
        with np.load(path) as z:
            states=z['states'];dt=float(z['dt']);force=z['contact_force_history'];stored_names=z['body_names'].tolist()
        t=spec['start_seconds']+(np.arange(len(states))+1)*dt
        indices=np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-1)
        ordered=[stored_names.index(name) for name in names]
        force=force[:,:,ordered]
        posture=posture_evidence(states,force,t,names)
        actual=measured_force_flags(force,threshold=1,minimum_steps=10)
        force_norm=np.linalg.norm(force,axis=-1).max(axis=1)
        diagnostic_norm=filter_segments(force_norm>1,minimum_steps=10)
        raw=measured_force_flags(force,threshold=1,minimum_steps=1)
        reference=flags[indices]
        for b,body in enumerate(bodies):
            body.update(reference_steps=int(reference[:,b].sum()),measured_raw_steps=int(raw[:,b].sum()),
                measured_filtered_steps=int(actual[:,b].sum()),mismatch_steps=int((reference[:,b]!=actual[:,b]).sum()),
                reference_segments=segments(reference[:,b],t,dt),actual_segments=segments(actual[:,b],t,dt),
                diagnostic_norm_steps=int(diagnostic_norm[:,b].sum()),peak_force_norm_n=float(force_norm[:,b].max()),
                peak_abs_fz_n=float(np.abs(force[:,:,b,2]).max()),norm_segments=segments(diagnostic_norm[:,b],t,dt))
            axes[row_index].broken_barh([(a-spec['start_seconds'],end-a) for a,end in body['reference_segments']],(b*3+.3,.9),facecolors='#007f82')
            axes[row_index].broken_barh([(a-spec['start_seconds'],end-a) for a,end in body['actual_segments']],(b*3+1.3,.9),facecolors='#d57232')
            axes[row_index].broken_barh([(a-spec['start_seconds'],end-a) for a,end in body['norm_segments']],(b*3+2.3,.6),facecolors='#395a91')
        axes[row_index].set(yticks=np.arange(9)*3+1.2,yticklabels=names,xlabel='从片段开头经过时间 s',
            title=task+' 代表回放：绿=参考几何，橙=法向力Fz，蓝=法向力向量模长（过滤0.2s）',xlim=(0,len(states)*dt))
        axes[row_index].grid(axis='x',alpha=.2)
        report.append(dict(task=task,clip=spec['clip'],seconds=len(states)*dt,bodies=bodies,
            geometric_valid_fraction=float(motion['geometry_valid'][window].mean()),
            reference_motion_sha256=hashlib.sha256(info['motion'].read_bytes()).hexdigest(),
            rollout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),posture=posture))
    fig.savefig(output/'contact_timeline.png',dpi=160);plt.close(fig)
    # Use the same task, seed, PPO, noise, gains and geometry in both arms.
    # Stage1Task changes ONLY the oracle point source, not reference kinematic
    # targets, critic body positions or reward landmark definitions.
    for source in ('body_landmark','environment_surface'):
        c=json.loads(json.dumps(config));c['oracle_contact_position_source']=source
        c['profile']='box_oracle_ablation_'+source
        c['engineering_choices']['reference_contact_input']='Next onset within 1.5s; position source='+source
        atomic_json(output/f'box_{source}.json',c)
    evidence=dict(source_run=str(run),iteration=iteration,reference_contacts='static geometry, not measured forces',
        measured_contacts='abs(world Fz)>1N in any 200Hz substep; retain >=10 control steps',tasks=report,
        force_semantics='ContactSensor.net_forces_w sums normal forces only, excluding tangential friction',
        caution='Single representative trajectory per task; horizontal impacts during a fall are not evidence of successful support; no causality or surface-source superiority established',
        stages='Surface-point ablation remains Stage 1; planner hindsight needs successful full episodes and actual body positions')
    atomic_json(output/'contact_audit.json',evidence)
    rows=''
    for task in report:
        for b in task['bodies']:
            mean='—' if b['offset_mean_mm'] is None else f"{b['offset_mean_mm']:.1f}"
            rows+=f"<tr><td>{task['task']}</td><td>{b['body']}</td><td>{b['onset_count']}</td><td>{mean}</td><td>{b['reference_steps']}</td><td>{b['measured_raw_steps']}</td><td>{b['measured_filtered_steps']}</td><td>{b['diagnostic_norm_steps']}</td><td>{b['peak_force_norm_n']:.1f}</td></tr>"
    src='data:image/png;base64,'+base64.b64encode((output/'contact_timeline.png').read_bytes()).decode()
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Stage 1 接触输入审计</title><style>body{{font:16px/1.8 system-ui;background:#eef2f5;color:#192e40;margin:30px auto;max-width:1200px;padding:20px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}code{{overflow-wrap:anywhere}}</style><h1>先审计 RL 配置和失稳，再验证接触接口</h1><p>本报告只审计已有参考几何标签和最终保存的单条代表PhysX回放，不是新的训练结果。</p><h2>当前证据</h2><p>爬箱手部onset的landmark到表面点有约6–7cm偏移；这并不自动证明landmark错误，因为它们本来就是不同的点。论文Stage 1使用参考接触，Stage 2 planner标签来自成功rollout。应将几何表面点作为明确的对照实验，而不是直接替换后宣称修复完成。</p><p>最终爬箱代表回放先前倾失稳，随后撞箱：右手在2.26秒出现约292N法向力，基座已倾斜46.5°；左手在2.98秒出现约85N法向力，基座倾斜39.8°。参考左右手onset分别约2.74／2.78秒。右手提前撞箱，左手发生在同一失败过程内，这些冲击不能当作正确撑箱。左右手参考接触帧为22／20，但实际竖直法向力大于1N的帧数均为0；不能据此说没有碰撞，也不能据此判断缺少支撑的原因。传感器net_forces_w只包含法向力，不含切向摩擦，之前“三维合力”的表述已纠正。此统计只覆盖3.16秒的代表失败轨迹。</p><img src="{src}" alt="接触时序对照"><table><tr><th>任务</th><th>部位</th><th>参考窗口onset数</th><th>onset平均偏移mm</th><th>回放内参考接触帧</th><th>真实Fz帧</th><th>过滤后Fz帧</th><th>过滤后法向力模长帧（诊断）</th><th>法向力模长峰值N</th></tr>{rows}</table><p>onset偏移统计覆盖整个参考任务窗口；参考／真实接触帧统计仅覆盖已保存的从头回放。两者时间范围不同，不能混算。真实接触连续至少10控制步才保留；onset=0的持续接触仍可能在窗口内有参考接触帧。</p><h2>修复与验证顺序</h2><ol><li>保留既有实验；增加显式oracle点来源开关，历史默认landmark，surface缺失时拒绝静默回退。</li><li>先完成RL配置逐项审计，补齐名义动力学／无延迟评估，检查失稳前的姿态、控制指令、力矩饱和与参考动态可跟踪性；终止阈值是工程选择，不能假定与作者一致。</li><li>再核对几何接触、竖直法向力、箱壁法向与身体接触位置。法向力向量模长仅作诊断，不直接替换论文Fz判据；摔倒碰撞不计作正确接触。</li><li>相同单爬箱任务、相同seed和动力学设置，对比body_landmark与environment_surface；只改actor的oracle坐标，不把critic／运动学奖励的身体位置一起替成地形点。</li><li>比较从头完成率、分任务CSI、2–6秒手部接触和力矩饱和，至少完成同预算试验再判断。若两组均失败，继续检查参考动态可跟踪性、控制与标签时序。</li><li>Tracker有成功完整rollout后，再按竖直力阈值和持续时间生成hindsight，满足前10%初始化、到达90%的参考成功条件后入buffer，供Stage 2使用；不能把失败或尾段成功充当完整成功。</li></ol><p>成对配置已生成：box_body_landmark.json、box_environment_surface.json。两组均使用<code>train --task box --training-config &lt;配置&gt;</code>，应从头训练且使用不同输出目录。GPU 0本次核查已有其他任务，未发起新仿真，也未停止任何他人进程。</p></html>'''
    (output/'index.html').write_text(page)
    print(output/'index.html')
