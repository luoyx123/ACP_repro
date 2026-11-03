"""离线汇总 Stage 1 评估，渲染已记录 PhysX 状态，生成可独立分享的 HTML。"""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import numpy as np
from acp.data.assets import atomic_json,run_path
from acp.data.playback import resolve_clip


def load_json(path):return json.loads(path.read_text())


def trajectory_metrics(path, duration, phase):
    with np.load(path) as z:
        states=z['states'];reference=z['reference'];dt=float(z['dt'])
    error=np.linalg.norm(states[:,:3]-reference[:,:3],axis=1)
    joint=np.sqrt(np.mean((states[:,13:]-reference[:,3:])**2,axis=1))
    # Same episode-end convention as Stage1Task; float tolerance handles NPZ dt.
    reached=phase*duration+len(states)*dt >= duration*.99-1e-5
    return dict(frames=len(states),seconds=len(states)*dt,root_mean_m=float(error.mean()),
        root_final_m=float(error[-1]),joint_rmse_rad=float(np.sqrt(np.mean(joint**2))),
        reached_end=bool(reached),root_threshold_crossed=bool(error[-1]>.75),
        finite=bool(np.isfinite(states).all()),representative_only=True)


def render(path, scene, destination, title, start_seconds):
    """MuJoCo only renders; dynamics and policy are never rerun here."""
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    with np.load(path) as z:
        states=z['states'];refs=z['reference'];dt=float(z['dt']);names=z['joint_names'].tolist()
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    addresses=[model.jnt_qposadr[model.joint(n).id] for n in names]
    renderer=mujoco.Renderer(model,height=480,width=640)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19)
    camera=mujoco.MjvCamera();camera.distance=2.8;camera.azimuth=130;camera.elevation=-18
    # Both panels use identical camera position, centered on the reference.
    try:
        with imageio.get_writer(destination,fps=1/(2*dt),codec='libx264',quality=7,
            ffmpeg_params=['-movflags','+faststart']) as writer:
            for i in range(0,len(states),2):
                state=states[i];ref=refs[i];camera.lookat[:]=ref[:3]
                data.qpos[:7]=state[:7];data.qpos[addresses]=state[13:]
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);actual=renderer.render().copy()
                # NPZ stores reference position/joints, but not reference quaternion.
                # Resolve it from original motion at the recorded clip time.
                q=render.reference.sample(np.array([start_seconds+(i+1)*dt]))['base_quaternion_wxyz'][0]
                data.qpos[:3]=ref[:3];data.qpos[3:7]=q;data.qpos[addresses]=ref[3:]
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);reference=renderer.render().copy()
                frame=Image.fromarray(np.concatenate((actual,reference),axis=1));draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,1280,64),fill='#142437')
                draw.text((12,5),title+' · 实际 PhysX 执行',font=font,fill='white')
                draw.text((652,5),'参考动作 · 仅作对照，不计策略成功',font=font,fill='white')
                error=np.linalg.norm(state[:3]-ref[:3])
                draw.text((12,34),f'片段时间 {start_seconds+(i+1)*dt:.2f}s  |  根误差 {error:.3f}m',font=font,fill='#bde9e8')
                draw.text((652,34),'同一场景与相机 · 1× 实时速度',font=font,fill='#bde9e8')
                writer.append_data(np.asarray(frame))
                if i==len(states)//2//2*2:frame.save(destination.with_suffix('.jpg'))
    finally:renderer.close()


def charts(rows, reports, run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11})
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors={'walk':'#007f82','box':'#d57232'};ev=[r for r in rows if 'evaluation_walk_full_success'in r]
    for task,label in [('walk','行走'),('box','爬箱')]:
        for suffix,style in [('full','-'),('tail','--')]:
            axes[0,0].plot([r['iteration'] for r in ev],[r[f'evaluation_{task}_{suffix}_success']*100 for r in ev],
                style,color=colors[task],label=label+('从头' if suffix=='full' else '尾部10%'),lw=1.7)
        axes[0,1].plot([r['iteration'] for r in reports],[r['results'][0]['per_task'][task]['mean_root_error_m'] for r in reports],color=colors[task],label=label)
    training=[r for r in rows if 'curriculum_mean'in r];x=[r['iteration'] for r in training]
    axes[1,0].plot(x,[r['curriculum_mean'] for r in training],label='CSI平均起始相位',color='#395a91')
    axes[1,0].plot(x,[r['curriculum_from_start_fraction'] for r in training],label='可从开头训练的环境比例',color='#007f82')
    axes[1,1].plot(x,[r['reward_per_control_step'] for r in training],color='#8499af',alpha=.3,lw=.6)
    values=np.array([r['reward_per_control_step'] for r in training]);window=min(100,len(values))
    axes[1,1].plot(x[window-1:],np.convolve(values,np.ones(window)/window,'valid'),color='#395a91',label=f'{window}次移动均值')
    for ax,title,ylabel in zip(axes.flat,['独立冻结策略评估','从头评估：根位置误差','混合课程进度（不能代替分任务指标）','训练奖励（不等于动作成功率）'],['成功率 %','平均根误差 m','相位 / 比例','每控制步奖励']):
        ax.set(title=title,xlabel='PPO 更新次数',ylabel=ylabel);ax.grid(alpha=.18);ax.legend(loc='best',fontsize=9)
    axes[0,0].set_ylim(-3,103);axes[1,0].set_ylim(-.03,1.03)
    fig.savefig(output/'training.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
    for task in ['walk','box']:
        with np.load(run/'evaluation/iteration_010000'/f'{task}_from_start.npz') as z:
            t=(np.arange(len(z['states']))+1)*float(z['dt']);err=np.linalg.norm(z['states'][:,:3]-z['reference'][:,:3],axis=1)
            axes[0].plot(t,err,label=task,color=colors[task]);axes[1].plot(t,z['states'][:,2],color=colors[task],label=task+'实际')
            axes[1].plot(t,z['reference'][:,2],'--',color=colors[task],label=task+'参考')
    axes[0].axhline(.75,color='#b74444',ls=':',label='根误差终止阈值 0.75m')
    for ax,title in zip(axes,['最终保存代表轨迹：根位置误差','最终保存代表轨迹：根高度']):
        ax.set(title=title,xlabel='从片段开头经过的时间 s',ylabel='m');ax.grid(alpha=.18);ax.legend(fontsize=9)
    fig.savefig(output/'diagnostics.png',dpi=160);plt.close(fig)


def inline(path,mime):return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();run=args.run.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True);os.environ.setdefault('MUJOCO_GL','egl')
    provenance=load_json(run/'provenance.json');config=provenance['config'];result=load_json(run/'result.json')
    rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    reports=[load_json(path) for path in sorted((run/'evaluation').glob('iteration_*/report.json'))]
    final=reports[-1];iteration=final['iteration']
    if iteration!=10000:raise ValueError('This two-clip report expects the completed 10,000-update run')
    checkpoint=load_json(run/'checkpoint_analysis.json');final_training=load_json(run/'progress.json')
    cases=[('walk',iteration,'from_start',0.,'行走：最终完整成功'),('box',iteration,'from_start',0.,'爬箱：最终从头失败'),
        ('box',iteration,'from_tail',.9,'爬箱：仅尾段成功'),('walk',0,'from_start',0.,'行走：初始策略失败')]
    from acp.envs.reference import ReferenceMotion
    videos=[]
    for task,it,label,phase,title in cases:
        spec=config['tasks'][task];duration=spec['end_seconds']-spec['start_seconds']
        path=run/'evaluation'/f'iteration_{it:06d}'/f'{task}_{label}.npz'
        info=resolve_clip(run_path(spec['run']),spec['clip'])
        render.reference=ReferenceMotion(info['motion'],provenance['robot_config']['joint_names'],provenance['robot_config']['body_names'])
        metrics=trajectory_metrics(path,duration,phase)
        video=out/f'{task}_{it:06d}_{label}.mp4'
        render(path,info['scene'],video,title,spec['start_seconds']+phase*duration)
        videos.append(dict(task=task,iteration=it,title=title,initial_phase=phase,file=video.name,**metrics,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(title,metrics,flush=True)
    charts(rows,reports,run,out)
    evidence=dict(result=result,final_evaluation=final,checkpoint=checkpoint,representatives=videos,
        evaluation_count=len(reports),box_best_full_success=max(r['results'][0]['per_task']['box']['success_rate'] for r in reports),
        run=str(run),source_hashes={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in
            ['result.json','provenance.json','metrics.jsonl','checkpoint_analysis.json']})
    atomic_json(out/'summary.json',evidence)
    table=''
    for index,label in [(0,'从头'),(1,'尾部10%')]:
        for task,name in [('walk','行走'),('box','爬箱')]:
            r=final['results'][index]['per_task'][task];rate=r['success_rate'];n=r['episodes']
            table+=f'<tr><td>{name} · {label}</td><td>{round(rate*n)}/{n}</td><td>{rate*100:.2f}%</td><td>{r["mean_root_error_m"]:.3f} m</td></tr>'
    video_html=''
    for v in videos:
        path=out/v['file'];scope='仅证明尾部约1.82秒可执行，不能证明爬箱成功。' if v['initial_phase'] else '代表环境单条回放；总体成功率以上表为准。'
        video_html+=f'<article><h3>{v["title"]}</h3><p>{v["seconds"]:.2f}秒 · 平均根误差 {v["root_mean_m"]:.3f}m · 最终根误差 {v["root_final_m"]:.3f}m。{scope}</p><video controls preload="metadata" playsinline poster="{inline(path.with_suffix(".jpg"),"image/jpeg")}" src="{inline(path,"video/mp4")}"></video></article>'
    css='''*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#192e40;font:16px/1.75 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1160px;margin:auto;padding:48px 28px}header{border-top:5px solid #007f82;padding:30px 0}small{letter-spacing:.12em;color:#4f697b}h1{font-size:38px;line-height:1.3;margin:12px 0}h2{margin-top:42px;font-size:24px}h3{margin:0 0 8px}p{margin:10px 0}.lead{font-size:20px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card,article,.panel{background:white;border:1px solid #d8e1e8;border-radius:12px;padding:24px;margin:18px 0}.number{font-size:36px;font-weight:700;color:#007f82}.bad{color:#b6532d}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:14px;border-bottom:1px solid #dde5eb}th{background:#e0e8ed}video,img{width:100%;display:block;border-radius:6px}video{background:#152533}code{overflow-wrap:anywhere;font-size:13px}.muted{color:#536b7c}.tag{color:#9b422a;font-weight:700}a{color:#007f82}details{margin-top:24px}footer{margin-top:40px;font-size:13px;color:#536b7c}@media(max-width:650px){main{padding:24px 16px}.cards{grid-template-columns:1fr}h1{font-size:28px}td,th{padding:8px;font-size:13px}}'''
    hours=final_training['elapsed_seconds']/3600
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Stage 1 · 行走与爬箱训练评估</title><style>{css}</style><main>
<header><small>ACP REPRO / STAGE 1 / 2026-11-03</small><h1>行走已学会，完整爬箱尚未学会</h1><p class="lead">10,000次PPO更新正常完成。流程完成不等于两项任务都收敛；当前 M3 尚不能验收。</p><p class="muted">单张 RTX A5000 · 2,048环境 · 共享策略 · 491,520,000 transitions · 训练循环约{hours:.2f}小时（含周期评估，不含最初初始化／评估）</p></header>
<div class="cards"><div class="card">行走 · 从头完成<div class="number">99.90%</div>1,023 / 1,024 环境</div><div class="card">爬箱 · 从头完成<div class="number bad">0%</div>0 / 1,024 环境</div><div class="card">爬箱 · 最后10%<div class="number">99.32%</div>1,017 / 1,024；仅代表尾段</div></div>
<h2>最终评估：完整动作与尾段分别计分</h2><table><thead><tr><th>任务／初始位置</th><th>成功数／环境数</th><th>完成率</th><th>平均根误差</th></tr></thead><tbody>{table}</tbody></table>
<p class="muted">评估冻结网络、使用确定性动作，关闭观测噪声和初态扰动，保留启动时采样的不同动力学及动作延迟。每任务1,024个固定随机物理实例，单一 seed、已见片段；不是1,024条不同动作，也不是未见地形泛化评估。成功定义为到达99%参考相位且未超过工程终止阈值（根误差0.75m或姿态误差1.5rad）。</p>
<h2>训练是否真正进步</h2><img alt="训练成功率、根误差、CSI与奖励曲线" src="{inline(out/'training.png','image/png')}">
<div class="panel"><p><b>行走：</b>最终CSI有1,021/1,024环境能从相位0开始训练，平均相位0.00015，完整动作完成率接近100%。</p><p><b>爬箱：</b>全部{len(reports)}次固定检查的从头完成率均为0%。最终CSI平均相位0.304，最小0.240；没有环境推进至开头。平均课程仍从约6.10秒开始，最靠前约4.82秒。尾部完成率高不能覆盖这段能力缺口。</p><p><b>混合指标偏差：</b>全局近期CSI episode成功率99.4%受两种任务的episode长度、难度和起始相位共同影响；不能解读为爬箱成功率99.4%。</p></div>
<h2>实际执行视频：左侧策略，右侧参考</h2><p>从训练保存的 PhysX 状态回放；MuJoCo只负责渲染，不重新执行动力学。关节顺序按名称映射；右侧是同时间参考姿态，不是策略输出。视频按固定顺序选择最终代表环境和初始策略，不筛选“最好看”的结果。</p>{video_html}
<h2>失败发生在哪里</h2><img alt="最终代表轨迹根误差与高度" src="{inline(out/'diagnostics.png','image/png')}"><div class="panel"><p>最终爬箱代表轨迹只执行3.16秒，根误差到0.777m，触发0.75m位置误差阈值；对应参考片段总长20.1秒。此处只能确认提前偏离参考，不能单凭日志断言是摩擦、奖励或动作延迟导致。</p><p>训练的主要瓶颈是爬箱前半段，并非训练进程崩溃。下一轮应首先检查失败前的接触目标与实际接触、动作与力矩饱和、参考是否能被当前执行器跟踪，再做单爬箱和随机化分组对照。当前证据不支持只延长混合训练就一定解决。</p></div>
<h2>配置审计与结论边界</h2><div class="panel"><p>已启用：本体／接触／目标／Tracker高程噪声，材质、质量、质心、PD、armature、关节零位、动作延迟，以及初态和里程计扰动。七项动力学写入回读及重置后保持检查均通过。Tracker扫描9×7，critic扫描17×11，actor468维、critic353维；50Hz策略／200HzPD。</p><p>仍有论文差异：仅两条动作、并行数量和GPU资源不同，PD与终止阈值属于工程选择，自碰撞关闭，传感误差注入有记录在案的解释。Stage 1使用oracle接触，没有ACP规划器和LiDAR神经建图；并非完整论文复现。两个数据片段仍保留原离线质量状态和明确的实验使用override。</p></div>
<details><summary>证据与复现信息</summary><p>源码依据：训练 provenance 记录的哈希；原训练配置被冻结在 provenance.json。训练正常完成、exit_code=0；累计100次训练后评估，含初始共101次。独立视频仅为每种任务第一个环境的记录，不能代替环境总体统计。</p><p><code>{html.escape(str(run))}</code></p><p>完整摘要和文件哈希另存 summary.json；本HTML内嵌全部视频与图片，离线单文件即可观看。外部导出的MP4与PNG用于单独分享。未进行新的种子训练或独立随机初态重新评估。</p></details><footer>实验室资产 · ACP_repro · 真实训练记录分析 / Matplotlib图表 / 已记录物理状态渲染</footer></main></html>'''
    (out/'index.html').write_text(body)
    print(out/'index.html')


if __name__=='__main__':main()
