"""实验室数据资产统一入口；在项目根目录运行 Python scripts/acp_pipeline.py -h。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'src'))
from acp.data.assets import atomic_json, machine, read_record, registry, resolve_path, run_path


def cfg_path(cfg,key):
    return (PROJECT/Path(cfg[key])).absolute()


def snapshot(run):
    counts=Counter()
    for path in (run/'clips').glob('*/manifest.json'):
        try:counts[read_record(path).get('status','unknown')]+=1
        except (json.JSONDecodeError,OSError):counts['writing']+=1
    data=dict(path=str(run),exists=run.is_dir(),clip_statuses=dict(counts))
    for filename in ('source_inventory_summary.json','quality_summary.json','progress.json'):
        try:
            if (run/filename).is_file():data[filename]=read_record(run/filename)
        except json.JSONDecodeError:data[filename]='writing'
    return data


def catalog():
    rows=[]
    for name,info in registry()['runs'].items():
        root=run_path(name)
        files=list(p for p in root.rglob('*') if p.is_file()) if root.is_dir() else []
        sizes=[]
        for p in files:
            try:sizes.append(p.stat().st_size)
            except FileNotFoundError:pass
        rows.append(dict(id=name,**info,exists=root.is_dir(),files=len(sizes),bytes=sum(sizes),
                         clip_statuses=snapshot(root)['clip_statuses']))
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),runs=rows,
                note='Snapshot of registered runs; live batches keep changing. No training admission implied.')
    atomic_json(PROJECT/'outputs/管理/资产目录.json',result)
    lines=['# 数据资产目录','',f"生成时间：{result['generated_at']}。处理完成不等于训练准入。",'',
           '| 稳定 ID | 数据集 | 用途 | 文件数 | GiB | 处理状态 | 位置 |','| --- | --- | --- | ---: | ---: | --- | --- |']
    for r in rows:
        rel=os.path.relpath(PROJECT/r['path'],PROJECT/'outputs/管理')
        lines.append(f"| `{r['id']}` | {r['dataset']} | {r['role']} | {r['files']} | {r['bytes']/2**30:.3f} | {r['clip_statuses']} | [{r['label']}]({rel}) |")
    (PROJECT/'outputs/管理/资产目录.md').write_text('\n'.join(lines)+'\n')
    print('已更新 outputs/管理/资产目录.json 与 .md')


def doctor(cfg):
    checks=[]
    for key,value in cfg.items():
        if key in ('path_overrides','simulation_gpu'):continue
        if key=='blender':ok=bool(shutil.which(value))
        else:ok=cfg_path(cfg,key).exists()
        checks.append(dict(name=key,ok=ok,value=value))
    for label,key,code in [('标注环境','label_python','import numpy,scipy,pydrake,trimesh'),
                           ('重定向环境','retarget_python','import mujoco,cvxpy,holosoma_retargeting')]:
        try:
            result=subprocess.run([str(cfg_path(cfg,key)),'-c',code],capture_output=True,text=True,timeout=30)
            checks.append(dict(name=label,ok=result.returncode==0,error=result.stderr[-1500:]))
        except (OSError,subprocess.TimeoutExpired) as exc:checks.append(dict(name=label,ok=False,error=str(exc)))
    for r in checks:print(('通过 ' if r['ok'] else '缺失 ')+r['name']+' '+str(r.get('value',r.get('error',''))))
    return 0 if all(r['ok'] for r in checks) else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,help='本机资源配置，默认 configs/machine.local.json')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='查看登记的数据集与稳定批次 ID')
    sub.add_parser('doctor',help='检查本机路径和两套 Python 环境')
    sub.add_parser('catalog',help='生成登记资产的文件数量、体积和状态快照')
    for name in ('status','view','evaluate'):
        sp=sub.add_parser(name,help={'status':'读取处理状态','view':'交互回放或导出 MP4/PNG','evaluate':'全量离线质量检查'}[name])
        sp.add_argument('asset',help='数据集名、稳定批次 ID 或目录')
        sp.add_argument('--workers',type=int,default=4)
        sp.add_argument('--clip')
        sp.add_argument('--output',type=Path)
        sp.add_argument('--show-source',action='store_true')
        sp.add_argument('--dry-run',action='store_true')
    sp=sub.add_parser('run',help='处理数据；已完成且指纹一致的结果会复用')
    sp.add_argument('dataset',choices=('omni','lafan1','mixamo','php'))
    sp.add_argument('--output',type=Path,help='试运行应指定新目录，避免覆盖主批次清单')
    sp.add_argument('--workers',type=int,default=4)
    sp.add_argument('--fps',type=float,default=50.)
    sp.add_argument('--clips',nargs='+')
    sp.add_argument('--all-variants',action='store_true',help='PHP：处理全部地形变体')
    sp.add_argument('--audit-only',action='store_true',help='PHP：仅生成格式清单')
    sp.add_argument('--dry-run',action='store_true')
    sp.add_argument('--background',action='store_true',help='后台执行，记录 PID、命令与日志')
    sp.add_argument('--label-timeout',type=int,default=120,help='每条 Drake 标注子进程最长秒数，默认 120')
    # Advanced functions share this same entry; their parsers own their options.
    operations={'sim-check':'acp.management.simulation','materialize':'acp.management.portable','structure':'acp.management.structure','check':'acp.management.check','repair':'acp.quality.repair',
                'finalize':'acp.quality.finalize','compare':'acp.quality.compare',
                'fit-scenes':'acp.pipeline.scene_fit','match-names':'acp.pipeline.names',
                'migrate':'acp.management.organize'}
    for name in operations:
        sub.add_parser(name,add_help=False,help='专项操作：'+name+' --help')
    args,extra=p.parse_known_args()
    if args.command in operations:
        if args.config:os.environ['ACP_MACHINE_CONFIG']=str(args.config.absolute())
        cfg=machine()
        python=sys.executable if args.command in ('check','migrate','structure','materialize','sim-check') else str(cfg_path(cfg,'retarget_python'))
        raise SystemExit(subprocess.call([python,'-m',operations[args.command],*extra],cwd=PROJECT,
            env={**os.environ,'PYTHONPATH':str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')}))
    if args.config:os.environ['ACP_MACHINE_CONFIG']=str(args.config.absolute())
    cfg=machine();reg=registry()
    if args.command=='list':
        for key,d in reg['datasets'].items():print(f"{key:8} {d['label']} → {d['run_id']} → {reg['runs'][d['run_id']]['path']}")
        print('\n其他稳定 ID：'+', '.join(reg['runs']));return
    if args.command=='doctor':raise SystemExit(doctor(cfg))
    if args.command=='catalog':catalog();return
    if args.command=='run':
        if args.label_timeout<1:p.error('label-timeout 必须为正数')
        d=reg['datasets'][args.dataset]
        run=(args.output.absolute() if args.output else run_path(d['run_id']))
        command=[str(cfg_path(cfg,'retarget_python')),str(PROJECT/d['script']),
                 '--dataset-root',str(cfg_path(cfg,d['source_key'])),'--output',str(run),
                 '--workers',str(args.workers),'--fps',str(args.fps)]
        if args.dataset in ('omni','lafan1','php'):
            command+=['--asap-root',str(cfg_path(cfg,'asap_root')),'--label-python',str(PROJECT/'src/acp/pipeline/label_python_bounded.sh')]
        if args.dataset in ('lafan1','php'):
            command+=['--reference-urdf',str(cfg_path(cfg,'omni_dataset')/'models/g1/g1_29dof_spherehand.urdf')]
        if args.dataset=='php':
            command+=['--php-root',str(cfg_path(cfg,'php_root'))]
            if args.all_variants:command+=['--all-variants']
            if args.audit_only:command+=['--audit-only']
        elif args.all_variants or args.audit_only:p.error('all-variants/audit-only 仅适用于 PHP')
        if args.dataset=='mixamo':
            command[1]=str(PROJECT/'src/acp/pipeline/mixamo_resources.py')
            support=PROJECT/'outputs/管理/运行配置/mixamo_supports.json'
            # Resolve historical manifest paths without editing source configuration evidence.
            if not args.dry_run:atomic_json(support,read_record(PROJECT/'configs/mixamo_test_scenes.json'))
            command+=['--extracted-root',str(PROJECT/'outputs/缓存/mixamo/fbx_skeletons_v1'),
                      '--support-scenes',str(support)]
        if args.clips:command+=['--clips',*args.clips]
    else:
        name=reg['datasets'].get(args.asset,{}).get('run_id',args.asset)
        run=run_path(name)
        if args.command=='status':
            print(json.dumps(snapshot(run),ensure_ascii=False,indent=2));return
        script='src/acp/visualization/viewer.py' if args.command=='view' else 'src/acp/quality/evaluate.py'
        command=[str(cfg_path(cfg,'retarget_python')),str(PROJECT/script),'--run',str(run)]
        if args.command=='view':
            command+=['--clip',args.clip] if args.clip else ['--list']
            if args.output:command+=['--output',str(args.output.absolute())]
            if args.show_source:command+=['--show-source']
        else:command+=['--workers',str(args.workers)]
    command+=extra
    print(shlex.join(command),flush=True)
    if args.dry_run:return
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    if not env.get('DISPLAY'):env.setdefault('MUJOCO_GL','egl')
    if args.command=='run':
        env['ACP_LABEL_REAL_PYTHON']=str(cfg_path(cfg,'label_python'))
        env['ACP_LABEL_TIMEOUT_SECONDS']=str(args.label_timeout)
        env['ACP_BOUNDED_LABEL_EXECUTABLE']=str(PROJECT/'src/acp/pipeline/label_python_bounded.sh')
        run.mkdir(parents=True,exist_ok=True)
        pidfile=run/'batch.pid'
        if pidfile.exists():
            try:os.kill(int(pidfile.read_text()),0)
            except (ValueError,ProcessLookupError):pass
            else:raise SystemExit(f'批次仍在运行：{pidfile.read_text().strip()}。请用 status 查看，或指定新的 --output。')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        record=dict(argv=command,cwd=str(PROJECT),configuration=cfg,started_at=stamp,
                    label_timeout_seconds=args.label_timeout,
                    launcher_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
                    resource_adapter_sha256={name:__import__('hashlib').sha256((PROJECT/name).read_bytes()).hexdigest()
                        for name in ('src/acp/pipeline/mixamo_resources.py','src/acp/pipeline/label_python_bounded.sh','src/acp/data/assets.py')})
        if args.background:
            log=run/f'batch_{stamp}.log'
            with log.open('w') as stream:
                child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(pid=child.pid,log=str(log));pidfile.write_text(str(child.pid)+'\n')
            atomic_json(run/'launches'/f'{stamp}.json',record)
            print(f'已启动 PID {child.pid}；日志 {log}');return
        atomic_json(run/'launches'/f'{stamp}.json',record)
    raise SystemExit(subprocess.call(command,cwd=PROJECT,env=env))

if __name__=='__main__':main()
