"""隔离单张空闲 GPU，检查 IsaacLab 启动、步进和正常退出。"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from acp.data.assets import PROJECT, atomic_json, machine


def select_gpu(selector):
    """Use physical indices/UUIDs from nvidia-smi, never a masked CUDA ordinal."""
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader,nounits'
    ], text=True)
    rows = [dict(zip(('index', 'uuid', 'name'), (x.strip() for x in row)))
            for row in csv.reader(io.StringIO(output)) if row]
    selected = [row for row in rows if selector in (row['index'], row['uuid'])]
    if len(selected) != 1:
        raise ValueError(f'无法唯一定位物理 GPU：{selector}')
    gpu = selected[0]
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv,noheader,nounits'
    ], text=True)
    desktop = {'gnome-remote-desktop-daemon', 'Xorg', 'Xwayland', 'gnome-shell'}
    blockers = []
    for row in csv.reader(io.StringIO(processes)):
        if len(row) != 3:
            raise ValueError('无法识别 GPU 进程清单，未启动仿真')
        uuid, pid, name = (x.strip() for x in row)
        if uuid == gpu['uuid'] and Path(name).name not in desktop:
            blockers.append(dict(pid=pid, process=name))
    if blockers:
        raise ValueError(f'GPU {gpu["index"]} 已有计算任务，未启动仿真：{blockers}')
    return gpu


def worker(gpu_index, steps, result_path):
    # AppLauncher forwards Kit arguments after SimulationApp's defaults. Rendering
    # uses a physical index; CUDA/PhysX use index 0 within the UUID visibility mask.
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(
        headless=True, device='cuda:0', multi_gpu=False, limit_cpu_threads=4,
        kit_args=f'--/renderer/activeGpu={gpu_index} '
                 '--/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1',
    )
    app = launcher.app
    sim = None
    try:
        import carb.settings
        import torch
        from isaaclab.sim import SimulationCfg, SimulationContext
        settings = carb.settings.get_settings()
        actual = {key: settings.get(key) for key in (
            '/renderer/multiGpu/enabled', '/renderer/multiGpu/autoEnable',
            '/renderer/activeGpu', '/physics/cudaDevice')}
        if (torch.cuda.device_count() != 1 or actual['/renderer/multiGpu/enabled']
                or actual['/renderer/multiGpu/autoEnable']
                or actual['/renderer/activeGpu'] != gpu_index
                or actual['/physics/cudaDevice'] != 0):
            raise RuntimeError(f'GPU 隔离配置未生效：{actual}')
        sim = SimulationContext(SimulationCfg(dt=0.005, device='cuda:0'))
        sim.reset()
        started = time.monotonic()
        for _ in range(steps):
            sim.step(render=False)
        torch.cuda.synchronize()
        atomic_json(result_path, dict(
            steps_completed=steps, step_seconds=time.monotonic()-started,
            settings=actual, cuda_device_count=torch.cuda.device_count(),
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            isaaclab_source=__import__('isaaclab').__file__,
            cleanup_started=True,
        ))
        print(f'ACP_SIM_STEPS_COMPLETED {steps}', flush=True)
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            # Unsubscribe the timeline STOP callback before close_stage triggers
            # STOP. This workspace's callback otherwise renders while paused.
            SimulationContext.clear_instance()
        app.close()
    # Some Kit versions exit from app.close(). The parent records the exit code.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', help='物理 nvidia-smi 编号或完整 GPU UUID；默认读取本机 simulation_gpu')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--output', type=Path, help='新建的检查目录，默认 outputs/管理/仿真预检/<时间>')
    p.add_argument('--_worker', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.steps < 1 or not 0 < args.timeout < float('inf'):
        p.error('steps 和 timeout 必须为有限正数')
    if args._worker is not None:
        # Keep ACP options out of Kit's argument parser.
        sys.argv = [sys.argv[0]]
        worker(args._worker, args.steps, args.output)
        return
    cfg = machine()
    selector = args.gpu or cfg.get('simulation_gpu')
    if not selector:
        p.error('请用 --gpu 指定空闲物理 GPU，或设置本机 simulation_gpu')
    try:
        gpu = select_gpu(str(selector))
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        p.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or PROJECT/'outputs/管理/仿真预检'/stamp).absolute()
    output.mkdir(parents=True, exist_ok=False)
    python = (PROJECT/Path(cfg['simulation_python'])).absolute()
    command = [str(python), '-u', '-m', 'acp.management.simulation', '--_worker',
               str(gpu['index']), '--steps', str(args.steps), '--output', str(output/'worker.json')]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
           'PYTHONPATH': str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH', '')}
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    started = time.monotonic()
    report = dict(gpu=gpu, command=command, steps_requested=args.steps, log=str(output/'run.log'))
    with (output/'run.log').open('w') as log:
        try:
            child = subprocess.run(command, env=env, cwd=PROJECT, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=args.timeout)
            report['exit_code'] = child.returncode
        except (subprocess.TimeoutExpired, OSError) as exc:
            report.update(exit_code=None, error=str(exc))
    if (output/'worker.json').is_file():
        report['worker'] = json.loads((output/'worker.json').read_text())
    report['passed'] = (report['exit_code'] == 0
                        and report.get('worker', {}).get('steps_completed') == args.steps)
    report['seconds'] = time.monotonic()-started
    report['note'] = 'Empty-scene environment check only; G1 dynamics and M2 acceptance remain separate.'
    atomic_json(output/'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
