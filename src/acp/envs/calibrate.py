"""M2 acceptance runner: real G1 reset/PD/contact/terrain on one selected GPU."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from acp.data.assets import PROJECT, atomic_json, machine, registry, run_path
from acp.management.simulation import select_gpu


def gpu_memory_mib(uuid):
    return int(subprocess.check_output(['nvidia-smi', '--id='+uuid,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5).strip())


def render_rollouts(folder):
    """Render recorded PhysX states with MuJoCo; never run MuJoCo dynamics."""
    import numpy as np
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from acp.data.playback import resolve_clip
    record = json.loads((folder/'worker.json').read_text())
    name = registry()['datasets'].get(record['asset'], {}).get('run_id', record['asset'])
    info = resolve_clip(run_path(name), record['clip'])
    model = mujoco.MjModel.from_xml_path(str(info['scene']));data = mujoco.MjData(model)
    model.vis.global_.offwidth = 768;model.vis.global_.offheight = 512
    joint_addresses = [model.jnt_qposadr[model.joint(n).id] for n in record['canonical_joints']]
    renderer = mujoco.Renderer(model, height=512, width=768)
    camera = mujoco.MjvCamera();camera.distance = 2.7;camera.azimuth = 135;camera.elevation = -20
    outputs = []
    try:
        for filename in ('stand_rollout', 'reference_replay'):
            with np.load(folder/(filename+'.npz')) as z: states=z['states'];dt=float(z['dt'])
            path = folder/(filename+'.mp4')
            with imageio.get_writer(path, fps=1/(2*dt), codec='libx264', quality=7) as video:
                for i in range(0, len(states), 2):
                    state = states[i];data.qpos[:7] = state[:7];data.qpos[joint_addresses] = state[13:]
                    mujoco.mj_forward(model, data);camera.lookat[:] = state[:3]
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render());draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 768, 48), fill=(20, 25, 30))
                    draw.text((10, 6), f'Recorded PhysX states | {filename} | t={(i+1)*dt:.2f}s', fill='white')
                    draw.text((10, 27), 'MuJoCo rendering only | untrained PD, no learned policy', fill='white')
                    video.append_data(np.asarray(frame))
                    if i == len(states)//2//2*2: frame.save(folder/(filename+'.png'))
            outputs.append(str(path))
    finally: renderer.close()
    atomic_json(folder/'render.json', dict(source='Recorded PhysX states; MuJoCo visualization only', videos=outputs))


def render_run(output, cfg, selector, timeout):
    gpu = select_gpu(str(selector))
    command = [str((PROJECT/Path(cfg['simulation_python'])).absolute()), '-m', 'acp.envs.calibrate',
               '--_render', '--output', str(output/'envs_1')]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':gpu['uuid'], 'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
           'PYTHONPATH':str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH', ''), 'MUJOCO_GL':'egl'}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    with (output/f'render_{stamp}.log').open('w') as log:
        result = subprocess.run(command, env=env, cwd=PROJECT,
            stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return result.returncode


def worker(args):
    import numpy as np
    from isaaclab.app import AppLauncher
    sys.argv = [sys.argv[0]]
    launcher = AppLauncher(headless=True, device='cuda:0', multi_gpu=False, limit_cpu_threads=4,
        kit_args=f'--/renderer/activeGpu={args._worker} --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1')
    app = launcher.app;env = None
    try:
        import torch
        from acp.data.playback import resolve_clip
        from acp.envs.g1 import G1ReferenceEnv
        name = registry()['datasets'].get(args.asset, {}).get('run_id', args.asset)
        info = resolve_clip(run_path(name), args.clip)
        cfg = json.loads((PROJECT/'configs/m2.json').read_text())
        env = G1ReferenceEnv(info, cfg, args.num_envs[0])
        report = dict(num_envs=env.num_envs, asset=args.asset, clip=args.clip,
                      simulator_joints=env.robot.joint_names, simulator_bodies=env.robot.body_names,
                      canonical_joints=env.joint_names, body_names=env.body_names,
                      removed_massless_links=env.removed_massless_links, checks={})
        import isaaclab
        import carb
        from acp.data.assets import resolve_path
        files = dict(config=PROJECT/'configs/m2.json', motion=info['motion'], scene=info['scene'],
                     source_urdf=resolve_path(env.manifest['target_urdf']), imported_urdf=env.urdf,
                     usd=env.cache/'usd/g1.usd')
        report['provenance'] = dict(config=cfg, files={k:dict(path=str(v), sha256=hashlib.sha256(v.read_bytes()).hexdigest())
            for k,v in files.items()}, torch=torch.__version__, isaaclab_import=isaaclab.__file__,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cuda_device_count=torch.cuda.device_count(),
            renderer_multi_gpu=carb.settings.get_settings().get('/renderer/multiGpu/enabled'))
        report['checks']['single_gpu'] = torch.cuda.device_count() == 1 and not report['provenance']['renderer_multi_gpu']
        env.reset()
        zero = torch.zeros((env.num_envs, 23), device=env.device)
        states, forces = [], []
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start = time.monotonic()
        min_height, max_tilt, min_geom_z = float('inf'), 0.0, float('inf')
        for i in range(args.policy_steps):
            obs = env.step(zero)
            if not torch.isfinite(obs).all(): raise ValueError('Nonfinite simulation observation')
            min_height = min(min_height, float(env.robot.data.root_link_pos_w[:, 2].min()))
            max_tilt = max(max_tilt, float((env.robot.data.projected_gravity_b[:, :2].norm(dim=-1)).max()))
            if i % 10 == 0: min_geom_z = min(min_geom_z, float(env.minimum_geometry_z().min()))
            states.append(torch.cat((env.robot.data.root_link_state_w[0], env.robot.data.joint_pos[0, env.joint_ids])).cpu().numpy())
            forces.append(env.force_history[0].cpu().numpy())
        torch.cuda.synchronize();elapsed = time.monotonic()-start
        stand = dict(minimum_root_height=min_height, maximum_gravity_xy=max_tilt,
                     minimum_geometry_z=min_geom_z, policy_steps=args.policy_steps,
                     elapsed_seconds=elapsed, environment_policy_steps_per_second=args.policy_steps*env.num_envs/elapsed,
                     peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                     final_contact_forces=env.contact_forces()[0].cpu().tolist())
        report['stand'] = stand
        report['checks']['standing'] = min_height > .5 and max_tilt < .3 and min_geom_z > -.01
        hits, heights, valid = env.height_scan()
        phits, pheights, pvalid = env.height_scan(planner=True)
        report['scan'] = dict(tracker_shape=list(heights.shape), planner_shape=list(phits.shape),
                              valid_fraction=float(valid.float().mean()), planner_valid_fraction=float(pvalid.float().mean()))
        report['checks']['scan'] = bool(valid.all() and pvalid.all() and heights.shape[-1] == 63 and phits.shape[1:] == (187, 3))
        report['checks']['observation'] = list(env.proprio().shape) == [env.num_envs, 75] and list(env.history.shape) == [env.num_envs, 5, 75]
        # Compare imported inertias and limits with the real source asset.
        mass = float(env.robot.root_physx_view.get_masses()[0].sum())
        source_mass = float(env.model.body_mass.sum())
        limits = env.robot.data.joint_effort_limits[:, env.joint_ids]
        source_joint_ids = [env.model.joint(n).id for n in cfg['joint_names']]
        position_limits = torch.tensor(env.model.jnt_range[source_joint_ids], device=env.device)
        report['asset_properties'] = dict(physx_mass=mass, source_mass=source_mass,
            maximum_effort_limit_error=float((limits-torch.tensor(cfg['effort_limits'], device=env.device)).abs().max()),
            maximum_position_limit_error=float((env.robot.data.joint_pos_limits[:, env.joint_ids]-position_limits).abs().max()),
            maximum_velocity_limit_error=float((env.robot.data.joint_vel_limits[:, env.joint_ids]-torch.tensor(cfg['velocity_limits'], device=env.device)).abs().max()))
        report['checks']['asset_properties'] = abs(mass-source_mass) < .002 and all(
            report['asset_properties'][key] < 1e-4 for key in
            ('maximum_effort_limit_error', 'maximum_position_limit_error', 'maximum_velocity_limit_error'))
        # Random-in-time reset + FK: no physics step before measurement, no teleport replay success claim.
        reset_errors = [];velocity_errors = [];fk_errors = []
        rng = np.random.default_rng(20261030)
        for _ in range(8):
            indices = rng.integers(0, len(env.reference.times), size=env.num_envs)
            times = env.reference.times[indices]
            expected_root, expected_q, expected_qd = env.reset(times)
            env.sim.forward();env.robot.update(0.0)
            values = env.reference.sample(times)
            reset_errors.append(float((env.robot.data.joint_pos[:, env.joint_ids]-expected_q).abs().max()))
            reset_errors.append(float((env.robot.data.root_link_pose_w-expected_root[:, :7]).abs().max()))
            velocity_errors.append(float((env.robot.data.root_link_vel_w-expected_root[:, 7:]).abs().max()))
            velocity_errors.append(float((env.robot.data.joint_vel[:, env.joint_ids]-expected_qd).abs().max()))
            expected_body = torch.tensor(values['body_landmarks_w'], device=env.device, dtype=torch.float32)+env.origins[:, None]
            fk_errors.append(float((env.body_positions()-expected_body).norm(dim=-1).max()))
        report['reference_reset'] = dict(position_error=max(reset_errors), velocity_error=max(velocity_errors),
                                        maximum_landmark_error=max(fk_errors), samples=8*env.num_envs)
        report['checks']['reference_reset'] = max(reset_errors) < 1e-4 and max(velocity_errors) < 1e-3 and max(fk_errors) < .002
        # Explicit actuator clipping must remain active even for extreme finite actions.
        env.reset();env.step(torch.full_like(zero, 100.0))
        applied = env.robot.data.applied_torque[:, env.joint_ids]
        computed = env.robot.data.computed_torque[:, env.joint_ids]
        limit = torch.tensor(cfg['effort_limits'], device=env.device)
        report['actuator'] = dict(maximum_applied_torque=float(applied.abs().max()),
            maximum_limit_excess=float((applied.abs()-limit).max()), saturated_joints=int((computed.abs()>limit).sum()))
        report['checks']['actuator'] = report['actuator']['maximum_limit_excess'] < 1e-4 and report['actuator']['saturated_joints'] > 0
        # Static scene query and the GPU scan must see the same triangle surfaces.
        env.reset(np.full(env.num_envs, env.reference.times[0]));env.sim.forward();env.robot.update(0.)
        map_hits = env.height_scan(planner=True)[0][0].clone().cpu().numpy()
        pose = env.robot.data.root_link_pose_w.clone();pose[:, 2] = -10
        env.robot.write_root_link_pose_to_sim(pose);env.sim.forward()
        import omni.physx
        query = omni.physx.get_physx_scene_query_interface()
        scan_errors = []
        for point in map_hits[::13]:
            origin = (float(point[0]), float(point[1]), float(point[2]+20))
            hit = query.raycast_closest(origin, (0., 0., -1.), 100.)
            if not hit['hit']: raise ValueError('PhysX terrain ray missed GPU scan surface')
            scan_errors.append(abs(float(hit['position'][2])-float(point[2])))
        report['scan']['maximum_physx_warp_height_error'] = max(scan_errors)
        report['checks']['terrain_registration'] = max(scan_errors) < .002
        # Test each group's accessible collision surface in isolation. Knee
        # fixtures overlap the top by 3mm; other groups close a 1mm gap by gravity.
        import mujoco
        from acp.envs.conversion import geom_mesh
        peaks = [];minimum_peaks = [];probe_vectors = [];all_group_peaks = []
        for group in range(9):
            env.park_probe();env.reset()
            pose = env.robot.data.root_link_pose_w.clone()
            pose[:, :3] = env.origins+torch.tensor([-5., -5., 1.5], device=env.device)
            env.robot.write_root_link_pose_to_sim(pose)
            env.mjdata.qpos[:] = np.r_[-5, -5, 1.5, 1, 0, 0, 0, cfg['default_joint_positions']]
            mujoco.mj_forward(env.model, env.mjdata)
            candidates = []
            for g in range(env.model.ngeom):
                name = env.model.body(env.model.geom_bodyid[g]).name
                if env.groups.get(name) != group or not (env.model.geom_contype[g] or env.model.geom_conaffinity[g]): continue
                v, _ = geom_mesh(env.model, env.mjdata, g);candidates.append(v)
            vertices = np.concatenate(candidates)
            # The knee group's bottom is occluded by the lower leg/foot. Test
            # its accessible top surface; test the other groups from below.
            above = group in (2, 3)
            top = vertices[np.argmax(vertices[:, 2]) if above else np.argmin(vertices[:, 2])]
            target = env.probe.data.default_root_state.clone()
            target[:, :3] = torch.tensor(top+[0., 0., .017 if above else -.021], device=env.device)+env.origins
            env.probe.write_root_pose_to_sim(target[:, :7])
            maximum = 0.0;vector = None;group_peaks = torch.zeros(9, device=env.device)
            per_env_peak = torch.zeros(env.num_envs, device=env.device)
            for _ in range(5):
                env.step(zero)
                f = env.force_history[:, :, group]
                norms = f.norm(dim=-1);value = float(norms.max())
                per_env_peak = torch.maximum(per_env_peak, norms.amax(dim=1))
                group_peaks = torch.maximum(group_peaks, env.force_history.norm(dim=-1).amax(dim=(0, 1)))
                if value > maximum:
                    maximum = value;vector = f.reshape(-1, 3)[int(norms.argmax())].cpu().tolist()
            peaks.append(maximum);minimum_peaks.append(float(per_env_peak.min()))
            probe_vectors.append(vector);all_group_peaks.append(group_peaks.cpu().tolist())
        env.park_probe()
        report['contact_fixtures'] = dict(peak_force_norms=peaks, peak_vectors=probe_vectors,
            minimum_peak_over_environments=minimum_peaks, body_order=cfg['body_names'],
            bottom_fixture_clearance_m=.001, knee_top_fixture_overlap_m=.003, all_group_peaks=all_group_peaks)
        report['checks']['nine_contact_groups'] = min(minimum_peaks) > 1.0
        report['checks']['support_force_direction'] = float(np.asarray(forces)[-50:, :, :2, 2].mean(axis=(0, 1)).sum()) > 1.0
        # A short untrained PD replay checks the live data/control path. Tracking
        # success belongs to Stage 1, not this interface acceptance test.
        env.reset(np.full(env.num_envs, env.reference.times[0]))
        replay = [];replay_reference = []
        before = env.sim.current_time
        for i in range(25):
            t = min(env.reference.times[0]+(i+1)*env.dt*env.decimation, env.reference.times[-1])
            ref = env.reference.sample(np.full(env.num_envs, t))
            q = torch.tensor(ref['joint_positions'], device=env.device, dtype=torch.float32)
            obs = env.step((q-env.default)/cfg['action_scale'])
            if not torch.isfinite(obs).all(): raise ValueError('Nonfinite reference replay')
            replay.append(torch.cat((env.robot.data.root_link_state_w[0], env.robot.data.joint_pos[0, env.joint_ids])).cpu().numpy())
            replay_reference.append(np.r_[ref['base_position_w'][0], ref['joint_positions'][0]])
        actual_dt = (env.sim.current_time-before)/25
        report['reference_replay'] = dict(policy_steps=25, measured_policy_dt=actual_dt, finite=True,
            joint_rmse=float(np.sqrt(np.mean((np.asarray(replay)[:, 13:]-np.asarray(replay_reference)[:, 3:])**2))),
            note='Untrained PD interface check; no motion success claim.')
        report['checks']['reference_replay'] = abs(actual_dt-.02) < 1e-8
        np.savez_compressed(args.output/'reference_replay.npz', states=np.asarray(replay), reference=np.asarray(replay_reference),
            dt=.02, joint_names=env.joint_names, body_names=env.body_names)
        report['passed'] = all(report['checks'].values())
        np.savez_compressed(args.output/'stand_rollout.npz', states=np.asarray(states), contact_force_history=np.asarray(forces),
                            dt=cfg['physics_dt']*cfg['decimation'], joint_names=env.joint_names, body_names=env.body_names)
        atomic_json(args.output/'worker.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        atomic_json(args.output/'worker.json', dict(passed=False, error=str(exc)))
    finally:
        from isaaclab.sim import SimulationContext
        if env is not None: env.close()
        elif SimulationContext.instance() is not None:
            SimulationContext.instance().clear_all_callbacks();SimulationContext.clear_instance()
        app.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu')
    p.add_argument('--asset', default='omni')
    p.add_argument('--clip', default='climb_00_z_scale_0.9')
    p.add_argument('--num-envs', type=int, nargs='+', default=[1, 16, 64])
    p.add_argument('--policy-steps', type=int, default=200)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--output', type=Path)
    p.add_argument('--render', action='store_true', help='导出单环境实际 PhysX 轨迹视频（MuJoCo 仅负责渲染）')
    p.add_argument('--render-only', action='store_true', help='仅重新导出 --output 中已有的单环境物理轨迹')
    p.add_argument('--_worker', type=int, help=argparse.SUPPRESS)
    p.add_argument('--_render', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.num_envs) < 1 or args.policy_steps < 1: p.error('Positive environment count/steps required')
    if args._render: render_rollouts(args.output);return
    if args._worker is not None: worker(args);return
    cfg = machine();selector = args.gpu or cfg.get('simulation_gpu')
    if not selector: p.error('Specify an idle physical GPU with --gpu')
    if args.render_only:
        if not args.output: p.error('--render-only requires --output pointing to an existing run')
        output = args.output.absolute();report = json.loads((output/'report.json').read_text())
        result = render_run(output, cfg, selector, args.timeout)
        report.setdefault('render_previous_exit_codes', []).append(report.get('render_exit_code'))
        report['render_exit_code'] = result
        report['passed'] = result == 0 and bool(report['results']) and all(r['passed'] for r in report['results'])
        atomic_json(output/'report.json', report)
        raise SystemExit(result)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or PROJECT/'outputs/实验归档/m2'/stamp).absolute()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for n in args.num_envs:
        gpu = select_gpu(str(selector))
        folder = output/f'envs_{n}';folder.mkdir()
        command = [str((PROJECT/Path(cfg['simulation_python'])).absolute()), '-u', '-m', 'acp.envs.calibrate',
            '--_worker', gpu['index'], '--asset', args.asset, '--clip', args.clip, '--num-envs', str(n),
            '--policy-steps', str(args.policy_steps), '--output', str(folder)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
               'PYTHONPATH': str(PROJECT/'src')+os.pathsep+os.environ.get('PYTHONPATH', ''),
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        row = dict(num_envs=n, gpu=gpu, command=command, log=str(folder/'run.log'))
        baseline = gpu_memory_mib(gpu['uuid']);memory_samples = [baseline];stop = threading.Event()
        def monitor():
            while not stop.wait(.5):
                try: memory_samples.append(gpu_memory_mib(gpu['uuid']))
                except (OSError, ValueError, subprocess.SubprocessError): pass
        thread = threading.Thread(target=monitor, daemon=True);thread.start()
        with (folder/'run.log').open('w') as log:
            try:
                child = subprocess.run(command, env=env, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                row['exit_code'] = child.returncode
            except (OSError, subprocess.TimeoutExpired) as exc: row.update(exit_code=None, error=str(exc))
        stop.set();thread.join(timeout=6)
        row.update(gpu_baseline_mib=baseline, gpu_peak_total_mib=max(memory_samples),
                   gpu_peak_increment_mib=max(memory_samples)-baseline)
        if (folder/'worker.json').is_file(): row['worker'] = json.loads((folder/'worker.json').read_text())
        row['passed'] = row['exit_code'] == 0 and row.get('worker', {}).get('passed', False)
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'worker'}, ensure_ascii=False), flush=True)
        if not row['passed']: break
    report = dict(results=results, passed=len(results) == len(args.num_envs) and all(r['passed'] for r in results))
    if args.render and report['passed'] and 1 in args.num_envs:
        report['render_exit_code'] = render_run(output, cfg, selector, args.timeout)
        report['passed'] = report['render_exit_code'] == 0
    atomic_json(output/'report.json', report)
    print(output/'report.json')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__': main()
