import json
from types import SimpleNamespace

import pytest

from acp.management import simulation


def test_select_physical_gpu_ignores_cuda_mask_and_refuses_other_work(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-busy')
    def query(command, **kwargs):
        if '--query-gpu=index,uuid,name' in command:
            return '0, GPU-free, NVIDIA RTX A5000\n1, GPU-busy, NVIDIA RTX A5000\n'
        return ('GPU-free, 22, /usr/libexec/gnome-remote-desktop-daemon\n'
                'GPU-busy, 33, /env/bin/python\n')
    monkeypatch.setattr(simulation.subprocess, 'check_output', query)
    assert simulation.select_gpu('GPU-free')['index'] == '0'
    with pytest.raises(ValueError, match='已有计算任务'):
        simulation.select_gpu('1')
    with pytest.raises(ValueError, match='无法唯一定位'):
        simulation.select_gpu('GPU-b')


def test_completed_steps_do_not_hide_shutdown_failure(tmp_path, monkeypatch):
    output = tmp_path/'probe'
    monkeypatch.setattr(simulation.sys, 'argv', ['sim-check', '--gpu', '0', '--output', str(output)])
    monkeypatch.setattr(simulation, 'machine', lambda: {'simulation_python': '/env/bin/python'})
    monkeypatch.setattr(simulation, 'select_gpu', lambda _: {'index': '0', 'uuid': 'GPU-free'})
    def crash_after_steps(command, **kwargs):
        (output/'worker.json').write_text(json.dumps({'steps_completed': 1000}))
        return SimpleNamespace(returncode=-11)
    monkeypatch.setattr(simulation.subprocess, 'run', crash_after_steps)
    with pytest.raises(SystemExit) as exc:
        simulation.main()
    assert exc.value.code == 1
    assert json.loads((output/'report.json').read_text())['passed'] is False
