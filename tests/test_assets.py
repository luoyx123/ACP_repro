import hashlib
import json
from pathlib import Path

from acp.data.assets import PROJECT, read_record, resolve_path


def test_relocation_uses_longest_prefix_and_path_boundaries(tmp_path):
    catalog={'original_project_root':'/old/project','relocations':{
        'outputs/run':'outputs/archive/run','outputs/run/extracted':'outputs/cache/bones'}}
    assert resolve_path('/old/project/outputs/run/extracted/a.npz',root=tmp_path,catalog=catalog)==tmp_path/'outputs/cache/bones/a.npz'
    assert resolve_path('outputs/runner/a.npz',root=tmp_path,catalog=catalog)==tmp_path/'outputs/runner/a.npz'
    assert resolve_path('/external/raw/a.npz',root=tmp_path,catalog=catalog)==Path('/external/raw/a.npz')


def test_historical_record_is_relocated_without_changing_evidence(tmp_path):
    path=tmp_path/'record.json'
    content={'scene':str(PROJECT/'outputs/omni_asap23/clips/x/scene.xml'),
             'sha256':'f'*64,'note':'prose mentioning outputs/omni_asap23 is not a path'}
    path.write_text(json.dumps(content))
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    result=read_record(path)
    assert result['scene']==str(PROJECT/'outputs/数据集/omni/asap23_baseline/clips/x/scene.xml')
    assert result['sha256']==content['sha256'] and result['note']==content['note']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
