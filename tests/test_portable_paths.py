import hashlib
from pathlib import Path
from acp.data import assets
from acp.management import portable


def test_external_source_and_runtime_models_preserve_original(tmp_path, monkeypatch):
    original=tmp_path/'outputs/数据集/demo/assets/robot.xml'
    original.parent.mkdir(parents=True)
    original.write_text('<mujoco><asset><mesh file="/old/raw/model.stl"/></asset></mujoco>')
    before=hashlib.sha256(original.read_bytes()).hexdigest()
    cfg={'path_overrides':{'/old/raw':str(tmp_path/'datasets/raw')},'portable_model_root':'outputs/缓存/portable'}
    monkeypatch.setattr(assets,'PROJECT',tmp_path)
    monkeypatch.setattr(assets,'REGISTRY',tmp_path/'registry.json')
    assets.REGISTRY.write_text('{"original_project_root":"/old/project","relocations":{}}')
    monkeypatch.setattr(assets,'machine',lambda:cfg)
    monkeypatch.setattr(portable,'PROJECT',tmp_path)
    monkeypatch.setattr(portable,'machine',lambda:cfg)
    portable.materialize()
    mapped=assets.resolve_path('/old/project/outputs/数据集/demo/assets/robot.xml')
    assert mapped!=original and str(tmp_path/'datasets/raw/model.stl') in mapped.read_text()
    assert assets.resolve_path('/old/raw/a.npz')==tmp_path/'datasets/raw/a.npz'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==before
