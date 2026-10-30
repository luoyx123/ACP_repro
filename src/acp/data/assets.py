"""Stable asset IDs and relocatable historical records, without changing evidence."""
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
REGISTRY = PROJECT / 'configs/assets_registry.json'


def registry():
    data = json.loads(REGISTRY.read_text()) if REGISTRY.is_file() else {'runs':{},'relocations':{}}
    data.setdefault('relocations', {}).update(machine().get('path_overrides', {}))
    return data


def machine():
    path=Path(os.environ.get('ACP_MACHINE_CONFIG',PROJECT/'configs/machine.local.json'))
    if not path.is_file():path=PROJECT/'configs/machine.example.json'
    return json.loads(path.read_text())


def resolve_path(value, *, root=None, catalog=None):
    """Map only path-prefix boundaries, longest first; never replace prose/hashes."""
    root=Path(root or PROJECT)
    catalog=catalog or registry()
    text=str(value)
    original_root=catalog.get('original_project_root',str(root))
    if text==original_root or text.startswith(original_root+'/'):
        text=text[len(original_root):].lstrip('/')
    elif text.startswith(str(root)+'/'):
        text=text[len(str(root))+1:]
    for before,after in sorted(catalog.get('relocations',{}).items(),key=lambda item:len(item[0]),reverse=True):
        if text==before or text.startswith(before+'/'):
            return materialized_path(root/(after+text[len(before):]))
    path=Path(text)
    if not path.is_absolute():return materialized_path(root/path)
    return materialized_path(path)


def materialized_path(path):
    """Use a separate machine-local model copy; retain original model evidence."""
    cache = machine().get('portable_model_root')
    if cache and path.suffix.lower() in ('.xml', '.urdf') and path.is_relative_to(PROJECT):
        candidate = PROJECT / cache / path.relative_to(PROJECT)
        if candidate.is_file():return candidate
    return path


def relocate_values(value, *, catalog=None):
    catalog=catalog or registry()
    if isinstance(value,dict):return {k:relocate_values(v,catalog=catalog) for k,v in value.items()}
    if isinstance(value,list):return [relocate_values(v,catalog=catalog) for v in value]
    if isinstance(value,str):
        prefixes=(catalog.get('original_project_root',str(PROJECT))+'/',str(PROJECT)+'/')
        if value.startswith(prefixes) or any(value==p or value.startswith(p+'/') for p in catalog.get('relocations',{})):
            return str(resolve_path(value,catalog=catalog))
    return value


def read_record(path):
    """Resolve locations at read time. Historical JSON bytes/SHA-256 stay intact."""
    return relocate_values(json.loads(resolve_path(path).read_text()))


def run_path(run_id):
    data=registry()
    if run_id in data['runs']:return PROJECT/data['runs'][run_id]['path']
    return resolve_path(run_id,catalog=data)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
