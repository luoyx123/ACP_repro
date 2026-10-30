"""为迁入机器生成独立模型路径副本，保留原模型和 manifest 字节。"""
import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from acp.data.assets import PROJECT, atomic_json, machine, registry


def materialize():
    cfg=machine();cache=PROJECT/cfg['portable_model_root']
    mappings={registry()['original_project_root']:str(PROJECT),**cfg.get('path_overrides',{})}
    rows=[]
    for source in sorted((PROJECT/'outputs/数据集').rglob('*')):
        if source.suffix.lower() not in ('.xml','.urdf'):continue
        tree=ET.parse(source);changed=False
        for element in tree.iter():
            for key,value in list(element.attrib.items()):
                if key not in ('filename','file','meshdir','texturedir'):continue
                for before,after in sorted(mappings.items(),key=lambda x:len(x[0]),reverse=True):
                    if value==before or value.startswith(before+'/'):
                        element.set(key,after+value[len(before):]);changed=True;break
        if not changed:continue
        destination=cache/source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        tree.write(destination,encoding='utf-8',xml_declaration=True)
        rows.append(dict(source=str(source.relative_to(PROJECT)),runtime_model=str(destination.relative_to(PROJECT)),
            original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            runtime_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    atomic_json(PROJECT/'outputs/管理/跨机器模型路径副本.json',dict(models=rows,path_mappings=mappings,
        note='Only file-location attributes changed. Original models/manifests remain intact. Runtime model hashes differ; physics calibration remains M2 work.'))
    print(f'生成 {len(rows)} 个机器路径副本；原模型、轨迹和 manifest 未改写')

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    materialize()
