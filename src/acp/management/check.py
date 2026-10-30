"""只读检查：文档链接、分类目录、登记资产及完成片段的关键路径。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'src'))
from acp.data.assets import read_record,registry,run_path


def check(manifests=False):
    errors=[];counts=Counter()
    files=[PROJECT/'README.md',PROJECT/'outputs/README.md',*(PROJECT/'docs').rglob('*.md')]
    for path in files:
        counts['markdown_files']+=1
        for match in re.finditer(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',path.read_text()):
            target=unquote(match.group(1)).split('#')[0].strip('<>')
            if not target or re.match(r'^[a-zA-Z]+:',target):continue
            counts['local_links']+=1
            if not (path.parent/target).exists():errors.append(f'链接缺失：{path.relative_to(PROJECT)} -> {target}')
    expected={'数据集','缓存','实验归档','可视化','管理'}
    for path in (PROJECT/'outputs').iterdir():
        if path.is_dir() and path.name not in expected:errors.append(f'未分类输出目录：{path.name}')
    for name,info in registry()['runs'].items():
        run=run_path(name);counts['registered_runs']+=1
        if not run.is_dir():errors.append(f'登记资产缺失：{name} -> {run}')
        if not manifests or info['role']!='dataset':continue
        for path in (run/'clips').glob('*/manifest.json'):
            try:record=read_record(path)
            except json.JSONDecodeError:counts['writing_manifests']+=1;continue
            if not record.get('status','').startswith('end_to_end'):continue
            counts['completed_manifests']+=1
            for key in ('source','scene','target_urdf','source_motion_npz','source_robot_xml'):
                if record.get(key) and not Path(record[key]).exists():errors.append(f'路径缺失：{path.relative_to(PROJECT)} {key} -> {record[key]}')
    return dict(counts=counts,errors=errors,ok=not errors)

def playback_audit():
    """Check every indexed motion schema and source overlay path, without GL."""
    import numpy as np
    from acp.data.playback import resolve_clip
    report={}
    for dataset,entry in registry()['datasets'].items():
        run=run_path(entry['run_id']);rows=read_record(run/'artifact_index.json') if (run/'artifact_index.json').exists() else []
        result=dict(indexed=len(rows),playable=0,source_overlay_ready=0,without_motion=0,errors=[])
        for row in rows:
            if not row.get('motion'):
                result['without_motion']+=1;continue
            try:
                motion=run/row['motion']
                artifact=motion.parent.parent
                record=read_record(artifact/'manifest.json')
                scene=Path(record['scene'])
                if not scene.is_file():raise FileNotFoundError(scene)
                info=dict(motion=motion,artifact=artifact,source=Path(record['source']))
                with np.load(info['motion'],allow_pickle=False) as z:
                    t=z['timestamps'];n=len(t)
                    if (n<2 or not np.all(np.diff(t)>0) or z['joint_positions'].shape!=(n,23)
                            or z['base_position_w'].shape!=(n,3) or z['base_quaternion_wxyz'].shape!=(n,4)
                            or len(z['joint_names'])!=23):raise ValueError('Invalid playback motion schema')
                result['playable']+=1
                record=read_record(info['artifact']/'manifest.json')
                if record.get('source_format')=='mixamo_fbx_world_bones':
                    ready=(info['artifact']/'human_reference.npz').is_file()
                elif record.get('source_motion_npz'):
                    ready=Path(record['source_motion_npz']).is_file() and Path(record['source_robot_xml']).is_file()
                else:ready=info['source'].is_file()
                if ready:result['source_overlay_ready']+=1
                else:raise FileNotFoundError('Source overlay is missing')
            except Exception as exc:result['errors'].append(dict(clip=row['clip'],error=f'{type(exc).__name__}: {exc}'))
        report[dataset]=result
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests',action='store_true')
    parser.add_argument('--playback',action='store_true',help='核验四类数据所有已索引轨迹的播放格式与源叠加文件')
    args=parser.parse_args()
    result=check(args.manifests)
    if args.playback:
        result['playback']=playback_audit()
        result['ok']=result['ok'] and all(not x['errors'] for x in result['playback'].values())
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['ok'] else 1)
