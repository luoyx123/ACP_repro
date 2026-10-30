"""Snapshot all-input extraction/retargeting counts without waiting for batch completion."""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from acp.data.assets import read_record, resolve_path, run_path
import argparse
from collections import Counter
import json
from pathlib import Path


def status(run, dataset):
    run=resolve_path(run)
    sources=sorted((dataset/'animation').glob('*.fbx'))
    counts=Counter();extraction=Counter();frames=0;failed=[]
    inventory={r['clip']:r for r in read_record(run/'inventory.json')} if (run/'inventory.json').is_file() else {}
    for source in sources:
        extracted=Path(inventory.get(source.stem,{}).get('extracted',run/'extracted'/(source.stem+'.npz')))
        metadata=extracted.with_suffix('.json')
        if metadata.is_file():
            meta=read_record(metadata);extraction[meta['status']]+=1
            frames+=meta.get('frames',0)
        else:extraction['pending']+=1
        manifest=run/'clips'/source.stem/'manifest.json'
        if not manifest.is_file():counts['pending']+=1;continue
        try:m=read_record(manifest)
        except json.JSONDecodeError:counts['writing']+=1;continue
        counts[m['status']]+=1
        if m['status']=='failed':failed.append({'clip':source.stem,'error':m.get('error')})
    summary={'expected_animations':len(sources),'extraction':dict(extraction),'extracted_source_frames':frames,
             'retargeting':dict(counts),'failures':failed,'stage1_training_admitted':False,
             'scope':'animation/*.fbx; exclude character assets and 2 static reference poses',
             'note':'Counts include pending/failed inputs. Processing completion does not mean training acceptance.'}
    quality=[]
    for path in (run/'clips').glob('*/quality/report.json'):
        r=read_record(path);quality.append(r)
    summary['evaluated_clips']=len(quality)
    summary['evaluated_frames']=sum(r.get('frames',0) for r in quality)
    summary['quality_statuses']=dict(Counter(r['status'] for r in quality))
    summary['quality_reason_counts']=dict(Counter(reason for r in quality for reason in r.get('reasons',[])))
    (run/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=run_path('mixamo_asap23_fixed'))
    p.add_argument('--dataset-root',type=Path,default=Path('/home/lyx/train/dataset/Mixamo_dataset'))
    a=p.parse_args();print(json.dumps(status(a.run,a.dataset_root),indent=2))
