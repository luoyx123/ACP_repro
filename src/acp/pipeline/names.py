"""Recover candidate semantic names by comparing evaluated, normalized skeletons.

Names are similarity hypotheses, not original-file identity. Both source files
and matching scores are retained so a reviewer can inspect ambiguous matches.
Run in the ACP environment after extracting the separately named FBX files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def fingerprint(path):
    with np.load(path) as z:
        p, names, rest = z['positions'], z['joint_names'].tolist(), z['rest_positions']
        scale = sum(np.linalg.norm(rest[names.index(a)]-rest[names.index(b)]) for a, b in
                    [('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot')])
        u = np.linspace(0, len(p)-1, 32)
        points = np.stack([np.interp(u, np.arange(len(p)), p[:, i, j])
                           for i in range(len(names)) for j in range(3)], axis=1).reshape(32, len(names), 3)
        points -= points[0, names.index('Hips')].copy()
        return points.ravel()/scale, (len(p)-1)/float(z['fps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extracted', type=Path, required=True)
    parser.add_argument('--named-extracted', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.extracted.glob('*.npz'))
    vectors, durations = zip(*(fingerprint(p) for p in files))
    vectors, durations = np.array(vectors), np.array(durations)
    prompts = json.loads(args.prompts.read_text())
    semantics, report = {}, []
    for named in sorted(args.named_extracted.glob('*.npz')):
        if named.with_suffix('.fbx').name not in prompts:
            continue
        vector, duration = fingerprint(named)
        errors = np.sqrt(np.mean((vectors-vector)**2, axis=1))
        order = np.argsort(errors)
        metadata = prompts[named.with_suffix('.fbx').name]
        description = metadata.get('description', '')
        text = (named.stem+' '+description).lower()
        kind = 'ladder' if 'ladder' in text else 'seat' if any(w in text for w in ('sitting', 'seated', 'sit down')) else None
        candidates = [{'clip':files[i].stem, 'normalized_rmse':float(errors[i]),
                       'duration_difference_s':float(abs(durations[i]-duration))} for i in order[:3]]
        report.append({'name':named.stem, 'description':description, 'matches':candidates})
        for i in order:
            if errors[i] > min(.07, errors[order[0]]+1e-6):
                break
            if abs(durations[i]-duration) > .08:
                continue
            entry = {'name':named.stem, 'description':description, 'motion_id':metadata.get('motion_id'),
                     'priority':bool(kind), 'support_kind':kind,
                     'name_status':'trajectory_similarity_candidate', 'normalized_rmse':float(errors[i]),
                     'duration_difference_s':float(abs(durations[i]-duration)),
                     'named_reference':str(named.absolute()),
                     'named_fbx_sha256':hashlib.sha256(named.with_suffix('.fbx').read_bytes()).hexdigest(),
                     'metadata_source':'https://huggingface.co/datasets/Linzhan/Mixamo-Animations-Characters'}
            clip = files[i].stem
            if clip not in semantics or entry['normalized_rmse'] < semantics[clip]['normalized_rmse']:
                semantics[clip] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(semantics, indent=2)+'\n')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'named_references':len(report), 'candidate_names':len(semantics),
                      'priority_clips':sum(r['priority'] for r in semantics.values()),
                      'ladder_candidates':{k:v['name'] for k,v in semantics.items() if v['support_kind']=='ladder'}}, indent=2))


if __name__ == '__main__':
    main()
