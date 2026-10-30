"""Make editable support hypotheses for reviewing priority Mixamo animations.

These are engineering test fixtures fitted to a first retargeting pass, not
recovered original environments. They are explicitly barred from admission
until geometry, support coverage and the name match have been reviewed.
"""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from acp.data.assets import read_record, resolve_path, run_path
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def fit_scene(artifact):
    manifest=read_record(artifact/'manifest.json')
    semantic=manifest.get('semantic') or {}
    kind=semantic.get('support_kind')
    if kind not in ('seat','ladder'):return None
    model=mujoco.MjModel.from_xml_path(manifest['scene']);data=mujoco.MjData(model)
    with np.load(next((artifact/'retargeted').glob('*.npz'))) as z:q=z['qpos'].copy()
    with np.load(artifact/'human_reference.npz') as z:
        source=z['positions'].copy();names=z['joint_names'].tolist();times=z['timestamps'].copy()
    toe_ids=[model.body(s+'_ankle_roll_sphere_5_link').id for s in ('left','right')]
    toes=[];pelvis_bottom=[]
    pelvis_geoms=[g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name=='pelvis_contour_link'
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
    ground=model.geom('ground').id
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_forward(model,data)
        toes.append(data.xpos[toe_ids].copy())
        pelvis_bottom.append(min(mujoco.mj_geomDistance(model,data,g,ground,5.,None) for g in pelvis_geoms))
    toes=np.array(toes)
    yaw=float(Rotation.from_quat(q[-1,[4,5,6,3]]).as_euler('xyz')[2])
    forward=np.array([np.cos(yaw),np.sin(yaw),0.])
    boxes=[];evidence={}
    if kind=='seat':
        end=max(5,len(q)//5)
        top=float(np.percentile(pelvis_bottom[-end:],10)-.002)
        if top<.1:raise ValueError('No plausible elevated seat in this reference')
        position=np.median(q[-end:,:3],axis=0)-forward*.07
        position[2]=top-.03
        boxes=[{'name':'seat','position':position.tolist(),'size':[.32,.42,.06],'rpy':[0.,0.,yaw]}]
        evidence={'fit':'terminal pelvis collision bottom minus 2mm', 'seat_top_m':top}
    else:
        anchors=[]
        for side,name in enumerate(('LeftToeBase','RightToeBase')):
            p=source[:,names.index(name)]
            speed=np.linalg.norm(np.gradient(p,times,axis=0),axis=1)
            mask=(speed<.15)&(p[:,2]>.06)
            edges=np.flatnonzero(np.diff(np.r_[False,mask,False]))
            for begin,end in zip(edges[::2],edges[1::2]):
                if end-begin>=4:
                    anchors.append(np.median(toes[begin:end,side],axis=0))
        if len(anchors)<2:raise ValueError('Insufficient elevated foot support intervals')
        anchors=sorted(anchors,key=lambda p:p[2])
        groups=[]
        for p in anchors:
            if groups and abs(p[2]-np.mean(groups[-1],axis=0)[2])<.07:groups[-1].append(p)
            else:groups.append([p])
        centres=np.array([np.mean(g,axis=0) for g in groups])
        if len(centres)<2:raise ValueError('Insufficient distinct rung heights')
        top=centres[:,2]-.007 # 5mm toe sphere + 2mm geometric clearance.
        spacing=float(np.median(np.diff(top)))
        if not .12<spacing<.55:raise ValueError(f'Ambiguous rung spacing {spacing}')
        # Feet alternate laterally: fit progression along heading, while fixing
        # a shared lateral centre instead of treating left/right as ladder tilt.
        along=centres@forward
        slope,intercept=np.polyfit(top,along,1)
        lateral=np.array([-forward[1],forward[0],0.])
        cross=float(np.mean(centres@lateral))
        max_hand=float(source[:,[names.index('LeftHand'),names.index('RightHand')],2].max())
        levels=np.arange(top[0],max_hand+.5*spacing,spacing)
        for level in levels:
            if level<.04:continue
            position=forward*(intercept+slope*level)+lateral*cross
            position[2]=level-.02
            boxes.append({'name':'rung_'+str(len(boxes)),'position':position.tolist(),
                          'size':[.06,.65,.04],'rpy':[0.,0.,yaw]})
        evidence={'fit':'regular horizontal rungs through stable retargeted toe supports',
                  'foot_anchor_centres_w':centres.tolist(),'rung_spacing_m':spacing,
                  'note':'hand grip geometry and rail placement require visual review'}
    return {'support_kind':kind,'review_status':'unconfirmed_engineering_test_scene',
            'contact_weight':2000. if kind=='ladder' else 200.,
            'step_size':.2,
            'original_scene_recovered':False,'approved_for_training':False,
            'fit_from_manifest':str((artifact/'manifest.json').absolute()),
            'evidence':evidence,'boxes':boxes}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();scenes={};errors={}
    for artifact in sorted((args.run/'clips').iterdir()):
        if not (artifact/'human_reference.npz').is_file():continue
        try:
            scene=fit_scene(artifact)
            if scene:scenes[artifact.name]=scene
        except Exception as exc:errors[artifact.name]=str(exc)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(scenes,indent=2)+'\n')
    args.output.with_suffix('.fit_report.json').write_text(json.dumps({'scenes':len(scenes),'errors':errors},indent=2)+'\n')
    print(json.dumps({'scenes':len(scenes),'errors':errors},indent=2))


if __name__=='__main__':main()
