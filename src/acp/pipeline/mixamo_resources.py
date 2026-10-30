"""Resource-only compatibility adapter for the unchanged, active Mixamo producer.

The original producer fingerprints its own bytes. Keep its solver/cache contract
stable while supplying machine-specific model, interpreter and Blender paths.
"""
import os
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'src'))
from acp.data.assets import machine
import acp.pipeline.datasets.mixamo as batch
from acp.robot.model import prepare


def configured_prepare(_source, _asap, destination):
    cfg=machine()
    return prepare(Path(cfg['omni_dataset'])/'models/g1/g1_29dof_spherehand.urdf',
                   Path(cfg['asap_root']),destination)


def configured_convert(row, run, _label_python, fps):
    # A top-level function stays picklable by ProcessPoolExecutor.
    label_python=os.environ.get('ACP_BOUNDED_LABEL_EXECUTABLE',
                                str((PROJECT/Path(machine()['label_python'])).absolute()))
    return original_convert_safe(row,run,label_python,fps)


original_convert_safe=batch.convert_safe
if __name__=='__main__':
    cfg=machine()
    blender=Path(cfg.get('blender','blender'))
    if blender.is_absolute():
        if blender.name!='blender':raise SystemExit('Configured Blender executable must be named blender')
        os.environ['PATH']=str(blender.parent)+os.pathsep+os.environ.get('PATH','')
    batch.prepare=configured_prepare
    batch.convert_safe=configured_convert
    batch.main()
