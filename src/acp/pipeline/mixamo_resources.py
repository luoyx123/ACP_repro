"""Resource-only compatibility adapter for the unchanged, active Mixamo producer.

The original producer fingerprints its own bytes. Keep its solver/cache contract
stable while supplying machine-specific model, interpreter and Blender paths.
"""
import os
from pathlib import Path
import sys
import time
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


def throttle_status(interval=15.):
    """Keep all-input snapshots without rereading 2453 metadata files per clip."""
    import acp.management.status as module
    original=module.status
    last=[-float('inf')]
    def snapshot(run,dataset):
        now=time.monotonic()
        if now-last[0]<interval:return None
        last[0]=now
        return original(run,dataset)
    module.status=snapshot
    return original


if __name__=='__main__':
    cfg=machine()
    blender=Path(cfg.get('blender','blender'))
    if blender.is_absolute():
        if blender.name!='blender':raise SystemExit('Configured Blender executable must be named blender')
        os.environ['PATH']=str(blender.parent)+os.pathsep+os.environ.get('PATH','')
    batch.prepare=configured_prepare
    batch.convert_safe=configured_convert
    full_status=throttle_status() if '--dof' in sys.argv and sys.argv[sys.argv.index('--dof')+1]=='29' else None
    batch.main()
    if full_status and '--output' in sys.argv:
        full_status(Path(sys.argv[sys.argv.index('--output')+1]),Path(cfg['mixamo_dataset']))
