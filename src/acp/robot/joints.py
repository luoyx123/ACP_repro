"""ASAP's joint subset: preserve three waist DoFs and fix all six wrist DoFs.

This is a joint-space baseline for an already retargeted G1 motion, not ASAP's
SMPL fitting optimizer and not a time-varying annealing algorithm.
"""

import xml.etree.ElementTree as ET

import numpy as np

WRIST_JOINTS = tuple(f"{side}_wrist_{axis}_joint"
                     for side in ("left", "right") for axis in ("roll", "pitch", "yaw"))


def moving_joint_names(urdf):
    return [joint.get("name") for joint in ET.parse(urdf).getroot().findall("joint")
            if joint.get("type") in ("revolute", "continuous", "prismatic")]


def project_joint_subset(qpos, source_names, target_names):
    """Canonical WXYZ,XYZ input/output. Root and retained joints stay exact."""
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != 7 + len(source_names) or not np.isfinite(qpos).all():
        raise ValueError("Expected finite canonical qpos matching source joint names")
    if len(source_names) != 29 or len(set(source_names)) != 29:
        raise ValueError("Expected 29 unique source joint names")
    if len(target_names) != 23 or len(set(target_names)) != 23:
        raise ValueError("Expected 23 unique target joint names")
    if set(source_names) - set(target_names) != set(WRIST_JOINTS) or not set(target_names) <= set(source_names):
        raise ValueError("ASAP subset must remove exactly the six wrist joints")
    return qpos[:, [*range(7), *(7 + source_names.index(name) for name in target_names)]].copy()
