"""源 G1 模型加载与命名关键点 FK，供所有机器人数据适配器和回放共用。"""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np


def load_source_model(path):
    """Compile the dataset's actual URDF, retaining fixed landmark bodies."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    compiler = root.find("mujoco/compiler")
    if compiler is None:
        extension = root.find("mujoco")
        if extension is None:
            extension = ET.SubElement(root, "mujoco")
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.clear()
    compiler.attrib.update(fusestatic="false", discardvisual="true", balanceinertia="true")
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((path.parent / mesh.get("filename")).resolve()))
    if root.find("link[@name='world']") is None:
        ET.SubElement(root, "link", name="world")
        joint = ET.SubElement(root, "joint", name="source_floating_base", type="floating")
        ET.SubElement(joint, "parent", link="world")
        ET.SubElement(joint, "child", link="pelvis")
    # The dataset URDF contains duplicate collision names; deterministic names
    # avoid parser renaming while retaining every collision shape and transform.
    for link in root.findall("link"):
        for index, collision in enumerate(link.findall("collision")):
            collision.set("name", f"{link.get('name')}_collision_{index}")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def fk_keypoints(model, qpos, demo_joints, source_mapping):
    if model.nq != 36:
        raise ValueError(f"Expected the dataset G1-29 model, got nq={model.nq}")
    data = mujoco.MjData(model)
    indices = [demo_joints.index(key) for key in source_mapping]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in source_mapping.values()]
    if min(body_ids) < 0:
        raise ValueError("The source FK mapping references an absent body")
    keypoints = np.zeros((len(qpos), len(demo_joints), 3))
    for index, canonical in enumerate(qpos):
        data.qpos[:] = np.r_[canonical[4:7], canonical[:4], canonical[7:]]
        mujoco.mj_forward(model, data)
        keypoints[index, indices] = data.xpos[body_ids]
    return keypoints
