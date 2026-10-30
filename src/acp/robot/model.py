"""Derive an ASAP-compatible 23-DoF model from the dataset's actual G1-29 URDF."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from acp.robot.joints import WRIST_JOINTS, moving_joint_names


def digest(path):
    return __import__('acp.management.fingerprints', fromlist=['digest']).digest(path)


def prepare(source, asap_root, destination):
    source, asap_root, destination = Path(source).resolve(), Path(asap_root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reference = asap_root / "humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.urdf"
    config = asap_root / "humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml"
    selected = moving_joint_names(reference)
    names = moving_joint_names(source)
    if len(selected) != 23 or set(names) - set(selected) != set(WRIST_JOINTS):
        raise ValueError("Local ASAP joint definition does not match the expected six-wrist reduction")
    root = ET.parse(source).getroot()
    root.set("name", "g1_asap23_spherehand")
    for joint in root.findall("joint"):
        if joint.get("name") in WRIST_JOINTS:
            joint.set("type", "fixed")
            for tag in ("limit", "dynamics"):
                node = joint.find(tag)
                if node is not None:
                    joint.remove(node)
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    for link in root.findall("link"):
        for kind in ("collision", "visual"):
            for index, geom in enumerate(link.findall(kind)):
                geom.set("name", f"{link.get('name')}_{kind}_{index}")
    # Fixed semantic frames do not alter collision geometry, masses or poses.
    for side in ("left", "right"):
        frame = f"acp_{side}_foot"
        ET.SubElement(root, "link", name=frame)
        joint = ET.SubElement(root, "joint", name=frame + "_fixed", type="fixed")
        ET.SubElement(joint, "parent", link=f"{side}_ankle_roll_link")
        ET.SubElement(joint, "child", link=frame)
        ET.SubElement(joint, "origin", xyz="0 0 -0.03", rpy="0 0 0")
    urdf = destination / "g1_asap23_spherehand.urdf"
    ET.indent(root)
    ET.ElementTree(root).write(urdf, encoding="utf-8", xml_declaration=True)
    native = compile_robot(urdf)
    xml = urdf.with_suffix(".xml")
    save_robot_xml(native, xml)
    actual = [native.joint(i).name for i in range(1, native.njnt)]
    if actual != moving_joint_names(urdf) or set(actual) != set(selected) or native.nq != 30:
        raise ValueError("Generated model has an incorrect joint order or subset")
    provenance = {"method": "asap_joint_subset_on_dataset_model", "source": str(source),
                  "source_sha256": digest(source), "asap_reference": str(reference),
                  "asap_reference_sha256": digest(reference), "asap_config_sha256": digest(config),
                  "target_joint_names": actual, "fixed_wrist_joints": list(WRIST_JOINTS),
                  "fixed_angles_rad": 0, "source_geometry_and_mass_preserved": True,
                  "target_urdf_sha256": digest(urdf), "target_xml_sha256": digest(xml),
                  "note": "ASAP-compatible joint definition; not the ASAP SMPL fitting algorithm."}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return urdf


def compile_robot(urdf):
    root = ET.parse(urdf).getroot()
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.clear()
    compiler.attrib.update(fusestatic="false", discardvisual="false", balanceinertia="true")
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((Path(urdf).parent / mesh.get("filename")).resolve()))
    for link in root.findall("link"):
        for kind in ("collision", "visual"):
            for index, geom in enumerate(link.findall(kind)):
                geom.set("name", f"{link.get('name')}_{kind}_{index}")
    ET.SubElement(root, "link", name="world")
    joint = ET.SubElement(root, "joint", name="source_floating_base", type="floating")
    ET.SubElement(joint, "parent", link="world")
    ET.SubElement(joint, "child", link="pelvis")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def save_robot_xml(model, output):
    """Avoid the XML writer's six-digit rounding of the kinematic tree."""
    mujoco.mj_saveLastXML(str(output), model)
    root = ET.parse(output).getroot()
    numbers = lambda values: " ".join(format(float(value), ".17g") for value in np.atleast_1d(values))
    for body in root.findall(".//body"):
        index = model.body(body.get("name")).id
        body.set("pos", numbers(model.body_pos[index]))
        body.set("quat", numbers(model.body_quat[index]))
        inertial = body.find("inertial")
        if inertial is not None:
            for name, values in [("pos", model.body_ipos[index]), ("quat", model.body_iquat[index]),
                                 ("mass", model.body_mass[index]), ("diaginertia", model.body_inertia[index])]:
                inertial.set(name, numbers(values))
    for joint in root.findall(".//joint"):
        if joint.get("name") is None:
            continue
        index = model.joint(joint.get("name")).id
        if model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE:
            joint.set("pos", numbers(model.jnt_pos[index]))
            joint.set("axis", numbers(model.jnt_axis[index]))
            joint.set("range", numbers(model.jnt_range[index]))
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def make_scene(robot_xml, terrain_urdf, output):
    root = ET.parse(robot_xml).getroot()
    root.set("model", "asap23_robot_terrain")
    asset, world = root.find("asset"), root.find("worldbody")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="floor_grid", type="2d", builtin="checker", width="512", height="512",
                  rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3")
    ET.SubElement(asset, "material", name="floor_material", texture="floor_grid", texrepeat="8 8", reflectance="0.05")
    ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 0.1", material="floor_material")
    ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1", directional="true")
    terrain_urdf = Path(terrain_urdf)
    terrain = ET.parse(terrain_urdf).getroot()
    # Preserve the actual fixed-link tree and mesh scales, including elevated terrain.
    links = {link.get("name"): link for link in terrain.findall("link")}
    parents = {"world": world}
    pending = list(terrain.findall("joint"))
    while pending:
        advanced = False
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in parents:
                continue
            if joint.get("type") != "fixed":
                raise ValueError("Only robot-terrain with static terrain is supported")
            origin = joint.find("origin")
            xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            xyzw = Rotation.from_euler("xyz", rpy).as_quat()
            body = ET.SubElement(parents[parent], "body", name="multi_boxes_" + child, pos=xyz,
                                 quat=" ".join(map(str, xyzw[[3, 0, 1, 2]])))
            parents[child] = body
            for index, collision in enumerate(links[child].findall("collision")):
                origin = collision.find("origin")
                xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
                rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
                xyzw = Rotation.from_euler("xyz", rpy).as_quat()
                mesh = collision.find("geometry/mesh")
                name = f"terrain_{child}_{index}"
                if mesh is not None:
                    ET.SubElement(asset, "mesh", name=name, file=str((terrain_urdf.parent / mesh.get("filename")).resolve()),
                                  scale=mesh.get("scale", "1 1 1"))
                    shape = dict(type="mesh", mesh=name)
                elif collision.find('geometry/box') is not None:
                    size = np.fromstring(collision.find('geometry/box').get('size'), sep=' ')
                    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                        raise ValueError('Invalid static terrain box dimensions')
                    shape = dict(type='box', size=' '.join(map(str, size/2)))
                elif collision.find('geometry/sphere') is not None:
                    shape = dict(type='sphere', size=collision.find('geometry/sphere').get('radius'))
                elif collision.find('geometry/cylinder') is not None:
                    cylinder = collision.find('geometry/cylinder')
                    shape = dict(type='cylinder', size=f"{cylinder.get('radius')} {float(cylinder.get('length'))/2}")
                else:
                    raise ValueError('Unsupported static terrain collision geometry')
                ET.SubElement(body, "geom", name="multi_boxes_" + name, **shape, pos=xyz,
                              quat=" ".join(map(str, xyzw[[3, 0, 1, 2]])), rgba="0.3 0.7 0.9 0.65")
            pending.remove(joint)
            advanced = True
        if not advanced:
            raise ValueError("Terrain has an unsupported or cyclic fixed-link tree")
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, required=True)
    parser.add_argument("--asap-root", type=Path, default=Path("/home/lyx/train/ASAP"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source_urdf, args.asap_root, args.output))
