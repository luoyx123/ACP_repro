"""Drake FK and geometry queries; no integration or invented contact forces."""

from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

from acp.data.assets import resolve_path

import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, Box, CoulombFriction, DiagramBuilder,
                        Parser, RigidTransform)

BODY_NAMES = ("left_foot", "right_foot", "left_knee", "right_knee", "left_hand",
              "right_hand", "left_elbow", "right_elbow", "pelvis")


class MotionScene:
    def __init__(self, robot_urdf, terrain_urdf=None):
        self.robot_urdf = Path(robot_urdf).resolve()
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        parser = Parser(self.plant)
        # Drake proximity queries require OBJ rather than Unitree's STL meshes.
        # Convert cached copies, preserving source meshes and all URDF transforms.
        robot_xml = ET.parse(self.robot_urdf).getroot()
        for mesh in robot_xml.findall(".//mesh"):
            mesh_path = Path(mesh.get("filename"))
            if not mesh_path.is_absolute():
                mesh_path = self.robot_urdf.parent / mesh_path
            if mesh_path.suffix.lower() == ".stl":
                import trimesh
                digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
                cache = resolve_path("outputs/mesh_cache")
                cache.mkdir(parents=True, exist_ok=True)
                converted = cache / f"{digest}.obj"
                if not converted.exists():
                    trimesh.load_mesh(mesh_path).export(converted)
                mesh_path = converted
            mesh.set("filename", mesh_path.resolve().as_uri())
        self.robot = parser.AddModelsFromString(ET.tostring(robot_xml, encoding="unicode"), "urdf")[0]
        if terrain_urdf is not None:
            parser.AddModels(str(Path(terrain_urdf).resolve()))
        self.plant.RegisterCollisionGeometry(self.plant.world_body(), RigidTransform([0, 0, -0.5]), Box(200, 200, 1),
                                             "ground", CoulombFriction(0.8, 0.8))
        self.plant.Finalize()
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.graph_context = self.scene_graph.GetMyContextFromRoot(self.context)
        self.position_names = self.plant.GetPositionNames(self.robot, add_model_instance_prefix=False)
        self.joint_names = [name.removesuffix("_q") for name in self.position_names[7:]]
        if len(self.position_names) != self.plant.num_positions():
            raise ValueError("Only static terrain is supported; dynamic objects require an explicit adapter")
        robot_xml = ET.parse(self.robot_urdf).getroot()
        links = {link.get("name") for link in robot_xml.findall("link")}
        hands = ([f"{s}_sphere_hand_link" for s in ["left", "right"]]
                 if "left_sphere_hand_link" in links
                 else [f"{s}_wrist_roll_rubber_hand" for s in ["left", "right"]])
        self.landmark_links = ["left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link",
                               "right_knee_link", *hands, "left_elbow_link", "right_elbow_link", "pelvis"]
        self.offsets = np.zeros((9, 3))
        self.offsets[:2, 2] = -0.03  # Nominal sole plane; actual geometry witnesses saved separately.
        if "rubber" in hands[0]:
            self.offsets[4:6, 0] = 0.15  # Engineering palm landmark, NOT a paper-disclosed point.
        # Generated Holosoma rev1.0 assets bind the same semantic landmarks in
        # URDF and MJCF. Geometry grouping below continues to use physical links.
        self.landmark_frame_links = list(self.landmark_links)
        for index, name in {0: "acp_left_foot", 1: "acp_right_foot", 4: "left_hand_target",
                            5: "right_hand_target", 8: "acp_pelvis"}.items():
            if name in links:
                self.landmark_frame_links[index] = name
                self.offsets[index] = 0
        self.frames = [self.plant.GetFrameByName(name, self.robot) for name in self.landmark_frame_links]
        # Add rigidly attached links, so pelvis_contour and sole spheres belong to their body group.
        groups = {name: i for i, name in enumerate(self.landmark_links)}
        changed = True
        while changed:
            changed = False
            for joint in robot_xml.findall("joint"):
                parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
                if joint.get("type") == "fixed" and parent in groups and child not in groups:
                    groups[child] = groups[parent]
                    changed = True
        self.geometry_group = {}
        self.robot_geometries = set()
        for idx in self.plant.GetBodyIndices(self.robot):
            body = self.plant.get_body(idx)
            for geom in self.plant.GetCollisionGeometriesForBody(body):
                self.robot_geometries.add(geom)
                if body.name() in groups:
                    self.geometry_group[geom] = groups[body.name()]

    def set_positions(self, q):
        self.plant.SetPositions(self.plant_context, q)

    def landmarks(self):
        world = self.plant.world_frame()
        return np.array([self.plant.CalcPointsPositions(self.plant_context, f, p, world).ravel()
                         for f, p in zip(self.frames, self.offsets)])

    def environment_pairs(self, max_distance=0.1):
        query = self.scene_graph.get_query_output_port().Eval(self.graph_context)
        pairs = query.ComputeSignedDistancePairwiseClosestPoints(max_distance)
        return query, [p for p in pairs if (p.id_A in self.robot_geometries) != (p.id_B in self.robot_geometries)]

    def contact_geometry(self, threshold=0.015):
        distances = np.full(9, np.inf)
        witnesses = np.zeros((9, 3))
        query, pairs = self.environment_pairs(threshold)
        for pair in pairs:
            robot_is_a = pair.id_A in self.robot_geometries
            geom = pair.id_A if robot_is_a else pair.id_B
            group = self.geometry_group.get(geom)
            if group is not None and pair.distance < distances[group]:
                distances[group] = pair.distance
                env_id = pair.id_B if robot_is_a else pair.id_A
                env_point = pair.p_BCb if robot_is_a else pair.p_ACa
                witnesses[group] = query.GetPoseInWorld(env_id) @ env_point
        minimum_distance = min((pair.distance for pair in pairs), default=np.inf)
        return distances, witnesses, minimum_distance
