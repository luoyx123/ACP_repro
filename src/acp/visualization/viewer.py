"""View indexed OmniRetarget G1-23 results, inspect failures, or export annotated MP4/PNG.

Run with the Holosoma Python environment. No simulation step is integrated.
"""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
from acp.data.assets import read_record, resolve_path, run_path

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from acp.data.playback import PlaybackClock, event_onsets, resolve_clip


class Clip:
    def __init__(self, resolved, *, show_source=False, dataset_root=None):
        import mujoco

        self.info = resolved
        self.manifest = read_record(resolved["artifact"] / "manifest.json")
        self.model = mujoco.MjModel.from_xml_path(str(resolved["scene"]))
        self.data = mujoco.MjData(self.model)
        artifact = resolved["artifact"]
        with np.load(resolved["motion"], allow_pickle=False) as z:
            self.times = z["timestamps"].copy()
            self.qpos = np.c_[z["base_position_w"], z["base_quaternion_wxyz"], z["joint_positions"]]
            self.flags = z["contact_flags"].copy()
            self.names = z["body_names"].tolist()
            self.joint_names = z["joint_names"].tolist()
            self.landmarks = z["body_landmarks_w"].copy()
            self.surfaces = z["contact_surface_positions_w"].copy()
            self.velocity_bad = ~z["velocity_limits_valid"].copy()
            self.fps = float(z["fps"])
        actual = [self.model.joint(i).name for i in range(self.model.njnt)
                  if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        if self.qpos.shape[1] != self.model.nq or actual != self.joint_names:
            raise ValueError("Recorded motion and scene have different joints")
        if len(self.qpos) != len(self.times):
            raise ValueError("Motion frame count differs from timestamps")
        self.quality = read_record(artifact / "quality/report.json")
        with np.load(artifact / "quality/diagnostics.npz") as z:
            self.self_min = z["self_contact_minimum_distances"].copy()
            self.unexpected_self_min = z['unexpected_self_contact_minimum_distances'].copy() if 'unexpected_self_contact_minimum_distances' in z else self.self_min.copy()
            self.nonlocal_min = z["nonlocal_self_contact_minimum_distances"].copy()
            self.foot_speeds = z["foot_planar_speeds"].copy()
        with np.load(artifact / "validation/mujoco_geometry.npz") as z:
            self.env_min = z["minimum_distances"].copy()
        for values in (self.self_min, self.nonlocal_min, self.env_min, self.foot_speeds):
            if len(values) != len(self.times):
                raise ValueError("Diagnostics differ from motion timestamps; rerun evaluation")
        events = [event_onsets(mask) for mask in
                  (self.unexpected_self_min < -0.003, self.env_min < -0.003, self.velocity_bad)]
        self.events = np.unique(np.concatenate(events))
        self.env_geoms = {i for i in range(self.model.ngeom)
                          if self.model.geom_bodyid[i] == 0
                          or (self.model.geom(i).name or "").startswith("multi_boxes_")}
        self.source_points, self.source_edges = None, []
        self.paired = None
        self.repair = None
        comparison = artifact / "comparison/diagnostics.npz"
        if comparison.is_file():
            with np.load(comparison) as z:
                self.paired = {name: z[name].copy() for name in
                               ("source_env", "source_self", "hand_position_errors")}
        repair = artifact / "arm_repair_diagnostics.npz"
        if repair.is_file():
            with np.load(repair) as z:
                self.repair = {name: z[name].copy() for name in
                               ("source_hand_distances", "before_hand_distances", "after_hand_distances", "frame_constraint_margins")}
        if show_source:
            self.load_source(dataset_root)

    def load_source(self, dataset_root):
        import mujoco
        human_path = self.info['artifact'] / 'human_reference.npz'
        if self.manifest.get('source_format') == 'mixamo_fbx_world_bones':
            with np.load(human_path) as z:
                names=z['joint_names'].tolist()
                mapped=z['mapped_joint_names'].tolist()
                self.source_points=z['positions'][:,[names.index(n) for n in mapped]].copy()
                np.testing.assert_allclose(self.times,z['timestamps'],atol=1e-12)
            edges=[('Hips','Spine')]
            for side in ('Left','Right'):
                edges += [('Hips',side+'UpLeg'),('Spine',side+'Arm')]
                edges += [(side+a,side+b) for a,b in [('UpLeg','Leg'),('Leg','Foot'),('Foot','ToeBase'),('Arm','ForeArm'),('ForeArm','Hand')]]
            self.source_edges=[(mapped.index(a),mapped.index(b)) for a,b in edges]
            return
        from scipy.spatial.transform import Rotation, Slerp
        from holosoma_retargeting.config_types.data_type import MotionDataConfig
        from acp.robot.transfer import load_source_model

        source = self.info["source"]
        if "source_motion_npz" in self.manifest:
            source = Path(self.manifest["source_motion_npz"])
            model = mujoco.MjModel.from_xml_path(self.manifest["source_robot_xml"])
        else:
            if dataset_root is not None:
                source = Path(dataset_root).resolve() / "robot-terrain" / source.name
            root = source.parent.parent
            model = load_source_model(root / "models/g1/g1_29dof_spherehand.urdf")
        data = mujoco.MjData(model)
        with np.load(source, allow_pickle=False) as z:
            q, fps = z["qpos"].copy(), float(z["fps"])
        if q.ndim != 2 or q.shape[1] != 36:
            raise ValueError("Source overlay requires robot-terrain G1-29 qpos")
        source_times = np.arange(len(q)) / fps
        if self.times[0] < -1e-9 or self.times[-1] > source_times[-1] + 1e-9:
            raise ValueError("Source clip does not span the target timestamps")
        query = np.clip(self.times, source_times[0], source_times[-1])
        quat = Slerp(source_times, Rotation.from_quat(q[:, [1, 2, 3, 0]]))(query).as_quat()
        interpolated = np.column_stack([np.interp(query, source_times, q[:, i]) for i in range(4, 36)])
        poses = np.c_[interpolated[:, :3], quat[:, [3, 0, 1, 2]], interpolated[:, 3:]]
        mapping = MotionDataConfig(data_format="mocap", robot_type="g1").resolved_joints_mapping
        names = list(mapping)
        ids = [model.body(name).id for name in mapping.values()]
        self.source_points = np.empty((len(poses), len(ids), 3))
        for frame, pose in enumerate(poses):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            self.source_points[frame] = data.xpos[ids]
        edges = [("Spine1", key) for key in ("LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm")]
        for side in ("Left", "Right"):
            edges += [(side + a, side + b) for a, b in
                      (("UpLeg", "Leg"), ("Leg", "Foot"), ("Foot", "ToeBase"),
                       ("Arm", "ForeArm"), ("ForeArm", "HandMiddle3"))]
        self.source_edges = [(names.index(a), names.index(b)) for a, b in edges]

    def update(self, frame):
        import mujoco
        self.data.qpos[:] = self.qpos[frame]
        mujoco.mj_forward(self.model, self.data)

    def collisions(self):
        pairs = {}
        for contact in self.data.contact:
            if contact.dist >= -0.003:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            environment = a in self.env_geoms or b in self.env_geoms
            if a in self.env_geoms and b in self.env_geoms:
                continue
            name = " / ".join(sorted(self.model.body(int(self.model.geom_bodyid[g])).name or "world"
                                      for g in (a, b)))
            policy = self.quality.get('collision_policy')
            if not environment and policy:
                from acp.robot.collision import joint_housing_overlap
                bodies=[self.model.body(int(self.model.geom_bodyid[g])).name for g in (a,b)]
                anchors={rule['joint']:self.data.xanchor[self.model.joint(rule['joint']).id]
                         for rule in policy['rules']}
                if joint_housing_overlap(*bodies,contact.pos,float(contact.dist),anchors,policy['rules']):
                    continue
            key = (environment, name)
            if key not in pairs or contact.dist < pairs[key][0]:
                pairs[key] = (float(contact.dist), contact.pos.copy())
        return sorted([(env, name, distance, point) for (env, name), (distance, point) in pairs.items()],
                      key=lambda item: item[2])

    def text(self, frame, *, speed=1.0, paused=False):
        active = ", ".join(name for name, flag in zip(self.names, self.flags[frame]) if flag) or "none"
        status = []
        if self.env_min[frame] < -0.003:
            status.append("ENV penetration")
        if self.unexpected_self_min[frame] < -0.003:
            status.append("SELF penetration")
        if self.velocity_bad[frame]:
            status.append("joint velocity limit")
        lines = [f"{self.info['clip']} | {frame + 1}/{len(self.times)} | t={self.times[frame]:.3f}s | {speed:g}x {'PAUSED' if paused else ''}",
                 f"Env min {1000*self.env_min[frame]:.1f} mm | Raw self {1000*self.self_min[frame]:.1f} mm | Unexpected self {1000*self.unexpected_self_min[frame]:.1f} mm",
                 f"Sole XY speed proxy L/R: {self.foot_speeds[frame, 0]:.3f} / {self.foot_speeds[frame, 1]:.3f} m/s",
                 "Contacts: " + active, "Flags: " + (", ".join(status) or "none of the frame-level checks")]
        lines += [f"{'ENV' if env else 'SELF'} {name}: {1000*distance:.1f} mm"
                  for env, name, distance, _ in self.collisions()[:2]]
        if self.paired is not None:
            reference_label = ("CSV on spherehand G1-29" if self.manifest.get("source_format") == "lafan_g1_csv_xyz_xyzw"
                               else "Source G1-29")
            lines += [f"{reference_label} native env/self: {1000*self.paired['source_env'][frame]:.1f} / {1000*self.paired['source_self'][frame]:.1f} mm",
                      "Wrist fixing hand shift L/R: " + " / ".join(
                          f"{1000*value:.1f}" for value in self.paired["hand_position_errors"][frame]) + " mm"]
        if self.repair is not None:
            for label, name in (("Source hand", "source_hand_distances"), ("Before repair", "before_hand_distances"),
                                ("After repair", "after_hand_distances")):
                lines += [label + " L/R distance: " + " / ".join(f"{1000*x:.1f}" for x in self.repair[name][frame]) + " mm"]
            lines += [f"Arm geometry constraint margin: {1000*self.repair['frame_constraint_margins'][frame]:.3f} mm"]
        return lines


def add_markers(scene, clip, frame, *, contacts=True, source=True, collisions=True):
    import mujoco

    def sphere(point, color, radius):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([radius]*3), point, np.eye(3).ravel(), np.array(color))
        scene.ngeom += 1

    def line(a, b, color, width=2):
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(b - a) < 1e-8:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                           np.ones(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, a, b)
        scene.ngeom += 1

    if contacts:
        for body in np.flatnonzero(clip.flags[frame]):
            sphere(clip.landmarks[frame, body], [0.1, 1, 0.15, 1], 0.018)
            sphere(clip.surfaces[frame, body], [1, 0.9, 0.1, 1], 0.012)
            line(clip.landmarks[frame, body], clip.surfaces[frame, body], [1, .9, .1, 1])
    if source and clip.source_points is not None:
        points = clip.source_points[frame]
        for point in points:
            sphere(point, [0.1, .85, 1, 1], 0.012)
        for a, b in clip.source_edges:
            line(points[a], points[b], [0.1, .85, 1, 1], width=4)
    if collisions:
        for env, _, _, point in clip.collisions()[:30]:
            sphere(point, [1, .45, .03, .9] if env else [1, .05, .15, .95], .022)


def choose_frame(clip, args):
    if args.frame is not None:
        if not 0 <= args.frame < len(clip.times):
            raise ValueError(f"--frame must be in [0, {len(clip.times)-1}]")
        return args.frame
    if args.seek == "worst-environment":
        return int(np.argmin(clip.env_min))
    if args.seek == "worst-self":
        return int(np.argmin(clip.self_min))
    if args.seek == "first-velocity":
        frames = np.flatnonzero(clip.velocity_bad)
        return int(frames[0]) if len(frames) else 0
    return 0


def camera_for(clip, args):
    import mujoco
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
    return camera


def export(clip, args, start):
    import mujoco
    from PIL import Image, ImageDraw

    output = args.output.resolve()
    if output.suffix.lower() not in (".png", ".mp4"):
        raise ValueError("Output must have a .png or .mp4 extension")
    if output.suffix.lower() == ".mp4" and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to export MP4")
    output.parent.mkdir(parents=True, exist_ok=True)
    clip.model.vis.global_.offwidth = max(clip.model.vis.global_.offwidth, args.width)
    clip.model.vis.global_.offheight = max(clip.model.vis.global_.offheight, args.height)
    camera = camera_for(clip, args)
    camera.lookat[:] = clip.qpos[start, :3] + [0, 0, .1]

    def draw(renderer, frame):
        clip.update(frame)
        if not args.fixed_camera:
            camera.lookat[:] = clip.qpos[frame, :3] + [0, 0, .1]
        renderer.update_scene(clip.data, camera=camera)
        add_markers(renderer.scene, clip, frame, contacts=not args.no_markers,
                    source=args.show_source, collisions=not args.no_markers)
        image = Image.fromarray(renderer.render())
        painter = ImageDraw.Draw(image)
        lines = clip.text(frame, speed=args.speed)
        source_label = ('scaled Mixamo human bones' if clip.manifest.get('source_format') == 'mixamo_fbx_world_bones'
                        else 'source G1-29 FK')
        lines += ["Green: body contact | Yellow: surface | Red: self penetration | Orange: environment | Cyan: " + source_label]
        painter.rectangle([0, 0, args.width, 17*len(lines)+12], fill=(15, 18, 23))
        for row, text in enumerate(lines):
            painter.text((10, 6+17*row), text, fill="white" if row else (150, 220, 255))
        return image

    renderer = mujoco.Renderer(clip.model, height=args.height, width=args.width)
    try:
        if output.suffix.lower() == ".png":
            draw(renderer, start).save(output)
        else:
            end = clip.times[-1] if args.seconds is None else min(clip.times[-1], clip.times[start] + args.seconds)
            duration = (end - clip.times[start]) / args.speed
            display_times = clip.times[start] + np.arange(int(np.floor(duration * args.video_fps + 1e-9)) + 1) * args.speed / args.video_fps
            frames = np.clip(np.searchsorted(clip.times, display_times, side="right") - 1, start, len(clip.times)-1)
            command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                       "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.video_fps), "-i", "-",
                       "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                try:
                    for number, frame in enumerate(frames):
                        image = draw(renderer, int(frame))
                        if number == 0:
                            image.save(output.with_suffix(".png"))
                        process.stdin.write(np.asarray(image).tobytes())
                finally:
                    process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("ffmpeg failed")
    finally:
        renderer.close()
    output.with_suffix(".json").write_text(json.dumps({
        "clip": clip.info["clip"], "motion": str(clip.info["motion"]),
        "scene": str(clip.info["scene"]), "route": clip.info["route"],
        "start_frame": start, "start_time_s": float(clip.times[start]),
        "source_overlay": args.show_source, "speed": args.speed,
        "note": "Kinematic playback; diagnostic markers are not contact-force validation.",
    }, indent=2) + "\n")
    print(output)


def interactive(clip, args, start):
    import mujoco
    import mujoco.viewer

    clock = PlaybackClock(clip.times, frame=start, speed=args.speed, playing=not args.paused, loop=not args.no_loop)
    lock = threading.Lock()
    state = {"contacts": not args.no_markers, "collisions": not args.no_markers, "source": args.show_source}

    def key(keycode):
        with lock:
            current = clock.frame()
            if keycode == 32:
                clock.toggle()
            elif keycode in (262, 263):
                clock.seek(current + (1 if keycode == 262 else -1), pause=True)
            elif keycode in (74, 75):
                events = clip.events[clip.events < current] if keycode == 74 else clip.events[clip.events > current]
                if len(events):
                    clock.seek(int(events[-1] if keycode == 74 else events[0]), pause=True)
            elif keycode == 82:
                clock.seek(0)
            elif keycode == 67:
                state["contacts"] = not state["contacts"]
            elif keycode == 68:
                state["collisions"] = not state["collisions"]
            elif keycode == 83 and clip.source_points is not None:
                state["source"] = not state["source"]
            elif keycode in (91, 93):
                clock.set_speed(float(np.clip(clock.speed * (.5 if keycode == 91 else 2), .125, 4)))

    print("Space: pause/play | Left/Right: step | J/K: previous/next issue onset | R: restart")
    print("C: contacts | D: penetration markers | S: source FK | [ / ]: speed | Mouse: camera")
    clip.update(start)
    began = time.monotonic()
    with mujoco.viewer.launch_passive(clip.model, clip.data, key_callback=key,
                                    show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = args.distance, args.azimuth, args.elevation
            viewer.cam.lookat[:] = clip.qpos[start, :3] + [0, 0, .1]
        while viewer.is_running():
            if args.seconds is not None and time.monotonic() - began >= args.seconds:
                break
            with lock:
                frame, playing, speed, toggles = clock.frame(), clock.playing, clock.speed, state.copy()
            with viewer.lock():
                clip.update(frame)
                if not args.fixed_camera:
                    viewer.cam.lookat[:] = clip.qpos[frame, :3] + [0, 0, .1]
                viewer.user_scn.ngeom = 0
                add_markers(viewer.user_scn, clip, frame, **toggles)
            viewer.set_texts([(None, None, "\n".join(clip.text(frame, speed=speed, paused=not playing)), ""),
                              (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               "Space play/pause | arrows step | J/K issue | C contacts | D penetration | S source | [ ] speed", "")])
            viewer.sync()
            time.sleep(1 / 120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=run_path("omni_asap23"))
    parser.add_argument("--clip")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path, help="Export .mp4 or a single .png; omit for an interactive window")
    parser.add_argument("--show-source", action="store_true", help="Overlay source G1-29 FK skeleton in cyan")
    parser.add_argument("--dataset-root", type=Path, help="Override source dataset path after moving machines")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--seek", choices=("start", "worst-environment", "worst-self", "first-velocity"), default="start")
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--seconds", type=float, help="MP4 source-time duration, or auto-close window after this wall time")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--fixed-camera", action="store_true")
    parser.add_argument("--no-markers", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--video-fps", type=float, default=25)
    parser.add_argument("--distance", type=float, default=3.2)
    parser.add_argument("--azimuth", type=float, default=135)
    parser.add_argument("--elevation", type=float, default=-20)
    args = parser.parse_args()
    if args.list:
        for row in read_record(args.run / "artifact_index.json"):
            print(f"{row['clip']:28s} {row['route']:42s} {', '.join(row['reasons'])}")
        return
    if not args.clip:
        parser.error("Specify --clip NAME, or use --list")
    if not np.isfinite(args.speed) or args.speed <= 0 or not np.isfinite(args.video_fps) or args.video_fps <= 0:
        parser.error("speed and video-fps must be finite and positive")
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("seconds must be finite and positive")
    if args.width < 320 or args.height < 240 or args.width % 2 or args.height % 2:
        parser.error("Video dimensions must be even, width >=320 and height >=240")
    os.environ.setdefault("MUJOCO_GL", "egl" if args.output else "glfw")
    resolved = resolve_clip(args.run, args.clip)
    clip = Clip(resolved, show_source=args.show_source, dataset_root=args.dataset_root)
    start = choose_frame(clip, args)
    if args.output:
        export(clip, args, start)
    else:
        interactive(clip, args, start)


if __name__ == "__main__":
    main()
