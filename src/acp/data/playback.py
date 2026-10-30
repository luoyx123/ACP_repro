"""Artifact lookup and timestamp-based playback, independent of a graphics backend."""

from acp.data.assets import read_record, resolve_path, run_path

import json
from pathlib import Path
import time

import numpy as np


def resolve_clip(run, clip):
    run = resolve_path(run).resolve()
    rows = read_record(run / "artifact_index.json")
    matches = [row for row in rows if row["clip"] == clip]
    if not matches:
        raise ValueError(f"Unknown clip {clip!r}; use --list to see available names")
    if not matches[0].get("motion"):
        raise ValueError(f"Clip {clip!r} has no playback artifact; status={matches[0].get('status', 'unknown')}")
    motion = (run / matches[0]["motion"]).resolve()
    if not motion.is_file():
        raise FileNotFoundError(motion)
    artifact = motion.parent.parent
    manifest = read_record(artifact / "manifest.json")
    scene = manifest.get("scene")
    if scene is None:
        for entry in manifest.get("commands", []):
            argv = entry["argv"]
            if "--scene" in argv:
                scene = argv[argv.index("--scene") + 1]
                break
    if scene is None:
        raise ValueError(f"No scene recorded in {artifact / 'manifest.json'}")
    scene = Path(scene)
    if not scene.is_file():
        raise FileNotFoundError(f"Native scene is missing: {scene}; regenerate assets on this machine")
    return {**matches[0], "artifact": artifact, "motion": motion, "scene": scene,
            "source": Path(manifest["source"])}


def event_onsets(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("Expected a one-dimensional event mask")
    return np.flatnonzero(mask & np.r_[True, ~mask[:-1]]) if len(mask) else np.array([], dtype=int)


class PlaybackClock:
    def __init__(self, timestamps, *, frame=0, speed=1.0, playing=True, loop=True, now=None):
        self.times = np.asarray(timestamps, dtype=float)
        if (self.times.ndim != 1 or len(self.times) < 2 or not np.isfinite(self.times).all()
                or not np.all(np.diff(self.times) > 0)):
            raise ValueError("Playback requires finite, strictly increasing timestamps")
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Playback speed must be finite and positive")
        self.speed, self.playing, self.loop = float(speed), playing, loop
        self.anchor = time.monotonic() if now is None else now
        self.position = self.times[int(np.clip(frame, 0, len(self.times) - 1))]

    def _time(self, now):
        value = self.position + (now - self.anchor) * self.speed if self.playing else self.position
        # Include the last sample's display interval before looping.
        period = self.times[-1] - self.times[0] + self.times[-1] - self.times[-2]
        if self.loop:
            value = self.times[0] + (value - self.times[0]) % period
        return float(np.clip(value, self.times[0], self.times[-1]))

    def frame(self, now=None):
        now = time.monotonic() if now is None else now
        return int(np.clip(np.searchsorted(self.times, self._time(now), side="right") - 1,
                           0, len(self.times) - 1))

    def seek(self, frame, *, pause=False, now=None):
        self.position = self.times[int(np.clip(frame, 0, len(self.times) - 1))]
        self.anchor = time.monotonic() if now is None else now
        if pause:
            self.playing = False

    def toggle(self, now=None):
        now = time.monotonic() if now is None else now
        self.position = self._time(now)
        self.anchor = now
        self.playing = not self.playing

    def set_speed(self, speed, now=None):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Playback speed must be finite and positive")
        now = time.monotonic() if now is None else now
        self.position, self.anchor = self._time(now), now
        self.speed = float(speed)
