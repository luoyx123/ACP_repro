import json

import numpy as np
import pytest

from acp.data.playback import PlaybackClock, event_onsets, resolve_clip


def test_playback_uses_irregular_timestamps_and_keeps_last_display_interval():
    clock = PlaybackClock([0, .04, .10], now=0)
    assert clock.frame(now=.03) == 0
    assert clock.frame(now=.07) == 1
    assert clock.frame(now=.14) == 2
    assert clock.frame(now=.17) == 0


def test_pause_resume_and_speed_changes_do_not_skip_paused_time():
    clock = PlaybackClock(np.arange(50) / 30, now=0, loop=False)
    clock.toggle(now=.105)
    assert clock.frame(now=10) == 3
    clock.toggle(now=10)
    assert clock.frame(now=10.04) == 4
    clock.set_speed(2, now=10.04)
    assert clock.frame(now=10.08) == 6
    clock.seek(12, pause=True, now=20)
    assert clock.frame(now=100) == 12
    clock.seek(-1, pause=True, now=100)
    assert clock.frame(now=100) == 0


def test_issue_jumps_use_segment_onsets():
    np.testing.assert_array_equal(event_onsets([False, True, True, False, True]), [1, 4])
    np.testing.assert_array_equal(event_onsets([True, True, False]), [0])
    assert len(event_onsets([])) == 0


def test_resolver_uses_selected_recovery_artifact_instead_of_failed_primary(tmp_path):
    artifact = tmp_path / "clips/example/attempts/robot_fk"
    (artifact / "labeled").mkdir(parents=True)
    motion = artifact / "labeled/motion.npz"
    motion.touch()
    scene = tmp_path / "actual_scene.xml"
    scene.write_text("<mujoco/>")
    (artifact / "manifest.json").write_text(json.dumps({"scene": str(scene), "source": "/raw/example.npz"}))
    (tmp_path / "clips/example/manifest.json").write_text(json.dumps({"status": "failed"}))
    (tmp_path / "artifact_index.json").write_text(json.dumps([
        {"clip": "example", "motion": str(motion.relative_to(tmp_path)), "route": "robot_fk"}]))
    resolved = resolve_clip(tmp_path, "example")
    assert resolved["artifact"] == artifact
    assert resolved["scene"] == scene
    with pytest.raises(ValueError, match="Unknown clip"):
        resolve_clip(tmp_path, "missing")
