"""Contact segmentation and next-onset labels for a single episode."""

import numpy as np
from scipy.spatial.transform import Rotation


def filter_segments(flags: np.ndarray, minimum_steps: int = 10) -> np.ndarray:
    flags = np.asarray(flags, dtype=bool)
    if flags.ndim != 2 or flags.shape[0] == 0 or minimum_steps < 1:
        raise ValueError("Expected [time, body] flags and positive minimum_steps")
    output = np.zeros_like(flags)
    for body in range(flags.shape[1]):
        edges = np.diff(np.r_[False, flags[:, body], False].astype(int))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start >= minimum_steps:
                output[start:stop, body] = True
    return output


def yaw_local(points_w, root_position_w, root_quaternion_wxyz):
    q = np.asarray(root_quaternion_wxyz)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid root quaternion")
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return (np.asarray(points_w) - root_position_w) @ Rotation.from_euler("z", yaw).as_matrix()


def next_onset_targets(flags, contact_positions_w, root_positions_w, root_quaternions_wxyz,
                       horizon_steps=75, minimum_steps=10):
    """Current onset is included; an ongoing segment is NOT a new onset.

    Horizon includes t+75. First future onset outside the footprint is masked,
    rather than skipped in favor of a later onset. Tail labels use recorded data
    only, and `future_complete` marks frames with a full lookahead window.
    Input flags may be geometric references or measured forces; caller records
    the provenance. This function does not establish episode success/admission.
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    flags = filter_segments(flags, minimum_steps)
    positions = np.asarray(contact_positions_w)
    t_count, bodies = flags.shape
    if (positions.shape != (t_count, bodies, 3)
            or np.shape(root_positions_w) != (t_count, 3)
            or np.shape(root_quaternions_wxyz) != (t_count, 4)
            or not all(np.all(np.isfinite(v)) for v in
                       [positions, root_positions_w, root_quaternions_wxyz])):
        raise ValueError("Invalid motion array shapes or nonfinite values")
    starts = flags & ~np.vstack([np.zeros((1, bodies), bool), flags[:-1]])
    targets = np.zeros_like(positions)
    occupancy = np.zeros_like(flags)
    onset_indices = np.full(flags.shape, -1, dtype=np.int64)
    upcoming = np.full(bodies, t_count, dtype=int)
    for t in range(t_count - 1, -1, -1):
        upcoming[starts[t]] = t
        for b in range(bodies):
            onset = upcoming[b]
            if onset < t_count and onset - t <= horizon_steps:
                point = yaw_local(positions[onset, b], root_positions_w[t], root_quaternions_wxyz[t])
                if abs(point[0]) <= 0.8 and abs(point[1]) <= 0.5:
                    targets[t, b] = point
                    occupancy[t, b] = True
                    onset_indices[t, b] = onset
    return {"contact_flags": flags, "target_positions_body_yaw": targets,
            "target_occupancy": occupancy, "target_onset_indices": onset_indices,
            "future_complete": np.arange(t_count) + horizon_steps < t_count}


def measured_force_flags(force_w, threshold=1.0, minimum_steps=10):
    """Accept [T,B,3] or [T,H,B,3] control-cycle force history in world axes."""
    forces = np.asarray(force_w)
    if forces.ndim not in (3, 4) or forces.shape[-1] != 3 or not np.all(np.isfinite(forces)):
        raise ValueError("Expected finite [T,B,3] or [T,H,B,3] forces")
    vertical = np.abs(forces[..., 2])
    if forces.ndim == 4:
        vertical = vertical.max(axis=1)
    return filter_segments(vertical > threshold, minimum_steps)
