"""Narrow, geometry-bound allowances for articulation housing overlaps."""
import numpy as np


def joint_housing_overlap(body_a, body_b, point, distance, anchors, rules):
    for rule in rules:
        if {body_a, body_b} != set(rule['body_pair']):
            continue
        if distance < -rule['maximum_penetration_m']:
            continue
        anchor = anchors[rule['joint']]
        if np.linalg.norm(np.asarray(point)-anchor) <= rule['maximum_contact_point_distance_to_joint_m']:
            return True
    return False
