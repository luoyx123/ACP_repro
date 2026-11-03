"""全身接触的环境过滤、几何准入与诊断；不改变 ACP 奖励或终止条件。"""
import torch


def reduce_contacts(geom, world, normal, frame, active, robot_geom, geom_group, n, num_groups=9):
    """Reduce packed contacts; self contacts never provide environment support.

    normal is the nonnegative local normal force. Return signed world normal
    vectors for the paper's existing vertical-force sensor, plus a scalar sum
    for diagnostics (opposite normals must not cancel this diagnostic).
    """
    valid = active & (geom >= 0).all(-1) & (geom < len(robot_geom)).all(-1)
    valid &= (world >= 0) & (world < n)
    safe_geom = geom.clamp(0, len(robot_geom)-1)
    robot = robot_geom[safe_geom]
    valid &= robot[:, 0] ^ robot[:, 1]
    force = normal.clamp_min(0)
    vector = force[:, None] * frame
    result = vector.new_zeros((n * num_groups, 3))
    magnitude = force.new_zeros(n * num_groups)
    for col, sign in ((0, -1), (1, 1)):
        group = geom_group[safe_geom[:, col]]
        selected = valid & robot[:, col] & (group >= 0)
        index = world.clamp(0, n-1) * num_groups + group.clamp_min(0)
        result.index_add_(0, index, torch.where(selected[:, None], sign * vector, 0.))
        magnitude.index_add_(0, index, torch.where(selected, force, 0.))
    return result.view(n, num_groups, 3), magnitude.view(n, num_groups)


def geometry_depths(geom, world, distance, active, robot_geom, n):
    """Maximum environment/self penetration per world, ignoring packed stale slots."""
    valid = active & (geom >= 0).all(-1) & (geom < len(robot_geom)).all(-1)
    valid &= (world >= 0) & (world < n)
    robot = robot_geom[geom.clamp(0, len(robot_geom)-1)]
    torch._assert_async((torch.isfinite(distance) | ~(valid & robot.any(-1))).all(),
                        'Non-finite robot contact distance')
    out = []
    for selected in (robot[:, 0] ^ robot[:, 1], robot.all(-1)):
        depths = torch.where(valid & selected, (-distance).clamp_min(0), 0.)
        out.append(distance.new_zeros(n).scatter_reduce_(0, world.clamp(0, n-1), depths,
                                                        reduce='amax', include_self=True))
    return tuple(out)


def contact_metrics(sim, names):
    """Separate paper Fz events from all-direction contact diagnostics."""
    measured = sim.measured_contacts()
    scalar = sim.normal_force_history.amax(1)
    env, self_depth = sim.geometry_depths()
    result = {'geometry/environment_penetration_max_m': env.max(),
              'geometry/self_penetration_max_m': self_depth.max(),
              'geometry/environment_over_3mm_fraction': (env > .003).float().mean(),
              'geometry/self_over_3mm_fraction': (self_depth > .003).float().mean()}
    for i, name in enumerate(names):
        result[f'contacts/{name}/paper_fz_fraction'] = measured[:, i].mean()
        result[f'contacts/{name}/normal_force_any_direction_fraction'] = (scalar[:, i] > 1.).float().mean()
        result[f'contacts/{name}/over_500N_fraction'] = (scalar[:, i] > 500.).float().mean()
    links = sim.link_force_history.amax(1)
    for i, name in enumerate(sim.link_names):
        result[f'all_links/{name}/normal_contact_fraction'] = (links[:, i] > 1.).float().mean()
        result[f'all_links/{name}/over_500N_fraction'] = (links[:, i] > 500.).float().mean()
    # One bounded transfer instead of one CUDA synchronization per scalar.
    values = torch.stack(list(result.values())).detach().cpu().tolist()
    return dict(zip(result, values))
