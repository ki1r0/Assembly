"""Keep every grasp option visible without repeating executor implementation data."""

_EXECUTOR_FIELDS = {
    'planned_waypoints_wxyz', 'planned_waypoint_joints', 'ik_pregrasp', 'ik_grasp',
    'measured_robot_tcp', 'part_component', 'measured_component',
}


def public_grasp_feedback(result):
    """Preserve estimates, screening and all candidate IDs; do not alter cached plans."""
    payload = {
        **result,
        'candidates': [
            {key: value for key, value in candidate.items() if key not in _EXECUTOR_FIELDS}
            for candidate in result.get('candidates', [])
        ],
        'candidate_payload_scope': 'All returned candidates. Public CAD/component geometry is shared once when identical. Cached IK joints and execution waypoints omitted from this model message; full records remain in the episode action log. These options are estimates, not grasp or retention guarantees.',
    }

    candidates = payload['candidates']
    # Public geometry remains available once, rather than repeated for every pose.
    for key in ('part_component', 'measured_component'):
        original = result.get('candidates', [])
        if original and key in original[0] and all(c.get(key) == original[0][key] for c in original):
            payload['shared_' + key] = original[0][key]
        else:
            for candidate, source in zip(candidates, original):
                if key in source:
                    candidate[key] = source[key]
    shared = {}
    for key in ('source', 'contact_width_source', 'gripper_geometry', 'feasibility', 'score_semantics', 'projection_source'):
        if candidates and key in candidates[0] and all(c.get(key) == candidates[0][key] for c in candidates):
            shared[key] = candidates[0][key]
            for candidate in candidates:
                candidate.pop(key)
    for field, keys in (
        ('assembly_compatibility', ('source', 'alignment_convention', 'release_clearance_scope')),
        ('contact_quality', ('model', 'margin_cost_semantics', 'assumed_friction_coefficient')),
    ):
        if not candidates or not all(field in c for c in candidates):
            continue
        for candidate in candidates:
            candidate[field] = dict(candidate[field])
        for key in keys:
            first = candidates[0][field]
            if key in first and all(c[field].get(key) == first[key] for c in candidates):
                shared[field + '.' + key] = first[key]
                for candidate in candidates:
                    candidate[field].pop(key)
    if shared:
        payload['metadata_shared_by_all_candidates'] = shared
    return payload
