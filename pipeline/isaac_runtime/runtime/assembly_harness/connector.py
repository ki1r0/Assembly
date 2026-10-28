"""Resolve public feature relationships into current TCP goals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

import numpy as np


SUPPORTED_RELATIONS = frozenset({
    "coaxial",
    "parallel",
    "centered_on",
    "point_on_plane",
    "remain_clear_of",
})
_REGIMES = frozenset({"free_space", "contact", "grasp", "release"})
_MAX_FEATURES = 64
_MAX_CONSTRAINTS = 8


class GoalResolutionError(ValueError):
    """A public goal is stale, malformed, unsupported, or geometrically unsafe."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Feature:
    ref: str
    kind: str
    frame: str
    geometry: Mapping[str, Any]
    position: np.ndarray
    quaternion: np.ndarray
    axis: np.ndarray | None
    normal: np.ndarray | None
    position_std_m: float
    axis_std_rad: float
    extent: np.ndarray | None


@dataclass(frozen=True)
class ResolvedGoal:
    observation_id: str
    submitted_observation_id: str
    feature_refs: tuple[str, ...]
    motion_regime: str
    intent: str
    stop_condition: str
    target_tcp_pose: tuple[float, ...]
    tcp_to_moving_feature_pose: tuple[float, ...]
    uncertainty_position_m: float
    uncertainty_axis_rad: float
    grasp_width_m: float | None
    moving_geometry: Mapping[str, Any]
    moving_frame: str
    constraints: tuple[Mapping[str, str], ...]
    allowed_contact_pairs: tuple[Mapping[str, Any], ...]
    aperture_entry_supported: bool = False
    geometry_diagnostics: tuple[Mapping[str, Any], ...] = ()

    def as_public_dict(self) -> dict[str, Any]:
        summary = f"Resolved {len(self.constraints)} public geometric relation(s) to a world-frame TCP pose."
        if self.allowed_contact_pairs:
            sources = "; ".join(
                f"{item['moving_feature_ref']}->{item['reference_feature_ref']}"
                for item in self.allowed_contact_pairs
            )
            summary += f" Expected contact is limited to current RGB-D feature pair(s): {sources}."
        if self.aperture_entry_supported:
            summary += " Aperture entry ends at the measured rim plane; no collision exemption or hidden depth is authorized."
        return {
            "observation_id": self.submitted_observation_id,
            "feature_refs": list(self.feature_refs),
            "constraints": [dict(item) for item in self.constraints],
            "target_frame": "world",
            "target_geometry": {
                "tcp_pose": {
                    "position_m": list(self.target_tcp_pose[:3]),
                    "quaternion_wxyz": list(self.target_tcp_pose[3:]),
                },
            },
            "summary": summary[:1024],
        }


class AssemblyConnector:
    """Pure public-geometry connector; it never receives an environment handle."""

    def carry_held_surface_track(
        self,
        goal: Mapping[str, Any],
        previous_observation: Mapping[str, Any],
        current_observation: Mapping[str, Any],
        *,
        moving_tcp_feature_pose: Sequence[float] | None,
        max_match_distance_m: float = 0.03,
    ) -> Mapping[str, Any]:
        """Carry a last-seen RGB-D patch for short free-space motion under verified grasp."""
        if (
            goal.get("motion_regime") != "free_space"
            or moving_tcp_feature_pose is None
            or not _bilateral_grasp_for_tracking(previous_observation)
            or not _bilateral_grasp_for_tracking(current_observation)
        ):
            return current_observation
        constraints = goal.get("constraints", ())
        moving_refs = {
            item.get("moving_feature") for item in constraints if isinstance(item, Mapping)
        }
        if len(moving_refs) != 1:
            return current_observation

        previous_id = _text(previous_observation.get("observation_id"), "previous observation_id")
        current_id = _text(current_observation.get("observation_id"), "current observation_id")
        previous_sensors = _robot_sensors(previous_observation)
        current_sensors = _robot_sensors(current_observation)
        previous_camera = _camera_observation_id(previous_observation, previous_sensors)
        current_camera = _camera_observation_id(current_observation, current_sensors)
        previous_tcp = _pose(previous_sensors.get("tcp_pose"), "previous tcp_pose")
        current_tcp = _pose(current_sensors.get("tcp_pose"), "current tcp_pose")
        old_ref = next(iter(moving_refs))
        old_raw = _features_by_ref(previous_observation).get(old_ref)
        if old_raw is None:
            return current_observation
        old = _parse_feature(old_raw, previous_id, previous_camera, previous_observation, previous_tcp[3:])
        if old.kind != "depth_surface_patch" or old.frame not in {"world", "world_m"}:
            return current_observation

        predicted_position, predicted_quaternion = _compose_tcp_feature(
            current_tcp, _pose(moving_tcp_feature_pose, "moving_tcp_feature_pose"),
        )
        old_geometry = old_raw["geometry"]
        origin_value = old_geometry.get("track_origin_position_m")
        origin_position = (
            old.position.copy() if origin_value is None
            else _vector(origin_value, 3, "track origin position")
        )
        if float(np.linalg.norm(predicted_position - origin_position)) > max_match_distance_m:
            return current_observation

        # A new RGB-D surface is preferred; same-camera features are stale robot-state companions.
        if current_camera != previous_camera:
            direct_matches = []
            for raw in current_observation.get("features", ()):
                if not isinstance(raw, Mapping) or raw.get("kind") != old.kind or raw.get("frame") != old.frame:
                    continue
                geometry = raw.get("geometry", {})
                if not isinstance(geometry, Mapping) or geometry.get("position_kind") == "kinematic_tcp_track_from_rgbd_surface":
                    continue
                try:
                    candidate = _parse_feature(raw, current_id, current_camera, current_observation, current_tcp[3:])
                except GoalResolutionError:
                    continue
                if old.extent is not None and candidate.extent is not None:
                    if float(np.linalg.norm(old.extent - candidate.extent)) > max(0.02, max_match_distance_m):
                        continue
                distance = float(np.linalg.norm(candidate.position - predicted_position))
                if distance <= max_match_distance_m:
                    direct_matches.append((distance, candidate.ref))
            direct_matches.sort()
            if direct_matches and (
                len(direct_matches) == 1 or direct_matches[1][0] - direct_matches[0][0] >= 0.002
            ):
                return current_observation

        stale_geometry_keys = {
            "pixel_bbox_xyxy", "pixel_centroid_xy", "model_image_pixel_centroid_xy",
            "model_image_pixel_bbox_xyxy", "image_ellipse_candidate_px", "image_circle_candidate_px",
            "pointcloud_world_m", "axis", "axis_unit", "principal_axis_unit", "normal",
            "normal_unit", "surface_normal_unit", "axis_kind", "normal_kind",
        }
        geometry = {key: value for key, value in old_geometry.items() if key not in stale_geometry_keys}
        geometry.update({
            "position_m": predicted_position.tolist(),
            "quaternion_wxyz": predicted_quaternion.tolist(),
            "position_kind": "kinematic_tcp_track_from_rgbd_surface",
            "track_origin_position_m": origin_position.tolist(),
            "track_origin_source_observation_id": old_geometry.get(
                "track_origin_source_observation_id", old_raw.get("source_observation_id"),
            ),
        })
        uncertainty = dict(old_raw.get("uncertainty", {}))
        uncertainty["position_source"] = "last public RGB-D patch carried by measured TCP; grip slip is unobserved"
        feature_ref = f"held_track:{current_id[-145:]}"
        tracked = {
            **{key: value for key, value in old_raw.items() if key not in {
                "feature_ref", "geometry", "description", "axis", "axis_unit", "normal", "normal_unit",
            }},
            "feature_ref": feature_ref,
            "description": "Short-range kinematic estimate of a previously observed RGB-D surface; not current pixel evidence or contact/axis evidence.",
            "frame": old.frame,
            "geometry": geometry,
            "uncertainty": uncertainty,
            "source_observation_id": current_id,
        }
        result = dict(current_observation)
        result["features"] = [
            *[feature for feature in current_observation.get("features", ())
              if not (current_camera == previous_camera and isinstance(feature, Mapping)
                      and feature.get("feature_ref") == old_ref)],
            tracked,
        ]
        return result

    def rebind_goal_features(
        self,
        goal: Mapping[str, Any],
        previous_observation: Mapping[str, Any],
        current_observation: Mapping[str, Any],
        *,
        moving_tcp_feature_pose: Sequence[float] | None = None,
        max_match_distance_m: float = 0.03,
    ) -> dict[str, Any]:
        """Rebind refs across camera frames by geometry, never by component rank."""
        previous_id = _text(previous_observation.get("observation_id"), "previous observation_id")
        current_id = _text(current_observation.get("observation_id"), "current observation_id")
        previous_sensors = _robot_sensors(previous_observation)
        current_sensors = _robot_sensors(current_observation)
        previous_camera = _camera_observation_id(previous_observation, previous_sensors)
        current_camera = _camera_observation_id(current_observation, current_sensors)
        previous_features = _features_by_ref(previous_observation)
        current_features = _features_by_ref(current_observation)
        current_tcp = _pose(current_sensors.get("tcp_pose"), "robot_state.sensors.tcp_pose")
        previous_tcp = _pose(previous_sensors.get("tcp_pose"), "robot_state.sensors.tcp_pose")
        bindings: dict[str, str] = {}

        for item in goal.get("constraints", ()):
            if not isinstance(item, Mapping):
                raise GoalResolutionError("invalid_goal", "constraint must be a mapping")
            for side in ("moving_feature", "reference_feature"):
                old_ref = _text(item.get(side), side)
                if old_ref in bindings:
                    continue
                old_raw = previous_features.get(old_ref)
                if old_raw is None:
                    raise GoalResolutionError("stale_feature", f"cannot track missing feature ref {old_ref!r}")
                old = _parse_feature(old_raw, previous_id, previous_camera, previous_observation, previous_tcp[3:])

                if old.kind == "robot_tcp":
                    tcp_candidates: list[tuple[_Feature, Mapping[str, Any]]] = []
                    for raw in current_features.values():
                        try:
                            candidate = _parse_feature(
                                raw, current_id, current_camera, current_observation, current_tcp[3:]
                            )
                        except GoalResolutionError:
                            continue
                        if (
                            candidate.kind == "robot_tcp"
                            and candidate.frame == old.frame
                            and _matches_robot_tcp_feature(candidate, current_tcp)
                        ):
                            tcp_candidates.append((candidate, raw))
                    if len(tcp_candidates) != 1:
                        code = "ambiguous_feature_tracking" if tcp_candidates else "feature_relocalization_required"
                        raise GoalResolutionError(code, "goal requires exactly one current FK-matched robot_tcp feature")
                    current_tcp_feature = tcp_candidates[0][0]
                    if current_tcp_feature.ref in bindings.values():
                        raise GoalResolutionError(
                            "ambiguous_feature_tracking", "two submitted refs matched one current feature"
                        )
                    bindings[old_ref] = current_tcp_feature.ref
                    continue

                predicted_position = old.position
                if side == "moving_feature" and old.frame == "world" and moving_tcp_feature_pose is not None:
                    if current_sensors.get("grasped") is True:
                        predicted_position = _compose_tcp_feature(current_tcp, moving_tcp_feature_pose)[0]
                elif old.frame == "tcp":
                    predicted_position = _rotate(np.asarray(current_tcp[3:]), _vector(
                        old_raw.get("geometry", {}).get("position_m", old_raw.get("geometry", {}).get("position")),
                        3, "feature.geometry.position_m"
                    )) + np.asarray(current_tcp[:3])

                candidates: list[tuple[float, _Feature, Mapping[str, Any]]] = []
                for raw in current_features.values():
                    try:
                        candidate = _parse_feature(
                            raw, current_id, current_camera, current_observation, current_tcp[3:]
                        )
                    except GoalResolutionError:
                        continue
                    if candidate.kind != old.kind or candidate.frame != old.frame:
                        continue
                    distance = float(np.linalg.norm(candidate.position - predicted_position))
                    if old.extent is not None and candidate.extent is not None:
                        extent_error = float(np.linalg.norm(old.extent - candidate.extent))
                        if extent_error > max(0.02, max_match_distance_m):
                            continue
                    candidates.append((distance, candidate, raw))
                candidates.sort(key=lambda entry: entry[0])
                if side == "moving_feature" and old.kind == "depth_surface_patch":
                    tracked = [
                        entry for entry in candidates
                        if entry[2].get("geometry", {}).get("position_kind")
                        == "kinematic_tcp_track_from_rgbd_surface"
                    ]
                    if tracked:
                        candidates = tracked
                if not candidates or candidates[0][0] > max_match_distance_m:
                    raise GoalResolutionError("feature_relocalization_required", f"no geometric match for {old_ref!r}")
                if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.002:
                    raise GoalResolutionError("ambiguous_feature_tracking", f"multiple current features match {old_ref!r}")
                if candidates[0][1].ref in bindings.values():
                    raise GoalResolutionError("ambiguous_feature_tracking", "two submitted refs matched one current feature")
                bindings[old_ref] = candidates[0][1].ref

        rebound = dict(goal)
        rebound["observation_id"] = current_id
        rebound["constraints"] = [
            {**item,
             "moving_feature": bindings[item["moving_feature"]],
             "reference_feature": bindings[item["reference_feature"]]}
            for item in goal["constraints"]
        ]
        return rebound

    def resolve(self, goal: Mapping[str, Any], observation: Mapping[str, Any]) -> ResolvedGoal:
        if not isinstance(goal, Mapping) or not isinstance(observation, Mapping):
            raise GoalResolutionError("invalid_input", "goal and observation must be mappings")
        if goal.get("tool", "execute_goal") != "execute_goal":
            raise GoalResolutionError("invalid_goal", "goal tool must be execute_goal")

        observation_id = _text(observation.get("observation_id"), "observation_id")
        submitted_id = _text(goal.get("observation_id"), "goal.observation_id")
        if submitted_id != observation_id:
            raise GoalResolutionError("stale_goal", "goal observation_id does not match current observation")

        regime = goal.get("motion_regime")
        if not isinstance(regime, str) or regime not in _REGIMES:
            raise GoalResolutionError("invalid_goal", f"motion_regime must be one of {sorted(_REGIMES)}")
        intent = _text(goal.get("intent"), "goal.intent", maximum=1024)
        stop_condition = _text(goal.get("stop_condition"), "goal.stop_condition", maximum=1024)

        raw_features = observation.get("features")
        if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
            raise GoalResolutionError("invalid_observation", "features must be a sequence")
        if len(raw_features) > _MAX_FEATURES:
            raise GoalResolutionError("invalid_observation", "too many public feature estimates")
        sensors = _robot_sensors(observation)
        tcp_pose = _pose(sensors.get("tcp_pose"), "robot_state.sensors.tcp_pose")
        camera_id = _camera_observation_id(observation, sensors)

        raw_features_by_ref: dict[str, Mapping[str, Any]] = {}
        for raw in raw_features:
            if not isinstance(raw, Mapping):
                raise GoalResolutionError("invalid_observation", "each feature must be a mapping")
            ref = _text(raw.get("feature_ref"), "feature.feature_ref")
            if ref in raw_features_by_ref:
                raise GoalResolutionError("invalid_observation", "feature_ref values must be unique")
            raw_features_by_ref[ref] = raw

        raw_constraints = goal.get("constraints")
        if (
            not isinstance(raw_constraints, Sequence)
            or isinstance(raw_constraints, (str, bytes))
            or not 1 <= len(raw_constraints) <= _MAX_CONSTRAINTS
        ):
            raise GoalResolutionError("invalid_goal", "constraints must contain one to eight relations")

        constraints: list[tuple[str, _Feature, _Feature]] = []
        moving_refs: set[str] = set()
        used_refs: list[str] = []
        for item in raw_constraints:
            if not isinstance(item, Mapping) or set(item) != {
                "relation", "moving_feature", "reference_feature",
            }:
                raise GoalResolutionError("invalid_goal", "each constraint requires relation and two feature refs")
            relation = item["relation"]
            if not isinstance(relation, str) or relation not in SUPPORTED_RELATIONS:
                raise GoalResolutionError("unsupported_relation", f"unsupported relation: {relation!r}")
            moving_ref = _text(item["moving_feature"], "moving_feature")
            reference_ref = _text(item["reference_feature"], "reference_feature")
            moving_raw = raw_features_by_ref.get(moving_ref)
            reference_raw = raw_features_by_ref.get(reference_ref)
            if moving_raw is None or reference_raw is None:
                raise GoalResolutionError("stale_feature", "goal refs must exist in the current observation")
            moving = _parse_feature(moving_raw, observation_id, camera_id, observation, tcp_pose[3:])
            reference = _parse_feature(reference_raw, observation_id, camera_id, observation, tcp_pose[3:])
            moving_refs.add(moving_ref)
            for ref in (moving_ref, reference_ref):
                if ref not in used_refs:
                    used_refs.append(ref)
            constraints.append((relation, moving, reference))
        if len(moving_refs) != 1:
            raise GoalResolutionError("unsupported_goal", "one goal may move exactly one feature frame")

        moving_ref = next(iter(moving_refs))
        moving_feature = _parse_feature(
            raw_features_by_ref[moving_ref], observation_id, camera_id, observation, tcp_pose[3:]
        )
        delta_q = np.array((1.0, 0.0, 0.0, 0.0), dtype=float)
        delta_t = np.zeros(3, dtype=float)
        for relation, moving, reference in constraints:
            current_position = _rotate(delta_q, moving.position) + delta_t
            current_axis = _rotate(delta_q, moving.axis) if moving.axis is not None else None
            if relation in {"coaxial", "parallel"}:
                if current_axis is None or reference.axis is None:
                    raise GoalResolutionError("missing_geometry", f"{relation} requires position and axis geometry")
                ref_axis = reference.axis.copy()
                if float(np.dot(current_axis, ref_axis)) < 0.0:
                    ref_axis = -ref_axis
                rotation = _quat_between(current_axis, ref_axis)
                rotated_position = _rotate(rotation, current_position)
                if relation == "coaxial":
                    target_position = reference.position + ref_axis * float(
                        np.dot(rotated_position - reference.position, ref_axis)
                    )
                else:
                    target_position = current_position
                step_translation = target_position - rotated_position
                delta_t = _rotate(rotation, delta_t) + step_translation
                delta_q = _quat_normalize(_quat_multiply(rotation, delta_q))
            elif relation == "centered_on":
                target_position = _center_target_position(reference, grasp=regime == "grasp")
                step_translation = target_position - current_position
                delta_t += step_translation
            elif relation == "point_on_plane":
                if reference.normal is None:
                    raise GoalResolutionError("missing_geometry", "point_on_plane requires a reference normal")
                signed_distance = float(np.dot(current_position - reference.position, reference.normal))
                delta_t -= reference.normal * signed_distance
            elif relation == "remain_clear_of":
                clearance_position = _clearance_target_position(moving, reference, current_position)
                delta_t += clearance_position - current_position

        final_position = _rotate(delta_q, moving_feature.position) + delta_t
        final_quaternion = _quat_normalize(_quat_multiply(delta_q, moving_feature.quaternion))
        _validate_final_relations(constraints, delta_q, delta_t, grasp=regime == "grasp")

        tcp_position = np.asarray(tcp_pose[:3], dtype=float)
        tcp_quaternion = np.asarray(tcp_pose[3:], dtype=float)
        tcp_to_feature_position = _rotate(_quat_inverse(tcp_quaternion), moving_feature.position - tcp_position)
        tcp_to_feature_quaternion = _quat_normalize(
            _quat_multiply(_quat_inverse(tcp_quaternion), moving_feature.quaternion)
        )
        goal_tcp_quaternion = _quat_normalize(
            _quat_multiply(final_quaternion, _quat_inverse(tcp_to_feature_quaternion))
        )
        goal_tcp_position = final_position - _rotate(goal_tcp_quaternion, tcp_to_feature_position)

        grasp_width = None
        if regime == "grasp":
            if not _matches_robot_tcp_feature(moving_feature, tcp_pose):
                raise GoalResolutionError(
                    "invalid_grasp_geometry",
                    "grasp goals require a current TCP-local or FK-matched robot_tcp moving_feature",
                )
            widths = [
                _grasp_width_for_axis(
                    reference,
                    _rotate(delta_q, moving.normal) if moving.normal is not None else None,
                )
                for _, moving, reference in constraints
                if reference.geometry.get("grasp_width_m") is not None
            ]
            if not widths or any(not isfinite(width) or width <= 0.0 for width in widths):
                raise GoalResolutionError("missing_geometry", "grasp requires a positive geometry-derived grasp_width_m")
            if max(widths) - min(widths) > 1e-6:
                raise GoalResolutionError("inconsistent_geometry", "referenced grasp widths disagree")
            grasp_width = widths[0]

        parsed_features = {
            feature.ref: feature
            for _, moving, reference in constraints
            for feature in (moving, reference)
        }
        uncertainty = max(parsed_features[ref].position_std_m for ref in used_refs)
        axis_refs: set[str] = set()
        for relation, moving, reference in constraints:
            if relation in {"coaxial", "parallel"}:
                axis_refs.update((moving.ref, reference.ref))
            elif relation == "point_on_plane":
                axis_refs.add(reference.ref)
            elif relation == "remain_clear_of":
                normal_kind = reference.geometry.get("normal_kind")
                if reference.normal is not None and (
                    reference.kind == "plane_candidate"
                    or normal_kind in {
                        "rgbd_plane_normal",
                        "surface_patch_normal_not_hole_axis",
                        "local_depth_surface_normal",
                        "fitted_rim_surface_normal",
                    }
                ):
                    axis_refs.update((moving.ref, reference.ref))
            if regime == "grasp" and moving.normal is not None:
                axis_refs.add(moving.ref)
            if regime == "grasp" and _uses_reliable_grasp_projection(reference):
                axis_refs.add(reference.ref)
        axis_uncertainty = max(
            (parsed_features[ref].axis_std_rad for ref in axis_refs), default=0.0,
        )
        tcp_to_moving = (*tcp_to_feature_position.tolist(), *tcp_to_feature_quaternion.tolist())
        public_constraints = tuple(
            {
                "relation": relation,
                "moving_feature": moving.ref,
                "reference_feature": reference.ref,
            }
            for relation, moving, reference in constraints
        )
        submitted_constraints = goal.get("_submitted_constraints", public_constraints)
        submitted_refs = goal.get("_submitted_feature_refs", used_refs)
        if not isinstance(submitted_constraints, Sequence) or not isinstance(submitted_refs, Sequence):
            raise GoalResolutionError("invalid_goal", "internal submitted-goal metadata is malformed")
        allowed_contacts = _expected_contact_pairs(
            regime, constraints, delta_q, delta_t, observation, sensors,
        )
        entry_constraints = [(moving, reference) for relation, moving, reference in constraints
                             if relation == "point_on_plane"]
        aperture_entry_supported = sensors.get("grasped") is True and bool(entry_constraints) and all(
            has_observed_aperture_geometry(raw_features_by_ref[reference.ref])
            and moving.frame in {"world", "world_m"}
            and moving.axis is not None
            and isinstance(moving.geometry.get("axis_kind"), str)
            and moving.geometry.get("axis_kind") not in {
                None, "largest_pca_extent_not_object_or_hole_axis",
                "fitted_circular_rim_plane_normal_not_semantic_part_axis",
            }
            and any(r == "coaxial" and m.ref == moving.ref and t.ref == reference.ref
                    for r, m, t in constraints)
            for moving, reference in entry_constraints
        )
        geometry_diagnostics = []
        for relation, moving, reference in constraints:
            if relation != "point_on_plane":
                continue
            point = _rotate(delta_q, moving.position) + delta_t
            points = _feature_points_world(reference, observation)
            distances = np.linalg.norm(points - point, axis=1)
            radius = min(0.008, max(0.003, 3.0 * reference.position_std_m))
            virtual = reference.kind in {"dark_aperture_candidate", "recessed_circular_candidate"}
            geometry_diagnostics.append({
                "moving_feature_ref": moving.ref,
                "reference_feature_ref": reference.ref,
                "moving_source_frame": moving.frame,
                "reference_source_frame": reference.frame,
                "frame": "world", "length_unit": "m", "time_unit": "s",
                "pose_convention": "xyz_m_quaternion_wxyz",
                "observation_id": observation_id,
                "camera_observation_id": camera_id,
                "camera_timestamp_s": sensors.get("rgbd_source_timestamp_s"),
                "robot_timestamp_s": sensors.get("robot_timestamp_s"),
                "moving_position_world_m": moving.position.tolist(),
                "reference_position_world_m": reference.position.tolist(),
                "goal_point_world_m": point.tolist(),
                "tcp_to_moving_feature_pose": list(tcp_to_moving),
                "reference_normal_world": reference.normal.tolist(),
                "moving_axis_world": moving.axis.tolist() if moving.axis is not None else None,
                "moving_axis_kind": moving.geometry.get("axis_kind"),
                "aperture_rim_supported": has_observed_aperture_geometry(raw_features_by_ref[reference.ref]),
                "aperture_entry_supported": aperture_entry_supported,
                "surface_sample_count": int(np.count_nonzero(distances <= radius)),
                "nearest_surface_sample_distance_m": float(distances.min()) if len(distances) else None,
                "surface_support_radius_m": radius,
                "reference_semantics": "virtual_aperture_plane" if virtual else "surface_plane",
                "surface_support": "not_required_for_localization" if virtual else (
                    "supported" if np.any(distances <= radius) else "unknown"
                ),
                "required_precondition": "observed_rim_reliable_moving_axis_and_coaxial_alignment" if virtual else "solid_samples_at_goal_point",
            })
        return ResolvedGoal(
            observation_id=observation_id,
            submitted_observation_id=_text(goal.get("_submitted_observation_id", submitted_id), "submitted observation_id"),
            feature_refs=tuple(str(ref) for ref in submitted_refs),
            motion_regime=regime,
            intent=intent,
            stop_condition=stop_condition,
            target_tcp_pose=(*goal_tcp_position.tolist(), *goal_tcp_quaternion.tolist()),
            tcp_to_moving_feature_pose=tuple(float(value) for value in tcp_to_moving),
            uncertainty_position_m=uncertainty,
            uncertainty_axis_rad=axis_uncertainty,
            grasp_width_m=grasp_width,
            moving_geometry=moving_feature.geometry,
            moving_frame=moving_feature.frame,
            constraints=tuple(dict(item) for item in submitted_constraints),
            allowed_contact_pairs=allowed_contacts,
            aperture_entry_supported=aperture_entry_supported,
            geometry_diagnostics=tuple(geometry_diagnostics),
        )


def has_observed_aperture_geometry(feature: Mapping[str, Any]) -> bool:
    """A measured rim locates a virtual entry plane; its center need not have depth."""
    if (feature.get("frame") not in {"world", "world_m"}
            or feature.get("kind") not in {"dark_aperture_candidate", "recessed_circular_candidate"}):
        return False
    geometry = feature.get("geometry", {})
    if not isinstance(geometry, Mapping) or geometry.get("normal_kind") != "fitted_rim_surface_normal":
        return False
    ellipse = geometry.get("image_ellipse_candidate_px", {})
    if not isinstance(ellipse, Mapping):
        return False
    if isinstance(ellipse.get("angular_coverage_fraction"), bool) or isinstance(geometry.get("rim_plane_fit_rms_m"), bool):
        return False
    try:
        coverage = float(ellipse.get("angular_coverage_fraction", 0.0))
        points = np.asarray(geometry.get("pointcloud_world_m", ()), dtype=float)
        normal = np.asarray(geometry.get("normal", ()), dtype=float)
        axis = np.asarray(geometry.get("axis", geometry.get("axis_unit", ())), dtype=float)
        center = np.asarray(geometry.get("position_m", ()), dtype=float)
        extent = np.asarray(geometry.get("extent_m", ()), dtype=float)
        plane_rms = float(geometry.get("rim_plane_fit_rms_m", float("nan")))
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(coverage) and 0.58 <= coverage <= 1.0
        and points.ndim == 2 and points.shape[1:] == (3,) and 12 <= len(points) <= 512
        and normal.shape == (3,) and axis.shape == (3,) and center.shape == (3,) and extent.shape == (3,)
        and np.isfinite(points).all() and np.isfinite(normal).all() and np.isfinite(center).all()
        and np.isfinite(axis).all() and np.linalg.norm(axis) > 1e-12
        and np.isfinite(extent).all() and np.all(extent[:2] > 0)
        and np.isfinite(plane_rms) and 0 <= plane_rms <= max(0.002, 0.25 * float(min(extent[:2])))
        and np.linalg.norm(normal) > 1e-12
        and np.allclose(abs(np.dot(axis, normal) / (np.linalg.norm(axis) * np.linalg.norm(normal))), 1.0)
        and np.linalg.matrix_rank(points - points.mean(axis=0)) >= 2
        and np.max(np.abs((points - center) @ (normal / np.linalg.norm(normal))))
            <= max(0.002, 3.0 * plane_rms)
    )


def has_observed_grasp_contact_geometry(
    observation: Mapping[str, Any], target_feature_ref: str,
) -> bool:
    """Return whether current public RGB-D supports both predicted finger sites."""
    features = observation.get("features") if isinstance(observation, Mapping) else None
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        return False
    tcp_refs = [
        feature.get("feature_ref")
        for feature in features
        if isinstance(feature, Mapping)
        and feature.get("kind") == "robot_tcp"
        and feature.get("frame") == "tcp"
        and isinstance(feature.get("feature_ref"), str)
    ]
    if len(tcp_refs) != 1 or not isinstance(target_feature_ref, str):
        return False
    goal = {
        "tool": "execute_goal",
        "observation_id": observation.get("observation_id"),
        "intent": "Check current observed grasp contact geometry.",
        "constraints": [{
            "relation": "centered_on",
            "moving_feature": tcp_refs[0],
            "reference_feature": target_feature_ref,
        }],
        "motion_regime": "grasp",
        "stop_condition": "Contact geometry is either supported or unavailable.",
    }
    try:
        resolved = AssemblyConnector().resolve(goal, observation)
    except (GoalResolutionError, TypeError, ValueError, ArithmeticError):
        return False
    if len(resolved.allowed_contact_pairs) != 1:
        return False

    try:
        observation_id = _text(observation.get("observation_id"), "observation_id")
        sensors = _robot_sensors(observation)
        camera_id = _camera_observation_id(observation, sensors)
        obstacle_map = sensors.get("rgbd_obstacle_map")
        if (
            not isinstance(obstacle_map, Mapping)
            or obstacle_map.get("source_observation_id") not in {observation_id, camera_id}
            or obstacle_map.get("frame") not in {"world", "world_m"}
            or obstacle_map.get("valid") is not True
            or obstacle_map.get("robot_self_filtered") is not True
        ):
            return False
        points = obstacle_map.get("points_m")
        if not isinstance(points, np.ndarray) and (
            not isinstance(points, Sequence) or isinstance(points, (str, bytes))
        ):
            return False
        if len(points) > 200_000:
            return False
        cloud = np.asarray(points, dtype=float)
        if cloud.size and (cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all()):
            return False
        bind_contact_surface_points(resolved.allowed_contact_pairs, cloud.reshape(-1, 3))
    except (GoalResolutionError, TypeError, ValueError, ArithmeticError):
        return False
    return True


def bind_contact_surface_points(
    pairs: Sequence[Mapping[str, Any]], obstacle_points: Sequence[Sequence[float]],
) -> tuple[Mapping[str, Any], ...]:
    """Bind predicted contact sites only to matching current RGB-D map samples."""
    cloud = np.asarray(obstacle_points, dtype=float).reshape(-1, 3)
    bound_pairs = []
    for pair in pairs:
        tolerance = float(pair["point_match_tolerance_m"])
        bound_sites = []
        for site in pair["contact_sites"]:
            samples = np.asarray(site["surface_points_world_m"], dtype=float)
            matched_indices: set[int] = set()
            for sample in samples:
                distances_sq = np.sum((cloud - sample) ** 2, axis=1)
                nearest = int(np.argmin(distances_sq)) if len(cloud) else -1
                if nearest >= 0 and distances_sq[nearest] <= tolerance * tolerance:
                    matched_indices.add(nearest)
            current_surface_points = cloud[sorted(matched_indices)] if matched_indices else np.empty((0, 3))
            if not len(current_surface_points):
                raise GoalResolutionError(
                    "contact_surface_missing",
                    f"RGB-D obstacle map does not preserve the current contact surface for {pair['reference_feature_ref']!r}",
                )
            bound_sites.append({
                "position_world_m": site["position_world_m"],
                "surface_points_world_m": current_surface_points.tolist(),
            })
        bound_pairs.append({
            **pair,
            "point_match_tolerance_m": 1e-6,
            "contact_sites": bound_sites,
        })
    return tuple(bound_pairs)


def _parse_feature(
    raw: Any,
    observation_id: str,
    camera_observation_id: str,
    observation: Mapping[str, Any],
    tcp_quaternion: np.ndarray,
) -> _Feature:
    if not isinstance(raw, Mapping):
        raise GoalResolutionError("invalid_observation", "each feature must be a mapping")
    ref = _text(raw.get("feature_ref"), "feature.feature_ref")
    source_id = _text(raw.get("source_observation_id"), "feature.source_observation_id")
    if source_id not in {observation_id, camera_observation_id}:
        raise GoalResolutionError("stale_feature", "feature source_observation_id is not from this robot/RGB-D snapshot")
    kind = _text(raw.get("kind"), "feature.kind")
    frame = _text(raw.get("frame"), "feature.frame")
    geometry = raw.get("geometry")
    uncertainty = raw.get("uncertainty", {})
    if not isinstance(geometry, Mapping) or not isinstance(uncertainty, Mapping):
        raise GoalResolutionError("invalid_observation", "feature geometry and uncertainty must be mappings")
    position = _vector(geometry.get("position_m", geometry.get("position")), 3, "feature.geometry.position_m")
    quaternion_value = geometry.get("quaternion_wxyz", geometry.get("quaternion"))
    quaternion = _quat_normalize(_vector(quaternion_value, 4, "feature.geometry.quaternion_wxyz")) \
        if quaternion_value is not None else np.asarray(tcp_quaternion, dtype=float).copy()
    axis_value = geometry.get("axis", geometry.get("axis_unit", raw.get("axis_unit")))
    normal_value = geometry.get(
        "normal", geometry.get("normal_unit", geometry.get("surface_normal_unit", raw.get("normal_unit")))
    )
    axis = _unit(_vector(axis_value, 3, "feature.geometry.axis"), "feature.geometry.axis") \
        if axis_value is not None else None
    normal = _unit(_vector(normal_value, 3, "feature.geometry.normal"), "feature.geometry.normal") \
        if normal_value is not None else None
    std = uncertainty.get("position_std_m", uncertainty.get("calibrated_measurement_error_m"))
    if isinstance(std, Sequence) and not isinstance(std, (str, bytes)):
        std_values = _vector(std, 3, "feature.uncertainty.position_std_m")
        if np.any(std_values < 0.0):
            raise GoalResolutionError("invalid_uncertainty", "position standard deviations must be non-negative")
        std = float(np.linalg.norm(std_values))
    if isinstance(std, bool) or not isinstance(std, (int, float)) or not isfinite(float(std)) or float(std) < 0.0:
        raise GoalResolutionError("invalid_uncertainty", "a calibrated scalar or xyz position_std_m is required")
    axis_std = uncertainty.get("axis_std_rad", 0.0)
    if isinstance(axis_std, bool) or not isinstance(axis_std, (int, float)) or not isfinite(float(axis_std)) or float(axis_std) < 0.0:
        raise GoalResolutionError("invalid_uncertainty", "uncertainty.axis_std_rad must be finite and non-negative")
    if (axis is not None or normal is not None) and "axis_std_rad" not in uncertainty:
        raise GoalResolutionError("invalid_uncertainty", "axis/normal features require uncertainty.axis_std_rad")

    frame_q, frame_p = _world_frame_transform(frame, observation)
    position = _rotate(frame_q, position) + frame_p
    quaternion = _quat_normalize(_quat_multiply(frame_q, quaternion))
    if axis is not None:
        axis = _unit(_rotate(frame_q, axis), "feature axis")
    if normal is not None:
        normal = _unit(_rotate(frame_q, normal), "feature normal")
    extent_value = geometry.get("extent_m")
    extent = _vector(extent_value, 3, "feature.geometry.extent_m") if extent_value is not None else None
    if extent is not None and np.any(extent < 0.0):
        raise GoalResolutionError("invalid_geometry", "feature.geometry.extent_m must be non-negative")
    return _Feature(ref, kind, frame, geometry, position, quaternion, axis, normal,
                    float(std), float(axis_std), extent)


def _expected_contact_pairs(
    regime: str,
    constraints: Sequence[tuple[str, _Feature, _Feature]],
    delta_q: np.ndarray,
    delta_t: np.ndarray,
    observation: Mapping[str, Any],
    sensors: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    if regime not in {"grasp", "contact", "release"}:
        return ()
    tcp_pose = _pose(_robot_sensors(observation).get("tcp_pose"), "robot_state.sensors.tcp_pose")
    pairs: list[Mapping[str, Any]] = []
    for relation, moving, reference in constraints:
        if relation == "remain_clear_of":
            continue
        if regime == "grasp":
            if moving.kind != "robot_tcp" or reference.geometry.get("grasp_width_m") is None:
                continue
            if moving.normal is None:
                continue
            axis = _unit(_rotate(delta_q, moving.normal), "gripper opening axis")
            width = _grasp_width_for_axis(reference, axis)
            activation_center = _rotate(delta_q, moving.position) + delta_t
            contact_centers = (
                activation_center - axis * (width * 0.5),
                activation_center + axis * (width * 0.5),
            )
            role = "gripper"
            activation_radius = width * 0.5 + 0.008
        elif relation == "point_on_plane" and reference.normal is not None:
            if reference.kind in {"dark_aperture_candidate", "recessed_circular_candidate"}:
                # A virtual entry plane never authorizes collision with rim, walls, or background.
                continue
            contact_center = _rotate(delta_q, moving.position) + delta_t
            contact_centers = (contact_center,)
            if sensors.get("grasped") is True and moving.frame in {"world", "world_m"}:
                role = "held_object"
            elif _matches_robot_tcp_feature(moving, tcp_pose):
                role = "gripper"
            else:
                continue
            activation_center = contact_center
            activation_radius = 0.008
        else:
            continue

        surface_points = _feature_points_world(reference, observation)
        if len(surface_points) == 0:
            continue
        contact_radius = min(0.008, max(0.003, 3.0 * reference.position_std_m))
        sites = []
        for center in contact_centers:
            distances = np.linalg.norm(surface_points - center, axis=1)
            site_points = surface_points[distances <= contact_radius]
            if len(site_points) == 0:
                sites = []
                break
            sites.append({
                "position_world_m": center.tolist(),
                "surface_points_world_m": site_points.tolist(),
            })
        if len(sites) != len(contact_centers):
            continue
        pairs.append({
            "moving_feature_ref": moving.ref,
            "reference_feature_ref": reference.ref,
            "moving_role": role,
            "activation_center_world_m": activation_center.tolist(),
            "activation_radius_m": activation_radius,
            "contact_radius_m": contact_radius,
            "point_match_tolerance_m": contact_radius,
            "contact_sites": sites,
        })
    return tuple(pairs)


def _matches_robot_tcp_feature(feature: _Feature, tcp_pose: Sequence[float]) -> bool:
    if feature.kind != "robot_tcp":
        return False
    if feature.frame == "tcp":
        return True
    if feature.frame not in {"world", "world_m"}:
        return False
    tcp_position = np.asarray(tcp_pose[:3], dtype=float)
    tcp_quaternion = _quat_normalize(np.asarray(tcp_pose[3:], dtype=float))
    position_error = float(np.linalg.norm(feature.position - tcp_position))
    orientation_dot = abs(float(np.dot(feature.quaternion, tcp_quaternion)))
    orientation_error = 2.0 * float(np.arccos(np.clip(orientation_dot, -1.0, 1.0)))
    return position_error <= 0.002 and orientation_error <= 0.02


def _feature_points_world(feature: _Feature, observation: Mapping[str, Any]) -> np.ndarray:
    points_are_world = (
        "contact_surface_points_world_m" in feature.geometry
        or "pointcloud_world_m" in feature.geometry
    )
    raw_points = feature.geometry.get(
        "contact_surface_points_world_m",
        feature.geometry.get("pointcloud_world_m", feature.geometry.get("surface_points_world_m")),
    )
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        return np.empty((0, 3), dtype=float)
    points = np.asarray(raw_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 512 or not np.isfinite(points).all():
        return np.empty((0, 3), dtype=float)
    if points_are_world or feature.frame in {"world", "world_m"}:
        return points
    frame_q, frame_p = _world_frame_transform(feature.frame, observation)
    return np.asarray([_rotate(frame_q, point) + frame_p for point in points])


def _positive_geometry_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or float(value) <= 0.0:
        raise GoalResolutionError("invalid_geometry", f"{name} must be positive and finite")
    return float(value)


def _grasp_width_for_axis(reference: _Feature, opening_axis: np.ndarray | None) -> float:
    candidate_width = _positive_geometry_number(reference.geometry.get("grasp_width_m"), "grasp_width_m")
    if opening_axis is None:
        return candidate_width

    if (
        reference.kind == "depth_surface_patch"
        and reference.axis_std_rad >= np.pi / 2.0 - 1e-6
        and isinstance(reference.geometry.get("image_circle_candidate_px"), Mapping)
    ):
        raw_points = reference.geometry.get("contact_surface_points_world_m")
        if isinstance(raw_points, Sequence) and not isinstance(raw_points, (str, bytes)):
            points = np.asarray(raw_points, dtype=np.float64)
            if (
                points.ndim == 2 and points.shape[1] == 3 and len(points) >= 2
                and np.isfinite(points).all()
            ):
                observed_width = float(np.ptp(points @ _unit(opening_axis, "gripper opening axis")))
                measured_width = observed_width - 2.0 * reference.position_std_m
                if 0.0 < measured_width <= candidate_width:
                    return measured_width

    if not _uses_reliable_grasp_projection(reference):
        return candidate_width

    # For an observed round patch, project its measured 3D ellipsoid bounds along
    # the current finger-separation direction instead of using its largest axis.
    axes = _rotation_axes(reference.quaternion)
    projected_radii = 0.5 * reference.extent * (axes.T @ _unit(opening_axis, "gripper opening axis"))
    measured_width = 2.0 * float(np.linalg.norm(projected_radii)) + 2.0 * reference.position_std_m
    if not isfinite(measured_width) or measured_width <= 0.0:
        raise GoalResolutionError("invalid_geometry", "projected grasp width must be positive and finite")
    return min(candidate_width, measured_width)


def _uses_reliable_grasp_projection(feature: _Feature) -> bool:
    return (
        feature.extent is not None
        and isinstance(feature.geometry.get("image_circle_candidate_px"), Mapping)
        and feature.axis_std_rad < np.pi / 2.0 - 1e-6
    )


def _robot_sensors(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, Mapping):
        raise GoalResolutionError("invalid_observation", "robot_state must be a public mapping")
    sensors = robot_state.get("sensors", {})
    if not isinstance(sensors, Mapping):
        raise GoalResolutionError("invalid_observation", "robot_state.sensors must be a mapping")
    return {**robot_state, **sensors}


def _bilateral_grasp_for_tracking(observation: Mapping[str, Any]) -> bool:
    try:
        sensors = _robot_sensors(observation)
    except GoalResolutionError:
        return False
    if sensors.get("grasped") is not True:
        return False
    contacts = [sensors[key] for key in ("finger_contacts", "finger_bolt_contacts") if key in sensors]
    forces = sensors.get("finger_contact_force_n")
    return bool(
        contacts
        and all(
            isinstance(values, Sequence) and not isinstance(values, (str, bytes))
            and len(values) == 2 and all(type(value) is bool and value for value in values)
            for values in contacts
        )
        and isinstance(forces, Sequence) and not isinstance(forces, (str, bytes))
        and len(forces) == 2
        and all(
            not isinstance(value, bool) and isinstance(value, (int, float, np.number))
            and isfinite(float(value)) and float(value) > 0.0
            for value in forces
        )
    )


def _world_frame_transform(frame: str, observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if frame in {"world", "world_m"}:
        return np.array((1.0, 0.0, 0.0, 0.0)), np.zeros(3)
    sensors = _robot_sensors(observation)
    if frame == "tcp":
        tcp = _pose(sensors.get("tcp_pose"), "robot_state.sensors.tcp_pose")
        return np.asarray(tcp[3:], dtype=float), np.asarray(tcp[:3], dtype=float)
    transforms = sensors.get("frame_transforms", sensors.get("camera_extrinsics"))
    transform: Any = transforms.get(frame) if isinstance(transforms, Mapping) else transforms
    if transform is None:
        raise GoalResolutionError("missing_frame_transform", f"no public world transform for frame {frame!r}")
    if isinstance(transform, Mapping):
        position = _vector(transform.get("position_m", transform.get("translation_m")), 3, "camera extrinsic position")
        quaternion = _quat_normalize(_vector(
            transform.get("quaternion_wxyz"), 4, "camera extrinsic quaternion_wxyz"
        ))
        return quaternion, position
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise GoalResolutionError("invalid_frame_transform", "camera extrinsic must be a pose mapping or 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise GoalResolutionError("invalid_frame_transform", "camera extrinsic has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < 0.999:
        raise GoalResolutionError("invalid_frame_transform", "camera extrinsic rotation must be orthonormal")
    return _matrix_quaternion(rotation), matrix[:3, 3].copy()


def _camera_observation_id(observation: Mapping[str, Any], sensors: Mapping[str, Any]) -> str:
    value = sensors.get("camera_observation_id", sensors.get("rgbd_source_observation_id"))
    if value is None:
        frames = observation.get("frames", {})
        if isinstance(frames, Mapping):
            for frame in frames.values():
                if isinstance(frame, Mapping):
                    value = frame.get("observation_id", frame.get("source_observation_id"))
                    if value is not None:
                        break
    return _text(value if value is not None else observation.get("observation_id"), "camera observation_id")


def _features_by_ref(observation: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    features = observation.get("features")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        raise GoalResolutionError("invalid_observation", "features must be a sequence")
    result: dict[str, Mapping[str, Any]] = {}
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        ref = feature.get("feature_ref")
        if isinstance(ref, str):
            result[ref] = feature
    return result


def _compose_tcp_feature(tcp_pose: Sequence[float], local_pose: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    tcp_position = np.asarray(tcp_pose[:3], dtype=float)
    tcp_quaternion = _quat_normalize(np.asarray(tcp_pose[3:], dtype=float))
    local_position = np.asarray(local_pose[:3], dtype=float)
    local_quaternion = _quat_normalize(np.asarray(local_pose[3:], dtype=float))
    return (
        _rotate(tcp_quaternion, local_position) + tcp_position,
        _quat_normalize(_quat_multiply(tcp_quaternion, local_quaternion)),
    )


def _validate_final_relations(
    constraints: Sequence[tuple[str, _Feature, _Feature]],
    delta_q: np.ndarray,
    delta_t: np.ndarray,
    *,
    grasp: bool = False,
) -> None:
    for relation, moving, reference in constraints:
        position = _rotate(delta_q, moving.position) + delta_t
        axis = _rotate(delta_q, moving.axis) if moving.axis is not None else None
        if relation == "coaxial":
            if axis is None or reference.axis is None:
                raise GoalResolutionError("missing_geometry", "coaxial requires axes")
            alignment = np.linalg.norm(np.cross(axis, reference.axis))
            line_error = np.linalg.norm(np.cross(position - reference.position, reference.axis))
            if alignment > 1e-5 or line_error > 1e-5:
                raise GoalResolutionError("inconsistent_constraints", "constraints do not admit one rigid target")
        elif relation == "parallel":
            if axis is None or reference.axis is None or np.linalg.norm(np.cross(axis, reference.axis)) > 1e-5:
                raise GoalResolutionError("inconsistent_constraints", "constraints do not admit parallel axes")
        elif relation == "centered_on" and np.linalg.norm(
            position - _center_target_position(reference, grasp=grasp)
        ) > 1e-5:
            raise GoalResolutionError("inconsistent_constraints", "constraints do not admit a shared center")
        elif relation == "point_on_plane":
            if reference.normal is None or abs(float(np.dot(position - reference.position, reference.normal))) > 1e-5:
                raise GoalResolutionError("inconsistent_constraints", "constraints do not admit plane contact")
        elif relation == "remain_clear_of":
            if np.linalg.norm(_clearance_target_position(moving, reference, position) - position) > 1e-8:
                raise GoalResolutionError("clearance_violation", "the resolved geometry does not satisfy remain_clear_of")


def _clearance_target_position(
    moving: _Feature,
    reference: _Feature,
    moving_position: np.ndarray,
) -> np.ndarray:
    normal_kind = reference.geometry.get("normal_kind")
    if reference.normal is not None and (
        reference.kind == "plane_candidate"
        or normal_kind in {
            "rgbd_plane_normal",
            "surface_patch_normal_not_hole_axis",
            "local_depth_surface_normal",
            "fitted_rim_surface_normal",
        }
    ):
        if reference.extent is None:
            raise GoalResolutionError("missing_clearance_geometry", "plane clearance requires measured extent_m")
        _clearance_radius(moving)
        reference_axes = _rotation_axes(reference.quaternion)
        normal_axis_index = int(np.argmax(np.abs(reference_axes.T @ reference.normal)))
        delta = moving_position - reference.position
        local_delta = reference_axes.T @ delta
        uncertainty = moving.position_std_m + reference.position_std_m
        moving_radius = _clearance_radius(moving)
        uncertainty += moving_radius * np.sin(min(moving.axis_std_rad, np.pi / 2.0))
        lateral_delta = delta - reference.normal * float(np.dot(delta, reference.normal))
        uncertainty += float(np.linalg.norm(lateral_delta)) * np.sin(
            min(reference.axis_std_rad, np.pi / 2.0)
        )

        # A plane's in-plane footprint is not a normal clearance distance.
        for index, axis in enumerate(reference_axes.T):
            if index == normal_axis_index:
                continue
            moving_support = _support_extent(moving, axis)
            if abs(float(local_delta[index])) > reference.extent[index] * 0.5 + moving_support + uncertainty:
                return moving_position

        signed_distance = float(np.dot(delta, reference.normal))
        if abs(signed_distance) <= moving.position_std_m + reference.position_std_m:
            raise GoalResolutionError(
                "missing_clearance_direction", "current geometry does not determine which side of the plane to clear"
            )
        plane_support = _support_extent(reference, reference.normal)
        moving_support = _support_extent(moving, reference.normal)
        required_distance = plane_support + moving_support + uncertainty
        target_distance = np.copysign(required_distance, signed_distance)
        if abs(signed_distance) >= required_distance:
            return moving_position
        return moving_position + reference.normal * float(target_distance - signed_distance)

    moving_radius = _clearance_radius(moving)
    reference_radius = _clearance_radius(reference)
    required_distance = moving_radius + reference_radius + moving.position_std_m + reference.position_std_m
    separation = moving_position - reference.position
    distance = float(np.linalg.norm(separation))
    if distance >= required_distance:
        return moving_position
    if distance <= 1e-9:
        raise GoalResolutionError(
            "missing_clearance_direction", "coincident bounded features do not determine a clearance direction"
        )
    return reference.position + separation * (required_distance / distance)


def _center_target_position(reference: _Feature, *, grasp: bool) -> np.ndarray:
    target = reference.position.copy()
    offset = reference.geometry.get("shape_bounds_center_offset_world_m")
    if grasp and reference.geometry.get("grasp_width_m") is not None and offset is not None:
        target += _vector(offset, 3, "feature.geometry.shape_bounds_center_offset_world_m")
    return target


def _clearance_radius(feature: _Feature) -> float:
    radius = feature.geometry.get("bounding_radius_m")
    if radius is None:
        raise GoalResolutionError("missing_clearance_geometry", "remain_clear_of requires measured bounding_radius_m values")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not isfinite(float(radius)) or radius < 0.0:
        raise GoalResolutionError("invalid_clearance_geometry", "bounding radii must be finite and non-negative")
    return float(radius)


def _rotation_axes(quaternion: np.ndarray) -> np.ndarray:
    return np.column_stack([_rotate(quaternion, np.eye(3)[:, index]) for index in range(3)])


def _support_extent(feature: _Feature, direction: np.ndarray) -> float:
    if feature.extent is None:
        return _clearance_radius(feature)
    axes = _rotation_axes(feature.quaternion)
    return float(np.dot(np.abs(axes.T @ direction), feature.extent * 0.5))


def _pose(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        position = value.get("position_m", value.get("position"))
        quaternion = value.get("quaternion_wxyz", value.get("quaternion"))
        return (*_vector(position, 3, f"{name}.position"), *_quat_normalize(_vector(quaternion, 4, f"{name}.quaternion")))
    result = _vector(value, 7, name)
    return (*result[:3], *_quat_normalize(np.asarray(result[3:], dtype=float)))


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise GoalResolutionError("missing_geometry", f"{name} must contain {size} numbers")
    values = np.asarray(value, dtype=float)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise GoalResolutionError("invalid_geometry", f"{name} must be finite")
    return values


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise GoalResolutionError("invalid_geometry", f"{name} cannot be zero")
    return vector / norm


def _quat_normalize(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not isfinite(norm) or norm <= 1e-9:
        raise GoalResolutionError("invalid_geometry", "quaternion cannot be zero or non-finite")
    result = quaternion / norm
    return -result if result[0] < 0.0 else result


def _quat_inverse(quaternion: np.ndarray) -> np.ndarray:
    q = _quat_normalize(quaternion)
    return np.array((q[0], -q[1], -q[2], -q[3]))


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w, x, y, z = left
    a, b, c, d = right
    return np.array((w*a - x*b - y*c - z*d, w*b + x*a + y*d - z*c,
                     w*c - x*d + y*a + z*b, w*d + x*c - y*b + z*a))


def _rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = _quat_normalize(quaternion)
    qv = q[1:]
    return vector + 2.0 * np.cross(qv, np.cross(qv, vector) + q[0] * vector)


def _quat_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a, b = _unit(source, "source axis"), _unit(target, "target axis")
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < -0.999999:
        basis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = _unit(np.cross(a, basis), "anti-parallel axis")
        return np.array((0.0, *axis))
    cross = np.cross(a, b)
    return _quat_normalize(np.array((1.0 + dot, *cross)))


def _matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        q = np.array((0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
                      (rotation[0, 2] - rotation[2, 0]) / scale,
                      (rotation[1, 0] - rotation[0, 1]) / scale))
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)) * 2.0
            q = np.array(((rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                          (rotation[0, 1] + rotation[1, 0]) / scale,
                          (rotation[0, 2] + rotation[2, 0]) / scale))
        elif index == 1:
            scale = sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)) * 2.0
            q = np.array(((rotation[0, 2] - rotation[2, 0]) / scale,
                          (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                          (rotation[1, 2] + rotation[2, 1]) / scale))
        else:
            scale = sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)) * 2.0
            q = np.array(((rotation[1, 0] - rotation[0, 1]) / scale,
                          (rotation[0, 2] + rotation[2, 0]) / scale,
                          (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale))
    return _quat_normalize(q)


def _text(value: Any, name: str, *, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise GoalResolutionError("invalid_input", f"{name} must be non-empty text of at most {maximum} chars")
    return value.strip()


__all__ = ["AssemblyConnector", "GoalResolutionError", "ResolvedGoal", "SUPPORTED_RELATIONS"]
