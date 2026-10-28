"""Receding-horizon geometric goal execution over a reusable robot driver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite
import logging
from typing import Any, Callable, Protocol

import numpy as np

from .connector import (
    AssemblyConnector,
    GoalResolutionError,
    ResolvedGoal,
    _camera_observation_id,
    bind_contact_surface_points as _bind_contact_surface_points,
)

_LOGGER = logging.getLogger(__name__)
_PRIVATE_PROGRESS_INTERVAL_STEPS = 50
_MAX_TCP_MOTION_DURATION_S = 30.0
_MAX_JOINT_STEP_BACKOFFS = 4
_JOINT_STEP_BACKOFF_FACTOR = 0.5
_PUBLIC_SWEEP_COLLISION_DIAGNOSTIC_FIELDS = frozenset({
    "collision_kind", "body_child", "body_parent", "sample_fraction",
    "obstacle_point_index", "obstacle_point_world_m", "closest_centerline_point_world_m",
    "distance_m", "capsule_radius_m", "sweep_margin_m", "required_radius_m",
    "signed_clearance_m", "map_source_observation_id", "camera_frame_id",
    "camera_timestamp_s", "robot_state_timestamp_s", "obstacle_pixel_xy",
})


class AssemblyDriver(Protocol):
    """Low-level Factory-compatible driver contract; all scene input is public RGB-D."""

    joint_limits: Sequence[Sequence[float]]
    max_gripper_width_m: float

    def solve_ik(self, tcp_pose: Sequence[float], seed_joint_positions: Sequence[float]) -> Sequence[float] | None: ...

    def check_robot_sweep_collision(
        self,
        start_tcp_pose: Sequence[float],
        end_tcp_pose: Sequence[float],
        start_joint_positions: Sequence[float],
        end_joint_positions: Sequence[float],
        obstacle_points_world_m: Sequence[Sequence[float]],
        moving_geometry: Mapping[str, Any],
    ) -> bool: ...

    def step(self, normalized_tcp_delta: Sequence[float]) -> Any: ...

    def hold(self) -> None: ...


@dataclass(frozen=True)
class ExecutorLimits:
    control_dt_s: float
    position_action_scale_m: tuple[float, float, float]
    rotation_action_scale_rad: tuple[float, float, float]
    max_contact_force_n: float
    max_gripper_force_n: float
    max_goal_uncertainty_m: float
    max_goal_axis_uncertainty_rad: float
    max_rgbd_age_s: float = 0.2
    max_feature_match_distance_m: float = 0.03
    max_steps: int = 200
    max_linear_speed_mps: float = 0.03
    max_contact_speed_mps: float = 0.004
    max_angular_speed_radps: float = 0.35
    max_joint_step_rad: float = 0.05
    position_tolerance_m: float = 0.001
    orientation_tolerance_rad: float = 0.02
    settled_linear_speed_mps: float = 0.01
    settled_angular_speed_radps: float = 0.1
    minimum_stable_duration_s: float = 0.1
    gripper_width_tolerance_m: float = 0.002
    gripper_stability_tolerance_m: float = 0.001

    def __post_init__(self) -> None:
        scalars = (
            self.control_dt_s, self.max_contact_force_n, self.max_gripper_force_n,
            self.max_goal_uncertainty_m, self.max_goal_axis_uncertainty_rad, self.max_linear_speed_mps,
            self.max_rgbd_age_s, self.max_feature_match_distance_m,
            self.max_contact_speed_mps, self.max_angular_speed_radps,
            self.max_joint_step_rad, self.position_tolerance_m,
            self.orientation_tolerance_rad, self.settled_linear_speed_mps,
            self.settled_angular_speed_radps, self.minimum_stable_duration_s,
            self.gripper_width_tolerance_m, self.gripper_stability_tolerance_m,
        )
        if any(isinstance(value, bool) or not isfinite(float(value)) or float(value) <= 0.0 for value in scalars):
            raise ValueError("executor limits must be positive finite values")
        if self.max_contact_force_n > 3.0:
            raise ValueError("max_contact_force_n cannot exceed the existing 3.0 N limit")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or not 1 <= self.max_steps <= 2000:
            raise ValueError("max_steps must be in [1, 2000]")
        if len(self.position_action_scale_m) != 3 or len(self.rotation_action_scale_rad) != 3:
            raise ValueError("action scales must contain three axes")
        if any(not isfinite(float(value)) or float(value) <= 0.0
               for value in (*self.position_action_scale_m, *self.rotation_action_scale_rad)):
            raise ValueError("action scales must be positive and finite")


class AssemblyExecutor:
    """Execute one model-submitted relation goal with feedback and safety checks.

    The driver supplies robot-only IK and local swept-volume checks for the actual
    commanded Cartesian segment. The executor supplies the current RGB-D obstacle
    cloud, held geometry, and narrowly scoped current-goal contact pairs to every check;
    it never requests target state from an environment or simulator stage.
    """

    def __init__(
        self,
        driver: AssemblyDriver,
        *,
        limits: ExecutorLimits,
        observation_provider: Callable[[], Mapping[str, Any]] | None = None,
        connector: AssemblyConnector | None = None,
        private_progress_recorder: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        required = ("solve_ik", "check_robot_sweep_collision", "step", "hold")
        missing = [name for name in required if not callable(getattr(driver, name, None))]
        if missing:
            raise TypeError(f"robot driver is missing required safety/control methods: {', '.join(missing)}")
        if not callable(observation_provider) and not callable(getattr(driver, "observe_public", None)):
            raise TypeError("driver.observe_public() or observation_provider is required for closed-loop execution")
        if not isinstance(limits, ExecutorLimits):
            raise TypeError("limits must be ExecutorLimits populated from robot capabilities/calibration")
        joint_limits = _joint_limits(driver)
        self.driver = driver
        self.limits = limits
        self.observation_provider = observation_provider
        self.connector = connector or AssemblyConnector()
        self.private_progress_recorder = private_progress_recorder
        self.joint_limits = joint_limits

    def execute_goal(self, goal: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
        diagnostics = {
            "driver_step_count": 0,
            "ik_call_count": 0,
            "sweep_check_count": 0,
        }
        result = self._execute_goal(goal, observation, diagnostics)
        result["execution_diagnostics"] = diagnostics
        return result

    def execute_tcp_delta(self, call: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
        diagnostics = {"driver_step_count": 0, "ik_call_count": 0, "sweep_check_count": 0}
        step_budget_details = ""

        def finish(result: dict[str, Any], target: Sequence[float] | None = None) -> dict[str, Any]:
            result["execution_diagnostics"] = diagnostics
            if target is not None:
                result.pop("resolved_goal", None)
                result["target_tcp_pose"] = {
                    "position_m": [float(value) for value in target[:3]],
                    "quaternion_wxyz": [float(value) for value in target[3:]],
                }
                result["feedback"] = (
                    f"{result['feedback']} {step_budget_details} "
                    f"Control steps={diagnostics['driver_step_count']}; "
                    f"target TCP position_m={result['target_tcp_pose']['position_m']} and "
                    f"quaternion_wxyz={result['target_tcp_pose']['quaternion_wxyz']}. "
                    "The final measured TCP pose is in the accompanying fresh observation."
                )[:1024]
            return result

        expected = {
            "tool", "observation_id", "frame", "translation_delta_m", "rotation_vector_rad",
            "max_duration_s", "motion_budget",
        }
        if not isinstance(call, Mapping) or set(call) != expected or call.get("tool") != "move_tcp":
            return finish(_result("invalid_goal", False, "move_tcp requires exactly its typed public fields"))
        if not isinstance(observation, Mapping) or call.get("observation_id") != observation.get("observation_id"):
            return finish(_result("stale_goal", False, "move_tcp must cite the supplied current observation"))
        if call["frame"] != "tcp":
            return finish(_result("invalid_goal", False, "move_tcp frame must be 'tcp'"))
        duration = call["max_duration_s"]
        if (
            isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not isfinite(float(duration)) or not 0.0 < float(duration) <= _MAX_TCP_MOTION_DURATION_S
        ):
            return finish(_result(
                "invalid_goal", False,
                f"max_duration_s must be finite and in (0, {_MAX_TCP_MOTION_DURATION_S:g}] seconds",
            ))
        duration = float(duration)
        budget = call["motion_budget"]
        if type(budget) is not int or not 1 <= budget <= 2000:
            return finish(_result(
                "motion_budget_exceeded", False,
                "motion_budget must be an integer in [1, 2000] control steps",
            ))
        try:
            translation_local = _strict_vector(call["translation_delta_m"], 3, "translation_delta_m")
            if not isfinite(float(np.linalg.norm(translation_local))):
                raise ValueError("translation_delta_m magnitude must be finite")
            rotation_local = _strict_vector(call["rotation_vector_rad"], 3, "rotation_vector_rad")
            angle = float(np.linalg.norm(rotation_local))
            if angle > np.pi + 1e-9:
                raise ValueError("rotation_vector_rad norm must be at most pi")
            sensors = _sensors(observation)
            if sensors.get("grasped") is not False:
                self._hold()
                return finish(_result(
                    "tcp_motion_requires_unheld", False,
                    "move_tcp requires public grasped=false so the swept check has no unmodeled held object",
                ))
            current_pose = _pose(sensors.get("tcp_pose"), "robot_state.sensors.tcp_pose")
            current_quaternion = _quat(current_pose[3:])
            if angle <= 1e-12:
                local_delta_quaternion = np.array((1.0, 0.0, 0.0, 0.0))
            else:
                local_delta_quaternion = np.concatenate((
                    np.asarray([np.cos(0.5 * angle)]),
                    rotation_local * (np.sin(0.5 * angle) / angle),
                ))
            target_position = (
                np.asarray(current_pose[:3], dtype=np.float64)
                + _quat_rotation_matrix(current_quaternion) @ translation_local
            )
            target_quaternion = _quat(_quat_multiply(current_quaternion, local_delta_quaternion))
            target_pose = (*target_position.tolist(), *target_quaternion.tolist())
            resolved = ResolvedGoal(
                observation_id=str(observation["observation_id"]),
                submitted_observation_id=str(observation["observation_id"]),
                feature_refs=(),
                motion_regime="free_space",
                intent="Move the TCP by one bounded local delta.",
                stop_condition="Reach and settle at the TCP pose computed from the submitted observation.",
                target_tcp_pose=target_pose,
                tcp_to_moving_feature_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                uncertainty_position_m=0.0,
                uncertainty_axis_rad=0.0,
                grasp_width_m=None,
                moving_geometry={},
                moving_frame="tcp",
                constraints=(),
                allowed_contact_pairs=(),
            )
            already_settled = (
                float(np.linalg.norm(np.asarray(target_pose[:3]) - np.asarray(current_pose[:3])))
                <= self.limits.position_tolerance_m
                and float(np.linalg.norm(_rotation_vector(target_pose[3:], current_pose[3:])))
                <= self.limits.orientation_tolerance_rad
                and _settled(sensors, self.limits)
            )
            calculated_step_budget = _motion_step_budget(
                current_pose, target_pose, "free_space", self.limits,
            )
            duration_step_budget = floor(duration / self.limits.control_dt_s + 1e-9)
            effective_step_budget = (
                1 if already_settled else min(budget, duration_step_budget, calculated_step_budget)
            )
            if effective_step_budget < 1:
                step_budget_details = (
                    f"Requested duration={duration:g}s allows no complete control period "
                    f"(control_dt_s={self.limits.control_dt_s:g}); no motion command was sent."
                )
                return finish(_result(
                    "motion_budget_exceeded", False,
                    "max_duration_s is shorter than one control period; no motion command was sent",
                    resolved,
                ), target_pose)
            step_budget_details = (
                f"Requested max_duration_s={duration:g}s and motion_budget={budget}; "
                f"effective control-step cap={effective_step_budget} "
                f"(min of model, duration={duration_step_budget}, and calculated/executor="
                f"{calculated_step_budget})."
            )
        except (TypeError, ValueError, ArithmeticError) as exc:
            self._hold()
            return finish(_result("invalid_public_state", True, str(exc)))

        result = self._execute_goal(
            {"tool": "move_tcp", "observation_id": observation["observation_id"]},
            observation,
            diagnostics,
            resolved_override=resolved,
            motion_budget_cap=effective_step_budget,
            force_guard=True,
        )
        return finish(result, target_pose)

    def _execute_goal(
        self,
        goal: Mapping[str, Any],
        observation: Mapping[str, Any],
        diagnostics: dict[str, int],
        *,
        resolved_override: ResolvedGoal | None = None,
        motion_budget_cap: int | None = None,
        force_guard: bool = False,
    ) -> dict[str, Any]:
        """Resolve and execute one current goal; intent text never selects control behavior."""
        if not isinstance(goal, Mapping) or not isinstance(observation, Mapping):
            return _result("invalid_goal", False, "goal and observation must be mappings")
        original_observation_id = observation.get("observation_id")
        if goal.get("observation_id") != original_observation_id:
            return _result("stale_goal", False, "goal must cite the supplied current observation")

        try:
            if resolved_override is None:
                active_goal = dict(goal)
                active_goal["_submitted_observation_id"] = original_observation_id
                active_goal["_submitted_constraints"] = [dict(item) for item in goal.get("constraints", ())]
                active_goal["_submitted_feature_refs"] = list(dict.fromkeys(
                    ref for item in goal.get("constraints", ())
                    for ref in (item.get("moving_feature"), item.get("reference_feature"))
                ))
                resolved = self.connector.resolve(active_goal, observation)
            else:
                if resolved_override.observation_id != original_observation_id:
                    return _result("stale_goal", False, "resolved TCP target is not bound to the current observation")
                active_goal = None
                resolved = resolved_override
            self._check_uncertainty(resolved)
            current = observation
            sensors = _sensors(current)
            if resolved.motion_regime == "release":
                if sensors.get("grasped") is not True:
                    return _result("release_state_conflict", False, "release requires a currently verified grasp", resolved)
                if sensors.get("release_authorized") is not True:
                    return _result("release_not_authorized", False, "release requires the public evaluator authorization boolean", resolved)
            if resolved.motion_regime == "grasp":
                grasped = sensors.get("grasped")
                if grasped is not False:
                    feedback = (
                        "refusing to open a gripper around an already-held object"
                        if grasped is True else
                        "grasp requires an explicitly verified unheld state; refusing to open the gripper"
                    )
                    return _result(
                        "grasp_state_conflict", False, feedback, resolved,
                    )
                if _force_limit_reached(sensors, self.limits.max_contact_force_n):
                    self._hold()
                    return _result("force_limit", False, "public force limit reached; robot held", resolved)
                if not resolved.allowed_contact_pairs:
                    self._hold()
                    return _result(
                        "missing_contact_geometry", False,
                        "Current public RGB-D points do not support both predicted finger contact sites on the selected feature; no motion command was issued.",
                        resolved,
                    )
                _bind_contact_surface_points(
                    resolved.allowed_contact_pairs,
                    _obstacle_points(current, max_age_s=self.limits.max_rgbd_age_s),
                )
                current = self._command_gripper(
                    float(getattr(self.driver, "max_gripper_width_m")), current,
                    closing=False, diagnostics=diagnostics,
                    goal=active_goal or goal, resolved=resolved,
                )
                active_goal = self.connector.rebind_goal_features(
                    active_goal, observation, current,
                    max_match_distance_m=self.limits.max_feature_match_distance_m,
                )
            elif resolved.motion_regime == "release":
                if not resolved.allowed_contact_pairs:
                    self._hold()
                    return _result("missing_contact_geometry", False, "current release goal does not bind a verifiable contact surface", resolved)
                _bind_contact_surface_points(
                    resolved.allowed_contact_pairs,
                    _obstacle_points(current, max_age_s=self.limits.max_rgbd_age_s),
                )

            contact_stable_samples = 0
            contact_stable_elapsed_s = 0.0
            if active_goal is not None:
                resolved = self.connector.resolve(active_goal, current)
                self._check_uncertainty(resolved)
            initial_tcp_pose = _pose(
                _sensors(current).get("tcp_pose"), "robot_state.sensors.tcp_pose",
            )
            motion_budget_steps = _motion_step_budget(
                initial_tcp_pose, resolved.target_tcp_pose, resolved.motion_regime, self.limits,
            )
            if motion_budget_cap is not None:
                motion_budget_steps = min(motion_budget_steps, motion_budget_cap)
            for _ in range(motion_budget_steps):
                if active_goal is not None:
                    resolved = self.connector.resolve(active_goal, current)
                    self._check_uncertainty(resolved)
                sensors = _sensors(current)
                if force_guard or resolved.motion_regime == "contact":
                    if _force_limit_reached(sensors, self.limits.max_contact_force_n):
                        self._hold()
                        return _result("force_limit", False, "public force limit reached; robot held", resolved)
                if (resolved.motion_regime == "contact" and not resolved.allowed_contact_pairs
                        and not resolved.aperture_entry_supported):
                    self._hold()
                    virtual_entry = any(item["reference_semantics"] == "virtual_aperture_plane"
                                        for item in resolved.geometry_diagnostics)
                    if virtual_entry:
                        return _result(
                            "missing_contact_geometry", False,
                            "Aperture entry prerequisites are unverified: measured rim geometry, a reliable "
                            "held-part axis, and coaxial plus point_on_plane to the same aperture are required. "
                            "No motion command was issued. A solid center sample is not required; "
                            "see geometry_diagnostics and obtain missing evidence or stop.", resolved,
                        )
                    return _result(
                        "missing_contact_geometry", False,
                        "Current RGB-D has no solid surface samples at the goal-bound contact point for this feature pair; "
                        "no motion command was issued. Choose a different supported relation or target instead of "
                        "repeating this contact. Support is unknown, not free space or a physical jam. "
                        "Aperture localization does not require a solid center sample; see geometry_diagnostics.", resolved,
                    )
                if (
                    resolved.moving_frame in {"world", "world_m"}
                    and resolved.motion_regime in {"free_space", "contact", "release"}
                    and sensors.get("grasped") is not True
                ):
                    self._hold()
                    return _result(
                        "object_not_held", False,
                        _object_not_held_feedback(
                            current,
                            sensors.get("grasped"),
                            driver_steps=int(diagnostics.get("driver_step_count", 0)),
                        ),
                        resolved,
                    )

                obstacle_points = _obstacle_points(current, max_age_s=self.limits.max_rgbd_age_s)
                if (
                    resolved.moving_frame in {"world", "world_m"}
                    and sensors.get("grasped") is True
                ):
                    # resolved.constraints preserves submitted refs; active_goal carries current rebound refs.
                    bound_constraints = (
                        active_goal.get("constraints", resolved.constraints)
                        if active_goal is not None else resolved.constraints
                    )
                    obstacle_points = _without_held_surface_samples(
                        obstacle_points, resolved.moving_geometry, current,
                        moving_feature_ref=bound_constraints[0]["moving_feature"],
                        protected_feature_refs=tuple(
                            constraint["reference_feature"] for constraint in bound_constraints
                        ),
                        sweep_margin_m=float(getattr(self.driver, "sweep_margin_m", 0.005)),
                    )
                sweep_contact_pairs: Sequence[Mapping[str, Any]] = ()
                if resolved.allowed_contact_pairs:
                    sweep_contact_pairs = _bind_contact_surface_points(
                        resolved.allowed_contact_pairs, obstacle_points,
                    )

                tcp_pose = _pose(sensors.get("tcp_pose"), "robot_state.sensors.tcp_pose")
                joint_positions = _vector(sensors.get("joint_positions"), len(self.joint_limits), "joint_positions")
                position_error = np.asarray(resolved.target_tcp_pose[:3]) - np.asarray(tcp_pose[:3])
                rotation_error = _rotation_vector(resolved.target_tcp_pose[3:], tcp_pose[3:])
                if (
                    float(np.linalg.norm(position_error)) <= self.limits.position_tolerance_m
                    and float(np.linalg.norm(rotation_error)) <= self.limits.orientation_tolerance_rad
                    and _settled(sensors, self.limits)
                ):
                    if resolved.motion_regime == "contact":
                        if contact_stable_samples:
                            contact_stable_elapsed_s += self.limits.control_dt_s
                        contact_stable_samples += 1
                        if contact_stable_elapsed_s + 1e-9 < self.limits.minimum_stable_duration_s:
                            previous = current
                            current = self._next_observation(
                                self._step((0.0,) * 6, diagnostics),
                                previous_id=previous.get("observation_id"),
                            )
                            if active_goal is not None:
                                active_goal = self.connector.rebind_goal_features(
                                    active_goal, previous, current,
                                    max_match_distance_m=self.limits.max_feature_match_distance_m,
                                )
                            self._record_private_progress(
                                current, active_goal, resolved, diagnostics,
                            )
                            continue
                    else:
                        contact_stable_samples = 0
                        contact_stable_elapsed_s = 0.0
                    if resolved.motion_regime == "grasp":
                        if resolved.grasp_width_m is None:
                            raise GoalResolutionError("missing_geometry", "grasp width was not resolved from geometry")
                        width_limit = float(getattr(
                            self.driver,
                            "measured_gripper_width_max_m",
                            getattr(self.driver, "max_gripper_width_m"),
                        ))
                        if resolved.grasp_width_m > width_limit:
                            raise GoalResolutionError("grasp_width_out_of_range", "geometry-derived grasp width exceeds gripper capability")
                        current = self._command_gripper(
                            resolved.grasp_width_m, current, closing=True,
                            diagnostics=diagnostics, goal=active_goal or goal, resolved=resolved,
                        )
                        return _result(
                            "grasp_contact_verified", False,
                            "bilateral finger contact and stable measured width verified; lift and retention are not verified",
                            resolved,
                        )
                    elif resolved.motion_regime == "release":
                        if _sensors(current).get("release_authorized") is not True:
                            return _result("release_not_authorized", False, "release authorization is no longer current", resolved)
                        width_limit = float(getattr(self.driver, "max_gripper_width_m"))
                        current = self._command_gripper(
                            width_limit, current, closing=False,
                            diagnostics=diagnostics, goal=active_goal or goal, resolved=resolved,
                        )
                        if _sensors(current).get("grasped") is not False:
                            self._hold()
                            return _result("release_unverified", False, "fresh public observation did not verify release", resolved)
                    return _result("completed", False, "geometric goal reached and settled", resolved)
                contact_stable_samples = 0
                contact_stable_elapsed_s = 0.0

                speed = (
                    self.limits.max_contact_speed_mps
                    if resolved.motion_regime == "contact"
                    else self.limits.max_linear_speed_mps
                )
                commanded_tcp_pose = _pose(
                    sensors.get("commanded_tcp_pose"),
                    "robot_state.sensors.commanded_tcp_pose",
                )
                action = _normalized_action(
                    commanded_tcp_pose,
                    resolved.target_tcp_pose,
                    dt_s=self.limits.control_dt_s,
                    max_linear_step_m=speed * self.limits.control_dt_s,
                    max_angular_step_rad=self.limits.max_angular_speed_radps * self.limits.control_dt_s,
                    position_scales=self.limits.position_action_scale_m,
                    rotation_scales=self.limits.rotation_action_scale_rad,
                )
                end_tcp_pose = None
                target_joints = None
                action_scale = 1.0
                for backoff_index in range(_MAX_JOINT_STEP_BACKOFFS + 1):
                    scaled_action = tuple(float(value) * action_scale for value in action)
                    end_tcp_pose = _commanded_tcp_pose(commanded_tcp_pose, scaled_action, self.limits)
                    diagnostics["ik_call_count"] += 1
                    target_joints = self.driver.solve_ik(end_tcp_pose, joint_positions.tolist())
                    if target_joints is None:
                        return _result("ik_unreachable", False, "current robot state has no IK solution for this goal", resolved)
                    target_joints = _vector(target_joints, len(self.joint_limits), "IK solution")
                    self._check_joint_limits(target_joints)
                    joint_step = float(np.max(np.abs(target_joints - joint_positions)))
                    if joint_step <= self.limits.max_joint_step_rad:
                        break
                    if backoff_index < _MAX_JOINT_STEP_BACKOFFS:
                        action_scale *= _JOINT_STEP_BACKOFF_FACTOR
                        continue
                    self._hold()
                    return _result(
                        "joint_step_limit", False,
                        "local Cartesian command remained above the joint-step limit after bounded adaptive "
                        "backoff; no arm step was sent and robot held",
                        resolved,
                    )
                if end_tcp_pose is None or target_joints is None:
                    self._hold()
                    return _result(
                        "joint_step_limit", False,
                        "adaptive Cartesian backoff produced no executable IK target; no arm step was sent and robot held",
                        resolved,
                    )
                diagnostics["sweep_check_count"] += 1
                if self.driver.check_robot_sweep_collision(
                    tcp_pose, end_tcp_pose, joint_positions.tolist(), target_joints.tolist(),
                    obstacle_points, {
                        **resolved.moving_geometry,
                        "_held_object": (
                            resolved.moving_frame in {"world", "world_m"}
                            and sensors.get("grasped") is True
                        ),
                        "_tcp_to_feature_pose": list(resolved.tcp_to_moving_feature_pose),
                        "_allowed_contact_pairs": [dict(pair) for pair in sweep_contact_pairs],
                    },
                ) is not True:
                    collision_diagnostic = _public_sweep_collision_diagnostic(
                        getattr(self.driver, "last_sweep_collision_diagnostic", None),
                        sensors,
                    )
                    retreat_direction_world = None
                    if resolved.motion_regime == "grasp" and sensors.get("grasped") is False:
                        retreat_direction_world = _grasp_approach_retreat_direction_world(
                            resolved.target_tcp_pose, tcp_pose,
                        )
                    collision_feedback = _observed_rgbd_collision_feedback(
                        getattr(self.driver, "last_sweep_collision_diagnostic", None), sensors,
                        retreat_direction_world=retreat_direction_world,
                    )
                    self._hold()
                    result = _result(
                        "collision_blocked", False,
                        collision_feedback or "RGB-D sweep check rejected the exact local servo step",
                        resolved,
                    )
                    if isinstance(collision_diagnostic, Mapping):
                        result["sweep_collision_diagnostic"] = collision_diagnostic
                    return result

                previous = current
                step_result = self._step(action, diagnostics)
                current = self._next_observation(step_result, previous_id=previous.get("observation_id"))
                if active_goal is not None:
                    current = self.connector.carry_held_surface_track(
                        active_goal, previous, current,
                        moving_tcp_feature_pose=resolved.tcp_to_moving_feature_pose,
                        max_match_distance_m=self.limits.max_feature_match_distance_m,
                    )
                    active_goal = self.connector.rebind_goal_features(
                        active_goal, previous, current,
                        moving_tcp_feature_pose=resolved.tcp_to_moving_feature_pose,
                        max_match_distance_m=self.limits.max_feature_match_distance_m,
                    )
                self._record_private_progress(
                    current, active_goal or goal, resolved, diagnostics,
                )

            self._hold()
            return _result("motion_timeout", False, "bounded online execution step budget exhausted; robot held", resolved)
        except GoalResolutionError as exc:
            self._record_private_progress(
                locals().get("current", observation),
                locals().get("active_goal") or goal,
                locals().get("resolved"),
                diagnostics,
                phase="executor_error",
                error_code=exc.code,
                force=True,
            )
            if exc.code in {
                "clearance_violation", "missing_clearance_geometry", "invalid_clearance_geometry",
                "missing_rgbd_map", "stale_rgbd_map", "invalid_rgbd_map", "stale_observation",
                "feature_relocalization_required", "ambiguous_feature_tracking",
                "force_limit", "gripper_timeout", "gripper_state_unavailable",
                "missing_contact_geometry", "contact_surface_missing",
            }:
                self._hold()
                return _result(exc.code, False, str(exc), locals().get("resolved"))
            return _result(exc.code, False, str(exc), locals().get("resolved"))
        except (TypeError, ValueError, ArithmeticError) as exc:
            self._record_private_progress(
                locals().get("current", observation),
                locals().get("active_goal") or goal,
                locals().get("resolved"),
                diagnostics,
                phase="executor_error",
                error_code="invalid_public_state",
                force=True,
            )
            self._hold()
            return _result("invalid_public_state", True, str(exc), locals().get("resolved"))
        except Exception:
            _LOGGER.exception("robot driver failed during goal execution")
            self._record_private_progress(
                locals().get("current", observation),
                locals().get("active_goal") or goal,
                locals().get("resolved"),
                diagnostics,
                phase="executor_error",
                error_code="driver_error",
                force=True,
            )
            self._hold()
            return _result("driver_error", True, "robot driver failed; robot held", locals().get("resolved"))

    def _record_private_progress(
        self,
        observation: Mapping[str, Any],
        goal: Mapping[str, Any],
        resolved: ResolvedGoal | None,
        diagnostics: Mapping[str, int],
        *,
        phase: str = "control",
        error_code: str | None = None,
        force: bool = False,
        measured_gripper_width_m: float | None = None,
    ) -> None:
        recorder = self.private_progress_recorder
        step_count = int(diagnostics.get("driver_step_count", 0))
        if not callable(recorder) or (
            not force
            and (step_count == 0 or step_count % _PRIVATE_PROGRESS_INTERVAL_STEPS != 0)
        ):
            return
        try:
            sensors = _sensors(observation)
            constraints = goal.get("constraints", ())
            active_refs = list(dict.fromkeys(
                ref
                for item in constraints
                if isinstance(item, Mapping)
                for ref in (item.get("moving_feature"), item.get("reference_feature"))
                if isinstance(ref, str)
            ))
            record = {
                "phase": phase,
                "error_code": error_code,
                "driver_step_count": step_count,
                "ik_call_count": int(diagnostics.get("ik_call_count", 0)),
                "sweep_check_count": int(diagnostics.get("sweep_check_count", 0)),
                "observation_id": observation.get("observation_id"),
                "camera_frame_id": _optional_finite_number(sensors.get("camera_frame_id")),
                "camera_timestamp_s": _optional_finite_number(sensors.get("rgbd_source_timestamp_s")),
                "robot_timestamp_s": _optional_finite_number(
                    sensors.get("robot_timestamp_s", sensors.get("robot_state_timestamp_s")),
                ),
                "active_feature_refs": active_refs,
                "tcp_pose": _progress_numbers(sensors.get("tcp_pose")),
                "target_tcp_pose": (
                    _progress_numbers(resolved.target_tcp_pose) if resolved is not None else None
                ),
                "joint_positions": _progress_numbers(sensors.get("joint_positions")),
                "grasp_width_m": resolved.grasp_width_m if resolved is not None else None,
                "grasped": sensors.get("grasped") if type(sensors.get("grasped")) is bool else None,
                "finger_contacts": _progress_booleans(sensors.get("finger_contacts")),
            }
            if measured_gripper_width_m is not None:
                record["measured_gripper_width_m"] = _optional_finite_number(measured_gripper_width_m)
            recorder(record)
        except Exception:
            _LOGGER.exception("failed to write private executor progress record")

    def _command_gripper(
        self,
        width_m: float,
        observation: Mapping[str, Any],
        *,
        closing: bool,
        diagnostics: dict[str, int],
        goal: Mapping[str, Any],
        resolved: ResolvedGoal | None,
    ) -> Mapping[str, Any]:
        setter = getattr(self.driver, "set_gripper_target_width", None)
        if not callable(setter):
            raise GoalResolutionError("driver_capability_missing", "robot driver has no gripper target interface")
        maximum = float(getattr(self.driver, "max_gripper_width_m"))
        if not isfinite(maximum) or maximum <= 0.0 or not 0.0 <= width_m <= maximum:
            raise GoalResolutionError("gripper_width_out_of_range", "gripper width is outside calibrated capability")
        measured_maximum = getattr(self.driver, "measured_gripper_width_max_m", maximum)
        if (
            isinstance(measured_maximum, bool)
            or not isinstance(measured_maximum, (int, float))
            or not isfinite(float(measured_maximum))
            or float(measured_maximum) <= 0.0
        ):
            raise GoalResolutionError("gripper_state_unavailable", "measured gripper width calibration is invalid")
        measured_maximum = float(measured_maximum)
        target_width = min(float(width_m), measured_maximum)
        setter(width_m, self.limits.max_gripper_force_n)
        current = observation
        stable_samples = 0
        stable_elapsed_s = 0.0
        previous_width: float | None = None
        for step_index in range(self.limits.max_steps + 1):
            sensors = _sensors(current)
            if _force_limit_reached(sensors, self.limits.max_contact_force_n):
                raise GoalResolutionError("force_limit", "public force limit reached during gripper motion")
            _obstacle_points(current, max_age_s=self.limits.max_rgbd_age_s)
            measured_width, contacts = _gripper_measurements(
                sensors, max_width_m=measured_maximum,
            )
            self._record_private_progress(
                current,
                goal,
                resolved,
                diagnostics,
                phase="gripper_closing" if closing else "gripper_opening",
                measured_gripper_width_m=measured_width,
            )
            contact_state_matches = all(contacts) if closing else not any(contacts)
            width_reached = abs(measured_width - target_width) <= self.limits.gripper_width_tolerance_m
            if contact_state_matches and width_reached:
                if (
                    previous_width is not None
                    and abs(measured_width - previous_width) <= self.limits.gripper_stability_tolerance_m
                ):
                    stable_samples += 1
                    stable_elapsed_s += self.limits.control_dt_s
                else:
                    stable_samples = 1
                    stable_elapsed_s = 0.0
                previous_width = measured_width
                if stable_samples >= 2 and stable_elapsed_s >= self.limits.minimum_stable_duration_s:
                    return current
            else:
                stable_samples = 0
                stable_elapsed_s = 0.0
                previous_width = None

            if step_index == self.limits.max_steps:
                break
            previous = current
            current = self._next_observation(
                self._step((0.0,) * 6, diagnostics),
                previous_id=previous.get("observation_id"),
            )
        raise GoalResolutionError(
            "gripper_timeout",
            "gripper did not reach stable measured width and bilateral contact state within the bounded step budget",
        )

    def _step(self, action: Sequence[float], diagnostics: dict[str, int]) -> Any:
        # Count executor-issued driver steps; safety hold calls are not motion steps.
        diagnostics["driver_step_count"] += 1
        return self.driver.step(action)

    def _next_observation(self, step_result: Any, *, previous_id: Any) -> Mapping[str, Any]:
        observation = step_result.get("observation") if isinstance(step_result, Mapping) else None
        if not isinstance(observation, Mapping):
            provider = self.observation_provider or getattr(self.driver, "observe_public", None)
            observation = provider()
        if not isinstance(observation, Mapping):
            raise GoalResolutionError("invalid_observation", "step did not produce a public observation")
        current_id = observation.get("observation_id")
        if not isinstance(current_id, str) or not current_id or current_id == previous_id:
            raise GoalResolutionError("stale_observation", "each control step requires a fresh observation_id")
        return observation

    def _check_uncertainty(self, resolved: ResolvedGoal) -> None:
        if resolved.uncertainty_position_m > self.limits.max_goal_uncertainty_m:
            raise GoalResolutionError("uncertainty_too_high", "feature position uncertainty exceeds configured robot precision")
        if resolved.uncertainty_axis_rad > self.limits.max_goal_axis_uncertainty_rad:
            raise GoalResolutionError("uncertainty_too_high", "feature axis uncertainty exceeds configured robot precision")

    def _check_joint_limits(self, positions: np.ndarray) -> None:
        for index, (value, (lower, upper)) in enumerate(zip(positions, self.joint_limits)):
            if value < lower - 1e-8 or value > upper + 1e-8:
                raise GoalResolutionError("joint_limit", f"IK/path waypoint exceeds joint limit {index}")

    def _hold(self) -> None:
        try:
            self.driver.hold()
        except Exception:
            pass


class FactoryAssemblyDriver:
    """Single-environment adapter for the existing Factory Cartesian controller.

    Link sweeps use capsule broad phase and per-collider conservative boxes for
    narrow phase. The Jacobian sweep still requires physical validation against
    the Factory articulation. Collision geometry comes from the robot asset;
    RGB-D points must already exclude the robot's own visible surface.
    """

    def __init__(
        self,
        env: Any,
        *,
        observe_public: Callable[[], Mapping[str, Any]],
        robot_link_radii_m: Mapping[str, float],
        robot_link_parents: Mapping[str, str],
        robot_link_collision_bounds_m: Mapping[
            str, Sequence[tuple[Sequence[float], Sequence[float]]]
        ] | None,
        ik_damping: float,
        sweep_margin_m: float,
        private_collision_recorder: Callable[[Mapping[str, Any]], None] | None = None,
        step_observation: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if getattr(env, "num_envs", None) != 1:
            raise ValueError("FactoryAssemblyDriver requires exactly one environment")
        if not callable(observe_public):
            raise TypeError("observe_public callback is required")
        if not isfinite(float(ik_damping)) or ik_damping <= 0.0:
            raise ValueError("ik_damping must be positive and finite")
        if not isfinite(float(sweep_margin_m)) or sweep_margin_m <= 0.0:
            raise ValueError("sweep_margin_m must be positive and finite")
        robot = getattr(env, "_robot", None)
        body_names = getattr(robot, "body_names", None)
        if not isinstance(body_names, Sequence) or not body_names:
            raise TypeError("Factory articulation must expose body_names")
        names = tuple(str(name) for name in body_names)
        if len(set(names)) != len(names):
            raise ValueError("Factory articulation body_names must be unique")
        if set(robot_link_radii_m) != set(names):
            raise ValueError("calibrated robot_link_radii_m must cover every Factory articulation body")
        parents = dict(robot_link_parents)
        if (
            len(parents) != len(names) - 1
            or not set(parents).issubset(names)
            or any(parent not in names or child == parent for child, parent in parents.items())
        ):
            raise ValueError("robot_link_parents must define one parent for each non-root body")
        roots = set(names) - set(parents)
        if len(roots) != 1:
            raise ValueError("robot_link_parents must define exactly one articulation root")
        root_name = next(iter(roots))
        for name in names:
            visited = set()
            current = name
            while current in parents:
                if current in visited:
                    raise ValueError("robot_link_parents must be acyclic")
                visited.add(current)
                current = parents[current]
            if current != root_name:
                raise ValueError("robot_link_parents must connect every body to the articulation root")
        radii = {name: float(value) for name, value in robot_link_radii_m.items()}
        if any(not isfinite(value) or value <= 0.0 for value in radii.values()):
            raise ValueError("robot link radii must be positive and finite")
        collision_bounds = _validated_link_collision_bounds(
            robot_link_collision_bounds_m, names, radii
        )

        self.env = env
        self._observe_public = observe_public
        self._step_observation = step_observation
        self.body_names = names
        self.robot_link_radii_m = radii
        self.robot_link_parents = parents
        self.robot_link_collision_bounds_m = collision_bounds
        self.fixed_base = _articulation_is_fixed_base(robot)
        self.root_body_name = root_name
        self._jacobian_body_names = tuple(name for name in names if name != root_name)
        children_by_parent: dict[str, list[str]] = {}
        for child, parent in parents.items():
            children_by_parent.setdefault(parent, []).append(child)
        gripper_leaves = tuple(sorted(
            name for name in names if name != root_name and not children_by_parent.get(name)
        ))
        finger_ids = getattr(env, "_finger_joint_ids", None)
        calibrated_fingers = finger_ids is not None and np.asarray(finger_ids).size == 2
        shared_parent = (
            len(gripper_leaves) == 2
            and parents.get(gripper_leaves[0]) == parents.get(gripper_leaves[1])
        )
        self._gripper_contact_bodies = gripper_leaves if calibrated_fingers and shared_parent else ()
        self.ik_damping = float(ik_damping)
        self.sweep_margin_m = float(sweep_margin_m)
        self.last_sweep_collision_diagnostic: dict[str, Any] | None = None
        self.last_sweep_collision_private_diagnostic: dict[str, Any] | None = None
        self._private_collision_recorder = private_collision_recorder
        self.joint_limits = self._read_joint_limits()
        if finger_ids is None:
            raise TypeError("Factory environment must expose calibrated finger joint ids")
        limits = _to_numpy(robot.data.joint_pos_limits)[0]
        self.max_gripper_width_m = 2.0 * float(np.min(limits[np.asarray(finger_ids, dtype=int), 1]))
        if not isfinite(self.max_gripper_width_m) or self.max_gripper_width_m <= 0.0:
            raise ValueError("Factory gripper width calibration is invalid")

    def observe_public(self) -> Mapping[str, Any]:
        return self._observe_public()

    def solve_ik(self, tcp_pose: Sequence[float], seed_joint_positions: Sequence[float]) -> Sequence[float] | None:
        target = _pose(tcp_pose, "IK target tcp_pose")
        joint_seed = _vector(seed_joint_positions, len(self.joint_limits), "IK seed")
        current_joints = _to_numpy(self.env._robot.data.joint_pos)[0, :len(self.joint_limits)]
        if np.max(np.abs(current_joints - joint_seed)) > 1e-3:
            return None
        current_position = _to_numpy(self.env.fingertip_midpoint_pos)[0]
        current_quaternion = _to_numpy(self.env.fingertip_midpoint_quat)[0]
        error = np.concatenate((
            np.asarray(target[:3]) - current_position,
            _rotation_vector(target[3:], current_quaternion),
        ))
        jacobian = _to_numpy(self.env.fingertip_midpoint_jacobian)[0, :, :len(self.joint_limits)]
        if jacobian.shape != (6, len(self.joint_limits)):
            return None
        damping_sq = self.ik_damping * self.ik_damping
        try:
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping_sq * np.eye(6), error)
        except np.linalg.LinAlgError:
            return None
        predicted_error = error - jacobian @ delta
        if (
            np.linalg.norm(predicted_error[:3]) > max(1e-4, 0.2 * np.linalg.norm(error[:3]))
            or np.linalg.norm(predicted_error[3:]) > max(1e-3, 0.2 * np.linalg.norm(error[3:]))
        ):
            return None
        return (joint_seed + delta).tolist()

    def check_robot_sweep_collision(
        self,
        start_tcp_pose: Sequence[float],
        end_tcp_pose: Sequence[float],
        start_joint_positions: Sequence[float],
        end_joint_positions: Sequence[float],
        obstacle_points_world_m: Sequence[Sequence[float]],
        moving_geometry: Mapping[str, Any],
    ) -> bool:
        self.last_sweep_collision_diagnostic = None
        self.last_sweep_collision_private_diagnostic = None
        cloud = np.asarray(obstacle_points_world_m, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            return False
        start = _vector(start_joint_positions, len(self.joint_limits), "sweep start joints")
        end = _vector(end_joint_positions, len(self.joint_limits), "sweep end joints")
        delta = end - start
        robot_data = self.env._robot.data
        origins = _to_numpy(self.env.scene.env_origins)[0]
        body_positions = _to_numpy(robot_data.body_pos_w)[0] - origins
        jacobians = _to_numpy(self.env._robot.root_physx_view.get_jacobians())[0]
        if (
            body_positions.shape != (len(self.body_names), 3)
            or jacobians.ndim != 3
            or jacobians.shape[0] < len(self._jacobian_body_names)
            or jacobians.shape[1] < 6
            or jacobians.shape[2] < len(delta)
        ):
            return False
        body_velocity = np.zeros_like(body_positions)
        body_rotation_delta = np.zeros_like(body_positions)
        for jacobian_index, body_name in enumerate(self._jacobian_body_names):
            body_index = self.body_names.index(body_name)
            body_velocity[body_index] = jacobians[jacobian_index, :3, :len(delta)] @ delta
            body_rotation_delta[body_index] = jacobians[jacobian_index, 3:6, :len(delta)] @ delta
        if not np.isfinite(body_velocity).all() or not np.isfinite(body_rotation_delta).all():
            return False

        try:
            body_quaternions = _to_numpy(robot_data.body_quat_w)[0]
        except (AttributeError, IndexError, TypeError, ValueError):
            return False
        if body_quaternions.shape != (len(self.body_names), 4) or not np.isfinite(body_quaternions).all():
            return False
        try:
            body_quaternions = np.asarray([_quat(value) for value in body_quaternions])
        except ValueError:
            return False

        root_stationary = self.fixed_base and _root_has_no_native_motion(
            robot_data, self.body_names.index(self.root_body_name), len(self.body_names)
        )

        tcp_start = _pose(start_tcp_pose, "sweep start tcp")
        tcp_end = _pose(end_tcp_pose, "sweep end tcp")
        contact_pairs = _validated_contact_pairs(moving_geometry.get("_allowed_contact_pairs"))
        if contact_pairs is None:
            return False
        local_feature_pose = moving_geometry.get("_tcp_to_feature_pose")
        if contact_pairs and (not isinstance(local_feature_pose, Sequence) or len(local_feature_pose) != 7):
            return False
        max_link_shift = max(
            float(np.linalg.norm(body_velocity[index]))
            + self.robot_link_radii_m[name] * float(np.linalg.norm(body_rotation_delta[index]))
            for index, name in enumerate(self.body_names)
        )
        max_tcp_shift = float(np.linalg.norm(np.asarray(tcp_end[:3]) - np.asarray(tcp_start[:3])))
        max_sample_shift = max(1e-4, 0.5 * self.sweep_margin_m)
        steps = max(1, int(np.ceil(max(max_link_shift, max_tcp_shift) / max_sample_shift)))
        body_sample_motion_bounds = {
            name: (
                float(np.linalg.norm(body_velocity[index]))
                + self.robot_link_radii_m[name]
                * float(np.linalg.norm(body_rotation_delta[index]))
            ) / steps
            for index, name in enumerate(self.body_names)
        }
        times = np.linspace(0.0, 1.0, steps + 1)
        for fraction in times:
            centers = body_positions + fraction * body_velocity
            tcp_pose = _interpolate_pose(tcp_start, tcp_end, float(fraction))
            moving_position = _compose_pose(tcp_pose, local_feature_pose)[0] if contact_pairs else None
            for child, parent in self.robot_link_parents.items():
                child_index, parent_index = self.body_names.index(child), self.body_names.index(parent)
                capsule_radius = max(self.robot_link_radii_m[child], self.robot_link_radii_m[parent])
                between_sample_padding = 0.5 * max(
                    body_sample_motion_bounds[child], body_sample_motion_bounds[parent],
                )
                radius = capsule_radius + self.sweep_margin_m + between_sample_padding
                collisions = _cloud_segment_collision_mask(
                    cloud, centers[parent_index], centers[child_index], radius,
                )
                if not np.any(collisions):
                    continue
                if self.robot_link_collision_bounds_m is not None:
                    narrow_phase = np.zeros(len(cloud), dtype=bool)
                    for body_name, body_index in ((parent, parent_index), (child, child_index)):
                        if body_name == self.root_body_name and root_stationary:
                            continue
                        rotation = _quat_rotation_matrix(_integrate_world_rotation(
                            body_quaternions[body_index],
                            fraction * body_rotation_delta[body_index],
                        ))
                        # Cover the body-point travel between adjacent Jacobian samples.
                        box_inflation = self.sweep_margin_m + 0.5 * body_sample_motion_bounds[body_name]
                        for lower, upper in self.robot_link_collision_bounds_m[body_name]:
                            narrow_phase |= _cloud_oriented_box_collision_mask(
                                cloud,
                                centers[body_index],
                                rotation,
                                lower,
                                upper,
                                box_inflation,
                            )
                    collisions &= narrow_phase
                    if not np.any(collisions):
                        continue
                if child in self._gripper_contact_bodies and moving_position is not None:
                    for pair in contact_pairs:
                        if pair["moving_role"] != "gripper" or not _moving_feature_at_contact(moving_position, pair):
                            continue
                        collisions &= ~_contact_surface_point_mask(cloud, pair)
                if np.any(collisions):
                    hit_index = int(np.flatnonzero(collisions)[0])
                    hit_point = cloud[hit_index]
                    hit_box = None
                    sweep_poses = []
                    for body_index, body_name in enumerate(self.body_names):
                        body_rotation = _quat_rotation_matrix(_integrate_world_rotation(
                            body_quaternions[body_index],
                            fraction * body_rotation_delta[body_index],
                        ))
                        body_quaternion = _integrate_world_rotation(
                            body_quaternions[body_index],
                            fraction * body_rotation_delta[body_index],
                        )
                        sweep_poses.append({
                            "body_name": body_name,
                            "position_world_m": centers[body_index].tolist(),
                            "quaternion_wxyz": body_quaternion.tolist(),
                        })
                    if self.robot_link_collision_bounds_m is not None:
                        for body_name, body_index in ((parent, parent_index), (child, child_index)):
                            if body_name == self.root_body_name and root_stationary:
                                continue
                            body_quaternion = _integrate_world_rotation(
                                body_quaternions[body_index],
                                fraction * body_rotation_delta[body_index],
                            )
                            rotation = _quat_rotation_matrix(body_quaternion)
                            inflation = (
                                self.sweep_margin_m
                                + 0.5 * body_sample_motion_bounds[body_name]
                            )
                            for box_index, (lower, upper) in enumerate(
                                self.robot_link_collision_bounds_m[body_name]
                            ):
                                local_point = rotation.T @ (hit_point - centers[body_index])
                                if np.all(local_point >= lower - inflation) and np.all(
                                    local_point <= upper + inflation
                                ):
                                    hit_box = {
                                        "body_name": body_name,
                                        "box_index": box_index,
                                        "local_point_m": local_point.tolist(),
                                        "lower_m": lower.tolist(),
                                        "upper_m": upper.tolist(),
                                        "inflation_m": float(inflation),
                                        "body_position_world_m": centers[body_index].tolist(),
                                        "body_quaternion_wxyz": body_quaternion.tolist(),
                                    }
                                    break
                            if hit_box is not None:
                                break
                    self.last_sweep_collision_diagnostic = _sweep_collision_details(
                        cloud, collisions, centers[parent_index], centers[child_index],
                        collision_kind="robot_link", body_child=child, body_parent=parent,
                        sample_fraction=float(fraction), capsule_radius_m=capsule_radius,
                        sweep_margin_m=self.sweep_margin_m,
                        between_sample_padding_m=between_sample_padding,
                    )
                    self.last_sweep_collision_private_diagnostic = {
                        "obstacle_point_index": hit_index,
                        "obstacle_point_world_m": hit_point.tolist(),
                        "sample_fraction": float(fraction),
                        "narrow_phase_active": self.robot_link_collision_bounds_m is not None,
                        "hit_obb": hit_box,
                        "sweep_body_poses": sweep_poses,
                    }
                    if self._private_collision_recorder is not None:
                        try:
                            self._private_collision_recorder(
                                self.last_sweep_collision_private_diagnostic
                            )
                        except Exception:
                            _LOGGER.exception("Could not persist private sweep-collision evidence")
                    return False
        if moving_geometry.get("_held_object") is True:
            radius_value = moving_geometry.get("collision_radius_m", moving_geometry.get("bounding_radius_m"))
            local_pose = moving_geometry.get("_tcp_to_feature_pose")
            if (
                isinstance(radius_value, bool) or not isinstance(radius_value, (int, float))
                or not isfinite(float(radius_value)) or float(radius_value) <= 0.0
                or not isinstance(local_pose, Sequence) or len(local_pose) != 7
            ):
                return False
            object_radius = float(radius_value)
            start_position = _compose_pose(tcp_start, local_pose)[0]
            start_distances = np.linalg.norm(cloud - start_position, axis=1)
            start_margin_only = (
                (start_distances > object_radius)
                & (start_distances <= object_radius + self.sweep_margin_m)
            )
            for fraction in times:
                tcp_pose = _interpolate_pose(tcp_start, tcp_end, float(fraction))
                object_position = _compose_pose(tcp_pose, local_pose)[0]
                distances = np.linalg.norm(cloud - object_position, axis=1)
                collisions = distances <= object_radius + self.sweep_margin_m
                # Let the held object leave a pre-existing margin-only overlap, but never a physical overlap or a worsening one.
                if fraction > 0.0:
                    collisions &= ~(start_margin_only & (distances >= start_distances - 1e-6))
                else:
                    collisions &= ~start_margin_only
                for pair in contact_pairs:
                    if pair["moving_role"] == "held_object" and _moving_feature_at_contact(object_position, pair):
                        collisions &= ~_contact_surface_point_mask(cloud, pair)
                if np.any(collisions):
                    self.last_sweep_collision_diagnostic = _sweep_collision_details(
                        cloud, collisions, object_position, object_position,
                        collision_kind="held_object", body_child="held_object", body_parent=None,
                        sample_fraction=float(fraction), capsule_radius_m=object_radius,
                        sweep_margin_m=self.sweep_margin_m,
                    )
                    return False
        return True

    def step(self, normalized_tcp_delta: Sequence[float]) -> Mapping[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Factory stepping requires the Isaac Lab torch runtime") from exc
        action = _vector(normalized_tcp_delta, 6, "normalized TCP action")
        self.env.step(torch.as_tensor([action.tolist()], dtype=torch.float32, device=self.env.device))
        observer = self._step_observation or self.observe_public
        return {"observation": observer()}

    def set_gripper_target_width(self, width_m: float, max_force_n: float) -> None:
        self.env.set_gripper_target_width(width_m, max_force_n)

    def hold(self) -> None:
        self.env.commanded_tcp_pos[:] = self.env.fingertip_midpoint_pos
        self.env.commanded_tcp_quat[:] = self.env.fingertip_midpoint_quat
        self.step((0.0,) * 6)

    def _read_joint_limits(self) -> tuple[tuple[float, float], ...]:
        limits = _to_numpy(self.env._robot.data.joint_pos_limits)[0, :7]
        if limits.shape != (7, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise TypeError("Factory Panda arm joint limits must be finite 7x2 bounds")
        return tuple((float(row[0]), float(row[1])) for row in limits)


def _obstacle_points(observation: Mapping[str, Any], *, max_age_s: float) -> np.ndarray:
    sensors = _sensors(observation)
    obstacle_map = sensors.get("rgbd_obstacle_map")
    observation_id = observation.get("observation_id")
    if not isinstance(obstacle_map, Mapping):
        raise GoalResolutionError("missing_rgbd_map", "online collision checking requires a public RGB-D obstacle map")
    camera_id = _camera_observation_id(observation, sensors)
    map_source_id = obstacle_map.get("source_observation_id")
    current_source_ids = {observation_id, camera_id}
    if (
        not isinstance(map_source_id, str)
        or map_source_id not in current_source_ids
        or obstacle_map.get("frame") not in {"world", "world_m"}
    ):
        raise GoalResolutionError("stale_rgbd_map", "RGB-D obstacle map must cite the current camera frame and world coordinates")
    if obstacle_map.get("valid") is not True:
        raise GoalResolutionError("invalid_rgbd_map", "RGB-D obstacle map is not marked valid")
    if obstacle_map.get("robot_self_filtered") is not True:
        raise GoalResolutionError("invalid_rgbd_map", "RGB-D collision points must exclude the robot's visible surface")
    robot_time = sensors.get("robot_timestamp_s", observation.get("timestamp"))
    camera_time = sensors.get("rgbd_source_timestamp_s", obstacle_map.get("timestamp_s"))
    if robot_time is None or camera_time is None:
        if camera_id != observation_id:
            raise GoalResolutionError("stale_rgbd_map", "asynchronous RGB-D and robot samples require separate timestamps")
    else:
        if (
            isinstance(robot_time, bool) or not isinstance(robot_time, (int, float)) or not isfinite(float(robot_time))
            or isinstance(camera_time, bool) or not isinstance(camera_time, (int, float)) or not isfinite(float(camera_time))
        ):
            raise GoalResolutionError("invalid_rgbd_map", "robot and camera timestamps must be finite seconds")
        age = float(robot_time) - float(camera_time)
        if age < -1e-6 or age > max_age_s + 1e-6:
            raise GoalResolutionError("stale_rgbd_map", "RGB-D obstacle map is future-dated or older than the configured age bound")
        map_time = obstacle_map.get("timestamp_s")
        if map_time is not None:
            if (
                isinstance(map_time, bool)
                or not isinstance(map_time, (int, float))
                or not isfinite(float(map_time))
            ):
                raise GoalResolutionError("invalid_rgbd_map", "obstacle map timestamp_s must be a finite number")
            if abs(float(map_time) - float(camera_time)) > 1e-6:
                raise GoalResolutionError("stale_rgbd_map", "obstacle map timestamp does not match its camera source timestamp")
    points = obstacle_map.get("points_m")
    if not isinstance(points, np.ndarray) and (
        not isinstance(points, Sequence) or isinstance(points, (str, bytes))
    ):
        raise GoalResolutionError("invalid_rgbd_map", "RGB-D obstacle points must be a bounded sequence")
    if len(points) > 200_000:
        raise GoalResolutionError("invalid_rgbd_map", "RGB-D obstacle points must be a bounded sequence")
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise GoalResolutionError("invalid_rgbd_map", "RGB-D obstacle points must be finite world xyz triples")
    return array


def _without_held_surface_samples(
    points: np.ndarray,
    geometry: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    moving_feature_ref: str | None = None,
    protected_feature_refs: Sequence[str] | None = None,
    sweep_margin_m: float = 0.005,
) -> np.ndarray:
    """Remove current held-surface samples and their local depth-connected continuation."""
    sensors = _sensors(observation)
    samples = geometry.get("contact_surface_points_world_m")
    frame_id = geometry.get("contact_surface_frame_id")
    sample_time = geometry.get("contact_surface_timestamp_s")
    camera_time = sensors.get("rgbd_source_timestamp_s")
    if (
        not isinstance(samples, Sequence)
        or isinstance(samples, (str, bytes))
        or not 1 <= len(samples) <= 512
        or frame_id != sensors.get("camera_frame_id")
        or isinstance(sample_time, bool)
        or not isinstance(sample_time, (int, float))
        or not isfinite(float(sample_time))
        or isinstance(camera_time, bool)
        or not isinstance(camera_time, (int, float))
        or not isfinite(float(camera_time))
        or abs(float(sample_time) - float(camera_time)) > 1e-6
    ):
        return points
    surface = np.asarray(samples, dtype=np.float64)
    if surface.ndim != 2 or surface.shape[1] != 3 or not np.isfinite(surface).all():
        return points

    # Feature and obstacle clouds come from the same RGB-D frame; 0.1 mm only
    # absorbs floating-point projection differences, not neighboring geometry.
    tolerance_m = 1e-4
    key_dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    cloud_keys = np.rint(points / tolerance_m).astype(np.int64).view(key_dtype).reshape(-1)
    surface_keys = np.rint(surface / tolerance_m).astype(np.int64).view(key_dtype).reshape(-1)
    protected_samples = []
    projected_protected_samples = []
    projected_protected_refs = None if protected_feature_refs is None else set(protected_feature_refs)
    features = observation.get("features", ())
    if isinstance(features, Sequence) and not isinstance(features, (str, bytes)):
        for feature in features:
            if not isinstance(feature, Mapping) or feature.get("feature_ref") == moving_feature_ref:
                continue
            feature_ref = feature.get("feature_ref")
            feature_geometry = feature.get("geometry")
            raw = feature_geometry.get("contact_surface_points_world_m") \
                if isinstance(feature_geometry, Mapping) else None
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not 1 <= len(raw) <= 512:
                continue
            candidate = np.asarray(raw, dtype=np.float64)
            if candidate.ndim == 2 and candidate.shape[1] == 3 and np.isfinite(candidate).all():
                protected_samples.append(candidate)
                if projected_protected_refs is None or feature_ref in projected_protected_refs:
                    projected_protected_samples.append(candidate)
    protected = np.zeros(len(points), dtype=bool)
    if protected_samples:
        protected_keys = np.rint(np.concatenate(protected_samples) / tolerance_m).astype(np.int64) \
            .view(key_dtype).reshape(-1)
        protected = np.isin(cloud_keys, protected_keys)
    matched = np.isin(cloud_keys, surface_keys) & ~protected
    keep = ~matched

    center = geometry.get("position_m")
    radius = geometry.get("bounding_radius_m")
    intrinsics = sensors.get("camera_intrinsics")
    extrinsics = sensors.get("camera_extrinsics")
    try:
        center_array = np.asarray(center, dtype=np.float64)
        radius_value = float(radius)
        margin_value = float(sweep_margin_m)
        k = np.asarray(intrinsics, dtype=np.float64)
        t_world_camera = np.asarray(extrinsics, dtype=np.float64)
        if (
            center_array.shape != (3,)
            or k.shape != (3, 3) or t_world_camera.shape != (4, 4)
            or not np.isfinite(center_array).all()
            or not np.isfinite(k).all() or not np.isfinite(t_world_camera).all()
            or not isfinite(radius_value) or not 0.0 < radius_value <= 0.05
            or not isfinite(margin_value) or not 0.0 < margin_value <= 0.02
        ):
            return points[keep]
        camera_from_world = np.linalg.inv(t_world_camera)
        camera_points = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
        projected = camera_points @ k.T
        in_front = (camera_points[:, 2] > 0.0) & (projected[:, 2] > 0.0)
        uv = projected[:, :2] / projected[:, 2:3]
        pixels = np.zeros((len(points), 2), dtype=np.int64)
        pixels[in_front] = np.rint(uv[in_front]).astype(np.int64)
        radius_bound_m = radius_value + margin_value + 0.002
        center_distance = np.linalg.norm(points - center_array, axis=1)
        local = in_front & (center_distance <= radius_bound_m)
        held_axis_extension = np.zeros(len(points), dtype=bool)
        try:
            tcp_pose = np.asarray(sensors.get("tcp_pose"), dtype=np.float64)
            if tcp_pose.shape == (7,) and np.isfinite(tcp_pose).all():
                tcp_axis = _quat_rotation_matrix(tcp_pose[3:])[:, 2]
                relative_points = points - center_array
                axial_distance = relative_points @ tcp_axis
                radial_distance = np.linalg.norm(
                    relative_points - axial_distance[:, None] * tcp_axis,
                    axis=1,
                )
                # shortcut: self-filter can occlude the shaft/head bridge; seed only the tight TCP +Z extension.
                held_axis_extension = (
                    in_front
                    & (center_distance <= radius_bound_m + 0.002)
                    & (axial_distance > radius_bound_m)
                    & (radial_distance <= 0.006)
                )
                local |= held_axis_extension
        except (TypeError, ValueError, ArithmeticError):
            pass
        local_indices = np.flatnonzero(local)
        if not len(local_indices):
            return points[keep]

        def pixel_stride(values: np.ndarray) -> int:
            differences = np.diff(np.unique(values[local]))
            differences = differences[differences > 0]
            stride = int(np.gcd.reduce(differences)) if len(differences) else 6
            return stride if 0 < stride <= 16 else 6

        step_x = pixel_stride(pixels[:, 0])
        step_y = pixel_stride(pixels[:, 1])
        index_by_pixel = {
            (int(pixels[index, 0]), int(pixels[index, 1])): int(index)
            for index in local_indices
        }
        protected_pixel_radius = 0.5 * np.hypot(step_x, step_y) + 1.0
        # shortcut: held-surface matching spans one missing RGB-D grid cell; keep the 3D bound authoritative.
        held_pixel_radius = max(protected_pixel_radius, float(np.hypot(step_x, step_y)))

        def projected_surface_matches(surface_points: np.ndarray, pixel_radius: float) -> np.ndarray:
            matches = np.zeros(len(points), dtype=bool)
            surface_camera = surface_points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
            surface_projected = surface_camera @ k.T
            valid_surface = (surface_camera[:, 2] > 0.0) & (surface_projected[:, 2] > 0.0)
            surface_uv = np.zeros((len(surface_points), 2), dtype=float)
            surface_uv[valid_surface] = surface_projected[valid_surface, :2] / surface_projected[valid_surface, 2:3]
            for sample_index in np.flatnonzero(valid_surface):
                uv_sample = surface_uv[sample_index]
                grid_x = int(np.rint(uv_sample[0] / step_x)) * step_x
                grid_y = int(np.rint(uv_sample[1] / step_y)) * step_y
                # The 6 px obstacle grid can place the nearest point over 10 mm from a dense feature sample.
                # User-approved 2 mm allowance bridges the observed held-part depth gap.
                tolerance_m = max(0.014, 0.01 * float(surface_camera[sample_index, 2]))
                for candidate_pixel in (
                    (grid_x + dx * step_x, grid_y + dy * step_y)
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                ):
                    candidate = index_by_pixel.get(candidate_pixel)
                    if candidate is None or protected[candidate]:
                        continue
                    if np.linalg.norm(pixels[candidate] - uv_sample) > pixel_radius:
                        continue
                    if np.linalg.norm(points[candidate] - surface_points[sample_index]) <= tolerance_m:
                        matches[candidate] = True
            return matches

        if projected_protected_samples:
            protected |= projected_surface_matches(
                np.concatenate(projected_protected_samples), protected_pixel_radius,
            )
        seeds = (matched | projected_surface_matches(surface, held_pixel_radius) | held_axis_extension) & ~protected
        visited = seeds.copy()
        connected = seeds.copy()
        queue = [int(index) for index in np.flatnonzero(seeds & local)]
        cursor = 0
        while cursor < len(queue):
            index = queue[cursor]
            cursor += 1
            x, y = pixels[index]
            threshold = max(0.010, 0.01 * float(camera_points[index, 2]))
            for neighbor_pixel in ((x - step_x, y), (x + step_x, y), (x, y - step_y), (x, y + step_y)):
                neighbor = index_by_pixel.get((int(neighbor_pixel[0]), int(neighbor_pixel[1])))
                if neighbor is None or visited[neighbor]:
                    continue
                visited[neighbor] = True
                if protected[neighbor] or np.linalg.norm(points[neighbor] - points[index]) > threshold:
                    continue
                connected[neighbor] = True
                queue.append(neighbor)
        keep &= ~connected
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError):
        return points[keep]
    return points[keep]


def _sensors(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, Mapping):
        raise GoalResolutionError("invalid_observation", "robot_state must be a public mapping")
    sensors = robot_state.get("sensors", {})
    if not isinstance(sensors, Mapping):
        raise GoalResolutionError("invalid_observation", "robot_state.sensors must be a mapping")
    return {**robot_state, **sensors}


def _object_not_held_feedback(
    observation: Mapping[str, Any],
    grasped: Any,
    *,
    driver_steps: int = 0,
) -> str:
    sensors = _sensors(observation)
    features = observation.get("features", ())
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        features = ()
    tcp_count = 0
    candidate_count = 0
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        frame = feature.get("frame")
        if feature.get("kind") == "robot_tcp" and frame == "tcp":
            tcp_count += 1
        geometry = feature.get("geometry")
        width = geometry.get("grasp_width_m") if isinstance(geometry, Mapping) else None
        if (
            frame in {"world", "world_m"}
            and not isinstance(width, bool)
            and isinstance(width, (int, float))
            and isfinite(float(width))
            and width > 0.0
        ):
            candidate_count += 1

    if driver_steps:
        motion_feedback = (
            f"Motion was paused after {driver_steps} control step(s) when bilateral grasp verification dropped; "
            "the part may have shifted, so re-ground from the returned fresh observation before another move."
        )
    else:
        motion_feedback = "No robot step was sent because bilateral grasp verification is not true."
    grasp_state = "public grasped state is false" if grasped is False else "public grasped state is not verified true"
    forces = sensors.get("finger_contact_force_n")
    force_feedback = ""
    if (
        isinstance(forces, Sequence)
        and not isinstance(forces, (str, bytes))
        and len(forces) == 2
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and isfinite(float(value))
            and float(value) >= 0.0
            for value in forces
        )
    ):
        force_feedback = (
            f"Per-finger contact forces at the stop were {float(forces[0]):.3f} N and "
            f"{float(forces[1]):.3f} N; treat them as contact evidence, not permission to move."
        )
    if tcp_count == 1 and candidate_count:
        operation_hint = (
            f"Current public features expose typed-grasp inputs: one TCP gripper and "
            f"{candidate_count} geometry-qualified candidate(s); select only a candidate grounded in the current image, or stop."
        )
    else:
        operation_hint = (
            "Current public features do not expose both one TCP gripper and a geometry-qualified candidate; "
            "a typed grasp is not available from this observation."
        )
    return f"{motion_feedback} World-frame object motion requires a verified grasp; {grasp_state}. {force_feedback} {operation_hint}"


def _force_limit_reached(sensors: Mapping[str, Any], max_force_n: float) -> bool:
    wrench = _vector(sensors.get("wrench"), 6, "robot_state.sensors.wrench")
    return float(np.linalg.norm(wrench[:3])) > max_force_n


def _gripper_measurements(
    sensors: Mapping[str, Any],
    *,
    max_width_m: float,
) -> tuple[float, tuple[bool, bool]]:
    width = sensors.get("gripper_width_m", sensors.get("gripper_width"))
    contacts = sensors.get("finger_contacts", sensors.get("contacts"))
    if (
        isinstance(width, bool) or not isinstance(width, (int, float))
        or not isfinite(float(width)) or float(width) < 0.0
        or float(width) > max_width_m + 1e-6
    ):
        raise GoalResolutionError("gripper_state_unavailable", "public measured gripper width is missing or invalid")
    if (
        not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes))
        or len(contacts) != 2 or any(type(value) is not bool for value in contacts)
    ):
        raise GoalResolutionError("gripper_state_unavailable", "public bilateral finger contacts are missing or invalid")
    return float(width), (contacts[0], contacts[1])


def _joint_limits(driver: Any) -> tuple[tuple[float, float], ...]:
    raw = getattr(driver, "joint_limits", None)
    values = np.asarray(raw, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise TypeError("driver.joint_limits must be finite [joint_count, 2] bounds")
    if np.any(values[:, 0] >= values[:, 1]):
        raise TypeError("each robot joint lower limit must be below its upper limit")
    return tuple((float(row[0]), float(row[1])) for row in values)


def _pose(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        position = _vector(value.get("position_m", value.get("position")), 3, f"{name}.position")
        quaternion = _quat(_vector(value.get("quaternion_wxyz", value.get("quaternion")), 4, f"{name}.quaternion"))
        return (*position.tolist(), *quaternion.tolist())
    pose = _vector(value, 7, name)
    return (*pose[:3].tolist(), *_quat(pose[3:]).tolist())


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _strict_vector(value: Any, size: int, name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} finite numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain exactly {size} finite numbers")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly {size} finite numbers")
    return result


def _quat(value: Sequence[float]) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    magnitude = float(np.linalg.norm(result))
    if magnitude <= 1e-9 or not isfinite(magnitude):
        raise ValueError("quaternion must be finite and non-zero")
    result /= magnitude
    return -result if result[0] < 0.0 else result


def _quat_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    w, x, y, z = left
    a, b, c, d = right
    return np.array((w*a - x*b - y*c - z*d, w*b + x*a + y*d - z*c,
                     w*c - x*d + y*a + z*b, w*d + x*c - y*b + z*a))


def _to_numpy(value: Any) -> np.ndarray:
    if callable(getattr(value, "detach", None)):
        value = value.detach()
    if callable(getattr(value, "cpu", None)):
        value = value.cpu()
    if callable(getattr(value, "numpy", None)):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _cloud_near_segment(
    points: np.ndarray, start: Sequence[float], end: Sequence[float], radius_m: float,
) -> bool:
    return bool(np.any(_cloud_segment_collision_mask(points, start, end, radius_m)))


def _validated_link_collision_bounds(
    bounds: Mapping[str, Sequence[tuple[Sequence[float], Sequence[float]]]] | None,
    body_names: Sequence[str],
    radii_m: dict[str, float],
) -> dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] | None:
    if not isinstance(bounds, Mapping) or set(bounds) != set(body_names):
        return None
    normalized: dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] = {}
    for name in body_names:
        body_boxes = bounds[name]
        if not isinstance(body_boxes, Sequence) or isinstance(body_boxes, (str, bytes)) or not body_boxes:
            return None
        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        body_radius = 0.0
        for box in body_boxes:
            try:
                corners = np.asarray(box, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                return None
            if (
                corners.shape != (2, 3)
                or not np.isfinite(corners).all()
                or not np.all(corners[0] < corners[1])
            ):
                return None
            lower, upper = corners[0].copy(), corners[1].copy()
            boxes.append((lower, upper))
            for mask in range(8):
                corner = np.where(
                    np.asarray([(mask >> axis) & 1 for axis in range(3)], dtype=bool),
                    upper,
                    lower,
                )
                body_radius = max(body_radius, float(np.linalg.norm(corner)))
        if not boxes or not isfinite(body_radius) or body_radius <= 0.0:
            return None
        # Keep the capsule broad phase a superset even for hand-built adapters.
        radii_m[name] = max(radii_m[name], body_radius)
        normalized[name] = tuple(boxes)
    return normalized


def _articulation_is_fixed_base(robot: Any) -> bool:
    # Use PhysX's composed articulation property, not the spawn config request.
    return getattr(robot, "is_fixed_base", None) is True


def _root_has_no_native_motion(robot_data: Any, root_index: int, body_count: int) -> bool:
    try:
        linear = _to_numpy(robot_data.body_lin_vel_w)[0]
        angular = _to_numpy(robot_data.body_ang_vel_w)[0]
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    if (
        linear.shape != (body_count, 3)
        or angular.shape != (body_count, 3)
        or not np.isfinite(linear).all()
        or not np.isfinite(angular).all()
    ):
        return False
    return (
        float(np.linalg.norm(linear[root_index])) <= 1e-6
        and float(np.linalg.norm(angular[root_index])) <= 1e-6
    )


def _integrate_world_rotation(
    quaternion_wxyz: Sequence[float], rotation_delta_world_rad: Sequence[float],
) -> np.ndarray:
    rotation_delta = np.asarray(rotation_delta_world_rad, dtype=np.float64)
    angle = float(np.linalg.norm(rotation_delta))
    if not isfinite(angle):
        raise ValueError("angular sweep delta must be finite")
    if angle <= 1e-12:
        return _quat(quaternion_wxyz)
    delta_quaternion = np.concatenate((
        np.asarray([np.cos(0.5 * angle)]),
        rotation_delta * (np.sin(0.5 * angle) / angle),
    ))
    return _quat(_quat_multiply(delta_quaternion, _quat(quaternion_wxyz)))


def _quat_rotation_matrix(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = _quat(quaternion_wxyz)
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float64)


def _cloud_oriented_box_collision_mask(
    points: np.ndarray,
    center_world_m: Sequence[float],
    rotation_world_from_body: np.ndarray,
    lower_body_m: Sequence[float],
    upper_body_m: Sequence[float],
    inflation_m: float,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    center = np.asarray(center_world_m, dtype=np.float64)
    rotation = np.asarray(rotation_world_from_body, dtype=np.float64)
    lower = np.asarray(lower_body_m, dtype=np.float64) - inflation_m
    upper = np.asarray(upper_body_m, dtype=np.float64) + inflation_m
    if (
        center.shape != (3,)
        or rotation.shape != (3, 3)
        or lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(rotation).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not isfinite(float(inflation_m))
        or inflation_m < 0.0
    ):
        return np.ones(len(points), dtype=bool)
    points_body = (points - center) @ rotation
    return np.all((points_body >= lower) & (points_body <= upper), axis=1)


def _cloud_segment_collision_mask(
    points: np.ndarray, start: Sequence[float], end: Sequence[float], radius_m: float,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    segment = b - a
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-16:
        distance_sq = np.sum((points - a) ** 2, axis=1)
    else:
        fraction = np.clip(((points - a) @ segment) / length_sq, 0.0, 1.0)
        closest = a + fraction[:, None] * segment
        distance_sq = np.sum((points - closest) ** 2, axis=1)
    return distance_sq <= radius_m * radius_m


def _sweep_collision_details(
    points: np.ndarray,
    collisions: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
    *,
    collision_kind: str,
    body_child: str,
    body_parent: str | None,
    sample_fraction: float,
    capsule_radius_m: float,
    sweep_margin_m: float,
    between_sample_padding_m: float = 0.0,
) -> dict[str, Any]:
    indices = np.flatnonzero(collisions)
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    segment = b - a
    length_sq = float(np.dot(segment, segment))
    candidates = points[indices]
    if length_sq <= 1e-16:
        closest = np.broadcast_to(a, candidates.shape)
    else:
        fraction = np.clip(((candidates - a) @ segment) / length_sq, 0.0, 1.0)
        closest = a + fraction[:, None] * segment
    distances = np.linalg.norm(candidates - closest, axis=1)
    selected = int(np.argmin(distances))
    distance_m = float(distances[selected])
    required_radius_m = float(capsule_radius_m + sweep_margin_m + between_sample_padding_m)
    return {
        "collision_kind": collision_kind,
        "body_child": body_child,
        "body_parent": body_parent,
        "sample_fraction": float(sample_fraction),
        "obstacle_point_index": int(indices[selected]),
        "obstacle_point_world_m": candidates[selected].tolist(),
        "closest_centerline_point_world_m": closest[selected].tolist(),
        "distance_m": distance_m,
        "capsule_radius_m": float(capsule_radius_m),
        "sweep_margin_m": float(sweep_margin_m),
        "required_radius_m": required_radius_m,
        "signed_clearance_m": distance_m - required_radius_m,
    }


def _optional_finite_number(value: Any) -> float | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.size != 1 or not np.isfinite(array).all():
        return None
    return float(array[0])


def _finite_xyz(value: Any) -> list[float] | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.shape != (3,) or not np.isfinite(array).all():
        return None
    return [float(number) for number in array]


def _grasp_approach_retreat_direction_world(
    target_tcp_pose: Sequence[float], current_tcp_pose: Sequence[float],
) -> list[float] | None:
    """Return the reverse of the remaining public TCP-to-grasp-target displacement."""
    try:
        target = np.asarray(target_tcp_pose, dtype=np.float64)
        current = np.asarray(current_tcp_pose, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if target.shape != (7,) or current.shape != (7,) or not np.isfinite(target).all() or not np.isfinite(current).all():
        return None
    retreat = current[:3] - target[:3]
    distance = float(np.linalg.norm(retreat))
    if not isfinite(distance) or distance <= 1e-6:
        return None
    return (retreat / distance).tolist()


def _public_sweep_collision_diagnostic(
    diagnostic: Any, sensors: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return only diagnostics that satisfy the harness's public result contract."""
    if not isinstance(diagnostic, Mapping):
        return None
    projected = dict(diagnostic)
    obstacle_point = _finite_xyz(projected.get("obstacle_point_world_m"))
    centerline_point = _finite_xyz(projected.get("closest_centerline_point_world_m"))
    if obstacle_point is None or centerline_point is None:
        return None

    obstacle_map = sensors.get("rgbd_obstacle_map")
    obstacle_map = obstacle_map if isinstance(obstacle_map, Mapping) else {}
    projected.update({
        "obstacle_point_world_m": obstacle_point,
        "closest_centerline_point_world_m": centerline_point,
        "map_source_observation_id": obstacle_map.get("source_observation_id"),
        "camera_frame_id": _optional_finite_number(sensors.get("camera_frame_id")),
        "camera_timestamp_s": _optional_finite_number(sensors.get("rgbd_source_timestamp_s")),
        "robot_state_timestamp_s": _optional_finite_number(
            sensors.get("robot_timestamp_s", sensors.get("robot_state_timestamp_s")),
        ),
        "obstacle_pixel_xy": _project_world_point_to_pixel(
            obstacle_point, sensors.get("camera_intrinsics"), sensors.get("camera_extrinsics"),
        ),
    })
    if set(projected) != _PUBLIC_SWEEP_COLLISION_DIAGNOSTIC_FIELDS:
        return None
    if (
        not isinstance(projected["collision_kind"], str)
        or projected["collision_kind"] not in {"robot_link", "held_object"}
    ):
        return None
    if not isinstance(projected["body_child"], str) or not projected["body_child"]:
        return None
    if projected["body_parent"] is not None and not isinstance(projected["body_parent"], str):
        return None
    if type(projected["obstacle_point_index"]) is not int or projected["obstacle_point_index"] < 0:
        return None
    for name in (
        "sample_fraction", "distance_m", "capsule_radius_m", "sweep_margin_m",
        "required_radius_m", "signed_clearance_m",
    ):
        value = _optional_finite_number(projected[name])
        if value is None:
            return None
        projected[name] = value
    if not 0.0 <= projected["sample_fraction"] <= 1.0:
        return None
    if projected["distance_m"] < 0.0 or projected["capsule_radius_m"] <= 0.0:
        return None
    if projected["sweep_margin_m"] <= 0.0 or projected["required_radius_m"] <= 0.0:
        return None
    if projected["signed_clearance_m"] > 0.0:
        return None
    if projected["map_source_observation_id"] is not None and not isinstance(
        projected["map_source_observation_id"], str,
    ):
        return None
    for name in ("camera_frame_id", "camera_timestamp_s", "robot_state_timestamp_s"):
        if projected[name] is not None and not isfinite(float(projected[name])):
            return None
    pixel = projected["obstacle_pixel_xy"]
    if pixel is not None and (
        not isinstance(pixel, (list, tuple)) or len(pixel) != 2
        or any(not isfinite(float(value)) for value in pixel)
    ):
        return None
    return projected


def _observed_rgbd_collision_feedback(
    diagnostic: Any, sensors: Mapping[str, Any], *,
    retreat_direction_world: Sequence[float] | None = None,
) -> str | None:
    """Give the model actionable location feedback for RGB-D-only sweep rejections."""
    if not isinstance(diagnostic, Mapping) or diagnostic.get("kind") != "rgbd_observed_surface":
        return None
    pixel = _project_world_point_to_pixel(
        diagnostic.get("obstacle_point_world_m"),
        sensors.get("camera_intrinsics"), sensors.get("camera_extrinsics"),
    )
    if pixel is None:
        return None
    obstacle_point = np.asarray(diagnostic["obstacle_point_world_m"], dtype=np.float64)
    frame_id = sensors.get("camera_frame_id")
    obstacle_map = sensors.get("rgbd_obstacle_map")
    source_id = obstacle_map.get("source_observation_id") if isinstance(obstacle_map, Mapping) else None
    timestamp = sensors.get("rgbd_source_timestamp_s")
    provenance = []
    if isinstance(frame_id, (int, float)) and not isinstance(frame_id, bool):
        provenance.append(f"camera_frame_id={frame_id:g}")
    if isinstance(source_id, (str, int, float)) and not isinstance(source_id, bool):
        provenance.append(f"map_source_observation_id={source_id}")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and isfinite(timestamp):
        provenance.append(f"camera_timestamp_s={timestamp:.3f}")
    source = f" ({', '.join(provenance)})" if provenance else ""
    preexisting = diagnostic.get("preexisting_overlap") is True
    direction_hint = ""
    approach_retreat_hint = ""
    retreat = _finite_xyz(retreat_direction_world)
    if retreat is not None:
        retreat_vector = np.asarray(retreat, dtype=np.float64)
        retreat_norm = float(np.linalg.norm(retreat_vector))
        if isfinite(retreat_norm) and retreat_norm > 1e-9:
            retreat_vector /= retreat_norm
            try:
                tcp_pose = np.asarray(sensors.get("tcp_pose"), dtype=np.float64)
                if tcp_pose.shape == (7,) and np.isfinite(tcp_pose).all():
                    local_retreat = _quat_rotation_matrix(_quat(tcp_pose[3:])).T @ retreat_vector
                    approach_retreat_hint = (
                        " Axial retreat candidate only (not lateral or verified clear):"
                        f" world_unit={np.round(retreat_vector, 3).tolist()}"
                        f" tcp_local_unit={np.round(local_retreat, 3).tolist()}"
                        f" example_0.01m_retreat_translation_delta_m={np.round(local_retreat * 0.01, 5).tolist()}."
                    )
            except (TypeError, ValueError, ArithmeticError):
                pass
    hint_source = "single-geometry clearance estimate"
    lateral_direction_hint = ""
    world_direction = _finite_xyz(diagnostic.get("separation_direction_world"))
    if world_direction is not None:
        direction = np.asarray(world_direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
    else:
        try:
            tcp_pose = np.asarray(sensors.get("tcp_pose"), dtype=np.float64)
            if tcp_pose.shape == (7,) and np.isfinite(tcp_pose).all():
                direction = np.asarray(tcp_pose[:3], dtype=np.float64) - obstacle_point
                norm = float(np.linalg.norm(direction))
                if norm <= 0.15:
                    hint_source = "approximate direction from the observed obstacle point toward the public TCP center"
                else:
                    direction = np.zeros(3, dtype=np.float64)
        except (TypeError, ValueError, ArithmeticError):
            direction = np.zeros(3, dtype=np.float64)
    if world_direction is not None or hint_source.startswith("approximate"):
        norm = float(np.linalg.norm(direction))
        if np.isfinite(norm) and norm > 1e-9:
            direction /= norm
            direction_text = f" world_unit={np.round(direction, 3).tolist()}"
            try:
                tcp_pose = np.asarray(sensors.get("tcp_pose"), dtype=np.float64)
                if tcp_pose.shape == (7,) and np.isfinite(tcp_pose).all():
                    local_direction = _quat_rotation_matrix(_quat(tcp_pose[3:])).T @ direction
                    direction_text += f" tcp_local_unit={np.round(local_direction, 3).tolist()}"
                    trial_delta = local_direction * 0.01
                    direction_text += (
                        f" example_0.01m_trial_translation_delta_m={np.round(trial_delta, 5).tolist()}"
                    )
                    if retreat is not None:
                        approach_axis = -np.asarray(retreat, dtype=np.float64)
                        lateral_direction = direction - float(np.dot(direction, approach_axis)) * approach_axis
                        lateral_norm = float(np.linalg.norm(lateral_direction))
                        if np.isfinite(lateral_norm) and lateral_norm >= 0.2:
                            lateral_direction /= lateral_norm
                            lateral_local = _quat_rotation_matrix(_quat(tcp_pose[3:])).T @ lateral_direction
                            lateral_direction_hint = (
                                " Cross-approach lateral candidate from one observed point only (unverified):"
                                f" world_unit={np.round(lateral_direction, 3).tolist()}"
                                f" tcp_local_unit={np.round(lateral_local, 3).tolist()}"
                                f" example_0.02m_lateral_trial_translation_delta_m="
                                f"{np.round(lateral_local * 0.02, 5).tolist()}."
                            )
            except (TypeError, ValueError, ArithmeticError):
                pass
            direction_hint = (
                f" Candidate direction ({hint_source}) for this one observed point only;"
                f" it is not a collision-free route:{direction_text}. The normal full RGB-D sweep, IK, and"
                " force checks still decide whether any proposed motion is allowed."
            )
    if lateral_direction_hint:
        # Keep the task-relevant lateral candidate inside the public feedback text budget.
        direction_hint = ""
    geometric_hint = (
        f" Public-RGB-D obstacle point world_m={np.round(obstacle_point, 4).tolist()}."
        f"{approach_retreat_hint}{direction_hint}{lateral_direction_hint}"
    )
    clearance_note = (
        " Current pose is inside the conservative sweep envelope; only measurable clearance-increasing motion "
        "from every such point is allowed. This step was not sent."
        if preexisting else ""
    )
    return (
        "RGB-D sweep blocked this local step (not sent): predicted robot-geometry overlap near image pixel "
        f"({pixel[0]:.1f}, {pixel[1]:.1f}){source}; this is not proof of physical contact. Do not repeat this grasp path."
        f"{geometric_hint}{clearance_note} Fresh RGB-D, sweep, IK, and force checks still govern any new move."
    )


def _progress_numbers(value: Any) -> list[float] | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if not array.size or not np.isfinite(array).all():
        return None
    return [float(number) for number in array]


def _progress_booleans(value: Any) -> list[bool] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if any(type(item) is not bool for item in value):
        return None
    return list(value)


def _project_world_point_to_pixel(
    point_world_m: Sequence[float], intrinsics: Any, extrinsics: Any,
) -> list[float] | None:
    try:
        point = _vector(point_world_m, 3, "collision point")
        k = np.asarray(intrinsics, dtype=np.float64)
        t_world_camera = np.asarray(extrinsics, dtype=np.float64)
        if k.shape != (3, 3) or t_world_camera.shape != (4, 4):
            return None
        camera_point = np.linalg.inv(t_world_camera) @ np.asarray((*point, 1.0))
        if not np.isfinite(camera_point).all() or camera_point[2] <= 0.0:
            return None
        projected = k @ camera_point[:3]
        if not np.isfinite(projected).all() or projected[2] <= 0.0:
            return None
        return [float(projected[0] / projected[2]), float(projected[1] / projected[2])]
    except (TypeError, ValueError, ArithmeticError, np.linalg.LinAlgError):
        return None


def _validated_contact_pairs(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) > 8:
        return None
    result = []
    for pair in value:
        if not isinstance(pair, Mapping) or pair.get("moving_role") not in {"gripper", "held_object"}:
            return None
        if not isinstance(pair.get("moving_feature_ref"), str) or not isinstance(pair.get("reference_feature_ref"), str):
            return None
        try:
            activation = _vector(pair.get("activation_center_world_m"), 3, "contact activation center")
            activation_radius = float(pair.get("activation_radius_m"))
            point_tolerance = float(pair.get("point_match_tolerance_m"))
        except (TypeError, ValueError):
            return None
        if (
            not isfinite(activation_radius) or not 0.0 < activation_radius <= 0.05
            or not isfinite(point_tolerance) or not 0.0 < point_tolerance <= 0.01
        ):
            return None
        sites = pair.get("contact_sites")
        if not isinstance(sites, Sequence) or isinstance(sites, (str, bytes)) or not 1 <= len(sites) <= 2:
            return None
        normalized_sites = []
        for site in sites:
            if not isinstance(site, Mapping):
                return None
            try:
                position = _vector(site.get("position_world_m"), 3, "contact site")
                points = np.asarray(site.get("surface_points_world_m"), dtype=float)
            except (TypeError, ValueError):
                return None
            if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 512 or not np.isfinite(points).all():
                return None
            normalized_sites.append({"position_world_m": position, "surface_points_world_m": points})
        result.append({
            "moving_feature_ref": pair["moving_feature_ref"],
            "reference_feature_ref": pair["reference_feature_ref"],
            "moving_role": pair["moving_role"],
            "activation_center_world_m": activation,
            "activation_radius_m": activation_radius,
            "point_match_tolerance_m": point_tolerance,
            "contact_sites": tuple(normalized_sites),
        })
    return tuple(result)


def _contact_surface_point_mask(points: np.ndarray, pair: Mapping[str, Any]) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    tolerance = float(pair["point_match_tolerance_m"])
    for site in pair["contact_sites"]:
        samples = site["surface_points_world_m"]
        distances = np.linalg.norm(points[:, None, :] - samples[None, :, :], axis=2)
        result |= np.min(distances, axis=1) <= tolerance
    return result


def _moving_feature_at_contact(position: np.ndarray, pair: Mapping[str, Any]) -> bool:
    delta = position - pair["activation_center_world_m"]
    return float(np.linalg.norm(delta)) <= float(pair["activation_radius_m"])


def _interpolate_pose(start: Sequence[float], end: Sequence[float], fraction: float) -> tuple[float, ...]:
    position = (1.0 - fraction) * np.asarray(start[:3]) + fraction * np.asarray(end[:3])
    rotation = _rotation_vector(end[3:], start[3:]) * fraction
    angle = float(np.linalg.norm(rotation))
    if angle <= 1e-12:
        delta_q = np.array((1.0, 0.0, 0.0, 0.0))
    else:
        delta_q = np.concatenate(((np.cos(angle / 2.0),), rotation * (np.sin(angle / 2.0) / angle)))
    quaternion = _quat(_quat_multiply(delta_q, _quat(start[3:])))
    return (*position.tolist(), *quaternion.tolist())


def _compose_pose(parent: Sequence[float], child: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    quaternion = _quat(parent[3:])
    local_position = np.asarray(child[:3], dtype=float)
    qv = quaternion[1:]
    rotated = local_position + 2.0 * np.cross(qv, np.cross(qv, local_position) + quaternion[0] * local_position)
    position = np.asarray(parent[:3], dtype=float) + rotated
    orientation = _quat(_quat_multiply(quaternion, _quat(child[3:])))
    return position, orientation


def _rotation_vector(target: Sequence[float], current: Sequence[float]) -> np.ndarray:
    q = _quat_multiply(_quat(target), np.array((current[0], -current[1], -current[2], -current[3])))
    q = _quat(q)
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm <= 1e-10:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(vector_norm, max(float(q[0]), 0.0)))
    return q[1:] * (angle / vector_norm)


def _normalized_action(
    current: Sequence[float],
    target: Sequence[float],
    *,
    dt_s: float,
    max_linear_step_m: float,
    max_angular_step_rad: float,
    position_scales: Sequence[float],
    rotation_scales: Sequence[float],
) -> tuple[float, ...]:
    position_error = np.asarray(target[:3], dtype=float) - np.asarray(current[:3], dtype=float)
    position_delta = _limit_norm(position_error, max_linear_step_m)
    rotation_delta = _limit_norm(_rotation_vector(target[3:], current[3:]), max_angular_step_rad)
    action = np.concatenate((position_delta / np.asarray(position_scales), rotation_delta / np.asarray(rotation_scales)))
    return tuple(float(value) for value in np.clip(action, -1.0, 1.0))


def _commanded_tcp_pose(
    current: Sequence[float], action: Sequence[float], limits: ExecutorLimits,
) -> tuple[float, ...]:
    normalized = np.asarray(action, dtype=float)
    position_delta = normalized[:3] * np.asarray(limits.position_action_scale_m, dtype=float)
    rotation_delta = normalized[3:] * np.asarray(limits.rotation_action_scale_rad, dtype=float)
    angle = float(np.linalg.norm(rotation_delta))
    if angle <= 1e-12:
        delta_quaternion = np.array((1.0, 0.0, 0.0, 0.0))
    else:
        delta_quaternion = np.concatenate(((np.cos(angle / 2.0),), rotation_delta * (np.sin(angle / 2.0) / angle)))
    current_quaternion = _quat(current[3:])
    target_quaternion = _quat(_quat_multiply(delta_quaternion, current_quaternion))
    target_position = np.asarray(current[:3], dtype=float) + position_delta
    return (*target_position.tolist(), *target_quaternion.tolist())


def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum or norm <= 1e-12 else vector * (maximum / norm)


def _settled(sensors: Mapping[str, Any], limits: ExecutorLimits) -> bool:
    velocity = sensors.get("tcp_velocity")
    if velocity is None:
        return False
    values = _vector(velocity, 6, "tcp_velocity")
    return (
        float(np.linalg.norm(values[:3])) <= limits.settled_linear_speed_mps
        and float(np.linalg.norm(values[3:])) <= limits.settled_angular_speed_radps
    )


def _motion_step_budget(
    start_tcp_pose: Sequence[float],
    target_tcp_pose: Sequence[float],
    motion_regime: str,
    limits: ExecutorLimits,
) -> int:
    """Bound a motion attempt by speed-limited travel time plus settling samples."""
    return min(limits.max_steps, _required_motion_step_budget(
        start_tcp_pose, target_tcp_pose, motion_regime, limits,
    ))


def _required_motion_step_budget(
    start_tcp_pose: Sequence[float],
    target_tcp_pose: Sequence[float],
    motion_regime: str,
    limits: ExecutorLimits,
) -> int:
    position_distance = max(
        0.0,
        float(np.linalg.norm(
            np.asarray(target_tcp_pose[:3], dtype=float)
            - np.asarray(start_tcp_pose[:3], dtype=float)
        )) - limits.position_tolerance_m,
    )
    angular_distance = max(
        0.0,
        float(np.linalg.norm(_rotation_vector(target_tcp_pose[3:], start_tcp_pose[3:])))
        - limits.orientation_tolerance_rad,
    )
    linear_speed = (
        limits.max_contact_speed_mps
        if motion_regime == "contact"
        else limits.max_linear_speed_mps
    )
    travel_duration_s = max(
        position_distance / linear_speed,
        angular_distance / limits.max_angular_speed_radps,
    )
    travel_steps = ceil(travel_duration_s / limits.control_dt_s)
    settling_steps = ceil(limits.minimum_stable_duration_s / limits.control_dt_s) + 1
    return max(1, travel_steps + settling_steps)


def _result(
    status: str,
    terminal: bool,
    feedback: str,
    resolved: ResolvedGoal | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "terminal": terminal, "feedback": feedback[:1024]}
    if resolved is not None:
        result["resolved_goal"] = resolved.as_public_dict()
        if resolved.geometry_diagnostics:
            result["geometry_diagnostics"] = [dict(item) for item in resolved.geometry_diagnostics]
    return result


__all__ = ["AssemblyDriver", "AssemblyExecutor", "ExecutorLimits", "FactoryAssemblyDriver"]
