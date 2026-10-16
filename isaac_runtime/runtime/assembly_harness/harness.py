"""Bounded model-to-goal loop; physical goal resolution stays behind an adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import acos, isfinite, sqrt
from typing import Any

from .agent import (
    MAX_CALLS,
    _json_value,
    compile_grasp_tool_call,
    parse_tool_call,
    project_observation,
)
from .checkpoint import CHECKPOINT_STATUSES
from .connector import AssemblyConnector, GoalResolutionError, bind_contact_surface_points
from .interfaces import GoalExecutor


_SUCCESS = {"completed", "succeeded"}
_CONTACT_GEOMETRY_REJECTIONS = {"missing_contact_geometry", "contact_surface_missing"}
_REPEATABLE_ZERO_STEP_REJECTIONS = _CONTACT_GEOMETRY_REJECTIONS | {
    "inconsistent_constraints", "uncertainty_too_high", "gripper_state_unavailable",
}
_SWEEP_COLLISION_DIAGNOSTIC_FIELDS = {
    "collision_kind", "body_child", "body_parent", "sample_fraction",
    "obstacle_point_index", "obstacle_point_world_m", "closest_centerline_point_world_m",
    "distance_m", "capsule_radius_m", "sweep_margin_m", "required_radius_m",
    "signed_clearance_m", "map_source_observation_id", "camera_frame_id",
    "camera_timestamp_s", "robot_state_timestamp_s", "obstacle_pixel_xy",
}


@dataclass(frozen=True)
class AssemblyHarnessResult:
    status: str
    exit_code: int
    calls: int
    completed: bool
    history: tuple[dict[str, Any], ...]
    error: str | None = None
    private_exception: BaseException | None = field(default=None, repr=False, compare=False)


class AssemblyHarnessLoop:
    """Ask the model for each goal; never infer or enforce task-stage ordering.

    The runtime adapter implements ``execute_goal(goal, observation)`` and is responsible for
    online goal resolution and the executor's native release/safety protections.
    A terminal action result ends the loop without another model action. The
    completion callback returns only the independent evaluator's boolean.
    """

    def __init__(
        self,
        agent: Any,
        executor: GoalExecutor,
        *,
        task_goal: str,
        observation_callback: Callable[[], Mapping[str, Any]],
        check_task_callback: Callable[[], bool],
        trace_callback: Callable[[dict[str, Any]], None],
        checkpoint_callback: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
        max_calls: int = MAX_CALLS,
        observe_only: bool = False,
    ) -> None:
        if not callable(getattr(agent, "next_tool_call", None)):
            raise TypeError("agent must expose next_tool_call()")
        if not callable(getattr(executor, "execute_goal", None)):
            raise TypeError("executor must expose execute_goal(goal, observation)")
        if not isinstance(task_goal, str) or not task_goal.strip():
            raise ValueError("task_goal must be non-empty natural-language text")
        if not callable(observation_callback) or not callable(check_task_callback) or not callable(trace_callback):
            raise TypeError("observation, check_task, and trace callbacks must be callable")
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise TypeError("checkpoint_callback must be callable when provided")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 1 <= max_calls <= MAX_CALLS:
            raise ValueError(f"max_calls must be in [1, {MAX_CALLS}]")
        self.agent = agent
        self.executor = executor
        self.task_goal = task_goal.strip()
        self.observation_callback = observation_callback
        self.check_task_callback = check_task_callback
        self.trace_callback = trace_callback
        self.checkpoint_callback = checkpoint_callback
        self.max_calls = max_calls
        if type(observe_only) is not bool:
            raise TypeError("observe_only must be a boolean")
        self.observe_only = observe_only
        self.history: list[dict[str, Any]] = []
        self.calls = 0
        self.completed = False
        self._started = False
        self._last_model_frame_id: int | str | None = None
        self._last_model_frame_timestamp_s: float | None = None
        self._last_zero_step_rejection: tuple[
            Mapping[str, Any], Mapping[str, Any], str,
        ] | None = None
        self._last_zero_step_rejection_tool: str | None = None
        self._last_grasp_collision_rejection: tuple[
            Mapping[str, Any], Mapping[str, Any],
        ] | None = None
        self._blocked_tcp_adjustments: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    def run(self) -> AssemblyHarnessResult:
        if self._started:
            raise RuntimeError("AssemblyHarnessLoop instances can run only once")
        self._started = True
        try:
            observation = self._observe()
            while self.calls < self.max_calls:
                self.calls += 1
                call = self.agent.next_tool_call(
                    self.task_goal, observation, self.history, completed=self.completed,
                )
                public_observation = project_observation(observation)[0]
                allowed_tools = {
                    "observe": {}, "grasp": {}, "move_tcp": {}, "execute_goal": {}, "stop": {},
                } if self.observe_only else None
                call = parse_tool_call(
                    call, observation=public_observation, completed=self.completed,
                    allowed_tools=allowed_tools,
                )
                tool = call["tool"]
                dispatch_goal = (
                    compile_grasp_tool_call(call, observation=public_observation)
                    if tool == "grasp" else call
                )

                if tool == "stop":
                    self._record(call, {"status": "stopped"})
                    return self._finish("stopped", error="model stopped before evaluator-confirmed completion")
                if tool == "finish":
                    self._record(call, {"status": "finished", "completed": self.completed})
                    return self._finish("success" if self.completed else "failed",
                                        error=None if self.completed else "finish requires completed=true")
                if tool == "observe":
                    observation = self._observe()
                    self._record(call, {"status": "completed", "observation": observation})
                    continue
                if tool == "check_task":
                    completed = self.check_task_callback()
                    if type(completed) is not bool:
                        raise TypeError("check_task callback must return only a boolean")
                    self.completed = completed
                    observation = self._observe()
                    self._record(call, {"completed": completed, "observation": observation})
                    continue

                if self.observe_only:
                    proposal_result = {
                        "status": "proposal_only",
                        "feedback": "observe-only mode validated this action proposal without executing it",
                    }
                    if tool == "grasp":
                        proposal_result["compiled_goal"] = dispatch_goal
                    self._record(call, proposal_result)
                    return self._finish("proposal_only", error="observe-only proposal was not executed")

                if self.completed:
                    self._record(call, {"status": "invalid_action"})
                    return self._finish("failed", error="goal execution is not allowed after task completion")

                if self._is_repeated_zero_step_rejection(dispatch_goal, observation):
                    if tool == "move_tcp":
                        feedback = (
                            "This TCP-local adjustment repeats a prior collision-blocked delta while the measured "
                            "TCP has not moved. It was not re-executed; choose a different bounded adjustment or "
                            "stop if no safe recovery is supported."
                        )
                    elif dispatch_goal.get("motion_regime") == "grasp":
                        feedback = (
                            "This grasp goal rebinds to the same physical target as a prior collision-blocked grasp, "
                            "and the measured TCP has not moved since that rejection. It was not re-executed; choose "
                            "a different bounded recovery move or stop."
                        )
                    else:
                        feedback = (
                            "This goal geometrically rebinds to the same target as a prior zero-motion-step "
                            "rejection, and the fresh public observation does not make it executable. It was not "
                            "re-executed; choose a different task-matching target or operation, obtain genuinely "
                            "new evidence that resolves the rejection, or stop."
                        )
                    repeated_result = {
                        "status": "repeated_no_progress",
                        "terminal": False,
                        "feedback": feedback,
                        "execution_diagnostics": {
                            "driver_step_count": 0,
                            "ik_call_count": 0,
                            "sweep_check_count": 0,
                        },
                        "observation": project_observation(observation)[0],
                    }
                    if tool == "grasp":
                        repeated_result["compiled_goal"] = dispatch_goal
                    self._record(call, repeated_result)
                    continue

                submitted_observation = observation
                if tool == "move_tcp":
                    execute_tcp_delta = getattr(self.executor, "execute_tcp_delta", None)
                    if not callable(execute_tcp_delta):
                        raise TypeError("executor must expose execute_tcp_delta(call, observation) for move_tcp")
                    action_result = execute_tcp_delta(dispatch_goal, observation)
                else:
                    action_result = self.executor.execute_goal(dispatch_goal, observation)
                result = self._project_action_result(action_result, dispatch_goal)
                if tool == "grasp":
                    result["compiled_goal"] = dispatch_goal
                if (
                    self.checkpoint_callback is not None
                    and result["status"] in CHECKPOINT_STATUSES
                ):
                    self.checkpoint_callback(call, result)
                if getattr(self.executor, "official_success_after_harness_action", False) is True:
                    self._record(call, result)
                    return self._finish(
                        "official_success_observed",
                        error="episode stopped after private evaluator observed task success",
                    )
                diagnostics = result.get("execution_diagnostics", {})
                gripper_geometry_rejection = (
                    tool == "grasp"
                    and result["status"] == "missing_geometry"
                    and isinstance(diagnostics, Mapping)
                    and diagnostics.get("ik_call_count") == 0
                    and diagnostics.get("sweep_check_count") == 0
                )
                grasp_collision_rejection = (
                    dispatch_goal.get("motion_regime") == "grasp"
                    and result["status"] == "collision_blocked"
                    and isinstance(diagnostics, Mapping)
                    and diagnostics.get("sweep_check_count", 0) >= 1
                )
                move_tcp_collision_rejection = (
                    tool == "move_tcp"
                    and result["status"] == "collision_blocked"
                    and isinstance(diagnostics, Mapping)
                    and diagnostics.get("sweep_check_count", 0) >= 1
                )
                if grasp_collision_rejection:
                    self._last_grasp_collision_rejection = (dict(dispatch_goal), submitted_observation)
                if move_tcp_collision_rejection:
                    self._blocked_tcp_adjustments.append((dict(dispatch_goal), submitted_observation))
                    self._blocked_tcp_adjustments = self._blocked_tcp_adjustments[-8:]
                if (
                    (
                        result["status"] in _REPEATABLE_ZERO_STEP_REJECTIONS
                        and isinstance(diagnostics, Mapping)
                        and diagnostics.get("driver_step_count") == 0
                    )
                    or gripper_geometry_rejection
                    or grasp_collision_rejection
                ):
                    self._last_zero_step_rejection = (
                        dict(dispatch_goal), submitted_observation, result["status"],
                    )
                    self._last_zero_step_rejection_tool = tool
                else:
                    self._last_zero_step_rejection = None
                    self._last_zero_step_rejection_tool = None
                observation = self._observe()
                if observation["observation_id"] == dispatch_goal["observation_id"]:
                    raise ValueError("goal execution must be followed by a fresh observation_id")
                result["observation"] = project_observation(observation)[0]
                self._record(call, result)
                if result["terminal"]:
                    return self._finish("safety_stop", error="executor marked this result terminal")

            return self._finish("max_calls", error="maximum model tool-call count reached")
        except Exception as exc:
            return self._finish("failed", error="harness operation failed", private_exception=exc)

    def _is_repeated_zero_step_rejection(
        self, goal: Mapping[str, Any], observation: Mapping[str, Any],
    ) -> bool:
        if goal.get("tool") == "move_tcp":
            retained_adjustments = []
            repeated_adjustment = False
            for previous_call, previous_observation in self._blocked_tcp_adjustments:
                if _tcp_pose_changed(previous_observation, observation, self.executor):
                    continue
                retained_adjustments.append((previous_call, previous_observation))
                if _same_tcp_adjustment(previous_call, goal):
                    repeated_adjustment = True
            self._blocked_tcp_adjustments = retained_adjustments
            if repeated_adjustment:
                return True

        grasp_rejection = self._last_grasp_collision_rejection
        if goal.get("motion_regime") == "grasp" and grasp_rejection is not None:
            previous_goal, previous_observation = grasp_rejection
            if _tcp_pose_changed(previous_observation, observation, self.executor):
                self._last_grasp_collision_rejection = None
            else:
                try:
                    rebound = AssemblyConnector().rebind_goal_features(
                        previous_goal, previous_observation, observation,
                    )
                except (GoalResolutionError, TypeError, ValueError, ArithmeticError):
                    rebound = None
                if rebound is not None and _constraints_key(rebound) == _constraints_key(goal):
                    return True

        previous = self._last_zero_step_rejection
        if previous is None:
            return False
        previous_goal, previous_observation, previous_status = previous
        previous_grasp_geometry_rejection = (
            previous_status == "missing_geometry"
            and (
                self._last_zero_step_rejection_tool == "grasp"
                or previous_goal.get("tool") == "grasp"
                or previous_goal.get("motion_regime") == "grasp"
            )
        )
        previous_grasp_collision_rejection = (
            previous_status == "collision_blocked"
            and (
                self._last_zero_step_rejection_tool == "grasp"
                or previous_goal.get("tool") == "grasp"
                or previous_goal.get("motion_regime") == "grasp"
            )
        )
        if (
            (
                previous_status not in _REPEATABLE_ZERO_STEP_REJECTIONS
                and not previous_grasp_geometry_rejection
                and not previous_grasp_collision_rejection
            )
            or goal.get("motion_regime") != previous_goal.get("motion_regime")
            or (
                previous_status in _CONTACT_GEOMETRY_REJECTIONS
                and goal.get("motion_regime") not in {"grasp", "contact"}
            )
            or (
                previous_grasp_geometry_rejection
                and goal.get("motion_regime") != "grasp"
            )
        ):
            return False
        try:
            rebound = AssemblyConnector().rebind_goal_features(
                previous_goal, previous_observation, observation,
            )
        except (GoalResolutionError, TypeError, ValueError, ArithmeticError):
            return False

        if _constraints_key(rebound) != _constraints_key(goal):
            return False

        if previous_status == "gripper_state_unavailable":
            return True

        def has_support(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
            try:
                resolved = AssemblyConnector().resolve(candidate, current)
                if resolved.aperture_entry_supported:
                    return True
                obstacle_map = current.get("robot_state", {}).get("sensors", {}).get("rgbd_obstacle_map", {})
                if not resolved.allowed_contact_pairs or obstacle_map.get("valid") is not True:
                    return False
                return bool(bind_contact_surface_points(resolved.allowed_contact_pairs, obstacle_map.get("points_m", [])))
            except (GoalResolutionError, TypeError, ValueError, ArithmeticError):
                return False

        if previous_status in _CONTACT_GEOMETRY_REJECTIONS:
            # New measured support permits reconsideration; new IDs alone do not.
            return not (has_support(goal, observation) and not has_support(previous_goal, previous_observation))
        if previous_grasp_geometry_rejection:
            # A gripper-only failure is repeatable until current geometry gains support;
            # re-binding the same physical grasp to a new feature ID is not new evidence.
            return not (has_support(goal, observation) and not has_support(previous_goal, previous_observation))
        if previous_grasp_collision_rejection:
            # Gripper opening can consume driver steps without changing the TCP. A genuinely
            # moved TCP is new execution state; fresh IDs alone are not.
            return not _tcp_pose_changed(previous_observation, observation, self.executor)
        if previous_status == "uncertainty_too_high":
            limits = _executor_limits(self.executor)
            if limits is None:
                return True
            try:
                resolved = AssemblyConnector().resolve(goal, observation)
            except (GoalResolutionError, TypeError, ValueError, ArithmeticError):
                return True
            return not (
                resolved.uncertainty_position_m <= limits.max_goal_uncertainty_m
                and resolved.uncertainty_axis_rad <= limits.max_goal_axis_uncertainty_rad
            )
        try:
            AssemblyConnector().resolve(goal, observation)
        except (GoalResolutionError, TypeError, ValueError, ArithmeticError):
            return True
        return False

    def _observe(self) -> Mapping[str, Any]:
        observation = self.observation_callback()
        if not isinstance(observation, Mapping):
            raise TypeError("observation callback must return a public mapping")
        projected, _ = project_observation(observation)
        sequence = projected.get("frame_sequence")
        if sequence:
            latest = sequence[-1]
            frame_id = latest["frame_id"]
            timestamp_s = latest["timestamp_s"]
            if self._last_model_frame_timestamp_s is not None and (
                timestamp_s <= self._last_model_frame_timestamp_s
                or frame_id == self._last_model_frame_id
            ):
                raise ValueError("observe_for_model must advance to a newer camera acquisition")
            self._last_model_frame_id = frame_id
            self._last_model_frame_timestamp_s = timestamp_s
        return projected

    def _project_action_result(self, result: Any, call: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError("executor execute_goal() must return a mapping")
        status = result.get("status")
        if not isinstance(status, str) or not status.strip() or len(status) > 64:
            raise ValueError("executor result requires a short public status")
        terminal = result.get("terminal")
        if type(terminal) is not bool:
            raise ValueError("executor result requires a boolean terminal safety marker")
        if terminal and status in _SUCCESS:
            raise ValueError("a terminal safety result cannot report action success")

        projected: dict[str, Any] = {"status": status, "terminal": terminal}
        feedback = result.get("feedback")
        if feedback is not None:
            if not isinstance(feedback, str) or len(feedback) > 1024:
                raise ValueError("executor feedback must be bounded public text")
            projected["feedback"] = feedback
        if "execution_diagnostics" in result:
            diagnostics = result["execution_diagnostics"]
            expected_diagnostics = {
                "driver_step_count", "ik_call_count", "sweep_check_count",
            }
            if not isinstance(diagnostics, Mapping) or set(diagnostics) != expected_diagnostics:
                raise ValueError(
                    f"execution_diagnostics must contain exactly {sorted(expected_diagnostics)}"
                )
            for name, count in diagnostics.items():
                if type(count) is not int or count < 0:
                    raise ValueError(f"execution_diagnostics.{name} must be a non-negative integer")
            projected["execution_diagnostics"] = dict(diagnostics)
        if "geometry_diagnostics" in result:
            geometry = result["geometry_diagnostics"]
            fields = {
                "moving_feature_ref", "reference_feature_ref", "moving_source_frame", "reference_source_frame",
                "frame", "length_unit", "time_unit", "pose_convention", "observation_id",
                "camera_observation_id", "camera_timestamp_s", "robot_timestamp_s",
                "moving_position_world_m", "reference_position_world_m", "goal_point_world_m",
                "tcp_to_moving_feature_pose", "reference_normal_world", "surface_sample_count",
                "moving_axis_world", "moving_axis_kind", "aperture_rim_supported", "aperture_entry_supported",
                "nearest_surface_sample_distance_m", "surface_support_radius_m", "reference_semantics",
                "surface_support", "required_precondition",
            }
            if (not isinstance(geometry, list) or not 1 <= len(geometry) <= 8
                    or any(not isinstance(item, Mapping) or set(item) != fields for item in geometry)):
                raise ValueError("geometry_diagnostics has an invalid public shape")
            projected["geometry_diagnostics"] = _json_value(geometry, "geometry_diagnostics")
        target_tcp_pose = result.get("target_tcp_pose")
        if call.get("tool") == "move_tcp":
            if target_tcp_pose is not None:
                if not isinstance(target_tcp_pose, Mapping) or set(target_tcp_pose) != {
                    "position_m", "quaternion_wxyz",
                }:
                    raise ValueError("move_tcp target_tcp_pose has an invalid public shape")
                position = target_tcp_pose["position_m"]
                quaternion = target_tcp_pose["quaternion_wxyz"]
                if (
                    not isinstance(position, (list, tuple)) or len(position) != 3
                    or not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4
                ):
                    raise ValueError("move_tcp target_tcp_pose must contain xyz and wxyz")
                for name, values in (("position_m", position), ("quaternion_wxyz", quaternion)):
                    if any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        or not isfinite(value) for value in values
                    ):
                        raise ValueError(f"move_tcp target_tcp_pose.{name} must be finite numeric values")
                quaternion_norm = sum(value * value for value in quaternion) ** 0.5
                if abs(quaternion_norm - 1.0) > 1e-4:
                    raise ValueError("move_tcp target TCP quaternion must be unit length")
                projected["target_tcp_pose"] = _json_value(dict(target_tcp_pose), "target_tcp_pose")
            elif status in _SUCCESS:
                raise ValueError("successful move_tcp execution must return its target TCP pose")
        elif target_tcp_pose is not None:
            raise ValueError("target_tcp_pose is only valid for move_tcp results")
        collision = result.get("sweep_collision_diagnostic")
        if collision is not None:
            if status != "collision_blocked":
                raise ValueError("sweep collision diagnostics are only valid for collision_blocked results")
            diagnostics = result.get("execution_diagnostics")
            if not isinstance(diagnostics, Mapping) or diagnostics.get("sweep_check_count", 0) < 1:
                raise ValueError("sweep collision diagnostics require a recorded sweep check")
            if not isinstance(collision, Mapping) or set(collision) != _SWEEP_COLLISION_DIAGNOSTIC_FIELDS:
                raise ValueError("sweep_collision_diagnostic has an invalid public shape")
            if collision["collision_kind"] not in {"robot_link", "held_object"}:
                raise ValueError("sweep collision kind is invalid")
            if not isinstance(collision["body_child"], str) or not collision["body_child"]:
                raise ValueError("sweep collision child body is invalid")
            if collision["body_parent"] is not None and not isinstance(collision["body_parent"], str):
                raise ValueError("sweep collision parent body is invalid")
            if type(collision["obstacle_point_index"]) is not int or collision["obstacle_point_index"] < 0:
                raise ValueError("sweep collision point index is invalid")
            for name in (
                "sample_fraction", "distance_m", "capsule_radius_m", "sweep_margin_m",
                "required_radius_m", "signed_clearance_m",
            ):
                value = collision[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                    raise ValueError(f"sweep collision {name} must be finite")
            if not 0.0 <= collision["sample_fraction"] <= 1.0:
                raise ValueError("sweep collision sample fraction is outside [0, 1]")
            if collision["distance_m"] < 0.0 or collision["capsule_radius_m"] <= 0.0:
                raise ValueError("sweep collision distance and capsule radius are invalid")
            if collision["sweep_margin_m"] <= 0.0 or collision["required_radius_m"] <= 0.0:
                raise ValueError("sweep collision margin and required radius are invalid")
            if collision["signed_clearance_m"] > 0.0:
                raise ValueError("sweep collision diagnostic must identify a colliding sample")
            for name in ("obstacle_point_world_m", "closest_centerline_point_world_m"):
                vector = collision[name]
                if not isinstance(vector, (list, tuple)) or len(vector) != 3:
                    raise ValueError(f"sweep collision {name} must contain xyz")
            if collision["map_source_observation_id"] is not None and not isinstance(
                collision["map_source_observation_id"], str,
            ):
                raise ValueError("sweep collision map source observation id is invalid")
            for name in ("camera_frame_id", "camera_timestamp_s", "robot_state_timestamp_s"):
                value = collision[name]
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
                ):
                    raise ValueError(f"sweep collision {name} must be finite or null")
            pixel = collision["obstacle_pixel_xy"]
            if pixel is not None and (
                not isinstance(pixel, (list, tuple)) or len(pixel) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
                       for value in pixel)
            ):
                raise ValueError("sweep collision pixel projection must contain finite uv or null")
            projected["sweep_collision_diagnostic"] = _json_value(
                dict(collision), "sweep_collision_diagnostic",
            )
        resolution = result.get("resolved_goal")
        if status in _SUCCESS and call.get("tool") != "move_tcp" and not isinstance(resolution, Mapping):
            raise ValueError("successful goal execution must return its public resolved_goal")
        if resolution is not None:
            if not isinstance(resolution, Mapping):
                raise ValueError("resolved_goal must be a public mapping when present")
            expected = {
                "observation_id", "feature_refs", "constraints", "target_frame",
                "target_geometry", "summary",
            }
            if set(resolution) != expected:
                raise ValueError(f"resolved_goal must contain exactly {sorted(expected)}")
            if resolution["observation_id"] != call["observation_id"]:
                raise ValueError("resolved_goal must cite the submitted observation_id")
            refs = resolution.get("feature_refs")
            submitted_refs = {
                ref for item in call["constraints"]
                for ref in (item["moving_feature"], item["reference_feature"])
            }
            if (not isinstance(refs, list) or not refs
                    or any(not isinstance(ref, str) for ref in refs)
                    or len(refs) != len(set(refs)) or not set(refs) <= submitted_refs):
                raise ValueError("resolved_goal.feature_refs must be a non-empty subset of submitted refs")
            if resolution["target_frame"] != "world":
                raise ValueError("resolved_goal.target_frame must be world")
            if not isinstance(resolution["summary"], str) or len(resolution["summary"]) > 1024:
                raise ValueError("resolved_goal.summary must be bounded public text")
            geometry = resolution["target_geometry"]
            if not isinstance(geometry, Mapping) or set(geometry) != {"tcp_pose"}:
                raise ValueError("resolved_goal.target_geometry must contain connector tcp_pose")
            pose = geometry["tcp_pose"]
            if not isinstance(pose, Mapping) or set(pose) != {"position_m", "quaternion_wxyz"}:
                raise ValueError("resolved_goal.target_geometry.tcp_pose has an invalid connector shape")
            position, quaternion = pose["position_m"], pose["quaternion_wxyz"]
            if not isinstance(position, (list, tuple)) or len(position) != 3:
                raise ValueError("resolved_goal target TCP position must contain three values")
            if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
                raise ValueError("resolved_goal target TCP quaternion must contain four values")
            _json_value(position, "resolved_goal.target_geometry.tcp_pose.position_m")
            _json_value(quaternion, "resolved_goal.target_geometry.tcp_pose.quaternion_wxyz")
            constraints = resolution["constraints"]
            if not isinstance(constraints, list) or not constraints:
                raise ValueError("resolved_goal.constraints must be a non-empty relation list")
            submitted_constraints = {
                (item["relation"], item["moving_feature"], item["reference_feature"])
                for item in call["constraints"]
            }
            for constraint in constraints:
                if (not isinstance(constraint, Mapping) or set(constraint) != {
                        "relation", "moving_feature", "reference_feature"}):
                    raise ValueError("resolved_goal constraint has an invalid connector shape")
                key = (constraint["relation"], constraint["moving_feature"], constraint["reference_feature"])
                if key not in submitted_constraints:
                    raise ValueError("resolved_goal constraints must match submitted relations and refs")
            projected["resolved_goal"] = _json_value(dict(resolution), "resolved_goal")
        return projected

    def _record(self, call: Mapping[str, Any], result: dict[str, Any]) -> None:
        public_result = {key: value for key, value in result.items() if key != "terminal"}
        if isinstance(public_result.get("observation"), Mapping):
            public_result["observation"] = _trace_observation(public_result["observation"])
        event = {"call": dict(call), "result": public_result}
        self.history.append(event)
        self.trace_callback({"event": "tool_result", **event})

    def _finish(
        self,
        status: str,
        *,
        error: str | None = None,
        private_exception: BaseException | None = None,
    ) -> AssemblyHarnessResult:
        success = status == "success" and self.completed
        result = AssemblyHarnessResult(
            status="success" if success else status,
            exit_code=0 if success else 1,
            calls=self.calls,
            completed=self.completed,
            history=tuple(dict(item) for item in self.history),
            error=None if success else error,
            private_exception=private_exception,
        )
        try:
            self.trace_callback({
                "event": "run_finished", "status": result.status, "exit_code": result.exit_code,
                "completed": result.completed, "calls": result.calls,
            })
        except Exception as exc:
            return AssemblyHarnessResult(
                status="failed", exit_code=1, calls=result.calls, completed=result.completed,
                history=result.history, error="trace callback failed", private_exception=exc,
            )
        return result


def _tcp_pose_changed(
    previous_observation: Mapping[str, Any],
    current_observation: Mapping[str, Any],
    executor: Any,
) -> bool:
    limits = _executor_limits(executor)
    position_tolerance = getattr(limits, "position_tolerance_m", None)
    orientation_tolerance = getattr(limits, "orientation_tolerance_rad", None)
    if (
        isinstance(position_tolerance, bool)
        or not isinstance(position_tolerance, (int, float))
        or not isfinite(float(position_tolerance))
        or float(position_tolerance) <= 0.0
        or isinstance(orientation_tolerance, bool)
        or not isinstance(orientation_tolerance, (int, float))
        or not isfinite(float(orientation_tolerance))
        or float(orientation_tolerance) <= 0.0
    ):
        return False

    def pose(observation: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        try:
            values = observation["robot_state"]["sensors"]["tcp_pose"]
            if not isinstance(values, (list, tuple)) or len(values) != 7:
                return None
            numbers = tuple(float(value) for value in values)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if any(not isfinite(value) for value in numbers):
            return None
        quaternion = numbers[3:]
        norm = sqrt(sum(value * value for value in quaternion))
        if not isfinite(norm) or norm <= 0.0:
            return None
        return numbers[:3], tuple(value / norm for value in quaternion)

    previous = pose(previous_observation)
    current = pose(current_observation)
    if previous is None or current is None:
        return False
    previous_position, previous_quaternion = previous
    current_position, current_quaternion = current
    translation = sqrt(sum(
        (current_position[index] - previous_position[index]) ** 2 for index in range(3)
    ))
    quaternion_dot = abs(sum(
        previous_quaternion[index] * current_quaternion[index] for index in range(4)
    ))
    orientation = 2.0 * acos(min(1.0, max(0.0, quaternion_dot)))
    return (
        translation > float(position_tolerance)
        or orientation > float(orientation_tolerance)
    )


def _executor_limits(executor: Any) -> Any:
    limits = getattr(executor, "limits", None)
    if limits is None:
        limits = getattr(getattr(executor, "executor", None), "limits", None)
    return limits


def _constraints_key(value: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted(
        (item["relation"], item["moving_feature"], item["reference_feature"])
        for item in value.get("constraints", ())
    ))


def _same_tcp_adjustment(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("tool") != "move_tcp" or right.get("tool") != "move_tcp":
        return False
    if left.get("frame") != right.get("frame"):
        return False
    for key in ("translation_delta_m", "rotation_vector_rad"):
        left_values = left.get(key)
        right_values = right.get(key)
        if (
            not isinstance(left_values, (list, tuple))
            or not isinstance(right_values, (list, tuple))
            or len(left_values) != len(right_values)
        ):
            return False
        try:
            differences = [float(a) - float(b) for a, b in zip(left_values, right_values)]
        except (TypeError, ValueError, OverflowError):
            return False
        if any(not isfinite(value) for value in differences) or any(abs(value) > 1e-6 for value in differences):
            return False
    return True


def _trace_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    projected, _ = project_observation(observation)
    frames = projected.get("frames", {})
    projected["frames"] = {
        name: value if not value.startswith("data:image/") else f"frame://{name}"
        for name, value in frames.items()
    }
    state = projected.get("robot_state")
    if isinstance(state, Mapping) and isinstance(state.get("sensors"), Mapping):
        state = dict(state)
        sensors = dict(state["sensors"])
        obstacle_map = sensors.get("rgbd_obstacle_map")
        if isinstance(obstacle_map, Mapping):
            sensors["rgbd_obstacle_map"] = {
                "source_observation_id": obstacle_map["source_observation_id"],
                "frame": obstacle_map["frame"],
                "valid": obstacle_map["valid"],
                "point_count": len(obstacle_map["points_m"]),
            }
        state["sensors"] = sensors
        projected["robot_state"] = state
    return projected


__all__ = ["AssemblyHarnessLoop", "AssemblyHarnessResult"]
