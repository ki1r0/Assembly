"""Bounded public TCP executor for bolt pickup, transport, and insertion."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, isfinite, sqrt
from typing import Any, Mapping, Sequence


MAX_SKILL_DURATION_S = 30.0
MAX_NUDGE_DURATION_S = 1.0
MAX_RETRACT_DURATION_S = 5.0
MAX_LINEAR_SPEED_MPS = 0.04
GRASP_APPROACH_SPEED_MPS = 0.02
NUDGE_SPEED_MPS = 0.005
FREE_SPACE_COMMAND_LOOKAHEAD_M = 0.010
MAX_ANGULAR_SPEED_RADPS = 0.5
POSITION_TOLERANCE_M = 0.002
LATERAL_SETTLED_TOLERANCE_M = 0.0005
LOADED_AXIAL_SETTLED_TOLERANCE_M = 0.006
ORIENTATION_TOLERANCE_RAD = 0.03
TCP_LINEAR_SETTLED_MPS = 0.01
TCP_ANGULAR_SETTLED_RADPS = 0.1
GRASP_POSITION_TOLERANCE_M = 0.0005
GRASP_ORIENTATION_TOLERANCE_RAD = 0.005
GRASP_TCP_LINEAR_SETTLED_MPS = 0.001
GRASP_JOINT_SETTLED_RADPS = 0.01
GRASP_VERIFIED_STEPS = 3
FINAL_XY_TOLERANCE_M = 0.0002
GRIPPER_WIDTH_TOLERANCE_M = 0.001
GRIPPER_WIDTH_STABLE_M = 0.0002
CONTACT_STABLE_STEPS = 3
PANDA_MAX_GRIPPER_WIDTH_M = 0.08
PANDA_ARM_JOINT_COUNT = 7
NOMINAL_GRASP_TCP_Z_M = 0.7338997358
NOMINAL_SAFE_CARRY_Z_M = 0.90
NOMINAL_PREINSERT_CLEAR_Z_M = 0.88585544
NOMINAL_SEAT_TCP_Z_M = 0.82185544
NUDGE_STEP_M = {"coarse": 0.001, "fine": 0.0001, "contact": 0.00005}
RETRACT_DISTANCE_M = {"short": 0.01}
MICRO_ACTION_POSITION_TOLERANCE_M = 0.00002
MICRO_ACTION_VERIFIED_STEPS = 3
MICRO_ACTION_TCP_SETTLED_MPS = 0.001
MICRO_ACTION_JOINT_SETTLED_RADPS = 0.01
MAX_INSERTION_SPEED_MPS = 0.005
NOMINAL_INSERTION_SPEED_MPS = 0.003
INSERTION_FORCE_RELIEF_M = 0.001
INSERTION_START_XY_TOLERANCE_M = 0.0005
INSERTION_START_Z_TOLERANCE_M = 0.006
INSERTION_POSITION_TOLERANCE_M = 0.0005
INSERTION_ORIENTATION_TOLERANCE_RAD = 0.005
INSERTION_TCP_SETTLED_MPS = 0.001
INSERTION_JOINT_SETTLED_RADPS = 0.01
INSERTION_VERIFIED_STEPS = 3
MIN_HELD_SEAT_DWELL_S = 0.2
NOMINAL_HELD_SEAT_DWELL_S = 0.25
INSERTION_PROGRESS_EPSILON_M = 0.00002
MAX_INSERTION_DIAGNOSTIC_SAMPLES = 2048
MAX_INSERTION_TRAVEL_M = 0.08
INSERTION_STALL_TIMEOUT_S = 1.0
TOP_DOWN_GRASP_QUATERNION_WXYZ = (0.0, 1.0, 0.0, 0.0)

_SKILL_MODES = {
    "pick": "default",
    "transport": "default",
    "insert_and_seat": "compliant",
    "release_and_retract": "default",
}

# Only the current calibrated task is supported: world and assembly axes coincide.
_WORLD_NUDGE_VECTORS = {
    "x_positive": (1.0, 0.0, 0.0),
    "x_negative": (-1.0, 0.0, 0.0),
    "y_positive": (0.0, 1.0, 0.0),
    "y_negative": (0.0, -1.0, 0.0),
}
_ASSEMBLY_NUDGE_VECTORS = {
    "x_positive": (1.0, 0.0, 0.0),
    "x_negative": (-1.0, 0.0, 0.0),
    "y_positive": (0.0, 1.0, 0.0),
    "y_negative": (0.0, -1.0, 0.0),
}
_NUDGE_VECTORS_BY_FRAME = {
    "world": _WORLD_NUDGE_VECTORS,
    "assembly": _ASSEMBLY_NUDGE_VECTORS,
}


class ExecutorInterfaceError(RuntimeError):
    """The environment does not expose the required public control evidence."""


@dataclass(frozen=True)
class TCPWaypoint:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or any(not isfinite(float(v)) for v in self.position_m):
            raise ValueError("waypoint position_m must contain three finite values")
        if len(self.quaternion_wxyz) != 4 or any(not isfinite(float(v)) for v in self.quaternion_wxyz):
            raise ValueError("waypoint quaternion_wxyz must contain four finite values")
        norm = sqrt(sum(float(v) ** 2 for v in self.quaternion_wxyz))
        if norm < 1e-8:
            raise ValueError("waypoint quaternion_wxyz must be non-zero")
        object.__setattr__(self, "position_m", tuple(float(v) for v in self.position_m))
        object.__setattr__(self, "quaternion_wxyz", tuple(float(v) / norm for v in self.quaternion_wxyz))


@dataclass(frozen=True)
class BoltMotionPlan:
    """Public, calibrated TCP targets; no simulator object pose is accepted."""

    pick_waypoints: tuple[TCPWaypoint, TCPWaypoint, TCPWaypoint]
    transport_waypoints: tuple[TCPWaypoint, ...]
    open_gripper_width_m: float
    grasp_gripper_width_m: float
    max_gripper_force_n: float
    seat_waypoint: TCPWaypoint | None = None
    retract_waypoint: TCPWaypoint | None = None
    max_insertion_force_n: float | None = None
    max_insertion_travel_m: float | None = None
    insertion_speed_mps: float = NOMINAL_INSERTION_SPEED_MPS
    insertion_stall_timeout_s: float = INSERTION_STALL_TIMEOUT_S
    held_seat_dwell_s: float = NOMINAL_HELD_SEAT_DWELL_S

    def __post_init__(self) -> None:
        if len(self.pick_waypoints) != 3 or any(not isinstance(w, TCPWaypoint) for w in self.pick_waypoints):
            raise ValueError("pick_waypoints must be (pregrasp, grasp, lift) TCP waypoints")
        if not self.transport_waypoints or any(not isinstance(w, TCPWaypoint) for w in self.transport_waypoints):
            raise ValueError("transport_waypoints must contain calibrated TCP waypoints")
        values = (self.open_gripper_width_m, self.grasp_gripper_width_m)
        if any(isinstance(v, bool) or not isfinite(float(v)) or float(v) < 0 for v in values):
            raise ValueError("gripper widths must be finite and non-negative")
        if self.open_gripper_width_m > PANDA_MAX_GRIPPER_WIDTH_M:
            raise ValueError("open gripper width exceeds Panda's 80 mm maximum")
        if self.open_gripper_width_m <= self.grasp_gripper_width_m:
            raise ValueError("open width must exceed grasp width")
        if (
            isinstance(self.max_gripper_force_n, bool)
            or not isfinite(float(self.max_gripper_force_n))
            or float(self.max_gripper_force_n) <= 0.0
        ):
            raise ValueError("max_gripper_force_n must be positive and finite")
        object.__setattr__(self, "open_gripper_width_m", float(self.open_gripper_width_m))
        object.__setattr__(self, "grasp_gripper_width_m", float(self.grasp_gripper_width_m))
        object.__setattr__(self, "max_gripper_force_n", float(self.max_gripper_force_n))

        insertion_values = (
            self.retract_waypoint,
            self.max_insertion_force_n,
            self.max_insertion_travel_m,
        )
        if self.seat_waypoint is None:
            if any(value is not None for value in insertion_values):
                raise ValueError("seat_waypoint is required when insertion settings are supplied")
            return
        if not isinstance(self.seat_waypoint, TCPWaypoint) or not isinstance(self.retract_waypoint, TCPWaypoint):
            raise ValueError("insertion requires calibrated seat and retract TCP waypoints")
        if (
            isinstance(self.max_insertion_force_n, bool)
            or self.max_insertion_force_n is None
            or not isfinite(float(self.max_insertion_force_n))
            or float(self.max_insertion_force_n) <= 0.0
        ):
            raise ValueError("max_insertion_force_n must be positive and finite")
        if (
            isinstance(self.max_insertion_travel_m, bool)
            or self.max_insertion_travel_m is None
            or not isfinite(float(self.max_insertion_travel_m))
            or not 0.0 < float(self.max_insertion_travel_m) <= MAX_INSERTION_TRAVEL_M
        ):
            raise ValueError(f"max_insertion_travel_m must be in (0, {MAX_INSERTION_TRAVEL_M}] m")
        if (
            isinstance(self.insertion_speed_mps, bool)
            or not isfinite(float(self.insertion_speed_mps))
            or not 0.0 < float(self.insertion_speed_mps) <= MAX_INSERTION_SPEED_MPS
        ):
            raise ValueError(f"insertion_speed_mps must be in (0, {MAX_INSERTION_SPEED_MPS}] m/s")
        if (
            isinstance(self.insertion_stall_timeout_s, bool)
            or not isfinite(float(self.insertion_stall_timeout_s))
            or not 0.0 < float(self.insertion_stall_timeout_s) <= MAX_SKILL_DURATION_S
        ):
            raise ValueError("insertion_stall_timeout_s must be positive and no greater than the skill limit")
        if (
            isinstance(self.held_seat_dwell_s, bool)
            or not isfinite(float(self.held_seat_dwell_s))
            or not MIN_HELD_SEAT_DWELL_S <= float(self.held_seat_dwell_s) <= MAX_SKILL_DURATION_S
        ):
            raise ValueError(f"held_seat_dwell_s must be in [{MIN_HELD_SEAT_DWELL_S}, {MAX_SKILL_DURATION_S}] s")

        preinsert = self.transport_waypoints[-1]
        travel = _norm(tuple(preinsert.position_m[i] - self.seat_waypoint.position_m[i] for i in range(3)))
        if travel <= 0.0 or travel > float(self.max_insertion_travel_m) + 1e-9:
            raise ValueError("seat waypoint must be reachable within max_insertion_travel_m")
        if (
            abs(preinsert.position_m[0] - self.seat_waypoint.position_m[0]) > 1e-9
            or abs(preinsert.position_m[1] - self.seat_waypoint.position_m[1]) > 1e-9
            or self.seat_waypoint.position_m[2] >= preinsert.position_m[2]
        ):
            raise ValueError("the bounded insertion path must descend vertically from preinsert")
        if self.retract_waypoint.position_m != preinsert.position_m:
            raise ValueError("retract_waypoint must return to the final preinsert waypoint")
        if _norm(_rotation_error(preinsert.quaternion_wxyz, self.seat_waypoint.quaternion_wxyz)) > 1e-8:
            raise ValueError("seat waypoint must preserve the preinsert orientation")
        if _norm(_rotation_error(preinsert.quaternion_wxyz, self.retract_waypoint.quaternion_wxyz)) > 1e-8:
            raise ValueError("retract waypoint must preserve the preinsert orientation")
        object.__setattr__(self, "max_insertion_force_n", float(self.max_insertion_force_n))
        object.__setattr__(self, "max_insertion_travel_m", float(self.max_insertion_travel_m))
        object.__setattr__(self, "insertion_speed_mps", float(self.insertion_speed_mps))
        object.__setattr__(self, "insertion_stall_timeout_s", float(self.insertion_stall_timeout_s))
        object.__setattr__(self, "held_seat_dwell_s", float(self.held_seat_dwell_s))


def make_nominal_bolt_motion_plan(
    *,
    max_gripper_force_n: float,
    max_insertion_force_n: float | None = None,
    held_seat_dwell_s: float = NOMINAL_HELD_SEAT_DWELL_S,
) -> BoltMotionPlan:
    """Return calibrated targets; omit insertion until its force cap is configured."""
    x, y = 0.30, 0.25
    grasp_z = NOMINAL_GRASP_TCP_Z_M
    q = TOP_DOWN_GRASP_QUATERNION_WXYZ
    safe_pick = TCPWaypoint((x, y, NOMINAL_SAFE_CARRY_Z_M), q)
    safe_socket = TCPWaypoint((0.58, -0.15, NOMINAL_SAFE_CARRY_Z_M), q)
    preinsert = TCPWaypoint((0.58, -0.15, NOMINAL_PREINSERT_CLEAR_Z_M), q)
    plan = BoltMotionPlan(
        pick_waypoints=(
            TCPWaypoint((x, y, grasp_z + 0.04), q),
            TCPWaypoint((x, y, grasp_z), q),
            TCPWaypoint((x, y, grasp_z + 0.08), q),
        ),
        transport_waypoints=(safe_pick, safe_socket, preinsert),
        open_gripper_width_m=PANDA_MAX_GRIPPER_WIDTH_M,
        grasp_gripper_width_m=0.012,
        max_gripper_force_n=max_gripper_force_n,
    )
    if max_insertion_force_n is None:
        return plan
    return BoltMotionPlan(
        pick_waypoints=plan.pick_waypoints,
        transport_waypoints=plan.transport_waypoints,
        open_gripper_width_m=plan.open_gripper_width_m,
        grasp_gripper_width_m=plan.grasp_gripper_width_m,
        max_gripper_force_n=plan.max_gripper_force_n,
        seat_waypoint=TCPWaypoint((0.58, -0.15, NOMINAL_SEAT_TCP_Z_M), q),
        retract_waypoint=preinsert,
        max_insertion_force_n=max_insertion_force_n,
        max_insertion_travel_m=MAX_INSERTION_TRAVEL_M,
        insertion_speed_mps=NOMINAL_INSERTION_SPEED_MPS,
        held_seat_dwell_s=held_seat_dwell_s,
    )


class BoltSkillExecutor:
    """Run bounded skills through a Factory-style 6D delta step.

    State and bilateral contact come from get_public_executor_state(). Gripper
    commands use set_gripper_target_width() with the plan's force cap. Contact
    is never inferred from commanded width. Release additionally requires the
    independent check_task boolean; reaching the seat TCP target is not success.
    """

    def __init__(self, env: Any, *, target_id: str, plan: BoltMotionPlan) -> None:
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id must be a non-empty string")
        if not isinstance(plan, BoltMotionPlan):
            raise TypeError("plan must be a BoltMotionPlan")
        if not callable(getattr(env, "get_public_executor_state", None)):
            raise ExecutorInterfaceError("environment must expose get_public_executor_state()")
        if not callable(getattr(env, "set_gripper_target_width", None)):
            raise ExecutorInterfaceError("environment must expose set_gripper_target_width(width_m, max_force_n)")
        if not callable(getattr(env, "step", None)):
            raise ExecutorInterfaceError("environment must expose step(delta_action)")
        self.env = env
        self.target_id = target_id
        self.plan = plan
        self._holding = False
        self._insertion_motion_completed = False
        self._environment_ended = False
        self._last_gripper_target_m = plan.open_gripper_width_m
        # Harness artifact integration may serialize this; it is never returned by a tool call.
        self.private_diagnostics: dict[str, Any] = {"insertion": None}
        self._control_dt_s, self._position_action_scale, self._rotation_action_scale = self._control_parameters()

    def execute_skill(
        self,
        skill: str,
        target_id: str,
        *,
        mode: str = "default",
        max_duration_s: float = MAX_SKILL_DURATION_S,
        completed: bool = False,
    ) -> dict[str, str]:
        """Run one bounded skill; ``completed`` is trusted only as release authorization."""
        if (
            target_id != self.target_id
            or not isinstance(skill, str)
            or skill not in _SKILL_MODES
            or mode != _SKILL_MODES.get(skill)
            or type(completed) is not bool
            or isinstance(max_duration_s, bool)
            or not isinstance(max_duration_s, (int, float))
            or not isfinite(float(max_duration_s))
            or not 0 < float(max_duration_s) <= MAX_SKILL_DURATION_S
        ):
            return {"status": "invalid_action"}
        if skill == "pick" and self._holding:
            return {"status": "invalid_action"}
        if skill in {"transport", "insert_and_seat", "release_and_retract"} and not self._holding:
            return {"status": "invalid_action"}
        if skill == "insert_and_seat" and self.plan.seat_waypoint is None:
            return {"status": "invalid_action"}
        if skill == "release_and_retract" and (
            not completed or not self._insertion_motion_completed or self.plan.retract_waypoint is None
        ):
            return {"status": "invalid_action"}

        max_steps = int(float(max_duration_s) / self._control_dt_s)
        reserved_steps = 2 if skill == "insert_and_seat" else 1
        if max_steps <= reserved_steps:
            return {"status": "invalid_action"}
        budget = max_steps - reserved_steps
        self._environment_ended = False
        try:
            state = self._read_state()
            initial_width = self.plan.open_gripper_width_m if skill == "pick" else self.plan.grasp_gripper_width_m
            if skill == "insert_and_seat":
                self._read_insertion_force_n()
            state, budget, armed = self._wait_for_motion_safety(state, budget, initial_width)
            if not armed:
                self._stop_and_hold()
                return {"status": "motion_timeout"}
            if skill == "pick":
                return self._pick(state, budget)
            if skill == "transport":
                return self._transport(state, budget)
            if skill == "insert_and_seat":
                self._insertion_motion_completed = False
                return self._insert_and_seat(state, budget)
            return self._release_and_retract(state, budget)
        except Exception:
            self._stop_and_hold()
            raise

    def nudge(
        self,
        *,
        frame: str,
        direction: str,
        step_class: str,
        max_duration_s: float,
    ) -> dict[str, str]:
        """Apply one bounded planar correction while retaining the verified grasp."""
        if (
            not self._holding
            or self.plan.max_insertion_force_n is None
            or not isinstance(frame, str)
            or not isinstance(direction, str)
            or not isinstance(step_class, str)
            or not _valid_duration(max_duration_s, MAX_NUDGE_DURATION_S)
        ):
            return {"status": "invalid_action"}
        direction_vector = _NUDGE_VECTORS_BY_FRAME.get(frame, {}).get(direction)
        step_m = NUDGE_STEP_M.get(step_class)
        if direction_vector is None or step_m is None:
            return {"status": "invalid_action"}
        translation = tuple(component * step_m for component in direction_vector)
        return self._execute_held_translation(
            translation,
            max_duration_s=float(max_duration_s),
            speed_mps=NUDGE_SPEED_MPS,
            rollback_on_force=True,
        )

    def retract(
        self,
        *,
        direction: str,
        distance_class: str,
        max_duration_s: float,
    ) -> dict[str, str]:
        """Retract only upward in world Z, using a finite calibrated distance class."""
        if (
            not self._holding
            or self.plan.max_insertion_force_n is None
            or not isinstance(direction, str)
            or direction != "z_positive"
            or not isinstance(distance_class, str)
            or not _valid_duration(max_duration_s, MAX_RETRACT_DURATION_S)
        ):
            return {"status": "invalid_action"}
        if distance_class == "short":
            distance_m = RETRACT_DISTANCE_M["short"]
        elif distance_class == "full":
            seat = self.plan.seat_waypoint
            if seat is None or self.plan.max_insertion_travel_m is None:
                return {"status": "invalid_action"}
            distance_m = self.plan.transport_waypoints[-1].position_m[2] - seat.position_m[2]
            if not 0.0 < distance_m <= self.plan.max_insertion_travel_m:
                return {"status": "invalid_action"}
        else:
            return {"status": "invalid_action"}
        return self._execute_held_translation(
            (0.0, 0.0, distance_m),
            max_duration_s=float(max_duration_s),
            speed_mps=MAX_LINEAR_SPEED_MPS,
            rollback_on_force=False,
        )

    def _pick(self, state: dict[str, Any], budget: int) -> dict[str, str]:
        pregrasp, grasp, lift = self.plan.pick_waypoints
        state, reached, budget = self._move_to(
            pregrasp,
            state,
            budget,
            self.plan.open_gripper_width_m,
            require_grasp=False,
            require_open=True,
            max_linear_speed_mps=MAX_LINEAR_SPEED_MPS,
            position_tracking_tolerance_m=self._reference_tolerance_for_lead(
                FREE_SPACE_COMMAND_LOOKAHEAD_M, MAX_LINEAR_SPEED_MPS
            ),
        )
        if not reached:
            self._stop_and_hold()
            return {"status": "motion_timeout"}

        state, reached, budget = self._move_to(
            grasp,
            state,
            budget,
            self.plan.open_gripper_width_m,
            require_grasp=False,
            require_open=True,
            position_tolerance_m=GRASP_POSITION_TOLERANCE_M,
            orientation_tolerance_rad=GRASP_ORIENTATION_TOLERANCE_RAD,
            tcp_linear_settled_mps=GRASP_TCP_LINEAR_SETTLED_MPS,
            joint_speed_tolerance_radps=GRASP_JOINT_SETTLED_RADPS,
            stable_steps_required=GRASP_VERIFIED_STEPS,
            max_linear_speed_mps=GRASP_APPROACH_SPEED_MPS,
            position_tracking_tolerance_m=POSITION_TOLERANCE_M,
        )
        if not reached:
            self._stop_and_hold()
            return {"status": "motion_timeout"}

        verified, budget = self._close_and_verify_grasp(budget)
        if not verified:
            self._stop_and_hold()
            return {"status": "motion_timeout"}
        self._holding = True
        self._last_gripper_target_m = self.plan.grasp_gripper_width_m
        state, reached, _ = self._move_to(
            lift, self._read_state(), budget, self.plan.grasp_gripper_width_m,
            require_grasp=True, require_open=False,
            position_tolerance_m=LOADED_AXIAL_SETTLED_TOLERANCE_M,
            xy_position_tolerance_m=LATERAL_SETTLED_TOLERANCE_M,
            max_linear_speed_mps=MAX_LINEAR_SPEED_MPS,
            position_tracking_tolerance_m=self._reference_tolerance_for_lead(
                FREE_SPACE_COMMAND_LOOKAHEAD_M, MAX_LINEAR_SPEED_MPS
            ),
        )
        if not reached:
            self._stop_and_hold()
            if not all(state["finger_bolt_contacts"]):
                return {"status": "grasp_lost"}
            return {"status": "motion_timeout"}
        return {"status": "completed"}

    def _transport(self, state: dict[str, Any], budget: int) -> dict[str, str]:
        final_index = len(self.plan.transport_waypoints) - 1
        for index, waypoint in enumerate(self.plan.transport_waypoints):
            state, reached, budget = self._move_to(
                waypoint, state, budget, self.plan.grasp_gripper_width_m,
                require_grasp=True,
                require_open=False,
                position_tolerance_m=LOADED_AXIAL_SETTLED_TOLERANCE_M,
                xy_position_tolerance_m=(
                    FINAL_XY_TOLERANCE_M if index == final_index else LATERAL_SETTLED_TOLERANCE_M
                ),
                orientation_tolerance_rad=(
                    INSERTION_ORIENTATION_TOLERANCE_RAD if index == final_index else ORIENTATION_TOLERANCE_RAD
                ),
                max_linear_speed_mps=MAX_LINEAR_SPEED_MPS,
                position_tracking_tolerance_m=self._reference_tolerance_for_lead(
                    FREE_SPACE_COMMAND_LOOKAHEAD_M, MAX_LINEAR_SPEED_MPS
                ),
            )
            if not reached:
                self._stop_and_hold()
                return {"status": "motion_timeout"}
        return {"status": "completed"}

    def _execute_held_translation(
        self,
        translation_m: Sequence[float],
        *,
        max_duration_s: float,
        speed_mps: float,
        rollback_on_force: bool,
    ) -> dict[str, str]:
        max_steps = int(max_duration_s / self._control_dt_s)
        budget = max_steps - 1
        rollback_steps = (
            ceil(_norm(translation_m) / (speed_mps * self._control_dt_s)) if rollback_on_force else 0
        )
        if budget <= rollback_steps:
            return {"status": "invalid_action"}

        self._environment_ended = False
        gripper_target = self._last_gripper_target_m
        try:
            state = self._read_state()
            if not all(state["finger_bolt_contacts"]):
                self._holding = False
                self._stop_and_hold()
                return {"status": "motion_timeout"}
            force_n = self._read_insertion_force_n()
            if force_n > float(self.plan.max_insertion_force_n):
                self._stop_and_hold()
                return {"status": "force_limit"}

            state, budget, armed = self._wait_for_motion_safety(state, budget, gripper_target)
            if not armed:
                self._stop_and_hold()
                return {"status": "motion_timeout"}
            if not all(state["finger_bolt_contacts"]):
                self._holding = False
                self._stop_and_hold()
                return {"status": "motion_timeout"}
            if budget <= rollback_steps:
                self._stop_and_hold()
                return {"status": "motion_timeout"}

            settled_samples = 0
            while budget > rollback_steps and settled_samples < MICRO_ACTION_VERIFIED_STEPS:
                if not all(state["finger_bolt_contacts"]):
                    self._holding = False
                    self._stop_and_hold()
                    return {"status": "motion_timeout"}
                force_n = self._read_insertion_force_n()
                if force_n > float(self.plan.max_insertion_force_n):
                    self._stop_and_hold()
                    return {"status": "force_limit"}
                stopped = (
                    _norm(state["tcp_velocity"][:3]) <= MICRO_ACTION_TCP_SETTLED_MPS
                    and _norm(state["tcp_velocity"][3:]) <= TCP_ANGULAR_SETTLED_RADPS
                    and max(abs(value) for value in state["joint_velocities"])
                    <= MICRO_ACTION_JOINT_SETTLED_RADPS
                )
                settled_samples = settled_samples + 1 if stopped else 0
                if settled_samples >= MICRO_ACTION_VERIFIED_STEPS:
                    break
                self._insertion_motion_completed = False
                state, ended = self._step((0.0,) * 6, gripper_target)
                budget -= 1
                if ended:
                    return {"status": "motion_timeout"}

            if settled_samples < MICRO_ACTION_VERIFIED_STEPS or budget <= rollback_steps:
                self._stop_and_hold()
                return {"status": "motion_timeout"}

            measured_start = state["tcp_pose"]
            command_start = state["commanded_tcp_pose"]
            measured_goal = tuple(measured_start[i] + float(translation_m[i]) for i in range(3))
            command_goal = tuple(command_start[i] + float(translation_m[i]) for i in range(3))
            command_waypoint = TCPWaypoint(command_goal, tuple(command_start[3:7]))
            start_waypoint = TCPWaypoint(tuple(command_start[:3]), tuple(command_start[3:7]))
            target_rotation_error = INSERTION_ORIENTATION_TOLERANCE_RAD
            stable_steps = 0
            movement_started = False
            tracking_tolerance = self._reference_tolerance_for_lead(
                FREE_SPACE_COMMAND_LOOKAHEAD_M, speed_mps
            )

            while budget > rollback_steps:
                if not all(state["finger_bolt_contacts"]):
                    self._holding = False
                    self._stop_and_hold()
                    return {"status": "motion_timeout"}
                force_n = self._read_insertion_force_n()
                if force_n > float(self.plan.max_insertion_force_n):
                    if rollback_on_force and movement_started and budget > 0:
                        state, budget = self._return_held_micro_to_start(
                            state,
                            budget,
                            start_waypoint,
                            measured_start,
                            gripper_target,
                            speed_mps,
                            tracking_tolerance,
                        )
                    self._stop_and_hold()
                    return {"status": "force_limit"}

                measured_error = _norm(
                    tuple(measured_goal[i] - state["tcp_pose"][i] for i in range(3))
                )
                commanded_error = _norm(
                    tuple(command_goal[i] - state["commanded_tcp_pose"][i] for i in range(3))
                )
                measured_rotation_error = _norm(
                    _rotation_error(command_start[3:7], state["tcp_pose"][3:7])
                )
                commanded_rotation_error = _norm(
                    _rotation_error(command_start[3:7], state["commanded_tcp_pose"][3:7])
                )
                stopped = (
                    _norm(state["tcp_velocity"][:3]) <= MICRO_ACTION_TCP_SETTLED_MPS
                    and _norm(state["tcp_velocity"][3:]) <= TCP_ANGULAR_SETTLED_RADPS
                    and max(abs(value) for value in state["joint_velocities"])
                    <= MICRO_ACTION_JOINT_SETTLED_RADPS
                )
                at_target = (
                    measured_error <= MICRO_ACTION_POSITION_TOLERANCE_M
                    and measured_rotation_error <= target_rotation_error
                    and commanded_error <= 1e-7
                    and commanded_rotation_error <= 1e-7
                )
                stable_steps = stable_steps + 1 if at_target and stopped else 0
                if stable_steps >= MICRO_ACTION_VERIFIED_STEPS:
                    return {"status": "completed"}

                action, _, _ = self._action(
                    command_waypoint,
                    state,
                    max_linear_speed_mps=speed_mps,
                    position_tracking_tolerance_m=tracking_tolerance,
                )
                self._insertion_motion_completed = False
                movement_started = movement_started or _norm(action) > 1e-12
                state, ended = self._step(action, gripper_target)
                budget -= 1
                if ended:
                    return {"status": "motion_timeout"}

            self._stop_and_hold()
            return {"status": "motion_timeout"}
        except Exception:
            self._stop_and_hold()
            raise

    def _return_held_micro_to_start(
        self,
        state: dict[str, Any],
        budget: int,
        start_waypoint: TCPWaypoint,
        measured_start: Sequence[float],
        gripper_target: float,
        speed_mps: float,
        tracking_tolerance: float,
    ) -> tuple[dict[str, Any], int]:
        """Undo a force-limited nudge only; this never introduces a new direction."""
        while budget > 0:
            if not all(state["finger_bolt_contacts"]):
                self._holding = False
                break
            measured_error = _norm(
                tuple(measured_start[i] - state["tcp_pose"][i] for i in range(3))
            )
            commanded_error = _norm(
                tuple(start_waypoint.position_m[i] - state["commanded_tcp_pose"][i] for i in range(3))
            )
            if (
                measured_error <= MICRO_ACTION_POSITION_TOLERANCE_M
                and commanded_error <= 1e-7
                and _norm(_rotation_error(start_waypoint.quaternion_wxyz, state["tcp_pose"][3:7]))
                <= INSERTION_ORIENTATION_TOLERANCE_RAD
            ):
                break
            action, _, _ = self._action(
                start_waypoint,
                state,
                max_linear_speed_mps=speed_mps,
                position_tracking_tolerance_m=tracking_tolerance,
            )
            state, ended = self._step(action, gripper_target)
            budget -= 1
            if ended:
                break
        return state, budget

    def _insert_and_seat(self, state: dict[str, Any], budget: int) -> dict[str, str]:
        diagnostics = self._start_insertion_diagnostics()
        seat = self.plan.seat_waypoint
        if seat is None or self.plan.max_insertion_force_n is None or self.plan.max_insertion_travel_m is None:
            diagnostics["exit_reason"] = "insertion_plan_unavailable"
            return {"status": "invalid_action"}
        preinsert = self.plan.transport_waypoints[-1]
        start_xy_error = _norm(
            tuple(state["tcp_pose"][i] - preinsert.position_m[i] for i in range(2))
        )
        start_z_error = abs(state["tcp_pose"][2] - preinsert.position_m[2])
        start_rotation_error = _norm(_rotation_error(preinsert.quaternion_wxyz, state["tcp_pose"][3:7]))
        travel_to_seat = _norm(tuple(seat.position_m[i] - state["tcp_pose"][i] for i in range(3)))
        if (
            start_xy_error > INSERTION_START_XY_TOLERANCE_M
            or start_z_error > INSERTION_START_Z_TOLERANCE_M
            or start_rotation_error > INSERTION_ORIENTATION_TOLERANCE_RAD
            or travel_to_seat > self.plan.max_insertion_travel_m + 1e-9
        ):
            self._finish_insertion_diagnostics(
                diagnostics, "preinsert_pose_outside_tolerance", state, seat, None, 0
            )
            return {"status": "invalid_action"}

        progress_origin_z = preinsert.position_m[2]
        elapsed_steps = 0
        progress_anchor = progress_origin_z - state["tcp_pose"][2]
        last_progress_step = 0
        stable_steps = 0
        seat_hold_since_step: int | None = None
        held_dwell_steps = ceil(self.plan.held_seat_dwell_s / self._control_dt_s)
        force_n: float | None = None
        while budget > 0:
            if not all(state["finger_bolt_contacts"]):
                self._holding = False
                self._stop_and_hold()
                self._finish_insertion_diagnostics(
                    diagnostics, "bilateral_contact_lost", state, seat, None, elapsed_steps
                )
                return {"status": "motion_timeout"}
            force_n = self._read_insertion_force_n()
            sample = self._insertion_diagnostic_sample(state, seat, force_n, elapsed_steps, "control")
            self._append_insertion_diagnostic_sample(diagnostics, sample)
            if force_n > self.plan.max_insertion_force_n:
                state, budget = self._relieve_insertion_target(state, budget)
                self._stop_and_hold()
                self._finish_insertion_diagnostics(
                    diagnostics, "insertion_force_limit", state, seat, force_n, elapsed_steps
                )
                return {"status": "motion_timeout" if self._environment_ended else "force_limit"}

            # Measure bounded insertion travel from the calibrated preinsert plane,
            # not a measured start that may still be converging within its tolerance.
            progress_m = progress_origin_z - state["tcp_pose"][2]
            if (
                progress_m < -INSERTION_POSITION_TOLERANCE_M
                or progress_m > self.plan.max_insertion_travel_m + INSERTION_POSITION_TOLERANCE_M
            ):
                state, budget = self._relieve_insertion_target(state, budget)
                self._stop_and_hold()
                self._finish_insertion_diagnostics(
                    diagnostics, "insertion_travel_out_of_bounds", state, seat, force_n, elapsed_steps
                )
                return {"status": "motion_timeout"}
            action, _, _ = self._action(
                seat,
                state,
                max_linear_speed_mps=self.plan.insertion_speed_mps,
                position_tracking_tolerance_m=INSERTION_POSITION_TOLERANCE_M,
            )
            at_seat = sample["at_seat"]
            stopped = sample["settled"]
            stable_steps = stable_steps + 1 if at_seat and stopped else 0
            if at_seat and stopped:
                if seat_hold_since_step is None:
                    seat_hold_since_step = elapsed_steps
            else:
                seat_hold_since_step = None
            if (
                stable_steps >= INSERTION_VERIFIED_STEPS
                and seat_hold_since_step is not None
                and elapsed_steps - seat_hold_since_step >= held_dwell_steps
            ):
                self._insertion_motion_completed = True
                self._finish_insertion_diagnostics(
                    diagnostics, "seat_pose_settled_and_dwell_verified", state, seat, force_n, elapsed_steps
                )
                return {"status": "completed"}

            if progress_m - progress_anchor >= INSERTION_PROGRESS_EPSILON_M:
                progress_anchor = progress_m
                last_progress_step = elapsed_steps
            elif not at_seat and (elapsed_steps - last_progress_step) * self._control_dt_s >= self.plan.insertion_stall_timeout_s:
                state, budget = self._relieve_insertion_target(state, budget)
                self._stop_and_hold()
                self._finish_insertion_diagnostics(
                    diagnostics, "no_progress_before_seat", state, seat, force_n, elapsed_steps
                )
                return {"status": "motion_timeout" if self._environment_ended else "stalled"}

            state, ended = self._step(action, self.plan.grasp_gripper_width_m)
            budget -= 1
            elapsed_steps += 1
            if ended:
                self._finish_insertion_diagnostics(
                    diagnostics, "environment_ended", state, seat, force_n, elapsed_steps
                )
                return {"status": "motion_timeout"}
        state, _ = self._relieve_insertion_target(state, 2)
        self._stop_and_hold()
        self._finish_insertion_diagnostics(
            diagnostics, "skill_budget_exhausted", state, seat, force_n, elapsed_steps
        )
        return {"status": "motion_timeout"}

    def _start_insertion_diagnostics(self) -> dict[str, Any]:
        previous = self.private_diagnostics.get("insertion")
        previous_attempts = []
        if isinstance(previous, Mapping):
            previous_attempts = [
                dict(attempt)
                for attempt in previous.get("previous_attempts", ())
                if isinstance(attempt, Mapping)
            ]
            previous_attempts.append(
                {key: value for key, value in previous.items() if key != "previous_attempts"}
            )
        diagnostics: dict[str, Any] = {
            "source": "public_executor_state_and_public_wrench",
            "exit_reason": "in_progress",
            "sample_limit": MAX_INSERTION_DIAGNOSTIC_SAMPLES,
            "dropped_sample_count": 0,
            "samples": [],
            "endpoint": None,
        }
        if previous_attempts:
            diagnostics["previous_attempts"] = previous_attempts
        self.private_diagnostics["insertion"] = diagnostics
        return diagnostics

    def _insertion_diagnostic_sample(
        self,
        state: Mapping[str, Any],
        seat: TCPWaypoint,
        force_n: float | None,
        step: int,
        phase: str,
    ) -> dict[str, Any]:
        tcp_pose = tuple(state["tcp_pose"])
        commanded_pose = tuple(state["commanded_tcp_pose"])
        tcp_position_error = tuple(seat.position_m[i] - tcp_pose[i] for i in range(3))
        command_position_error = tuple(seat.position_m[i] - commanded_pose[i] for i in range(3))
        tcp_rotation_error = _norm(_rotation_error(seat.quaternion_wxyz, tcp_pose[3:7]))
        command_rotation_error = _norm(_rotation_error(seat.quaternion_wxyz, commanded_pose[3:7]))
        tcp_velocity = tuple(state["tcp_velocity"])
        joint_velocities = tuple(state["joint_velocities"])
        arm_joint_velocities = joint_velocities[:PANDA_ARM_JOINT_COUNT]
        gripper_joint_velocities = joint_velocities[PANDA_ARM_JOINT_COUNT:]
        tcp_linear_settled = _norm(tcp_velocity[:3]) <= INSERTION_TCP_SETTLED_MPS
        tcp_angular_settled = _norm(tcp_velocity[3:]) <= TCP_ANGULAR_SETTLED_RADPS
        arm_joints_settled = max(abs(value) for value in arm_joint_velocities) <= INSERTION_JOINT_SETTLED_RADPS
        gripper_joints_settled = max(abs(value) for value in gripper_joint_velocities) <= INSERTION_JOINT_SETTLED_RADPS
        actual_pose_at_seat = (
            _norm(tcp_position_error[:2]) <= INSERTION_POSITION_TOLERANCE_M
            and abs(tcp_position_error[2]) <= INSERTION_POSITION_TOLERANCE_M
            and tcp_rotation_error <= INSERTION_ORIENTATION_TOLERANCE_RAD
        )
        command_pose_at_seat = (
            _norm(command_position_error) <= 1e-7 and command_rotation_error <= 1e-7
        )
        return {
            "step": int(step),
            "phase": phase,
            "target_tcp_pose": (*seat.position_m, *seat.quaternion_wxyz),
            "tcp_pose": tcp_pose,
            "commanded_tcp_pose": commanded_pose,
            "tcp_position_error_m_xyz": tcp_position_error,
            "tcp_position_error_norm_m": _norm(tcp_position_error),
            "command_position_error_m_xyz": command_position_error,
            "command_position_error_norm_m": _norm(command_position_error),
            "tcp_rotation_error_rad": tcp_rotation_error,
            "command_rotation_error_rad": command_rotation_error,
            "tcp_velocity": tcp_velocity,
            "tcp_linear_speed_mps": _norm(tcp_velocity[:3]),
            "tcp_angular_speed_radps": _norm(tcp_velocity[3:]),
            "joint_positions": tuple(state["joint_positions"]),
            "joint_velocities": joint_velocities,
            "arm_joints_settled": arm_joints_settled,
            "gripper_joint_velocities": gripper_joint_velocities,
            "gripper_joints_settled": gripper_joints_settled,
            "force_norm_n": force_n,
            "finger_bolt_contacts": tuple(state["finger_bolt_contacts"]),
            "actual_pose_at_seat": actual_pose_at_seat,
            "command_pose_at_seat": command_pose_at_seat,
            "at_seat": actual_pose_at_seat and command_pose_at_seat,
            "tcp_linear_settled": tcp_linear_settled,
            "tcp_angular_settled": tcp_angular_settled,
            "settled": tcp_linear_settled and tcp_angular_settled and arm_joints_settled,
        }

    def _append_insertion_diagnostic_sample(
        self, diagnostics: dict[str, Any], sample: dict[str, Any]
    ) -> None:
        samples = diagnostics["samples"]
        if len(samples) < MAX_INSERTION_DIAGNOSTIC_SAMPLES:
            samples.append(sample)
        else:
            diagnostics["dropped_sample_count"] += 1
            samples[-1] = sample
        diagnostics["endpoint"] = sample

    def _finish_insertion_diagnostics(
        self,
        diagnostics: dict[str, Any],
        exit_reason: str,
        state: Mapping[str, Any],
        seat: TCPWaypoint,
        force_n: float | None,
        step: int,
    ) -> None:
        diagnostics["exit_reason"] = exit_reason
        diagnostics["endpoint"] = self._insertion_diagnostic_sample(
            state, seat, force_n, step, "endpoint"
        )

    def _relieve_insertion_target(
        self, state: dict[str, Any], budget: int
    ) -> tuple[dict[str, Any], int]:
        """Back out up to 1 mm toward preinsert before holding on force stop."""
        preinsert_z = self.plan.transport_waypoints[-1].position_m[2]
        current_z = state["tcp_pose"][2]
        relief_z = current_z + min(INSERTION_FORCE_RELIEF_M, max(preinsert_z - current_z, 0.0))
        retreat = TCPWaypoint(
            (state["tcp_pose"][0], state["tcp_pose"][1], relief_z),
            tuple(state["tcp_pose"][3:7]),
        )
        while budget > 1:
            measured_error = _norm(tuple(retreat.position_m[i] - state["tcp_pose"][i] for i in range(3)))
            commanded_error = _norm(
                tuple(retreat.position_m[i] - state["commanded_tcp_pose"][i] for i in range(3))
            )
            if measured_error <= 0.0001 and commanded_error <= 1e-7:
                break
            action, _, _ = self._action(
                retreat,
                state,
                max_linear_speed_mps=self.plan.insertion_speed_mps,
                position_tracking_tolerance_m=INSERTION_POSITION_TOLERANCE_M,
            )
            state, ended = self._step(action, self.plan.grasp_gripper_width_m)
            budget -= 1
            if ended:
                break
        return state, budget

    def _release_and_retract(self, state: dict[str, Any], budget: int) -> dict[str, str]:
        if not all(state["finger_bolt_contacts"]):
            self._holding = False
            self._stop_and_hold()
            return {"status": "motion_timeout"}
        seat = self.plan.seat_waypoint
        if seat is None:
            return {"status": "invalid_action"}
        seat_error = _norm(tuple(seat.position_m[i] - state["tcp_pose"][i] for i in range(3)))
        seat_rotation_error = _norm(_rotation_error(seat.quaternion_wxyz, state["tcp_pose"][3:7]))
        if (
            seat_error > INSERTION_POSITION_TOLERANCE_M
            or seat_rotation_error > INSERTION_ORIENTATION_TOLERANCE_RAD
            or _norm(state["tcp_velocity"][:3]) > INSERTION_TCP_SETTLED_MPS
            or _norm(state["tcp_velocity"][3:]) > TCP_ANGULAR_SETTLED_RADPS
            or max(abs(v) for v in state["joint_velocities"]) > INSERTION_JOINT_SETTLED_RADPS
        ):
            self._stop_and_hold()
            return {"status": "motion_timeout"}

        clear_steps = 0
        while budget > 0:
            state, ended = self._step((0.0,) * 6, self.plan.open_gripper_width_m)
            budget -= 1
            if ended:
                return {"status": "motion_timeout"}
            released = (
                abs(state["gripper_width_m"] - self.plan.open_gripper_width_m) <= GRIPPER_WIDTH_TOLERANCE_M
                and not any(state["finger_bolt_contacts"])
            )
            clear_steps = clear_steps + 1 if released else 0
            if clear_steps >= CONTACT_STABLE_STEPS:
                self._holding = False
                self._last_gripper_target_m = self.plan.open_gripper_width_m
                break
        else:
            self._stop_and_hold()
            return {"status": "motion_timeout"}

        retract = self.plan.retract_waypoint
        if retract is None:
            self._stop_and_hold()
            return {"status": "invalid_action"}
        state, reached, _ = self._move_to(
            retract,
            state,
            budget,
            self.plan.open_gripper_width_m,
            require_grasp=False,
            require_open=True,
        )
        if not reached:
            self._stop_and_hold()
            return {"status": "motion_timeout"}
        return {"status": "completed"}

    def _move_to(
        self,
        waypoint: TCPWaypoint,
        state: dict[str, Any],
        budget: int,
        gripper_width_m: float,
        *,
        require_grasp: bool,
        require_open: bool,
        position_tolerance_m: float = POSITION_TOLERANCE_M,
        xy_position_tolerance_m: float | None = None,
        orientation_tolerance_rad: float = ORIENTATION_TOLERANCE_RAD,
        tcp_linear_settled_mps: float = TCP_LINEAR_SETTLED_MPS,
        joint_speed_tolerance_radps: float = 0.05,
        stable_steps_required: int = 1,
        max_linear_speed_mps: float = MAX_LINEAR_SPEED_MPS,
        position_tracking_tolerance_m: float = POSITION_TOLERANCE_M,
    ) -> tuple[dict[str, Any], bool, int]:
        stable_steps = 0
        while budget > 0:
            if require_grasp and not all(state["finger_bolt_contacts"]):
                self._holding = False
                return state, False, budget
            action, _, rotation_error = self._action(
                waypoint,
                state,
                max_linear_speed_mps=max_linear_speed_mps,
                position_tracking_tolerance_m=position_tracking_tolerance_m,
            )
            position_error = tuple(waypoint.position_m[i] - state["tcp_pose"][i] for i in range(3))
            position_reached = (
                _norm(position_error[:2]) <= xy_position_tolerance_m
                and abs(position_error[2]) <= position_tolerance_m
                if xy_position_tolerance_m is not None
                else _norm(position_error) <= position_tolerance_m
            )
            joints_stopped = max(abs(v) for v in state["joint_velocities"]) <= joint_speed_tolerance_radps
            tcp_stopped = (
                _norm(state["tcp_velocity"][:3]) <= tcp_linear_settled_mps
                and _norm(state["tcp_velocity"][3:]) <= TCP_ANGULAR_SETTLED_RADPS
            )
            gripper_ready = (
                not require_open
                or abs(state["gripper_width_m"] - gripper_width_m) <= GRIPPER_WIDTH_TOLERANCE_M
            )
            pose_verified = (
                position_reached
                and rotation_error <= orientation_tolerance_rad
                and joints_stopped
                and tcp_stopped
                and gripper_ready
            )
            stable_steps = stable_steps + 1 if pose_verified else 0
            if stable_steps >= stable_steps_required:
                return state, True, budget
            state, ended = self._step(action, gripper_width_m)
            budget -= 1
            if ended:
                return state, False, budget
        return state, False, budget

    def _close_and_verify_grasp(self, budget: int) -> tuple[bool, int]:
        previous_width: float | None = None
        stable_steps = 0
        while budget > 0:
            state, ended = self._step((0.0,) * 6, self.plan.grasp_gripper_width_m)
            budget -= 1
            if ended:
                return False, budget
            width = state["gripper_width_m"]
            contacts = state["finger_bolt_contacts"]
            if all(contacts) and previous_width is not None and abs(width - previous_width) <= GRIPPER_WIDTH_STABLE_M:
                stable_steps += 1
            else:
                stable_steps = 0
            previous_width = width
            if stable_steps >= CONTACT_STABLE_STEPS:
                return True, budget
        return False, budget

    def _wait_for_motion_safety(
        self, state: dict[str, Any], budget: int, gripper_width_m: float
    ) -> tuple[dict[str, Any], int, bool]:
        while budget > 0 and not state["motion_safety_armed"]:
            state, ended = self._step((0.0,) * 6, gripper_width_m)
            budget -= 1
            if ended:
                return state, budget, False
        return state, budget, state["motion_safety_armed"]

    def _action(
        self,
        waypoint: TCPWaypoint,
        state: Mapping[str, Any],
        *,
        max_linear_speed_mps: float = MAX_LINEAR_SPEED_MPS,
        position_tracking_tolerance_m: float = POSITION_TOLERANCE_M,
    ) -> tuple[tuple[float, ...], float, float]:
        current = state["tcp_pose"]
        commanded = state["commanded_tcp_pose"]
        position_error = tuple(waypoint.position_m[i] - current[i] for i in range(3))
        rotation_error = _rotation_error(waypoint.quaternion_wxyz, current[3:7])
        max_position_step = max_linear_speed_mps * self._control_dt_s
        command_position_error = tuple(waypoint.position_m[i] - commanded[i] for i in range(3))
        position_delta = _limit_norm(command_position_error, max_position_step)
        position_delta = _limit_delta_by_tracking_error(
            tuple(commanded[i] - current[i] for i in range(3)),
            position_delta,
            position_tracking_tolerance_m + max_position_step,
        )
        max_rotation_step = MAX_ANGULAR_SPEED_RADPS * self._control_dt_s
        command_rotation_error = _rotation_error(waypoint.quaternion_wxyz, commanded[3:7])
        rotation_delta = _limit_norm(command_rotation_error, max_rotation_step)
        rotation_delta = _limit_delta_by_tracking_error(
            _rotation_error(commanded[3:7], current[3:7]),
            rotation_delta,
            ORIENTATION_TOLERANCE_RAD + max_rotation_step,
        )
        action = tuple(
            max(-1.0, min(1.0, position_delta[i] / self._position_action_scale[i])) for i in range(3)
        ) + tuple(
            max(-1.0, min(1.0, rotation_delta[i] / self._rotation_action_scale[i])) for i in range(3)
        )
        return action, _norm(position_error), _norm(rotation_error)

    def _step(self, action: Sequence[float], gripper_width_m: float) -> tuple[dict[str, Any], bool]:
        self._last_gripper_target_m = gripper_width_m
        self._set_gripper_target(gripper_width_m)
        result = self.env.step(self._factory_action(action))
        ended = False
        if isinstance(result, tuple) and len(result) >= 5:
            ended = _single_bool(result[2]) or _single_bool(result[3])
        self._environment_ended = ended
        return self._read_state(), ended

    def _read_state(self) -> dict[str, Any]:
        public_state = self.env.get_public_executor_state()
        if not isinstance(public_state, Mapping):
            raise ExecutorInterfaceError("get_public_executor_state() must return a mapping")
        pose = _numbers(public_state.get("tcp_pose"), "tcp_pose", 7)
        commanded_pose = _numbers(public_state.get("commanded_tcp_pose"), "commanded_tcp_pose", 7)
        quaternion_norm = _norm(pose[3:7])
        if quaternion_norm < 1e-8:
            raise ExecutorInterfaceError("tcp_pose contains a zero quaternion")
        pose = pose[:3] + tuple(value / quaternion_norm for value in pose[3:7])
        commanded_quaternion_norm = _norm(commanded_pose[3:7])
        if commanded_quaternion_norm < 1e-8:
            raise ExecutorInterfaceError("commanded_tcp_pose contains a zero quaternion")
        commanded_pose = commanded_pose[:3] + tuple(
            value / commanded_quaternion_norm for value in commanded_pose[3:7]
        )
        velocity = _numbers(public_state.get("tcp_velocity"), "tcp_velocity", 6)
        joints = _numbers(public_state.get("joint_positions"), "joint_positions", 9)
        joint_velocities = _numbers(public_state.get("joint_velocities"), "joint_velocities", 9)
        motion_safety_armed = _single_bool(public_state.get("motion_safety_armed"))
        width_value = public_state.get("gripper_width_m")
        if isinstance(width_value, bool):
            raise ExecutorInterfaceError("gripper_width_m must be a finite measured width")
        try:
            width = float(width_value)
        except (TypeError, ValueError) as exc:
            raise ExecutorInterfaceError("gripper_width_m must be a finite measured width") from exc
        if not isfinite(width) or width < 0:
            raise ExecutorInterfaceError("gripper_width_m must be a finite non-negative measured width")
        raw_contacts = public_state.get("finger_bolt_contacts")
        if not isinstance(raw_contacts, Sequence) or isinstance(raw_contacts, (str, bytes)) or len(raw_contacts) != 2:
            raise ExecutorInterfaceError("finger_bolt_contacts must contain measured left/right booleans")
        contacts = (_single_bool(raw_contacts[0]), _single_bool(raw_contacts[1]))
        return {
            "tcp_pose": pose,
            "commanded_tcp_pose": commanded_pose,
            "tcp_velocity": velocity,
            "joint_positions": joints,
            "joint_velocities": joint_velocities,
            "motion_safety_armed": motion_safety_armed,
            "gripper_width_m": float(width),
            "finger_bolt_contacts": contacts,
        }

    def _read_insertion_force_n(self) -> float:
        wrench_reader = getattr(self.env, "public_wrench", None)
        if not callable(wrench_reader):
            raise ExecutorInterfaceError("force-bounded motion requires public_wrench() with a calibrated zero")
        baseline = getattr(self.env, "wrench_baseline_stats", None)
        if not isinstance(baseline, Mapping) or baseline.get("stationary_zero_reference_only") is not True:
            raise ExecutorInterfaceError("call set_stationary_wrench_baseline() before force-bounded motion")
        wrench = _numbers(wrench_reader(), "public_wrench", 6)
        return _norm(wrench[:3])

    def _control_parameters(self) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
        try:
            cfg = self.env.cfg
            dt = float(cfg.sim.dt) * int(cfg.decimation)
            position_scale = _three(cfg.ctrl.pos_action_threshold, "pos_action_threshold")
            rotation_scale = _three(cfg.ctrl.rot_action_threshold, "rot_action_threshold")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ExecutorInterfaceError("environment must expose Factory control timing and action thresholds") from exc
        if not isfinite(dt) or dt <= 0 or min(*position_scale, *rotation_scale) <= 0:
            raise ExecutorInterfaceError("Factory control timing and action thresholds must be positive")
        return dt, position_scale, rotation_scale

    def _reference_tolerance_for_lead(self, lead_m: float, speed_mps: float) -> float:
        """Account for _action's one-step allowance so the total lead stays bounded."""
        return max(0.0, lead_m - speed_mps * self._control_dt_s)

    def _factory_action(self, action: Sequence[float]) -> Any:
        try:
            import torch
        except ImportError:
            if hasattr(self.env, "device"):
                raise ExecutorInterfaceError("torch is required by the Factory environment step API")
            return [list(action)]
        return torch.tensor([list(action)], dtype=torch.float32, device=getattr(self.env, "device", "cpu"))

    def _stop_and_hold(self) -> None:
        if self._environment_ended:
            return
        self._set_gripper_target(self._last_gripper_target_m)
        self.env.step(self._factory_action((0.0,) * 6))

    def _set_gripper_target(self, width_m: float) -> None:
        if not isfinite(width_m) or not 0 <= width_m <= PANDA_MAX_GRIPPER_WIDTH_M:
            raise ExecutorInterfaceError("gripper width target is outside Panda's 0-80 mm range")
        self.env.set_gripper_target_width(width_m, self.plan.max_gripper_force_n)


def _valid_duration(value: Any, maximum_s: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
        and 0.0 < float(value) <= maximum_s
    )


def _numbers(value: Any, name: str, size: int) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 1:
        if not isinstance(value[0], Sequence) or isinstance(value[0], (str, bytes)):
            break
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ExecutorInterfaceError(f"{name} must contain {size} measured values")
    if any(isinstance(item, bool) for item in value):
        raise ExecutorInterfaceError(f"{name} must contain numeric values, not booleans")
    result = tuple(float(item) for item in value)
    if any(not isfinite(item) for item in result):
        raise ExecutorInterfaceError(f"{name} contains a non-finite value")
    return result


def _three(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values = (float(value),) * 3
    else:
        values = _numbers(value, name, 3)
    if any(not isfinite(item) or item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive finite values")
    return values


def _single_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) != 1:
            raise ExecutorInterfaceError("Factory single-environment done flags must have one value")
        value = value[0]
    if type(value) is not bool:
        raise ExecutorInterfaceError("Factory single-environment done flags must be booleans")
    return value


def _norm(value: Sequence[float]) -> float:
    return sqrt(sum(float(item) ** 2 for item in value))


def _limit_norm(value: Sequence[float], maximum: float) -> tuple[float, ...]:
    norm = _norm(value)
    scale = min(1.0, maximum / norm) if norm > 1e-12 else 0.0
    return tuple(float(item) * scale for item in value)


def _limit_delta_by_tracking_error(
    tracking_error: Sequence[float], delta: Sequence[float], max_tracking_error: float
) -> tuple[float, ...]:
    """Bound command-target lead over measured TCP while allowing recovery toward it."""
    lead = tuple(float(item) for item in tracking_error)
    step = tuple(float(item) for item in delta)
    current_norm = _norm(lead)
    candidate = tuple(lead[i] + step[i] for i in range(len(step)))
    candidate_norm = _norm(candidate)
    if candidate_norm <= max_tracking_error or candidate_norm < current_norm:
        return step
    if current_norm >= max_tracking_error or _norm(step) <= 1e-12:
        return (0.0,) * len(step)

    step_sq = sum(item * item for item in step)
    lead_step_dot = sum(lead[i] * step[i] for i in range(len(step)))
    constant = current_norm * current_norm - max_tracking_error * max_tracking_error
    discriminant = max(0.0, lead_step_dot * lead_step_dot - step_sq * constant)
    fraction = min(1.0, max(0.0, (-lead_step_dot + sqrt(discriminant)) / step_sq))
    return tuple(fraction * item for item in step)


def _rotation_error(target: Sequence[float], current: Sequence[float]) -> tuple[float, float, float]:
    tw, tx, ty, tz = target
    cw, cx, cy, cz = current
    error = (
        tw * cw + tx * cx + ty * cy + tz * cz,
        -tw * cx + tx * cw - ty * cz + tz * cy,
        -tw * cy + tx * cz + ty * cw - tz * cx,
        -tw * cz - tx * cy + ty * cx + tz * cw,
    )
    if error[0] < 0:
        error = tuple(-value for value in error)
    vector_norm = _norm(error[1:])
    if vector_norm < 1e-10:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * atan2(vector_norm, max(0.0, error[0]))
    return tuple(error[i + 1] * angle / vector_norm for i in range(3))


__all__ = [
    "BoltMotionPlan",
    "BoltSkillExecutor",
    "ExecutorInterfaceError",
    "TCPWaypoint",
    "make_nominal_bolt_motion_plan",
]
