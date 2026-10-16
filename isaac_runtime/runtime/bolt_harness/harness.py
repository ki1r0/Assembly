"""Serial model-to-executor loop for the bolt-insertion harness."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .agent import (
    BoltHarnessAgent, _TOOL_STATUSES, _available_evidence_refs, _project_observation, parse_tool_call,
)


MAX_TOOL_CALLS = 32
_ACTION_SUCCESS = {"completed", "succeeded"}


@dataclass(frozen=True)
class BoltHarnessResult:
    """Runner outcome; held/seat_ready are the latest pre-release check values."""

    status: str
    exit_code: int
    calls: int
    held: bool
    seat_ready: bool
    task_success: bool
    history: tuple[dict[str, Any], ...]
    error: str | None = None
    private_exception: BaseException | None = field(default=None, repr=False, compare=False)
    help_request: dict[str, Any] | None = None


class BoltHarnessLoop:
    """Run bounded public model decisions against synchronous executor methods.

    ``observation_callback`` returns the current public RGB frames and public robot /
    wrench state. ``check_task_callback`` returns exactly ``{held, seat_ready}``
    before release and exactly ``{task_success}`` afterwards. These are independent
    booleans; only their phase-specific completion projection enters model history.
    ``private_exception`` retains a caught exception and traceback for runner
    diagnostics; it is never copied into the public trace.

    The executor must implement ``execute_skill``. Nudge and retract calls are
    dispatched only when the corresponding executor method exists. Model calls
    and actions are serial: the environment must remain paused during inference.
    """

    def __init__(
        self,
        agent: BoltHarnessAgent,
        executor: Any,
        *,
        task_spec: Mapping[str, Any],
        observation_callback: Callable[[], Mapping[str, Any]],
        check_task_callback: Callable[[], Mapping[str, Any]],
        trace_callback: Callable[[dict[str, Any]], None],
        help_request_callback: Callable[[Mapping[str, Any]], None] | None = None,
        max_calls: int = 16,
    ) -> None:
        if not callable(getattr(agent, "next_tool_call", None)):
            raise TypeError("agent must expose next_tool_call()")
        if not callable(getattr(executor, "execute_skill", None)):
            raise TypeError("executor must expose execute_skill()")
        if not isinstance(task_spec, Mapping):
            raise TypeError("task_spec must be a public mapping")
        if not callable(observation_callback) or not callable(check_task_callback):
            raise TypeError("observation and check_task callbacks must be callable")
        if not callable(trace_callback):
            raise TypeError("trace_callback must be callable")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or not 1 <= max_calls <= MAX_TOOL_CALLS:
            raise ValueError(f"max_calls must be in [1, {MAX_TOOL_CALLS}]")

        self.agent = agent
        self.executor = executor
        self.task_spec = dict(task_spec)
        self.observation_callback = observation_callback
        self.check_task_callback = check_task_callback
        self.trace_callback = trace_callback
        if help_request_callback is not None and not callable(help_request_callback):
            raise TypeError("help_request_callback must be callable")
        self.help_request_callback = help_request_callback
        self.max_calls = max_calls
        self._history: list[dict[str, Any]] = []
        self._calls = 0
        self._held = False
        self._seat_ready = False
        self._task_success = False
        self._help_request: dict[str, Any] | None = None
        self._released = False
        self._release_authorized = False
        self._completed_signal = False
        self._started = False

        previous_trace = getattr(agent, "trace_callback", None)

        def trace_model_event(event: dict[str, Any]) -> None:
            if callable(previous_trace) and previous_trace is not trace_callback:
                previous_trace(event)
            trace_callback(event)

        try:
            agent.trace_callback = trace_model_event
        except (AttributeError, TypeError) as exc:
            raise TypeError("agent must allow installation of its raw request/response trace callback") from exc

    def run(self) -> BoltHarnessResult:
        """Run one episode loop; calls counts decisions, with agent retry still bounded to one."""
        if self._started:
            raise RuntimeError("BoltHarnessLoop instances can run only once")
        self._started = True
        try:
            observation = self._observe()
            while self._calls < self.max_calls:
                self._calls += 1
                raw_call = self.agent.next_tool_call(
                    observation,
                    self.task_spec,
                    self._history,
                    completed=self._completed_signal,
                )
                call = parse_tool_call(
                    json.dumps(raw_call),
                    completed=self._completed_signal,
                    task_spec=self.task_spec,
                    available_evidence_refs=_available_evidence_refs(observation, self._history),
                )
                tool = call["tool"]

                if tool == "stop":
                    self._record(call, {"status": "stopped"})
                    return self._finish("stopped", error="model stopped before evaluator-confirmed task success")

                if tool == "observe":
                    observation = self._observe()
                    self._record(call, {"status": "completed", "observation": observation})
                    continue

                if tool == "check_task":
                    completed = self._check_task()
                    self._record(call, {"completed": completed})
                    if self._released and self._task_success:
                        return self._finish("success")
                    continue

                if tool == "send_help_request":
                    if self.help_request_callback is None:
                        self._record(call, {"status": "invalid_action"})
                        return self._finish("failed", error="local help-request outbox is unavailable")
                    request = {
                        "request_id": f"help-{self._calls:02d}",
                        **{key: value for key, value in call.items() if key != "tool"},
                    }
                    self.help_request_callback(request)
                    self._help_request = request
                    self._record(call, {"status": "completed", "delivery": "local_outbox", "request": request})
                    return self._finish("help_requested")

                if tool == "execute_skill" and call["skill"] == "release_and_retract":
                    if not (self._release_authorized and self._held and self._seat_ready):
                        self._record(call, {"status": "invalid_action"})
                        return self._finish("failed", error="release requires a current held and seat_ready check")

                action_result = self._dispatch_action(call)
                status = action_result["status"]
                if (status in {"force_limit", "stalled"} and tool == "execute_skill"
                        and call.get("skill") == "insert_and_seat"):
                    self._release_authorized = False
                    self._completed_signal = False
                    observation = self._observe()
                    self._record(call, {"status": status, "observation": observation})
                    contacts = observation.get("finger_bolt_contacts")
                    if (not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes))
                            or len(contacts) != 2 or any(type(contact) is not bool for contact in contacts)
                            or not all(contacts)):
                        return self._finish(
                            "failed", error=f"insertion {status} did not preserve bilateral public grasp contact"
                        )
                    continue
                if status not in _ACTION_SUCCESS:
                    self._record(call, {"status": status})
                    return self._finish("failed", error=f"executor returned {status}")

                self._release_authorized = False
                self._completed_signal = False
                if tool == "execute_skill" and call["skill"] == "release_and_retract":
                    self._released = True
                observation = self._observe()
                self._record(call, {"status": status, "observation": observation})

            return self._finish("max_calls", error="maximum model tool-call count reached")
        except Exception as exc:
            return self._finish("failed", error="harness operation failed", private_exception=exc)

    def _observe(self) -> Mapping[str, Any]:
        observation = self.observation_callback()
        if not isinstance(observation, Mapping):
            raise TypeError("observation callback must return a public mapping")
        return observation

    def _check_task(self) -> bool:
        state = self.check_task_callback()
        if not isinstance(state, Mapping):
            raise TypeError("check_task callback must return a mapping of independent booleans")
        if self._released:
            if set(state) != {"task_success"} or type(state["task_success"]) is not bool:
                raise ValueError("post-release check_task must return only boolean task_success")
            self._task_success = state["task_success"]
            self._completed_signal = self._task_success
            return self._task_success

        if set(state) != {"held", "seat_ready"}:
            raise ValueError("pre-release check_task must return only held and seat_ready booleans")
        if type(state["held"]) is not bool or type(state["seat_ready"]) is not bool:
            raise ValueError("pre-release held and seat_ready must be booleans")
        self._held = state["held"]
        self._seat_ready = state["seat_ready"]
        self._release_authorized = self._held and self._seat_ready
        self._completed_signal = self._release_authorized
        return self._completed_signal

    def _dispatch_action(self, call: Mapping[str, Any]) -> dict[str, str]:
        tool = call["tool"]
        if tool == "execute_skill":
            result = self.executor.execute_skill(
                call["skill"],
                call["target_id"],
                mode=call["mode"],
                max_duration_s=call["max_duration_s"],
                completed=self._completed_signal,
            )
        elif tool in {"nudge", "retract"}:
            method = getattr(self.executor, tool, None)
            if not callable(method):
                return {"status": "invalid_action"}
            arguments = {key: value for key, value in call.items() if key != "tool"}
            result = method(**arguments)
        else:
            raise ValueError(f"unsupported action tool: {tool}")

        if not isinstance(result, Mapping):
            raise TypeError(f"executor {tool}() must return a status mapping")
        status = result.get("status")
        if not isinstance(status, str) or status not in _TOOL_STATUSES:
            raise ValueError(f"executor {tool}() returned an invalid status")
        return {"status": status}

    def _record(self, call: Mapping[str, Any], result: dict[str, Any]) -> None:
        item = {"call": dict(call), "result": result}
        self._history.append(item)
        trace_result = dict(result)
        if "observation" in trace_result:
            trace_result["observation"] = _trace_observation(trace_result["observation"])
        self.trace_callback(
            {
                "event": "tool_result",
                "call": dict(call),
                "result": trace_result,
                "history": _trace_history(self._history),
            }
        )

    def _finish(
        self,
        status: str,
        *,
        error: str | None = None,
        private_exception: BaseException | None = None,
    ) -> BoltHarnessResult:
        success = status == "success" and self._released and self._task_success
        result = BoltHarnessResult(
            status="success" if success else status,
            exit_code=0 if success else 1,
            calls=self._calls,
            held=self._held,
            seat_ready=self._seat_ready,
            task_success=self._task_success if self._released else False,
            history=tuple(dict(item) for item in self._history),
            error=None if success else error,
            private_exception=private_exception,
            help_request=self._help_request,
        )
        try:
            self.trace_callback({
                "event": "run_finished", "status": result.status, "exit_code": result.exit_code,
                "held": result.held, "seat_ready": result.seat_ready,
                "task_success": result.task_success, "calls": result.calls,
            })
        except Exception as exc:
            return BoltHarnessResult(
                status="failed", exit_code=1, calls=result.calls, held=result.held,
                seat_ready=result.seat_ready, task_success=result.task_success, history=result.history,
                error="trace callback failed",
                private_exception=private_exception if private_exception is not None else exc,
                help_request=result.help_request,
            )
        return result


def _trace_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return _project_observation(observation)[0]


def _trace_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_history = []
    for item in history:
        result = dict(item["result"])
        if "observation" in result:
            result["observation"] = _trace_observation(result["observation"])
        safe_history.append({"call": dict(item["call"]), "result": result})
    return safe_history


__all__ = ["BoltHarnessLoop", "BoltHarnessResult", "MAX_TOOL_CALLS"]
