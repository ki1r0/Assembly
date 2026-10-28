"""Stable integration contracts for perception and goal execution adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np


class GoalExecutor(Protocol):
    """Execute typed VLM actions against one current public observation."""

    def execute_goal(
        self,
        goal: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Match ``AssemblyExecutor.execute_goal(goal, observation)`` exactly.

        Successful results carry the connector's current
        ``ResolvedGoal.as_public_dict()`` fields: ``observation_id``,
        ``feature_refs``, ``constraints``, ``target_frame``,
        ``target_geometry`` and ``summary``. Native release and safety guards
        remain inside the executor; ``terminal=True`` means no further model
        action.
        """

    def execute_tcp_delta(
        self,
        call: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Execute a bounded TCP-local adjustment through the same safety path."""


class AssemblyRuntimeAdapter(GoalExecutor, Protocol):
    """Live bridge composed by a runtime factory for the CLI entry point."""

    def observe_public(self) -> Mapping[str, Any]:
        """Return the current public RGB-D, calibration, robot-state, and features."""

    def check_task(self) -> bool:
        """Return only the independent evaluator's task-completion boolean."""

    def subscribe_rgb_frames(
        self,
        listener: Callable[[np.ndarray, float], None],
    ) -> Callable[[], None] | None:
        """Subscribe to new RGB frames with simulation-time timestamps."""

    def close(self) -> None:
        """Release runtime resources without modifying episode outcome evidence."""


__all__ = ["AssemblyRuntimeAdapter", "GoalExecutor"]
