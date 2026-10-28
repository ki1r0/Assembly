"""Private, recoverable peg process scoring, independent of policy tool calls."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class PegCriteria:
    version: str = "cape-peg-process-v0.1-dev"
    support_evidence: str = "force"
    aperture_evidence: str = "analytic"
    stable_s: float = 1.0
    max_sample_gap_s: float = 1 / 30
    event_dwell_s: float = 0.10
    radial_m: float = 0.0002
    tilt_rad: float = 0.05
    entry_m: float = 0.010
    support_gap_m: float = 0.00025
    penetration_m: float = 0.0001
    linear_speed_mps: float = 0.002
    angular_speed_radps: float = 0.01

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in {"version", "support_evidence", "aperture_evidence"} and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.support_evidence not in {'force', 'collider_geometry'}:
            raise ValueError('Unknown support evidence type')
        if self.aperture_evidence not in {'analytic', 'collider_sweep'}:
            raise ValueError('Unknown aperture evidence type')


CHECKPOINTS = (
    "controlled", "lifted_retained", "transported", "mouth_aligned",
    "entered", "seated", "released_seated", "withdrawn_stable",
)


def scene_query_hit(hit, axis, ray_index):
    """Normalize this installed PhysX RaycastHit API (collision, not collider)."""
    return {'ray': ray_index, 'collider': str(hit.collision),
        'gap_m': float(hit.distance) - .002,
        'normal_alignment': sum(float(v) * float(a) for v, a in zip(hit.normal, axis)),
        'point_world_m': [float(v) for v in hit.position]}


def diagnostic_load_force(mode, enabled, simulation_time_s):
    if not enabled:
        return 0.
    if mode != 'peg-seat-probe':
        raise ValueError('External test load is only allowed in a privileged seat probe')
    return .2 if 0 <= simulation_time_s < .5 else 0.


def conjunction(*values):
    if any(value is False for value in values):
        return False
    return None if any(value is None for value in values) else True


def boolean(sample, key):
    value = sample.get(key)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{key} must be bool or null")
    return value


def compare(sample, key, bound, *, minimum=False, absolute=False):
    value = sample.get(key)
    if value is None:
        return None
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{key} must be finite or null")
    if absolute:
        value = abs(value)
    return value >= bound if minimum else value <= bound


class PegProcessEvaluator:
    """Track current and historical states without latching recoverable failures.

    Input is private measurement evidence. Unknown samples break dwell; no elapsed
    time is inferred across gaps. Calibration probes never gain process provenance.
    """

    def __init__(self, criteria=None, *, calibrated=False, privileged=False, initial_state_valid=None):
        self.criteria = criteria or PegCriteria()
        self.calibrated = calibrated
        self.privileged = privileged
        if initial_state_valid is not None and not isinstance(initial_state_valid, bool):
            raise ValueError('Initial-state validity must be bool or null')
        self.initial_state_valid = initial_state_valid
        self.last_time = None
        self.seen_control = False
        self.events = []
        self.window = deque()
        self.window_interrupted = False
        self.states = {name: {"current": None, "ever_reached": False,
            "first_reached_time_s": None, "last_reached_time_s": None,
            "total_valid_duration_s": 0.0, "max_contiguous_duration_s": 0.0,
            "lost_count": 0, "since_s": None} for name in CHECKPOINTS}
        self.latest = {}
        self.max_entry_m = None
        self.max_valid_entry_m = None
        self.minimum_radial_m = None

    def observe(self, sample: Mapping):
        t = float(sample["time_s"])
        if not isfinite(t) or (self.last_time is not None and t <= self.last_time):
            raise ValueError("Measurement times must be finite and strictly increasing")
        c = self.criteria
        dt = 0.0 if self.last_time is None else t - self.last_time
        gap = self.last_time is not None and dt > c.max_sample_gap_s + 1e-9
        if gap:
            self.window.clear()
            self.window_interrupted = True
            self.events.append({"time_s": t, "event": "measurement_gap", "duration_s": dt})
        controlled = boolean(sample, "target_controlled")
        control_since = self.states['controlled']['since_s']
        if controlled is True and not gap and control_since is not None:
            self.seen_control |= t - control_since >= c.event_dwell_s - 1e-9
        actual_aperture = c.aperture_evidence == 'collider_sweep'
        fit = boolean(sample, "casing_sweep_clear" if actual_aperture else "entered_selected_casing_opening")
        correct = conjunction(boolean(sample, "through_selected_cover_hole"), fit)
        radial_aligned = compare(sample, "shaft_radial_error_m", c.radial_m)
        if actual_aperture and fit is True and compare(sample, 'casing_entry_depth_m', c.entry_m, minimum=True) is True:
            radial_aligned = True
        aligned = conjunction(radial_aligned,
                              compare(sample, "axis_tilt_rad", c.tilt_rad))
        entered = conjunction(correct, compare(sample, "casing_entry_depth_m", c.entry_m, minimum=True))
        seated = conjunction(entered, aligned,
            compare(sample, "cap_support_gap_m", c.support_gap_m, absolute=True),
            boolean(sample, "cap_support_contact" if c.support_evidence == 'force' else 'cap_bearing_geometry_supported'),
            compare(sample, "non_casing_penetration_bound_m" if actual_aperture else "penetration_m", c.penetration_m),
            None if boolean(sample, "premature_bottoming") is None else not sample["premature_bottoming"])
        provenance = self.seen_control and not self.privileged
        released = conjunction(provenance, boolean(sample, "robot_released"))
        stable = conjunction(seated, boolean(sample, "robot_released"), boolean(sample, "robot_withdrawn"),
            compare(sample, "relative_linear_speed_mps", c.linear_speed_mps, absolute=True),
            compare(sample, "relative_angular_speed_radps", c.angular_speed_radps, absolute=True))
        self.window.append((t, stable))
        # Retain the sample bracketing the beginning of the requested time window.
        while len(self.window) > 1 and self.window[1][0] <= t - c.stable_s:
            self.window.popleft()
        window_ready = (t - self.window[0][0] >= c.stable_s - 1e-9)
        if window_ready:
            self.window_interrupted = False
        settled = conjunction(stable, window_ready,
                              conjunction(*(v for _, v in self.window)))
        if self.window_interrupted and stable is not False:
            settled = None
        seat_predicate_stable = settled
        settled = conjunction(settled, provenance)
        values = (controlled, conjunction(controlled, boolean(sample, "lifted_from_support")),
            conjunction(controlled, boolean(sample, "in_target_neighborhood")), aligned,
            entered, seated, conjunction(seated, released), settled)
        for name, value in zip(CHECKPOINTS, values):
            state = self.states[name]
            previous = state["current"]
            if gap:
                state["since_s"] = None
            if value is True:
                if previous is not True or state["since_s"] is None:
                    state["since_s"] = t
                duration = t - state["since_s"]
                if previous is True and not gap:
                    state["total_valid_duration_s"] += dt
                state["max_contiguous_duration_s"] = max(state["max_contiguous_duration_s"], duration)
                dwell = 0 if name == "withdrawn_stable" else c.event_dwell_s
                if duration + 1e-9 >= dwell:
                    if not state["ever_reached"]:
                        state["first_reached_time_s"] = t
                        self.events.append({"time_s": t, "checkpoint": name, "event": "reached"})
                    state["ever_reached"] = True
                    state["last_reached_time_s"] = t
            else:
                if previous is True:
                    state["lost_count"] += 1
                    self.events.append({"time_s": t, "checkpoint": name,
                                        "event": "lost" if value is False else "evidence_unknown"})
                state["since_s"] = None
            state["current"] = value
        for key, attr, fn in (("casing_entry_depth_m", "max_entry_m", max),
                              ("shaft_radial_error_m", "minimum_radial_m", min)):
            value = sample.get(key)
            if value is not None:
                value = float(value)
                if not isfinite(value):
                    raise ValueError(f"{key} must be finite")
                old = getattr(self, attr)
                setattr(self, attr, value if old is None else fn(old, value))
        if correct is True and sample.get('casing_entry_depth_m') is not None:
            depth = float(sample['casing_entry_depth_m'])
            self.max_valid_entry_m = depth if self.max_valid_entry_m is None else max(self.max_valid_entry_m, depth)
        goals = [correct, conjunction(entered, aligned), seated, released, settled]
        valid = (self.calibrated and not self.privileged and self.initial_state_valid is True
                 and all(v is not None for v in goals))
        self.latest = {"time_s": t, "assembly_success": settled if valid else None,
            "evaluation_status": "privileged_diagnostic" if self.privileged else
                "invalid_scene" if self.initial_state_valid is False else
                "uncalibrated" if not self.calibrated else "valid" if valid else "insufficient_evidence",
            "candidate_physical_success": settled, "goal_predicates": goals,
            "seat_predicate_stable": seat_predicate_stable,
            "q_goal_final_bounds": [sum(v is True for v in goals) / 5,
                                    sum(v is not False for v in goals) / 5]}
        self.last_time = t
        return dict(self.latest)

    def result(self, *, termination_reason=None):
        return {"evaluator_version": self.criteria.version,
            "criteria": asdict(self.criteria), **self.latest,
            "termination_reason": termination_reason,
            "process_provenance": self.seen_control and not self.privileged,
            "initial_state_valid": self.initial_state_valid,
            "checkpoints": {key: {k: v for k, v in value.items() if k != "since_s"}
                            for key, value in self.states.items()},
            "max_entry_m": self.max_entry_m, "minimum_radial_m": self.minimum_radial_m,
            "max_valid_entry_m": self.max_valid_entry_m,
            "max_entry_scope": "max_entry_m is raw axial projection, including off-hole poses; max_valid_entry_m requires the selected cover hole and supported casing fit"}


def rescore_run(run_dir, output_dir):
    """Rescore recorded measurements only; never infer missing contact evidence."""
    import html
    import json
    from pathlib import Path

    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    source = run / "private/peg-process-measurements.jsonl"
    summary = json.loads((run / "episode/backend-summary.json").read_text())
    if summary.get("task") != "peg":
        raise ValueError("Peg scorer cannot score another task")
    instance_path = run / 'checks/peg-mvp-instance.json'
    instance = json.loads(instance_path.read_text()) if instance_path.exists() else {}
    evaluator = PegProcessEvaluator(privileged=bool(summary.get("privileged_setup_diagnostic")) or instance.get('privileged') is True)
    criteria_path = run / 'checks/peg-mvp-criteria.json'
    if criteria_path.exists():
        evaluator.criteria = PegCriteria(**json.loads(criteria_path.read_text()))
    plot_rows = []
    with source.open() as stream:
        for line in stream:
            sample = json.loads(line)
            if evaluator.last_time is None:
                evaluator.initial_state_valid = boolean(sample, 'initial_state_valid')
                if not criteria_path.exists():
                    mode = sample.get('support_evidence', 'force')
                    evaluator.criteria = PegCriteria(support_evidence=mode,
                        version='cape-peg-process-v0.2-geometry-dev' if mode == 'collider_geometry' else 'cape-peg-process-v0.1-dev')
            evaluator.observe(sample)
            if not plot_rows or sample['time_s'] - plot_rows[-1]['time_s'] >= .2 - 1e-9:
                plot_rows.append(sample)
    if evaluator.last_time is None:
        raise ValueError("No recorded measurements; no score produced")
    result = evaluator.result(termination_reason=(summary.get("terminal") or {}).get("source"))
    if summary.get('error') or result['termination_reason'] == 'backend_error':
        result.update(evaluation_status='runtime_error', assembly_success=None)
    physics_path = run / 'checks/physics-backend.json'
    physics = json.loads(physics_path.read_text()) if physics_path.exists() else {}
    control_path = run / 'private/control.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else {}
    terminal_time = (summary.get('terminal') or {}).get('monotonic')
    decision_time = control.get('decision_started_monotonic')
    dt = physics.get('physics_dt_s')
    steps = summary.get('physics_steps')
    result['timing'] = {'physics_steps': steps, 'physics_dt_s': dt,
        'control_dt_s': physics.get('control_dt_s'),
        'simulation_time_s': steps * dt if steps is not None and dt is not None else None,
        'decision_wall_time_s': terminal_time - decision_time
            if terminal_time is not None and decision_time is not None else None}
    result['final_measurements'] = {key: sample.get(key) for key in (
        'through_selected_cover_hole', 'entered_selected_casing_opening',
        'casing_entry_depth_m', 'shaft_radial_error_m', 'axis_tilt_rad',
        'cap_support_gap_m', 'penetration_m', 'cap_support_contact',
        'cap_bearing_geometry_supported', 'target_finger_forces_n')}
    result.update(source_run=str(run), legacy_native_success=summary.get("native_success"),
                  evidence_scope="Recorded measurements; calibration pending, not a certified success rate")
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (output / "events.jsonl").open("x") as stream:
        for event in evaluator.events:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
        (name, state["current"], state["ever_reached"], state["first_reached_time_s"],
         state["max_contiguous_duration_s"], state["lost_count"])) + "</tr>"
        for name, state in result["checkpoints"].items())
    charts = []
    for key, label, scale in (
            ('casing_entry_depth_m', 'Entry depth in selected hole / mm', 1000.),
            ('shaft_radial_error_m', 'Radial error / mm', 1000.),
            ('cap_support_gap_m', 'Cap support gap / mm', 1000.),
            ('target_finger_forces_n', 'Maximum target-filtered finger force / N', 1.)):
        segments = [[]]
        for row in plot_rows:
            value = row.get(key)
            applicable = key != 'casing_entry_depth_m' or (
                row.get('through_selected_cover_hole') and row.get('entered_selected_casing_opening'))
            if value is None or not applicable:
                if segments[-1]:
                    segments.append([])
                continue
            if isinstance(value, list):
                value = max(value, default=0.)
            segments[-1].append((row['time_s'], float(value) * scale))
        points = [point for segment in segments for point in segment]
        if not points:
            charts.append('<p>' + html.escape(label) + ': no applicable samples</p>')
            continue
        t0, t1 = plot_rows[0]['time_s'], plot_rows[-1]['time_s']
        low, high = min(v for _, v in points), max(v for _, v in points)
        paths = [' '.join(f'{40 + 680 * (t - t0) / max(t1 - t0, 1e-9):.2f},'
                         f'{140 - 110 * (v - low) / max(high - low, 1e-9):.2f}' for t, v in segment)
                 for segment in segments if segment]
        charts.append('<h2>' + html.escape(label) + '</h2><svg viewBox="0 0 760 180" role="img" aria-label="'
            + html.escape(label) + '"><path d="M40 20 V140 H730" fill="none" stroke="#777"/>'
            + ''.join('<polyline points="' + path + '" fill="none" stroke="#245d79"/>' for path in paths)
            + f'<text x="40" y="165">{t0:.2f}s to {t1:.2f}s; range {low:.5g} to {high:.5g}</text></svg>')
    text = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Peg episode</title>"
        "<h1>CAPE Peg development report</h1><p>"
        + html.escape(str({key: result.get(key) for key in
            ("evaluation_status", "assembly_success", "candidate_physical_success", "termination_reason")}))
        + "</p><p>Not a calibrated score. Raw evidence remains in the source run. Curves are sampled at 5Hz; event timing uses full-rate measurements.</p>"
        + "<table border='1'><tr><th>Checkpoint</th><th>Current</th><th>Ever reached</th>"
        "<th>First time/s</th><th>Longest duration/s</th><th>Lost count</th></tr>"
        + rows + '</table>' + ''.join(charts) + "</html>")
    (output / "episode_report.html").write_text(text)
    return result
