"""Instance-driven orchestration of the latest ours launcher, not a new policy."""
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

from .peg_mvp import rescore_run


def selected_instances(path, split, limit=None, instance_ids=None):
    rows = json.loads(Path(path).read_text())["instances"]
    identifiers = [row["instance_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate instance IDs")
    selected = [row for row in rows if row["split"] == split]
    if instance_ids:
        if set(instance_ids)-{row['instance_id'] for row in selected}:
            raise ValueError('Requested instance is absent from the selected split')
        selected=[row for row in selected if row['instance_id'] in instance_ids]
    for row in selected:
        if row["case"] not in {"nominal-aligned", "bolt-horizontal-pose"}:
            raise ValueError("Unsupported peg variant")
        xy = row["xy_m"]
        if len(xy) != 2 or not all(math.isfinite(v) and abs(v) <= .01 for v in xy):
            raise ValueError("Invalid initial XY offset")
        if Path(row["instance_id"]).name != row["instance_id"] or row["instance_id"] in {".", ".."}:
            raise ValueError("Instance IDs must be plain directory names")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    if not selected:
        raise ValueError("No selected instances")
    return selected[:limit] if limit else selected


def evaluate(args):
    rows = selected_instances(args.instances, args.split, args.limit, args.instance_id)
    return _run_trials(args, rows, privileged=False)


def calibration_rows(split):
    if split not in {'development', 'acceptance'}:
        raise ValueError('Unknown calibration split')
    counts = (10, 10, 10) if split == 'development' else (5, 3, 2)
    seed = 0 if split == 'development' else 100
    rows = []
    for label, count, mode, inverted, expected in (
            ('positive', counts[0], 'peg-seat-probe', False, True),
            ('offset', counts[1], 'peg-offset-probe', False, False),
            ('inverted', counts[2], 'peg-seat-probe', True, False)):
        for index in range(count):
            rows.append({'instance_id':f'{split}-{label}-{index:02d}',
                'case':'nominal-aligned', 'xy_m':[0.,0.], 'probe_mode':mode,
                'probe_seed':seed, 'probe_inverted':inverted, 'expected_seat':expected})
            seed += 1
    return rows


def calibrate(args):
    rows = calibration_rows(args.split)
    if getattr(args, 'instance_id', None):
        if set(args.instance_id)-{row['instance_id'] for row in rows}:
            raise ValueError('Unknown calibration instance')
        rows = [row for row in rows if row['instance_id'] in args.instance_id]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('Positive limit required')
        rows = rows[:args.limit]
    return _run_trials(args, rows, privileged=True)


def endpoint_calibration_match(record):
    if record.get('returncode') != 0 or record.get('termination_reason') != 'evaluator_calibration_complete':
        return None
    actual = record.get('seat_predicate_stable')
    return actual is record['expected_seat'] if isinstance(actual, bool) else None


def aggregate_trials(records, planned_count):
    scored=[row for row in records if row.get('evaluation_status')=='valid'
            and isinstance(row.get('assembly_success'),bool)]
    successes=sum(row['assembly_success'] for row in scored)
    return {'planned_count':planned_count,'ended_count':sum(row.get('status')!='running' for row in records),
        'valid_scored_count':len(scored),'verified_success_count':successes,
        'unscored_ended_count':sum(row.get('status')!='running' and row not in scored for row in records),
        'success_rate':successes/len(scored) if scored else None,
        'success_rate_wilson_95':_wilson_interval(successes,len(scored)),
        'scored_coverage':len(scored)/planned_count,
        'verified_delivery_rate':successes/planned_count}


def _run_trials(args, rows, *, privileged):
    experiment = Path(args.experiment_root).resolve()
    if not (experiment / "launch.py").is_file() or not (experiment / ".venv/bin/python").is_file():
        raise ValueError("Latest experiment launcher and its Python environment are required")
    if not all(math.isfinite(value) and value > 0 for value in (args.budget_s, args.sim_budget_s)):
        raise ValueError("Positive finite wall-clock and simulation budgets required")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for line in ((experiment / ".env").read_text().splitlines() if (experiment / ".env").is_file() else []):
        fields = shlex.split(line, comments=True)
        if fields and fields[0] == "export":
            fields = fields[1:]
        if fields and "=" in fields[0]:
            key, value = fields[0].split("=", 1)
            env[key] = value
    env["ASSEMBLY_ASSET_ROOT"] = str(Path(args.asset_root).resolve())
    records = []; finished = False
    def save():
        temporary = root / "batch-status.json.tmp"
        temporary.write_text(json.dumps({"suite": "cape_assembly_peg_mvp_v01",
            "policy": "none_privileged_initializer" if privileged else "latest_ours",
            "status": "finished_requires_review" if finished else "running",
            "split": args.split, "trials": records,
            "scope_detail": "Endpoint predicate calibration, not robot assembly or process validation" if privileged else "Policy rollout",
            "metrics":None if privileged else aggregate_trials(records,len(rows)),
            "endpoint_calibration_review":{'planned':len(rows),
                'matching_expected':sum(row.get('calibration_match') is True for row in records),
                'mismatching_expected':sum(row.get('calibration_match') is False for row in records),
                'note':'Automatic matches still require physical review; this does not certify process success'} if privileged else None,
            "scope": "Physical attempts; calibration-pending scores are UNKNOWN"}, indent=2))
        temporary.replace(root / "batch-status.json")
    for row in rows:
        run = root / row["instance_id"]
        command = [str(experiment / ".venv/bin/python"), str(experiment / "launch.py"),
            "--peg-mvp", "--peg-mvp-module", str(Path(__file__).with_name('peg_mvp.py').resolve()),
            "--robot", args.robot, "--task", "peg", "--method", "ours",
            "--controller", args.controller, "--case", row["case"],
            "--peg-offset-m", *map(str, row["xy_m"]), "--gpu", str(args.gpu),
            "--peg-sim-budget-s", str(args.sim_budget_s),
            "--budget-s", str(args.budget_s), "--run-dir", str(run)]
        if privileged:
            command += ['--verifier-calibration',row['probe_mode'],'--peg-probe-seed',str(row['probe_seed'])]
            if row['probe_inverted']:
                command += ['--peg-probe-inverted']
        else:
            command += ['--rollout']
        record = {**row, "status": "running", "run_dir": str(run), "command": command}
        records.append(record)
        save()
        with (root / (row["instance_id"] + ".log")).open("x") as stream:
            result = subprocess.run(command, env=env, cwd=experiment, stdout=stream, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, status="ended")
        try:
            score = rescore_run(run, root / (row["instance_id"] + "-score"))
            record.update({key: score.get(key) for key in
                ("assembly_success", "evaluation_status", "termination_reason", "seat_predicate_stable", "evaluator_version")})
            if privileged:
                record['calibration_match']=endpoint_calibration_match(record)
        except (OSError, ValueError, KeyError) as error:
            record.update(status="scoring_unavailable", assembly_success=None, error=str(error))
        save()
    finished = True
    save()
    with (root / "batch_summary.csv").open("x", newline="") as stream:
        fields = ["instance_id", "case", "status", "assembly_success", "evaluation_status", "termination_reason", "seat_predicate_stable", "expected_seat", "calibration_match"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return 0


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials <= 0:
        return None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]
