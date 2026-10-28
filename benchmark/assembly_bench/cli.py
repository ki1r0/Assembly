"""Portable entry point for peg evaluation and private calibration probes."""
import argparse
import json
from pathlib import Path
from .peg_mvp import rescore_run
from .peg_mvp_runner import evaluate, calibrate


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    rescore = commands.add_parser('rescore-peg')
    rescore.add_argument('--run', required=True)
    rescore.add_argument('--output', required=True)
    for name in ('evaluate-peg-mvp', 'calibrate-peg-mvp'):
        command = commands.add_parser(name)
        command.add_argument('--experiment-root', default=str(root / 'pipeline'))
        command.add_argument('--asset-root', required=True)
        command.add_argument('--robot', choices=('panda', 'xarm7'), default='xarm7')
        command.add_argument('--controller', choices=('reference_ik_high_pd', 'isaac_osc'), default='reference_ik_high_pd')
        command.add_argument('--split', choices=('dev', 'test') if name == 'evaluate-peg-mvp' else ('development', 'acceptance'), required=True)
        command.add_argument('--instances', default=str(root / 'benchmark/data/instances.json'))
        command.add_argument('--instance-id', action='append')
        command.add_argument('--limit', type=int)
        command.add_argument('--gpu', default='0')
        command.add_argument('--budget-s', type=float, default=1800.)
        command.add_argument('--sim-budget-s', type=float, default=600.)
        command.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'rescore-peg':
        print(json.dumps(rescore_run(args.run, args.output), indent=2))
        return 0
    if args.robot == 'xarm7' and args.controller == 'isaac_osc':
        parser.error('isaac_osc is Panda-only; xArm uses reference_ik_high_pd')
    return evaluate(args) if args.command == 'evaluate-peg-mvp' else calibrate(args)


if __name__ == '__main__':
    raise SystemExit(main())
