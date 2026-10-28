"""Static public-USD export; does not import or launch Isaac/PhysX."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1_gripper_geometry import extract_r1_gripper_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "checks/r1_gripper_geometry.json")
    args = parser.parse_args()
    geometry = {arm: extract_r1_gripper_geometry(args.usd_path, arm).to_dict()
                for arm in ("left", "right")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(geometry, indent=2) + "\n")
    print(f"Exported both arms to {args.output}; pad/contact calibration unknown")


if __name__ == "__main__":
    main()
