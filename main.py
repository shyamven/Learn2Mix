import argparse
import sys

from learn2mix.experiments.registry import EXPERIMENTS, SUPPORTED_METHODS
from learn2mix.experiments.runner import run_experiment, run_interactive_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learn2Mix experiment runner.",
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENTS.keys()),
        required=False,
        help="Experiment to run.",
    )
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default=None,
        help="Training method for classification experiments.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and methods and exit.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name in sorted(EXPERIMENTS.keys()):
            print(f"  - {name}")
        print("\nAvailable methods:")
        for method in SUPPORTED_METHODS:
            print(f"  - {method}")
        return 0

    if args.experiment is None:
        parser.error("--experiment is required unless --list is used.")

    if args.method is None and EXPERIMENTS[args.experiment]["kind"] == "classification":
        return run_interactive_experiment(args.experiment)
    return run_experiment(args.experiment, args.method)


if __name__ == "__main__":
    sys.exit(main())

