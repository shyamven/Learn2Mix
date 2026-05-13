from typing import Optional

from .registry import EXPERIMENTS, SUPPORTED_METHODS
from .classification_engine import run_classification_experiment
from .l2m_regression_engine import run_l2m_regression_experiment


def run_experiment(experiment: str, method: Optional[str] = None) -> int:
    if experiment not in EXPERIMENTS:
        available = ", ".join(sorted(EXPERIMENTS.keys()))
        raise ValueError(f"Unknown experiment '{experiment}'. Available: {available}")

    if method is not None and method not in SUPPORTED_METHODS:
        available_methods = ", ".join(SUPPORTED_METHODS)
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: {available_methods}"
        )

    exp_cfg = EXPERIMENTS[experiment]
    kind = exp_cfg["kind"]

    if kind == "classification":
        if method is None:
            raise ValueError(
                f"Experiment '{experiment}' requires --method. Supported: {', '.join(SUPPORTED_METHODS)}"
            )
        return run_classification_experiment(experiment, method)
    if kind == "l2m_regression":
        if method is not None:
            raise ValueError(f"Experiment '{experiment}' does not support --method.")
        return run_l2m_regression_experiment(experiment)

    raise ValueError(f"Unsupported experiment kind '{kind}' for '{experiment}'")


def run_interactive_experiment(experiment: str) -> int:
    exp_cfg = EXPERIMENTS[experiment]
    if exp_cfg["kind"] != "classification":
        raise ValueError(f"Experiment '{experiment}' does not support interactive method selection.")
    chosen_method = input(f"Choose method ({', '.join(SUPPORTED_METHODS)}): ")
    if chosen_method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{chosen_method}'")
    return run_classification_experiment(experiment, chosen_method)

