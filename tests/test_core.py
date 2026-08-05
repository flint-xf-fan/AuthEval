from pathlib import Path

import numpy as np
import pandas as pd

from apeval.evaluation import evaluate_authority_log


ROOT = Path(__file__).resolve().parents[1]


def test_example_log_obeys_factorization():
    frame = pd.read_csv(ROOT / "examples" / "authority_log_example.csv")
    result = evaluate_authority_log(frame, reps=100, seed=7)
    metrics = result["metrics"]
    assert np.isclose(
        metrics["Delta_authority"]["estimate"],
        metrics["intervention_rate"]["estimate"] * metrics["delta"]["estimate"],
    )


def test_observable_only_marks_counterfactual_gap_unavailable():
    frame = pd.read_csv(ROOT / "examples" / "authority_log_example.csv").drop(
        columns=["utility_proposed_local", "utility_proposed_global"]
    )
    result = evaluate_authority_log(frame, reps=100, seed=7)
    assert result["mode"] == "observable_only"
    assert not result["metrics"]["Delta_authority"]["available"]
