"""Generic authority-preserving evaluation from role-separated action logs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_LOG_COLUMNS = {
    "case_id",
    "proposed_action",
    "executed_action",
    "utility_executed_local",
}
OPTIONAL_UTILITY_COLUMNS = {
    "utility_proposed_local",
    "utility_proposed_global",
}


def _cluster_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    reps: int,
    quantiles: tuple[float, float],
) -> tuple[float, float]:
    frame = pd.DataFrame({"value": values.astype(float), "cluster_id": cluster_ids.astype(str)})
    grouped = frame.groupby("cluster_id", sort=True)["value"].agg(["sum", "size"])
    if grouped.empty:
        return float("nan"), float("nan")
    sums = grouped["sum"].to_numpy(dtype=float)
    sizes = grouped["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    for index in range(reps):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        estimates[index] = sums[sampled].sum() / sizes[sampled].sum()
    lower, upper = np.quantile(estimates, quantiles, method="linear")
    return float(lower), float(upper)


def _metric(
    name: str,
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    reps: int,
    quantiles: tuple[float, float],
) -> dict[str, Any]:
    lower, upper = _cluster_ci(
        values,
        cluster_ids,
        seed=seed,
        reps=reps,
        quantiles=quantiles,
    )
    return {
        "name": name,
        "estimate": float(np.mean(values)),
        "ci_lower": lower,
        "ci_upper": upper,
        "n": int(len(values)),
        "available": True,
    }


def _unavailable(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "estimate": None,
        "ci_lower": None,
        "ci_upper": None,
        "n": 0,
        "available": False,
        "reason": reason,
    }


def validate_authority_log(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_LOG_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required authority-log columns: {missing}")
    if frame.empty:
        raise ValueError("Authority log is empty")
    if frame["case_id"].isna().any() or frame["case_id"].duplicated().any():
        raise ValueError("case_id must be non-null and unique")
    for column in ("proposed_action", "executed_action"):
        if frame[column].isna().any():
            raise ValueError(f"{column} must be non-null")
    numeric_columns = ["utility_executed_local"] + [
        column for column in OPTIONAL_UTILITY_COLUMNS if column in frame.columns
    ]
    output = frame.copy()
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
        if output[column].isna().any() or not np.isfinite(output[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")
    if "utility_proposed_local" in output:
        same_action = output["proposed_action"].astype(str).eq(output["executed_action"].astype(str))
        same_score = np.isclose(
            output["utility_proposed_local"].to_numpy(dtype=float),
            output["utility_executed_local"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        if bool((same_action & ~same_score).any()):
            raise ValueError("Identical proposed and selected actions must have identical same-rule scores")
    if "cluster_id" not in output:
        output["cluster_id"] = output["case_id"]
    elif output["cluster_id"].isna().any():
        raise ValueError("cluster_id must be non-null when present")
    return output


def evaluate_authority_log(
    frame: pd.DataFrame,
    *,
    seed: int = 20260710,
    reps: int = 5000,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> dict[str, Any]:
    if reps <= 0:
        raise ValueError("reps must be positive")
    data = validate_authority_log(frame)
    clusters = data["cluster_id"].astype(str).to_numpy()
    proposed = data["proposed_action"].astype(str).to_numpy()
    executed = data["executed_action"].astype(str).to_numpy()
    intervened = proposed != executed
    executed_local = data["utility_executed_local"].to_numpy(dtype=float)

    metrics = {
        "J": _metric(
            "J", executed_local, clusters,
            seed=seed, reps=reps, quantiles=quantiles,
        ),
        "intervention_rate": _metric(
            "intervention_rate", intervened.astype(float), clusters,
            seed=seed + 1, reps=reps, quantiles=quantiles,
        ),
    }

    if "utility_proposed_local" in data:
        proposed_local = data["utility_proposed_local"].to_numpy(dtype=float)
        authority_values = proposed_local - executed_local
        metrics["J_auth"] = _metric(
            "J_auth", proposed_local, clusters,
            seed=seed + 2, reps=reps, quantiles=quantiles,
        )
        metrics["Delta_authority"] = _metric(
            "Delta_authority", authority_values, clusters,
            seed=seed + 3, reps=reps, quantiles=quantiles,
        )
        if intervened.any():
            metrics["delta"] = _metric(
                "delta",
                authority_values[intervened],
                clusters[intervened],
                seed=seed + 4,
                reps=reps,
                quantiles=quantiles,
            )
        else:
            metrics["delta"] = _unavailable("delta", "no interventions observed")
    else:
        reason = "utility_proposed_local not supplied"
        metrics["J_auth"] = _unavailable("J_auth", reason)
        metrics["Delta_authority"] = (
            _metric(
                "Delta_authority", np.zeros(len(data), dtype=float), clusters,
                seed=seed + 3, reps=reps, quantiles=quantiles,
            )
            if not intervened.any()
            else _unavailable("Delta_authority", reason)
        )
        metrics["delta"] = _unavailable("delta", reason)

    if "utility_proposed_global" in data:
        proposed_global = data["utility_proposed_global"].to_numpy(dtype=float)
        metrics["J_std"] = _metric(
            "J_std", proposed_global, clusters,
            seed=seed + 5, reps=reps, quantiles=quantiles,
        )
        metrics["Delta_total"] = _metric(
            "Delta_total", proposed_global - executed_local, clusters,
            seed=seed + 6, reps=reps, quantiles=quantiles,
        )
        if "utility_proposed_local" in data:
            metrics["Delta_utility"] = _metric(
                "Delta_utility", proposed_global - data["utility_proposed_local"].to_numpy(dtype=float),
                clusters,
                seed=seed + 7,
                reps=reps,
                quantiles=quantiles,
            )
        else:
            metrics["Delta_utility"] = _unavailable(
                "Delta_utility", "utility_proposed_local not supplied"
            )
    else:
        reason = "utility_proposed_global not supplied"
        metrics["J_std"] = _unavailable("J_std", reason)
        metrics["Delta_total"] = _unavailable("Delta_total", reason)
        metrics["Delta_utility"] = _unavailable("Delta_utility", reason)

    return {
        "mode": (
            "full" if "utility_proposed_local" in data
            else "observable_plus_global" if "utility_proposed_global" in data
            else "observable_only"
        ),
        "n": int(len(data)),
        "n_clusters": int(data["cluster_id"].nunique()),
        "seed": seed,
        "bootstrap_reps": reps,
        "quantiles": list(quantiles),
        "metrics": metrics,
    }
