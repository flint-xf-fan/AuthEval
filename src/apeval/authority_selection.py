"""Locked decision-level authority-selection experiment utilities.

The assistant consumes public image inputs. The gate consumes a separate public
metadata proxy and deterministic private workflow context. Outcomes enter only
after execution, for readout fitting or post-hoc utility evaluation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from apeval.models.vlm_cache import RISK_LEVELS
from apeval.sim import (
    ACTIONS,
    LEVEL,
    PMULT,
    SITES,
    W_BAR,
    admissible_band,
    apply_patient_overrides,
    effective_weights,
    execute,
    riskproxy_bin,
    utility,
)


PROB_COLUMNS = tuple(f"p_{level}" for level in RISK_LEVELS)
LABEL_COLUMN = "ideal_action_level"
PRIVATE_CONTEXT_COLUMNS = ("sensitivity_preference", "follow_up_burden", "invasiveness_aversion")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(seed: int, namespace: str, value: object) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()


def _stable_choice(seed: int, namespace: str, stable_key: str, choices: Sequence[str]) -> str:
    digest = _stable_digest(seed, namespace, stable_key)
    return choices[int(digest[:12], 16) % len(choices)]


def _deterministic_mode(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    winners = counts[counts.eq(counts.max())].index.astype(str)
    return sorted(winners)[0]


def pilot_image_ids(paths: Iterable[str | Path]) -> set[str]:
    ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Frozen pilot manifest does not exist: {path}")
        frame = pd.read_csv(path, usecols=["image_id"])
        ids.update(frame["image_id"].astype(str))
    return ids


def deterministic_validation_partition(
    manifest: pd.DataFrame,
    *,
    design_fraction: float,
    seed: int,
    forced_design_image_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Partition complete validation clusters, forcing prior pilot clusters to design."""
    required = {"image_id", "cluster_id", "split"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Validation manifest missing columns: {missing}")
    if not 0.0 < design_fraction < 1.0:
        raise ValueError("design_fraction must be strictly between zero and one")
    if set(manifest["split"].astype(str).unique()) != {"val"}:
        raise ValueError("Validation partition accepts val rows only")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Validation manifest contains duplicate image_id values")

    frame = manifest[["image_id", "cluster_id", "split"]].copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    forced_ids = set(map(str, forced_design_image_ids))
    missing_forced = sorted(forced_ids - set(frame["image_id"]))
    if missing_forced:
        raise ValueError(f"Pilot IDs absent from validation manifest: {missing_forced[:5]}")
    forced_clusters = set(frame.loc[frame["image_id"].isin(forced_ids), "cluster_id"])

    clusters = sorted(frame["cluster_id"].unique())
    target = max(len(forced_clusters), int(round(len(clusters) * design_fraction)))
    candidates = sorted(
        (cluster for cluster in clusters if cluster not in forced_clusters),
        key=lambda cluster: (_stable_digest(seed, "validation_partition", cluster), cluster),
    )
    design_clusters = forced_clusters | set(candidates[: max(0, target - len(forced_clusters))])
    frame["partition"] = np.where(frame["cluster_id"].isin(design_clusters), "design", "lock")
    if set(frame["partition"]) != {"design", "lock"}:
        raise ValueError("Deterministic partition did not produce both design and lock rows")
    if frame.groupby("cluster_id")["partition"].nunique().max() != 1:
        raise AssertionError("A validation cluster crossed design and lock")
    if forced_ids and not frame.loc[frame["image_id"].isin(forced_ids), "partition"].eq("design").all():
        raise AssertionError("A frozen pilot row entered the lock partition")
    return frame.sort_values("image_id").reset_index(drop=True)


def deterministic_shuffle_manifest(
    lock_manifest: pd.DataFrame,
    *,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """Pair public metadata rows with different lock images for the vision control."""
    required = {"image_id", "image_path", "split"}
    missing = sorted(required - set(lock_manifest))
    if missing:
        raise ValueError(f"Lock manifest missing shuffle columns: {missing}")
    if len(lock_manifest) < 2:
        raise ValueError("Image-shuffle control requires at least two lock rows")
    selected = lock_manifest.assign(
        _shuffle_key=lock_manifest["image_id"].astype(str).map(
            lambda image_id: _stable_digest(seed, "shuffle_control", image_id)
        )
    ).sort_values(["_shuffle_key", "image_id"]).head(min(size, len(lock_manifest))).copy()
    if len(selected) < 2:
        raise ValueError("Image-shuffle control selected fewer than two rows")

    source_ids = np.roll(selected["image_id"].astype(str).to_numpy(), 1)
    source_paths = np.roll(selected["image_path"].astype(str).to_numpy(), 1)
    selected["shuffled_from_image_id"] = source_ids
    selected["image_path"] = source_paths
    if selected["image_id"].astype(str).eq(selected["shuffled_from_image_id"]).any():
        raise AssertionError("Shuffle control contains an identity image pairing")
    public_columns = [
        "image_id", "image_path", "age", "sex", "anatomic_site", "split",
        "shuffled_from_image_id",
    ]
    return selected[[column for column in public_columns if column in selected]].reset_index(drop=True)


def validate_score_cache(
    scores: pd.DataFrame,
    expected_image_ids: Iterable[str],
    *,
    prompt_id: str,
    model_name: str,
) -> None:
    required = {"image_id", "cache_status", "prompt_id", "model_name", *PROB_COLUMNS}
    missing = sorted(required - set(scores))
    if missing:
        raise ValueError(f"Score cache missing columns: {missing}")
    if scores["image_id"].duplicated().any():
        raise ValueError("Score cache contains duplicate image IDs")
    expected = set(map(str, expected_image_ids))
    observed = set(scores["image_id"].astype(str))
    if observed != expected:
        raise ValueError(
            f"Score-cache coverage mismatch: missing={len(expected - observed)}, "
            f"extra={len(observed - expected)}"
        )
    if not scores["cache_status"].eq("ok").all():
        raise ValueError("Score cache contains non-ok rows")
    if set(scores["prompt_id"].astype(str)) != {prompt_id}:
        raise ValueError("Score cache prompt does not match the frozen prompt")
    if set(scores["model_name"].astype(str)) != {model_name}:
        raise ValueError("Score cache model does not match the frozen model")


def score_features(scores: pd.DataFrame, probability_clip: float) -> np.ndarray:
    missing = sorted(set(PROB_COLUMNS) - set(scores))
    if missing:
        raise ValueError(f"Score rows missing probability columns: {missing}")
    probs = scores[list(PROB_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("Score probabilities must be finite and nonnegative")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Score probabilities do not sum to one")
    return np.log(np.clip(probs, probability_clip, 1.0))


def prediction_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "ordinal_mae": float(np.mean(np.abs(labels - predictions))),
        "accuracy": float(np.mean(labels == predictions)),
    }


def best_constant_mae(labels: np.ndarray) -> tuple[int, float]:
    labels = np.asarray(labels, dtype=int)
    candidates = [(float(np.mean(np.abs(labels - level))), level) for level in range(4)]
    mae, level = min(candidates)
    return int(level), float(mae)


def _joined_labeled_scores(scores: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"image_id", LABEL_COLUMN, "cluster_id"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Labeled manifest missing columns: {missing}")
    joined = scores[["image_id", *PROB_COLUMNS]].merge(
        manifest[["image_id", LABEL_COLUMN, "cluster_id"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(scores):
        raise ValueError("Labeled manifest does not cover all score rows")
    return joined


def fit_calibrated_readout(
    design_scores: pd.DataFrame,
    design_manifest: pd.DataFrame,
    readout_cfg: Mapping,
    *,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Select C by design-only grouped CV and fit one frozen multinomial readout."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    joined = _joined_labeled_scores(design_scores, design_manifest)
    labels = joined[LABEL_COLUMN].to_numpy(dtype=int)
    if set(labels) != {0, 1, 2, 3}:
        raise ValueError("Design rows must contain all four action labels")
    features = score_features(joined, float(readout_cfg["probability_clip"]))
    groups = joined["cluster_id"].astype(str).to_numpy()
    folds = int(readout_cfg["cross_validation_folds"])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(features, labels, groups))

    rows = []
    for c_value in map(float, readout_cfg["regularization_c"]):
        predictions = np.full(len(joined), -1, dtype=int)
        for train_idx, val_idx in splits:
            model = LogisticRegression(
                C=c_value,
                solver=str(readout_cfg["solver"]),
                class_weight=readout_cfg["class_weight"],
                max_iter=int(readout_cfg["max_iter"]),
                random_state=seed,
            )
            model.fit(features[train_idx], labels[train_idx])
            predictions[val_idx] = model.predict(features[val_idx]).astype(int)
        if (predictions < 0).any():
            raise AssertionError("Cross-validation left rows without predictions")
        rows.append({"C": c_value, **prediction_metrics(labels, predictions)})
    cv = pd.DataFrame(rows)
    selected = sorted(
        rows,
        key=lambda row: (
            row["ordinal_mae"],
            -row["balanced_accuracy"],
            -row["macro_f1"],
            row["C"],
        ),
    )[0]
    final = LogisticRegression(
        C=float(selected["C"]),
        solver=str(readout_cfg["solver"]),
        class_weight=readout_cfg["class_weight"],
        max_iter=int(readout_cfg["max_iter"]),
        random_state=seed,
    )
    final.fit(features, labels)
    if list(map(int, final.classes_)) != [0, 1, 2, 3]:
        raise ValueError(f"Frozen readout classes are not 0..3: {final.classes_}")
    payload = {
        "type": str(readout_cfg["type"]),
        "feature_order": list(PROB_COLUMNS),
        "feature_transform": "log_clipped_probability",
        "probability_clip": float(readout_cfg["probability_clip"]),
        "classes": [int(value) for value in final.classes_],
        "coef": final.coef_.astype(float).tolist(),
        "intercept": final.intercept_.astype(float).tolist(),
        "selected_C": float(selected["C"]),
        "selection_metrics": {key: float(selected[key]) for key in ("ordinal_mae", "balanced_accuracy", "macro_f1", "accuracy")},
        "solver": str(readout_cfg["solver"]),
        "class_weight": readout_cfg["class_weight"],
        "max_iter": int(readout_cfg["max_iter"]),
        "seed": int(seed),
    }
    return payload, cv


def apply_readout(readout: Mapping, scores: pd.DataFrame) -> np.ndarray:
    features = score_features(scores, float(readout["probability_clip"]))
    coef = np.asarray(readout["coef"], dtype=float)
    intercept = np.asarray(readout["intercept"], dtype=float)
    if coef.shape != (4, 4) or intercept.shape != (4,):
        raise ValueError(f"Unexpected readout dimensions: coef={coef.shape}, intercept={intercept.shape}")
    return features @ coef.T + intercept


def competence_gate(
    lock_scores: pd.DataFrame,
    lock_manifest: pd.DataFrame,
    shuffled_scores: pd.DataFrame,
    readout: Mapping,
    competence_cfg: Mapping,
) -> dict:
    lock = _joined_labeled_scores(lock_scores, lock_manifest)
    class_scores = apply_readout(readout, lock)
    predictions = np.argmax(class_scores, axis=1).astype(int)
    labels = lock[LABEL_COLUMN].to_numpy(dtype=int)
    metrics = prediction_metrics(labels, predictions)
    constant_level, constant_mae = best_constant_mae(labels)
    counts = np.bincount(predictions, minlength=4)

    shuffled = shuffled_scores[["image_id", *PROB_COLUMNS]].merge(
        lock_manifest[["image_id", LABEL_COLUMN]],
        on="image_id",
        validate="one_to_one",
    )
    shuffled_predictions = np.argmax(apply_readout(readout, shuffled), axis=1).astype(int)
    shuffled_labels = shuffled[LABEL_COLUMN].to_numpy(dtype=int)
    shuffled_metrics = prediction_metrics(shuffled_labels, shuffled_predictions)
    original_subset = lock_scores[lock_scores["image_id"].isin(shuffled["image_id"])][["image_id", *PROB_COLUMNS]].merge(
        lock_manifest[["image_id", LABEL_COLUMN]], on="image_id", validate="one_to_one"
    )
    original_subset_predictions = np.argmax(apply_readout(readout, original_subset), axis=1).astype(int)
    original_subset_metrics = prediction_metrics(
        original_subset[LABEL_COLUMN].to_numpy(dtype=int), original_subset_predictions
    )

    observed = {
        **metrics,
        "n": int(len(lock)),
        "n_actions": int(np.sum(counts > 0)),
        "max_action_share": float(counts.max() / len(predictions)),
        "best_constant_action": constant_level,
        "best_constant_mae": constant_mae,
        "mae_improvement_over_best_constant": float(constant_mae - metrics["ordinal_mae"]),
        "shuffle_n": int(len(shuffled)),
        "shuffle_balanced_accuracy": shuffled_metrics["balanced_accuracy"],
        "shuffle_ordinal_mae": shuffled_metrics["ordinal_mae"],
        "shuffle_bacc_decrease": float(
            original_subset_metrics["balanced_accuracy"] - shuffled_metrics["balanced_accuracy"]
        ),
        "shuffle_mae_increase": float(
            shuffled_metrics["ordinal_mae"] - original_subset_metrics["ordinal_mae"]
        ),
        "original_subset_balanced_accuracy": original_subset_metrics["balanced_accuracy"],
        "original_subset_ordinal_mae": original_subset_metrics["ordinal_mae"],
        **{f"n_a{level}": int(counts[level]) for level in range(4)},
    }
    threshold = float(competence_cfg["min_shuffle_mae_increase_or_bacc_decrease"])
    checks = {
        "min_actions": observed["n_actions"] >= int(competence_cfg["min_actions"]),
        "max_action_share": observed["max_action_share"] <= float(competence_cfg["max_action_share"]),
        "balanced_accuracy": observed["balanced_accuracy"] >= float(competence_cfg["min_balanced_accuracy"]),
        "mae_improvement": observed["mae_improvement_over_best_constant"] >= float(
            competence_cfg["min_mae_improvement_over_best_constant"]
        ),
        "image_shuffle": (
            observed["shuffle_mae_increase"] >= threshold
            or observed["shuffle_bacc_decrease"] >= threshold
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "metrics": observed,
        "failure_reasons": [key for key, value in checks.items() if not value],
    }


def policy_id(bias: float) -> str:
    sign = "m" if bias < 0 else "p"
    return f"bias_{sign}{abs(int(round(float(bias) * 100))):03d}"


def rank_policy_actions(class_scores: np.ndarray, bias: float) -> np.ndarray:
    scores = np.asarray(class_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError(f"Class scores must have shape (n, 4), got {scores.shape}")
    adjusted = scores + float(bias) * np.arange(4, dtype=float)[None, :]
    return np.argsort(-adjusted, axis=1, kind="stable").astype(int)


def build_authority_context(
    manifest: pd.DataFrame,
    gate_proxy: pd.DataFrame,
    image_ids: Sequence[str],
    *,
    seed: int,
) -> pd.DataFrame:
    required_manifest = {"image_id", "cluster_id", "risk_stratum", "site_profile"}
    required_proxy = {"image_id", "s_gate"}
    if required_manifest - set(manifest):
        raise ValueError(f"Manifest missing authority columns: {sorted(required_manifest - set(manifest))}")
    if required_proxy - set(gate_proxy):
        raise ValueError(f"Gate proxy missing columns: {sorted(required_proxy - set(gate_proxy))}")
    if gate_proxy["image_id"].duplicated().any():
        raise ValueError("Gate proxy contains duplicate image IDs")
    order = pd.DataFrame({"image_id": list(map(str, image_ids)), "_order": range(len(image_ids))})
    context = order.merge(
        manifest[["image_id", "cluster_id", "risk_stratum", "site_profile"]],
        on="image_id", how="left", validate="one_to_one",
    ).merge(
        gate_proxy[["image_id", "s_gate"]],
        on="image_id", how="left", validate="one_to_one",
    ).sort_values("_order")
    if context[["cluster_id", "risk_stratum", "site_profile", "s_gate"]].isna().any().any():
        raise ValueError("Authority context is incomplete for scored image IDs")
    context["cluster_id"] = context["cluster_id"].astype(str)
    canonical = context.groupby("cluster_id", sort=True).agg(
        site_profile=("site_profile", _deterministic_mode),
        s_gate=("s_gate", "mean"),
    )
    context["site_profile"] = context["cluster_id"].map(canonical["site_profile"])
    context["s_gate"] = context["cluster_id"].map(canonical["s_gate"])

    unknown = sorted(set(context["site_profile"]) - set(SITES))
    if unknown:
        raise ValueError(f"Unknown synthetic workflow profiles: {unknown}")
    context["sensitivity_preference"] = context["cluster_id"].map(
        lambda cluster_id: _stable_choice(seed, "sens", cluster_id, list(PMULT["sensitivity_preference"]))
    )
    context["follow_up_burden"] = context["cluster_id"].map(
        lambda cluster_id: _stable_choice(seed, "burden", cluster_id, list(PMULT["follow_up_burden"]))
    )
    context["invasiveness_aversion"] = context["cluster_id"].map(
        lambda cluster_id: _stable_choice(seed, "aversion", cluster_id, list(PMULT["invasiveness_aversion"]))
    )

    shared_columns = ("site_profile", "s_gate", *PRIVATE_CONTEXT_COLUMNS)
    variation = context.groupby("cluster_id", sort=True)[list(shared_columns)].nunique(dropna=False)
    if variation.to_numpy().max(initial=0) != 1:
        raise AssertionError("Authority context is not constant within cluster_id")
    return context.reset_index(drop=True)


def evaluate_ranked_policy(
    context: pd.DataFrame,
    ranked_levels: np.ndarray,
    *,
    regime: str,
    slate_size: int,
    policy_name: str,
) -> tuple[dict, pd.DataFrame]:
    ranks = np.asarray(ranked_levels, dtype=int)
    if ranks.shape != (len(context), 4):
        raise ValueError(f"Rank matrix must have shape ({len(context)}, 4), got {ranks.shape}")
    if slate_size not in (2, 4):
        raise ValueError("Only precommitted slate sizes 2 and 4 are allowed")
    records = []
    for index, row in context.iterrows():
        slate = [ACTIONS[level] for level in ranks[index, :slate_size]]
        head = slate[0]
        proxy_bin = riskproxy_bin(float(row["s_gate"]))
        site = SITES[row["site_profile"]]
        band = admissible_band(regime, proxy_bin, int(site["cap"]), str(site["conservatism"]))
        band, _ = apply_patient_overrides(
            band,
            proxy_bin,
            str(row["follow_up_burden"]),
            str(row["invasiveness_aversion"]),
        )
        executed, intervention_type = execute(slate, band)
        w_under, w_over = effective_weights(
            str(row["site_profile"]),
            str(row["sensitivity_preference"]),
            str(row["follow_up_burden"]),
            str(row["invasiveness_aversion"]),
        )
        risk = str(row["risk_stratum"])
        global_head = utility(risk, head, W_BAR["w_under"], W_BAR["w_over"])
        local_head = utility(risk, head, w_under, w_over)
        local_executed = utility(risk, executed, w_under, w_over)
        records.append({
            "image_id": row["image_id"],
            "cluster_id": row["cluster_id"],
            "policy_id": policy_name,
            "slate_size": int(slate_size),
            "a_head": head,
            "a_exec": executed,
            "intervention_type": intervention_type,
            "intervened": bool(head != executed),
            "Ug_head": float(global_head),
            "Uz_head": float(local_head),
            "Uz_exec": float(local_executed),
            "delta_authority_i": float(local_head - local_executed),
            "delta_utility_i": float(global_head - local_head),
        })
    trace = pd.DataFrame(records)
    intervention = trace["intervened"].to_numpy(dtype=bool)
    delta_authority = trace["delta_authority_i"].to_numpy(dtype=float)
    summary = {
        "policy_id": policy_name,
        "slate_size": int(slate_size),
        "n": int(len(trace)),
        "n_clusters": int(trace["cluster_id"].nunique()),
        "J_std": float(trace["Ug_head"].mean()),
        "J_auth": float(trace["Uz_head"].mean()),
        "J": float(trace["Uz_exec"].mean()),
        "intervention_rate": float(intervention.mean()),
        "delta": float(delta_authority[intervention].mean()) if intervention.any() else float("nan"),
        "Delta_authority": float(delta_authority.mean()),
        "Delta_utility": float(trace["delta_utility_i"].mean()),
        "Delta_total": float((trace["delta_authority_i"] + trace["delta_utility_i"]).mean()),
    }
    return summary, trace


def evaluate_policy_grid(
    class_scores: np.ndarray,
    image_ids: Sequence[str],
    manifest: pd.DataFrame,
    gate_proxy: pd.DataFrame,
    biases: Sequence[float],
    *,
    regime: str,
    seed: int,
    slate_sizes: Sequence[int] = (4, 2),
) -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame]]:
    context = build_authority_context(manifest, gate_proxy, image_ids, seed=seed)
    rows: list[dict] = []
    traces: dict[tuple[int, str], pd.DataFrame] = {}
    for bias in map(float, biases):
        name = policy_id(bias)
        ranks = rank_policy_actions(class_scores, bias)
        for slate_size in slate_sizes:
            summary, trace = evaluate_ranked_policy(
                context,
                ranks,
                regime=regime,
                slate_size=int(slate_size),
                policy_name=name,
            )
            rows.append({"bias": bias, **summary})
            traces[(int(slate_size), name)] = trace
    return pd.DataFrame(rows), traces


def select_policy_winner(grid: pd.DataFrame, *, slate_size: int, objective: str) -> dict:
    if objective not in {"J_std", "J"}:
        raise ValueError(f"Unsupported selection objective: {objective}")
    eligible = grid[grid["slate_size"].eq(int(slate_size))].to_dict("records")
    if not eligible:
        raise ValueError(f"No candidate policies for slate_size={slate_size}")
    return sorted(eligible, key=lambda row: (-float(row[objective]), str(row["policy_id"])))[0]


def paired_cluster_bootstrap(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    reps: int,
    seed: int,
    quantiles: tuple[float, float],
) -> tuple[float, float]:
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float), "cluster_id": np.asarray(cluster_ids).astype(str)})
    grouped = frame.groupby("cluster_id", sort=True)["value"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(dtype=float)
    sizes = grouped["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(reps), dtype=float)
    for index in range(int(reps)):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        estimates[index] = sums[sampled].sum() / sizes[sampled].sum()
    lower, upper = np.quantile(estimates, quantiles)
    return float(lower), float(upper)


def authority_summary_with_cluster_bootstrap(
    trace: pd.DataFrame,
    *,
    reps: int,
    seed: int,
    quantiles: tuple[float, float],
) -> tuple[dict, dict]:
    """Summarize an image-level authority trace with whole-cluster uncertainty."""
    required = {
        "cluster_id",
        "intervened",
        "Uz_head",
        "Uz_exec",
        "delta_authority_i",
    }
    missing = sorted(required - set(trace))
    if missing:
        raise ValueError(f"Authority trace missing columns: {missing}")
    if trace.empty:
        raise ValueError("Authority trace is empty")

    frame = trace[list(required)].copy()
    frame["cluster_id"] = frame["cluster_id"].astype(str)
    frame["intervened"] = frame["intervened"].astype(float)
    grouped = frame.groupby("cluster_id", sort=True).agg(
        n=("cluster_id", "size"),
        J_auth_sum=("Uz_head", "sum"),
        J_sum=("Uz_exec", "sum"),
        intervention_sum=("intervened", "sum"),
        authority_sum=("delta_authority_i", "sum"),
    )

    total_n = float(grouped["n"].sum())
    total_interventions = float(grouped["intervention_sum"].sum())
    primary = {
        "estimand": "image_weighted_primary",
        "n_images": int(total_n),
        "n_clusters": int(len(grouped)),
        "J_auth": float(grouped["J_auth_sum"].sum() / total_n),
        "J": float(grouped["J_sum"].sum() / total_n),
        "intervention_rate": float(total_interventions / total_n),
        "delta": float(grouped["authority_sum"].sum() / total_interventions)
        if total_interventions
        else float("nan"),
        "Delta_authority": float(grouped["authority_sum"].sum() / total_n),
    }
    if total_interventions and not np.isclose(
        primary["Delta_authority"], primary["intervention_rate"] * primary["delta"]
    ):
        raise AssertionError("Delta_authority does not equal intervention_rate * delta")
    if not total_interventions and not np.isclose(primary["Delta_authority"], 0.0):
        raise AssertionError("Zero interventions must imply zero Delta_authority")

    arrays = {
        column: grouped[column].to_numpy(dtype=float)
        for column in (
            "n",
            "J_auth_sum",
            "J_sum",
            "intervention_sum",
            "authority_sum",
        )
    }
    bootstrap = {
        key: np.empty(int(reps), dtype=float)
        for key in ("J_auth", "J", "intervention_rate", "delta", "Delta_authority")
    }
    rng = np.random.default_rng(seed)
    for index in range(int(reps)):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        sampled_n = arrays["n"][sampled].sum()
        sampled_interventions = arrays["intervention_sum"][sampled].sum()
        sampled_authority = arrays["authority_sum"][sampled].sum()
        bootstrap["J_auth"][index] = arrays["J_auth_sum"][sampled].sum() / sampled_n
        bootstrap["J"][index] = arrays["J_sum"][sampled].sum() / sampled_n
        bootstrap["intervention_rate"][index] = sampled_interventions / sampled_n
        bootstrap["delta"][index] = (
            sampled_authority / sampled_interventions if sampled_interventions else np.nan
        )
        bootstrap["Delta_authority"][index] = sampled_authority / sampled_n

    for key, estimates in bootstrap.items():
        if np.isnan(estimates).all():
            lower, upper = np.nan, np.nan
        else:
            lower, upper = np.nanquantile(estimates, quantiles)
        primary[f"{key}_ci_lower"] = float(lower)
        primary[f"{key}_ci_upper"] = float(upper)

    cluster_authority = grouped["authority_sum"] / grouped["n"]
    sensitivity_bootstrap = np.empty(int(reps), dtype=float)
    sensitivity_rng = np.random.default_rng(seed)
    values = cluster_authority.to_numpy(dtype=float)
    for index in range(int(reps)):
        sampled = sensitivity_rng.integers(0, len(values), size=len(values))
        sensitivity_bootstrap[index] = values[sampled].mean()
    sensitivity_lower, sensitivity_upper = np.quantile(sensitivity_bootstrap, quantiles)
    sensitivity = {
        "estimand": "cluster_equal_sensitivity",
        "n_images": int(total_n),
        "n_clusters": int(len(grouped)),
        "Delta_authority": float(values.mean()),
        "Delta_authority_ci_lower": float(sensitivity_lower),
        "Delta_authority_ci_upper": float(sensitivity_upper),
    }
    return primary, sensitivity
