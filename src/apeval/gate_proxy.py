"""Assistant-independent public gate proxies for the controlled ISIC study."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from apeval.sim import RISK_SCORE_VECTOR, operational_head_from_score


PUBLIC_METADATA_FEATURES = ("age", "sex", "anatomic_site")
FORBIDDEN_GATE_FEATURES = {
    "diagnosis", "risk_stratum", "site_profile", "site_id", "ideal_action_level",
    "label", "y", "r_true", "ell_star", "utility", "s_assistant",
}


def _normalize_category(series: pd.Series) -> pd.Series:
    normalized = series.fillna("__missing__").astype(str).str.strip().str.lower()
    return normalized.replace({"": "__missing__", "nan": "__missing__", "none": "__missing__"})


@dataclass(frozen=True)
class MetadataSchema:
    age_mean: float
    age_scale: float
    sex_categories: tuple[str, ...]
    anatomic_site_categories: tuple[str, ...]
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "age_mean": self.age_mean,
            "age_scale": self.age_scale,
            "sex_categories": list(self.sex_categories),
            "anatomic_site_categories": list(self.anatomic_site_categories),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "MetadataSchema":
        return cls(
            age_mean=float(payload["age_mean"]),
            age_scale=float(payload["age_scale"]),
            sex_categories=tuple(payload["sex_categories"]),
            anatomic_site_categories=tuple(payload["anatomic_site_categories"]),
            feature_names=tuple(payload["feature_names"]),
        )


def fit_metadata_schema(train: pd.DataFrame) -> MetadataSchema:
    missing = sorted(set(PUBLIC_METADATA_FEATURES) - set(train.columns))
    if missing:
        raise ValueError(f"Metadata gate training frame missing public features: {missing}")
    age = pd.to_numeric(train["age"], errors="coerce")
    age_mean = float(age.mean()) if age.notna().any() else 0.0
    age_scale = float(age.std(ddof=0)) if age.notna().any() else 1.0
    if not np.isfinite(age_scale) or age_scale < 1e-8:
        age_scale = 1.0
    sex_categories = tuple(sorted(set(_normalize_category(train["sex"])) - {"__missing__"}))
    site_categories = tuple(
        sorted(set(_normalize_category(train["anatomic_site"])) - {"__missing__"})
    )
    names = ["intercept", "age_z", "age_missing", "sex_missing", "anatomic_site_missing"]
    names.extend(f"sex={value}" for value in sex_categories)
    names.append("sex=__other__")
    names.extend(f"anatomic_site={value}" for value in site_categories)
    names.append("anatomic_site=__other__")
    return MetadataSchema(
        age_mean=age_mean,
        age_scale=age_scale,
        sex_categories=sex_categories,
        anatomic_site_categories=site_categories,
        feature_names=tuple(names),
    )


def metadata_design_matrix(frame: pd.DataFrame, schema: MetadataSchema) -> np.ndarray:
    """Build a design matrix by reading only declared public metadata fields."""
    missing = sorted(set(PUBLIC_METADATA_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Metadata gate scoring frame missing public features: {missing}")
    public = frame.loc[:, PUBLIC_METADATA_FEATURES].copy()
    age = pd.to_numeric(public["age"], errors="coerce")
    sex = _normalize_category(public["sex"])
    site = _normalize_category(public["anatomic_site"])
    columns = [
        np.ones(len(frame), dtype=float),
        age.fillna(schema.age_mean).to_numpy(dtype=float) / schema.age_scale
        - schema.age_mean / schema.age_scale,
        age.isna().to_numpy(dtype=float),
        sex.eq("__missing__").to_numpy(dtype=float),
        site.eq("__missing__").to_numpy(dtype=float),
    ]
    columns.extend(sex.eq(value).to_numpy(dtype=float) for value in schema.sex_categories)
    columns.append((~sex.isin((*schema.sex_categories, "__missing__"))).to_numpy(dtype=float))
    columns.extend(site.eq(value).to_numpy(dtype=float) for value in schema.anatomic_site_categories)
    columns.append((~site.isin((*schema.anatomic_site_categories, "__missing__"))).to_numpy(dtype=float))
    matrix = np.column_stack(columns)
    if matrix.shape[1] != len(schema.feature_names):
        raise AssertionError("Metadata gate feature schema and design matrix diverged")
    return matrix


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _prediction_metrics(targets: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=int)
    pred = np.asarray(pred, dtype=int)
    confusion = np.zeros((4, 4), dtype=int)
    for actual, guessed in zip(targets, pred):
        confusion[actual, guessed] += 1
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    precision = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(np.mean(targets == pred)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "ordinal_mae": float(np.mean(np.abs(targets - pred))),
    }


def gate_proxy_metrics(targets: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=int)
    probs = np.asarray(probs, dtype=float)
    scores = probs @ RISK_SCORE_VECTOR
    proxy_pred = operational_head_from_score(scores, (0.25, 0.50, 0.75))
    argmax_pred = probs.argmax(axis=1)
    one_hot = np.eye(4)[targets]
    confidence = probs.max(axis=1)
    correct = (argmax_pred == targets).astype(float)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean() - correct[mask].mean()))
    return {
        **_prediction_metrics(targets, proxy_pred),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "expected_calibration_error_10bin": ece,
        "mean_s_gate": float(scores.mean()),
        **{
            f"argmax_{key}": value
            for key, value in _prediction_metrics(targets, argmax_pred).items()
        },
    }


def _fit_ridge(X: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    y = np.eye(4)[np.asarray(targets, dtype=int)]
    counts = np.bincount(np.asarray(targets, dtype=int), minlength=4).astype(float)
    class_weight = len(targets) / (4.0 * np.maximum(counts, 1.0))
    row_weight = np.sqrt(class_weight[np.asarray(targets, dtype=int)])
    weighted_x = X * row_weight[:, None]
    weighted_y = y * row_weight[:, None]
    penalty = np.eye(X.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    lhs = weighted_x.T @ weighted_x + penalty
    rhs = weighted_x.T @ weighted_y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def fit_metadata_gate_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    alphas: Iterable[float],
    temperatures: Iterable[float],
) -> tuple[dict, dict]:
    """Fit on train and choose ridge/temperature on validation only."""
    for name, frame in (("train", train), ("validation", val)):
        if "ideal_action_level" not in frame:
            raise ValueError(f"{name} frame lacks ideal_action_level for declared model fitting")
    schema = fit_metadata_schema(train)
    train_x = metadata_design_matrix(train, schema)
    val_x = metadata_design_matrix(val, schema)
    train_y = train["ideal_action_level"].to_numpy(dtype=int)
    val_y = val["ideal_action_level"].to_numpy(dtype=int)
    candidates = []
    best = None
    for alpha in alphas:
        weights = _fit_ridge(train_x, train_y, float(alpha))
        val_logits = val_x @ weights
        for temperature in temperatures:
            probs = _softmax(val_logits, float(temperature))
            metrics = gate_proxy_metrics(val_y, probs)
            key = (
                metrics["ordinal_mae"],
                metrics["multiclass_brier"],
                -metrics["balanced_accuracy"],
                float(alpha),
                float(temperature),
            )
            candidates.append({"alpha": float(alpha), "temperature": float(temperature), **metrics})
            if best is None or key < best[0]:
                best = (key, weights.copy(), float(alpha), float(temperature), metrics)
    if best is None:
        raise ValueError("Metadata gate model search received no candidates")
    _, weights, alpha, temperature, val_metrics = best
    model = {
        "model_type": "class_balanced_multiclass_ridge",
        "input_features": list(PUBLIC_METADATA_FEATURES),
        "forbidden_features": sorted(FORBIDDEN_GATE_FEATURES),
        "schema": schema.to_dict(),
        "weights": weights.tolist(),
        "alpha": alpha,
        "temperature": temperature,
        "selection_split": "val",
        "selection_metric": "ordinal_mae_then_multiclass_brier_then_balanced_accuracy",
    }
    search = {"selected_validation_metrics": val_metrics, "candidates": candidates}
    return model, search


def predict_metadata_gate(frame: pd.DataFrame, model: Mapping) -> tuple[np.ndarray, np.ndarray]:
    schema = MetadataSchema.from_dict(model["schema"])
    matrix = metadata_design_matrix(frame, schema)
    weights = np.asarray(model["weights"], dtype=float)
    probs = _softmax(matrix @ weights, float(model["temperature"]))
    return probs @ RISK_SCORE_VECTOR, probs


def stable_gate_noise(image_id: str, seed: int, width: float = 0.02) -> float:
    digest = hashlib.sha256(f"{image_id}|vlm_gate|{seed}".encode("utf-8")).hexdigest()
    unit = int(digest[:12], 16) / float(16**12 - 1)
    return (unit - 0.5) * 2.0 * float(width)


def fixed_model_gate_proxy(
    scores: pd.DataFrame,
    *,
    proxy_name: str,
    seed: int,
    noise_width: float,
) -> pd.DataFrame:
    required = {"image_id", "split", "s_assistant"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Fixed model proxy source is missing columns: {missing}")
    output = scores[["image_id", "split"]].copy()
    output["s_gate"] = [
        float(np.clip(score + stable_gate_noise(str(image_id), seed, noise_width), 0.0, 1.0))
        for image_id, score in zip(scores["image_id"], scores["s_assistant"])
    ]
    output["gate_proxy"] = proxy_name
    output["proxy_source"] = "fixed_model_score_plus_deterministic_noise"
    output["seed"] = int(seed)
    output["noise_width"] = float(noise_width)
    return output
