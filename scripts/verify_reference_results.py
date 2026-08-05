#!/usr/bin/env python3
"""Reconstruct all released summary values from the packaged case traces."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apeval.authority_selection import authority_summary_with_cluster_bootstrap  # noqa: E402
from apeval.evaluation import evaluate_authority_log  # noqa: E402


REPS = 5000
SEED = 20260711
QUANTILES = (0.025, 0.975)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(observed: float, expected: float, label: str) -> None:
    if not np.isclose(float(observed), float(expected), rtol=0.0, atol=1e-12):
        raise AssertionError(f"{label}: observed={observed}, expected={expected}")


def verify_authority_directory(path: Path) -> None:
    trace_path = path / "case_trace.csv"
    summary_path = path / "authority_summary.csv"
    receipt_path = path / "receipt.yaml"
    trace = pd.read_csv(trace_path, dtype={"image_id": str, "cluster_id": str})
    expected = pd.read_csv(summary_path)
    primary, sensitivity = authority_summary_with_cluster_bootstrap(
        trace,
        reps=REPS,
        seed=SEED,
        quantiles=QUANTILES,
    )
    observed_rows = {
        "image_weighted_primary": primary,
        "cluster_equal_sensitivity": sensitivity,
    }
    for _, row in expected.iterrows():
        observed = observed_rows[str(row["estimand"])]
        for column in expected.columns:
            if column == "estimand" or pd.isna(row[column]):
                continue
            assert_close(observed[column], row[column], f"{path.name}:{row['estimand']}:{column}")
    assert_close(
        primary["Delta_authority"],
        primary["intervention_rate"] * primary["delta"],
        f"{path.name}:p*delta",
    )

    receipt = yaml.safe_load(receipt_path.read_text())
    output_paths = {
        "trace": trace_path,
        "summary_csv": summary_path,
        "summary_md": path / "summary.md",
    }
    for key, expected_hash in receipt["output_sha256"].items():
        if sha256_file(output_paths[key]) != str(expected_hash):
            raise AssertionError(f"{path.name}: output hash mismatch for {key}")


def grouped_trace(path: Path) -> pd.DataFrame:
    trace = pd.read_csv(path, dtype={"image_id": str, "cluster_id": str})
    return trace.groupby("cluster_id", sort=True).agg(
        n=("cluster_id", "size"),
        authority_sum=("delta_authority_i", "sum"),
    )


def reconstructed_contrast(low: pd.DataFrame, high: pd.DataFrame) -> pd.DataFrame:
    if not low.index.equals(high.index):
        raise AssertionError("Proposal traces use different clusters")
    n = low["n"].to_numpy(float)
    if not np.array_equal(n, high["n"].to_numpy(float)):
        raise AssertionError("Proposal traces use different cluster sizes")
    low_sum = low["authority_sum"].to_numpy(float)
    high_sum = high["authority_sum"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    image = np.empty(REPS)
    cluster = np.empty(REPS)
    for index in range(REPS):
        sampled = rng.integers(0, len(n), size=len(n))
        sampled_n = n[sampled].sum()
        image[index] = (
            high_sum[sampled].sum() / sampled_n
            - low_sum[sampled].sum() / sampled_n
        )
        cluster[index] = (
            (high_sum[sampled] / n[sampled]).mean()
            - (low_sum[sampled] / n[sampled]).mean()
        )
    rows = []
    for unit, point, draws in (
        ("image_weighted", high_sum.sum() / n.sum() - low_sum.sum() / n.sum(), image),
        ("cluster_equal", (high_sum / n).mean() - (low_sum / n).mean(), cluster),
    ):
        lower, upper = np.quantile(draws, QUANTILES)
        rows.append({"unit": unit, "difference": point, "ci_lower": lower, "ci_upper": upper})
    return pd.DataFrame(rows)


def verify_proposal_dependence(path: Path) -> None:
    names = {
        "LLaVA-Med head": "llava_med_head",
        "MedGemma readout": "medgemma_readout",
        "CLIP probe": "clip_probe",
    }
    expected = pd.read_csv(path / "proposal_dependence_summary.csv")
    for proposal, slug in names.items():
        trace_path = path / f"case_trace_{slug}.csv"
        trace = pd.read_csv(trace_path, dtype={"image_id": str, "cluster_id": str})
        primary, sensitivity = authority_summary_with_cluster_bootstrap(
            trace,
            reps=REPS,
            seed=SEED,
            quantiles=QUANTILES,
        )
        row = expected.loc[expected["proposal"].eq(proposal)].iloc[0]
        mapping = {
            "n_images": primary["n_images"],
            "n_clusters": primary["n_clusters"],
            "intervention_rate": primary["intervention_rate"],
            "delta": primary["delta"],
            "Delta_authority_image": primary["Delta_authority"],
            "Delta_authority_image_ci_lower": primary["Delta_authority_ci_lower"],
            "Delta_authority_image_ci_upper": primary["Delta_authority_ci_upper"],
            "Delta_authority_cluster": sensitivity["Delta_authority"],
            "Delta_authority_cluster_ci_lower": sensitivity["Delta_authority_ci_lower"],
            "Delta_authority_cluster_ci_upper": sensitivity["Delta_authority_ci_upper"],
        }
        for column, observed in mapping.items():
            assert_close(observed, row[column], f"proposal:{proposal}:{column}")

    observed = reconstructed_contrast(
        grouped_trace(path / "case_trace_llava_med_head.csv"),
        grouped_trace(path / "case_trace_clip_probe.csv"),
    )
    expected_contrast = pd.read_csv(path / "paired_contrast.csv")
    for _, row in expected_contrast.iterrows():
        actual = observed.loc[observed["unit"].eq(row["unit"])].iloc[0]
        for column in ("difference", "ci_lower", "ci_upper"):
            assert_close(actual[column], row[column], f"proposal:{row['unit']}:{column}")

    receipt = yaml.safe_load((path / "receipt.yaml").read_text())
    for name, expected_hash in receipt["outputs_sha256"].items():
        if sha256_file(path / name) != str(expected_hash):
            raise AssertionError(f"proposal: output hash mismatch for {name}")


def verify_example_log() -> None:
    frame = pd.read_csv(ROOT / "examples" / "authority_log_example.csv")
    result = evaluate_authority_log(frame, seed=20260710, reps=200)
    metrics = result["metrics"]
    assert_close(
        metrics["Delta_authority"]["estimate"],
        metrics["intervention_rate"]["estimate"] * metrics["delta"]["estimate"],
        "example:p*delta",
    )


def verify_frozen_inputs() -> None:
    receipt = yaml.safe_load(
        (ROOT / "artifacts" / "frozen_proposal_heads_receipt.yaml").read_text()
    )
    artifact = ROOT / str(receipt["artifact"])
    if sha256_file(artifact) != str(receipt["sha256"]):
        raise AssertionError("Frozen proposal-head artifact hash mismatch")
    heads = pd.read_csv(artifact, dtype={"image_id": str})
    if len(heads) != int(receipt["rows"]) or heads["image_id"].duplicated().any():
        raise AssertionError("Frozen proposal-head artifact has invalid coverage")


def main() -> None:
    reference = ROOT / "reference_results"
    verify_authority_directory(reference / "medgemma_workshop_minimal")
    verify_authority_directory(reference / "medgemma_regime_decomposition" / "capacity_ceiling")
    verify_authority_directory(reference / "medgemma_regime_decomposition" / "safety_floor")
    verify_proposal_dependence(reference / "proposal_dependence")
    verify_example_log()
    verify_frozen_inputs()
    print("PASS: all reference summaries, bootstrap intervals, identities, and output hashes verified")


if __name__ == "__main__":
    main()
