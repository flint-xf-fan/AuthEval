#!/usr/bin/env python3
"""Compare frozen proposal mechanisms under one fixed authority process."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apeval.authority_selection import (  # noqa: E402
    PRIVATE_CONTEXT_COLUMNS,
    apply_readout,
    authority_summary_with_cluster_bootstrap,
    build_authority_context,
    evaluate_ranked_policy,
    rank_policy_actions,
    validate_score_cache,
)
from apeval.sim import LEVEL, make_slate  # noqa: E402


def path_at_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranks_from_heads(values: pd.Series) -> np.ndarray:
    if values.isna().any() or not set(values.astype(str)).issubset(set(LEVEL)):
        raise ValueError("Frozen proposal heads contain missing or unknown actions")
    return np.asarray(
        [[LEVEL[action] for action in make_slate(LEVEL[str(head)])] for head in values],
        dtype=int,
    )


def summarize(
    name: str,
    reported_bacc: float,
    context: pd.DataFrame,
    ranks: np.ndarray,
    cfg: dict,
) -> dict:
    point, trace = evaluate_ranked_policy(
        context,
        ranks,
        regime=str(cfg["regime"]),
        slate_size=int(cfg["slate_size"]),
        policy_name=name,
    )
    bootstrap = cfg["bootstrap"]
    primary, sensitivity = authority_summary_with_cluster_bootstrap(
        trace,
        reps=int(bootstrap["reps"]),
        seed=int(bootstrap["seed"]),
        quantiles=tuple(map(float, bootstrap["quantiles"])),
    )
    if not np.isclose(float(primary["Delta_authority"]), float(point["Delta_authority"])):
        raise AssertionError(f"{name}: bootstrap summary disagrees with the point estimate")
    grouped = trace.groupby("cluster_id", sort=True).agg(
        n=("cluster_id", "size"),
        authority_sum=("delta_authority_i", "sum"),
    )
    return {
        "row": {
            "proposal": name,
            "test_balanced_accuracy": float(reported_bacc),
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
        },
        "grouped": grouped,
        "trace": trace,
    }


def paired_contrast(low: pd.DataFrame, high: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not low.index.equals(high.index):
        raise AssertionError("Paired contrast requires identical clusters")
    n = low["n"].to_numpy(float)
    if not np.array_equal(n, high["n"].to_numpy(float)):
        raise AssertionError("Paired contrast requires identical cluster sizes")
    low_sum = low["authority_sum"].to_numpy(float)
    high_sum = high["authority_sum"].to_numpy(float)
    bootstrap = cfg["bootstrap"]
    reps = int(bootstrap["reps"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    image_draws = np.empty(reps)
    cluster_draws = np.empty(reps)
    for index in range(reps):
        sampled = rng.integers(0, len(n), size=len(n))
        sampled_n = n[sampled].sum()
        image_draws[index] = (
            high_sum[sampled].sum() / sampled_n
            - low_sum[sampled].sum() / sampled_n
        )
        cluster_draws[index] = (
            (high_sum[sampled] / n[sampled]).mean()
            - (low_sum[sampled] / n[sampled]).mean()
        )
    quantiles = tuple(map(float, bootstrap["quantiles"]))
    rows = []
    for unit, point, draws in (
        ("image_weighted", high_sum.sum() / n.sum() - low_sum.sum() / n.sum(), image_draws),
        ("cluster_equal", (high_sum / n).mean() - (low_sum / n).mean(), cluster_draws),
    ):
        lower, upper = np.quantile(draws, quantiles)
        rows.append({
            "contrast": "CLIP probe minus LLaVA-Med head",
            "unit": unit,
            "difference": float(point),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "excludes_zero": bool(lower > 0 or upper < 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/proposal_dependence.yaml")
    args = parser.parse_args()
    config_path = path_at_root(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    paths = {
        key: path_at_root(cfg[key])
        for key in ("manifest", "gate_proxy", "medgemma_scores", "readout_lock", "frozen_heads")
    }
    manifest = pd.read_csv(paths["manifest"], dtype={"image_id": str, "cluster_id": str})
    image_ids = manifest["image_id"].astype(str).tolist()
    proxy = pd.read_csv(paths["gate_proxy"], dtype={"image_id": str})
    context = build_authority_context(
        manifest,
        proxy,
        image_ids,
        seed=int(cfg["context_seed"]),
    )
    shared = ["site_profile", "s_gate", *PRIVATE_CONTEXT_COLUMNS]
    if context.groupby("cluster_id")[shared].nunique(dropna=False).to_numpy().max() != 1:
        raise AssertionError("Authority context is not constant within cluster")

    heads = pd.read_csv(paths["frozen_heads"], dtype={"image_id": str})
    heads = manifest[["image_id"]].merge(heads, on="image_id", validate="one_to_one")
    lock = yaml.safe_load(paths["readout_lock"].read_text())
    scores = pd.read_csv(paths["medgemma_scores"], dtype={"image_id": str})
    validate_score_cache(
        scores,
        image_ids,
        prompt_id=str(lock["model"]["prompt_id"]),
        model_name=str(lock["model"]["model_name"]),
    )
    scores = manifest[["image_id"]].merge(scores, on="image_id", validate="one_to_one")
    medgemma_ranks = rank_policy_actions(apply_readout(lock["readout"], scores), 0.0)

    bacc = cfg["reported_balanced_accuracy"]
    arms = {
        "LLaVA-Med head": summarize(
            "LLaVA-Med head", bacc["LLaVA-Med head"], context,
            ranks_from_heads(heads["llava_med_head"]), cfg,
        ),
        "MedGemma readout": summarize(
            "MedGemma readout", bacc["MedGemma readout"], context,
            medgemma_ranks, cfg,
        ),
        "CLIP probe": summarize(
            "CLIP probe", bacc["CLIP probe"], context,
            ranks_from_heads(heads["clip_probe_head"]), cfg,
        ),
    }
    reference = arms["LLaVA-Med head"]["trace"]
    for name, arm in arms.items():
        trace = arm["trace"]
        if not trace[["image_id", "cluster_id"]].equals(reference[["image_id", "cluster_id"]]):
            raise AssertionError(f"{name}: cases or clusters differ from the reference arm")

    summary = pd.DataFrame([arm["row"] for arm in arms.values()])
    paired = paired_contrast(
        arms["LLaVA-Med head"]["grouped"],
        arms["CLIP probe"]["grouped"],
        cfg,
    )
    output_root = path_at_root(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "proposal_dependence_summary.csv", index=False)
    paired.to_csv(output_root / "paired_contrast.csv", index=False)
    trace_paths = {}
    for name, arm in arms.items():
        slug = name.lower().replace(" ", "_").replace("-", "_")
        trace_path = output_root / f"case_trace_{slug}.csv"
        arm["trace"].to_csv(trace_path, index=False)
        trace_paths[slug] = trace_path
    receipt = {
        "analysis_id": cfg["analysis_id"],
        "analysis_status": cfg["analysis_status"],
        "context_seed": int(cfg["context_seed"]),
        "regime": cfg["regime"],
        "slate_size": int(cfg["slate_size"]),
        "bootstrap": dict(cfg["bootstrap"]),
        "source_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "output_sha256": {
            "proposal_dependence_summary.csv": sha256_file(output_root / "proposal_dependence_summary.csv"),
            "paired_contrast.csv": sha256_file(output_root / "paired_contrast.csv"),
            **{path.name: sha256_file(path) for path in trace_paths.values()},
        },
    }
    (output_root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nPaired contrast\n", paired.to_string(index=False))


if __name__ == "__main__":
    main()
