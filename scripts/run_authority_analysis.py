#!/usr/bin/env python3
"""Run the fixed MedGemma authority-gap analyses reported in the paper."""
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


def path_at_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run(config_path: Path) -> list[dict]:
    cfg = yaml.safe_load(config_path.read_text())
    manifest_path = path_at_root(cfg["manifest"])
    scores_path = path_at_root(cfg["scores"])
    proxy_path = path_at_root(cfg["gate_proxy"])
    lock_path = path_at_root(cfg["readout_lock"])
    output_root = path_at_root(cfg["output_root"])

    manifest = pd.read_csv(manifest_path, dtype={"image_id": str, "cluster_id": str})
    scores = pd.read_csv(scores_path, dtype={"image_id": str})
    proxy = pd.read_csv(proxy_path, dtype={"image_id": str})
    lock = yaml.safe_load(lock_path.read_text())
    validate_score_cache(
        scores,
        manifest["image_id"],
        prompt_id=str(lock["model"]["prompt_id"]),
        model_name=str(lock["model"]["model_name"]),
    )

    policy = cfg["policy"]
    if float(policy["bias"]) != 0.0 or int(policy["slate_size"]) != 4:
        raise ValueError("The camera-ready analysis uses zero bias and a four-action slate")
    class_scores = apply_readout(lock["readout"], scores)
    ranks = rank_policy_actions(class_scores, float(policy["bias"]))
    context = build_authority_context(
        manifest,
        proxy,
        scores["image_id"].astype(str).tolist(),
        seed=int(policy["context_seed"]),
    )
    shared = ["site_profile", "s_gate", *PRIVATE_CONTEXT_COLUMNS]
    if context.groupby("cluster_id")[shared].nunique(dropna=False).to_numpy().max() != 1:
        raise AssertionError("Authority context is not constant within cluster")

    bootstrap = cfg["bootstrap"]
    rows: list[dict] = []
    for regime in policy["regimes"]:
        point, trace = evaluate_ranked_policy(
            context,
            ranks,
            regime=str(regime),
            slate_size=int(policy["slate_size"]),
            policy_name=str(policy["policy_id"]),
        )
        primary, sensitivity = authority_summary_with_cluster_bootstrap(
            trace,
            reps=int(bootstrap["reps"]),
            seed=int(bootstrap["seed"]),
            quantiles=tuple(map(float, bootstrap["quantiles"])),
        )
        for key in ("J_auth", "J", "intervention_rate", "delta", "Delta_authority"):
            if not np.isclose(float(primary[key]), float(point[key])):
                raise AssertionError(f"{regime}: point estimate disagrees for {key}")

        regime_dir = output_root / str(regime)
        trace_path = regime_dir / "case_trace.csv"
        summary_path = regime_dir / "authority_summary.csv"
        atomic_csv(
            trace[[
                "image_id", "cluster_id", "policy_id", "slate_size", "a_head",
                "a_exec", "intervention_type", "intervened", "Uz_head", "Uz_exec",
                "delta_authority_i",
            ]],
            trace_path,
        )
        atomic_csv(pd.DataFrame([primary, sensitivity]), summary_path)
        receipt = {
            "analysis_id": cfg["analysis_id"],
            "analysis_status": cfg["analysis_status_by_regime"][str(regime)],
            "regime": str(regime),
            "policy": dict(policy),
            "bootstrap": dict(bootstrap),
            "source_sha256": {
                "manifest": sha256_file(manifest_path),
                "scores": sha256_file(scores_path),
                "gate_proxy": sha256_file(proxy_path),
                "readout_lock": sha256_file(lock_path),
            },
            "output_sha256": {
                "case_trace.csv": sha256_file(trace_path),
                "authority_summary.csv": sha256_file(summary_path),
            },
        }
        (regime_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        rows.append({"regime": str(regime), **primary})

    combined = pd.DataFrame(rows)
    atomic_csv(combined, output_root / "camera_ready_results.csv")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/authority_analysis.yaml")
    args = parser.parse_args()
    rows = run(path_at_root(args.config))
    for row in rows:
        print(
            f"[{row['regime']}] p={row['intervention_rate']:.6f} "
            f"delta={row['delta']:.6f} Delta_authority={row['Delta_authority']:.6f} "
            f"CI=[{row['Delta_authority_ci_lower']:.6f},"
            f"{row['Delta_authority_ci_upper']:.6f}]"
        )


if __name__ == "__main__":
    main()
