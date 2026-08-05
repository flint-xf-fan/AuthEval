#!/usr/bin/env python3
"""Create a resumable MedGemma option-likelihood cache for one prepared split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apeval.authority_selection import validate_score_cache  # noqa: E402
from apeval.models.vlm_cache import build_backend, score_manifest  # noqa: E402


def path_at_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def validate_partial(frame: pd.DataFrame, manifest: pd.DataFrame, cfg: dict) -> None:
    required = {
        "image_id", "cache_status", "prompt_id", "model_name",
        "p_very_low", "p_low", "p_moderate", "p_high",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Partial cache missing columns: {missing}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Partial cache contains duplicate image IDs")
    if not set(frame["image_id"].astype(str)).issubset(set(manifest["image_id"].astype(str))):
        raise ValueError("Partial cache contains IDs outside the selected manifest")
    if not frame["cache_status"].eq("ok").all():
        raise ValueError("Partial cache contains non-ok rows")
    if set(frame["prompt_id"].astype(str)) != {str(cfg["prompt_id"])}:
        raise ValueError("Partial cache uses a different prompt")
    if set(frame["model_name"].astype(str)) != {str(cfg["model_name"])}:
        raise ValueError("Partial cache uses a different model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/medgemma.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    cfg = yaml.safe_load(path_at_root(args.config).read_text())
    manifest_path = path_at_root(cfg["manifests"][args.split])
    output_path = path_at_root(args.output or cfg["outputs"][args.split])
    manifest = pd.read_csv(
        manifest_path,
        dtype={"image_id": str, "cluster_id": str},
    )
    if manifest["image_id"].duplicated().any():
        raise ValueError("Prepared manifest contains duplicate image IDs")
    missing_images = manifest.loc[
        ~manifest["image_path"].map(lambda value: path_at_root(value).exists()),
        "image_path",
    ]
    if len(missing_images):
        raise FileNotFoundError(
            f"{len(missing_images)} image files are missing; first example: {missing_images.iloc[0]}"
        )
    manifest = manifest.copy()
    manifest["image_path"] = manifest["image_path"].map(lambda value: str(path_at_root(value)))

    existing = (
        pd.read_csv(output_path, dtype={"image_id": str})
        if output_path.exists()
        else pd.DataFrame()
    )
    if not existing.empty:
        validate_partial(existing, manifest, cfg)
    completed = set(existing["image_id"].astype(str)) if not existing.empty else set()
    missing = manifest.loc[~manifest["image_id"].isin(completed)]
    if missing.empty:
        validate_score_cache(
            existing,
            manifest["image_id"],
            prompt_id=str(cfg["prompt_id"]),
            model_name=str(cfg["model_name"]),
        )
        print(f"[score_medgemma] complete cache already exists: {output_path}")
        return

    print(
        f"[score_medgemma] loading {cfg['model_name']} at revision {cfg['revision']}; "
        f"completed={len(completed)} missing={len(missing)}",
        flush=True,
    )
    backend = build_backend(cfg)
    chunks = [existing] if not existing.empty else []
    pending: list[pd.DataFrame] = []
    order = {image_id: index for index, image_id in enumerate(manifest["image_id"])}
    for offset, (_, row) in enumerate(missing.iterrows(), start=1):
        pending.append(score_manifest(pd.DataFrame([row]), backend, int(cfg["seed"])))
        if offset % args.progress_every == 0 or offset == len(missing):
            combined = pd.concat([*chunks, *pending], ignore_index=True)
            combined["_order"] = combined["image_id"].astype(str).map(order)
            combined = combined.sort_values("_order").drop(columns="_order")
            atomic_csv(combined, output_path)
            chunks, pending = [combined], []
            print(
                f"[score_medgemma] rows={len(completed) + offset}/{len(manifest)} "
                f"output={output_path}",
                flush=True,
            )

    final = pd.read_csv(output_path, dtype={"image_id": str})
    validate_score_cache(
        final,
        manifest["image_id"],
        prompt_id=str(cfg["prompt_id"]),
        model_name=str(cfg["model_name"]),
    )


if __name__ == "__main__":
    main()
