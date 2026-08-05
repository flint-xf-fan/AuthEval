import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apeval.gate_proxy import (  # noqa: E402
    PUBLIC_METADATA_FEATURES,
    fit_metadata_gate_model,
    fixed_model_gate_proxy,
    gate_proxy_metrics,
    predict_metadata_gate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: str, expected_split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image_id", "split", "ideal_action_level", *PUBLIC_METADATA_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if set(frame["split"]) != {expected_split}:
        raise ValueError(f"{path} is not exclusively the {expected_split} split")
    return frame


def _score_metadata(frame: pd.DataFrame, model: dict, split: str):
    score, probs = predict_metadata_gate(frame, model)
    output = frame[["image_id", "split"]].copy()
    output["s_gate"] = score
    output["gate_proxy"] = "metadata"
    output["proxy_source"] = "train_fitted_public_metadata_model"
    metrics = gate_proxy_metrics(frame["ideal_action_level"].to_numpy(dtype=int), probs)
    return output, {"split": split, "n": int(len(frame)), **metrics}


def _write_fixed_model_proxy(spec: dict, *, seed: int, noise_width: float) -> dict:
    frames = []
    legacy_differences = []
    for split, source in spec["score_files"].items():
        scores = pd.read_csv(source)
        proxy = fixed_model_gate_proxy(
            scores,
            proxy_name=spec["name"],
            seed=seed,
            noise_width=noise_width,
        )
        frames.append(proxy)
        if "s_gate" in scores:
            joined = proxy.merge(
                scores[["image_id", "s_gate"]],
                on="image_id",
                suffixes=("_new", "_legacy"),
                validate="one_to_one",
            )
            legacy_differences.append(float((joined["s_gate_new"] - joined["s_gate_legacy"]).abs().max()))
    output = pd.concat(frames, ignore_index=True)
    out_path = Path(spec["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    return {
        "name": spec["name"],
        "output": str(out_path),
        "rows": int(len(output)),
        "sha256": _sha256(out_path),
        "max_abs_difference_from_legacy_coupled_column": (
            max(legacy_differences) if legacy_differences else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and freeze assistant-independent gate proxies.")
    parser.add_argument("--config", default="configs/model/gate_proxy.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    manifests = cfg["data"]["manifests"]
    train = _load_manifest(manifests["train"], "train")
    val = _load_manifest(manifests["val"], "val")
    model, search = fit_metadata_gate_model(
        train,
        val,
        alphas=cfg["metadata_model"]["alpha_candidates"],
        temperatures=cfg["metadata_model"]["temperature_candidates"],
    )
    model.update({
        "seed": int(cfg["seed"]),
        "fit_split": "train",
        "fit_n": int(len(train)),
        "validation_n": int(len(val)),
        "test_used_for_fitting_or_selection": False,
    })

    model_path = Path(cfg["outputs"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2) + "\n")

    train_proxy, train_metrics = _score_metadata(train, model, "train")
    val_proxy, val_metrics = _score_metadata(val, model, "val")

    # Test is not read until the complete selected model has been serialized.
    test = _load_manifest(manifests["test"], "test")
    test_proxy, test_metrics = _score_metadata(test, model, "test")
    metadata_proxy = pd.concat([train_proxy, val_proxy, test_proxy], ignore_index=True)
    metadata_path = Path(cfg["outputs"]["metadata_proxy"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_proxy.to_csv(metadata_path, index=False)

    fixed_reports = [
        _write_fixed_model_proxy(
            spec,
            seed=int(cfg["fixed_model_proxies"]["seed"]),
            noise_width=float(cfg["fixed_model_proxies"]["noise_width"]),
        )
        for spec in cfg["fixed_model_proxies"]["sources"]
    ]
    summary = {
        "design_status": "post_hoc_measurement_repair",
        "canonical_proxy": "metadata",
        "metadata_model": {
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "proxy_path": str(metadata_path),
            "proxy_sha256": _sha256(metadata_path),
            "selected_alpha": model["alpha"],
            "selected_temperature": model["temperature"],
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "test_used_for_fitting_or_selection": False,
        },
        "fixed_model_proxies": fixed_reports,
        "candidate_count": len(search["candidates"]),
    }
    summary_path = Path(cfg["outputs"]["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
