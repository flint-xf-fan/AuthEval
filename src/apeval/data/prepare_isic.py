"""ISIC metadata ingestion + leakage-safe manifest generation.
Labels (y) appear in manifests for training/eval ONLY; the gate never reads them."""
import hashlib, os, warnings, numpy as np, pandas as pd, yaml
from pathlib import Path
from apeval.sim import ISIC_TO_STRATUM, IDEAL, SITES
from apeval.data.isic_schema import detect_columns, collapse_onehot_diagnosis
from apeval.data.splits import build_split_clusters, leakage_safe_split

DIAGS = list(ISIC_TO_STRATUM)

def generate_synthetic_metadata(n=1200, seed=20260706, with_source=True):
    rng = np.random.default_rng(seed)
    n_pat = n // 3
    patient = rng.integers(0, n_pat, n)
    lesion  = patient * 10 + rng.integers(0, 3, n)          # lesions nested within patients
    dx      = rng.choice(DIAGS, n, p=_diag_probs())
    rows = pd.DataFrame({
        "isic_id":  [f"ISIC_{i:07d}" for i in range(n)],
        "jpg_path": [f"images/ISIC_{i:07d}.jpg" for i in range(n)],
        "patient_id": [f"P{p:05d}" for p in patient],
        "lesion_id":  [f"L{l:06d}" for l in lesion],
        "dx": dx,
        "sex": rng.choice(["male", "female"], n),
        "age_approx": rng.choice([25, 35, 45, 55, 65, 75, 85], n),
        "anatom_site_general": rng.choice(
            ["head/neck", "upper extremity", "lower extremity", "torso", "palms/soles"], n),
    })
    if with_source:
        rows["dataset"] = rng.choice(
            ["BCN_2019", "HAM10000_vidir", "MSK_component", "unknown_source"], n,
            p=[0.35, 0.35, 0.20, 0.10])
    # inject a few exact-duplicate images (same content, new id)
    dup = rows.sample(n=max(5, n // 100), random_state=seed).copy()
    dup_ids = [f"ISIC_D{i:06d}" for i in range(len(dup))]
    dup_pairs = pd.DataFrame({"image_id_1": rows.loc[dup.index, "isic_id"].values,
                              "image_id_2": dup_ids})
    dup["isic_id"] = dup_ids
    return pd.concat([rows, dup], ignore_index=True), dup_pairs

def _diag_probs():
    # benign-majority, but every stratum supported
    base = {"NV": 0.34, "BKL": 0.14, "MEL": 0.14, "BCC": 0.12, "AK": 0.07,
            "AKIEC": 0.05, "SCC": 0.05, "DF": 0.05, "VASC": 0.04}
    return [base[d] for d in DIAGS]

def ingest(df, cfg):
    cols = detect_columns(df.columns, cfg["column_aliases"])
    out = pd.DataFrame(index=df.index)
    for canon in ("image_id", "image_path", "patient_id", "lesion_id",
                  "sex", "age", "anatomic_site", "source"):
        if cols.get(canon):
            out[canon] = df[cols[canon]].values
    if cols.get("diagnosis"):
        out["diagnosis"] = df[cols["diagnosis"]].values
    else:
        oh = collapse_onehot_diagnosis(df, cfg.get("onehot_diagnosis", []))
        if oh is not None:
            out["diagnosis"] = oh.values
    if "age" in out:
        out["age_bin"] = pd.cut(pd.to_numeric(out["age"], errors="coerce"),
                                [0, 30, 50, 70, 200],
                                labels=["<30", "30-50", "50-70", "70+"]).astype(str)
    return out

def remove_duplicates(df, dup_pairs):
    if dup_pairs is None or len(dup_pairs) == 0:
        return df, pd.DataFrame(columns=["removed_image_id", "kept_image_id"])
    c = dup_pairs.columns.tolist()
    a, b = (c[0], c[1])
    removed, report = set(), []
    have = set(df["image_id"])
    for _, row in dup_pairs.iterrows():
        i1, i2 = row[a], row[b]
        if i1 in have and i2 in have:          # drop the secondary deterministically
            removed.add(i2); report.append({"removed_image_id": i2, "kept_image_id": i1})
    clean = df[~df["image_id"].isin(removed)].copy()
    return clean, pd.DataFrame(report)

def assign_risk(df):
    df = df.copy()
    df["risk_stratum"] = df.get("diagnosis", pd.Series(index=df.index)).map(ISIC_TO_STRATUM)
    df["ideal_action_level"] = df["risk_stratum"].map(IDEAL)
    return df

def _profile_from_source(val, kw):
    v = str(val).lower()
    for profile, words in kw.items():
        if any(w in v for w in words):
            return profile
    return "RURAL_like"

def _stable_bucket(value, n):
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % n

def assign_sites(df, cfg):
    df = df.copy()
    if "source" in df.columns and df["source"].notna().mean() > 0.5:
        df["site_id"] = df["source"].astype(str)
        df["site_profile"] = df["source"].map(lambda v: _profile_from_source(v, cfg["site_keyword_map"]))
    else:
        warnings.warn("No source column; constructing synthetic sites.")
        profiles = list(SITES)
        # Synthetic workflow profiles must be independent of labels/outcomes.
        # Use stable public case-mix metadata only. Image identifiers are used
        # only when both public case-mix fields are absent, to avoid turning
        # workflow construction into a per-image random assignment.
        age_sig = df.get("age_bin", pd.Series("", index=df.index)).fillna("__missing_age__").astype(str)
        site_sig = df.get("anatomic_site", pd.Series("", index=df.index)).fillna("__missing_site__").astype(str)
        sig = age_sig + "|" + site_sig
        no_case_mix = age_sig.eq("__missing_age__") & site_sig.eq("__missing_site__")
        if no_case_mix.any() and "image_id" in df.columns:
            sig = sig.copy()
            sig.loc[no_case_mix] = "image|" + df.loc[no_case_mix, "image_id"].astype(str)
        idx = sig.map(lambda s: _stable_bucket(s, len(profiles)))
        df["site_profile"] = idx.map(lambda i: profiles[i])
        df["site_id"] = "SYN_" + df["site_profile"]
    assert set(df["site_profile"]).issubset(set(SITES)), "site_profile not in frozen config"
    return df

def _detect_image_id_column(df, cfg, role):
    cols = detect_columns(df.columns, cfg["column_aliases"])
    image_col = cols.get("image_id")
    if image_col is None:
        raise ValueError(f"Could not detect image_id column in {role}.")
    return image_col

def _merge_raw_labels(raw, labels, cfg):
    raw_key = _detect_image_id_column(raw, cfg, "raw_metadata")
    label_key = _detect_image_id_column(labels, cfg, "raw_labels")
    if labels[label_key].duplicated().any():
        dupes = labels.loc[labels[label_key].duplicated(), label_key].head(5).tolist()
        raise ValueError(f"raw_labels contains duplicate image ids, e.g. {dupes}")

    label_cols = [c for c in labels.columns if c == label_key or c not in raw.columns]
    labels_for_merge = labels[label_cols].copy()
    if label_key != raw_key:
        labels_for_merge = labels_for_merge.rename(columns={label_key: raw_key})
    return raw.merge(labels_for_merge, on=raw_key, how="left", validate="many_to_one")

def _add_image_paths(raw, cfg):
    image_dir = cfg.get("paths", {}).get("image_dir")
    if not image_dir:
        return raw
    cols = detect_columns(raw.columns, cfg["column_aliases"])
    if cols.get("image_path"):
        return raw
    image_col = _detect_image_id_column(raw, cfg, "raw_metadata")
    raw = raw.copy()
    raw["jpg_path"] = raw[image_col].astype(str).map(lambda image_id: str(Path(image_dir) / f"{image_id}.jpg"))
    return raw

def _add_forced_duplicate_groups(df, cfg):
    path = cfg.get("paths", {}).get("duplicate_components")
    if not path:
        return df, None
    path = Path(path)
    if not path.exists():
        warnings.warn(f"duplicate_components file not found; ignoring: {path}")
        return df, None
    groups = pd.read_csv(path)
    group_col = "exact_group_id" if "exact_group_id" in groups.columns else "component_id"
    if "image_id" not in groups.columns or group_col not in groups.columns:
        raise ValueError(f"duplicate_components must contain image_id and {group_col}: {path}")
    if groups["image_id"].duplicated().any():
        dupes = groups.loc[groups["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ValueError(f"duplicate_components contains duplicate image IDs, e.g. {dupes}")
    mapping = groups.set_index("image_id")[group_col].astype(str)
    out = df.copy()
    out["duplicate_group"] = out["image_id"].map(mapping)
    return out, "duplicate_group" if out["duplicate_group"].notna().any() else None


def _split_freeze_path(cfg, out_dir):
    configured = cfg.get("paths", {}).get("split_freeze")
    return Path(configured) if configured else Path(out_dir) / "split_freeze.csv"


def _validate_frozen_assignments(df, frozen, force_group_col, path):
    required = {"image_id", "lesion_id", "cluster_id", "split"}
    missing = sorted(required - set(frozen.columns))
    if missing:
        raise ValueError(f"split freeze missing columns {missing}: {path}")
    if frozen["image_id"].duplicated().any():
        dupes = frozen.loc[frozen["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ValueError(f"split freeze contains duplicate image IDs, e.g. {dupes}: {path}")
    allowed_splits = {"train", "val", "test"}
    unknown_splits = sorted(set(frozen["split"].dropna()) - allowed_splits)
    if unknown_splits:
        raise ValueError(f"split freeze contains unknown splits {unknown_splits}: {path}")

    current_ids = set(df["image_id"].astype(str))
    frozen_ids = set(frozen["image_id"].astype(str))
    missing_ids = sorted(current_ids - frozen_ids)
    extra_ids = sorted(frozen_ids - current_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            "split freeze image-ID set does not match ingested data: "
            f"missing={len(missing_ids)} extra={len(extra_ids)} "
            f"examples_missing={missing_ids[:3]} examples_extra={extra_ids[:3]}"
        )

    expected_clusters, split_key = build_split_clusters(df, force_group_col=force_group_col)
    expected = pd.Series(expected_clusters, index=df["image_id"].astype(str)).sort_index()
    observed = frozen.assign(image_id=frozen["image_id"].astype(str)).set_index("image_id")["cluster_id"].astype(str).sort_index()
    mismatch = expected.ne(observed)
    if mismatch.any():
        examples = expected.index[mismatch][:5].tolist()
        raise ValueError(
            f"split freeze cluster IDs do not match current lesion/duplicate grouping "
            f"for {int(mismatch.sum())} images, e.g. {examples}: {path}"
        )

    by_cluster = frozen.groupby("cluster_id")["split"].nunique()
    if (by_cluster > 1).any():
        raise ValueError(
            f"split freeze has {int((by_cluster > 1).sum())} clusters crossing splits: {path}"
        )
    return split_key


def write_split_freeze(df, path, force_group_col=None, *, preserve_existing_split=True):
    """Write an ID-only, row-order-stable split freeze from an assigned frame."""
    clusters, _ = build_split_clusters(df, force_group_col=force_group_col)
    if preserve_existing_split and "split" not in df:
        raise ValueError("Cannot preserve a split that is absent from the input frame")
    freeze = pd.DataFrame({
        "image_id": df["image_id"].astype(str),
        "lesion_id": df.get("lesion_id", pd.Series(pd.NA, index=df.index)),
        "cluster_id": clusters,
        "split": df["split"] if preserve_existing_split else pd.NA,
    }).sort_values("image_id")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    freeze.to_csv(path, index=False)
    return freeze


def _assign_frozen_or_new_split(df, cfg, out_dir, force_group_col):
    freeze_path = _split_freeze_path(cfg, out_dir)
    if freeze_path.exists():
        frozen = pd.read_csv(freeze_path, dtype={"image_id": str, "cluster_id": str, "split": str})
        split_key = _validate_frozen_assignments(df, frozen, force_group_col, freeze_path)
        lookup = frozen.set_index("image_id")
        image_ids = df["image_id"].astype(str)
        assigned = df.copy()
        assigned["cluster_id"] = image_ids.map(lookup["cluster_id"])
        assigned["split"] = image_ids.map(lookup["split"])
        return assigned, f"frozen_{split_key}"

    split, clusters, split_key = leakage_safe_split(
        df,
        cfg["split_ratios"],
        cfg["seed"],
        force_group_col=force_group_col,
        return_cluster_id=True,
    )
    assigned = df.copy()
    assigned["cluster_id"] = clusters
    assigned["split"] = split
    write_split_freeze(assigned, freeze_path, force_group_col=force_group_col)
    return assigned, split_key

def prepare(cfg, dry_run=True, injected=None, out_dir=None):
    out_dir = Path(out_dir or cfg["paths"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    dup_pairs = None
    if injected is not None:
        raw = injected
    elif dry_run:
        raw, dup_pairs = generate_synthetic_metadata(seed=cfg["seed"])
    else:
        raw_metadata = Path(cfg["paths"]["raw_metadata"])
        if not raw_metadata.exists():
            raise FileNotFoundError(f"raw metadata not found: {raw_metadata}")
        raw = pd.read_csv(raw_metadata)
        raw_labels = cfg["paths"].get("raw_labels")
        if raw_labels:
            raw_labels = Path(raw_labels)
            if not raw_labels.exists():
                raise FileNotFoundError(f"raw labels not found: {raw_labels}")
            raw = _merge_raw_labels(raw, pd.read_csv(raw_labels), cfg)
        dp = cfg["paths"].get("duplicates")
        if dp and Path(dp).exists():
            dup_pairs = pd.read_csv(dp)

    raw = _add_image_paths(raw, cfg)
    df = ingest(raw, cfg)
    df, dup_report = remove_duplicates(df, dup_pairs)
    df = assign_risk(df)
    df = assign_sites(df, cfg)
    df, force_group_col = _add_forced_duplicate_groups(df, cfg)
    df, split_key = _assign_frozen_or_new_split(df, cfg, out_dir, force_group_col)

    # write manifests (labels present, for train/eval only)
    df.to_csv(out_dir / "all_clean.csv", index=False)
    for sp in ("train", "val", "test"):
        df[df["split"] == sp].to_csv(out_dir / f"{sp}.csv", index=False)
    dup_report.to_csv(out_dir / "duplicate_removal_report.csv", index=False)
    site_summary = (df.groupby(["site_id", "site_profile"])
                      .agg(n=("image_id", "size")).reset_index())
    site_summary.to_csv(out_dir / "site_summary.csv", index=False)
    _write_report(out_dir, df, split_key, dup_report, cfg)
    return df, split_key

def _markdown_table(table):
    table = table.copy()
    if isinstance(table.index, pd.MultiIndex) or table.index.name is not None:
        table = table.reset_index()
    headers = [str(c) for c in table.columns]
    rows = [[str(v) for v in row] for row in table.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)

def _checksums_block(cfg):
    raw_metadata = cfg.get("paths", {}).get("raw_metadata")
    if not raw_metadata:
        return []
    checksum_path = Path(raw_metadata).expanduser().parent / "CHECKSUMS.txt"
    if not checksum_path.exists():
        return []
    text = checksum_path.read_text().strip()
    if not text:
        return []
    return ["\n## Raw-data checksums", "", "```text", text, "```"]

def _phash_report_block(out_dir):
    summary_path = Path(out_dir) / "phash_duplicate_summary.md"
    if not summary_path.exists():
        return []
    interesting = []
    for line in summary_path.read_text().splitlines():
        if line.startswith("- Exact hash") or line.startswith("- Near-duplicate") or line.startswith("- Cross-split") or line.startswith("- Hash failures"):
            interesting.append(line)
    if not interesting:
        return []
    return [
        "\n## Perceptual duplicate report",
        "",
        f"- Summary: `{summary_path}`",
        *interesting,
        "",
        "Configured exact pHash duplicate components are used for split enforcement.",
        "Near-duplicate pHash components remain diagnostic unless exact duplicates are confirmed.",
        "Active split leakage after configured duplicate-component enforcement is reported in the leakage checks above.",
    ]

def _write_report(out_dir, df, split_key, dup_report, cfg):
    strata_present = sorted(x for x in df["risk_stratum"].dropna().unique())
    miss = {c: round(float(df[c].isna().mean()), 3) for c in ("age", "sex", "anatomic_site") if c in df}
    missing_cols = [c for c in ("age", "sex", "anatomic_site") if c in df]
    missing_by_split = (
        df.groupby("split")[missing_cols].apply(lambda g: g.isna().mean().round(3)).reset_index()
        if missing_cols else pd.DataFrame({"split": sorted(df["split"].unique())})
    )
    split_risk = pd.crosstab(df["split"], df["risk_stratum"])
    split_site = pd.crosstab(df["split"], df["site_profile"])
    site_risk = pd.crosstab(df["site_profile"], df["risk_stratum"])
    site_statement = (
        "Synthetic workflow profiles assigned from public case-mix metadata; "
        "source/provenance column was unavailable or too sparse."
        if df["site_id"].astype(str).str.startswith("SYN_").all()
        else "Source-derived site IDs mapped to frozen workflow profiles."
    )
    lines = [
        "# ISIC split report\n",
        f"- Split key: **{split_key}**",
        f"- Rows (clean): {len(df)}",
        f"- Duplicates removed: {len(dup_report)}",
        f"- Risk strata present: {strata_present}  (all four: {set(strata_present)>= {'very_low','low','moderate','high'}})",
        f"- Site profiles: {sorted(df['site_profile'].unique())}",
        f"- Site profile construction: {site_statement}",
        f"- Metadata missingness: {miss}",
        f"- Split sizes: {df['split'].value_counts().to_dict()}",
        f"- Bootstrap clusters: {df['cluster_id'].nunique()}",
        "\n## Leakage checks",
    ]
    for k in ("patient_id", "lesion_id", "image_id"):
        if k in df:
            cross = df.groupby(k)["split"].nunique()
            lines.append(f"- {k}: {int((cross>1).sum())} keys crossing splits (must be 0 for the split key)")
    if "duplicate_group" in df:
        duplicate_groups = df[df["duplicate_group"].notna()]
        if len(duplicate_groups):
            cross = duplicate_groups.groupby("duplicate_group")["split"].nunique()
            lines.append(
                f"- duplicate_group: {int((cross>1).sum())} groups crossing splits "
                "(must be 0 when duplicate components are configured)"
            )
    lines.extend([
        "\n## Split x risk stratum",
        _markdown_table(split_risk),
        "\n## Split x site profile",
        _markdown_table(split_site),
        "\n## Site profile x risk stratum",
        _markdown_table(site_risk),
        "\n## Metadata missingness by split",
        _markdown_table(missing_by_split),
    ])
    lines.extend(_phash_report_block(out_dir))
    lines.extend(_checksums_block(cfg))
    (out_dir / "split_report.md").write_text("\n".join(lines) + "\n")
