import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPORT_COLUMNS = [
    "component_id",
    "component_size",
    "component_crosses_split",
    "threshold",
    "exact_hash_duplicate",
    "image_id",
    "split",
    "site_id",
    "site_profile",
    "risk_stratum",
    "image_hash",
    "image_path",
]


def _load_gray_pixels(path: Path, width: int, height: int) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        with Image.open(path) as img:
            img = img.convert("L").resize((width, height))
            return img.tobytes()

    convert = shutil.which("convert")
    if convert is None:
        raise RuntimeError("Image duplicate check requires Pillow or ImageMagick `convert`.")
    proc = subprocess.run(
        [
            convert,
            str(path),
            "-auto-orient",
            "-resize",
            f"{width}x{height}!",
            "-colorspace",
            "Gray",
            "-depth",
            "8",
            "gray:-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


@lru_cache(maxsize=None)
def _dct_matrix(n: int) -> np.ndarray:
    mat = np.empty((n, n), dtype=float)
    for k in range(n):
        alpha = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for i in range(n):
            mat[k, i] = alpha * np.cos(np.pi * (i + 0.5) * k / n)
    return mat


def phash(path: str | Path, hash_size: int = 8, transform_size: int = 32) -> int:
    """Compute a 64-bit perceptual hash without copying raw images."""
    path = Path(path)
    width = height = transform_size
    pixels = _load_gray_pixels(path, width, height)
    expected = width * height
    if len(pixels) != expected:
        raise ValueError(f"Expected {expected} grayscale bytes from {path}, got {len(pixels)}")

    arr = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width)).astype(float)
    dct = _dct_matrix(transform_size) @ arr @ _dct_matrix(transform_size).T
    low = dct[:hash_size, :hash_size].reshape(-1)
    median = float(np.median(low[1:])) if len(low) > 1 else float(low[0])
    value = 0
    for bit, coeff in enumerate(low):
        if float(coeff) > median:
            value |= 1 << bit
    return value


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


@dataclass
class _BKNode:
    value: int
    indices: list[int] = field(default_factory=list)
    children: dict[int, "_BKNode"] = field(default_factory=dict)


class _BKTree:
    def __init__(self):
        self.root: _BKNode | None = None

    def insert(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value=value, indices=[index])
            return
        node = self.root
        while True:
            dist = hamming(value, node.value)
            if dist == 0:
                node.indices.append(index)
                return
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = _BKNode(value=value, indices=[index])
                return
            node = child

    def query(self, value: int, threshold: int) -> list[int]:
        if self.root is None:
            return []
        out = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = hamming(value, node.value)
            if dist <= threshold:
                out.extend(node.indices)
            low = dist - threshold
            high = dist + threshold
            stack.extend(child for d, child in node.children.items() if low <= d <= high)
        return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def build_duplicate_report(
    manifest: pd.DataFrame,
    *,
    threshold: int = 4,
    hash_size: int = 8,
    transform_size: int = 32,
    limit: int | None = None,
    progress_every: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if "image_id" not in manifest or "image_path" not in manifest:
        raise ValueError("manifest must contain image_id and image_path columns")

    work = manifest.copy()
    if limit is not None:
        work = work.head(limit).copy()
    work = work.reset_index(drop=True)

    hashes: list[int | None] = []
    failures = []
    total = len(work)
    for idx, row in work.iterrows():
        image_path = Path(str(row["image_path"]))
        try:
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            hashes.append(phash(image_path, hash_size=hash_size, transform_size=transform_size))
        except Exception as exc:
            hashes.append(None)
            failures.append({
                "image_id": row.get("image_id"),
                "image_path": str(image_path),
                "error": str(exc),
            })
        if progress_every > 0 and ((idx + 1) % progress_every == 0 or idx + 1 == total):
            print(f"[check_image_duplicates] hashed={idx + 1}/{total}", flush=True)

    work["image_hash_int"] = hashes
    valid = work[work["image_hash_int"].notna()].copy().reset_index(drop=True)
    valid["image_hash_int"] = valid["image_hash_int"].astype("uint64")
    valid["image_hash"] = valid["image_hash_int"].map(lambda x: f"{int(x):016x}")

    uf = _UnionFind(len(valid))
    tree = _BKTree()
    values = valid["image_hash_int"].map(int).tolist()
    for idx, value in enumerate(values):
        for other in tree.query(value, threshold):
            uf.union(idx, other)
        tree.insert(value, idx)

    roots = [uf.find(i) for i in range(len(valid))]
    root_to_members: dict[int, list[int]] = {}
    for idx, root in enumerate(roots):
        root_to_members.setdefault(root, []).append(idx)
    components = [members for members in root_to_members.values() if len(members) > 1]
    components.sort(key=lambda members: (-len(members), valid.loc[members[0], "image_id"]))

    hash_counts = valid["image_hash"].value_counts().to_dict()
    exact = valid[valid["image_hash"].map(hash_counts).gt(1)].copy()
    if len(exact) and "split" in exact:
        exact_split_n = exact.groupby("image_hash")["split"].nunique(dropna=True)
        exact_cross_hashes = set(exact_split_n.index[exact_split_n > 1])
    else:
        exact_cross_hashes = set()
    rows = []
    for comp_idx, members in enumerate(components, start=1):
        comp = valid.loc[members]
        split_count = comp["split"].nunique(dropna=True) if "split" in comp else 0
        component_id = f"PHASH_{comp_idx:06d}"
        for _, row in comp.sort_values("image_id").iterrows():
            rows.append({
                "component_id": component_id,
                "component_size": len(comp),
                "component_crosses_split": bool(split_count > 1),
                "threshold": threshold,
                "exact_hash_duplicate": bool(hash_counts.get(row["image_hash"], 0) > 1),
                "image_id": row.get("image_id"),
                "split": row.get("split"),
                "site_id": row.get("site_id"),
                "site_profile": row.get("site_profile"),
                "risk_stratum": row.get("risk_stratum"),
                "image_hash": row.get("image_hash"),
                "image_path": row.get("image_path"),
            })

    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    failures_df = pd.DataFrame(failures, columns=["image_id", "image_path", "error"])
    exact_groups = int((valid["image_hash"].value_counts() > 1).sum())
    exact_images = int(valid["image_hash"].map(hash_counts).gt(1).sum())
    exact_cross_split_groups = int(len(exact_cross_hashes))
    exact_cross_split_images = int(exact["image_hash"].isin(exact_cross_hashes).sum()) if len(exact) else 0
    cross_split_components = (
        int(report.groupby("component_id")["component_crosses_split"].first().sum())
        if len(report)
        else 0
    )
    summary = {
        "rows_scanned": int(len(work)),
        "images_hashed": int(len(valid)),
        "failures": int(len(failures_df)),
        "threshold": int(threshold),
        "hash_size": int(hash_size),
        "transform_size": int(transform_size),
        "algorithm": "phash_dct",
        "exact_hash_groups": exact_groups,
        "exact_hash_images": exact_images,
        "exact_cross_split_groups": exact_cross_split_groups,
        "exact_cross_split_images": exact_cross_split_images,
        "near_duplicate_components": int(len(components)),
        "near_duplicate_images": int(len(report)),
        "cross_split_components": cross_split_components,
    }
    return report, failures_df, summary


def exact_cross_split_report(report: pd.DataFrame) -> pd.DataFrame:
    """Return exact-hash duplicate rows whose exact hash crosses splits."""
    columns = ["exact_group_id", *REPORT_COLUMNS]
    if report.empty or "image_hash" not in report or "split" not in report:
        return pd.DataFrame(columns=columns)
    exact = report[report["exact_hash_duplicate"]].copy()
    if exact.empty:
        return pd.DataFrame(columns=columns)
    split_n = exact.groupby("image_hash")["split"].nunique(dropna=True)
    cross_hashes = list(split_n.index[split_n > 1])
    if not cross_hashes:
        return pd.DataFrame(columns=columns)
    group_ids = {h: f"EXACT_PHASH_{i:06d}" for i, h in enumerate(sorted(cross_hashes), start=1)}
    out = exact[exact["image_hash"].isin(cross_hashes)].copy()
    out["exact_group_id"] = out["image_hash"].map(group_ids)
    out = out[columns].sort_values(["exact_group_id", "split", "image_id"]).reset_index(drop=True)
    return out


def exact_duplicate_components_report(report: pd.DataFrame) -> pd.DataFrame:
    """Return all exact-hash duplicate rows for future split enforcement."""
    columns = ["exact_group_id", *REPORT_COLUMNS]
    if report.empty or "image_hash" not in report:
        return pd.DataFrame(columns=columns)
    exact = report[report["exact_hash_duplicate"]].copy()
    if exact.empty:
        return pd.DataFrame(columns=columns)
    group_ids = {
        h: f"EXACT_PHASH_{i:06d}"
        for i, h in enumerate(sorted(exact["image_hash"].dropna().unique()), start=1)
    }
    exact["exact_group_id"] = exact["image_hash"].map(group_ids)
    return exact[columns].sort_values(["exact_group_id", "image_id"]).reset_index(drop=True)


def remap_report_to_manifest(report: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Refresh split/site/risk metadata in an existing pHash report from final manifests."""
    if report.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    refresh_cols = [c for c in ["split", "site_id", "site_profile", "risk_stratum", "image_path"] if c in manifest]
    meta = manifest[["image_id", *refresh_cols]].drop_duplicates("image_id")
    base = report.drop(columns=[c for c in refresh_cols if c in report.columns], errors="ignore")
    out = base.merge(meta, on="image_id", how="left", validate="many_to_one")
    for col in REPORT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if "split" in out and "component_id" in out:
        crosses = out.groupby("component_id")["split"].transform(lambda s: s.nunique(dropna=True) > 1)
        out["component_crosses_split"] = crosses.fillna(False).astype(bool)
    return out[REPORT_COLUMNS].sort_values(["component_id", "image_id"]).reset_index(drop=True)


def _summary_from_report(
    report: pd.DataFrame,
    *,
    rows_scanned: int,
    images_hashed: int,
    failures: int,
    threshold: int,
    hash_size: int,
    transform_size: int,
) -> dict:
    exact = report[report["exact_hash_duplicate"]].copy() if len(report) else pd.DataFrame()
    if len(exact) and "split" in exact:
        exact_split_n = exact.groupby("image_hash")["split"].nunique(dropna=True)
        exact_cross_hashes = set(exact_split_n.index[exact_split_n > 1])
    else:
        exact_cross_hashes = set()
    cross_split_components = (
        int(report.groupby("component_id")["component_crosses_split"].first().sum())
        if len(report)
        else 0
    )
    return {
        "rows_scanned": int(rows_scanned),
        "images_hashed": int(images_hashed),
        "failures": int(failures),
        "threshold": int(threshold),
        "hash_size": int(hash_size),
        "transform_size": int(transform_size),
        "algorithm": "phash_dct",
        "exact_hash_groups": int(exact["image_hash"].nunique()) if len(exact) else 0,
        "exact_hash_images": int(len(exact)),
        "exact_cross_split_groups": int(len(exact_cross_hashes)),
        "exact_cross_split_images": int(exact["image_hash"].isin(exact_cross_hashes).sum()) if len(exact) else 0,
        "near_duplicate_components": int(report["component_id"].nunique()) if len(report) else 0,
        "near_duplicate_images": int(len(report)),
        "cross_split_components": cross_split_components,
    }


def _write_summary(
    path: Path,
    report_path: Path,
    failures_path: Path,
    exact_cross_path: Path,
    exact_components_path: Path,
    summary: dict,
    *,
    final_split_source: Path | None = None,
) -> None:
    lines = [
        "# ISIC perceptual duplicate report",
        "",
        "This report uses 64-bit DCT pHash perceptual hashes over existing image files. It does not copy raw images or alter splits.",
        "",
        f"- Algorithm: {summary['algorithm']}",
        f"- Rows scanned: {summary['rows_scanned']}",
        f"- Images hashed: {summary['images_hashed']}",
        f"- Hash failures: {summary['failures']}",
        f"- Hamming threshold: {summary['threshold']}",
        f"- Transform size: {summary['transform_size']}",
        f"- Exact hash duplicate groups: {summary['exact_hash_groups']}",
        f"- Exact hash duplicate images: {summary['exact_hash_images']}",
        f"- Exact hash cross-split groups: {summary['exact_cross_split_groups']}",
        f"- Exact hash cross-split images: {summary['exact_cross_split_images']}",
        f"- Near-duplicate components: {summary['near_duplicate_components']}",
        f"- Near-duplicate images: {summary['near_duplicate_images']}",
        f"- Cross-split near-duplicate components: {summary['cross_split_components']}",
        f"- Component report: `{report_path}`",
        f"- All exact duplicate components: `{exact_components_path}`",
        f"- Exact cross-split report: `{exact_cross_path}`",
        f"- Failure report: `{failures_path}`",
        "",
    ]
    if final_split_source is not None:
        lines.extend([
            f"- Split metadata source: `{final_split_source}`",
            "",
        ])
    if summary["exact_cross_split_groups"]:
        lines.append("WARNING: exact pHash duplicate groups still cross active splits; resplit before freezing.")
    elif summary["cross_split_components"]:
        lines.append(
            "Exact pHash cross-split groups are zero. Cross-split near-duplicate "
            "components remain diagnostic at the configured Hamming threshold."
        )
    else:
        lines.append("No exact or near-duplicate components cross active splits at the configured threshold.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/isic.yaml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--hash-size", type=int, default=8)
    ap.add_argument("--transform-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--output", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--failures", default=None)
    ap.add_argument("--exact-cross-output", default=None)
    ap.add_argument("--exact-components-output", default=None)
    ap.add_argument(
        "--use-final-split",
        action="store_true",
        help="Refresh an existing pHash report with the current manifest split metadata before summarizing.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["paths"]["out_dir"])
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "all_clean.csv"
    output_path = Path(args.output) if args.output else out_dir / "phash_duplicate_report.csv"
    summary_path = Path(args.summary) if args.summary else out_dir / "phash_duplicate_summary.md"
    failures_path = Path(args.failures) if args.failures else out_dir / "phash_duplicate_failures.csv"
    exact_cross_path = (
        Path(args.exact_cross_output)
        if args.exact_cross_output
        else out_dir / (
            "phash_final_exact_cross_split_duplicates.csv"
            if args.use_final_split
            else "phash_exact_cross_split_duplicates.csv"
        )
    )
    exact_components_path = (
        Path(args.exact_components_output)
        if args.exact_components_output
        else out_dir / "phash_exact_duplicate_components.csv"
    )

    manifest = pd.read_csv(manifest_path)
    final_split_source = manifest_path if args.use_final_split else None
    if args.use_final_split and output_path.exists() and args.limit is None:
        report = remap_report_to_manifest(pd.read_csv(output_path), manifest)
        if failures_path.exists():
            failures = pd.read_csv(failures_path)
        else:
            failures = pd.DataFrame(columns=["image_id", "image_path", "error"])
        summary = _summary_from_report(
            report,
            rows_scanned=len(manifest),
            images_hashed=len(manifest) - len(failures),
            failures=len(failures),
            threshold=args.threshold,
            hash_size=args.hash_size,
            transform_size=args.transform_size,
        )
    else:
        report, failures, summary = build_duplicate_report(
            manifest,
            threshold=args.threshold,
            hash_size=args.hash_size,
            transform_size=args.transform_size,
            limit=args.limit,
            progress_every=args.progress_every,
        )
    exact_cross = exact_cross_split_report(report)
    exact_components = exact_duplicate_components_report(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    failures.to_csv(failures_path, index=False)
    exact_cross.to_csv(exact_cross_path, index=False)
    exact_components.to_csv(exact_components_path, index=False)
    _write_summary(
        summary_path,
        output_path,
        failures_path,
        exact_cross_path,
        exact_components_path,
        summary,
        final_split_source=final_split_source,
    )
    print(
        f"[check_image_duplicates] rows={summary['rows_scanned']} hashed={summary['images_hashed']} "
        f"components={summary['near_duplicate_components']} exact_cross_split={summary['exact_cross_split_groups']} "
        f"near_cross_split={summary['cross_split_components']} "
        f"wrote={output_path}"
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    main()
