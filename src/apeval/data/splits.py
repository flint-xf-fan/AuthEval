"""Leakage-safe, group-respecting, stratified splits.
Priority: patient_id -> lesion_id -> lesion_id/image_id hybrid -> image_id.
Optional forced groups, such as exact duplicate components, are unioned with
the normal split key so duplicate grouping cannot break patient/lesion grouping.
"""
from collections import defaultdict
import hashlib
import warnings

import numpy as np

def choose_split_key_values(df):
    for key in ("patient_id", "lesion_id"):
        if key in df.columns and df[key].notna().mean() > 0.99:
            return df[key].astype(str), key
    if ("lesion_id" in df.columns and "image_id" in df.columns
            and df["lesion_id"].notna().any()):
        warnings.warn("Partial lesion_id coverage; splitting by lesion_id where available and image_id otherwise.")
        has_lesion = df["lesion_id"].notna()
        key_values = np.where(
            has_lesion,
            "lesion:" + df["lesion_id"].astype(str),
            "image:" + df["image_id"].astype(str),
        )
        return key_values, "lesion_id_or_image_id"
    warnings.warn("No patient_id/lesion_id with full coverage; splitting by image_id (leakage risk).")
    return df["image_id"].astype(str), "image_id"

def choose_split_key(df):
    _, key = choose_split_key_values(df)
    return key

class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

def _union_forced_groups(base_values, forced_values):
    uf = _UnionFind()
    for base, forced in zip(base_values, forced_values):
        base_key = f"base:{base}"
        uf.find(base_key)
        if forced is not None and str(forced) != "" and str(forced).lower() != "nan":
            uf.union(base_key, f"forced:{forced}")
    # Union-find roots depend on row traversal. Convert each component to a
    # content-derived identifier so the persisted cluster id is invariant to
    # input row order.
    members = defaultdict(set)
    base_nodes = [f"base:{base}" for base in base_values]
    for node in set(base_nodes):
        members[uf.find(node)].add(node)
    stable = {}
    for root, component in members.items():
        payload = "\x1f".join(sorted(component))
        stable[root] = f"cluster:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"
    return np.asarray([stable[uf.find(node)] for node in base_nodes], dtype=object)


def build_split_clusters(df, force_group_col=None):
    """Return stable leakage clusters and the human-readable grouping rule."""
    key_values, key_name = choose_split_key_values(df)
    base_values = np.asarray(key_values, dtype=object)
    if force_group_col and force_group_col in df.columns and df[force_group_col].notna().any():
        clusters = _union_forced_groups(base_values, df[force_group_col].to_numpy())
        key_name = f"{key_name}+{force_group_col}"
    else:
        clusters = _union_forced_groups(base_values, np.full(len(df), None, dtype=object))
    return clusters, key_name

def leakage_safe_split(
    df,
    ratios,
    seed,
    strat_cols=("risk_stratum", "site_profile"),
    force_group_col=None,
    *,
    return_cluster_id=False,
):
    """Assign each unique key to exactly one split; rows inherit their key's split.
    Groups are stratified approximately by strat_cols at the group level."""
    work = df.copy()
    clusters, key_name = build_split_clusters(df, force_group_col=force_group_col)
    work["_split_key"] = clusters
    # one representative stratum-signature per key (the group's first row)
    strat = [c for c in strat_cols if c in df.columns]
    grp = work.groupby("_split_key")[strat].first() if strat else work.groupby("_split_key").size().to_frame("n")
    grp = grp.reset_index()
    grp["_sig"] = grp[strat].astype(str).agg("|".join, axis=1) if strat else "all"
    rng = np.random.default_rng(seed)
    split_of = {}
    for _, sub in grp.groupby("_sig"):
        keys = sub["_split_key"].to_numpy(); rng.shuffle(keys)
        n = len(keys); n_tr = int(round(n * ratios["train"])); n_va = int(round(n * ratios["val"]))
        for k in keys[:n_tr]:        split_of[k] = "train"
        for k in keys[n_tr:n_tr+n_va]: split_of[k] = "val"
        for k in keys[n_tr+n_va:]:   split_of[k] = "test"
    split = work["_split_key"].map(split_of)
    if return_cluster_id:
        return split, work["_split_key"].copy(), key_name
    return split, key_name
