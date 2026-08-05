"""Robust ISIC metadata schema detection (alias-tolerant, y stays a label only)."""
import pandas as pd

def detect_columns(columns, aliases):
    """Map each canonical name to the first matching actual column (case-insensitive)."""
    lower = {c.lower(): c for c in columns}
    out = {}
    for canon, alts in aliases.items():
        out[canon] = next((lower[a.lower()] for a in alts if a.lower() in lower), None)
    return out

def collapse_onehot_diagnosis(df, onehot_cols):
    """If no single diagnosis column exists but one-hot label columns do, argmax them."""
    present = [c for c in onehot_cols if c in df.columns]
    if not present:
        return None
    return df[present].astype(float).idxmax(axis=1)
