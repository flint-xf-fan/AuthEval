import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse, yaml
from pathlib import Path
from apeval.data.prepare_isic import prepare

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/isic.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", help="Override output directory. Dry-run defaults to <configured_out_dir>_dryrun.")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out_dir = a.out_dir
    if a.dry_run and out_dir is None:
        configured = Path(cfg["paths"]["out_dir"])
        out_dir = str(configured.with_name(f"{configured.name}_dryrun"))
    df, key = prepare(cfg, dry_run=a.dry_run, out_dir=out_dir)
    written = out_dir or cfg["paths"]["out_dir"]
    print(f"[prepare_isic] rows={len(df)} split_key={key} "
          f"sites={sorted(df['site_profile'].unique())} "
          f"strata={sorted(x for x in df['risk_stratum'].dropna().unique())}")
    print(f"[prepare_isic] wrote manifests to {written}")
