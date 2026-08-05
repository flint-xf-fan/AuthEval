import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apeval.evaluation import evaluate_authority_log  # noqa: E402


REPORT_ORDER = (
    "J_std", "J_auth", "J", "intervention_rate", "delta",
    "Delta_utility", "Delta_authority", "Delta_total",
)


def write_markdown(path: Path, result: dict, source: str) -> None:
    lines = [
        "# Authority-Preserving Evaluation Report",
        "",
        f"Source log: `{source}`",
        f"Mode: **{result['mode']}**",
        f"Cases/clusters: `{result['n']}` / `{result['n_clusters']}`",
        f"Cluster bootstrap: `{result['bootstrap_reps']}` replicates, seed `{result['seed']}`.",
        "",
        "| Metric | Estimate | CI lower | CI upper | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for name in REPORT_ORDER:
        metric = result["metrics"][name]
        if metric["available"]:
            lines.append(
                f"| {name} | {metric['estimate']:.6f} | {metric['ci_lower']:.6f} | "
                f"{metric['ci_upper']:.6f} | available |"
            )
        else:
            lines.append(f"| {name} | NA | NA | NA | {metric['reason']} |")
    lines.extend([
        "",
        "`J_std` scores the proposal under global utility; `J_auth` scores the same "
        "proposal under the local rule; `J` scores the workflow-selected action under that "
        "utility. Positive total or authority gaps mean proposal scoring overstates "
        "the selected-action score; negative gaps mean it understates it.",
        "",
    ])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a role-separated recommendation log")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--lower", type=float, default=0.025)
    parser.add_argument("--upper", type=float, default=0.975)
    args = parser.parse_args()

    result = evaluate_authority_log(
        pd.read_csv(args.input),
        seed=args.seed,
        reps=args.reps,
        quantiles=(args.lower, args.upper),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_markdown(output_dir / "summary.md", result, args.input)
    print(f"[evaluate_authority_log] mode={result['mode']} wrote={output_dir}")


if __name__ == "__main__":
    main()
