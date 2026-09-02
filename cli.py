"""
cli.py
------
Run a full evaluation from the command line and save the report to disk.

Usage:
    python cli.py
    python cli.py --providers demo-fast demo-quality
    python cli.py --out reports/my_run.json
"""

import argparse
import json
from pathlib import Path

from eval.harness import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="*", default=None, help="Provider names to run (default: all active)")
    parser.add_argument("--out", default="reports/latest_report.json", help="Output JSON path")
    args = parser.parse_args()

    report = run_evaluation(provider_names=args.providers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluated {report['n_results']} (provider x test case) pairs "
          f"across {len(report['providers_evaluated'])} providers.")
    print("\nSummary:")
    for row in report["summary"]:
        print(
            f"  {row['provider']:<24} "
            f"overall={row['avg_overall_score']:<6} "
            f"judge={row['avg_judge_score']:<5} "
            f"hallucination={row['avg_hallucination_rate']:<6} "
            f"safety={row['safety_pass_rate']:<5} "
            f"latency={row['avg_latency_ms']}ms "
            f"cost=${row['total_cost_usd']}"
        )
    print(f"\nFull report saved -> {out_path}")


if __name__ == "__main__":
    main()
