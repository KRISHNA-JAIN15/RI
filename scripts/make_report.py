"""Assemble RESULTS.md from a finished run (training summary, Table I, figures).

Usage: python scripts/make_report.py --run runs/ddpg_paper --out RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="RESULTS.md")
    args = p.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "train_summary.json").read_text())
    table = (run / "eval" / "table1.md").read_text()
    rows = json.loads((run / "eval" / "table1.json").read_text())
    extra = [r for r in rows if "(ours)" in r["algorithm"]]
    figs = sorted((run / "eval").glob("fig5_*.png"))
    scenes = sorted((run / "eval").glob("scene_*.png"))

    lines = [
        "# Results",
        "",
        f"Run directory: `{run}`",
        "",
        "## Training (Sec. IV-B)",
        "",
        f"* rounds: {summary['episodes']}, env steps: {summary['total_env_steps']}, gradient updates: {summary['updates']}",
        f"* wall time: {summary['wall_time_s'] / 60:.1f} min (CPU)",
        f"* mean return over the last 100 rounds: {summary['final_100_mean_return']:.1f}",
        f"* success rate over the last 100 training rounds (with exploration noise): {100 * summary['final_100_success_rate']:.1f} %",
        "",
        f"![Fig. 4]({run / 'fig4_training_curve.png'})",
        "",
        "## Simulation test (Sec. IV-C, Table I)",
        "",
        table,
        "",
        "Additional statistics (ours):",
        "",
        "| Algorithm | collision rate (%) | mean steps / nodes | mean return |",
        "|---|---|---|---|",
    ]
    for r in extra:
        lines.append(f"| {r['algorithm']} | {r['collision_rate_percent']:.1f} | {r['mean_steps']:.1f} | {r['mean_return']:.1f} |")
    lines += ["", "## Fig. 5 style trajectories", ""]
    for f in figs + scenes:
        lines.append(f"![{f.name}]({f})")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
