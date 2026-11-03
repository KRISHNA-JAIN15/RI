"""Re-draw Fig. 4 (training curve) from a train_log.csv.

Usage: python scripts/plot_results.py --run runs/ddpg [--bin 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_rl.visualize import plot_training_curve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--bin", type=int, default=100)
    args = p.parse_args()
    run = Path(args.run)
    plot_training_curve(run / "train_log.csv", run / "fig4_training_curve.png", args.bin)
    print(run / "fig4_training_curve.png")


if __name__ == "__main__":
    main()
