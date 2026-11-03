"""Table I metrics: S, L_avg, sigma_L, T_avg, sigma_T, E_avg, sigma_E."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeResult:
    success: bool
    path_length: float     # end-effector path length [m]
    plan_time_ms: float    # planning / inference time [ms]
    final_error: float     # ||EE - target|| at the end [m]
    collided: bool = False
    steps: int = 0
    episode_return: float = 0.0
    min_clearance: float = float("nan")


def summarize(results: list[EpisodeResult], name: str, successful_only: bool = True) -> dict:
    """Aggregate per-episode results into the Table I columns.

    ``L``, ``T`` and ``E`` are computed over successful rounds when
    ``successful_only`` is set (a failed plan has no meaningful path / error),
    ``S`` is always over all rounds.
    """
    n = len(results)
    ok = [r for r in results if r.success]
    pool = ok if (successful_only and ok) else results

    def stats(values):
        arr = np.asarray(values, dtype=np.float64)
        return (float(arr.mean()), float(arr.std())) if arr.size else (float("nan"), float("nan"))

    L, sL = stats([r.path_length for r in pool])
    T, sT = stats([r.plan_time_ms for r in pool])
    E, sE = stats([r.final_error for r in pool])
    return {
        "algorithm": name,
        "episodes": n,
        "S_percent": 100.0 * len(ok) / max(n, 1),
        "L_avg_m": L,
        "sigma_L_m": sL,
        "T_avg_ms": T,
        "sigma_T_ms": sT,
        "E_avg_m": E,
        "sigma_E_m": sE,
        "collision_rate_percent": 100.0 * sum(r.collided for r in results) / max(n, 1),
        "mean_steps": float(np.mean([r.steps for r in results])) if n else float("nan"),
        "mean_return": float(np.mean([r.episode_return for r in results])) if n else float("nan"),
        "min_clearance_m": float(np.nanmean([r.min_clearance for r in results])) if n else float("nan"),
        "timeout_rate_percent": 100.0 * sum((not r.success) and (not r.collided) for r in results) / max(n, 1),
    }


PAPER_TABLE_I = [
    {"algorithm": "DDPG (paper)", "S_percent": 87.3, "L_avg_m": 1.11, "sigma_L_m": 0.65, "T_avg_ms": 9.55, "sigma_T_ms": 8.17, "E_avg_m": 0.031, "sigma_E_m": 0.007},
    {"algorithm": "RRT (paper)", "S_percent": 36.8, "L_avg_m": 1.05, "sigma_L_m": 0.35, "T_avg_ms": 3425.2, "sigma_T_ms": 369.7, "E_avg_m": 0.032, "sigma_E_m": 0.005},
]


def format_table(rows: list[dict]) -> str:
    header = "| Algorithm | S(%) | Lavg(m) | σL(m) | Tavg(ms) | σT(ms) | Eavg(m) | σE(m) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['algorithm']} | {r['S_percent']:.1f} | {r['L_avg_m']:.2f} | {r['sigma_L_m']:.2f} | "
            f"{r['T_avg_ms']:.2f} | {r['sigma_T_ms']:.2f} | {r['E_avg_m']:.3f} | {r['sigma_E_m']:.3f} |"
        )
    return "\n".join(lines)
