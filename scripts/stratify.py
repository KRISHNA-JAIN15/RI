"""Stratify evaluation results by persistent-homology scene difficulty (C5).

Regenerates the evaluation scenes (same config + seed as scripts/evaluate.py), computes
``tau`` for each with :func:`arm_rl.topology.scene_complexity`, joins them with
``ddpg_episodes.json`` (and ``rrt_episodes.json`` if present) and reports success rate,
collision rate and mean path length per ``tau`` quantile bin, the Spearman correlation
between ``tau`` and success, and, if ``paths.npz`` exists (``evaluate.py --save-paths``),
the distribution of path homotopy signatures per algorithm.

Usage:
    python scripts/stratify.py --eval-dir runs/ddpg/eval --config runs/ddpg/config.yaml --seed 12345 [--bins 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.topology import path_homotopy_signature, scene_complexity, signature_stats  # noqa: E402
from arm_rl.utils import load_config, write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True, help="the --seed used by evaluate.py")
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=4000, help="grid points for scene_complexity")
    p.add_argument("--max-dim", type=int, default=1)
    p.add_argument("--pad", type=float, default=0.15)
    return p.parse_args()


def quantile_bins(tau: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign each tau to one of ``n_bins`` quantile bins (ties broken by rank)."""
    order = np.argsort(tau, kind="stable")
    ranks = np.empty(len(tau), dtype=np.int64)
    ranks[order] = np.arange(len(tau))
    bins = np.minimum((ranks * n_bins) // max(1, len(tau)), n_bins - 1)
    edges = np.quantile(tau, np.linspace(0.0, 1.0, n_bins + 1)) if len(tau) else np.zeros(n_bins + 1)
    return bins, edges


def stratify(episodes: list[dict], tau: np.ndarray, bins: np.ndarray, n_bins: int) -> list[dict]:
    rows = []
    for b in range(n_bins):
        idx = np.where(bins[: len(episodes)] == b)[0]
        ep = [episodes[i] for i in idx]
        if not ep:
            rows.append({"bin": b, "n": 0})
            continue
        rows.append({
            "bin": b,
            "n": len(ep),
            "tau_min": float(tau[idx].min()),
            "tau_max": float(tau[idx].max()),
            "tau_mean": float(tau[idx].mean()),
            "success_rate": float(np.mean([e["success"] for e in ep])),
            "collision_rate": float(np.mean([e["collided"] for e in ep])),
            "mean_path_length": float(np.mean([e["path_length"] for e in ep])),
            "mean_final_error": float(np.mean([e["final_error"] for e in ep])),
        })
    return rows


def correlations(episodes: list[dict], tau: np.ndarray) -> dict:
    n = len(episodes)
    succ = np.array([float(e["success"]) for e in episodes])
    coll = np.array([float(e["collided"]) for e in episodes])
    out = {}
    for name, y in (("success", succ), ("collided", coll)):
        if n > 2 and y.std() > 0 and tau[:n].std() > 0:
            rho, pval = spearmanr(tau[:n], y)
            out[name] = {"spearman_rho": float(rho), "p_value": float(pval)}
        else:
            out[name] = {"spearman_rho": float("nan"), "p_value": float("nan")}
    return out


def md_table(rows: list[dict]) -> str:
    head = "| bin | n | tau range | mean tau | success | collision | mean path (m) | final err (m) |\n|---|---|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        if r["n"] == 0:
            body += f"| {r['bin']} | 0 | - | - | - | - | - | - |\n"
            continue
        body += (f"| {r['bin']} | {r['n']} | {r['tau_min']:.3f}-{r['tau_max']:.3f} | {r['tau_mean']:.3f} | "
                 f"{100 * r['success_rate']:.1f} % | {100 * r['collision_rate']:.1f} % | {r['mean_path_length']:.3f} | {r['mean_final_error']:.3f} |\n")
    return head + body


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    cfg = load_config(args.config)
    results = {}
    for algo, fname in (("ddpg", "ddpg_episodes.json"), ("rrt", "rrt_episodes.json")):
        if (eval_dir / fname).exists():
            results[algo] = json.load(open(eval_dir / fname))
    if not results:
        sys.exit(f"no ddpg_episodes.json / rrt_episodes.json in {eval_dir}")
    n_scenes = max(len(v) for v in results.values())

    # Regenerate the evaluation scenes exactly as scripts/evaluate.py does.
    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    scene_rng = np.random.default_rng(args.seed)
    scenes = [env.scene_gen.sample(scene_rng) for _ in range(n_scenes)]
    start = env.arm.ee_position(env.arm.home)

    comp = [scene_complexity(s, env.arm.shoulder, env.arm.reach, n_samples=args.n_samples, max_dim=args.max_dim, start=start, pad=args.pad)
            for s in scenes]
    tau = np.array([c["tau"] for c in comp])
    bins, edges = quantile_bins(tau, args.bins)

    report = {
        "eval_dir": str(eval_dir), "config": args.config, "seed": args.seed, "n_scenes": n_scenes, "bins": args.bins,
        "tau_settings": {"n_samples": args.n_samples, "max_dim": args.max_dim, "pad": args.pad},
        "tau": {"mean": float(tau.mean()), "std": float(tau.std()), "min": float(tau.min()), "max": float(tau.max()),
                "edges": [float(e) for e in edges],
                "mean_components": {k: float(np.mean([c[k] for c in comp])) for k in ("tau_h0", "tau_bottleneck", "tau_h1", "tau_h2")},
                "compute_time_s": float(sum(c["time_s"] for c in comp))},
        "per_scene_tau": [float(t) for t in tau],
        "algorithms": {},
    }
    md = [f"# Results stratified by scene difficulty tau\n",
          f"eval dir `{eval_dir}`, config `{args.config}`, seed {args.seed}, {n_scenes} scenes, {args.bins} quantile bins.\n",
          f"tau: mean {tau.mean():.3f}, std {tau.std():.3f}, range {tau.min():.3f}-{tau.max():.3f} "
          f"(mean components: H0 pockets {report['tau']['mean_components']['tau_h0']:.3f}, bottleneck {report['tau']['mean_components']['tau_bottleneck']:.3f}, "
          f"H1 {report['tau']['mean_components']['tau_h1']:.3f}, H2 {report['tau']['mean_components']['tau_h2']:.3f}).\n"]

    # Homotopy signatures of the saved paths (optional).
    paths = np.load(eval_dir / "paths.npz") if (eval_dir / "paths.npz").exists() else None

    for algo, episodes in results.items():
        rows = stratify(episodes, tau, bins, args.bins)
        corr = correlations(episodes, tau)
        entry = {"n": len(episodes), "overall_success": float(np.mean([e["success"] for e in episodes])), "bins": rows, "correlation": corr}
        md.append(f"\n## {algo.upper()} (n = {len(episodes)}, overall success {100 * entry['overall_success']:.1f} %)\n")
        md.append(md_table(rows))
        md.append(f"\nSpearman(tau, success) = {corr['success']['spearman_rho']:.3f} (p = {corr['success']['p_value']:.3g}); "
                  f"Spearman(tau, collision) = {corr['collided']['spearman_rho']:.3f} (p = {corr['collided']['p_value']:.3g})\n")
        if paths is not None and f"{algo}_paths" in paths:
            P, L = paths[f"{algo}_paths"], paths[f"{algo}_lengths"]
            sigs, sigs_success = [], []
            for i in range(min(len(P), len(episodes), n_scenes)):
                if L[i] < 2:
                    continue
                sig = path_homotopy_signature(P[i, : L[i]], scenes[i])
                sigs.append(sig)
                if episodes[i]["success"]:
                    sigs_success.append(sig)
            st_all, st_succ = signature_stats(sigs), signature_stats(sigs_success)
            entry["homotopy"] = {"all": st_all, "successful": st_succ}
            md.append(f"\nHomotopy signatures (all {st_all['n']} paths): {st_all['n_classes']} distinct classes, most common class share "
                      f"{100 * st_all['top_share']:.1f} %, {100 * st_all['nontrivial_fraction']:.1f} % of paths pass at least one obstacle on a definite side "
                      f"(|swept angle| >= pi/2). Successful paths only ({st_succ['n']}): {st_succ['n_classes']} classes, top share {100 * st_succ['top_share']:.1f} %.\n")
        report["algorithms"][algo] = entry

    # DDPG vs RRT agreement of the route choice on scenes both solved.
    if paths is not None and "ddpg_paths" in paths and "rrt_paths" in paths and "ddpg" in results and "rrt" in results:
        agree, total = 0, 0
        for i in range(min(len(results["ddpg"]), len(results["rrt"]), n_scenes)):
            if results["ddpg"][i]["success"] and results["rrt"][i]["success"] and paths["ddpg_lengths"][i] >= 2 and paths["rrt_lengths"][i] >= 2:
                total += 1
                sd = path_homotopy_signature(paths["ddpg_paths"][i, : paths["ddpg_lengths"][i]], scenes[i])
                sr = path_homotopy_signature(paths["rrt_paths"][i, : paths["rrt_lengths"][i]], scenes[i])
                agree += int(sd == sr)
        report["route_agreement"] = {"n_both_success": total, "same_signature_fraction": (agree / total) if total else float("nan")}
        if total:
            md.append(f"\nDDPG and RRT take the same homotopy signature on {agree}/{total} = {100 * agree / total:.1f} % of the scenes both solved.\n")

    text = "".join(md)
    print(text)
    (eval_dir / "stratified.md").write_text(text)
    write_json(report, eval_dir / "stratified.json")
    print(f"saved to {eval_dir / 'stratified.md'} and {eval_dir / 'stratified.json'}")


if __name__ == "__main__":
    main()
