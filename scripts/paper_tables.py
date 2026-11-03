"""Compact LaTeX tables + number macros for paper/main.tex from the benchmark results.

    python scripts/paper_tables.py [--bench runs/bench] [--out paper/tables]

Writes
  results_all.tex     one table: variants x scenes (S, collision) + cost columns
  cost.tex            per-step cost attribution (wall, env steps, extra sim steps, plugin time, inference)
  numbers.tex         \\newcommand macros used in the text (so the prose never goes stale)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_benchmark import aggregate, collect, fmt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHORT = {
    "paper_ddpg": "DDPG (paper)", "apf_only": "APF only", "shield_eval": "DDPG + shield$_{2\\,\\mathrm{cm}}$ (test)",
    "shield_eval_1cm": "DDPG + shield$_{1\\,\\mathrm{cm}}$ (test)", "td3": "TD3", "td3_her": "TD3+HER", "td3_her_obl": "TD3+HER+OBL",
    "td3_her_apfshape": "TD3+HER+APF$_\\Phi$", "td3_her_apf": "TD3+HER+APF", "td3_her_apf_shield": "TD3+HER+APF+shield",
    "full_set": "\\quad + set encoder", "full_curriculum": "\\quad + set + curriculum", "curriculum_mlp": "\\quad + curriculum (MLP)",
    "cem_rl": "CEM-RL (TD3+HER)",
}
SCENE_SHORT = {"static": "Static tree", "wind": "Tree in wind", "ur5e": "UR5e cell"}


def pm(t, nd=1):
    m, h, n = t
    if math.isnan(m):
        return "--"
    return f"{m:.{nd}f}" if (h == 0 or math.isnan(h)) else f"{m:.{nd}f}$\\pm${h:.{nd}f}"


def macro(name, value):
    import re

    name = re.sub(r"[^A-Za-z]", "", name)  # TeX macro names: letters only
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="runs/bench")
    p.add_argument("--matrix", default="configs/benchmark.yaml")
    p.add_argument("--out", default="paper/tables")
    args = p.parse_args()
    bench = yaml.safe_load(open(ROOT / args.matrix))
    rows = collect(ROOT / args.bench)
    agg = aggregate(rows, bench)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    scenes = [s for s in bench["scenes"] if any(k[0] == s for k in agg)]
    variants = [v for v in bench["variants"] if any((s, v) in agg for s in scenes)]

    # ---------------- results_all.tex
    cols = "l" + "rr" * len(scenes)
    L = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Test performance over 1000 rounds per seed (mean$\\pm$95\\,\\% CI over seeds): success rate $S$ and collision rate $C$ in \\%. "
         "Rows are the ablation ladder; each rung adds one component to the rung above it. RRT plans on the $t{=}0$ snapshot and its path is executed in the moving world.}",
         "\\label{tab:results}", "\\footnotesize", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
         "Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SCENE_SHORT.get(s, s)}}}" for s in scenes) + " \\\\",
         " & " + " & ".join("$S$ & $C$" for _ in scenes) + " \\\\", "\\midrule"]
    rrt_row = None
    for v in variants:
        cells = []
        for s in scenes:
            a = agg.get((s, v))
            cells += [pm(a["S"]), pm(a["collision"])] if a else ["--", "--"]
        L.append(f"{SHORT.get(v, v)} & " + " & ".join(cells) + " \\\\")
        if v == "paper_ddpg":
            rr = []
            for s in scenes:
                a = agg.get((s, v))
                rr += [pm(a["rrt"]["S_percent"]), pm(a["rrt"]["collision_rate_percent"])] if a and "rrt" in a else ["--", "--"]
            rrt_row = "RRT (executed) & " + " & ".join(rr) + " \\\\"
    if rrt_row:
        L.insert(L.index("\\midrule") + 1, rrt_row)
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "results_all.tex").write_text("\n".join(L))

    # ---------------- cost.tex (static scene = cheapest, plus ur5e)
    L = ["\\begin{table}[t]", "\\centering",
         "\\caption{What each step costs (static tree, mean over seeds): training wall time, environment steps, extra simulator steps charged to plugins, time inside plugins, and inference time per test round (policy + shield QP). $\\Delta S$ is the success change w.r.t.\\ the rung the row builds on.}",
         "\\label{tab:cost}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\linewidth}{!}{", "\\begin{tabular}{lrrrrrr}", "\\toprule",
         "Method & $\\Delta S$ & wall (min) & env steps (k) & extra sim (k) & plugin (s) & infer (ms) \\\\", "\\midrule"]
    sc = "static" if "static" in scenes else scenes[0]
    for v in variants:
        a = agg.get((sc, v))
        if not a:
            continue
        b = agg.get((sc, bench["variants"][v].get("base"))) if bench["variants"][v].get("base") else None
        dS = f"{a['S'][0] - b['S'][0]:+.1f}" if b else "--"
        eo = bench["variants"][v].get("eval_only")
        wall = "--" if eo else f"{a['train_wall_s'][0] / 60:.1f}"
        env_steps = "--" if eo else f"{a['env_steps'][0] / 1e3:.0f}"
        extra = "--" if eo else f"{a['extra_sim_steps'][0] / 1e3:.0f}"
        plug = f"{a['plugin_time_s'][0]:.0f}" if not eo else "--"
        L.append(f"{SHORT.get(v, v)} & {dS} & {wall} & {env_steps} & {extra} & {plug} & {a['T_ms'][0]:.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    (out / "cost.tex").write_text("\n".join(L))

    # ---------------- numbers.tex
    M = []
    for s in scenes:
        base = agg.get((s, "paper_ddpg"))
        if base:
            M.append(macro(f"S{s}base", f"{base['S'][0]:.1f}"))
            M.append(macro(f"C{s}base", f"{base['collision'][0]:.1f}"))
            if "rrt" in base:
                M.append(macro(f"S{s}rrt", f"{base['rrt']['S_percent'][0]:.1f}"))
                M.append(macro(f"T{s}rrt", f"{base['rrt']['T_avg_ms'][0]:.0f}"))
            M.append(macro(f"T{s}base", f"{base['T_ms'][0]:.2f}"))
        best = max([(agg[(s, v)]["S"][0], -agg[(s, v)]["collision"][0], v) for v in variants if (s, v) in agg and not bench["variants"][v].get("eval_only")], default=None)
        if best:
            M.append(macro(f"S{s}best", f"{best[0]:.1f}"))
            M.append(macro(f"C{s}best", f"{-best[1]:.1f}"))
            M.append(macro(f"V{s}best", SHORT.get(best[2], best[2])))
        for v in ["td3_her_apf_shield", "shield_eval", "shield_eval_1cm", "td3_her", "td3", "td3_her_obl", "cem_rl", "apf_only", "full_set", "full_curriculum", "curriculum_mlp", "td3_her_apf"]:
            a = agg.get((s, v))
            if a:
                key = v.replace("_", "")
                M.append(macro(f"S{s}{key}", f"{a['S'][0]:.1f}"))
                M.append(macro(f"C{s}{key}", f"{a['collision'][0]:.1f}"))
                M.append(macro(f"T{s}{key}", f"{a['T_ms'][0]:.2f}"))
    used = yaml.safe_load(open(ROOT / args.bench / "matrix_used.yaml")) if (ROOT / args.bench / "matrix_used.yaml").exists() else {}
    M.append(macro("nseeds", str(len(used.get("seeds", [])) or 3)))
    M.append(macro("ntrain", str(used.get("episodes", 5000))))
    M.append(macro("neval", str(used.get("eval_episodes", 1000))))
    shield_stats = [r["plugins"].get("shield", {}) for r in rows if "shield" in (r.get("plugins") or {})]
    if shield_stats:
        import numpy as np
        M.append(macro("shieldms", f"{np.mean([s.get('mean_solve_ms', float('nan')) for s in shield_stats]):.2f}"))
        M.append(macro("shieldact", f"{100 * np.mean([s.get('activation_rate', float('nan')) for s in shield_stats]):.0f}"))
    (out / "numbers.tex").write_text("\n".join(M) + "\n")
    print(out / "results_all.tex", out / "cost.tex", out / "numbers.tex")


if __name__ == "__main__":
    main()
