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

import numpy as np
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
    "rrt_replan": "RRT re-planning / 10 steps", "her_k1": "TD3+HER ($k{=}1$)", "apf_shield_eval": "APF + shield (no learning)",
    "paper_ddpg_10k": "DDPG (paper), 10k rounds", "full_10k": "TD3+HER+APF+shield, 10k rounds",
    "cem_s4": "CEM-RL $\\sigma_0{=}10^{-4}$", "cem_lr": "CEM-RL lr$_a{=}10^{-3}$", "cem_s4_lr": "CEM-RL $\\sigma_0{=}10^{-4}$, lr$_a{=}10^{-3}$",
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
        if v == "rrt_replan":
            continue
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
    rr2 = []
    for s_ in scenes:
        a = agg.get((s_, "rrt_replan"))
        rr2 += [pm(a["rrt_replan"]["S_percent"]), pm(a["rrt_replan"]["collision_rate_percent"])] if a and "rrt_replan" in a else ["--", "--"]
    if any(c != "--" for c in rr2):
        L.insert(L.index("\\midrule") + 2, "RRT re-planning / 10 steps & " + " & ".join(rr2) + " \\\\")
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

    # ---------------- efficiency.tex: sample efficiency, timeouts, clearance, shield statistics
    L = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Sample efficiency and safety statistics (mean over seeds). Rounds$\\to$90: training rounds until the 200-round moving success rate first reaches 90\\,\\% (fraction of seeds that reach it in parentheses when below 1); "
         "timeout = rounds that neither reach nor collide; clearance = mean over rounds of the minimum link--obstacle clearance; act.\\ = fraction of steps on which the shield QP changed the action; QP = mean solve time.}",
         "\\label{tab:eff}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{l" + "rrrrr" * len(scenes) + "}", "\\toprule",
         "Method & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{SCENE_SHORT.get(s, s)}}}" for s in scenes) + " \\\\",
         " & " + " & ".join("rounds$\\to$90 & timeout (\\%) & $L$ (m) & clear.\\ (cm) & act.\\,\\% / QP ms" for _ in scenes) + " \\\\", "\\midrule"]
    for v in variants:
        if v == "rrt_replan":
            continue
        cells = []
        for s_ in scenes:
            a = agg.get((s_, v))
            if not a:
                cells += ["--"] * 5
                continue
            eo = bench["variants"][v].get("eval_only")
            if eo or "rounds_to_90" not in a:
                r90 = "--"
            elif a.get("reach90_frac", 0) == 0:
                r90 = "never"
            else:
                r90 = f"{a['rounds_to_90'][0]:.0f}" + (f" ({a['reach90_frac']:.2f})" if a.get("reach90_frac", 1) < 1 else "")
            sh = [r for r in rows if r["scene"] == s_ and r["variant"] == v and "shield" in (r.get("plugins") or {})]
            if sh:
                act = 100 * float(np.mean([r["plugins"]["shield"].get("activation_rate", float("nan")) for r in sh]))
                qp = float(np.mean([r["plugins"]["shield"].get("mean_solve_ms", float("nan")) for r in sh]))
                shc = f"{act:.0f} / {qp:.2f}"
            else:
                shc = "--"
            cells += [r90, f"{a['timeout'][0]:.1f}", f"{a['L'][0]:.2f}", f"{100 * a['min_clearance'][0]:.1f}", shc]
        L.append(f"{SHORT.get(v, v)} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "efficiency.tex").write_text("\n".join(L))

    # ---------------- gains.tex: relative effect of each component vs the rung it builds on
    L = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Effect of each component relative to the rung it builds on (mean over seeds): change in success $\\Delta S$ and collision rate $\\Delta C$ in percentage points, "
         "change in training wall time $\\Delta W$ in \\%, and inference time per test round as a multiple of the base rung ($\\times T$). "
         "Test-time rows add the component to the trained DDPG policy without retraining.}",
         "\\label{tab:gains}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{ll" + "rrrr" * len(scenes) + "}", "\\toprule",
         "Component & vs. & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{SCENE_SHORT.get(s, s)}}}" for s in scenes) + " \\\\",
         " & & " + " & ".join("$\\Delta S$ & $\\Delta C$ & $\\Delta W$ & $\\times T$" for _ in scenes) + " \\\\", "\\midrule"]
    G = []
    for v in variants:
        bv = bench["variants"][v].get("base")
        if not bv:
            continue
        cells = []
        for sc in scenes:
            a, b = agg.get((sc, v)), agg.get((sc, bv))
            if not a or not b:
                cells += ["--"] * 4
                continue
            dS, dC = a["S"][0] - b["S"][0], a["collision"][0] - b["collision"][0]
            eo = bench["variants"][v].get("eval_only")
            dW = "--" if eo else f"{100 * (a['train_wall_s'][0] / b['train_wall_s'][0] - 1):+.0f}"
            xT = a["T_ms"][0] / b["T_ms"][0]
            cells += [f"{dS:+.1f}", f"{dC:+.1f}", dW, f"{xT:.1f}"]
            G += [macro(f"gS{sc}{v}", f"{dS:+.1f}"), macro(f"gC{sc}{v}", f"{dC:+.1f}"), macro(f"gW{sc}{v}", dW if dW == "--" else dW.replace("+", "+")), macro(f"gT{sc}{v}", f"{xT:.1f}")]
        L.append(f"{SHORT.get(v, v)} & {SHORT.get(bv, bv)} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "gains.tex").write_text("\n".join(L))

    # ---------------- numbers.tex
    M = list(G)
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
        rp = agg.get((s, "rrt_replan"))
        if rp and "rrt_replan" in rp:
            M += [macro(f"S{s}replan", f"{rp['rrt_replan']['S_percent'][0]:.1f}"), macro(f"C{s}replan", f"{rp['rrt_replan']['collision_rate_percent'][0]:.1f}"), macro(f"T{s}replan", f"{rp['rrt_replan']['T_avg_ms'][0]:.0f}")]
        for v in ["td3_her_apf_shield", "shield_eval", "shield_eval_1cm", "td3_her", "td3", "td3_her_obl", "cem_rl", "apf_only", "full_set", "full_curriculum", "curriculum_mlp", "td3_her_apf", "td3_her_apfshape",
                  "her_k1", "apf_shield_eval", "paper_ddpg_10k", "full_10k", "cem_s4", "cem_lr", "cem_s4_lr"]:
            a = agg.get((s, v))
            if a:
                key = v.replace("_", "")
                M.append(macro(f"S{s}{key}", f"{a['S'][0]:.1f}"))
                M.append(macro(f"C{s}{key}", f"{a['collision'][0]:.1f}"))
                M.append(macro(f"T{s}{key}", f"{a['T_ms'][0]:.2f}"))
                M.append(macro(f"O{s}{key}", f"{a['timeout'][0]:.1f}"))
                M.append(macro(f"L{s}{key}", f"{a['L'][0]:.2f}"))
                M.append(macro(f"N{s}{key}", str(a['n'])))
                if "rounds_to_90" in a and a.get("reach90_frac", 0) > 0:
                    M.append(macro(f"R{s}{key}", f"{a['rounds_to_90'][0]:.0f}"))
                else:
                    M.append(macro(f"R{s}{key}", "never"))
    used = yaml.safe_load(open(ROOT / args.bench / "matrix_used.yaml")) if (ROOT / args.bench / "matrix_used.yaml").exists() else {}
    M.append(macro("nseeds", str(len(used.get("seeds", [])) or 3)))
    M.append(macro("ntrain", str(used.get("episodes", 5000))))
    M.append(macro("neval", str(used.get("eval_episodes", 1000))))
    shield_stats = [r["plugins"].get("shield", {}) for r in rows if "shield" in (r.get("plugins") or {})]
    if shield_stats:
        M.append(macro("shieldms", f"{np.mean([s.get('mean_solve_ms', float('nan')) for s in shield_stats]):.2f}"))
        M.append(macro("shieldact", f"{100 * np.mean([s.get('activation_rate', float('nan')) for s in shield_stats]):.0f}"))
    (out / "numbers.tex").write_text("\n".join(M) + "\n")
    print(out / "results_all.tex", out / "cost.tex", out / "numbers.tex")


if __name__ == "__main__":
    main()
