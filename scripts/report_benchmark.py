"""Aggregate runs/bench into an IEEE-ready report: tables (Markdown + LaTeX), figures, JSON.

    python scripts/report_benchmark.py [--bench runs/bench] [--out paper]

Outputs
  <bench>/results.json              every (scene, variant, seed) with train cost + eval metrics
  <bench>/REPORT.md                 main tables, ablation ladder with deltas and costs, stratified results
  <out>/tables/*.tex                booktabs tables for the paper
  <out>/figures/*.png|pdf           learning curves (mean +- 95 % CI over seeds), cost-vs-success, ladder
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# validated categorical palette (dataviz skill reference instance), fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e", "#8a8f99", "#7a4f1d", "#0b0b0b", "#1a8ca0", "#b5179e"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e4e0"


# ------------------------------------------------------------------ collection
def read_curve(path: Path, bin_size: int = 100) -> dict:
    rows = list(csv.DictReader(open(path)))
    succ = np.array([int(r["reached"]) for r in rows], dtype=float)
    ret = np.array([float(r["return"]) for r in rows])
    steps = np.array([int(r["env_steps"]) for r in rows]) if "env_steps" in rows[0] else np.cumsum([int(r["steps"]) for r in rows])
    n = len(rows) // bin_size
    out = {
        "episode": [int((i + 1) * bin_size) for i in range(n)],
        "success": [float(succ[i * bin_size:(i + 1) * bin_size].mean()) for i in range(n)],
        "return": [float(ret[i * bin_size:(i + 1) * bin_size].mean()) for i in range(n)],
        "env_steps": [int(steps[(i + 1) * bin_size - 1]) for i in range(n)],
    }
    # rounds / env steps to reach a 90 % (moving 200-episode) success rate
    mov = np.convolve(succ, np.ones(200) / 200, mode="valid") if len(succ) >= 200 else np.array([])
    hit = np.argmax(mov >= 0.9) if mov.size and (mov >= 0.9).any() else None
    out["rounds_to_90"] = int(hit + 200) if hit is not None else None
    out["steps_to_90"] = int(steps[hit + 199]) if hit is not None else None
    out["final_success_last500"] = float(succ[-500:].mean())
    return out


def collect(bench_dir: Path) -> list[dict]:
    rows = []
    for table in sorted(bench_dir.glob("*/*/seed*/eval/table1.json")):
        d = table.parents[1]
        scene, variant, seed = d.parts[-3], d.parts[-2], int(d.parts[-1][4:])
        summ = json.load(open(d / "train_summary.json")) if (d / "train_summary.json").exists() else {}
        t1 = json.load(open(table))
        main = t1[0]
        rec = {
            "scene": scene, "variant": variant, "seed": seed, "label": main["algorithm"],
            "S": main["S_percent"], "collision": main.get("collision_rate_percent", float("nan")),
            "timeout": main.get("timeout_rate_percent", float("nan")),
            "L": main["L_avg_m"], "sL": main["sigma_L_m"], "T_ms": main["T_avg_ms"], "sT_ms": main["sigma_T_ms"],
            "E": main["E_avg_m"], "sE": main["sigma_E_m"], "min_clearance": main.get("min_clearance_m", float("nan")),
            "mean_return": main.get("mean_return", float("nan")), "mean_steps": main.get("mean_steps", float("nan")),
            "train_wall_s": summ.get("wall_time_s", float("nan")), "env_steps": summ.get("total_env_steps", float("nan")),
            "updates": summ.get("updates", float("nan")),
            "plugin_time_s": sum(float(p.get("time_s", 0.0)) for p in (summ.get("plugins") or {}).values()),
            "extra_sim_steps": sum(int(p.get("extra_sim_steps", 0)) for p in (summ.get("plugins") or {}).values()),
            "extra_transitions": sum(int(p.get("extra_transitions", 0)) for p in (summ.get("plugins") or {}).values()),
            "plugins": summ.get("plugins", {}),
        }
        rrt = [r for r in t1 if r["algorithm"].startswith("RRT (ours)")]
        if rrt:
            rec["rrt"] = rrt[0]
        if (d / "train_log.csv").exists():
            rec["curve"] = read_curve(d / "train_log.csv")
        if (d / "eval" / "stratified.json").exists():
            rec["stratified"] = json.load(open(d / "eval" / "stratified.json"))
        rows.append(rec)
    return rows


# ------------------------------------------------------------------ aggregation
def rel(p: Path) -> str:
    """Path relative to the repo when possible (an --out outside the repo stays absolute)."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def ci95(values):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    if v.size == 1:
        return float(v[0]), 0.0, 1
    return float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(v.size)), int(v.size)


def fmt(m, h, nd=1):
    if math.isnan(m):
        return "n/a"
    return f"{m:.{nd}f}" if (h == 0 or math.isnan(h)) else f"{m:.{nd}f} ± {h:.{nd}f}"


def aggregate(rows, bench):
    agg = {}
    for sc in bench["scenes"]:
        for var in bench["variants"]:
            rs = [r for r in rows if r["scene"] == sc and r["variant"] == var]
            if not rs:
                continue
            a = {"n": len(rs), "label": bench["variants"][var]["label"], "method_label": rs[0]["label"]}
            for k in ["S", "collision", "timeout", "L", "T_ms", "E", "min_clearance", "mean_return", "mean_steps",
                      "train_wall_s", "env_steps", "updates", "plugin_time_s", "extra_sim_steps", "extra_transitions"]:
                a[k] = ci95([r[k] for r in rs])
            curves = [r["curve"] for r in rs if "curve" in r]
            if curves:
                a["rounds_to_90"] = ci95([c["rounds_to_90"] for c in curves if c["rounds_to_90"] is not None]) if any(c["rounds_to_90"] for c in curves) else (float("nan"), float("nan"), 0)
                a["reach90_frac"] = float(np.mean([c["rounds_to_90"] is not None for c in curves]))
                a["final_success_last500"] = ci95([c["final_success_last500"] for c in curves])
            rrts = [r["rrt"] for r in rs if "rrt" in r]
            if rrts:
                a["rrt"] = {k: ci95([x[k] for x in rrts]) for k in ["S_percent", "L_avg_m", "T_avg_ms", "E_avg_m", "collision_rate_percent"] if k in rrts[0]}
            agg[(sc, var)] = a
    return agg


# ------------------------------------------------------------------ tables
def md_main_table(agg, bench, sc):
    lines = [f"| Method | S (%) | collision (%) | timeout (%) | L (m) | T (ms) | E (m) | min clearance (m) | rounds→90 % | train wall (min) | env steps |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for var in bench["variants"]:
        a = agg.get((sc, var))
        if not a:
            continue
        r90 = a.get("rounds_to_90", (float("nan"), float("nan"), 0))
        r90s = ("never" if a.get("reach90_frac", 0) == 0 else fmt(r90[0], r90[1], 0) + (f" ({a['reach90_frac']:.0%})" if a.get("reach90_frac", 1) < 1 else "")) if "rounds_to_90" in a else "n/a"
        lines.append(f"| {a['label']} | {fmt(*a['S'][:2])} | {fmt(*a['collision'][:2])} | {fmt(*a['timeout'][:2])} | {fmt(*a['L'][:2], 2)} | {fmt(*a['T_ms'][:2], 2)} | {fmt(*a['E'][:2], 3)} | {fmt(*a['min_clearance'][:2], 3)} | {r90s} | {fmt(a['train_wall_s'][0] / 60, a['train_wall_s'][1] / 60)} | {fmt(a['env_steps'][0] / 1e3, a['env_steps'][1] / 1e3, 0)}k |")
        if "rrt" in a:
            rr = a["rrt"]
            lines.append(f"| RRT (plan on t=0 snapshot, executed) | {fmt(*rr['S_percent'][:2])} | {fmt(*rr['collision_rate_percent'][:2])} | – | {fmt(*rr['L_avg_m'][:2], 2)} | {fmt(*rr['T_avg_ms'][:2], 1)} | {fmt(*rr['E_avg_m'][:2], 3)} | – | – | – | – |")
    return "\n".join(lines)


def md_ladder(agg, bench, sc):
    lines = ["| Step | adds | S (%) | ΔS vs previous | collision (%) | Δcoll | return | Δreturn | train wall (min) | Δwall | env steps (k) | extra sim steps (k) | plugin time (s) | T infer (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for var, v in bench["variants"].items():
        a = agg.get((sc, var))
        if not a:
            continue
        b = agg.get((sc, v["base"])) if v.get("base") else None

        def delta(key, nd=1, scale=1.0):
            if not b:
                return "–"
            d = (a[key][0] - b[key][0]) * scale
            return f"{d:+.{nd}f}"

        lines.append(f"| {var} | {v['label']} | {fmt(*a['S'][:2])} | {delta('S')} | {fmt(*a['collision'][:2])} | {delta('collision')} | {fmt(*a['mean_return'][:2])} | {delta('mean_return')} | {fmt(a['train_wall_s'][0] / 60, a['train_wall_s'][1] / 60)} | {delta('train_wall_s', 1, 1 / 60)} | {a['env_steps'][0] / 1e3:.0f} | {a['extra_sim_steps'][0] / 1e3:.0f} | {a['plugin_time_s'][0]:.0f} | {fmt(*a['T_ms'][:2], 2)} |")
    return "\n".join(lines)


def tex_escape(s: str) -> str:
    return s.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_").replace("→", "$\\rightarrow$").replace("±", "$\\pm$").replace("Δ", "$\\Delta$")


def md_to_tex(md: str, caption: str, label: str) -> str:
    lines = [l for l in md.splitlines() if l.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    body = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
    cols = "l" + "r" * (len(header) - 1)
    out = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{tex_escape(caption)}}}", f"\\label{{{label}}}", "\\small",
           f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(tex_escape(h) for h in header) + " \\\\", "\\midrule"]
    for row in body:
        out.append(" & ".join(tex_escape(c).replace("–", "--") for c in row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out)


# ------------------------------------------------------------------ figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def fig_curves(rows, bench, sc, out: Path, key="success"):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style(ax)
    for i, var in enumerate(bench["variants"]):
        curves = [r["curve"] for r in rows if r["scene"] == sc and r["variant"] == var and "curve" in r and not bench["variants"][var].get("eval_only")]
        if not curves:
            continue
        n = min(len(c["episode"]) for c in curves)
        x = np.array(curves[0]["episode"][:n])
        y = np.array([c[key][:n] for c in curves])
        m = y.mean(0)
        h = 1.96 * y.std(0, ddof=1) / math.sqrt(len(y)) if len(y) > 1 else np.zeros_like(m)
        col = PALETTE[i % len(PALETTE)]
        ax.plot(x, m * (100 if key == "success" else 1), color=col, linewidth=2, label=bench["variants"][var]["label"])
        ax.fill_between(x, (m - h) * (100 if key == "success" else 1), (m + h) * (100 if key == "success" else 1), color=col, alpha=0.12, linewidth=0)
    ax.set_xlabel("training rounds")
    if key == "success":
        ax.set_ylim(-2, 102)  # CI bands of a 0/1 statistic can leave [0, 100]
    ax.set_ylabel("success rate (%) per 100 rounds" if key == "success" else "average return per 100 rounds")
    ax.set_title(bench["scenes"][sc]["label"], fontsize=10, color=TEXT, loc="left")
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_ladder(agg, bench, out: Path):
    scenes = list(bench["scenes"])
    variants = [v for v in bench["variants"] if any((sc, v) in agg for sc in scenes)]
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.6 * len(scenes), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scenes):
        style(ax)
        ys = np.arange(len(variants))
        for k, var in enumerate(variants):
            a = agg.get((sc, var))
            if not a:
                continue
            ax.barh(k, a["S"][0], xerr=a["S"][1], color=PALETTE[k % len(PALETTE)], height=0.62, error_kw={"ecolor": MUTED, "capsize": 2, "linewidth": 1})
            ax.text(a["S"][0] + a["S"][1] + 1.5, k, f"{a['S'][0]:.1f}", va="center", fontsize=7, color=TEXT)
        ax.set_yticks(ys)
        ax.set_yticklabels([bench["variants"][v]["label"] for v in variants], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 122)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("success (%)")
        ax.set_title(textwrap.fill(bench["scenes"][sc]["label"], 30), fontsize=9, loc="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_cost(agg, bench, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax in axes:
        style(ax)
    for i, var in enumerate(bench["variants"]):
        pts = [agg[(sc, var)] for sc in bench["scenes"] if (sc, var) in agg and not bench["variants"][var].get("eval_only")]
        if not pts:
            continue
        col = PALETTE[i % len(PALETTE)]
        axes[0].scatter([p["train_wall_s"][0] / 60 for p in pts], [p["S"][0] for p in pts], color=col, s=36, label=bench["variants"][var]["label"], edgecolor="white", linewidth=0.8)
        axes[1].scatter([p["T_ms"][0] for p in pts], [p["collision"][0] for p in pts], color=col, s=36, edgecolor="white", linewidth=0.8)
    axes[0].set_xlabel("training wall time (min)")
    axes[0].set_ylabel("test success (%)")
    axes[1].set_xlabel("inference time per round (ms)")
    axes[1].set_ylabel("collision rate (%)")
    fig.legend(fontsize=6.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_curves_all(rows, bench, out: Path, key="success"):
    """One panel per world, shared legend below (paper Fig.)."""
    scenes = [sc for sc in bench["scenes"] if any(r["scene"] == sc and "curve" in r for r in rows)]
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.4 * len(scenes), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    handles = {}
    for ax, sc in zip(axes, scenes):
        style(ax)
        for i, var in enumerate(bench["variants"]):
            if bench["variants"][var].get("eval_only"):
                continue
            curves = [r["curve"] for r in rows if r["scene"] == sc and r["variant"] == var and "curve" in r]
            if not curves:
                continue
            n = min(len(c["episode"]) for c in curves)
            x = np.array(curves[0]["episode"][:n])
            y = np.array([c[key][:n] for c in curves]) * (100 if key == "success" else 1)
            m = y.mean(0)
            h = 1.96 * y.std(0, ddof=1) / math.sqrt(len(y)) if len(y) > 1 else np.zeros_like(m)
            col = PALETTE[i % len(PALETTE)]
            (ln,) = ax.plot(x, m, color=col, linewidth=1.8)
            ax.fill_between(x, m - h, m + h, color=col, alpha=0.10, linewidth=0)
            handles[bench["variants"][var]["label"]] = ln
        ax.set_title(bench["scenes"][sc]["label"], fontsize=9, loc="left", color=TEXT)
        ax.set_xlabel("training rounds")
    axes[0].set_ylabel("success (%) per 100 rounds" if key == "success" else "return per 100 rounds")
    fig.legend(handles.values(), handles.keys(), fontsize=7, frameon=False, ncol=min(5, len(handles)), loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_stratified(rows, bench, out: Path, variants=("paper_ddpg", "td3", "td3_her_apf", "td3_her_apf_shield", "shield_eval_1cm")):
    """Success per TDA-difficulty quartile, grouped bars, one panel per world."""
    scenes = [sc for sc in bench["scenes"] if any(r["scene"] == sc and "stratified" in r for r in rows)]
    if not scenes:
        return
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.4 * len(scenes), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scenes):
        style(ax)
        vs = [v for v in variants if any(r["scene"] == sc and r["variant"] == v and "stratified" in r for r in rows)]
        width = 0.8 / max(len(vs), 1)
        for k, v in enumerate(vs):
            strat = [r["stratified"] for r in rows if r["scene"] == sc and r["variant"] == v and "stratified" in r]
            bins = [st["algorithms"]["ddpg"]["bins"] for st in strat if "ddpg" in st.get("algorithms", {})]
            nb = min(len(b) for b in bins)
            succ = [100 * np.mean([b[i]["success_rate"] for b in bins]) for i in range(nb)]
            xs = np.arange(nb) + (k - (len(vs) - 1) / 2) * width
            ax.bar(xs, succ, width=width * 0.92, color=PALETTE[list(bench["variants"]).index(v) % len(PALETTE)], label=bench["variants"][v]["label"])
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels([f"Q{i + 1}" for i in range(4)])
        ax.set_xlabel("scene difficulty quartile (τ)")
        ax.set_title(bench["scenes"][sc]["label"], fontsize=9, loc="left", color=TEXT)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("success (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, ncol=min(5, len(l)), loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


# ------------------------------------------------------------------ stratified
def md_stratified(rows, bench, sc):
    """Per-variant success per TDA-difficulty bin, averaged over seeds (format of scripts/stratify.py)."""
    out = []
    nb = 0
    for var in bench["variants"]:
        strat = [r["stratified"] for r in rows if r["scene"] == sc and r["variant"] == var and "stratified" in r]
        if not strat:
            continue
        algs = sorted({a for st in strat for a in st.get("algorithms", {})})
        for alg in algs:
            if alg == "rrt" and var != "paper_ddpg":
                continue
            bins = [st["algorithms"][alg]["bins"] for st in strat if alg in st.get("algorithms", {})]
            k = min(len(b) for b in bins)
            nb = max(nb, k)
            succ = [np.mean([b[i]["success_rate"] for b in bins]) for i in range(k)]
            coll = [np.mean([b[i]["collision_rate"] for b in bins]) for i in range(k)]
            rho = np.nanmean([st["algorithms"][alg].get("correlation", {}).get("success", {}).get("spearman_rho", float("nan")) for st in strat if alg in st.get("algorithms", {})])
            hom = [st["algorithms"][alg].get("homotopy", {}).get("all", {}).get("n_classes", float("nan")) for st in strat if alg in st.get("algorithms", {})]
            name = bench["variants"][var]["label"] if alg == "ddpg" else "RRT"
            out.append(f"| {name} | " + " | ".join(f"{100 * a:.0f} / {100 * c:.0f}" for a, c in zip(succ, coll)) + f" | {rho:.2f} | {np.nanmean(hom):.1f} |")
    if not out:
        return ""
    header = "| Method | " + " | ".join(f"τ bin {i + 1}: S / coll (%)" for i in range(nb)) + " | Spearman(τ, success) | homotopy classes |"
    return "\n".join([header, "|" + "---|" * (nb + 3)] + out)


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="runs/bench")
    p.add_argument("--matrix", default="configs/benchmark.yaml")
    p.add_argument("--out", default="paper")
    args = p.parse_args()
    bench_dir = ROOT / args.bench
    bench = yaml.safe_load(open(ROOT / args.matrix))
    out = ROOT / args.out
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows = collect(bench_dir)
    if not rows:
        print("no finished evaluations under", bench_dir)
        return
    json.dump([{k: v for k, v in r.items() if k != "curve"} for r in rows], open(bench_dir / "results.json", "w"), indent=1)
    agg = aggregate(rows, bench)
    used = yaml.safe_load(open(bench_dir / "matrix_used.yaml")) if (bench_dir / "matrix_used.yaml").exists() else {}

    md = [f"# Benchmark report", "",
          f"Matrix: {used.get('episodes', '?')} training rounds, {used.get('eval_episodes', '?')} test rounds, seeds {used.get('seeds', '?')}. "
          f"Values are mean ± 95 % CI over seeds. RRT plans on the t = 0 snapshot and its path is executed in the (moving) world.", ""]
    for sc in bench["scenes"]:
        if not any(k[0] == sc for k in agg):
            continue
        md += [f"## {bench['scenes'][sc]['label']}", "", "### Main results", "", md_main_table(agg, bench, sc), "",
               "### Ablation ladder (each row vs. the rung it builds on)", "", md_ladder(agg, bench, sc), ""]
        st = md_stratified(rows, bench, sc)
        if st:
            md += ["### Stratified by TDA scene complexity τ (quantile bins)", "", st, ""]
        md += [f"![learning curves]({rel(out / 'figures' / f'curves_{sc}.png')})", ""]
        (out / "tables" / f"main_{sc}.tex").write_text(md_to_tex(md_main_table(agg, bench, sc), f"Results on {bench['scenes'][sc]['label']}", f"tab:main_{sc}"))
        (out / "tables" / f"ladder_{sc}.tex").write_text(md_to_tex(md_ladder(agg, bench, sc), f"Ablation ladder on {bench['scenes'][sc]['label']}", f"tab:ladder_{sc}"))
        fig_curves(rows, bench, sc, out / "figures" / f"curves_{sc}")
        fig_curves(rows, bench, sc, out / "figures" / f"returns_{sc}", key="return")
    fig_ladder(agg, bench, out / "figures" / "ladder")
    fig_cost(agg, bench, out / "figures" / "cost")
    fig_curves_all(rows, bench, out / "figures" / "curves_all")
    fig_curves_all(rows, bench, out / "figures" / "returns_all", key="return")
    fig_stratified(rows, bench, out / "figures" / "stratified")
    # plugin-level stats (shield activations etc.)
    md += ["## Plugin statistics (mean over seeds and scenes)", "", "| variant | plugin | stat | value |", "|---|---|---|---|"]
    stats = defaultdict(list)
    for r in rows:
        for pname, st in (r.get("plugins") or {}).items():
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    stats[(r["variant"], pname, k)].append(float(v))
    for (var, pname, k), vals in sorted(stats.items()):
        md.append(f"| {var} | {pname} | {k} | {np.mean(vals):.4g} |")
    md += ["", f"Figures: `{rel(out / 'figures')}`; LaTeX tables: `{rel(out / 'tables')}`; raw: `{rel(bench_dir / 'results.json')}`."]
    (bench_dir / "REPORT.md").write_text("\n".join(md) + "\n")
    print(bench_dir / "REPORT.md")


if __name__ == "__main__":
    main()
