"""Paired per-scene significance tests between variants that were evaluated on identical scenes.

Every variant is tested on the same 1000 scenes per evaluation seed, so the success indicators
can be paired scene by scene and pooled over training seeds. For each (scene world, variant,
base) we report the paired difference in success rate with a bootstrap 95 % CI and an exact
McNemar p-value on the discordant pairs, plus the same for the collision indicator.

    python scripts/paired_stats.py [--bench runs/bench] [--out paper/tables]
Writes paired.md, paired.json (in the bench dir), paired.tex and macros appended to numbers.tex.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_tables import SCENE_SHORT, SHORT  # noqa: E402


def load_episodes(d: Path, name="ddpg_episodes.json"):
    f = d / "eval" / name
    return json.load(open(f)) if f.exists() else None


def paired(a: np.ndarray, b: np.ndarray, n_boot=5000, seed=0):
    """a, b: 0/1 arrays of equal length (variant, base). Returns diff (pp), CI, McNemar p."""
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
    return 100 * d.mean(), 100 * lo, 100 * hi, p, n10, n01


def macro(name, value):
    return "\\newcommand{\\%s}{%s}" % (re.sub(r"[^A-Za-z]", "", name), value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="runs/bench")
    ap.add_argument("--matrix", default="configs/benchmark.yaml")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--vs-baseline", action="store_true", help="also compare every variant with paper_ddpg")
    args = ap.parse_args()
    bench = yaml.safe_load(open(ROOT / args.matrix))
    bdir = ROOT / args.bench
    results, md, macros = [], [], []
    comparisons = []
    for v, spec in bench["variants"].items():
        if spec.get("base"):
            comparisons.append((v, spec["base"]))
    comparisons += [(v, "paper_ddpg") for v in bench["variants"] if v != "paper_ddpg"]
    seen = set()
    for sc in bench["scenes"]:
        md += [f"## {bench['scenes'][sc]['label']}", "", "| variant | vs | n pairs | ΔS (pp) [95 % CI] | p (McNemar) | wins/losses | ΔC (pp) [95 % CI] | p |", "|---|---|---|---|---|---|---|---|"]
        for v, b in comparisons:
            if (sc, v, b) in seen:
                continue
            seen.add((sc, v, b))
            A, B, CA, CB = [], [], [], []
            for sd in sorted((bdir / sc / v).glob("seed*")) if (bdir / sc / v).exists() else []:
                ea = load_episodes(sd)
                eb = load_episodes(bdir / sc / b / sd.name)
                if ea is None or eb is None or len(ea) != len(eb):
                    continue
                A += [e["success"] for e in ea]
                B += [e["success"] for e in eb]
                CA += [e["collided"] for e in ea]
                CB += [e["collided"] for e in eb]
            if not A:
                continue
            dS, lo, hi, p, w, l = paired(np.array(A), np.array(B))
            dC, clo, chi, pc, _, _ = paired(np.array(CA), np.array(CB), seed=1)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            results.append({"scene": sc, "variant": v, "base": b, "n": len(A), "dS": dS, "dS_lo": lo, "dS_hi": hi, "p_S": p, "wins": w, "losses": l,
                            "dC": dC, "dC_lo": clo, "dC_hi": chi, "p_C": pc})
            md.append(f"| {v} | {b} | {len(A)} | {dS:+.1f} [{lo:+.1f}, {hi:+.1f}]{star} | {p:.2g} | {w}/{l} | {dC:+.1f} [{clo:+.1f}, {chi:+.1f}] | {pc:.2g} |")
            macros += [macro(f"pS{sc}{v}vs{b}", f"{dS:+.1f}"), macro(f"pSlo{sc}{v}vs{b}", f"{lo:+.1f}"), macro(f"pShi{sc}{v}vs{b}", f"{hi:+.1f}"),
                       macro(f"pP{sc}{v}vs{b}", f"{p:.3f}" if p >= 0.001 else "<0.001"), macro(f"pC{sc}{v}vs{b}", f"{dC:+.1f}"), macro(f"pPC{sc}{v}vs{b}", f"{pc:.3f}" if pc >= 0.001 else "<0.001")]
        md.append("")
    json.dump(results, open(bdir / "paired.json", "w"), indent=1)
    (bdir / "paired.md").write_text("# Paired per-scene tests (pooled over seeds)\n\n" + "\n".join(md))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    # LaTeX: ladder comparisons only (variant vs its base), S and C with CI and p
    scenes = [sc for sc in bench["scenes"] if any(r["scene"] == sc for r in results)]
    L = ["\\begin{table*}[t]", "\\centering",
         "\\caption{Paired per-scene tests. Every variant is evaluated on the same scenes as the rung it builds on, so success and collision indicators are paired scene by scene and pooled over seeds ($n$ pairs). "
         "Entries: paired difference in percentage points with a bootstrap 95\\,\\% CI; $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (exact McNemar).}",
         "\\label{tab:paired}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{ll" + "rr" * len(scenes) + "}", "\\toprule",
         "Variant & vs. & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SCENE_SHORT.get(s, s)}}}" for s in scenes) + " \\\\",
         " & & " + " & ".join("$\\Delta S$ & $\\Delta C$" for _ in scenes) + " \\\\", "\\midrule"]
    def cell(r, key):
        d, lo, hi, p = r[f"d{key}"], r[f"d{key}_lo"], r[f"d{key}_hi"], r[f"p_{key}"]
        star = "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""
        return f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]{star}"
    for v, b in [(v, bench["variants"][v]["base"]) for v in bench["variants"] if bench["variants"][v].get("base")]:
        rs = {r["scene"]: r for r in results if r["variant"] == v and r["base"] == b}
        if not rs:
            continue
        cells = []
        for sc in scenes:
            r = rs.get(sc)
            cells += [cell(r, "S"), cell(r, "C")] if r else ["--", "--"]
        L.append(f"{SHORT.get(v, v)} & {SHORT.get(b, b)} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (out / "paired.tex").write_text("\n".join(L))
    with open(out / "numbers.tex", "a") as f:
        f.write("\n% paired tests\n" + "\n".join(macros) + "\n")
    print(bdir / "paired.md", out / "paired.tex", f"{len(results)} comparisons")


if __name__ == "__main__":
    main()
