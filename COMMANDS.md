# Commands

All commands run from `ddpg_arm_planning/` with the venv active:

```bash
cd /Users/krishnajain/Desktop/RI/ddpg_arm_planning
source ../.venv/bin/activate            # python 3.13 venv (torch, mujoco, matplotlib, pyyaml, pypdf)
```

## 0. Install / test

```bash
python3.13 -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests                # 11 test files: env, geometry, kinematics, mujoco, shield, apf, td3/her/obl, topology, cem_rl, set encoder
```

## 1. Train

```bash
# Paper baseline (Li et al. 2024): static fruit tree, DDPG, 10 000 rounds x 100 steps (~5-20 min CPU)
python scripts/train.py --config configs/default.yaml --out runs/ddpg_paper

# Shorter run, fixed seed, resume
python scripts/train.py --config configs/default.yaml --out runs/ddpg_5k --episodes 5000 --seed 0
python scripts/train.py --config configs/default.yaml --out runs/ddpg_5k --resume runs/ddpg_5k/checkpoints/ddpg_ep005000.pt

# Dynamic worlds
python scripts/train.py --config configs/fruit_tree_wind.yaml --out runs/fruit_tree_wind     # tree in wind, swaying fruit
python scripts/train.py --config configs/ur5e_industrial.yaml --out runs/ur5e_industrial     # MuJoCo UR5e cell (~40 min)
python scripts/train.py --config configs/ur5e_warehouse.yaml  --out runs/ur5e_warehouse
python scripts/train.py --config configs/ur5e_industrial.yaml --set env.sim.control=dynamic --out runs/ur5e_dyn   # torque servo instead of kinematic

# Any config key can be overridden with --set SECTION.KEY=VALUE (repeatable)
python scripts/train.py --set method.algo=td3 --set 'method.plugins=[her,apf,shield]' --set method.apf.prior=true --out runs/td3_her_apf_shield
python scripts/train.py --set 'method.plugins=[her]' --set method.her.k=1 --out runs/her_k1
python scripts/train.py --set 'method.plugins=[curriculum]' --out runs/tda_curriculum
python scripts/train.py --set method.encoder=set --set env.obs.mode=all_pairs --out runs/set_encoder
python scripts/train.py --set train.gamma=0.99 --set env.collision_mode=penalty --out runs/paper_literal   # diverges (critic Q>0)

# CEM-RL (population learner)
python scripts/train_cemrl.py --config configs/default.yaml --out runs/cemrl --episodes 5000

# Several short runs in parallel: sweep.sh EPISODES name:"k=v k=v" ...
scripts/sweep.sh 2000 tau1:"train.tau=0.001" tau5:"train.tau=0.005" g99:"train.gamma=0.99"
```

Outputs per run: `runs/<run>/ddpg_final.pt`, `checkpoints/ddpg_epXXXXXX.pt` (every 250 rounds), `train_log.csv`, `config.yaml`, `viewer/`.

## 2. Evaluate (Table I numbers)

```bash
# 1000 test rounds, fixed eval seed, plus open-loop RRT comparison
python scripts/evaluate.py --checkpoint runs/ddpg_paper/ddpg_final.pt --episodes 1000 --out runs/ddpg_paper/eval

# Skip RRT / fewer RRT rounds (RRT is the slow part)
python scripts/evaluate.py --checkpoint runs/ddpg_paper/ddpg_final.pt --episodes 1000 --no-rrt --out runs/ddpg_paper/eval
python scripts/evaluate.py --checkpoint runs/ddpg_paper/ddpg_final.pt --rrt-episodes 100 --out runs/ddpg_paper/eval

# Closed-loop RRT: re-plan every 10 steps
python scripts/evaluate.py --checkpoint runs/ddpg_paper/ddpg_final.pt --rrt-replan 10 --out runs/ddpg_paper/eval_replan

# Test-time CBF shield / APF on an unchanged policy: evaluate.py has no --set, plugins come from the config.
# Either copy the run's config.yaml, edit `method.plugins` (e.g. [shield], method.shield.d_safe: 0.01) and pass --config,
# or use the benchmark helper: SRC run, DST dir, JSON overrides, eval rounds, eval seed [, rrt_replan]
python scripts/benchmark.py --_eval-only runs/ddpg_paper runs/ddpg_paper_shield1cm '{"method.plugins":"[shield]","method.shield.d_safe":0.01,"method.label":"DDPG+shield 1cm"}' 1000 12345
python scripts/benchmark.py --_eval-only runs/ddpg_paper runs/apf_shield_eval '{"method.plugins":"[apf,shield]","method.apf.prior_only":true,"method.label":"APF+shield"}' 1000 12345
# (writes DST/config.yaml with the overrides, symlinks the checkpoint, then runs evaluate.py; the 98.1 % / 0 collision number is the first one on runs/bench/static/paper_ddpg_10k)

# Save per-round paths (for paired stats / stratification)
python scripts/evaluate.py --checkpoint runs/ddpg_paper/ddpg_final.pt --save-paths --out runs/ddpg_paper/eval
```

Outputs: `<out>/table1.json`, `<out>/table1.md`, `<out>/episodes.csv`, example path figures (`--n-figures`, default 3).

## 3. Plots, demo, single-run report

```bash
python scripts/plot_results.py --run runs/ddpg_paper --bin 100                     # learning curve (paper Fig. 4)
python scripts/demo.py --checkpoint runs/ddpg_paper/ddpg_final.pt --out runs/demo --every 5   # matplotlib 3-D frames
python scripts/make_report.py --run runs/ddpg_paper --out runs/ddpg_paper/RESULTS.md
python scripts/mujoco_demo.py --config configs/ur5e_industrial.yaml --checkpoint runs/ur5e_industrial/ddpg_final.pt --video runs/ur5e_industrial/demo.gif --episodes 3 --fps 10
```

## 4. 3-D viewer (Three.js, browser)

```bash
python scripts/serve_viewer.py --run runs/ddpg_paper              # opens http://localhost:8000/viewer/?run=runs/ddpg_paper
python scripts/serve_viewer.py --run runs/ur5e_industrial --port 8080 --no-browser

# Build viewer data for a run that was trained without it (one rollout set per checkpoint)
python scripts/export_viewer.py --run runs/ddpg_paper --random-baseline      # --random-baseline adds the untrained policy as round 0 (ghost overlay)
python scripts/export_viewer.py --run runs/ddpg_paper --episodes 8
```

Start the server while training: the page polls `runs/<run>/viewer/manifest.json` and jumps to the newest checkpoint (LIVE badge).
Keys: space play/pause, left/right step, `[` `]` previous/next checkpoint, 1-9 scene.

## 5. Benchmark (ablation matrix, 3 worlds x variants x 3 seeds)

```bash
python scripts/benchmark.py --dry-run                               # list jobs and what is already done
python scripts/benchmark.py --workers 8                             # run / resume the whole matrix (resumable; ~hours)
python scripts/benchmark.py --workers 4 --scenes static,wind        # subset of worlds
python scripts/benchmark.py --variants paper_ddpg,shield_eval_1cm,td3_her_apf_shield --seeds 0
python scripts/benchmark.py --episodes 2000 --eval-episodes 200 --out runs/bench_smoke   # quick smoke matrix
python scripts/benchmark.py --train-only                            # train, skip evaluation
```

Variant names (`configs/benchmark.yaml`): paper_ddpg, apf_only, shield_eval, shield_eval_1cm, td3, td3_her, td3_her_obl,
td3_her_apfshape, td3_her_apf, td3_her_apf_shield, full_set, full_curriculum, curriculum_mlp, cem_rl, rrt_replan,
her_k1, apf_shield_eval, paper_ddpg_10k, full_10k, cem_s4, cem_lr, cem_s4_lr, shield_eval_1cm_10k.
Layout: `runs/bench/<scene>/<variant>/seed<k>/{ddpg_final.pt,train_log.csv,eval/table1.json}`, log in `runs/bench_full.log`.

## 6. Results -> report, tables, figures, paper

```bash
# Difficulty stratification (persistent-homology tau) for every finished eval
for d in runs/bench/*/*/seed*/eval; do python scripts/stratify.py --eval-dir $d --config $d/../config.yaml --seed 12345 --bins 4; done

python scripts/report_benchmark.py --out paper      # runs/bench/{REPORT.md,results.json}, paper/figures/*.pdf, paper/tables/*.md
python scripts/paper_tables.py                      # paper/tables/{results_all,gains,efficiency,numbers}.tex
python scripts/paired_stats.py                      # paired per-scene bootstrap CI + McNemar -> paired.{md,json,tex}, p-value macros (~8 min)
python scripts/paired_stats.py --vs-baseline        # every variant against paper_ddpg instead of its ladder parent

# Whole chain in one go (waits for a running benchmark first)
scripts/finalize_paper.sh
```

## 7. Paper

```bash
cd paper && tectonic -X compile main.tex && cd ..                    # -> paper/main.pdf
python -c "from pypdf import PdfReader; print(len(PdfReader('paper/main.pdf').pages))"
cp paper/main.pdf runs/bench/paper_draft.pdf
```

`paper/main.tex` follows the IEEE conference template (`paper/sample.tex`, class `paper/IEEEtran.cls` = `paper/sample.cls`), reads
`paper/tables/numbers.tex` (macros like `\Sstaticbase`) and carries its reference list inline as `\bibitem`s, as the template does.
Re-run step 6 before compiling after any new result. After editing `paper/refs.bib`, regenerate the inline list:

```bash
cd paper && python - <<'EOF2'
import re; t=open('main.tex').read()
b0=t.index('\\begin{thebibliography}'); b1=t.index('\\end{thebibliography}')+len('\\end{thebibliography}')
open('main_bib.tex','w').write(t[:b0]+'\\bibliographystyle{IEEEtran}\n\\bibliography{refs}\n'+t[b1:])
EOF2
tectonic -X compile --keep-intermediates main_bib.tex && python - <<'EOF2'
t=open('main.tex').read(); b=open('main_bib.bbl').read()
b=b[b.index('\\begin{thebibliography}'):b.index('\\end{thebibliography}')+len('\\end{thebibliography}')]
b0=t.index('\\begin{thebibliography}'); b1=t.index('\\end{thebibliography}')+len('\\end{thebibliography}')
open('main.tex','w').write(t[:b0]+b+t[b1:])
EOF2
rm -f main_bib.*; cd ..
```

## 8. Where things live

| What | Path |
|---|---|
| Paper-faithful config, `[paper]`/`[assumed]` tags | `configs/default.yaml` |
| Dynamic scenes | `configs/fruit_tree_wind.yaml`, `configs/ur5e_*.yaml`, `configs/scenes/*.yaml` |
| Ablation matrix | `configs/benchmark.yaml` |
| Plugins (shield, apf, her, obl, curriculum) | `arm_rl/plugins/` |
| Benchmark outputs | `runs/bench/REPORT.md`, `SUMMARY.md`, `results.json`, `paired.md` |
| Paper | `paper/main.tex`, `paper/refs.bib`, `paper/main.pdf` |
| Research ideas | `docs/research_directions.md` |
