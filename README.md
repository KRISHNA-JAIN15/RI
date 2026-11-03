# DDPG path planning for a six-axis arm in a multi-obstacle environment

End-to-end re-implementation of

> Z. Li, Z. Xiong, K. Tian, T. Gao, K. Cai, **"Deep Reinforcement Learning for Robotic Arm
> Path Planning in Multi-Obstacle Environments"**, 2024 9th Asia-Pacific Conference on
> Intelligent Robot Systems (ACIRS). DOI 10.1109/ACIRS62330.2024.10684932

The paper trains DDPG to drive a six-axis picking arm to a fruit while avoiding the
trunk and branches of a tree. Its contribution is the 27-D state (Sec. III-A): for each
arm link, the direction vector of the shortest distance to the nearest obstacle, plus
joint angles, end-effector position and target position. This repository reproduces the
state, action, reward, network architecture, training schedule and the Table I
evaluation protocol. The ROS/RViz simulator is replaced by an exact analytical model
(numpy forward kinematics + cylinder/cylinder closest-point geometry) and matplotlib.

## What is in here beyond the paper

* **Dynamic, configurable worlds** (`arm_rl/scene.py`, `configs/scenes/*.yaml`): obstacles are
  capsules with motion laws (oscillate, linear with bounce, conveyor wrap, orbit, swing,
  random walk); targets can move (conveyor parts, swaying fruit); scenes are YAML presets
  with per-object noise and optional random clutter. The state can carry obstacle and
  target velocities (45-D).
* **Industrial simulator** (`arm_rl/mujoco_sim.py`): the Universal Robots UR5e from MuJoCo
  Menagerie in MuJoCo 3, with the model's real per-link collision capsules, position-servo
  actuators (`sim.control: dynamic`) and `mj_geomDistance` closest points as the paper's
  state vectors. Native MuJoCo viewer / video via `scripts/mujoco_demo.py`.
* **3-D web viewer** (`viewer/`): replay every checkpoint on the same test scenes, live
  while training (see below).
* **Research directions** for a follow-up paper: `docs/research_directions.md`.

## Layout

```
arm_rl/
  geometry.py       eq. 5-7   closest points between two cylinder axes (vectorised, exact)
  kinematics.py               six-axis DH forward kinematics, reach 61.2 cm, link cylinders
  scene.py          Sec. IV   fruit tree (trunk + branches, r = 4 cm), target sphere (r = 3 cm), noise
  env.py            Sec. III  Gymnasium env: 27-D state (eq. 8-12), +-5 deg actions, reward eq. 13
  networks.py       Sec. III-D / Fig. 2  Actor 128-128-tanh, Critic 128-128-linear
  replay_buffer.py            uniform replay, 100 000 transitions
  ddpg.py           eq. 1-4   DDPG with soft target updates
  rrt.py            Table I   joint-space RRT baseline sharing the same collision checker
  metrics.py        Table I   S, Lavg, sigmaL, Tavg, sigmaT, Eavg, sigmaE
  visualize.py      Fig. 3-5  scene rendering, training curve, EE trajectories
  utils.py                    config / seeding / CSV logging
configs/default.yaml          every hyper-parameter, tagged [paper] or [assumed]
scripts/train.py              Sec. IV-B training (10 000 rounds x 100 steps)
scripts/evaluate.py           Sec. IV-C: 1000 rounds DDPG vs RRT -> Table I
scripts/plot_results.py       Fig. 4 from a training log
scripts/demo.py               render one episode (PNG + GIF)
scripts/export_viewer.py      roll out every checkpoint on fixed scenes -> viewer data
scripts/serve_viewer.py       local server for the 3-D viewer
arm_rl/export.py              rollout -> JSON for the viewer (also called by train.py at each checkpoint)
arm_rl/mujoco_sim.py          MuJoCo backend: UR5e model, capsule obstacles on mocap bodies, contacts
scripts/mujoco_demo.py        native MuJoCo viewer (mjpython) or offscreen video of a policy
configs/scenes/               industrial_cell.yaml, warehouse_shelf.yaml (dynamic presets)
configs/ur5e_industrial.yaml  UR5e + industrial cell;  configs/ur5e_warehouse.yaml;  configs/fruit_tree_wind.yaml
third_party/mujoco_menagerie/ UR5e model (Apache-2.0, google-deepmind/mujoco_menagerie)
docs/research_directions.md   six proposed contributions for a follow-up paper
viewer/                       Three.js 3-D viewer (index.html, app.js, style.css)
tests/                        geometry, kinematics, environment unit tests
```

## Install

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

## Run

```bash
# 1. train with the paper's schedule (10 000 rounds, ~20 min on a laptop CPU)
python scripts/train.py --config configs/default.yaml --out runs/ddpg_paper

# 2. evaluate 1000 rounds of DDPG and RRT on identical scenes -> Table I + Fig. 5 plots
python scripts/evaluate.py --checkpoint runs/ddpg_paper/ddpg_final.pt --episodes 1000 --out runs/ddpg_paper/eval

# 3. figures
python scripts/plot_results.py --run runs/ddpg_paper          # Fig. 4
python scripts/demo.py --checkpoint runs/ddpg_paper/ddpg_final.pt --out runs/ddpg_paper/demo
```

Outputs: `train_log.csv`, `fig4_training_curve.png`, `ddpg_final.pt`, `checkpoints/`,
`viewer/` (rollouts per checkpoint), `eval/table1.md`, `eval/table1.json`,
`eval/*_episodes.json`, `eval/fig5_*.png`.

## Dynamic scenes and the UR5e simulator

```bash
# paper's tree, but branches swing in the wind and the fruit sways (analytic arm, 45-D state)
python scripts/train.py --config configs/fruit_tree_wind.yaml --out runs/fruit_tree_wind
# UR5e in MuJoCo: conveyor target, sweeping gantry, passing AGV, random clutter
python scripts/train.py --config configs/ur5e_industrial.yaml --out runs/ur5e_industrial
# UR5e warehouse shelf with a person walking through the aisle
python scripts/train.py --config configs/ur5e_warehouse.yaml --out runs/ur5e_warehouse
# physics instead of kinematics (position servos, tracking lag, contacts during motion)
python scripts/train.py --config configs/ur5e_industrial.yaml --set env.sim.control=dynamic --out runs/ur5e_dyn
# native MuJoCo viewer (macOS needs mjpython) or a video
mjpython scripts/mujoco_demo.py --config configs/ur5e_industrial.yaml --checkpoint runs/ur5e_industrial/ddpg_final.pt
python  scripts/mujoco_demo.py --config configs/ur5e_industrial.yaml --checkpoint runs/ur5e_industrial/ddpg_final.pt --video runs/ur5e_industrial/demo.gif
```

Scene presets (`configs/scenes/*.yaml`) list obstacles as capsules `p1 -> p2` with a
radius, optional `noise` (random shift per round) and an optional `motion` law:

```yaml
obstacles:
  - {p1: [0.45, 0.30, 0.80], p2: [0.95, 0.30, 0.80], radius: 0.045,
     motion: {type: oscillate, axis: [0, -1, 0], amplitude: 0.35, period: 7.0, phase: random}}
  - {box: {center: [0.95, 0.0, 0.25], size: [0.30, 0.40, 0.50]}, radius: 0.03,
     motion: {type: linear, axis: [0, 1, 0], speed: 0.25, bounds: [-0.6, 0.6]}}
target:
  region: {center: [0.55, 0.0, 0.16], extent: [0.10, 0.0, 0.04]}
  motion: {type: conveyor, axis: [0, 1, 0], speed: 0.06, bounds: [-0.30, 0.30]}
randomize: {count: [0, 2], region: {center: [0.45, 0, 0.45], extent: [0.4, 0.8, 0.5]}, radius: [0.025, 0.05], length: [0.1, 0.35]}
```

Motion types: `static`, `oscillate`, `linear` (bounce between `bounds`), `conveyor` (wrap),
`orbit`, `swing` (rotation about `pivot`), `random_walk`. `env.dt` seconds of world motion
elapse per control step; `env.include_velocities: true` appends the nearest-obstacle and
target velocities to the state. Any preset works with both the analytic arm and the UR5e.

## 3-D viewer

```bash
python scripts/serve_viewer.py --run runs/ddpg_paper        # opens http://localhost:8000/viewer/?run=runs/ddpg_paper
```

* Every `train.checkpoint_every` rounds (default 250) `train.py` replays the current policy
  on `train.viewer_episodes` fixed test scenes (same scenes for every checkpoint) and writes
  `runs/<run>/viewer/ckpt_XXXXXX.json` + `manifest.json`. Start the server while training
  runs: the page polls the manifest and jumps to the newest checkpoint ("LIVE" badge).
* For an existing run build the data afterwards:
  `python scripts/export_viewer.py --run runs/ddpg_paper --random-baseline`
  (`--random-baseline` adds the untrained policy as round 0, used by the ghost overlay).
* The page shows the arm (blue cylinders, or the UR5e's real collision capsules for MuJoCo
  runs), obstacles (green capsules, moving if the scene is dynamic), picking target (red
  sphere, gold when reached, moving on conveyors), end-effector trail, and the paper's state vectors S_l drawn as
  arrows from each link to its nearest obstacle (coloured by clearance). The panel has the
  training curve with clickable checkpoint markers, a checkpoint slider, per-scene outcome
  tabs, playback controls, a live HUD (distance, reward, return, clearance, joint angles) and
  a ghost overlay of the untrained policy on the same scene.
* Keys: space play/pause, ←/→ step, `[` `]` previous/next checkpoint, 1-9 scene.
* Three.js is loaded from jsdelivr (pinned 0.160.0); the rest is static, served from the repo.

## Paper -> code mapping

| Paper | Value | Where |
|---|---|---|
| State S = [Sl, Sa, Se, St] (eq. 8-12) | 15 + 6 + 3 + 3 = 27 dims | `env.py::_observation` |
| Direction vector V = Q' - P' (eq. 5-7) | exact closest points of two segments | `geometry.py` |
| Action | incremental joint angles in [-5 deg, 5 deg] (tanh x 5 deg) | `env.py::step` |
| Reward (eq. 13) | -abs(d) if d > 3 cm; 0 if d < 3 cm; -50 on collision; d = 10 x distance | `env.py::reward_for` |
| Round length | 100 steps or target touched | `env.max_steps` |
| Actor / Critic | 2 x FC128 ReLU, tanh output / linear output | `networks.py` |
| Optimiser | Adam, actor 1e-5, critic 1e-4, L2 regularisation | `ddpg.py` |
| Replay / batch | 100 000 / 64 | `ddpg.py` |
| Action noise | 0..2 deg (Gaussian sigma, linearly decayed 2 -> 0 deg) | `train.py` |
| Scene noise | 0..10 (cm / deg) on obstacle poses, 0..5 cm on target centre | `scene.py` |
| Arm | six axes, reach 61.2 cm, links as cylinders | `kinematics.py` |
| Obstacles / target | cylinders r = 4 cm with varying heights / sphere r = 3 cm | `scene.py` |
| Training | 10 000 rounds x 100 steps | `configs/default.yaml` |
| Test | 1000 rounds, DDPG vs RRT, Table I metrics | `scripts/evaluate.py`, `metrics.py` |

## Assumptions (not stated in the paper)

* **State dimension.** Sec. III-A and Fig. 2 give 27 dims (five link vectors). Sec. III-D
  says "input layer with a state dimension of 30". We follow eq. 12 / Fig. 2 (27). Setting
  `env.monitored_links: [0,1,2,3,4,5]` gives the 30-D variant.
* **Arm geometry.** No specific robot is named. We use a UR-style DH table scaled so that
  the maximum shoulder-to-end-effector reach is 0.612 m; link radius 4 cm; joint limits
  +-170 deg; a fixed folded home pose at the start of every round.
* **Tree.** Trunk 0.9 m tall at 0.65 m in front of the base, four branches of 15-28 cm
  rooted at 0.20-0.48 m height that fan towards the arm. A fruit hangs at each branch tip;
  one reachable fruit is the picking target of the round.
* **Collision handling.** The paper only mentions ending a round on 100 steps or target
  touched. Letting the round continue after a collision does not learn (<= 9 % success after
  10 000 rounds in our first run), so the default is `collision_mode: terminate` (-50 and the
  round ends), which is also how most DDPG manipulator papers treat collisions.
  `continue` (-50, round goes on) is still available.
* **gamma = 0.95, tau = 0.005, gradient-norm clip 1.0, weight decay 1e-4 (critic), 1000
  random warm-up steps.** With gamma = 0.99 the critic diverges on this reward (Q grows
  positive although every reward is <= 0); the sweep in `scripts/sweep.sh` /
  `runs/sweep/` documents this. All other optimiser settings are the paper's.
* **"Noise 0..10" on obstacle poses** is read as up to 10 cm translation of the tree root and
  up to 10 deg rotation of each branch.
* **RRT baseline.** Joint-space RRT (5 deg steps, 2.5 deg edge checks, 3000 iterations max)
  with 10 % goal bias implemented as a Jacobian pseudo-inverse step towards the Cartesian
  target (there is no IK in the paper's description). This baseline is considerably stronger
  than the one reported in the paper (see results).
* **Success** in evaluation = end effector within 3 cm of the target and no collision during
  the round. L, T, E statistics are computed over successful rounds; T for DDPG is the actor
  inference time summed over the round, T for RRT is the planning time.

## Results (seed 0, 10 000 rounds, 1000 test rounds, Apple M-series CPU)

| Algorithm | S(%) | Lavg(m) | σL(m) | Tavg(ms) | σT(ms) | Eavg(m) | σE(m) |
|---|---|---|---|---|---|---|---|
| DDPG (ours) | 98.2 | 0.45 | 0.06 | 0.42 | 0.09 | 0.025 | 0.004 |
| RRT (ours) | 86.4 | 0.84 | 0.43 | 55.33 | 119.81 | 0.019 | 0.007 |
| DDPG (paper) | 87.3 | 1.11 | 0.65 | 9.55 | 8.17 | 0.031 | 0.007 |
| RRT (paper) | 36.8 | 1.05 | 0.35 | 3425.20 | 369.70 | 0.032 | 0.005 |

* The qualitative claims of the paper reproduce: DDPG with the proposed 27-D state reaches
  the target far more reliably than RRT and plans two orders of magnitude faster
  (0.42 ms vs 55 ms here; 9.55 ms vs 3425 ms in the paper), while landing within the 3 cm
  picking range (E ~ 2.5-3 cm).
* Absolute numbers differ because the arm, tree and RRT implementation are not published:
  our scenes are easier for both planners (higher S, shorter L), our hardware is faster,
  and our RRT uses a Jacobian goal step that the paper's "traditional RRT" apparently
  lacks.
* Training curve (`runs/ddpg_paper/fig4_training_curve.png`) has the same shape as Fig. 4:
  steep rise over the first ~1500 rounds then a plateau; the plateau level is higher (-26
  instead of -120) because rounds end early on reaching the target and gamma/collision
  handling differ (see Assumptions).
* Full details, per-round JSON and Fig. 5 style trajectories: `RESULTS.md`, `runs/ddpg_paper/eval/`.

### Dynamic scenes (same protocol, 1000 test rounds, RRT plans on the t = 0 snapshot and its path is executed in the moving world)

| Run | Simulator | World | DDPG S(%) | DDPG collisions (%) | RRT S(%) | DDPG T (ms) | RRT T (ms) |
|---|---|---|---|---|---|---|---|
| `runs/fruit_tree_wind` | analytic | branches swing +-10 deg, fruit sways, 45-D state | 97.2 | 1.5 | 40.9 | 0.48 | 44 |
| `runs/ur5e_industrial` | MuJoCo UR5e | conveyor target 6 cm/s, sweeping gantry, AGV, clutter | 75.9 | 14.8 | 1.8 | 0.47 | 3.9 |

The closed-loop policy keeps working when the world moves; an open-loop RRT plan does not
(the conveyor carries the part away from the planned goal). Two things the numbers make
obvious for the follow-up paper (`docs/research_directions.md`): the 14.8 % collision rate
in the industrial cell calls for a safety shield (C1), and RRT needs a re-planning variant
to be a fair baseline in dynamic scenes. Per-run reports: `runs/*/RESULTS.md`; MuJoCo video
of the trained UR5e policy: `runs/ur5e_industrial/demo.gif`.

## Benchmark

`configs/benchmark.yaml` defines the ablation ladder (paper DDPG → TD3 → HER → APF shaping →
APF prior → CBF shield → set encoder → TDA curriculum, plus APF-only, test-time shield and
CEM-RL variants) over three scenes (static fruit tree, fruit tree in wind, UR5e MuJoCo cell),
3 seeds, 5000 training rounds and 1000 test rounds. `scripts/benchmark.py` runs it with
parallel, resumable workers; `scripts/stratify.py` adds per-run difficulty stratification;
`scripts/report_benchmark.py` aggregates everything.

* `runs/bench/REPORT.md`: main tables, ablation ladder with per-step deltas and costs,
  stratified results, plugin statistics.
* `runs/bench/SUMMARY.md`: the matrix actually run (which variants on which scenes), the
  headline numbers, what helped/hurt/cost, failed or dropped variants, compute, and a
  reviewer's assessment.
* `paper/tables/*.tex` and `paper/figures/*.{png,pdf}`: booktabs tables, learning curves,
  ladder and cost figures used by `paper/main.tex`.
* `runs/bench/results.json`: every (scene, variant, seed) record.

```bash
python scripts/benchmark.py --dry-run          # list jobs
python scripts/benchmark.py --workers 8        # run / resume the matrix
for d in runs/bench/*/*/seed*/eval; do python scripts/stratify.py --eval-dir $d --config $d/../config.yaml --seed 12345 --bins 4; done
python scripts/report_benchmark.py             # REPORT.md, paper/tables, paper/figures
```
