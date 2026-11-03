# Research directions: beyond Li et al. (ACIRS 2024)

Working notes for a follow-up paper on DRL path planning for a six-axis arm in
multi-obstacle environments. Everything below is written against this repository, so
each idea names the module it plugs into and the experiment that would demonstrate it.

## 0. Where the baseline stands and what it leaves open

What this reproduction established (see `README.md`, `RESULTS.md`):

* The paper's 27-D state (closest-point direction vectors per link + joints + EE + target)
  with the paper's DDPG hyper-parameters reaches 98 % success on static fruit-tree scenes
  once two unspecified details are fixed: **gamma = 0.95** and **terminate on collision**.
  With gamma = 0.99 the critic diverges (Q > 0 although every reward is <= 0). That is a
  finding in itself: the paper's reward (-|d| per step, d in dm) produces returns of order
  -300 and plain DDPG overestimates on it.
* First dynamic-scene numbers (1000 test rounds each, `README.md`): fruit tree with swinging
  branches, DDPG 97.2 % vs RRT 40.9 %; UR5e industrial cell with a conveyor target, sweeping
  gantry and AGV, DDPG 75.9 % with **14.8 % collisions** vs RRT 1.8 % (open-loop plan on a
  snapshot). The collision rate is the number C1 below is designed to drive to zero; the RRT
  number says the baseline must become a *re-planning* RRT (plan every k steps on the current
  snapshot) to be fair in dynamic worlds.
* Already implemented here but absent from the paper: moving obstacles and targets
  (`arm_rl/scene.py` motion laws), velocity-augmented 45-D state, an industrial UR5e in
  MuJoCo with real collision geometry and position-servo dynamics (`arm_rl/mujoco_sim.py`),
  configurable scene presets (`configs/scenes/`).

Open problems the paper does not address, which define the contribution space:

| Gap | Consequence |
|---|---|
| No safety guarantee: collision is only penalised | 1-2 % test collisions remain; unacceptable on a real cell |
| Static world, 100 steps, no dynamics | not representative of conveyors, co-workers, other robots |
| Fixed "5 nearest" vectors | state cannot represent >1 close obstacle per link, no obstacle velocity |
| Plain DDPG | overestimation, seed sensitivity, slow early learning |
| Evaluation = 1000 rounds, one scene family | no difficulty stratification, no confidence intervals |

## 1. Recommended contributions (ranked by rigour x expected gain)

### C1. Control-barrier-function (CBF) safety shield on the learned policy

**Idea.** Keep DDPG/TD3 as the task policy, but filter every action through a quadratic
program that enforces forward invariance of the safe set
`h_i(q) = clearance_i(q) - d_safe >= 0` for every link-obstacle pair. The paper's state
already contains the needed quantities: the closest-point vector `V_i` gives both the
clearance and its gradient direction, and the link Jacobian at the closest point `P'_i`
turns it into a constraint on joint increments:

```
min_{dq}  || dq - dq_RL ||^2
s.t.      n_i^T J_i(P'_i) dq  >=  -alpha * h_i(q) - n_i^T v_obs,i * dt     for all monitored pairs
          |dq| <= 5 deg
```

`n_i = V_i / |V_i|`, `v_obs,i` the obstacle velocity (we already export it), `alpha` the
class-K gain. Six variables, ~5-30 linear constraints: solvable in < 0.2 ms with `qpsolvers`/OSQP.

**Why it is relevant now.** Safe RL and CBF shields are the standard way 2023-2026 papers
bring RL manipulators onto hardware (Cheng et al. 2019; Ames et al. 2019; ISO/TS 15066
speed-and-separation monitoring is literally a distance-based barrier).

**Rigour.** Gives a *theorem* (forward invariance of the safe set under the filtered
action, standard CBF result) instead of a penalty. Cleanly separates task learning from safety.

**Where.** New `arm_rl/safety.py` (`CBFShield.filter(dq, state)`), called in
`ArmPathPlanningEnv.step` or in the agent's `act`. Needs the Jacobian at the closest point:
`SixAxisArm.position_jacobian` (analytic) / `mj_jac` (MuJoCo) at `P'`.

**Experiments.** Collision rate and success with/without the shield at train time and at
test time; speed of the arm vs clearance (shield should slow the arm only near obstacles);
dynamic scenes with increasing obstacle speed (0-0.5 m/s). Expected: collision rate 0 % in
all static scenes and near-0 in dynamic ones, success unchanged or higher, minor increase in
path length.

### C2. Potential-field prior: residual RL over an artificial-potential-field controller

**Idea.** The attractive/repulsive APF of Khatib (1986) is computable from the paper's state
alone: attractive term from `S_e - S_t`, repulsive term from each `V_i` (the closest-point
vectors *are* the repulsive gradients). Use it twice:

1. **Residual policy**: `dq = dq_APF(s) + pi_theta(s)`; the network only learns the correction
   (Johannink et al. 2019). Learning starts from a controller that already reaches ~50 % of
   targets, so exploration is focused on the local minima APF gets stuck in.
2. **Potential-based reward shaping**: `F = gamma * Phi(s') - Phi(s)` with
   `Phi = -k_a |EE - T| + k_r sum_i max(0, 1/|V_i| - 1/rho_0)`. Ng, Harada & Russell (1999)
   prove this leaves the optimal policy unchanged, so the paper's reward semantics are
   preserved while learning is accelerated.

**Rigour.** Policy-invariance theorem for the shaping; the residual formulation gives a
measurable "value of learning over the prior" (RL minus APF-only).

**Where.** `arm_rl/apf.py` (gradient from the state), `env.reward_for` gains the shaping term,
`DDPGAgent.act` adds the prior. APF-only becomes a third baseline in Table I.

**Experiments.** Rounds-to-90 %-success (expect >= 2x fewer), success in cluttered scenes
where APF alone fails (local minima), ablation prior-only / shaping-only / both.

### C3. Stabilised off-policy learner: TD3 + hindsight experience replay + opposition-based augmentation

**Idea.** Replace DDPG's single critic by TD3 (clipped double-Q, delayed actor updates, target
policy smoothing; Fujimoto et al. 2018). Add HER (Andrychowicz et al. 2017): the task is
goal-conditioned (`S_t` is part of the state), so every failed round can be relabelled with
the EE position actually reached, turning the 3 cm "picking range" into dense successful
experience. Add **opposition-based learning (OBL)** as transition augmentation: for a stored
`(s, a, r, s')`, also simulate the *opposite* action `a_opp = -a` from `s` (the simulator is
exact and cheap, 0.1-0.7 ms) and store `(s, a_opp, r_opp, s'_opp)`. OBL (Tizhoosh 2005;
Rahnamayan et al. 2008) argues that evaluating the opposite of a guess halves the expected
distance to the optimum; in RL it doubles buffer coverage around the current policy at zero
extra environment cost.

**Rigour / honesty.** TD3 and HER are well-established and directly address two defects we
measured (overestimation; sparse 3 cm success region). OBL's theoretical basis is for
population optimisation, not RL; present it as an *ablation* with the same budget, not as the
headline. If it does not help, say so: that is still a result.

**Where.** `arm_rl/ddpg.py` -> `arm_rl/td3.py` (second critic, delayed update);
`ReplayBuffer` gains `relabel_goal()` (HER needs the target index in the state: last 3 dims)
and `add_opposite()`; `scripts/train.py` calls `env.backend` to simulate the opposite action.

**Experiments.** Learning curves (mean +- 95 % CI over 5 seeds) DDPG / TD3 / TD3+HER /
TD3+HER+OBL, same 10 000 rounds; final success, collision rate, final error.

### C4. Swarm / population exploration: CEM-RL (evolutionary + gradient actors)

**Idea.** Maintain a population ("swarm") of actor networks. Each generation: sample actors
from a Gaussian over parameters (cross-entropy method), roll them out in parallel
environments (`gymnasium.vector` over our analytic env runs > 10 k steps/s per core), update
the sampling distribution from the elite half, and apply TD3 gradient steps to half of the
population using the *shared* replay buffer (Pourchot & Sigaud 2019; Khadka & Tumer 2018).
The gradient half exploits, the evolutionary half explores by parameter-space perturbation,
which is exactly what dynamic obstacle scenes need (action-space Gaussian noise of 2 deg is
too local to discover "wait for the gantry to pass").

**Why "swarm".** CEM-RL is a particle-population method; PSO can be substituted for CEM
(the update rule is the only change) if the paper wants the swarm-intelligence framing.
A second, more ambitious swarm angle is **two UR5e arms sharing the cell** (cooperative
picking, each arm an obstacle for the other): the state representation extends naturally
because the other arm's links are just more capsules.

**Rigour.** Population methods are seed-robust and give variance estimates for free;
compare on equal *environment-step* budgets, not wall-clock.

**Where.** `arm_rl/cem_rl.py` + `scripts/train_cemrl.py`; env already supports batched scenes
via `options={"scene": ...}`.

**Experiments.** Success vs environment steps; dispersion across seeds; dynamic scenes with
periodic obstacles where timing matters.

### C5. Topological data analysis for scene difficulty, curriculum and path classification

**Idea.** Two uses of persistent homology (Edelsbrunner & Harer 2010; `ripser`/`giotto-tda`):

1. **Scene complexity index.** Sample the free workspace (points in the reachable shell not
   inside any obstacle), compute Vietoris-Rips persistence; the number and lifetime of
   H1/H2 features (loops/voids the EE must go around or through) is a scalar difficulty
   `tau(scene)`. Use it (a) to build a **curriculum** (train from low to high `tau`) and
   (b) to **stratify results** (success vs `tau` bins) instead of a single Table I number.
2. **Homotopy class of trajectories.** Classify DDPG vs RRT paths by homotopy class
   (Bhattacharya et al. 2012; Pokorny et al. 2016). Reports *which* way around the obstacles
   the learned policy goes and whether it is consistent across scenes, a qualitative result
   no success rate shows.

**Rigour / honesty.** TDA is defensible only if it drives a decision (the curriculum) or an
evaluation axis (stratification). As a state feature it is expensive and unlikely to beat
the closest-point vectors; do not sell it that way.

**Where.** `arm_rl/topology.py` (offline, per scene, cached in the scene JSON),
`scripts/train.py --curriculum tau`, `scripts/evaluate.py --stratify`.

**Experiments.** Success vs `tau` for all methods; curriculum vs uniform sampling at equal
budget; distribution of homotopy classes for DDPG vs RRT.

### C6. Dynamic environments with a set-based state encoder and a real robot model

**Idea.** Replace the fixed "5 nearest vectors" by a permutation-invariant encoder over *all*
link-obstacle pairs: each pair is a token `(V_ij, |V_ij|, v_obs,j, link one-hot)`, processed
by a Deep-Sets/attention pooling (Zaheer et al. 2017; Vaswani et al. 2017) into a fixed
embedding fed to the actor/critic. This removes the two representational limits of the
paper (one obstacle per link; no velocities) and generalises across obstacle *counts*.
Evaluate on the dynamic presets already in this repo (`industrial_cell`: conveyor target,
sweeping gantry, AGV; `warehouse_shelf`: person walking through; `fruit_tree_wind`) and on
the UR5e MuJoCo model in **servo-dynamics** mode (`sim.control: dynamic`), reporting
cycle time and tracking lag, which the kinematic paper setting hides.

**Rigour.** Sim-to-sim transfer (analytic -> MuJoCo UR5e) and domain randomisation (Tobin
et al. 2017) are the accepted evidence for sim-to-real readiness; the UR5e RTDE interface
makes a hardware demo a short step.

**Where.** `arm_rl/networks.py` (`SetEncoder`), `env._observation` returns the token set;
everything else exists.

**Experiments.** Success/collision vs obstacle speed; vs number of obstacles (train on 5-10,
test on 20-25); kinematic vs dynamic control; analytic-trained policy evaluated zero-shot on
UR5e.

## 2. Suggested paper package

* **Title angle.** "Safe and sample-efficient DRL path planning for industrial arms in
  dynamic multi-obstacle cells" (C1 + C2 + C3 as the core; C6 as the setting; C4/C5 as
  secondary studies). Three contributions is the sweet spot; six is a thesis.
* **Baselines.** Paper's DDPG (reproduced), APF-only, RRT / RRT-Connect with periodic
  re-planning for dynamic scenes, TD3 (no extras).
* **Metrics.** Success, collision rate, final error, EE path length, planning/inference time,
  **minimum clearance**, **cycle time** (dynamic control), robustness vs obstacle speed,
  all as mean +- 95 % CI over 5 seeds x 1000 rounds; stratified by TDA difficulty.
* **Ablations.** Shield on/off; prior / shaping / both; HER / OBL; population size.
* **Theory boxes.** CBF forward invariance (C1); policy invariance of shaping (C2).

## 3. Implementation status in this repository

| Item | Status |
|---|---|
| Reproduction of the paper (state, reward, DDPG, Table I) | done |
| Dynamic obstacles / targets, motion laws, presets, velocities in state | done |
| UR5e in MuJoCo, real collision geometry, kinematic + servo control | done |
| 3-D viewer with per-checkpoint replay, live training | done |
| C1 CBF shield | to do (~1 day) |
| C2 APF prior + shaping | to do (~0.5 day) |
| C3 TD3 + HER + OBL | to do (~1 day) |
| C4 CEM-RL | to do (~1-2 days, needs vectorised envs) |
| C5 TDA curriculum / stratification | to do (~1-2 days) |
| C6 set encoder | to do (~1 day) |

## References

* Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., Tabuada, P. (2019). Control barrier functions: theory and applications. ECC.
* Cheng, R., Orosz, G., Murray, R. M., Burdick, J. W. (2019). End-to-end safe reinforcement learning through barrier functions for safety-critical continuous control tasks. AAAI.
* ISO/TS 15066:2016. Robots and robotic devices - Collaborative robots.
* Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots. IJRR.
* Ng, A. Y., Harada, D., Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. ICML.
* Johannink, T. et al. (2019). Residual reinforcement learning for robot control. ICRA.
* Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing function approximation error in actor-critic methods (TD3). ICML.
* Haarnoja, T. et al. (2018). Soft actor-critic. ICML.
* Andrychowicz, M. et al. (2017). Hindsight experience replay. NeurIPS.
* Tizhoosh, H. R. (2005). Opposition-based learning: a new scheme for machine intelligence. CIMCA.
* Rahnamayan, S., Tizhoosh, H. R., Salama, M. M. A. (2008). Opposition-based differential evolution. IEEE TEVC.
* Khadka, S., Tumer, K. (2018). Evolution-guided policy gradient in reinforcement learning. NeurIPS.
* Pourchot, A., Sigaud, O. (2019). CEM-RL: combining evolutionary and gradient-based methods for policy search. ICLR.
* Edelsbrunner, H., Harer, J. (2010). Computational Topology: An Introduction. AMS.
* Bhattacharya, S., Likhachev, M., Kumar, V. (2012). Topological constraints in search-based robot path planning. Autonomous Robots.
* Pokorny, F. T., Hawasly, M., Ramamoorthy, S. (2016). Topological trajectory classification with filtrations of simplicial complexes and persistent homology. IJRR.
* Zaheer, M. et al. (2017). Deep sets. NeurIPS.
* Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.
* Tobin, J. et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. IROS.
* Li, Z., Xiong, Z., Tian, K., Gao, T., Cai, K. (2024). Deep reinforcement learning for robotic arm path planning in multi-obstacle environments. ACIRS.
