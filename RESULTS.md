# Results

Run directory: `runs/ddpg_paper`

## Training (Sec. IV-B)

* rounds: 10000, env steps: 228626, gradient updates: 227627
* wall time: 5.0 min (CPU)
* mean return over the last 100 rounds: -25.5
* success rate over the last 100 training rounds (with exploration noise): 99.0 %

![Fig. 4](runs/ddpg_paper/fig4_training_curve.png)

## Simulation test (Sec. IV-C, Table I)

TABLE I. ALGORITHM PERFORMANCE

| Algorithm | S(%) | Lavg(m) | σL(m) | Tavg(ms) | σT(ms) | Eavg(m) | σE(m) |
|---|---|---|---|---|---|---|---|
| DDPG (ours) | 98.2 | 0.45 | 0.06 | 0.42 | 0.09 | 0.025 | 0.004 |
| RRT (ours) | 86.4 | 0.84 | 0.43 | 55.33 | 119.81 | 0.019 | 0.007 |
| DDPG (paper) | 87.3 | 1.11 | 0.65 | 9.55 | 8.17 | 0.031 | 0.007 |
| RRT (paper) | 36.8 | 1.05 | 0.35 | 3425.20 | 369.70 | 0.032 | 0.005 |


Additional statistics (ours):

| Algorithm | collision rate (%) | mean steps / nodes | mean return |
|---|---|---|---|
| DDPG (ours) | 1.3 | 14.0 | -26.4 |
| RRT (ours) | 0.0 | 28.8 | 0.0 |

## Fig. 5 style trajectories

![fig5_ddpg_path_0.png](runs/ddpg_paper/eval/fig5_ddpg_path_0.png)
![fig5_ddpg_path_1.png](runs/ddpg_paper/eval/fig5_ddpg_path_1.png)
![fig5_ddpg_path_2.png](runs/ddpg_paper/eval/fig5_ddpg_path_2.png)
![fig5_rrt_path_0.png](runs/ddpg_paper/eval/fig5_rrt_path_0.png)
![fig5_rrt_path_1.png](runs/ddpg_paper/eval/fig5_rrt_path_1.png)
![scene_ddpg_0.png](runs/ddpg_paper/eval/scene_ddpg_0.png)
![scene_ddpg_1.png](runs/ddpg_paper/eval/scene_ddpg_1.png)
![scene_ddpg_2.png](runs/ddpg_paper/eval/scene_ddpg_2.png)
