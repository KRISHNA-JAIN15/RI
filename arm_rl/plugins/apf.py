"""C2: artificial-potential-field prior (residual control) and potential-based reward shaping.

Options under ``method.apf``:

    prior: true          residual control  a_out = clip(a_rl + a_apf, -1, 1)
    prior_only: false    a_out = a_apf (pure APF controller, baseline; a_rl ignored)
    prior_gain: 1.0      max |dq_apf| = prior_gain * env.max_delta
    k_att, k_rep, rho0, d_switch      APF gains (see arm_rl/apf.py)
    shaping: true        F = gamma * Phi(s') - Phi(s) added to the training reward
    k_phi, k_phi_rep, shaping_scale   potential gains (Phi(s') = 0 on terminal transitions)

``gamma`` is the agent's discount (``self.agent.gamma``), falling back to ``cfg.gamma``.
The shaping is potential-based (Ng, Harada & Russell 1999), so the optimal policy of the
paper's MDP is unchanged.  The residual prior changes the policy class, not the reward.
"""

from __future__ import annotations

import numpy as np

from arm_rl import apf
from arm_rl.plugins.base import Plugin as BasePlugin


class Plugin(BasePlugin):
    name = "apf"

    def __init__(self, cfg: dict, env, agent, phase: str = "train"):
        super().__init__(cfg, env, agent, phase)
        c = self.cfg
        self.prior = bool(c.get("prior", True))
        self.prior_only = bool(c.get("prior_only", False))
        self.prior_gain = float(c.get("prior_gain", 1.0))
        self.k_att = float(c.get("k_att", 1.0))
        self.k_rep = float(c.get("k_rep", 0.02))
        self.rho0 = float(c.get("rho0", 0.15))
        self.d_switch = float(c.get("d_switch", 0.1))
        self.shaping = bool(c.get("shaping", True))
        self.k_phi = float(c.get("k_phi", 1.0))
        self.k_phi_rep = float(c.get("k_phi_rep", 0.01))
        self.shaping_scale = float(c.get("shaping_scale", 1.0))
        self.gamma = float(getattr(agent, "gamma", c.get("gamma", 0.95)))
        # analytic ``sl`` vectors join the capsule axes -> subtract both radii to get the
        # surface clearance; MuJoCo's join the surfaces already.
        if getattr(env.backend, "name", "analytic") == "analytic":
            obs_r = float(getattr(env.scene_gen, "obstacle_radius", 0.04))
            self.margin = float(env.arm.link_radius) + obs_r
        else:
            self.margin = 0.0
        self.cost.update({"calls": 0, "shaping_calls": 0})

    # ------------------------------------------------------------------ prior
    def apf_action(self) -> np.ndarray:
        """APF joint increment at the live configuration, normalised by env.max_delta."""
        dq, _ = apf.apf_joint_step(self.env, self.k_att, self.k_rep, self.rho0, self.d_switch,
                                   dq_max=self.env.max_delta * self.prior_gain)
        return dq / self.env.max_delta

    def transform_action(self, obs: np.ndarray, action: np.ndarray, deterministic: bool) -> np.ndarray:
        if not (self.prior or self.prior_only):
            return action
        with self.timed():
            self.cost["calls"] += 1
            a_apf = self.apf_action()
            if self.prior_only:
                out = a_apf
            else:
                out = np.asarray(action, dtype=np.float64) + a_apf
            return np.clip(out, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ shaping
    def phi(self, obs: np.ndarray) -> float:
        return apf.potential(obs, self.env.obs_layout, self.k_phi, self.k_phi_rep, self.rho0,
                             self.env.distance_scale, self.margin)

    def shape_reward(self, obs, action, reward: float, next_obs, terminated: bool, info: dict) -> float:
        if not self.shaping:
            return reward
        with self.timed():
            self.cost["shaping_calls"] += 1
            phi_s = self.phi(obs)
            phi_s2 = 0.0 if terminated else self.phi(next_obs)
            return float(reward + self.shaping_scale * (self.gamma * phi_s2 - phi_s))
