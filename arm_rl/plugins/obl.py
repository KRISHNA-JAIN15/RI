"""Opposition-based transition augmentation (Tizhoosh 2005; Rahnamayan et al. 2008) --
research direction C3.

After every real step ``(s, a) -> s'`` the simulator is rewound to ``s`` and the opposite
action is executed as a counterfactual step, producing a second transition
``(s, a_opp, r_opp, s'_opp)`` for the replay buffer.  The environment is then restored
exactly to its post-step state, so the training rollout is unaffected.

Options (``method.obl``):
    prob: 1.0          fraction of steps that are augmented
    mode: opposite     opposite -> a_opp = -a ; quasi -> a_opp ~ U(-a, 0) per joint
    seed: 0            rng seed (prob / quasi sampling)

No-op in phase "eval".  Extra simulator calls are charged to ``cost["extra_sim_steps"]``.
"""

from __future__ import annotations

import numpy as np

from arm_rl.plugins.base import Plugin


class Plugin(Plugin):
    name = "obl"

    def __init__(self, cfg: dict, env, agent, phase: str = "train"):
        super().__init__(cfg, env, agent, phase)
        self.prob = float(self.cfg.get("prob", 1.0))
        self.mode = str(self.cfg.get("mode", "opposite")).lower()
        assert self.mode in ("opposite", "quasi"), f"obl.mode must be opposite|quasi, got {self.mode}"
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        self._pre = None  # env snapshot taken before the real step

    def opposite_action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32)
        if self.mode == "opposite":
            return (-a).astype(np.float32)
        # quasi-opposition: uniform between the opposite point and the centre (0)
        return (-a * self.rng.uniform(0.0, 1.0, size=a.shape)).astype(np.float32)

    # ------------------------------------------------------------------ hooks
    def transform_action(self, obs: np.ndarray, action: np.ndarray, deterministic: bool) -> np.ndarray:
        if self.phase == "train":
            with self.timed():
                self._pre = self.env.snapshot()
        return action

    def extra_transitions(self, obs, action, reward, next_obs, terminated: bool, info: dict) -> list[tuple]:
        if self.phase != "train" or self._pre is None:
            return []
        if self.prob < 1.0 and self.rng.uniform() >= self.prob:
            self._pre = None
            return []
        with self.timed():
            env = self.env
            a_opp = self.opposite_action(action)
            post = env.snapshot()
            env.restore(self._pre)
            obs_o, r_o, term_o, _trunc_o, _info_o = env.step(a_opp)
            env.restore(post)
            # restore() trims ee_history to the post-step length; the last entry now holds
            # the counterfactual EE, put the real one back.
            if env.ee_history:
                env.ee_history[-1] = post["ee"].copy()
            self._pre = None
        self.cost["extra_sim_steps"] += 1
        self.cost["extra_transitions"] += 1
        return [(np.asarray(obs, dtype=np.float32), a_opp, float(r_o), np.asarray(obs_o, dtype=np.float32), bool(term_o))]
