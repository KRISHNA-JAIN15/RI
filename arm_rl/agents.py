"""Agent factory: ``method.algo`` selects ddpg (paper) or td3 (C3)."""

from __future__ import annotations


def make_agent(state_dim: int, action_dim: int, cfg: dict, device="cpu", seed: int = 0, env=None):
    method = dict(cfg.get("method") or {})
    algo = str(method.get("algo", "ddpg")).lower()
    algo_cfg = dict(cfg.get("ddpg") or {})
    algo_cfg.update(dict(cfg.get(algo) or {}) if algo != "ddpg" else {})
    algo_cfg["encoder"] = method.get("encoder", "mlp")
    algo_cfg["obs_layout"] = getattr(env, "obs_layout", None)
    if algo == "ddpg":
        from arm_rl.ddpg import DDPGAgent

        return DDPGAgent(state_dim, action_dim, algo_cfg, device=device, seed=seed)
    if algo == "td3":
        from arm_rl.td3 import TD3Agent

        return TD3Agent(state_dim, action_dim, algo_cfg, device=device, seed=seed)
    raise ValueError(f"unknown algo {algo}")
