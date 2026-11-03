"""Twin Delayed DDPG (TD3, Fujimoto et al. 2018) -- research direction C3.

Differences from :class:`arm_rl.ddpg.DDPGAgent`:

* two critics, the target uses the minimum of both (clipped double-Q)
* target policy smoothing: clipped Gaussian noise on the target action
* delayed actor / target updates (every ``policy_delay`` critic updates)

The public interface is identical to DDPGAgent (act / random_action / store / ready /
update / save / load / n_updates / actor / buffer / rng) so train.py, evaluate.py and
export.py can use either agent.  ``load`` also accepts a DDPG checkpoint: only the actor
(and its target) are restored and a warning is logged.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from arm_rl.networks import make_actor, make_critic
from arm_rl.replay_buffer import ReplayBuffer

log = logging.getLogger(__name__)


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, cfg: dict, device: str | torch.device = "cpu", seed: int = 0):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.warmup_steps = int(cfg.get("warmup_steps", 1000))
        self.grad_clip = cfg.get("grad_clip")
        # TD3-specific (normalised action units, 1.0 == max_delta)
        self.policy_noise = float(cfg.get("policy_noise", 0.2))
        self.noise_clip = float(cfg.get("noise_clip", 0.5))
        self.policy_delay = max(1, int(cfg.get("policy_delay", 2)))

        torch.manual_seed(seed)
        self.actor = make_actor(state_dim, action_dim, cfg).to(self.device)
        self.critic = make_critic(state_dim, action_dim, cfg).to(self.device)
        self.critic2 = make_critic(state_dim, action_dim, cfg).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic2_target = copy.deepcopy(self.critic2)
        for net in (self.actor_target, self.critic_target, self.critic2_target):
            for p in net.parameters():
                p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.get("actor_lr", 1e-5)))
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.critic2.parameters()),
            lr=float(cfg.get("critic_lr", 1e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
        self.buffer = ReplayBuffer(state_dim, action_dim, int(cfg.get("buffer_size", 100_000)))
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0
        self.n_actor_updates = 0
        self._last_actor_loss = float("nan")

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            a = a + self.rng.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def random_action(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

    # ------------------------------------------------------------------ learning
    def store(self, s, a, r, s2, done: bool) -> None:
        self.buffer.add(s, a, r, s2, done)

    def ready(self) -> bool:
        return len(self.buffer) >= max(self.batch_size, self.warmup_steps)

    def update(self) -> dict[str, float]:
        s, a, r, s2, done = self.buffer.sample(self.batch_size, self.rng, self.device)

        # ---- Critics: clipped double-Q with target policy smoothing
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-1.0, 1.0)
            target_q = torch.min(self.critic_target(s2, a2), self.critic2_target(s2, a2))
            target_q = r + self.gamma * (1.0 - done) * target_q
        q1 = self.critic(s, a)
        q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()) + list(self.critic2.parameters()), float(self.grad_clip))
        self.critic_opt.step()
        self.n_updates += 1

        # ---- Delayed actor + target updates
        if self.n_updates % self.policy_delay == 0:
            actor_loss = -self.critic(s, self.actor(s)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.grad_clip))
            self.actor_opt.step()
            self._last_actor_loss = float(actor_loss.item())
            self.n_actor_updates += 1
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.critic2, self.critic2_target)
            self._soft_update(self.actor, self.actor_target)

        return {"critic_loss": float(critic_loss.item()), "actor_loss": self._last_actor_loss, "q_mean": float(q1.mean().item())}

    @torch.no_grad()
    def _soft_update(self, net: torch.nn.Module, target: torch.nn.Module) -> None:
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.mul_(1.0 - self.tau).add_(self.tau * p)

    def stats(self) -> dict:
        return {"algo": "td3", "updates": self.n_updates, "actor_updates": self.n_actor_updates,
                "policy_noise": self.policy_noise, "noise_clip": self.noise_clip, "policy_delay": self.policy_delay}

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algo": "td3",
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic2": self.critic2.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "n_updates": self.n_updates,
                "n_actor_updates": self.n_actor_updates,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        try:
            self.critic.load_state_dict(ckpt["critic"])
            self.critic2.load_state_dict(ckpt["critic2"])
            self.critic_target.load_state_dict(ckpt["critic_target"])
            self.critic2_target.load_state_dict(ckpt["critic2_target"])
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.n_updates = int(ckpt.get("n_updates", 0))
            self.n_actor_updates = int(ckpt.get("n_actor_updates", 0))
        except (KeyError, RuntimeError, ValueError) as e:
            log.warning("TD3Agent.load(%s): checkpoint is not a TD3 checkpoint (%s: %s); loaded the actor only", path, type(e).__name__, e)
            # keep the critics / optimisers freshly initialised
