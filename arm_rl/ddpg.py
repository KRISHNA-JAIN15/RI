"""Deep Deterministic Policy Gradient (paper Sec. II-B, eq. 1-4).

* Critic: minimise MSE( Q(s,a), r + gamma * Q'(s', mu'(s')) )          (eq. 2)
* Actor : maximise Q(s, mu(s))  <=>  grad J = 1/N sum grad_mu grad_a Q   (eq. 1)
* Target networks: soft update  theta' <- delta*theta + (1-delta)*theta' (eq. 3-4)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from arm_rl.networks import Actor, Critic
from arm_rl.replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: dict, device: str | torch.device = "cpu", seed: int = 0):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.warmup_steps = int(cfg.get("warmup_steps", 1000))
        self.grad_clip = cfg.get("grad_clip")
        hidden = list(cfg.get("hidden_sizes", [128, 128]))

        torch.manual_seed(seed)
        self.actor = Actor(state_dim, action_dim, hidden).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for p in list(self.actor_target.parameters()) + list(self.critic_target.parameters()):
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(cfg.get("actor_lr", 1e-5)))
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(cfg.get("critic_lr", 1e-4)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),  # "regularization is added"
        )
        self.buffer = ReplayBuffer(state_dim, action_dim, int(cfg.get("buffer_size", 100_000)))
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """Deterministic action in [-1, 1] plus optional Gaussian exploration noise.

        ``noise_std`` is expressed in the normalised action unit (1.0 == max_delta).
        """
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

        # ---- Critic (eq. 2)
        with torch.no_grad():
            target_q = r + self.gamma * (1.0 - done) * self.critic_target(s2, self.actor_target(s2))
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.grad_clip))
        self.critic_opt.step()

        # ---- Actor (eq. 1)
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.grad_clip))
        self.actor_opt.step()

        # ---- Soft target update (eq. 3-4)
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        self.n_updates += 1
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()), "q_mean": float(q.mean().item())}

    @torch.no_grad()
    def _soft_update(self, net: torch.nn.Module, target: torch.nn.Module) -> None:
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.mul_(1.0 - self.tau).add_(self.tau * p)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "n_updates": self.n_updates,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str | Path, map_location: str | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.n_updates = int(ckpt.get("n_updates", 0))
