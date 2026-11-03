"""Actor / Critic networks (paper Sec. III-D, Fig. 2).

Actor : state -> FC128-ReLU -> FC128-ReLU -> FC(action)-tanh
Critic: [state, action] -> FC128-ReLU -> FC128-ReLU -> FC1 (linear)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: list[int] | None = None):
        super().__init__()
        self.net = _mlp(state_dim, hidden or [128, 128], action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: list[int] | None = None):
        super().__init__()
        self.net = _mlp(state_dim + action_dim, hidden or [128, 128], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


def make_actor(state_dim: int, action_dim: int, cfg: dict) -> nn.Module:
    """Build the actor. ``cfg['encoder']``: 'mlp' (paper) or 'set' (C6, see SetEncoder)."""
    enc = str(cfg.get("encoder", "mlp"))
    hidden = list(cfg.get("hidden_sizes", [128, 128]))
    if enc == "mlp":
        return Actor(state_dim, action_dim, hidden)
    raise ValueError(f"unknown encoder {enc}")


def make_critic(state_dim: int, action_dim: int, cfg: dict) -> nn.Module:
    enc = str(cfg.get("encoder", "mlp"))
    hidden = list(cfg.get("hidden_sizes", [128, 128]))
    if enc == "mlp":
        return Critic(state_dim, action_dim, hidden)
    raise ValueError(f"unknown encoder {enc}")
