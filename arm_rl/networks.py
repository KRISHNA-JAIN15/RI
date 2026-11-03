"""Actor / Critic networks (paper Sec. III-D, Fig. 2).

Actor : state -> FC128-ReLU -> FC128-ReLU -> FC(action)-tanh
Critic: [state, action] -> FC128-ReLU -> FC128-ReLU -> FC1 (linear)

Research direction C6 (docs/research_directions.md): ``SetEncoder`` replaces the paper's
fixed "5 nearest vectors" by a permutation-invariant encoding of ALL link-obstacle pair
tokens produced by ``ArmPathPlanningEnv(all_pairs=True)``. ``SetActor`` / ``SetCritic``
share the 128-128 head of the paper; only the input changes.
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


# ============================================================================ C6: set encoder
class SetEncoder(nn.Module):
    """Permutation-invariant encoder over link-obstacle pair tokens (Deep Sets + attention pooling).

    Input: the flat state produced by ``ArmPathPlanningEnv(all_pairs=True)``; ``obs_layout``
    (env.obs_layout) tells where the ``pairs`` block (n_links x max_obstacles x pair_dim) and
    the ``mask`` block (n_links x max_obstacles) live. Each token is concatenated with a one-hot
    of its link index, passed through a per-token MLP (pair_dim + n_links -> 64 -> 64), and pooled
    with multi-head attention against a learned query (padded tokens get -1e9 before the softmax).
    An all-masked set yields a zero embedding. Output: ``(B, set_embed)``.

    Implementation note: ``[token, onehot] @ W`` is computed as ``token @ W_tok + E[link]`` with a
    learned per-link embedding ``E`` (exactly equivalent, avoids materialising the concatenation).
    """

    def __init__(self, obs_layout: dict, set_embed: int = 64, token_hidden: int = 64, n_heads: int = 4):
        super().__init__()
        if not obs_layout or "pairs" not in obs_layout or "mask" not in obs_layout:
            raise ValueError(
                "encoder 'set' needs the all-pairs observation: set env.all_pairs=true "
                "(obs_layout has no 'pairs'/'mask' block)"
            )
        self.n_links = int(obs_layout["n_links"])
        self.max_obstacles = int(obs_layout["max_obstacles"])
        self.pair_dim = int(obs_layout["pair_dim"])
        self.pairs_slice = tuple(obs_layout["pairs"])
        self.mask_slice = tuple(obs_layout["mask"])
        assert self.pairs_slice[1] - self.pairs_slice[0] == self.n_links * self.max_obstacles * self.pair_dim
        assert self.mask_slice[1] - self.mask_slice[0] == self.n_links * self.max_obstacles
        # everything that is not a pair token (sl, sa, se, st, vl, vt) is passed through untouched
        skip = {"pairs", "mask"}
        base = [v for k, v in obs_layout.items() if isinstance(v, (tuple, list)) and k not in skip]
        self.base_slices = sorted((int(a), int(b)) for a, b in base)
        self.base_dim = sum(b - a for a, b in self.base_slices)

        assert token_hidden % n_heads == 0, "token_hidden must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = token_hidden // n_heads
        self.token_in = nn.Linear(self.pair_dim, token_hidden)
        self.link_embed = nn.Embedding(self.n_links, token_hidden)  # == the one-hot columns of the first layer
        self.token_out = nn.Linear(token_hidden, token_hidden)
        self.key = nn.Linear(token_hidden, token_hidden, bias=False)
        self.value = nn.Linear(token_hidden, token_hidden, bias=False)
        self.query = nn.Parameter(torch.randn(n_heads, self.head_dim) / self.head_dim ** 0.5)
        self.out = nn.Linear(token_hidden, set_embed)
        self.set_embed = set_embed
        # token n = (link n // max_obstacles, obstacle n % max_obstacles)
        self.register_buffer("link_ids", torch.arange(self.n_links).repeat_interleave(self.max_obstacles), persistent=False)

    @property
    def out_dim(self) -> int:
        return self.base_dim + self.set_embed

    def split(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(base (B, base_dim), tokens (B, N, pair_dim), mask (B, N)) with N = n_links * max_obstacles."""
        base = torch.cat([state[..., a:b] for a, b in self.base_slices], dim=-1)
        tokens = state[..., self.pairs_slice[0]:self.pairs_slice[1]].reshape(state.shape[0], -1, self.pair_dim)
        mask = state[..., self.mask_slice[0]:self.mask_slice[1]]
        return base, tokens, mask

    def pool(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked attention pooling of ``tokens`` (B, N, pair_dim) with ``mask`` (B, N) in {0, 1}."""
        B, N, _ = tokens.shape
        h = torch.relu(self.token_in(tokens) + self.link_embed(self.link_ids[:N]).unsqueeze(0))
        h = torch.relu(self.token_out(h))  # (B, N, H*dh)
        k = self.key(h).view(B, N, self.n_heads, self.head_dim)
        v = self.value(h).view(B, N, self.n_heads, self.head_dim)
        scores = (k * self.query).sum(-1) / self.head_dim ** 0.5  # (B, N, H)
        valid = mask > 0.5
        scores = scores.masked_fill(~valid.unsqueeze(-1), -1e9)
        w = torch.softmax(scores, dim=1)  # over tokens; padded tokens get exactly 0 when any token is valid
        pooled = (w.unsqueeze(-1) * v).sum(1).reshape(B, -1)  # (B, H*dh)
        pooled = pooled * valid.any(dim=-1, keepdim=True).to(pooled.dtype)  # empty set -> zeros
        return self.out(pooled)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Flat state (B, state_dim) -> head input (B, base_dim + set_embed)."""
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
        base, tokens, mask = self.split(state)
        out = torch.cat([base, self.pool(tokens, mask)], dim=-1)
        return out.squeeze(0) if squeeze else out


class SetActor(nn.Module):
    def __init__(self, obs_layout: dict, action_dim: int, hidden: list[int] | None = None, **enc_kw):
        super().__init__()
        self.encoder = SetEncoder(obs_layout, **enc_kw)
        self.net = _mlp(self.encoder.out_dim, hidden or [128, 128], action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(self.encoder(state)))


class SetCritic(nn.Module):
    def __init__(self, obs_layout: dict, action_dim: int, hidden: list[int] | None = None, **enc_kw):
        super().__init__()
        self.encoder = SetEncoder(obs_layout, **enc_kw)
        self.net = _mlp(self.encoder.out_dim + action_dim, hidden or [128, 128], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.encoder(state), action], dim=-1))


def _set_kwargs(cfg: dict) -> dict:
    return {
        "set_embed": int(cfg.get("set_embed", 64)),
        "token_hidden": int(cfg.get("set_token_hidden", 64)),
        "n_heads": int(cfg.get("set_heads", 4)),
    }


def _check_layout(state_dim: int, cfg: dict) -> dict:
    layout = cfg.get("obs_layout")
    if not layout or "pairs" not in layout:
        raise ValueError(
            "encoder 'set' requires cfg['obs_layout'] with a 'pairs' block: build the env with "
            "env.all_pairs=true (e.g. --set env.all_pairs=true --set method.encoder=set)"
        )
    end = max(b for v in layout.values() if isinstance(v, (tuple, list)) for b in v[1:])
    if end != state_dim:
        raise ValueError(f"obs_layout covers {end} dims but state_dim is {state_dim}")
    return layout


def make_actor(state_dim: int, action_dim: int, cfg: dict) -> nn.Module:
    """Build the actor. ``cfg['encoder']``: 'mlp' (paper) or 'set' (C6, see SetEncoder)."""
    enc = str(cfg.get("encoder", "mlp"))
    hidden = list(cfg.get("hidden_sizes", [128, 128]))
    if enc == "mlp":
        return Actor(state_dim, action_dim, hidden)
    if enc == "set":
        return SetActor(_check_layout(state_dim, cfg), action_dim, hidden, **_set_kwargs(cfg))
    raise ValueError(f"unknown encoder {enc}")


def make_critic(state_dim: int, action_dim: int, cfg: dict) -> nn.Module:
    enc = str(cfg.get("encoder", "mlp"))
    hidden = list(cfg.get("hidden_sizes", [128, 128]))
    if enc == "mlp":
        return Critic(state_dim, action_dim, hidden)
    if enc == "set":
        return SetCritic(_check_layout(state_dim, cfg), action_dim, hidden, **_set_kwargs(cfg))
    raise ValueError(f"unknown encoder {enc}")
