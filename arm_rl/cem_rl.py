"""CEM-RL: cross-entropy method over actor parameters combined with a gradient learner (C4).

Pourchot & Sigaud, "CEM-RL: Combining evolutionary and gradient-based methods for policy
search", ICLR 2019.  Each generation

1. samples ``pop_size`` actor parameter vectors from N(mu, diag(sigma^2)),
2. applies gradient (DDPG / TD3) steps to ``rl_fraction`` of them, using the critic of the
   gradient learner and the *shared* replay buffer filled by the whole population,
3. rolls every actor out for ``episodes_per_actor`` episodes (score = mean return), storing
   all transitions in the shared buffer,
4. moves (mu, sigma) towards the elite half (standard CEM with an additive noise term
   ``sigma^2 += noise_decay^gen * sigma_init`` that keeps the search from collapsing).

The gradient learner (``arm_rl.agents.make_agent``) is only used for its critic, buffer and
optimiser; its ``actor`` serves as the parameter template and as scratch space for the RL
steps.  The best-ever actor is copied into it before every ``save()`` so the checkpoint is
loadable by ``scripts/evaluate.py`` with the same ``method.algo``.

Configuration lives under ``method.cem`` (all keys optional)::

    method:
      cem:
        pop_size: 10            # actors per generation
        elite_frac: 0.5         # fraction of the population used for the CEM update
        sigma_init: 1.0e-3      # initial *variance* of every parameter (as in the authors' code)
        noise_decay: 0.999      # eps_gen = noise_decay^gen * sigma_init added to the variance
        episodes_per_actor: 1   # rollouts per actor; score = mean return
        rl_fraction: 0.5        # fraction of the population that receives gradient steps
        max_grad_steps: 2000    # upper bound on gradient steps per RL actor and generation
        importance_mixing: false
        checkpoint_actor: best  # best | mu   -> which actor the checkpoints hold
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

CEM_DEFAULTS = {
    "pop_size": 10,
    "elite_frac": 0.5,
    "sigma_init": 1e-3,
    "noise_decay": 0.999,
    "episodes_per_actor": 1,
    "rl_fraction": 0.5,
    "max_grad_steps": 2000,
    "importance_mixing": False,
    "importance_alpha": 0.1,
    "checkpoint_actor": "best",
}


def cem_config(cfg: dict) -> dict:
    out = dict(CEM_DEFAULTS)
    out.update(dict((cfg.get("method") or {}).get("cem") or {}))
    return out


# ---------------------------------------------------------------------- parameter vectors
def flat_params(module: torch.nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).cpu().numpy().astype(np.float64)


@torch.no_grad()
def set_flat_params(module: torch.nn.Module, vec: np.ndarray) -> None:
    i = 0
    for p in module.parameters():
        n = p.numel()
        p.copy_(torch.as_tensor(vec[i:i + n], dtype=p.dtype, device=p.device).view_as(p))
        i += n
    if i != vec.size:
        raise ValueError(f"parameter vector has {vec.size} entries, module needs {i}")


# ---------------------------------------------------------------------- CEM
class CEM:
    """Diagonal-Gaussian cross-entropy method over a flat parameter vector."""

    def __init__(self, mu: np.ndarray, pop_size: int, elite_frac: float = 0.5, sigma_init: float = 1e-3,
                 noise_decay: float = 0.999, rng: np.random.Generator | None = None):
        self.mu = np.asarray(mu, dtype=np.float64).copy()
        self.n = self.mu.size
        self.pop_size = int(pop_size)
        self.n_elite = max(1, int(round(float(elite_frac) * self.pop_size)))
        self.sigma_init = float(sigma_init)
        self.noise_decay = float(noise_decay)
        self.var = np.full(self.n, self.sigma_init, dtype=np.float64)  # diag(sigma^2)
        self.generation = 0
        self.rng = rng or np.random.default_rng(0)

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(self.var)

    def sample(self, n: int | None = None) -> np.ndarray:
        n = self.pop_size if n is None else int(n)
        eps = self.rng.standard_normal((n, self.n))
        return self.mu[None, :] + eps * self.sigma[None, :]

    def log_density(self, x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Log N(x; mu, diag(var)) without the constant, batched over rows of x."""
        return -0.5 * np.sum((x - mu[None, :]) ** 2 / var[None, :] + np.log(var)[None, :], axis=1)

    def update(self, params: np.ndarray, scores: np.ndarray) -> dict:
        """Standard CEM step: mu <- mean of elites, sigma^2 <- their spread around the old mu
        plus the decaying noise term.  Returns bookkeeping for the log."""
        params = np.asarray(params, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-scores)
        elite = params[order[: self.n_elite]]
        elite_mean = elite.mean(axis=0)
        eps = self.noise_decay ** self.generation * self.sigma_init
        old_mu = self.mu
        self.var = np.mean((elite - old_mu[None, :]) ** 2, axis=0) + eps
        self.mu = elite_mean
        self.generation += 1
        return {
            "elite_idx": order[: self.n_elite].tolist(),
            "elite_mean_score": float(scores[order[: self.n_elite]].mean()),
            "noise_eps": float(eps),
            "sigma_mean": float(np.sqrt(self.var).mean()),
        }

    def importance_mix(self, old_params: np.ndarray, old_mu: np.ndarray, old_var: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Importance mixing (Sun et al. 2009): keep old samples that are likely under the new
        distribution and complete the population with fresh ones.

        Returns (params, reused) where ``reused[i]`` is the index into ``old_params`` the i-th
        row came from, or -1 for a fresh sample."""
        keep_p = np.minimum(1.0, (1.0 - alpha) * np.exp(self.log_density(old_params, self.mu, self.var) - self.log_density(old_params, old_mu, old_var)))
        keep = self.rng.random(old_params.shape[0]) < keep_p
        rows = [old_params[i] for i in np.flatnonzero(keep)]
        reused = [int(i) for i in np.flatnonzero(keep)]
        while len(rows) < self.pop_size:
            x = self.sample(1)[0]
            acc_p = max(alpha, 1.0 - np.exp(self.log_density(x[None], old_mu, old_var) - self.log_density(x[None], self.mu, self.var))[0])
            if self.rng.random() < acc_p:
                rows.append(x)
                reused.append(-1)
        return np.stack(rows[: self.pop_size]), np.asarray(reused[: self.pop_size], dtype=np.int64)


# ---------------------------------------------------------------------- trainer
class CEMRLTrainer:
    """Drives the generations, the shared buffer and the artifact conventions of scripts/train.py."""

    CSV_FIELDS = [
        "episode", "return", "shaped_return", "steps", "reached", "collisions", "final_distance", "min_clearance", "noise_deg",
        "critic_loss", "actor_loss", "q_mean", "updates", "env_steps", "wall_time_s", "plugin_time_s",
        "generation", "actor", "rl_actor", "score", "sigma_mean",
    ]

    def __init__(self, cfg: dict, env, agent, plugins: list, out: Path, seed: int = 0, quiet: bool = True, logger=None):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.plugins = plugins
        self.out = Path(out)
        self.quiet = quiet
        self.cem_cfg = cem_config(cfg)
        self.rng = np.random.default_rng(seed + 1_000_003)
        self.worker = copy.deepcopy(agent.actor)  # scratch actor for rollouts
        for p in self.worker.parameters():
            p.requires_grad_(False)
        self.worker.eval()
        self.device = next(self.worker.parameters()).device
        c = self.cem_cfg
        self.pop_size = int(c["pop_size"])
        self.n_rl = int(round(float(c["rl_fraction"]) * self.pop_size))
        self.episodes_per_actor = max(1, int(c["episodes_per_actor"]))
        self.cem = CEM(flat_params(agent.actor), self.pop_size, float(c["elite_frac"]), float(c["sigma_init"]), float(c["noise_decay"]), rng=self.rng)

        tr = cfg["train"]
        self.n_episodes = int(tr["episodes"])
        self.checkpoint_every = int(tr.get("checkpoint_every", 500))
        self.log_every = int(tr.get("log_every", 50))
        max_delta_deg = float(cfg["env"]["max_delta_deg"])
        self.max_delta_deg = max_delta_deg
        ddpg_cfg = cfg.get("ddpg") or {}
        self.noise_start = float(ddpg_cfg.get("action_noise_deg_start", 2.0)) / max_delta_deg
        self.noise_end = float(ddpg_cfg.get("action_noise_deg_end", 0.0)) / max_delta_deg

        self.logger = logger
        self.episode = 0
        self.total_steps = 0
        self.t_start = time.time()
        self.returns: list[float] = []
        self.success: list[float] = []
        self.best_params = flat_params(agent.actor)
        self.best_score = -np.inf
        self.best_generation = -1
        self.generation_log: list[dict] = []
        self.grad_steps_total = 0
        self.prev_gen_steps = 0
        self.on_checkpoint = None  # callable(episode) for viewer export

    # ------------------------------------------------------------------ helpers
    def noise_std(self, episode: int) -> float:
        frac = (episode - 1) / max(1, self.n_episodes - 1)
        return self.noise_start + (self.noise_end - self.noise_start) * frac

    @torch.no_grad()
    def _act(self, obs: np.ndarray, noise_std: float) -> np.ndarray:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.worker(s).squeeze(0).cpu().numpy()
        if noise_std > 0.0:
            a = a + self.rng.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _load_into_agent(self, vec: np.ndarray) -> None:
        set_flat_params(self.agent.actor, vec)
        tgt = getattr(self.agent, "actor_target", None)
        if tgt is not None:
            set_flat_params(tgt, vec)
        opt = getattr(self.agent, "actor_opt", None)
        if opt is not None:
            opt.state.clear()  # fresh Adam moments for each sampled actor (as in the authors' code)

    def checkpoint_params(self) -> np.ndarray:
        return self.cem.mu if str(self.cem_cfg["checkpoint_actor"]) == "mu" else self.best_params

    def save(self, path: Path) -> None:
        self._load_into_agent(self.checkpoint_params())
        self.agent.save(path)

    # ------------------------------------------------------------------ rollout
    def rollout(self, params: np.ndarray, gen: int, actor_idx: int, is_rl: bool, losses: list[dict]) -> float:
        """Run ``episodes_per_actor`` episodes with ``params``; log one CSV row each; return mean return."""
        set_flat_params(self.worker, params)
        env, agent, plugins = self.env, self.agent, self.plugins
        rets = []
        for _ in range(self.episodes_per_actor):
            if self.episode >= self.n_episodes:
                break
            self.episode += 1
            episode = self.episode
            noise_std = self.noise_std(episode)
            scene = None
            for p in plugins:
                scene = p.choose_scene(episode) or scene
            obs, info = env.reset(options={"scene": scene} if scene is not None else None)
            for p in plugins:
                p.on_episode_start(episode, obs, info)
            ep_return, ep_shaped, ep_steps, reached = 0.0, 0.0, 0, False
            min_clear = float("inf")
            transitions = []
            while True:
                action = self._act(obs, noise_std)
                for p in plugins:
                    action = p.transform_action(obs, action, deterministic=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                r_train = reward
                for p in plugins:
                    r_train = p.shape_reward(obs, action, r_train, next_obs, terminated, info)
                agent.store(obs, action, r_train, next_obs, terminated)
                for p in plugins:
                    for tr_ in p.extra_transitions(obs, action, r_train, next_obs, terminated, info):
                        agent.store(*tr_)
                transitions.append((obs, action, reward, next_obs, terminated, info))
                obs = next_obs
                ep_return += reward
                ep_shaped += r_train
                ep_steps += 1
                self.total_steps += 1
                min_clear = min(min_clear, env.min_obstacle_clearance())
                if terminated or truncated:
                    reached = bool(info["reached"])
                    break
            for p in plugins:
                for tr_ in p.on_episode_end(transitions, episode):
                    agent.store(*tr_)
            rets.append(ep_return)
            self.returns.append(ep_return)
            self.success.append(float(reached))
            row = {
                "episode": episode,
                "return": ep_return,
                "shaped_return": ep_shaped,
                "steps": ep_steps,
                "reached": int(reached),
                "collisions": info["n_collisions"],
                "final_distance": info["distance"],
                "min_clearance": min_clear,
                "noise_deg": noise_std * self.max_delta_deg,
                "updates": agent.n_updates,
                "env_steps": self.total_steps,
                "wall_time_s": time.time() - self.t_start,
                "plugin_time_s": sum(p.cost["time_s"] for p in plugins),
                "generation": gen,
                "actor": actor_idx,
                "rl_actor": int(is_rl),
                "sigma_mean": float(self.cem.sigma.mean()),
            }
            if losses:
                for k in ("critic_loss", "actor_loss", "q_mean"):
                    vals = [l[k] for l in losses if k in l]
                    if vals:
                        row[k] = float(np.mean(vals))
            if self.logger is not None:
                self.logger.log(row)
        return float(np.mean(rets)) if rets else -np.inf

    # ------------------------------------------------------------------ generation
    def rl_steps(self, vec: np.ndarray, n_steps: int, losses: list[dict]) -> np.ndarray:
        """Copy ``vec`` into the gradient learner, run ``n_steps`` updates, return the new vector."""
        if n_steps <= 0 or not self.agent.ready():
            return vec
        self._load_into_agent(vec)
        for _ in range(n_steps):
            losses.append(self.agent.update())
        self.grad_steps_total += n_steps
        return flat_params(self.agent.actor)

    def run_generation(self, gen: int) -> dict:
        c = self.cem_cfg
        if bool(c["importance_mixing"]) and gen > 0 and hasattr(self, "_prev"):
            old_params, old_mu, old_var, old_scores = self._prev
            params, reused = self.cem.importance_mix(old_params, old_mu, old_var, float(c["importance_alpha"]))
        else:
            params = self.cem.sample()
            reused = np.full(self.pop_size, -1, dtype=np.int64)
        mu_before, var_before = self.cem.mu.copy(), self.cem.var.copy()

        n_grad = min(int(c["max_grad_steps"]), self.prev_gen_steps // max(1, self.pop_size)) if gen > 0 else 0
        steps_before = self.total_steps
        scores = np.full(self.pop_size, -np.inf)
        losses_per_actor: list[list[dict]] = [[] for _ in range(self.pop_size)]
        evaluated = 0
        for i in range(self.pop_size):
            if self.episode >= self.n_episodes:
                break
            is_rl = i < self.n_rl
            if is_rl:
                params[i] = self.rl_steps(params[i], n_grad, losses_per_actor[i])
            if reused[i] >= 0 and not is_rl:
                scores[i] = old_scores[reused[i]]  # importance mixing: fitness carried over
                continue
            scores[i] = self.rollout(params[i], gen, i, is_rl, losses_per_actor[i])
            evaluated += 1
        n_valid = int(np.isfinite(scores).sum())
        complete = n_valid == self.pop_size

        # best-ever actor
        if n_valid:
            j = int(np.argmax(scores))
            if scores[j] > self.best_score:
                self.best_score, self.best_params, self.best_generation = float(scores[j]), params[j].copy(), gen
        info = {"generation": gen, "episodes_end": self.episode, "n_grad_steps_per_actor": n_grad, "evaluated": evaluated,
                "mean_score": float(np.mean(scores[np.isfinite(scores)])) if n_valid else None,
                "max_score": float(np.max(scores[np.isfinite(scores)])) if n_valid else None, "complete": complete}
        if complete:
            info.update(self.cem.update(params, scores))
            self._prev = (params, mu_before, var_before, scores)
        self.prev_gen_steps = self.total_steps - steps_before
        self.generation_log.append(info)
        return info

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict:
        gen = 0
        last_ckpt = 0
        while self.episode < self.n_episodes:
            info = self.run_generation(gen)
            # checkpoints crossed during this generation (train.checkpoint_every in episodes)
            k = (last_ckpt // self.checkpoint_every + 1) * self.checkpoint_every
            while k <= self.episode:
                self.save(self.out / "checkpoints" / f"ddpg_ep{k:06d}.pt")
                self.save(self.out / "ddpg_final.pt")
                if self.on_checkpoint is not None:
                    self.on_checkpoint(k)
                last_ckpt = k
                k += self.checkpoint_every
            if not self.quiet and (gen % max(1, self.log_every // max(1, self.pop_size * self.episodes_per_actor)) == 0):
                w = self.log_every
                print(f"gen {gen:4d} ep {self.episode:6d} steps {self.total_steps:7d} ret {np.mean(self.returns[-w:]):7.1f} "
                      f"succ {100 * np.mean(self.success[-w:]):3.0f}% best {self.best_score:7.1f} "
                      f"sigma {self.cem.sigma.mean():.2e} grad/actor {info['n_grad_steps_per_actor']}", flush=True)
            gen += 1
        self.save(self.out / "ddpg_final.pt")
        if self.on_checkpoint is not None and self.n_episodes % self.checkpoint_every != 0:
            self.on_checkpoint(self.n_episodes)
        return self.summary()

    def summary(self) -> dict:
        return {
            "episodes": self.n_episodes,
            "total_env_steps": self.total_steps,
            "updates": self.agent.n_updates,
            "wall_time_s": time.time() - self.t_start,
            "final_100_mean_return": float(np.mean(self.returns[-100:])) if self.returns else None,
            "final_100_success_rate": float(np.mean(self.success[-100:])) if self.success else None,
            "method": self.cfg.get("method", {}),
            "plugins": {p.name: p.stats() for p in self.plugins},
            "agent": getattr(self.agent, "stats", lambda: {})(),
            "population": {
                "pop_size": self.pop_size,
                "n_rl_actors": self.n_rl,
                "elite_frac": float(self.cem_cfg["elite_frac"]),
                "episodes_per_actor": self.episodes_per_actor,
                "generations": len(self.generation_log),
                "grad_steps_total": self.grad_steps_total,
                "best_score": float(self.best_score) if np.isfinite(self.best_score) else None,
                "best_generation": self.best_generation,
                "checkpoint_actor": str(self.cem_cfg["checkpoint_actor"]),
                "sigma_mean_final": float(self.cem.sigma.mean()),
                "n_params": int(self.cem.n),
                "generation_log": self.generation_log,
            },
        }
