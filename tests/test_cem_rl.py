import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arm_rl.agents import make_agent  # noqa: E402
from arm_rl.cem_rl import CEM, CEMRLTrainer, flat_params, set_flat_params  # noqa: E402
from arm_rl.env import ArmPathPlanningEnv  # noqa: E402
from arm_rl.utils import CSVLogger, load_config  # noqa: E402

CFG_PATH = ROOT / "configs" / "default.yaml"


def test_cem_update_moves_mu_to_elite_mean():
    rng = np.random.default_rng(0)
    cem = CEM(np.zeros(5), pop_size=4, elite_frac=0.5, sigma_init=1e-2, noise_decay=0.5, rng=rng)
    params = cem.sample()
    scores = np.array([1.0, 3.0, 2.0, 0.0])  # elites: rows 1 and 2
    mu_old = cem.mu.copy()
    info = cem.update(params, scores)
    elite_mean = params[[1, 2]].mean(axis=0)
    assert np.allclose(cem.mu, elite_mean)
    assert np.linalg.norm(cem.mu - elite_mean) < np.linalg.norm(mu_old - elite_mean)
    assert info["elite_idx"] == [1, 2]
    assert np.all(cem.var >= 1e-2)  # noise term noise_decay^0 * sigma_init added
    cem.update(params, scores)
    assert info["noise_eps"] == 1e-2 and cem.generation == 2


def test_flat_params_roundtrip():
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    v = flat_params(net)
    v2 = np.random.default_rng(1).standard_normal(v.size)
    set_flat_params(net, v2)
    assert np.allclose(flat_params(net), v2, atol=1e-6)


def _run(tmp_path: Path, episodes: int = 8, pop_size: int = 4):
    cfg = load_config(CFG_PATH)
    cfg["train"]["episodes"] = episodes
    cfg["train"]["checkpoint_every"] = 4
    cfg["train"]["viewer_episodes"] = 0
    cfg["ddpg"]["warmup_steps"] = 64  # let the RL half actually take gradient steps in generation 2
    cfg["method"]["cem"] = {"pop_size": pop_size, "elite_frac": 0.5, "rl_fraction": 0.5}
    env = ArmPathPlanningEnv(cfg["arm"], cfg["scene"], cfg["env"])
    env.reset(seed=0)
    agent = make_agent(env.state_dim, env.action_space.shape[0], cfg, device="cpu", seed=0, env=env)
    logger = CSVLogger(tmp_path / "train_log.csv", CEMRLTrainer.CSV_FIELDS)
    trainer = CEMRLTrainer(cfg, env, agent, [], tmp_path, seed=0, quiet=True, logger=logger)
    summary = trainer.train()
    logger.close()
    return cfg, env, trainer, summary


def test_one_generation_artifacts(tmp_path):
    cfg, env, trainer, summary = _run(tmp_path, episodes=8, pop_size=4)
    rows = list(csv.DictReader(open(tmp_path / "train_log.csv")))
    assert len(rows) == 8
    assert [int(r["episode"]) for r in rows] == list(range(1, 9))
    for k in ("return", "reached", "steps", "collisions", "final_distance", "min_clearance", "env_steps", "wall_time_s", "updates"):
        assert all(r[k] != "" for r in rows), k
    assert int(rows[-1]["env_steps"]) == summary["total_env_steps"] == sum(int(r["steps"]) for r in rows)
    assert summary["population"]["generations"] == 2
    assert summary["population"]["grad_steps_total"] > 0  # RL half updated in generation 2
    assert summary["updates"] == summary["population"]["grad_steps_total"]
    assert (tmp_path / "checkpoints" / "ddpg_ep000004.pt").exists()
    assert (tmp_path / "checkpoints" / "ddpg_ep000008.pt").exists()
    json.dumps(summary)  # serialisable

    # best actor stored in ddpg_final.pt, loadable through make_agent with the same algo
    fresh = make_agent(env.state_dim, env.action_space.shape[0], cfg, device="cpu", seed=5, env=env)
    fresh.load(tmp_path / "ddpg_final.pt")
    assert np.allclose(flat_params(fresh.actor), trainer.best_params, atol=1e-6)
    obs, _ = env.reset()
    a = fresh.act(obs, 0.0)
    assert a.shape == (6,) and np.all(np.abs(a) <= 1.0)


def test_cli_smoke(tmp_path):
    out = tmp_path / "run"
    cmd = [sys.executable, str(ROOT / "scripts" / "train_cemrl.py"), "--config", str(CFG_PATH), "--episodes", "8", "--out", str(out),
           "--quiet", "--set", "method.cem.pop_size=4", "--set", "train.checkpoint_every=4", "--set", "train.viewer_episodes=1"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    assert (out / "config.yaml").exists() and (out / "ddpg_final.pt").exists()
    s = json.loads((out / "train_summary.json").read_text())
    assert s["episodes"] == 8 and s["population"]["pop_size"] == 4
    assert (out / "viewer" / "manifest.json").exists()
