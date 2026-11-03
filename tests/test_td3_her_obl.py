"""Fast tests for research contribution C3: TD3 learner, HER and OBL plugins."""

import logging

import numpy as np
import torch

from arm_rl.agents import make_agent
from arm_rl.ddpg import DDPGAgent
from arm_rl.env import ArmPathPlanningEnv
from arm_rl.plugins import load_plugins
from arm_rl.td3 import TD3Agent
from arm_rl.utils import load_config

CFG = load_config("configs/default.yaml")


def make_env(**env_over):
    env_cfg = dict(CFG["env"], **env_over)
    return ArmPathPlanningEnv(CFG["arm"], CFG["scene"], env_cfg)


def td3_cfg(**over):
    cfg = dict(CFG["ddpg"], warmup_steps=32, batch_size=16, policy_delay=2, encoder="mlp")
    cfg.update(over)
    return cfg


def fill_buffer(agent, env, n=64, seed=0):
    env.reset(seed=seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n):
        a = agent.random_action()
        obs2, r, term, trunc, info = env.step(a)
        agent.store(obs, a, r, obs2, term)
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()


def snap_equal(a: dict, b: dict) -> bool:
    if not (np.allclose(a["q"], b["q"]) and np.allclose(a["ee"], b["ee"])):
        return False
    for key in ("step_count", "ee_path_length", "n_collisions", "n_hist"):
        if a[key] != b[key]:
            return False
    sa, sb = a["scene"], b["scene"]
    return np.allclose(sa.target, sb.target) and np.allclose(sa.obs_q1, sb.obs_q1) and np.allclose(sa.obs_q2, sb.obs_q2) and np.isclose(sa.t, sb.t)


# ------------------------------------------------------------------ TD3
def test_td3_update_runs_and_losses_finite():
    env = make_env()
    agent = TD3Agent(env.state_dim, 6, td3_cfg(), seed=0)
    assert not agent.ready()
    fill_buffer(agent, env)
    assert agent.ready()
    out = None
    for _ in range(4):
        out = agent.update()
    assert set(out) >= {"critic_loss", "actor_loss", "q_mean"}
    assert all(np.isfinite(v) for v in out.values())
    assert agent.n_updates == 4 and agent.n_actor_updates == 2
    a = agent.act(env.reset(seed=1)[0], noise_std=0.1)
    assert a.shape == (6,) and np.all(np.abs(a) <= 1.0)


def test_td3_delayed_actor_update():
    env = make_env()
    agent = TD3Agent(env.state_dim, 6, td3_cfg(policy_delay=3, actor_lr=1e-2), seed=0)
    fill_buffer(agent, env)
    w0 = [p.detach().clone() for p in agent.actor.parameters()]
    agent.update()
    agent.update()
    assert all(torch.equal(a, b) for a, b in zip(w0, agent.actor.parameters()))
    agent.update()
    assert any(not torch.equal(a, b) for a, b in zip(w0, agent.actor.parameters()))


def test_td3_save_load_roundtrip(tmp_path):
    env = make_env()
    agent = TD3Agent(env.state_dim, 6, td3_cfg(), seed=0)
    fill_buffer(agent, env)
    for _ in range(3):
        agent.update()
    path = tmp_path / "td3.pt"
    agent.save(path)
    other = TD3Agent(env.state_dim, 6, td3_cfg(), seed=99)
    other.load(path)
    obs, _ = env.reset(seed=5)
    assert np.allclose(agent.act(obs), other.act(obs))
    s = torch.zeros(1, env.state_dim)
    a = torch.zeros(1, 6)
    assert torch.allclose(agent.critic2(s, a), other.critic2(s, a))
    assert other.n_updates == 3 and other.n_actor_updates == 1


def test_td3_loads_ddpg_checkpoint_actor_only(tmp_path, caplog):
    env = make_env()
    ddpg = DDPGAgent(env.state_dim, 6, td3_cfg(), seed=3)
    path = tmp_path / "ddpg.pt"
    ddpg.save(path)
    td3 = TD3Agent(env.state_dim, 6, td3_cfg(), seed=7)
    c0 = [p.detach().clone() for p in td3.critic2.parameters()]
    with caplog.at_level(logging.WARNING, logger="arm_rl.td3"):
        td3.load(path)
    assert any("actor only" in r.getMessage() for r in caplog.records)
    obs, _ = env.reset(seed=5)
    assert np.allclose(ddpg.act(obs), td3.act(obs))
    assert all(torch.equal(a, b) for a, b in zip(c0, td3.critic2.parameters()))


def test_factory_builds_td3():
    env = make_env()
    cfg = dict(CFG, method={"algo": "td3"}, td3={"policy_delay": 3})
    agent = make_agent(env.state_dim, 6, cfg, env=env)
    assert isinstance(agent, TD3Agent) and agent.policy_delay == 3


# ------------------------------------------------------------------ HER
def rollout(env, n_steps=20, seed=0):
    obs, _ = env.reset(seed=seed)
    transitions = []
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        a = rng.uniform(-1, 1, size=6).astype(np.float32)
        obs2, r, term, trunc, info = env.step(a)
        transitions.append((obs, a, r, obs2, term, info))
        obs = obs2
        if term or trunc:
            break
    return transitions


def test_her_future_relabels_k_per_transition():
    env = make_env()
    transitions = rollout(env)
    assert all(not t[5]["collision"] for t in transitions), "test rollout should be collision free"
    cfg = dict(CFG, method={"plugins": ["her"], "her": {"k": 4, "strategy": "future"}})
    (her,) = load_plugins(cfg, env, agent=None, phase="train")
    extra = her.on_episode_end(transitions, episode=1)
    assert len(extra) == 4 * len(transitions)
    assert her.cost["extra_transitions"] == len(extra)
    se, st = slice(*env.obs_layout["se"]), slice(*env.obs_layout["st"])
    achieved = np.stack([t[3][se] for t in transitions])
    for i, (obs, a, r, obs2, term) in enumerate(extra):
        t = i // 4
        goal = obs2[st]
        assert np.allclose(obs[st], goal)
        # goal is an achieved EE of the same or a later step
        dists = np.linalg.norm(achieved[t:] - goal, axis=1)
        assert dists.min() < 1e-6
        d = float(np.linalg.norm(obs2[se] - goal))
        assert np.isclose(r, env.reward_for(d, False))
        assert term == (d < env.reach_threshold)
        # everything but the goal slice is untouched
        mask = np.ones(env.state_dim, bool)
        mask[st] = False
        assert np.allclose(obs[mask], transitions[t][0][mask]) and np.allclose(obs2[mask], transitions[t][3][mask])
    # relabelling with the transition's own achieved EE gives the success reward
    n_success = sum(1 for e in extra if e[2] == 0.0 and e[4])
    assert n_success >= len(transitions)  # at least the last step (future index == itself)


def test_her_final_and_self_goal_reward_zero():
    env = make_env()
    transitions = rollout(env)
    cfg = dict(CFG, method={"plugins": ["her"], "her": {"k": 2, "strategy": "final"}})
    (her,) = load_plugins(cfg, env, agent=None, phase="train")
    extra = her.relabel(transitions)
    assert len(extra) == 2 * len(transitions)
    for e in extra[-2:]:  # last transition relabelled with its own achieved EE
        assert e[2] == 0.0 and e[4] is True
    # goal == achieved EE of the transition itself -> reward 0, terminated
    se = slice(*env.obs_layout["se"])
    o, a, r, o2, term = her.relabel_one(transitions[0], transitions[0][3][se])
    assert r == 0.0 and term


def test_her_sets_target_velocity_to_zero_and_is_noop_in_eval():
    env = make_env(include_velocities=True)
    transitions = rollout(env)
    cfg = dict(CFG, method={"plugins": ["her"], "her": {"k": 1}})
    (her,) = load_plugins(cfg, env, agent=None, phase="train")
    vt = slice(*env.obs_layout["vt"])
    for o, a, r, o2, term in her.on_episode_end(transitions, 1):
        assert np.all(o[vt] == 0) and np.all(o2[vt] == 0)
    (her_eval,) = load_plugins(cfg, env, agent=None, phase="eval")
    assert her_eval.on_episode_end(transitions, 1) == []


def test_her_keeps_collision_penalty():
    env = make_env()
    transitions = rollout(env)
    obs, a, r, obs2, term, info = transitions[0]
    info = dict(info, collision=True)
    cfg = dict(CFG, method={"plugins": ["her"], "her": {"k": 1}})
    (her,) = load_plugins(cfg, env, agent=None, phase="train")
    se = slice(*env.obs_layout["se"])
    o, _, r_new, o2, term_new = her.relabel_one((obs, a, -50.0, obs2, True, info), obs2[se])
    assert r_new == env.collision_penalty and term_new


# ------------------------------------------------------------------ OBL
def test_obl_opposite_transition_matches_env_and_restores_state():
    env = make_env()
    cfg = dict(CFG, method={"plugins": ["obl"], "obl": {"prob": 1.0, "mode": "opposite"}})
    (obl,) = load_plugins(cfg, env, agent=None, phase="train")
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(5):
        a = rng.uniform(-1, 1, size=6).astype(np.float32)
        pre = env.snapshot()
        a_t = obl.transform_action(obs, a, deterministic=False)
        assert np.array_equal(a_t, a)
        obs2, r, term, trunc, info = env.step(a)
        post = env.snapshot()
        hist_post = [h.copy() for h in env.ee_history]
        extra = obl.extra_transitions(obs, a, r, obs2, term, info)
        assert len(extra) == 1
        o, a_opp, r_o, o2, term_o = extra[0]
        assert np.allclose(o, obs) and np.allclose(a_opp, -a)
        # env is back exactly to its post-step state
        after = env.snapshot()
        assert snap_equal(post, after)
        assert len(env.ee_history) == len(hist_post) and all(np.allclose(x, y) for x, y in zip(env.ee_history, hist_post))
        assert np.allclose(env._observation(), obs2)
        # reference: an independent env replaying the opposite action from the pre-step state
        ref = make_env()
        ref.reset(seed=4)
        ref.restore(pre)
        o2_ref, r_ref, term_ref, _, _ = ref.step(-a)
        assert np.allclose(o2, o2_ref, atol=1e-6) and np.isclose(r_o, r_ref) and term_o == term_ref
        obs = obs2
        if term or trunc:
            break
    assert obl.cost["extra_sim_steps"] == obl.cost["extra_transitions"] >= 1
    assert obl.cost["time_s"] > 0


def test_obl_quasi_prob_and_eval_noop():
    env = make_env()
    cfg = dict(CFG, method={"plugins": ["obl"], "obl": {"prob": 0.0, "mode": "quasi"}})
    (obl,) = load_plugins(cfg, env, agent=None, phase="train")
    obs, _ = env.reset(seed=1)
    a = np.ones(6, dtype=np.float32) * 0.5
    q = obl.opposite_action(a)
    assert np.all(q <= 0) and np.all(q >= -0.5)
    obl.transform_action(obs, a, False)
    obs2, r, term, trunc, info = env.step(a)
    assert obl.extra_transitions(obs, a, r, obs2, term, info) == []  # prob 0 -> never augmented
    cfg_eval = dict(CFG, method={"plugins": ["obl"], "obl": {"prob": 1.0}})
    (obl_eval,) = load_plugins(cfg_eval, env, agent=None, phase="eval")
    obl_eval.transform_action(obs, a, True)
    assert obl_eval._pre is None
    assert obl_eval.extra_transitions(obs, a, r, obs2, term, info) == []
