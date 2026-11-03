"""Training/evaluation plugins (research contributions C1-C6).

A plugin is a class named ``Plugin`` in ``arm_rl/plugins/<name>.py``.  It is enabled by
listing ``<name>`` under ``method.plugins`` in the config, with its options under
``method.<name>``.  See :mod:`arm_rl.plugins.base` for the hook contract.
"""

from __future__ import annotations

import importlib

from arm_rl.plugins.base import Plugin


def load_plugins(cfg: dict, env, agent, phase: str = "train") -> list[Plugin]:
    method = dict(cfg.get("method") or {})
    plugins: list[Plugin] = []
    for name in method.get("plugins", []) or []:
        mod = importlib.import_module(f"arm_rl.plugins.{name}")
        opts = dict(method.get(name) or {})
        opts.setdefault("episodes", int((cfg.get("train") or {}).get("episodes", 0)) or None)  # budget for schedules
        p = mod.Plugin(opts, env, agent, phase=phase)
        plugins.append(p)
    return plugins
