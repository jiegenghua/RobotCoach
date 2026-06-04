"""MetaWorld environment helpers."""

from __future__ import annotations

import gymnasium as gym
import metaworld
import numpy as np

from embocoach.tasks import EmboTask


def make_env(
    task: EmboTask,
    seed: int = 42,
    *,
    render_mode: str | None = None,
    visualize: bool = False,
):
    """Create a Meta-World MT1 env for training or evaluation.

    Returns ``(env, None)`` — the second value is kept for API compatibility.
    """
    del visualize
    env = gym.make(
        "Meta-World/MT1",
        env_name=task.env_name,
        seed=seed,
        render_mode=render_mode,
    )
    return env, None


def render_frame(env, task_index: int | None) -> None:
    """Draw one frame from the env used during evaluation."""
    del task_index
    env.render()


def info_get(info, key: str, index: int | None = None):
    """Read a field from MetaWorld step info (dict or list)."""
    del index
    if isinstance(info, dict):
        val = info.get(key, info.get(f"_{key}"))
        if val is None:
            return False
        return val
    if isinstance(info, list):
        item = info[0]
        return item.get(key, item.get(f"_{key}", False))
    return False


def env_success(info, index: int | None = None) -> bool:
    del index
    val = info_get(info, "success")
    if isinstance(val, (list, np.ndarray)):
        return bool(val[0])
    return bool(val)
