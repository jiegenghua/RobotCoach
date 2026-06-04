"""Physics-informed reward shaping (primary agent edit target).

The baseline uses intentionally weak shaping so agents can improve success rates
through the Draft-Debug-Improve loop.
"""

from __future__ import annotations

import numpy as np


# Tunable coefficients — agents typically increase these during optimization.
DISTANCE_SCALE = 0.3
SUCCESS_BONUS = 2.0
ACTION_PENALTY = 0.001
USE_DENSE_SHAPING = True


def compute_shaped_reward(
    obs: np.ndarray,
    action: np.ndarray,
    env_reward: float,
    info,
    task_index: int | None = None,
) -> float:
    """Return shaped reward used for PPO updates."""
    del task_index
    base = float(env_reward)
    if not USE_DENSE_SHAPING:
        return base

    state = np.asarray(obs, dtype=np.float32)
    # MetaWorld obs: [0:3] gripper, [4:7] object, [36:39] goal.
    if state.shape[-1] >= 27:
        gripper = state[0:3]
        obj = state[4:7]
        goal = state[36:39]
        dist = float(np.linalg.norm(gripper - goal))
        shaped = -DISTANCE_SCALE * dist
    else:
        shaped = 0.0

    from embocoach.substrate.metaworld.env import env_success

    success = env_success(info)

    bonus = SUCCESS_BONUS if success else 0.0
    action_cost = ACTION_PENALTY * float(np.sum(np.square(action)))
    return base + shaped + bonus - action_cost
