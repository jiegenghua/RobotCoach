"""Gymnasium wrappers for MetaWorld EmboCoach training."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np


class ShapedRewardWrapper(gym.Wrapper):
    """Replace env rewards with workspace ``reward.compute_shaped_reward`` output."""

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Callable[..., float],
    ):
        super().__init__(env)
        self.reward_fn = reward_fn
        self._last_obs: np.ndarray | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        assert self._last_obs is not None
        shaped = self.reward_fn(self._last_obs, action, float(env_reward), info)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, shaped, terminated, truncated, info
