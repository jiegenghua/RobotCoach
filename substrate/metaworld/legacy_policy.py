"""Legacy custom ActorCritic for loading pre-SB3 ``.pt`` checkpoints."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from embocoach.substrate.metaworld.policy import PolicyConfig


def _activation(name: str):
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: tuple[int, ...], activation: str):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        act = _activation(activation)
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), act])
            prev = dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.actor = MLP(obs_dim, action_dim, cfg.hidden_dims, cfg.activation)
        self.critic = MLP(obs_dim, 1, cfg.hidden_dims, cfg.activation)
        self.log_std = nn.Parameter(torch.full((action_dim,), cfg.log_std_init))

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        dist = self._distribution(obs_t)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs_t).squeeze(-1)
        return action.detach().numpy(), log_prob.item(), value.item()


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        nn.init.zeros_(module.bias)
