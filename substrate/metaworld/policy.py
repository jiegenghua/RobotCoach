"""Policy network settings for Stable-Baselines3 PPO (MlpPolicy)."""

from __future__ import annotations

from dataclasses import dataclass

import torch as th


@dataclass
class PolicyConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    log_std_init: float = -0.5


def activation_fn(name: str) -> type[th.nn.Module]:
    if name == "tanh":
        return th.nn.Tanh
    if name == "relu":
        return th.nn.ReLU
    raise ValueError(f"Unsupported activation: {name}")


def policy_kwargs_from_config(cfg: PolicyConfig) -> dict:
    """Build SB3 ``policy_kwargs`` from workspace policy config."""
    hidden = list(cfg.hidden_dims)
    return {
        "net_arch": dict(pi=hidden, vf=hidden),
        "activation_fn": activation_fn(cfg.activation),
        "log_std_init": cfg.log_std_init,
    }


def policy_config_from_dict(raw: dict) -> PolicyConfig:
    return PolicyConfig(
        hidden_dims=tuple(raw["hidden_dims"]),
        activation=str(raw.get("activation", "tanh")),
        log_std_init=float(raw.get("log_std_init", -0.5)),
    )
