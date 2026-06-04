"""PPO training with Stable-Baselines3 for MetaWorld EmboCoach tasks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from embocoach.substrate.metaworld.env import make_env
from embocoach.substrate.metaworld.evaluate import evaluate_policy

from embocoach.substrate.metaworld.policy import policy_config_from_dict, policy_kwargs_from_config
from embocoach.substrate.metaworld.workspace_load import load_policy, load_reward
from embocoach.substrate.metaworld.wrappers import ShapedRewardWrapper
from embocoach.tasks import get_task


def checkpoint_path(workspace: Path, task_id: str) -> Path:
    return workspace / "checkpoints" / f"{task_id}.zip"


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


class TrainTimeLimitCallback(BaseCallback):
    """Stop ``model.learn`` when a wall-clock budget is exceeded."""

    def __init__(self, limit_sec: float, verbose: int = 0):
        super().__init__(verbose)
        self.limit_sec = limit_sec
        self._start = 0.0

    def _on_training_start(self) -> None:
        self._start = time.time()

    def _on_step(self) -> bool:
        return (time.time() - self._start) < self.limit_sec


def _make_env(task, seed: int, reward_mod):
    env, _ = make_env(task, seed=seed)
    return ShapedRewardWrapper(env, reward_mod.compute_shaped_reward)


def _build_ppo(env, cfg: dict, policy_mod) -> PPO:
    t_cfg = cfg["training"]
    p_cfg = policy_config_from_dict(cfg["policy"])
    if hasattr(policy_mod, "PolicyConfig"):
        p_cfg = policy_mod.PolicyConfig(
            hidden_dims=tuple(cfg["policy"]["hidden_dims"]),
            activation=cfg["policy"]["activation"],
            log_std_init=float(cfg["policy"]["log_std_init"]),
        )
    policy_kwargs = (
        policy_mod.policy_kwargs_from_config(p_cfg)
        if hasattr(policy_mod, "policy_kwargs_from_config")
        else policy_kwargs_from_config(p_cfg)
    )
    device = cfg.get("device", "cpu")
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=float(t_cfg["learning_rate"]),
        n_steps=int(t_cfg["rollout_steps"]),
        batch_size=int(t_cfg["minibatch_size"]),
        n_epochs=int(t_cfg["epochs"]),
        gamma=float(t_cfg["gamma"]),
        gae_lambda=float(t_cfg["gae_lambda"]),
        clip_range=float(t_cfg["clip_range"]),
        vf_coef=float(t_cfg["vf_coef"]),
        ent_coef=float(t_cfg["ent_coef"]),
        max_grad_norm=float(t_cfg["max_grad_norm"]),
        policy_kwargs=policy_kwargs,
        seed=int(cfg.get("seed", 42)),
        device=device,
        verbose=0,
    )


def train(
    task_id: str,
    workspace: Path,
    output: Path | None = None,
    *,
    run_eval: bool = True,
) -> dict:
    task = get_task(task_id)
    workspace = workspace.resolve()
    reward_mod = load_reward(workspace)
    policy_mod = load_policy(workspace)
    cfg = load_config(workspace / "config.yaml")
    seed = int(cfg.get("seed", 42))
    set_random_seed(seed)
    np.random.seed(seed)

    env = _make_env(task, seed, reward_mod)
    model = _build_ppo(env, cfg, policy_mod)

    t_cfg = cfg["training"]
    total_timesteps = int(t_cfg["total_timesteps"])
    time_limit = float(t_cfg.get("train_time_limit_sec", 3600))
    callback = TrainTimeLimitCallback(time_limit)

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    out = output or checkpoint_path(workspace, task_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out.with_suffix("")))

    metrics = {
        "timesteps": total_timesteps,
        "checkpoint": str(out),
    }
    env.close()

    if run_eval:
        eval_metrics = evaluate_policy(task_id, workspace, checkpoint=out)
        metrics.update(eval_metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    metrics = train(args.task, args.workspace.resolve(), args.output)
    print(yaml.safe_dump(metrics))


if __name__ == "__main__":
    main()
