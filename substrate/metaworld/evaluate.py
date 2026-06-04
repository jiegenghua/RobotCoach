"""Policy evaluation using native MetaWorld success checks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from embocoach.substrate.metaworld.env import env_success, make_env, render_frame
from embocoach.substrate.metaworld.legacy_policy import ActorCritic
from embocoach.substrate.metaworld.policy import PolicyConfig
from embocoach.tasks import get_task


def checkpoint_path(workspace: Path, task_id: str) -> Path:
    return workspace / "checkpoints" / f"{task_id}.zip"


def _legacy_checkpoint_path(workspace: Path, task_id: str) -> Path:
    return workspace / "checkpoints" / f"{task_id}.pt"


def resolve_checkpoint(workspace: Path, task_id: str, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint
    preferred = checkpoint_path(workspace, task_id)
    if preferred.exists():
        return preferred
    legacy = _legacy_checkpoint_path(workspace, task_id)
    if legacy.exists():
        return legacy
    return preferred


def load_sb3_model(checkpoint: Path, device: str) -> PPO:
    return PPO.load(str(checkpoint), device=device)


def load_legacy_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy_cfg = PolicyConfig(
        hidden_dims=tuple(payload["policy"]["hidden_dims"]),
        activation=payload["policy"]["activation"],
        log_std_init=float(payload["policy"]["log_std_init"]),
    )
    model = ActorCritic(payload["obs_dim"], payload["action_dim"], policy_cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def evaluate_policy(
    task_id: str,
    workspace: Path,
    checkpoint: Path | None = None,
    num_episodes: int | None = None,
    *,
    render: bool = False,
) -> dict:
    task = get_task(task_id)
    workspace = workspace.resolve()
    with (workspace / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg["evaluation"]
    if num_episodes is None:
        episodes = int(
            eval_cfg.get("render_episodes", 5) if render else eval_cfg["num_episodes"]
        )
    else:
        episodes = num_episodes
    max_steps = int(eval_cfg["max_episode_steps"])
    render_delay = float(eval_cfg.get("render_delay_sec", 0.02))
    device_str = str(cfg.get("device", "cpu"))
    ckpt = resolve_checkpoint(workspace, task_id, checkpoint)
    use_sb3 = ckpt.suffix == ".zip" or str(ckpt).endswith(".zip")

    render_mode = "human" if render else None
    env, _ = make_env(
        task,
        seed=int(cfg.get("seed", 42)),
        render_mode=render_mode,
        visualize=render,
    )

    if use_sb3:
        model = load_sb3_model(ckpt, device_str)
    else:
        model = load_legacy_model(ckpt, torch.device(device_str))

    successes = 0
    returns = []

    for _ in range(episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        for _ in range(max_steps):
            if use_sb3:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _, _ = model.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            done = bool(terminated or truncated)
            success = env_success(info)

            if render:
                render_frame(env, None)
                if render_delay > 0:
                    time.sleep(render_delay)

            if success:
                successes += 1
                break
            if done:
                break
        returns.append(ep_return)

    env.close()
    success_rate = successes / episodes
    metrics = {
        "success_rate": success_rate,
        "mean_return": float(np.mean(returns)),
        "episodes": episodes,
        "human_baseline": task.human_baseline,
        "delta_vs_human": success_rate - task.human_baseline,
    }
    if render:
        metrics["rendered"] = True
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a MuJoCo viewer while evaluating (defaults to render_episodes)",
    )
    args = parser.parse_args()
    metrics = evaluate_policy(
        args.task,
        args.workspace.resolve(),
        checkpoint=args.checkpoint,
        num_episodes=args.episodes,
        render=args.render,
    )
    print(yaml.safe_dump(metrics))


if __name__ == "__main__":
    main()
