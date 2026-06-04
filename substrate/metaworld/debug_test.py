"""Debug helpers: smoke training (LLM debug action) and short train+eval (CLI)."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

from embocoach.substrate.metaworld.train import train


def debug_smoke_train(task_id: str, workspace: Path) -> dict:
    """Short PPO run to verify reward/policy code loads and trains (no evaluation)."""
    cfg_path = workspace / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    debug_cfg = cfg.get("debug", {})
    backup = cfg_path.read_text()
    try:
        cfg["training"]["total_timesteps"] = int(debug_cfg.get("total_timesteps", 5000))
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f)
        metrics = train(task_id, workspace, run_eval=False)
        metrics["status"] = "ok"
        return metrics
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        cfg_path.write_text(backup)


def debug_test(task_id: str, workspace: Path) -> dict:
    cfg_path = workspace / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    debug_cfg = cfg.get("debug", {})
    original = cfg["training"]["total_timesteps"]
    cfg["training"]["total_timesteps"] = int(debug_cfg.get("total_timesteps", 5000))
    cfg["evaluation"]["num_episodes"] = int(debug_cfg.get("num_episodes", 10))
    cfg["evaluation"]["max_episode_steps"] = int(debug_cfg.get("max_episode_steps", 200))

    backup = cfg_path.read_text()
    try:
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f)
        metrics = train(task_id, workspace)
        metrics["status"] = "ok"
        return metrics
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        cfg_path.write_text(backup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    args = parser.parse_args()
    result = debug_test(args.task, args.workspace.resolve())
    print(yaml.safe_dump(result))
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
