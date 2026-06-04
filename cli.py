"""CLI for EmboCoach-Bench MetaWorld."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embocoach.agent.workflow import EmboCoachWorkflow
from embocoach.tasks import TASKS
from embocoach.workspace import init_workspace, workspace_path


def main():
    parser = argparse.ArgumentParser(description="EmboCoach-Bench (MetaWorld)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-tasks", help="List MetaWorld benchmark tasks")
    p_list.set_defaults(func=cmd_list_tasks)

    p_init = sub.add_parser("init", help="Initialize task workspace (C_env)")
    p_init.add_argument("--task", required=True)
    p_init.add_argument("--force", action="store_true")
    p_init.add_argument("--workspace-root", type=Path, default=None)
    p_init.set_defaults(func=cmd_init)

    p_debug = sub.add_parser("debug-test", help="Run quick validation (10 episodes)")
    p_debug.add_argument("--task", required=True)
    p_debug.add_argument("--workspace-root", type=Path, default=None)
    p_debug.set_defaults(func=cmd_debug)

    p_train = sub.add_parser("train", help="Full PPO training run")
    p_train.add_argument("--task", required=True)
    p_train.add_argument("--workspace-root", type=Path, default=None)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate trained checkpoint")
    p_eval.add_argument("--task", required=True)
    p_eval.add_argument("--workspace-root", type=Path, default=None)
    p_eval.add_argument("--episodes", type=int, default=None)
    p_eval.add_argument(
        "--render",
        action="store_true",
        help="Show MuJoCo viewer during evaluation (uses render_episodes from config)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_run = sub.add_parser("run-agent", help="Run Draft-Debug-Improve agent loop")
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--iterations", type=int, default=3)
    p_run.add_argument("--workspace-root", type=Path, default=None)
    p_run.set_defaults(func=cmd_run_agent)

    p_llm = sub.add_parser("run-llm-agent", help="Run LLM-driven Draft-Debug-Improve loop")
    p_llm.add_argument("--task", required=True)
    p_llm.add_argument(
        "--provider",
        default="openai",
        choices=("openai", "claude", "qwen", "volcano", "ark", "volcengine", "mimo"),
        help=(
            "LLM backend: openai (OpenAI-compatible), claude (Anthropic), "
            "qwen (DashScope compatible-mode), volcano/ark (Volcengine 方舟), "
            "mimo (Xiaomi MiMo)"
        ),
    )
    p_llm.add_argument(
        "--model",
        default=None,
        help="Model or endpoint id (defaults per provider; see embocoach.agent.llm_agent.create_llm_agent)",
    )
    p_llm.add_argument(
        "--api-key",
        default=None,
        dest="api_key",
        help="API key (otherwise OPENAI_API_KEY, MIMO_API_KEY, DASHSCOPE_API_KEY, ARK_API_KEY, …)",
    )
    p_llm.add_argument(
        "--base-url",
        default=None,
        dest="base_url",
        help="Override base URL for OpenAI-compatible providers (openai, qwen, volcano, mimo)",
    )
    p_llm.add_argument("--max-turns", type=int, default=10)
    p_llm.add_argument("--workspace-root", type=Path, default=None)
    p_llm.set_defaults(func=cmd_run_llm_agent)

    p_wf = sub.add_parser(
        "run-llm-workflow",
        help="Run structured LLM workflow (view/edit -> debug -> train/eval per iteration)",
    )
    p_wf.add_argument("--task", required=True)
    p_wf.add_argument(
        "--provider",
        default="openai",
        choices=("openai", "claude", "qwen", "volcano", "ark", "volcengine", "mimo"),
    )
    p_wf.add_argument("--model", default=None)
    p_wf.add_argument("--api-key", default=None, dest="api_key")
    p_wf.add_argument("--base-url", default=None, dest="base_url")
    p_wf.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum workflow iterations (default: 5)",
    )
    p_wf.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override evaluation episode count (default: from config.yaml)",
    )
    p_wf.add_argument("--workspace-root", type=Path, default=None)
    p_wf.set_defaults(func=cmd_run_llm_workflow)

    args = parser.parse_args()
    args.func(args)


def cmd_list_tasks(_args):
    for task_id, task in sorted(TASKS.items()):
        print(
            f"{task_id:30} protocol={task.protocol:4}  "
            f"setting={task.setting.value:7}  human={task.human_baseline:.2f}"
        )


def cmd_init(args):
    path = init_workspace(args.task, args.workspace_root, force=args.force)
    print(path)


def cmd_debug(args):
    from embocoach.substrate.metaworld.debug_test import debug_test

    ws = workspace_path(args.task, args.workspace_root)
    if not ws.exists():
        init_workspace(args.task, args.workspace_root)
    result = debug_test(args.task, ws)
    print(json.dumps(result, indent=2))


def cmd_train(args):
    from embocoach.substrate.metaworld.train import train

    ws = workspace_path(args.task, args.workspace_root)
    if not ws.exists():
        init_workspace(args.task, args.workspace_root)
    result = train(args.task, ws)
    print(json.dumps(result, indent=2))


def cmd_eval(args):
    from embocoach.substrate.metaworld.evaluate import evaluate_policy

    ws = workspace_path(args.task, args.workspace_root)
    result = evaluate_policy(
        args.task, ws, num_episodes=args.episodes, render=args.render
    )
    print(json.dumps(result, indent=2))


def cmd_run_agent(args):
    flow = EmboCoachWorkflow(args.task, args.workspace_root)
    result = flow.run_agentic(max_iterations=args.iterations)
    print(json.dumps(
        {
            "task_id": result.task_id,
            "baseline_success_rate": result.baseline.get("success_rate"),
            "final_success_rate": result.final.get("success_rate"),
            "human_baseline": flow.task.human_baseline,
            "iterations": result.iterations,
            "log": str(result.log_path),
        },
        indent=2,
    ))


def cmd_run_llm_agent(args):
    from embocoach.agent.llm_agent import LLMCoachSession, create_llm_agent
    from embocoach.tasks import get_task

    ws = workspace_path(args.task, args.workspace_root)
    if not ws.exists():
        init_workspace(args.task, args.workspace_root)
    task = get_task(args.task)
    agent = create_llm_agent(
        args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    session = LLMCoachSession(task, ws, agent, max_turns=args.max_turns)
    result = session.run()

    last_train_eval = None
    for turn in reversed(result.turns):
        data = turn.get("data") or {}
        if (
            turn.get("action", {}).get("action") == "train"
            and data.get("success_rate") is not None
        ):
            last_train_eval = data
            break

    print(json.dumps(
        {
            "task_id": result.task_id,
            "stopped_reason": result.stopped_reason,
            "turns": len(result.turns),
            "human_baseline": task.human_baseline,
            "last_eval_success_rate": (
                last_train_eval.get("success_rate") if last_train_eval else None
            ),
            "log": str(result.log_path),
        },
        indent=2,
    ))


def cmd_run_llm_workflow(args):
    from embocoach.agent.llm_agent import create_llm_agent
    from embocoach.agent.llm_workflow import LLMWorkflowSession
    from embocoach.tasks import get_task

    ws = workspace_path(args.task, args.workspace_root)
    if not ws.exists():
        init_workspace(args.task, args.workspace_root)
    task = get_task(args.task)
    agent = create_llm_agent(
        args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    session = LLMWorkflowSession(
        task,
        ws,
        agent,
        max_iter=args.max_iter,
        eval_episodes=args.eval_episodes,
    )
    result = session.run()

    print(json.dumps(
        {
            "task_id": result.task_id,
            "stopped_reason": result.stopped_reason,
            "human_baseline": result.human_baseline,
            "last_eval_success_rate": result.last_eval_success_rate,
            "best_eval_success_rate": result.best_eval_success_rate,
            "iterations": len(result.iterations),
            "log": str(result.log_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
