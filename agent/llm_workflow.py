"""Structured Draft-Debug-Improve workflow driven by an LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embocoach.agent.llm_agent import BaseLLMAgent, _parse_action_json
from embocoach.agent.tools import DebugTestTool, FileEditorTool
from embocoach.substrate.metaworld.evaluate import evaluate_policy
from embocoach.substrate.metaworld.train import train
from embocoach.tasks import EmboTask


def build_workflow_system_prompt(task: EmboTask) -> str:
    editable = ", ".join(task.editable_files)
    return f"""You are a Senior Robotics Learning Engineer optimizing MetaWorld RL policies.

Task: {task.id} ({task.env_name})
Human baseline success rate: {task.human_baseline:.2f}
Editable files: {editable}

You ONLY propose file edits. You do NOT control the workflow.
The system runs a fixed loop after each proposal:
  view files -> apply your edits -> debug-test -> (if debug passes) train + evaluate

You receive current file contents plus a memory log of prior iterations (edits and post-training eval success rates).
Use that history to decide what to change next.

Respond with ONLY a single raw JSON object — no markdown fences, no explanation outside JSON.
Schema:

{{
  "rationale": "brief explanation of this iteration's changes",
  "edits": [
    {{"path": "reward.py", "old_str": "exact text to replace", "new_str": "replacement"}}
  ]
}}

Primary levers: reward shaping in reward.py, policy.hidden_dims and training hyperparameters in config.yaml.
Success is measured by native MetaWorld info["success"] during evaluation (success_rate).
"""


def _format_memory_log(memory: list[dict[str, Any]]) -> str:
    if not memory:
        return "(no prior iterations)"

    lines: list[str] = []
    for entry in memory:
        lines.append(f"### Iteration {entry['iteration']}")
        if entry.get("rationale"):
            lines.append(f"Rationale: {entry['rationale']}")
        for edit in entry.get("edits", []):
            path = edit.get("path", "?")
            old_preview = (edit.get("old_str") or "")[:80].replace("\n", "\\n")
            new_preview = (edit.get("new_str") or "")[:80].replace("\n", "\\n")
            ok = "applied" if edit.get("ok") else "failed"
            lines.append(f"- edit {path} ({ok}): {old_preview!r} -> {new_preview!r}")
        if entry.get("debug_ok") is False:
            lines.append(f"Debug: FAILED ({entry.get('error', 'see logs')})")
        elif entry.get("eval_success_rate") is not None:
            lines.append(f"Success rate (post-training eval): {entry['eval_success_rate']:.3f}")
        elif entry.get("error"):
            lines.append(f"Error: {entry['error']}")
        lines.append("")
    return "\n".join(lines).strip()


def _parse_edit_proposal(text: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    return _parse_action_json(text)


@dataclass
class LLMWorkflowResult:
    task_id: str
    human_baseline: float
    baseline_debug: dict[str, Any]
    iterations: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = "max_iter"
    last_eval_success_rate: float | None = None
    best_eval_success_rate: float = 0.0
    log_path: Path | None = None


class LLMWorkflowSession:
    """Fixed loop: view -> LLM edit -> debug -> (if pass) train/eval -> memory."""

    def __init__(
        self,
        task: EmboTask,
        workspace: Path,
        agent: BaseLLMAgent,
        *,
        max_iter: int = 5,
        edit_retries: int = 3,
        eval_episodes: int | None = None,
    ):
        self.task = task
        self.workspace = workspace
        self.agent = agent
        self.max_iter = max_iter
        self.edit_retries = edit_retries
        self.eval_episodes = eval_episodes
        self.system_prompt = build_workflow_system_prompt(task)
        self.files = FileEditorTool(workspace, task.editable_files)
        self.debug = DebugTestTool(workspace, task.id)
        self.memory: list[dict[str, Any]] = []
        self.prd = (workspace / "PRD.md").read_text()

    def _view_context(self) -> str:
        parts: list[str] = []
        for path in ("reward.py", "config.yaml"):
            if path in self.task.editable_files:
                result = self.files.view(path)
                parts.append(f"--- {path} ---\n{result.output}")
        return "\n\n".join(parts)

    def _propose_edits(self, iteration: int) -> dict[str, Any]:
        user_content = (
            f"{self.prd}\n\n"
            f"## Memory log (prior iterations)\n{_format_memory_log(self.memory)}\n\n"
            f"## Current files (iteration {iteration}/{self.max_iter})\n"
            f"{self._view_context()}\n\n"
            "Propose edits for this iteration. The system will apply them, run debug-test, "
            "and if debug passes, run full training and evaluation."
        )
        messages = [{"role": "user", "content": user_content}]
        return _parse_edit_proposal(
            self.agent.next_action(messages, system_prompt=self.system_prompt)
        )

    def _apply_edits(self, edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        snapshots: dict[str, str] = {}
        for edit in edits:
            path = edit.get("path", "")
            if path and path not in snapshots:
                rel = Path(path).name if "/" not in path else str(Path(path).name)
                if rel in self.files.editable:
                    snapshots[rel] = (self.workspace / rel).read_text()

        applied: list[dict[str, Any]] = []
        for edit in edits:
            path = edit.get("path", "")
            old_str = edit.get("old_str", "")
            new_str = edit.get("new_str", "")
            record = {"path": path, "old_str": old_str, "new_str": new_str}
            try:
                result = self.files.str_replace(path, old_str, new_str)
                record["ok"] = result.ok
                record["feedback"] = result.output
            except (PermissionError, KeyError, FileNotFoundError) as exc:
                record["ok"] = False
                record["feedback"] = str(exc)
            applied.append(record)
            if not record["ok"]:
                for rel, content in snapshots.items():
                    (self.workspace / rel).write_text(content)
                break
        return applied

    def _run_train_eval(self) -> tuple[dict[str, Any], dict[str, Any]]:
        train_metrics = train(self.task.id, self.workspace)
        eval_metrics = evaluate_policy(
            self.task.id,
            self.workspace,
            num_episodes=self.eval_episodes,
        )
        return train_metrics, eval_metrics

    def _request_and_apply_edits(
        self, iteration: int
    ) -> tuple[str, list[dict[str, Any]] | None, str | None]:
        """Ask LLM for edits and apply them. Returns (rationale, applied_edits, error)."""
        rationale = ""
        for attempt in range(self.edit_retries):
            try:
                proposal = self._propose_edits(iteration)
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt + 1 >= self.edit_retries:
                    return "", None, f"Failed to parse LLM proposal: {exc}"
                continue

            rationale = str(proposal.get("rationale", ""))
            edits = proposal.get("edits") or []
            if not edits:
                if attempt + 1 >= self.edit_retries:
                    return rationale, None, "LLM returned no edits"
                continue

            applied = self._apply_edits(edits)
            if all(e.get("ok") for e in applied):
                return rationale, applied, None

            if attempt + 1 < self.edit_retries:
                failed = [
                    f"{e['path']}: {e.get('feedback', 'failed')}"
                    for e in applied
                    if not e.get("ok")
                ]
                self.memory.append(
                    {
                        "iteration": iteration,
                        "rationale": rationale,
                        "edits": applied,
                        "note": "edit_retry",
                        "feedback": "; ".join(failed),
                    }
                )
        return rationale, None, "Edits could not be applied after retries"

    def _run_iteration(self, iteration: int) -> dict[str, Any] | None:
        """One fixed workflow step. Returns entry dict, or None to stop (human baseline)."""
        rationale, applied_edits, edit_error = self._request_and_apply_edits(iteration)
        if edit_error:
            entry = {
                "iteration": iteration,
                "rationale": rationale,
                "edits": applied_edits or [],
                "error": edit_error,
            }
            self.memory.append(entry)
            return entry

        debug_result = self.debug.run()
        debug_ok = debug_result.ok

        entry: dict[str, Any] = {
            "iteration": iteration,
            "rationale": rationale,
            "edits": applied_edits,
            "debug_ok": debug_ok,
            "debug": debug_result.data,
        }

        if not debug_ok:
            entry["error"] = debug_result.data.get("error", "debug test failed")
            self.memory.append(entry)
            return entry

        try:
            train_metrics, eval_metrics = self._run_train_eval()
            eval_sr = float(eval_metrics.get("success_rate", 0.0))
        except Exception as exc:
            entry["error"] = str(exc)
            self.memory.append(entry)
            return entry

        entry["eval_success_rate"] = eval_sr
        entry["train"] = train_metrics
        entry["eval"] = eval_metrics
        self.memory.append(entry)

        if eval_sr >= self.task.human_baseline:
            return entry  # caller checks human baseline

        return entry

    def run(self) -> LLMWorkflowResult:
        baseline_debug = self.debug.run().data
        best_sr = 0.0
        last_eval_sr: float | None = None
        stopped_reason = "max_iter"
        iterations: list[dict[str, Any]] = []

        for iteration in range(1, self.max_iter + 1):
            entry = self._run_iteration(iteration)
            iterations.append(entry)

            eval_sr = entry.get("eval_success_rate")
            if eval_sr is not None:
                last_eval_sr = float(eval_sr)
                best_sr = max(best_sr, last_eval_sr)
                if last_eval_sr >= self.task.human_baseline:
                    stopped_reason = "human_baseline"
                    break

        result = LLMWorkflowResult(
            task_id=self.task.id,
            human_baseline=self.task.human_baseline,
            baseline_debug=baseline_debug,
            iterations=iterations,
            stopped_reason=stopped_reason,
            last_eval_success_rate=last_eval_sr,
            best_eval_success_rate=best_sr,
        )
        log_path = self.workspace / "logs" / "llm_workflow.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "task_id": result.task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "human_baseline": result.human_baseline,
                    "stopped_reason": result.stopped_reason,
                    "last_eval_success_rate": result.last_eval_success_rate,
                    "best_eval_success_rate": result.best_eval_success_rate,
                    "baseline_debug": result.baseline_debug,
                    "memory": self.memory,
                    "iterations": result.iterations,
                },
                indent=2,
            )
        )
        result.log_path = log_path
        return result
