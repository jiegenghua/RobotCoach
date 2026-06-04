"""Draft-Debug-Improve workflow orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from embocoach.agent.coach import RuleBasedCoach
from embocoach.agent.tools import DebugTestTool
from embocoach.tasks import get_task
from embocoach.workspace import init_workspace


@dataclass
class WorkflowResult:
    task_id: str
    baseline: dict
    final: dict
    iterations: list[dict] = field(default_factory=list)
    log_path: Path | None = None


class EmboCoachWorkflow:
    """Closed-loop embodied policy engineering session."""

    def __init__(self, task_id: str, workspace_root: Path | None = None):
        self.task = get_task(task_id)
        self.workspace = init_workspace(task_id, workspace_root)

    def run_baseline(self) -> dict:
        debug = DebugTestTool(self.workspace, self.task.id)
        return debug.run().data

    def run_agentic(self, max_iterations: int = 3) -> WorkflowResult:
        baseline = self.run_baseline()
        coach = RuleBasedCoach(self.task, self.workspace)
        coach_result = coach.run(max_iterations=max_iterations)
        final_metrics = coach_result.final_metrics or baseline

        result = WorkflowResult(
            task_id=self.task.id,
            baseline=baseline,
            final=final_metrics,
            iterations=[
                {"step": s.name, "ok": s.success, "detail": s.detail}
                for s in coach_result.steps
            ],
        )
        log_path = self.workspace / "logs" / "workflow.json"
        log_path.write_text(
            json.dumps(
                {
                    "task_id": result.task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "baseline": result.baseline,
                    "final": result.final,
                    "iterations": result.iterations,
                },
                indent=2,
            )
        )
        result.log_path = log_path
        return result

