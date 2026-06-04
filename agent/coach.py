"""Rule-based coach agent demonstrating Draft-Debug-Improve on MetaWorld."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from embocoach.agent.tools import DebugTestTool, FileEditorTool, TaskTrackerTool
from embocoach.substrate.metaworld.evaluate import evaluate_policy
from embocoach.substrate.metaworld.train import checkpoint_path, train
from embocoach.tasks import EmboTask, TaskSetting


@dataclass
class CoachStep:
    name: str
    success: bool
    detail: str


@dataclass
class CoachRunResult:
    steps: list[CoachStep]
    final_metrics: dict


IMPROVEMENTS = (
    {
        "name": "stronger_distance_shaping",
        "reward_old": "DISTANCE_SCALE = 0.3",
        "reward_new": "DISTANCE_SCALE = 1.0",
        "config_old": "hidden_dims: [128, 128]",
        "config_new": "hidden_dims: [256, 256]",
    },
    {
        "name": "higher_success_bonus",
        "reward_old": "SUCCESS_BONUS = 2.0",
        "reward_new": "SUCCESS_BONUS = 5.0",
        "config_old": "total_timesteps: 200000",
        "config_new": "total_timesteps: 300000",
    },
    {
        "name": "lower_action_penalty",
        "reward_old": "ACTION_PENALTY = 0.001",
        "reward_new": "ACTION_PENALTY = 0.0001",
        "config_old": "learning_rate: 0.0003",
        "config_new": "learning_rate: 0.0005",
    },
)


class RuleBasedCoach:
    """Iteratively improves reward shaping and hyperparameters."""

    def __init__(self, task: EmboTask, workspace: Path):
        self.task = task
        self.workspace = workspace
        self.files = FileEditorTool(workspace, task.editable_files)
        self.debug = DebugTestTool(workspace, task.id)
        self.tracker = TaskTrackerTool(workspace)

    def _best_checkpoint_path(self) -> Path:
        return self.workspace / "checkpoints" / f"{self.task.id}_best.zip"

    def run(self, max_iterations: int = 3, eval_episodes: int = 20) -> CoachRunResult:
        steps: list[CoachStep] = []
        self.tracker.add_task("Analyze PRD and baseline substrate")
        self.tracker.add_task("Draft reward/policy improvements", "in_progress")

        if self.task.setting == TaskSetting.SCRATCH:
            self._restore_scratch_implementation()
            steps.append(
                CoachStep("restore_shaping", True, "Enabled dense reward shaping for scratch task")
            )

        best_sr = -1.0
        best_ckpt: Path | None = None
        final_metrics: dict = {}

        for i, patch in enumerate(IMPROVEMENTS[:max_iterations]):
            self.files.str_replace("reward.py", patch["reward_old"], patch["reward_new"])
            self.files.str_replace("config.yaml", patch["config_old"], patch["config_new"])

            debug_result = self.debug.run()
            debug_sr = float(debug_result.data.get("success_rate", 0.0))

            train(self.task.id, self.workspace)
            ckpt = checkpoint_path(self.workspace, self.task.id)
            eval_metrics = evaluate_policy(
                self.task.id,
                self.workspace,
                checkpoint=ckpt,
                num_episodes=eval_episodes,
            )
            sr = float(eval_metrics.get("success_rate", 0.0))

            steps.append(
                CoachStep(
                    f"iteration_{i + 1}_{patch['name']}",
                    debug_result.ok,
                    f"debug_sr={debug_sr:.3f} eval_sr={sr:.3f}",
                )
            )
            if sr >= best_sr:
                best_sr = sr
                best_dest = self._best_checkpoint_path()
                best_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt, best_dest)
                best_ckpt = best_dest
                final_metrics = {
                    **eval_metrics,
                    "debug_success_rate": debug_sr,
                    "best_checkpoint": str(best_dest),
                }
            else:
                self.files.str_replace("reward.py", patch["reward_new"], patch["reward_old"])
                self.files.str_replace("config.yaml", patch["config_new"], patch["config_old"])

        if best_ckpt is not None and best_ckpt.exists():
            shutil.copy2(best_ckpt, checkpoint_path(self.workspace, self.task.id))

        self.tracker.update_task(1, "completed")
        self.tracker.add_note(f"Best eval success rate: {best_sr:.3f}")
        return CoachRunResult(steps=steps, final_metrics=final_metrics)

    def _restore_scratch_implementation(self) -> None:
        self.files.str_replace("reward.py", "USE_DENSE_SHAPING = False", "USE_DENSE_SHAPING = True")
