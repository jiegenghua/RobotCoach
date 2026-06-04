"""Task registry for MetaWorld EmboCoach-Bench tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class TaskSetting(str, Enum):
    IMPROVE = "improve"
    SCRATCH = "scratch"


@dataclass(frozen=True)
class EmboTask:
    """One benchmark task instance T = (D_prd, P_sys, C_env)."""

    id: str
    env_name: str
    description: str
    setting: TaskSetting
    protocol: Literal["mt1"]
    human_baseline: float
    editable_files: tuple[str, ...]


IMPROVE_EDITABLE = (
    "policy.py",
    "reward.py",
    "config.yaml",
    "train.py",
)

SCRATCH_EDITABLE = (
    "policy.py",
    "reward.py",
    "config.yaml",
    "train.py",
)


def _mt1_task(
    task_id: str,
    env_suffix: str,
    description: str,
    human: float,
    setting: TaskSetting,
) -> EmboTask:
    return EmboTask(
        id=task_id,
        env_name=f"{env_suffix}-v3",
        description=description,
        setting=setting,
        protocol="mt1",
        human_baseline=human,
        editable_files=SCRATCH_EDITABLE if setting == TaskSetting.SCRATCH else IMPROVE_EDITABLE,
    )


def _st_task(task_id: str, env_suffix: str, description: str, human: float) -> EmboTask:
    return EmboTask(
        id=task_id,
        env_name=f"{env_suffix}-v3",
        description=description,
        setting=TaskSetting.IMPROVE,
        protocol="mt1",
        human_baseline=human,
        editable_files=IMPROVE_EDITABLE,
    )


TASKS: dict[str, EmboTask] = {
    "peg-insert-side-mt1": _mt1_task(
        "peg-insert-side-mt1",
        "peg-insert-side",
        "Insert a peg sideways. Randomize peg and goal positions.",
        0.66,
        TaskSetting.IMPROVE,
    ),
    "pick-place-mt1": _mt1_task(
        "pick-place-mt1",
        "pick-place",
        "Pick and place a puck to a goal. Randomize puck and goal positions.",
        0.36,
        TaskSetting.IMPROVE,
    ),
    "push-mt1": _mt1_task(
        "push-mt1",
        "push",
        "Push the puck to a goal. Randomize puck and goal positions.",
        0.54,
        TaskSetting.IMPROVE,
    ),
    "button-press-topdown-mt1": _mt1_task(
        "button-press-topdown-mt1",
        "button-press-topdown",
        "Press a button from the top. Randomize button positions.",
        1.00,
        TaskSetting.SCRATCH,
    ),
    "door-open-mt1": _mt1_task(
        "door-open-mt1",
        "door-open",
        "Open a door with a revolving joint. Randomize door positions.",
        1.00,
        TaskSetting.SCRATCH,
    ),
    "drawer-close-mt1": _mt1_task(
        "drawer-close-mt1",
        "drawer-close",
        "Push and close a drawer. Randomize drawer positions.",
        1.00,
        TaskSetting.SCRATCH,
    ),
    "drawer-open-mt1": _mt1_task(
        "drawer-open-mt1",
        "drawer-open",
        "Open a drawer. Randomize drawer positions.",
        1.00,
        TaskSetting.SCRATCH,
    ),
    "reach-mt1": _mt1_task(
        "reach-mt1",
        "reach",
        "Reach a goal position. Randomize goal positions.",
        1.00,
        TaskSetting.SCRATCH,
    ),
    "window-close-mt1": _mt1_task(
        "window-close-mt1",
        "window-close",
        "Push and close a window. Randomize window positions.",
        1.00,
        TaskSetting.SCRATCH,
    ),
    "window-open-mt1": _mt1_task(
        "window-open-mt1",
        "window-open",
        "Push and open a window. Randomize window positions.",
        1.00,
        TaskSetting.SCRATCH,
    ),
    "hand-insert-st": _st_task(
        "hand-insert-st",
        "hand-insert",
        "Insert the gripper into a hole.",
        0.50,
    ),
    "pick-out-of-hole-st": _st_task(
        "pick-out-of-hole-st",
        "pick-out-of-hole",
        "Pick up a puck from a hole. Randomize puck and goal positions.",
        0.50,
    ),
    "coffee-pull-st": _st_task(
        "coffee-pull-st",
        "coffee-pull",
        "Grasp a mug and pull it across the table to a designated target location.",
        0.15,
    ),
    "pick-place-st": _st_task(
        "pick-place-st",
        "pick-place",
        "Pick and place a puck to a goal. Randomize puck and goal positions.",
        0.85,
    ),
}

# Legacy MT10 task IDs (same tasks, now trained with Meta-World/MT1).
TASK_ALIASES: dict[str, str] = {
    "peg-insert-side-mt10": "peg-insert-side-mt1",
    "pick-place-mt10": "pick-place-mt1",
    "push-mt10": "push-mt1",
    "button-press-topdown-mt10": "button-press-topdown-mt1",
    "door-open-mt10": "door-open-mt1",
    "drawer-close-mt10": "drawer-close-mt1",
    "drawer-open-mt10": "drawer-open-mt1",
    "reach-mt10": "reach-mt1",
    "window-close-mt10": "window-close-mt1",
    "window-open-mt10": "window-open-mt1",
}


def get_task(task_id: str) -> EmboTask:
    canonical = TASK_ALIASES.get(task_id, task_id)
    if canonical not in TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Available: {', '.join(sorted(TASKS))}")
    return TASKS[canonical]
