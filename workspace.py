"""Workspace management for development substrate (C_env)."""

from __future__ import annotations

import shutil
from pathlib import Path

from embocoach.tasks import EmboTask, TaskSetting, get_task

PACKAGE_ROOT = Path(__file__).resolve().parent
SUBSTRATE_TEMPLATE = PACKAGE_ROOT / "substrate" / "metaworld"
PRD_DIR = PACKAGE_ROOT / "prd"


def default_workspaces_root() -> Path:
    return Path.cwd() / "workspaces"


def workspace_path(task_id: str, root: Path | None = None) -> Path:
    return (root or default_workspaces_root()) / task_id


def init_workspace(task_id: str, root: Path | None = None, force: bool = False) -> Path:
    task = get_task(task_id)
    dest = workspace_path(task_id, root)
    if dest.exists():
        if not force:
            return dest
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for item in SUBSTRATE_TEMPLATE.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)

    if task.setting == TaskSetting.SCRATCH:
        _apply_scratch_skeleton(dest)

    (dest / "checkpoints").mkdir(exist_ok=True)
    (dest / "logs").mkdir(exist_ok=True)
    prd = render_prd(task)
    (dest / "PRD.md").write_text(prd)
    return dest


def _apply_scratch_skeleton(workspace: Path) -> None:
    """Disable dense shaping so agents must implement reward engineering."""
    reward_path = workspace / "reward.py"
    reward_path.write_text(
        reward_path.read_text().replace("USE_DENSE_SHAPING = True", "USE_DENSE_SHAPING = False")
    )


def _task_prd_body(task: EmboTask) -> str:
    path = PRD_DIR / f"{task.id}.md"
    if path.exists():
        return path.read_text()
    return (
        f"**Task:** {task.id} (`{task.env_name}`)\n\n"
        f"{task.description}\n\n"
        f"**Human baseline:** {task.human_baseline:.2f}\n"
    )


def render_prd(task: EmboTask) -> str:
    shared = (PRD_DIR / "_shared.md").read_text()
    task_doc = _task_prd_body(task)
    editable = "\n".join(f"  - `{path}`" for path in task.editable_files)
    return (
        f"# Product Requirement Document: {task.id}\n\n"
        f"{shared}\n\n"
        f"## Task-Specific Domain Scaffolding\n\n"
        f"{task_doc}\n\n"
        f"## Editable Files (strict)\n\n{editable}\n\n"
        f"## Acceptance Criteria\n\n"
        f"- Maximize native MetaWorld success rate on **{task.env_name}**.\n"
        f"- Human-engineered baseline: **{task.human_baseline:.2f}**.\n"
        f"- Evaluate over **100 episodes** after training (`evaluate.py`).\n"
        f"- Run `debug_test.py` (10 episodes) before each full training submission.\n"
    )
