"""Agent action space: Terminal, FileEditor, TaskTracker, DebugTest."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embocoach.substrate.metaworld.debug_test import debug_test


@dataclass
class ToolResult:
    ok: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)


class FileEditorTool:
    def __init__(self, workspace: Path, editable: tuple[str, ...]):
        self.workspace = workspace
        self.editable = set(editable)

    def view(self, path: str) -> ToolResult:
        rel = self._check(path)
        content = (self.workspace / rel).read_text()
        return ToolResult(True, content, {"path": rel, "lines": content.count("\n") + 1})

    def str_replace(self, path: str, old_str: str, new_str: str) -> ToolResult:
        rel = self._check(path)
        file_path = self.workspace / rel
        content = file_path.read_text()
        if old_str not in content:
            return ToolResult(False, f"old_str not found in {rel}")
        updated = content.replace(old_str, new_str, 1)
        file_path.write_text(updated)
        return ToolResult(True, f"Updated {rel}", {"path": rel})

    def _check(self, path: str) -> str:
        rel = Path(path).name if "/" not in path else str(Path(path).name)
        if rel not in self.editable:
            raise PermissionError(f"Editing '{rel}' is not allowed. Editable: {sorted(self.editable)}")
        return rel


class TerminalTool:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def run(self, command: str, timeout: int = 3600) -> ToolResult:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult(proc.returncode == 0, output, {"returncode": proc.returncode})


class TaskTrackerTool:
    def __init__(self, workspace: Path):
        self.path = workspace / "logs" / "task_tracker.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"tasks": [], "notes": []}

    def add_task(self, description: str, status: str = "pending") -> ToolResult:
        self.state["tasks"].append({"description": description, "status": status})
        self._save()
        return ToolResult(True, f"Added task: {description}")

    def update_task(self, index: int, status: str) -> ToolResult:
        self.state["tasks"][index]["status"] = status
        self._save()
        return ToolResult(True, f"Task {index} -> {status}")

    def add_note(self, note: str) -> ToolResult:
        self.state["notes"].append(note)
        self._save()
        return ToolResult(True, note)

    def summary(self) -> ToolResult:
        return ToolResult(True, json.dumps(self.state, indent=2), self.state)

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2))


class DebugTestTool:
    def __init__(self, workspace: Path, task_id: str):
        self.workspace = workspace
        self.task_id = task_id

    def run(self) -> ToolResult:
        metrics = debug_test(self.task_id, self.workspace)
        ok = metrics.get("status", "ok") == "ok"
        return ToolResult(ok, json.dumps(metrics, indent=2), metrics)
