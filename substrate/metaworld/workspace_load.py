"""Load editable modules from a task workspace."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_workspace_module(workspace: Path, module_name: str) -> ModuleType | None:
    path = workspace / f"{module_name}.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"embocoach_workspace_{workspace.name}_{module_name}",
        path,
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_reward(workspace: Path):
    from embocoach.substrate.metaworld import reward as default

    return load_workspace_module(workspace, "reward") or default


def load_policy(workspace: Path):
    from embocoach.substrate.metaworld import policy as default

    return load_workspace_module(workspace, "policy") or default
