"""LLM backends for EmboCoach-Bench: OpenAI-compatible APIs and Anthropic Claude."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embocoach.agent.tools import FileEditorTool, TaskTrackerTool, ToolResult
from embocoach.substrate.metaworld.debug_test import debug_smoke_train
from embocoach.substrate.metaworld.train import train
from embocoach.tasks import EmboTask


def _parse_action_json(text: str) -> dict[str, Any]:
    """Parse model output into a JSON action dict (handles markdown fences / extra prose)."""
    raw = text.strip()
    if not raw:
        return {}

    def _try_load(candidate: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    direct = _try_load(raw)
    if direct is not None:
        return direct

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        fenced = _try_load(fence.group(1).strip())
        if fenced is not None:
            return fenced

    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    for match in re.finditer(r"\{", raw):
        try:
            obj, _ = decoder.raw_decode(raw, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            best = obj
    if best is not None:
        return best

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        sliced = _try_load(raw[start : end + 1])
        if sliced is not None:
            return sliced

    raise ValueError(f"Model did not return valid JSON (first 400 chars): {raw[:400]!r}")


def build_system_prompt(task: EmboTask) -> str:
    editable = ", ".join(task.editable_files)
    return f"""You are a Senior Robotics Learning Engineer optimizing MetaWorld RL policies.

Task: {task.id} ({task.env_name})
Human baseline success rate: {task.human_baseline:.2f}
Editable files: {editable}

Follow the Draft-Debug-Improve loop:
1. view files before editing (especially reward.py and config.yaml)
2. edit via exact string replacement (one match per edit action; chain multiple edits if needed)
3. Run debug after each meaningful change (short training smoke test — verifies code runs; no eval)
4. Run train after debug passes — runs full training then evaluation; feedback includes success_rate for tuning
5. Use track to record subgoals and notes

Respond with ONLY a single raw JSON object — no markdown fences, no explanation before or after.
Available actions:

view:   {{"action": "view", "path": "reward.py"}}
edit:   {{"action": "edit", "path": "reward.py", "old_str": "exact text", "new_str": "replacement"}}
debug:  {{"action": "debug"}}
train:  {{"action": "train"}}
track:  {{"action": "track", "op": "add", "description": "...", "status": "pending|in_progress|completed"}}
        {{"action": "track", "op": "note", "text": "..."}}
        {{"action": "track", "op": "summary"}}
done:   {{"action": "done", "reason": "..."}}

Optimization levers (use both):
- **reward.py** — redesign `compute_shaped_reward`, not only scalar constants. You may change
  which distances/terms you use (gripper vs object vs goal), add multi-stage shaping, grasp/contact
  proxies, success bonuses, action penalties, or toggle `USE_DENSE_SHAPING`. Keep the function
  signature `(obs, action, env_reward, info, task_index=None) -> float` and import
  `env_success` from `embocoach.substrate.metaworld.env` when using success in shaping.
  MetaWorld obs layout (when len >= 39): gripper [0:3], object [4:7], goal [36:39].
- **config.yaml** — `policy.hidden_dims`, `training.*` hyperparameters (LR, timesteps, rollout, etc.)

Do not change evaluation success criteria; benchmark success_rate still uses native info["success"] only.
"""


class BaseLLMAgent(ABC):
    @abstractmethod
    def next_action(
        self, messages: list[dict[str, str]], *, system_prompt: str
    ) -> dict[str, Any]:
        ...


class OpenAIAgent(BaseLLMAgent):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        use_json_response_format: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install optional LLM support: pip install 'embocoach-metaworld[llm]'"
            ) from exc
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self._use_json_response_format = use_json_response_format

    def next_action(
        self, messages: list[dict[str, str]], *, system_prompt: str
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "max_tokens": 4096,
        }
        if self._use_json_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"
        if self._use_json_response_format:
            return json.loads(content)
        return _parse_action_json(content)


class AnthropicAgent(BaseLLMAgent):
    """Anthropic Messages API (Claude)."""

    def __init__(self, model: str, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Claude support requires anthropic: pip install 'embocoach-metaworld[llm]'"
            ) from exc
        kwargs: dict[str, str] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model

    def next_action(
        self, messages: list[dict[str, str]], *, system_prompt: str
    ) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=messages,
        )
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return _parse_action_json("".join(parts))


def create_llm_agent(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseLLMAgent:
    """
    Factory for LLM backends.

    * openai — ``OPENAI_API_KEY``, optional ``OPENAI_BASE_URL``, optional ``OPENAI_MODEL``.
    * qwen — Alibaba DashScope OpenAI-compatible mode: ``DASHSCOPE_API_KEY`` or
      ``QWEN_API_KEY``, default base ``https://dashscope.aliyuncs.com/compatible-mode/v1``.
    * volcano — Volcengine Ark (方舟) OpenAI-compatible: ``ARK_API_KEY`` or
      ``VOLCENGINE_API_KEY``, default base ``https://ark.cn-beijing.volces.com/api/v3``.
    * mimo — Xiaomi MiMo OpenAI-compatible: ``MIMO_API_KEY``, default base
      ``https://token-plan-cn.xiaomimimo.com/v1``, default model ``mimo-v2.5-pro``.
    * claude — ``ANTHROPIC_API_KEY``, optional ``ANTHROPIC_MODEL``.
    """
    p = (provider or "openai").strip().lower()
    if p == "claude":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Claude requires an API key: use --api-key or set ANTHROPIC_API_KEY"
            )
        m = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        return AnthropicAgent(model=m, api_key=key)

    if p == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        m = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        return OpenAIAgent(model=m, api_key=key, base_url=url, use_json_response_format=True)

    if p == "qwen":
        key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        if not key:
            raise ValueError(
                "Qwen requires an API key: use --api-key or set DASHSCOPE_API_KEY / QWEN_API_KEY"
            )
        url = base_url or os.environ.get(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        m = model or os.environ.get("QWEN_MODEL", "qwen-plus")
        return OpenAIAgent(
            model=m, api_key=key, base_url=url, use_json_response_format=False
        )

    if p in ("volcano", "ark", "volcengine"):
        key = api_key or os.environ.get("ARK_API_KEY") or os.environ.get("VOLCENGINE_API_KEY")
        if not key:
            raise ValueError(
                "Volcano Ark requires an API key: use --api-key or set "
                "ARK_API_KEY / VOLCENGINE_API_KEY"
            )
        url = base_url or os.environ.get(
            "VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
        )
        m = model or os.environ.get("VOLCENGINE_MODEL")
        if not m:
            raise ValueError(
                "Volcano Ark requires a model or endpoint id: use --model or set VOLCENGINE_MODEL"
            )
        return OpenAIAgent(
            model=m, api_key=key, base_url=url, use_json_response_format=False
        )

    if p == "mimo":
        key = api_key or os.environ.get("MIMO_API_KEY")
        if not key:
            raise ValueError(
                "MiMo requires an API key: use --api-key or set MIMO_API_KEY"
            )
        url = base_url or os.environ.get(
            "MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1"
        )
        m = model or os.environ.get("MIMO_MODEL", "mimo-v2.5-pro")
        return OpenAIAgent(
            model=m, api_key=key, base_url=url, use_json_response_format=False
        )

    raise ValueError(
        f"Unknown provider {provider!r}. Use: openai, claude, qwen, volcano, mimo"
    )


@dataclass
class LLMSessionResult:
    task_id: str
    turns: list[dict[str, Any]] = field(default_factory=list)
    log_path: Path | None = None
    stopped_reason: str = "max_turns"


class LLMCoachSession:
    """Runs an LLM-driven engineering session with tool use."""

    def __init__(self, task: EmboTask, workspace: Path, agent: BaseLLMAgent, max_turns: int = 10):
        self.task = task
        self.workspace = workspace
        self.agent = agent
        self.max_turns = max_turns
        self.system_prompt = build_system_prompt(task)
        self.files = FileEditorTool(workspace, task.editable_files)
        self.tracker = TaskTrackerTool(workspace)
        prd = (workspace / "PRD.md").read_text()
        self.messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"{prd}\n\nBegin by viewing reward.py and config.yaml. "
                    f"You may edit reward shaping design and/or hyperparameters. "
                    f"After each change run debug, then train; use success_rate to iterate."
                ),
            },
        ]

    def _run_debug(self) -> ToolResult:
        metrics = debug_smoke_train(self.task.id, self.workspace)
        ok = metrics.get("status", "ok") == "ok"
        return ToolResult(ok, json.dumps(metrics, indent=2), metrics)

    def _format_train_eval_feedback(self, metrics: dict[str, Any]) -> str:
        sr = metrics.get("success_rate")
        baseline = self.task.human_baseline
        delta = (float(sr) - baseline) if sr is not None else None
        lines = [
            "Full training and evaluation finished.",
            f"success_rate: {sr}",
            f"human_baseline: {baseline:.2f}",
        ]
        if delta is not None:
            lines.append(f"delta_vs_human: {delta:+.3f}")
        lines.append("Use success_rate to decide the next reward/hyperparameter edits.")
        lines.append("")
        lines.append(json.dumps(metrics, indent=2))
        return "\n".join(lines)

    def _run_train_eval(self) -> ToolResult:
        try:
            metrics = train(self.task.id, self.workspace, run_eval=True)
            metrics["status"] = "ok"
        except Exception as exc:
            metrics = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return ToolResult(False, json.dumps(metrics, indent=2), metrics)
        return ToolResult(
            True,
            self._format_train_eval_feedback(metrics),
            metrics,
        )

    def _dispatch(self, action: dict[str, Any]) -> Any:
        kind = action.get("action")
        if kind == "view":
            return self.files.view(action["path"])
        if kind == "edit":
            return self.files.str_replace(action["path"], action["old_str"], action["new_str"])
        if kind == "debug":
            return self._run_debug()
        if kind == "train":
            return self._run_train_eval()
        if kind == "evaluate":
            return type("R", (), {
                "ok": False,
                "output": 'Use {"action": "train"} — training always includes evaluation.',
            })()
        if kind == "track":
            op = action.get("op", "summary")
            if op == "add":
                return self.tracker.add_task(
                    action["description"], action.get("status", "pending")
                )
            if op == "note":
                return self.tracker.add_note(action["text"])
            return self.tracker.summary()
        if kind == "done":
            return None
        return type("R", (), {"ok": False, "output": f"Unknown action: {kind}"})()

    def run(self) -> LLMSessionResult:
        log: list[dict[str, Any]] = []
        stopped_reason = "max_turns"

        for turn in range(self.max_turns):
            try:
                action = self.agent.next_action(
                    self.messages, system_prompt=self.system_prompt
                )
            except ValueError as exc:
                entry = {
                    "turn": turn,
                    "action": {"action": "parse_error"},
                    "ok": False,
                    "feedback": (
                        f"{exc}\n\nReply with ONLY one JSON object (no prose). "
                        "Example: {\"action\": \"debug\"}"
                    ),
                }
                log.append(entry)
                self.messages.append(
                    {
                        "role": "user",
                        "content": entry["feedback"],
                    }
                )
                continue
            except Exception as exc:
                entry = {
                    "turn": turn,
                    "action": {"action": "error"},
                    "ok": False,
                    "feedback": f"{type(exc).__name__}: {exc}",
                }
                log.append(entry)
                stopped_reason = "agent_error"
                break

            kind = action.get("action")
            entry: dict[str, Any] = {"turn": turn, "action": action}

            if kind == "done":
                entry["ok"] = True
                entry["feedback"] = action.get("reason", "done")
                log.append(entry)
                stopped_reason = "done"
                break

            try:
                result = self._dispatch(action)
            except (PermissionError, KeyError, FileNotFoundError) as exc:
                entry["ok"] = False
                entry["feedback"] = str(exc)
                log.append(entry)
                self.messages.append({"role": "assistant", "content": json.dumps(action)})
                self.messages.append({"role": "user", "content": entry["feedback"]})
                continue

            if result is None:
                entry["ok"] = True
                entry["feedback"] = action.get("reason", "done")
                log.append(entry)
                stopped_reason = "done"
                break

            entry["ok"] = result.ok
            entry["feedback"] = result.output
            if hasattr(result, "data") and result.data:
                entry["data"] = result.data
            log.append(entry)

            self.messages.append({"role": "assistant", "content": json.dumps(action)})
            self.messages.append({"role": "user", "content": entry["feedback"]})

        result = LLMSessionResult(
            task_id=self.task.id,
            turns=log,
            stopped_reason=stopped_reason,
        )
        log_path = self.workspace / "logs" / "llm_session.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "task_id": result.task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stopped_reason": result.stopped_reason,
                    "human_baseline": self.task.human_baseline,
                    "turns": result.turns,
                },
                indent=2,
            )
        )
        result.log_path = log_path
        return result
