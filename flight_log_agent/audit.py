from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents import RunHooks


DEFAULT_DEV_LOG_ROOT = Path(".dev_logs") / "flight_agent_runs"


class DeveloperAuditLogger:
    def __init__(
        self,
        root_dir: Path = DEFAULT_DEV_LOG_ROOT,
        *,
        run_id: Optional[str] = None,
        max_string_chars: int = 50_000,
    ) -> None:
        self.run_id = run_id or _new_run_id()
        self.root_dir = Path(root_dir)
        self.run_dir = self.root_dir / self.run_id
        self.events_path = self.run_dir / "run_events.jsonl"
        self.usage_path = self.run_dir / "usage.json"
        self.metadata_path = self.run_dir / "metadata.json"
        self.max_string_chars = max_string_chars
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: str, **fields: Any) -> None:
        entry = {
            "ts": _utc_now_iso(),
            "event": event,
            "run_id": self.run_id,
            **fields,
        }
        self._append_jsonl(self.events_path, entry)

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "created_at": _utc_now_iso(),
            **metadata,
        }
        self._write_json(self.metadata_path, payload)

    def save_usage(self, usage: Any) -> None:
        self._write_json(self.usage_path, serialize_usage(usage))

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        serialized = self._json_dumps(payload)
        with path.open("a", encoding="utf-8") as file:
            file.write(serialized)
            file.write("\n")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(self._json_dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _json_dumps(self, payload: Any, *, indent: Optional[int] = None) -> str:
        safe_payload = make_json_safe(payload, max_string_chars=self.max_string_chars)
        return json.dumps(safe_payload, indent=indent, sort_keys=True)


class AgentRunAuditHooks(RunHooks):
    def __init__(self, audit_logger: DeveloperAuditLogger) -> None:
        self.audit_logger = audit_logger
        self._tool_starts: dict[str, float] = {}

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: Optional[str],
        input_items: list[Any],
    ) -> None:
        self.audit_logger.log_event(
            "llm.started",
            agent_name=_agent_name(agent),
            input_item_count=len(input_items),
            system_prompt_chars=len(system_prompt or ""),
        )

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        self.audit_logger.log_event(
            "llm.finished",
            agent_name=_agent_name(agent),
            response_id=getattr(response, "response_id", None),
            request_id=getattr(response, "request_id", None),
            usage=serialize_usage(getattr(response, "usage", None)),
        )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        tool_call_id = _tool_call_id(context)
        self._tool_starts[tool_call_id] = time.perf_counter()
        self.audit_logger.log_event(
            "tool.started",
            agent_name=_agent_name(agent),
            tool_name=_tool_name(context, tool),
            tool_call_id=tool_call_id,
            input=_parse_json_maybe(getattr(context, "tool_arguments", None)),
        )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: str) -> None:
        tool_call_id = _tool_call_id(context)
        started_at = self._tool_starts.pop(tool_call_id, None)
        duration_ms = (
            round((time.perf_counter() - started_at) * 1000, 3)
            if started_at is not None
            else None
        )
        self.audit_logger.log_event(
            "tool.finished",
            agent_name=_agent_name(agent),
            tool_name=_tool_name(context, tool),
            tool_call_id=tool_call_id,
            output=_parse_json_maybe(result),
            duration_ms=duration_ms,
        )


def log_run_items(audit_logger: DeveloperAuditLogger, items: list[Any]) -> None:
    for item in items:
        item_type = getattr(item, "type", None)
        raw_item = getattr(item, "raw_item", None)
        raw_type = _raw_item_type(raw_item)
        is_tool_search_item = item_type in {"tool_search_call_item", "tool_search_output_item"}
        is_hosted_tool_item = (
            item_type in {"tool_call_item", "tool_call_output_item"}
            and raw_type not in {"function_call", "function_call_output"}
        )
        if not is_tool_search_item and not is_hosted_tool_item:
            continue

        audit_logger.log_event(
            "hosted_tool.item",
            item_type=item_type,
            raw_item=raw_item,
        )


def serialize_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "requests": getattr(usage, "requests", 0),
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
        "cached_input_tokens": getattr(input_details, "cached_tokens", 0) or 0,
        "cache_write_tokens": (
            getattr(input_details, "cache_write_tokens", 0) or 0
        ),
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", 0) or 0,
        "request_usage_entries": [
            serialize_usage(entry)
            for entry in (getattr(usage, "request_usage_entries", None) or [])
        ],
    }


def make_json_safe(value: Any, *, max_string_chars: int = 50_000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_string(value, max_string_chars)

    if isinstance(value, bytes):
        return _truncate_string(value.decode("utf-8", errors="replace"), max_string_chars)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item, max_string_chars=max_string_chars)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item, max_string_chars=max_string_chars) for item in value]

    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump(), max_string_chars=max_string_chars)

    if is_dataclass(value):
        return make_json_safe(asdict(value), max_string_chars=max_string_chars)

    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist(), max_string_chars=max_string_chars)

    return _truncate_string(repr(value), max_string_chars)


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _truncate_string(value: str, max_string_chars: int) -> Any:
    if len(value) <= max_string_chars:
        return value

    return {
        "truncated": True,
        "original_length": len(value),
        "preview": value[:max_string_chars],
    }


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _tool_call_id(context: Any) -> str:
    return str(getattr(context, "tool_call_id", None) or "unknown")


def _tool_name(context: Any, tool: Any) -> Optional[str]:
    return (
        getattr(context, "tool_name", None)
        or getattr(context, "qualified_tool_name", None)
        or getattr(tool, "name", None)
    )


def _agent_name(agent: Any) -> Optional[str]:
    return getattr(agent, "name", None) or getattr(agent, "kwargs", {}).get("name")


def _raw_item_type(raw_item: Any) -> Optional[str]:
    if isinstance(raw_item, dict):
        return raw_item.get("type")
    return getattr(raw_item, "type", None)
