from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PX4_SOURCE_PATH = Path(__file__).resolve().parents[2] / "ref" / "PX4-Autopilot"
PRIMITIVE_TYPES = {
    "bool",
    "byte",
    "char",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
}
FIELD_RE = re.compile(r"^(?P<type>[A-Za-z][A-Za-z0-9_]*(?:\[[0-9]*\])?)\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)")


def load_px4_msg_schema(source_path: Optional[str | Path] = None) -> dict[str, list[str]]:
    source_root = Path(source_path) if source_path else DEFAULT_PX4_SOURCE_PATH
    return _load_px4_msg_schema_cached(str(source_root.resolve()))


def is_valid_topic_field(signal: str, source_path: Optional[str | Path] = None) -> bool:
    if "." not in signal:
        return False
    topic, field = signal.split(".", 1)
    fields = load_px4_msg_schema(source_path).get(topic)
    return bool(fields and field in fields)


def resolve_topic_field(
    topic: Optional[str],
    field: str,
    source_path: Optional[str | Path] = None,
) -> Optional[str]:
    if not topic or not field:
        return None
    candidate = f"{topic}.{field}"
    if is_valid_topic_field(candidate, source_path):
        return candidate

    schema = load_px4_msg_schema(source_path)
    fields = schema.get(topic, [])
    suffix = f".{field}"
    matches = [schema_field for schema_field in fields if schema_field.endswith(suffix)]
    if len(matches) == 1:
        return f"{topic}.{matches[0]}"
    return None


def field_or_flattened_prefix_present(field: str, fields: Iterable[str]) -> bool:
    if not field:
        return False
    known_fields = set(fields)
    return field in known_fields or any(item.startswith(f"{field}.") for item in known_fields)


@lru_cache(maxsize=8)
def _load_px4_msg_schema_cached(source_root: str) -> dict[str, list[str]]:
    msg_dir = Path(source_root) / "msg"
    if not msg_dir.exists():
        return {}

    messages = {
        path.stem: _parse_msg_file(path)
        for path in msg_dir.glob("*.msg")
    }
    schema: dict[str, list[str]] = {}
    for message_name in messages:
        fields = sorted(_expand_message_fields(message_name, messages, seen=set()))
        topics = messages[message_name]["topics"] or [_camel_to_snake(message_name)]
        for topic in topics:
            schema[topic] = fields
    return schema


def _parse_msg_file(path: Path) -> dict[str, list[tuple[str, str]] | list[str]]:
    fields: list[tuple[str, str]] = []
    topics: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# TOPICS"):
            topics.extend(line.split()[2:])
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = FIELD_RE.match(line)
        if not match:
            continue
        field_type = match.group("type").split("[", 1)[0]
        field_name = match.group("name")
        if field_name.isupper() or "=" in line:
            continue
        fields.append((field_type, field_name))
    return {"fields": fields, "topics": topics}


def _expand_message_fields(
    message_name: str,
    messages: dict[str, dict[str, list]],
    *,
    seen: set[str],
) -> set[str]:
    if message_name in seen:
        return set()
    seen = {*seen, message_name}
    message = messages.get(message_name)
    if not message:
        return set()

    expanded: set[str] = set()
    for field_type, field_name in message["fields"]:
        if field_type in PRIMITIVE_TYPES or field_type not in messages:
            expanded.add(field_name)
            continue
        for nested in _expand_message_fields(field_type, messages, seen=seen):
            expanded.add(f"{field_name}.{nested}")
    return expanded


def _camel_to_snake(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()
