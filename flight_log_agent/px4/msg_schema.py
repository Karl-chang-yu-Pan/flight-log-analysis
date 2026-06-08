from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from flight_log_agent.source_path import resolve_source_path

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
CONSTANT_RE = re.compile(
    r"^(?P<type>[A-Za-z][A-Za-z0-9_]*(?:\[[0-9]*\])?)\s+"
    r"(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>-?(?:0x[0-9A-Fa-f]+|\d+))\b"
)


def load_px4_msg_schema(source_path: Optional[str | Path] = None) -> dict[str, list[str]]:
    source_root = resolve_source_path(source_path)
    if source_root is None:
        return {}
    return _load_px4_msg_schema_cached(str(source_root.resolve()))


def load_px4_msg_enum_registry(source_path: Optional[str | Path] = None) -> dict[str, dict[str, dict[str, int]]]:
    source_root = resolve_source_path(source_path)
    if source_root is None:
        return {}
    return _load_px4_msg_enum_registry_cached(str(source_root.resolve()))


def normalize_px4_enum_value(signal: str, value: Any, source_path: Optional[str | Path] = None) -> Any:
    registry = load_px4_msg_enum_registry(source_path)
    entry = registry.get(signal)
    if not entry or not isinstance(value, str):
        return value
    normalized_values = [_normalize_enum_label(value)]
    if "::" in value:
        normalized_values.append(_normalize_enum_label(value.rsplit("::", 1)[-1]))
    for normalized in dedupe_keep_order(normalized_values):
        if normalized in entry["aliases"]:
            return entry["aliases"][normalized]
    return value


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


@lru_cache(maxsize=8)
def _load_px4_msg_enum_registry_cached(source_root: str) -> dict[str, dict[str, dict[str, int]]]:
    msg_dir = Path(source_root) / "msg"
    if not msg_dir.exists():
        return {}

    messages = {
        path.stem: _parse_msg_file(path)
        for path in msg_dir.glob("*.msg")
    }
    registry: dict[str, dict[str, dict[str, int]]] = {}
    for message_name, message in messages.items():
        topics = message["topics"] or [_camel_to_snake(message_name)]
        enum_fields = _message_enum_fields(message_name, messages, seen=set())
        for topic in topics:
            for field_path, constants in enum_fields.items():
                aliases = _enum_aliases(constants)
                if aliases:
                    registry[f"{topic}.{field_path}"] = {
                        "constants": {constant["name"]: constant["value"] for constant in constants},
                        "aliases": aliases,
                    }
    return registry


def _parse_msg_file(path: Path) -> dict[str, Any]:
    fields: list[tuple[str, str]] = []
    field_meta: dict[str, dict[str, Any]] = {}
    constants: list[dict[str, Any]] = []
    topics: list[str] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        raw = raw_line.strip()
        if not raw:
            continue
        if raw.startswith("# TOPICS"):
            topics.extend(raw.split()[2:])
            continue
        line, _, comment = raw.partition("#")
        line = line.strip()
        comment = comment.strip()
        if not line:
            continue
        constant_match = CONSTANT_RE.match(line)
        if constant_match:
            constants.append(
                {
                    "type": constant_match.group("type").split("[", 1)[0],
                    "name": constant_match.group("name"),
                    "value": int(constant_match.group("value"), 0),
                    "line": line_no,
                    "comment": comment,
                }
            )
            continue
        match = FIELD_RE.match(line)
        if not match:
            continue
        field_type = match.group("type").split("[", 1)[0]
        field_name = match.group("name")
        if field_name.isupper() or "=" in line:
            continue
        fields.append((field_type, field_name))
        field_meta[field_name] = {
            "type": field_type,
            "line": line_no,
            "comment": comment,
        }
    return {"fields": fields, "field_meta": field_meta, "constants": constants, "topics": topics}


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


def _message_enum_fields(
    message_name: str,
    messages: dict[str, dict[str, Any]],
    *,
    seen: set[str],
) -> dict[str, list[dict[str, Any]]]:
    if message_name in seen:
        return {}
    seen = {*seen, message_name}
    message = messages.get(message_name)
    if not message:
        return {}

    enum_fields: dict[str, list[dict[str, Any]]] = {}
    for field_type, field_name in message["fields"]:
        if field_type in messages and field_type not in PRIMITIVE_TYPES:
            for nested_field, constants in _message_enum_fields(field_type, messages, seen=seen).items():
                enum_fields[f"{field_name}.{nested_field}"] = constants
            continue

        constants = _constants_for_field(field_name, message)
        if constants:
            enum_fields[field_name] = constants
    return enum_fields


def _constants_for_field(field_name: str, message: dict[str, Any]) -> list[dict[str, Any]]:
    constants = message["constants"]
    if not constants:
        return []

    prefixes = _enum_prefixes_for_field(field_name, message["field_meta"].get(field_name, {}))
    matches = [
        constant for constant in constants
        if any(_constant_has_prefix(constant["name"], prefix) for prefix in prefixes)
    ]
    return matches


def _enum_prefixes_for_field(field_name: str, meta: dict[str, Any]) -> list[str]:
    prefixes = [_normalize_enum_label(field_name)]
    comment = str(meta.get("comment") or "")
    prefixes.extend(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", comment))
    return dedupe_keep_order(prefixes)


def _constant_has_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}_")


def _enum_aliases(constants: list[dict[str, Any]]) -> dict[str, int]:
    aliases: dict[str, int] = {}
    full_names = [constant["name"] for constant in constants]
    common_prefix = _common_enum_prefix(full_names)
    for constant in constants:
        aliases[constant["name"]] = constant["value"]

    candidate_aliases: dict[str, list[int]] = {}
    for constant in constants:
        alias = _short_enum_alias(constant["name"], common_prefix)
        if alias:
            candidate_aliases.setdefault(alias, []).append(constant["value"])
    for alias, values in candidate_aliases.items():
        unique_values = set(values)
        if len(values) == 1 and len(unique_values) == 1:
            aliases[alias] = values[0]
    return aliases


def _common_enum_prefix(names: list[str]) -> str:
    if not names:
        return ""
    token_lists = [name.split("_") for name in names]
    prefix: list[str] = []
    for tokens in zip(*token_lists):
        if len(set(tokens)) != 1:
            break
        prefix.append(tokens[0])
    return "_".join(prefix)


def _short_enum_alias(name: str, common_prefix: str) -> str:
    prefix = f"{common_prefix}_"
    if common_prefix and name.startswith(prefix):
        return name[len(prefix):]
    return ""


def _normalize_enum_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _camel_to_snake(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()
