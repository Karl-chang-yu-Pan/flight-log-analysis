from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel

from flight_log_agent.px4.source_snapshot import SourceInput, SourceSnapshot, source_handle
from flight_log_agent.utils import dedupe_keep_order

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

_INTEGER_TYPES = {
    "byte", "char",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
}
_FLOAT_TYPES = {"float32", "float64"}

# ---------------------------------------------------------------------------
# Per-signal resampling / interpolation policy
#
# The interval evaluator that turns ULog samples into per-branch active
# windows needs to know how to fill a signal's value *between* its logged
# samples. That fill rule is derivable from the ``.msg`` field's declared
# type, its unit comment, and whether it owns enum constants — all of which
# the profiler already parses. This section is the single source of truth
# for that derivation; inspect :func:`derive_signal_policy` for the full
# taxonomy and the traps it guards against.
# ---------------------------------------------------------------------------

SignalInterpolation = Literal[
    "discrete_hold",     # never interpolate; hold the last sample (bool/enum/counter/id/flags/int)
    "linear",            # linear interpolation between samples (continuous physical scalars)
    "angle_wrap",        # unwrap -> linear -> rewrap to [-pi, pi] (absolute headings/euler angles in rad)
    "quaternion_slerp",  # spherical linear interpolation over a float32[4] quaternion
]


class SignalPolicy(BaseModel):
    """How a logged signal should be resampled between its native samples.

    ``method`` is the fill rule. ``nan_segmented`` is always True: a NaN
    endpoint is a semantic sentinel in PX4 (disarmed / unknown / hold-current
    / no-limit), so a NaN breaks the interpolation segment and is never
    bridged, regardless of ``method``. ``confidence`` flags fields that fell
    through to a default (unitless floats) so a caller can apply overrides.
    """

    method: SignalInterpolation
    unit: Optional[str] = None
    nan_segmented: bool = True
    confidence: Literal["high", "low"] = "high"
    reason: str = ""


# Field-name roots denoting an absolute orientation angle, cyclic on
# [-pi, pi]. Only fire angle_wrap when the unit is also radians, so angular
# *rates* (rad/s) and geodetic degrees (lat/lon) are excluded.
_ANGLE_NAME_ROOTS = ("heading", "yaw", "roll", "pitch", "cog_rad", "course", "bearing", "azimuth")

# Name shapes that force discrete hold regardless of numeric type: counters,
# identifiers, indices, and bitfields / flag words.
_DISCRETE_NAME_RE = re.compile(
    r"(?:_counter|_count|reset_count|_id|_ids|_index|_indices|_bitfield|_bitmask|_flags|_flag)$"
)

# Unit spellings observed across the PX4 msg tree, normalized to a canonical
# token. Angular rates and absolute angles are kept distinct — the classifier
# depends on that split (rad wraps, rad/s does not).
_UNIT_ALIASES = {
    "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m",
    "m/s": "m/s", "metres/sec": "m/s", "meters/sec": "m/s",
    "metres/second": "m/s", "meters/second": "m/s", "metres/s": "m/s", "meters/s": "m/s",
    "m/s^2": "m/s^2", "metres/sec^2": "m/s^2", "meters/sec^2": "m/s^2", "metres/second^2": "m/s^2",
    "rad": "rad", "radian": "rad", "radians": "rad",
    "rad/s": "rad/s", "rad/sec": "rad/s", "radians/s": "rad/s", "radians/sec": "rad/s",
    "deg/s": "deg/s", "deg per second": "deg/s", "degrees/s": "deg/s", "degrees per second": "deg/s",
    "deg": "deg", "degree": "deg", "degrees": "deg",
    "degc": "degC", "celsius": "degC",
    "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "us": "us", "microsecond": "us", "microseconds": "us", "ms": "ms", "milliseconds": "ms",
    "pa": "Pa", "hpa": "hPa", "a": "A", "v": "V", "w": "W", "rpm": "rpm", "hz": "Hz",
}
_RATE_UNITS = {"rad/s", "deg/s"}


def _normalize_unit_token(token: str) -> str:
    return re.sub(r"\s+", " ", token.strip().lower())


def _iter_unit_tokens(comment: str) -> Iterable[str]:
    """Yield bracketed then parenthesized comment groups (e.g. ``[m/s]``,
    ``(metres)``). Bracket style is unambiguous; paren style is noisier
    (prose like ``(negative altitude)``), so callers only accept a group
    that normalizes to a known unit."""
    for match in re.finditer(r"\[([^\]]+)\]", comment):
        yield match.group(1)
    for match in re.finditer(r"\(([^)]+)\)", comment):
        yield match.group(1)


def _parse_unit_kind(comment: str) -> tuple[Optional[str], str]:
    """Return ``(canonical_unit, kind)`` where kind is one of
    ``"angle"`` (absolute radians), ``"rate"`` (rad/s, deg/s),
    ``"physical"`` (any other recognized unit), or ``"none"``."""
    for token in _iter_unit_tokens(comment):
        canonical = _UNIT_ALIASES.get(_normalize_unit_token(token))
        if canonical is None:
            continue
        if canonical in _RATE_UNITS:
            return canonical, "rate"
        if canonical == "rad":
            return canonical, "angle"
        return canonical, "physical"
    return None, "none"


def _is_reset_delta(name: str, comment: str) -> bool:
    lowered = comment.lower()
    return name.startswith("delta_") or "amount by which" in lowered or "reset delta" in lowered


def _is_quaternion(base_type: str, name: str, comment: str, array_size: Optional[int]) -> bool:
    if base_type != "float32" or array_size != 4:
        return False
    lowered = comment.lower()
    return (
        name == "q"
        or name == "q_d"
        or name.startswith("q_")
        or name.endswith("_q")
        or "quaternion" in lowered
        or "quaterion" in lowered  # PX4 spells it this way in DistanceSensor.msg
    )


def _has_cyclic_hint(name: str, comment: str) -> bool:
    """True when the field looks cyclic: an explicit ``-PI..+PI`` range in the
    comment, or an angle-name root (heading / yaw / roll / pitch / ...)."""
    if re.search(r"[-+]?\s*PI\s*\.\.", comment) or re.search(r"[-+]\s*PI\b", comment):
        return True
    root = (name or "").lower()
    tokens = root.split("_")
    return any(root == a or root.startswith(a + "_") or a in tokens for a in _ANGLE_NAME_ROOTS)


def _is_absolute_angle(name: str, comment: str) -> bool:
    """True when the field is an absolute orientation angle that wraps on
    [-pi, pi].

    Requires a radian indication AND a cyclic hint. The radian check scans
    the whole comment (``\\brad\\b`` matches both ``[rad]`` and bare
    "in radians"), because PX4 writes the unit as prose on many setpoint
    ``yaw`` fields rather than in brackets. Angular *rates* (``rad/s``,
    ``deg/s``, "per second") are excluded first so a rate is never wrapped.
    """
    lowered = comment.lower()
    if re.search(r"rad(?:ians?)?\s*/\s*s|deg(?:rees?)?\s*/\s*s|per\s+second", lowered):
        return False
    has_radian = bool(re.search(r"\brad(?:ians?)?\b", lowered))
    return has_radian and _has_cyclic_hint(name, comment)


def derive_signal_policy(
    field_type: str,
    field_name: str,
    comment: str = "",
    *,
    is_enum: bool = False,
    array_size: Optional[int] = None,
) -> SignalPolicy:
    """Derive the resample/interpolation policy for one ``.msg`` field.

    Ordered decision tree (first match wins):

    1. ``bool`` -> discrete_hold.
    2. enum / bitfield (owns constants, or in enum registry, or a
       ``*_flags`` / ``*_bitfield`` name) -> discrete_hold.
    3. discrete name shape (``*_counter`` / ``reset_count*`` / ``*_id`` /
       ``*_index`` / ``*_flags``) -> discrete_hold.
    4. reset delta (``delta_*`` / "amount by which ... reset") -> discrete_hold
       (an impulse quantity, ~0 except at a reset event).
    5. ``float32[4]`` quaternion -> quaternion_slerp.
    6. any remaining integer type -> discrete_hold (continuous integers are
       vanishingly rare in PX4).
    7. float + absolute-radian unit + cyclic hint -> angle_wrap.
    8. float + any other recognized unit (incl. rad/s rate, geodetic degrees)
       -> linear.
    9. float with no recognized unit -> linear, confidence="low".

    Traps this guards against, all seen in the real msg tree:

    * counter "wrap" ("allow to wrap if count exceeds 255") is uint8 mod-256
      rollover, NOT angular wrap — angle_wrap fires only on float + radians.
    * ``rad/s`` / "deg per second" are angular *rates*: continuous, non-cyclic
      -> linear, never wrapped.
    * lat/lon in degrees are absolute geodetic -> linear, not +/-pi cyclic.
    * reset deltas (``delta_q_reset``, ``delta_heading``) are impulses -> hold,
      never slerp/interpolate.
    * NaN is a pervasive sentinel (disarmed / hold-current / unknown), so every
      policy is ``nan_segmented`` (the segment breaks at a NaN endpoint).
    """
    name = field_name or ""
    text = comment or ""

    if field_type == "bool":
        return SignalPolicy(method="discrete_hold", reason="boolean state")
    if is_enum:
        return SignalPolicy(method="discrete_hold", reason="enumerated / bitfield field")
    if _DISCRETE_NAME_RE.search(name):
        return SignalPolicy(method="discrete_hold", reason="counter / id / index / flags by name")
    if _is_reset_delta(name, text):
        return SignalPolicy(method="discrete_hold", reason="reset delta (impulse; hold between resets)")
    if _is_quaternion(field_type, name, text, array_size):
        return SignalPolicy(method="quaternion_slerp", unit="quaternion", reason="float32[4] quaternion")
    if field_type in _INTEGER_TYPES:
        return SignalPolicy(method="discrete_hold", reason="integer field (discrete)")
    if field_type in _FLOAT_TYPES:
        unit, kind = _parse_unit_kind(text)
        if _is_absolute_angle(name, text):
            return SignalPolicy(
                method="angle_wrap", unit=unit or "rad",
                reason="absolute angle in radians; wraps at +/-pi",
            )
        if kind != "none":
            return SignalPolicy(
                method="linear", unit=unit,
                reason=f"continuous quantity ({unit})",
            )
        return SignalPolicy(
            method="linear", unit=None, confidence="low",
            reason="float with no recognized unit; defaulted to linear",
        )
    return SignalPolicy(
        method="discrete_hold", confidence="low",
        reason=f"unclassified type {field_type}",
    )


def load_px4_msg_schema(source_path: SourceInput = None) -> dict[str, list[str]]:
    if isinstance(source_path, SourceSnapshot):
        return _load_px4_msg_schema_snapshot_cached(
            str(source_path.repository_path),
            source_path.commit_sha,
        )
    source = source_handle(source_path)
    if source is None:
        return {}
    return _build_schema(_load_messages(source))


def load_px4_msg_enum_registry(source_path: SourceInput = None) -> dict[str, dict[str, dict[str, int]]]:
    if isinstance(source_path, SourceSnapshot):
        return _load_px4_msg_enum_registry_snapshot_cached(
            str(source_path.repository_path),
            source_path.commit_sha,
        )
    source = source_handle(source_path)
    if source is None:
        return {}
    return _build_enum_registry(_load_messages(source))


def load_px4_signal_policies(source_path: SourceInput = None) -> dict[str, SignalPolicy]:
    """Return ``{topic.field: SignalPolicy}`` for every logged signal.

    Parallel to :func:`load_px4_msg_enum_registry`; the interval evaluator
    consults it to pick each signal's between-samples fill rule. Enum
    membership is taken from the enum registry so those fields classify as
    ``discrete_hold`` even when their name alone wouldn't reveal it.
    """
    if isinstance(source_path, SourceSnapshot):
        return _load_px4_signal_policies_snapshot_cached(
            str(source_path.repository_path),
            source_path.commit_sha,
        )
    source = source_handle(source_path)
    if source is None:
        return {}
    return _build_signal_policy_registry(_load_messages(source))


def signal_policy_for(signal: str, source_path: SourceInput = None) -> Optional[SignalPolicy]:
    """Return the :class:`SignalPolicy` for a ``topic.field`` signal, or None
    when the signal isn't in the schema."""
    return load_px4_signal_policies(source_path).get(signal)


def normalize_px4_enum_value(signal: str, value: Any, source_path: SourceInput = None) -> Any:
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


def is_valid_topic_field(signal: str, source_path: SourceInput = None) -> bool:
    if "." not in signal:
        return False
    topic, field = signal.split(".", 1)
    fields = load_px4_msg_schema(source_path).get(topic)
    return bool(fields and field in fields)


def resolve_topic_field(
    topic: Optional[str],
    field: str,
    source_path: SourceInput = None,
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


@lru_cache(maxsize=32)
def _load_px4_msg_schema_snapshot_cached(
    repository_path: str,
    commit_sha: str,
) -> dict[str, list[str]]:
    snapshot = SourceSnapshot(Path(repository_path), commit_sha)
    return _build_schema(_load_messages(snapshot))


@lru_cache(maxsize=32)
def _load_px4_msg_enum_registry_snapshot_cached(
    repository_path: str,
    commit_sha: str,
) -> dict[str, dict[str, dict[str, int]]]:
    snapshot = SourceSnapshot(Path(repository_path), commit_sha)
    return _build_enum_registry(_load_messages(snapshot))


@lru_cache(maxsize=32)
def _load_px4_signal_policies_snapshot_cached(
    repository_path: str,
    commit_sha: str,
) -> dict[str, SignalPolicy]:
    snapshot = SourceSnapshot(Path(repository_path), commit_sha)
    return _build_signal_policy_registry(_load_messages(snapshot))


def _load_messages(source: Any) -> dict[str, dict[str, Any]]:
    return {
        Path(path).stem: _parse_msg_text(source.read_text(path))
        for path in source.list_files("msg", patterns=["*.msg"])
    }


def _parse_msg_text(text: str) -> dict[str, Any]:
    fields: list[tuple[str, str]] = []
    field_meta: dict[str, dict[str, Any]] = {}
    constants: list[dict[str, Any]] = []
    topics: list[str] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
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
        raw_type = match.group("type")
        field_type = raw_type.split("[", 1)[0]
        arity_match = re.search(r"\[(\d+)\]", raw_type)
        array_size = int(arity_match.group(1)) if arity_match else None
        field_name = match.group("name")
        if field_name.isupper() or "=" in line:
            continue
        fields.append((field_type, field_name))
        field_meta[field_name] = {
            "type": field_type,
            "array_size": array_size,
            "line": line_no,
            "comment": comment,
        }
    return {"fields": fields, "field_meta": field_meta, "constants": constants, "topics": topics}


def _build_schema(messages: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    schema: dict[str, list[str]] = {}
    for message_name in messages:
        fields = sorted(_expand_message_fields(message_name, messages, seen=set()))
        topics = messages[message_name]["topics"] or [_camel_to_snake(message_name)]
        for topic in topics:
            schema[topic] = fields
    return schema


def _build_enum_registry(messages: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
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


def _build_signal_policy_registry(messages: dict[str, dict[str, Any]]) -> dict[str, SignalPolicy]:
    registry: dict[str, SignalPolicy] = {}
    for message_name, message in messages.items():
        topics = message["topics"] or [_camel_to_snake(message_name)]
        policies = _message_signal_policies(message_name, messages, seen=set())
        for topic in topics:
            for field_path, policy in policies.items():
                registry[f"{topic}.{field_path}"] = policy
    return registry


def _message_signal_policies(
    message_name: str,
    messages: dict[str, dict[str, Any]],
    *,
    seen: set[str],
) -> dict[str, SignalPolicy]:
    if message_name in seen:
        return {}
    seen = {*seen, message_name}
    message = messages.get(message_name)
    if not message:
        return {}

    # Direct (leaf) fields of THIS message that own enum constants — nested
    # message fields resolve their own enum status inside the recursion.
    enum_leaf_fields = {
        field_path
        for field_path in _message_enum_fields(message_name, messages, seen=set())
        if "." not in field_path
    }

    policies: dict[str, SignalPolicy] = {}
    for field_type, field_name in message["fields"]:
        if field_type in messages and field_type not in PRIMITIVE_TYPES:
            for nested_path, policy in _message_signal_policies(field_type, messages, seen=seen).items():
                policies[f"{field_name}.{nested_path}"] = policy
            continue
        meta = message["field_meta"].get(field_name, {})
        policies[field_name] = derive_signal_policy(
            field_type,
            field_name,
            str(meta.get("comment") or ""),
            is_enum=field_name in enum_leaf_fields,
            array_size=meta.get("array_size"),
        )
    return policies


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




def _camel_to_snake(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()
