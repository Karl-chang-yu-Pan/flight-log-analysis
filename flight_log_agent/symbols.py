from __future__ import annotations

import re
from typing import Optional


_BRACKET_INDEX_RE = re.compile(r"\[[^\]]+\]")

_TOPIC_RE = r"[a-z][a-z0-9_]*"
_INSTANCE_RE = r"(?:\[\d+\])?"
_FIELD_PART_STRICT_RE = r"[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?"
_FIELD_PART_PERMISSIVE_RE = r"_?[A-Za-z][A-Za-z0-9_]*(?:\[\d+\])?"

_STRICT_SIGNAL_RE = re.compile(
    rf"{_TOPIC_RE}{_INSTANCE_RE}\.{_FIELD_PART_STRICT_RE}(?:\.{_FIELD_PART_STRICT_RE})*"
)
_PERMISSIVE_SIGNAL_RE = re.compile(
    rf"_?[A-Za-z][A-Za-z0-9_]*(?:(?:\.|->){_FIELD_PART_PERMISSIVE_RE})+"
)
_PARSE_SIGNAL_RE = re.compile(
    rf"(?P<topic>{_TOPIC_RE})(?:\[(?P<instance>\d+)\])?\.(?P<field>.+)"
)
_ENUM_CONSTANT_LEAF_RE = re.compile(r"[A-Z][A-Z0-9_]+")


def normalize_symbol(value: str) -> str:
    """Canonicalize a PX4 source or log symbol for matching.

    Rules:
      - ``->`` and ``::`` collapse to ``.``
      - whitespace and outer ``&``/``*`` are stripped
      - bracket indices like ``[0]`` are removed (lossy fuzzy match)
      - a single leading ``_`` (PX4 private-member convention) is stripped
        from the start of the whole string; nested underscores are preserved
    """
    normalized = str(value or "").strip().replace("->", ".").replace("::", ".").replace(" ", "").strip("&*")
    normalized = _BRACKET_INDEX_RE.sub("", normalized)
    if normalized.startswith("_"):
        normalized = normalized[1:]
    return normalized


def _topic_is_struct_type(reference: str) -> bool:
    """PX4 struct type names end in ``_s``; topic names do not.

    Rejects e.g. ``vehicle_status_s.VEHICLE_TYPE_X`` so C++ struct-constant
    references aren't misclassified as logged signals.
    """
    topic = reference.split(".", 1)[0].split("[", 1)[0]
    return topic.endswith("_s")


def is_signal_reference(value: str) -> bool:
    """Strict canonical-form check: ``topic[.instance]?.field[.subfield]*``."""
    if not isinstance(value, str) or not _STRICT_SIGNAL_RE.fullmatch(value):
        return False
    return not _topic_is_struct_type(value)


def looks_like_signal_reference(value: str) -> bool:
    """Permissive check for a pre-normalization reference (allows ``->`` and a leading ``_``)."""
    if not isinstance(value, str) or not _PERMISSIVE_SIGNAL_RE.fullmatch(value):
        return False
    return not _topic_is_struct_type(value.replace("->", "."))


def parse_signal_reference(reference: str) -> Optional[tuple[str, Optional[int], str]]:
    """Parse a canonical ``topic[.instance]?.field`` reference into its parts."""
    match = _PARSE_SIGNAL_RE.fullmatch(str(reference or "").strip())
    if match is None:
        return None
    topic = match.group("topic")
    if topic.endswith("_s"):
        return None
    instance_raw = match.group("instance")
    instance = int(instance_raw) if instance_raw is not None else None
    return topic, instance, match.group("field")


def looks_like_enum_constant(value: str) -> bool:
    """Leaf field matches ``[A-Z][A-Z0-9_]+`` (e.g. ``VEHICLE_TYPE_ROTARY_WING``)."""
    if not isinstance(value, str) or "." not in value:
        return False
    leaf = value.rsplit(".", 1)[-1]
    return bool(_ENUM_CONSTANT_LEAF_RE.fullmatch(leaf))


def parse_simple_signal(value: str) -> Optional[tuple[str, str]]:
    """Split a ``"topic.field"`` string into ``(topic, field)`` or return None.

    A stripped-down counterpart to :func:`parse_signal_reference` that
    doesn't try to extract a multi-instance index. Used by the ULog
    sample-reading path where signals are already in canonical form.
    """
    if not isinstance(value, str) or "." not in value:
        return None
    topic, field = value.split(".", 1)
    if not topic or not field:
        return None
    return topic, field
