from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Optional, TypeVar


T = TypeVar("T")


def dedupe_keep_order(items: Iterable[T], *, drop_empty: bool = True) -> list[T]:
    """Return items with duplicates removed, preserving insertion order.

    By default empty strings and ``None`` are dropped; set ``drop_empty=False``
    to keep them.
    """
    seen: set[T] = set()
    result: list[T] = []
    for item in items:
        if drop_empty and not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def stable_id(prefix: str, value: Any) -> str:
    """Deterministic 12-hex-char identifier derived from ``value``."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def model_dump(value: Any) -> Any:
    """Serialize a Pydantic model or arbitrary value to a JSON-safe shape.

    Pydantic models are dumped with ``exclude_none=True``. Containers are
    walked recursively so nested models are also dumped consistently. Scalars
    pass through unchanged.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: model_dump(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [model_dump(item) for item in value]
    if isinstance(value, tuple):
        return tuple(model_dump(item) for item in value)
    return value


def copy_model(value: Any, *, update: dict[str, Any] | None = None) -> Any:
    """Return a copy of a Pydantic model with optional field updates."""
    update = update or {}
    if hasattr(value, "model_copy"):
        return value.model_copy(update=update)
    data = value.model_dump() if hasattr(value, "model_dump") else dict(vars(value))
    data.update(update)
    return value.__class__(**data)


def json_safe_value(value: Any) -> Any:
    """Decode bytes, unwrap numpy scalars, pass everything else through.

    PX4 ULogs frequently expose bytes (string fields, null-padded) and
    numpy scalars (from pyulog's pandas-style series). This helper makes
    both safe for JSON serialization and for downstream consumers that
    expect Python primitives.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if hasattr(value, "item"):
        return value.item()
    return value


def timestamp_to_seconds(timestamp: Any) -> float:
    """Convert a PX4 ULog microsecond timestamp to seconds (6 dp).

    Six decimal places matches microsecond precision and the existing
    inventory convention. Callers that want millisecond rounding wrap
    with ``round(..., 3)`` themselves; callers that want unrounded
    values can do ``float(json_safe_value(t)) / 1_000_000``.
    """
    return round(float(json_safe_value(timestamp)) / 1_000_000, 6)


def safe_float(value: Any) -> Optional[float]:
    """Coerce ``value`` to a finite ``float``; return None when impossible.

    Booleans and non-finite floats (NaN / +Inf / -Inf) are rejected so
    callers don't accidentally treat them as numeric. Numpy scalars get
    unwrapped via ``.item()`` before coercion.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> Optional[int]:
    """Coerce ``value`` to an ``int``; return None when impossible.

    Numpy scalars get unwrapped via ``.item()`` first. Floats coerce via
    Python's normal ``int(...)`` truncation. Strings are accepted as long
    as ``int(...)`` would have accepted them.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_number(value: Any) -> bool:
    """Whether ``value`` is a finite number per :func:`safe_float`."""
    return safe_float(value) is not None


def round_float(value: float) -> float:
    """Round a float to 6 decimal places, the metrics/plots convention."""
    return round(float(value), 6)


def compare(left: Any, op: str, right: Any) -> bool:
    """Apply a comparison operator to ``left`` and ``right``.

    Supports ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``. The generic
    Python operators apply, so numeric and string comparisons both work.
    Numeric-tolerance comparisons live closer to their call site
    (``signature_evaluator._compare_literal``) because the tolerance
    semantics are domain-specific.
    """
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    raise ValueError(f"unsupported operator: {op}")
