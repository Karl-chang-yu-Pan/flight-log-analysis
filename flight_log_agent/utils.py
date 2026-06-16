from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, TypeVar


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
