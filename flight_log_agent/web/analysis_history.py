from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def resolve_history_directory(
    log_path: str | Path,
    *,
    upload_root: str | Path,
    fallback_root: str | Path,
    history_key: str | None = None,
) -> Path:
    """Return durable history storage without writing beside arbitrary local logs."""
    resolved_log = Path(log_path).expanduser().resolve()
    resolved_upload_root = Path(upload_root).expanduser().resolve()

    try:
        upload_relative = resolved_log.relative_to(resolved_upload_root)
    except ValueError:
        upload_relative = None

    if upload_relative is not None and len(upload_relative.parts) == 2:
        return resolved_log.parent / "analysis_history"

    identity = str(history_key or resolved_log)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return Path(fallback_root).expanduser().resolve() / digest


def write_history_record(
    history_dir: str | Path,
    record: Mapping[str, Any],
) -> Path:
    run_id = _validated_run_id(record.get("run_id"))
    directory = Path(history_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{run_id}.json"
    temporary = directory / f".{run_id}.{uuid.uuid4().hex}.tmp"
    serialized = json.dumps(dict(record), indent=2, sort_keys=True) + "\n"

    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination


def load_history_records(history_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(history_dir)
    if not directory.is_dir():
        return []

    records: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        try:
            _validated_run_id(record.get("run_id"))
        except ValueError:
            continue
        records.append(record)

    return sorted(
        records,
        key=lambda record: (
            _sortable_timestamp(record.get("created_at")),
            str(record.get("run_id") or ""),
        ),
    )


def _validated_run_id(value: Any) -> str:
    run_id = str(value or "")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid analysis run_id")
    return run_id


def _sortable_timestamp(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
