from __future__ import annotations

from pathlib import Path
from typing import Optional


DEFAULT_PX4_SOURCE_PATH = Path(__file__).resolve().parents[1] / "ref" / "PX4-Autopilot"


def resolve_source_path(
    source_path: Optional[str | Path],
    *,
    default_source_path: Path | None = None,
) -> Optional[Path]:
    if source_path:
        return Path(source_path)
    default_source_path = default_source_path or DEFAULT_PX4_SOURCE_PATH
    if default_source_path.exists():
        return default_source_path
    return None
