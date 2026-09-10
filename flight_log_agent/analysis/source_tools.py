from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

from agents import FunctionTool, function_tool

from flight_log_agent.px4.source import read_source_file
from flight_log_agent.px4.source_snapshot import SourceSnapshot


MAX_SOURCE_EXCERPT_CHARS = 16_000
DUPLICATE_SOURCE_EXCERPT_MESSAGE = (
    "Duplicate immutable snapshot source excerpt suppressed; reuse the "
    "earlier result for this exact range."
)


def make_read_px4_source_file_tool(
    resolve_source: Callable[[str], SourceSnapshot],
) -> FunctionTool:
    """Build one run-scoped, commit-pinned source excerpt tool."""

    active_or_completed_reads: set[tuple[str, str, int, int]] = set()
    reads_lock = threading.Lock()

    @function_tool(name_override="read_px4_source_file")
    async def read_px4_source_file(
        repository: str,
        path: str,
        start_line: int,
        end_line: int,
    ) -> str:
        """Read a bounded line range from the exact PX4 source snapshot.

        Use this after a commit-pinned Git search identifies a relevant file
        and line. The result contains numbered source text and continuation
        metadata. At most 200 lines and 16000 source characters are returned.

        Args:
            repository: Virtual Git directory, such as PX4-Autopilot or an
                initialized submodule beneath it.
            path: File path relative to that virtual Git directory.
            start_line: First one-based source line to return, inclusive.
            end_line: Last one-based source line requested, inclusive.
        """

        cache_key = (repository, path, start_line, end_line)
        with reads_lock:
            if cache_key in active_or_completed_reads:
                return _compact_json(
                    {
                        "duplicate": True,
                        "message": DUPLICATE_SOURCE_EXCERPT_MESSAGE,
                        "file": f"{repository}:{path}",
                        "requested_range": [start_line, end_line],
                    }
                )
            active_or_completed_reads.add(cache_key)

        succeeded = False
        try:
            raw_result = _read_source_lines(
                resolve_source,
                repository,
                path,
                start_line,
                end_line,
            )
            if "error" in raw_result:
                return _compact_json({"error": raw_result["error"]})

            result = _compact_source_excerpt(
                raw_result,
                repository=repository,
                requested_end_line=end_line,
            )
            succeeded = "error" not in result
            return _compact_json(result)
        except Exception as exc:
            return _compact_json(
                {
                    "error": str(exc) or type(exc).__name__,
                }
            )
        finally:
            if not succeeded:
                with reads_lock:
                    active_or_completed_reads.discard(cache_key)

    return read_px4_source_file


def _read_source_lines(
    resolve_source: Callable[[str], SourceSnapshot],
    repository: str,
    path: str,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    source = resolve_source(repository)
    return read_source_file(
        source,
        path,
        start_line=start_line,
        end_line=end_line,
    )


def _compact_source_excerpt(
    raw_result: dict[str, Any],
    *,
    repository: str,
    requested_end_line: int,
) -> dict[str, Any]:
    total_lines = int(raw_result.get("total_lines") or 0)
    raw_lines = list(raw_result.get("lines") or [])
    if (
        not raw_lines
        and total_lines
        and raw_result.get("start_line", 1) > total_lines
    ):
        return {
            "error": (
                f"start_line {raw_result['start_line']} exceeds the file's "
                f"{total_lines} lines."
            ),
            "file": f"{repository}:{raw_result['file']}",
            "total_lines": total_lines,
        }

    rendered_lines: list[str] = []
    rendered_chars = 0
    next_start_line: int | None = None
    truncated_line: int | None = None

    for item in raw_lines:
        line_number = int(item["line"])
        rendered = f"{line_number}: {item['text']}"
        separator_chars = 1 if rendered_lines else 0
        if rendered_chars + separator_chars + len(rendered) <= MAX_SOURCE_EXCERPT_CHARS:
            rendered_lines.append(rendered)
            rendered_chars += separator_chars + len(rendered)
            continue

        if not rendered_lines:
            prefix = f"{line_number}: "
            marker = " [line truncated]"
            available = max(
                0,
                MAX_SOURCE_EXCERPT_CHARS - len(prefix) - len(marker),
            )
            rendered_lines.append(f"{prefix}{str(item['text'])[:available]}{marker}")
            truncated_line = line_number
        else:
            next_start_line = line_number
        break

    returned_range: list[int] | None = None
    if rendered_lines:
        first_line = int(raw_lines[0]["line"])
        last_line = (
            truncated_line
            if truncated_line is not None
            else int(raw_lines[len(rendered_lines) - 1]["line"])
        )
        returned_range = [first_line, last_line]
        requested_last_available = min(requested_end_line, total_lines)
        if next_start_line is None and last_line < requested_last_available:
            next_start_line = last_line + 1

    result: dict[str, Any] = {
        "file": f"{repository}:{raw_result['file']}",
        "range": returned_range,
        "total_lines": total_lines,
        "next_start_line": next_start_line,
        "text": "\n".join(rendered_lines),
    }
    if truncated_line is not None:
        result["truncated_line"] = truncated_line
    return result


def _compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
