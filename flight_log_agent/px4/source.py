from __future__ import annotations

from pathlib import Path

from flight_log_agent.px4.source_snapshot import DirectorySource, SourceInput, UnavailableSource, source_handle


DEFAULT_READ_LINE_LIMIT = 200


def search_source(source_path: SourceInput, query: str, max_results: int = 8) -> list[dict]:
    try:
        source = source_handle(source_path)
        if source is None or isinstance(source, UnavailableSource):
            return [{"error": "PX4 source is unavailable."}]
        matches = source.search(query)[:max_results]
    except Exception as exc:
        return [{"error": str(exc)}]
    return [{
        "query": query,
        "matches": [f"{match.file}:{match.line}:{match.text}" for match in matches],
    }]


def read_source_file(
    source_path: SourceInput,
    relative_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict:
    if start_line < 1:
        return {"error": "start_line must be >= 1."}

    if end_line is not None and end_line < start_line:
        return {"error": "end_line must be >= start_line."}

    try:
        source = source_handle(source_path)
        if source is None or isinstance(source, UnavailableSource):
            return {"error": "PX4 source is unavailable."}
        requested_path = Path(relative_path)
        if requested_path.is_absolute() and isinstance(source, DirectorySource):
            try:
                normalized_relative_path = requested_path.resolve().relative_to(source.root).as_posix()
            except ValueError:
                return {"error": "relative_path escapes the PX4 source path."}
        else:
            normalized_relative_path = str(relative_path).replace("\\", "/")
        text = source.read_text(normalized_relative_path)
    except Exception as exc:
        if "escapes repository" in str(exc):
            return {"error": "relative_path escapes the PX4 source path."}
        return {"error": str(exc)}
    requested_end_line = end_line or start_line + DEFAULT_READ_LINE_LIMIT - 1
    capped_end_line = min(requested_end_line, start_line + DEFAULT_READ_LINE_LIMIT - 1)
    file_lines = text.splitlines()
    selected_lines = [
        {
            "line": line_number,
            "text": text,
        }
        for line_number, text in enumerate(file_lines, start=1)
        if start_line <= line_number <= capped_end_line
    ]

    return {
        "file": normalized_relative_path,
        "path": (
            str((source.root / normalized_relative_path).resolve())
            if isinstance(source, DirectorySource)
            else f"{source.identity}:{normalized_relative_path}"
        ),
        "start_line": start_line,
        "end_line": min(capped_end_line, len(file_lines)),
        "total_lines": len(file_lines),
        "truncated": capped_end_line < requested_end_line or capped_end_line < len(file_lines),
        "lines": selected_lines,
    }
