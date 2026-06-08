from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from flight_log_agent.px4.source import read_source_file, search_source
from flight_log_agent.models import CodeRef, SourceEvidenceBundle, SourceHit, SourceSearchContext
from flight_log_agent.source_path import resolve_source_path


def bounded_source_search(
    source_path: Optional[Path],
    search_context: SourceSearchContext,
    max_hits_total: int = 40,
    max_hits_per_query: int = 8,
    max_snippet_chars: int = 1600,
) -> SourceEvidenceBundle:
    source_path = resolve_source_path(source_path)
    if source_path is None:
        return SourceEvidenceBundle(
            search_context=search_context,
            hits=[],
            warnings=["No PX4 source path provided."],
        )

    hits: list[SourceHit] = []
    warnings: list[str] = []

    queries = dedupe_keep_order(search_context.question_intent.source_queries)
    for query in queries:
        if len(hits) >= max_hits_total:
            break

        raw_results = search_source(source_path, query, max_results=max_hits_per_query)
        for item in flatten_source_results(query, raw_results):
            if len(hits) >= max_hits_total:
                break
            item.snippet = item.snippet[:max_snippet_chars]
            hits.append(item)

    read_snippets: list[CodeRef] = []
    for file in search_context.question_intent.likely_source_files[:6]:
        try:
            result = read_source_file(source_path, file, 1, 220)
            read_snippets.append(
                CodeRef(
                    file=file,
                    start_line=1,
                    end_line=220,
                    snippet=json.dumps(result, default=str)[:max_snippet_chars],
                    explanation="Bounded top-of-file/context read from likely source file.",
                )
            )
        except Exception as exc:
            warnings.append(f"Failed to read likely source file {file}: {exc!r}")

    return SourceEvidenceBundle(
        search_context=search_context,
        hits=hits,
        read_snippets=read_snippets,
        warnings=warnings,
    )


def flatten_source_results(query: str, raw_results: Any) -> list[SourceHit]:
    if not isinstance(raw_results, list):
        return [SourceHit(query=query, file="unknown", line=None, snippet=str(raw_results))]

    hits: list[SourceHit] = []
    for item in raw_results:
        if isinstance(item, dict):
            file = str(item.get("file") or item.get("path") or "unknown")
            line = _maybe_int(item.get("line") or item.get("line_number"))
            snippet = str(item.get("snippet") or item.get("text") or item.get("match") or item)
        else:
            file = "unknown"
            line = None
            snippet = str(item)
        hits.append(SourceHit(query=query, file=file, line=line, snippet=snippet))
    return hits


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
