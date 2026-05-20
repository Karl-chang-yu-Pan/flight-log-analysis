from __future__ import annotations

"""
Mechanism cache support for the PX4 flight-log analysis runner.

This module intentionally knows nothing about a specific ULog. It stores and
validates reusable PX4 source-code mechanisms only:

    normalized question intent -> mechanism file retrieval
    mechanism file -> source footprint validation against a PX4 checkout
    resolver output -> mechanism cache write

Per-log applicability and log-signature verification should stay in the runner.
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


MECHANISM_CACHE_SCHEMA_VERSION = 1

SourceValidationStatus = Literal[
    "exact_git_match",
    "source_footprint_match",
    "function_match_file_changed",
    "snippet_match_function_changed",
    "source_changed_revalidate_required",
    "source_ref_missing",
    "source_unavailable_revalidation_required",
]


class MechanismCacheConfig(BaseModel):
    cache_root: Path = Path(".flightlog_cache/mechanisms")
    schema_version: int = MECHANISM_CACHE_SCHEMA_VERSION
    max_snippet_chars: int = 4000


class SourceIdentity(BaseModel):
    px4_git_hash: Optional[str] = None
    px4_version: Optional[str] = None
    px4_tag: Optional[str] = None


class MechanismSourceRef(BaseModel):
    file: str
    function: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    snippet: Optional[str] = None
    explanation: str = ""

    file_sha256: Optional[str] = None
    function_sha256: Optional[str] = None
    snippet_sha256: Optional[str] = None
    normalized_snippet: Optional[str] = None


class MechanismRecord(BaseModel):
    schema_version: int = MECHANISM_CACHE_SCHEMA_VERSION
    mechanism_id: str
    name: str
    summary: str
    vehicle_control_domain: str
    source_identity: SourceIdentity
    source_refs: list[MechanismSourceRef] = Field(default_factory=list)

    question_intents: list[str] = Field(default_factory=list)
    source_queries: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)

    # This is the runner's source-level MechanismCandidate payload. It is kept
    # intact so the runner can reconstruct the original Pydantic model without
    # asking the resolver agent again.
    candidate_payload: dict[str, Any]

    created_at_unix: float = Field(default_factory=time.time)
    updated_at_unix: float = Field(default_factory=time.time)


class MechanismRetrievalResult(BaseModel):
    records: list[MechanismRecord] = Field(default_factory=list)
    scanned_files: int = 0
    selected_files: list[str] = Field(default_factory=list)
    rejected_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SourceRefValidation(BaseModel):
    file: str
    function: Optional[str] = None
    status: SourceValidationStatus
    usable: bool
    reason: str
    current_file_sha256: Optional[str] = None
    current_function_sha256: Optional[str] = None
    current_snippet_sha256: Optional[str] = None


class MechanismSourceValidation(BaseModel):
    mechanism_id: str
    status: SourceValidationStatus
    usable: bool
    reason: str
    checked_refs: list[SourceRefValidation] = Field(default_factory=list)


class MechanismRetriever:
    """Searches mechanism files using only source-intent / airframe context."""

    def __init__(self, config: MechanismCacheConfig | str | Path | None = None):
        if config is None:
            self.config = MechanismCacheConfig()
        elif isinstance(config, MechanismCacheConfig):
            self.config = config
        else:
            self.config = MechanismCacheConfig(cache_root=Path(config))

    def retrieve(self, source_search_context: Any, max_records: int = 8) -> MechanismRetrievalResult:
        context = _to_plain_dict(source_search_context)
        cache_root = self.config.cache_root
        result = MechanismRetrievalResult()

        if not cache_root.exists():
            result.warnings.append(f"Mechanism cache root does not exist: {cache_root}")
            return result

        scored: list[tuple[int, Path, MechanismRecord]] = []
        for path in sorted(cache_root.rglob("*.json")):
            result.scanned_files += 1
            try:
                record = MechanismRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                result.rejected_files.append(str(path))
                result.warnings.append(f"Failed to load mechanism file {path}: {exc!r}")
                continue

            if record.schema_version != self.config.schema_version:
                result.rejected_files.append(str(path))
                continue

            score = self._score(record, context)
            if score <= 0:
                result.rejected_files.append(str(path))
                continue
            scored.append((score, path, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        for _score, path, record in scored[:max_records]:
            result.records.append(record)
            result.selected_files.append(str(path))

        return result

    def _score(self, record: MechanismRecord, context: dict[str, Any]) -> int:
        airframe = context.get("airframe") or {}
        intent = context.get("question_intent") or {}

        query_text_parts = [
            intent.get("original_question"),
            intent.get("problem_domain"),
            intent.get("concise_intent"),
            " ".join(intent.get("source_queries") or []),
            " ".join(intent.get("likely_modules") or []),
            " ".join(intent.get("likely_source_files") or []),
        ]
        query_text = " ".join(str(x) for x in query_text_parts if x)
        query_tokens = set(_tokenize(query_text))

        record_text_parts = [
            record.mechanism_id,
            record.name,
            record.summary,
            record.vehicle_control_domain,
            " ".join(record.question_intents),
            " ".join(record.source_queries),
            " ".join(record.search_terms),
            " ".join(ref.file for ref in record.source_refs),
            " ".join(ref.function or "" for ref in record.source_refs),
        ]
        record_text = " ".join(str(x) for x in record_text_parts if x)
        record_tokens = set(_tokenize(record_text))

        score = 0
        overlap = query_tokens & record_tokens
        score += min(len(overlap), 20)

        vehicle_type = str(airframe.get("vehicle_type") or "").lower()
        problem_domain = str(intent.get("problem_domain") or "").lower()
        domain = record.vehicle_control_domain.lower()
        if vehicle_type and vehicle_type in domain:
            score += 8
        if problem_domain and (problem_domain in domain or domain in problem_domain):
            score += 8

        current_git = str(airframe.get("px4_git_hash") or "")
        cached_git = str(record.source_identity.px4_git_hash or "")
        if current_git and cached_git and current_git == cached_git:
            score += 10

        likely_files = {str(x) for x in intent.get("likely_source_files") or []}
        if likely_files:
            for ref in record.source_refs:
                if ref.file in likely_files or any(ref.file.endswith(f) or f.endswith(ref.file) for f in likely_files):
                    score += 5

        # Phrase matches are useful for PX4 symbols that tokenization splits poorly.
        lowered_query = query_text.lower()
        for term in record.search_terms:
            if term and term.lower() in lowered_query:
                score += 4

        return score


class MechanismSourceValidator:
    """Validates whether cached mechanism source refs still match a PX4 checkout."""

    def __init__(self, source_path: Optional[str | Path], current_git_hash: Optional[str] = None):
        self.source_path = Path(source_path) if source_path else None
        self.current_git_hash = current_git_hash

    def validate_record(self, record: MechanismRecord) -> MechanismSourceValidation:
        cached_git = record.source_identity.px4_git_hash
        if self.current_git_hash and cached_git and self.current_git_hash == cached_git:
            return MechanismSourceValidation(
                mechanism_id=record.mechanism_id,
                status="exact_git_match",
                usable=True,
                reason="Current PX4 git hash exactly matches the mechanism source identity.",
                checked_refs=[],
            )

        if self.source_path is None:
            return MechanismSourceValidation(
                mechanism_id=record.mechanism_id,
                status="source_unavailable_revalidation_required",
                usable=False,
                reason="PX4 git hash did not exactly match and no source_path was provided for source-footprint comparison.",
                checked_refs=[],
            )

        checked = [self._validate_ref(ref) for ref in record.source_refs]
        if not checked:
            return MechanismSourceValidation(
                mechanism_id=record.mechanism_id,
                status="source_changed_revalidate_required",
                usable=False,
                reason="Mechanism has no source references to validate.",
                checked_refs=[],
            )

        if any(not item.usable for item in checked):
            worst = _worst_status([item.status for item in checked])
            return MechanismSourceValidation(
                mechanism_id=record.mechanism_id,
                status=worst,
                usable=False,
                reason="At least one mechanism source reference changed or is missing; source re-resolution is required.",
                checked_refs=checked,
            )

        status = _worst_status([item.status for item in checked])
        return MechanismSourceValidation(
            mechanism_id=record.mechanism_id,
            status=status,
            usable=True,
            reason="All mechanism source references are still valid for the current PX4 source tree.",
            checked_refs=checked,
        )

    def _validate_ref(self, ref: MechanismSourceRef) -> SourceRefValidation:
        assert self.source_path is not None
        path = (self.source_path / ref.file).resolve()
        try:
            if not path.exists() or not path.is_file():
                return SourceRefValidation(
                    file=ref.file,
                    function=ref.function,
                    status="source_ref_missing",
                    usable=False,
                    reason="Referenced source file is missing in the current PX4 source tree.",
                )

            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return SourceRefValidation(
                file=ref.file,
                function=ref.function,
                status="source_ref_missing",
                usable=False,
                reason=f"Failed to read referenced source file: {exc!r}",
            )

        current_file_hash = _sha256_text(text)
        if ref.file_sha256 and current_file_hash == ref.file_sha256:
            return SourceRefValidation(
                file=ref.file,
                function=ref.function,
                status="source_footprint_match",
                usable=True,
                reason="Referenced source file hash is unchanged.",
                current_file_sha256=current_file_hash,
            )

        function_text = _extract_function_body(text, ref.function) if ref.function else None
        current_function_hash = _sha256_normalized(function_text) if function_text else None
        if ref.function_sha256 and current_function_hash == ref.function_sha256:
            return SourceRefValidation(
                file=ref.file,
                function=ref.function,
                status="function_match_file_changed",
                usable=True,
                reason="File changed, but referenced function body is unchanged after normalization.",
                current_file_sha256=current_file_hash,
                current_function_sha256=current_function_hash,
            )

        snippet_source = None
        if ref.start_line and ref.end_line:
            snippet_source = _line_window(text, ref.start_line, ref.end_line)
        current_snippet_hash = _sha256_normalized(snippet_source) if snippet_source else None
        if ref.snippet_sha256 and current_snippet_hash == ref.snippet_sha256:
            return SourceRefValidation(
                file=ref.file,
                function=ref.function,
                status="snippet_match_function_changed",
                usable=True,
                reason="Function/file changed, but the original referenced line window is unchanged after normalization.",
                current_file_sha256=current_file_hash,
                current_function_sha256=current_function_hash,
                current_snippet_sha256=current_snippet_hash,
            )

        cached_snippet = ref.normalized_snippet or (_normalize_cpp_like_text(ref.snippet) if ref.snippet else None)
        searchable_text = _normalize_cpp_like_text(function_text or text)
        if cached_snippet and cached_snippet in searchable_text:
            return SourceRefValidation(
                file=ref.file,
                function=ref.function,
                status="snippet_match_function_changed",
                usable=True,
                reason="Cached mechanism snippet is still present in the current source footprint.",
                current_file_sha256=current_file_hash,
                current_function_sha256=current_function_hash,
                current_snippet_sha256=_sha256_text(cached_snippet),
            )

        return SourceRefValidation(
            file=ref.file,
            function=ref.function,
            status="source_changed_revalidate_required",
            usable=False,
            reason="Referenced source file exists, but neither file, function, nor snippet fingerprint matches.",
            current_file_sha256=current_file_hash,
            current_function_sha256=current_function_hash,
            current_snippet_sha256=current_snippet_hash,
        )


class MechanismCacheWriter:
    """Writes resolver-produced mechanism candidates into durable mechanism files."""

    def __init__(self, config: MechanismCacheConfig | str | Path | None = None):
        if config is None:
            self.config = MechanismCacheConfig()
        elif isinstance(config, MechanismCacheConfig):
            self.config = config
        else:
            self.config = MechanismCacheConfig(cache_root=Path(config))

    def write_candidate(
        self,
        candidate_payload: Any,
        airframe_context: Any,
        question_intent: Any,
        source_path: Optional[str | Path] = None,
        source_evidence: Optional[Any] = None,
    ) -> MechanismRecord:
        candidate = _to_plain_dict(candidate_payload)
        airframe = _to_plain_dict(airframe_context)
        intent = _to_plain_dict(question_intent)

        mechanism_id = _slugify(candidate.get("mechanism_id") or candidate.get("name") or candidate.get("summary") or "mechanism")
        now = time.time()

        source_identity = SourceIdentity(
            px4_git_hash=airframe.get("px4_git_hash"),
            px4_version=airframe.get("px4_version"),
            px4_tag=airframe.get("px4_tag"),
        )

        source_refs = [
            self._fingerprint_source_ref(ref, source_path)
            for ref in candidate.get("source_refs", [])
        ]

        vehicle_control_domain = str(
            intent.get("problem_domain")
            or airframe.get("vehicle_type")
            or "unknown"
        )

        search_terms = _derive_search_terms(candidate, intent, source_evidence)
        record = MechanismRecord(
            schema_version=self.config.schema_version,
            mechanism_id=mechanism_id,
            name=str(candidate.get("name") or mechanism_id),
            summary=str(candidate.get("summary") or ""),
            vehicle_control_domain=vehicle_control_domain,
            source_identity=source_identity,
            source_refs=source_refs,
            question_intents=[
                str(x) for x in [
                    intent.get("problem_domain"),
                    intent.get("concise_intent"),
                    *(intent.get("likely_modules") or []),
                ] if x
            ],
            source_queries=[str(x) for x in (intent.get("source_queries") or []) if x],
            search_terms=search_terms,
            candidate_payload=candidate,
            created_at_unix=now,
            updated_at_unix=now,
        )

        path = self._record_path(record)
        if path.exists():
            try:
                existing = MechanismRecord.model_validate_json(path.read_text(encoding="utf-8"))
                record.created_at_unix = existing.created_at_unix
            except Exception:
                pass
            record.updated_at_unix = now

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)
        return record

    def _record_path(self, record: MechanismRecord) -> Path:
        git = _slugify(record.source_identity.px4_git_hash or "unknown_git")[:24]
        domain = _slugify(record.vehicle_control_domain or "unknown_domain")
        return (
            self.config.cache_root
            / f"schema_v{self.config.schema_version}"
            / domain
            / record.mechanism_id
            / f"{git}.json"
        )

    def _fingerprint_source_ref(self, ref_payload: Any, source_path: Optional[str | Path]) -> MechanismSourceRef:
        ref = _to_plain_dict(ref_payload)
        source_ref = MechanismSourceRef(
            file=str(ref.get("file") or ""),
            function=ref.get("function"),
            start_line=_maybe_int(ref.get("start_line")),
            end_line=_maybe_int(ref.get("end_line")),
            snippet=ref.get("snippet"),
            explanation=str(ref.get("explanation") or ""),
        )

        if source_path is None or not source_ref.file:
            return source_ref

        path = Path(source_path) / source_ref.file
        if not path.exists() or not path.is_file():
            return source_ref

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return source_ref

        source_ref.file_sha256 = _sha256_text(text)

        function_text = _extract_function_body(text, source_ref.function) if source_ref.function else None
        if function_text:
            source_ref.function_sha256 = _sha256_normalized(function_text)

        snippet = source_ref.snippet
        if not snippet and source_ref.start_line and source_ref.end_line:
            snippet = _line_window(text, source_ref.start_line, source_ref.end_line)
        if snippet:
            snippet = snippet[: self.config.max_snippet_chars]
            source_ref.snippet = snippet
            source_ref.normalized_snippet = _normalize_cpp_like_text(snippet)
            source_ref.snippet_sha256 = _sha256_text(source_ref.normalized_snippet)

        return source_ref


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return dict(value)


def _derive_search_terms(candidate: dict[str, Any], intent: dict[str, Any], source_evidence: Optional[Any]) -> list[str]:
    terms: list[str] = []
    for key in ("name", "summary"):
        if candidate.get(key):
            terms.append(str(candidate[key]))
    for key in ("required_parameters", "required_signals", "vehicle_type_gates", "mode_state_gates"):
        terms.extend(str(x) for x in candidate.get(key, []) if x)
    terms.extend(str(x) for x in intent.get("source_queries", []) if x)
    terms.extend(str(x) for x in intent.get("likely_modules", []) if x)
    terms.extend(str(x) for x in intent.get("likely_source_files", []) if x)

    evidence = _to_plain_dict(source_evidence) if source_evidence is not None else {}
    for hit in evidence.get("hits", [])[:20]:
        hit_dict = _to_plain_dict(hit)
        if hit_dict.get("query"):
            terms.append(str(hit_dict["query"]))
        if hit_dict.get("file"):
            terms.append(str(hit_dict["file"]))

    return _dedupe_keep_order([t.strip() for t in terms if t and t.strip()])[:80]


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "mechanism"


def _tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_./:-]+", value.lower())
    return [t for t in tokens if len(t) >= 3]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sha256_normalized(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return _sha256_text(_normalize_cpp_like_text(text))


def _normalize_cpp_like_text(text: Optional[str]) -> str:
    if not text:
        return ""
    # Remove comments and collapse whitespace so line wrapping/comment-only edits
    # do not invalidate mechanism footprints.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _line_window(text: str, start_line: int, end_line: int) -> str:
    lines = text.splitlines()
    start = max(start_line - 1, 0)
    end = min(max(end_line, start_line), len(lines))
    return "\n".join(lines[start:end])


def _extract_function_body(text: str, function: Optional[str]) -> Optional[str]:
    if not function:
        return None

    function = function.strip()
    candidates = _function_search_terms(function)
    for candidate in candidates:
        match = _find_function_match(text, candidate)
        if match is None:
            continue
        opening = text.find("{", match.end())
        if opening < 0 or opening - match.end() > 2000:
            continue
        closing = _find_matching_brace(text, opening)
        if closing is None:
            continue
        return text[match.start() : closing + 1]
    return None


def _function_search_terms(function: str) -> list[str]:
    terms = [function]
    no_args = function.split("(", 1)[0].strip()
    if no_args and no_args not in terms:
        terms.append(no_args)
    leaf = no_args.split("::")[-1].strip()
    if leaf and leaf not in terms:
        terms.append(leaf)
    return _dedupe_keep_order([t for t in terms if t])


def _find_function_match(text: str, term: str) -> Optional[re.Match[str]]:
    if "::" in term:
        pattern = re.compile(re.escape(term) + r"\s*\(")
    else:
        pattern = re.compile(r"\b" + re.escape(term) + r"\s*\(")
    for match in pattern.finditer(text):
        # Skip obvious calls by requiring that the previous non-space character
        # is not one that commonly appears in expressions.
        prefix = text[max(0, match.start() - 200) : match.start()]
        prefix_tail = prefix.rstrip()[-1:] if prefix.rstrip() else ""
        if prefix_tail in ".>=" and "::" not in term:
            continue
        return match
    return None


def _find_matching_brace(text: str, opening_index: int) -> Optional[int]:
    depth = 0
    i = opening_index
    in_string: Optional[str] = None
    escaped = False
    while i < len(text):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
        else:
            if ch in ('"', "'"):
                in_string = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def _worst_status(statuses: list[SourceValidationStatus]) -> SourceValidationStatus:
    order: dict[str, int] = {
        "exact_git_match": 0,
        "source_footprint_match": 1,
        "function_match_file_changed": 2,
        "snippet_match_function_changed": 3,
        "source_changed_revalidate_required": 4,
        "source_ref_missing": 5,
        "source_unavailable_revalidation_required": 6,
    }
    if not statuses:
        return "source_changed_revalidate_required"
    return max(statuses, key=lambda s: order.get(s, 99))


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None

