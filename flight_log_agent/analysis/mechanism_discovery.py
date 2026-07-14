"""Deterministic mechanism discovery — the profiler→DAG production seam.

Stage 1 of the discovery rework (#73): adapt the profiler's per-file facts
(Layer 1 ``SourceFileFacts``) into the inputs :func:`build_mechanism_dag`
expects, and load those facts through the Layer 1 disk cache so repeated
discovery never re-parses an unchanged PX4 file.

The binding mapping was validated end-to-end against real PX4 v1.14.3
source (airspeed / NPFG / weathervane / terrain mechanisms) before being
promoted here from the test shim.

Later stages grow this module into the discovery fixpoint (seeds → search
→ rank → load → build DAG → resolve gaps → repeat) and the judge-LLM
checkpoints. No LLM code belongs in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from flight_log_agent.analysis.mechanism_dag import (
    MechanismDAG,
    _DAGBuilder,
    build_mechanism_dag,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.px4.source_facts_cache import (
    SourceFileFacts,
    get_or_extract_facts,
)
from flight_log_agent.symbols import (
    exact_symbol,
    strip_symbol_indices,
    symbol_produces_reference,
)
from flight_log_agent.utils import dedupe_keep_order


def _as_dict(ref: Any) -> dict[str, Any]:
    if isinstance(ref, dict):
        return ref
    if hasattr(ref, "model_dump"):
        return ref.model_dump(exclude_none=True)
    return dict(vars(ref))


def binding_from_assignment(assignment: Any) -> dict[str, Any]:
    """Convert one profiler ``SourceAssignmentRef`` into a DAG binding dict.

    The DAG builder reads ``target_symbol`` / ``source_symbol`` /
    ``assignment_path`` / ``logged_signal`` — not the profiler's flat
    ``target`` / ``expression`` / ``file`` / ``line``. A profiler-derived
    topic/field pair establishes declaration compatibility only. The
    aggregation pass fills ``logged_signal`` only after source structure
    proves that the target object crosses a publication boundary.
    """
    ref = _as_dict(assignment)
    topic = ref.get("target_topic")
    field_name = ref.get("target_field")
    return {
        "target_symbol": str(ref.get("target") or ""),
        "source_symbol": str(ref.get("expression") or ""),
        "function": str(ref.get("function") or ""),
        "callable_id": str(ref.get("callable_id") or ""),
        "function_parameters": list(ref.get("function_parameters") or []),
        "declaration_kind": str(ref.get("declaration_kind") or ""),
        "assignment_path": [
            {
                "file": str(ref.get("file") or ""),
                "line": int(ref.get("line") or 0),
                "expression": str(ref.get("expression") or ""),
            }
        ],
        # A struct type proves declaration compatibility, not publication.
        # ``dag_inputs_from_facts`` fills this only after a publish boundary
        # ties the target object to a topic.
        "declared_signal": f"{topic}.{field_name}" if topic and field_name else "",
        "logged_signal": "",
        "control_predicates": list(ref.get("control_predicates") or []),
        "control_predicate_lines": list(ref.get("control_predicate_lines") or []),
        "reachability_exact": bool(ref.get("reachability_exact", True)),
        "struct_variables": dict(ref.get("struct_variables") or {}),
        "symbol_bindings": dict(ref.get("symbol_bindings") or {}),
    }


@dataclass
class DAGInputs:
    """Aggregated, deduplicated ``build_mechanism_dag`` inputs.

    One instance per discovery round, built from the union of every
    loaded file's Layer 1 facts.
    """

    bindings: list[dict[str, Any]] = field(default_factory=list)
    helper_expressions: list[dict[str, Any]] = field(default_factory=list)
    parameter_predicates: list[dict[str, Any]] = field(default_factory=list)
    parameter_aliases: dict[str, str] = field(default_factory=dict)
    parameter_names: set[str] = field(default_factory=set)
    call_statements: list[dict[str, Any]] = field(default_factory=list)
    boundary_bindings: list[dict[str, Any]] = field(default_factory=list)


def dag_inputs_from_facts(facts: Iterable[Any]) -> DAGInputs:
    """Aggregate per-file facts into deduplicated DAG-builder inputs.

    Layer 1 entries are per-file, so duplicates only arise when the same
    file is passed twice or two files declare identical facts (headers);
    both are deduplicated on content keys. ``parameter_aliases`` maps the
    PX4 parameter *member* (``_param_wv_roll_min``) to its canonical name
    (``WV_ROLL_MIN``) from ``ParameterRef.member`` / ``.name``.
    """
    inputs = DAGInputs()
    seen_bindings: set[tuple[str, str, str, int]] = set()
    seen_helpers: set[tuple[str, str, int]] = set()
    seen_predicates: set[tuple[str, str, int]] = set()
    seen_calls: set[tuple[str, str, str, int]] = set()

    entries = [_as_dict(facts_entry) for facts_entry in facts]

    topic_refs_by_variable: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        for direction_key in ("subscribed_topics", "published_topics"):
            direction = "subscribe" if direction_key == "subscribed_topics" else "publish"
            for raw_ref in entry.get(direction_key) or []:
                ref = _as_dict(raw_ref)
                variable = str(ref.get("variable") or "")
                topic = str(ref.get("topic") or "")
                if not variable or not topic:
                    continue
                topic_refs_by_variable.setdefault(variable, []).append(
                    {
                        "source_symbol": variable,
                        "topic": topic,
                        "instance": ref.get("instance"),
                        "direction": direction,
                        "file": str(ref.get("file") or entry.get("file") or ""),
                        "function": "",
                        "callable_id": "",
                        "provenance": str(ref.get("api") or direction_key),
                    }
                )

    def unique_boundary(
        variable: str,
        direction: str,
        source_file: str,
    ) -> tuple[str, Any] | None:
        candidates = [
            item
            for item in topic_refs_by_variable.get(variable, [])
            if item.get("direction") == direction and item.get("topic")
        ]
        exact_file = [
            item for item in candidates if str(item.get("file") or "") == source_file
        ]
        if exact_file:
            candidates = exact_file
        elif source_file:
            source_directory = source_file.rpartition("/")[0]
            same_directory = [
                item
                for item in candidates
                if str(item.get("file") or "").rpartition("/")[0]
                == source_directory
            ]
            if same_directory:
                candidates = same_directory
        placements = {
            (str(item.get("topic") or ""), item.get("instance"))
            for item in candidates
        }
        return next(iter(placements)) if len(placements) == 1 else None

    inputs.boundary_bindings.extend(
        item
        for refs in topic_refs_by_variable.values()
        for item in refs
    )

    for entry in entries:
        for raw_call in entry.get("function_calls") or []:
            call = _as_dict(raw_call)
            name = str(call.get("name") or "").rsplit("::", 1)[-1]
            receiver = str(call.get("receiver") or "")
            args = [str(arg) for arg in (call.get("args") or [])]
            if not receiver or not args:
                continue
            direction = "subscribe" if name in {"copy", "update"} else (
                "publish" if name == "publish" else ""
            )
            if not direction:
                continue
            file = str(call.get("file") or entry.get("file") or "")
            placement = unique_boundary(receiver, direction, file)
            if not placement:
                continue
            topic, instance = placement
            source_symbol = args[0].lstrip("&*").strip()
            if not source_symbol:
                continue
            inputs.boundary_bindings.append(
                {
                    "source_symbol": source_symbol,
                    "topic": topic,
                    "instance": instance,
                    "direction": direction,
                    "file": file,
                    "function": str(call.get("function") or ""),
                    "callable_id": str(call.get("callable_id") or ""),
                    "provenance": f"{receiver}.{name}",
                }
            )

    publish_candidates: dict[tuple[str, str, str], set[tuple[str, Any]]] = {}
    for item in inputs.boundary_bindings:
        callable_id = str(item.get("callable_id") or "")
        if item.get("direction") != "publish" or not callable_id:
            continue
        key = (
            str(item.get("source_symbol") or ""),
            str(item.get("file") or ""),
            callable_id,
        )
        publish_candidates.setdefault(key, set()).add(
            (str(item.get("topic") or ""), item.get("instance"))
        )
    publish_roots = {
        key: next(iter(placements))
        for key, placements in publish_candidates.items()
        if len(placements) == 1
    }

    for entry in entries:

        for assignment in entry.get("source_assignments") or []:
            ref = _as_dict(assignment)
            key = (
                str(ref.get("target") or ""),
                str(ref.get("expression") or ""),
                str(ref.get("file") or ""),
                int(ref.get("line") or 0),
            )
            if key in seen_bindings:
                continue
            seen_bindings.add(key)
            binding = binding_from_assignment(ref)
            target = str(binding.get("target_symbol") or "")
            root, dot, field_path = target.replace("->", ".").partition(".")
            file = str(ref.get("file") or entry.get("file") or "")
            callable_id = str(ref.get("callable_id") or "")
            placement = publish_roots.get((root, file, callable_id))
            if placement and dot and field_path:
                topic, instance = placement
                topic_identity = (
                    f"{topic}[{instance}]" if instance is not None else topic
                )
                binding["logged_signal"] = f"{topic_identity}.{field_path}"
            inputs.bindings.append(binding)

        for helper in entry.get("helper_expressions") or []:
            helper_dict = _as_dict(helper)
            key = (
                str(helper_dict.get("name") or ""),
                str(helper_dict.get("file") or ""),
                int(helper_dict.get("line") or 0),
            )
            if key in seen_helpers:
                continue
            seen_helpers.add(key)
            inputs.helper_expressions.append(helper_dict)

        for predicate in entry.get("parameter_predicates") or []:
            predicate_dict = _as_dict(predicate)
            key = (
                str(predicate_dict.get("predicate") or ""),
                str(predicate_dict.get("file") or ""),
                int(predicate_dict.get("line") or 0),
            )
            if key in seen_predicates:
                continue
            seen_predicates.add(key)
            inputs.parameter_predicates.append(predicate_dict)

        for call in entry.get("function_calls") or []:
            call_dict = _as_dict(call)
            if not call_dict.get("args"):
                continue
            key = (
                str(call_dict.get("name") or ""),
                str(call_dict.get("receiver") or ""),
                str(call_dict.get("file") or ""),
                int(call_dict.get("line") or 0),
            )
            if key in seen_calls:
                continue
            seen_calls.add(key)
            inputs.call_statements.append(call_dict)

        for parameter in entry.get("referenced_parameters") or []:
            parameter_dict = _as_dict(parameter)
            name = parameter_dict.get("name")
            member = parameter_dict.get("member")
            if name:
                inputs.parameter_names.add(str(name))
                if member:
                    inputs.parameter_aliases.setdefault(str(member), str(name))

    return inputs


def load_facts(
    profiler: MechanismSourceProfiler,
    cache_root: Union[str, Path],
    files: Sequence[str],
    source_hash: str,
    *,
    source_root: Optional[Union[str, Path]] = None,
    git_path: str = "git",
) -> list[SourceFileFacts]:
    """Load Layer 1 facts for ``files``, extracting-and-caching on miss.

    Thin loop over :func:`get_or_extract_facts` — the production consumer
    the Layer 1 cache was built for. Duplicate paths in ``files`` load
    once; order is preserved.
    """
    facts: list[SourceFileFacts] = []
    seen: set[str] = set()
    for file_path in files:
        normalized = str(file_path)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        facts.append(
            get_or_extract_facts(
                profiler,
                cache_root,
                normalized,
                source_hash,
                source_root=source_root,
                git_path=git_path,
            )
        )
    return facts


# ---------------------------------------------------------------------------
# Terminal validation (Phase 0: trust the question and terminals)
# ---------------------------------------------------------------------------


def split_terminal_qualifier(terminal: str) -> tuple[str, str]:
    """Split ``Class::member`` into ``(qualifier, bare member)``.

    The qualifier is SCOPE, not spelling noise: validation uses it to
    pick the write file whose class family matches, instead of
    discarding it. The bare member is what writers key on at the
    assignment site. Member paths and indices are preserved."""
    text = str(terminal or "").strip()
    qualifier, sep, bare = text.rpartition("::")
    return (qualifier.strip() if sep else "", bare.strip())


def canonicalize_terminal(terminal: str) -> str:
    """The sliceable terminal form: the bare member as written at the
    assignment site. The class qualifier is handled separately as scope
    evidence by :func:`validate_terminal` — see
    :func:`split_terminal_qualifier`."""
    return split_terminal_qualifier(terminal)[1]


@dataclass
class TerminalValidation:
    """Deterministic verdict on one candidate terminal, produced BEFORE
    any DAG is built from it.

    ``status``:

    * ``valid`` — the terminal has write targets in the loaded facts
      (scoped to ``resolved_file`` when one could be determined) or is
      an exact member of the observed logged catalogue.
    * ``absent`` — no write target anywhere in the loaded facts and not
      a logged output.
    * ``absent_in_scope`` — write targets exist, but none in the
      declared terminal file's family or module.
    * ``ambiguous`` — write targets span several unrelated locations
      and no terminal file was declared to pick one; slicing would fuse
      foreign modules, so nothing is built.
    """

    terminal: str
    status: str
    logged: bool = False
    write_files: list[str] = field(default_factory=list)
    resolved_file: Optional[str] = None
    reason: Optional[str] = None


def validate_terminal(
    terminal: str,
    bindings: Iterable[dict[str, Any]],
    logged_signals: Iterable[str],
    terminal_file: Optional[str] = None,
) -> TerminalValidation:
    """Validate a candidate terminal against actual write targets and
    the observed catalogue — never by prompt trust or name shape.

    Matching mirrors the DAG builder's terminal lookup exactly (the
    EXACT-identity union of written targets and resolved logged signals,
    index-compatible spellings included), so a terminal validated here
    is one the slicer can act on. Scope rules mirror the walk's
    visibility conventions: a member-shaped root (leading or trailing
    underscore) is visible across its module directory, a local only
    within its file family — write targets spread wider than that
    without a declared terminal file are ambiguous, not sliceable. A
    ``Class::member`` qualifier is used as scope evidence: it selects
    the write file whose family matches the class name.
    """
    qualifier, canonical = split_terminal_qualifier(terminal)
    norm = exact_symbol(canonical)
    if not norm:
        return TerminalValidation(
            terminal=canonical, status="absent", reason="empty terminal"
        )

    logged = norm in {exact_symbol(str(s)) for s in logged_signals if s}

    shape = strip_symbol_indices(norm)
    matches: list[dict[str, Any]] = []
    for binding in bindings:
        target = exact_symbol(
            str(binding.get("target_symbol") or binding.get("target") or "")
        )
        published = exact_symbol(str(binding.get("logged_signal") or ""))
        for candidate in (target, published):
            if strip_symbol_indices(candidate) == shape and symbol_produces_reference(
                candidate, norm
            ):
                matches.append(binding)
                break

    writes_per_file: dict[str, int] = {}
    for binding in matches:
        file = _DAGBuilder._binding_first_file(binding)
        if file:
            writes_per_file[file] = writes_per_file.get(file, 0) + 1
    write_files = sorted(writes_per_file)

    if not matches:
        if logged:
            return TerminalValidation(
                terminal=canonical,
                status="valid",
                logged=True,
                reason="observed logged output; no publisher loaded yet",
            )
        return TerminalValidation(
            terminal=canonical,
            status="absent",
            reason="no write target in loaded facts",
        )

    def best_file(files: Iterable[str]) -> Optional[str]:
        ranked = sorted(set(files), key=lambda f: (-writes_per_file.get(f, 0), f))
        return ranked[0] if ranked else None

    if qualifier and not terminal_file:
        # The class qualifier is scope evidence: pick the write files
        # whose family stem matches the class name (same underscore/case
        # convention as receiver↔class affinity). No match falls through
        # to the unqualified rules — the qualifier could name a base
        # class whose file is not loaded yet.
        affinity = qualifier.rsplit("::", 1)[-1].replace("_", "").lower()
        class_files = [
            f
            for f in write_files
            if _DAGBuilder._file_family(f)[1].replace("_", "").lower() == affinity
        ]
        if class_files:
            return TerminalValidation(
                terminal=canonical,
                status="valid",
                logged=logged,
                write_files=write_files,
                resolved_file=best_file(class_files),
            )

    if terminal_file:
        if terminal_file in writes_per_file:
            return TerminalValidation(
                terminal=canonical,
                status="valid",
                logged=logged,
                write_files=write_files,
                resolved_file=terminal_file,
            )
        declared_family = _DAGBuilder._file_family(terminal_file)
        family_files = [
            f
            for f in write_files
            if _DAGBuilder._file_family(f) == declared_family
        ]
        if not family_files:
            # Member widening stays within the module (same directory),
            # matching the walk's visibility convention.
            root = canonical.split(".", 1)[0].split("->", 1)[0].strip().strip("&*")
            if root.startswith("_") or root.endswith("_"):
                family_files = [
                    f
                    for f in write_files
                    if _DAGBuilder._file_family(f)[0] == declared_family[0]
                ]
        if family_files:
            return TerminalValidation(
                terminal=canonical,
                status="valid",
                logged=logged,
                write_files=write_files,
                resolved_file=best_file(family_files),
            )
        return TerminalValidation(
            terminal=canonical,
            status="absent_in_scope",
            logged=logged,
            write_files=write_files,
            reason=(
                f"no write target in declared file {terminal_file};"
                f" written in: {', '.join(write_files[:4])}"
            ),
        )

    root = canonical.split(".", 1)[0].split("->", 1)[0].strip().strip("&*")
    if root.startswith("_") or root.endswith("_"):
        scopes = {_DAGBuilder._file_family(f)[0] for f in write_files}
        scope_kind = "modules"
    else:
        scopes = {_DAGBuilder._file_family(f) for f in write_files}
        scope_kind = "file families"
    if len(scopes) <= 1:
        return TerminalValidation(
            terminal=canonical,
            status="valid",
            logged=logged,
            write_files=write_files,
        )
    if logged:
        # An exact observed logged output is resolvable through the
        # catalogue alone; multiple publisher sites are legitimate.
        return TerminalValidation(
            terminal=canonical,
            status="valid",
            logged=True,
            write_files=write_files,
        )
    return TerminalValidation(
        terminal=canonical,
        status="ambiguous",
        logged=logged,
        write_files=write_files,
        reason=(
            f"write targets span {len(scopes)} {scope_kind} with no terminal"
            f" file declared: {', '.join(write_files[:4])}"
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic discovery fixpoint
# ---------------------------------------------------------------------------


def _definition_queries(symbol: str) -> list[str]:
    """Search queries likely to hit the file that *defines* ``symbol``.

    Unresolved entries can be dotted accessor chains
    (``_scale_check_groundspeed.isAllFinite``); the writable entity is the
    root, so query on an assignment-shaped root pattern. Empty for roots
    too short to search meaningfully (a two-letter root matches half the
    tree and only burns the round's file budget).
    """
    root = symbol.split(".", 1)[0].split("->", 1)[0].strip().strip("&*")
    queries: list[str] = []
    if len(root) >= 4:
        queries.append(f"{root} =")
    if "." in symbol or "->" in symbol:
        # Accessor gap (``_obj.getThing``): the definition lives under the
        # METHOD name, not the receiver — ``root =`` greps the wrong thing.
        tail = symbol.replace("->", ".").rsplit(".", 1)[-1].strip().strip("()")
        if len(tail) >= 4:
            queries.append(f"{tail}(")
    return queries


def _gap_definition_files(
    profiler: MechanismSourceProfiler,
    symbols: Iterable[str],
    *,
    max_files_per_gap: int = 2,
    max_matched_files: int = 10,
    max_total: int = 16,
) -> list[str]:
    """Candidate definition files for unresolved symbols, noise-guarded.

    One ranked search per gap so a single gap can never monopolize the
    round's file budget (``max_files_per_gap``). A query matching more
    than ``max_matched_files`` distinct files is *ungreppable* — a generic
    name like ``scale`` matches half the tree — and is dropped entirely:
    loading its top hits would pull unrelated modules whose bindings then
    collide with slice-local names (measured on RTL: rotation.h/Dual.hpp
    contributed ~⅓ of the DAG before this guard). Specificity is decided
    by measured hit count, not name shape.
    """
    files: list[str] = []
    for symbol in sorted({str(s) for s in symbols if s}):
        if len(files) >= max_total:
            break
        for query in _definition_queries(symbol):
            hits = profiler.search_related_source_files(
                [query], max_files=max_matched_files + 1
            )
            if len(hits) > max_matched_files:
                continue
            # The ranker penalizes test/vendored paths below zero but
            # still returns them when nothing else matches — a negative
            # score means "known junk", never load it.
            positive = [hit for hit in hits if hit.score > 0]
            files.extend(hit.file for hit in positive[:max_files_per_gap])
    return dedupe_keep_order(files)[:max_total]


def make_helper_body_provider(
    profiler: MechanismSourceProfiler,
    fetched_files: list[str],
    *,
    max_files_per_helper: int = 2,
) -> Callable[[str], Any]:
    """On-demand cross-file helper loader for :func:`build_mechanism_dag`.

    Searches for the callee's definition (PX4 methods define as
    ``Class::name(``; the bare ``name(`` form catches free functions),
    extracts helper records from the top hits, and appends those files to
    ``fetched_files`` so the discovery loop can load their full facts in
    the next round. The DAG builder memoizes probes, so an unknown name
    costs at most one search per build.

    Noise control differs from :func:`_gap_definition_files` on purpose:
    hits with a non-positive ranking score (test/vendored paths) are never
    extracted from, but there is NO hit-count ambiguity guard here — a
    real mechanism helper (``get_distance_to_next_waypoint``) is *called*
    from dozens of files, and extraction already filters to definitions
    of the requested name, so caller-heavy hits are harmless while the
    guard measurably severed the RTL→geo.cpp haversine subgraph.
    """

    def provider(helper_name: str) -> Any:
        hits = profiler.search_related_source_files(
            [f"::{helper_name}(", f"{helper_name}("],
            max_files=max_files_per_helper,
        )
        files = [hit.file for hit in hits if hit.score > 0]
        if not files:
            return []
        for file_path in files:
            if file_path not in fetched_files:
                fetched_files.append(file_path)
        return profiler.extract_helper_expressions_from_source(
            files, helper_names=[helper_name]
        )

    return provider


@dataclass
class DiscoveryRound:
    """Per-round trace of the fixpoint, kept for judge/debug consumption."""

    index: int
    new_files: list[str]
    unresolved_symbols: list[str]
    vertices: int
    edges: int


@dataclass
class DiscoveryResult:
    dag: Optional[MechanismDAG]
    inputs: DAGInputs
    files_loaded: list[str]
    rounds: list[DiscoveryRound]
    terminal_validation: Optional[TerminalValidation] = None


def discover_mechanism_dag(
    profiler: MechanismSourceProfiler,
    cache_root: Union[str, Path],
    seeds: Sequence[str],
    terminal: str,
    source_hash: str,
    *,
    source_root: Optional[Union[str, Path]] = None,
    terminal_file: Optional[str] = None,
    max_rounds: int = 3,
    max_files_per_round: int = 8,
    max_files_total: int = 24,
    inventory: Optional[dict[str, Any]] = None,
    schema_signals: Optional[Iterable[str]] = None,
    logged_signals: Optional[Iterable[str]] = None,
    parameter_values: Optional[dict[str, Any]] = None,
    enum_registry: Optional[dict[str, dict[str, Any]]] = None,
) -> DiscoveryResult:
    """Deterministic discovery fixpoint: the DAG's own gaps drive the search.

    Round 0 seeds the file set from ``seeds`` (+ the terminal itself, so
    an empty seed list can still bootstrap). Every round loads new files
    through Layer 1, rebuilds the DAG from the union of facts, then turns
    the DAG's ``unresolved_symbols`` into definition searches for the next
    round's files. Cross-file helpers resolve *within* a round via the
    on-demand provider; the provider's fetched files join the next round
    so their assignments bind too.

    Stops when: nothing is unresolved, a round makes no progress (same
    gap set and no provider fetches), no new files remain, or budgets run
    out. No LLM anywhere — seed selection and sufficiency judgment are
    the caller's problem (the judge stage).

    Every terminal — seeder-proposed, judge-proposed, or replayed from
    the Layer 4 cache — passes :func:`validate_terminal` against the
    round's loaded facts before anything is built from it. An absent
    terminal keeps loading pending files but never builds; an ambiguous
    or out-of-scope one stops the fixpoint with the structured reason in
    ``terminal_validation`` — building would fuse unrelated modules.
    """
    terminal_as_given = str(terminal or "").strip()
    terminal = canonicalize_terminal(terminal)
    if logged_signals is not None:
        # Materialized once: consumed by per-round validation AND the
        # builder, so a one-shot iterable must not exhaust in between.
        logged_signals = {str(s) for s in logged_signals}

    if enum_registry is None:
        # Schema-derived message enums, flattened PER MESSAGE (the scope
        # the reference itself carries) — loaded once per discovery.
        from flight_log_agent.px4.msg_schema import load_px4_declared_constant_registry

        try:
            enum_registry = load_px4_declared_constant_registry(profiler.source)
        except Exception:
            enum_registry = {}

    seed_queries = dedupe_keep_order([*(str(s) for s in seeds if s), terminal])
    hits = profiler.search_related_source_files(
        seed_queries, max_files=max_files_per_round
    )
    pending: list[str] = [hit.file for hit in hits]
    if terminal_file:
        # The declared write file is provenance, not a search guess —
        # load it first so validation decides with it in evidence.
        pending = dedupe_keep_order([terminal_file, *pending])

    loaded: list[str] = []
    facts_by_file: dict[str, SourceFileFacts] = {}
    rounds: list[DiscoveryRound] = []
    dag: Optional[MechanismDAG] = None
    inputs = DAGInputs()
    validation: Optional[TerminalValidation] = None
    previous_unresolved: Optional[set[str]] = None

    for index in range(max_rounds):
        budget_left = max(0, max_files_total - len(loaded))
        new_files = [f for f in pending if f not in facts_by_file]
        new_files = new_files[: min(max_files_per_round, budget_left)]
        if not new_files and index > 0:
            break

        for facts in load_facts(
            profiler,
            cache_root,
            new_files,
            source_hash,
            source_root=source_root,
        ):
            facts_by_file[facts.file] = facts
        loaded.extend(new_files)

        inputs = dag_inputs_from_facts(facts_by_file.values())

        # Gate the build on terminal validation. A verdict scoped to a
        # resolved file is locked — later rounds load foreign files whose
        # same-named writers cannot hijack a scoped slice. An UNSCOPED
        # valid verdict is re-checked every round: writers arriving from
        # gap searches can reveal the terminal as genuinely ambiguous.
        locked = (
            validation is not None
            and validation.status == "valid"
            and bool(validation.resolved_file or terminal_file or validation.logged)
        )
        if not locked:
            validation = validate_terminal(
                terminal_as_given, inputs.bindings, logged_signals or (), terminal_file
            )
        if validation.status != "valid":
            dag = None
            rounds.append(
                DiscoveryRound(
                    index=index,
                    new_files=list(new_files),
                    unresolved_symbols=[],
                    vertices=0,
                    edges=0,
                )
            )
            # Ambiguity only grows with more files; an out-of-scope
            # verdict is final once the declared file itself is loaded.
            # Plain absence keeps draining pending files — the writer
            # may live in a file the seed search found but the round
            # budget deferred.
            if validation.status == "ambiguous" or (
                validation.status == "absent_in_scope"
                and terminal_file in facts_by_file
            ):
                break
            continue

        fetched_files: list[str] = []
        provider = make_helper_body_provider(profiler, fetched_files)
        dag = build_mechanism_dag(
            inputs.bindings,
            terminal,
            inventory=inventory,
            schema_signals=schema_signals,
            logged_signals=logged_signals,
            helper_expressions=inputs.helper_expressions,
            helper_body_provider=provider,
            parameter_predicates=inputs.parameter_predicates,
            parameter_values=parameter_values,
            parameter_names=inputs.parameter_names,
            parameter_aliases=inputs.parameter_aliases,
            terminal_file=validation.resolved_file or terminal_file,
            call_statements=inputs.call_statements,
            boundary_bindings=inputs.boundary_bindings,
            enum_registry=enum_registry,
        )

        unresolved = set(dag.unresolved_symbols)
        rounds.append(
            DiscoveryRound(
                index=index,
                new_files=list(new_files),
                unresolved_symbols=sorted(unresolved),
                vertices=len(dag.vertices),
                edges=len(dag.edges),
            )
        )

        if not unresolved:
            break
        if unresolved == previous_unresolved and not fetched_files:
            break
        previous_unresolved = unresolved

        gap_files = _gap_definition_files(
            profiler,
            unresolved,
            max_files_per_gap=2,
            max_total=max_files_per_round * 2,
        )
        pending = dedupe_keep_order(gap_files + fetched_files)

    return DiscoveryResult(
        dag=dag,
        inputs=inputs,
        files_loaded=loaded,
        rounds=rounds,
        terminal_validation=validation,
    )
