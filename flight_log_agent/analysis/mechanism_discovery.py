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

from flight_log_agent.analysis.mechanism_dag import MechanismDAG, build_mechanism_dag
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.px4.source_facts_cache import (
    SourceFileFacts,
    get_or_extract_facts,
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
    ``target`` / ``expression`` / ``file`` / ``line``. ``logged_signal``
    composes from ``target_topic`` + ``target_field`` when the profiler
    resolved the write to a published topic; otherwise it stays empty and
    the binding participates only via ``target_symbol``.
    """
    ref = _as_dict(assignment)
    topic = ref.get("target_topic")
    field_name = ref.get("target_field")
    return {
        "target_symbol": str(ref.get("target") or ""),
        "source_symbol": str(ref.get("expression") or ""),
        "assignment_path": [
            {
                "file": str(ref.get("file") or ""),
                "line": int(ref.get("line") or 0),
                "expression": str(ref.get("expression") or ""),
            }
        ],
        "logged_signal": f"{topic}.{field_name}" if topic and field_name else "",
        "control_predicates": list(ref.get("control_predicates") or []),
        "struct_variables": dict(ref.get("struct_variables") or {}),
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

    for facts_entry in facts:
        entry = _as_dict(facts_entry)

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
            inputs.bindings.append(binding_from_assignment(ref))

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
    if len(root) < 4:
        return []
    return [f"{root} ="]


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
    """

    def provider(helper_name: str) -> Any:
        # Same rationale as _definition_queries: a too-short name
        # (``get``, ``max``, ``sin``) matches half the tree and only
        # drags in junk definitions.
        if len(helper_name) < 4:
            return []
        hits = profiler.search_related_source_files(
            [f"::{helper_name}(", f"{helper_name}("],
            max_files=max_files_per_helper,
        )
        files = [hit.file for hit in hits]
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
    """
    seed_queries = dedupe_keep_order([*(str(s) for s in seeds if s), terminal])
    hits = profiler.search_related_source_files(
        seed_queries, max_files=max_files_per_round
    )
    pending: list[str] = [hit.file for hit in hits]

    loaded: list[str] = []
    facts_by_file: dict[str, SourceFileFacts] = {}
    rounds: list[DiscoveryRound] = []
    dag: Optional[MechanismDAG] = None
    inputs = DAGInputs()
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
            terminal_file=terminal_file,
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

        gap_queries = dedupe_keep_order(
            query
            for symbol in sorted(unresolved)
            for query in _definition_queries(symbol)
        )
        gap_hits = (
            profiler.search_related_source_files(
                gap_queries, max_files=max_files_per_round
            )
            if gap_queries
            else []
        )
        pending = dedupe_keep_order(
            [hit.file for hit in gap_hits] + fetched_files
        )

    return DiscoveryResult(
        dag=dag,
        inputs=inputs,
        files_loaded=loaded,
        rounds=rounds,
    )
