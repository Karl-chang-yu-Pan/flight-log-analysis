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
from typing import Any, Iterable, Optional, Sequence, Union

from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.px4.source_facts_cache import (
    SourceFileFacts,
    get_or_extract_facts,
)


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
