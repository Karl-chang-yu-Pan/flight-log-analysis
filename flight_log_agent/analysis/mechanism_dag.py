"""Source-mechanism DAG builder.

Constructs a compact, LLM-friendly DAG of the source-derived dependency
graph reaching a given terminal symbol. Three vertex kinds
(``evidence`` / ``operation`` / ``branch``), two edge kinds
(``data`` / ``control``). See ``memory/dag_design.md`` for the design
rationale.

Sections:

* Milestone 1 — DAG models and ``build_mechanism_dag`` (backward slice
  from a terminal, no-collapse helper nesting, snippet embedding).
* Milestone 2 — ``evaluate_feasibility`` with constant reduction and
  optional dead-subgraph pruning.
* Milestone 3 — interval evaluation over ULog signal samples,
  populating ``active_windows`` per branch.
* Milestone 4 — disk cache (Layer 2/3) and ``split_by_terminal``
  subgraph partitioning for downstream LLM presentation.
"""

from __future__ import annotations

import re
import math
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from flight_log_agent.analysis.dag_value import (
    DAGValueProgram,
    DAGValueSession,
)
from flight_log_agent.analysis.parameter_lookup import (
    CXX_STDLIB_CONSTANTS,
)
from flight_log_agent.analysis.source_expression import (
    alias_dotted_names,
    source_expression_names,
)
from flight_log_agent.analysis.source_expansion import (
    SourceStructureIndex,
    SourceSymbolIdentity,
    UnresolvedSourceReference,
    reference_receiver_is_source_boundary,
    source_reference_resolution_key,
)
from flight_log_agent.px4.mechanism_source_profiler import (
    callable_accepts_argument_count,
    callable_arguments_with_defaults,
    callable_parameter_count,
    split_top_level_args,
    substitute_expression_symbols,
)
from flight_log_agent.symbols import (
    exact_symbol,
    is_signal_reference,
    parse_signal_reference,
    source_storage_produces_reference,
    strip_symbol_indices,
    symbol_produces_reference,
)
from flight_log_agent.utils import dedupe_keep_order, stable_id
from flight_log_agent.ulog.inventory import observed_signals_from_inventory


VertexKind = Literal["evidence", "operation", "branch"]
EvidenceSubKind = Literal["logged_signal", "parameter", "constant", "opaque_symbol"]
OperationSubKind = Literal["assign", "reduction", "helper_call", "external_call", "unresolved"]
EdgeKind = Literal["data", "control", "selection"]

_CALL_SCOPE_MARKER = "::@call:"
SourceReadScope = tuple[str, str, Optional[int], Optional[int]]


@dataclass(frozen=True)
class _ExpressionCall:
    """One syntactic call occurrence inside a source expression."""

    name: str
    receiver: str
    args: tuple[str, ...]
    argument_expressions: tuple[Any, ...]
    offset: int
    source_site_id: str
    call_scope: str = ""
    helper_key: Optional[tuple[str, str]] = None
    call_text: str = ""
    result_text: str = ""
    result_path: str = ""


def _base_callable_scope(callable_id: str) -> str:
    """Return the source callable behind a call-site-instantiated scope."""
    return str(callable_id or "").split(_CALL_SCOPE_MARKER, 1)[0]


def _callable_instance_scope(
    callee_callable: str,
    *,
    caller_callable: str,
    source_site_id: str,
) -> str:
    """Create a stable lexical scope for one invocation of ``callee``."""
    token = stable_id(
        "call",
        (str(caller_callable or ""), str(source_site_id or ""), callee_callable),
    )
    return f"{callee_callable}{_CALL_SCOPE_MARKER}{token}"


def _source_site_order(source_site_id: str) -> Optional[int]:
    """Compatibility fallback for source facts predating ``source_order``."""
    parts = str(source_site_id or "").rsplit(":", 3)
    if len(parts) != 4:
        return None
    if parts[-1].startswith("legacy_"):
        return int(parts[-2]) if parts[-2].isdigit() else None
    return int(parts[-3]) if parts[-3].isdigit() else None


def _record_source_order(
    record: dict[str, Any],
    *,
    order_key: str = "source_order",
    site_key: str = "source_site_id",
) -> Optional[int]:
    raw = record.get(order_key)
    if isinstance(raw, (int, float)):
        return int(raw)
    return _source_site_order(str(record.get(site_key) or ""))

# Fixed language/evaluator grammar, not domain semantics. These tokens can be
# followed by parentheses in lowered source but never identify source helpers.
_NON_CALL_SYNTAX = {
    "and",
    "bool",
    "catch",
    "char",
    "const_cast",
    "double",
    "dynamic_cast",
    "else",
    "float",
    "for",
    "if",
    "int",
    "long",
    "not",
    "or",
    "reinterpret_cast",
    "return",
    "short",
    "signed",
    "sizeof",
    "static_cast",
    "switch",
    "unsigned",
    "void",
    "while",
}


class DAGVertex(BaseModel):
    """A single DAG node.

    ``kind`` is the primary discriminator. ``sub_kind`` narrows the role
    within a kind. Kind-specific fields are optional so a single model
    covers all three kinds without per-kind subclasses.
    """

    id: str
    kind: VertexKind
    sub_kind: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    snippet: Optional[str] = None

    # Operation-specific fields.
    variable: Optional[str] = None
    expression: Optional[str] = None
    lowered_expression: Optional[str] = None
    operator_kind: Optional[str] = None
    provenance: Optional[str] = None
    unresolved_reason: Optional[str] = None

    # Branch-specific fields. Filled by later milestones.
    predicate_raw: Optional[str] = None
    predicate_lowered: Optional[str] = None
    feasibility_verdict: Optional[str] = None
    active_windows: list[tuple[float, float]] = Field(default_factory=list)

    # Evidence-specific fields.
    signal_name: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class DAGEdge(BaseModel):
    id: str
    source_id: str
    target_id: str
    kind: EdgeKind
    role: Optional[str] = None
    via: Optional[str] = None


class MechanismDAG(BaseModel):
    dag_id: str
    terminal: str
    vertices: list[DAGVertex] = Field(default_factory=list)
    edges: list[DAGEdge] = Field(default_factory=list)
    unresolved_symbols: list[str] = Field(default_factory=list)
    # Internal expansion frontier. Excluded from serialized DAG/report schemas;
    # ``unresolved_symbols`` remains the stable external representation.
    unresolved_references: list[UnresolvedSourceReference] = Field(
        default_factory=list, exclude=True
    )
    pending_construction: list[str] = Field(default_factory=list, exclude=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _as_binding_dict(binding: Any) -> dict[str, Any]:
    if isinstance(binding, dict):
        return binding
    if hasattr(binding, "model_dump"):
        return binding.model_dump(exclude_none=True)
    return dict(vars(binding))


def _source_expression_metadata(
    expression_ref: Any,
    fallback_text: str,
) -> dict[str, Any]:
    """Serialize parser-proven local expression semantics on a DAG vertex."""
    if hasattr(expression_ref, "model_dump"):
        expression_ref = expression_ref.model_dump(exclude_none=True)
    if not isinstance(expression_ref, dict):
        return {
            "source_expression": str(fallback_text or ""),
            "expression_inputs_exact": False,
        }
    record = {
        "text": str(expression_ref.get("text") or fallback_text or ""),
        "lowered_text": str(
            expression_ref.get("lowered_text")
            or expression_ref.get("text")
            or fallback_text
            or ""
        ),
        "input_symbols": [
            str(value)
            for value in (expression_ref.get("input_symbols") or [])
            if str(value)
        ],
        "input_identities": dict(expression_ref.get("input_identities") or {}),
        "call_results": [
            (
                value.model_dump(exclude_none=True)
                if hasattr(value, "model_dump")
                else dict(value)
            )
            for value in (expression_ref.get("call_results") or [])
        ],
        "exact": bool(expression_ref.get("exact", False)),
    }
    return {
        "source_expression": record["text"],
        "source_expression_ref": record,
        "expression_inputs_exact": record["exact"],
    }


def _helper_source_return_sites(helper: dict[str, Any]) -> list[dict[str, Any]]:
    """Return parser-emitted source return sites only."""
    return [
        _as_binding_dict(site) for site in (helper.get("return_sites") or [])
    ]


def _logged_signals_from_inventory(inventory: Optional[dict[str, Any]]) -> set[str]:
    """Derive the ``topic.field`` logged-signal set from an inventory.

    Reimplemented here (not imported from BindingIndex) so the DAG builder
    has no dependency on that module.
    """
    return observed_signals_from_inventory(inventory)


def observed_signal_placements(
    reference: str, observed_signals: Iterable[str]
) -> tuple[str, ...]:
    """Return exact observed placements compatible with one reference."""
    observed = {exact_symbol(value) for value in observed_signals if value}
    symbol = exact_symbol(reference)
    if symbol in observed:
        return (symbol,)
    parsed = parse_signal_reference(symbol)
    if parsed is None:
        return ()
    topic, requested_instance, field = parsed
    candidates: list[str] = []
    for candidate in observed:
        candidate_parts = parse_signal_reference(candidate)
        if candidate_parts is None:
            continue
        candidate_topic, candidate_instance, candidate_field = candidate_parts
        if candidate_topic != topic or candidate_field != field:
            continue
        if (
            requested_instance is not None
            and candidate_instance != requested_instance
        ):
            continue
        candidates.append(candidate)
    return tuple(sorted(candidates))


def resolve_observed_signal_placement(
    reference: str, observed_signals: Iterable[str]
) -> Optional[str]:
    """Resolve one reference only when its observed placement is unique."""
    candidates = observed_signal_placements(reference, observed_signals)
    return candidates[0] if len(candidates) == 1 else None


def _parse_numeric_literal(
    expression: str,
    known_constants: Optional[dict[str, Any]] = None,
) -> Optional[Any]:
    """Return the numeric value of a compile-time-constant expression, else None.

    Handles bare int/float/hex literals and simple constant arithmetic
    (``(1 << 5)``) via the restricted evaluator. References resolve only from
    ``known_constants`` supplied by other source-proven declarations.
    """
    text = (expression or "").strip().rstrip(";").strip()
    if not text:
        return None
    # Strip a trailing C float suffix (6371000.0f -> 6371000.0).
    text = re.sub(r"(?<=[0-9.])[fFuUlL]+\b", "", text)
    try:
        return int(text, 0)
    except (TypeError, ValueError):
        pass
    try:
        return float(text)
    except (TypeError, ValueError):
        pass
    from flight_log_agent.analysis.safe_eval import (
        ExpressionEvaluationError,
        eval_expression,
    )
    try:
        value = eval_expression(text, dict(known_constants or {}))
    except (ExpressionEvaluationError, TypeError, ValueError, ZeroDivisionError, SyntaxError):
        return None
    return value if isinstance(value, (int, float, bool)) else None


def build_mechanism_dag(
    source_bindings: Sequence[Any],
    terminal: str,
    *,
    inventory: Optional[dict[str, Any]] = None,
    schema_signals: Optional[Iterable[str]] = None,
    helper_expressions: Sequence[Any] = (),
    helper_body_provider: Optional[Callable[[str], Any]] = None,
    parameter_predicates: Sequence[Any] = (),
    parameter_values: Optional[dict[str, Any]] = None,
    source_root: Optional[str | Path] = None,
    logged_signals: Optional[Iterable[str]] = None,
    parameter_names: Optional[Iterable[str]] = None,
    parameter_bindings: Sequence[Any] = (),
    snippet_context_lines: int = 3,
    terminal_file: Optional[str] = None,
    terminal_identity: Optional[SourceSymbolIdentity | dict[str, Any]] = None,
    call_statements: Sequence[Any] = (),
    boundary_bindings: Sequence[Any] = (),
    enum_registry: Optional[dict[str, dict[str, Any]]] = None,
    source_structure: Optional[SourceStructureIndex] = None,
    construction_checkpoint: Optional[Callable[[MechanismDAG], set[str]]] = None,
) -> MechanismDAG:
    """Build a mechanism DAG for ``terminal``.

    ``source_bindings`` are the profiler's source-assignment records
    (``target_symbol``, ``source_symbol``, ``control_predicates``,
    ``assignment_path``, ``struct_variables``).
    The builder walks them backward from ``terminal`` **natively** — it
    indexes and traverses them itself rather than delegating to
    ``BindingIndex`` — so the graph is built by one interleaved fixpoint
    rather than re-nesting a flattened reach list. ``inventory`` supplies
    the logged-signal catalogue (``topic.field``); ``schema_signals`` is
    the optional msg-schema catalogue used to validate derived
    ``topic.field`` references. ``helper_expressions`` are the profiler's
    ``HelperExpressionRef`` records used to expand helper subgraphs
    without collapsing intermediates. ``helper_body_provider`` (optional)
    is invoked lazily when the builder encounters a helper call whose
    body isn't in ``helper_expressions``, so cross-file helper resolution
    happens on-demand instead of requiring pre-flattening via
    ``extract_helper_expressions_recursive``. ``parameter_predicates``
    are the profiler's ``ParameterPredicateRef`` records — the builder
    consults them at branch emission to attach operator/threshold
    metadata so downstream evaluators don't re-run
    ``parse_parameter_predicate``. ``parameter_values`` (from the
    ULog inventory) let evidence:constant vertices carry the resolved
    numeric value when the source references a PX4 parameter name as a
    literal. ``parameter_bindings`` retain the source-declared member,
    owner, and canonical parameter name so identical member spellings in
    unrelated classes cannot alias. ``source_root`` enables per-vertex
    snippet embedding; omit it to keep tests hermetic. ``terminal_file``
    (optional) is part of terminal identity. No writer in that file is an
    unresolved terminal, never permission to widen to a same-named symbol
    elsewhere.

    Identity is the EXACT symbol spelling (``exact_symbol``): indices,
    instances, and the leading-underscore member marker all distinguish.
    Source-object to logged-topic equivalence requires an explicit
    ``boundary_bindings`` entry.

    ``construction_checkpoint`` enables staged construction and disables inline
    helper fetching. At coherent dependency boundaries it receives this DAG
    with internal pending-operation IDs and returns only IDs whose inactivity
    has been proven from exact source controls over the comparison domain.
    Missing definitions remain typed source requests; known pending values are
    resumed in this builder, without reconstructing their invocation scopes.
    """
    builder = _DAGBuilder(
        source_bindings=[_as_binding_dict(b) for b in source_bindings],
        terminal=terminal,
        helper_index=_index_helpers(helper_expressions),
        helper_body_provider=None if construction_checkpoint is not None else helper_body_provider,
        parameter_predicates=list(parameter_predicates),
        parameter_values=dict(parameter_values or {}),
        source_root=Path(source_root) if source_root else None,
        logged_signals=set(logged_signals or ()) | _logged_signals_from_inventory(inventory),
        schema_signals=set(schema_signals or ()),
        parameter_names=set(parameter_names or ()),
        parameter_bindings=[_as_binding_dict(b) for b in parameter_bindings],
        snippet_context_lines=snippet_context_lines,
        terminal_file=terminal_file,
        terminal_identity=terminal_identity,
        call_statements=[_as_binding_dict(c) for c in call_statements],
        boundary_bindings=[_as_binding_dict(b) for b in boundary_bindings],
        enum_registry=dict(enum_registry or {}),
        source_structure=source_structure or SourceStructureIndex(),
    )
    return builder.build(construction_checkpoint=construction_checkpoint)


# ---------------------------------------------------------------------------
# Builder internals
# ---------------------------------------------------------------------------


class _DAGBuilder:
    def __init__(
        self,
        *,
        source_bindings: list[dict[str, Any]],
        terminal: str,
        helper_index: dict[tuple[str, str], dict[str, Any]],
        helper_body_provider: Optional[Callable[[str], Any]],
        parameter_predicates: list[Any],
        parameter_values: dict[str, Any],
        source_root: Optional[Path],
        logged_signals: set[str],
        schema_signals: set[str],
        parameter_names: set[str],
        parameter_bindings: list[dict[str, Any]],
        snippet_context_lines: int,
        terminal_file: Optional[str] = None,
        terminal_identity: Optional[SourceSymbolIdentity | dict[str, Any]] = None,
        call_statements: Optional[list[dict[str, Any]]] = None,
        boundary_bindings: Optional[list[dict[str, Any]]] = None,
        enum_registry: Optional[dict[str, dict[str, Any]]] = None,
        source_structure: Optional[SourceStructureIndex] = None,
    ) -> None:
        self.terminal_raw = terminal
        self.terminal = exact_symbol(terminal)
        self.terminal_file = str(terminal_file) if terminal_file else None
        self.terminal_identity = (
            terminal_identity
            if isinstance(terminal_identity, SourceSymbolIdentity)
            else SourceSymbolIdentity.model_validate(terminal_identity)
            if terminal_identity
            else None
        )
        self._call_statements = list(call_statements or [])
        self._calls_by_site: dict[
            tuple[str, str, int, str, tuple[str, ...]], list[dict[str, Any]]
        ] = defaultdict(list)
        self._calls_by_location: dict[
            tuple[str, str, int, str], list[dict[str, Any]]
        ] = defaultdict(list)
        self._calls_by_source_site_id: dict[str, dict[str, Any]] = {}
        for call in self._call_statements:
            key = (
                str(call.get("file") or ""),
                _base_callable_scope(
                    str(call.get("callable_id") or call.get("function") or "")
                ),
                int(call.get("line") or 0),
                str(call.get("name") or "").rsplit("::", 1)[-1],
                tuple(str(arg).strip() for arg in (call.get("args") or [])),
            )
            self._calls_by_site[key].append(call)
            self._calls_by_location[key[:4]].append(call)
            source_site_id = str(call.get("source_site_id") or "")
            if source_site_id:
                self._calls_by_source_site_id.setdefault(source_site_id, call)
        self._boundary_bindings = list(boundary_bindings or [])
        # Schema-derived message enums, scoped per message
        # (``{message: {CONSTANT: value}}``) — resolved at wiring time
        # like every other constant, never as a flat global table.
        self._enum_registry = dict(enum_registry or {})
        self._source_structure = source_structure or SourceStructureIndex()
        self.helper_index = helper_index
        self._helper_keys_by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for helper_key in helper_index:
            self._helper_keys_by_name[helper_key[0]].append(helper_key)
        self.helper_body_provider = helper_body_provider
        # Helpers already probed via the provider so a repeated call for an
        # unknown name doesn't re-fetch on every backward-walk pass.
        self._helper_provider_probed: set[tuple[Any, ...]] = set()
        self.source_root = source_root
        # Catalogues key on exact identity. Schema shape lookup allows an
        # aggregate declaration to validate an indexed element; observed
        # placement resolution keeps field indices exact and permits an
        # omitted topic instance only when the log has one candidate.
        self.logged_signals = {exact_symbol(s) for s in logged_signals if s}
        self.parameter_names = {p for p in parameter_names if p}
        self._parameter_bindings_by_member: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)
        for raw_binding in parameter_bindings or []:
            binding = dict(raw_binding)
            member = exact_symbol(str(binding.get("member") or ""))
            name = str(binding.get("name") or "")
            if not member or not name:
                continue
            root = member.replace("->", ".").split(".", 1)[0]
            self._parameter_bindings_by_member[root].append(binding)
            self.parameter_names.add(name)
        self.snippet_context_lines = snippet_context_lines
        self._parameter_values = {
            str(k).upper(): v for k, v in (parameter_values or {}).items()
        }
        self._parameter_predicate_by_site: dict[
            tuple[str, str, int, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for record in parameter_predicates or []:
            entry = record if isinstance(record, dict) else (
                record.model_dump(exclude_none=True) if hasattr(record, "model_dump") else dict(vars(record))
            )
            predicate = str(entry.get("predicate") or "").strip()
            if predicate:
                self._parameter_predicate_by_site[
                    (
                        _canonical_predicate(predicate),
                        str(entry.get("file") or ""),
                        int(entry.get("line") or 0),
                        str(entry.get("source_site_id") or ""),
                    )
                ].append(entry)

        self._schema_signals = {exact_symbol(s) for s in schema_signals if s}
        self._schema_shapes = {strip_symbol_indices(s) for s in self._schema_signals}

        # Native backward-walk indexes over the source bindings — the DAG
        # owns the walk rather than delegating to BindingIndex. ``_by_target``
        # keys on the written symbol using exact identity, with index-erased shape maps
        # so an index-free reference still finds its indexed writers (and
        # vice versa) without ever fusing distinct indices.
        self._source_bindings: list[dict[str, Any]] = list(source_bindings)
        self._all_bindings: list[dict[str, Any]] = list(self._source_bindings)
        self._registered_call_instances: set[str] = set()
        self._projected_bindings: dict[tuple[int, str], dict[str, Any]] = {}
        self._by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._target_shapes: dict[str, list[str]] = defaultdict(list)
        self._reference_identities_by_scope: dict[
            tuple[str, str, str], list[SourceSymbolIdentity]
        ] = defaultdict(list)
        self._reference_identities_by_site: dict[
            tuple[str, str, int, str], list[SourceSymbolIdentity]
        ] = defaultdict(list)
        self._reference_identities_by_file: dict[
            tuple[str, str], list[SourceSymbolIdentity]
        ] = defaultdict(list)
        self._reference_identities_by_callable: dict[
            tuple[str, str], list[SourceSymbolIdentity]
        ] = defaultdict(list)
        self._reference_identities_by_symbol: dict[
            str, list[SourceSymbolIdentity]
        ] = defaultdict(list)
        for binding in self._all_bindings:
            target = exact_symbol(
                str(binding.get("target_symbol") or binding.get("target") or "")
            )
            if target:
                self._index_binding(self._by_target, self._target_shapes, target, binding)
            self._index_reference_identities(binding)

        # Source-defined numeric constants (enum entry / ``#define`` /
        # ``constexpr``), resolved natively from the bindings: a single
        # unconditional declaration whose RHS is a compile-time expression.
        # A fixed point derives implicit enumerators represented as
        # ``PREVIOUS + 1`` from their preceding declaration.
        # Replaces the old dependency on ``BindingIndex.assignment_resolutions``
        # (which stored SliceResult objects the DAG mis-typed).
        self._source_constants: dict[str, Any] = {}
        self._source_constants_by_entity: dict[str, Any] = {}
        constant_values_by_alias: dict[str, dict[str, Any]] = defaultdict(dict)
        constant_bindings: list[
            tuple[str, dict[str, Any], tuple[str, ...], str]
        ] = []
        for binding in self._all_bindings:
            target = exact_symbol(
                str(binding.get("target_symbol") or binding.get("target") or "")
            )
            if (
                not target
                or binding.get("declaration_kind") not in {"enum", "define", "constexpr"}
            ):
                continue
            if binding.get("control_predicates"):
                continue
            identity = self._binding_target_identity(binding)
            scopes = {
                exact_symbol(str(value))
                for value in (binding.get("constant_scopes") or [])
                if exact_symbol(str(value))
            }
            if identity is not None:
                scopes.update(
                    exact_symbol(value)
                    for value in (
                        identity.class_owner,
                        identity.namespace_owner,
                    )
                    if exact_symbol(value)
                )
            entity = (
                identity.declaration_id
                if identity is not None and identity.declaration_id
                else stable_id(
                    "constant",
                    (
                        target,
                        self._binding_first_file(binding),
                        self._binding_target_line(binding) or 0,
                    ),
                )
            )
            constant_bindings.append((target, binding, tuple(sorted(scopes)), entity))

        def register_constant(
            target: str, scopes: tuple[str, ...], entity: str, value: Any
        ) -> None:
            self._source_constants_by_entity[entity] = value
            leaf = target.rsplit(".", 1)[-1]
            aliases = {
                *(f"{scope}.{leaf}" for scope in scopes),
                *([target] if "." in target else []),
            }
            if not self._source_structure.authoritative_declarations:
                aliases.add(target)
            for alias in aliases:
                candidates = constant_values_by_alias[alias]
                candidates[entity] = value
                if len(candidates) == 1:
                    self._source_constants[alias] = next(iter(candidates.values()))
                else:
                    self._source_constants.pop(alias, None)

        pending_constants = list(constant_bindings)
        while pending_constants:
            unresolved: list[
                tuple[str, dict[str, Any], tuple[str, ...], str]
            ] = []
            progress = False
            for target, binding, scopes, entity in pending_constants:
                known_constants = dict(self._source_constants)
                expression_ref = binding.get("expression_ref") or {}
                if hasattr(expression_ref, "model_dump"):
                    expression_ref = expression_ref.model_dump(
                        exclude_none=True
                    )
                if isinstance(expression_ref, dict):
                    for symbol, raw_identity in (
                        expression_ref.get("input_identities") or {}
                    ).items():
                        try:
                            input_identity = SourceSymbolIdentity.model_validate(
                                raw_identity
                            )
                        except (TypeError, ValueError):
                            continue
                        input_value = self._source_constants_by_entity.get(
                            input_identity.declaration_id
                        )
                        if input_value is not None:
                            known_constants[exact_symbol(str(symbol))] = input_value
                for scope in scopes:
                    prefix = f"{scope}."
                    for alias, alias_value in self._source_constants.items():
                        if not alias.startswith(prefix):
                            continue
                        relative = alias[len(prefix) :]
                        if "." not in relative:
                            known_constants[relative] = alias_value
                value = _parse_numeric_literal(
                    str(
                        binding.get("source_symbol")
                        or binding.get("expression")
                        or ""
                    ),
                    known_constants,
                )
                if value is None:
                    unresolved.append((target, binding, scopes, entity))
                    continue
                register_constant(target, scopes, entity, value)
                progress = True
            if not progress:
                break
            pending_constants = unresolved

        self.vertices: dict[str, DAGVertex] = {}
        self.edges: dict[tuple[str, str, str, str, str], DAGEdge] = {}
        self.unresolved_symbols: set[str] = set()
        self._unresolved_references: dict[tuple[Any, ...], UnresolvedSourceReference] = {}
        self._terminal_reference_keys: set[tuple[Any, ...]] = set()

        # Memoization tables.
        self._evidence_by_signal: dict[tuple[Any, ...], str] = {}
        # Branch vertices key on predicate plus the parser's source-node ID.
        # File/line remain the fallback for legacy facts.
        self._branch_by_site: dict[tuple[str, str, int, str, str], str] = {}
        # Assignment-target index: EXACT target symbol → list of vertex ids
        # that produce it, plus the index-erased shape map for
        # index-compatible lookups.
        self._producers_by_symbol: dict[str, list[str]] = {}
        self._producer_shapes: dict[str, list[str]] = defaultdict(list)

        # Snippet cache to avoid re-reading a file many times.
        self._file_lines_cache: dict[str, list[str]] = {}
        self._callers_by_callee: dict[
            str, list[tuple[dict[str, Any], tuple[str, str]]]
        ] = defaultdict(list)
        self._resolved_helper_key_by_call_site: dict[str, tuple[str, str]] = {}
        self._caller_paths_cache: dict[
            str, list[list[tuple[dict[str, Any], tuple[str, str]]]]
        ] = {}
        self._index_source_call_edges()

    # ------------------------------------------------------------
    # Exact-identity indexes
    # ------------------------------------------------------------
    #
    # Every index keys on ``exact_symbol`` — the lossless identity.
    # Lookups are directional: an aggregate writer may produce one of its
    # elements, but an element writer never proves its enclosing aggregate
    # and two explicit indices never fuse. This replaces the old lossy
    # ``normalize_symbol`` keys
    # that erased indices and the leading-underscore member marker.

    @staticmethod
    def _index_binding(
        index: dict[str, list[dict[str, Any]]],
        shapes: dict[str, list[str]],
        key: str,
        binding: dict[str, Any],
    ) -> None:
        index[key].append(binding)
        shape_bucket = shapes[strip_symbol_indices(key)]
        if key not in shape_bucket:
            shape_bucket.append(key)

    @staticmethod
    def _matching_keys(
        shapes: dict[str, list[str]], symbol_exact: str
    ) -> list[str]:
        return [
            key
            for key in shapes.get(strip_symbol_indices(symbol_exact), ())
            if symbol_produces_reference(key, symbol_exact)
        ]

    @staticmethod
    def _matching_source_keys(
        shapes: dict[str, list[str]], symbol_exact: str
    ) -> list[str]:
        """Find exact or aggregate source writers for one source read."""
        parts = exact_symbol(symbol_exact).split(".")
        matches: list[str] = []
        for length in range(len(parts), 0, -1):
            prefix = ".".join(parts[:length])
            for key in shapes.get(strip_symbol_indices(prefix), ()):
                if (
                    key not in matches
                    and source_storage_produces_reference(key, symbol_exact)
                ):
                    matches.append(key)
        return matches

    def _targets_matching(self, symbol_exact: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for key in self._matching_source_keys(
            self._target_shapes, symbol_exact
        ):
            for binding in self._by_target.get(key, ()):
                if id(binding) not in seen:
                    seen.add(id(binding))
                    out.append(binding)
        return out

    def _index_producer(self, key: str, op_id: str) -> None:
        self._producers_by_symbol.setdefault(key, []).append(op_id)
        shape_bucket = self._producer_shapes[strip_symbol_indices(key)]
        if key not in shape_bucket:
            shape_bucket.append(key)

    def _producers_matching(self, symbol_exact: str) -> list[str]:
        out: list[str] = []
        for key in self._matching_keys(self._producer_shapes, symbol_exact):
            for op_id in self._producers_by_symbol.get(key, ()):
                if op_id not in out:
                    out.append(op_id)
        return out

    def _declared_signal_known(self, symbol_exact: str) -> bool:
        """Whether schema structure declares this exact value or aggregate."""
        if symbol_exact in self._schema_signals:
            return True
        shape = strip_symbol_indices(symbol_exact)
        return shape in self._schema_shapes and any(
            strip_symbol_indices(candidate) == shape
            and symbol_produces_reference(candidate, symbol_exact)
            for candidate in self._schema_signals
        )

    def _observed_signal_placement(self, reference: str) -> Optional[str]:
        """Resolve a reference to one exact observed topic-instance placement.

        Field and array identity must match exactly. An omitted topic instance
        may resolve only when the current log has one compatible instance;
        multiple instances remain ambiguous.
        """
        return resolve_observed_signal_placement(reference, self.logged_signals)

    def _member_observation_leaf(
        self, symbol_norm: str, identity: SourceSymbolIdentity
    ) -> Optional[str]:
        """Ground a struct-member read from the flight observation.

        A member whose DECLARED TYPE is a uORB topic struct (e.g.
        ``vehicle_attitude_setpoint_s _att_sp`` -> topic
        ``vehicle_attitude_setpoint``) is recorded in the log. The logged value
        is the ground truth of what the member held, so when the member is READ
        we prefer that observation over tracing its source writes, which for a
        published output are one-per-flight-mode and cannot be disambiguated
        without grounding every mode branch. The topic is derived from the
        declared type, not the member name, so no spelling heuristic applies;
        grounding only happens when the resulting ``topic.field`` is actually
        logged, otherwise this returns ``None`` and source resolution proceeds.
        """
        if identity is None or identity.kind != "member":
            return None
        root, separator, field = symbol_norm.partition(".")
        if not separator or not field:
            return None
        class_owner = identity.class_owner or identity.declaring_class or ""
        member_type = self._source_structure.member_receiver_type(
            class_owner, root
        )
        if not member_type or not member_type.endswith("_s"):
            return None
        placement = self._observed_signal_placement(
            f"{member_type[:-2]}.{field}"
        )
        if placement is None:
            return None
        return self._emit_evidence(
            "logged_signal",
            placement,
            file=None,
            line=None,
            metadata={"grounded_via": "declared_type"},
        )

    def _source_constant_value(
        self,
        symbol: str,
        identity: SourceSymbolIdentity,
        scope_function: str = "",
    ) -> Optional[Any]:
        """Resolve a source constant under its declaration-derived scope.

        A parser-proven declaration identity is authoritative. Qualified names
        and lexical class lineage are structural fallbacks. A globally unique
        spelling is accepted only for non-authoritative legacy facts.
        """
        canonical = exact_symbol(symbol)
        if identity.declaration_proven and identity.declaration_id:
            entity_value = self._source_constants_by_entity.get(
                identity.declaration_id
            )
            if entity_value is not None:
                return entity_value
        if "." in canonical and canonical in self._source_constants:
            return self._source_constants[canonical]
        owner = self._source_structure.callable_owner(
            _base_callable_scope(scope_function)
        )
        for candidate_owner in self._source_structure.lineage(owner):
            qualified = f"{exact_symbol(candidate_owner)}.{canonical}"
            if qualified in self._source_constants:
                return self._source_constants[qualified]
        if (
            not self._source_structure.authoritative_declarations
            and canonical in self._source_constants
        ):
            return self._source_constants[canonical]
        return None

    # ------------------------------------------------------------
    # Build
    # ------------------------------------------------------------

    def build(
        self,
        *,
        construction_checkpoint: Optional[Callable[[MechanismDAG], set[str]]] = None,
    ) -> MechanismDAG:
        """Build the DAG via one interleaved backward fixpoint.

        Starting from the terminal, every helper invocation receives a private
        callable scope. Formal bindings, source assignments, and each source
        return are ordinary writers in that scope and feed the same backward
        symbol frontier. No helper formula is substituted into its caller;
        producer edges and source reachability select the active return.
        """
        # Every frontier symbol carries the scope of the site that
        # referenced it — writers are then resolved under C++-faithful
        # visibility instead of a global by-name index, so a local named
        # ``dt`` in one module can never bind to another module's ``dt``.
        Scope = SourceReadScope  # (source unit, callable, read line, byte order)
        terminal_scope: Scope = (
            self.terminal_file
            or (self.terminal_identity.file if self.terminal_identity else ""),
            (
                self.terminal_identity.callable_id
                if self.terminal_identity
                and self.terminal_identity.kind == "local"
                else ""
            ),
            None,
            None,
        )
        frontier: deque[tuple[str, Any, Scope, str]] = deque(
            [("terminal", self.terminal_raw, terminal_scope, "")]
        )
        walked: set[tuple[Any, ...]] = set()
        materialized_helpers: set[tuple[str, str]] = set()
        self._emitted_ids: set[int] = set()
        self._emitted_bindings: list[dict[str, Any]] = []
        deferred: dict[str, dict[str, Any]] = {}
        discharged: set[str] = set()

        def enqueue_value(binding: dict[str, Any]) -> None:
            if not binding.get("external_source_signal"):
                enqueue_expression(
                    str(binding.get("source_symbol") or binding.get("expression") or ""),
                    self._binding_walk_scope(binding), binding.get("expression_ref"),
                    origin_vertex_id=self._binding_operation_id(binding)[0],
                    read_before_write_target=self._binding_read_before_write_target(binding),
                    before_call_site_id=(str(binding.get("call_site_id") or "")
                                         if binding.get("synthetic_call_binding") else ""),
                )

        def enqueue_expression(
            expression: str,
            scope: Scope,
            expression_ref: Any = None,
            *,
            read_before_write_target: str = "",
            before_call_site_id: str = "",
            origin_vertex_id: str = "",
        ) -> None:
            if not expression:
                return
            target_norm = exact_symbol(read_before_write_target)
            for raw in self._wire_symbols(expression, expression_ref):
                if raw:
                    symbol_scope = scope
                    if (
                        target_norm
                        and exact_symbol(raw) == target_norm
                        and scope[2] is not None
                    ):
                        # Compound writes read the value reaching the source
                        # site before they write the new value. Excluding the
                        # current line prevents the write from producing its
                        # own input while retaining prior local definitions.
                        symbol_scope = (
                            scope[0],
                            scope[1],
                            scope[2] if scope[3] is not None else scope[2] - 1,
                            scope[3] - 1 if scope[3] is not None else None,
                        )
                    symbol_payload: Any = (
                        (raw, before_call_site_id)
                        if before_call_site_id
                        else raw
                    )
                    frontier.append(("symbol", symbol_payload, symbol_scope, origin_vertex_id))
            for invocation in self._find_helper_calls(
                expression,
                scope[0],
                scope_function=scope[1],
                line=scope[2],
                expression_ref=expression_ref,
                origin_vertex_id=origin_vertex_id,
            ):
                invocation_key = (
                    invocation.source_site_id,
                    invocation.result_path,
                )
                if invocation_key not in materialized_helpers:
                    frontier.append(("helper", invocation, scope, origin_vertex_id))

        while frontier or deferred or discharged:
            if not frontier:
                snapshot = self._construction_snapshot(set(deferred) | discharged)
                inactive = construction_checkpoint(snapshot) if construction_checkpoint else set()
                # Newly materialized definitions can change guard evaluation.
                # A prior discharge is not a permanent pruning certificate.
                resumed = discharged - inactive
                for binding in self._emitted_bindings:
                    vertex_id = self._binding_operation_id(binding)[0]
                    if vertex_id in resumed:
                        deferred[vertex_id] = binding
                discharged.difference_update(resumed)
                # Only the checkpoint's source/domain-proven inactivity may
                # discharge pending value work. Unknown paths are resumed.
                for vertex_id in set(deferred) & inactive:
                    discharged.add(vertex_id)
                    del deferred[vertex_id]
                if not deferred:
                    del snapshot
                    break
                incoming: dict[str, list[str]] = defaultdict(list)
                for edge in snapshot.edges:
                    incoming[edge.target_id].append(edge.source_id)
                pending_controls = [v.id for v in snapshot.vertices if v.kind == "branch"]
                control_dependencies: set[str] = set()
                while pending_controls:
                    vertex_id = pending_controls.pop()
                    if vertex_id in control_dependencies:
                        continue
                    control_dependencies.add(vertex_id)
                    pending_controls.extend(incoming[vertex_id])
                needed = set(deferred) & control_dependencies
                for vertex_id in sorted(needed or set(deferred)):
                    enqueue_value(deferred.pop(vertex_id))
                del snapshot
                continue
            kind, payload, scope, origin_vertex_id = frontier.popleft()
            if kind in {"symbol", "terminal"}:
                is_terminal_root = kind == "terminal"
                raw, before_call_site_id = (
                    payload if isinstance(payload, tuple) else (payload, "")
                )
                norm = exact_symbol(raw)
                walk_key = (kind, norm, scope, before_call_site_id, origin_vertex_id)
                if not norm or walk_key in walked:
                    continue
                walked.add(walk_key)
                if (
                    norm != self.terminal
                    and self._source_constant_value(
                        norm,
                        self._reference_identity(
                            raw, scope[0], scope[1], scope[2]
                        ),
                        scope[1],
                    )
                    is not None
                ):
                    # Source-defined constants resolve as value-carrying
                    # evidence leaves at wiring time; walking their single
                    # literal write would demote them to bare operations.
                    continue
                if is_terminal_root:
                    writers = self._writers_of(norm)
                    call_writers = self._writers_from_relevant_calls(
                        norm, raw, scope, terminal=True, origin_vertex_id=origin_vertex_id
                    )
                    writers = self._dedupe_bindings([*writers, *call_writers])
                    if writers and self.terminal_identity is not None:
                        writers = (
                            [
                                binding
                                for binding in writers
                                if (
                                    producer := self._binding_target_identity(binding)
                                )
                                is not None
                                and self._source_structure.storage_compatible(
                                    self.terminal_identity, producer
                                )
                            ]
                            if (
                                self.terminal_identity.declaration_proven
                                or not self._source_structure.authoritative_declarations
                            )
                            else []
                        )
                    elif writers and self.terminal_file:
                        # A validated file is part of terminal identity. No
                        # match is an unresolved terminal, not permission to
                        # widen to same-spelled writers elsewhere.
                        writers = [
                            b for b in writers
                            if self._binding_first_file(b) == self.terminal_file
                        ]
                    writers = self._terminal_call_context_writers(writers)
                else:
                    writers = self._scoped_writers(
                        norm,
                        raw,
                        scope,
                        excluded_call_effect_site=before_call_site_id,
                    )
                    call_writers = self._writers_from_relevant_calls(
                        norm,
                        raw,
                        scope,
                        terminal=False,
                        origin_vertex_id=origin_vertex_id,
                    )
                    writers = self._dedupe_bindings([*writers, *call_writers])
                writers = self._project_source_writers(writers, norm)
                if writers:
                    for binding in writers:
                        newly_emitted = id(binding) not in self._emitted_ids
                        if newly_emitted:
                            self._emitted_ids.add(id(binding))
                            self._emitted_bindings.append(binding)
                            self._emit_operation_vertex(
                                binding,
                                is_terminal=self._binding_reaches_terminal(binding),
                            )
                        if construction_checkpoint is not None:
                            if newly_emitted:
                                deferred[self._binding_operation_id(binding)[0]] = binding
                        else:
                            enqueue_value(binding)
                        # A branch's inputs are part of the mechanism:
                        # walking predicate symbols emits the internal-state
                        # writers that feasibility grounding later follows.
                        for position, predicate in enumerate(
                            binding.get("control_predicates") or []
                        ):
                            enqueue_expression(
                                str(predicate),
                                self._binding_predicate_scope(binding, position),
                                (
                                    (binding.get("control_expression_refs") or [])[position]
                                    if position
                                    < len(binding.get("control_expression_refs") or [])
                                    else None
                                ),
                                origin_vertex_id=self._binding_operation_id(binding)[0],
                            )
            else:  # helper
                invocation = payload
                invocation_key = (
                    invocation.source_site_id,
                    invocation.result_path,
                )
                if invocation_key in materialized_helpers:
                    continue
                materialized_helpers.add(invocation_key)
                helper_key = self._helper_key_for_invocation(
                    invocation,
                    scope_file=scope[0],
                    scope_function=scope[1],
                )
                helper = self.helper_index.get(helper_key) if helper_key else None
                if not helper:
                    continue
                body_scope: Scope = (
                    str(helper.get("file") or ""),
                    invocation.call_scope,
                    None,
                    None,
                )
                return_symbol = (
                    f"__return__{invocation.result_path}"
                    if invocation.result_path.startswith("[")
                    else (
                        f"__return__.{invocation.result_path}"
                        if invocation.result_path
                        else "__return__"
                    )
                )
                frontier.append(("symbol", return_symbol, body_scope, origin_vertex_id))

        return self._construction_snapshot(discharged)

    def _construction_snapshot(self, pending: set[str]) -> MechanismDAG:
        """Wire a coherent view without retaining provisional missing inputs.

        Source operations and invocation identities survive suspension. Derived
        edges/leaves are rewired against the current producer index so an early
        opaque leaf or source request cannot survive after its producer arrives.
        """
        self.vertices = {key: value.model_copy(deep=True) for key, value in self.vertices.items()
                         if value.kind == "operation"}
        self.edges = {}
        self._evidence_by_signal = {}
        self._branch_by_site = {}
        self._producers_by_symbol = {}
        self._producer_shapes = defaultdict(list)
        for vertex in self.vertices.values():
            self._index_producer(exact_symbol(vertex.variable or ""), vertex.id)
        walk_references = dict(self._unresolved_references)
        walk_symbols = set(self.unresolved_symbols)
        for binding in self._emitted_bindings:
            self._wire_binding_edges(binding, controls_only=self._binding_operation_id(binding)[0] in pending)

        terminal_ids = [v.id for v in self.vertices.values() if v.metadata.get("is_terminal")]
        for key in self._terminal_reference_keys:
            reference = self._unresolved_references[key]
            self._unresolved_references[key] = reference.model_copy(update={
                "origin_vertex_ids": dedupe_keep_order([*reference.origin_vertex_ids, *terminal_ids]),
            })

        snapshot = MechanismDAG(
            dag_id=stable_id("dag", (self.terminal, tuple(sorted(self.vertices.keys())))),
            terminal=self.terminal_raw,
            vertices=[self.vertices[key] for key in self.vertices],
            edges=[self.edges[key] for key in self.edges],
            unresolved_symbols=sorted(self.unresolved_symbols),
            unresolved_references=list(self._unresolved_references.values()),
            pending_construction=sorted(pending),
        )
        self._unresolved_references = walk_references
        self.unresolved_symbols = walk_symbols
        return snapshot

    # ------------------------------------------------------------
    # Native backward-walk helpers
    # ------------------------------------------------------------

    @staticmethod
    def _structured_expression_symbols(expression_ref: Any) -> Optional[list[str]]:
        """Return parser-proven value inputs, or ``None`` for fallback."""
        if hasattr(expression_ref, "model_dump"):
            expression_ref = expression_ref.model_dump(exclude_none=True)
        if not isinstance(expression_ref, dict) or not bool(
            expression_ref.get("exact", False)
        ):
            return None
        return dedupe_keep_order(
            str(value)
            for value in (expression_ref.get("input_symbols") or [])
            if str(value)
        )

    def _walk_symbols(
        self, expression: str, expression_ref: Any = None
    ) -> list[str]:
        """Normalized symbols referenced by ``expression``.

        C++ operators/casts are normalized first so ``&&`` / ``->`` / ``::``
        and cast-wrapped arguments still yield their symbols.
        """
        structured = self._structured_expression_symbols(expression_ref)
        if structured is not None:
            return dedupe_keep_order(
                exact_symbol(name) for name in structured if exact_symbol(name)
            )
        if not expression:
            return []
        normalized = _normalize_cpp_expression(str(expression))
        return dedupe_keep_order(
            exact_symbol(name) for name in source_expression_names(normalized)
        )

    def _wire_symbols(
        self, expression: str, expression_ref: Any = None
    ) -> list[str]:
        """Raw symbol names referenced by ``expression``, extracted after
        normalizing C++ syntax.

        The edge-wiring paths need the *raw* name (for edge ``role`` and for
        :meth:`_resolve_symbol_producer`, which matches accessor chains /
        struct vars / parameters on the un-normalized form). But extraction
        must run on the C++-normalized expression, or a ``::``-qualified RHS
        like ``matrix::Eulerf(-vehicle_local_position.heading)`` yields no
        names at all and the operation's real inputs — including logged
        signals — are silently dropped from the graph. Mirrors the
        normalization :meth:`_walk_symbols` already applies during the walk.
        """
        structured = self._structured_expression_symbols(expression_ref)
        if structured is not None:
            return structured
        if not expression:
            return []
        return dedupe_keep_order(
            source_expression_names(_normalize_cpp_expression(str(expression)))
        )

    def _read_before_write_target(
        self,
        target: str,
        expression: str,
        expression_ref: Any,
        assignment_operator: str,
    ) -> str:
        """Return ``target`` when the source write reads its prior value.

        Compound assignment implies the read. A plain assignment can also
        read the prior value (``value = choose(value, floor)``); in that case
        the RHS inputs extracted from source provide the proof. This source
        fact, rather than the operator spelling alone, determines whether
        producer lookup must stop before the current write site.
        """
        target_norm = exact_symbol(target)
        if not target_norm:
            return ""
        if str(assignment_operator or "=") != "=":
            return target
        if any(
            exact_symbol(symbol) == target_norm
            for symbol in self._wire_symbols(expression, expression_ref)
        ):
            return target
        return ""

    def _binding_read_before_write_target(
        self, binding: dict[str, Any]
    ) -> str:
        if binding.get("synthetic_pointer_output_binding") or binding.get(
            "synthetic_member_output_binding"
        ):
            return ""
        return self._read_before_write_target(
            str(binding.get("target_symbol") or ""),
            str(
                binding.get("source_symbol")
                or binding.get("expression")
                or ""
            ),
            binding.get("expression_ref"),
            str(binding.get("assignment_operator") or "="),
        )

    @staticmethod
    def _dedupe_bindings(
        bindings: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Preserve source order while de-duplicating binding objects."""
        seen: set[int] = set()
        result: list[dict[str, Any]] = []
        for binding in bindings:
            marker = id(binding)
            if marker in seen:
                continue
            seen.add(marker)
            result.append(binding)
        return result

    def _binding_formal_parameters(
        self, binding: dict[str, Any]
    ) -> list[str]:
        """Derive a binding's formals from loaded source structure."""
        direct = [
            str(value)
            for value in (binding.get("function_parameters") or [])
            if str(value)
        ]
        if direct:
            return direct
        callable_id = _base_callable_scope(
            self._binding_target_scope(binding)[1]
        )
        callable_record = self._source_structure.callables_by_id.get(callable_id)
        if callable_record:
            parameters = [
                str(value)
                for value in (callable_record.get("parameters") or [])
                if str(value)
            ]
            if parameters:
                return parameters
        for helper_key, helper in self.helper_index.items():
            if self._helper_callable_id(helper_key, helper) != callable_id:
                continue
            return [
                str(value)
                for value in (helper.get("parameters") or [])
                if str(value)
            ]
        return []

    def _binding_reads_formal(self, binding: dict[str, Any]) -> bool:
        """Whether a writer's value or reachability reads a formal."""
        formals = {
            exact_symbol(value)
            for value in self._binding_formal_parameters(binding)
            if exact_symbol(value)
        }
        if not formals:
            return False
        expressions: list[tuple[str, Any]] = [
            (
                str(
                    binding.get("source_symbol")
                    or binding.get("expression")
                    or ""
                ),
                binding.get("expression_ref"),
            )
        ]
        control_refs = binding.get("control_expression_refs") or []
        expressions.extend(
            (
                str(value),
                control_refs[position] if position < len(control_refs) else None,
            )
            for position, value in enumerate(
                binding.get("control_predicates") or []
            )
        )
        for expression, expression_ref in expressions:
            for symbol in self._wire_symbols(expression, expression_ref):
                root = symbol.replace("->", ".").split(".", 1)[0]
                if exact_symbol(root) in formals:
                    return True
        return False

    def _terminal_call_context_writers(
        self, writers: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Instantiate terminal writers through every source-proven caller path.

        Every source-proven invocation path receives a private callable scope,
        even when the writer does not read a formal. Arguments determine value
        aliases, but the call site itself determines whether the callee ran and
        therefore contributes caller reachability. Recursive source call paths
        stop at the repeated callable identity; no depth or call-count budget
        affects which acyclic paths are kept.
        """
        contextualized: list[dict[str, Any]] = []
        for writer in writers:
            writer_file, writer_callable = self._binding_target_scope(writer)
            base_callable = _base_callable_scope(writer_callable)
            if (
                _CALL_SCOPE_MARKER in writer_callable
                or not base_callable
            ):
                contextualized.append(writer)
                continue

            call_paths = self._caller_paths_to_callable(base_callable)
            if not call_paths:
                contextualized.append(writer)
                continue

            clones: list[dict[str, Any]] = []
            target = exact_symbol(
                str(writer.get("target_symbol") or writer.get("target") or "")
            )
            expression = str(
                writer.get("source_symbol") or writer.get("expression") or ""
            )
            writer_line = self._binding_target_line(writer)
            for call_path in call_paths:
                call_scope = ""
                for source_call, helper_key in call_path:
                    args = [
                        str(value).strip()
                        for value in (source_call.get("args") or [])
                    ]
                    source_caller = str(
                        source_call.get("callable_id")
                        or source_call.get("function")
                        or ""
                    )
                    caller_callable = call_scope or source_caller
                    runtime_call = dict(source_call)
                    runtime_call["callable_id"] = caller_callable
                    if _CALL_SCOPE_MARKER in caller_callable:
                        runtime_call["source_site_id"] = stable_id(
                            "invocation",
                            (caller_callable, self._call_site_id(source_call)),
                        )
                    helper = self.helper_index.get(helper_key) or {}
                    call_scope = self._register_call_instance(
                        runtime_call,
                        helper_key,
                        helper,
                        args,
                        caller_callable=caller_callable,
                    )
                    if not call_scope:
                        break
                if not call_scope:
                    continue
                clones.extend(
                    candidate
                    for candidate in self._targets_matching(target)
                    if self._binding_target_scope(candidate)
                    == (writer_file, call_scope)
                    and str(
                        candidate.get("source_symbol")
                        or candidate.get("expression")
                        or ""
                    )
                    == expression
                    and self._binding_target_line(candidate) == writer_line
                )
            contextualized.extend(clones or [writer])
        return self._dedupe_bindings(contextualized)

    def _source_calls_to_callable(
        self, callee_callable: str
    ) -> list[tuple[dict[str, Any], tuple[str, str]]]:
        """Return loaded call sites resolving to one exact callable."""
        return list(self._callers_by_callee.get(callee_callable, ()))

    def _index_source_call_edges(self) -> None:
        """Resolve each loaded call site once into the reverse call graph."""
        for source_call in self._call_statements:
            call_site_id = self._call_site_id(source_call)
            if call_site_id in self._resolved_helper_key_by_call_site:
                continue
            args = [str(value).strip() for value in source_call.get("args") or []]
            call_file = str(source_call.get("file") or "")
            caller_callable = str(
                source_call.get("callable_id")
                or source_call.get("function")
                or ""
            )
            resolved_callable_id = str(
                source_call.get("resolved_callable_id") or ""
            )
            exact_keys = [
                key
                for key, helper in self.helper_index.items()
                if resolved_callable_id
                and self._helper_callable_id(key, helper)
                == resolved_callable_id
            ]
            helper_key = exact_keys[0] if len(exact_keys) == 1 else None
            if helper_key is None:
                helper_key = self._pick_helper_key(
                    str(source_call.get("name") or ""),
                    call_file,
                    scope_function=caller_callable,
                    receiver=str(source_call.get("receiver") or ""),
                    receiver_type_hint=str(
                        source_call.get("receiver_type") or ""
                    ),
                    resolved_callable_id_hint=resolved_callable_id,
                    resolved_callable_file_hint=str(
                        source_call.get("resolved_callable_file") or ""
                    ),
                    resolved_callable_owner_hint=str(
                        source_call.get("resolved_callable_owner") or ""
                    ),
                    argument_count=len(args),
                    allow_provider=False,
                )
            if helper_key is None:
                continue
            helper = self.helper_index.get(helper_key) or {}
            callee_callable = self._helper_callable_id(helper_key, helper)
            self._resolved_helper_key_by_call_site[call_site_id] = helper_key
            entry = (source_call, helper_key)
            if entry not in self._callers_by_callee[callee_callable]:
                self._callers_by_callee[callee_callable].append(entry)
        self._caller_paths_cache.clear()

    def _caller_paths_to_callable(
        self,
        callee_callable: str,
        active: Optional[frozenset[str]] = None,
    ) -> list[list[tuple[dict[str, Any], tuple[str, str]]]]:
        """Return outermost-to-innermost acyclic call paths to ``callee``."""
        if active is None and callee_callable in self._caller_paths_cache:
            return [list(path) for path in self._caller_paths_cache[callee_callable]]
        root_request = active is None
        active = active or frozenset()
        if callee_callable in active:
            return []
        next_active = active | {callee_callable}
        paths: list[list[tuple[dict[str, Any], tuple[str, str]]]] = []
        for source_call, helper_key in self._source_calls_to_callable(
            callee_callable
        ):
            caller_callable = _base_callable_scope(
                str(
                    source_call.get("callable_id")
                    or source_call.get("function")
                    or ""
                )
            )
            outer_paths = (
                self._caller_paths_to_callable(caller_callable, next_active)
                if caller_callable and caller_callable not in next_active
                else []
            )
            call_entry = (source_call, helper_key)
            if outer_paths:
                paths.extend([*outer, call_entry] for outer in outer_paths)
            else:
                paths.append([call_entry])
        if root_request:
            self._caller_paths_cache[callee_callable] = [list(path) for path in paths]
        return paths

    @staticmethod
    def _call_argument_storage(
        argument: str, expression_ref: Any = None
    ) -> str:
        if hasattr(expression_ref, "model_dump"):
            expression_ref = expression_ref.model_dump(exclude_none=True)
        if isinstance(expression_ref, dict) and bool(
            expression_ref.get("exact")
        ):
            return exact_symbol(
                str(expression_ref.get("direct_storage") or "")
            )
        text = str(argument or "").strip()
        while text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        text = text.lstrip("&*").strip()
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*|\[[^\]]+\]|\(\))*",
            text,
        ):
            return ""
        return exact_symbol(text)

    @staticmethod
    def _storage_contains_reference(storage: str, reference: str) -> bool:
        """Whether an actual object can contain the referenced output.

        Pointer/reference calls receive aggregate storage (``out``) while
        source-proven effects commonly target a member (``out.alt``). This is
        directional: a field actual does not prove its enclosing aggregate.
        """
        storage_parts = exact_symbol(storage).split(".")
        reference_parts = exact_symbol(reference).split(".")
        if not storage_parts or len(storage_parts) > len(reference_parts):
            return False
        reference_prefix = ".".join(reference_parts[: len(storage_parts)])
        return symbol_produces_reference(
            ".".join(storage_parts), reference_prefix
        )

    def _helper_writes_receiver_member(
        self,
        helper_key: tuple[str, str],
        helper: dict[str, Any],
    ) -> bool:
        """Whether loaded source proves a member write by ``helper``."""
        helper_scope = (
            str(helper.get("file") or ""),
            self._helper_callable_id(helper_key, helper),
        )
        return any(
            self._binding_target_scope(binding) == helper_scope
            and (identity := self._binding_target_identity(binding)) is not None
            and identity.kind == "member"
            for binding in self._source_bindings
        )

    def _symbol_is_receiver_call_result(
        self,
        symbol_norm: str,
        scope: SourceReadScope,
    ) -> bool:
        """Whether ``symbol`` is the parser's identity for a method call.

        Expression-name extraction represents ``obj.read().field`` as
        ``obj.read``. That is a call result, not storage owned by ``obj``;
        helper expansion handles its return value separately.
        """
        scope_file, scope_callable = scope[:2]
        scope_line = scope[2] if len(scope) > 2 else None
        scope_order = scope[3] if len(scope) > 3 else None
        source_callable = _base_callable_scope(scope_callable)
        for call in self._call_statements:
            call_file = str(call.get("file") or "")
            caller = _base_callable_scope(
                str(call.get("callable_id") or call.get("function") or "")
            )
            if scope_file and call_file != scope_file:
                continue
            if source_callable and caller != source_callable:
                continue
            call_line = int(call.get("line") or 0) or None
            call_order = _record_source_order(call)
            if (
                scope_line is not None
                and call_file == scope_file
                and call_line is not None
                and (
                    call_line > scope_line
                    or (
                        call_line == scope_line
                        and scope_order is not None
                        and call_order is not None
                        and call_order > scope_order
                    )
                )
            ):
                continue
            receiver = str(call.get("receiver") or "")
            if not receiver:
                continue
            name = str(call.get("name") or "").rsplit("::", 1)[-1]
            if exact_symbol(f"{receiver}.{name}") == symbol_norm:
                return True
        return False

    def _writers_from_relevant_calls(
        self,
        symbol_norm: str,
        symbol_raw: str,
        scope: SourceReadScope,
        *,
        terminal: bool,
        origin_vertex_id: str = "",
    ) -> list[dict[str, Any]]:
        """Resolve calls whose source-proven effects can produce ``symbol``.

        Passing a value to a call does not make the call its producer. Loaded
        callees must prove either a write through a pointer/reference formal
        or a receiver-member write. Missing callees become typed discovery
        gaps instead of being fetched inline based only on textual overlap.
        """
        scope_file, scope_callable = scope[:2]
        scope_line = scope[2] if len(scope) > 2 else None
        scope_order = scope[3] if len(scope) > 3 else None
        source_callable = _base_callable_scope(scope_callable)
        if self._symbol_is_receiver_call_result(symbol_norm, scope):
            return []
        admitted = False
        for source_call in self._call_statements:
            call_file = str(source_call.get("file") or "")
            caller = _base_callable_scope(
                str(
                    source_call.get("callable_id")
                    or source_call.get("function")
                    or ""
                )
            )
            if scope_file and call_file != scope_file:
                continue
            if source_callable and caller != source_callable:
                continue
            call_line = int(source_call.get("line") or 0) or None
            call_order = _record_source_order(source_call)
            if (
                scope_line is not None
                and call_file == scope_file
                and call_line is not None
                and (
                    call_line > scope_line
                    or (
                        call_line == scope_line
                        and scope_order is not None
                        and call_order is not None
                        and call_order > scope_order
                    )
                )
            ):
                continue
            name = str(source_call.get("name") or "")
            if not name or source_call.get("evaluation_intrinsic"):
                continue
            receiver = exact_symbol(str(source_call.get("receiver") or ""))
            if receiver and self._match_parameter(
                f"{receiver}.{name.rsplit('::', 1)[-1]}"
            ) is not None:
                continue
            args = [str(arg).strip() for arg in (source_call.get("args") or [])]
            argument_refs = list(
                source_call.get("argument_expressions") or []
            )
            storages = [
                self._call_argument_storage(
                    str(arg),
                    argument_refs[index]
                    if index < len(argument_refs)
                    else None,
                )
                for index, arg in enumerate(args)
            ]
            possible_argument_match = any(
                storage
                and self._storage_contains_reference(storage, symbol_norm)
                for storage in storages
            )
            possible_receiver_match = bool(
                receiver
                and self._storage_contains_reference(receiver, symbol_norm)
            )
            if not possible_argument_match and not possible_receiver_match:
                continue
            call_reference = UnresolvedSourceReference(
                symbol=name,
                kind="callable",
                file=call_file,
                line=call_line,
                callable_id=caller,
                class_owner=self._source_structure.callable_owner(caller),
                receiver=str(source_call.get("receiver") or ""),
                argument_count=len(args),
                source_site_id=str(source_call.get("source_site_id") or ""),
            )
            if reference_receiver_is_source_boundary(
                call_reference,
                self._boundary_bindings,
                self._source_structure,
            ):
                continue
            helper_key = self._pick_helper_key(
                name,
                call_file,
                scope_function=scope_callable or caller,
                receiver=str(source_call.get("receiver") or ""),
                receiver_type_hint=str(
                    source_call.get("receiver_type") or ""
                ),
                resolved_callable_id_hint=str(
                    source_call.get("resolved_callable_id") or ""
                ),
                resolved_callable_file_hint=str(
                    source_call.get("resolved_callable_file") or ""
                ),
                resolved_callable_owner_hint=str(
                    source_call.get("resolved_callable_owner") or ""
                ),
                argument_count=len(args),
                allow_provider=False,
            )
            if helper_key is None:
                reference_key = self._record_unresolved(
                    name,
                    kind="callable",
                    file=call_file,
                    line=int(source_call.get("line") or 0) or None,
                    scope_function=scope_callable or caller,
                    source_expression=str(source_call.get("evidence") or ""),
                    receiver=str(source_call.get("receiver") or ""),
                    receiver_type=str(source_call.get("receiver_type") or ""),
                    resolved_callable_id=str(
                        source_call.get("resolved_callable_id") or ""
                    ),
                    resolved_callable_file=str(
                        source_call.get("resolved_callable_file") or ""
                    ),
                    resolved_callable_owner=str(
                        source_call.get("resolved_callable_owner") or ""
                    ),
                    argument_count=len(args),
                    source_site_id=str(source_call.get("source_site_id") or ""),
                    origin_vertex_id=origin_vertex_id,
                    origin_operand=symbol_raw,
                )
                if terminal:
                    self._terminal_reference_keys.add(reference_key)
                continue
            helper = self.helper_index.get(helper_key) or {}
            output_positions = set(_pointer_param_positions(helper).values())
            output_argument_match = any(
                index in output_positions
                and storage
                and self._storage_contains_reference(storage, symbol_norm)
                for index, storage in enumerate(storages)
            )
            receiver_output_match = bool(
                possible_receiver_match
                and self._helper_writes_receiver_member(helper_key, helper)
            )
            if not output_argument_match and not receiver_output_match:
                continue
            runtime_call = dict(source_call)
            runtime_call["callable_id"] = scope_callable or caller
            if _CALL_SCOPE_MARKER in scope_callable:
                runtime_call["source_site_id"] = stable_id(
                    "invocation",
                    (scope_callable, self._call_site_id(source_call)),
                )
            self._register_call_instance(
                runtime_call,
                helper_key,
                helper,
                args,
                caller_callable=scope_callable or caller,
            )
            admitted = True
        if not admitted:
            return []
        if terminal:
            return self._writers_of(symbol_norm)
        return self._scoped_writers(symbol_norm, symbol_raw, scope)

    @staticmethod
    def _call_site_id(call: dict[str, Any]) -> str:
        explicit = str(call.get("source_site_id") or "")
        if explicit:
            return explicit
        return stable_id(
            "callsite",
            (
                str(call.get("file") or ""),
                _base_callable_scope(
                    str(call.get("callable_id") or call.get("function") or "")
                ),
                int(call.get("line") or 0),
                str(call.get("name") or ""),
                str(call.get("receiver") or ""),
                tuple(str(arg).strip() for arg in (call.get("args") or [])),
            ),
        )

    def _identity_in_call_scope(
        self,
        raw: Any,
        source_callable: str,
        call_scope: str,
        *,
        call: Optional[dict[str, Any]] = None,
        caller_callable: str = "",
    ) -> Any:
        if not raw:
            return raw
        try:
            identity = SourceSymbolIdentity.model_validate(raw)
        except (TypeError, ValueError):
            return raw
        if identity.kind == "local" and identity.callable_id == source_callable:
            identity = identity.model_copy(update={"callable_id": call_scope})
        if (
            identity.kind == "member"
            and call is not None
            and str(call.get("receiver_access") or "unknown") == "value"
        ):
            receiver = exact_symbol(str(call.get("receiver") or ""))
            if receiver and receiver != "this":
                raw_receiver_identity = call.get("receiver_identity")
                try:
                    receiver_identity = SourceSymbolIdentity.model_validate(
                        raw_receiver_identity
                    )
                except (TypeError, ValueError):
                    receiver_identity = self._source_structure.symbol_identity(
                        receiver,
                        file=str(call.get("file") or ""),
                        callable_id=_base_callable_scope(caller_callable),
                        function_name=str(call.get("function") or ""),
                    )
                if receiver_identity.declaration_proven:
                    member_symbol = identity.symbol.replace("this.", "")
                    composite_symbol = f"{receiver}.{member_symbol}"
                    composite_declaration = (
                        f"{receiver_identity.declaration_id}::subobject::"
                        f"{identity.declaration_id or identity.declaring_class}"
                    )
                    composite_owner = (
                        f"{receiver_identity.declaration_id}::subobject::"
                        f"{identity.declaring_class or identity.class_owner}"
                    )
                    identity = SourceSymbolIdentity(
                        kind=receiver_identity.kind,
                        symbol=composite_symbol,
                        root=receiver_identity.root,
                        file=str(call.get("file") or receiver_identity.file),
                        callable_id=(
                            _base_callable_scope(caller_callable)
                            if receiver_identity.kind == "local"
                            else ""
                        ),
                        class_owner=receiver_identity.class_owner,
                        declaring_class=(
                            composite_owner
                            if receiver_identity.kind == "member"
                            else receiver_identity.declaring_class
                        ),
                        namespace_owner=receiver_identity.namespace_owner,
                        declaration_id=composite_declaration,
                        declaration_proven=True,
                    )
        return identity.model_dump()

    def _expression_ref_in_call_scope(
        self,
        raw: Any,
        source_callable: str,
        call_scope: str,
        *,
        call: dict[str, Any],
        caller_callable: str,
    ) -> Any:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(exclude_none=True)
        if not isinstance(raw, dict):
            return raw
        record = dict(raw)
        record["input_identities"] = {
            symbol: self._identity_in_call_scope(
                identity,
                source_callable,
                call_scope,
                call=call,
                caller_callable=caller_callable,
            )
            for symbol, identity in (record.get("input_identities") or {}).items()
        }
        return record

    def _register_call_instance(
        self,
        call: dict[str, Any],
        helper_key: tuple[str, str],
        helper: dict[str, Any],
        args: Sequence[str],
        *,
        caller_callable: str,
    ) -> str:
        helper_file = str(helper.get("file") or "")
        helper_callable = self._helper_callable_id(helper_key, helper)
        source_site_id = self._call_site_id(call)
        call_scope = _callable_instance_scope(
            helper_callable,
            caller_callable=caller_callable,
            source_site_id=source_site_id,
        )
        explicit_count = len(args)
        effective_args = callable_arguments_with_defaults(helper, args)
        if effective_args is None:
            return ""
        args = effective_args
        if call_scope in self._registered_call_instances:
            return call_scope
        self._registered_call_instances.add(call_scope)

        caller_file = str(call.get("file") or "")
        caller_line = int(call.get("line") or 0)
        caller_predicates = [
            str(value) for value in (call.get("control_predicates") or [])
        ]
        caller_predicate_lines = list(call.get("control_predicate_lines") or [])
        caller_predicate_sites = list(
            call.get("control_predicate_site_ids") or []
        )
        caller_predicate_orders = list(
            call.get("control_predicate_orders") or []
        )
        caller_predicate_refs = list(
            call.get("control_expression_refs") or []
        )
        argument_refs = list(call.get("argument_expressions") or [])
        instance_bindings: list[dict[str, Any]] = []

        # Clone source-definition bindings into this invocation scope. The
        # source locations remain the callee's; only lexical/runtime identity
        # changes. Caller reachability is conjoined with callee reachability.
        for original in self._source_bindings:
            if self._binding_target_scope(original) != (
                helper_file,
                helper_callable,
            ):
                continue
            clone = dict(original)
            callee_predicates = [
                str(value)
                for value in (original.get("control_predicates") or [])
            ]
            callee_lines = list(original.get("control_predicate_lines") or [])
            callee_sites = list(
                original.get("control_predicate_site_ids") or []
            )
            callee_orders = list(
                original.get("control_predicate_orders") or []
            )
            callee_refs = list(original.get("control_expression_refs") or [])
            clone.update(
                {
                    "scope_file": helper_file,
                    "scope_function": call_scope,
                    "expression_scope_file": helper_file,
                    "expression_scope_function": call_scope,
                    "control_scope_file": helper_file,
                    "control_scope_function": call_scope,
                    "control_predicates": [
                        *caller_predicates,
                        *callee_predicates,
                    ],
                    "control_predicate_lines": [
                        *caller_predicate_lines,
                        *callee_lines,
                    ],
                    "control_predicate_site_ids": [
                        *caller_predicate_sites,
                        *callee_sites,
                    ],
                    "control_predicate_orders": [
                        *caller_predicate_orders,
                        *callee_orders,
                    ],
                    "control_expression_refs": [
                        *caller_predicate_refs,
                        *callee_refs,
                    ],
                    "control_predicate_files": [
                        *(caller_file for _ in caller_predicates),
                        *(helper_file for _ in callee_predicates),
                    ],
                    "control_predicate_callables": [
                        *(caller_callable for _ in caller_predicates),
                        *(call_scope for _ in callee_predicates),
                    ],
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    )
                    and bool(original.get("reachability_exact", True)),
                    "invocation_control_predicate_count": len(
                        caller_predicates
                    ),
                    "call_site_id": source_site_id,
                    "call_instance_scope": call_scope,
                    "callable_id": call_scope,
                }
            )
            clone["target_identity"] = self._identity_in_call_scope(
                original.get("target_identity"),
                helper_callable,
                call_scope,
                call=call,
                caller_callable=caller_callable,
            )
            clone["reference_identities"] = {
                symbol: self._identity_in_call_scope(
                    identity,
                    helper_callable,
                    call_scope,
                    call=call,
                    caller_callable=caller_callable,
                )
                for symbol, identity in (
                    original.get("reference_identities") or {}
                ).items()
            }
            self._register_synthetic_binding(clone)
            instance_bindings.append(clone)

        self._register_receiver_member_inputs(
            call,
            helper,
            instance_bindings,
            caller_callable=caller_callable,
            call_file=caller_file,
            call_line=caller_line,
            call_scope=call_scope,
        )

        formals = [str(value) for value in (helper.get("parameters") or [])]
        output_formals = {
            str(write.get("param") or "")
            for write in (helper.get("pointer_output_writes") or [])
        }
        for index, (formal, actual) in enumerate(zip(formals, args)):
            if not exact_symbol(formal):
                continue
            if formal in output_formals and not self._helper_reads_formal(
                helper_callable, helper, formal
            ):
                continue
            formal_identity = self._source_structure.symbol_identity(
                formal,
                file=helper_file,
                callable_id=helper_callable,
                function_name=str(helper.get("name") or ""),
                function_parameters=formals,
            )
            uses_default = index >= explicit_count
            expression_file = helper_file if uses_default else caller_file
            expression_callable = call_scope if uses_default else caller_callable
            expression_line = (
                int(helper.get("line") or 0) if uses_default else caller_line
            )
            self._register_synthetic_binding(
                {
                    "target_symbol": formal,
                    "source_symbol": actual,
                    "assignment_operator": "=",
                    "expression_ref": (
                        argument_refs[index]
                        if index < explicit_count and index < len(argument_refs)
                        else None
                    ),
                    "assignment_path": [
                        {
                            "file": expression_file,
                            "line": expression_line,
                            "expression": actual,
                        }
                    ],
                    "logged_signal": "",
                    "control_predicates": caller_predicates,
                    "control_predicate_lines": caller_predicate_lines,
                    "control_predicate_site_ids": caller_predicate_sites,
                    "control_predicate_orders": caller_predicate_orders,
                    "control_expression_refs": caller_predicate_refs,
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    ),
                    "invocation_control_predicate_count": len(
                        caller_predicates
                    ),
                    "struct_variables": {},
                    "scope_file": helper_file,
                    "scope_function": call_scope,
                    "scope_line": int(helper.get("line") or 0),
                    "scope_source_order": 0,
                    "expression_scope_file": expression_file,
                    "expression_scope_function": expression_callable,
                    "expression_scope_source_order": (
                        _record_source_order(call)
                        if not uses_default
                        else _record_source_order(helper)
                    ),
                    "control_scope_file": caller_file,
                    "control_scope_function": caller_callable,
                    "function": str(call.get("function") or ""),
                    "callable_id": caller_callable,
                    "target_identity": formal_identity.model_copy(
                        update={"callable_id": call_scope}
                    ).model_dump(),
                    "synthetic_call_binding": True,
                    "call_site_id": source_site_id,
                    "call_instance_scope": call_scope,
                }
            )
        self._register_helper_return_bindings(
            call,
            helper_key,
            helper,
            caller_callable=caller_callable,
            caller_file=caller_file,
            call_scope=call_scope,
        )
        self._register_pointer_output_call(
            call,
            helper_key,
            helper,
            args,
            caller_callable=caller_callable,
            call_file=caller_file,
            call_line=caller_line,
            call_scope=call_scope,
        )
        self._register_member_output_calls(
            call,
            helper,
            instance_bindings,
            caller_callable=caller_callable,
            call_file=caller_file,
            call_line=caller_line,
            call_scope=call_scope,
        )
        return call_scope

    def _register_helper_return_bindings(
        self,
        call: dict[str, Any],
        helper_key: tuple[str, str],
        helper: dict[str, Any],
        *,
        caller_callable: str,
        caller_file: str,
        call_scope: str,
    ) -> None:
        """Register each source return as a writer in this call instance."""
        helper_file = str(helper.get("file") or "")
        helper_callable = self._helper_callable_id(helper_key, helper)
        caller_predicates = [
            str(value) for value in (call.get("control_predicates") or [])
        ]
        caller_lines = list(call.get("control_predicate_lines") or [])
        caller_sites = list(call.get("control_predicate_site_ids") or [])
        caller_orders = list(call.get("control_predicate_orders") or [])
        caller_refs = list(call.get("control_expression_refs") or [])
        return_identity = SourceSymbolIdentity(
            kind="local",
            symbol="__return__",
            root="__return__",
            file=helper_file,
            callable_id=call_scope,
            class_owner=str(helper.get("owner") or ""),
            declaration_id=f"{helper_callable}:return",
            declaration_proven=True,
        )
        self._reference_identities_by_scope[
            (helper_file, call_scope, "__return__")
        ].append(return_identity)
        self._reference_identities_by_callable[
            (call_scope, "__return__")
        ].append(return_identity)
        for site in _helper_source_return_sites(helper):
            expression = str(site.get("expression") or "").strip()
            if not expression:
                continue
            return_file = str(site.get("file") or helper_file)
            return_line = int(site.get("line") or helper.get("line") or 0)
            return_predicates = [
                str(value)
                for value in (site.get("control_predicates") or [])
            ]
            return_lines = list(site.get("control_predicate_lines") or [])
            return_sites = list(site.get("control_predicate_site_ids") or [])
            return_orders = list(site.get("control_predicate_orders") or [])
            return_refs = [
                self._expression_ref_in_call_scope(
                    value,
                    helper_callable,
                    call_scope,
                    call=call,
                    caller_callable=caller_callable,
                )
                for value in (site.get("control_expression_refs") or [])
            ]
            expression_ref = self._expression_ref_in_call_scope(
                site.get("expression_ref"),
                helper_callable,
                call_scope,
                call=call,
                caller_callable=caller_callable,
            )
            self._register_synthetic_binding(
                {
                    "target_symbol": "__return__",
                    "source_symbol": expression,
                    "assignment_operator": "=",
                    "assignment_path": [
                        {
                            "file": return_file,
                            "line": return_line,
                            "expression": expression,
                        }
                    ],
                    "logged_signal": "",
                    "control_predicates": [
                        *caller_predicates,
                        *return_predicates,
                    ],
                    "control_predicate_lines": [
                        *caller_lines,
                        *return_lines,
                    ],
                    "control_predicate_site_ids": [
                        *caller_sites,
                        *return_sites,
                    ],
                    "control_predicate_orders": [
                        *caller_orders,
                        *return_orders,
                    ],
                    "control_expression_refs": [
                        *caller_refs,
                        *return_refs,
                    ],
                    "control_predicate_files": [
                        *(caller_file for _ in caller_predicates),
                        *(return_file for _ in return_predicates),
                    ],
                    "control_predicate_callables": [
                        *(caller_callable for _ in caller_predicates),
                        *(call_scope for _ in return_predicates),
                    ],
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    )
                    and bool(site.get("reachability_exact", False)),
                    "scope_file": helper_file,
                    "scope_function": call_scope,
                    "scope_line": return_line,
                    "scope_source_order": _record_source_order(site),
                    "expression_scope_file": return_file,
                    "expression_scope_function": call_scope,
                    "expression_scope_line": return_line,
                    "expression_scope_source_order": _record_source_order(site),
                    "control_scope_file": return_file,
                    "control_scope_function": call_scope,
                    "function": str(helper.get("name") or ""),
                    "callable_id": call_scope,
                    "target_identity": return_identity.model_dump(),
                    "expression_ref": expression_ref,
                    "reference_identities": dict(
                        (expression_ref or {}).get("input_identities") or {}
                    )
                    if isinstance(expression_ref, dict)
                    else {},
                    "synthetic_helper_return_binding": True,
                    "source_site_id": str(site.get("source_site_id") or ""),
                    "source_order": _record_source_order(site),
                    "call_site_id": self._call_site_id(call),
                    "call_instance_scope": call_scope,
                    "provenance": (
                        f"helper_return:{helper_key[0]}@{helper_key[1]}"
                    ),
                }
            )

    def _register_helper_return_projection(
        self,
        helper: dict[str, Any],
        call_scope: str,
        result_path: str,
    ) -> None:
        """Materialize a demanded helper-result field inside its call scope."""
        clean_path = str(result_path or "").strip().lstrip(".")
        if not clean_path:
            return
        requested = (
            f"__return__{clean_path}"
            if clean_path.startswith("[")
            else f"__return__.{clean_path}"
        )
        helper_file = str(helper.get("file") or "")
        base_returns = [
            binding
            for binding in self._targets_matching("__return__")
            if self._binding_target_scope(binding)
            == (helper_file, call_scope)
        ]
        self._project_source_writers(base_returns, requested)

    def _register_receiver_member_inputs(
        self,
        call: dict[str, Any],
        helper: dict[str, Any],
        instance_bindings: Sequence[dict[str, Any]],
        *,
        caller_callable: str,
        call_file: str,
        call_line: int,
        call_scope: str,
    ) -> None:
        """Bind callee member reads to the exact receiver subobject state."""
        receiver = exact_symbol(str(call.get("receiver") or ""))
        if not receiver or receiver == "this":
            return
        helper_file = str(helper.get("file") or "")
        read_identities: dict[str, SourceSymbolIdentity] = {}
        for binding in instance_bindings:
            for symbol, raw_identity in (
                binding.get("reference_identities") or {}
            ).items():
                try:
                    identity = SourceSymbolIdentity.model_validate(raw_identity)
                except (TypeError, ValueError):
                    continue
                if identity.declaration_proven and exact_symbol(identity.symbol).startswith(
                    f"{receiver}."
                ):
                    read_identities.setdefault(exact_symbol(str(symbol)), identity)

        helper_refs: list[Any] = [
            site.get("expression_ref")
            for site in _helper_source_return_sites(helper)
        ]
        helper_callable = _base_callable_scope(call_scope)
        for expression_ref in helper_refs:
            record = (
                expression_ref.model_dump(exclude_none=True)
                if hasattr(expression_ref, "model_dump")
                else expression_ref
            )
            if not isinstance(record, dict):
                continue
            for symbol, raw_identity in (
                record.get("input_identities") or {}
            ).items():
                transformed = self._identity_in_call_scope(
                    raw_identity,
                    helper_callable,
                    call_scope,
                    call=call,
                    caller_callable=caller_callable,
                )
                try:
                    identity = SourceSymbolIdentity.model_validate(transformed)
                except (TypeError, ValueError):
                    continue
                if identity.declaration_proven and exact_symbol(identity.symbol).startswith(
                    f"{receiver}."
                ):
                    read_identities.setdefault(exact_symbol(str(symbol)), identity)

        for member_symbol, identity in read_identities.items():
            source_symbol = exact_symbol(identity.symbol)
            expression_ref = {
                "text": source_symbol,
                "lowered_text": source_symbol,
                "input_symbols": [source_symbol],
                "input_identities": {source_symbol: identity.model_dump()},
                "call_results": [],
                "exact": True,
            }
            self._register_synthetic_binding(
                {
                    "target_symbol": member_symbol,
                    "source_symbol": source_symbol,
                    "assignment_path": [
                        {
                            "file": call_file,
                            "line": call_line,
                            "expression": source_symbol,
                        }
                    ],
                    "logged_signal": "",
                    "control_predicates": list(
                        call.get("control_predicates") or []
                    ),
                    "control_predicate_lines": list(
                        call.get("control_predicate_lines") or []
                    ),
                    "control_predicate_site_ids": list(
                        call.get("control_predicate_site_ids") or []
                    ),
                    "control_predicate_orders": list(
                        call.get("control_predicate_orders") or []
                    ),
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    ),
                    "scope_file": helper_file,
                    "scope_function": call_scope,
                    "scope_line": 0,
                    "expression_scope_file": call_file,
                    "expression_scope_function": caller_callable,
                    "expression_scope_line": call_line,
                    "expression_scope_source_order": _record_source_order(call),
                    "source_site_id": self._call_site_id(call),
                    "source_order": _record_source_order(call),
                    "target_identity": identity.model_dump(),
                    "expression_ref": expression_ref,
                    "reference_identities": {
                        source_symbol: identity.model_dump(),
                        member_symbol: identity.model_dump(),
                    },
                    "synthetic_call_binding": True,
                    "synthetic_receiver_input_binding": True,
                    "call_site_id": self._call_site_id(call),
                    "call_instance_scope": call_scope,
                }
            )
            self._reference_identities_by_scope[
                (helper_file, call_scope, member_symbol)
            ].append(identity)
            self._reference_identities_by_callable[
                (call_scope, member_symbol)
            ].append(identity)

    def _helper_reads_formal(
        self, helper_callable: str, helper: dict[str, Any], formal: str
    ) -> bool:
        """Whether source facts prove a value read through ``formal``."""
        for binding in self._source_bindings:
            if self._binding_target_scope(binding)[1] != helper_callable:
                continue
            texts: list[tuple[str, Any]] = [
                (
                    str(
                        binding.get("source_symbol")
                        or binding.get("expression")
                        or ""
                    ),
                    binding.get("expression_ref"),
                )
            ]
            control_refs = binding.get("control_expression_refs") or []
            texts.extend(
                (
                    str(value),
                    control_refs[position]
                    if position < len(control_refs)
                    else None,
                )
                for position, value in enumerate(
                    binding.get("control_predicates") or []
                )
            )
            for text, expression_ref in texts:
                for symbol in self._wire_symbols(text, expression_ref):
                    root = symbol.replace("->", ".").split(".", 1)[0]
                    if exact_symbol(root) == exact_symbol(formal):
                        return True
        return_texts: list[tuple[str, Any]] = [
            (str(site.get("expression") or ""), site.get("expression_ref"))
            for site in _helper_source_return_sites(helper)
        ]
        return any(
            any(
                exact_symbol(symbol.replace("->", ".").split(".", 1)[0])
                == exact_symbol(formal)
                for symbol in self._wire_symbols(text, expression_ref)
            )
            for text, expression_ref in return_texts
        )

    def _register_member_output_calls(
        self,
        call: dict[str, Any],
        helper: dict[str, Any],
        instance_bindings: Sequence[dict[str, Any]],
        *,
        caller_callable: str,
        call_file: str,
        call_line: int,
        call_scope: str,
    ) -> None:
        """Expose source-proven receiver-member writes at the call site."""
        receiver = str(call.get("receiver") or "").replace("->", ".").strip()
        if not receiver:
            # An unqualified same-object call writes the same member storage
            # already represented by the callee's source binding. Projecting
            # it back onto itself duplicates the entire callee once per caller
            # and adds no storage identity information.
            return
        helper_callable = _base_callable_scope(call_scope)
        helper_file = str(helper.get("file") or "")
        formals = [str(value) for value in (helper.get("parameters") or [])]
        seen: set[tuple[str, str]] = set()
        for binding in instance_bindings:
            target = str(
                binding.get("target_symbol") or binding.get("target") or ""
            )
            if not target:
                continue
            identity = self._binding_target_identity(binding)
            if identity is None:
                identity = self._source_structure.symbol_identity(
                    target,
                    file=helper_file,
                    callable_id=helper_callable,
                    function_name=str(helper.get("name") or ""),
                    function_parameters=formals,
                )
            if identity.kind != "member":
                continue
            member_path = target.replace("->", ".")
            if member_path.startswith("this."):
                member_path = member_path[5:]
            caller_target = (
                f"{receiver}.{member_path}" if receiver else member_path
            )
            key = (exact_symbol(caller_target), exact_symbol(target))
            if key in seen:
                continue
            seen.add(key)
            target_identity = identity.model_copy(
                update={"symbol": exact_symbol(caller_target)}
            )
            expression_ref = {
                "text": target,
                "lowered_text": target,
                "input_symbols": [target],
                "input_identities": {target: identity.model_dump()},
                "call_results": [],
                "exact": True,
            }
            self._register_synthetic_binding(
                {
                    "target_symbol": caller_target,
                    "source_symbol": target,
                    "assignment_path": [
                        {
                            "file": call_file,
                            "line": call_line,
                            "expression": target,
                        }
                    ],
                    "logged_signal": "",
                    "control_predicates": list(
                        call.get("control_predicates") or []
                    ),
                    "control_predicate_lines": list(
                        call.get("control_predicate_lines") or []
                    ),
                    "control_predicate_site_ids": list(
                        call.get("control_predicate_site_ids") or []
                    ),
                    "control_predicate_orders": list(
                        call.get("control_predicate_orders") or []
                    ),
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    ),
                    "struct_variables": {},
                    "scope_file": call_file,
                    "scope_function": caller_callable,
                    "scope_line": call_line,
                    "scope_source_order": _record_source_order(call),
                    "expression_scope_file": helper_file,
                    "expression_scope_function": call_scope,
                    "expression_scope_unordered": True,
                    "control_scope_file": call_file,
                    "control_scope_function": caller_callable,
                    "function": str(call.get("function") or ""),
                    "callable_id": caller_callable,
                    "target_identity": target_identity.model_dump(),
                    "expression_ref": expression_ref,
                    "reference_identities": {target: identity.model_dump()},
                    "synthetic_member_output_binding": True,
                    "call_site_id": self._call_site_id(call),
                    "source_site_id": self._call_site_id(call),
                    "source_order": _record_source_order(call),
                    "call_instance_scope": call_scope,
                }
            )

    def _register_synthetic_binding(self, binding: dict[str, Any]) -> None:
        target = exact_symbol(
            str(binding.get("target_symbol") or binding.get("target") or "")
        )
        if not target:
            return
        if not binding.get("reference_identities"):
            expression_ref = binding.get("expression_ref") or {}
            if hasattr(expression_ref, "model_dump"):
                expression_ref = expression_ref.model_dump(exclude_none=True)
            if isinstance(expression_ref, dict):
                binding["reference_identities"] = {
                    exact_symbol(str(symbol)): dict(identity)
                    for symbol, identity in (
                        expression_ref.get("input_identities") or {}
                    ).items()
                    if exact_symbol(str(symbol)) and isinstance(identity, dict)
                }
        self._all_bindings.append(binding)
        self._index_binding(
            self._by_target, self._target_shapes, target, binding
        )
        self._index_reference_identities(binding)

    @staticmethod
    def _source_subobject_suffix(target: str, reference: str) -> str:
        target_exact = exact_symbol(target)
        reference_exact = exact_symbol(reference)
        if (
            target_exact == reference_exact
            or not source_storage_produces_reference(
                target_exact, reference_exact
            )
        ):
            return ""
        return reference_exact[len(target_exact) :]

    @staticmethod
    def _append_result_path(path: str, suffix: str) -> str:
        clean_suffix = str(suffix or "")
        if not clean_suffix:
            return str(path or "")
        if path:
            return f"{path}{clean_suffix}"
        return clean_suffix.lstrip(".")

    @staticmethod
    def _project_identity(raw: Any, projected_symbol: str) -> Any:
        if not raw:
            return raw
        try:
            identity = SourceSymbolIdentity.model_validate(raw)
        except (TypeError, ValueError):
            return raw
        return identity.model_copy(
            update={"symbol": exact_symbol(projected_symbol)}
        ).model_dump()

    def _project_binding_to_reference(
        self,
        binding: dict[str, Any],
        reference: str,
    ) -> Optional[dict[str, Any]]:
        """Derive one explicit source-subobject writer from an aggregate copy.

        The parser proves whether an assignment's whole RHS is a storage copy
        or a call result. Only those two structural cases can carry a field or
        element demand backwards without guessing expression semantics.
        """
        target = exact_symbol(
            str(binding.get("target_symbol") or binding.get("target") or "")
        )
        requested = exact_symbol(reference)
        if target == requested:
            return binding
        suffix = self._source_subobject_suffix(target, requested)
        if not suffix or str(binding.get("assignment_operator") or "=") != "=":
            return None
        cache_key = (id(binding), requested)
        cached = self._projected_bindings.get(cache_key)
        if cached is not None:
            return cached

        raw_ref = binding.get("expression_ref")
        if hasattr(raw_ref, "model_dump"):
            raw_ref = raw_ref.model_dump(exclude_none=True)
        expression_ref = dict(raw_ref) if isinstance(raw_ref, dict) else {}
        if not bool(expression_ref.get("exact")):
            return None

        direct_storage = exact_symbol(
            str(expression_ref.get("direct_storage") or "")
        )
        direct_call = expression_ref.get("direct_call_result")
        projected_expression = ""
        projected_ref = dict(expression_ref)
        reference_identities: dict[str, Any] = {}
        if direct_storage:
            projected_expression = f"{direct_storage}{suffix}"
            raw_identities = dict(expression_ref.get("input_identities") or {})
            raw_identity = raw_identities.get(direct_storage)
            projected_identity = self._project_identity(
                raw_identity, projected_expression
            )
            projected_ref.update(
                {
                    "text": projected_expression,
                    "lowered_text": projected_expression,
                    "input_symbols": [projected_expression],
                    "input_identities": (
                        {projected_expression: projected_identity}
                        if projected_identity
                        else {}
                    ),
                    "call_results": [],
                    "direct_storage": projected_expression,
                    "direct_call_result": None,
                }
            )
            if projected_identity:
                reference_identities[projected_expression] = projected_identity
        elif isinstance(direct_call, dict):
            projected_call = dict(direct_call)
            projected_call["result_path"] = self._append_result_path(
                str(projected_call.get("result_path") or ""), suffix
            )
            call_text = str(projected_call.get("text") or "").strip()
            call_text = call_text.lstrip("&*").strip()
            if not call_text:
                return None
            projected_expression = f"{call_text}{suffix}"
            projected_ref.update(
                {
                    "text": projected_expression,
                    "lowered_text": projected_expression,
                    "input_symbols": [],
                    "input_identities": {},
                    "call_results": [projected_call],
                    "direct_storage": None,
                    "direct_call_result": projected_call,
                }
            )
        else:
            return None

        clone = dict(binding)
        clone.update(
            {
                "target_symbol": requested,
                "source_symbol": projected_expression,
                "expression_ref": projected_ref,
                "reference_identities": reference_identities,
                "target_identity": self._project_identity(
                    binding.get("target_identity"), requested
                ),
                "synthetic_source_projection": True,
                "projection_of_target": target,
                "_projection_origin_token": binding.get(
                    "_projection_origin_token", id(binding)
                ),
                "source_site_id": stable_id(
                    "source-projection",
                    (
                        str(binding.get("source_site_id") or ""),
                        target,
                        requested,
                    ),
                ),
            }
        )
        assignment_path = [
            dict(item) for item in (binding.get("assignment_path") or [])
        ]
        if assignment_path:
            assignment_path[0]["expression"] = projected_expression
            clone["assignment_path"] = assignment_path
        self._projected_bindings[cache_key] = clone
        self._register_synthetic_binding(clone)
        return clone

    def _project_source_writers(
        self,
        writers: Sequence[dict[str, Any]],
        reference: str,
    ) -> list[dict[str, Any]]:
        projected: list[dict[str, Any]] = []
        most_specific: dict[Any, dict[str, Any]] = {}
        for binding in writers:
            origin = binding.get("_projection_origin_token", id(binding))
            prior = most_specific.get(origin)
            target = exact_symbol(
                str(binding.get("target_symbol") or binding.get("target") or "")
            )
            prior_target = exact_symbol(
                str(
                    (prior or {}).get("target_symbol")
                    or (prior or {}).get("target")
                    or ""
                )
            )
            if prior is None or len(target) > len(prior_target):
                most_specific[origin] = binding
        for binding in most_specific.values():
            candidate = self._project_binding_to_reference(binding, reference)
            if candidate is not None and candidate not in projected:
                projected.append(candidate)
        return projected

    def _index_reference_identities(self, binding: dict[str, Any]) -> None:
        """Index source-proven identities where an expression reads them."""
        site_file, site_callable = self._binding_site_scope(binding)
        site_line = self._binding_walk_scope(binding)[2] or 0
        for raw_symbol, raw_identity in (
            binding.get("reference_identities") or {}
        ).items():
            try:
                identity = SourceSymbolIdentity.model_validate(raw_identity)
            except (TypeError, ValueError):
                continue
            symbol = exact_symbol(str(raw_symbol))
            if not symbol:
                continue
            self._reference_identities_by_scope[
                (site_file, site_callable, symbol)
            ].append(identity)
            self._reference_identities_by_site[
                (site_file, site_callable, site_line, symbol)
            ].append(identity)
            self._reference_identities_by_file[(site_file, symbol)].append(identity)
            self._reference_identities_by_callable[
                (site_callable, symbol)
            ].append(identity)
            self._reference_identities_by_symbol[symbol].append(identity)

    def _register_pointer_output_call(
        self,
        call: dict[str, Any],
        helper_key: tuple[str, str],
        helper: dict[str, Any],
        args: Sequence[str],
        *,
        caller_callable: str,
        call_file: str,
        call_line: int,
        call_scope: str,
    ) -> None:
        pointer_writes = list(helper.get("pointer_output_writes") or [])
        pointer_params = _pointer_param_positions(helper)
        if not pointer_writes or not pointer_params:
            return
        helper_file = str(helper.get("file") or "")
        argument_refs = list(call.get("argument_expressions") or [])
        actual_storages = [
            self._call_argument_storage(
                str(argument),
                argument_refs[index]
                if index < len(argument_refs)
                else None,
            )
            for index, argument in enumerate(args)
        ]
        substituted = derive_pointer_output_bindings(
            pointer_writes, pointer_params, actual_storages
        )
        seen_effects: set[tuple[str, str]] = set()
        for entry in substituted:
            formal_target = (
                f"{entry['param']}.{entry['field']}"
                if entry["field"]
                else entry["param"]
            )
            effect_key = (exact_symbol(entry["target"]), exact_symbol(formal_target))
            if effect_key in seen_effects:
                continue
            seen_effects.add(effect_key)
            predicates = list(call.get("control_predicates") or [])
            predicate_lines = list(call.get("control_predicate_lines") or [])
            predicate_sites = list(
                call.get("control_predicate_site_ids") or []
            )
            predicate_orders = list(
                call.get("control_predicate_orders") or []
            )
            target_identity = self._source_structure.symbol_identity(
                entry["target"],
                file=call_file,
                callable_id=caller_callable,
                function_name=str(call.get("function") or ""),
            )
            self._register_synthetic_binding(
                {
                    "target_symbol": entry["target"],
                    # The actual output is produced by the helper's formal
                    # field write. Keeping that intermediate operation makes
                    # callee-local data and controls explicit in the DAG.
                    "source_symbol": formal_target,
                    "assignment_path": [
                        {
                            "file": call_file,
                            "line": call_line,
                            "expression": formal_target,
                        }
                    ],
                    "logged_signal": "",
                    "control_predicates": predicates,
                    "control_predicate_lines": predicate_lines,
                    "control_predicate_site_ids": predicate_sites,
                    "control_predicate_orders": predicate_orders,
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    ),
                    "struct_variables": {},
                    "scope_file": call_file,
                    "scope_function": caller_callable,
                    "scope_line": call_line,
                    "scope_source_order": _record_source_order(call),
                    "expression_scope_file": helper_file,
                    "expression_scope_function": call_scope,
                    "expression_scope_unordered": True,
                    "control_scope_file": call_file,
                    "control_scope_function": caller_callable,
                    "function": str(call.get("function") or ""),
                    "callable_id": caller_callable,
                    "target_identity": target_identity.model_dump(),
                    "synthetic_pointer_output_binding": True,
                    "call_site_id": self._call_site_id(call),
                    "source_site_id": self._call_site_id(call),
                    "source_order": _record_source_order(call),
                    "call_instance_scope": call_scope,
                }
            )

    @staticmethod
    def _binding_first_file(binding: dict[str, Any]) -> str:
        path = binding.get("assignment_path") or []
        first = path[0] if path else {}
        return str((first or {}).get("file") or "")

    @staticmethod
    def _bare_function(name: Any) -> str:
        """Return a stable callable identity without erasing ownership.

        Kept under the historical method name to avoid a broad mechanical
        rename. Class qualification and profiler-generated callable IDs are
        semantically significant and must not be reduced to a short name.
        """
        return str(name or "").strip()

    @staticmethod
    def _helper_callable_id(
        helper_key: Optional[tuple[str, str]], helper: dict[str, Any]
    ) -> str:
        """Derive a callable identity from the helper definition record."""
        if helper.get("callable_id"):
            return str(helper["callable_id"])
        context = str(helper.get("name") or "") or (
            helper_key[1] if helper_key else ""
        )
        return ":".join(
            [
                str(helper.get("file") or ""),
                str(helper.get("line") or 0),
                context,
                ",".join(str(p) for p in (helper.get("parameters") or [])),
            ]
        )

    def _binding_target_scope(self, binding: dict[str, Any]) -> tuple[str, str]:
        """Where the binding's TARGET symbol lives (visibility scope).

        Synthesized call bindings carry explicit ``scope_file`` /
        ``scope_function`` keys pointing at the callee (the formal lives
        there even though the write site is the caller); ordinary
        bindings default to their write site.
        """
        file = str(binding.get("scope_file") or self._binding_first_file(binding) or "")
        function = self._bare_function(
            binding.get("scope_function")
            or binding.get("callable_id")
            or binding.get("function")
            or ""
        )
        return (file, function)

    @staticmethod
    def _binding_target_identity(
        binding: dict[str, Any],
    ) -> Optional[SourceSymbolIdentity]:
        raw = binding.get("target_identity")
        if not raw:
            return None
        try:
            return SourceSymbolIdentity.model_validate(raw)
        except (TypeError, ValueError):
            return None

    def _reference_identity(
        self,
        symbol_raw: str,
        file: Optional[str],
        scope_function: str,
        line: Optional[int] = None,
    ) -> SourceSymbolIdentity:
        canonical = exact_symbol(symbol_raw)
        source_callable = _base_callable_scope(scope_function)
        callable_scopes = list(
            dict.fromkeys(
                value for value in (scope_function, source_callable) if value
            )
        )
        if file and scope_function and line is not None:
            candidates = []
            for callable_scope in callable_scopes:
                candidates = self._reference_identities_by_site.get(
                    (str(file), callable_scope, int(line), canonical), []
                )
                if candidates:
                    break
            if not candidates:
                for callable_scope in callable_scopes:
                    candidates = self._reference_identities_by_scope.get(
                        (str(file), callable_scope, canonical), []
                    )
                    if candidates:
                        break
        elif file and scope_function:
            candidates = []
            for callable_scope in callable_scopes:
                candidates = self._reference_identities_by_scope.get(
                    (str(file), callable_scope, canonical), []
                )
                if candidates:
                    break
        elif file:
            candidates = self._reference_identities_by_file.get(
                (str(file), canonical), []
            )
        elif scope_function:
            candidates = []
            for callable_scope in callable_scopes:
                candidates = self._reference_identities_by_callable.get(
                    (callable_scope, canonical), []
                )
                if candidates:
                    break
        else:
            candidates = self._reference_identities_by_symbol.get(canonical, [])
        if not candidates and "." in canonical:
            parent_symbol = canonical.rsplit(".", 1)[0]
            parent_identity = self._reference_identity(
                parent_symbol, file, scope_function, line
            )
            if parent_identity.declaration_proven:
                identity = parent_identity.model_copy(
                    update={"symbol": canonical}
                )
                if identity.kind == "local" and scope_function != source_callable:
                    return identity.model_copy(
                        update={"callable_id": scope_function}
                    )
                return identity
        keys = {candidate.key() for candidate in candidates}
        if len(keys) == 1:
            identity = candidates[0]
        else:
            callable_record = (
                self._source_structure.callables_by_id.get(source_callable) or {}
            )
            identity = self._source_structure.symbol_identity(
                symbol_raw,
                file=str(file or ""),
                callable_id=source_callable,
                function_name=str(callable_record.get("name") or ""),
                function_parameters=[
                    str(value)
                    for value in callable_record.get("parameters") or []
                ],
            )
        if identity.kind == "local" and scope_function != source_callable:
            return identity.model_copy(update={"callable_id": scope_function})
        return identity

    def _record_unresolved(
        self,
        symbol: str,
        *,
        kind: str = "symbol",
        file: Optional[str] = None,
        line: Optional[int] = None,
        scope_function: str = "",
        source_expression: str = "",
        receiver: str = "",
        receiver_type: str = "",
        resolved_callable_id: str = "",
        resolved_callable_file: str = "",
        resolved_callable_owner: str = "",
        argument_count: Optional[int] = None,
        source_site_id: str = "",
        origin_vertex_id: str = "",
        origin_operand: str = "",
    ) -> tuple[Any, ...]:
        identity = (
            self._reference_identity(symbol, file, scope_function, line)
            if kind in {"symbol", "member_writers", "storage_writers"}
            else None
        )
        owner = self._source_structure.callable_owner(
            _base_callable_scope(scope_function)
        )
        reference = UnresolvedSourceReference(
            symbol=symbol,
            kind=kind,
            file=str(file or ""),
            line=line,
            callable_id=scope_function,
            class_owner=owner,
            receiver=receiver,
            receiver_type=receiver_type,
            resolved_callable_id=resolved_callable_id,
            resolved_callable_file=resolved_callable_file,
            resolved_callable_owner=resolved_callable_owner,
            argument_count=argument_count,
            source_expression=source_expression,
            source_site_id=source_site_id,
            identity=identity,
            origin_vertex_ids=(
                [origin_vertex_id] if origin_vertex_id else []
            ),
            origin_operands=([origin_operand] if origin_operand else []),
        )
        reference_key: tuple[Any, ...] = reference.visit_key()
        if (
            kind == "storage_writers"
            and identity is not None
            and identity.declaration_proven
        ):
            reference_key = (
                kind,
                identity.kind,
                identity.declaration_id,
            )
        existing = self._unresolved_references.get(reference_key)
        if existing is None:
            self._unresolved_references[reference_key] = reference
        elif origin_vertex_id or origin_operand:
            self._unresolved_references[reference_key] = existing.model_copy(
                update={
                    "origin_vertex_ids": dedupe_keep_order(
                        value
                        for value in [
                            *existing.origin_vertex_ids,
                            origin_vertex_id,
                        ]
                        if value
                    ),
                    "origin_operands": dedupe_keep_order(
                        value
                        for value in [
                            *existing.origin_operands,
                            origin_operand,
                        ]
                        if value
                    ),
                }
            )
        if kind not in {"member_writers", "storage_writers"}:
            self.unresolved_symbols.add(symbol)
        return reference_key

    def _binding_site_scope(self, binding: dict[str, Any]) -> tuple[str, str]:
        """Where the binding's EXPRESSION text lives — the scope its
        referenced symbols are resolved in."""
        return (
            str(
                binding.get("expression_scope_file")
                or self._binding_first_file(binding)
                or ""
            ),
            self._bare_function(
                binding.get("expression_scope_function")
                or binding.get("callable_id")
                or binding.get("function")
                or ""
            ),
        )

    def _binding_control_scope(self, binding: dict[str, Any]) -> tuple[str, str]:
        """Where the binding's governing predicates are evaluated."""
        site_file, site_function = self._binding_site_scope(binding)
        return (
            str(binding.get("control_scope_file") or site_file),
            self._bare_function(
                binding.get("control_scope_function") or site_function
            ),
        )

    def _binding_predicate_scope(
        self, binding: dict[str, Any], position: int
    ) -> SourceReadScope:
        files = list(binding.get("control_predicate_files") or [])
        callables = list(binding.get("control_predicate_callables") or [])
        lines = list(binding.get("control_predicate_lines") or [])
        site_ids = list(binding.get("control_predicate_site_ids") or [])
        orders = list(binding.get("control_predicate_orders") or [])
        default_file, default_callable = self._binding_control_scope(binding)
        return (
            str(files[position]) if position < len(files) else default_file,
            self._bare_function(
                callables[position]
                if position < len(callables)
                else default_callable
            ),
            (
                int(lines[position])
                if position < len(lines)
                and isinstance(lines[position], (int, float))
                else None
            ),
            (
                int(orders[position])
                if position < len(orders)
                and isinstance(orders[position], (int, float))
                else _source_site_order(
                    str(site_ids[position]) if position < len(site_ids) else ""
                )
            ),
        )

    def _binding_walk_scope(
        self, binding: dict[str, Any]
    ) -> SourceReadScope:
        file, callable_id = self._binding_site_scope(binding)
        if binding.get("expression_scope_unordered"):
            return file, callable_id, None, None
        raw_source_order = binding.get("expression_scope_source_order")
        source_order = (
            int(raw_source_order)
            if isinstance(raw_source_order, (int, float))
            else _record_source_order(binding)
        )
        raw_expression_line = binding.get("expression_scope_line")
        if isinstance(raw_expression_line, (int, float)):
            return file, callable_id, int(raw_expression_line), source_order
        path = binding.get("assignment_path") or []
        first = path[0] if path else {}
        raw_line = (first or {}).get("line")
        line = int(raw_line) if isinstance(raw_line, (int, float)) else None
        return file, callable_id, line, source_order

    def _binding_target_line(self, binding: dict[str, Any]) -> Optional[int]:
        """Source-order position of the target in its visibility scope.

        Ordinary assignments live and execute in the same callable, so their
        assignment line is sufficient. A synthesized formal-to-actual binding
        is visible from the callee entry even though its expression is
        evaluated at the caller's call site; ``scope_line`` preserves that
        distinction.
        """
        raw_line = binding.get("scope_line")
        if isinstance(raw_line, (int, float)):
            return int(raw_line)
        return self._binding_walk_scope(binding)[2]

    def _binding_target_order(self, binding: dict[str, Any]) -> Optional[int]:
        raw = binding.get("scope_source_order")
        if isinstance(raw, (int, float)):
            return int(raw)
        return _record_source_order(binding)

    def _scoped_writers(
        self,
        symbol_norm: str,
        symbol_raw: str,
        scope: SourceReadScope,
        *,
        excluded_call_effect_site: str = "",
    ) -> list[dict[str, Any]]:
        """Writers of a symbol under C++-faithful visibility.

        Declaration identity selects locals, members, and globals. Logged
        topic inputs are evidence leaves, while explicit publish operations
        are ordinary terminal writers. No naming convention or file
        proximity participates in visibility.
        """
        target_writers = self._filter_visible_writers(
            self._targets_matching(symbol_norm),
            symbol_raw,
            scope,
            excluded_call_effect_site=excluded_call_effect_site,
        )
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for binding in target_writers:
            if id(binding) not in seen:
                seen.add(id(binding))
                out.append(binding)
        return out

    def _filter_visible_writers(
        self,
        target_writers: list[dict[str, Any]],
        symbol_raw: str,
        scope: tuple[str, str] | SourceReadScope,
        *,
        excluded_call_effect_site: str = "",
    ) -> list[dict[str, Any]]:
        if excluded_call_effect_site:
            target_writers = [
                binding
                for binding in target_writers
                if not (
                    str(binding.get("call_site_id") or "")
                    == excluded_call_effect_site
                    and (
                        binding.get("synthetic_pointer_output_binding")
                        or binding.get("synthetic_member_output_binding")
                    )
                )
            ]
        scope_file, scope_function = scope[:2]
        scope_line = scope[2] if len(scope) > 2 else None
        scope_order = scope[3] if len(scope) > 3 else None
        reference = (
            self._reference_identity(
                symbol_raw, scope_file, scope_function, scope_line
            )
            if scope_file
            else None
        )
        if (
            _CALL_SCOPE_MARKER in scope_function
            and (reference is None or reference.kind in {"local", "unknown"})
        ):
            # Runtime call instances share source identity but not storage for
            # locals/formals. Members retain declaring-class storage identity
            # across method invocations and are filtered structurally below.
            target_writers = [
                binding
                for binding in target_writers
                if self._binding_target_scope(binding)
                == (scope_file, scope_function)
            ]
        if scope_file and target_writers:
            if reference is None:
                reference = self._reference_identity(
                    symbol_raw, scope_file, scope_function, scope_line
                )
            structurally_scoped: list[dict[str, Any]] = []
            for binding in target_writers:
                producer = self._binding_target_identity(binding)
                if producer is not None and self._source_structure.compatible(
                    reference, producer
                ):
                    structurally_scoped.append(binding)
            if reference.kind == "unknown":
                # A syntax-aware extractor found the use but no declaration.
                # Same spelling and source proximity do not establish storage.
                target_writers = []
            # A structurally classified local/member must never widen. When
            # declaration coverage is incomplete, same-callable fallback is
            # conservative and explicitly prevents cross-method fusion.
            elif reference.kind in {"local", "member"}:
                if reference.declaration_proven:
                    target_writers = structurally_scoped
                elif structurally_scoped:
                    target_writers = structurally_scoped
                else:
                    target_writers = [
                        binding
                        for binding in target_writers
                        if self._binding_target_scope(binding)
                        == (scope_file, scope_function)
                    ]
            elif structurally_scoped:
                producer_identities = {
                    identity.key()
                    for binding in structurally_scoped
                    if (identity := self._binding_target_identity(binding))
                }
                target_writers = (
                    structurally_scoped
                    if len(producer_identities) == 1
                    else []
                )
            else:
                # Unknown/global references are admitted only when source
                # lookup leaves one exact writer identity. Ambiguity remains
                # unresolved instead of being settled by file proximity.
                identities = {
                    identity.key()
                    for binding in target_writers
                    if (identity := self._binding_target_identity(binding))
                }
                if not identities:
                    same_scope = [
                        binding
                        for binding in target_writers
                        if self._binding_target_scope(binding)
                        == (scope_file, scope_function)
                    ]
                    if not same_scope and not scope_function:
                        same_scope = [
                            binding
                            for binding in target_writers
                            if self._binding_target_scope(binding)[0] == scope_file
                        ]
                    target_writers = same_scope
                elif len(identities) != 1:
                    target_writers = []
            if reference.kind == "member":
                # Source order is meaningful within the current callable even
                # though it is meaningless across methods. A same-callable
                # unconditional write dominates older object state; a
                # conditional write retains cross-method state as a fallback.
                if scope_line is None:
                    return target_writers
                same_scope_prior = [
                    binding
                    for binding in target_writers
                    if self._binding_target_scope(binding)
                    == (scope_file, scope_function)
                    and (
                        not self._binding_target_line(binding)
                        or self._binding_target_line(binding) <= scope_line
                    )
                    and not (
                        self._binding_target_line(binding) == scope_line
                        and scope_order is not None
                        and self._binding_target_order(binding) is not None
                        and self._binding_target_order(binding) > scope_order
                    )
                ]
                same_scope_prior.sort(
                    key=lambda binding: (
                        self._binding_target_line(binding) or 0,
                        self._binding_target_order(binding) or -1,
                    ),
                    reverse=True,
                )
                reaching: list[dict[str, Any]] = []
                dominated = False
                for binding in same_scope_prior:
                    reaching.append(binding)
                    if bool(binding.get("reachability_exact", True)) and not binding.get(
                        "control_predicates"
                    ):
                        dominated = True
                        break
                if dominated:
                    return reaching
                fallback_state = [
                    binding
                    for binding in target_writers
                    if binding not in reaching
                ]
                return [
                    *reaching,
                    *fallback_state,
                ]
            if scope_line is not None and target_writers:
                prior = [
                    binding
                    for binding in target_writers
                    if self._binding_target_scope(binding)[0] != scope_file
                    or not self._binding_target_line(binding)
                    or self._binding_target_line(binding) <= scope_line
                ]
                prior = [
                    binding
                    for binding in prior
                    if not (
                        self._binding_target_scope(binding)[0] == scope_file
                        and self._binding_target_line(binding) == scope_line
                        and scope_order is not None
                        and self._binding_target_order(binding) is not None
                        and self._binding_target_order(binding) > scope_order
                    )
                ]
                prior.sort(
                    key=lambda binding: (
                        self._binding_target_line(binding) or 0,
                        self._binding_target_order(binding) or -1,
                    ),
                    reverse=True,
                )
                reaching: list[dict[str, Any]] = []
                for binding in prior:
                    reaching.append(binding)
                    if (
                        self._binding_target_scope(binding)[0] == scope_file
                        and bool(binding.get("reachability_exact", True))
                        and not binding.get("control_predicates")
                    ):
                        break
                target_writers = reaching
        return target_writers

    def _writers_of(self, symbol_norm: str) -> list[dict[str, Any]]:
        """Bindings that explicitly write ``symbol_norm``."""
        return self._targets_matching(symbol_norm)

    # ------------------------------------------------------------
    # Vertex emission
    # ------------------------------------------------------------

    def _binding_operation_id(self, binding: dict[str, Any]) -> tuple[str, str, Optional[str], Optional[int], str, str]:
        """Deterministic identity used by both build passes."""
        target_raw = str(binding.get("target_symbol") or "")
        target_norm = exact_symbol(target_raw)
        expression = str(binding.get("source_symbol") or "").strip()
        assignment_path = binding.get("assignment_path") or []
        first_hop = assignment_path[0] if assignment_path else {}
        file = str(first_hop.get("file") or "") or None
        line_raw = first_hop.get("line")
        line = int(line_raw) if isinstance(line_raw, (int, float)) else None
        target_scope = self._binding_target_scope(binding)
        op_id = self._make_id(
            "op",
            (
                target_norm,
                expression,
                file or "",
                line or 0,
                target_scope[0],
                target_scope[1],
                str(binding.get("source_site_id") or ""),
            ),
        )
        return op_id, target_raw, file, line, target_norm, expression

    def _emit_operation_vertex(self, binding: dict[str, Any], *, is_terminal: bool) -> str:
        op_id, target_raw, file, line, target_norm, expression = self._binding_operation_id(binding)
        if op_id not in self.vertices:
            metadata: dict[str, Any] = {}
            if is_terminal:
                metadata["is_terminal"] = True
            # The operation's complete reachability: the conjunction of
            # its governing predicates (sibling negations already folded
            # into the else/else-if predicate strings), with an explicit
            # exactness flag — an unmodeled construct (switch, loops)
            # makes the formula unresolved, never silently partial.
            metadata["reachability"] = {
                "all_of": [
                    str(p)
                    for p in (binding.get("control_predicates") or [])
                    if str(p).strip()
                ],
                "exact": bool(binding.get("reachability_exact", True)),
            }
            metadata["assignment_operator"] = str(
                binding.get("assignment_operator") or "="
            )
            source_site_id = str(binding.get("source_site_id") or "")
            if source_site_id:
                metadata["source_site_id"] = source_site_id
                source_order = _record_source_order(binding)
                if source_order is not None:
                    metadata["source_order"] = source_order
            metadata.update(
                _source_expression_metadata(
                    binding.get("expression_ref"), expression
                )
            )
            target_file, target_callable = self._binding_target_scope(binding)
            site_file, site_callable = self._binding_site_scope(binding)
            control_file, control_callable = self._binding_control_scope(binding)
            function = target_callable
            if function:
                # The target's declaring callable — wiring visibility for
                # locals compares against it.
                metadata["function"] = function
            metadata["target_scope"] = {
                "file": target_file,
                "callable": target_callable,
                "line": self._binding_target_line(binding),
                "source_order": self._binding_target_order(binding),
            }
            metadata["site_scope"] = {
                "file": site_file,
                "callable": site_callable,
                "source_order": _record_source_order(binding),
            }
            metadata["control_scope"] = {
                "file": control_file,
                "callable": control_callable,
            }
            if binding.get("target_identity"):
                metadata["target_identity"] = dict(binding["target_identity"])
            if binding.get("synthetic_call_binding"):
                metadata["synthetic_call_binding"] = True
            if binding.get("synthetic_pointer_output_binding"):
                metadata["synthetic_pointer_output_binding"] = True
            if binding.get("synthetic_member_output_binding"):
                metadata["synthetic_member_output_binding"] = True
            if binding.get("synthetic_receiver_input_binding"):
                metadata["synthetic_receiver_input_binding"] = True
            if binding.get("synthetic_helper_return_binding"):
                metadata["synthetic_helper_return_binding"] = True
            if binding.get("call_site_id"):
                metadata["call_site_id"] = str(binding["call_site_id"])
            if binding.get("call_instance_scope"):
                metadata["call_instance_scope"] = str(
                    binding["call_instance_scope"]
                )
            if binding.get("synthetic_boundary_transfer"):
                metadata["synthetic_boundary_transfer"] = True
                metadata["boundary_direction"] = str(
                    binding.get("boundary_direction") or ""
                )
            if binding.get("external_target_signal"):
                metadata["external_target_signal"] = (
                    self._observed_signal_placement(target_norm) or target_norm
                )
                metadata["external_target_observation"] = (
                    "observed"
                    if self._observed_signal_placement(target_norm) is not None
                    else "unobserved"
                )
            self.vertices[op_id] = DAGVertex(
                id=op_id,
                kind="operation",
                sub_kind=(
                    "helper_call"
                    if binding.get("synthetic_helper_return_binding")
                    else self._classify_operation(expression)
                ),
                file=file,
                line=line,
                snippet=self._snippet(file, line),
                variable=target_raw,
                expression=expression,
                provenance=str(binding.get("provenance") or "") or None,
                metadata=metadata,
            )
            self._index_producer(target_norm, op_id)
        return op_id

    def _wire_binding_edges(self, binding: dict[str, Any], *, controls_only: bool = False) -> None:
        op_id, _target_raw, file, line, target_norm, expression = self._binding_operation_id(binding)

        # Attach control-predicate branches at their OWN source sites —
        # the branch's identity is where the control statement lives,
        # not where the gated assignment does.
        predicate_sites = binding.get("control_predicate_lines") or []
        predicate_site_ids = binding.get("control_predicate_site_ids") or []
        predicate_orders = binding.get("control_predicate_orders") or []
        predicate_files = binding.get("control_predicate_files") or []
        predicate_callables = binding.get("control_predicate_callables") or []
        control_file, control_callable = self._binding_control_scope(binding)
        for position, predicate in enumerate(binding.get("control_predicates") or []):
            control_refs = binding.get("control_expression_refs") or []
            condition_ref = (
                control_refs[position] if position < len(control_refs) else None
            )
            site = (
                int(predicate_sites[position])
                if position < len(predicate_sites)
                else line
            )
            branch_id = self._emit_branch(
                str(predicate),
                file=(
                    str(predicate_files[position])
                    if position < len(predicate_files)
                    else control_file or file
                ),
                line=site,
                scope_function=(
                    str(predicate_callables[position])
                    if position < len(predicate_callables)
                    else control_callable
                ),
                source_site_id=(
                    str(predicate_site_ids[position])
                    if position < len(predicate_site_ids)
                    else ""
                ),
                source_order=(
                    int(predicate_orders[position])
                    if position < len(predicate_orders)
                    and isinstance(predicate_orders[position], (int, float))
                    else None
                ),
                condition_ref=condition_ref,
            )
            self._add_edge(branch_id, op_id, kind="control")

        if controls_only:
            return

        # Wire each source-expression symbol as an incoming data edge,
        # resolved in the binding's own callable scope.
        expression_scope = self._binding_walk_scope(binding)
        expression_file, scope_function = expression_scope[:2]
        expression_line = expression_scope[2]
        expression_order = expression_scope[3]
        excluded_call_effect_site = (
            str(binding.get("call_site_id") or "")
            if binding.get("synthetic_call_binding")
            else ""
        )
        prior_target = exact_symbol(
            self._binding_read_before_write_target(binding)
        )
        for symbol in self._wire_symbols(expression, binding.get("expression_ref")):
            normalized = exact_symbol(symbol)
            reads_prior_target = (
                normalized == target_norm
                and normalized == prior_target
            )
            if not normalized or (
                normalized == target_norm
                and not reads_prior_target
                and not binding.get("synthetic_pointer_output_binding")
                and not binding.get("synthetic_member_output_binding")
            ):
                continue
            resolution_line = expression_line
            resolution_order = expression_order
            if reads_prior_target and resolution_line is not None:
                if resolution_order is not None:
                    resolution_order -= 1
                else:
                    resolution_line -= 1
            if binding.get("external_source_signal"):
                producer_ids = [
                    self._emit_evidence(
                        "logged_signal",
                        self._observed_signal_placement(normalized) or normalized,
                        file=None,
                        line=None,
                        metadata={
                            "source_form": symbol,
                            "boundary": "source_proven",
                            "boundary_provenance": str(
                                binding.get("provenance") or "subscribe"
                            ),
                        },
                    )
                ]
            else:
                producer_ids = self._resolve_symbol_producers(
                    normalized,
                    symbol,
                    expression,
                    expression_file or file,
                    resolution_line,
                    scope_function=scope_function,
                    source_order=resolution_order,
                    excluded_call_effect_site=excluded_call_effect_site,
                    origin_vertex_id=op_id,
                    origin_operand=symbol,
                )
            if binding.get("synthetic_pointer_output_binding"):
                producer_ids = [
                    producer_id
                    for producer_id in producer_ids
                    if not bool(
                        (self.vertices[producer_id].metadata or {}).get(
                            "synthetic_call_binding"
                        )
                    )
                ]
            for producer_id in producer_ids:
                self._add_edge(producer_id, op_id, kind="data", role=symbol)

        self._wire_expression_helper_calls(
            expression,
            binding.get("expression_ref"),
            op_id,
            file=expression_file or file,
            line=expression_line,
            scope_function=scope_function,
        )

    def _wire_expression_helper_calls(
        self,
        expression: str,
        expression_ref: Any,
        target_id: str,
        *,
        file: Optional[str],
        line: Optional[int],
        scope_function: str,
    ) -> None:
        """Attach source-resolved call return writers to one operation."""
        for invocation in self._find_helper_calls(
            expression,
            file,
            scope_function=scope_function,
            line=line,
            expression_ref=expression_ref,
            origin_vertex_id=target_id,
        ):
            helper_key = self._helper_key_for_invocation(
                invocation,
                scope_file=file,
                scope_function=scope_function,
            )
            if helper_key is None:
                continue
            self._record_call_grounding_roles(target_id, invocation)
            helper = self.helper_index.get(helper_key) or {}
            return_symbol = (
                f"__return__{invocation.result_path}"
                if invocation.result_path.startswith("[")
                else (
                    f"__return__.{invocation.result_path}"
                    if invocation.result_path
                    else "__return__"
                )
            )
            return_producers = self._resolve_symbol_producers(
                exact_symbol(return_symbol),
                return_symbol,
                expression,
                str(helper.get("file") or ""),
                None,
                scope_function=invocation.call_scope,
                origin_vertex_id=target_id,
                origin_operand=(
                    f"call-result:{invocation.result_path}"
                    if invocation.result_path
                    else f"call:{invocation.name}"
                ),
            )
            for helper_return_id in return_producers:
                self._add_edge(
                    helper_return_id,
                    target_id,
                    kind="data",
                    role=(
                        f"call-result:{invocation.result_path}"
                        if invocation.result_path
                        else f"call:{invocation.name}"
                    ),
                    via=invocation.source_site_id,
                )

    def _record_call_grounding_roles(
        self, target_id: str, invocation: _ExpressionCall
    ) -> None:
        """Keep exact source call spellings with the operation that reads them.

        Edge ``role`` remains the stable semantic label used by renderers.
        Grounding uses this source-site map to replace the actual call syntax
        rather than trying to reconstruct receivers and arguments from a
        short helper name.
        """
        vertex = self.vertices.get(target_id)
        if vertex is None or not invocation.call_text:
            return
        metadata = dict(vertex.metadata or {})
        call_roles = dict(metadata.get("source_call_roles") or {})
        call_roles[invocation.source_site_id] = {
            "call": invocation.call_text,
            "result": invocation.result_text or invocation.call_text,
        }
        metadata["source_call_roles"] = call_roles
        vertex.metadata = metadata

    def _visible_reaching_producers(
        self,
        producers: list[str],
        symbol_raw: str,
        file: Optional[str],
        line: Optional[int],
        scope_function: str,
        source_order: Optional[int] = None,
        *,
        excluded_call_effect_site: str = "",
    ) -> list[str]:
        """Return source-visible definitions that may reach a read site.

        Conditional definitions are alternatives and are all retained until
        an unconditional prior definition dominates earlier source-order
        candidates. A same-callable read can never bind to a later local
        write. Member state is conservatively retained across methods because
        textual order does not describe inter-method execution order.
        """
        if not producers or not file:
            return list(producers)

        def metadata(vertex_id: str) -> dict[str, Any]:
            return self.vertices[vertex_id].metadata or {}

        if excluded_call_effect_site:
            producers = [
                producer_id
                for producer_id in producers
                if not (
                    str(metadata(producer_id).get("call_site_id") or "")
                    == excluded_call_effect_site
                    and (
                        metadata(producer_id).get(
                            "synthetic_pointer_output_binding"
                        )
                        or metadata(producer_id).get(
                            "synthetic_member_output_binding"
                        )
                    )
                )
            ]

        reference = self._reference_identity(
            symbol_raw, file, scope_function, line
        )
        if _CALL_SCOPE_MARKER in scope_function:
            instance_producers = [
                producer_id
                for producer_id in producers
                if str(
                    (metadata(producer_id).get("target_scope") or {}).get(
                        "callable"
                    )
                    or ""
                )
                == scope_function
            ]
            if instance_producers:
                producers = instance_producers
        visible: list[str] = []
        has_structured_producer = False
        for producer_id in producers:
            raw_identity = metadata(producer_id).get("target_identity")
            if raw_identity:
                has_structured_producer = True
                try:
                    producer_identity = SourceSymbolIdentity.model_validate(raw_identity)
                except (TypeError, ValueError):
                    continue
                if self._source_structure.compatible(reference, producer_identity):
                    visible.append(producer_id)
                continue
            if self._source_structure.authoritative_declarations:
                continue
            target_scope = metadata(producer_id).get("target_scope") or {}
            target_file = str(target_scope.get("file") or self.vertices[producer_id].file or "")
            target_callable = str(
                target_scope.get("callable") or metadata(producer_id).get("function") or ""
            )
            if target_file != file:
                continue
            if scope_function and target_callable and target_callable != scope_function:
                continue
            visible.append(producer_id)

        if has_structured_producer and reference.kind in {"local", "member"}:
            visible = [
                producer_id
                for producer_id in visible
                if metadata(producer_id).get("target_identity")
            ]
        if reference.kind == "member":
            if line is None:
                return visible
            same_scope_prior = []
            for producer_id in visible:
                item = self.vertices[producer_id]
                target_scope = metadata(producer_id).get("target_scope") or {}
                target_file = str(
                    target_scope.get("file") or item.file or ""
                )
                target_callable = str(target_scope.get("callable") or "")
                target_line = target_scope.get("line")
                if (
                    target_file == file
                    and target_callable == scope_function
                    and (
                        not isinstance(target_line, (int, float))
                        or int(target_line) <= line
                    )
                    and not (
                        isinstance(target_line, (int, float))
                        and int(target_line) == line
                        and source_order is not None
                        and isinstance(target_scope.get("source_order"), (int, float))
                        and int(target_scope["source_order"]) > source_order
                    )
                ):
                    same_scope_prior.append(producer_id)
            same_scope_prior.sort(
                key=lambda producer_id: int(
                    (
                        metadata(producer_id).get("target_scope") or {}
                    ).get("line")
                    or 0,
                )
                * 1_000_000_000
                + int(
                    (
                        metadata(producer_id).get("target_scope") or {}
                    ).get("source_order")
                    or 0
                ),
                reverse=True,
            )
            reaching: list[str] = []
            dominated = False
            for producer_id in same_scope_prior:
                reaching.append(producer_id)
                reachability = metadata(producer_id).get("reachability") or {}
                if reachability.get("exact") and not reachability.get("all_of"):
                    dominated = True
                    break
            if dominated:
                return reaching
            fallback_state = [
                producer_id
                for producer_id in visible
                if producer_id not in same_scope_prior
            ]
            return [*reaching, *fallback_state]

        if line is None:
            return visible

        prior: list[str] = []
        unordered: list[str] = []
        for producer_id in visible:
            item = self.vertices[producer_id]
            site_scope = metadata(producer_id).get("site_scope") or {}
            site_file = str(site_scope.get("file") or item.file or "")
            synthetic = bool(metadata(producer_id).get("synthetic_call_binding"))
            if synthetic or site_file != file or item.line is None:
                unordered.append(producer_id)
            elif item.line < line or (
                item.line == line
                and (
                    source_order is None
                    or not isinstance(site_scope.get("source_order"), (int, float))
                    or int(site_scope["source_order"]) <= source_order
                )
            ):
                prior.append(producer_id)

        prior.sort(
            key=lambda producer_id: (
                int(self.vertices[producer_id].line or 0),
                int(
                    (
                        metadata(producer_id).get("site_scope") or {}
                    ).get("source_order")
                    or 0
                ),
            ),
            reverse=True,
        )
        reaching: list[str] = []
        for producer_id in prior:
            reaching.append(producer_id)
            reachability = metadata(producer_id).get("reachability") or {}
            if reachability.get("exact") and not reachability.get("all_of"):
                break
        return reaching + [producer_id for producer_id in unordered if producer_id not in reaching]

    def _resolve_symbol_producers(
        self,
        symbol_norm: str,
        symbol_raw: str,
        source_expression: str,
        file: Optional[str],
        line: Optional[int],
        scope_function: str = "",
        source_order: Optional[int] = None,
        emit_opaque: bool = True,
        excluded_call_effect_site: str = "",
        origin_vertex_id: str = "",
        origin_operand: str = "",
    ) -> list[str]:
        """Resolve a source-expression symbol to all reaching producers.

        If a prior operation produces ``symbol_norm``, connect to it. Otherwise
        emit an evidence leaf. Classification order (each consults a
        profiler-emitted source-of-truth so the vertex carries the
        resolved runtime handle rather than the raw source form):

        1. Direct producer via ``_producers_by_symbol`` (in-DAG operation).
        2. Source enum resolution — the symbol is defined in source as a
           numeric constant; emit ``constant`` with ``metadata['value']``.
        3. C stdlib constant (``CXX_STDLIB_CONSTANTS``) — same shape.
        4. Parameter accessor resolved through source-scoped declarations.
        5. Otherwise: typed unresolved source symbol.

        No flat ``symbol_bindings`` table is consulted anywhere — every
        source→logged mapping is derived from graph structure.
        """
        identity = self._reference_identity(
            symbol_raw, file, scope_function, line
        )
        if identity.declaration_proven and identity.kind in {"member", "global"}:
            self._record_unresolved(
                identity.root,
                kind="storage_writers",
                file=file,
                line=line,
                scope_function=scope_function,
                source_expression=source_expression,
                origin_vertex_id=origin_vertex_id,
                origin_operand=origin_operand,
            )
        producers = self._visible_reaching_producers(
            self._producers_matching(symbol_norm),
            symbol_raw,
            file,
            line,
            scope_function,
            source_order,
            excluded_call_effect_site=excluded_call_effect_site,
        )
        if len(producers) == 1:
            return producers
        # A member whose declared type is a logged uORB topic, whose source
        # writes are ambiguous (one-per-flight-mode, not disambiguable) or
        # absent, reads as an observation: the recorded value is ground truth.
        # A single unambiguous producer above stays a traced source edge.
        observation = self._member_observation_leaf(symbol_norm, identity)
        if observation is not None:
            return [observation]
        if producers:
            return producers

        # 2. Source enum / #define resolution.
        enum_value = self._source_constant_value(
            symbol_norm, identity, scope_function
        )
        if enum_value is not None:
            return [self._emit_evidence(
                "constant",
                symbol_raw,
                file=None,
                line=None,
                metadata={
                    "value": enum_value,
                    "source": "enum",
                    "source_scope": (
                        identity.declaration_id
                        or _base_callable_scope(scope_function)
                    ),
                },
            )]

        # 2b. Schema message enum — the reference names its own scope
        # (``position_setpoint_s.SETPOINT_TYPE_LAND`` → message
        # ``position_setpoint``); no global name table.
        enum_root, _, enum_tail = symbol_raw.replace("::", ".").rpartition(".")
        enum_root = enum_root.split(".", 1)[0].strip()
        if enum_root.endswith("_s") and enum_tail.isupper():
            schema_enum = self._enum_registry.get(enum_root[:-2], {}).get(enum_tail)
            if schema_enum is not None:
                return [self._emit_evidence(
                    "constant",
                    symbol_raw,
                    file=None,
                    line=None,
                    metadata={"value": schema_enum, "source": "msg_schema"},
                )]

        # 3. C stdlib constant.
        cxx_value = CXX_STDLIB_CONSTANTS.get(symbol_raw.upper())
        if cxx_value is not None:
            return [self._emit_evidence(
                "constant",
                symbol_raw.upper(),
                file=None,
                line=None,
                metadata={"value": cxx_value, "source": "cxx_stdlib"},
            )]

        # 4. Parameter accessor resolved from source-scoped declarations.
        parameter_alias = self._match_parameter(
            symbol_raw,
            file=file,
            scope_function=scope_function,
            line=line,
        )
        if parameter_alias is not None:
            return [self._emit_evidence("parameter", parameter_alias, file=None, line=None)]

        # 5. Unclassified. Preserve source identity and fail closed.
        if not emit_opaque:
            return []
        if self._symbol_is_receiver_call_result(
            symbol_norm,
            (str(file or ""), scope_function, line, source_order),
        ) or re.search(rf"{re.escape(symbol_raw)}\s*\(", source_expression):
            # The callable frontier owns expansion of a call result. Keeping
            # its parser placeholder as a second symbol gap causes searches
            # for names such as ``receiver.method`` and cannot find storage.
            return [
                self._emit_evidence(
                    "opaque_symbol",
                    symbol_raw,
                    file=file,
                    line=line,
                    metadata={
                        "source_identity": identity.model_dump(exclude_none=True)
                    },
                )
            ]
        if identity.declaration_proven and identity.kind == "local":
            self.unresolved_symbols.add(symbol_raw)
        elif identity.declaration_proven and identity.kind in {"member", "global"}:
            self.unresolved_symbols.add(symbol_raw)
        elif not (
            identity.declaration_proven
            and identity.kind in {"member", "global"}
        ):
            self._record_unresolved(
                symbol_raw,
                file=file,
                line=line,
                scope_function=scope_function,
                source_expression=source_expression,
                origin_vertex_id=origin_vertex_id,
                origin_operand=origin_operand,
            )
        return [
            self._emit_evidence(
                "opaque_symbol",
                symbol_raw,
                file=file,
                line=line,
                metadata={
                    "source_identity": identity.model_dump(exclude_none=True)
                },
            )
        ]

    def _lower_predicate(
        self, canonical: str
    ) -> tuple[str, dict[str, str]]:
        """Normalize only language syntax; values remain on DAG edges."""
        return _normalize_cpp_expression(canonical), {}

    def _branch_metadata_from_parameter_predicate(
        self,
        canonical: str,
        *,
        file: Optional[str],
        line: Optional[int],
        source_site_id: str,
    ) -> dict[str, Any]:
        """Return branch metadata sourced from a ``ParameterPredicateRef``.

        The profiler already extracts operator + compared_value at profile
        time; consuming that record here means downstream evaluators don't
        re-parse the predicate to recover the same structure. Absent match
        returns an empty dict — the branch stays a plain source-predicate
        node whose feasibility path handles evaluation.
        """
        exact_key = (
            canonical,
            str(file or ""),
            int(line or 0),
            str(source_site_id or ""),
        )
        records = self._parameter_predicate_by_site.get(exact_key, [])
        if not records and source_site_id:
            records = self._parameter_predicate_by_site.get(
                (canonical, str(file or ""), int(line or 0), ""), []
            )
        identities = {
            (
                str(record.get("name") or ""),
                str(record.get("member") or ""),
                str(record.get("operator") or ""),
                str(record.get("compared_value") or ""),
            )
            for record in records
        }
        if len(identities) != 1:
            return {}
        record = records[0]
        metadata: dict[str, Any] = {}
        parameter = record.get("name") or record.get("parameter")
        if parameter:
            metadata["parameter"] = str(parameter)
        operator = record.get("operator")
        if operator:
            metadata["operator"] = str(operator)
        compared = record.get("compared_value")
        if compared is not None and compared != "":
            metadata["compared_value"] = compared
        member = record.get("member")
        if member:
            metadata["member"] = str(member)
        return metadata

    def _emit_evidence(
        self,
        sub_kind: str,
        signal: str,
        *,
        file: Optional[str],
        line: Optional[int],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        if sub_kind == "logged_signal":
            signal = self._observed_signal_placement(signal) or signal
        vertex_metadata = dict(metadata) if metadata else {}
        if sub_kind in {"logged_signal", "parameter"}:
            key: tuple[Any, ...] = (sub_kind, signal)
        elif sub_kind == "opaque_symbol":
            raw_identity = vertex_metadata.get("source_identity") or {}
            try:
                identity_key = SourceSymbolIdentity.model_validate(
                    raw_identity
                ).storage_key()
            except (TypeError, ValueError):
                identity_key = ()
            key = (
                sub_kind,
                signal,
                identity_key or (str(file or ""), int(line or 0)),
            )
        elif sub_kind == "constant" and vertex_metadata.get("source") == "enum":
            key = (
                sub_kind,
                signal,
                str(vertex_metadata.get("source_scope") or ""),
            )
        else:
            key = (sub_kind, signal)
        existing = self._evidence_by_signal.get(key)
        if existing is not None:
            return existing
        vertex_id = self._make_id("ev", key)
        if sub_kind == "logged_signal":
            # Declaration validity and observation are INDEPENDENT: a
            # schema/type-derived placement is a source-proven boundary
            # that may stop the walk, but only presence in THIS flight's
            # log makes it flight evidence. Downstream consumers must
            # never read an unobserved leaf as observed data.
            observation = (
                "observed"
                if self._observed_signal_placement(exact_symbol(signal)) is not None
                else "unobserved"
            )
            declaration = (
                "valid"
                if self._declared_signal_known(exact_symbol(signal))
                else "unknown"
            )
            boundary = str(vertex_metadata.get("boundary") or "not_proven")
            vertex_metadata["observation"] = observation
            vertex_metadata["observation_status"] = observation
            vertex_metadata["declaration_status"] = declaration
            vertex_metadata["boundary_status"] = boundary
        self.vertices[vertex_id] = DAGVertex(
            id=vertex_id,
            kind="evidence",
            sub_kind=sub_kind,
            file=file,
            line=line,
            snippet=self._snippet(file, line),
            signal_name=signal,
            metadata=vertex_metadata,
        )
        self._evidence_by_signal[key] = vertex_id
        return vertex_id

    def _emit_branch(
        self,
        predicate: str,
        *,
        file: Optional[str],
        line: Optional[int],
        scope_function: str = "",
        source_site_id: str = "",
        source_order: Optional[int] = None,
        condition_ref: Any = None,
    ) -> str:
        canonical = _canonical_predicate(predicate)
        # Branch identity is the SOURCE SITE plus the canonical predicate
        # — identical text at two different sites is two branches. The
        # canonical text alone keys only predicate SEMANTICS (parameter-
        # predicate metadata, lowering), never vertex identity.
        site_key = (
            canonical,
            str(file or ""),
            int(line or 0),
            str(source_site_id or ""),
            str(scope_function or ""),
        )
        existing = self._branch_by_site.get(site_key)
        if existing is not None:
            return existing
        vertex_id = self._make_id("br", site_key)
        metadata = self._branch_metadata_from_parameter_predicate(
            canonical,
            file=file,
            line=line,
            source_site_id=source_site_id,
        )
        metadata.update(_source_expression_metadata(condition_ref, predicate))
        if source_order is not None:
            metadata["source_order"] = int(source_order)
        lowered, variables = self._lower_predicate(canonical)
        if variables:
            metadata["variables"] = variables
        self.vertices[vertex_id] = DAGVertex(
            id=vertex_id,
            kind="branch",
            file=file,
            line=line,
            snippet=self._snippet(file, line),
            predicate_raw=predicate,
            predicate_lowered=lowered,
            feasibility_verdict="unknown",
            metadata=metadata,
        )
        self._branch_by_site[site_key] = vertex_id

        # A branch's own predicate depends on the symbols it reads. Wire
        # data edges from those producers so pre-evaluation later has
        # every input in the graph. Normalize C++ operators/casts first so
        # ``&&``/``->``/``::`` predicates yield their symbols (parameters,
        # accessor chains) instead of failing ast extraction wholesale.
        symbol_source = _normalize_cpp_expression(predicate)
        for symbol in self._wire_symbols(predicate, condition_ref):
            normalized = exact_symbol(symbol)
            if not normalized:
                continue
            producer_ids = self._resolve_symbol_producers(
                normalized,
                symbol,
                symbol_source,
                file,
                line,
                scope_function=scope_function,
                source_order=(
                    source_order
                    if source_order is not None
                    else _source_site_order(source_site_id)
                ),
                origin_vertex_id=vertex_id,
                origin_operand=symbol,
            )
            for producer_id in producer_ids:
                self._add_edge(producer_id, vertex_id, kind="data", role=symbol)

        self._wire_expression_helper_calls(
            predicate,
            (
                condition_ref.model_dump(exclude_none=True)
                if hasattr(condition_ref, "model_dump")
                else condition_ref
            ),
            vertex_id,
            file=file,
            line=line,
            scope_function=scope_function,
        )

        return vertex_id

    # ------------------------------------------------------------
    # Helper subgraph nesting
    # ------------------------------------------------------------

    def _find_helper_calls(
        self,
        expression: str,
        scope_file: Optional[str] = None,
        *,
        scope_function: str = "",
        line: Optional[int] = None,
        expression_ref: Optional[dict[str, Any]] = None,
        origin_vertex_id: str = "",
    ) -> list[_ExpressionCall]:
        """Return each source-resolved call occurrence in ``expression``.

        Resolution is frontier-driven: this is the only path that may ask
        the provider for a missing expression callee. Every occurrence keeps
        a distinct site identity, including repeated calls with the same name
        on one source line.
        """
        matches: list[_ExpressionCall] = []
        raw_expression_ref = expression_ref
        if hasattr(raw_expression_ref, "model_dump"):
            raw_expression_ref = raw_expression_ref.model_dump(exclude_none=True)
        structured_expression = (
            raw_expression_ref
            if isinstance(raw_expression_ref, dict)
            and bool(raw_expression_ref.get("exact"))
            else None
        )
        if structured_expression is not None:
            for raw_result in structured_expression.get("call_results") or []:
                result = (
                    raw_result.model_dump(exclude_none=True)
                    if hasattr(raw_result, "model_dump")
                    else dict(raw_result)
                )
                source_site_id = str(result.get("call_source_site_id") or "")
                source_record = self._calls_by_source_site_id.get(source_site_id)
                if source_record is None:
                    self._record_unresolved(
                        str(result.get("text") or source_site_id or "call"),
                        kind="callable",
                        file=scope_file,
                        line=line,
                        scope_function=scope_function,
                        source_expression=expression,
                        origin_vertex_id=origin_vertex_id,
                        origin_operand=str(
                            result.get("text") or source_site_id or "call"
                        ),
                    )
                    continue
                structured_call = self._call_from_structured_result(
                    source_record,
                    result_path=str(result.get("result_path") or ""),
                    expression=expression,
                    scope_file=scope_file,
                    scope_function=scope_function,
                    line=line,
                    origin_vertex_id=origin_vertex_id,
                )
                if structured_call is not None:
                    matches.append(structured_call)
            return matches

        structured_records: dict[
            tuple[str, str, tuple[str, ...]],
            list[tuple[dict[str, Any], str]],
        ] = defaultdict(list)
        for raw_result in (expression_ref or {}).get("call_results") or []:
            result = (
                raw_result.model_dump(exclude_none=True)
                if hasattr(raw_result, "model_dump")
                else dict(raw_result)
            )
            source_site_id = str(result.get("call_source_site_id") or "")
            source_record = self._calls_by_source_site_id.get(source_site_id)
            if source_record is None:
                continue
            candidate = str(source_record.get("name") or "").rsplit("::", 1)[-1]
            receiver = str(source_record.get("receiver") or "")
            args = tuple(
                str(value).strip() for value in source_record.get("args") or []
            )
            structured_records[(candidate, receiver, args)].append(
                (
                    source_record,
                    str(result.get("result_path") or ""),
                )
            )
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression):
            candidate = match.group(1)
            if candidate in _NON_CALL_SYNTAX:
                continue
            prefix = expression[: match.start()]
            receiver_match = re.search(
                r"([A-Za-z_][A-Za-z0-9_.]*(?:\[[^\]]+\])?)\s*(?:\.|->)\s*$",
                prefix,
            )
            receiver = receiver_match.group(1) if receiver_match else ""
            if receiver and self._match_parameter(
                f"{receiver}.{candidate}",
                file=scope_file,
                scope_function=scope_function,
                line=line,
            ) is not None:
                continue
            close = self._matching_parenthesis(expression, match.end() - 1)
            if close is None:
                continue
            call_start = (
                receiver_match.start(1) if receiver_match is not None else match.start()
            )
            call_text = expression[call_start : close + 1].strip()
            raw_args = expression[match.end() : close]
            args = tuple(
                arg.strip()
                for arg in split_top_level_args(raw_args)
                if arg.strip()
            )
            structured = structured_records.get((candidate, receiver, args)) or []
            structured_entry = structured.pop(0) if structured else None
            lookup_key = (
                str(scope_file or ""),
                _base_callable_scope(scope_function),
                int(line or 0),
                candidate,
                args,
            )
            source_record = (
                structured_entry[0]
                if structured_entry is not None
                else next(
                    (
                        call
                        for call in self._calls_by_site.get(lookup_key, ())
                        if str(call.get("receiver") or "") == receiver
                    ),
                    None,
                )
            )
            if source_record is None:
                location_candidates = [
                    call
                    for call in self._calls_by_location.get(lookup_key[:4], ())
                    if str(call.get("receiver") or "") == receiver
                    and len(call.get("args") or []) == len(args)
                ]
                if len(location_candidates) == 1:
                    source_record = location_candidates[0]
            if source_record is not None and source_record.get(
                "evaluation_intrinsic"
            ):
                continue
            source_site_id = (
                self._call_site_id(source_record)
                if source_record is not None
                else stable_id(
                    "callsite",
                    (
                        str(scope_file or ""),
                        _base_callable_scope(scope_function),
                        (
                            0
                            if _CALL_SCOPE_MARKER in scope_function
                            else int(line or 0)
                        ),
                        match.start(),
                        candidate,
                        receiver,
                        args,
                    ),
                )
            )
            if source_record is None:
                self._record_unresolved(
                    f"{receiver}.{candidate}" if receiver else candidate,
                    kind="callable",
                    file=scope_file,
                    line=line,
                    scope_function=scope_function,
                    source_expression=expression,
                    receiver=receiver,
                    argument_count=len(args),
                    source_site_id=source_site_id,
                    origin_vertex_id=origin_vertex_id,
                    origin_operand=(
                        f"{receiver}.{candidate}" if receiver else candidate
                    ),
                )
                continue
            # A source site nested under two different parent invocations is
            # two runtime instances even though its AST node is shared.
            runtime_site_id = (
                stable_id("invocation", (scope_function, source_site_id))
                if _CALL_SCOPE_MARKER in scope_function
                else source_site_id
            )
            call_reference = UnresolvedSourceReference(
                symbol=candidate,
                kind="callable",
                file=str(scope_file or ""),
                line=line,
                callable_id=_base_callable_scope(scope_function),
                class_owner=self._source_structure.callable_owner(
                    _base_callable_scope(scope_function)
                ),
                receiver=receiver,
                argument_count=len(args),
                source_expression=expression,
                source_site_id=source_site_id,
            )
            if reference_receiver_is_source_boundary(
                call_reference,
                self._boundary_bindings,
                self._source_structure,
            ):
                # The source-site transfer operation already owns this call.
                # Treating the same syntax as an unresolved helper would add
                # a second, non-value-flow expansion frontier.
                continue
            helper_key = self._pick_helper_key(
                candidate,
                scope_file,
                scope_function=scope_function,
                receiver=receiver,
                receiver_type_hint=str(
                    (source_record or {}).get("receiver_type") or ""
                ),
                resolved_callable_id_hint=str(
                    (source_record or {}).get("resolved_callable_id") or ""
                ),
                resolved_callable_file_hint=str(
                    (source_record or {}).get("resolved_callable_file") or ""
                ),
                resolved_callable_owner_hint=str(
                    (source_record or {}).get("resolved_callable_owner") or ""
                ),
                argument_count=len(args),
                line=line,
                source_site_id=source_site_id,
                allow_provider=True,
            )
            if helper_key is None:
                self._record_unresolved(
                    f"{receiver}.{candidate}" if receiver else candidate,
                    kind="callable",
                    file=scope_file,
                    line=line,
                    scope_function=scope_function,
                    source_expression=expression,
                    receiver=receiver,
                    receiver_type=str((source_record or {}).get("receiver_type") or ""),
                    resolved_callable_id=str(
                        (source_record or {}).get("resolved_callable_id") or ""
                    ),
                    resolved_callable_file=str(
                        (source_record or {}).get("resolved_callable_file") or ""
                    ),
                    resolved_callable_owner=str(
                        (source_record or {}).get("resolved_callable_owner") or ""
                    ),
                    argument_count=len(args),
                    source_site_id=source_site_id,
                    origin_vertex_id=origin_vertex_id,
                    origin_operand=(
                        f"{receiver}.{candidate}" if receiver else candidate
                    ),
                )
                continue
            helper = self.helper_index.get(helper_key) or {}
            runtime_call = dict(source_record or {})
            runtime_call.update(
                {
                    "name": candidate,
                    "receiver": receiver,
                    "args": list(args),
                    "file": str(scope_file or ""),
                    "line": int(line or 0),
                    "callable_id": scope_function,
                    "source_site_id": runtime_site_id,
                }
            )
            result_path = structured_entry[1] if structured_entry else ""
            runtime_call["result_path"] = result_path
            call_scope = self._register_call_instance(
                runtime_call,
                helper_key,
                helper,
                args,
                caller_callable=scope_function,
            )
            self._register_helper_return_projection(
                helper, call_scope, result_path
            )
            result_text = call_text
            if result_path:
                suffix = (
                    result_path
                    if result_path.startswith("[")
                    else f".{result_path}"
                )
                if not result_text.endswith(suffix):
                    result_text += suffix
            matches.append(
                _ExpressionCall(
                    name=candidate,
                    receiver=receiver,
                    args=args,
                    argument_expressions=tuple(
                        (source_record or {}).get("argument_expressions") or ()
                    ),
                    offset=match.start(),
                    source_site_id=runtime_site_id,
                    call_scope=call_scope,
                    helper_key=helper_key,
                    call_text=call_text,
                    result_text=result_text,
                    result_path=result_path,
                )
            )
        for entries in structured_records.values():
            for source_record, result_path in entries:
                structured_call = self._call_from_structured_result(
                    source_record,
                    result_path=result_path,
                    expression=expression,
                    scope_file=scope_file,
                    scope_function=scope_function,
                    line=line,
                    origin_vertex_id=origin_vertex_id,
                )
                if structured_call is not None:
                    matches.append(structured_call)
        return matches

    def _call_from_structured_result(
        self,
        source_record: dict[str, Any],
        *,
        result_path: str,
        expression: str,
        scope_file: Optional[str],
        scope_function: str,
        line: Optional[int],
        origin_vertex_id: str = "",
    ) -> Optional[_ExpressionCall]:
        """Recover an aliased call result without searching expression text."""
        candidate = str(source_record.get("name") or "").rsplit("::", 1)[-1]
        receiver = str(source_record.get("receiver") or "")
        args = tuple(
            str(value).strip() for value in (source_record.get("args") or [])
        )
        if (
            not candidate
            or candidate in _NON_CALL_SYNTAX
            or source_record.get("evaluation_intrinsic")
        ):
            return None
        call_file = str(source_record.get("file") or scope_file or "")
        call_line = int(source_record.get("line") or line or 0)
        source_site_id = self._call_site_id(source_record)
        runtime_site_id = (
            stable_id("invocation", (scope_function, source_site_id))
            if _CALL_SCOPE_MARKER in scope_function
            else source_site_id
        )
        call_reference = UnresolvedSourceReference(
            symbol=candidate,
            kind="callable",
            file=call_file,
            line=call_line or None,
            callable_id=_base_callable_scope(scope_function),
            class_owner=self._source_structure.callable_owner(
                _base_callable_scope(scope_function)
            ),
            receiver=receiver,
            argument_count=len(args),
            source_expression=expression,
            source_site_id=source_site_id,
        )
        if reference_receiver_is_source_boundary(
            call_reference,
            self._boundary_bindings,
            self._source_structure,
        ):
            return None
        helper_key = self._pick_helper_key(
            candidate,
            call_file,
            scope_function=scope_function,
            receiver=receiver,
            receiver_type_hint=str(source_record.get("receiver_type") or ""),
            resolved_callable_id_hint=str(
                source_record.get("resolved_callable_id") or ""
            ),
            resolved_callable_file_hint=str(
                source_record.get("resolved_callable_file") or ""
            ),
            resolved_callable_owner_hint=str(
                source_record.get("resolved_callable_owner") or ""
            ),
            argument_count=len(args),
            line=call_line or None,
            source_site_id=source_site_id,
            allow_provider=True,
        )
        if helper_key is None:
            self._record_unresolved(
                f"{receiver}.{candidate}" if receiver else candidate,
                kind="callable",
                file=call_file,
                line=call_line or None,
                scope_function=scope_function,
                source_expression=expression,
                receiver=receiver,
                receiver_type=str(source_record.get("receiver_type") or ""),
                resolved_callable_id=str(
                    source_record.get("resolved_callable_id") or ""
                ),
                resolved_callable_file=str(
                    source_record.get("resolved_callable_file") or ""
                ),
                resolved_callable_owner=str(
                    source_record.get("resolved_callable_owner") or ""
                ),
                argument_count=len(args),
                source_site_id=source_site_id,
                origin_vertex_id=origin_vertex_id,
                origin_operand=(
                    f"{receiver}.{candidate}" if receiver else candidate
                ),
            )
            return None
        helper = self.helper_index.get(helper_key) or {}
        runtime_call = dict(source_record)
        runtime_call.update(
            {
                "name": candidate,
                "receiver": receiver,
                "args": list(args),
                "file": call_file,
                "line": call_line,
                "callable_id": scope_function,
                "source_site_id": runtime_site_id,
                "result_path": result_path,
            }
        )
        call_scope = self._register_call_instance(
            runtime_call,
            helper_key,
            helper,
            args,
            caller_callable=scope_function,
        )
        self._register_helper_return_projection(
            helper, call_scope, result_path
        )
        separator = (
            "->"
            if str(source_record.get("receiver_access") or "") == "pointer"
            else "."
        )
        call_text = (
            f"{receiver}{separator}{candidate}({', '.join(args)})"
            if receiver
            else f"{candidate}({', '.join(args)})"
        )
        result_text = call_text
        suffix = (
            result_path
            if result_path.startswith("[")
            else f".{result_path}" if result_path else ""
        )
        if suffix and not result_text.endswith(suffix):
            result_text += suffix
        return _ExpressionCall(
            name=candidate,
            receiver=receiver,
            args=args,
            argument_expressions=tuple(
                source_record.get("argument_expressions") or ()
            ),
            offset=-1,
            source_site_id=runtime_site_id,
            call_scope=call_scope,
            helper_key=helper_key,
            call_text=call_text,
            result_text=result_text,
            result_path=result_path,
        )

    @staticmethod
    def _matching_parenthesis(text: str, opening: int) -> Optional[int]:
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def _helper_key_for_invocation(
        self,
        invocation: _ExpressionCall,
        *,
        scope_file: Optional[str] = None,
        scope_function: str = "",
    ) -> Optional[tuple[str, str]]:
        """Return the source-resolved callable retained by an invocation.

        Expression discovery has already resolved the callable with the full
        source record, including receiver type and exact callable identity.
        Later graph passes must not discard that proof and try to infer the
        owner again from whichever source files happen to remain loaded.
        """
        retained = invocation.helper_key or self._resolved_helper_key_by_call_site.get(
            invocation.source_site_id
        )
        if retained in self.helper_index:
            return retained
        return None

    def _pick_helper_key(
        self,
        helper_name: str,
        scope_file: Optional[str] = None,
        *,
        scope_function: str = "",
        receiver: str = "",
        receiver_type_hint: str = "",
        resolved_callable_id_hint: str = "",
        resolved_callable_file_hint: str = "",
        resolved_callable_owner_hint: str = "",
        argument_count: Optional[int] = None,
        line: Optional[int] = None,
        source_site_id: str = "",
        allow_provider: bool = False,
    ) -> Optional[tuple[str, str]]:
        """Resolve a callable from receiver type, owner lineage, and arity.

        File proximity and sorted-first selection are deliberately absent.
        Ambiguous overloads remain unresolved until source structure proves a
        unique callable identity.
        """

        def matching_keys() -> list[tuple[str, str]]:
            found = list(self._helper_keys_by_name.get(helper_name, ()))
            if argument_count is not None:
                found = [
                    key
                    for key in found
                    if callable_accepts_argument_count(
                        self.helper_index[key], argument_count
                    )
                ]
            return found

        matches = matching_keys()
        if resolved_callable_id_hint:
            exact_matches = [
                key
                for key in matches
                if self._helper_callable_id(
                    key, self.helper_index.get(key) or {}
                )
                == resolved_callable_id_hint
            ]
            if len(exact_matches) == 1:
                return exact_matches[0]
        if not matches:
            if not allow_provider or self.helper_body_provider is None:
                return None
            reference = UnresolvedSourceReference(
                symbol=helper_name,
                kind="callable",
                file=str(scope_file or ""),
                line=line,
                callable_id=_base_callable_scope(scope_function),
                class_owner=self._source_structure.callable_owner(
                    _base_callable_scope(scope_function)
                ),
                receiver=receiver,
                receiver_type=receiver_type_hint,
                resolved_callable_id=resolved_callable_id_hint,
                resolved_callable_file=resolved_callable_file_hint,
                resolved_callable_owner=resolved_callable_owner_hint,
                argument_count=argument_count,
                source_site_id=source_site_id,
            )
            probe_key = source_reference_resolution_key(
                reference, self._source_structure
            )
            if probe_key in self._helper_provider_probed:
                return None
            self._helper_provider_probed.add(probe_key)
            try:
                fetched = self.helper_body_provider(helper_name, reference)
            except TypeError:
                fetched = self.helper_body_provider(helper_name)
            added_helpers = False
            for helper in _coerce_helpers(fetched):
                key_iter = _index_helpers([helper])
                for key, value in key_iter.items():
                    if key not in self.helper_index:
                        self.helper_index[key] = value
                        self._helper_keys_by_name[key[0]].append(key)
                        added_helpers = True
            if added_helpers and hasattr(self, "_callers_by_callee"):
                self._index_source_call_edges()
            matches = matching_keys()
            if resolved_callable_id_hint:
                exact_matches = [
                    key
                    for key in matches
                    if self._helper_callable_id(
                        key, self.helper_index.get(key) or {}
                    )
                    == resolved_callable_id_hint
                ]
                if len(exact_matches) == 1:
                    return exact_matches[0]
        if not matches:
            return None
        caller_owner = self._source_structure.callable_owner(
            _base_callable_scope(scope_function)
        )
        receiver_type = str(receiver_type_hint or "").rstrip("*& ").split("<", 1)[0].strip()
        receiver_root = receiver.replace("->", ".").split(".", 1)[0].lstrip("&*")
        if not receiver_type and caller_owner and receiver_root:
            receiver_type = self._source_structure.member_receiver_type(
                caller_owner,
                receiver_root,
            )
        elif receiver_type:
            receiver_type = self._source_structure.resolve_class_name(
                receiver_type,
                lexical_owner=caller_owner,
            )
        if receiver and not receiver_type:
            return None
        expected_owners = (
            set(self._source_structure.lineage(receiver_type))
            if receiver_type
            else set(self._source_structure.lineage(caller_owner))
        )
        if expected_owners:
            owned = [
                key
                for key in matches
                if str(self.helper_index[key].get("owner") or "") in expected_owners
            ]
            if len(owned) == 1:
                return owned[0]
        if len(matches) == 1 and not receiver:
            return matches[0]
        return None

    @staticmethod
    def _helper_parameter_count(helper: dict[str, Any]) -> int:
        return callable_parameter_count(helper)

    # ------------------------------------------------------------
    # Edges and IDs
    # ------------------------------------------------------------

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        *,
        kind: EdgeKind,
        role: Optional[str] = None,
        via: Optional[str] = None,
    ) -> None:
        key = (source_id, target_id, kind, role or "", via or "")
        if key in self.edges:
            return
        edge_id = self._make_id(
            "edge", (source_id, target_id, kind, role or "", via or "")
        )
        self.edges[key] = DAGEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            role=role,
            via=via,
        )

    @staticmethod
    def _make_id(prefix: str, key: Any) -> str:
        return stable_id(prefix, key)

    # ------------------------------------------------------------
    # Classification and snippets
    # ------------------------------------------------------------

    @staticmethod
    def _classify_operation(expression: str) -> str:
        expr = expression.strip()
        if re.match(r"^\s*(max|min|fmax|fmin|clamp|constrain)\s*\(", expr):
            return "reduction"
        if re.match(r"^\s*[^?]+\?[^:]+:", expr):
            return "reduction"
        if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_:]*\s*\(", expr):
            return "helper_call"
        return "assign"

    def _match_parameter(
        self,
        symbol: str,
        *,
        file: Optional[str] = None,
        scope_function: str = "",
        line: Optional[int] = None,
    ) -> Optional[str]:
        """Resolve a parameter from its source-declared storage identity."""
        raw = exact_symbol(
            symbol.replace(".get()", "").replace(".get", "")
        )
        if raw.startswith("this."):
            raw = raw[5:]
        root = raw.replace("->", ".").split(".", 1)[0]
        candidates = list(self._parameter_bindings_by_member.get(root, ()))
        if not candidates:
            return None
        reference = self._reference_identity(
            root, str(file or ""), scope_function, line
        )
        owner = str(reference.declaring_class or reference.class_owner or "")
        if owner:
            lineage = set(self._source_structure.lineage(owner)) or {owner}
            owned = [
                item
                for item in candidates
                if str(item.get("owner") or "") in lineage
            ]
            candidates = owned
        elif self._source_structure.authoritative_declarations:
            return None

        names = {str(item.get("name") or "") for item in candidates}
        names.discard("")
        return next(iter(names)) if len(names) == 1 else None

    def _binding_reaches_terminal(self, binding: dict[str, Any]) -> bool:
        target = exact_symbol(str(binding.get("target_symbol") or ""))
        return target == self.terminal

    def _snippet(self, file: Optional[str], line: Optional[int]) -> Optional[str]:
        if not file or not line or self.source_root is None:
            return None
        try:
            lines = self._file_lines_cache.get(file)
            if lines is None:
                text = (self.source_root / file).read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                self._file_lines_cache[file] = lines
        except (OSError, ValueError):
            return None
        start = max(0, line - 1 - self.snippet_context_lines)
        end = min(len(lines), line + self.snippet_context_lines)
        return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# Helper index
# ---------------------------------------------------------------------------


def _index_helpers(helpers: Sequence[Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index helpers by short name and stable callable identity."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for helper in helpers:
        as_dict = helper if isinstance(helper, dict) else _helper_to_dict(helper)
        name = str(as_dict.get("name") or "")
        if not name:
            continue
        short = name.split("::")[-1]
        callable_id = str(as_dict.get("callable_id") or "") or ":".join(
            [
                str(as_dict.get("file") or ""),
                str(as_dict.get("line") or 0),
                name,
                ",".join(str(value) for value in as_dict.get("parameters") or []),
            ]
        )
        owner, separator, _method = name.rpartition("::")
        if not as_dict.get("owner"):
            as_dict["owner"] = owner if separator else ""
        as_dict.setdefault("callable_id", callable_id)
        index[(short, callable_id)] = as_dict
    return index


def _helper_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(vars(value))


def _coerce_helpers(fetched: Any) -> list[Any]:
    """Normalize helper-provider return values into an iterable of helpers.

    Providers may return a single ``HelperExpressionRef`` / dict, an
    iterable of them, or ``None`` when the name has no body.
    """
    if fetched is None:
        return []
    if isinstance(fetched, (list, tuple, set)):
        return [item for item in fetched if item is not None]
    return [fetched]


_CLASS_CONTEXT_RE = re.compile(r"(?P<klass>[A-Za-z_][A-Za-z0-9_]*)::(?P<fn>[A-Za-z_][A-Za-z0-9_]*)")


def _class_context_from_name_or_evidence(name: str, evidence: str) -> str:
    """Extract ``Class::function`` from a helper name or its evidence string."""
    match = _CLASS_CONTEXT_RE.search(name)
    if match:
        return f"{match.group('klass')}::{match.group('fn')}"
    match = _CLASS_CONTEXT_RE.search(evidence)
    if match:
        return f"{match.group('klass')}::{match.group('fn')}"
    # Free function: use empty class marker.
    short = name.split("::")[-1]
    return f"::{short}"


# ---------------------------------------------------------------------------
# Predicate canonicalization
# ---------------------------------------------------------------------------


def _format_lowered_value(value: Any) -> str:
    """Render a resolved constant into a predicate-embeddable string.

    Booleans render as ``True``/``False`` (Python-shaped so the safe-eval
    branch consumers can parse them without extra substitution); numbers
    render via ``str`` so the safe-eval numeric path handles them.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


# C-style casts (``(float)x``, ``(int32_t)x``) that break ast parsing.
_CAST_RE = re.compile(
    r"\(\s*(?:const\s+)?(?:unsigned\s+|signed\s+)?"
    r"(?:float|double|bool|char|short|int|long|u?int(?:8|16|32|64)_t|size_t)"
    r"\s*\)"
)


def _normalize_cpp_expression(text: str) -> str:
    """Rewrite C++ operators/casts into a form ``source_expression_names``
    can ast-parse, so symbols inside branch predicates and cast-wrapped
    arguments are extractable.

    ``&&``/``||``/``!`` → Python boolean ops, ``->`` and ``::`` collapsed to
    ``.``, and C-style casts stripped. Without this a predicate like
    ``_param_x.get() > 0 && a->b == C::D`` yields no symbols at all (the
    ``&&`` fails ``ast.parse``), so no evidence edges — including the
    parameters and logged signals the branch reads.
    """
    result = text.replace("&&", " and ").replace("||", " or ")
    result = re.sub(r"!(?!=)", " not ", result)
    result = result.replace("->", ".").replace("::", ".")
    return _CAST_RE.sub(" ", result)


def _canonical_predicate(predicate: str) -> str:
    """Canonicalize a predicate string for branch deduplication.

    Whitespace-collapse + strip trailing punctuation. Keeps the string
    human-readable; a semantic canonicalizer (e.g. commutativity of
    ``&&`` operands) is Milestone 2 work.
    """
    canonical = " ".join(predicate.split())
    canonical = canonical.strip("(){} ;")
    return canonical


# ---------------------------------------------------------------------------
# Feasibility pre-evaluation (Milestone 2)
# ---------------------------------------------------------------------------


_PARAM_ACCESSOR_RE = re.compile(r"_param_(?P<name>[A-Za-z0-9_]+)\.get\(\s*\)")


def ground_expression_via_edges(
    expression: str,
    vertex_id: str,
    dag: MechanismDAG,
    *,
    enum_values: Optional[dict[str, Any]] = None,
    max_depth: int = 6,
) -> Optional[str]:
    """Render ``expression`` with graph producers for diagnostics only.

    This compatibility renderer is not used for feasibility or replay.
    Those paths evaluate producer values through :class:`DAGValueProgram` so
    symbol identity and producer selection remain graph-native. For display,
    each symbol the vertex reads (its incoming data edges' roles) is
    substituted with its producer's grounded form: a logged-signal leaf
    substitutes its signal name, a constant/parameter leaf its value or
    name, and an operation recursively grounds its own expression.
    Opaque producers stay as-is (the caller's evaluator then fails
    honestly). ``struct_s::NAME`` enum references substitute from
    ``enum_values``. Purely structural — works for any vertex kind on
    any module.
    """
    vertices_by_id = {v.id: v for v in dag.vertices}
    edges_by_target: dict[str, list[DAGEdge]] = {}
    for edge in dag.edges:
        if edge.kind == "data" and edge.role:
            edges_by_target.setdefault(edge.target_id, []).append(edge)

    def producer_form(producer_id: str, depth: int, seen: frozenset[str]) -> Optional[str]:
        vertex = vertices_by_id.get(producer_id)
        if vertex is None or producer_id in seen or depth > max_depth:
            return None
        if vertex.kind == "evidence":
            if vertex.sub_kind == "logged_signal" and vertex.signal_name:
                return str(vertex.signal_name)
            value = (vertex.metadata or {}).get("value")
            if value is not None:
                return _format_lowered_value(value)
            if vertex.sub_kind == "parameter" and vertex.signal_name:
                return str(vertex.signal_name)
            return None
        if vertex.kind == "operation" and vertex.expression:
            return ground(
                str(vertex.expression), producer_id, depth + 1, seen | {producer_id}
            )
        return None

    def ground(text: str, target_id: str, depth: int, seen: frozenset[str]) -> Optional[str]:
        # Align source spellings before matching edge roles. Tree-sitter
        # preserves qualified names exactly, while fallback extraction may
        # already have collapsed them.
        result = text.replace("->", ".").replace("::", ".")
        edges_by_role: dict[str, list[DAGEdge]] = defaultdict(list)
        target = vertices_by_id.get(target_id)
        call_roles = (
            dict((target.metadata or {}).get("source_call_roles") or {})
            if target
            else {}
        )
        for edge in edges_by_target.get(target_id, []):
            role = str(edge.role)
            if edge.via and role.startswith("call:"):
                role = str((call_roles.get(edge.via) or {}).get("call") or role)
            elif edge.via and role.startswith("call-result:"):
                role = str((call_roles.get(edge.via) or {}).get("result") or role)
            canonical_role = role.replace("->", ".").replace("::", ".")
            edges_by_role[canonical_role].append(edge)
        # A projected call is longer than its base call. Replace it first so
        # a source-proven logged projection wins before recursive helper
        # grounding considers the base result.
        for role in sorted(edges_by_role, key=len, reverse=True):
            role_edges = edges_by_role[role]
            replacements = {
                replacement
                for edge in role_edges
                if (replacement := producer_form(edge.source_id, depth, seen)) is not None
            }
            # Multiple reaching definitions require piecewise replay. A flat
            # grounded expression cannot choose one without losing control
            # semantics, so leave the expression explicitly unevaluable.
            if len(replacements) > 1:
                return None
            if len(replacements) == 1:
                result = substitute_expression_symbols(
                    result, [role], [next(iter(replacements))]
                )
        return result

    grounded = ground(str(expression or ""), vertex_id, 0, frozenset({vertex_id}))
    return grounded


def _freshness_gate_verdict(predicate: str) -> Optional[bool]:
    """TEMPORARY stopgap verdict for a standalone uORB freshness/timeout gate.

    PX4 guards paths with ``hrt_elapsed_time(&_last) < TIMEOUT`` (recent) or
    ``> TIMEOUT`` (stale). Both the elapsed time and the ``#define`` timeout are
    not yet grounded, so such a gate is otherwise unresolved and drags its
    guarded operation to unknown. As a permissive stand-in we assume the data
    is fresh (elapsed time ~ 0): a ``<``/``<=`` check is satisfied and a
    ``>``/``>=`` check is not, with a leading negation flipping the result so a
    gate and its complement stay consistent. Only a *standalone* comparison is
    resolved; a boolean combination keeps its real (unresolved) verdict.

    Unlike a dataman read this value IS in the log; replace this with
    log-derived freshness (a topic's last sample time vs the resolved timeout)
    so a genuine data gap can make the gate false.
    """
    text = predicate.strip()
    negated = False
    while text.startswith("!"):
        negated = not negated
        text = text[1:].strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
    if "hrt_elapsed_time" not in text and "hrt_absolute_time" not in text:
        return None
    if "&&" in text or "||" in text:
        return None
    if "<" in text:
        fresh = True
    elif ">" in text:
        fresh = False
    else:
        return None
    return (not fresh) if negated else fresh


def evaluate_feasibility(
    dag: MechanismDAG,
    *,
    parameter_values: Optional[dict[str, Any]] = None,
    enum_values: Optional[dict[str, Any]] = None,
    signal_samples: Optional[dict[str, list[tuple[float, Any]]]] = None,
    signal_policies: Optional[dict[str, Any]] = None,
    prepared_signal_series: Optional[dict[str, "PreparedSignalSeries"]] = None,
    value_program: Optional[DAGValueProgram] = None,
    value_session: Optional[DAGValueSession] = None,
    prune_dead: bool = True,
    dynamic_branch_ids: Optional[set[str]] = None,
    allow_assumptions: bool = True,
    stream_timestamps: bool = False,
) -> MechanismDAG:
    """Pre-evaluate each ``branch`` vertex against known constants and
    optionally against time-varying signal samples.

    Constants come from ``parameter_values`` and ``enum_values`` and are
    handled the same way as Milestone 2.

    When ``signal_samples`` is provided, branches whose predicates
    reference time-varying logged signals are additionally evaluated
    over the union of sample timestamps (hold-last policy between
    samples). The resulting True-intervals populate the vertex's
    ``active_windows``. Verdict semantics:

    * empty windows → ``always_false``
    * windows cover the entire sample span → ``always_true``
    * mixed → verdict stays ``unknown`` with populated ``active_windows``

    If ``prune_dead`` is set, operations whose only gating branches all
    resolve to ``always_false`` are removed along with the branches
    themselves (and their orphaned edges).

    ``dynamic_branch_ids`` restricts timestamp work to structurally ready
    checkpoint gates; static evaluation still visits every branch.
    ``allow_assumptions=False`` requires derived evidence for every verdict.
    ``stream_timestamps`` releases pure dynamic memoization after each shared
    timestamp batch without changing resampling or branch-window semantics.
    """
    samples = signal_samples or {}
    policies = signal_policies or {}
    prepared_series = (
        prepared_signal_series
        if prepared_signal_series is not None
        else prepare_signal_series(samples, policies)
    )
    program = value_program or (
        value_session.program if value_session is not None else DAGValueProgram(dag)
    )
    if value_session is not None and value_session.program is not program:
        raise ValueError("value_session must be bound to value_program")

    def resolve_sample(signal: str, timestamp: float) -> Optional[Any]:
        return sample_prepared_signal(prepared_series, signal, timestamp)

    session = value_session or program.bind(
        parameter_values=parameter_values,
        enum_values=enum_values,
        sample_resolver=resolve_sample,
    )
    branches = [vertex for vertex in dag.vertices if vertex.kind == "branch"]
    static_results = session.evaluate_many(
        tuple(vertex.id for vertex in branches), None
    )

    schedules: dict[str, tuple[float, ...]] = {}
    spans: dict[str, tuple[float, float]] = {}
    policy_summaries: dict[str, dict[str, str]] = {}
    policies_complete: dict[str, bool] = {}
    dynamic_roots_by_timestamp: dict[float, list[str]] = defaultdict(list)
    for vertex in branches:
        if static_results[vertex.id].status == "value":
            continue
        if dynamic_branch_ids is not None and vertex.id not in dynamic_branch_ids:
            continue
        referenced = tuple(
            signal
            for signal in program.observable_inputs_for(vertex.id)
            if signal in prepared_series
        )
        summary = {
            signal: str(
                ((prepared_series.get(signal).policy or {}).get("method") or "unknown")
                if prepared_series.get(signal) is not None
                else "unknown"
            )
            for signal in referenced
        }
        policy_summaries[vertex.id] = summary
        policies_complete[vertex.id] = bool(referenced) and all(
            prepared_series.get(signal) is not None
            and prepared_series[signal].policy is not None
            for signal in referenced
        )
        if not referenced or any(
            prepared_series.get(signal) is None
            or prepared_series[signal].span is None
            for signal in referenced
        ):
            continue
        start = max(prepared_series[signal].span[0] for signal in referenced)
        end = min(prepared_series[signal].span[1] for signal in referenced)
        if start > end:
            continue
        span = (start, end)
        timeline = {start, end}
        for signal in referenced:
            timeline.update(
                timestamp
                for timestamp in prepared_series[signal].times
                if start <= timestamp <= end
            )
        scheduled = tuple(sorted(timeline))
        if not scheduled:
            continue
        spans[vertex.id] = span
        schedules[vertex.id] = scheduled
        for timestamp in scheduled:
            dynamic_roots_by_timestamp[timestamp].append(vertex.id)

    dynamic_results: dict[str, list[tuple[float, Any]]] = defaultdict(list)
    dynamic_failures: dict[str, list[str]] = defaultdict(list)
    for timestamp in sorted(dynamic_roots_by_timestamp):
        timestamp_results = session.evaluate_many(
            tuple(dynamic_roots_by_timestamp[timestamp]), timestamp
        )
        for vertex_id, result in timestamp_results.items():
            if result.status == "value":
                dynamic_results[vertex_id].append((timestamp, result.value))
            else:
                dynamic_failures[vertex_id].append(result.reason or result.status)
        if stream_timestamps:
            session.release_timestamp_values()

    updated_vertices: list[DAGVertex] = []
    verdicts: dict[str, str] = {}
    for vertex in dag.vertices:
        if vertex.kind != "branch":
            updated_vertices.append(vertex)
            continue
        static_result = static_results[vertex.id]
        verdict = "unknown"
        windows: list[tuple[float, float]] = []
        if static_result.status == "value":
            verdict = "always_true" if bool(static_result.value) else "always_false"
        elif vertex.id in schedules:
            evaluated = dynamic_results.get(vertex.id, [])
            windows = boolean_sample_windows(evaluated)
            complete = len(evaluated) == len(schedules[vertex.id])
            if complete and policies_complete.get(vertex.id, False):
                if not windows:
                    verdict = "always_false"
                elif _covers_span(windows, spans[vertex.id]):
                    verdict = "always_true"

        # TEMPORARY stopgap: resolve a standalone uORB freshness/timeout gate
        # under the "data is fresh" assumption (see _freshness_gate_verdict).
        fresh_verdict: Optional[bool] = None
        if allow_assumptions and verdict == "unknown" and not windows:
            fresh_verdict = _freshness_gate_verdict(
                vertex.predicate_raw or vertex.predicate_lowered or ""
            )
        if fresh_verdict is not None:
            verdict = "always_true" if fresh_verdict else "always_false"

        verdicts[vertex.id] = verdict
        metadata = dict(vertex.metadata or {})
        metadata["evaluation_mode"] = "dag_value_plan"
        metadata["static_evaluation"] = (
            {
                "status": "value",
                "assumed": True,
                "reason": "temporary: uORB freshness gate assumed fresh",
            }
            if fresh_verdict is not None
            else {
                "status": static_result.status,
                "reason": static_result.reason,
            }
        )
        if vertex.id in spans:
            metadata["evaluation_domain"] = list(spans[vertex.id])
            metadata["sampling_policies"] = policy_summaries.get(vertex.id, {})
        if dynamic_failures.get(vertex.id):
            metadata["evaluation_failures"] = list(
                dict.fromkeys(dynamic_failures[vertex.id])
            )
        updated_vertices.append(
            vertex.model_copy(
                update={
                    "feasibility_verdict": verdict,
                    "active_windows": windows,
                    "metadata": metadata,
                }
            )
        )

    annotated = dag.model_copy(update={"vertices": updated_vertices, "edges": list(dag.edges)})
    return prune_infeasible_operations(annotated) if prune_dead else annotated


def prune_infeasible_operations(dag: MechanismDAG) -> MechanismDAG:
    """Apply existing conjunction pruning without reevaluating the graph."""
    dead_branches = {
        vertex.id for vertex in dag.vertices
        if vertex.kind == "branch" and vertex.feasibility_verdict == "always_false"
    }
    # A single false control conjunct makes its operation unreachable.
    dead_operations = {
        edge.target_id for edge in dag.edges
        if edge.kind == "control" and edge.source_id in dead_branches
    }
    kept = {vertex.id for vertex in dag.vertices} - dead_branches - dead_operations
    return dag.model_copy(update={
        "vertices": [vertex for vertex in dag.vertices if vertex.id in kept],
        "edges": [edge for edge in dag.edges
                  if edge.source_id in kept and edge.target_id in kept],
    })


def _reduce_predicate(
    predicate: str,
    parameter_values: dict[str, Any],
    enum_values: dict[str, Any],
) -> str:
    """Attempt to reduce ``predicate`` to ``always_true`` / ``always_false``.

    Returns ``"unknown"`` when the predicate cannot be reduced (missing
    values, unsupported syntax, method calls, etc.).
    """
    if not predicate.strip():
        return "unknown"

    from flight_log_agent.analysis.safe_eval import (
        ExpressionEvaluationError,
        eval_expression,
    )

    text = _substitute_predicate_syntax(predicate)

    env: dict[str, Any] = {}
    env.update(parameter_values)
    env.update(enum_values)

    try:
        result = eval_expression(text, env)
    except (ExpressionEvaluationError, TypeError, ValueError, ZeroDivisionError):
        return "unknown"

    if isinstance(result, bool):
        return "always_true" if result else "always_false"
    if isinstance(result, (int, float)):
        # C-style truthiness: non-zero is true.
        return "always_true" if result != 0 else "always_false"
    return "unknown"


# ---------------------------------------------------------------------------
# Interval evaluation over signal time-series (Milestone 3)
# ---------------------------------------------------------------------------


def _sample_span(
    signal_samples: dict[str, list[tuple[float, Any]]],
) -> Optional[tuple[float, float]]:
    """Return ``(min_ts, max_ts)`` across every signal, or None if empty."""
    all_ts: list[float] = []
    for samples in signal_samples.values():
        for ts, _ in samples:
            all_ts.append(ts)
    if not all_ts:
        return None
    return min(all_ts), max(all_ts)


def _predicate_signal_references(
    predicate: str,
    signal_samples: dict[str, list[tuple[float, Any]]],
) -> tuple[str, dict[str, str]]:
    """Return a safe-eval expression and alias-to-signal references.

    Exact token-boundary aliasing preserves array indices and prevents a
    short signal name from matching inside an unrelated longer reference.
    """
    lowered = _substitute_predicate_syntax(predicate)
    return alias_dotted_names(lowered, signal_samples.keys())


@dataclass(frozen=True)
class PreparedSignalSeries:
    """One required signal normalized once for all branch evaluations."""

    samples: tuple[tuple[float, Any], ...]
    times: tuple[float, ...]
    span: Optional[tuple[float, float]]
    policy: Optional[dict[str, Any]]


@dataclass(frozen=True)
class DAGValueSeries:
    """Timestamped values reconstructed from one MechanismDAG vertex."""

    samples: tuple[tuple[float, Any], ...]
    span: Optional[tuple[float, float]]
    referenced_signals: tuple[str, ...]
    policy_summary: dict[str, str]
    policies_complete: bool
    complete: bool
    reason: str = ""


def prepare_signal_series(
    signal_samples: dict[str, list[tuple[float, Any]]],
    signal_policies: Optional[dict[str, Any]] = None,
) -> dict[str, PreparedSignalSeries]:
    """Sort only supplied DAG signals, once per analysis run."""
    policies = signal_policies or {}
    prepared: dict[str, PreparedSignalSeries] = {}
    for signal, samples in signal_samples.items():
        ordered = tuple(sorted(samples, key=lambda item: float(item[0])))
        times = tuple(float(timestamp) for timestamp, _value in ordered)
        prepared[signal] = PreparedSignalSeries(
            samples=ordered,
            times=times,
            span=(times[0], times[-1]) if times else None,
            policy=_signal_policy(signal, policies),
        )
    return prepared


@dataclass(frozen=True)
class _PreparedPredicate:
    text: str
    alias_to_signal: dict[str, str]
    referenced: tuple[str, ...]
    span: Optional[tuple[float, float]]
    policies: dict[str, Optional[dict[str, Any]]]
    series: dict[str, PreparedSignalSeries]


def _prepare_predicate(
    predicate: str,
    signal_samples: dict[str, list[tuple[float, Any]]],
    signal_policies: Optional[dict[str, Any]] = None,
    *,
    prepared_series: Optional[dict[str, PreparedSignalSeries]] = None,
) -> _PreparedPredicate:
    text, aliases = _predicate_signal_references(predicate, signal_samples)
    referenced = tuple(dict.fromkeys(aliases.values()))
    span: Optional[tuple[float, float]] = None
    available_series = prepared_series or {}
    missing = {
        signal: signal_samples.get(signal, [])
        for signal in referenced
        if signal not in available_series
    }
    newly_prepared = (
        prepare_signal_series(missing, signal_policies or {}) if missing else {}
    )
    referenced_series = {
        signal: available_series.get(signal) or newly_prepared[signal]
        for signal in referenced
        if signal in available_series or signal in newly_prepared
    }
    if referenced and all(
        referenced_series.get(signal) is not None
        and referenced_series[signal].span is not None
        for signal in referenced
    ):
        start = max(referenced_series[signal].span[0] for signal in referenced)
        end = min(referenced_series[signal].span[1] for signal in referenced)
        if start <= end:
            span = (start, end)
    policies = {
        signal: referenced_series[signal].policy
        if signal in referenced_series
        else _signal_policy(signal, signal_policies or {})
        for signal in referenced
    }
    return _PreparedPredicate(
        text=text,
        alias_to_signal=aliases,
        referenced=referenced,
        span=span,
        policies=policies,
        series=referenced_series,
    )


def _predicate_sample_span(
    predicate: str,
    signal_samples: dict[str, list[tuple[float, Any]]],
    *,
    prepared: Optional[_PreparedPredicate] = None,
) -> Optional[tuple[float, float]]:
    """Common observed domain of the signals referenced by a predicate."""
    return (prepared or _prepare_predicate(predicate, signal_samples)).span


def _signal_policy(
    signal: str,
    signal_policies: dict[str, Any],
) -> Optional[dict[str, Any]]:
    policy = signal_policies.get(signal)
    if policy is None:
        schema_signal = re.sub(r"^([^.[\]]+)\[\d+\]", r"\1", signal)
        policy = signal_policies.get(schema_signal)
    if policy is None:
        return None
    if isinstance(policy, dict):
        return policy
    if hasattr(policy, "model_dump"):
        return policy.model_dump()
    return dict(vars(policy))


def _predicate_policies_complete(
    predicate: str,
    signal_samples: dict[str, list[tuple[float, Any]]],
    signal_policies: dict[str, Any],
    *,
    prepared: Optional[_PreparedPredicate] = None,
    prepared_series: Optional[dict[str, PreparedSignalSeries]] = None,
) -> bool:
    prepared = prepared or _prepare_predicate(
        predicate,
        signal_samples,
        signal_policies,
        prepared_series=prepared_series,
    )
    return bool(prepared.referenced) and all(
        prepared.policies.get(signal) is not None
        for signal in prepared.referenced
    )


def _predicate_policy_summary(
    predicate: str,
    signal_samples: dict[str, list[tuple[float, Any]]],
    signal_policies: dict[str, Any],
    *,
    prepared: Optional[_PreparedPredicate] = None,
) -> dict[str, str]:
    prepared = prepared or _prepare_predicate(
        predicate, signal_samples, signal_policies
    )
    return {
        signal: str((prepared.policies.get(signal) or {}).get("method") or "unknown")
        for signal in prepared.referenced
    }


def _sample_value_at(
    ordered: Sequence[tuple[float, Any]],
    times: Sequence[float],
    timestamp: float,
    policy: Optional[dict[str, Any]],
) -> Optional[Any]:
    """Resample one time-ordered series according to its schema-derived
    policy. ``ordered``/``times`` are pre-sorted ONCE by the caller — a
    per-timestamp sort made interval evaluation quadratic on real logs."""
    if not ordered:
        return None
    position = bisect_right(times, timestamp)
    if position and times[position - 1] == timestamp:
        return ordered[position - 1][1]
    if position == 0 or position >= len(ordered):
        return None

    before_t, before = ordered[position - 1]
    after_t, after = ordered[position]
    method = str((policy or {}).get("method") or "discrete_hold")
    if method == "discrete_hold":
        return before
    if method == "quaternion_slerp":
        # Indexed quaternion components cannot be interpolated independently.
        # A vector-aware evaluator is required for non-sample timestamps.
        return None
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    before_f = float(before)
    after_f = float(after)
    if math.isnan(before_f) or math.isnan(after_f) or after_t == before_t:
        return None
    fraction = (timestamp - before_t) / (after_t - before_t)
    if method == "angle_wrap":
        delta = (after_f - before_f + math.pi) % (2 * math.pi) - math.pi
        return (before_f + fraction * delta + math.pi) % (2 * math.pi) - math.pi
    if method == "linear":
        return before_f + fraction * (after_f - before_f)
    return None


def sample_prepared_signal(
    prepared_signal_series: dict[str, PreparedSignalSeries],
    signal: str,
    timestamp: float,
) -> Optional[Any]:
    """Read one already-prepared signal without sorting it again."""
    series = prepared_signal_series.get(signal)
    if series is None:
        return None
    return _sample_value_at(
        series.samples, series.times, timestamp, series.policy
    )


def evaluate_dag_vertex_series(
    dag: MechanismDAG,
    vertex_id: str,
    *,
    parameter_values: Optional[dict[str, Any]] = None,
    enum_values: Optional[dict[str, Any]] = None,
    signal_samples: Optional[dict[str, list[tuple[float, Any]]]] = None,
    signal_policies: Optional[dict[str, Any]] = None,
    prepared_signal_series: Optional[dict[str, PreparedSignalSeries]] = None,
    timestamps: Optional[Iterable[float]] = None,
    evaluation_windows: Optional[list[tuple[float, float]]] = None,
    value_program: Optional[DAGValueProgram] = None,
    value_session: Optional[DAGValueSession] = None,
) -> DAGValueSeries:
    """Evaluate one vertex from graph edges over its observed input domain.

    The operation text applies only local operators. Every operand value is
    selected from an incoming DAG edge by :class:`DAGValueProgram`; this function
    supplies timestamp alignment and schema-derived resampling policy.
    """
    samples = signal_samples or {}
    policies = signal_policies or {}
    program = value_program or (
        value_session.program if value_session is not None else DAGValueProgram(dag)
    )
    if value_session is not None and value_session.program is not program:
        raise ValueError("value_session must be bound to value_program")
    referenced = tuple(
        signal
        for signal in program.observable_inputs_for(vertex_id)
        if signal in samples or signal in (prepared_signal_series or {})
    )
    prepared = dict(prepared_signal_series or {})
    missing = {
        signal: samples.get(signal, [])
        for signal in referenced
        if signal not in prepared and signal in samples
    }
    if missing:
        prepared.update(prepare_signal_series(missing, policies))
    unavailable = [signal for signal in referenced if signal not in prepared]
    policy_summary = {
        signal: str((prepared.get(signal).policy or {}).get("method") or "unknown")
        if prepared.get(signal) is not None
        else "unknown"
        for signal in referenced
    }
    policies_complete = bool(referenced) and all(
        prepared.get(signal) is not None
        and prepared[signal].policy is not None
        for signal in referenced
    )
    if unavailable:
        return DAGValueSeries(
            samples=(),
            span=None,
            referenced_signals=referenced,
            policy_summary=policy_summary,
            policies_complete=False,
            complete=False,
            reason=f"missing observed inputs: {unavailable}",
        )

    span: Optional[tuple[float, float]] = None
    if referenced:
        spans = [prepared[signal].span for signal in referenced]
        if any(item is None for item in spans):
            return DAGValueSeries(
                samples=(),
                span=None,
                referenced_signals=referenced,
                policy_summary=policy_summary,
                policies_complete=policies_complete,
                complete=False,
                reason="one or more observed inputs have no samples",
            )
        start = max(item[0] for item in spans if item is not None)
        end = min(item[1] for item in spans if item is not None)
        if start > end:
            return DAGValueSeries(
                samples=(),
                span=None,
                referenced_signals=referenced,
                policy_summary=policy_summary,
                policies_complete=policies_complete,
                complete=False,
                reason="observed input domains do not overlap",
            )
        span = (start, end)

    requested_windows = list(evaluation_windows or [])
    timeline = {float(timestamp) for timestamp in (timestamps or ())}
    for signal in referenced:
        timeline.update(prepared[signal].times)
    if span is not None:
        timeline.update(span)
    for start, end in requested_windows:
        timeline.update((float(start), float(end)))

    def in_domain(timestamp: float) -> bool:
        if span is not None and not (span[0] <= timestamp <= span[1]):
            return False
        return not requested_windows or any(
            start <= timestamp <= end for start, end in requested_windows
        )

    scheduled = sorted(timestamp for timestamp in timeline if in_domain(timestamp))
    if not scheduled:
        return DAGValueSeries(
            samples=(),
            span=span,
            referenced_signals=referenced,
            policy_summary=policy_summary,
            policies_complete=policies_complete,
            complete=False,
            reason="no timestamps in the common evaluation domain",
        )

    values: list[tuple[float, Any]] = []
    failures: list[str] = []

    def resolve(signal: str, timestamp: float) -> Optional[Any]:
        return sample_prepared_signal(prepared, signal, timestamp)

    session = value_session or program.bind(
        parameter_values=parameter_values,
        enum_values=enum_values,
        sample_resolver=resolve,
    )

    for timestamp in scheduled:
        result = session.evaluate(vertex_id, timestamp)
        if result.status == "value":
            values.append((timestamp, result.value))
        else:
            failures.append(result.reason or result.status)
    return DAGValueSeries(
        samples=tuple(values),
        span=span,
        referenced_signals=referenced,
        policy_summary=policy_summary,
        policies_complete=policies_complete,
        complete=len(values) == len(scheduled),
        reason="; ".join(dict.fromkeys(failures)),
    )


def boolean_sample_windows(
    samples: Sequence[tuple[float, Any]],
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    current_start: Optional[float] = None
    last_timestamp: Optional[float] = None
    for timestamp, value in samples:
        last_timestamp = timestamp
        if bool(value) and current_start is None:
            current_start = timestamp
        elif not bool(value) and current_start is not None:
            windows.append((current_start, timestamp))
            current_start = None
    if current_start is not None and last_timestamp is not None:
        windows.append((current_start, last_timestamp))
    return windows


def _covers_span(
    windows: list[tuple[float, float]],
    span: tuple[float, float],
) -> bool:
    """True when the windows contain the whole span (single interval, aligned)."""
    if len(windows) != 1:
        return False
    start, end = windows[0]
    return start <= span[0] and end >= span[1]


def _substitute_predicate_syntax(predicate: str) -> str:
    """Lower a C++ predicate to safe-eval form.

    Routes through :func:`normalize_source_expression` — the centralized
    C++→Python lowering (macro spellings like ``PX4_ISFINITE``, float
    suffixes, ``true``/``false``, ternaries) — after substituting PX4
    parameter accessors and collapsing ``->``/``::``. Duplicating a
    partial subset here left this path failing on tokens the legacy
    evaluator handled.
    """
    from flight_log_agent.analysis.source_expression import (
        normalize_source_expression,
    )

    text = _PARAM_ACCESSOR_RE.sub(lambda m: m.group("name").upper(), predicate)
    text = text.replace("->", ".").replace("::", ".")
    return normalize_source_expression(text)


def _evaluate_predicate_intervals(
    predicate: str,
    parameter_values: dict[str, Any],
    enum_values: dict[str, Any],
    signal_samples: dict[str, list[tuple[float, Any]]],
    signal_policies: Optional[dict[str, Any]] = None,
    *,
    prepared: Optional[_PreparedPredicate] = None,
    prepared_series: Optional[dict[str, PreparedSignalSeries]] = None,
) -> Optional[list[tuple[float, float]]]:
    """Evaluate ``predicate`` per timestamp, return True-intervals.

    Uses the supplied schema-derived policy for each signal. A missing
    policy falls back to hold-last for window construction, but callers
    must not promote those windows to an always-true/false verdict.
    Returns None when no referenced signal has samples or when the
    predicate fails to evaluate at every timestamp.
    """
    if not predicate.strip():
        return None

    from flight_log_agent.analysis.safe_eval import (
        ExpressionEvaluationError,
        eval_expression,
    )

    prepared = prepared or _prepare_predicate(
        predicate,
        signal_samples,
        signal_policies,
        prepared_series=prepared_series,
    )
    text = prepared.text
    alias_to_signal = prepared.alias_to_signal
    signal_key_map = {signal: alias for alias, signal in alias_to_signal.items()}
    referenced = list(prepared.referenced)

    if not referenced:
        return None

    span = prepared.span
    if span is None:
        return None

    # Union of timestamps from referenced signals, restricted to their
    # common observed domain.
    all_ts: set[float] = set()
    for signal in referenced:
        for ts, _ in prepared.series[signal].samples:
            if span[0] <= ts <= span[1]:
                all_ts.add(ts)
    all_ts.update(span)
    if not all_ts:
        return None
    ts_sorted = sorted(all_ts)

    intervals: list[tuple[float, float]] = []
    current_start: Optional[float] = None
    ever_evaluated = False
    last_evaluated: Optional[float] = None

    for t in ts_sorted:
        resampled = {
            signal_key_map[signal]: _sample_value_at(
                series.samples, series.times, t, series.policy
            )
            for signal, series in prepared.series.items()
        }
        if any(value is None for value in resampled.values()):
            continue

        env = {**parameter_values, **enum_values, **resampled}
        try:
            result = eval_expression(text, env)
        except (ExpressionEvaluationError, TypeError, ValueError, ZeroDivisionError):
            return None

        ever_evaluated = True
        last_evaluated = t
        truthy = bool(result) if isinstance(result, (bool, int, float)) else False

        if truthy and current_start is None:
            current_start = t
        elif not truthy and current_start is not None:
            intervals.append((current_start, t))
            current_start = None

    if not ever_evaluated:
        return None
    if current_start is not None and last_evaluated is not None:
        intervals.append((current_start, last_evaluated))
    return intervals


# ---------------------------------------------------------------------------
# Milestone 4 — subgraph slicing and disk cache
# ---------------------------------------------------------------------------


def split_by_terminal(dag: MechanismDAG) -> list[MechanismDAG]:
    """Split a DAG into one subgraph per terminal operation.

    A terminal operation is any ``operation`` vertex whose
    ``metadata['is_terminal']`` is truthy — the DAG builder marks these
    when the operation's target matches the requested terminal symbol.

    Each returned subgraph contains the terminal operation and every
    vertex reachable backward from it via any edge kind. Vertices
    shared by multiple terminals (helper subgraph reuse, common
    parameters) appear in each subgraph.

    If no operation is marked terminal, returns ``[dag]`` unchanged so
    callers don't need to special-case that.
    """
    terminal_ops = [
        v for v in dag.vertices
        if v.kind == "operation" and v.metadata.get("is_terminal")
    ]
    if len(terminal_ops) <= 1:
        return [dag]

    from collections import defaultdict as _defaultdict

    edges_by_target: dict[str, list[DAGEdge]] = _defaultdict(list)
    for edge in dag.edges:
        edges_by_target[edge.target_id].append(edge)

    vertices_by_id = {v.id: v for v in dag.vertices}

    subgraphs: list[MechanismDAG] = []
    for index, terminal_op in enumerate(terminal_ops):
        reachable = _backward_reachable(terminal_op.id, edges_by_target)
        selected_vertices = [vertices_by_id[vid] for vid in reachable if vid in vertices_by_id]
        selected_edges = [
            edge for edge in dag.edges
            if edge.source_id in reachable and edge.target_id in reachable
        ]
        # Unresolved symbols surfaced only if any selected evidence vertex
        # references them — keeps subgraph payloads tight.
        surviving_symbols = {
            v.signal_name for v in selected_vertices
            if v.kind == "evidence" and v.sub_kind == "opaque_symbol" and v.signal_name
        }
        subgraphs.append(
            MechanismDAG(
                dag_id=f"{dag.dag_id}_sub{index}",
                terminal=dag.terminal,
                vertices=selected_vertices,
                edges=selected_edges,
                unresolved_symbols=sorted(surviving_symbols),
                unresolved_references=[
                    reference for reference in dag.unresolved_references
                    if not reference.origin_vertex_ids or reachable.intersection(reference.origin_vertex_ids)
                ],
                pending_construction=[vertex_id for vertex_id in dag.pending_construction if vertex_id in reachable],
            )
        )
    return subgraphs


def _backward_reachable(
    start_id: str,
    edges_by_target: dict[str, list[DAGEdge]],
) -> set[str]:
    """Return every vertex id reachable backward from ``start_id`` via any edge."""
    seen = {start_id}
    frontier = [start_id]
    while frontier:
        current = frontier.pop()
        for edge in edges_by_target.get(current, []):
            if edge.source_id not in seen:
                seen.add(edge.source_id)
                frontier.append(edge.source_id)
    return seen


# --- Disk cache -----------------------------------------------------------


_FS_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _terminal_slug(terminal: str) -> str:
    """Filesystem-safe slug for a terminal symbol.

    ``position_setpoint_triplet.current.alt`` → ``position_setpoint_triplet__current__alt``.
    """
    slug = terminal.strip().strip(".").replace(".", "__")
    slug = _FS_SANITIZE_RE.sub("_", slug)
    return slug or "unknown"


def layer2_cache_path(cache_root: Path, source_hash: str, terminal: str) -> Path:
    """Layer 2 (unresolved DAG) path: keyed by source + terminal."""
    return Path(cache_root) / "dag" / source_hash / f"{_terminal_slug(terminal)}.json"


def layer3_cache_path(
    cache_root: Path,
    source_hash: str,
    ulog_hash: str,
    terminal: str,
) -> Path:
    """Layer 3 (flight-annotated DAG) path: keyed by source + ulog + terminal."""
    return (
        Path(cache_root)
        / "dag_annotated"
        / source_hash
        / ulog_hash
        / f"{_terminal_slug(terminal)}.json"
    )


def write_dag_to_cache(dag: MechanismDAG, path: Path) -> None:
    """Serialize ``dag`` to ``path`` atomically.

    Writes to ``path.tmp`` first, then renames over ``path`` so a
    partial write can't corrupt an existing cache entry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dag.model_dump_json(), encoding="utf-8")
    tmp.replace(path)


def read_dag_from_cache(path: Path) -> Optional[MechanismDAG]:
    """Deserialize a DAG from ``path``.

    Returns ``None`` on missing file or unparseable payload. Callers
    treat that as a cache miss and rebuild.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return MechanismDAG.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Argument parsing for helper call sites
# ---------------------------------------------------------------------------


def _pointer_param_positions(helper: dict[str, Any]) -> dict[str, int]:
    """Return positions of source-proven pointer/reference outputs.

    The profiler stamps ``pointer_output_writes`` with the FORMAL name; the
    DAG builder needs the positional index to look up the caller's actual
    argument. Positions are inferred from ``parameters`` order; type info
    ``*`` vs ``&`` is intentionally not needed here: the profiler only emits
    an entry after syntax proves alias-capable storage and a write through it.
    """
    positions: dict[str, int] = {}
    formals = helper.get("parameters") or []
    referenced = {str(write.get("param") or "") for write in (helper.get("pointer_output_writes") or [])}
    for index, formal in enumerate(formals):
        name = str(formal).strip()
        if name and name in referenced:
            positions[name] = index
    return positions


def derive_pointer_output_bindings(
    pointer_writes: Sequence[dict[str, str]],
    pointer_params: dict[str, int],
    call_args: Sequence[str],
) -> list[dict[str, str]]:
    """Substitute call-site arguments into a helper's pointer-output writes.

    Returns entries ``{"param": <formal>, "field": <field>,
    "target": <arg.field>, "expression": <RHS>}``. DAG construction uses
    these records to emit an explicit actual-storage operation fed by the
    callee's formal write. The profiler also retains this helper for legacy
    non-DAG consumers, but flattened records are excluded from DAG facts.

    ``call_args`` are treated as already stripped of language-specific
    prefixes (C++ type declarations, ``&`` address-of). Callers that
    receive raw text from source (the profiler when it scans function
    definitions) preprocess before invoking; callers with pre-extracted
    argument expressions (the DAG builder over caller binding
    ``source_symbol``) pass args unmodified.
    """
    if not pointer_writes or not pointer_params:
        return []
    results: list[dict[str, str]] = []
    for write in pointer_writes:
        param = str(write.get("param") or "")
        field = str(write.get("field") or "")
        expression = str(write.get("expression") or "")
        index = pointer_params.get(param)
        if index is None or index >= len(call_args) or not expression:
            continue
        arg = str(call_args[index]).strip().lstrip("&").strip()
        if not arg:
            continue
        results.append({
            "param": param,
            "field": field,
            "target": f"{arg}.{field}" if field else arg,
            "expression": expression,
        })
    return results


def _extract_call_arguments(func_name: str, expression: str) -> list[str]:
    """Return the top-level, comma-separated arguments of the first
    ``func_name(...)`` occurrence in ``expression``.

    Nested parens count toward depth so commas inside sub-calls stay
    grouped. Returns an empty list when the call cannot be located or
    has no arguments.
    """
    index = expression.find(func_name)
    while index >= 0:
        after = index + len(func_name)
        # Skip whitespace; must land on `(`.
        cursor = after
        while cursor < len(expression) and expression[cursor].isspace():
            cursor += 1
        if cursor < len(expression) and expression[cursor] == "(":
            # Boundary check on the left so ``get_absolute_altitude_for_item``
            # doesn't match inside ``suffix_get_absolute_altitude_for_item``.
            left_char = expression[index - 1] if index > 0 else ""
            if left_char.isalnum() or left_char == "_":
                index = expression.find(func_name, after)
                continue
            depth = 0
            args: list[str] = []
            current: list[str] = []
            for char in expression[cursor:]:
                if char == "(":
                    depth += 1
                    if depth > 1:
                        current.append(char)
                    continue
                if char == ")":
                    depth -= 1
                    if depth == 0:
                        text = "".join(current).strip()
                        if text:
                            args.append(text)
                        return args
                    current.append(char)
                    continue
                if char == "," and depth == 1:
                    args.append("".join(current).strip())
                    current = []
                    continue
                current.append(char)
            return args
        index = expression.find(func_name, after)
    return []
