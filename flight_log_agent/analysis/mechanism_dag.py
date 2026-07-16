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

from flight_log_agent.analysis.parameter_lookup import (
    CXX_STDLIB_CONSTANTS,
    is_px4_parameter_name,
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
from flight_log_agent.expression_math import is_safe_math_function_name
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
    normalize_symbol,
    parse_signal_reference,
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


@dataclass(frozen=True)
class _ExpressionCall:
    """One syntactic call occurrence inside a source expression."""

    name: str
    receiver: str
    args: tuple[str, ...]
    offset: int
    source_site_id: str


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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _as_binding_dict(binding: Any) -> dict[str, Any]:
    if isinstance(binding, dict):
        return binding
    if hasattr(binding, "model_dump"):
        return binding.model_dump(exclude_none=True)
    return dict(vars(binding))


def _logged_signals_from_inventory(inventory: Optional[dict[str, Any]]) -> set[str]:
    """Derive the ``topic.field`` logged-signal set from an inventory.

    Reimplemented here (not imported from BindingIndex) so the DAG builder
    has no dependency on that module.
    """
    return observed_signals_from_inventory(inventory)


def _parse_numeric_literal(expression: str) -> Optional[Any]:
    """Return the numeric value of a compile-time-constant expression, else None.

    Handles bare int/float/hex literals and simple constant arithmetic
    (``(1 << 5)``) via the restricted evaluator. References to other names
    fail (empty env), so only genuine literals resolve — which is exactly
    what a source-defined constant (enum entry / ``#define`` / ``constexpr``)
    should be.
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
        value = eval_expression(text, {})
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
    parameter_aliases: Optional[dict[str, str]] = None,
    snippet_context_lines: int = 3,
    terminal_file: Optional[str] = None,
    terminal_identity: Optional[SourceSymbolIdentity | dict[str, Any]] = None,
    call_statements: Sequence[Any] = (),
    boundary_bindings: Sequence[Any] = (),
    enum_registry: Optional[dict[str, dict[str, Any]]] = None,
    source_structure: Optional[SourceStructureIndex] = None,
) -> MechanismDAG:
    """Build a mechanism DAG for ``terminal``.

    ``source_bindings`` are the profiler's source-assignment records
    (``target_symbol``, ``source_symbol``, ``control_predicates``,
    ``assignment_path``, ``struct_variables``, optional ``logged_signal``).
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
    literal. ``parameter_aliases`` maps a PX4 parameter *member*
    (``_param_rtl_cone_half_angle_deg``) to its canonical name
    (``RTL_CONE_ANG``) — built from ``ParameterRef.member`` / ``.name``
    (the ``DEFINE_PARAMETERS`` map). Without it the builder can only
    resolve params whose member name equals the param name; PX4 members
    routinely drop the module prefix, so the alias map is what resolves
    the rest. ``source_root`` enables per-vertex snippet embedding —
    omit to keep tests hermetic. ``terminal_file`` (optional) scopes the
    terminal's writers to that file when any exist there — a
    multi-module binding set can contain a same-named but unrelated
    variable from another class; the hint keeps the slice on the module
    the caller actually asked about.

    Identity is the EXACT symbol spelling (``exact_symbol``): indices,
    instances, and the leading-underscore member marker all distinguish.
    Source-object to logged-topic equivalence requires an explicit
    ``boundary_bindings`` entry.
    """
    builder = _DAGBuilder(
        source_bindings=[_as_binding_dict(b) for b in source_bindings],
        terminal=terminal,
        helper_index=_index_helpers(helper_expressions),
        helper_body_provider=helper_body_provider,
        parameter_predicates=list(parameter_predicates),
        parameter_values=dict(parameter_values or {}),
        source_root=Path(source_root) if source_root else None,
        logged_signals=set(logged_signals or ()) | _logged_signals_from_inventory(inventory),
        schema_signals=set(schema_signals or ()),
        parameter_names=set(parameter_names or ()),
        parameter_aliases=dict(parameter_aliases or {}),
        snippet_context_lines=snippet_context_lines,
        terminal_file=terminal_file,
        terminal_identity=terminal_identity,
        call_statements=[_as_binding_dict(c) for c in call_statements],
        boundary_bindings=[_as_binding_dict(b) for b in boundary_bindings],
        enum_registry=dict(enum_registry or {}),
        source_structure=source_structure or SourceStructureIndex(),
    )
    return builder.build()


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
        parameter_aliases: dict[str, str],
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
        self._boundary_bindings = list(boundary_bindings or [])
        # Guards dotted-root rebinding recursion against alias cycles.
        self._rebinding_stack: set[tuple[str, str, str]] = set()
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
        # PX4 parameter member → canonical name (DEFINE_PARAMETERS map).
        # Keyed by both the raw member and its normalized form so
        # ``_match_parameter`` can look up either. Param names imply an
        # identity alias so alias lookup alone covers every known param.
        self._parameter_aliases: dict[str, str] = {}
        for member, name in (parameter_aliases or {}).items():
            if not member or not name:
                continue
            self._parameter_aliases[member] = name
            self._parameter_aliases[normalize_symbol(member)] = name
            self.parameter_names.add(name)
        self.snippet_context_lines = snippet_context_lines
        self._parameter_values = {
            str(k).upper(): v for k, v in (parameter_values or {}).items()
        }
        self._parameter_predicate_by_predicate: dict[str, dict[str, Any]] = {}
        for record in parameter_predicates or []:
            entry = record if isinstance(record, dict) else (
                record.model_dump(exclude_none=True) if hasattr(record, "model_dump") else dict(vars(record))
            )
            predicate = str(entry.get("predicate") or "").strip()
            if predicate:
                self._parameter_predicate_by_predicate.setdefault(
                    _canonical_predicate(predicate), entry
                )

        self._schema_signals = {exact_symbol(s) for s in schema_signals if s}
        self._schema_shapes = {strip_symbol_indices(s) for s in self._schema_signals}

        # Native backward-walk indexes over the source bindings — the DAG
        # owns the walk rather than delegating to BindingIndex. ``_by_output``
        # keys on the resolved logged signal, ``_by_target`` on the written
        # symbol — both on the EXACT identity, with index-erased shape maps
        # so an index-free reference still finds its indexed writers (and
        # vice versa) without ever fusing distinct indices.
        self._source_bindings: list[dict[str, Any]] = list(source_bindings)
        self._all_bindings: list[dict[str, Any]] = list(self._source_bindings)
        self._registered_call_instances: set[str] = set()
        self._by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._target_shapes: dict[str, list[str]] = defaultdict(list)
        self._output_shapes: dict[str, list[str]] = defaultdict(list)
        self._reference_identities_by_scope: dict[
            tuple[str, str, str], list[SourceSymbolIdentity]
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
        writes_per_target: dict[str, int] = defaultdict(int)
        for binding in self._all_bindings:
            logged = exact_symbol(str(binding.get("logged_signal") or ""))
            target = exact_symbol(
                str(binding.get("target_symbol") or binding.get("target") or "")
            )
            if logged:
                self._index_binding(self._by_output, self._output_shapes, logged, binding)
            if target:
                self._index_binding(self._by_target, self._target_shapes, target, binding)
                writes_per_target[target] += 1
            site_file, site_callable = self._binding_site_scope(binding)
            for raw_symbol, raw_identity in (
                binding.get("reference_identities") or {}
            ).items():
                try:
                    identity = SourceSymbolIdentity.model_validate(raw_identity)
                except (TypeError, ValueError):
                    continue
                symbol = exact_symbol(str(raw_symbol))
                self._reference_identities_by_scope[
                    (site_file, site_callable, symbol)
                ].append(identity)
                self._reference_identities_by_file[(site_file, symbol)].append(identity)
                self._reference_identities_by_callable[
                    (site_callable, symbol)
                ].append(identity)
                self._reference_identities_by_symbol[symbol].append(identity)

        # Source-defined numeric constants (enum entry / ``#define`` /
        # ``constexpr``), resolved natively from the bindings: a single
        # unconditional write whose RHS is a compile-time numeric literal.
        # Replaces the old dependency on ``BindingIndex.assignment_resolutions``
        # (which stored SliceResult objects the DAG mis-typed).
        self._source_constants: dict[str, Any] = {}
        for binding in self._all_bindings:
            target = exact_symbol(
                str(binding.get("target_symbol") or binding.get("target") or "")
            )
            if (
                not target
                or writes_per_target[target] != 1
                or binding.get("declaration_kind") not in {"enum", "define", "constexpr"}
            ):
                continue
            if binding.get("control_predicates"):
                continue
            value = _parse_numeric_literal(
                str(binding.get("source_symbol") or binding.get("expression") or "")
            )
            if value is not None:
                self._source_constants.setdefault(target, value)

        # Struct-typed variable → struct type map, aggregated from every
        # source binding and helper record. Used by
        # :meth:`_resolve_struct_var_field` to derive ``var.field →
        # topic.field`` graph-natively via
        # :func:`_derive_topic_from_return_type`.
        self._struct_variables: dict[str, str] = {}
        for helper in helper_index.values():
            for name, struct_type in (helper.get("struct_variables") or {}).items():
                if name and struct_type:
                    self._struct_variables.setdefault(str(name), str(struct_type))
        for binding in self._all_bindings:
            for name, struct_type in (binding.get("struct_variables") or {}).items():
                if name and struct_type:
                    self._struct_variables.setdefault(str(name), str(struct_type))

        self.vertices: dict[str, DAGVertex] = {}
        self.edges: dict[tuple[str, str, str, str], DAGEdge] = {}
        self.unresolved_symbols: set[str] = set()
        self._unresolved_references: dict[tuple[Any, ...], UnresolvedSourceReference] = {}

        # Memoization tables.
        self._evidence_by_signal: dict[tuple[str, str], str] = {}
        # Branch vertices key on predicate plus the parser's source-node ID.
        # File/line remain the fallback for legacy facts.
        self._branch_by_site: dict[tuple[str, str, int, str, str], str] = {}
        self._helper_subgraph_return_id: dict[tuple[str, str, str], str] = {}
        self._helper_instance_site: dict[tuple[str, str, str], str] = {}
        # (helper key, call site) -> ordered formal parameter vertices.
        # Callers wire their i-th argument's producer to the i-th formal
        # vertex; helper body operations that reference the formal look
        # it up in this map (scoped to the helper).
        self._helper_parameter_vertices: dict[
            tuple[str, str, str], list[tuple[str, str]]
        ] = {}
        # Assignment-target index: EXACT target symbol → list of vertex ids
        # that produce it, plus the index-erased shape map for
        # index-compatible lookups.
        self._producers_by_symbol: dict[str, list[str]] = {}
        self._producer_shapes: dict[str, list[str]] = defaultdict(list)

        # Snippet cache to avoid re-reading a file many times.
        self._file_lines_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------
    # Exact-identity indexes
    # ------------------------------------------------------------
    #
    # Every index keys on ``exact_symbol`` — the lossless identity.
    # Lookups accept any index-compatible spelling of the SAME shape
    # (an index-free reference reads its indexed writers and an indexed
    # reference reads whole-object writers), but two explicit indices
    # never fuse. This replaces the old lossy ``normalize_symbol`` keys
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

    def _targets_matching(self, symbol_exact: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for key in self._matching_keys(self._target_shapes, symbol_exact):
            for binding in self._by_target.get(key, ()):
                if id(binding) not in seen:
                    seen.add(id(binding))
                    out.append(binding)
        return out

    def _outputs_matching(self, symbol_exact: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for key in self._matching_keys(self._output_shapes, symbol_exact):
            for binding in self._by_output.get(key, ()):
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
        symbol = exact_symbol(reference)
        if symbol in self.logged_signals:
            return symbol
        parsed = parse_signal_reference(symbol)
        if parsed is None:
            return None
        topic, requested_instance, field = parsed
        candidates: list[str] = []
        for candidate in self.logged_signals:
            candidate_parts = parse_signal_reference(candidate)
            if candidate_parts is None:
                continue
            candidate_topic, candidate_instance, candidate_field = candidate_parts
            if candidate_topic != topic or candidate_field != field:
                continue
            if requested_instance is not None and candidate_instance != requested_instance:
                continue
            candidates.append(candidate)
        return candidates[0] if len(candidates) == 1 else None

    def _boundary_topic_for(
        self,
        source_symbol: str,
        *,
        file: Optional[str] = None,
        scope_function: str = "",
        direction: str = "subscribe",
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Resolve a source object to a topic through profiler-proven uORB flow.

        Ambiguous bindings remain unresolved. File/callable context narrows a
        copied local; module-level subscription objects can still resolve from
        companion headers when their variable-to-topic mapping is unique.
        """
        symbol = exact_symbol(source_symbol)
        candidates = [
            item
            for item in self._boundary_bindings
            if exact_symbol(str(item.get("source_symbol") or "")) == symbol
            and str(item.get("direction") or "") == direction
        ]
        if file:
            exact_file = [item for item in candidates if str(item.get("file") or "") == file]
            if exact_file:
                candidates = exact_file
        if scope_function:
            scoped = [
                item
                for item in candidates
                if not item.get("function")
                or str(item.get("function")) == scope_function
                or str(item.get("callable_id") or "") == scope_function
            ]
            if scoped:
                candidates = scoped
            else:
                consumer_owner = self._callable_owner(scope_function)
                owner_scoped = [
                    item
                    for item in candidates
                    if consumer_owner
                    and self._same_source_owner(
                        str(item.get("source_owner") or ""), consumer_owner
                    )
                ]
                if owner_scoped:
                    candidates = owner_scoped
                elif any(
                    item.get("function") or item.get("callable_id")
                    for item in candidates
                ):
                    # A method-scoped copy with no structurally proven common
                    # class owner is a local. It must not cross methods merely
                    # because the destination has the same spelling.
                    return None
        placements = {
            (str(item.get("topic") or ""), item.get("instance"))
            for item in candidates
            if item.get("topic")
        }
        if len(placements) != 1:
            return None
        topic, instance = next(iter(placements))
        provenance = next(
            item
            for item in candidates
            if str(item.get("topic") or "") == topic
            and item.get("instance") == instance
        )
        if instance is not None:
            topic = f"{topic}[{instance}]"
        else:
            observed_instances = {
                parsed[1]
                for signal in self.logged_signals
                if (parsed := parse_signal_reference(signal)) is not None
                and parsed[0] == topic
                and parsed[1] is not None
            }
            if len(observed_instances) == 1:
                topic = f"{topic}[{next(iter(observed_instances))}]"
        return topic, provenance

    @staticmethod
    def _callable_owner(callable_identity: str) -> str:
        """Extract a qualified callable's owner from a profiler identity."""
        matches = re.findall(
            r"(?<![A-Za-z0-9_])"
            r"(?P<qualified>[A-Za-z_][A-Za-z0-9_]*"
            r"(?:::[A-Za-z_][A-Za-z0-9_]*)+)",
            str(callable_identity or ""),
        )
        if not matches:
            return ""
        owner, separator, _method = matches[-1].rpartition("::")
        return owner if separator else ""

    @staticmethod
    def _same_source_owner(proven_owner: str, consumer_owner: str) -> bool:
        """Compare source-derived owners while tolerating namespace context."""
        left = str(proven_owner or "").strip(":")
        right = str(consumer_owner or "").strip(":")
        return bool(
            left
            and right
            and (
                left == right
                or left.endswith(f"::{right}")
                or right.endswith(f"::{left}")
            )
        )

    # ------------------------------------------------------------
    # Build
    # ------------------------------------------------------------

    def build(self) -> MechanismDAG:
        """Build the DAG via one interleaved backward fixpoint.

        Starting from the terminal, we emit operation vertices and helper
        subgraphs *as we discover them*, driving a single frontier of
        symbols. Crucially, when a helper subgraph is materialized, the
        symbols in its body are fed back into the same frontier — so a
        writer that is only referenced *inside* a helper (e.g.
        ``_destination.lat`` inside a distance helper) still gets pulled
        in. There is no flattened intermediate reach list; the graph is
        constructed directly, and struct-root field expansion is folded
        into the same walk.
        """
        # Every frontier symbol carries the scope of the site that
        # referenced it — writers are then resolved under C++-faithful
        # visibility instead of a global by-name index, so a local named
        # ``dt`` in one module can never bind to another module's ``dt``.
        Scope = tuple[str, str, Optional[int]]  # (source unit, callable, read line)
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
        )
        frontier: deque[tuple[str, Any, Scope]] = deque(
            [("symbol", self.terminal_raw, terminal_scope)]
        )
        walked: set[tuple[str, Scope]] = set()
        materialized_helpers: set[str] = set()
        self._emitted_ids: set[int] = set()
        self._emitted_bindings: list[dict[str, Any]] = []

        def enqueue_expression(expression: str, scope: Scope) -> None:
            if not expression:
                return
            normalized_expr = _normalize_cpp_expression(str(expression))
            for raw in dedupe_keep_order(source_expression_names(normalized_expr)):
                if raw:
                    frontier.append(("symbol", raw, scope))
            for invocation in self._find_helper_calls(
                expression,
                scope[0],
                scope_function=scope[1],
                line=scope[2],
            ):
                if invocation.source_site_id not in materialized_helpers:
                    frontier.append(("helper", invocation, scope))

        while frontier:
            kind, payload, scope = frontier.popleft()
            if kind == "symbol":
                raw = payload
                norm = exact_symbol(raw)
                if not norm or (norm, scope) in walked:
                    continue
                walked.add((norm, scope))
                if norm != self.terminal and norm in self._source_constants:
                    # Source-defined constants resolve as value-carrying
                    # evidence leaves at wiring time; walking their single
                    # literal write would demote them to bare operations.
                    continue
                if (
                    norm != self.terminal
                    and self._observed_signal_placement(norm) is not None
                ):
                    # Known ground: the log records this exact field at one
                    # unambiguous topic-instance placement, so it is an
                    # evidence leaf — walking its
                    # publisher would widen into another module.
                    continue
                if norm == self.terminal:
                    writers = self._writers_of(norm)
                    call_writers = self._writers_from_relevant_calls(
                        norm, raw, scope, terminal=True
                    )
                    writers = self._dedupe_bindings([*writers, *call_writers])
                    if writers and self.terminal_identity is not None:
                        writers = [
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
                    elif writers and self.terminal_file:
                        # Scope the terminal to its module; fall back to
                        # all writers when none live in the hinted file.
                        scoped = [
                            b for b in writers
                            if self._binding_first_file(b) == self.terminal_file
                        ]
                        if scoped:
                            writers = scoped
                    writers = self._terminal_call_context_writers(writers)
                else:
                    writers = self._scoped_writers(norm, raw, scope)
                    call_writers = self._writers_from_relevant_calls(
                        norm,
                        raw,
                        scope,
                        terminal=False,
                    )
                    writers = self._dedupe_bindings([*writers, *call_writers])
                if writers:
                    for binding in writers:
                        if id(binding) not in self._emitted_ids:
                            self._emitted_ids.add(id(binding))
                            self._emitted_bindings.append(binding)
                            self._emit_operation_vertex(
                                binding,
                                is_terminal=self._binding_reaches_terminal(binding),
                            )
                        enqueue_expression(
                            str(binding.get("source_symbol") or binding.get("expression") or ""),
                            self._binding_walk_scope(binding),
                        )
                        # A branch's inputs are part of the mechanism:
                        # walking predicate symbols emits the internal-state
                        # writers that feasibility grounding later follows.
                        for position, predicate in enumerate(
                            binding.get("control_predicates") or []
                        ):
                            enqueue_expression(
                                str(predicate),
                                self._binding_predicate_scope(binding, position),
                            )
                elif "." in norm:
                    # A member read can be rooted in a source assignment to
                    # the aggregate itself (most importantly a call formal
                    # bound to its caller actual). Emit those root writers so
                    # wiring can perform the exact ``root.field`` rebinding.
                    root_raw = raw.replace("->", ".").split(".", 1)[0]
                    root_writers = self._scoped_writers(
                        exact_symbol(root_raw), root_raw, scope
                    )
                    for binding in root_writers:
                        if id(binding) not in self._emitted_ids:
                            self._emitted_ids.add(id(binding))
                            self._emitted_bindings.append(binding)
                            self._emit_operation_vertex(binding, is_terminal=False)
                        enqueue_expression(
                            str(
                                binding.get("source_symbol")
                                or binding.get("expression")
                                or ""
                            ),
                            self._binding_walk_scope(binding),
                        )
                elif (
                    "." not in norm
                    and norm != self.terminal
                    and norm not in self.logged_signals
                    and self._match_parameter(norm) is None
                ):
                    # Bare struct root with no direct writer — pull its
                    # field writes (``_mission_item`` → ``_mission_item.*``)
                    # under the SAME visibility rules as direct writers:
                    # unscoped, this globally pulled every module's
                    # same-named struct locals (mavlink's mission_item.*).
                    field_writers = self._filter_visible_writers(
                        self._field_writers_of(norm), raw, scope
                    )
                    for field_binding in field_writers:
                        field_target = str(
                            field_binding.get("target_symbol")
                            or field_binding.get("target")
                            or ""
                        )
                        if field_target:
                            frontier.append(
                                (
                                    "symbol",
                                    field_target,
                                    self._binding_walk_scope(field_binding),
                                )
                            )
            else:  # helper
                invocation = payload
                if invocation.source_site_id in materialized_helpers:
                    continue
                materialized_helpers.add(invocation.source_site_id)
                self._materialize_helper_subgraph(
                    invocation,
                    wire_edges=False,
                    scope_file=scope[0],
                    scope_function=scope[1],
                )
                helper_key = self._pick_helper_key(
                    invocation.name,
                    scope[0],
                    scope_function=scope[1],
                    receiver=invocation.receiver,
                    argument_count=len(invocation.args),
                )
                helper = self.helper_index.get(helper_key) if helper_key else None
                if not helper:
                    continue
                body_expressions = [str(v) for v in (helper.get("assignments") or {}).values()]
                return_expression = (
                    helper.get("lowered_return_expression")
                    or helper.get("return_expression")
                    or ""
                )
                if return_expression:
                    body_expressions.append(str(return_expression))
                for pointer_write in helper.get("pointer_output_writes") or []:
                    body_expressions.append(str(pointer_write.get("expression") or ""))
                body_scope: Scope = (
                    str(helper.get("file") or ""),
                    _callable_instance_scope(
                        self._helper_callable_id(helper_key, helper),
                        caller_callable=scope[1],
                        source_site_id=invocation.source_site_id,
                    ),
                    None,
                )
                for expression in body_expressions:
                    # The (b) fix: helper-body symbols drive the same frontier,
                    # so their writers get emitted instead of going opaque.
                    enqueue_expression(expression, body_scope)

        # Wire edges now that every producer vertex has been emitted.
        for binding in self._emitted_bindings:
            self._wire_binding_edges(binding)
        for helper_key in list(self._helper_subgraph_return_id.keys()):
            self._wire_helper_subgraph_edges(helper_key)

        return MechanismDAG(
            dag_id=stable_id("dag", (self.terminal, tuple(sorted(self.vertices.keys())))),
            terminal=self.terminal_raw,
            vertices=[self.vertices[key] for key in self.vertices],
            edges=[self.edges[key] for key in self.edges],
            unresolved_symbols=sorted(self.unresolved_symbols),
            unresolved_references=list(self._unresolved_references.values()),
        )

    # ------------------------------------------------------------
    # Native backward-walk helpers
    # ------------------------------------------------------------

    def _walk_symbols(self, expression: str) -> list[str]:
        """Normalized symbols referenced by ``expression``.

        C++ operators/casts are normalized first so ``&&`` / ``->`` / ``::``
        and cast-wrapped arguments still yield their symbols.
        """
        if not expression:
            return []
        normalized = _normalize_cpp_expression(str(expression))
        return dedupe_keep_order(
            exact_symbol(name) for name in source_expression_names(normalized)
        )

    def _wire_symbols(self, expression: str) -> list[str]:
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
        if not expression:
            return []
        return dedupe_keep_order(
            source_expression_names(_normalize_cpp_expression(str(expression)))
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
        expressions = [
            str(binding.get("source_symbol") or binding.get("expression") or ""),
            *(
                str(value)
                for value in (binding.get("control_predicates") or [])
            ),
        ]
        for expression in expressions:
            for symbol in self._wire_symbols(expression):
                root = symbol.replace("->", ".").split(".", 1)[0]
                if exact_symbol(root) in formals:
                    return True
        return False

    def _terminal_call_context_writers(
        self, writers: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Instantiate only terminal writers that need caller formals.

        A member writer that depends only on class state represents the same
        storage at every call and stays at its source definition. A writer
        whose value or control predicate reads a formal needs the caller's
        actual argument, so each source-proven invocation gets a private
        callable scope. No provider search is performed here: the terminal's
        writer and callable definition are already loaded facts.
        """
        contextualized: list[dict[str, Any]] = []
        for writer in writers:
            writer_file, writer_callable = self._binding_target_scope(writer)
            base_callable = _base_callable_scope(writer_callable)
            if (
                _CALL_SCOPE_MARKER in writer_callable
                or not base_callable
                or not self._binding_reads_formal(writer)
            ):
                contextualized.append(writer)
                continue

            matching_helpers = {
                helper_key
                for helper_key, helper in self.helper_index.items()
                if self._helper_callable_id(helper_key, helper) == base_callable
            }
            if not matching_helpers:
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
            for source_call in self._call_statements:
                args = [
                    str(value).strip()
                    for value in (source_call.get("args") or [])
                ]
                call_file = str(source_call.get("file") or "")
                caller_callable = str(
                    source_call.get("callable_id")
                    or source_call.get("function")
                    or ""
                )
                helper_key = self._pick_helper_key(
                    str(source_call.get("name") or ""),
                    call_file,
                    scope_function=caller_callable,
                    receiver=str(source_call.get("receiver") or ""),
                    receiver_type_hint=str(
                        source_call.get("receiver_type") or ""
                    ),
                    argument_count=len(args),
                    allow_provider=False,
                )
                if helper_key not in matching_helpers:
                    continue
                helper = self.helper_index.get(helper_key) or {}
                call_scope = self._register_call_instance(
                    source_call,
                    helper_key,
                    helper,
                    args,
                    caller_callable=caller_callable,
                )
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

    @staticmethod
    def _call_argument_storage(argument: str) -> str:
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
        scope: tuple[str, str, Optional[int]],
    ) -> bool:
        """Whether ``symbol`` is the parser's identity for a method call.

        Expression-name extraction represents ``obj.read().field`` as
        ``obj.read``. That is a call result, not storage owned by ``obj``;
        helper expansion handles its return value separately.
        """
        scope_file, scope_callable = scope[:2]
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
        scope: tuple[str, str, Optional[int]],
        *,
        terminal: bool,
    ) -> list[dict[str, Any]]:
        """Resolve calls whose source-proven effects can produce ``symbol``.

        Passing a value to a call does not make the call its producer. Loaded
        callees must prove either a write through a pointer/reference formal
        or a receiver-member write. Missing callees become typed discovery
        gaps instead of being fetched inline based only on textual overlap.
        """
        scope_file, scope_callable = scope[:2]
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
            name = str(source_call.get("name") or "")
            if not name or is_safe_math_function_name(name):
                continue
            receiver = exact_symbol(str(source_call.get("receiver") or ""))
            if receiver and self._match_parameter(
                f"{receiver}.{name.rsplit('::', 1)[-1]}"
            ) is not None:
                continue
            args = [str(arg).strip() for arg in (source_call.get("args") or [])]
            storages = [
                self._call_argument_storage(str(arg))
                for arg in args
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
            helper_key = self._pick_helper_key(
                name,
                call_file,
                scope_function=scope_callable or caller,
                receiver=str(source_call.get("receiver") or ""),
                receiver_type_hint=str(
                    source_call.get("receiver_type") or ""
                ),
                argument_count=len(args),
                allow_provider=False,
            )
            if helper_key is None:
                self._record_unresolved(
                    name,
                    kind="callable",
                    file=call_file,
                    line=int(source_call.get("line") or 0) or None,
                    scope_function=scope_callable or caller,
                    source_expression=str(source_call.get("evidence") or ""),
                    receiver=str(source_call.get("receiver") or ""),
                    argument_count=len(args),
                )
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

    @staticmethod
    def _identity_in_call_scope(
        raw: Any, source_callable: str, call_scope: str
    ) -> Any:
        if not raw:
            return raw
        try:
            identity = SourceSymbolIdentity.model_validate(raw)
        except (TypeError, ValueError):
            return raw
        if identity.kind == "local" and identity.callable_id == source_callable:
            identity = identity.model_copy(update={"callable_id": call_scope})
        return identity.model_dump()

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
                    "call_site_id": source_site_id,
                    "call_instance_scope": call_scope,
                    "callable_id": call_scope,
                }
            )
            clone["target_identity"] = self._identity_in_call_scope(
                original.get("target_identity"), helper_callable, call_scope
            )
            clone["reference_identities"] = {
                symbol: self._identity_in_call_scope(
                    identity, helper_callable, call_scope
                )
                for symbol, identity in (
                    original.get("reference_identities") or {}
                ).items()
            }
            self._register_synthetic_binding(clone)
            instance_bindings.append(clone)

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
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    ),
                    "struct_variables": {},
                    "scope_file": helper_file,
                    "scope_function": call_scope,
                    "scope_line": int(helper.get("line") or 0),
                    "expression_scope_file": expression_file,
                    "expression_scope_function": expression_callable,
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

    def _helper_reads_formal(
        self, helper_callable: str, helper: dict[str, Any], formal: str
    ) -> bool:
        """Whether source facts prove a value read through ``formal``."""
        for binding in self._source_bindings:
            if self._binding_target_scope(binding)[1] != helper_callable:
                continue
            texts = [
                str(
                    binding.get("source_symbol")
                    or binding.get("expression")
                    or ""
                ),
                *(str(value) for value in binding.get("control_predicates") or []),
            ]
            for text in texts:
                for symbol in self._wire_symbols(text):
                    root = symbol.replace("->", ".").split(".", 1)[0]
                    if exact_symbol(root) == exact_symbol(formal):
                        return True
        helper_texts = [
            str(helper.get("return_expression") or ""),
            str(helper.get("lowered_return_expression") or ""),
            *(str(value) for value in (helper.get("assignments") or {}).values()),
        ]
        return any(
            any(
                exact_symbol(symbol.replace("->", ".").split(".", 1)[0])
                == exact_symbol(formal)
                for symbol in self._wire_symbols(text)
            )
            for text in helper_texts
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
            target_identity = self._source_structure.symbol_identity(
                caller_target,
                file=call_file,
                callable_id=_base_callable_scope(caller_callable),
                function_name=str(call.get("function") or ""),
            )
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
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    ),
                    "struct_variables": {},
                    "scope_file": call_file,
                    "scope_function": caller_callable,
                    "scope_line": call_line,
                    "expression_scope_file": helper_file,
                    "expression_scope_function": call_scope,
                    "expression_scope_unordered": True,
                    "control_scope_file": call_file,
                    "control_scope_function": caller_callable,
                    "function": str(call.get("function") or ""),
                    "callable_id": caller_callable,
                    "target_identity": target_identity.model_dump(),
                    "synthetic_member_output_binding": True,
                    "call_site_id": self._call_site_id(call),
                    "call_instance_scope": call_scope,
                }
            )

    def _register_synthetic_binding(self, binding: dict[str, Any]) -> None:
        target = exact_symbol(
            str(binding.get("target_symbol") or binding.get("target") or "")
        )
        if not target:
            return
        self._all_bindings.append(binding)
        self._index_binding(
            self._by_target, self._target_shapes, target, binding
        )

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
        helper_line = int(helper.get("line") or 0)
        substituted = derive_pointer_output_bindings(
            pointer_writes, pointer_params, args
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
            helper_writers = [
                binding
                for binding in self._targets_matching(exact_symbol(formal_target))
                if self._binding_target_scope(binding)
                == (helper_file, call_scope)
            ]
            if not helper_writers:
                # Provider-fetched helper records can arrive without their
                # file's ordinary source bindings. Preserve the effect while
                # marking its internal reachability unresolved rather than
                # dropping the output or claiming an unconditional write.
                self._register_synthetic_binding(
                    {
                        "target_symbol": formal_target,
                        "source_symbol": entry["expression"],
                        "assignment_path": [
                            {
                                "file": helper_file,
                                "line": helper_line,
                                "expression": entry["expression"],
                            }
                        ],
                        "logged_signal": "",
                        "control_predicates": [],
                        "reachability_exact": False,
                        "struct_variables": dict(
                            helper.get("struct_variables") or {}
                        ),
                        "function": str(helper.get("name") or ""),
                        "scope_file": helper_file,
                        "scope_function": call_scope,
                        "expression_scope_file": helper_file,
                        "expression_scope_function": call_scope,
                        "callable_id": call_scope,
                        "call_site_id": self._call_site_id(call),
                        "synthetic_pointer_helper_write": True,
                    }
                )
            predicates = list(call.get("control_predicates") or [])
            predicate_lines = list(call.get("control_predicate_lines") or [])
            predicate_sites = list(
                call.get("control_predicate_site_ids") or []
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
                    "reachability_exact": bool(
                        call.get("reachability_exact", True)
                    ),
                    "struct_variables": {},
                    "scope_file": call_file,
                    "scope_function": caller_callable,
                    "scope_line": call_line,
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

    def _helper_instance_key(
        self,
        helper_key: tuple[str, str],
        invocation: _ExpressionCall,
        caller_callable: str,
    ) -> tuple[str, str, str]:
        helper = self.helper_index[helper_key]
        call_scope = _callable_instance_scope(
            self._helper_callable_id(helper_key, helper),
            caller_callable=caller_callable,
            source_site_id=invocation.source_site_id,
        )
        return helper_key[0], helper_key[1], call_scope

    @staticmethod
    def _file_family(path: str) -> tuple[str, str]:
        """(directory, stem) — the class file family (``rtl.cpp``/``rtl.h``)."""
        text = str(path or "")
        directory, _, base = text.rpartition("/")
        return (directory, base.split(".", 1)[0])

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
    ) -> SourceSymbolIdentity:
        canonical = exact_symbol(symbol_raw)
        source_callable = _base_callable_scope(scope_function)
        if file and scope_function:
            candidates = self._reference_identities_by_scope.get(
                (str(file), source_callable, canonical), []
            )
        elif file:
            candidates = self._reference_identities_by_file.get(
                (str(file), canonical), []
            )
        elif scope_function:
            candidates = self._reference_identities_by_callable.get(
                (source_callable, canonical), []
            )
        else:
            candidates = self._reference_identities_by_symbol.get(canonical, [])
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
        argument_count: Optional[int] = None,
    ) -> None:
        identity = (
            self._reference_identity(symbol, file, scope_function)
            if kind == "symbol"
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
            argument_count=argument_count,
            source_expression=source_expression,
            identity=identity,
        )
        self._unresolved_references.setdefault(reference.visit_key(), reference)
        self.unresolved_symbols.add(symbol)

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
    ) -> tuple[str, str, Optional[int]]:
        files = list(binding.get("control_predicate_files") or [])
        callables = list(binding.get("control_predicate_callables") or [])
        lines = list(binding.get("control_predicate_lines") or [])
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
        )

    def _binding_walk_scope(
        self, binding: dict[str, Any]
    ) -> tuple[str, str, Optional[int]]:
        file, callable_id = self._binding_site_scope(binding)
        if binding.get("expression_scope_unordered"):
            return file, callable_id, None
        raw_expression_line = binding.get("expression_scope_line")
        if isinstance(raw_expression_line, (int, float)):
            return file, callable_id, int(raw_expression_line)
        path = binding.get("assignment_path") or []
        first = path[0] if path else {}
        raw_line = (first or {}).get("line")
        line = int(raw_line) if isinstance(raw_line, (int, float)) else None
        return file, callable_id, line

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

    def _scoped_writers(
        self,
        symbol_norm: str,
        symbol_raw: str,
        scope: tuple[str, str, Optional[int]],
    ) -> list[dict[str, Any]]:
        """Writers of a symbol under C++-faithful visibility.

        Logged-output writers bypass scoping — uORB topics are the one
        legitimate cross-module channel. Target writers are filtered by
        the PX4 naming convention on the symbol's root:

        * member (``_``-prefixed): same class file family; when the
          family has no writer, widen to all (inheritance and cross-file
          member flows stay reachable).
        * local (bare): same file, and same function when both sides
          know theirs. Locals never widen — a stranger's same-named
          local is a different variable, which is exactly the fusion
          this prevents.

        ``_by_output`` (the logged-signal publisher index) is deliberately
        NOT consulted here: a logged input is an evidence LEAF — its
        values come from the log, and its publisher is a different
        module's mechanism across the uORB boundary. Only the terminal
        enters source through its publishers (see :meth:`build`).
        Unioning publishers at every step walked the airspeed slice
        through Commander → vehicle_command → Mavlink and beyond.
        """
        target_writers = self._filter_visible_writers(
            self._targets_matching(symbol_norm), symbol_raw, scope
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
        scope: tuple[str, str] | tuple[str, str, Optional[int]],
    ) -> list[dict[str, Any]]:
        scope_file, scope_function = scope[:2]
        scope_line = scope[2] if len(scope) > 2 else None
        if _CALL_SCOPE_MARKER in scope_function:
            # Runtime call instances share source identity but not storage for
            # locals/formals. Resolve inside the exact invocation before the
            # structural source-identity checks below.
            target_writers = [
                binding
                for binding in target_writers
                if self._binding_target_scope(binding)
                == (scope_file, scope_function)
            ]
        if scope_file and target_writers:
            reference = self._reference_identity(
                symbol_raw, scope_file, scope_function
            )
            structurally_scoped: list[dict[str, Any]] = []
            for binding in target_writers:
                producer = self._binding_target_identity(binding)
                if producer is not None and self._source_structure.compatible(
                    reference, producer
                ):
                    structurally_scoped.append(binding)
            # A structurally classified local/member must never widen. When
            # declaration coverage is incomplete, same-callable fallback is
            # conservative and explicitly prevents cross-method fusion.
            if reference.kind in {"local", "member"}:
                if structurally_scoped:
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
                # Textual order between methods has no runtime reaching-
                # definition meaning; all owner-compatible member writers
                # remain alternatives for later control/feasibility pruning.
                return target_writers
            if scope_line is not None and target_writers:
                prior = [
                    binding
                    for binding in target_writers
                    if self._binding_target_scope(binding)[0] != scope_file
                    or not self._binding_target_line(binding)
                    or self._binding_target_line(binding) <= scope_line
                ]
                prior.sort(
                    key=lambda binding: self._binding_target_line(binding) or 0,
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
        """Bindings that write ``symbol_norm`` (as a logged output or a
        source target). Union of both indexes, de-duplicated by identity."""
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for binding in self._outputs_matching(symbol_norm) + self._targets_matching(
            symbol_norm
        ):
            if id(binding) not in seen:
                seen.add(id(binding))
                out.append(binding)
        return out

    def _field_writers_of(self, root_norm: str) -> list[dict[str, Any]]:
        """Every binding whose target begins with ``{root_norm}.`` — the
        field writes of a bare struct root."""
        needle = f"{root_norm}."
        out: list[dict[str, Any]] = []
        for target_key, bindings in self._by_target.items():
            if target_key.startswith(needle):
                out.extend(bindings)
        return out

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
            }
            metadata["site_scope"] = {
                "file": site_file,
                "callable": site_callable,
            }
            metadata["control_scope"] = {
                "file": control_file,
                "callable": control_callable,
            }
            if binding.get("target_identity"):
                metadata["target_identity"] = dict(binding["target_identity"])
            if binding.get("synthetic_call_binding"):
                metadata["synthetic_call_binding"] = True
            if binding.get("call_site_id"):
                metadata["call_site_id"] = str(binding["call_site_id"])
            if binding.get("call_instance_scope"):
                metadata["call_instance_scope"] = str(
                    binding["call_instance_scope"]
                )
            logged_signal = exact_symbol(str(binding.get("logged_signal") or ""))
            if logged_signal:
                observed_placement = self._observed_signal_placement(logged_signal)
                metadata["logged_signal"] = observed_placement or logged_signal
                metadata["logged_observation"] = (
                    "observed" if observed_placement is not None else "unobserved"
                )
            self.vertices[op_id] = DAGVertex(
                id=op_id,
                kind="operation",
                sub_kind=self._classify_operation(expression),
                file=file,
                line=line,
                snippet=self._snippet(file, line),
                variable=target_raw,
                expression=expression,
                metadata=metadata,
            )
            self._index_producer(target_norm, op_id)
        return op_id

    def _wire_binding_edges(self, binding: dict[str, Any]) -> None:
        op_id, _target_raw, file, line, target_norm, expression = self._binding_operation_id(binding)

        # Attach control-predicate branches at their OWN source sites —
        # the branch's identity is where the control statement lives,
        # not where the gated assignment does.
        predicate_sites = binding.get("control_predicate_lines") or []
        predicate_site_ids = binding.get("control_predicate_site_ids") or []
        predicate_files = binding.get("control_predicate_files") or []
        predicate_callables = binding.get("control_predicate_callables") or []
        control_file, control_callable = self._binding_control_scope(binding)
        for position, predicate in enumerate(binding.get("control_predicates") or []):
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
            )
            self._add_edge(branch_id, op_id, kind="control")

        # Wire each source-expression symbol as an incoming data edge,
        # resolved in the binding's own callable scope.
        expression_file, scope_function = self._binding_site_scope(binding)
        expression_line = self._binding_walk_scope(binding)[2]
        for symbol in self._wire_symbols(expression):
            normalized = exact_symbol(symbol)
            if not normalized or (
                normalized == target_norm
                and not binding.get("synthetic_pointer_output_binding")
                and not binding.get("synthetic_member_output_binding")
            ):
                continue
            producer_ids = self._resolve_symbol_producers(
                normalized,
                symbol,
                expression,
                expression_file or file,
                expression_line,
                scope_function=scope_function,
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

        # Helper-call inputs.
        for invocation in self._find_helper_calls(
            expression,
            expression_file or file,
            scope_function=scope_function,
            line=expression_line,
        ):
            helper_key = self._pick_helper_key(
                invocation.name,
                expression_file or file,
                scope_function=scope_function,
                receiver=invocation.receiver,
                argument_count=len(invocation.args),
            )
            if helper_key is None:
                continue
            instance_key = self._helper_instance_key(
                helper_key, invocation, scope_function
            )
            helper_return_id = self._helper_subgraph_return_id.get(instance_key)
            if helper_return_id:
                self._add_edge(
                    helper_return_id,
                    op_id,
                    kind="data",
                    role=f"call:{invocation.name}",
                    via=invocation.source_site_id,
                )
            # Wire the caller's actual arguments to the helper's formal
            # parameter vertices — one edge per positional match.
            self._wire_helper_call_arguments(
                invocation,
                instance_key,
                expression_file or file,
                expression_line,
                scope_function=scope_function,
            )
            # Emit pointer-output writes from the helper as ops with the
            # caller's actual arg substituted for the pointer formal. Ops
            # dedupe with any source_assignments-derived vertex that
            # already covers the same call site.
            self._emit_helper_pointer_output_writes(
                invocation,
                instance_key,
                file=expression_file or file,
                line=expression_line,
                scope_function=scope_function,
            )

    def _emit_helper_pointer_output_writes(
        self,
        invocation: _ExpressionCall,
        instance_key: tuple[str, str, str],
        *,
        file: Optional[str],
        line: Optional[int],
        scope_function: str = "",
    ) -> None:
        """Graph-native equivalent of the profiler's pointer-output routing.

        For each ``pointer_output_writes`` entry on the resolved helper, emit
        an operation vertex whose target is ``{caller_actual_arg}.{field}``
        with the helper's formal output field as its expression. The helper's
        ordinary assignment operations retain the internal RHS and control
        predicates, and feed this call-site effect through a data edge. Ops
        share the (target, expression, file, line) identity used by
        :meth:`_emit_operation_vertex` so a source_assignments-derived binding
        for the same call site does not double-emit.
        """
        helper_key = self._pick_helper_key(
            invocation.name,
            file,
            scope_function=scope_function,
            receiver=invocation.receiver,
            argument_count=len(invocation.args),
        )
        if helper_key is None:
            return
        helper = self.helper_index.get(helper_key)
        if not helper:
            return
        effective_args = callable_arguments_with_defaults(
            helper, invocation.args
        )
        if effective_args is None:
            return
        pointer_writes = helper.get("pointer_output_writes") or []
        if not pointer_writes:
            return
        pointer_params = _pointer_param_positions(helper)
        if not pointer_params:
            return
        substituted = derive_pointer_output_bindings(
            pointer_writes, pointer_params, effective_args
        )
        helper_file = str(helper.get("file") or "")
        helper_callable = instance_key[2]
        helper_line = int(helper.get("line") or 0) or None
        for entry in substituted:
            target = entry["target"]
            expression = f"{entry['param']}.{entry['field']}"
            target_norm = exact_symbol(target)
            op_id = self._make_id(
                "op",
                (
                    target_norm,
                    expression,
                    file or "",
                    line or 0,
                    instance_key[2],
                ),
            )
            if op_id not in self.vertices:
                self.vertices[op_id] = DAGVertex(
                    id=op_id,
                    kind="operation",
                    sub_kind=self._classify_operation(expression),
                    file=file,
                    line=line,
                    snippet=self._snippet(file, line),
                    variable=target,
                    expression=expression,
                    provenance=f"pointer_output:{helper_key[0]}@{helper_key[1]}",
                    metadata={
                        "call_site_id": invocation.source_site_id,
                        "call_instance_scope": instance_key[2],
                    },
                )
                self._index_producer(target_norm, op_id)
            producer_ids = self._resolve_symbol_producers(
                exact_symbol(expression),
                expression,
                expression,
                helper_file,
                helper_line,
                scope_function=helper_callable,
            )
            for producer_id in producer_ids:
                self._add_edge(
                    producer_id, op_id, kind="data", role=expression
                )

    def _wire_helper_call_arguments(
        self,
        invocation: _ExpressionCall,
        instance_key: tuple[str, str, str],
        file: Optional[str],
        line: Optional[int],
        *,
        scope_function: str = "",
    ) -> None:
        """Wire one call occurrence's actuals to its private formals."""
        helper_key = self._pick_helper_key(
            invocation.name,
            file,
            scope_function=scope_function,
            receiver=invocation.receiver,
            argument_count=len(invocation.args),
        )
        if helper_key is None:
            return
        helper = self.helper_index.get(helper_key)
        if not helper:
            return
        effective_args = callable_arguments_with_defaults(
            helper, invocation.args
        )
        if effective_args is None:
            return
        formals = self._helper_parameter_vertices.get(instance_key)
        if not formals:
            return
        explicit_count = len(invocation.args)
        helper_file = str(helper.get("file") or "")
        helper_line = int(helper.get("line") or 0) or None
        for index, ((formal_name, formal_vertex_id), arg_text) in enumerate(
            zip(formals, effective_args)
        ):
            uses_default = index >= explicit_count
            argument_file = helper_file if uses_default else file
            argument_line = helper_line if uses_default else line
            argument_scope = instance_key[2] if uses_default else scope_function
            for symbol in self._wire_symbols(arg_text):
                normalized = exact_symbol(symbol)
                if not normalized:
                    continue
                producer_ids = self._resolve_symbol_producers(
                    normalized,
                    symbol,
                    arg_text,
                    argument_file,
                    argument_line,
                    scope_function=argument_scope,
                )
                for producer_id in producer_ids:
                    self._add_edge(
                        producer_id,
                        formal_vertex_id,
                        kind="data",
                        role=f"arg:{formal_name}",
                        via=invocation.source_site_id,
                    )

    def _visible_reaching_producers(
        self,
        producers: list[str],
        symbol_raw: str,
        file: Optional[str],
        line: Optional[int],
        scope_function: str,
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

        reference = self._reference_identity(symbol_raw, file, scope_function)
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
            # Source order across methods does not describe runtime order.
            return visible

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
            elif item.line <= line:
                prior.append(producer_id)

        prior.sort(key=lambda producer_id: int(self.vertices[producer_id].line or 0), reverse=True)
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
        emit_opaque: bool = True,
    ) -> list[str]:
        """Resolve a source-expression symbol to all reaching producers.

        If a prior operation produces ``symbol_norm``, connect to it. Otherwise
        emit an evidence leaf. Classification order (each consults a
        profiler-emitted source-of-truth so the vertex carries the
        resolved runtime handle rather than the raw source form):

        1. Direct producer via ``_producers_by_symbol`` (in-DAG operation).
        2a. Graph-native helper-chain derivation — a ``symbol().field``
           chain resolves via the helper's ``return_type`` and the PX4 msg
           schema; emit ``logged_signal`` with the derived ``topic.field``
           and the source form on ``metadata['source_form']``.
        2b. Graph-native struct-variable derivation — ``var.field`` where
           ``var`` is struct-typed (local or class-member) resolves via the
           same ``foo_s`` topic convention; emit ``logged_signal``.
        3. Source enum resolution — the symbol is defined in source as a
           numeric constant; emit ``constant`` with ``metadata['value']``.
        4. C stdlib constant (``CXX_STDLIB_CONSTANTS``) — same shape.
        5. Direct logged-signal set membership (fallback for canonical
           references that don't need a binding).
        6. Parameter accessor resolved through profiler-derived aliases.
        7. Otherwise: typed unresolved source symbol.

        No flat ``symbol_bindings`` table is consulted anywhere — every
        source→logged mapping is derived from graph structure.
        """
        producers = self._visible_reaching_producers(
            self._producers_matching(symbol_norm),
            symbol_raw,
            file,
            line,
            scope_function,
        )
        if producers:
            return producers

        # 1b. Dotted reference whose ROOT is rebound by a unique simple
        # writer (a formal bound to its actual, a reference alias):
        # rewrite root -> actual and re-resolve, so ``formal.field``
        # reaches the actual's logged placement
        # (``triplet.current.field``) instead of degrading to the nested
        # message name.
        rebinding_key = (symbol_norm, str(file or ""), scope_function)
        if "." in symbol_raw and rebinding_key not in self._rebinding_stack:
            root_raw, _, tail = symbol_raw.replace("->", ".").partition(".")
            root_norm = exact_symbol(root_raw)
            rebinders = [
                b
                for b in self._filter_visible_writers(
                    self._targets_matching(root_norm),
                    root_raw,
                    (file or "", scope_function),
                )
                if not b.get("control_predicates")
                and re.fullmatch(
                    r"[A-Za-z_][\w.]*",
                    str(b.get("source_symbol") or "").strip(),
                )
                # Pass-through forwarding (formal bound to a same-named
                # actual) is an identity, not a rebinding.
                and exact_symbol(str(b.get("source_symbol") or "")) != root_norm
            ]
            targets = {
                (
                    str(binding.get("source_symbol")).strip(),
                    self._binding_walk_scope(binding),
                )
                for binding in rebinders
            }
            if len(targets) == 1:
                target, target_scope = next(iter(targets))
                rewritten = f"{target}.{tail}"
                if exact_symbol(rewritten) != symbol_norm:
                    self._rebinding_stack.add(rebinding_key)
                    try:
                        # A failed rewrite must not hijack resolution:
                        # when the rewritten form classifies to nothing,
                        # the ORIGINAL symbol continues its own sequence
                        # below instead of surfacing the rewrite's dead
                        # end as the answer.
                        rebound = self._resolve_symbol_producers(
                            exact_symbol(rewritten),
                            rewritten,
                            source_expression,
                            target_scope[0],
                            target_scope[2],
                            scope_function=target_scope[1],
                            emit_opaque=False,
                        )
                    finally:
                        self._rebinding_stack.discard(rebinding_key)
                    if rebound:
                        return rebound
                    if any(
                        binding.get("synthetic_call_binding")
                        for binding in rebinders
                    ):
                        # A formal-to-actual binding is syntax-proven, so an
                        # unresolved rewritten leaf belongs to the caller's
                        # storage identity, not to the callee formal name.
                        return self._resolve_symbol_producers(
                            exact_symbol(rewritten),
                            rewritten,
                            source_expression,
                            target_scope[0],
                            target_scope[2],
                            scope_function=target_scope[1],
                            emit_opaque=True,
                        )

        # 2a. Graph-native derivation: if ``source_expression`` contains a
        # ``symbol_raw().field`` chain, resolve it via the helper's
        # ``return_type`` and the PX4 msg schema.
        chain_resolved = self._resolve_symbol_via_chain(
            symbol_raw, source_expression, file=file, scope_function=scope_function
        )
        if chain_resolved is not None:
            signal, boundary = chain_resolved
            return [self._emit_evidence(
                "logged_signal",
                signal,
                file=None,
                line=None,
                metadata={
                    "source_form": symbol_raw,
                    "derivation": "source_boundary",
                    "boundary": "source_proven",
                    "boundary_provenance": boundary.get("provenance"),
                },
            )]

        # 2b. Graph-native struct-variable derivation: ``var.field`` where
        # ``var`` is struct-typed (local declaration or class member).
        # Same PX4 ``foo_s`` convention as helper return types; no flat
        # side-table.
        struct_resolved = self._resolve_symbol_via_struct_var(
            symbol_raw, source_expression, file=file, scope_function=scope_function
        )
        if struct_resolved is not None:
            signal, boundary = struct_resolved
            return [self._emit_evidence(
                "logged_signal",
                signal,
                file=None,
                line=None,
                metadata={
                    "source_form": symbol_raw,
                    "derivation": "source_boundary",
                    "boundary": "source_proven",
                    "boundary_provenance": boundary.get("provenance"),
                },
            )]

        # 3. Source enum / #define resolution.
        enum_value = self._source_constants.get(symbol_norm)
        if enum_value is not None:
            return [self._emit_evidence(
                "constant",
                symbol_raw,
                file=None,
                line=None,
                metadata={"value": enum_value, "source": "enum"},
            )]

        # 3b. Schema message enum — the reference names its own scope
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

        # 4. C stdlib constant.
        cxx_value = CXX_STDLIB_CONSTANTS.get(symbol_raw.upper())
        if cxx_value is not None:
            return [self._emit_evidence(
                "constant",
                symbol_raw.upper(),
                file=None,
                line=None,
                metadata={"value": cxx_value, "source": "cxx_stdlib"},
            )]

        # 5. Canonical logged signal at one exact observed placement.
        observed_placement = self._observed_signal_placement(symbol_norm)
        if observed_placement is not None:
            return [
                self._emit_evidence(
                    "logged_signal", observed_placement, file=None, line=None
                )
            ]

        # 6. Parameter accessor resolved from profiler-derived aliases or an
        # exact declaration/inventory name.
        parameter_alias = self._match_parameter(symbol_raw)
        if parameter_alias is not None:
            return [self._emit_evidence("parameter", parameter_alias, file=None, line=None)]

        # 7a. Bare PX4-parameter-shaped name resolved through the ULog
        # parameter inventory. Handles source RHSes like ``FW_AIRSPD_TRIM``.
        parameter_value = self._parameter_values.get(symbol_raw.upper())
        if parameter_value is not None and is_px4_parameter_name(symbol_raw.upper()):
            return [self._emit_evidence(
                "constant",
                symbol_raw,
                file=None,
                line=None,
                metadata={"value": parameter_value, "source": "parameter"},
            )]

        # 8. Unclassified. Callers probing an alternative spelling
        # (rebinding) suppress the fallback so the original symbol keeps
        # its own resolution sequence.
        if not emit_opaque:
            return []
        if self._symbol_is_receiver_call_result(
            symbol_norm,
            (str(file or ""), scope_function, line),
        ) or re.search(rf"{re.escape(symbol_raw)}\s*\(", source_expression):
            # The callable frontier owns expansion of a call result. Keeping
            # its parser placeholder as a second symbol gap causes searches
            # for names such as ``receiver.method`` and cannot find storage.
            return [
                self._emit_evidence(
                    "opaque_symbol", symbol_raw, file=file, line=line
                )
            ]
        self._record_unresolved(
            symbol_raw,
            file=file,
            line=line,
            scope_function=scope_function,
            source_expression=source_expression,
        )
        return [self._emit_evidence("opaque_symbol", symbol_raw, file=file, line=line)]

    def _lower_predicate(
        self, canonical: str
    ) -> tuple[str, dict[str, str]]:
        """Return ``(lowered_expression, variables)`` for a canonical predicate.

        Fully graph-native — no flat ``symbol_bindings`` table anywhere.
        Three passes:

        1. **Helper-chain resolution** — ``chain().field`` gets replaced with
           ``topic.field`` only when source facts prove that the receiver is a
           subscribed message boundary.
        2. **Struct-variable resolution** — ``var.field`` gets replaced only
           when a source-proven subscription copy/update binds ``var`` to a
           topic.
        3. **Enum-shaped constants** → resolved values from source
           (``assignment_resolutions`` covers enum entries and object-like
           ``#define``) or the C-stdlib table.

        The ``variables`` dict records each source-form → logged
        substitution so downstream evaluators share one lowering owner
        instead of re-running ``lower_control_predicates``.
        """
        if not canonical:
            return canonical, {}
        lowered = canonical
        variables: dict[str, str] = {}

        # Pass 1: helper-chain resolution via return_type + msg schema.
        lowered, chain_variables = self._substitute_helper_chains(lowered)
        variables.update(chain_variables)

        # Pass 2: struct-variable field access — same derivation as helper
        # chains but for ``var.field`` patterns where var is struct-typed.
        lowered, struct_variables = self._substitute_struct_var_fields(lowered)
        for key, value in struct_variables.items():
            variables.setdefault(key, value)

        # Pass 3: enum-shaped constants → resolved values.
        for token in sorted(
            set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", lowered)),
            key=len,
            reverse=True,
        ):
            value = self._source_constants.get(exact_symbol(token))
            if value is None:
                value = CXX_STDLIB_CONSTANTS.get(token)
            if isinstance(value, (int, float, bool)):
                lowered = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
                    _format_lowered_value(value),
                    lowered,
                )

        return lowered, variables

    def _substitute_struct_var_fields(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace ``var.field`` occurrences whose ``var`` is struct-typed.

        Same derivation path as :meth:`_substitute_helper_chains`. Uses a
        text-level ``\\bvar.field`` scan against the aggregated
        struct-variables map; substitutes when the topic resolves via
        the PX4 ``foo_s`` convention AND ``topic.field`` is in the
        trusted signal catalogue. Longest var names go first so a prefix
        doesn't overwrite a more specific match.
        """
        variables: dict[str, str] = {}
        if not self._boundary_bindings:
            return text, variables

        subscribed_variables = {
            str(binding.get("source_symbol") or "")
            for binding in self._boundary_bindings
            if binding.get("direction") == "subscribe"
        }
        for var in sorted(subscribed_variables, key=len, reverse=True):
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_])(?:_?)({re.escape(var)})\s*(?:\.|->)\s*"
                rf"(?P<field>[A-Za-z_][A-Za-z0-9_]*)"
            )
            def make_replacement(matched_var: str):
                def replace(match: re.Match) -> str:
                    field = match.group("field")
                    resolved = self._resolve_struct_var_field(matched_var, field)
                    if resolved is None:
                        return match.group(0)
                    signal, _provenance = resolved
                    variables[f"{matched_var}.{field}"] = signal
                    return signal
                return replace
            text = pattern.sub(make_replacement(var), text)
        return text, variables

    def _substitute_helper_chains(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace ``chain().field`` occurrences with ``topic.field``.

        For each match, walks the helper subgraph to recover the return
        type and derives the topic through the PX4 struct-``_s``
        convention. Cross-checked against the trusted signal catalogue
        (``schema_signals`` ∪ ``logged_signals``) before substitution so
        an unknown topic silently falls through instead of producing a
        phantom binding.
        """
        variables: dict[str, str] = {}

        def replace(match: re.Match) -> str:
            chain = match.group("chain").strip()
            field = match.group("field").strip()
            resolved = self._resolve_helper_chain(chain, field)
            if resolved is None:
                return match.group(0)
            signal, _provenance = resolved
            variables[match.group(0)] = signal
            return signal

        substituted = _HELPER_CHAIN_RE.sub(replace, text)
        return substituted, variables

    def _resolve_symbol_via_struct_var(
        self,
        symbol_raw: str,
        source_expression: str,
        *,
        file: Optional[str] = None,
        scope_function: str = "",
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Try to graph-derive a ``topic.field`` binding for ``symbol_raw``
        via the struct-variable map.

        ``source_expression_names`` returns the dotted ``var.field`` intact
        when there's no intervening ``()``. If the whole symbol matches a
        ``struct_var.field`` shape, this routes directly to
        :meth:`_resolve_struct_var_field`. Otherwise falls back to
        scanning the source expression for the pattern.
        """
        if not symbol_raw:
            return None
        # Fast path: the symbol itself is already the ``var.field`` form.
        parts = symbol_raw.rsplit(".", 1)
        if len(parts) == 2:
            var_candidate, field = parts
            resolved = self._resolve_struct_var_field(
                var_candidate, field, file=file, scope_function=scope_function
            )
            if resolved is not None:
                return resolved
        # Nested placement: ``var.member.field`` — the struct variable is
        # the ROOT and the field path is everything after it.
        root, _, nested_tail = symbol_raw.partition(".")
        if nested_tail and "." in nested_tail:
            resolved = self._resolve_struct_var_field(
                root, nested_tail, file=file, scope_function=scope_function
            )
            if resolved is not None:
                return resolved
        if not source_expression:
            return None
        # Slow path: find ``symbol_raw.field`` in the surrounding expression.
        pattern = re.compile(
            rf"{re.escape(symbol_raw)}\s*(?:\.|->)\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)"
        )
        match = pattern.search(source_expression)
        if not match:
            return None
        return self._resolve_struct_var_field(
            symbol_raw,
            match.group("field"),
            file=file,
            scope_function=scope_function,
        )

    def _resolve_symbol_via_chain(
        self,
        symbol_raw: str,
        source_expression: str,
        *,
        file: Optional[str] = None,
        scope_function: str = "",
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Try to graph-derive a ``topic.field`` binding for ``symbol_raw``.

        ``source_expression_names`` truncates at ``()``, so a symbol like
        ``_navigator.get_vstatus`` reaches classification without its
        trailing ``.vehicle_type``. This helper walks the surrounding
        ``source_expression`` for the ``symbol_raw().field`` chain, then
        delegates to :meth:`_resolve_helper_chain` for the actual
        return-type + msg-schema lookup. Returns ``None`` when no chain
        pattern matches, when the helper isn't in the index, or when the
        resulting topic.field isn't in the trusted signal catalogue.
        """
        if not symbol_raw or not source_expression:
            return None
        # Find ``symbol_raw().field`` in the surrounding expression.
        pattern = re.compile(
            rf"{re.escape(symbol_raw)}\s*\(\s*\)\s*(?:\.|->)\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)"
        )
        match = pattern.search(source_expression)
        if not match:
            return None
        return self._resolve_helper_chain(
            symbol_raw,
            match.group("field"),
            file=file,
            scope_function=scope_function,
        )

    def _resolve_struct_var_field(
        self,
        var: str,
        field: str,
        *,
        file: Optional[str] = None,
        scope_function: str = "",
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Return ``topic.field`` when ``var.field`` is graph-derivable.

        Resolves ``var`` through a source-proven subscription boundary and
        validates the resulting ``topic.field`` against the trusted signal
        catalogue. A struct declaration or naming convention alone is not a
        message-boundary proof.
        """
        boundary = self._boundary_topic_for(
            var, file=file, scope_function=scope_function, direction="subscribe"
        )
        if boundary is None:
            return None
        topic, provenance = boundary
        signal = f"{topic}.{field}"
        if (
            self._observed_signal_placement(signal) is not None
            or self._declared_signal_known(signal)
        ):
            return signal, provenance
        return None

    def _resolve_helper_chain(
        self,
        chain: str,
        field: str,
        *,
        file: Optional[str] = None,
        scope_function: str = "",
    ) -> Optional[tuple[str, dict[str, Any]]]:
        """Return ``topic.field`` when ``chain().field`` is graph-derivable.

        Splits the chain on ``.``/``->``, takes the final segment as the
        helper's short name, looks up the helper in ``helper_index``, and
        derives the topic from :func:`_derive_topic_from_return_type`.
        Returns ``None`` when the helper isn't in the index, the return
        type doesn't follow the ``foo_s`` convention, or the resulting
        ``topic.field`` isn't in the trusted signal catalogue.
        """
        segments = [seg for seg in re.split(r"[.>]+", chain) if seg]
        if not segments:
            return None
        receiver = segments[0]
        boundary = self._boundary_topic_for(
            receiver, file=file, scope_function=scope_function, direction="subscribe"
        )
        if boundary is not None:
            topic, provenance = boundary
            signal = f"{topic}.{field}"
            if (
                self._observed_signal_placement(signal) is not None
                or self._declared_signal_known(signal)
            ):
                return signal, provenance
        helper_name = segments[-1]
        helper_key = self._pick_helper_key(helper_name)
        if helper_key is None:
            return None
        helper = self.helper_index.get(helper_key)
        if not helper:
            return None
        return_expression = str(
            helper.get("return_expression")
            or helper.get("lowered_return_expression")
            or ""
        ).strip()
        return_names = dedupe_keep_order(
            source_expression_names(_normalize_cpp_expression(return_expression))
        )
        if len(return_names) == 1:
            helper_boundary = self._boundary_topic_for(
                return_names[0],
                file=str(helper.get("file") or "") or None,
                scope_function=self._helper_callable_id(helper_key, helper),
                direction="subscribe",
            )
            if helper_boundary is not None:
                topic, provenance = helper_boundary
                signal = f"{topic}.{field}"
                if (
                    self._observed_signal_placement(signal) is not None
                    or self._declared_signal_known(signal)
                ):
                    return signal, provenance
        # A return type establishes schema compatibility, not that the value
        # came from uORB. Helpers without source-proven boundary provenance
        # remain unresolved.
        return None

    def _branch_metadata_from_parameter_predicate(self, canonical: str) -> dict[str, Any]:
        """Return branch metadata sourced from a ``ParameterPredicateRef``.

        The profiler already extracts operator + compared_value at profile
        time; consuming that record here means downstream evaluators don't
        re-parse the predicate to recover the same structure. Absent match
        returns an empty dict — the branch stays a plain source-predicate
        node whose feasibility path handles evaluation.
        """
        record = self._parameter_predicate_by_predicate.get(canonical)
        if not record:
            return {}
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
        key = (sub_kind, signal)
        existing = self._evidence_by_signal.get(key)
        if existing is not None:
            return existing
        vertex_id = self._make_id("ev", key)
        vertex_metadata = dict(metadata) if metadata else {}
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
        metadata = self._branch_metadata_from_parameter_predicate(canonical)
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
        for symbol in dedupe_keep_order(source_expression_names(symbol_source)):
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
            )
            for producer_id in producer_ids:
                self._add_edge(producer_id, vertex_id, kind="data", role=symbol)

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
    ) -> list[_ExpressionCall]:
        """Return each source-resolved call occurrence in ``expression``.

        Resolution is frontier-driven: this is the only path that may ask
        the provider for a missing expression callee. Every occurrence keeps
        a distinct site identity, including repeated calls with the same name
        on one source line.
        """
        matches: list[_ExpressionCall] = []
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression):
            candidate = match.group(1)
            if candidate in _NON_CALL_SYNTAX or is_safe_math_function_name(candidate):
                continue
            prefix = expression[: match.start()]
            receiver_match = re.search(
                r"([A-Za-z_][A-Za-z0-9_.]*(?:\[[^\]]+\])?)\s*(?:\.|->)\s*$",
                prefix,
            )
            receiver = receiver_match.group(1) if receiver_match else ""
            if receiver and self._match_parameter(
                f"{receiver}.{candidate}"
            ) is not None:
                continue
            close = self._matching_parenthesis(expression, match.end() - 1)
            if close is None:
                continue
            raw_args = expression[match.end() : close]
            args = tuple(
                arg.strip()
                for arg in split_top_level_args(raw_args)
                if arg.strip()
            )
            lookup_key = (
                str(scope_file or ""),
                _base_callable_scope(scope_function),
                int(line or 0),
                candidate,
                args,
            )
            source_record = next(
                (
                    call
                    for call in self._calls_by_site.get(lookup_key, ())
                    if str(call.get("receiver") or "") == receiver
                ),
                None,
            )
            source_site_id = (
                self._call_site_id(source_record)
                if source_record is not None
                else stable_id(
                    "callsite",
                    (
                        str(scope_file or ""),
                        _base_callable_scope(scope_function),
                        int(line or 0),
                        match.start(),
                        candidate,
                        receiver,
                        args,
                    ),
                )
            )
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
            )
            if reference_receiver_is_source_boundary(
                call_reference,
                self._boundary_bindings,
                self._source_structure,
            ):
                self._record_unresolved(
                    f"{receiver}.{candidate}" if receiver else candidate,
                    kind="callable",
                    file=scope_file,
                    line=line,
                    scope_function=scope_function,
                    source_expression=expression,
                    receiver=receiver,
                    argument_count=len(args),
                )
                continue
            helper_key = self._pick_helper_key(
                candidate,
                scope_file,
                scope_function=scope_function,
                receiver=receiver,
                receiver_type_hint=str(
                    (source_record or {}).get("receiver_type") or ""
                ),
                argument_count=len(args),
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
                    argument_count=len(args),
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
            self._register_call_instance(
                runtime_call,
                helper_key,
                helper,
                args,
                caller_callable=scope_function,
            )
            matches.append(
                _ExpressionCall(
                    name=candidate,
                    receiver=receiver,
                    args=args,
                    offset=match.start(),
                    source_site_id=runtime_site_id,
                )
            )
        return matches

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

    def _materialize_helper_subgraph(
        self,
        invocation: _ExpressionCall,
        *,
        wire_edges: bool = True,
        scope_file: Optional[str] = None,
        scope_function: str = "",
    ) -> Optional[str]:
        """Emit the helper's body as nested vertices, return its output vertex id.

        Memoized by source callable plus call site. Preserves every
        intermediate assignment as its own vertex — no collapsing.

        When ``wire_edges`` is False, only vertices are emitted (pass 1).
        Edges are wired by :meth:`_wire_helper_subgraph_edges` in pass 2.
        """
        helper_key = self._pick_helper_key(
            invocation.name,
            scope_file,
            scope_function=scope_function,
            receiver=invocation.receiver,
            argument_count=len(invocation.args),
        )
        if helper_key is None:
            return None
        instance_key = self._helper_instance_key(
            helper_key, invocation, scope_function
        )
        self._helper_instance_site[instance_key] = invocation.source_site_id
        cached = self._helper_subgraph_return_id.get(instance_key)
        if cached is not None:
            return cached or None

        helper = self.helper_index[helper_key]
        file = helper.get("file")
        line = helper.get("line")
        scope_function = instance_key[2]

        # Emit a helper_parameter vertex for each formal so callers can
        # wire their actual arguments into shared entry points and helper
        # body operations can resolve references to formals against them.
        formals: list[tuple[str, str]] = []
        for formal in helper.get("parameters") or []:
            formal_str = str(formal)
            if not formal_str:
                continue
            param_id = self._make_id(
                "ev",
                (
                    "helper_param",
                    helper_key[0],
                    helper_key[1],
                    invocation.source_site_id,
                    formal_str,
                ),
            )
            if param_id not in self.vertices:
                self.vertices[param_id] = DAGVertex(
                    id=param_id,
                    kind="evidence",
                    sub_kind="helper_parameter",
                    file=file,
                    line=line,
                    signal_name=formal_str,
                    metadata={
                        "helper": f"{helper_key[0]}@{helper_key[1]}",
                        "call_site_id": invocation.source_site_id,
                        "call_instance_scope": scope_function,
                    },
                )
            formals.append((formal_str, param_id))
        self._helper_parameter_vertices[instance_key] = formals

        for var, expression in (helper.get("assignments") or {}).items():
            var_norm = exact_symbol(var)
            op_id = self._make_id(
                "op",
                (
                    helper_key[0],
                    helper_key[1],
                    invocation.source_site_id,
                    var_norm,
                    expression,
                ),
            )
            if op_id not in self.vertices:
                self.vertices[op_id] = DAGVertex(
                    id=op_id,
                    kind="operation",
                    sub_kind="assign",
                    file=file,
                    line=line,
                    snippet=self._snippet(file, line),
                    variable=var,
                    expression=expression,
                    provenance=f"helper_body:{helper_key[0]}@{helper_key[1]}",
                    metadata={
                        "function": scope_function,
                        "target_scope": {"file": str(file or ""), "callable": scope_function},
                        "site_scope": {"file": str(file or ""), "callable": scope_function},
                        "reachability": {"all_of": [], "exact": False},
                        "call_site_id": invocation.source_site_id,
                        "call_instance_scope": scope_function,
                    },
                )
                self._index_producer(var_norm, op_id)

        return_expression = (
            helper.get("lowered_return_expression")
            or helper.get("return_expression")
        )
        if not return_expression:
            self._helper_subgraph_return_id[instance_key] = ""
            return None

        terminal_id = self._make_id(
            "op",
            (
                helper_key[0],
                helper_key[1],
                invocation.source_site_id,
                "__return__",
                return_expression,
            ),
        )
        if terminal_id not in self.vertices:
            self.vertices[terminal_id] = DAGVertex(
                id=terminal_id,
                kind="operation",
                sub_kind="helper_call",
                file=file,
                line=line,
                snippet=self._snippet(file, line),
                variable=f"{scope_function}::__return__",
                expression=return_expression,
                lowered_expression=return_expression,
                provenance=f"helper_return:{helper_key[0]}@{helper_key[1]}",
                metadata={
                    "function": scope_function,
                    "target_scope": {"file": str(file or ""), "callable": scope_function},
                    "site_scope": {"file": str(file or ""), "callable": scope_function},
                    "reachability": {"all_of": [], "exact": False},
                    "call_site_id": invocation.source_site_id,
                    "call_instance_scope": scope_function,
                },
            )
        self._helper_subgraph_return_id[instance_key] = terminal_id

        if wire_edges:
            self._wire_helper_subgraph_edges(instance_key)
        return terminal_id

    def _wire_helper_subgraph_edges(
        self, instance_key: tuple[str, str, str]
    ) -> None:
        helper_key = instance_key[:2]
        helper = self.helper_index.get(helper_key)
        if not helper:
            return
        file = helper.get("file")
        line = helper.get("line")
        scope_function = instance_key[2]
        call_site_id = self._helper_instance_site.get(instance_key, "")

        # Helper-scoped local resolver: formal parameter names bind to
        # their formal-parameter vertex before falling back to the global
        # producer index. This preserves helper-local scoping when two
        # helpers happen to share a formal name (``float x``).
        local_scope = {
            exact_symbol(formal): vertex_id
            for formal, vertex_id in self._helper_parameter_vertices.get(instance_key, ())
        }

        for var, expression in (helper.get("assignments") or {}).items():
            var_norm = exact_symbol(var)
            op_id = self._make_id(
                "op",
                (
                    helper_key[0],
                    helper_key[1],
                    call_site_id,
                    var_norm,
                    expression,
                ),
            )
            for symbol in self._wire_symbols(str(expression)):
                normalized = exact_symbol(symbol)
                if not normalized or normalized == var_norm:
                    continue
                producer_ids = (
                    [local_scope[normalized]]
                    if normalized in local_scope
                    else self._resolve_symbol_producers(
                        normalized,
                        symbol,
                        str(expression),
                        file,
                        line,
                        scope_function=scope_function,
                    )
                )
                for producer_id in producer_ids:
                    self._add_edge(producer_id, op_id, kind="data", role=symbol)

        # Helper return operation, and its data inputs.
        return_expression = (
            helper.get("lowered_return_expression")
            or helper.get("return_expression")
        )
        terminal_id: Optional[str] = None
        if return_expression:
            terminal_id = self._make_id(
                "op",
                (
                    helper_key[0],
                    helper_key[1],
                    call_site_id,
                    "__return__",
                    return_expression,
                ),
            )
            for symbol in self._wire_symbols(str(return_expression)):
                normalized = exact_symbol(symbol)
                if not normalized:
                    continue
                producer_ids = (
                    [local_scope[normalized]]
                    if normalized in local_scope
                    else self._resolve_symbol_producers(
                        normalized,
                        symbol,
                        str(return_expression),
                        file,
                        line,
                        scope_function=scope_function,
                    )
                )
                for producer_id in producer_ids:
                    self._add_edge(producer_id, terminal_id, kind="data", role=symbol)

        # Conditional returns: each branch GATES the helper return. Wire a
        # control edge from the branch vertex to the return op (so it isn't
        # an orphan) and the branch's return-value symbols as data inputs of
        # the return (so the alternate value's producers are in the graph
        # even when the lowered expression didn't capture them).
        for branch in helper.get("branches") or []:
            condition = str(branch.get("condition") or "").strip()
            if not condition:
                continue
            branch_id = self._emit_branch(
                condition,
                file=str(branch.get("file") or file or "") or None,
                line=int(branch.get("line") or line or 0) or None,
                scope_function=scope_function,
                source_site_id=str(branch.get("source_site_id") or ""),
            )
            if terminal_id is not None:
                self._add_edge(branch_id, terminal_id, kind="selection")
            value_expression = str(branch.get("expression") or "")
            if terminal_id is None or not value_expression:
                continue
            for symbol in self._wire_symbols(value_expression):
                normalized = exact_symbol(symbol)
                if not normalized:
                    continue
                producer_ids = (
                    [local_scope[normalized]]
                    if normalized in local_scope
                    else self._resolve_symbol_producers(
                        normalized,
                        symbol,
                        value_expression,
                        file,
                        line,
                        scope_function=scope_function,
                    )
                )
                for producer_id in producer_ids:
                    self._add_edge(producer_id, terminal_id, kind="data", role=f"branch:{symbol}")

    def _pick_helper_key(
        self,
        helper_name: str,
        scope_file: Optional[str] = None,
        *,
        scope_function: str = "",
        receiver: str = "",
        receiver_type_hint: str = "",
        argument_count: Optional[int] = None,
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
        if not matches:
            if not allow_provider or self.helper_body_provider is None:
                return None
            reference = UnresolvedSourceReference(
                symbol=helper_name,
                kind="callable",
                file=str(scope_file or ""),
                callable_id=_base_callable_scope(scope_function),
                class_owner=self._source_structure.callable_owner(
                    _base_callable_scope(scope_function)
                ),
                receiver=receiver,
                argument_count=argument_count,
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
            for helper in _coerce_helpers(fetched):
                key_iter = _index_helpers([helper])
                for key, value in key_iter.items():
                    if key not in self.helper_index:
                        self.helper_index[key] = value
                        self._helper_keys_by_name[key[0]].append(key)
            matches = matching_keys()
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
        key = (source_id, target_id, kind, role or "")
        if key in self.edges:
            return
        edge_id = self._make_id("edge", (source_id, target_id, kind, role or ""))
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
    # Argument parsing
    # ------------------------------------------------------------


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

    def _match_parameter(self, symbol: str) -> Optional[str]:
        """Return canonical parameter name if ``symbol`` looks like one.

        Accepts both the pre-normalized form (`_param_rtl_return_alt`) and
        normalized (`param_rtl_return_alt`), and strips a trailing
        ``.get()`` / ``.get`` accessor before matching. Resolution order:

        1. ``parameter_aliases`` (the ``DEFINE_PARAMETERS`` member→name map)
           — the only path that resolves members whose name differs from the
           param name (``_param_rtl_cone_half_angle_deg`` → ``RTL_CONE_ANG``).
        2. Direct membership / the ``_param_<snake>→UPPER`` heuristic, which
           only works when member name equals param name.
        """
        raw = symbol.replace(".get()", "").replace(".get", "")

        # 1. Member → canonical name via the DEFINE_PARAMETERS alias map.
        aliased = self._parameter_aliases.get(raw) or self._parameter_aliases.get(
            normalize_symbol(raw)
        )
        if aliased is not None:
            return aliased

        if not self.parameter_names:
            return None
        upper = raw.upper()
        if upper in self.parameter_names:
            return upper
        # 2. PX4-style: `_param_rtl_return_alt` → `RTL_RETURN_ALT` (member==name).
        match = re.fullmatch(r"_?param_([a-zA-Z0-9_]+)", raw)
        if match:
            candidate = match.group(1).upper()
            if candidate in self.parameter_names:
                return candidate
        return None

    def _binding_reaches_terminal(self, binding: dict[str, Any]) -> bool:
        logged = exact_symbol(str(binding.get("logged_signal") or ""))
        target = exact_symbol(str(binding.get("target_symbol") or ""))
        return logged == self.terminal or target == self.terminal

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


# Matches ``chain().field`` and ``chain()->field`` where ``chain`` is a
# dotted / arrow-separated identifier path (``_navigator.get_vstatus``,
# ``_navigator->get_vstatus``). Group ``chain`` captures the accessor path
# without the trailing ``()``; ``field`` captures the single trailing field.
_HELPER_CHAIN_RE = re.compile(
    r"(?P<chain>[A-Za-z_][A-Za-z0-9_]*(?:\s*(?:\.|->)\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*\(\s*\)\s*(?:\.|->)\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)"
)


def _derive_topic_from_return_type(return_type: Optional[str]) -> Optional[str]:
    """Derive a PX4 topic name from a C++ return-type expression.

    PX4 convention: struct types corresponding to uORB topics end in
    ``_s``, with the topic name being the struct name without the
    suffix. Pointer/reference marks and nested namespaces are stripped.
    ``vehicle_status_s *`` → ``vehicle_status``; ``float`` → ``None``.
    Returns ``None`` when the type doesn't fit the convention so
    callers know to skip graph-native binding derivation for this helper.
    """
    if not return_type:
        return None
    text = return_type.replace("*", "").replace("&", "").strip()
    text = text.rstrip(";").strip()
    if not text:
        return None
    text = text.rsplit("::", 1)[-1].strip()
    if text.endswith("_s") and len(text) > 2:
        return text[:-2]
    return None


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
    """Lower ``expression`` toward logged form using the graph itself.

    Each symbol the vertex reads (its incoming data edges' roles) is
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
        # Edge roles carry the ``.``-collapsed form; align the text so
        # ``struct_s::NAME`` and ``obj->field`` references substitute.
        result = text.replace("->", ".").replace("::", ".")
        edges_by_role: dict[str, list[DAGEdge]] = defaultdict(list)
        for edge in edges_by_target.get(target_id, []):
            edges_by_role[str(edge.role)].append(edge)
        for role, role_edges in edges_by_role.items():
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


def evaluate_feasibility(
    dag: MechanismDAG,
    *,
    parameter_values: Optional[dict[str, Any]] = None,
    enum_values: Optional[dict[str, Any]] = None,
    signal_samples: Optional[dict[str, list[tuple[float, Any]]]] = None,
    signal_policies: Optional[dict[str, Any]] = None,
    prepared_signal_series: Optional[dict[str, "PreparedSignalSeries"]] = None,
    prune_dead: bool = True,
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
    """
    params = {k.upper(): v for k, v in (parameter_values or {}).items()}
    enums = dict(enum_values or {})
    samples = signal_samples or {}
    policies = signal_policies or {}
    prepared_series = (
        prepared_signal_series
        if prepared_signal_series is not None
        else prepare_signal_series(samples, policies)
    )
    updated_vertices: list[DAGVertex] = []
    verdicts: dict[str, str] = {}
    for vertex in dag.vertices:
        if vertex.kind != "branch":
            updated_vertices.append(vertex)
            continue

        predicate = vertex.predicate_raw or vertex.predicate_lowered or ""
        evaluated_predicate = predicate
        prepared_predicate: Optional[_PreparedPredicate] = None
        verdict = _reduce_predicate(predicate, params, enums)
        windows: list[tuple[float, float]] = []

        if verdict == "unknown":
            if samples:
                prepared_predicate = _prepare_predicate(
                    predicate, samples, policies, prepared_series=prepared_series
                )
            evaluated = (
                _evaluate_predicate_intervals(
                    predicate,
                    params,
                    enums,
                    samples,
                    policies,
                    prepared=prepared_predicate,
                    prepared_series=prepared_series,
                )
                if samples
                else None
            )
            if evaluated is None:
                # Internal-state predicate (``_flare_states.flaring``) —
                # no direct param/logged reference. Ground it through the
                # graph: substitute each symbol with its producer's logged
                # form via the branch's own data edges, then retry.
                grounded = ground_expression_via_edges(
                    predicate, vertex.id, dag, enum_values=enums
                )
                if grounded and grounded != predicate:
                    evaluated_predicate = grounded
                    verdict = _reduce_predicate(grounded, params, enums)
                    if verdict == "unknown" and samples:
                        prepared_predicate = _prepare_predicate(
                            grounded,
                            samples,
                            policies,
                            prepared_series=prepared_series,
                        )
                        evaluated = _evaluate_predicate_intervals(
                            grounded,
                            params,
                            enums,
                            samples,
                            policies,
                            prepared=prepared_predicate,
                            prepared_series=prepared_series,
                        )
            if verdict == "unknown" and evaluated is not None:
                windows = evaluated
                if prepared_predicate is None:
                    prepared_predicate = _prepare_predicate(
                        evaluated_predicate,
                        samples,
                        policies,
                        prepared_series=prepared_series,
                    )
                predicate_span = prepared_predicate.span
                policies_complete = _predicate_policies_complete(
                    evaluated_predicate,
                    samples,
                    policies,
                    prepared=prepared_predicate,
                )
                if not windows and policies_complete:
                    verdict = "always_false"
                elif (
                    policies_complete
                    and predicate_span is not None
                    and _covers_span(windows, predicate_span)
                ):
                    verdict = "always_true"

        verdicts[vertex.id] = verdict
        metadata = dict(vertex.metadata or {})
        if samples and prepared_predicate is None:
            prepared_predicate = _prepare_predicate(
                evaluated_predicate,
                samples,
                policies,
                prepared_series=prepared_series,
            )
        evaluation_span = prepared_predicate.span if prepared_predicate else None
        if evaluation_span is not None:
            metadata["evaluation_domain"] = list(evaluation_span)
            metadata["sampling_policies"] = _predicate_policy_summary(
                evaluated_predicate,
                samples,
                policies,
                prepared=prepared_predicate,
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

    updated_edges = list(dag.edges)
    kept_ids = {v.id for v in updated_vertices}

    if prune_dead:
        # Incoming control edges encode the operation's reachability
        # conjunction. One false conjunct makes the operation unreachable.
        control_by_op: dict[str, list[str]] = {}
        for edge in updated_edges:
            if edge.kind == "control":
                control_by_op.setdefault(edge.target_id, []).append(edge.source_id)

        dead_op_ids: set[str] = set()
        for op_id, branch_ids in control_by_op.items():
            if branch_ids and any(verdicts.get(bid) == "always_false" for bid in branch_ids):
                dead_op_ids.add(op_id)

        # Also mark always_false branches as dead once every operation
        # they gate is gone (which is by definition here).
        dead_branch_ids = {bid for bid, verdict in verdicts.items() if verdict == "always_false"}

        kept_ids -= dead_op_ids | dead_branch_ids
        updated_vertices = [v for v in updated_vertices if v.id in kept_ids]
        updated_edges = [
            e for e in updated_edges
            if e.source_id in kept_ids and e.target_id in kept_ids
        ]

    return MechanismDAG(
        dag_id=dag.dag_id,
        terminal=dag.terminal,
        vertices=updated_vertices,
        edges=updated_edges,
        unresolved_symbols=dag.unresolved_symbols,
    )


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
    "target": <arg.field>, "expression": <RHS>}`` suitable for emission as either DAG operation
    vertices or profiler ``SourceAssignmentRef`` records. Single semantic
    owner for the substitution — both the profiler's flatten pass and
    the DAG builder's graph-native emission call this function, so a
    helper called from multiple sites produces the same per-site
    bindings regardless of which caller drove the derivation.

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
