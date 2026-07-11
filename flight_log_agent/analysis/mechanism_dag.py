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
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from flight_log_agent.analysis.parameter_lookup import (
    CXX_STDLIB_CONSTANTS,
    is_px4_parameter_name,
)
from flight_log_agent.analysis.source_expression import source_expression_names
from flight_log_agent.expression_math import is_safe_math_function_name
from flight_log_agent.px4.mechanism_source_profiler import substitute_expression_symbols
from flight_log_agent.symbols import (
    is_signal_reference,
    looks_like_enum_constant,
    normalize_symbol,
)
from flight_log_agent.utils import dedupe_keep_order, stable_id


VertexKind = Literal["evidence", "operation", "branch"]
EvidenceSubKind = Literal["logged_signal", "parameter", "constant", "opaque_symbol"]
OperationSubKind = Literal["assign", "reduction", "helper_call", "external_call", "unresolved"]
EdgeKind = Literal["data", "control"]


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
    out: set[str] = set()
    for topic, fields in ((inventory or {}).get("topic_fields") or {}).items():
        if not isinstance(topic, str):
            continue
        for field in fields or []:
            if isinstance(field, str) and field:
                out.add(f"{topic}.{field}")
    return out


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
    call_statements: Sequence[Any] = (),
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
    terminal's writers to that file when any exist there —
    ``normalize_symbol`` strips the leading ``_`` (member ↔ logged
    convention), so a multi-module binding set can contain a same-named
    but unrelated variable from another class (NPFG ``lateral_accel``
    vs L1 ``_lateral_accel``); the hint keeps the slice on the module
    the caller actually asked about.
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
        call_statements=[_as_binding_dict(c) for c in call_statements],
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
        call_statements: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.terminal_raw = terminal
        self.terminal = normalize_symbol(terminal)
        self.terminal_file = str(terminal_file) if terminal_file else None
        self._call_statements = list(call_statements or [])
        self.helper_index = helper_index
        self.helper_body_provider = helper_body_provider
        # Helpers already probed via the provider so a repeated call for an
        # unknown name doesn't re-fetch on every backward-walk pass.
        self._helper_provider_probed: set[str] = set()
        self.source_root = source_root
        self.logged_signals = {normalize_symbol(s) for s in logged_signals if s}
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

        self._schema_signals = {normalize_symbol(s) for s in schema_signals if s}

        # Native backward-walk indexes over the source bindings — the DAG
        # owns the walk rather than delegating to BindingIndex. ``_by_output``
        # keys on the resolved logged signal, ``_by_target`` on the written
        # symbol; the interleaved build in :meth:`build` traverses them.
        self._all_bindings: list[dict[str, Any]] = list(source_bindings)
        self._by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        writes_per_target: dict[str, int] = defaultdict(int)
        for binding in self._all_bindings:
            logged = normalize_symbol(str(binding.get("logged_signal") or ""))
            target = normalize_symbol(
                str(binding.get("target_symbol") or binding.get("target") or "")
            )
            if logged:
                self._by_output[logged].append(binding)
            if target:
                self._by_target[target].append(binding)
                writes_per_target[target] += 1

        # Source-defined numeric constants (enum entry / ``#define`` /
        # ``constexpr``), resolved natively from the bindings: a single
        # unconditional write whose RHS is a compile-time numeric literal.
        # Replaces the old dependency on ``BindingIndex.assignment_resolutions``
        # (which stored SliceResult objects the DAG mis-typed).
        self._source_constants: dict[str, Any] = {}
        for binding in self._all_bindings:
            target = normalize_symbol(
                str(binding.get("target_symbol") or binding.get("target") or "")
            )
            if not target or writes_per_target[target] != 1:
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

        # Memoization tables.
        self._evidence_by_signal: dict[tuple[str, str], str] = {}
        self._branch_by_predicate: dict[str, str] = {}
        self._helper_subgraph_return_id: dict[tuple[str, str], str] = {}
        # helper_key -> ordered list of (formal_name, param_vertex_id).
        # Callers wire their i-th argument's producer to the i-th formal
        # vertex; helper body operations that reference the formal look
        # it up in this map (scoped to the helper).
        self._helper_parameter_vertices: dict[
            tuple[str, str], list[tuple[str, str]]
        ] = {}
        # Assignment-target index: normalized target symbol → list of vertex ids
        # that produce it. Used to link consumers back to producing operations.
        self._producers_by_symbol: dict[str, list[str]] = {}

        # Snippet cache to avoid re-reading a file many times.
        self._file_lines_cache: dict[str, list[str]] = {}

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
        self._register_call_statement_bindings()

        # Every frontier symbol carries the scope of the site that
        # referenced it — writers are then resolved under C++-faithful
        # visibility instead of a global by-name index, so a local named
        # ``dt`` in one module can never bind to another module's ``dt``.
        Scope = tuple[str, str]  # (file, bare function)
        terminal_scope: Scope = (self.terminal_file or "", "")
        frontier: deque[tuple[str, str, Scope]] = deque(
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
            for helper_name in self._find_helper_calls(expression):
                if helper_name not in materialized_helpers:
                    frontier.append(("helper", helper_name, scope))

        while frontier:
            kind, payload, scope = frontier.popleft()
            if kind == "symbol":
                raw = payload
                norm = normalize_symbol(raw)
                if not norm or (norm, scope) in walked:
                    continue
                walked.add((norm, scope))
                if norm != self.terminal and norm in self._source_constants:
                    # Source-defined constants resolve as value-carrying
                    # evidence leaves at wiring time; walking their single
                    # literal write would demote them to bare operations.
                    continue
                if norm == self.terminal:
                    writers = self._writers_of(norm)
                    if writers and self.terminal_file:
                        # Scope the terminal to its module; fall back to
                        # all writers when none live in the hinted file.
                        scoped = [
                            b for b in writers
                            if self._binding_first_file(b) == self.terminal_file
                        ]
                        if scoped:
                            writers = scoped
                else:
                    writers = self._scoped_writers(norm, raw, scope)
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
                            self._binding_site_scope(binding),
                        )
                        # A branch's inputs are part of the mechanism:
                        # walking predicate symbols emits the internal-state
                        # writers that feasibility grounding later follows.
                        for predicate in binding.get("control_predicates") or []:
                            enqueue_expression(
                                str(predicate), self._binding_site_scope(binding)
                            )
                elif (
                    "." not in norm
                    and norm != self.terminal
                    and norm not in self.logged_signals
                    and self._match_parameter(norm) is None
                ):
                    # Bare struct root with no direct writer — pull its
                    # field writes (``_mission_item`` → ``_mission_item.*``).
                    for field_binding in self._field_writers_of(norm):
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
                                    self._binding_site_scope(field_binding),
                                )
                            )
            else:  # helper
                helper_name = payload
                if helper_name in materialized_helpers:
                    continue
                materialized_helpers.add(helper_name)
                self._materialize_helper_subgraph(helper_name, wire_edges=False)
                helper_key = self._pick_helper_key(helper_name)
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
                    self._bare_function(helper_key[0] if helper_key else helper_name),
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
            normalize_symbol(name) for name in source_expression_names(normalized)
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

    def _register_call_statement_bindings(self) -> None:
        """Model side-effect argument flow across bare call statements.

        ``obj.method(a, b);`` passes each actual into the callee's formal,
        which the callee body reads as ordinary state — but the statement
        is not an assignment, so no binding exists and the backward walk
        cannot cross the argument hop (a caller local feeding a member
        object's published field stays invisible). Synthesize one binding
        ``formal <- actual`` per resolvable call; the callee body's own
        assignments are ordinary source bindings already.

        Only IN-SET callees are considered: an unloaded callee has no body
        bindings to walk, so the synthesized hop would be inert — and
        probing the provider with generic statement names (``update``)
        drags junk definitions. When several loaded classes define the
        name, a callee whose class context matches the receiver's struct
        type wins; else the deterministic first match (the Milestone 2
        receiver-typing note on :meth:`_pick_helper_key` applies here too).
        """
        for call in self._call_statements:
            name = str(call.get("name") or "")
            args = [str(a) for a in (call.get("args") or []) if str(a).strip()]
            if not name or len(name) < 4 or not args:
                continue
            if is_safe_math_function_name(name):
                continue
            matches = [key for key in self.helper_index if key[0] == name]
            if not matches:
                continue
            receiver = str(call.get("receiver") or "")
            receiver_type = (
                self._struct_variables.get(receiver)
                or self._struct_variables.get(receiver.lstrip("_"))
                if receiver
                else None
            )
            # Bind only an UNAMBIGUOUS callee — a generic statement name
            # (``update``) matched by sorted-first would spray one class's
            # formals with every caller's actuals and fuse unrelated
            # modules. Resolution: exact receiver struct type, then
            # receiver↔class name affinity (member ``_tecs`` ↔ class
            # ``TECS``, underscores/case ignored), then a unique name.
            helper_key = None
            if receiver_type:
                exact = [k for k in matches if str(k[1]) == receiver_type]
                if exact:
                    helper_key = sorted(exact)[0]
            if helper_key is None and receiver:
                affinity = receiver.strip("_").replace("_", "").lower()
                akin = [
                    k
                    for k in matches
                    if str(k[1] or "").replace("_", "").lower() == affinity
                ]
                if akin:
                    helper_key = sorted(akin)[0]
            if helper_key is None and len(matches) == 1:
                helper_key = matches[0]
            if helper_key is None:
                continue
            helper = self.helper_index.get(helper_key) or {}
            formals = [str(f) for f in (helper.get("parameters") or [])]
            if not formals:
                continue
            file = str(call.get("file") or "")
            line_raw = call.get("line")
            line = int(line_raw) if isinstance(line_raw, (int, float)) else 0
            predicates = list(call.get("control_predicates") or [])
            for formal, actual in zip(formals, args):
                formal_norm = normalize_symbol(formal)
                if not formal_norm:
                    continue
                self._by_target[formal_norm].append(
                    {
                        "target_symbol": formal,
                        "source_symbol": actual,
                        "assignment_path": [
                            {"file": file, "line": line, "expression": actual}
                        ],
                        "logged_signal": "",
                        "control_predicates": predicates,
                        "struct_variables": {},
                        # The formal lives in the CALLEE; the actual's
                        # symbols resolve at the call site (see
                        # _binding_target_scope / _binding_site_scope).
                        "scope_file": str(helper.get("file") or ""),
                        "scope_function": name,
                        "function": "",
                    }
                )

    @staticmethod
    def _binding_first_file(binding: dict[str, Any]) -> str:
        path = binding.get("assignment_path") or []
        first = path[0] if path else {}
        return str((first or {}).get("file") or "")

    @staticmethod
    def _bare_function(name: Any) -> str:
        return str(name or "").rsplit("::", 1)[-1].strip()

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
            binding.get("scope_function") or binding.get("function") or ""
        )
        return (file, function)

    def _binding_site_scope(self, binding: dict[str, Any]) -> tuple[str, str]:
        """Where the binding's EXPRESSION text lives — the scope its
        referenced symbols are resolved in."""
        return (
            self._binding_first_file(binding),
            self._bare_function(binding.get("function") or ""),
        )

    def _scoped_writers(
        self,
        symbol_norm: str,
        symbol_raw: str,
        scope: tuple[str, str],
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
        """
        output_writers = list(self._by_output.get(symbol_norm, []))
        target_writers = list(self._by_target.get(symbol_norm, []))
        scope_file, scope_function = scope
        if scope_file and target_writers:
            root = symbol_raw.split(".", 1)[0].split("->", 1)[0].strip().strip("&*")
            # PX4 marks members with a LEADING underscore in modules and a
            # TRAILING one in libraries (``airspeed_ref_``) — both are
            # class members, visible across the class file family.
            if root.startswith("_") or root.endswith("_"):
                family = self._file_family(scope_file)
                same_family = [
                    b
                    for b in target_writers
                    if self._file_family(self._binding_target_scope(b)[0]) == family
                ]
                target_writers = same_family or target_writers
            else:
                scoped = []
                for binding in target_writers:
                    b_file, b_function = self._binding_target_scope(binding)
                    if b_file != scope_file:
                        continue
                    if scope_function and b_function and b_function != scope_function:
                        continue
                    scoped.append(binding)
                target_writers = scoped
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for binding in output_writers + target_writers:
            if id(binding) not in seen:
                seen.add(id(binding))
                out.append(binding)
        return out

    def _writers_of(self, symbol_norm: str) -> list[dict[str, Any]]:
        """Bindings that write ``symbol_norm`` (as a logged output or a
        source target). Union of both indexes, de-duplicated by identity."""
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for binding in list(self._by_output.get(symbol_norm, [])) + list(
            self._by_target.get(symbol_norm, [])
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
        target_norm = normalize_symbol(target_raw)
        expression = str(binding.get("source_symbol") or "").strip()
        assignment_path = binding.get("assignment_path") or []
        first_hop = assignment_path[0] if assignment_path else {}
        file = str(first_hop.get("file") or "") or None
        line_raw = first_hop.get("line")
        line = int(line_raw) if isinstance(line_raw, (int, float)) else None
        op_id = self._make_id("op", (target_norm, expression, file or "", line or 0))
        return op_id, target_raw, file, line, target_norm, expression

    def _emit_operation_vertex(self, binding: dict[str, Any], *, is_terminal: bool) -> str:
        op_id, target_raw, file, line, target_norm, expression = self._binding_operation_id(binding)
        if op_id not in self.vertices:
            self.vertices[op_id] = DAGVertex(
                id=op_id,
                kind="operation",
                sub_kind=self._classify_operation(expression),
                file=file,
                line=line,
                snippet=self._snippet(file, line),
                variable=target_raw,
                expression=expression,
                metadata={"is_terminal": is_terminal} if is_terminal else {},
            )
            self._producers_by_symbol.setdefault(target_norm, []).append(op_id)
        return op_id

    def _wire_binding_edges(self, binding: dict[str, Any]) -> None:
        op_id, _target_raw, file, line, target_norm, expression = self._binding_operation_id(binding)

        # Attach control-predicate branches.
        for predicate in binding.get("control_predicates") or []:
            branch_id = self._emit_branch(str(predicate), file=file, line=line)
            self._add_edge(branch_id, op_id, kind="control")

        # Wire each source-expression symbol as an incoming data edge.
        for symbol in self._wire_symbols(expression):
            normalized = normalize_symbol(symbol)
            if not normalized or normalized == target_norm:
                continue
            producer_id = self._resolve_symbol_producer(normalized, symbol, expression, file, line)
            if producer_id is not None:
                self._add_edge(producer_id, op_id, kind="data", role=symbol)

        # Helper-call inputs.
        for helper_call in self._find_helper_calls(expression):
            helper_key = self._pick_helper_key(helper_call)
            if helper_key is None:
                continue
            helper_return_id = self._helper_subgraph_return_id.get(helper_key)
            if helper_return_id:
                self._add_edge(
                    helper_return_id,
                    op_id,
                    kind="data",
                    role=f"call:{helper_call}",
                    via=helper_call,
                )
            # Wire the caller's actual arguments to the helper's formal
            # parameter vertices — one edge per positional match.
            self._wire_helper_call_arguments(helper_call, expression, file, line)
            # Emit pointer-output writes from the helper as ops with the
            # caller's actual arg substituted for the pointer formal. Ops
            # dedupe with any source_assignments-derived vertex that
            # already covers the same call site.
            self._emit_helper_pointer_output_writes(
                helper_call, expression, file=file, line=line
            )

    def _emit_helper_pointer_output_writes(
        self,
        helper_call: str,
        caller_expression: str,
        *,
        file: Optional[str],
        line: Optional[int],
    ) -> None:
        """Graph-native equivalent of the profiler's pointer-output routing.

        For each ``pointer_output_writes`` entry on the resolved helper, emit
        an operation vertex whose target is ``{caller_actual_arg}.{field}``
        with the write's RHS as the expression. Wires RHS symbols as data
        edges. Ops share the (target, expression, file, line) identity used
        by :meth:`_emit_operation_vertex` so a source_assignments-derived
        binding for the same call site does not double-emit.
        """
        helper_key = self._pick_helper_key(helper_call)
        if helper_key is None:
            return
        helper = self.helper_index.get(helper_key)
        if not helper:
            return
        pointer_writes = helper.get("pointer_output_writes") or []
        if not pointer_writes:
            return
        pointer_params = _pointer_param_positions(helper)
        if not pointer_params:
            return
        args = _extract_call_arguments(helper_call, caller_expression)
        substituted = derive_pointer_output_bindings(pointer_writes, pointer_params, args)
        for entry in substituted:
            target = entry["target"]
            expression = entry["expression"]
            target_norm = normalize_symbol(target)
            op_id = self._make_id("op", (target_norm, expression, file or "", line or 0))
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
                )
                self._producers_by_symbol.setdefault(target_norm, []).append(op_id)
            for symbol in self._wire_symbols(expression):
                normalized = normalize_symbol(symbol)
                if not normalized or normalized == target_norm:
                    continue
                producer_id = self._resolve_symbol_producer(
                    normalized, symbol, expression, file, line
                )
                if producer_id is not None:
                    self._add_edge(producer_id, op_id, kind="data", role=symbol)

    def _wire_helper_call_arguments(
        self,
        helper_call: str,
        caller_expression: str,
        file: Optional[str],
        line: Optional[int],
    ) -> None:
        """Parse the ``helper_call(...)`` in ``caller_expression`` and wire
        each positional argument to the helper's matching formal parameter
        vertex.

        Uses the first occurrence of ``helper_call(...)`` in the expression.
        Multiple invocations in one expression get handled if they appear
        as separate entries in the ``_find_helper_calls`` result — the
        function is invoked once per name; a limitation on today's parser,
        good enough for the RTL/airspeed cases.
        """
        helper_key = self._pick_helper_key(helper_call)
        if helper_key is None:
            return
        formals = self._helper_parameter_vertices.get(helper_key)
        if not formals:
            return
        args = _extract_call_arguments(helper_call, caller_expression)
        for (formal_name, formal_vertex_id), arg_text in zip(formals, args):
            for symbol in self._wire_symbols(arg_text):
                normalized = normalize_symbol(symbol)
                if not normalized:
                    continue
                producer_id = self._resolve_symbol_producer(
                    normalized, symbol, arg_text, file, line
                )
                if producer_id is not None:
                    self._add_edge(
                        producer_id,
                        formal_vertex_id,
                        kind="data",
                        role=f"arg:{formal_name}",
                        via=helper_call,
                    )

    def _resolve_symbol_producer(
        self,
        symbol_norm: str,
        symbol_raw: str,
        source_expression: str,
        file: Optional[str],
        line: Optional[int],
    ) -> Optional[str]:
        """Link a source-expression symbol to a producer vertex.

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
        6. Parameter accessor match — heuristic; will retire when we
           consume profiler ``ReferencedParameterRef`` records directly.
        7. Enum-shaped name (``looks_like_enum_constant``) — heuristic;
           retires when source constant coverage is complete.
        8. Otherwise: opaque symbol.

        No flat ``symbol_bindings`` table is consulted anywhere — every
        source→logged mapping is derived from graph structure.
        """
        producers = self._producers_by_symbol.get(symbol_norm)
        if producers:
            # Prefer a producer from the consumer's own file — the
            # leading-underscore strip in ``normalize_symbol`` can fuse a
            # member with a same-named local from another module, and
            # same-file linkage is the strongest disambiguation available
            # without full class scoping. Fall back to the last producer.
            if file:
                same_file = [
                    p for p in producers if self.vertices[p].file == file
                ]
                if same_file:
                    return same_file[-1]
            return producers[-1]

        # 2a. Graph-native derivation: if ``source_expression`` contains a
        # ``symbol_raw().field`` chain, resolve it via the helper's
        # ``return_type`` and the PX4 msg schema.
        chain_resolved = self._resolve_symbol_via_chain(symbol_raw, source_expression)
        if chain_resolved is not None:
            return self._emit_evidence(
                "logged_signal",
                chain_resolved,
                file=None,
                line=None,
                metadata={"source_form": symbol_raw, "derivation": "helper_return_type"},
            )

        # 2b. Graph-native struct-variable derivation: ``var.field`` where
        # ``var`` is struct-typed (local declaration or class member).
        # Same PX4 ``foo_s`` convention as helper return types; no flat
        # side-table.
        struct_resolved = self._resolve_symbol_via_struct_var(symbol_raw, source_expression)
        if struct_resolved is not None:
            return self._emit_evidence(
                "logged_signal",
                struct_resolved,
                file=None,
                line=None,
                metadata={"source_form": symbol_raw, "derivation": "struct_variable"},
            )

        # 3. Source enum / #define resolution.
        enum_value = self._source_constants.get(symbol_norm)
        if enum_value is not None:
            return self._emit_evidence(
                "constant",
                symbol_raw,
                file=None,
                line=None,
                metadata={"value": enum_value, "source": "enum"},
            )

        # 4. C stdlib constant.
        cxx_value = CXX_STDLIB_CONSTANTS.get(symbol_raw.upper())
        if cxx_value is not None:
            return self._emit_evidence(
                "constant",
                symbol_raw.upper(),
                file=None,
                line=None,
                metadata={"value": cxx_value, "source": "cxx_stdlib"},
            )

        # 5. Canonical logged signal (direct set membership).
        if symbol_norm in self.logged_signals:
            return self._emit_evidence("logged_signal", symbol_raw, file=None, line=None)

        # 6. Parameter accessor heuristic.
        parameter_alias = self._match_parameter(symbol_raw)
        if parameter_alias is not None:
            return self._emit_evidence("parameter", parameter_alias, file=None, line=None)

        # 7a. Bare PX4-parameter-shaped name resolved through the ULog
        # parameter inventory. Handles source RHSes like ``FW_AIRSPD_TRIM``
        # that aren't dotted (so ``looks_like_enum_constant`` skips them).
        parameter_value = self._parameter_values.get(symbol_raw.upper())
        if parameter_value is not None and is_px4_parameter_name(symbol_raw.upper()):
            return self._emit_evidence(
                "constant",
                symbol_raw,
                file=None,
                line=None,
                metadata={"value": parameter_value, "source": "parameter"},
            )

        # 7b. Enum-shaped name (no value known).
        if looks_like_enum_constant(symbol_raw):
            return self._emit_evidence("constant", symbol_raw, file=None, line=None)

        # 8. Unclassified.
        self.unresolved_symbols.add(symbol_raw)
        return self._emit_evidence("opaque_symbol", symbol_raw, file=file, line=line)

    def _lower_predicate(
        self, canonical: str
    ) -> tuple[str, dict[str, str]]:
        """Return ``(lowered_expression, variables)`` for a canonical predicate.

        Fully graph-native — no flat ``symbol_bindings`` table anywhere.
        Three passes:

        1. **Helper-chain resolution** — ``chain().field`` gets replaced with
           ``topic.field`` by looking up the helper's ``return_type`` and
           cross-referencing the PX4 msg schema.
        2. **Struct-variable resolution** — ``var.field`` gets replaced when
           ``var`` was declared struct-typed (local or class-member); the
           topic is derived via the same PX4 ``foo_s`` convention.
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
            value = self._source_constants.get(normalize_symbol(token))
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
        if not self._struct_variables:
            return text, variables

        for var in sorted(self._struct_variables, key=len, reverse=True):
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
                    variables[f"{matched_var}.{field}"] = resolved
                    return resolved
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
            variables[match.group(0)] = resolved
            return resolved

        substituted = _HELPER_CHAIN_RE.sub(replace, text)
        return substituted, variables

    def _resolve_symbol_via_struct_var(
        self, symbol_raw: str, source_expression: str
    ) -> Optional[str]:
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
            resolved = self._resolve_struct_var_field(var_candidate, field)
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
        return self._resolve_struct_var_field(symbol_raw, match.group("field"))

    def _resolve_symbol_via_chain(
        self, symbol_raw: str, source_expression: str
    ) -> Optional[str]:
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
        return self._resolve_helper_chain(symbol_raw, match.group("field"))

    def _resolve_struct_var_field(self, var: str, field: str) -> Optional[str]:
        """Return ``topic.field`` when ``var.field`` is graph-derivable.

        Looks up ``var`` in the aggregated struct-variables map, derives the
        topic from the struct type via :func:`_derive_topic_from_return_type`
        (identical convention to helper return types), and validates the
        resulting ``topic.field`` against the trusted signal catalogue.
        Returns ``None`` when ``var`` isn't struct-typed, when the type
        doesn't follow the ``foo_s`` convention, or when the topic.field
        isn't in the catalogue.
        """
        struct_type = self._struct_variables.get(var)
        if not struct_type:
            struct_type = self._struct_variables.get(var.lstrip("_"))
        if not struct_type:
            return None
        topic = _derive_topic_from_return_type(struct_type)
        if not topic:
            return None
        signal = f"{topic}.{field}"
        schema_signals = self._schema_signals
        if signal in self.logged_signals or signal in schema_signals:
            return signal
        return None

    def _resolve_helper_chain(self, chain: str, field: str) -> Optional[str]:
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
        helper_name = segments[-1]
        helper_key = self._pick_helper_key(helper_name)
        if helper_key is None:
            return None
        helper = self.helper_index.get(helper_key)
        if not helper:
            return None
        topic = _derive_topic_from_return_type(helper.get("return_type"))
        if not topic:
            return None
        signal = f"{topic}.{field}"
        schema_signals = self._schema_signals
        if signal in self.logged_signals or signal in schema_signals:
            return signal
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
        key = (sub_kind, signal)
        existing = self._evidence_by_signal.get(key)
        if existing is not None:
            return existing
        vertex_id = self._make_id("ev", key)
        self.vertices[vertex_id] = DAGVertex(
            id=vertex_id,
            kind="evidence",
            sub_kind=sub_kind,
            file=file,
            line=line,
            snippet=self._snippet(file, line),
            signal_name=signal,
            metadata=dict(metadata) if metadata else {},
        )
        self._evidence_by_signal[key] = vertex_id
        return vertex_id

    def _emit_branch(
        self,
        predicate: str,
        *,
        file: Optional[str],
        line: Optional[int],
    ) -> str:
        canonical = _canonical_predicate(predicate)
        existing = self._branch_by_predicate.get(canonical)
        if existing is not None:
            return existing
        vertex_id = self._make_id("br", canonical)
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
        self._branch_by_predicate[canonical] = vertex_id

        # A branch's own predicate depends on the symbols it reads. Wire
        # data edges from those producers so pre-evaluation later has
        # every input in the graph. Normalize C++ operators/casts first so
        # ``&&``/``->``/``::`` predicates yield their symbols (parameters,
        # accessor chains) instead of failing ast extraction wholesale.
        symbol_source = _normalize_cpp_expression(predicate)
        for symbol in dedupe_keep_order(source_expression_names(symbol_source)):
            normalized = normalize_symbol(symbol)
            if not normalized:
                continue
            producer_id = self._resolve_symbol_producer(normalized, symbol, symbol_source, file, line)
            if producer_id is not None:
                self._add_edge(producer_id, vertex_id, kind="data", role=symbol)

        return vertex_id

    # ------------------------------------------------------------
    # Helper subgraph nesting
    # ------------------------------------------------------------

    def _find_helper_calls(self, expression: str) -> list[str]:
        """Return helper names invoked in ``expression`` that we can expand.

        A name is expandable when it appears in ``helper_index`` under some
        class context, or when the on-demand provider can fetch it — the
        latter check invokes :meth:`_pick_helper_key` which memoizes probes
        so an unknown name is asked at most once per build.

        Two guards:

        * Math-function names (the ``expression_math`` vocabulary) never
          expand — they're evaluator-native, and a template-math overload
          (``Dual<S,N> sqrt(...)``) is not the mechanism's helper.
        * A **dotted short** head (``.get(``, ``->dot(``) is a method on
          another object — a param/uORB accessor or vector op — and a
          bare-name match against the whole loaded helper set would let a
          getter literally named ``get`` from an unrelated class claim
          it. Long dotted heads (``_mission.get_landing_alt()``) stay
          eligible, as do short *bare* heads; full receiver-class vs
          ``class_context`` disambiguation is the Milestone 2 work noted
          on :meth:`_pick_helper_key`.
        """
        matches: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression):
            candidate = match.group(1)
            if candidate in seen:
                continue
            if is_safe_math_function_name(candidate):
                continue
            preceding = expression[max(0, match.start() - 2):match.start()]
            if len(candidate) < 4 and (
                preceding.endswith(".") or preceding.endswith("->")
            ):
                continue
            seen.add(candidate)
            if self._pick_helper_key(candidate) is not None:
                matches.append(candidate)
        return matches

    def _materialize_helper_subgraph(self, helper_name: str, *, wire_edges: bool = True) -> Optional[str]:
        """Emit the helper's body as nested vertices, return its output vertex id.

        Memoized by ``(helper_name, class_context)``. Preserves every
        intermediate assignment as its own vertex — no collapsing.

        When ``wire_edges`` is False, only vertices are emitted (pass 1).
        Edges are wired by :meth:`_wire_helper_subgraph_edges` in pass 2.
        """
        helper_key = self._pick_helper_key(helper_name)
        if helper_key is None:
            return None
        cached = self._helper_subgraph_return_id.get(helper_key)
        if cached is not None:
            return cached or None

        helper = self.helper_index[helper_key]
        file = helper.get("file")
        line = helper.get("line")

        # Emit a helper_parameter vertex for each formal so callers can
        # wire their actual arguments into shared entry points and helper
        # body operations can resolve references to formals against them.
        formals: list[tuple[str, str]] = []
        for formal in helper.get("parameters") or []:
            formal_str = str(formal)
            if not formal_str:
                continue
            param_id = self._make_id("ev", ("helper_param", helper_key[0], helper_key[1], formal_str))
            if param_id not in self.vertices:
                self.vertices[param_id] = DAGVertex(
                    id=param_id,
                    kind="evidence",
                    sub_kind="helper_parameter",
                    file=file,
                    line=line,
                    signal_name=formal_str,
                    metadata={"helper": f"{helper_key[0]}@{helper_key[1]}"},
                )
            formals.append((formal_str, param_id))
        self._helper_parameter_vertices[helper_key] = formals

        for var, expression in (helper.get("assignments") or {}).items():
            var_norm = normalize_symbol(var)
            op_id = self._make_id("op", (helper_key[0], helper_key[1], var_norm, expression))
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
                )
                self._producers_by_symbol.setdefault(var_norm, []).append(op_id)

        return_expression = (
            helper.get("lowered_return_expression")
            or helper.get("return_expression")
        )
        if not return_expression:
            self._helper_subgraph_return_id[helper_key] = ""
            return None

        terminal_id = self._make_id(
            "op",
            (helper_key[0], helper_key[1], "__return__", return_expression),
        )
        if terminal_id not in self.vertices:
            self.vertices[terminal_id] = DAGVertex(
                id=terminal_id,
                kind="operation",
                sub_kind="helper_call",
                file=file,
                line=line,
                snippet=self._snippet(file, line),
                variable=f"{helper_key[1]}::__return__",
                expression=return_expression,
                lowered_expression=return_expression,
                provenance=f"helper_return:{helper_key[0]}@{helper_key[1]}",
            )
        self._helper_subgraph_return_id[helper_key] = terminal_id

        if wire_edges:
            self._wire_helper_subgraph_edges(helper_key)
        return terminal_id

    def _wire_helper_subgraph_edges(self, helper_key: tuple[str, str]) -> None:
        helper = self.helper_index.get(helper_key)
        if not helper:
            return
        file = helper.get("file")
        line = helper.get("line")

        # Helper-scoped local resolver: formal parameter names bind to
        # their formal-parameter vertex before falling back to the global
        # producer index. This preserves helper-local scoping when two
        # helpers happen to share a formal name (``float x``).
        local_scope = {
            normalize_symbol(formal): vertex_id
            for formal, vertex_id in self._helper_parameter_vertices.get(helper_key, ())
        }

        for var, expression in (helper.get("assignments") or {}).items():
            var_norm = normalize_symbol(var)
            op_id = self._make_id("op", (helper_key[0], helper_key[1], var_norm, expression))
            for symbol in self._wire_symbols(str(expression)):
                normalized = normalize_symbol(symbol)
                if not normalized or normalized == var_norm:
                    continue
                producer_id = local_scope.get(normalized) or self._resolve_symbol_producer(
                    normalized, symbol, str(expression), file, line
                )
                if producer_id is not None:
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
                (helper_key[0], helper_key[1], "__return__", return_expression),
            )
            for symbol in self._wire_symbols(str(return_expression)):
                normalized = normalize_symbol(symbol)
                if not normalized:
                    continue
                producer_id = local_scope.get(normalized) or self._resolve_symbol_producer(
                    normalized, symbol, str(return_expression), file, line
                )
                if producer_id is not None:
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
            branch_id = self._emit_branch(condition, file=file, line=line)
            if terminal_id is not None:
                self._add_edge(branch_id, terminal_id, kind="control")
            value_expression = str(branch.get("expression") or "")
            if terminal_id is None or not value_expression:
                continue
            for symbol in self._wire_symbols(value_expression):
                normalized = normalize_symbol(symbol)
                if not normalized:
                    continue
                producer_id = local_scope.get(normalized) or self._resolve_symbol_producer(
                    normalized, symbol, value_expression, file, line
                )
                if producer_id is not None:
                    self._add_edge(producer_id, terminal_id, kind="data", role=f"branch:{symbol}")

    def _pick_helper_key(self, helper_name: str) -> Optional[tuple[str, str]]:
        """Choose one ``(name, class_context)`` for a call like ``foo(...)``.

        Milestone 1: if multiple class contexts define ``foo``, pick the
        first deterministically. Multi-context disambiguation via the
        caller's class scope is Milestone 2 work.

        On miss, consult the on-demand helper provider (if any) — the
        cross-file callee isn't in the pre-flattened helper set but the
        provider can locate and lower it. Result gets merged into
        ``helper_index`` so subsequent lookups skip the provider call.
        """
        matches = [key for key in self.helper_index if key[0] == helper_name]
        if matches:
            return sorted(matches)[0]
        if self.helper_body_provider is None or helper_name in self._helper_provider_probed:
            return None
        self._helper_provider_probed.add(helper_name)
        fetched = self.helper_body_provider(helper_name)
        for helper in _coerce_helpers(fetched):
            key_iter = _index_helpers([helper])
            for key, value in key_iter.items():
                self.helper_index.setdefault(key, value)
        matches = [key for key in self.helper_index if key[0] == helper_name]
        if not matches:
            return None
        return sorted(matches)[0]

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
        logged = normalize_symbol(str(binding.get("logged_signal") or ""))
        target = normalize_symbol(str(binding.get("target_symbol") or ""))
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
    """Index ``helper_expressions`` by ``(short_name, class_context)``.

    ``class_context`` combines the enclosing class (if any) with the
    function name, i.e. ``"RTL::calculate_return_alt_from_cone_half_angle"``
    or ``"::free_function_name"`` when there's no class.
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for helper in helpers:
        as_dict = helper if isinstance(helper, dict) else _helper_to_dict(helper)
        name = str(as_dict.get("name") or "")
        if not name:
            continue
        short = name.split("::")[-1]
        class_context = _class_context_from_name_or_evidence(name, str(as_dict.get("evidence") or ""))
        index[(short, class_context)] = as_dict
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

    enums = {str(k): v for k, v in (enum_values or {}).items()}

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
        result = text
        for edge in edges_by_target.get(target_id, []):
            role = str(edge.role)
            replacement = producer_form(edge.source_id, depth, seen)
            if replacement is None:
                continue
            result = substitute_expression_symbols(result, [role], [replacement])
        # Enum constants (``launch_detection_status_s::STATE_X``).
        def enum_sub(match: "re.Match[str]") -> str:
            name = match.group("name")
            value = enums.get(name)
            return _format_lowered_value(value) if value is not None else match.group(0)

        result = re.sub(
            r"\b[A-Za-z_]\w*_s::(?P<name>[A-Z][A-Z0-9_]+)\b", enum_sub, result
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
    full_span = _sample_span(samples)

    updated_vertices: list[DAGVertex] = []
    verdicts: dict[str, str] = {}
    for vertex in dag.vertices:
        if vertex.kind != "branch":
            updated_vertices.append(vertex)
            continue

        predicate = vertex.predicate_raw or vertex.predicate_lowered or ""
        verdict = _reduce_predicate(predicate, params, enums)
        windows: list[tuple[float, float]] = []

        if verdict == "unknown":
            evaluated = (
                _evaluate_predicate_intervals(predicate, params, enums, samples)
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
                    verdict = _reduce_predicate(grounded, params, enums)
                    if verdict == "unknown" and samples:
                        evaluated = _evaluate_predicate_intervals(
                            grounded, params, enums, samples
                        )
            if verdict == "unknown" and evaluated is not None:
                windows = evaluated
                if not windows:
                    verdict = "always_false"
                elif full_span is not None and _covers_span(windows, full_span):
                    verdict = "always_true"

        verdicts[vertex.id] = verdict
        updated_vertices.append(
            vertex.model_copy(update={"feasibility_verdict": verdict, "active_windows": windows})
        )

    updated_edges = list(dag.edges)
    kept_ids = {v.id for v in updated_vertices}

    if prune_dead:
        # Find operations whose incoming control edges are ALL always_false.
        control_by_op: dict[str, list[str]] = {}
        for edge in updated_edges:
            if edge.kind == "control":
                control_by_op.setdefault(edge.target_id, []).append(edge.source_id)

        dead_op_ids: set[str] = set()
        for op_id, branch_ids in control_by_op.items():
            if branch_ids and all(verdicts.get(bid) == "always_false" for bid in branch_ids):
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

    # Substitute PX4 parameter accessors: `_param_rtl_cone_ang.get()` →
    # bare identifier `RTL_CONE_ANG` that the eval env can bind.
    text = _PARAM_ACCESSOR_RE.sub(lambda m: m.group("name").upper(), predicate)
    # C++ boolean operators → Python.
    text = text.replace("&&", " and ").replace("||", " or ")
    # Unary not, but NOT `!=`.
    text = re.sub(r"!(?!=)", " not ", text)
    # C++ member and scope access → Python attribute.
    text = text.replace("->", ".").replace("::", ".")

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
    """Apply the same C++ → Python substitutions used by ``_reduce_predicate``."""
    text = _PARAM_ACCESSOR_RE.sub(lambda m: m.group("name").upper(), predicate)
    text = text.replace("&&", " and ").replace("||", " or ")
    text = re.sub(r"!(?!=)", " not ", text)
    return text.replace("->", ".").replace("::", ".")


def _evaluate_predicate_intervals(
    predicate: str,
    parameter_values: dict[str, Any],
    enum_values: dict[str, Any],
    signal_samples: dict[str, list[tuple[float, Any]]],
) -> Optional[list[tuple[float, float]]]:
    """Evaluate ``predicate`` per timestamp, return True-intervals.

    Uses hold-last policy: each signal's value between samples is the
    last-observed value. Returns None when no referenced signal has any
    sample or when the predicate fails to evaluate at every timestamp.
    """
    if not predicate.strip():
        return None

    from flight_log_agent.analysis.safe_eval import (
        ExpressionEvaluationError,
        eval_expression,
    )

    text = _substitute_predicate_syntax(predicate)

    # Rewrite ``topic.field`` → ``topic__field`` so ``safe_eval`` can bind
    # against a flat env key (attribute nodes are unsupported).
    signal_key_map: dict[str, str] = {}
    referenced: list[str] = []
    for signal in signal_samples:
        if signal in text:
            flat = signal.replace(".", "__")
            text = text.replace(signal, flat)
            signal_key_map[signal] = flat
            referenced.append(signal)

    if not referenced:
        return None

    # Union of timestamps from referenced signals.
    all_ts: set[float] = set()
    for signal in referenced:
        for ts, _ in signal_samples[signal]:
            all_ts.add(ts)
    if not all_ts:
        return None
    ts_sorted = sorted(all_ts)

    # Hold-last cursors per signal.
    cursor: dict[str, int] = {signal: 0 for signal in referenced}
    hold_last: dict[str, Any] = {}

    intervals: list[tuple[float, float]] = []
    current_start: Optional[float] = None
    ever_evaluated = False

    for t in ts_sorted:
        for signal in referenced:
            samples = signal_samples[signal]
            while cursor[signal] < len(samples) and samples[cursor[signal]][0] <= t:
                hold_last[signal_key_map[signal]] = samples[cursor[signal]][1]
                cursor[signal] += 1

        # Skip timestamps before we have any value for a referenced signal.
        if any(signal_key_map[s] not in hold_last for s in referenced):
            continue

        env = {**parameter_values, **enum_values, **hold_last}
        try:
            result = eval_expression(text, env)
        except (ExpressionEvaluationError, TypeError, ValueError, ZeroDivisionError):
            return None

        ever_evaluated = True
        truthy = bool(result) if isinstance(result, (bool, int, float)) else False

        if truthy and current_start is None:
            current_start = t
        elif not truthy and current_start is not None:
            intervals.append((current_start, t))
            current_start = None

    if not ever_evaluated:
        return None
    if current_start is not None:
        intervals.append((current_start, ts_sorted[-1]))
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
    """Return {formal_name → positional index} for pointer parameters.

    The profiler stamps ``pointer_output_writes`` with the FORMAL name; the
    DAG builder needs the positional index to look up the caller's actual
    argument. Positions are inferred from ``parameters`` order; type info
    ``*`` vs ``&`` is dropped since the profiler already gated on it when
    it decided the write was routable.
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

    Returns entries ``{"param": <formal>, "target": <arg.field>,
    "expression": <RHS>}`` suitable for emission as either DAG operation
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
        if index is None or index >= len(call_args) or not field or not expression:
            continue
        arg = str(call_args[index]).strip().lstrip("&").strip()
        if not arg:
            continue
        results.append({
            "param": param,
            "target": f"{arg}.{field}",
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
