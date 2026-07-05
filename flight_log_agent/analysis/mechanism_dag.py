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
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from flight_log_agent.analysis.binding_index import BindingIndex
from flight_log_agent.analysis.source_expression import source_expression_names
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


def build_mechanism_dag(
    binding_index: BindingIndex,
    terminal: str,
    *,
    helper_expressions: Sequence[Any] = (),
    source_root: Optional[str | Path] = None,
    logged_signals: Optional[Iterable[str]] = None,
    parameter_names: Optional[Iterable[str]] = None,
    snippet_context_lines: int = 3,
) -> MechanismDAG:
    """Build a mechanism DAG for ``terminal``.

    ``binding_index`` supplies the backward walk over materialized
    source assignments. ``helper_expressions`` are the profiler's
    ``HelperExpressionRef`` records used to expand helper subgraphs
    without collapsing intermediates. ``source_root`` enables per-vertex
    snippet embedding — omit to keep tests hermetic.
    """
    builder = _DAGBuilder(
        binding_index=binding_index,
        terminal=terminal,
        helper_index=_index_helpers(helper_expressions),
        source_root=Path(source_root) if source_root else None,
        logged_signals=set(logged_signals or ()) | set(binding_index.logged_signals),
        parameter_names=set(parameter_names or ()),
        snippet_context_lines=snippet_context_lines,
    )
    return builder.build()


# ---------------------------------------------------------------------------
# Builder internals
# ---------------------------------------------------------------------------


class _DAGBuilder:
    def __init__(
        self,
        *,
        binding_index: BindingIndex,
        terminal: str,
        helper_index: dict[tuple[str, str], dict[str, Any]],
        source_root: Optional[Path],
        logged_signals: set[str],
        parameter_names: set[str],
        snippet_context_lines: int,
    ) -> None:
        self.binding_index = binding_index
        self.terminal_raw = terminal
        self.terminal = normalize_symbol(terminal)
        self.helper_index = helper_index
        self.source_root = source_root
        self.logged_signals = {normalize_symbol(s) for s in logged_signals if s}
        self.parameter_names = {p for p in parameter_names if p}
        self.snippet_context_lines = snippet_context_lines

        self.vertices: dict[str, DAGVertex] = {}
        self.edges: dict[tuple[str, str, str, str], DAGEdge] = {}
        self.unresolved_symbols: set[str] = set()

        # Memoization tables.
        self._evidence_by_signal: dict[tuple[str, str], str] = {}
        self._branch_by_predicate: dict[str, str] = {}
        self._helper_subgraph_return_id: dict[tuple[str, str], str] = {}
        # Assignment-target index: normalized target symbol → list of vertex ids
        # that produce it. Used to link consumers back to producing operations.
        self._producers_by_symbol: dict[str, list[str]] = {}

        # Snippet cache to avoid re-reading a file many times.
        self._file_lines_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------
    # Build
    # ------------------------------------------------------------

    def build(self) -> MechanismDAG:
        reaching = self.binding_index.bindings_reaching(self.terminal)
        reaching = self._expand_struct_roots(reaching)

        # Two-pass build. Pass 1 emits every operation vertex so
        # ``_producers_by_symbol`` is fully populated before any edge is
        # wired. Otherwise consumers processed before their producers
        # fall through to opaque-symbol leaves.
        for binding in reaching:
            self._emit_operation_vertex(binding, is_terminal=self._binding_reaches_terminal(binding))
        # Pass 1b: pre-emit every helper subgraph referenced anywhere so
        # helper-body intermediates land in the producer index before
        # edges are wired.
        for binding in reaching:
            expression = str(binding.get("source_symbol") or "")
            for helper_name in self._find_helper_calls(expression):
                self._materialize_helper_subgraph(helper_name, wire_edges=False)

        # Pass 2: wire edges now that every producer is known.
        for binding in reaching:
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
    # Struct-root expansion
    # ------------------------------------------------------------

    def _expand_struct_roots(self, reaching: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add field-level bindings for bare struct roots referenced but
        not directly assigned.

        A backward walk that lands on ``get_absolute_altitude_for_item(_mission_item)``
        stops at ``_mission_item`` because nothing writes to the bare
        struct. This pass fetches every ``_mission_item.*`` field write
        via :meth:`BindingIndex.bindings_writing_prefix` and continues
        the reach walk from each field. Fixpoint over any struct roots
        the newly-added bindings introduce.
        """
        seen_ids = {id(binding) for binding in reaching}
        result = list(reaching)
        pending_roots: list[str] = []

        def scan(binding: dict[str, Any]) -> None:
            for name in source_expression_names(str(binding.get("source_symbol") or "")):
                normalized = normalize_symbol(name)
                if not normalized or "." in normalized:
                    continue  # only bare roots — dotted names go through the normal walk
                if normalized == self.terminal:
                    continue
                if normalized in self.logged_signals or self._match_parameter(name):
                    continue
                pending_roots.append(normalized)

        for binding in reaching:
            scan(binding)

        seen_roots: set[str] = set()
        while pending_roots:
            root = pending_roots.pop()
            if root in seen_roots:
                continue
            seen_roots.add(root)
            for field_binding in self.binding_index.bindings_writing_prefix(root):
                if id(field_binding) in seen_ids:
                    continue
                seen_ids.add(id(field_binding))
                result.append(field_binding)
                # Reaching further backward: walk the newly-added binding's
                # own dependencies.
                for reached in self.binding_index.bindings_reaching(
                    str(field_binding.get("target_symbol") or "")
                ):
                    if id(reached) in seen_ids:
                        continue
                    seen_ids.add(id(reached))
                    result.append(reached)
                    scan(reached)
                scan(field_binding)

        return result

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
        for symbol in dedupe_keep_order(source_expression_names(expression)):
            normalized = normalize_symbol(symbol)
            if not normalized or normalized == target_norm:
                continue
            producer_id = self._resolve_symbol_producer(normalized, symbol, expression, file, line)
            if producer_id is not None:
                self._add_edge(producer_id, op_id, kind="data", role=symbol)

        # Helper-call inputs.
        for helper_call in self._find_helper_calls(expression):
            helper_return_id = self._helper_subgraph_return_id.get(self._pick_helper_key(helper_call) or ("", ""))
            if helper_return_id:
                self._add_edge(
                    helper_return_id,
                    op_id,
                    kind="data",
                    role=f"call:{helper_call}",
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
        emit an evidence leaf classified by whether the symbol is logged,
        a parameter, an enum constant, or an unresolved opaque reference.

        ``symbol_raw`` preserves the pre-normalization form for display on
        the evidence vertex.
        """
        producers = self._producers_by_symbol.get(symbol_norm)
        if producers:
            return producers[-1]

        # Evidence leaf classification. Uses the raw form for display.
        if symbol_norm in self.logged_signals:
            return self._emit_evidence("logged_signal", symbol_raw, file=None, line=None)

        parameter_alias = self._match_parameter(symbol_raw)
        if parameter_alias is not None:
            return self._emit_evidence("parameter", parameter_alias, file=None, line=None)

        if looks_like_enum_constant(symbol_raw):
            return self._emit_evidence("constant", symbol_raw, file=None, line=None)

        # Unclassified. Emit as opaque symbol so downstream tools can see
        # the reference but know it wasn't grounded.
        self.unresolved_symbols.add(symbol_raw)
        return self._emit_evidence("opaque_symbol", symbol_raw, file=file, line=line)

    def _emit_evidence(
        self,
        sub_kind: str,
        signal: str,
        *,
        file: Optional[str],
        line: Optional[int],
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
        self.vertices[vertex_id] = DAGVertex(
            id=vertex_id,
            kind="branch",
            file=file,
            line=line,
            snippet=self._snippet(file, line),
            predicate_raw=predicate,
            predicate_lowered=canonical,
            feasibility_verdict="unknown",
        )
        self._branch_by_predicate[canonical] = vertex_id

        # A branch's own predicate depends on the symbols it reads. Wire
        # data edges from those producers so pre-evaluation later has
        # every input in the graph.
        for symbol in dedupe_keep_order(source_expression_names(predicate)):
            normalized = normalize_symbol(symbol)
            if not normalized:
                continue
            producer_id = self._resolve_symbol_producer(normalized, symbol, predicate, file, line)
            if producer_id is not None:
                self._add_edge(producer_id, vertex_id, kind="data", role=symbol)

        return vertex_id

    # ------------------------------------------------------------
    # Helper subgraph nesting
    # ------------------------------------------------------------

    def _find_helper_calls(self, expression: str) -> list[str]:
        """Return helper names invoked in ``expression`` that we can expand.

        Only names that appear in ``helper_index`` under some class
        context — otherwise there's no body to inline.
        """
        matches: list[str] = []
        for candidate in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression):
            for name, _class_ctx in self.helper_index:
                if name == candidate:
                    matches.append(candidate)
                    break
        return dedupe_keep_order(matches)

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

        for var, expression in (helper.get("assignments") or {}).items():
            var_norm = normalize_symbol(var)
            op_id = self._make_id("op", (helper_key[0], helper_key[1], var_norm, expression))
            for symbol in dedupe_keep_order(source_expression_names(str(expression))):
                normalized = normalize_symbol(symbol)
                if not normalized or normalized == var_norm:
                    continue
                producer_id = self._resolve_symbol_producer(normalized, symbol, str(expression), file, line)
                if producer_id is not None:
                    self._add_edge(producer_id, op_id, kind="data", role=symbol)

        for branch in helper.get("branches") or []:
            condition = str(branch.get("condition") or "").strip()
            if condition:
                self._emit_branch(condition, file=file, line=line)

        return_expression = (
            helper.get("lowered_return_expression")
            or helper.get("return_expression")
        )
        if not return_expression:
            return
        terminal_id = self._make_id(
            "op",
            (helper_key[0], helper_key[1], "__return__", return_expression),
        )
        for symbol in dedupe_keep_order(source_expression_names(str(return_expression))):
            normalized = normalize_symbol(symbol)
            if not normalized:
                continue
            producer_id = self._resolve_symbol_producer(normalized, symbol, str(return_expression), file, line)
            if producer_id is not None:
                self._add_edge(producer_id, terminal_id, kind="data", role=symbol)

    def _pick_helper_key(self, helper_name: str) -> Optional[tuple[str, str]]:
        """Choose one ``(name, class_context)`` for a call like ``foo(...)``.

        Milestone 1: if multiple class contexts define ``foo``, pick the
        first deterministically. Multi-context disambiguation via the
        caller's class scope is Milestone 2 work.
        """
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
        ``.get()`` / ``.get`` accessor before matching.
        """
        if not self.parameter_names:
            return None
        raw = symbol.replace(".get()", "").replace(".get", "")
        upper = raw.upper()
        if upper in self.parameter_names:
            return upper
        # PX4-style: `_param_rtl_return_alt` or `param_rtl_return_alt` → `RTL_RETURN_ALT`.
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

        if verdict == "unknown" and samples:
            evaluated = _evaluate_predicate_intervals(predicate, params, enums, samples)
            if evaluated is not None:
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
