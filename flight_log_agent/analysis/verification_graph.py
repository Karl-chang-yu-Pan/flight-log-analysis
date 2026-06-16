from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, Field

from flight_log_agent.analysis.control_predicate import lower_control_predicates
from flight_log_agent.analysis.source_expression import source_expression_names
from flight_log_agent.models import CodeRef, MechanismCandidate


class VerificationGraphNode(BaseModel):
    node_id: str
    kind: Literal["evidence", "operation", "branch", "output", "comparison"]
    label: str
    symbol: Optional[str] = None
    logged_signal: Optional[str] = None
    binding_id: Optional[str] = None
    check_type: Optional[str] = None
    source_refs: list[CodeRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationGraphEdge(BaseModel):
    source_id: str
    target_id: str
    kind: Literal["data", "control", "comparison"]


class VerificationGraph(BaseModel):
    graph_id: str
    candidate_name: str
    terminal_output: str
    nodes: list[VerificationGraphNode] = Field(default_factory=list)
    edges: list[VerificationGraphEdge] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


NON_SYMBOL_NAMES = {
    "True",
    "False",
    "F",
    "and",
    "abs",
    "constrain",
    "cos",
    "fabs",
    "fmax",
    "fmin",
    "isfinite",
    "max",
    "min",
    "not",
    "or",
    "round",
    "sin",
    "sqrt",
    "tan",
    "f",
}


def compile_verification_graphs(
    candidate: MechanismCandidate,
    output_bindings: Iterable[Any] = (),
    *,
    source_path: Optional[str] = None,
) -> list[VerificationGraph]:
    bindings = [_binding_dict(binding) for binding in output_bindings]
    return [
        compile_verification_graph(candidate, terminal, bindings, source_path=source_path)
        for terminal in terminal_outputs(candidate, bindings)
    ]


def terminal_outputs(
    candidate: MechanismCandidate,
    output_bindings: Iterable[Any] = (),
) -> list[str]:
    bound_outputs = {
        normalize_symbol(str(_get(binding, "logged_signal") or ""))
        for binding in output_bindings
        if _get(binding, "logged_signal")
    }
    primary_outputs = dedupe(
        normalize_symbol(signal)
        for signal in candidate.primary_output_signals
        if signal
    )
    return [signal for signal in primary_outputs if not bound_outputs or signal in bound_outputs]


def compile_verification_graph(
    candidate: MechanismCandidate,
    terminal_output: str,
    output_bindings: Iterable[Any] = (),
    *,
    source_path: Optional[str] = None,
) -> VerificationGraph:
    terminal = normalize_symbol(terminal_output)
    bindings = [_binding_dict(binding) for binding in output_bindings]
    signal_bindings = source_signal_bindings(bindings)
    known_logged_signals = {
        normalize_symbol(str(binding.get("logged_signal") or ""))
        for binding in bindings
        if binding.get("logged_signal")
    }
    relevant_bindings = backward_binding_slice(terminal, bindings)
    nodes: dict[str, VerificationGraphNode] = {}
    edges: dict[tuple[str, str, str], VerificationGraphEdge] = {}
    unresolved: list[str] = []

    actual_id = add_node(
        nodes,
        kind="evidence",
        label=f"actual logged output: {terminal}",
        symbol=terminal,
        logged_signal=terminal,
        metadata={"evidence_kind": "observed"},
    )
    output_id = add_node(
        nodes,
        kind="output",
        label=f"expected source output: {terminal}",
        symbol=terminal,
        logged_signal=terminal,
        metadata={"actual_input_id": actual_id, "producer_controls": {}},
    )
    comparison_id = add_node(
        nodes,
        kind="comparison",
        label=f"compare expected and actual: {terminal}",
        symbol=terminal,
        logged_signal=terminal,
    )
    add_edge(edges, output_id, comparison_id, "comparison")
    add_edge(edges, actual_id, comparison_id, "comparison")

    if not relevant_bindings:
        unresolved.append(f"No primary source binding reaches terminal output {terminal}.")

    binding_operations: dict[tuple[str, ...], str] = {}
    operations_by_target: dict[str, list[str]] = defaultdict(list)
    for binding in relevant_bindings:
        source_expression = str(binding.get("source_symbol") or "").strip()
        source_symbol = normalize_symbol(source_expression)
        target_symbol = normalize_symbol(str(binding.get("target_symbol") or ""))
        operation_id = add_node(
            nodes,
            kind="operation",
            label=f"{target_symbol} = {source_expression}",
            symbol=target_symbol,
            binding_id=str(binding.get("binding_id") or "") or None,
            source_refs=assignment_path_source_refs(binding.get("assignment_path") or []),
            metadata={
                "operation": "source_assignment",
                "source_expression": source_expression,
                "source_symbol": source_symbol,
                "target_symbol": target_symbol,
            },
        )
        binding_operations[binding_key(binding)] = operation_id
        operations_by_target[target_symbol].append(operation_id)
        if normalize_symbol(str(binding.get("logged_signal") or "")) == terminal:
            add_edge(edges, operation_id, output_id, "data")
            predicates = [str(item) for item in binding.get("control_predicates") or [] if item]
            control_id = None
            if predicates:
                lowered_predicates = lower_control_predicates(
                    predicates,
                    signal_bindings,
                    source_path=source_path,
                )
                control_id = add_node(
                    nodes,
                    kind="branch",
                    label=f"producer control: {target_symbol} = {source_expression}",
                    symbol=str(binding.get("binding_id") or "") or target_symbol,
                    metadata={
                        "source_predicates": predicates,
                        "lowered_predicates": [item.model_dump() for item in lowered_predicates],
                        "producer_id": operation_id,
                    },
                )
                add_edge(edges, control_id, output_id, "control")
                lowered_dependencies = []
                lowered_dependency_signals: dict[str, str] = {}
                for lowered in lowered_predicates:
                    if lowered.status != "log_verifiable" or not lowered.expression:
                        continue
                    lowered_dependencies.extend(expression_symbols(lowered.expression))
                    lowered_dependency_signals.update(lowered.variables)
                for dependency in dedupe(lowered_dependencies):
                    evidence_id = add_node(
                        nodes,
                        kind="evidence",
                        label=f"producer control dependency: {dependency}",
                        symbol=dependency,
                        logged_signal=lowered_dependency_signals.get(dependency),
                        metadata={"evidence_kind": "producer_control"},
                    )
                    add_edge(edges, evidence_id, control_id, "data")
            nodes[output_id].metadata["producer_controls"][operation_id] = control_id
    add_edge(edges, actual_id, output_id, "control")

    for binding in relevant_bindings:
        source_expression = str(binding.get("source_symbol") or "").strip()
        operation_id = binding_operations[binding_key(binding)]
        for dependency in expression_symbols(source_expression):
            producers = operations_by_target.get(dependency, [])
            if producers:
                for producer_id in producers:
                    add_edge(edges, producer_id, operation_id, "data")
                continue
            evidence_id = add_node(
                nodes,
                kind="evidence",
                label=f"source dependency: {dependency}",
                symbol=dependency,
                logged_signal=(
                    signal_bindings.get(dependency)
                    or (dependency if dependency in known_logged_signals else None)
                ),
                metadata={"evidence_kind": "unresolved_input"},
            )
            add_edge(edges, evidence_id, operation_id, "data")

    graph = VerificationGraph(
        graph_id=stable_id("graph", {"candidate": candidate.name, "terminal": terminal}),
        candidate_name=candidate.name,
        terminal_output=terminal,
        nodes=list(nodes.values()),
        edges=list(edges.values()),
        unresolved_dependencies=dedupe(unresolved),
    )
    graph.validation_errors = validate_verification_graph(graph)
    return graph


def backward_binding_slice(terminal_output: str, bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal = normalize_symbol(terminal_output)
    by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        logged_signal = normalize_symbol(str(binding.get("logged_signal") or ""))
        target_symbol = normalize_symbol(str(binding.get("target_symbol") or ""))
        if logged_signal:
            by_output[logged_signal].append(binding)
        if target_symbol:
            by_target[target_symbol].append(binding)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    frontier = deque(by_output.get(terminal, []))
    while frontier:
        binding = frontier.popleft()
        key = binding_key(binding)
        if key in seen:
            continue
        seen.add(key)
        selected.append(binding)
        source_expression = str(binding.get("source_symbol") or "")
        for symbol in expression_symbols(source_expression):
            frontier.extend(by_target.get(symbol, []))
    return selected


def source_signal_bindings(bindings: list[dict[str, Any]]) -> dict[str, str]:
    signal_bindings: dict[str, str] = {}
    for binding in bindings:
        for source, signal in dict(binding.get("symbol_bindings") or {}).items():
            raw_source_symbol = str(source or "").strip()
            source_symbol = normalize_symbol(raw_source_symbol)
            logged = normalize_symbol(str(signal or ""))
            if raw_source_symbol and logged:
                signal_bindings[raw_source_symbol] = logged
            if source_symbol and logged:
                signal_bindings[source_symbol] = logged
        logged_signal = normalize_symbol(str(binding.get("logged_signal") or ""))
        if not logged_signal:
            continue
        for symbol in (
            str(binding.get("target_symbol") or ""),
            normalize_symbol(str(binding.get("target_symbol") or "")),
        ):
            if symbol:
                signal_bindings[symbol] = logged_signal
    return signal_bindings


def validate_verification_graph(graph: VerificationGraph) -> list[str]:
    errors: list[str] = []
    node_ids = {node.node_id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            errors.append(f"Edge references missing node: {edge.source_id} -> {edge.target_id}.")

    outputs = [node for node in graph.nodes if node.kind == "output" and node.logged_signal == graph.terminal_output]
    comparisons = [node for node in graph.nodes if node.kind == "comparison" and node.logged_signal == graph.terminal_output]
    if len(outputs) != 1:
        errors.append(f"Expected exactly one terminal output node, found {len(outputs)}.")
    if len(comparisons) != 1:
        errors.append(f"Expected exactly one terminal comparison node, found {len(comparisons)}.")

    if comparisons:
        reachable = reverse_reachable_node_ids(comparisons[0].node_id, graph.edges)
        unrelated = sorted(node_ids - reachable)
        if unrelated:
            errors.append(f"Graph contains nodes unrelated to the terminal comparison: {unrelated}.")
    return dedupe(errors)


def reverse_reachable_node_ids(target_id: str, edges: list[VerificationGraphEdge]) -> set[str]:
    by_target: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        by_target[edge.target_id].append(edge.source_id)
    reachable = {target_id}
    frontier = [target_id]
    while frontier:
        current = frontier.pop()
        for source in by_target.get(current, []):
            if source not in reachable:
                reachable.add(source)
                frontier.append(source)
    return reachable


def expression_symbols(expression: str) -> list[str]:
    parsed_names = source_expression_names(expression)
    if parsed_names:
        return dedupe(normalize_symbol(name) for name in parsed_names)
    symbols = []
    for token in re.findall(
        r"\b_?[A-Za-z][A-Za-z0-9_]*(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)*\b",
        expression,
    ):
        normalized = normalize_symbol(token)
        if normalized and normalized not in NON_SYMBOL_NAMES and not normalized.isdigit():
            symbols.append(normalized)
    return dedupe(symbols)


def assignment_path_source_refs(path: list[dict[str, Any]]) -> list[CodeRef]:
    refs = []
    for step in path:
        file = step.get("file")
        if not file:
            continue
        line = step.get("line")
        refs.append(
            CodeRef(
                file=str(file),
                function=step.get("function"),
                start_line=line,
                end_line=line,
                snippet=step.get("evidence"),
                explanation="Primary source assignment path.",
            )
        )
    return refs


def add_node(
    nodes: dict[str, VerificationGraphNode],
    *,
    kind: str,
    label: str,
    symbol: Optional[str] = None,
    logged_signal: Optional[str] = None,
    binding_id: Optional[str] = None,
    check_type: Optional[str] = None,
    source_refs: Optional[list[CodeRef]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    identity = {
        "kind": kind,
        "symbol": symbol,
        "logged_signal": logged_signal,
        "binding_id": binding_id,
        "check_type": check_type,
        "label": label,
        "metadata": metadata or {},
    }
    node_id = stable_id("node", identity)
    nodes.setdefault(
        node_id,
        VerificationGraphNode(
            node_id=node_id,
            kind=kind,
            label=label,
            symbol=symbol,
            logged_signal=logged_signal,
            binding_id=binding_id,
            check_type=check_type,
            source_refs=source_refs or [],
            metadata=metadata or {},
        ),
    )
    return node_id


def add_edge(
    edges: dict[tuple[str, str, str], VerificationGraphEdge],
    source_id: str,
    target_id: str,
    kind: str,
) -> None:
    key = (source_id, target_id, kind)
    edges.setdefault(key, VerificationGraphEdge(source_id=source_id, target_id=target_id, kind=kind))


def normalize_symbol(value: str) -> str:
    normalized = str(value or "").strip().replace("->", ".").replace("::", ".").replace(" ", "").strip("&*")
    normalized = re.sub(r"\[[^\]]+\]", "", normalized)
    if normalized.startswith("_"):
        normalized = normalized[1:]
    return normalized


def binding_key(binding: dict[str, Any]) -> tuple[str, ...]:
    return (
        normalize_symbol(str(binding.get("source_symbol") or "")),
        normalize_symbol(str(binding.get("target_symbol") or "")),
        normalize_symbol(str(binding.get("logged_signal") or "")),
        str(binding.get("binding_id") or ""),
        *[str(item) for item in binding.get("control_predicates") or []],
    )


def stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _binding_dict(binding: Any) -> dict[str, Any]:
    return _model_dump(binding)


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value) if isinstance(value, dict) else dict(vars(value))


def _get(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)
