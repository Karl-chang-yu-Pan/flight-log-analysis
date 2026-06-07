from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, Field

from flight_log_agent.models import CodeRef, MechanismCandidate, RelationshipCheckSpec


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


CHECK_SIGNAL_FIELDS = ("signal", "first", "second", "actual", "setpoint")
NON_SYMBOL_NAMES = {
    "True",
    "False",
    "abs",
    "constrain",
    "cos",
    "fabs",
    "fmax",
    "fmin",
    "isfinite",
    "max",
    "min",
    "round",
    "sin",
    "sqrt",
    "tan",
}


def compile_verification_graphs(
    candidate: MechanismCandidate,
    output_bindings: Iterable[Any] = (),
) -> list[VerificationGraph]:
    bindings = [_binding_dict(binding) for binding in output_bindings]
    return [
        compile_verification_graph(candidate, terminal, bindings)
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
) -> VerificationGraph:
    terminal = normalize_symbol(terminal_output)
    bindings = [_binding_dict(binding) for binding in output_bindings]
    relevant_bindings = backward_binding_slice(terminal, bindings)
    reachable_symbols = binding_slice_symbols(terminal, relevant_bindings)
    owned_checks = [
        (branch_name, check)
        for branch_name, check in candidate_checks(candidate)
        if check_owned_by_terminal(check, terminal, reachable_symbols)
    ]

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

    binding_operations: dict[tuple[str, str, str], str] = {}
    operations_by_target: dict[str, list[str]] = defaultdict(list)
    for binding in relevant_bindings:
        source_symbol = normalize_symbol(str(binding.get("source_symbol") or ""))
        target_symbol = normalize_symbol(str(binding.get("target_symbol") or ""))
        operation_id = add_node(
            nodes,
            kind="operation",
            label=f"{target_symbol} = {source_symbol}",
            symbol=target_symbol,
            binding_id=str(binding.get("binding_id") or "") or None,
            source_refs=assignment_path_source_refs(binding.get("assignment_path") or []),
            metadata={
                "operation": "source_assignment",
                "source_symbol": source_symbol,
                "target_symbol": target_symbol,
            },
        )
        binding_operations[binding_key(binding)] = operation_id
        operations_by_target[target_symbol].append(operation_id)
        if normalize_symbol(str(binding.get("logged_signal") or "")) == terminal:
            add_edge(edges, operation_id, output_id, "data")

    for binding in relevant_bindings:
        source_symbol = normalize_symbol(str(binding.get("source_symbol") or ""))
        operation_id = binding_operations[binding_key(binding)]
        for dependency in expression_symbols(source_symbol):
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
                logged_signal=dependency if looks_like_logged_signal(dependency) else None,
                metadata={"evidence_kind": "unresolved_input"},
            )
            add_edge(edges, evidence_id, operation_id, "data")

    for branch_name, check in owned_checks:
        check_id = add_node(
            nodes,
            kind="operation",
            label=check.description or check.supports or check.type,
            check_type=check.type,
            metadata={
                "operation": "verification_check",
                "check": _model_dump(check),
                "branch_name": branch_name,
            },
        )
        add_edge(edges, check_id, comparison_id, "comparison")
        for dependency in check_input_symbols(check, terminal):
            evidence_id = add_node(
                nodes,
                kind="evidence",
                label=f"check dependency: {dependency}",
                symbol=dependency,
                logged_signal=dependency if looks_like_logged_signal(dependency) else None,
                metadata={"evidence_kind": "input"},
            )
            add_edge(edges, evidence_id, check_id, "data")
        if branch_name:
            branch_id = add_node(
                nodes,
                kind="branch",
                label=f"branch: {branch_name}",
                symbol=branch_name,
            )
            add_edge(edges, branch_id, check_id, "control")

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
        source = normalize_symbol(str(binding.get("source_symbol") or ""))
        for symbol in expression_symbols(source):
            frontier.extend(by_target.get(symbol, []))
    return selected


def binding_slice_symbols(terminal_output: str, bindings: list[dict[str, Any]]) -> set[str]:
    symbols = {normalize_symbol(terminal_output)}
    for binding in bindings:
        symbols.add(normalize_symbol(str(binding.get("logged_signal") or "")))
        symbols.add(normalize_symbol(str(binding.get("target_symbol") or "")))
        symbols.update(expression_symbols(str(binding.get("source_symbol") or "")))
    return {symbol for symbol in symbols if symbol}


def check_owned_by_terminal(
    check: RelationshipCheckSpec,
    terminal_output: str,
    reachable_symbols: set[str],
) -> bool:
    primary_outputs = check_primary_outputs(check)
    if terminal_output not in primary_outputs:
        return False
    dependencies = set(check_input_symbols(check, terminal_output))
    return all(
        dependency in reachable_symbols
        or not looks_like_source_symbol(dependency)
        or dependency.isupper()
        for dependency in dependencies
    )


def check_primary_outputs(check: RelationshipCheckSpec) -> list[str]:
    outputs: list[str] = []
    actual = normalize_symbol(str(check.actual or ""))
    if actual and looks_like_logged_signal(actual):
        outputs.append(actual)
    for variable in check.variables:
        name = str(_get(variable, "name") or "")
        source = normalize_symbol(str(_get(variable, "source") or ""))
        if name == "actual" and looks_like_logged_signal(source):
            outputs.append(source)
    if not outputs and check.type not in {"topic_field_present", "parameter_equals", "branch_parameter_satisfied"}:
        signal = normalize_symbol(str(check.signal or ""))
        if signal and looks_like_logged_signal(signal):
            outputs.append(signal)
    return dedupe(outputs)


def check_input_symbols(check: RelationshipCheckSpec, terminal_output: str) -> list[str]:
    primary = set(check_primary_outputs(check))
    inputs = []
    for field in CHECK_SIGNAL_FIELDS:
        value = normalize_symbol(str(getattr(check, field, None) or ""))
        if value and value not in primary:
            inputs.append(value)
    for variable in check.variables:
        source = normalize_symbol(str(_get(variable, "source") or ""))
        if source and source not in primary:
            inputs.append(source)
    for expression in (check.expression, check.expected_expression):
        inputs.extend(expression_symbols(str(expression or "")))
    return dedupe(item for item in inputs if item != terminal_output)


def mechanism_defining_checks(candidate: MechanismCandidate) -> list[RelationshipCheckSpec]:
    return [
        check
        for _, check in candidate_checks(candidate)
        if check.type not in {"topic_field_present", "parameter_equals", "branch_parameter_satisfied"}
    ]


def candidate_checks(candidate: MechanismCandidate) -> list[tuple[Optional[str], RelationshipCheckSpec]]:
    checks = [(None, check) for check in [*candidate.numeric_checks, *candidate.exclusion_checks]]
    for group in candidate.branch_groups:
        checks.extend((group.name, check) for check in [*group.numeric_checks, *group.exclusion_checks])
    return checks


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


def looks_like_logged_signal(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", value))


def looks_like_source_symbol(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", value))


def binding_key(binding: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_symbol(str(binding.get("source_symbol") or "")),
        normalize_symbol(str(binding.get("target_symbol") or "")),
        normalize_symbol(str(binding.get("logged_signal") or "")),
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
