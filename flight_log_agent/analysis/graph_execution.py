from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from flight_log_agent.analysis.log_evidence import EvidenceSample, ULogEvidenceIndex
from flight_log_agent.analysis.verification_graph import (
    VerificationGraph,
    VerificationGraphNode,
    expression_symbols,
    normalize_symbol,
)


class GraphNodeExecution(BaseModel):
    node_id: str
    status: Literal[
        "observed",
        "reconstructed",
        "supported",
        "contradicted",
        "ambiguous",
        "unavailable",
        "pending",
    ]
    value: Any = None
    samples: list[EvidenceSample] = Field(default_factory=list)
    reason: Optional[str] = None
    provenance: list[str] = Field(default_factory=list)


class GraphExecutionResult(BaseModel):
    graph_id: str
    terminal_output: str
    verdict: Literal["supported", "contradicted", "unresolved"]
    node_results: list[GraphNodeExecution] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)


def execute_verification_graph(
    graph: VerificationGraph,
    evidence_index: ULogEvidenceIndex,
    *,
    tolerance: float = 1e-6,
) -> GraphExecutionResult:
    nodes = {node.node_id: node for node in graph.nodes}
    predecessors: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        predecessors[edge.target_id].append(edge.source_id)

    results: dict[str, GraphNodeExecution] = {}
    for node_id in topological_order(graph):
        node = nodes[node_id]
        inputs = [results[source_id] for source_id in predecessors.get(node_id, []) if source_id in results]
        results[node_id] = execute_node(node, inputs, evidence_index, tolerance=tolerance)

    comparison = next(
        (
            results[node.node_id]
            for node in graph.nodes
            if node.kind == "comparison" and node.logged_signal == graph.terminal_output
        ),
        None,
    )
    verdict = (
        "supported" if comparison and comparison.status == "supported"
        else "contradicted" if comparison and comparison.status == "contradicted"
        else "unresolved"
    )
    unresolved = [
        result.reason
        for result in results.values()
        if result.status in {"ambiguous", "unavailable", "pending"} and result.reason
    ]
    return GraphExecutionResult(
        graph_id=graph.graph_id,
        terminal_output=graph.terminal_output,
        verdict=verdict,
        node_results=list(results.values()),
        unresolved_dependencies=list(dict.fromkeys([*graph.unresolved_dependencies, *unresolved])),
    )


def execute_node(
    node: VerificationGraphNode,
    inputs: list[GraphNodeExecution],
    evidence_index: ULogEvidenceIndex,
    *,
    tolerance: float,
) -> GraphNodeExecution:
    if node.kind == "evidence":
        return execute_evidence_node(node, evidence_index)
    if node.kind == "operation":
        return execute_operation_node(node, inputs)
    if node.kind == "output":
        return propagate_single_input(node, inputs, "terminal output has multiple or unavailable producers")
    if node.kind == "comparison":
        return execute_comparison_node(node, inputs, tolerance=tolerance)
    return GraphNodeExecution(
        node_id=node.node_id,
        status="pending",
        reason=f"branch execution is not implemented for {node.label}",
    )


def execute_evidence_node(node: VerificationGraphNode, evidence_index: ULogEvidenceIndex) -> GraphNodeExecution:
    if node.logged_signal:
        resolution = evidence_index.resolve_signal(node.logged_signal)
        if resolution.status == "observed" and resolution.series is not None:
            return GraphNodeExecution(
                node_id=node.node_id,
                status="observed",
                samples=resolution.series.samples,
                provenance=[resolution.series.signal],
            )
        return GraphNodeExecution(
            node_id=node.node_id,
            status=resolution.status,
            reason=resolution.reason,
            provenance=resolution.candidates,
        )
    if node.symbol and node.symbol.isupper():
        parameter = evidence_index.resolve_parameter(node.symbol)
        return GraphNodeExecution(
            node_id=node.node_id,
            status=parameter.status,
            value=parameter.value,
            reason=parameter.reason,
            provenance=[node.symbol] if parameter.status == "observed" else [],
        )
    return GraphNodeExecution(
        node_id=node.node_id,
        status="unavailable",
        reason=f"No exact log or parameter binding for source dependency {node.symbol or node.label}.",
    )


def execute_operation_node(node: VerificationGraphNode, inputs: list[GraphNodeExecution]) -> GraphNodeExecution:
    operation = node.metadata.get("operation")
    if operation != "source_assignment":
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Operation execution is not implemented for {operation or node.label}.",
        )
    source_symbol = normalize_symbol(str(node.metadata.get("source_symbol") or ""))
    dependencies = expression_symbols(source_symbol)
    if len(dependencies) != 1 or dependencies[0] != source_symbol:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Source assignment requires expression execution: {node.label}.",
        )
    usable = [item for item in inputs if item.status in {"observed", "reconstructed"}]
    matching = [
        item
        for item in usable
        if source_symbol in item.provenance
        or len(usable) == 1
    ]
    if len(matching) == 1:
        source = matching[0]
        return GraphNodeExecution(
            node_id=node.node_id,
            status="reconstructed",
            value=source.value,
            samples=source.samples,
            provenance=list(dict.fromkeys([*source.provenance, source_symbol])),
        )
    if len(matching) > 1:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="ambiguous",
            reason=f"Source assignment {node.label} has multiple executable producers.",
        )
    failed = [item for item in inputs if item.reason]
    return GraphNodeExecution(
        node_id=node.node_id,
        status="pending",
        reason=failed[0].reason if failed else f"Source assignment requires expression execution: {node.label}.",
    )


def propagate_single_input(
    node: VerificationGraphNode,
    inputs: list[GraphNodeExecution],
    failure_reason: str,
) -> GraphNodeExecution:
    usable = [item for item in inputs if item.status in {"observed", "reconstructed"}]
    if len(usable) == 1:
        source = usable[0]
        return GraphNodeExecution(
            node_id=node.node_id,
            status="reconstructed",
            value=source.value,
            samples=source.samples,
            provenance=source.provenance,
        )
    return GraphNodeExecution(
        node_id=node.node_id,
        status="ambiguous" if len(usable) > 1 else "pending",
        reason=failure_reason,
    )


def execute_comparison_node(
    node: VerificationGraphNode,
    inputs: list[GraphNodeExecution],
    *,
    tolerance: float,
) -> GraphNodeExecution:
    expected = next((item for item in inputs if item.status == "reconstructed"), None)
    actual = next((item for item in inputs if item.status == "observed"), None)
    if expected is None or actual is None:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Terminal comparison lacks reconstructed expected or observed actual values for {node.logged_signal}.",
        )
    errors = aligned_errors(expected.samples, actual.samples)
    if not errors:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Terminal comparison has no aligned samples for {node.logged_signal}.",
        )
    max_error = max(errors)
    return GraphNodeExecution(
        node_id=node.node_id,
        status="supported" if max_error <= tolerance else "contradicted",
        value={"sample_count": len(errors), "max_error": max_error, "tolerance": tolerance},
        provenance=list(dict.fromkeys([*expected.provenance, *actual.provenance])),
    )


def aligned_errors(expected: list[EvidenceSample], actual: list[EvidenceSample]) -> list[float]:
    if not expected or not actual:
        return []
    expected_by_time = {sample.time_s: sample.value for sample in expected}
    errors = []
    for sample in actual:
        if sample.time_s not in expected_by_time:
            continue
        expected_value = numeric_value(expected_by_time[sample.time_s])
        actual_value = numeric_value(sample.value)
        if expected_value is not None and actual_value is not None:
            errors.append(abs(expected_value - actual_value))
    return errors


def topological_order(graph: VerificationGraph) -> list[str]:
    node_ids = [node.node_id for node in graph.nodes]
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.source_id in indegree and edge.target_id in indegree:
            indegree[edge.target_id] += 1
            outgoing[edge.source_id].append(edge.target_id)
    frontier = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    ordered = []
    while frontier:
        node_id = frontier.popleft()
        ordered.append(node_id)
        for target_id in outgoing.get(node_id, []):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                frontier.append(target_id)
    return ordered if len(ordered) == len(node_ids) else node_ids


def numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None
