from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from flight_log_agent.analysis.log_evidence import EvidenceSample, ULogEvidenceIndex
from flight_log_agent.analysis.source_expression import (
    SourceExpressionError,
    evaluate_source_expression,
)
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
        "applicable",
        "excluded",
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
        return execute_output_node(node, inputs)
    if node.kind == "comparison":
        return execute_comparison_node(node, inputs, tolerance=tolerance)
    return execute_branch_node(node, inputs)


def execute_evidence_node(node: VerificationGraphNode, evidence_index: ULogEvidenceIndex) -> GraphNodeExecution:
    if node.logged_signal:
        resolution = evidence_index.resolve_signal(node.logged_signal)
        if resolution.status == "observed" and resolution.series is not None:
            return GraphNodeExecution(
                node_id=node.node_id,
                status="observed",
                samples=resolution.series.samples,
                provenance=list(dict.fromkeys([resolution.series.signal, node.symbol or ""])),
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
    source_expression = str(
        node.metadata.get("source_expression")
        or node.metadata.get("source_symbol")
        or ""
    )
    source_symbol = normalize_symbol(str(node.metadata.get("source_symbol") or source_expression))
    dependencies = expression_symbols(source_expression)
    input_by_dependency = match_inputs_to_dependencies(dependencies, inputs)
    missing = [dependency for dependency in dependencies if dependency not in input_by_dependency]
    if missing:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Source assignment is missing executable dependencies {missing}: {node.label}.",
        )
    if len(dependencies) == 1 and dependencies[0] == source_symbol:
        source = input_by_dependency[source_symbol]
        return GraphNodeExecution(
            node_id=node.node_id,
            status="reconstructed",
            value=source.value,
            samples=source.samples,
            provenance=list(dict.fromkeys([*source.provenance, source_symbol])),
        )
    try:
        value, samples = evaluate_expression_inputs(source_expression, input_by_dependency)
    except SourceExpressionError as exc:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Source assignment expression is not executable: {node.label}: {exc}.",
        )
    return GraphNodeExecution(
        node_id=node.node_id,
        status="reconstructed",
        value=value,
        samples=samples,
        provenance=list(dict.fromkeys([
            *(item for result in input_by_dependency.values() for item in result.provenance),
            source_symbol,
        ])),
    )


def execute_branch_node(node: VerificationGraphNode, inputs: list[GraphNodeExecution]) -> GraphNodeExecution:
    lowered_predicates = [
        item
        for item in node.metadata.get("lowered_predicates") or []
        if isinstance(item, dict)
    ]
    if lowered_predicates:
        blocked = [
            item
            for item in lowered_predicates
            if item.get("status") != "log_verifiable" or not item.get("expression")
        ]
        if blocked:
            return GraphNodeExecution(
                node_id=node.node_id,
                status="pending",
                reason=(
                    f"Branch has non-log-verifiable control predicates: "
                    f"{[(item.get('raw'), item.get('status'), item.get('unresolved_symbols')) for item in blocked]}."
                ),
            )
        predicates = [str(item["expression"]) for item in lowered_predicates]
    else:
        predicates = [str(item) for item in node.metadata.get("source_predicates") or [] if item]
    if not predicates:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Branch has no executable predicates: {node.label}.",
        )
    expression = " and ".join(f"({predicate})" for predicate in predicates)
    dependencies = expression_symbols(expression)
    input_by_dependency = match_inputs_to_dependencies(dependencies, inputs)
    missing = [dependency for dependency in dependencies if dependency not in input_by_dependency]
    if missing:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Branch is missing executable dependencies {missing}: {node.label}.",
        )
    try:
        value, samples = evaluate_expression_inputs(expression, input_by_dependency)
    except SourceExpressionError as exc:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason=f"Branch predicate is not executable: {node.label}: {exc}.",
        )
    applicable = bool(value) if not samples else any(bool(sample.value) for sample in samples)
    return GraphNodeExecution(
        node_id=node.node_id,
        status="applicable" if applicable else "excluded",
        value=value,
        samples=samples,
        provenance=list(dict.fromkeys(
            item for result in input_by_dependency.values() for item in result.provenance
        )),
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


def execute_output_node(
    node: VerificationGraphNode,
    inputs: list[GraphNodeExecution],
) -> GraphNodeExecution:
    producer_controls = node.metadata.get("producer_controls") or {}
    producers = {
        result.node_id: result
        for result in inputs
        if result.node_id in producer_controls
    }
    if len(producers) == 1 and not next(iter(producer_controls.values()), None):
        return propagate_single_input(node, list(producers.values()), "terminal output producer is unavailable")
    if not producers:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason="terminal output has no executable producers",
        )

    uncontrolled = [
        producer_id
        for producer_id, control_id in producer_controls.items()
        if producer_id in producers and not control_id
    ]
    if uncontrolled:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="ambiguous",
            reason=f"terminal output has competing producers without control predicates: {uncontrolled}",
        )

    by_id = {result.node_id: result for result in inputs}
    actual = by_id.get(str(node.metadata.get("actual_input_id") or ""))
    timeline = (
        [sample.time_s for sample in actual.samples]
        if actual is not None and actual.samples
        else sorted({
            sample.time_s
            for result in producers.values()
            for sample in result.samples
        })
    )
    if not timeline:
        return GraphNodeExecution(
            node_id=node.node_id,
            status="pending",
            reason="terminal producer selection has no timestamped evidence",
        )

    selected_samples: list[EvidenceSample] = []
    selected_provenance: list[str] = []
    for time_s in timeline:
        active: list[GraphNodeExecution] = []
        for producer_id, producer in producers.items():
            control = by_id.get(str(producer_controls.get(producer_id) or ""))
            if control is None or branch_value_at(control, time_s) is not True:
                continue
            if execution_value_at(producer, time_s) is not None:
                active.append(producer)
        if len(active) != 1:
            return GraphNodeExecution(
                node_id=node.node_id,
                status="ambiguous",
                reason=f"terminal producer selection found {len(active)} active producers at {time_s:.6f}s",
            )
        selected = active[0]
        selected_samples.append(EvidenceSample(time_s=time_s, value=execution_value_at(selected, time_s)))
        selected_provenance.extend(selected.provenance)

    return GraphNodeExecution(
        node_id=node.node_id,
        status="reconstructed",
        samples=selected_samples,
        provenance=list(dict.fromkeys(selected_provenance)),
    )


def branch_value_at(result: GraphNodeExecution, time_s: float) -> Optional[bool]:
    if result.status not in {"applicable", "excluded"}:
        return None
    if not result.samples:
        return result.status == "applicable"
    sample = prior_sample(result.samples, [item.time_s for item in result.samples], time_s)
    return bool(sample.value) if sample is not None else None


def execution_value_at(result: GraphNodeExecution, time_s: float) -> Any:
    if result.status != "reconstructed":
        return None
    if not result.samples:
        return result.value
    sample = prior_sample(result.samples, [item.time_s for item in result.samples], time_s)
    return sample.value if sample is not None else None


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
    errors = aligned_errors(expected.samples, actual.samples, expected_value=expected.value)
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


def aligned_errors(
    expected: list[EvidenceSample],
    actual: list[EvidenceSample],
    *,
    expected_value: Any = None,
) -> list[float]:
    if not actual:
        return []
    scalar_expected = numeric_value(expected_value)
    if not expected:
        if scalar_expected is None:
            return []
        return [
            abs(scalar_expected - actual_value)
            for sample in actual
            if (actual_value := numeric_value(sample.value)) is not None
        ]
    expected_times = [sample.time_s for sample in expected]
    errors = []
    for sample in actual:
        expected_sample = prior_sample(expected, expected_times, sample.time_s)
        if expected_sample is None:
            continue
        expected_value = numeric_value(expected_sample.value)
        actual_value = numeric_value(sample.value)
        if expected_value is not None and actual_value is not None:
            errors.append(abs(expected_value - actual_value))
    return errors


def match_inputs_to_dependencies(
    dependencies: list[str],
    inputs: list[GraphNodeExecution],
) -> dict[str, GraphNodeExecution]:
    usable = [item for item in inputs if item.status in {"observed", "reconstructed"}]
    matched = {}
    for dependency in dependencies:
        candidates = [item for item in usable if dependency in item.provenance]
        if len(candidates) == 1:
            matched[dependency] = candidates[0]
    return matched


def evaluate_expression_inputs(
    expression: str,
    inputs: dict[str, GraphNodeExecution],
) -> tuple[Any, list[EvidenceSample]]:
    series_inputs = {
        name: result.samples
        for name, result in inputs.items()
        if result.samples
    }
    scalar_inputs = {
        name: result.value
        for name, result in inputs.items()
        if not result.samples
    }
    if not series_inputs:
        return evaluate_source_expression(expression, scalar_inputs), []
    times = sorted({
        sample.time_s
        for samples in series_inputs.values()
        for sample in samples
    })
    series_times = {
        name: [sample.time_s for sample in samples]
        for name, samples in series_inputs.items()
    }
    results = []
    for time_s in times:
        env = dict(scalar_inputs)
        for name, samples in series_inputs.items():
            sample = prior_sample(samples, series_times[name], time_s)
            if sample is None:
                break
            env[name] = sample.value
        else:
            results.append(EvidenceSample(time_s=time_s, value=evaluate_source_expression(expression, env)))
    if not results:
        raise SourceExpressionError("no aligned input samples")
    return None, results


def prior_sample(
    samples: list[EvidenceSample],
    times: list[float],
    time_s: float,
) -> Optional[EvidenceSample]:
    import bisect

    index = bisect.bisect_right(times, time_s) - 1
    return samples[index] if index >= 0 else None


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
