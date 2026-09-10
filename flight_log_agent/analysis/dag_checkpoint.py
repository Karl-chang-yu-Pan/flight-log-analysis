"""Source-backed checkpoint assessment over an existing mechanism graph.

Requirements describe why this graph cannot yet verify an observation. They
are not a second mechanism or a verification plan: source requests retain the
constructor's exact frontier identities, and numerical work uses DAG replay.
"""

from __future__ import annotations

import ast
import json
import math
from collections import defaultdict
from typing import Any, Iterable, Optional, Sequence

from flight_log_agent.analysis.dag_replay import (
    EvaluationScope,
    observed_checkpoint_roots,
    replay_dag_roots,
)
from flight_log_agent.analysis.dag_value import DAGValueProgram, DAGValueSession
from flight_log_agent.analysis.mechanism_dag import MechanismDAG, PreparedSignalSeries
from flight_log_agent.utils import stable_id


def _policy(value: Any) -> dict[str, Any]:
    return value.model_dump() if hasattr(value, "model_dump") else value or {}


def _same_source_value(left: Any, right: Any) -> bool:
    # A declared NaN is stable source metadata, not a changed program.
    return left is right or left == right or (
        isinstance(left, float) and isinstance(right, float)
        and math.isnan(left) and math.isnan(right)
    )


def _unbound_operands(expression: Any) -> list[str]:
    """Inspect the compiled AST, without reparsing or guessing source bindings."""
    aliases = dict(expression.alias_to_operand)
    callees = {
        id(node.func) for node in ast.walk(expression.tree)
        if isinstance(node, ast.Call)
    }
    return sorted({
        node.id for node in ast.walk(expression.tree)
        if isinstance(node, ast.Name) and id(node) not in callees
        and node.id not in aliases
    })


def assess_checkpoint(
    annotated: MechanismDAG,
    root_ids: Sequence[str],
    observed: Optional[str],
    *,
    source_dag: Optional[MechanismDAG] = None,
    observed_signals: Iterable[str] = (),
    parameter_values: Optional[dict[str, Any]] = None,
    signal_samples: dict[str, list[tuple[float, Any]]],
    signal_policies: Optional[dict[str, Any]] = None,
    prepared_signal_series: Optional[dict[str, PreparedSignalSeries]] = None,
    value_program: Optional[DAGValueProgram] = None,
    value_session: Optional[DAGValueSession] = None,
    scope: Optional[EvaluationScope] = None,
    attempt_replay: bool = True,
) -> dict[str, Any]:
    """Identify graph-local obligations before attempting numerical replay.

    A verified result covers known writers in the supplied graph and window,
    not the completeness of source discovery or the answer to the question.
    This function performs no source search, log I/O, or model calls.
    """
    source = source_dag if source_dag is not None else annotated
    vertices = {vertex.id: vertex for vertex in annotated.vertices}
    requested_roots = set(root_ids)
    # Include every known alternative publication, even if the caller omitted it.
    roots = requested_roots | set(observed_checkpoint_roots(annotated).get(observed, ()))
    incoming: dict[str, list[Any]] = defaultdict(list)
    outgoing: dict[str, list[Any]] = defaultdict(list)
    for edge in annotated.edges:
        incoming[edge.target_id].append(edge)
        outgoing[edge.source_id].append(edge)
    requirements: dict[str, dict[str, Any]] = {}

    def require(kind: str, reason: str, vertex_id: str = "", **context: Any) -> None:
        vertex = vertices.get(vertex_id)
        payload = {
            "kind": kind, "reason": reason, "vertex_id": vertex_id,
            "file": vertex.file if vertex else None,
            "line": vertex.line if vertex else None,
            **context,
        }
        key = stable_id("checkpoint-need", (json.dumps(payload, sort_keys=True),))
        requirements[key] = {"id": key, **payload}

    if not roots:
        require("observation_binding", "no source writer identifies the checkpoint")
    if not observed:
        require("observation_binding", "terminal has no source-proven publication binding")
    if scope is not None and (scope.windows is None or not scope.windows or scope.error):
        require("evaluation_scope", scope.error or "comparison has no evaluation windows")
    if scope is not None and scope.assumptions:
        require("evaluation_scope", "questioned-condition assumptions require validation",
                assumptions=list(scope.assumptions))

    observed_set = set(observed_signals)
    pending_construction = set(source.pending_construction)
    parameters = (value_session.parameters if value_session is not None else
                  {str(name).upper(): value for name, value in (parameter_values or {}).items()})
    policies = signal_policies or {}
    prepared = prepared_signal_series or {}

    def series(signal: str) -> Sequence[tuple[float, Any]]:
        item = prepared.get(signal)
        return item.samples if item is not None else signal_samples.get(signal, ())

    output_samples = series(observed) if observed else ()
    windows = list(scope.windows or ()) if scope is not None else (
        [(output_samples[0][0], output_samples[-1][0])] if len(output_samples) >= 2 else []
    )

    def known_domain(branch: Any) -> bool:
        metadata = branch.metadata or {}
        if metadata.get("static_evaluation", {}).get("assumed"):
            return False
        if metadata.get("static_evaluation", {}).get("status") == "value":
            return True
        domain = metadata.get("evaluation_domain") or []
        sampling = metadata.get("sampling_policies") or {}
        return bool(
            windows and len(domain) == 2 and sampling
            and all(method and method != "unknown" for method in sampling.values())
            and not metadata.get("evaluation_failures")
            and all(domain[0] <= start <= end <= domain[1] for start, end in windows)
        )

    # Only exact, annotated false gates can discharge an inactive writer.
    # Unknown and assumption-based gates keep every alternative in the view.
    inactive: dict[str, str] = {}
    for vertex in annotated.vertices:
        if vertex.kind != "operation":
            continue
        if (vertex.metadata.get("synthetic_boundary_transfer")
                and vertex.metadata.get("boundary_direction") == "subscribe"):
            # Skipping a copy retains receiver state; it does not prove that
            # the receiver is inactive or equals the latest topic sample.
            continue
        for edge in incoming[vertex.id]:
            branch = vertices.get(edge.source_id)
            if (edge.kind == "control" and branch is not None
                    and branch.kind == "branch"
                    and branch.feasibility_verdict == "always_false"
                    and branch.metadata.get("expression_inputs_exact")
                    and known_domain(branch)):
                inactive[vertex.id] = branch.id
                break

    relevant: set[str] = set()
    pending = sorted(roots)
    selected_edges: dict[str, Any] = {}
    while pending:
        vertex_id = pending.pop()
        if vertex_id in relevant:
            continue
        relevant.add(vertex_id)
        if vertex_id not in vertices:
            require("source_linkage", "dependency vertex is absent from the graph", vertex_id)
            continue
        edges = incoming[vertex_id]
        if vertex_id in inactive:
            edges = [edge for edge in edges if edge.source_id == inactive[vertex_id]
                     and edge.kind == "control"]
        for edge in edges:
            selected_edges[edge.id] = edge
            pending.append(edge.source_id)

    view = annotated.model_copy(update={
        "vertices": [vertex for vertex in annotated.vertices if vertex.id in relevant],
        "edges": list(selected_edges.values()),
    })
    program = value_program or (value_session.program if value_session else DAGValueProgram(view))
    if value_session is not None and value_session.program is not program:
        raise ValueError("value_session must be bound to value_program")
    # The compiled program can be shared with the full source graph, but it
    # must not belong to a previous round with reused vertex IDs.
    source_vertices = {vertex.id: vertex for vertex in source.vertices}
    source_incoming: dict[str, list[Any]] = defaultdict(list)
    for edge in source.edges:
        source_incoming[edge.target_id].append(edge)
    semantic_metadata = (
        "source_expression_ref", "expression_inputs_exact", "source_call_roles",
        "reachability", "target_scope", "source_identity", "value",
        "synthetic_boundary_transfer", "boundary_direction", "external_target_signal",
    )
    for vertex_id in relevant & vertices.keys():
        expected = source_vertices.get(vertex_id)
        actual = program.vertices.get(vertex_id)
        if expected is None or actual is None or any(
            getattr(actual, key) != getattr(expected, key)
            for key in ("kind", "sub_kind", "variable", "expression", "lowered_expression",
                        "predicate_raw", "signal_name", "file", "line")
        ) or any(not _same_source_value(actual.metadata.get(key), expected.metadata.get(key))
                 for key in semantic_metadata):
            raise ValueError("value_program does not describe the checkpoint source graph")
        if value_program is not None or value_session is not None:
            expected_data = {(e.source_id, e.role, e.via) for e in source_incoming[vertex_id] if e.kind == "data"}
            actual_data = {(e.source_id, e.role, e.via) for e in program.data_by_target.get(vertex_id, ())}
            expected_controls = {e.source_id for e in source_incoming[vertex_id] if e.kind == "control"}
            expected_selections = {e.source_id for e in source_incoming[vertex_id] if e.kind == "selection"}
            if (expected_data != actual_data
                    or expected_controls != set(program.controls_by_target.get(vertex_id, ()))
                    or expected_selections != set(program.selections_by_target.get(vertex_id, ()))):
                raise ValueError("value_program does not describe the checkpoint source graph")

    source_requests: dict[tuple[Any, ...], Any] = {}
    requests_by_origin: dict[str, list[Any]] = defaultdict(list)
    for reference in source.unresolved_references:
        origins = set(reference.origin_vertex_ids) & relevant
        origins -= set(inactive)
        if not origins:
            if not reference.origin_vertex_ids:
                require("source_linkage", "frontier reference lacks a graph origin",
                        source_reference=reference.model_dump(mode="json"))
            continue
        source_requests[reference.visit_key()] = reference
        for origin in sorted(origins):
            requests_by_origin[origin].append(reference)
        require("source_lookup", "source discovery must resolve this exact reference", min(origins),
                origin_vertex_ids=sorted(origins), source_reference=reference.model_dump(mode="json"))

    def check_signal(signal: str, vertex_id: str = "") -> None:
        samples = series(signal)
        if signal not in observed_set or len(samples) < 2:
            require("observation_data", "exact signal is absent or has insufficient samples",
                    vertex_id, signal=signal)
        policy = _policy(policies.get(signal) or (
            prepared[signal].policy if signal in prepared else None
        ))
        if not policy.get("method") or policy["method"] == "unknown":
            require("sampling_policy", "signal has no derived sampling policy", vertex_id, signal=signal)

    if observed:
        check_signal(observed)
    for root in sorted(roots):
        vertex = vertices.get(root)
        metadata = vertex.metadata if vertex else {}
        if (vertex is None or vertex.kind != "operation" or not observed
                or metadata.get("external_target_signal") != observed
                or not metadata.get("synthetic_boundary_transfer")
                or metadata.get("boundary_direction") != "publish"
                or not vertex.file or not vertex.line):
            require("observation_binding", "writer lacks a source-located publication transfer", root)

    for vertex_id in sorted(relevant & vertices.keys()):
        vertex = vertices[vertex_id]
        metadata = vertex.metadata or {}
        if vertex_id in inactive:
            continue
        if vertex_id in pending_construction:
            require("construction", "source value dependencies have not been materialized", vertex_id)
            continue
        if vertex.kind == "evidence":
            if vertex.sub_kind == "parameter":
                if metadata.get("value") is None and parameters.get(str(vertex.signal_name or "").upper()) is None:
                    require("parameter_data", "required parameter value is unavailable", vertex_id, parameter=vertex.signal_name)
            elif vertex.sub_kind == "logged_signal":
                signal = str(vertex.signal_name or "")
                check_signal(signal, vertex_id)
                transfers = [vertices[edge.target_id] for edge in outgoing[vertex_id]
                             if edge.id in selected_edges and edge.kind == "data"
                             and edge.target_id in vertices]
                if (metadata.get("boundary") != "source_proven"
                        or metadata.get("grounded_via") == "declared_type"
                        or metadata.get("observation") != "observed"
                        or not transfers or any(
                            not transfer.metadata.get("synthetic_boundary_transfer")
                            or transfer.metadata.get("boundary_direction") != "subscribe"
                            or not transfer.file or not transfer.line
                            for transfer in transfers)):
                    require("observation_binding", "input lacks an explicit source transfer to storage",
                            vertex_id, signal=signal)
                if signal == observed:
                    require("observation_binding", "comparison output is also a checkpoint input",
                            vertex_id, signal=signal)
            elif vertex.sub_kind == "opaque_symbol":
                linked_request = requests_by_origin.get(vertex_id) or any(
                    edge.role in reference.origin_operands
                    for edge in outgoing[vertex_id] if edge.id in selected_edges
                    for reference in requests_by_origin.get(edge.target_id, ())
                )
                if not linked_request:
                    require("source_linkage", "source storage has no producer or scoped expansion request",
                            vertex_id, operand=vertex.signal_name,
                            source_identity=metadata.get("source_identity"))
            continue

        compiled = program.compiled_vertices.get(vertex_id)
        if compiled is None or compiled.compile_error:
            require("expression", compiled.compile_error if compiled else "no compiled source expression",
                    vertex_id)
        elif compiled.expression is not None:
            for operand in _unbound_operands(compiled.expression):
                require("source_linkage", "expression operand has no DAG binding", vertex_id, operand=operand)
        controls = [edge for edge in incoming[vertex_id] if edge.kind == "control"]
        if vertex.kind == "operation":
            reachability = metadata.get("reachability") or {}
            if not reachability.get("exact") or (reachability.get("all_of") and not controls):
                require("control_flow", "operation reachability is not fully source-linked", vertex_id)
            if (metadata.get("synthetic_boundary_transfer")
                    and metadata.get("boundary_direction") == "subscribe"
                    and (controls or reachability.get("all_of"))):
                require("state_alignment", "conditional input transfer requires receiver-state and transfer-time evidence",
                        vertex_id)
        elif vertex.kind == "branch":
            if metadata.get("static_evaluation", {}).get("assumed"):
                require("control_flow", "gate verdict depends on an assumption", vertex_id)
            elif not known_domain(vertex):
                require("control_flow", "gate has no complete evaluation over the comparison domain", vertex_id)

    # No numeric upgrade is allowed while graph-local proof obligations remain.
    # In particular, missing storage writers cannot be erased by a local match.
    replay: Optional[dict[str, Any]] = None
    if not requirements and attempt_replay:
        replay = replay_dag_roots(
            view, sorted(roots), str(observed), parameter_values=parameter_values,
            signal_samples=signal_samples, signal_policies=signal_policies,
            prepared_signal_series=prepared_signal_series, value_program=program,
            value_session=value_session, scope=scope,
        )
        if not replay.get("complete"):
            require("numerical_replay", replay.get("reason") or "replay is incomplete",
                    operation_results=replay.get("results", []))
    result = dict(replay) if replay is not None else {
        "status": "not_attempted", "complete": False,
        "reason": "source-backed checkpoint has outstanding analysis requirements",
    }
    return {
        **result,
        "checkpoint_id": stable_id("checkpoint", (source.terminal, observed or "")),
        "observed": observed,
        "root_vertex_ids": sorted(roots),
        "dependency_vertex_ids": sorted(relevant),
        "inactive_writer_ids": sorted(inactive.keys() & relevant),
        "verification_scope": "known_graph",
        "authorizes_discovery_stop": False,
        "analysis_requirements": [requirements[key] for key in sorted(requirements)],
        "source_requests": sorted(
            (reference.model_dump(mode="json") for reference in source_requests.values()),
            key=lambda item: json.dumps(item, sort_keys=True),
        ),
    }


def dependency_view(dag: MechanismDAG, roots: Iterable[str]) -> MechanismDAG:
    """An ancestor-closed graph view, preserving original vertex/edge identities."""
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in dag.edges:
        incoming[edge.target_id].append(edge.source_id)
    relevant = set(roots)
    pending = list(relevant)
    while pending:
        for predecessor in incoming[pending.pop()]:
            if predecessor not in relevant:
                relevant.add(predecessor)
                pending.append(predecessor)
    return dag.model_copy(update={
        "vertices": [v for v in dag.vertices if v.id in relevant],
        "edges": [e for e in dag.edges if e.source_id in relevant and e.target_id in relevant],
    })
