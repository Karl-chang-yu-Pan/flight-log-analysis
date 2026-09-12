"""Checkpoint-driven source discovery over the one mechanism DAG.

No question is reinterpreted here. An exact observed question target, or the
explicit discovery terminal when no comparison is supplied, defines scope.
A verified stop closes that source investigation; the judge still decides
whether the established mechanism answers the user's question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from flight_log_agent.analysis.dag_checkpoint import assess_checkpoint, dependency_view
from flight_log_agent.analysis.dag_replay import EvaluationScope, observed_checkpoint_roots
from flight_log_agent.analysis.dag_value import DAGValueProgram
from flight_log_agent.analysis.dag_observation import evaluate_local_observed_equations
from flight_log_agent.analysis.mechanism_dag import (
    ConstructionDemand, MechanismDAG, evaluate_feasibility, prepare_signal_series, sample_prepared_signal,
)
from flight_log_agent.analysis.source_expansion import UnresolvedSourceReference


@dataclass
class CheckpointRound:
    action: Literal["continue", "verified", "unresolved"]
    annotated: MechanismDAG
    references: list[UnresolvedSourceReference]
    summary: dict[str, Any]
    construction: ConstructionDemand = ConstructionDemand()


def evaluate_checkpoint_round(
    dag: MechanismDAG,
    *,
    parameter_values: dict[str, Any],
    observed_signals: set[str],
    signal_policies: dict[str, Any],
    load_samples: Callable[[MechanismDAG, tuple[str, ...]], dict[str, list[tuple[float, Any]]]],
    scope: Optional[EvaluationScope] = None,
    question_target: Optional[str] = None,
) -> CheckpointRound:
    """Preflight first, evaluate ready gates, replay, then select exact needs.

    No file/round budget, guessed binding, or numerical-only stop. An unresolved
    source-writer request remains a proof obligation even after a local match.
    """
    all_groups = observed_checkpoint_roots(dag)
    groups = all_groups
    terminal_ids = {v.id for v in dag.vertices if v.metadata.get("is_terminal")}
    if question_target is not None:
        groups = {signal: roots for signal, roots in groups.items() if signal == question_target}
    else:
        groups = {signal: roots for signal, roots in groups.items() if terminal_ids.intersection(roots)}
    targets = groups or {None: tuple(sorted(terminal_ids))}
    roots = {root for ids in targets.values() for root in ids}
    view = dependency_view(dag, roots)
    target_dependencies = {v.id for v in view.vertices}
    observed_dependencies = {item["signal"] for item in dag.observation_witnesses
                             if item["value_id"] in target_dependencies}
    intermediate_groups = {
        signal: ids for signal, ids in all_groups.items()
        if signal not in groups and (target_dependencies.intersection(ids) or signal in observed_dependencies)
    }
    # A local comparison must include alternative writers of that same
    # publication, not merely whichever writer the terminal walk encountered.
    view = dependency_view(dag, roots | {root for ids in intermediate_groups.values() for root in ids})
    samples = load_samples(view, tuple(s for s in (*groups, *intermediate_groups) if s in observed_signals))
    prepared = prepare_signal_series(samples, signal_policies)
    program = DAGValueProgram(view)
    session = program.bind(
        parameter_values=parameter_values,
        sample_resolver=lambda signal, timestamp: sample_prepared_signal(prepared, signal, timestamp),
    )
    kwargs = dict(
        source_dag=dag, parameter_values=parameter_values,
        observed_signals=observed_signals, signal_samples=samples,
        signal_policies=signal_policies, prepared_signal_series=prepared,
        value_program=program, value_session=session, scope=scope,
    )
    preflight = {signal: assess_checkpoint(view, ids, signal, attempt_replay=False, **kwargs)
                 for signal, ids in targets.items()}

    # Only graph-complete gate inputs can justify dynamic scheduling. Opaque
    # dependencies do not consume a whole log merely to fail at each timestamp.
    blocked: set[str] = set()
    unscoped_frontier = False
    for checkpoint in preflight.values():
        for requirement in checkpoint["analysis_requirements"]:
            if requirement["reason"] == "gate has no complete evaluation over the comparison domain":
                continue
            if requirement["kind"] == "observation_binding" and not requirement.get("vertex_id"):
                continue
            vertex_id = requirement.get("vertex_id")
            blocked.update(requirement.get("origin_vertex_ids") or ())
            if vertex_id:
                blocked.add(vertex_id)
            elif requirement.get("source_reference"):
                unscoped_frontier = True
    ready: set[str] = set()
    if not unscoped_frontier:
        for vertex in view.vertices:
            if vertex.kind != "branch":
                continue
            dependencies = dependency_view(view, [vertex.id])
            if not blocked.intersection(v.id for v in dependencies.vertices):
                ready.add(vertex.id)
    annotated_view = evaluate_feasibility(
        view, parameter_values=parameter_values, signal_samples=samples,
        signal_policies=signal_policies, prepared_signal_series=prepared,
        value_program=program, value_session=session, prune_dead=False,
        dynamic_branch_ids=ready, allow_assumptions=False, stream_timestamps=True,
    )
    checkpoints = {signal: assess_checkpoint(annotated_view, ids, signal, **kwargs)
                   for signal, ids in targets.items()}
    intermediates = {signal: assess_checkpoint(annotated_view, ids, signal, **kwargs)
                     for signal, ids in intermediate_groups.items()}
    local_equations = evaluate_local_observed_equations(
        view, program, signal_samples=samples, parameter_values=parameter_values,
        signal_policies=signal_policies, scope=scope, relevant_ids=target_dependencies,
    )
    selected = next(iter(checkpoints.values())) if len(checkpoints) == 1 else None
    verified = bool(
        selected and selected["observed"] and selected["status"] == "matched"
        and selected["complete"] and not selected["analysis_requirements"]
        and not selected["source_requests"]
        and not any(item["status"] == "mismatched" for item in intermediates.values())
    )
    if verified:
        selected["authorizes_discovery_stop"] = True
        selected["verification_scope"] = "questioned_signal" if question_target is not None else "terminal"
    wanted = {UnresolvedSourceReference.model_validate(raw).visit_key()
              for checkpoint in checkpoints.values() for raw in checkpoint["source_requests"]}
    references = [r for r in dag.unresolved_references if r.visit_key() in wanted]
    inactive = {vertex_id for checkpoint in checkpoints.values()
                for vertex_id in checkpoint["inactive_writer_ids"]}
    required = {requirement["vertex_id"] for checkpoint in checkpoints.values()
                for requirement in checkpoint["analysis_requirements"]
                if requirement["kind"] == "construction"}
    relevant = {vertex_id for checkpoint in checkpoints.values()
                for vertex_id in checkpoint["dependency_vertex_ids"]}
    gate_roots = [v.id for v in annotated_view.vertices if v.id in relevant and v.kind == "branch"
                  and v.feasibility_verdict == "unknown"]
    gate_inputs = {v.id for v in dependency_view(annotated_view, gate_roots).vertices}
    searchable = [reference for reference in references
                  if reference.visit_key() not in dag.exhausted_source_requests]
    gate_requests = [reference for reference in searchable
                     if gate_inputs.intersection(reference.origin_vertex_ids)]
    gate_work = required & gate_inputs
    local_needs = [need for check in local_equations
                   for need in check.get("input_requirements", [])]
    local_work = {need["vertex_id"] for need in local_needs if need["kind"] == "construction"}
    local_work &= set(dag.pending_construction) - inactive
    local_keys = {UnresolvedSourceReference.model_validate(raw).visit_key()
                  for need in local_needs for raw in need["source_requests"]
                  if need["vertex_id"] not in inactive}
    local_requests = [reference for reference in dag.unresolved_references
                      if reference.visit_key() in local_keys
                      and reference.visit_key() not in dag.exhausted_source_requests]
    # A missing guard helper is source work, not permission to expand its
    # guarded values. If that exact search is exhausted, retain the unknown
    # obligation but allow other useful work to proceed.
    # Conditional equation construction is not an applicability decision.
    # Its exact value prerequisites may proceed while unrelated guard source
    # is unknown. Only proven inactivity can discharge either kind of work.
    if local_work:
        materialize, next_references, kind = local_work, [], "local_calculation_construction"
    elif local_requests:
        materialize, next_references, kind = set(), local_requests, "local_calculation_source"
    elif gate_work:
        materialize, next_references, kind = gate_work, [], "guard_construction"
    elif gate_requests:
        materialize, next_references, kind = set(), gate_requests, "guard_source"
    else:
        local_inputs = {vertex_id for item in intermediates.values()
                        for vertex_id in item["dependency_vertex_ids"]}
        local_work = required & local_inputs
        materialize = local_work or required
        next_references = [] if materialize else searchable
        kind = "calculation_construction" if materialize else "source" if searchable else "blocked"
    construction = ConstructionDemand(
        materialize=frozenset(materialize),
        inactive=frozenset(inactive),
    )
    action = "verified" if verified else "continue" if next_references or materialize else "unresolved"
    updates = {v.id: v for v in annotated_view.vertices}
    annotated = dag.model_copy(update={"vertices": [updates.get(v.id, v) for v in dag.vertices]})
    return CheckpointRound(action, annotated, next_references, {
        "action": action, "question_target": question_target,
        "checkpoints": {signal: value for signal, value in checkpoints.items() if signal is not None},
        "terminal_checkpoint": checkpoints.get(None),
        "selected_checkpoint": selected,
        "intermediate_checkpoints": intermediates,
        "local_equation_checks": local_equations,
        "next_analysis": {
            "kind": kind, "operation_ids": sorted(materialize),
            "guard_ids": gate_roots,
            "source_requests": [reference.model_dump(mode="json") for reference in next_references],
        },
        "preflight": {signal or "internal_terminal": value for signal, value in preflight.items()},
        "dynamic_gate_count": len(ready),
        "pending_construction_count": len(dag.pending_construction),
        "reason": "source-backed checkpoint verified" if verified else "checkpoint has outstanding analysis requirements",
    }, construction=construction)
