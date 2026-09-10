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
from flight_log_agent.analysis.mechanism_dag import (
    MechanismDAG, evaluate_feasibility, prepare_signal_series, sample_prepared_signal,
)
from flight_log_agent.analysis.source_expansion import UnresolvedSourceReference


@dataclass
class CheckpointRound:
    action: Literal["continue", "verified", "unresolved"]
    annotated: MechanismDAG
    references: list[UnresolvedSourceReference]
    summary: dict[str, Any]


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
    groups = observed_checkpoint_roots(dag)
    terminal_ids = {v.id for v in dag.vertices if v.metadata.get("is_terminal")}
    if question_target is not None:
        groups = {signal: roots for signal, roots in groups.items() if signal == question_target}
    else:
        groups = {signal: roots for signal, roots in groups.items() if terminal_ids.intersection(roots)}
    targets = groups or {None: tuple(sorted(terminal_ids))}
    roots = {root for ids in targets.values() for root in ids}
    view = dependency_view(dag, roots)
    samples = load_samples(view, tuple(s for s in groups if s in observed_signals))
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
    selected = next(iter(checkpoints.values())) if len(checkpoints) == 1 else None
    verified = bool(
        selected and selected["observed"] and selected["status"] == "matched"
        and selected["complete"] and not selected["analysis_requirements"]
        and not selected["source_requests"]
    )
    if verified:
        selected["authorizes_discovery_stop"] = True
        selected["verification_scope"] = "questioned_signal" if question_target is not None else "terminal"
    wanted = {UnresolvedSourceReference.model_validate(raw).visit_key()
              for checkpoint in checkpoints.values() for raw in checkpoint["source_requests"]}
    references = [r for r in dag.unresolved_references if r.visit_key() in wanted]
    action = "verified" if verified else "continue" if references else "unresolved"
    updates = {v.id: v for v in annotated_view.vertices}
    annotated = dag.model_copy(update={"vertices": [updates.get(v.id, v) for v in dag.vertices]})
    return CheckpointRound(action, annotated, references, {
        "action": action, "question_target": question_target,
        "checkpoints": {signal: value for signal, value in checkpoints.items() if signal is not None},
        "terminal_checkpoint": checkpoints.get(None),
        "selected_checkpoint": selected,
        "preflight": {signal or "internal_terminal": value for signal, value in preflight.items()},
        "dynamic_gate_count": len(ready),
        "pending_construction_count": len(dag.pending_construction),
        "reason": "source-backed checkpoint verified" if verified else "checkpoint has outstanding analysis requirements",
    })
