"""Graph-native numerical replay shared by terminal and diagnostic checkpoints.

Checkpoint results describe only the supplied graph and evaluation domain.
They do not establish discovery completeness or authorize early termination.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional, Sequence

from flight_log_agent.analysis.dag_value import DAGValueProgram, DAGValueSession
from flight_log_agent.analysis.mechanism_dag import (
    DAGValueSeries,
    MechanismDAG,
    PreparedSignalSeries,
    boolean_sample_windows,
    evaluate_dag_vertex_series,
    prepare_signal_series,
    sample_prepared_signal,
)


@dataclass(frozen=True)
class EvaluationScope:
    """A resolved comparison domain; None windows mean unresolved, not all time."""

    windows: Optional[tuple[tuple[float, float], ...]]
    signal: str = ""
    reference: str = ""
    units: str = ""
    frame: str = ""
    assumptions: tuple[str, ...] = ()
    error: str = ""

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> "EvaluationScope":
        raw_windows = result.get("windows")
        error = str(result.get("error") or "")
        windows = None
        if raw_windows is not None and not error:
            try:
                parsed = [(float(start), float(end)) for start, end in raw_windows]
                if any(
                    not math.isfinite(start) or not math.isfinite(end) or end < start
                    for start, end in parsed
                ):
                    raise ValueError("invalid evaluation window")
                windows = tuple(_merge_windows(parsed))
            except (TypeError, ValueError):
                error = "invalid evaluation windows"
        if windows is None and not error:
            error = "questioned condition is unresolved"
        return cls(
            windows=windows,
            signal=str(result.get("signal") or ""),
            reference=str(result.get("reference") or ""),
            units=str(result.get("units") or ""),
            frame=str(result.get("frame") or ""),
            assumptions=tuple(str(item) for item in result.get("assumptions") or ()),
            error=error,
        )

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def observed_checkpoint_roots(dag: MechanismDAG) -> dict[str, tuple[str, ...]]:
    """Group only source-proven publication operations, never names or types."""
    roots: dict[str, list[str]] = {}
    for vertex in dag.vertices:
        signal = str((vertex.metadata or {}).get("external_target_signal") or "")
        if vertex.kind == "operation" and signal:
            roots.setdefault(signal, []).append(vertex.id)
    return {signal: tuple(ids) for signal, ids in sorted(roots.items())}


# The five distinct replay states (core correctness invariant):
# ``matched`` is supporting evidence and ``mismatched`` contradictory
# evidence ONLY when replay completeness (writer reachability domains,
# ordering, retained state, alignment, coverage) is trustworthy.
# ``partial`` and ``unevaluable`` are unresolved evidence;
# ``not_attempted`` is neutral.
ReplayStatus = Literal[
    "not_attempted", "unevaluable", "partial", "matched", "mismatched"
]


def _intersect_windows(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    for first_start, first_end in first:
        for second_start, second_end in second:
            start = max(first_start, second_start)
            end = min(first_end, second_end)
            if start <= end:
                intersections.append((start, end))
    return _merge_windows(intersections)


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(windows):
        if end < start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _window_duration(windows: list[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in _merge_windows(windows))


def _replay_tolerance(
    samples: list[tuple[float, Any]],
    policy: Optional[Any],
) -> float:
    """Derive a numerical tolerance from data scale and declared policy."""
    values = [
        float(value)
        for _timestamp, value in samples
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not values:
        return 1e-9
    if policy is not None and hasattr(policy, "model_dump"):
        policy = policy.model_dump()
    if str((policy or {}).get("method") or "") == "discrete_hold":
        return 1e-9
    ordered = sorted(abs(value) for value in values)
    scale = ordered[len(ordered) // 2]
    sorted_values = sorted(values)
    median_value = sorted_values[len(sorted_values) // 2]
    deviations = sorted(abs(value - median_value) for value in values)
    robust_variation = deviations[len(deviations) // 2]
    return max(1e-6, 1e-3 * max(scale, robust_variation))


def replay_dag_roots(
    annotated: MechanismDAG,
    root_ids: Sequence[str],
    observed: str,
    *,
    parameter_values: Optional[dict[str, Any]] = None,
    signal_samples: dict[str, list[tuple[float, Any]]],
    signal_policies: Optional[dict[str, Any]] = None,
    prepared_signal_series: Optional[dict[str, PreparedSignalSeries]] = None,
    value_program: Optional[DAGValueProgram] = None,
    value_session: Optional[DAGValueSession] = None,
    scope: Optional[EvaluationScope] = None,
) -> dict[str, Any]:
    """Compare source writers with independent observations through DAG edges.

    With no scope, replay retains the full observed-output domain. A resolved
    scope restricts the comparison, not input history or source extraction.
    This routine performs no I/O and never substitutes observations for writers.
    """
    def not_attempted(reason: str) -> dict[str, Any]:
        return {"status": "not_attempted", "complete": False, "reason": reason}

    if scope is not None and scope.windows is None:
        return not_attempted(scope.error or "questioned condition is unresolved")
    if scope is not None and not scope.windows:
        return not_attempted("questioned condition has no evaluation windows")
    by_id = {vertex.id: vertex for vertex in annotated.vertices}
    ids = tuple(dict.fromkeys(root_ids))
    terminal_ops = [by_id[root] for root in ids if root in by_id]
    if not ids or len(terminal_ops) != len(ids) or any(
        op.kind != "operation"
        or (op.metadata or {}).get("external_target_signal") != observed
        for op in terminal_ops
    ):
        return not_attempted("roots lack exact publication provenance for the observation")
    samples = signal_samples
    prepared_series = dict(prepared_signal_series or {})
    missing = {
        signal: series for signal, series in samples.items()
        if signal not in prepared_series
    }
    if missing:
        prepared_series.update(prepare_signal_series(missing, signal_policies))
    if observed not in prepared_series:
        return not_attempted(f"{observed} not observed in this log")
    observed_samples = list(prepared_series[observed].samples)
    if len(observed_samples) < 2:
        return not_attempted(f"{observed} has too few samples")
    observed_policy = (signal_policies or {}).get(observed)
    observed_span = (observed_samples[0][0], observed_samples[-1][0])
    requested = list(scope.windows) if scope is not None else [observed_span]
    comparison_domain = _intersect_windows(requested, [observed_span])
    if _window_duration(comparison_domain) <= 0:
        return not_attempted("no positive-duration observed evaluation domain")
    tolerance = _replay_tolerance(
        [(t, v) for t, v in observed_samples
         if any(start <= t <= end for start, end in comparison_domain)],
        observed_policy,
    )
    value_program = value_program or (
        value_session.program if value_session is not None else DAGValueProgram(annotated)
    )
    if value_session is not None and value_session.program is not value_program:
        raise ValueError("value_session must be bound to value_program")

    def resolve_sample(signal: str, timestamp: float) -> Optional[Any]:
        return sample_prepared_signal(prepared_series, signal, timestamp)

    value_session = value_session or value_program.bind(
        parameter_values=parameter_values,
        sample_resolver=resolve_sample,
    )

    vertices = by_id
    controls_by_op: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for edge in annotated.edges:
        incoming.setdefault(edge.target_id, []).append(edge.source_id)
        if edge.kind == "control":
            controls_by_op.setdefault(edge.target_id, []).append(edge.source_id)

    ancestors = set(ids)
    pending = list(ids)
    while pending:
        for predecessor in incoming.get(pending.pop(), ()):
            if predecessor not in ancestors:
                ancestors.add(predecessor)
                pending.append(predecessor)
    circular = [
        vertex.id for vertex in annotated.vertices
        if vertex.id in ancestors and vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal" and vertex.signal_name == observed
    ]
    if circular:
        return {**not_attempted("comparison output is also an input; independent replay unavailable"),
                "blocking_vertex_ids": circular}
    unproven = [
        vertex.id for vertex in annotated.vertices
        if vertex.id in ancestors and vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal"
        and ((vertex.metadata or {}).get("grounded_via") == "declared_type"
             or (vertex.metadata or {}).get("observation") == "unlogged")
    ]
    if unproven:
        return {**not_attempted("input observation lacks proven runtime binding"),
                "blocking_vertex_ids": unproven}

    results: list[dict[str, Any]] = []
    writer_domains: list[tuple[str, list[tuple[float, float]]]] = []
    all_writers = set(observed_checkpoint_roots(annotated).get(observed, ()))
    complete = all_writers == set(ids)
    for op in terminal_ops:
        reachability = (op.metadata or {}).get("reachability") or {}
        domain = list(comparison_domain)
        writer_complete = bool(reachability.get("exact", False))
        for branch_id in controls_by_op.get(op.id, []):
            branch = vertices.get(branch_id)
            if branch is None or branch.kind != "branch":
                writer_complete = False
                continue
            if branch.feasibility_verdict == "always_false":
                domain = []
            elif branch.feasibility_verdict == "always_true":
                branch_domain = [observed_span]
            elif branch.active_windows:
                branch_domain = list(branch.active_windows)
                policies_used = (branch.metadata or {}).get("sampling_policies") or {}
                if not policies_used or "unknown" in policies_used.values():
                    writer_complete = False
            else:
                branch_domain = []
                writer_complete = False
            if (branch.metadata or {}).get("static_evaluation", {}).get("assumed", False):
                writer_complete = False
            if domain:
                domain = _intersect_windows(domain, branch_domain)

        if not domain:
            results.append(
                {
                    "operation_id": op.id,
                    "expression": op.expression,
                    "grounded": None,
                    "evaluation_mode": "dag_value_plan",
                    "evaluable": True,
                    "active_windows": domain,
                    "active_duration": 0.0,
                    "matched_duration": 0.0,
                    "match_fraction": None,
                }
            )
            writer_domains.append((op.id, domain))
            complete = complete and writer_complete
            continue

        reconstructed: DAGValueSeries = evaluate_dag_vertex_series(
            annotated,
            op.id,
            parameter_values=parameter_values,
            signal_samples=samples,
            signal_policies=signal_policies,
            prepared_signal_series=prepared_series,
            timestamps=(timestamp for timestamp, _value in observed_samples),
            evaluation_windows=domain,
            value_program=value_program,
            value_session=value_session,
        )
        comparison_samples: list[tuple[float, bool]] = []
        errors: list[float] = []
        comparison_complete = reconstructed.complete
        for timestamp, expected_value in reconstructed.samples:
            observed_value = sample_prepared_signal(
                prepared_series, observed, timestamp
            )
            if not isinstance(expected_value, (int, float)) or not isinstance(
                observed_value, (int, float)
            ):
                comparison_complete = False
                continue
            error = abs(float(expected_value) - float(observed_value))
            if not math.isfinite(error):
                comparison_complete = False
                continue
            errors.append(error)
            comparison_samples.append(
                (
                    timestamp,
                    error <= tolerance,
                )
            )
        if not comparison_samples:
            results.append(
                {
                    "operation_id": op.id,
                    "expression": op.expression,
                    "grounded": None,
                    "evaluation_mode": "dag_value_plan",
                    "evaluable": False,
                    "active_windows": domain,
                    "reason": reconstructed.reason,
                }
            )
            complete = False
            continue
        input_policies_complete = (
            reconstructed.policies_complete
            if reconstructed.referenced_signals
            else True
        )
        if not input_policies_complete or observed_policy is None:
            writer_complete = False
        writer_complete = writer_complete and comparison_complete
        # Disjoint windows cannot lend each other a held comparison value.
        match_windows = [
            window for start, end in domain
            for window in boolean_sample_windows([
                (timestamp, matched) for timestamp, matched in comparison_samples
                if start <= timestamp <= end
            ])
        ]
        active_duration = _window_duration(domain)
        matched_duration = _window_duration(_intersect_windows(match_windows, domain))
        results.append(
            {
                "operation_id": op.id,
                "expression": op.expression,
                "grounded": None,
                "evaluation_mode": "dag_value_plan",
                "evaluable": True,
                "active_windows": domain,
                "active_duration": round(active_duration, 6),
                "matched_duration": round(matched_duration, 6),
                "sample_count": len(comparison_samples),
                "mismatched_samples": sum(not match for _time, match in comparison_samples),
                "mean_abs_error": sum(errors) / len(errors),
                "max_abs_error": max(errors),
                "reason": reconstructed.reason,
                "match_fraction": (
                    round(matched_duration / active_duration, 3)
                    if active_duration
                    else None
                ),
            }
        )
        writer_domains.append((op.id, domain))
        complete = complete and writer_complete

    merged_domain = _merge_windows(
        [window for _operation_id, domain in writer_domains for window in domain]
    )
    summed_writer_duration = sum(
        _window_duration(domain) for _operation_id, domain in writer_domains
    )
    covered_duration = _window_duration(merged_domain)
    non_overlapping = abs(summed_writer_duration - covered_duration) <= 1e-6
    covers_output = _intersect_windows(merged_domain, requested) == _merge_windows(requested)
    complete = complete and non_overlapping and covers_output and covered_duration > 0

    if not results:
        status: ReplayStatus = "unevaluable"
        reason = "no terminal operation was available for DAG replay"
    elif not any(r.get("evaluable") for r in results):
        status = "unevaluable"
        reason = "no terminal operation was evaluable through DAG edges"
    elif not complete:
        status = "partial"
        reason = (
            "writer reachability, policy, non-overlap, or output-domain "
            "coverage is incomplete"
        )
    else:
        matched_duration = sum(
            float(result.get("matched_duration") or 0.0)
            for result in results
            if result.get("evaluable")
        )
        match_fraction = matched_duration / covered_duration if covered_duration else 0.0
        status = "matched" if match_fraction >= 0.95 else "mismatched"
        reason = f"complete piecewise replay match fraction {match_fraction:.3f}"
    return {
        "status": status,
        "complete": complete,
        "reason": reason,
        "observed": observed,
        "evaluation_windows": requested,
        "missing_writer_ids": sorted(all_writers - set(ids)),
        "dependency_issues": [
            {"vertex_id": vertex_id, "reason": compiled.compile_error}
            for vertex_id, compiled in value_program.compiled_vertices.items()
            if vertex_id in ancestors and compiled.compile_error
        ],
        "tolerance": round(tolerance, 6),
        "results": results,
    }
