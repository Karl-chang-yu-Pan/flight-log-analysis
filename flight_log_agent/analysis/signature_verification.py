"""Candidate log-signature verification.

Two complementary pipelines run for every candidate:

- **Flat path** (``evaluate_verification_plan`` / ``evaluate_log_signature``):
  dispatches each typed check (``threshold``, ``state_equals``,
  ``tracks_setpoint``, ``derived_expression``, …) to a dedicated handler
  in :mod:`flight_log_agent.ulog.signature_evaluator`. Produces the
  per-branch / per-window / per-check breakdown that the report renders.

- **Graph path** (``compile_verification_graphs`` / ``execute_verification_graph``):
  builds one DAG per ``candidate.primary_output_signals[i]`` that
  reconstructs the terminal output from the backward binding slice and
  compares it to the logged actual. Produces a single conclusive verdict
  per terminal when the source bindings are complete.

These answer different questions about the same candidate, so they are
not redundant: the flat path supplies the per-check granularity; the
graph path supplies the primary-source reconstruction verdict.
``merge_graph_results`` overlays the graph verdict on top of the flat
evaluation — promoting unresolved to supported / contradicted when the
graph is conclusive, and capping the confidence ceiling so a graph
promotion can never exceed the strength of the underlying evidence.

Why static checks (``parameter_equals``, ``branch_parameter_satisfied``,
``topic_field_present``) stay on the flat path even though the graph
could in principle express them:

- The graph executor only runs when the candidate declares
  ``primary_output_signals`` *and* output bindings reach those signals.
  Moving static checks into the graph would silently lose them for
  candidates without a primary terminal (mode/state/parameter questions
  that aren't about reconstructing a downstream signal).
- The three flat handlers together total ~36 LOC and share the same
  helpers (``_get_parameter``, ``_compare_literal``, ``_parse_signal``)
  as the eleven other typed check kinds, so moving them does not delete
  any infrastructure — it would add new node types to
  ``graph_execution.execute_node`` while keeping the shared helpers
  alive for the remaining handlers.
- The result is net +LOC, lost coverage, and no simplification.
  Keep them flat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

from flight_log_agent.analysis.graph_execution import GraphExecutionResult, execute_verification_graph
from flight_log_agent.analysis.log_evidence import ULogEvidenceIndex
from flight_log_agent.analysis.verdict import (
    aggregate as _aggregate_verdict,
    branch_result as _branch_result,
    ceiling_for,
    combine_verdicts,
)
from flight_log_agent.analysis.verification_graph import compile_verification_graphs
from flight_log_agent.models import ApplicabilityResult, MechanismCandidate, SignatureEvaluation, VerificationPlan
from flight_log_agent.symbols import is_signal_reference
from flight_log_agent.ulog.signature_evaluator import evaluate_log_signature
from flight_log_agent.utils import dedupe_keep_order as dedupe
from flight_log_agent.utils import model_dump as _model_to_dict


def evaluate_candidate_log_signature(
    log_path: Path,
    candidate: MechanismCandidate,
    applicability: ApplicabilityResult,
    verification_plan: VerificationPlan | None = None,
    source_path: Path | None = None,
    output_bindings: Iterable[Any] = (),
    graph_evidence_index: ULogEvidenceIndex | None = None,
    verification_graphs: Iterable[Any] | None = None,
) -> SignatureEvaluation:
    if verification_plan is not None:
        evaluation = evaluate_verification_plan(
            log_path,
            candidate,
            applicability,
            verification_plan,
            source_path=source_path,
        )
    else:
        required_signals = [
            signal
            for signal in candidate.required_signals
            if is_signal_reference(signal)
        ]
        raw = evaluate_log_signature(
            log_path,
            candidate.name,
            [_model_to_dict(x) for x in candidate.expected_logged_signature],
            [_model_to_dict(x) for x in applicability.candidate_windows],
            required_signals,
            [_model_to_dict(x) for x in candidate.exclusion_checks],
            [_model_to_dict(x) for x in candidate.numeric_checks],
            source_path=source_path,
        )
        evaluation = normalize_signature_evaluation(candidate.name, raw, applicability)
    return apply_graph_verification(
        log_path,
        candidate,
        evaluation,
        output_bindings,
        source_path=source_path,
        evidence_index=graph_evidence_index,
        graphs=verification_graphs,
    )


def apply_graph_verification(
    log_path: Path,
    candidate: MechanismCandidate,
    evaluation: SignatureEvaluation,
    output_bindings: Iterable[Any],
    *,
    source_path: Path | None = None,
    evidence_index: ULogEvidenceIndex | None = None,
    graphs: Iterable[Any] | None = None,
) -> SignatureEvaluation:
    bindings = list(output_bindings)
    if not candidate.primary_output_signals or not bindings:
        return evaluation
    try:
        compiled_graphs = (
            list(graphs)
            if graphs is not None
            else compile_verification_graphs(candidate, bindings, source_path=source_path)
        )
        if not compiled_graphs:
            return merge_graph_results(evaluation, [])
        index = evidence_index or ULogEvidenceIndex.from_path(log_path)
        results = [execute_verification_graph(graph, index) for graph in compiled_graphs]
    except Exception as exc:
        return evaluation.model_copy(update={
            "warnings": dedupe([*evaluation.warnings, f"Primary-source graph verification failed: {exc}"]),
        })
    return merge_graph_results(evaluation, results)


def merge_graph_results(
    evaluation: SignatureEvaluation,
    results: list[GraphExecutionResult],
) -> SignatureEvaluation:
    conclusive = [result for result in results if result.verdict != "unresolved"]
    raw = dict(evaluation.raw)
    raw["verification_graphs"] = [_model_to_dict(result) for result in results]
    if not conclusive:
        return evaluation.model_copy(update={"raw": raw})

    graph_verdict = _aggregate_verdict(conclusive, level="graph")
    graph_checks = []
    evidence = list(evaluation.evidence)
    contradictions = list(evaluation.contradictions)
    for result in conclusive:
        supported = result.verdict == "supported"
        message = (
            f"Primary-source reconstruction matched logged terminal output {result.terminal_output}."
            if supported
            else f"Primary-source reconstruction contradicted logged terminal output {result.terminal_output}."
        )
        graph_checks.append({
            "type": "primary_source_terminal_comparison",
            "role": "mechanism_defining",
            "status": "passed" if supported else "failed",
            "message": message,
            "value": {"terminal_output": result.terminal_output, "graph_id": result.graph_id},
        })
        (evidence if supported else contradictions).append(message)

    verdict = combine_verdicts(evaluation.verdict, graph_verdict)
    if verdict == "supported" and contradictions:
        verdict = "mixed"
    elif verdict == "contradicted" and evidence:
        verdict = "mixed"

    return evaluation.model_copy(update={
        "verdict": verdict,
        "confidence_ceiling": ceiling_for(
            verdict,
            max_ceiling=(
                evaluation.confidence_ceiling
                if evaluation.verdict == "supported"
                else "medium"
            ),
        ),
        "evidence": dedupe(evidence),
        "contradictions": dedupe(contradictions),
        "check_results": [*evaluation.check_results, *graph_checks],
        "raw": raw,
    })


def evaluate_verification_plan(
    log_path: Path,
    candidate: MechanismCandidate,
    applicability: ApplicabilityResult,
    plan: VerificationPlan,
    *,
    source_path: Path | None = None,
) -> SignatureEvaluation:
    branch_results: list[dict[str, Any]] = []
    all_check_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for branch in plan.branches:
        if not branch.applicable:
            branch_results.append({
                "branch_id": branch.branch_id,
                "name": branch.name,
                "verdict": "excluded",
                "excluded_by": list(branch.excluded_by),
                "window_results": [],
            })
            continue
        window_results = []
        for window in branch.windows:
            numeric_checks = []
            exclusion_checks = []
            for planned in branch.checks:
                if not planned.executable:
                    continue
                check = _planned_check_dict(planned, window.name)
                (exclusion_checks if planned.category == "exclusion" else numeric_checks).append(check)
            raw = evaluate_log_signature(
                log_path,
                f"{candidate.name}:{branch.name}",
                [],
                [_model_to_dict(window)],
                list(branch.required_signals),
                exclusion_checks,
                numeric_checks,
                source_path=source_path,
            )
            check_results = raw.get("check_results") if isinstance(raw.get("check_results"), list) else []
            all_check_results.extend(check_results)
            window_results.append({
                "window": _model_to_dict(window),
                "verdict": _aggregate_verdict(check_results, level="check_list"),
                "check_results": check_results,
            })
        unresolved_checks = [
            {
                "check_id": planned.check_id,
                "branch_id": branch.branch_id,
                "role": planned.role,
                "type": planned.check.type,
                "status": "unresolved",
                "message": "; ".join(planned.unresolved_dependencies),
            }
            for planned in branch.checks
            if not planned.executable
        ]
        all_check_results.extend(unresolved_checks)
        branch_results.append(_branch_result(branch, window_results, unresolved_checks))

    verdict = _aggregate_verdict(branch_results, level="mechanism")
    evidence_branch_ids = {
        result["branch_id"]
        for result in branch_results
        if result["verdict"] in {"supported", "mixed"}
    }
    contradiction_branch_ids = {
        result["branch_id"]
        for result in branch_results
        if result["verdict"] in {"contradicted", "mixed"}
    }
    evidence = dedupe([
        result["message"]
        for result in all_check_results
        if result.get("role") == "mechanism_defining"
        and result.get("status") == "passed"
        and result.get("branch_id") in evidence_branch_ids
    ])
    contradictions = dedupe([
        result["message"]
        for result in all_check_results
        if result.get("role") == "mechanism_defining"
        and result.get("status") == "failed"
        and result.get("branch_id") in contradiction_branch_ids
    ])
    unresolved_defining = [
        result for result in all_check_results
        if result.get("role") == "mechanism_defining" and result.get("status") == "unresolved"
    ]
    if applicability.missing_required_signals:
        warnings.append(f"Missing required signals: {applicability.missing_required_signals}")
    ceiling = ceiling_for(
        verdict,
        has_unresolved_defining=bool(unresolved_defining),
    )
    return SignatureEvaluation(
        candidate_name=candidate.name,
        verdict=verdict,
        confidence_ceiling=ceiling,
        evidence=evidence,
        contradictions=contradictions,
        check_results=all_check_results,
        warnings=warnings,
        raw={
            "verification_plan": {"mechanism_id": plan.mechanism_id},
            "branch_results": branch_results,
        },
    )


def _planned_check_dict(planned: Any, window_name: str) -> dict[str, Any]:
    data = _model_to_dict(planned.check)
    data.update({
        "window": window_name,
        "check_id": planned.check_id,
        "branch_id": planned.branch_id,
        "role": planned.role,
    })
    return data


def normalize_signature_evaluation(
    candidate_name: str,
    raw: Any,
    applicability: ApplicabilityResult,
) -> SignatureEvaluation:
    if not isinstance(raw, dict):
        raw = {"raw": raw}

    evidence = _extract_list(raw, ["evidence", "supporting_evidence", "supports"])
    contradictions = _extract_list(raw, ["contradictions", "contradicting_evidence", "contradicts"])
    warnings = _extract_list(raw, ["warnings"])

    if applicability.missing_required_signals:
        warnings.append(f"Missing required signals: {applicability.missing_required_signals}")

    verdict_raw = str(raw.get("verdict") or raw.get("status") or "").lower()
    if "support" in verdict_raw:
        verdict = "supported"
    elif "contrad" in verdict_raw:
        verdict = "contradicted"
    elif "mixed" in verdict_raw:
        verdict = "mixed"
    else:
        from flight_log_agent.analysis.verdict import verdict_from_counts
        verdict = verdict_from_counts(
            supported=len(evidence),
            contradicted=len(contradictions),
        )

    ceiling = ceiling_for(
        verdict,
        missing_required_signals=bool(applicability.missing_required_signals),
    )

    check_results = raw.get("check_results") if isinstance(raw.get("check_results"), list) else []

    return SignatureEvaluation(
        candidate_name=candidate_name,
        verdict=verdict,
        confidence_ceiling=ceiling,
        evidence=[str(x) for x in evidence],
        contradictions=[str(x) for x in contradictions],
        check_results=check_results,
        warnings=[str(x) for x in warnings],
        raw=raw,
    )


def derive_confidence(
    applicability: ApplicabilityResult,
    evaluation: SignatureEvaluation,
) -> Literal["high", "medium", "low", "unresolved"]:
    if not applicability.applicable:
        return "low"
    if applicability.missing_required_signals and not evaluation.raw.get("verification_plan"):
        return "low"
    return evaluation.confidence_ceiling


def _extract_list(raw: dict[str, Any], keys: list[str]) -> list[Any]:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, list):
            return value
        if value:
            return [value]
    return []
