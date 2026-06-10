from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from flight_log_agent.models import ApplicabilityResult, MechanismCandidate, SignatureEvaluation, VerificationPlan
from flight_log_agent.ulog.signature_evaluator import evaluate_log_signature


def evaluate_candidate_log_signature(
    log_path: Path,
    candidate: MechanismCandidate,
    applicability: ApplicabilityResult,
    verification_plan: VerificationPlan | None = None,
    source_path: Path | None = None,
) -> SignatureEvaluation:
    if verification_plan is not None:
        return evaluate_verification_plan(
            log_path,
            candidate,
            applicability,
            verification_plan,
            source_path=source_path,
        )
    required_signals = [
        signal
        for signal in candidate.required_signals
        if _is_logged_signal_reference(signal)
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
    return normalize_signature_evaluation(candidate.name, raw, applicability)


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
                "verdict": verdict_from_role_results(check_results),
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
        branch_result = aggregate_branch_results(branch, window_results, unresolved_checks)
        branch_results.append(branch_result)

    verdict = aggregate_mechanism_verdict(branch_results)
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
    ceiling = confidence_ceiling_for_plan(verdict, unresolved_defining)
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


def verdict_from_role_results(results: list[dict[str, Any]]) -> str:
    applicability_results = [result for result in results if result.get("role") == "branch_applicability"]
    if any(result.get("status") == "failed" for result in applicability_results):
        return "excluded"
    if any(result.get("status") == "unresolved" for result in applicability_results):
        return "unresolved"
    defining = [result for result in results if result.get("role") == "mechanism_defining"]
    if not defining or any(result.get("status") == "unresolved" for result in defining):
        return "unresolved"
    passed = any(result.get("status") == "passed" for result in defining)
    failed = any(result.get("status") == "failed" for result in defining)
    if passed and failed:
        return "mixed"
    if failed:
        return "contradicted"
    if all(result.get("status") == "passed" for result in defining):
        return "supported"
    return "unresolved"


def aggregate_branch_results(branch: Any, windows: list[dict[str, Any]], unresolved_checks: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [window["verdict"] for window in windows]
    if branch.unresolved_dependencies or any(item.get("role") == "mechanism_defining" for item in unresolved_checks):
        verdict = "unresolved"
    elif "supported" in verdicts and any(item in {"contradicted", "mixed"} for item in verdicts):
        verdict = "mixed"
    elif "supported" in verdicts:
        verdict = "supported"
    elif "mixed" in verdicts:
        verdict = "mixed"
    elif verdicts and all(item == "excluded" for item in verdicts):
        verdict = "excluded"
    elif "contradicted" in verdicts:
        verdict = "contradicted"
    else:
        verdict = "unresolved"
    return {
        "branch_id": branch.branch_id,
        "name": branch.name,
        "verdict": verdict,
        "has_mechanism_defining_checks": any(
            planned.role == "mechanism_defining"
            for planned in branch.checks
        ),
        "unresolved_dependencies": list(branch.unresolved_dependencies),
        "window_results": windows,
    }


def aggregate_mechanism_verdict(branch_results: list[dict[str, Any]]) -> str:
    verdicts = [
        result["verdict"]
        for result in branch_results
        if result["verdict"] != "excluded" and result.get("has_mechanism_defining_checks")
    ]
    if "supported" in verdicts and any(item in {"contradicted", "mixed"} for item in verdicts):
        return "mixed"
    if "supported" in verdicts:
        return "supported"
    if "mixed" in verdicts:
        return "mixed"
    if verdicts and all(item == "contradicted" for item in verdicts):
        return "contradicted"
    return "unresolved"


def confidence_ceiling_for_plan(verdict: str, unresolved_defining: list[dict[str, Any]]) -> str:
    if verdict == "supported":
        return "medium" if unresolved_defining else "high"
    if verdict == "mixed":
        return "medium"
    if verdict == "contradicted":
        return "low"
    return "unresolved"


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


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
    elif evidence and contradictions:
        verdict = "mixed"
    elif evidence:
        verdict = "supported"
    elif contradictions:
        verdict = "contradicted"
    else:
        verdict = "unresolved"

    if applicability.missing_required_signals:
        ceiling = "low"
    elif verdict == "supported":
        ceiling = "high"
    elif verdict == "mixed":
        ceiling = "medium"
    elif verdict == "contradicted":
        ceiling = "low"
    else:
        ceiling = "unresolved"

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


def _model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return value
    return {"value": value}


def _extract_list(raw: dict[str, Any], keys: list[str]) -> list[Any]:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, list):
            return value
        if value:
            return [value]
    return []


def _is_logged_signal_reference(value: str) -> bool:
    if not isinstance(value, str) or "." not in value:
        return False
    topic, field = value.split(".", 1)
    if not topic or not field:
        return False
    invalid_chars = set(" +-*/()")
    return not any(char in invalid_chars for char in value)
