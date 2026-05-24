from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from flight_log_agent.models import ApplicabilityResult, MechanismCandidate, SignatureEvaluation
from flight_log_agent.ulog.signature_evaluator import evaluate_log_signature


def evaluate_candidate_log_signature(
    log_path: Path,
    candidate: MechanismCandidate,
    applicability: ApplicabilityResult,
) -> SignatureEvaluation:
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
    )
    return normalize_signature_evaluation(candidate.name, raw, applicability)


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
    if applicability.missing_required_signals:
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
