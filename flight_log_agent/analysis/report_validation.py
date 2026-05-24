from __future__ import annotations

from flight_log_agent.models import FlightLogReport, ValidationIssue, ValidationResult


def validate_report(report: FlightLogReport) -> ValidationResult:
    issues: list[ValidationIssue] = []

    for i, hyp in enumerate(report.ranked_hypotheses):
        path = f"ranked_hypotheses[{i}]"
        if hyp.confidence in ("high", "medium") and not hyp.source_refs:
            issues.append(ValidationIssue(
                severity="error",
                path=f"{path}.source_refs",
                message="Medium/high confidence mechanism has no source references.",
            ))
        if hyp.confidence in ("high", "medium") and not hyp.numeric_checks:
            issues.append(ValidationIssue(
                severity="error",
                path=f"{path}.numeric_checks",
                message="Medium/high confidence mechanism has no numeric checks.",
            ))
        if hyp.confidence in ("high", "medium") and hyp.applicability.missing_required_signals:
            issues.append(ValidationIssue(
                severity="error",
                path=f"{path}.applicability.missing_required_signals",
                message="Medium/high confidence mechanism is missing required signals.",
            ))
        if not hyp.expected_logged_signature:
            issues.append(ValidationIssue(
                severity="error",
                path=f"{path}.expected_logged_signature",
                message="Mechanism has no expected logged signature.",
            ))

    return ValidationResult(
        passed=not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


def enforce_validation_downgrades(report: FlightLogReport, validation: ValidationResult) -> FlightLogReport:
    for issue in validation.issues:
        if issue.severity != "error":
            continue
        if not issue.path.startswith("ranked_hypotheses["):
            continue
        idx_str = issue.path.split("[", 1)[1].split("]", 1)[0]
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        if 0 <= idx < len(report.ranked_hypotheses):
            hyp = report.ranked_hypotheses[idx]
            if hyp.confidence in ("high", "medium"):
                hyp.confidence = "low"
                hyp.contradicting_evidence.append(f"Confidence downgraded by validation: {issue.message}")
    return report
