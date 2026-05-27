from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CodeRef(BaseModel):
    file: str
    function: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    snippet: Optional[str] = None
    explanation: str = ""


class PlotOverlay(BaseModel):
    start_s: float
    end_s: Optional[float] = None
    label: Optional[str] = None
    kind: Optional[str] = None
    source: Optional[str] = None
    color: Optional[str] = None
    alpha: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None


class PlotRef(BaseModel):
    title: str
    path: str = ""
    purpose: str
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    signals: list[str] = Field(default_factory=list)
    plot_type: str = "timeseries"
    bins: int = 50
    overlays: list[PlotOverlay] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AirframeContext(BaseModel):
    """
    Minimal aircraft context for mechanism discovery.
    vehicle_type should already include VTOL subtype, e.g.:
      - "fixed_wing"
      - "multicopter"
      - "vtol_standard"
      - "vtol_tailsitter"
      - "vtol_tiltrotor"
    """
    px4_git_hash: Optional[str] = None
    px4_version: Optional[str] = None
    px4_tag: Optional[str] = None
    vehicle_type: str = "unknown"
    sys_autostart: Optional[int] = None
    airframe_name: Optional[str] = None
    control_surface_summary: str = "unknown"


class QuestionIntent(BaseModel):
    """
    Produced from natural-language user question.
    No parameters/log data are used to verify anything here.
    """
    original_question: str
    problem_domain: str
    concise_intent: str
    source_queries: list[str]
    likely_modules: list[str] = Field(default_factory=list)
    likely_source_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SourceSearchContext(BaseModel):
    """
    The only context allowed into source mechanism discovery.
    No parameter values, no topic catalog, no detailed timeline.
    """
    airframe: AirframeContext
    question_intent: QuestionIntent


class SourceHit(BaseModel):
    query: str
    file: str
    line: Optional[int] = None
    snippet: str


class SourceEvidenceBundle(BaseModel):
    search_context: SourceSearchContext
    hits: list[SourceHit]
    read_snippets: list[CodeRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExpectedSignatureItem(BaseModel):
    name: str
    description: str
    signal: Optional[str] = None
    expected_behavior: Optional[str] = None


class WindowSpec(BaseModel):
    name: str
    start_s: float
    end_s: float
    reason: Optional[str] = None


class RelationshipCheckSpec(BaseModel):
    type: Literal[
        "threshold",
        "transition_occurs",
        "no_transition",
        "state_equals",
        "state_not_equals",
        "tracks_setpoint",
        "diverges_from_setpoint",
        "monotonic_change",
        "same_direction_change",
        "parameter_equals",
        "branch_parameter_satisfied",
        "tracks_parameter_value",
        "topic_field_present",
        "custom",
    ]
    window: Optional[str] = None
    signal: Optional[str] = None
    parameter: Optional[str] = None
    source_predicate: Optional[str] = None
    first: Optional[str] = None
    second: Optional[str] = None
    actual: Optional[str] = None
    setpoint: Optional[str] = None
    metric: Optional[str] = None
    op: Optional[Literal[">", ">=", "<", "<=", "==", "!=", "between", "outside"]] = None
    value: Optional[float | int | str | bool] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    from_value: Optional[float | int | str | bool] = None
    to_value: Optional[float | int | str | bool] = None
    mode: Optional[Literal["any", "all"]] = None
    direction: Optional[Literal["increase", "decrease", "above", "below", "absolute"]] = None
    max_error: Optional[float] = None
    min_error: Optional[float] = None
    min_delta: Optional[float] = None
    supports: Optional[str] = None
    contradicts: Optional[str] = None
    description: Optional[str] = None


class MechanismCandidate(BaseModel):
    """
    Candidate produced from PX4 source. This is still only a possible mechanism.
    Parameters/topics/log data have not verified it yet.
    """
    name: str
    summary: str
    source_refs: list[CodeRef]

    vehicle_type_gates: list[str] = Field(default_factory=list)
    airframe_gates: list[str] = Field(default_factory=list)
    mode_state_gates: list[str] = Field(default_factory=list)
    parameter_gates: list[str] = Field(default_factory=list)

    required_parameters: list[str] = Field(default_factory=list)
    required_signals: list[str] = Field(default_factory=list)
    expected_logged_signature: list[ExpectedSignatureItem] = Field(default_factory=list)
    exclusion_checks: list[RelationshipCheckSpec] = Field(default_factory=list)
    numeric_checks: list[RelationshipCheckSpec] = Field(default_factory=list)
    plot_requests: list[PlotRef] = Field(default_factory=list)


class MechanismCandidateSet(BaseModel):
    candidates: list[MechanismCandidate]
    rejected_source_paths: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class ApplicabilityResult(BaseModel):
    candidate_name: str
    applicable: bool
    supported_conditions: list[str] = Field(default_factory=list)
    excluded_by: list[str] = Field(default_factory=list)
    unresolved_conditions: list[str] = Field(default_factory=list)
    relevant_parameters: dict[str, Any] = Field(default_factory=dict)
    candidate_windows: list[WindowSpec] = Field(default_factory=list)
    available_required_signals: list[str] = Field(default_factory=list)
    missing_required_signals: list[str] = Field(default_factory=list)


class SignatureEvaluation(BaseModel):
    candidate_name: str
    verdict: Literal["supported", "contradicted", "mixed", "unresolved"]
    confidence_ceiling: Literal["high", "medium", "low", "unresolved"]
    evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    check_results: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class VerifiedMechanismResult(BaseModel):
    candidate: MechanismCandidate
    applicability: ApplicabilityResult
    evaluation: SignatureEvaluation
    final_confidence: Literal["high", "medium", "low", "unresolved"]


class ParameterValue(BaseModel):
    name: str
    value: str


class ApplicabilityReport(BaseModel):
    applicable: bool
    supported_conditions: list[str] = Field(default_factory=list)
    excluded_by: list[str] = Field(default_factory=list)
    unresolved_conditions: list[str] = Field(default_factory=list)
    relevant_parameters: list[ParameterValue] = Field(default_factory=list)
    available_required_signals: list[str] = Field(default_factory=list)
    missing_required_signals: list[str] = Field(default_factory=list)


class HypothesisReportItem(BaseModel):
    title: str
    known_px4_mechanism: str
    mechanism: str
    source_refs: list[CodeRef]
    expected_logged_signature: list[ExpectedSignatureItem]
    applicability: ApplicabilityReport
    evidence: list[str]
    contradicting_evidence: list[str]
    exclusion_checks: list[RelationshipCheckSpec]
    numeric_checks: list[RelationshipCheckSpec]
    confidence: Literal["high", "medium", "low", "unresolved"]
    plots: list[PlotRef] = Field(default_factory=list)


class FlightLogReport(BaseModel):
    airframe_summary: str
    question_intent_summary: str
    ranked_hypotheses: list[HypothesisReportItem]
    excluded_mechanisms: list[str]
    confirmed: list[str]
    unconfirmed: list[str]
    final_summary: str


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    path: str
    message: str


class ValidationResult(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
