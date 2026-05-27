from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from flight_log_agent.models import CodeRef, RelationshipCheckSpec


class SourceDiscoveryLogContext(BaseModel):
    """
    Static log facts available to source discovery.

    The iterative source resolver may use these facts to narrow source branches,
    but it must not perform dynamic time-series verification.
    """
    parameters: dict[str, Any] = Field(default_factory=dict)
    topic_fields: dict[str, list[str]] = Field(default_factory=dict)
    available_topics: list[str] = Field(default_factory=list)
    vehicle_type: Optional[str] = None
    mode_state_constraints: dict[str, Any] = Field(default_factory=dict)


class TopicFieldRef(BaseModel):
    topic: str
    field: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None


class SourceFieldRef(BaseModel):
    field: str
    topic: Optional[str] = None
    variable: Optional[str] = None
    source_file: str
    source_line: int


class ParameterRequirement(BaseModel):
    name: str
    role: Literal["branch_selector", "threshold", "tuning_or_shaping", "unknown"]
    source_predicate: Optional[str] = None
    actual_value: float | int | str | bool | None = None
    gate_result: Literal["satisfied", "contradicted", "unknown", "verification_required"]
    effect: str
    source_file: str
    source_line: Optional[int] = None


class SourceSnippet(BaseModel):
    file: str
    start_line: int
    end_line: int
    text: str


class SourceBackedParameterPredicate(BaseModel):
    name: str
    role: Literal["branch_selector", "threshold", "tuning_or_shaping", "unknown"]
    predicate: str
    operator: Optional[Literal[">", ">=", "<", "<=", "==", "!="]] = None
    compared_value: float | int | str | bool | None = None
    effect: str = ""
    source_file: Optional[str] = None
    source_line: Optional[int] = None


class SourceBackedVerificationCheck(BaseModel):
    check: RelationshipCheckSpec
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    rationale: str = ""


class SourceMechanismCandidate(BaseModel):
    title: str
    source_mechanism: str
    source_chain: list[CodeRef] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)
    controlling_parameters: list[ParameterRequirement] = Field(default_factory=list)
    published_topics: list[TopicFieldRef] = Field(default_factory=list)
    subscribed_topics: list[TopicFieldRef] = Field(default_factory=list)
    relevant_fields: list[SourceFieldRef] = Field(default_factory=list)
    branch_conditions: list[str] = Field(default_factory=list)
    expected_log_signature: list[str] = Field(default_factory=list)
    required_log_evidence: list[str] = Field(default_factory=list)
    interpreted_parameter_predicates: list[SourceBackedParameterPredicate] = Field(default_factory=list)
    verification_checks: list[SourceBackedVerificationCheck] = Field(default_factory=list)
    contradiction_checks: list[str] = Field(default_factory=list)
    source_confidence: Literal["low", "medium", "high"] = "low"
    resolver_notes: list[str] = Field(default_factory=list)


class SourceMechanismCandidateSet(BaseModel):
    candidates: list[SourceMechanismCandidate] = Field(default_factory=list)
    expansion_queries: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class SourceDiscoveryCandidateDraft(BaseModel):
    title: str
    source_mechanism: str
    source_files: list[str] = Field(default_factory=list)
    source_chain: list[CodeRef] = Field(default_factory=list)
    controlling_parameter_names: list[str] = Field(default_factory=list)
    relevant_signals: list[str] = Field(default_factory=list)
    branch_conditions: list[str] = Field(default_factory=list)
    expected_log_signature: list[str] = Field(default_factory=list)
    required_log_evidence: list[str] = Field(default_factory=list)
    interpreted_parameter_predicates: list[SourceBackedParameterPredicate] = Field(default_factory=list)
    verification_checks: list[SourceBackedVerificationCheck] = Field(default_factory=list)
    contradiction_checks: list[str] = Field(default_factory=list)
    source_confidence: Literal["low", "medium", "high"] = "low"
    resolver_notes: list[str] = Field(default_factory=list)


class SourceDiscoveryIterationPacket(BaseModel):
    user_question: str
    depth: int
    active_queries: list[str] = Field(default_factory=list)
    visited_files: list[str] = Field(default_factory=list)
    new_files: list[str] = Field(default_factory=list)
    source_profile: dict[str, Any] = Field(default_factory=dict)
    parameter_requirements: list[ParameterRequirement] = Field(default_factory=list)
    static_log_context: dict[str, Any] = Field(default_factory=dict)
    prior_decision_notes: list[str] = Field(default_factory=list)


class SourceDiscoveryDecision(BaseModel):
    relevant_files: list[str] = Field(default_factory=list)
    expansion_queries: list[str] = Field(default_factory=list)
    stop: bool = False
    candidate_drafts: list[SourceDiscoveryCandidateDraft] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
