from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from agents import (
    Agent,
    ModelSettings,
    Runner,
    ShellTool,
    Usage,
    WebSearchTool,
)
from pydantic import BaseModel, Field

from flight_log_agent.audit import (
    AgentRunAuditHooks,
    DeveloperAuditLogger,
    log_run_items,
)
from flight_log_agent.models import (
    ApplicabilityReport,
    CodeRef,
    ExpectedSignatureItem,
    FlightLogReport,
    HypothesisReportItem,
    ParameterValue,
    PlotRef,
    RelationshipCheckSpec,
)
from flight_log_agent.px4.source_snapshot import SourceSnapshot
from flight_log_agent.symbols import parse_signal_reference


EvidenceStateName = Literal[
    "sufficient",
    "needs_web_research",
    "unresolved",
]
ShellToolFactory = Callable[[bool], ShellTool]
WEB_FALLBACK_MAX_TURNS = 6
MAX_WEB_QUERIES = 3
MAX_WEB_QUERY_CHARS = 240
MAX_RECEIPT_STDOUT_CHARS = 600_000
MAX_PUBLIC_SOURCE_OPERATION_CHARS = 240
SOURCE_ROOT_ALIAS = "PX4-Autopilot"
T = TypeVar("T", bound=BaseModel)
_CPP_RAW_STRING_OPENING_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:u8|u|U|L)?R\""
    r"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\("
)


@dataclass(frozen=True)
class _LoggedFieldComponent:
    name: str
    indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _SourceFieldComponent:
    name: str
    index_expressions: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SchemaFieldDefinition:
    type_name: str
    array_extents: tuple[str, ...] = ()


class LogObservation(BaseModel):
    finding: str
    evidence: str
    topic_fields: list[str] = Field(default_factory=list)
    time_or_window: str | None = None


class NamedValue(BaseModel):
    name: str
    value: str


class ShellExecutionReceipt(BaseModel):
    command: str
    exit_code: int | None = None
    timed_out: bool = False
    receipt_id: str = ""
    stdout_sha256: str = ""
    stdout_chars: int = 0
    stdout: str = Field(default="", exclude=True)


class InitialInspection(BaseModel):
    question_is_causal: bool
    question_focus: str
    firmware_and_version: str
    log_duration_and_timebase: str
    observations: list[LogObservation] = Field(default_factory=list)
    relevant_parameters: list[NamedValue] = Field(default_factory=list)
    relevant_topic_fields: list[str] = Field(default_factory=list)
    events_and_messages: list[str] = Field(default_factory=list)
    likely_windows: list[str] = Field(default_factory=list)
    source_search_terms: list[str] = Field(default_factory=list)
    missing_or_uncertain: list[str] = Field(default_factory=list)
    execution_receipts: list[ShellExecutionReceipt] = Field(
        default_factory=list
    )


class SourceLineageStep(BaseModel):
    sequence: int
    file: str
    symbol: str | None = None
    lines: str
    operation: str
    source_excerpt: str
    input_or_state: str
    output_or_effect: str
    runtime_conditions: list[str] = Field(default_factory=list)
    execution_commands: list[str] = Field(default_factory=list)
    role: Literal[
        "origin",
        "transformation",
        "constraint",
        "branch",
        "propagation",
        "publication",
    ]


class CandidateMechanism(BaseModel):
    candidate_id: str
    title: str
    mechanism: str
    explained_logged_value: str
    upstream_assignment_path: list[SourceLineageStep] = Field(default_factory=list)
    causal_transformation: str
    downstream_propagation: list[str] = Field(default_factory=list)
    runtime_conditions: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    expected_log_signature: list[str] = Field(default_factory=list)
    contradicting_log_signature: list[str] = Field(default_factory=list)
    required_topic_fields: list[str] = Field(default_factory=list)
    required_parameters: list[str] = Field(default_factory=list)
    discriminating_calculations: list[str] = Field(default_factory=list)
    source_gaps: list[str] = Field(default_factory=list)


class SchemaSourceBinding(BaseModel):
    schema_file: str
    schema_lines: str
    schema_excerpt: str
    schema_execution_commands: list[str] = Field(default_factory=list)


class PublicationBinding(SchemaSourceBinding):
    logged_value: str
    message_identifier: str
    message_type_identifier: str
    declaration_file: str
    declaration_lines: str
    declaration_excerpt: str
    declaration_execution_commands: list[str] = Field(default_factory=list)
    nested_schema_bindings: list[SchemaSourceBinding] = Field(
        default_factory=list
    )


class SourceInvestigation(BaseModel):
    source_available: bool
    snapshot_identity: str | None = None
    explained_logged_value: str
    publishing_assignment: SourceLineageStep | None = None
    publication_binding: PublicationBinding | None = None
    candidates: list[CandidateMechanism] = Field(default_factory=list)
    propagation_only_matches: list[str] = Field(default_factory=list)
    unresolved_reason: str | None = None
    execution_receipts: list[ShellExecutionReceipt] = Field(
        default_factory=list
    )


class EvidenceCheck(BaseModel):
    check_id: str
    candidate_id: str
    evidence_role: Literal[
        "causal_discriminator",
        "downstream_consistency",
        "observation",
        "contradiction",
    ]
    topic_fields: list[str] = Field(default_factory=list)
    window: str
    source_prediction: str
    method: str
    result: str
    evaluated_requirement_ids: list[str] = Field(default_factory=list)
    execution_commands: list[str] = Field(default_factory=list)
    assessment: Literal[
        "supports",
        "contradicts",
        "ambiguous",
        "not_evaluable",
    ]


class TargetedLogParse(BaseModel):
    explained_logged_value: str
    event_windows: list[str] = Field(default_factory=list)
    parameter_values: list[NamedValue] = Field(default_factory=list)
    checks: list[EvidenceCheck] = Field(default_factory=list)
    missing_topic_fields: list[str] = Field(default_factory=list)
    additional_source_questions: list[str] = Field(default_factory=list)
    external_context_questions: list[str] = Field(default_factory=list)
    execution_receipts: list[ShellExecutionReceipt] = Field(
        default_factory=list
    )


class ReviewedEvidenceCheck(BaseModel):
    check_id: str
    assessment: Literal[
        "supports",
        "contradicts",
        "ambiguous",
        "not_evaluable",
    ]
    independent_result: str
    execution_commands: list[str] = Field(default_factory=list)


class ReviewedObservation(BaseModel):
    observation_index: int
    assessment: Literal[
        "supports",
        "contradicts",
        "ambiguous",
        "not_evaluable",
    ]
    independent_result: str
    execution_commands: list[str] = Field(default_factory=list)


class ReviewedPublicationBinding(PublicationBinding):
    assessment: Literal["verified", "unresolved", "contradicted"]
    reasoning: str


class ReviewedSourceLineageStep(BaseModel):
    sequence: int
    file: str
    source_excerpt: str
    input_tokens: list[str] = Field(default_factory=list)
    output_tokens: list[str] = Field(default_factory=list)
    direction_assessment: Literal[
        "verified",
        "reversed",
        "ambiguous",
    ]
    reasoning: str


class CandidateReview(BaseModel):
    candidate_id: str
    verdict: Literal[
        "supported",
        "partially_supported",
        "contradicted",
        "unresolved",
    ]
    confidence: Literal["high", "medium", "low", "unresolved"]
    source_path_complete: bool
    source_verification_commands: list[str] = Field(default_factory=list)
    reviewed_source_lineage: list[ReviewedSourceLineageStep] = Field(
        default_factory=list
    )
    reviewed_checks: list[ReviewedEvidenceCheck] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    reasoning: str


class EvidenceReview(BaseModel):
    question_is_causal: bool
    explained_logged_value: str
    local_observations_sufficient: bool
    reviewed_observations: list[ReviewedObservation] = Field(
        default_factory=list
    )
    reviewed_publication_binding: ReviewedPublicationBinding | None = None
    candidate_reviews: list[CandidateReview] = Field(default_factory=list)
    preferred_candidate_id: str | None = None
    requested_state: EvidenceStateName
    reason: str
    web_gap_kind: Literal[
        "none",
        "external_context",
        "local_source",
        "local_log",
    ] = "none"
    web_queries: list[str] = Field(default_factory=list)
    next_source_queries: list[str] = Field(default_factory=list)
    execution_receipts: list[ShellExecutionReceipt] = Field(
        default_factory=list
    )


class EvidenceState(BaseModel):
    cycle: int
    status: EvidenceStateName
    question_is_causal: bool
    accepted_candidate_ids: list[str] = Field(default_factory=list)
    accepted_candidate_titles: list[str] = Field(default_factory=list)
    accepted_observation_indices: list[int] = Field(default_factory=list)
    excluded_candidate_ids: list[str] = Field(default_factory=list)
    excluded_candidate_titles: list[str] = Field(default_factory=list)
    reason: str
    web_queries: list[str] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)


class WebFinding(BaseModel):
    title: str
    url: str
    source_kind: Literal[
        "px4_docs",
        "px4_github",
        "px4_forum",
        "primary",
        "secondary",
    ]
    version_scope: str
    relevant_claim: str


class WebResearch(BaseModel):
    queries_used: list[str] = Field(default_factory=list)
    observed_queries: list[str] = Field(default_factory=list)
    findings: list[WebFinding] = Field(default_factory=list)
    source_followup_queries: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    observed_citation_urls: list[str] = Field(default_factory=list)
    error: str | None = None


class InvestigationCycle(BaseModel):
    cycle: int
    source: SourceInvestigation
    targeted_log: TargetedLogParse
    review: EvidenceReview
    evidence_state: EvidenceState
    web_research: WebResearch | None = None


@dataclass
class StagedAnalysisResult:
    report: FlightLogReport
    usage: Usage
    evidence_state: EvidenceState


@dataclass(frozen=True)
class ModelRequestBudget:
    limit: int


class _UsageTrackingAuditHooks(AgentRunAuditHooks):
    def __init__(self, audit_logger: DeveloperAuditLogger) -> None:
        super().__init__(audit_logger)
        self.observed_usage = Usage()

    async def on_llm_end(
        self,
        context: Any,
        agent: Any,
        response: Any,
    ) -> None:
        await super().on_llm_end(context, agent, response)
        response_usage = getattr(response, "usage", None)
        if isinstance(response_usage, Usage):
            self.observed_usage.add(response_usage)


def derive_evidence_state(
    *,
    initial: InitialInspection,
    source: SourceInvestigation,
    targeted: TargetedLogParse,
    review: EvidenceReview,
    cycle: int,
    allow_web_research: bool,
) -> EvidenceState:
    """Turn model-authored review records into a fail-closed workflow state."""

    _canonicalize_source_investigation_files(source)
    issues: list[str] = []
    fatal_issues: list[str] = []
    question_is_causal = (
        initial.question_is_causal or review.question_is_causal
    )
    if initial.question_is_causal != review.question_is_causal:
        issue = (
            "causal classification changed between initial inspection and "
            "independent review"
        )
        issues.append(issue)
        fatal_issues.append(issue)

    expected_logged_value = initial.question_focus.strip()
    for stage_name, stage_value in (
        ("source investigation", source.explained_logged_value),
        ("targeted log parse", targeted.explained_logged_value),
        ("independent review", review.explained_logged_value),
    ):
        if stage_value.strip() != expected_logged_value:
            issue = (
                f"{stage_name} changed the analysis target from "
                f"{expected_logged_value!r} to {stage_value.strip()!r}"
            )
            issues.append(issue)
            fatal_issues.append(issue)

    candidate_by_id: dict[str, CandidateMechanism] = {}
    candidate_ids_with_invalid_lineage: set[str] = set()
    candidate_ids_with_invalid_evidence: set[str] = set()
    candidate_id_by_title: dict[str, str] = {}
    for stage_name, receipts in (
        ("initial inspection", initial.execution_receipts),
        ("source investigation", source.execution_receipts),
        ("targeted log parse", targeted.execution_receipts),
        ("independent review", review.execution_receipts),
    ):
        for receipt in receipts:
            if _receipt_integrity_valid(receipt):
                continue
            issue = f"{stage_name} contains a modified execution receipt"
            issues.append(issue)
            fatal_issues.append(issue)
    source_commands = _successful_execution_commands(
        source.execution_receipts
    )
    initial_commands = _successful_execution_commands(
        initial.execution_receipts
    )
    targeted_commands = _successful_execution_commands(
        targeted.execution_receipts
    )
    review_commands = _successful_execution_commands(
        review.execution_receipts
    )
    if not any(
        _command_program(command) in {"python", "python3"}
        for command in initial_commands
    ):
        issue = "initial inspection has no successful ULog Python execution"
        issues.append(issue)
        fatal_issues.append(issue)
    initial_target_references = {
        field
        for observation in initial.observations
        for field in observation.topic_fields
    } | set(initial.relevant_topic_fields)
    initial_target_text = [
        *(observation.finding for observation in initial.observations),
        *(observation.evidence for observation in initial.observations),
        *initial.events_and_messages,
    ]
    if (
        not expected_logged_value
        or (
            expected_logged_value not in initial_target_references
            and not any(
                expected_logged_value in text
                for text in initial_target_text
            )
        )
    ):
        issue = (
            "initial analysis target is not bound to an observed topic, "
            "event, or message"
        )
        issues.append(issue)
        fatal_issues.append(issue)
    if not any(
        (
            expected_logged_value in observation.topic_fields
            or _text_contains_focus(
                observation.finding,
                expected_logged_value,
            )
            or _text_contains_focus(
                observation.evidence,
                expected_logged_value,
            )
        )
        and _receipts_contain_json_record(
            initial.execution_receipts,
            name="observation",
            expected=_observation_receipt_payload(
                observation_index,
                observation,
                question_focus=expected_logged_value,
            ),
            programs={"python", "python3"},
        )
        for observation_index, observation in enumerate(initial.observations)
    ):
        issue = (
            "initial analysis target was not present in an exact successful "
            "ULog observation record"
        )
        issues.append(issue)
        fatal_issues.append(issue)
    if not any(
        _command_program(command) in {"python", "python3"}
        for command in targeted_commands
    ):
        issue = "targeted log parse has no successful ULog Python execution"
        issues.append(issue)
        fatal_issues.append(issue)
    if not any(
        _command_program(command) in {"python", "python3"}
        for command in review_commands
    ):
        issue = "independent review has no successful ULog Python execution"
        issues.append(issue)
        fatal_issues.append(issue)

    for candidate in source.candidates:
        if candidate.candidate_id in candidate_by_id:
            issue = f"duplicate candidate id: {candidate.candidate_id}"
            issues.append(issue)
            fatal_issues.append(issue)
            continue
        candidate_by_id[candidate.candidate_id] = candidate
        previous_title_id = candidate_id_by_title.get(candidate.title)
        if previous_title_id is not None:
            issue = (
                f"duplicate candidate title: {candidate.title} "
                f"({previous_title_id}, {candidate.candidate_id})"
            )
            issues.append(issue)
            fatal_issues.append(issue)
            candidate_ids_with_invalid_lineage.update(
                {previous_title_id, candidate.candidate_id}
            )
        else:
            candidate_id_by_title[candidate.title] = candidate.candidate_id

        lineage = candidate.upstream_assignment_path
        lineage_issue = False
        if not source.source_available or source.snapshot_identity is None:
            issues.append(
                f"candidate {candidate.candidate_id} has no exact source snapshot"
            )
            lineage_issue = True
        if source.publishing_assignment is None:
            issues.append(
                f"candidate {candidate.candidate_id} has no publishing assignment"
            )
            lineage_issue = True
        if not lineage:
            issues.append(
                f"candidate {candidate.candidate_id} has an empty source lineage"
            )
            lineage_issue = True
        else:
            sequences = [step.sequence for step in lineage]
            if sequences != sorted(sequences) or len(sequences) != len(
                set(sequences)
            ):
                issues.append(
                    f"candidate {candidate.candidate_id} has an invalid lineage order"
                )
                lineage_issue = True
            if not _lineage_has_connected_data_flow(lineage):
                issues.append(
                    f"candidate {candidate.candidate_id} source lineage has "
                    "disconnected or reversed data flow"
                )
                lineage_issue = True
            if not any(
                step.role
                in {"origin", "transformation", "constraint", "branch"}
                for step in lineage
            ):
                issues.append(
                    f"candidate {candidate.candidate_id} has only propagation "
                    "or publication steps"
                )
                lineage_issue = True
            for step in lineage:
                source_file_commands = [
                    command
                    for command in step.execution_commands
                    if _source_command_reads_file(command, step.file)
                ]
                if not step.execution_commands:
                    issues.append(
                        f"candidate {candidate.candidate_id} lineage step "
                        f"{step.sequence} has no source execution command"
                    )
                    lineage_issue = True
                elif not set(step.execution_commands).issubset(
                    source_commands
                ):
                    issues.append(
                        f"candidate {candidate.candidate_id} lineage step "
                        f"{step.sequence} cites an unexecuted source command"
                    )
                    lineage_issue = True
                elif not any(
                    _command_program(command) == "git"
                    for command in step.execution_commands
                ):
                    issues.append(
                        f"candidate {candidate.candidate_id} lineage step "
                        f"{step.sequence} is not backed by a pinned Git read"
                    )
                    lineage_issue = True
                elif not source_file_commands:
                    issues.append(
                        f"candidate {candidate.candidate_id} lineage step "
                        f"{step.sequence} was not read from its cited file"
                    )
                    lineage_issue = True
                elif not step.source_excerpt.strip() or not (
                    _commands_contain_exact_text(
                        source.execution_receipts,
                        source_file_commands,
                        step.source_excerpt,
                    )
                ):
                    issues.append(
                        f"candidate {candidate.candidate_id} lineage step "
                        f"{step.sequence} excerpt was not returned by its "
                        "source command"
                    )
                    lineage_issue = True
                elif not _source_line_range_matches(
                    source.execution_receipts,
                    source_file_commands,
                    step.source_excerpt,
                    step.lines,
                ):
                    issues.append(
                        f"candidate {candidate.candidate_id} lineage step "
                        f"{step.sequence} line range does not match the "
                        "returned source excerpt"
                    )
                    lineage_issue = True
                elif not _source_step_fields_are_grounded(step):
                    issues.append(
                        f"candidate {candidate.candidate_id} lineage step "
                        f"{step.sequence} operation or data flow is not "
                        "grounded in its source excerpt"
                    )
                    lineage_issue = True
                elif not _source_assignment_direction_is_valid(step):
                    issues.append(
                        f"candidate {candidate.candidate_id} lineage step "
                        f"{step.sequence} reverses the source assignment "
                        "direction"
                    )
                    lineage_issue = True
            if source.publishing_assignment is not None:
                endpoint = lineage[-1]
                publishing = source.publishing_assignment
                if not _source_lineage_steps_match(
                    endpoint,
                    publishing,
                ):
                    issues.append(
                        f"candidate {candidate.candidate_id} does not terminate "
                        "at the recorded publishing assignment"
                    )
                    lineage_issue = True
        if (
            candidate.explained_logged_value.strip()
            != source.explained_logged_value.strip()
        ):
            issues.append(
                f"candidate {candidate.candidate_id} explains a different "
                "logged value"
            )
            lineage_issue = True
        if candidate.explained_logged_value.strip() != expected_logged_value:
            issues.append(
                f"candidate {candidate.candidate_id} changed the analysis target"
            )
            lineage_issue = True
        if candidate.source_gaps:
            issues.append(
                f"candidate {candidate.candidate_id} has unresolved source gaps"
            )
            lineage_issue = True
        if lineage_issue:
            candidate_ids_with_invalid_lineage.add(candidate.candidate_id)

    publishing_field = _logged_value_field_identifier(
        expected_logged_value
    )
    if (
        question_is_causal
        and source.source_available
        and publishing_field is not None
        and (
            source.publishing_assignment is None
            or publishing_field
            not in _source_data_tokens(
                source.publishing_assignment.output_or_effect
            )
            or publishing_field
            not in _source_data_tokens(
                source.publishing_assignment.source_excerpt
            )
        )
    ):
        issue = (
            "the recorded publishing assignment does not contain the "
            "source field for the initial logged value"
        )
        issues.append(issue)
        fatal_issues.append(issue)
    if question_is_causal and source.source_available:
        binding = source.publication_binding
        if (
            binding is None
            or source.publishing_assignment is None
            or not _publication_binding_is_valid(
                binding,
                publishing_assignment=source.publishing_assignment,
                expected_logged_value=expected_logged_value,
                receipts=source.execution_receipts,
                successful_commands=source_commands,
            )
        ):
            issue = (
                "source investigation did not bind the publishing message "
                "type and schema to the initial logged value"
            )
            issues.append(issue)
            fatal_issues.append(issue)
        reviewed_binding = review.reviewed_publication_binding
        if (
            binding is None
            or reviewed_binding is None
            or reviewed_binding.assessment != "verified"
            or not _publication_bindings_match(
                binding,
                reviewed_binding,
            )
            or source.publishing_assignment is None
            or not _publication_binding_is_valid(
                reviewed_binding,
                publishing_assignment=source.publishing_assignment,
                expected_logged_value=expected_logged_value,
                receipts=review.execution_receipts,
                successful_commands=review_commands,
            )
        ):
            issue = (
                "independent review did not verify the publishing message "
                "type and schema binding"
            )
            issues.append(issue)
            fatal_issues.append(issue)

    seen_parameter_names: set[str] = set()
    for parameter in targeted.parameter_values:
        if parameter.name in seen_parameter_names:
            issue = (
                "targeted log parse contains duplicate parameter value: "
                f"{parameter.name}"
            )
            issues.append(issue)
            fatal_issues.append(issue)
        seen_parameter_names.add(parameter.name)

    targeted_check_by_id: dict[str, EvidenceCheck] = {}
    invalid_check_ids: set[str] = set()
    for check in targeted.checks:
        if check.check_id in targeted_check_by_id:
            issue = f"duplicate targeted check id: {check.check_id}"
            issues.append(issue)
            fatal_issues.append(issue)
            continue
        targeted_check_by_id[check.check_id] = check
        if check.candidate_id not in candidate_by_id:
            issue = (
                f"targeted check {check.check_id} refers to unknown candidate "
                f"{check.candidate_id}"
            )
            issues.append(issue)
            fatal_issues.append(issue)
            continue
        candidate_requirement_ids = set(
            _candidate_runtime_requirement_ids(
                candidate_by_id[check.candidate_id]
            )
        )
        unknown_requirement_ids = (
            set(check.evaluated_requirement_ids)
            - candidate_requirement_ids
        )
        if unknown_requirement_ids:
            issues.append(
                f"targeted check {check.check_id} cites unknown runtime "
                f"requirements: {sorted(unknown_requirement_ids)}"
            )
            candidate_ids_with_invalid_evidence.add(check.candidate_id)
            invalid_check_ids.add(check.check_id)
        if not all(
            value.strip()
            for value in (
                check.window,
                check.source_prediction,
                check.method,
                check.result,
            )
        ):
            issues.append(
                f"targeted check {check.check_id} has incomplete performed evidence"
            )
            candidate_ids_with_invalid_evidence.add(check.candidate_id)
            invalid_check_ids.add(check.check_id)
        if not check.execution_commands:
            issues.append(
                f"targeted check {check.check_id} has no execution command"
            )
            candidate_ids_with_invalid_evidence.add(check.candidate_id)
            invalid_check_ids.add(check.check_id)
        elif not set(check.execution_commands).issubset(targeted_commands):
            issues.append(
                f"targeted check {check.check_id} cites an unexecuted command"
            )
            candidate_ids_with_invalid_evidence.add(check.candidate_id)
            invalid_check_ids.add(check.check_id)
        elif not any(
            _command_program(command) in {"python", "python3"}
            for command in check.execution_commands
        ):
            issues.append(
                f"targeted check {check.check_id} was not computed from the ULog"
            )
            candidate_ids_with_invalid_evidence.add(check.candidate_id)
            invalid_check_ids.add(check.check_id)
        elif not _commands_contain_json_record(
            targeted.execution_receipts,
            check.execution_commands,
            name="check",
            expected=_targeted_check_receipt_payload(
                check,
                candidate_by_id[check.candidate_id],
                targeted,
            ),
            programs={"python", "python3"},
        ):
            issues.append(
                f"targeted check {check.check_id} result was not returned "
                "with all cited fields and requirements by one Python command"
            )
            candidate_ids_with_invalid_evidence.add(check.candidate_id)
            invalid_check_ids.add(check.check_id)
        for requirement_id in check.evaluated_requirement_ids:
            if not _commands_contain_json_record(
                targeted.execution_receipts,
                check.execution_commands,
                name="check",
                expected=_targeted_check_receipt_payload(
                    check,
                    candidate_by_id[check.candidate_id],
                    targeted,
                ),
                programs={"python", "python3"},
            ):
                issues.append(
                    f"targeted check {check.check_id} runtime requirement "
                    f"{requirement_id} was not returned by its Python command"
                )
                candidate_ids_with_invalid_evidence.add(check.candidate_id)
                invalid_check_ids.add(check.check_id)
        if (
            check.evidence_role == "causal_discriminator"
            and expected_logged_value not in check.topic_fields
        ):
            issues.append(
                f"targeted check {check.check_id} does not evaluate the "
                "analysis target"
            )
            candidate_ids_with_invalid_evidence.add(check.candidate_id)
            invalid_check_ids.add(check.check_id)
        if check.assessment == "contradicts":
            candidate_ids_with_invalid_evidence.add(check.candidate_id)

    accepted_ids: list[str] = []
    evidence_excluded_ids: set[str] = set()
    seen_review_ids: set[str] = set()
    review_by_candidate_id: dict[str, CandidateReview] = {}
    for candidate_review in review.candidate_reviews:
        candidate_id = candidate_review.candidate_id
        if candidate_id in seen_review_ids:
            issue = f"duplicate candidate review: {candidate_id}"
            issues.append(issue)
            fatal_issues.append(issue)
            continue
        seen_review_ids.add(candidate_id)
        review_by_candidate_id[candidate_id] = candidate_review

        if candidate_id not in candidate_by_id:
            issue = f"review refers to unknown candidate: {candidate_id}"
            issues.append(issue)
            fatal_issues.append(issue)
            continue

        candidate = candidate_by_id[candidate_id]
        source_review_valid = True
        if not candidate_review.source_verification_commands:
            issues.append(
                f"candidate review {candidate_id} has no independent source "
                "verification command"
            )
            source_review_valid = False
        elif not set(
            candidate_review.source_verification_commands
        ).issubset(review_commands):
            issues.append(
                f"candidate review {candidate_id} cites an unexecuted source "
                "verification command"
            )
            source_review_valid = False
        else:
            for step in candidate.upstream_assignment_path:
                step_review_commands = [
                    command
                    for command in (
                        candidate_review.source_verification_commands
                    )
                    if any(
                        _commands_have_same_argv(command, source_command)
                        for source_command in step.execution_commands
                    )
                    and _source_command_reads_file(command, step.file)
                ]
                if not step_review_commands or not (
                    _commands_contain_exact_text(
                        review.execution_receipts,
                        step_review_commands,
                        step.source_excerpt,
                    )
                ):
                    issues.append(
                        f"candidate review {candidate_id} did not independently "
                        f"verify lineage step {step.sequence}"
                    )
                    source_review_valid = False
        reviewed_lineage_by_sequence: dict[
            int,
            ReviewedSourceLineageStep,
        ] = {}
        for reviewed_step in candidate_review.reviewed_source_lineage:
            if reviewed_step.sequence in reviewed_lineage_by_sequence:
                issues.append(
                    f"candidate review {candidate_id} repeats source lineage "
                    f"step {reviewed_step.sequence}"
                )
                source_review_valid = False
                continue
            reviewed_lineage_by_sequence[reviewed_step.sequence] = reviewed_step
        for step in candidate.upstream_assignment_path:
            reviewed_step = reviewed_lineage_by_sequence.get(step.sequence)
            if reviewed_step is None:
                issues.append(
                    f"candidate review {candidate_id} did not review the "
                    f"direction of lineage step {step.sequence}"
                )
                source_review_valid = False
                continue
            reviewed_input_tokens = _source_data_tokens(
                " ".join(reviewed_step.input_tokens)
            )
            reviewed_output_tokens = _source_data_tokens(
                " ".join(reviewed_step.output_tokens)
            )
            excerpt_tokens = _source_data_tokens(step.source_excerpt)
            candidate_input_tokens = _grounded_step_tokens(
                step,
                step.input_or_state,
            )
            candidate_output_tokens = _grounded_step_tokens(
                step,
                step.output_or_effect,
            )
            direction_is_verified = (
                reviewed_step.file == step.file
                and reviewed_step.source_excerpt.strip()
                == step.source_excerpt.strip()
                and reviewed_step.direction_assessment == "verified"
                and bool(reviewed_output_tokens)
                and reviewed_output_tokens.issubset(excerpt_tokens)
                and (
                    bool(reviewed_input_tokens)
                    or (
                        step.role == "origin"
                        and not candidate_input_tokens
                    )
                )
                and reviewed_input_tokens.issubset(excerpt_tokens)
                and candidate_input_tokens.issubset(
                    reviewed_input_tokens
                )
                and candidate_output_tokens.issubset(
                    reviewed_output_tokens
                )
            )
            if not direction_is_verified:
                issues.append(
                    f"candidate review {candidate_id} did not independently "
                    f"verify the data-flow direction of lineage step "
                    f"{step.sequence}"
                )
                source_review_valid = False
        if set(reviewed_lineage_by_sequence) - {
            step.sequence for step in candidate.upstream_assignment_path
        }:
            issues.append(
                f"candidate review {candidate_id} contains an unknown source "
                "lineage step"
            )
            source_review_valid = False
        if not source_review_valid:
            candidate_ids_with_invalid_evidence.add(candidate_id)

        reviewed_check_ids: set[str] = set()
        independently_supported_check_ids: set[str] = set()
        independently_contradicted_check_ids: set[str] = set()
        has_discriminator = False
        for reviewed_check in candidate_review.reviewed_checks:
            check_id = reviewed_check.check_id
            reviewed_check_valid = True
            if check_id in reviewed_check_ids:
                issues.append(
                    f"candidate review {candidate_id} repeats check {check_id}"
                )
                candidate_ids_with_invalid_evidence.add(candidate_id)
                continue
            reviewed_check_ids.add(check_id)
            targeted_check = targeted_check_by_id.get(check_id)
            if targeted_check is None:
                issue = (
                    f"candidate review {candidate_id} refers to unknown targeted "
                    f"check {check_id}"
                )
                issues.append(issue)
                fatal_issues.append(issue)
                continue
            if targeted_check.candidate_id != candidate_id:
                issue = (
                    f"candidate review {candidate_id} refers to check {check_id} "
                    f"owned by {targeted_check.candidate_id}"
                )
                issues.append(issue)
                fatal_issues.append(issue)
                continue
            if check_id in invalid_check_ids:
                issues.append(
                    f"candidate review {candidate_id} relies on invalid "
                    f"targeted check {check_id}"
                )
                candidate_ids_with_invalid_evidence.add(candidate_id)
                reviewed_check_valid = False
            if not reviewed_check.execution_commands:
                issues.append(
                    f"reviewed check {check_id} has no independent execution "
                    "command"
                )
                candidate_ids_with_invalid_evidence.add(candidate_id)
                reviewed_check_valid = False
            elif not set(reviewed_check.execution_commands).issubset(
                review_commands
            ):
                issues.append(
                    f"reviewed check {check_id} cites an unexecuted review "
                    "command"
                )
                candidate_ids_with_invalid_evidence.add(candidate_id)
                reviewed_check_valid = False
            elif not _commands_contain_json_record(
                review.execution_receipts,
                reviewed_check.execution_commands,
                name="reviewed_check",
                expected=_reviewed_check_receipt_payload(
                    reviewed_check,
                    targeted_check,
                ),
                programs={"python", "python3"},
            ):
                issues.append(
                    f"reviewed check {check_id} result was not returned with "
                    "all fields and requirements by one independent Python "
                    "command"
                )
                candidate_ids_with_invalid_evidence.add(candidate_id)
                reviewed_check_valid = False
            if (
                targeted_check.assessment == "contradicts"
                or reviewed_check.assessment == "contradicts"
            ):
                candidate_ids_with_invalid_evidence.add(candidate_id)
                if (
                    reviewed_check_valid
                    and targeted_check.assessment == "contradicts"
                    and reviewed_check.assessment == "contradicts"
                ):
                    independently_contradicted_check_ids.add(check_id)
            if (
                reviewed_check_valid
                and targeted_check.assessment == "supports"
                and reviewed_check.assessment == "supports"
                and reviewed_check.independent_result.strip()
            ):
                independently_supported_check_ids.add(check_id)
                if targeted_check.evidence_role == "causal_discriminator":
                    has_discriminator = True

        check_topic_fields = {
            field
            for check in targeted.checks
            if check.candidate_id == candidate_id
            and check.check_id not in invalid_check_ids
            and check.check_id in independently_supported_check_ids
            and check.assessment == "supports"
            for field in check.topic_fields
        }
        evaluated_requirement_ids = {
            requirement_id
            for check in targeted.checks
            if check.candidate_id == candidate_id
            and check.check_id not in invalid_check_ids
            and check.check_id in independently_supported_check_ids
            and check.assessment == "supports"
            for requirement_id in check.evaluated_requirement_ids
        }
        missing_runtime_requirement_ids = set(
            _candidate_runtime_requirement_ids(candidate)
        ) - evaluated_requirement_ids
        missing_required_fields = (
            set(candidate.required_topic_fields) - check_topic_fields
        ) | (
            set(candidate.required_topic_fields)
            & set(targeted.missing_topic_fields)
        )
        available_parameter_names = {
            parameter.name
            for parameter in targeted.parameter_values
            if any(
                check.candidate_id == candidate_id
                and check.check_id not in invalid_check_ids
                and check.check_id in independently_supported_check_ids
                and check.assessment == "supports"
                and [
                    parameter.name,
                    parameter.value,
                ]
                in _targeted_check_receipt_payload(
                    check,
                    candidate,
                    targeted,
                )["parameters"]
                for check in targeted.checks
            )
        }
        missing_required_parameters = (
            set(candidate.required_parameters) - available_parameter_names
        )
        if missing_required_fields:
            issues.append(
                f"candidate {candidate_id} did not evaluate required topic "
                f"fields: {sorted(missing_required_fields)}"
            )
            candidate_ids_with_invalid_evidence.add(candidate_id)
        if missing_required_parameters:
            issues.append(
                f"candidate {candidate_id} did not evaluate required "
                f"parameters: {sorted(missing_required_parameters)}"
            )
            candidate_ids_with_invalid_evidence.add(candidate_id)
        if missing_runtime_requirement_ids:
            issues.append(
                f"candidate {candidate_id} did not evaluate required runtime "
                "conditions: "
                f"{sorted(missing_runtime_requirement_ids)}"
            )
            candidate_ids_with_invalid_evidence.add(candidate_id)
        if candidate_review.contradicting_evidence:
            issues.append(
                f"candidate {candidate_id} retains contradicting evidence"
            )
            candidate_ids_with_invalid_evidence.add(candidate_id)

        if candidate_review.verdict == "contradicted":
            candidate_requirement_ids = set(
                _candidate_runtime_requirement_ids(candidate)
            )
            valid_exclusion_check_ids: set[str] = set()
            for check in targeted.checks:
                if (
                    check.check_id
                    not in independently_contradicted_check_ids
                    or check.evidence_role != "contradiction"
                ):
                    continue
                check_fields = set(check.topic_fields)
                parameters_are_bound = all(
                    any(
                        parameter.name == required_parameter
                        and _commands_contain_json_record(
                            targeted.execution_receipts,
                            check.execution_commands,
                            name="check",
                            expected=_targeted_check_receipt_payload(
                                check,
                                candidate,
                                targeted,
                            ),
                            programs={"python", "python3"},
                        )
                        and [
                            parameter.name,
                            parameter.value,
                        ]
                        in _targeted_check_receipt_payload(
                            check,
                            candidate,
                            targeted,
                        )["parameters"]
                        for parameter in targeted.parameter_values
                    )
                    for required_parameter in candidate.required_parameters
                )
                data_contradiction = (
                    expected_logged_value in check_fields
                    and set(candidate.required_topic_fields).issubset(
                        check_fields
                    )
                    and candidate_requirement_ids.issubset(
                        set(check.evaluated_requirement_ids)
                    )
                    and parameters_are_bound
                )
                runtime_contradiction = bool(
                    candidate_requirement_ids
                    & set(check.evaluated_requirement_ids)
                )
                if data_contradiction or runtime_contradiction:
                    valid_exclusion_check_ids.add(check.check_id)
            log_contradiction = (
                source_review_valid
                and bool(valid_exclusion_check_ids)
                and bool(candidate_review.contradicting_evidence)
            )
            if log_contradiction:
                evidence_excluded_ids.add(candidate_id)
            else:
                issues.append(
                    f"candidate {candidate_id} was contradicted without "
                    "output-bound contradiction evidence"
                )
                candidate_ids_with_invalid_evidence.add(candidate_id)

        if (
            source.source_available
            and source.snapshot_identity is not None
            and candidate_review.verdict == "supported"
            and candidate_review.confidence in {"high", "medium"}
            and candidate_review.source_path_complete
            and has_discriminator
            and candidate_review.supporting_evidence
            and not candidate_review.unresolved_dependencies
            and candidate_id not in candidate_ids_with_invalid_lineage
            and candidate_id not in candidate_ids_with_invalid_evidence
            and source_review_valid
            and review.local_observations_sufficient
            and review.requested_state == "sufficient"
        ):
            accepted_ids.append(candidate_id)

    missing_reviews = set(candidate_by_id) - seen_review_ids
    for candidate_id in sorted(missing_reviews):
        issue = f"source candidate was not reviewed: {candidate_id}"
        issues.append(issue)
        if (
            question_is_causal
            and candidate_id not in candidate_ids_with_invalid_lineage
        ):
            fatal_issues.append(issue)

    viable_unresolved_candidates = {
        candidate_id
        for candidate_id in candidate_by_id
        if candidate_id not in candidate_ids_with_invalid_lineage
        and candidate_id not in accepted_ids
        and candidate_id not in evidence_excluded_ids
    }
    if accepted_ids and viable_unresolved_candidates:
        issue = (
            "viable competing candidates remain unresolved: "
            f"{sorted(viable_unresolved_candidates)}"
        )
        issues.append(issue)
        fatal_issues.append(issue)
        accepted_ids = []

    if review.preferred_candidate_id is not None:
        if review.preferred_candidate_id not in candidate_by_id:
            issue = (
                "preferred candidate is unknown: "
                f"{review.preferred_candidate_id}"
            )
            issues.append(issue)
            fatal_issues.append(issue)
        elif (
            review.requested_state == "sufficient"
            and review.preferred_candidate_id not in accepted_ids
        ):
            issue = (
                "preferred candidate did not satisfy the causal evidence gate: "
                f"{review.preferred_candidate_id}"
            )
            issues.append(issue)
            fatal_issues.append(issue)

    accepted_observation_indices: list[int] = []
    if not question_is_causal:
        seen_observation_indices: set[int] = set()
        for reviewed_observation in review.reviewed_observations:
            observation_index = reviewed_observation.observation_index
            if observation_index in seen_observation_indices:
                issue = (
                    "independent review repeats observation "
                    f"{observation_index}"
                )
                issues.append(issue)
                fatal_issues.append(issue)
                continue
            seen_observation_indices.add(observation_index)
            if not 0 <= observation_index < len(initial.observations):
                issue = (
                    "independent review refers to unknown observation "
                    f"{observation_index}"
                )
                issues.append(issue)
                fatal_issues.append(issue)
                continue
            observation = initial.observations[observation_index]
            observation_focus_is_bound = (
                expected_logged_value in observation.topic_fields
                or _text_contains_focus(
                    observation.finding,
                    expected_logged_value,
                )
                or _text_contains_focus(
                    observation.evidence,
                    expected_logged_value,
                )
            )
            observation_valid = (
                reviewed_observation.assessment == "supports"
                and bool(observation.evidence.strip())
                and observation_focus_is_bound
                and _receipts_contain_json_record(
                    initial.execution_receipts,
                    name="observation",
                    expected=_observation_receipt_payload(
                        observation_index,
                        observation,
                        question_focus=expected_logged_value,
                    ),
                    programs={"python", "python3"},
                )
                and bool(reviewed_observation.independent_result.strip())
                and bool(reviewed_observation.execution_commands)
                and set(
                    reviewed_observation.execution_commands
                ).issubset(review_commands)
                and _commands_contain_json_record(
                    review.execution_receipts,
                    reviewed_observation.execution_commands,
                    name="reviewed_observation",
                    expected=_reviewed_observation_receipt_payload(
                        reviewed_observation,
                        observation,
                        question_focus=expected_logged_value,
                    ),
                    programs={"python", "python3"},
                )
            )
            if observation_valid:
                accepted_observation_indices.append(observation_index)
            else:
                issue = (
                    "independent review did not verify observation "
                    f"{observation_index} from output-bound ULog evidence"
                )
                issues.append(issue)
                fatal_issues.append(issue)
        if (
            review.requested_state == "sufficient"
            and not accepted_observation_indices
        ):
            issue = (
                "observational sufficiency was requested without an "
                "independently verified ULog observation"
            )
            issues.append(issue)
            fatal_issues.append(issue)

    accepted_titles = [
        _candidate_public_title(candidate_by_id[candidate_id])
        for candidate_id in accepted_ids
        if candidate_id in candidate_by_id
    ]

    if (
        review.requested_state == "needs_web_research"
        and not review.web_queries
    ):
        issues.append("web research was requested without a targeted query")
    if (
        review.requested_state == "sufficient"
        and not review.local_observations_sufficient
    ):
        issues.append(
            "sufficient state was requested while local observations were "
            "marked insufficient"
        )

    external_context_prerequisite = (
        not question_is_causal
        or any(
            candidate_id not in candidate_ids_with_invalid_lineage
            and candidate_id not in candidate_ids_with_invalid_evidence
            for candidate_id in candidate_by_id
        )
    )
    source_discovery_prerequisite = (
        source.source_available
        and source.snapshot_identity is not None
        and (
            not candidate_by_id
            or bool(candidate_ids_with_invalid_lineage)
            or bool(review.next_source_queries)
        )
    )
    web_eligible = (
        allow_web_research
        and review.requested_state == "needs_web_research"
        and bool(review.web_queries)
        and not fatal_issues
        and not targeted.missing_topic_fields
        and (
            (
                review.web_gap_kind == "external_context"
                and external_context_prerequisite
            )
            or (
                review.web_gap_kind == "local_source"
                and source_discovery_prerequisite
            )
        )
    )
    if (
        review.requested_state == "needs_web_research"
        and review.web_gap_kind
        not in {"external_context", "local_source"}
    ):
        issues.append(
            "web research was requested for a gap that is not external "
            "context or local source discovery"
        )

    if question_is_causal:
        if accepted_ids:
            status: EvidenceStateName = "sufficient"
            reason = (
                "At least one source-grounded candidate passed an independent "
                "same-window causal discriminator."
            )
        elif (
            web_eligible
        ):
            status = "needs_web_research"
            reason = review.reason
        else:
            status = "unresolved"
            reason = review.reason
    elif (
        accepted_observation_indices
        and review.local_observations_sufficient
        and review.requested_state == "sufficient"
    ):
        status = "sufficient"
        reason = review.reason
    elif (
        web_eligible
    ):
        status = "needs_web_research"
        reason = review.reason
    else:
        status = "unresolved"
        reason = review.reason

    if status == "sufficient" and fatal_issues:
        status = "unresolved"
        accepted_ids = []
        accepted_titles = []
        accepted_observation_indices = []
        evidence_excluded_ids = set()
        reason = "Cross-stage evidence validation failed."

    excluded_ids = [
        candidate_id
        for candidate_id in candidate_by_id
        if candidate_id in evidence_excluded_ids
    ]
    return EvidenceState(
        cycle=cycle,
        status=status,
        question_is_causal=question_is_causal,
        accepted_candidate_ids=accepted_ids,
        accepted_candidate_titles=accepted_titles,
        accepted_observation_indices=accepted_observation_indices,
        excluded_candidate_ids=excluded_ids,
        excluded_candidate_titles=[
            _candidate_public_title(candidate_by_id[candidate_id])
            for candidate_id in excluded_ids
        ],
        reason=reason,
        web_queries=list(dict.fromkeys(review.web_queries)),
        validation_issues=issues,
    )


def apply_evidence_state_to_report(
    report: FlightLogReport,
    *,
    evidence_state: EvidenceState,
    review: EvidenceReview,
    source: SourceInvestigation | None = None,
    targeted: TargetedLogParse | None = None,
    initial: InitialInspection | None = None,
    airframe_summary: str | None = None,
) -> FlightLogReport:
    """Bind confident public claims to the deterministic evidence gate."""

    report.airframe_summary = (
        airframe_summary
        or "Airframe context unavailable from deterministic inventory."
    )
    report.question_intent_summary = (
        f"Analysis focus: {initial.question_focus}"
        if initial is not None
        else "Analysis focus unavailable."
    )
    report.excluded_mechanisms = []
    review_confidence = {
        candidate_review.candidate_id: candidate_review.confidence
        for candidate_review in review.candidate_reviews
    }
    title_to_id = {
        title: candidate_id
        for candidate_id, title in zip(
            evidence_state.accepted_candidate_ids,
            evidence_state.accepted_candidate_titles,
        )
    }
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in (source.candidates if source is not None else [])
    }

    if evidence_state.status != "sufficient":
        report.ranked_hypotheses = []
    elif evidence_state.question_is_causal:
        accepted_ids = set(evidence_state.accepted_candidate_ids)
        original_title_to_id = {
            candidate.title: candidate_id
            for candidate_id, candidate in candidate_by_id.items()
            if candidate_id in accepted_ids
        }
        canonical_title_by_id = {
            candidate_id: title
            for candidate_id, title in zip(
                evidence_state.accepted_candidate_ids,
                evidence_state.accepted_candidate_titles,
            )
        }
        draft_plots_by_candidate_id: dict[str, list[PlotRef]] = {}
        for hypothesis in report.ranked_hypotheses:
            candidate_id = title_to_id.get(
                hypothesis.title
            ) or original_title_to_id.get(hypothesis.title)
            if candidate_id not in accepted_ids:
                continue
            draft_plots_by_candidate_id.setdefault(
                candidate_id,
                [],
            ).extend(hypothesis.plots)

        projected_hypotheses: list[HypothesisReportItem] = []
        for candidate_id in evidence_state.accepted_candidate_ids:
            candidate = candidate_by_id.get(candidate_id)
            canonical_title = canonical_title_by_id.get(candidate_id)
            if candidate is None or canonical_title is None:
                continue
            source_refs = _candidate_code_refs(candidate)
            numeric_checks = _candidate_numeric_checks(
                candidate=candidate,
                candidate_id=candidate_id,
                candidate_title=canonical_title,
                targeted=targeted,
                review=review,
            )
            expected_signatures = _candidate_expected_signatures(
                candidate=candidate,
                candidate_id=candidate_id,
                targeted=targeted,
                review=review,
            )
            confidence = review_confidence.get(candidate_id, "low")
            if not source_refs or not numeric_checks:
                confidence = "low"
            projected_hypotheses.append(
                HypothesisReportItem(
                    title=canonical_title,
                    known_px4_mechanism=(
                        "Commit-pinned PX4 source lineage"
                    ),
                    mechanism=_candidate_grounded_mechanism(candidate),
                    source_refs=source_refs,
                    expected_logged_signature=expected_signatures,
                    applicability=_candidate_applicability(
                        candidate,
                        candidate_id=candidate_id,
                        targeted=targeted,
                        review=review,
                    ),
                    evidence=_candidate_report_evidence(
                        candidate_id=candidate_id,
                        targeted=targeted,
                        review=review,
                    ),
                    contradicting_evidence=[],
                    unresolved_evidence=(
                        []
                        if source_refs and numeric_checks
                        else [
                            "Accepted evidence could not be projected into "
                            "the stable report fields."
                        ]
                    ),
                    exclusion_checks=[],
                    numeric_checks=numeric_checks,
                    confidence=confidence,
                    plots=_canonical_plot_refs(
                        draft_plots_by_candidate_id.get(
                            candidate_id,
                            [],
                        )
                    ),
                )
            )
        report.ranked_hypotheses = projected_hypotheses
        report.excluded_mechanisms = list(
            evidence_state.excluded_candidate_titles
        )
    elif initial is not None:
        report = _project_observational_report(
            report,
            evidence_state=evidence_state,
            initial=initial,
            review=review,
        )
    else:
        for hypothesis in report.ranked_hypotheses:
            if hypothesis.confidence in {"high", "medium"}:
                hypothesis.confidence = "low"
        report.final_summary = (
            "The observational evidence was not available for deterministic "
            "report projection."
        )

    gated_report = reconcile_report_with_evidence_state(
        report,
        evidence_state=evidence_state,
    )
    if (
        evidence_state.status == "sufficient"
        and evidence_state.question_is_causal
        and gated_report.confirmed
        and source is not None
    ):
        candidate_by_title = {
            title: candidate_by_id[candidate_id]
            for candidate_id, title in zip(
                evidence_state.accepted_candidate_ids,
                evidence_state.accepted_candidate_titles,
            )
            if candidate_id in candidate_by_id
        }
        confirmed_mechanisms = [
            (
                f"{title}: "
                f"{_candidate_grounded_mechanism(candidate_by_title[title])}"
                if title in candidate_by_title
                else title
            )
            for title in gated_report.confirmed
        ]
        gated_report.final_summary = (
            "Confirmed by commit-pinned source lineage and independently "
            "replayed ULog evidence: "
            + "; ".join(confirmed_mechanisms)
        )
    return gated_report


def _project_observational_report(
    report: FlightLogReport,
    *,
    evidence_state: EvidenceState,
    initial: InitialInspection,
    review: EvidenceReview,
) -> FlightLogReport:
    review_by_index = {
        reviewed.observation_index: reviewed
        for reviewed in review.reviewed_observations
        if reviewed.assessment == "supports"
    }
    records = [
        (
            index,
            initial.observations[index],
            review_by_index[index],
        )
        for index in evidence_state.accepted_observation_indices
        if 0 <= index < len(initial.observations)
        and index in review_by_index
    ]
    if not records:
        for hypothesis in report.ranked_hypotheses:
            hypothesis.confidence = "low"
        return report

    title_material = "\0".join(
        [
            initial.question_focus,
            *(
                f"{observation.evidence}\0{reviewed.independent_result}"
                for _index, observation, reviewed in records
            ),
        ]
    )
    digest = hashlib.sha256(
        title_material.encode("utf-8")
    ).hexdigest()[:10]
    title = f"ULog observation {digest}: {initial.question_focus}"
    evidence = [
        (
            f"{observation.evidence} Independent review: "
            f"{reviewed.independent_result}"
        )
        for _index, observation, reviewed in records
    ]
    report.question_intent_summary = (
        f"Direct observation of {initial.question_focus}"
    )
    report.ranked_hypotheses = [
        HypothesisReportItem(
            title=title,
            known_px4_mechanism="Direct ULog observation",
            mechanism="Direct ULog evidence: " + " ".join(evidence),
            source_refs=[],
            expected_logged_signature=[
                ExpectedSignatureItem(
                    name=f"observation_{index}",
                    description="Value returned by the ULog inspection.",
                    signal=", ".join(observation.topic_fields) or None,
                    expected_behavior=observation.evidence,
                )
                for index, observation, _reviewed in records
            ],
            applicability=ApplicabilityReport(
                applicable=True,
                available_required_signals=list(dict.fromkeys(
                    field
                    for _index, observation, _reviewed in records
                    for field in observation.topic_fields
                )),
            ),
            evidence=evidence,
            contradicting_evidence=[],
            unresolved_evidence=[],
            exclusion_checks=[],
            numeric_checks=[
                RelationshipCheckSpec(
                    type="custom",
                    actual=", ".join(observation.topic_fields) or None,
                    description=(
                        f"Observed: {observation.evidence} "
                        f"Independent review: "
                        f"{reviewed.independent_result}"
                    ),
                )
                for _index, observation, reviewed in records
            ],
            confidence="low",
            plots=[],
        )
    ]
    report.final_summary = (
        "Direct, independently replayed ULog observation: "
        + " ".join(evidence)
    )
    return report


def _canonical_plot_refs(plots: list[PlotRef]) -> list[PlotRef]:
    canonical: list[PlotRef] = []
    seen_paths: set[str] = set()
    for plot in plots:
        if not plot.path or plot.path in seen_paths:
            continue
        seen_paths.add(plot.path)
        artifact_name = Path(plot.path).name
        canonical.append(
            PlotRef(
                title=f"Analysis plot: {artifact_name}",
                path=plot.path,
                purpose=(
                    "Plot artifact produced during the evidence workflow."
                ),
            )
        )
    return canonical


def _candidate_public_title(candidate: CandidateMechanism) -> str:
    lineage_material = "\0".join(
        [
            candidate.candidate_id,
            *(
                f"{step.file}:{step.lines}:{step.source_excerpt}"
                for step in candidate.upstream_assignment_path
            ),
        ]
    )
    digest = hashlib.sha256(
        lineage_material.encode("utf-8")
    ).hexdigest()[:10]
    endpoint = candidate.upstream_assignment_path[-1]
    operation = _bounded_public_source_operation(endpoint.source_excerpt)
    return f"PX4 source path {digest}: {operation}"


def _candidate_source_operations(
    candidate: CandidateMechanism,
) -> list[str]:
    return [
        _bounded_public_source_operation(step.source_excerpt)
        for step in candidate.upstream_assignment_path
        if _normalize_evidence_text(step.source_excerpt)
    ]


def _candidate_grounded_mechanism(
    candidate: CandidateMechanism,
) -> str:
    return (
        "The commit-pinned source lineage executes: "
        + " -> ".join(_candidate_source_operations(candidate))
    )


def _bounded_public_source_operation(operation: str) -> str:
    normalized = _normalize_evidence_text(operation)
    if len(normalized) <= MAX_PUBLIC_SOURCE_OPERATION_CHARS:
        return normalized
    return normalized[:MAX_PUBLIC_SOURCE_OPERATION_CHARS] + "…"


def _candidate_code_refs(
    candidate: CandidateMechanism,
) -> list[CodeRef]:
    refs: list[CodeRef] = []
    seen: set[tuple[str, int, int, str]] = set()
    for step in candidate.upstream_assignment_path:
        line_numbers = [
            int(value)
            for value in re.findall(r"\d+", step.lines)
        ]
        if len(line_numbers) not in {1, 2}:
            continue
        start_line = line_numbers[0]
        end_line = line_numbers[-1]
        key = (
            step.file,
            start_line,
            end_line,
            step.source_excerpt,
        )
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            CodeRef(
                file=step.file,
                function=None,
                start_line=start_line,
                end_line=end_line,
                snippet=step.source_excerpt,
                explanation=(
                    "Verified commit-pinned source lineage excerpt."
                ),
            )
        )
    return refs


def _candidate_numeric_checks(
    *,
    candidate: CandidateMechanism | None,
    candidate_id: str,
    candidate_title: str,
    targeted: TargetedLogParse | None,
    review: EvidenceReview,
) -> list[RelationshipCheckSpec]:
    if candidate is None or targeted is None:
        return []
    review_by_check_id = {
        check.check_id: check
        for candidate_review in review.candidate_reviews
        if candidate_review.candidate_id == candidate_id
        for check in candidate_review.reviewed_checks
        if check.assessment == "supports"
    }
    checks: list[RelationshipCheckSpec] = []
    for check in targeted.checks:
        reviewed = review_by_check_id.get(check.check_id)
        if (
            check.candidate_id != candidate_id
            or check.assessment != "supports"
            or reviewed is None
        ):
            continue
        checks.append(
            RelationshipCheckSpec(
                type="custom",
                window=check.window,
                actual=", ".join(check.topic_fields) or None,
                source_predicate=" -> ".join(
                    _candidate_source_operations(candidate)
                ),
                description=(
                    f"{candidate_title} — {check.check_id}: "
                    f"Result: {check.result} "
                    f"Independent review: {reviewed.independent_result}"
                ),
            )
        )
    return checks


def _candidate_expected_signatures(
    *,
    candidate: CandidateMechanism,
    candidate_id: str,
    targeted: TargetedLogParse | None,
    review: EvidenceReview,
) -> list[ExpectedSignatureItem]:
    if targeted is None:
        return []
    reviewed_check_ids = {
        check.check_id
        for candidate_review in review.candidate_reviews
        if candidate_review.candidate_id == candidate_id
        for check in candidate_review.reviewed_checks
        if check.assessment == "supports"
    }
    source_behavior = " -> ".join(
        _candidate_source_operations(candidate)
    )
    return [
        ExpectedSignatureItem(
            name=check.check_id,
            description=(
                "Expected behavior derived from the commit-pinned source "
                "operations."
            ),
            signal=", ".join(check.topic_fields) or None,
            expected_behavior=source_behavior,
        )
        for check in targeted.checks
        if check.candidate_id == candidate_id
        and check.assessment == "supports"
        and check.check_id in reviewed_check_ids
    ]


def _candidate_applicability(
    candidate: CandidateMechanism,
    *,
    candidate_id: str,
    targeted: TargetedLogParse | None,
    review: EvidenceReview,
) -> ApplicabilityReport:
    reviewed_check_ids = {
        check.check_id
        for candidate_review in review.candidate_reviews
        if candidate_review.candidate_id == candidate_id
        for check in candidate_review.reviewed_checks
        if check.assessment == "supports"
    }
    supported_checks = [
        check
        for check in (targeted.checks if targeted is not None else [])
        if check.candidate_id == candidate_id
        and check.assessment == "supports"
        and check.check_id in reviewed_check_ids
    ]
    parameter_by_name = {
        parameter.name: parameter.value
        for parameter in (
            targeted.parameter_values if targeted is not None else []
        )
    }
    return ApplicabilityReport(
        applicable=True,
        supported_conditions=[
            f"Runtime requirement {requirement_id} was independently evaluated."
            for requirement_id in dict.fromkeys(
                requirement_id
                for check in supported_checks
                for requirement_id in check.evaluated_requirement_ids
            )
        ],
        excluded_by=[],
        unresolved_conditions=[],
        relevant_parameters=[
            ParameterValue(
                name=parameter_name,
                value=parameter_by_name[parameter_name],
            )
            for parameter_name in candidate.required_parameters
            if parameter_name in parameter_by_name
        ],
        available_required_signals=list(dict.fromkeys(
            field
            for check in supported_checks
            for field in check.topic_fields
        )),
        missing_required_signals=[],
    )


def _candidate_report_evidence(
    *,
    candidate_id: str,
    targeted: TargetedLogParse | None,
    review: EvidenceReview,
) -> list[str]:
    if targeted is None:
        return []
    review_by_check_id = {
        check.check_id: check
        for candidate_review in review.candidate_reviews
        if candidate_review.candidate_id == candidate_id
        for check in candidate_review.reviewed_checks
        if check.assessment == "supports"
    }
    return [
        (
            f"{check.check_id}: {check.result} "
            f"Independent review: "
            f"{review_by_check_id[check.check_id].independent_result}"
        )
        for check in targeted.checks
        if check.candidate_id == candidate_id
        and check.assessment == "supports"
        and check.check_id in review_by_check_id
    ]


def reconcile_report_with_evidence_state(
    report: FlightLogReport,
    *,
    evidence_state: EvidenceState,
    unresolved_reason: str | None = None,
) -> FlightLogReport:
    """Reconcile the public report after generation or validation."""

    accepted_titles = set(evidence_state.accepted_candidate_titles)
    if evidence_state.status != "sufficient":
        report.confirmed = []
    else:
        report.confirmed = [
            hypothesis.title
            for hypothesis in report.ranked_hypotheses
            if hypothesis.confidence in {"high", "medium"}
            and (
                not evidence_state.question_is_causal
                or hypothesis.title in accepted_titles
            )
        ]
    report.unconfirmed = [
        hypothesis.title
        for hypothesis in report.ranked_hypotheses
        if hypothesis.title not in report.confirmed
    ]

    if evidence_state.status != "sufficient" or (
        evidence_state.question_is_causal and not report.confirmed
    ):
        reason = (
            unresolved_reason
            or (
                "The structured report did not retain an evidence-gated "
                "causal hypothesis."
                if evidence_state.status == "sufficient"
                else (
                    "The available source and flight-log evidence did not "
                    "pass the deterministic evidence gate."
                )
            )
        ).strip()
        prefix = (
            "No causal explanation was confirmed."
            if evidence_state.question_is_causal
            else "The requested conclusion remains unresolved."
        )
        report.final_summary = prefix + (f" {reason}" if reason else "")

    return report


async def run_staged_analysis(
    *,
    user_question: str,
    inventory: dict[str, Any],
    source_snapshot: SourceSnapshot | None,
    mission_path: Path | None,
    work_dir: Path,
    plots_dir: Path,
    model: str,
    max_turns: int,
    max_total_requests: int | None = None,
    project_instructions: str,
    base_instructions: str,
    audit_logger: DeveloperAuditLogger,
    shell_tool_factory: ShellToolFactory,
    enable_web_fallback: bool = True,
    web_search_context: Literal["low", "medium", "high"] = "medium",
) -> StagedAnalysisResult:
    """Run the staged evidence workflow behind the stable report contract."""

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    request_budget = ModelRequestBudget(
        limit=max_total_requests if max_total_requests is not None else max_turns
    )
    if request_budget.limit < 1:
        raise ValueError("max_total_requests must be at least 1")

    usage = Usage()
    stages_dir = work_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    mission_note = (
        "/inputs/mission.plan is available."
        if mission_path is not None
        else "No mission file was supplied."
    )
    source_note = (
        f"Exact source snapshot commit: {source_snapshot.commit_sha}"
        if source_snapshot is not None
        else "No exact source snapshot is available."
    )
    shared_context = {
        "question": user_question,
        "inventory": _compact_inventory(inventory, source_snapshot),
        "mission": mission_note,
        "source": source_note,
        "plots_dir": "/plots",
    }

    initial = await _run_initial_inspection(
        context=shared_context,
        model=model,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=(7 if source_snapshot is not None else 5),
        project_instructions=project_instructions,
        base_instructions=base_instructions,
        audit_logger=audit_logger,
        usage=usage,
        shell_tool=shell_tool_factory(False),
        stages_dir=stages_dir,
    )

    cycles: list[InvestigationCycle] = []
    first_source = await _run_source_investigation(
        cycle=1,
        context=shared_context,
        initial=initial,
        prior_cycle=None,
        web_research=None,
        source_snapshot=source_snapshot,
        model=model,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=5,
        project_instructions=project_instructions,
        base_instructions=base_instructions,
        audit_logger=audit_logger,
        usage=usage,
        shell_tool_factory=shell_tool_factory,
        stages_dir=stages_dir,
    )
    first_targeted = await _run_targeted_log_parse(
        cycle=1,
        context=shared_context,
        initial=initial,
        source=first_source,
        prior_cycle=None,
        model=model,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=3,
        project_instructions=project_instructions,
        base_instructions=base_instructions,
        audit_logger=audit_logger,
        usage=usage,
        shell_tool=shell_tool_factory(False),
        stages_dir=stages_dir,
    )
    first_review = await _run_evidence_review(
        cycle=1,
        context=shared_context,
        initial=initial,
        source=first_source,
        targeted=first_targeted,
        web_research=None,
        model=model,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=1,
        project_instructions=project_instructions,
        base_instructions=base_instructions,
        audit_logger=audit_logger,
        usage=usage,
        shell_tool=shell_tool_factory(source_snapshot is not None),
        stages_dir=stages_dir,
    )
    first_state = derive_evidence_state(
        initial=initial,
        source=first_source,
        targeted=first_targeted,
        review=first_review,
        cycle=1,
        allow_web_research=(
            enable_web_fallback
            and (
                source_snapshot is not None
                or not (
                    initial.question_is_causal
                    or first_review.question_is_causal
                )
            )
        ),
    )
    _record_evidence_state(
        stages_dir / "04_evidence_state_cycle_1.json",
        first_state,
        audit_logger,
    )
    cycles.append(
        InvestigationCycle(
            cycle=1,
            source=first_source,
            targeted_log=first_targeted,
            review=first_review,
            evidence_state=first_state,
        )
    )

    final_review = first_review
    final_state = first_state

    web_retry_minimum = 9 if source_snapshot is not None else 7
    remaining_requests = request_budget.limit - usage.requests
    if (
        first_state.status == "needs_web_research"
        and enable_web_fallback
        and remaining_requests < web_retry_minimum
    ):
        final_state = first_state.model_copy(
            update={
                "status": "unresolved",
                "web_queries": [],
                "reason": (
                    "The local evidence was insufficient and the remaining "
                    "model-request budget could not complete a bounded web "
                    "retry plus local verification."
                ),
            }
        )
        cycles[-1].evidence_state = final_state
        audit_logger.log_event(
            "agent.shell_analysis_web_fallback.skipped",
            reason="insufficient_request_budget",
            output={
                "remaining_requests": remaining_requests,
                "required_requests": web_retry_minimum,
            },
        )
    elif first_state.status == "needs_web_research" and enable_web_fallback:
        web_research = await _run_web_research_or_record_failure(
            queries=first_state.web_queries,
            model=model,
            max_turns=max_turns,
            request_budget=request_budget,
            reserve_requests=(
                7 if source_snapshot is not None else 5
            ),
            audit_logger=audit_logger,
            usage=usage,
            stages_dir=stages_dir,
            web_search_context=web_search_context,
        )
        cycles[-1].web_research = web_research

        if web_research.error is None and web_research.findings:
            try:
                second_source = await _run_source_investigation(
                    cycle=2,
                    context=shared_context,
                    initial=initial,
                    prior_cycle=cycles[-1],
                    web_research=web_research,
                    source_snapshot=source_snapshot,
                    model=model,
                    max_turns=max_turns,
                    request_budget=request_budget,
                    reserve_requests=5,
                    project_instructions=project_instructions,
                    base_instructions=base_instructions,
                    audit_logger=audit_logger,
                    usage=usage,
                    shell_tool_factory=shell_tool_factory,
                    stages_dir=stages_dir,
                )
                second_targeted = await _run_targeted_log_parse(
                    cycle=2,
                    context=shared_context,
                    initial=initial,
                    source=second_source,
                    prior_cycle=cycles[-1],
                    model=model,
                    max_turns=max_turns,
                    request_budget=request_budget,
                    reserve_requests=3,
                    project_instructions=project_instructions,
                    base_instructions=base_instructions,
                    audit_logger=audit_logger,
                    usage=usage,
                    shell_tool=shell_tool_factory(False),
                    stages_dir=stages_dir,
                )
                second_review = await _run_evidence_review(
                    cycle=2,
                    context=shared_context,
                    initial=initial,
                    source=second_source,
                    targeted=second_targeted,
                    web_research=web_research,
                    model=model,
                    max_turns=max_turns,
                    request_budget=request_budget,
                    reserve_requests=1,
                    project_instructions=project_instructions,
                    base_instructions=base_instructions,
                    audit_logger=audit_logger,
                    usage=usage,
                    shell_tool=shell_tool_factory(
                        source_snapshot is not None
                    ),
                    stages_dir=stages_dir,
                )
                second_state = derive_evidence_state(
                    initial=initial,
                    source=second_source,
                    targeted=second_targeted,
                    review=second_review,
                    cycle=2,
                    allow_web_research=False,
                )
                _record_evidence_state(
                    stages_dir / "08_evidence_state_cycle_2.json",
                    second_state,
                    audit_logger,
                )
                cycles.append(
                    InvestigationCycle(
                        cycle=2,
                        source=second_source,
                        targeted_log=second_targeted,
                        review=second_review,
                        evidence_state=second_state,
                        web_research=web_research,
                    )
                )
                final_review = second_review
                final_state = second_state
            except Exception as exc:
                audit_logger.log_event(
                    "agent.shell_analysis_optional_retry.failed",
                    error=repr(exc),
                )
                final_state = first_state.model_copy(
                    update={
                        "status": "unresolved",
                        "accepted_candidate_ids": [],
                        "accepted_candidate_titles": [],
                        "reason": (
                            "The first local evidence review was insufficient "
                            "and the optional local retry failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "web_queries": [],
                    }
                )
                cycles[-1].evidence_state = final_state
        else:
            web_failure = (
                web_research.error
                or "the bounded web fallback returned no usable findings"
            )
            final_state = first_state.model_copy(
                update={
                    "status": "unresolved",
                    "accepted_candidate_ids": [],
                    "accepted_candidate_titles": [],
                    "reason": (
                        "The first local evidence review was insufficient and "
                        f"the bounded web fallback failed: {web_failure}"
                    ),
                    "web_queries": [],
                }
            )
            cycles[-1].evidence_state = final_state

    _record_evidence_state(
        stages_dir / "08_final_evidence_state.json",
        final_state,
        audit_logger,
        final=True,
    )
    available_plot_artifacts = _available_plot_artifacts(plots_dir)
    report = await _run_final_report(
        context=shared_context,
        initial=initial,
        cycles=cycles,
        final_state=final_state,
        available_plot_artifacts=available_plot_artifacts,
        model=model,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=0,
        project_instructions=project_instructions,
        audit_logger=audit_logger,
        usage=usage,
        stages_dir=stages_dir,
    )
    report = _normalize_report_plot_paths(
        report,
        plots_dir=plots_dir,
        available_plot_artifacts=available_plot_artifacts,
    )
    report = apply_evidence_state_to_report(
        report,
        evidence_state=final_state,
        review=final_review,
        source=cycles[-1].source,
        targeted=cycles[-1].targeted_log,
        initial=initial,
        airframe_summary=_inventory_airframe_summary(inventory),
    )
    _write_stage_artifact(stages_dir / "09_gated_report.json", report)

    return StagedAnalysisResult(
        report=report,
        usage=usage,
        evidence_state=final_state,
    )


async def _run_initial_inspection(
    *,
    context: dict[str, Any],
    model: str,
    max_turns: int,
    request_budget: ModelRequestBudget,
    reserve_requests: int,
    project_instructions: str,
    base_instructions: str,
    audit_logger: DeveloperAuditLogger,
    usage: Usage,
    shell_tool: ShellTool,
    stages_dir: Path,
) -> InitialInspection:
    agent = Agent(
        name="PX4 Initial ULog Inspector",
        model=model,
        instructions=f"""
{base_instructions}

Stage role: neutral ULog orientation.
- Use pyulog and the shell to inspect FLIGHT_LOG_ULOG
  (/inputs/flight.ulg).
- Identify the exact logged value or event the question asks about, relevant
  windows, available fields, parameters, messages, and state transitions.
- For every recorded observation, make one Python command print the exact
  finding and evidence text together with its topic fields, then copy those
  printed values verbatim into the observation. Print one line as
  observation_json=<JSON object> with exactly these keys:
  observation_index, question_focus, finding, evidence, topic_fields, and
  time_or_window. Use the zero-based output-list index and compact JSON.
- An observation used to answer a logged-value question must include
  question_focus exactly in topic_fields. An event/message observation may
  instead include question_focus verbatim in finding or evidence; include
  the underlying topic fields when they exist.
- Set question_focus to that exact logged value or event identifier; later
  stages must preserve it verbatim.
- Do not select a firmware mechanism or answer the question yet.
- Produce source-search terms derived from the question and observed names.
- Treat missing data as missing; do not invent an observation to fill a field.

Project instructions:
{project_instructions or "(none)"}
""",
        tools=[shell_tool],
        model_settings=ModelSettings(tool_choice="shell"),
        output_type=InitialInspection,
    )
    output = await _run_agent_stage(
        stage="shell_analysis_initial",
        artifact_name="01_initial_inspection.json",
        agent=agent,
        prompt=json.dumps(context, indent=2),
        expected_type=InitialInspection,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=reserve_requests,
        audit_logger=audit_logger,
        usage=usage,
        stages_dir=stages_dir,
    )
    return output


async def _run_source_investigation(
    *,
    cycle: int,
    context: dict[str, Any],
    initial: InitialInspection,
    prior_cycle: InvestigationCycle | None,
    web_research: WebResearch | None,
    source_snapshot: SourceSnapshot | None,
    model: str,
    max_turns: int,
    request_budget: ModelRequestBudget,
    reserve_requests: int,
    project_instructions: str,
    base_instructions: str,
    audit_logger: DeveloperAuditLogger,
    usage: Usage,
    shell_tool_factory: ShellToolFactory,
    stages_dir: Path,
) -> SourceInvestigation:
    if source_snapshot is None:
        audit_logger.log_event(
            f"agent.shell_analysis_source_{cycle}.started",
            skipped=True,
        )
        output = SourceInvestigation(
            source_available=False,
            snapshot_identity=None,
            explained_logged_value=initial.question_focus,
            unresolved_reason=(
                "No exact source snapshot is available; source-backed causal "
                "candidates cannot be established."
            ),
        )
        _write_stage_artifact(
            stages_dir / f"02_source_cycle_{cycle}.json",
            output,
        )
        audit_logger.log_event(
            f"agent.shell_analysis_source_{cycle}.finished",
            output=output.model_dump(),
            skipped=True,
        )
        return output

    agent = Agent(
        name=f"PX4 Pinned Source Investigator Cycle {cycle}",
        model=model,
        instructions=f"""
{base_instructions}

Stage role: source lineage and candidate discovery.
- Use only the exact commit-pinned Git interface for PX4 source.
- Start at the publishing assignment for the logged value being explained.
- Verify that the publishing assignment belongs to the questioned ULog
  topic/message type, and that its exact source field matches the field part
  of question_focus. Trace through the publisher/type declaration when the
  assignment alone does not establish the topic mapping.
- Populate publication_binding with two exact, commit-pinned Git reads:
  (1) the declaration that binds the publishing assignment's message
  identifier to its generated <SchemaName>_s type, and (2) the corresponding
  .msg field declaration. Cite each complete excerpt, canonical line range,
  file, and successful Git show command. The .msg filename or its # TOPICS
  declaration must derive the ULog topic name; do not infer it from a similar
  variable name.
- When the ULog field is flattened from nested messages or arrays, populate
  nested_schema_bindings in field-path order with one exact, commit-pinned
  .msg read for every referenced nested message type. Derive each next schema
  from the preceding field's declared type; do not guess a type or skip an
  intermediate field.
- For a C++ designated initializer such as `type_s message {{
  .field = value }}`, make declaration_excerpt span from the message
  declaration through the exact cited member initializer. This lets the
  executor verify that the otherwise object-less `.field` belongs to that
  message.
- Trace backward in execution order through assignments, constraints,
  branches, and helpers that can change that value.
- Record every discovered value-changing step in upstream_assignment_path.
- Copy the exact successful Git command that exposed each recorded lineage
  step into that step's execution_commands. The executor fills
  execution_receipts; leave that field empty.
- Copy a concise, verbatim code excerpt from that command's output into
  source_excerpt. Read the cited file with Git show; a search hit alone is not
  enough to establish a lineage step.
- Record file relative to the PX4 repository root. For a submodule read,
  prefix the Git show path with the submodule alias path below
  PX4-Autopilot.
- Record the excerpt's actual one-based line or line range from the Git show
  output. The excerpt must be the complete selected line or lines, not a
  substring. Copy that complete excerpt into operation as well, and bind its
  input_or_state and output_or_effect to only the exact source identifiers or
  literals consumed and produced by the operation. Every token in those two
  fields must occur in the excerpt. Do not put a logged-topic alias in
  output_or_effect when it is not source text; explained_logged_value carries
  the cross-stage analysis target. Do not add explanatory prose or other
  context tokens to those two fields.
- Separate downstream copying or tracking into downstream_propagation; it is
  provenance evidence, not proof of what created the deviation.
- Derive competing candidates and, for each, runtime conditions plus a
  same-window calculation or branch test that could distinguish it.
- Do not claim that a source mechanism occurred in this flight.
- If the source path remains incomplete, preserve the gap explicitly.

Project instructions:
{project_instructions or "(none)"}
""",
        tools=[shell_tool_factory(True)],
        model_settings=ModelSettings(tool_choice="shell"),
        output_type=SourceInvestigation,
    )
    payload = {
        "context": context,
        "initial_inspection": initial.model_dump(),
        "prior_cycle": prior_cycle.model_dump() if prior_cycle else None,
        "web_research": web_research.model_dump() if web_research else None,
        "required_snapshot_identity": source_snapshot.commit_sha,
        "candidate_id_prefix": f"c{cycle}_",
    }

    def normalize_source_output(
        output: SourceInvestigation,
    ) -> SourceInvestigation:
        output.source_available = True
        output.snapshot_identity = source_snapshot.commit_sha
        _canonicalize_source_investigation_files(output)
        return output

    output = await _run_agent_stage(
        stage=f"shell_analysis_source_{cycle}",
        artifact_name=f"02_source_cycle_{cycle}.json",
        agent=agent,
        prompt=json.dumps(payload, indent=2),
        expected_type=SourceInvestigation,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=reserve_requests,
        audit_logger=audit_logger,
        usage=usage,
        stages_dir=stages_dir,
        output_normalizer=normalize_source_output,
    )
    return output


async def _run_targeted_log_parse(
    *,
    cycle: int,
    context: dict[str, Any],
    initial: InitialInspection,
    source: SourceInvestigation,
    prior_cycle: InvestigationCycle | None,
    model: str,
    max_turns: int,
    request_budget: ModelRequestBudget,
    reserve_requests: int,
    project_instructions: str,
    base_instructions: str,
    audit_logger: DeveloperAuditLogger,
    usage: Usage,
    shell_tool: ShellTool,
    stages_dir: Path,
) -> TargetedLogParse:
    agent = Agent(
        name=f"PX4 Targeted ULog Verifier Cycle {cycle}",
        model=model,
        instructions=f"""
{base_instructions}

Stage role: candidate-directed ULog checks.
- Use pyulog and the shell to test the source-derived candidates against the
  actual flight.
- Align comparisons to the same event window and compatible representation.
- For each viable causal candidate, attempt a causal_discriminator: replay a
  source expression with logged operands, verify its activating branch, or
  perform a counterfactual/exclusion test.
- Treat every supplied runtime requirement as necessary evidence. Reference
  its exact requirement_id in evaluated_requirement_ids only when the Python
  command actually evaluated it, and print that ID with the result.
- Label signal agreement that only shows copying or tracking as
  downstream_consistency, never causal_discriminator.
- Record missing fields and not-evaluable checks explicitly.
- Preserve initial_inspection.question_focus verbatim in
  explained_logged_value.
- For every performed check, copy the exact successful Python command that
  computed it into execution_commands. The executor fills execution_receipts;
  leave that field empty.
- Make the command print a compact structured summary containing the checked
  window, topic fields, required parameter values, and numeric/branch result.
  Print one line as check_json=<JSON object> with exactly these keys:
  check_id, candidate_id, evidence_role, window, topic_fields,
  requirement_ids, parameters, result, and assessment. Encode parameters as
  [name, value] pairs for the candidate's required parameters. Use compact
  JSON, and copy all decoded values verbatim into the structured output.
- Do not formulate the final answer.

Project instructions:
{project_instructions or "(none)"}
""",
        tools=[shell_tool],
        model_settings=ModelSettings(tool_choice="shell"),
        output_type=TargetedLogParse,
    )
    payload = {
        "context": context,
        "initial_inspection": initial.model_dump(),
        "source_investigation": source.model_dump(),
        "runtime_evidence_requirements": [
            {
                "candidate_id": candidate.candidate_id,
                "condition": condition,
                "requirement_id": _runtime_requirement_id(
                    candidate.candidate_id,
                    condition,
                ),
            }
            for candidate in source.candidates
            for condition in _candidate_runtime_conditions(candidate)
        ],
        "prior_cycle": prior_cycle.model_dump() if prior_cycle else None,
    }
    return await _run_agent_stage(
        stage=f"shell_analysis_targeted_{cycle}",
        artifact_name=f"03_targeted_log_cycle_{cycle}.json",
        agent=agent,
        prompt=json.dumps(payload, indent=2),
        expected_type=TargetedLogParse,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=reserve_requests,
        audit_logger=audit_logger,
        usage=usage,
        stages_dir=stages_dir,
    )


async def _run_evidence_review(
    *,
    cycle: int,
    context: dict[str, Any],
    initial: InitialInspection,
    source: SourceInvestigation,
    targeted: TargetedLogParse,
    web_research: WebResearch | None,
    model: str,
    max_turns: int,
    request_budget: ModelRequestBudget,
    reserve_requests: int,
    project_instructions: str,
    base_instructions: str,
    audit_logger: DeveloperAuditLogger,
    usage: Usage,
    shell_tool: ShellTool,
    stages_dir: Path,
) -> EvidenceReview:
    agent = Agent(
        name=f"PX4 Independent Causal Reviewer Cycle {cycle}",
        model=model,
        instructions=f"""
{base_instructions}

Stage role: fresh-context falsification and evidence review.
- Use the shell. Independently check the proposed source lineage and ULog
  evidence rather than accepting the prior agents' interpretation.
- Start from the questioned logged value and verify the direction and order of
  the cited source path.
- Independently verify that the publishing endpoint belongs to the questioned
  ULog topic/message type and publishes its exact field; a same-named field in
  an unrelated message is not a valid binding.
- Populate reviewed_publication_binding by independently re-reading the exact
  message declaration and .msg schema cited by publication_binding. Copy the
  same binding fields, cite your own successful Git show commands, and mark it
  verified only when the generated message type, schema topic, field, and
  publishing message identifier all agree.
- For a flattened nested or array field, independently re-read every
  nested_schema_bindings entry in field-path order and verify that each next
  .msg filename is derived from the preceding field's declared type.
- For each candidate, copy the exact successful Git show commands that
  independently verified its cited lineage files into
  source_verification_commands.
- For every proposed source step, independently identify the exact source
  tokens consumed and produced by that operation. Record them in
  reviewed_source_lineage with the same sequence, canonical file, and
  verbatim excerpt. Mark direction_assessment verified only when the proposed
  upstream-to-downstream direction is correct; do not copy the prior
  input/output descriptions without checking the code.
- Look for an earlier assignment, constraint, branch, or helper that could
  create the deviation.
- Treat equality between a downstream consumer and its input as propagation.
- A causal candidate is supported only when the source path is complete and a
  same-window causal_discriminator supports it.
- Independently re-check targeted evidence and reference its exact check_id in
  reviewed_checks. Do not create a new check in the review. Record your own
  result and assessment for each referenced check.
- Preserve initial_inspection.question_focus verbatim in
  explained_logged_value. Copy the exact successful commands used for each
  independent re-check into that reviewed check's execution_commands. The
  executor fills execution_receipts; leave that field empty.
- Make one independent Python command print its compact result together with
  the checked window, topic fields, and runtime requirement IDs; copy the
  decoded result verbatim into independent_result. Print one line as
  reviewed_check_json=<JSON object> with exactly these keys: check_id,
  independent_result, assessment, window, topic_fields, and requirement_ids.
  Use compact JSON.
- For a non-causal question, independently re-read each observation used in
  the answer. Record its zero-based initial observation index, exact
  independent result, assessment, and successful Python command in
  reviewed_observations. That command must print one line as
  reviewed_observation_json=<JSON object> with exactly these keys:
  observation_index, question_focus, independent_result, assessment, and
  topic_fields. Use compact JSON.
- Attempt to falsify the preferred candidate and consider at least one
  competing explanation when the source exposes one.
- Do not mark a competing candidate contradicted without output-bound
  contradiction evidence from a performed contradicted log check. If the
  independent source read instead makes a proposed path incomplete, leave
  that source conflict unresolved rather than using free text to exclude it.
- Request web research only for a precise external-context gap or a
  local_source discovery gap where round one could not locate or complete the
  mechanism in the available exact snapshot. Web search may suggest source
  terms or history, but cannot replace the next pinned-source and ULog tests.
- Set web_gap_kind to external_context for documentation/version context, or
  local_source for the bounded discovery case. Use local_log when missing
  flight evidence is the blocker; web cannot repair that.
- If missing local signals prevent discrimination, mark the result unresolved.
- Do not write the final FlightLogReport.

Project instructions:
{project_instructions or "(none)"}
""",
        tools=[shell_tool],
        model_settings=ModelSettings(tool_choice="shell"),
        output_type=EvidenceReview,
    )
    payload = {
        "context": context,
        "initial_inspection": initial.model_dump(),
        "source_investigation": source.model_dump(),
        "targeted_log_parse": targeted.model_dump(),
        "web_research": web_research.model_dump() if web_research else None,
        "cycle": cycle,
    }
    return await _run_agent_stage(
        stage=f"shell_analysis_review_{cycle}",
        artifact_name=f"04_review_cycle_{cycle}.json",
        agent=agent,
        prompt=json.dumps(payload, indent=2),
        expected_type=EvidenceReview,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=reserve_requests,
        audit_logger=audit_logger,
        usage=usage,
        stages_dir=stages_dir,
    )


async def _run_web_research_or_record_failure(
    *,
    queries: list[str],
    model: str,
    max_turns: int,
    request_budget: ModelRequestBudget,
    reserve_requests: int,
    audit_logger: DeveloperAuditLogger,
    usage: Usage,
    stages_dir: Path,
    web_search_context: Literal["low", "medium", "high"],
) -> WebResearch:
    approved_queries = _bounded_web_queries(queries)
    if not approved_queries:
        output = WebResearch(
            error="No bounded technical web query was available."
        )
        _write_stage_artifact(stages_dir / "05_web_research.json", output)
        return output

    agent = Agent(
        name="PX4 Targeted Web Researcher",
        model=model,
        instructions=f"""
Stage role: bounded source-discovery or external-context research after an
insufficient local pass.
- You MUST use web search.
- Search only the precise technical gaps supplied by the evidence reviewer.
- Prefer official PX4 documentation, the PX4-Autopilot repository and its
  commits/pull requests/issues, then PX4 Discuss and primary references.
- Record version scope and exact URLs.
- Do not search for or disclose flight filenames, coordinates, identifiers,
  or unrelated logged details.
- Web results are supplementary. Do not claim they prove what happened in the
  flight.
- Produce source follow-up queries that the next pinned-source pass can test.
""",
        tools=[
            WebSearchTool(
                search_context_size=web_search_context,
                external_web_access=True,
            )
        ],
        model_settings=ModelSettings(tool_choice="web_search"),
        output_type=WebResearch,
    )
    payload = {
        "approved_queries": approved_queries,
        "privacy_boundary": (
            "No flight-derived context is supplied to this stage. Use only "
            "the approved technical queries."
        ),
    }
    try:
        result = await _run_agent_stage(
            stage="shell_analysis_web_fallback",
            artifact_name="05_web_research.json",
            agent=agent,
            prompt=json.dumps(payload, indent=2),
            expected_type=WebResearch,
            max_turns=min(max_turns, WEB_FALLBACK_MAX_TURNS),
            request_budget=request_budget,
            reserve_requests=reserve_requests,
            audit_logger=audit_logger,
            usage=usage,
            stages_dir=stages_dir,
            return_result=True,
        )
        output, raw_result = result
        provenance_items = [
            *(getattr(raw_result, "raw_responses", []) or []),
            *(getattr(raw_result, "new_items", []) or []),
        ]
        output.observed_citation_urls = _extract_urls(
            provenance_items
        )
        output.observed_queries = _extract_search_queries(provenance_items)
        unapproved_reported_queries = sorted(
            set(output.queries_used) - set(approved_queries)
        )
        unapproved_observed_queries = sorted(
            set(output.observed_queries) - set(approved_queries)
        )
        finding_urls = {finding.url for finding in output.findings}
        unobserved_finding_urls = sorted(
            finding_urls - set(output.observed_citation_urls)
        )
        provenance_errors: list[str] = []
        if unapproved_reported_queries:
            provenance_errors.append(
                "reported unapproved queries: "
                + ", ".join(unapproved_reported_queries)
            )
        if unapproved_observed_queries:
            provenance_errors.append(
                "observed unapproved queries: "
                + ", ".join(unapproved_observed_queries)
            )
        if unobserved_finding_urls:
            provenance_errors.append(
                "finding URLs were not observed in tool citations: "
                + ", ".join(unobserved_finding_urls)
            )
        if provenance_errors:
            output.error = "; ".join(provenance_errors)
            output.findings = []
            output.source_followup_queries = []
        audit_logger.log_event(
            "web_research.provenance_checked",
            output={
                "approved_queries": approved_queries,
                "observed_queries": output.observed_queries,
                "observed_citation_urls": output.observed_citation_urls,
                "error": output.error,
            },
        )
        _write_stage_artifact(stages_dir / "05_web_research.json", output)
        return output
    except Exception as exc:
        output = WebResearch(
            queries_used=approved_queries,
            error=f"{type(exc).__name__}: {exc}",
        )
        audit_logger.log_event(
            "agent.shell_analysis_web_fallback.failed",
            error=output.error,
        )
        _write_stage_artifact(stages_dir / "05_web_research.json", output)
        return output


async def _run_final_report(
    *,
    context: dict[str, Any],
    initial: InitialInspection,
    cycles: list[InvestigationCycle],
    final_state: EvidenceState,
    available_plot_artifacts: list[str],
    model: str,
    max_turns: int,
    request_budget: ModelRequestBudget,
    reserve_requests: int,
    project_instructions: str,
    audit_logger: DeveloperAuditLogger,
    usage: Usage,
    stages_dir: Path,
) -> FlightLogReport:
    agent = Agent(
        name="PX4 Evidence-Gated Report Writer",
        model=model,
        instructions=f"""
Write the existing FlightLogReport from the completed evidence workflow.
You have no tools.

Rules:
- Do not introduce a mechanism, value, source reference, or calculation that
  is absent from the supplied stage records.
- For causal questions, only candidates listed in
  final_evidence_state.accepted_candidate_titles may be medium/high confidence
  or appear in confirmed.
- Use the exact canonical title string from
  final_evidence_state.accepted_candidate_titles as the hypothesis title, not
  the earlier source-investigation title.
- If the final evidence state is unresolved, say so directly and present
  remaining candidates only as low/unresolved.
- Preserve downstream consistency as corroboration, not causal proof.
- Clearly separate ULog observations, exact-source behavior, web context,
  contradictions, and missing evidence.
- Web findings may explain context or suggest a source path, but only the
  subsequent pinned-source and ULog cycle can support the flight conclusion.
- Answer the user's actual question directly.
- Reference a plot only by an exact path from available_plot_artifacts. Do not
  invent a plot path.

Project instructions:
{project_instructions or "(none)"}
""",
        tools=[],
        model_settings=ModelSettings(tool_choice="none"),
        output_type=FlightLogReport,
    )
    payload = {
        "context": context,
        "initial_inspection": initial.model_dump(),
        "cycles": [cycle.model_dump() for cycle in cycles],
        "final_evidence_state": final_state.model_dump(),
        "available_plot_artifacts": available_plot_artifacts,
    }
    return await _run_agent_stage(
        stage="shell_analysis_final_report",
        artifact_name="09_draft_report.json",
        agent=agent,
        prompt=json.dumps(payload, indent=2),
        expected_type=FlightLogReport,
        max_turns=max_turns,
        request_budget=request_budget,
        reserve_requests=reserve_requests,
        audit_logger=audit_logger,
        usage=usage,
        stages_dir=stages_dir,
    )


async def _run_agent_stage(
    *,
    stage: str,
    artifact_name: str,
    agent: Agent[Any],
    prompt: str,
    expected_type: type[T],
    max_turns: int,
    request_budget: ModelRequestBudget,
    reserve_requests: int,
    audit_logger: DeveloperAuditLogger,
    usage: Usage,
    stages_dir: Path,
    return_result: bool = False,
    output_normalizer: Callable[[T], T] | None = None,
) -> T | tuple[T, Any]:
    remaining_requests = request_budget.limit - usage.requests
    allowed_turns = min(
        max_turns,
        remaining_requests - reserve_requests,
    )
    if allowed_turns < 1:
        raise RuntimeError(
            "Model request budget exhausted before "
            f"{stage} (limit={request_budget.limit}, used={usage.requests})."
        )

    started_at = time.perf_counter()
    audit_logger.log_event(f"agent.{stage}.started")
    hooks = _UsageTrackingAuditHooks(audit_logger)
    usage_recorded = False
    try:
        result = await Runner.run(
            agent,
            prompt,
            max_turns=allowed_turns,
            hooks=hooks,
        )
        log_run_items(audit_logger, getattr(result, "new_items", []) or [])
        stage_usage = getattr(
            getattr(result, "context_wrapper", None),
            "usage",
            None,
        )
        if isinstance(stage_usage, Usage):
            usage.add(stage_usage)
            usage_recorded = True
        elif hooks.observed_usage.requests:
            stage_usage = hooks.observed_usage
            usage.add(stage_usage)
            usage_recorded = True
        audit_logger.save_usage(usage)

        output = (
            result.final_output
            if isinstance(result.final_output, expected_type)
            else expected_type.model_validate(result.final_output)
        )
        if hasattr(output, "execution_receipts"):
            output.execution_receipts = _extract_execution_receipts(
                getattr(result, "new_items", []) or []
            )
        if output_normalizer is not None:
            output = output_normalizer(output)
        _write_stage_artifact(stages_dir / artifact_name, output)
        audit_logger.log_event(
            f"agent.{stage}.finished",
            output=output.model_dump(),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            usage=stage_usage,
        )
        if return_result:
            return output, result
        return output
    except Exception as exc:
        if not usage_recorded and hooks.observed_usage.requests:
            usage.add(hooks.observed_usage)
            audit_logger.save_usage(usage)
        audit_logger.log_event(
            f"agent.{stage}.failed",
            error=repr(exc),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
        raise


def _compact_inventory(
    inventory: dict[str, Any],
    source_snapshot: SourceSnapshot | None,
) -> dict[str, Any]:
    return {
        "firmware_version": inventory.get("firmware_version"),
        "firmware_branch": inventory.get("firmware_branch"),
        "logged_git_hash": inventory.get("git_hash"),
        "resolved_source_commit": (
            source_snapshot.commit_sha if source_snapshot else None
        ),
        "airframe": inventory.get("airframe"),
        "duration_s": inventory.get("duration_s"),
        "available_topics": inventory.get("available_topics") or [],
        "missing_topics": inventory.get("missing_topics") or [],
        "warnings": inventory.get("warnings") or [],
    }


def _inventory_airframe_summary(inventory: dict[str, Any]) -> str:
    parts: list[str] = []
    firmware_version = inventory.get("firmware_version")
    git_hash = inventory.get("git_hash")
    airframe = inventory.get("airframe")
    if firmware_version:
        parts.append(f"PX4 {firmware_version}")
    if git_hash:
        parts.append(f"git {str(git_hash)[:12]}")
    if airframe:
        parts.append(
            "airframe="
            + json.dumps(
                airframe,
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            )
        )
    return (
        "; ".join(parts)
        if parts
        else "Airframe context unavailable from deterministic inventory."
    )


def _write_stage_artifact(path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json")
    execution_receipts = getattr(value, "execution_receipts", None)
    if isinstance(execution_receipts, list):
        payload["execution_receipts"] = [
            {
                "command": receipt.command,
                "exit_code": receipt.exit_code,
                "timed_out": receipt.timed_out,
                "receipt_id": receipt.receipt_id,
                "stdout_sha256": receipt.stdout_sha256,
                "stdout_chars": receipt.stdout_chars,
                "stdout": receipt.stdout,
            }
            for receipt in execution_receipts
            if isinstance(receipt, ShellExecutionReceipt)
        ]
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _record_evidence_state(
    path: Path,
    state: EvidenceState,
    audit_logger: DeveloperAuditLogger,
    *,
    final: bool = False,
) -> None:
    _write_stage_artifact(path, state)
    audit_logger.log_event(
        "evidence_state.final" if final else "evidence_state.transition",
        output=state.model_dump(),
    )


def _bounded_web_queries(queries: list[str]) -> list[str]:
    bounded: list[str] = []
    for query in queries:
        normalized = " ".join(query.split())[:MAX_WEB_QUERY_CHARS].strip()
        if normalized and normalized not in bounded:
            bounded.append(normalized)
        if len(bounded) == MAX_WEB_QUERIES:
            break
    return bounded


def _available_plot_artifacts(plots_dir: Path) -> list[str]:
    plot_root = plots_dir.resolve()
    artifacts: list[str] = []
    if not plot_root.is_dir():
        return artifacts
    for candidate in sorted(plot_root.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(plot_root)
        except ValueError:
            continue
        artifacts.append(f"/plots/{relative.as_posix()}")
    return artifacts


def _normalize_report_plot_paths(
    report: FlightLogReport,
    *,
    plots_dir: Path,
    available_plot_artifacts: list[str],
) -> FlightLogReport:
    plot_root = plots_dir.resolve()
    alias_to_host_path = {
        alias: str((plot_root / alias.removeprefix("/plots/")).resolve())
        for alias in available_plot_artifacts
    }
    for hypothesis in report.ranked_hypotheses:
        for plot in hypothesis.plots:
            if not plot.path:
                continue
            host_path = alias_to_host_path.get(plot.path)
            if host_path is None:
                plot.warnings.append(
                    "The report referenced a plot that was not produced "
                    "during this analysis."
                )
                plot.path = ""
                continue
            plot.path = host_path
    return report


def _extract_execution_receipts(
    items: list[Any],
) -> list[ShellExecutionReceipt]:
    receipts: list[ShellExecutionReceipt] = []
    for item in items:
        raw_item = _model_mapping(getattr(item, "raw_item", None))
        if raw_item.get("type") not in {
            "shell_call_output",
            "local_shell_call_output",
        }:
            continue
        raw_entries = raw_item.get("shell_output") or raw_item.get("output")
        if not isinstance(raw_entries, list):
            continue
        for raw_entry in raw_entries:
            entry = _model_mapping(raw_entry)
            command = entry.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            outcome = _model_mapping(entry.get("outcome"))
            exit_code = entry.get("exit_code", outcome.get("exit_code"))
            if not isinstance(exit_code, int):
                exit_code = None
            raw_stdout = (
                entry.get("stdout")
                if isinstance(entry.get("stdout"), str)
                else ""
            )
            bounded_stdout = _bounded_receipt_stdout(raw_stdout)
            stdout_sha256 = hashlib.sha256(
                bounded_stdout.encode("utf-8")
            ).hexdigest()
            timed_out = (
                entry.get("status") == "timeout"
                or outcome.get("type") == "timeout"
            )
            receipt_material = (
                f"{command}\0{exit_code}\0{timed_out}\0{stdout_sha256}"
            )
            receipts.append(
                ShellExecutionReceipt(
                    command=command,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    receipt_id=hashlib.sha256(
                        receipt_material.encode("utf-8")
                    ).hexdigest()[:24],
                    stdout_sha256=stdout_sha256,
                    stdout_chars=len(bounded_stdout),
                    stdout=bounded_stdout,
                )
            )
    return receipts


def _bounded_receipt_stdout(stdout: str) -> str:
    if len(stdout) <= MAX_RECEIPT_STDOUT_CHARS:
        return stdout
    omitted = len(stdout) - MAX_RECEIPT_STDOUT_CHARS
    return (
        stdout[:MAX_RECEIPT_STDOUT_CHARS]
        + f"\n[receipt stdout truncated; {omitted} characters omitted]"
    )


def _successful_execution_commands(
    receipts: list[ShellExecutionReceipt],
) -> set[str]:
    return {
        receipt.command
        for receipt in receipts
        if not receipt.timed_out and receipt.exit_code == 0
    }


def _observation_receipt_payload(
    observation_index: int,
    observation: LogObservation,
    *,
    question_focus: str,
) -> dict[str, Any]:
    return {
        "observation_index": observation_index,
        "question_focus": question_focus,
        "finding": observation.finding,
        "evidence": observation.evidence,
        "topic_fields": observation.topic_fields,
        "time_or_window": observation.time_or_window,
    }


def _targeted_check_receipt_payload(
    check: EvidenceCheck,
    candidate: CandidateMechanism,
    targeted: TargetedLogParse,
) -> dict[str, Any]:
    required_parameter_names = set(candidate.required_parameters)
    return {
        "check_id": check.check_id,
        "candidate_id": check.candidate_id,
        "evidence_role": check.evidence_role,
        "window": check.window,
        "topic_fields": check.topic_fields,
        "requirement_ids": check.evaluated_requirement_ids,
        "parameters": [
            [parameter.name, parameter.value]
            for parameter in targeted.parameter_values
            if parameter.name in required_parameter_names
        ],
        "result": check.result,
        "assessment": check.assessment,
    }


def _reviewed_check_receipt_payload(
    reviewed_check: ReviewedEvidenceCheck,
    targeted_check: EvidenceCheck,
) -> dict[str, Any]:
    return {
        "check_id": reviewed_check.check_id,
        "independent_result": reviewed_check.independent_result,
        "assessment": reviewed_check.assessment,
        "window": targeted_check.window,
        "topic_fields": targeted_check.topic_fields,
        "requirement_ids": targeted_check.evaluated_requirement_ids,
    }


def _reviewed_observation_receipt_payload(
    reviewed_observation: ReviewedObservation,
    observation: LogObservation,
    *,
    question_focus: str,
) -> dict[str, Any]:
    return {
        "observation_index": reviewed_observation.observation_index,
        "question_focus": question_focus,
        "independent_result": reviewed_observation.independent_result,
        "assessment": reviewed_observation.assessment,
        "topic_fields": observation.topic_fields,
    }


def _receipts_contain_json_record(
    receipts: list[ShellExecutionReceipt],
    *,
    name: str,
    expected: dict[str, Any],
    programs: set[str] | None = None,
) -> bool:
    prefix = f"{name}_json="
    for receipt in receipts:
        if (
            receipt.timed_out
            or receipt.exit_code != 0
            or (
                programs is not None
                and _command_program(receipt.command) not in programs
            )
        ):
            continue
        for line in receipt.stdout.splitlines():
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            try:
                record = json.loads(stripped[len(prefix):])
            except (TypeError, ValueError):
                continue
            if record == expected:
                return True
    return False


def _commands_contain_json_record(
    receipts: list[ShellExecutionReceipt],
    commands: list[str],
    *,
    name: str,
    expected: dict[str, Any],
    programs: set[str] | None = None,
) -> bool:
    command_set = set(commands)
    return _receipts_contain_json_record(
        [
            receipt
            for receipt in receipts
            if receipt.command in command_set
        ],
        name=name,
        expected=expected,
        programs=programs,
    )


def _receipt_integrity_valid(receipt: ShellExecutionReceipt) -> bool:
    has_integrity_metadata = bool(
        receipt.receipt_id
        or receipt.stdout_sha256
        or receipt.stdout_chars
    )
    if not has_integrity_metadata:
        return True
    if not receipt.receipt_id or not receipt.stdout_sha256:
        return False
    if receipt.stdout_chars != len(receipt.stdout):
        return False
    stdout_sha256 = hashlib.sha256(
        receipt.stdout.encode("utf-8")
    ).hexdigest()
    if stdout_sha256 != receipt.stdout_sha256:
        return False
    receipt_material = (
        f"{receipt.command}\0{receipt.exit_code}\0{receipt.timed_out}\0"
        f"{stdout_sha256}"
    )
    expected_receipt_id = hashlib.sha256(
        receipt_material.encode("utf-8")
    ).hexdigest()[:24]
    return receipt.receipt_id == expected_receipt_id


def _command_program(command: str) -> str:
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return ""
    return Path(argv[0]).name if argv else ""


def _commands_have_same_argv(first: str, second: str) -> bool:
    try:
        return shlex.split(first, posix=True) == shlex.split(
            second,
            posix=True,
        )
    except ValueError:
        return False


def _commands_contain_exact_text(
    receipts: list[ShellExecutionReceipt],
    commands: list[str],
    text: str,
) -> bool:
    expected = text.strip()
    command_set = set(commands)
    if not expected:
        return False
    return any(
        not receipt.timed_out
        and receipt.exit_code == 0
        and receipt.command in command_set
        and expected in receipt.stdout
        for receipt in receipts
    )


def _source_command_reads_file(command: str, file_name: str) -> bool:
    canonical_file = _canonical_source_file_from_command(command)
    if canonical_file is None:
        return False
    expected = file_name.replace("\\", "/").lstrip("/")
    return canonical_file == expected


def _canonical_source_file_from_command(command: str) -> str | None:
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return None
    if (
        len(argv) != 5
        or argv[0] != "git"
        or argv[1] != "-C"
        or argv[3] != "show"
    ):
        return None
    alias_parts = argv[2].replace("\\", "/").split("/")
    if (
        not alias_parts
        or alias_parts[0] != SOURCE_ROOT_ALIAS
        or any(part in {"", ".", ".."} for part in alias_parts)
    ):
        return None
    prefix = "SNAPSHOT:"
    if not argv[4].startswith(prefix):
        return None
    snapshot_parts = argv[4][len(prefix):].replace("\\", "/").split("/")
    if (
        not snapshot_parts
        or any(part in {"", ".", ".."} for part in snapshot_parts)
    ):
        return None
    return "/".join([*alias_parts[1:], *snapshot_parts])


def _canonicalize_source_investigation_files(
    source: SourceInvestigation,
) -> None:
    steps = [
        *(
            [source.publishing_assignment]
            if source.publishing_assignment is not None
            else []
        ),
        *(
            step
            for candidate in source.candidates
            for step in candidate.upstream_assignment_path
        ),
    ]
    for step in steps:
        canonical_files = {
            canonical
            for command in step.execution_commands
            if (
                canonical := _canonical_source_file_from_command(command)
            )
            is not None
        }
        if len(canonical_files) == 1:
            step.file = canonical_files.pop()
    binding = source.publication_binding
    if binding is None:
        return
    for file_attribute, commands in (
        (
            "declaration_file",
            binding.declaration_execution_commands,
        ),
        ("schema_file", binding.schema_execution_commands),
    ):
        canonical_files = {
            canonical
            for command in commands
            if (
                canonical := _canonical_source_file_from_command(command)
            )
            is not None
        }
        if len(canonical_files) == 1:
            setattr(binding, file_attribute, canonical_files.pop())
    for nested_binding in binding.nested_schema_bindings:
        canonical_files = {
            canonical
            for command in nested_binding.schema_execution_commands
            if (
                canonical := _canonical_source_file_from_command(command)
            )
            is not None
        }
        if len(canonical_files) == 1:
            nested_binding.schema_file = canonical_files.pop()


def _source_line_range_matches(
    receipts: list[ShellExecutionReceipt],
    commands: list[str],
    source_excerpt: str,
    claimed_lines: str,
) -> bool:
    claimed_match = re.fullmatch(
        r"(?P<start>[1-9][0-9]*)(?:-(?P<end>[1-9][0-9]*))?",
        claimed_lines.strip(),
    )
    if claimed_match is None:
        return False
    claimed_start = int(claimed_match.group("start"))
    claimed_end = int(
        claimed_match.group("end") or claimed_match.group("start")
    )
    if claimed_start > claimed_end:
        return False
    expected = source_excerpt.strip()
    command_set = set(commands)
    for receipt in receipts:
        if (
            receipt.timed_out
            or receipt.exit_code != 0
            or receipt.command not in command_set
        ):
            continue
        source_lines = receipt.stdout.splitlines()
        if claimed_end > len(source_lines):
            continue
        selected_excerpt = "\n".join(
            source_lines[claimed_start - 1:claimed_end]
        ).strip()
        if selected_excerpt == expected:
            return True
    return False


def _parse_source_line_range(
    claimed_lines: str,
) -> tuple[int, int] | None:
    claimed_match = re.fullmatch(
        r"(?P<start>[1-9][0-9]*)(?:-(?P<end>[1-9][0-9]*))?",
        claimed_lines.strip(),
    )
    if claimed_match is None:
        return None
    start = int(claimed_match.group("start"))
    end = int(claimed_match.group("end") or start)
    return (start, end) if start <= end else None


def _publication_binding_is_valid(
    binding: PublicationBinding,
    *,
    publishing_assignment: SourceLineageStep,
    expected_logged_value: str,
    receipts: list[ShellExecutionReceipt],
    successful_commands: set[str],
) -> bool:
    if binding.logged_value.strip() != expected_logged_value:
        return False
    logged_path = _logged_value_path_components(expected_logged_value)
    if logged_path is None:
        return False
    logged_topic, logged_components = logged_path
    logged_fields = [
        component.name for component in logged_components
    ]
    declaration_commands = _valid_binding_commands(
        binding.declaration_execution_commands,
        file_name=binding.declaration_file,
        excerpt=binding.declaration_excerpt,
        lines=binding.declaration_lines,
        receipts=receipts,
        successful_commands=successful_commands,
    )
    if not declaration_commands:
        return False
    publishing_lhs = _publishing_lhs_path(publishing_assignment)
    if publishing_lhs is None:
        source_components = _designated_initializer_lhs_path(
            publishing_assignment
        )
        if (
            source_components is None
            or not _designated_initializer_is_bound(
                binding,
                publishing_assignment=publishing_assignment,
            )
        ):
            return False
        publishing_identifier = binding.message_identifier
    else:
        publishing_identifier, source_components = publishing_lhs
    if (
        publishing_identifier != binding.message_identifier
        or [component.name for component in source_components]
        != logged_fields
        or len(source_components) != len(logged_components)
    ):
        return False
    for source_component, logged_component in zip(
        source_components,
        logged_components,
    ):
        if len(source_component.index_expressions) != len(
            logged_component.indices
        ):
            return False
        for expression, logged_index in zip(
            source_component.index_expressions,
            logged_component.indices,
        ):
            source_index = _cpp_literal_index(expression)
            if source_index is not None and source_index != logged_index:
                return False
    declaration_tokens = _source_data_tokens(
        binding.declaration_excerpt
    )
    if not {
        binding.message_identifier,
        binding.message_type_identifier,
    }.issubset(declaration_tokens):
        return False
    if not _declaration_binds_message_type(
        binding.declaration_excerpt,
        message_identifier=binding.message_identifier,
        message_type_identifier=binding.message_type_identifier,
    ):
        return False
    schema_path = Path(binding.schema_file)
    if schema_path.suffix != ".msg":
        return False
    schema_name = _camel_to_snake(schema_path.stem)
    if binding.message_type_identifier != f"{schema_name}_s":
        return False
    schema_bindings: list[SchemaSourceBinding] = [
        binding,
        *binding.nested_schema_bindings,
    ]
    if len(schema_bindings) != len(logged_fields):
        return False
    validated_schemas: list[
        tuple[
            list[str],
            dict[str, _SchemaFieldDefinition],
            dict[str, _SchemaFieldDefinition],
        ]
    ] = []
    for schema_binding in schema_bindings:
        validated_schema = _validated_schema_binding(
            schema_binding,
            receipts=receipts,
            successful_commands=successful_commands,
        )
        if validated_schema is None:
            return False
        validated_schemas.append(validated_schema)

    root_schema_texts, _, _ = validated_schemas[0]
    declared_topic_names = {
        topic_name
        for schema_text in root_schema_texts
        for topic_name in _schema_declared_topic_names(schema_text)
    }
    if logged_topic not in (declared_topic_names or {schema_name}):
        return False

    for index, logged_component in enumerate(logged_components):
        field_name = logged_component.name
        _, schema_fields, excerpt_fields = validated_schemas[index]
        field_definition = schema_fields.get(field_name)
        if (
            field_definition is None
            or excerpt_fields.get(field_name) != field_definition
            or not _schema_array_shape_matches(
                field_definition,
                logged_component,
            )
        ):
            return False
        if index == len(logged_fields) - 1:
            if _schema_message_reference_name(
                field_definition.type_name
            ) is not None:
                return False
            continue
        referenced_schema_name = _schema_message_reference_name(
            field_definition.type_name
        )
        if referenced_schema_name is None:
            return False
        next_schema_path = Path(schema_bindings[index + 1].schema_file)
        if (
            next_schema_path.suffix != ".msg"
            or next_schema_path.stem != referenced_schema_name
        ):
            return False
    return True


def _source_lineage_steps_match(
    candidate_step: SourceLineageStep,
    publishing_step: SourceLineageStep,
) -> bool:
    return (
        candidate_step.file == publishing_step.file
        and candidate_step.lines.strip() == publishing_step.lines.strip()
        and candidate_step.source_excerpt.strip()
        == publishing_step.source_excerpt.strip()
        and candidate_step.role == publishing_step.role
    )


def _valid_binding_commands(
    commands: list[str],
    *,
    file_name: str,
    excerpt: str,
    lines: str,
    receipts: list[ShellExecutionReceipt],
    successful_commands: set[str],
) -> set[str]:
    if (
        not commands
        or not set(commands).issubset(successful_commands)
        or not all(_command_program(command) == "git" for command in commands)
    ):
        return set()
    file_commands = {
        command
        for command in commands
        if _source_command_reads_file(command, file_name)
    }
    if not file_commands:
        return set()
    if not _commands_contain_exact_text(
        receipts,
        list(file_commands),
        excerpt,
    ):
        return set()
    if not _source_line_range_matches(
        receipts,
        list(file_commands),
        excerpt,
        lines,
    ):
        return set()
    return file_commands


def _validated_schema_binding(
    binding: SchemaSourceBinding,
    *,
    receipts: list[ShellExecutionReceipt],
    successful_commands: set[str],
) -> tuple[
    list[str],
    dict[str, _SchemaFieldDefinition],
    dict[str, _SchemaFieldDefinition],
] | None:
    if Path(binding.schema_file).suffix != ".msg":
        return None
    schema_commands = _valid_binding_commands(
        binding.schema_execution_commands,
        file_name=binding.schema_file,
        excerpt=binding.schema_excerpt,
        lines=binding.schema_lines,
        receipts=receipts,
        successful_commands=successful_commands,
    )
    if not schema_commands:
        return None
    schema_texts = [
        receipt.stdout
        for receipt in receipts
        if (
            not receipt.timed_out
            and receipt.exit_code == 0
            and receipt.command in schema_commands
        )
    ]
    if not schema_texts:
        return None
    schema_fields: dict[str, _SchemaFieldDefinition] = {}
    for schema_text in schema_texts:
        for field_name, field_definition in _schema_field_types(
            schema_text
        ).items():
            prior_definition = schema_fields.get(field_name)
            if (
                prior_definition is not None
                and prior_definition != field_definition
            ):
                return None
            schema_fields[field_name] = field_definition
    return (
        schema_texts,
        schema_fields,
        _schema_field_types(binding.schema_excerpt),
    )


def _publication_bindings_match(
    source_binding: PublicationBinding,
    reviewed_binding: ReviewedPublicationBinding,
) -> bool:
    fields = (
        "logged_value",
        "message_identifier",
        "message_type_identifier",
        "declaration_file",
        "declaration_lines",
        "declaration_excerpt",
        "schema_file",
        "schema_lines",
        "schema_excerpt",
    )
    if not all(
        getattr(source_binding, field) == getattr(reviewed_binding, field)
        for field in fields
    ):
        return False
    if len(source_binding.nested_schema_bindings) != len(
        reviewed_binding.nested_schema_bindings
    ):
        return False
    nested_fields = (
        "schema_file",
        "schema_lines",
        "schema_excerpt",
    )
    return all(
        all(
            getattr(source_nested, field)
            == getattr(reviewed_nested, field)
            for field in nested_fields
        )
        for source_nested, reviewed_nested in zip(
            source_binding.nested_schema_bindings,
            reviewed_binding.nested_schema_bindings,
        )
    )


def _declaration_binds_message_type(
    source_excerpt: str,
    *,
    message_identifier: str,
    message_type_identifier: str,
) -> bool:
    message = re.escape(message_identifier)
    message_type = re.escape(message_type_identifier)
    namespace = r"(?:::)?(?:[A-Za-z_][A-Za-z0-9_]*::)*"
    qualifiers = (
        r"(?:(?:const|volatile|static|constexpr|thread_local)\s+)*"
    )
    direct_declaration = re.compile(
        rf"^\s*{qualifiers}(?:struct\s+)?{namespace}{message_type}"
        rf"\s*[*&]*\s*{message}\b"
    )
    auto_declaration = re.compile(
        rf"^\s*{qualifiers}auto\s+{message}\s*=\s*"
        rf"{namespace}{message_type}\b"
    )
    structural_excerpt = _mask_source_non_code(source_excerpt)
    return any(
        direct_declaration.search(line) is not None
        or auto_declaration.search(line) is not None
        for line in structural_excerpt.splitlines()
    )


def _designated_initializer_is_bound(
    binding: PublicationBinding,
    *,
    publishing_assignment: SourceLineageStep,
) -> bool:
    if publishing_assignment.file != binding.declaration_file:
        return False
    declaration_range = _parse_source_line_range(
        binding.declaration_lines
    )
    publishing_range = _parse_source_line_range(
        publishing_assignment.lines
    )
    if (
        declaration_range is None
        or publishing_range is None
        or declaration_range[0] > publishing_range[0]
        or declaration_range[1] < publishing_range[1]
    ):
        return False
    declaration_text = _mask_source_non_code(
        binding.declaration_excerpt
    )
    publishing_text = _mask_source_non_code(
        publishing_assignment.source_excerpt
    ).strip()
    if not publishing_text:
        return False
    message = re.escape(binding.message_identifier)
    message_type = re.escape(binding.message_type_identifier)
    namespace = r"(?:::)?(?:[A-Za-z_][A-Za-z0-9_]*::)*"
    declaration_pattern = re.compile(
        rf"{namespace}{message_type}\s*[*&]*\s*{message}\s*\{{"
    )
    for initializer_match in re.finditer(
        re.escape(publishing_text),
        declaration_text,
    ):
        declarations = [
            match
            for match in declaration_pattern.finditer(
                declaration_text,
                0,
                initializer_match.start(),
            )
        ]
        if not declarations:
            continue
        opening = declarations[-1]
        opening_brace = declaration_text.find(
            "{",
            opening.start(),
            opening.end(),
        )
        if opening_brace < 0:
            continue
        containing_prefix = declaration_text[
            opening_brace:initializer_match.start()
        ]
        if (
            containing_prefix.count("{")
            - containing_prefix.count("}")
            == 1
        ):
            return True
    return False


def _camel_to_snake(value: str) -> str:
    with_word_boundaries = re.sub(
        r"(.)([A-Z][a-z]+)",
        r"\1_\2",
        value,
    )
    return re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        with_word_boundaries,
    ).lower()


def _schema_declared_topic_names(schema_text: str) -> set[str]:
    topics: set[str] = set()
    for line in schema_text.splitlines():
        prefix = "# TOPICS "
        if not line.startswith(prefix):
            continue
        topics.update(
            token
            for token in line[len(prefix):].split()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token)
        )
    return topics


def _schema_field_types(
    schema_text: str,
) -> dict[str, _SchemaFieldDefinition]:
    fields: dict[str, _SchemaFieldDefinition] = {}
    for line in schema_text.splitlines():
        declaration = line.split("#", 1)[0].strip()
        if not declaration or "=" in declaration:
            continue
        match = re.fullmatch(
            r"(?P<type>[A-Za-z][A-Za-z0-9_/]*)"
            r"(?P<arrays>(?:\[[^\[\]]+\])*)"
            r"\s+(?P<field>[A-Za-z_][A-Za-z0-9_]*)",
            declaration,
        )
        if match is not None:
            field_name = match.group("field")
            if field_name in fields:
                return {}
            fields[field_name] = _SchemaFieldDefinition(
                type_name=match.group("type"),
                array_extents=tuple(
                    extent.strip()
                    for extent in re.findall(
                        r"\[([^\[\]]+)\]",
                        match.group("arrays"),
                    )
                ),
            )
    return fields


def _schema_field_names(schema_text: str) -> set[str]:
    return set(_schema_field_types(schema_text))


def _schema_message_reference_name(type_name: str) -> str | None:
    unqualified = type_name.rsplit("/", 1)[-1]
    return (
        unqualified
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", unqualified)
        else None
    )


def _schema_array_shape_matches(
    field_definition: _SchemaFieldDefinition,
    logged_component: _LoggedFieldComponent,
) -> bool:
    if len(field_definition.array_extents) != len(
        logged_component.indices
    ):
        return False
    for extent, index in zip(
        field_definition.array_extents,
        logged_component.indices,
    ):
        if extent.isdecimal() and index >= int(extent):
            return False
    return True


def _source_step_fields_are_grounded(step: SourceLineageStep) -> bool:
    excerpt = _normalize_evidence_text(step.source_excerpt)
    operation = _normalize_evidence_text(step.operation)
    if not excerpt or not operation or operation != excerpt:
        return False
    excerpt_tokens = _source_data_tokens(step.source_excerpt)
    declared_input_tokens = _source_data_tokens(step.input_or_state)
    declared_output_tokens = _source_data_tokens(step.output_or_effect)
    return bool(
        (
            declared_input_tokens.issubset(excerpt_tokens)
            or (
                step.role == "origin"
                and not declared_input_tokens
            )
        )
        and declared_output_tokens
        and declared_output_tokens.issubset(excerpt_tokens)
    )


def _grounded_step_tokens(
    step: SourceLineageStep,
    field_value: str,
) -> set[str]:
    return _source_data_tokens(step.source_excerpt) & _source_data_tokens(
        field_value
    )


def _source_data_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.finditer(
        r"(?P<raw_string>(?<![A-Za-z0-9_])"
        r"(?:u8|u|U|L)?R\"(?P<raw_delimiter>"
        r"[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=raw_delimiter)\")"
        r"|(?P<comment>//[^\n]*|/\*.*?\*/)"
        r"|(?P<string>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
        r"|(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)"
        r"|(?P<number>(?:"
        r"0[xX](?:"
        r"[0-9A-Fa-f](?:'?[0-9A-Fa-f])*"
        r"(?:\.(?:[0-9A-Fa-f](?:'?[0-9A-Fa-f])*)?)?"
        r"|\.[0-9A-Fa-f](?:'?[0-9A-Fa-f])*"
        r")(?:[pP][+-]?\d(?:'?\d)*)?"
        r"|0[bB][01](?:'?[01])*"
        r"|(?:\d(?:'?\d)*(?:\.(?:\d(?:'?\d)*)?)?"
        r"|\.\d(?:'?\d)*)(?:[eE][+-]?\d(?:'?\d)*)?"
        r")[fFlLuU]*)",
        value,
        flags=re.DOTALL,
    ):
        if match.group("comment") is not None:
            continue
        token = match.group()
        if (
            match.group("raw_string") is not None
            or match.group("string") is not None
        ):
            tokens.add(f"literal:{token}")
        else:
            tokens.add(token)
    return tokens


def _source_assignment_direction_is_valid(
    step: SourceLineageStep,
) -> bool:
    if not _assignment_operator_matches(step.source_excerpt):
        return True
    assignment = _simple_assignment_tokens(step.source_excerpt)
    if assignment is None:
        return False
    produced_tokens, consumed_tokens, is_compound = assignment
    claimed_input_tokens = _grounded_step_tokens(
        step,
        step.input_or_state,
    )
    claimed_output_tokens = _grounded_step_tokens(
        step,
        step.output_or_effect,
    )
    permitted_input_tokens = (
        produced_tokens | consumed_tokens
        if is_compound
        else consumed_tokens
    )
    return bool(
        claimed_output_tokens
        and claimed_output_tokens.issubset(produced_tokens)
        and claimed_input_tokens.issubset(permitted_input_tokens)
    )


def _simple_assignment_tokens(
    operation: str,
) -> tuple[set[str], set[str], bool] | None:
    matches = _assignment_operator_matches(operation)
    if not matches:
        return None
    if len(matches) > 1 and any(
        match.group("operator") != "=" for match in matches
    ):
        return None
    produced_tokens: set[str] = set()
    address_or_control_tokens: set[str] = set()
    previous_end = 0
    for index, match in enumerate(matches):
        left_side = operation[previous_end:match.start()]
        lhs_tokens = _assignment_lhs_tokens(
            left_side,
            strict=index > 0,
        )
        if lhs_tokens is None:
            return None
        lhs_produced, lhs_inputs = lhs_tokens
        produced_tokens.update(lhs_produced)
        address_or_control_tokens.update(lhs_inputs)
        if match.group("operator") != "=":
            address_or_control_tokens.update(lhs_produced)
        previous_end = match.end()
    right_tokens = _source_data_tokens(operation[previous_end:])
    if not produced_tokens or not right_tokens:
        return None
    return (
        produced_tokens,
        right_tokens | address_or_control_tokens,
        any(match.group("operator") != "=" for match in matches),
    )


def _assignment_operator_matches(
    operation: str,
) -> list[re.Match[str]]:
    code_only = _mask_source_non_code(operation)
    return list(
        re.finditer(
            r"(?P<operator><<=|>>=|\+=|-=|\*=|/=|%=|&=|\|=|\^="
            r"|(?<![=!<>])=(?!=))",
            code_only,
        )
    )


def _mask_source_non_code(value: str) -> str:
    """Mask C/C++ comments and quoted literals without changing offsets."""

    masked = list(value)
    index = 0
    state: Literal[
        "code",
        "line_comment",
        "block_comment",
        "single_quote",
        "double_quote",
    ] = "code"
    while index < len(value):
        char = value[index]
        next_char = value[index + 1] if index + 1 < len(value) else ""
        if state == "code":
            raw_string_end = _cpp_raw_string_end(value, index)
            if raw_string_end is not None:
                for raw_index in range(index, raw_string_end):
                    masked[raw_index] = (
                        "\n" if value[raw_index] == "\n" else " "
                    )
                index = raw_string_end
                continue
            if char == "/" and next_char == "/":
                masked[index] = " "
                masked[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if char == "/" and next_char == "*":
                masked[index] = " "
                masked[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if char == "'":
                if not _is_cpp_digit_separator(value, index):
                    masked[index] = " "
                    state = "single_quote"
            elif char == '"':
                masked[index] = " "
                state = "double_quote"
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                masked[index] = " "
            index += 1
            continue
        if state == "block_comment":
            masked[index] = "\n" if char == "\n" else " "
            if char == "*" and next_char == "/":
                masked[index + 1] = " "
                state = "code"
                index += 2
            else:
                index += 1
            continue
        masked[index] = "\n" if char == "\n" else " "
        if char == "\\" and next_char:
            masked[index + 1] = (
                "\n" if next_char == "\n" else " "
            )
            index += 2
            continue
        if (
            state == "single_quote"
            and char == "'"
        ) or (
            state == "double_quote"
            and char == '"'
        ):
            state = "code"
        index += 1
    return "".join(masked)


def _cpp_raw_string_end(value: str, index: int) -> int | None:
    opening = _CPP_RAW_STRING_OPENING_RE.match(value, index)
    if opening is None:
        return None
    closing = ")" + opening.group("delimiter") + '"'
    content_start = opening.end()
    closing_start = value.find(closing, content_start)
    return (
        len(value)
        if closing_start < 0
        else closing_start + len(closing)
    )


def _is_cpp_digit_separator(value: str, index: int) -> bool:
    if (
        index <= 0
        or index + 1 >= len(value)
        or not value[index - 1].isalnum()
        or not value[index + 1].isalnum()
    ):
        return False
    token_start = index - 1
    while (
        token_start > 0
        and value[token_start - 1] in "0123456789abcdefABCDEFxXbB.'+-eEpP"
    ):
        token_start -= 1
    numeric_prefix = value[token_start:index].replace("'", "")
    next_char = value[index + 1]
    if re.fullmatch(r"0[xX][0-9A-Fa-f]+", numeric_prefix):
        return bool(re.fullmatch(r"[0-9A-Fa-f]", next_char))
    if re.fullmatch(r"0[bB][01]+", numeric_prefix):
        return next_char in "01"
    if re.fullmatch(
        r"0[xX](?:"
        r"[0-9A-Fa-f]+(?:\.[0-9A-Fa-f]*)?"
        r"|\.[0-9A-Fa-f]+"
        r")(?:[pP][+-]?\d*)?",
        numeric_prefix,
    ):
        return bool(
            re.fullmatch(
                r"[0-9A-Fa-f]",
                next_char,
            )
            if "p" not in numeric_prefix.lower()
            else next_char.isdigit()
        )
    return bool(
        re.fullmatch(
            r"(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eEpP][+-]?\d*)?",
            numeric_prefix,
        )
        and next_char.isdigit()
    )


def _assignment_lhs_tokens(
    left_side: str,
    *,
    strict: bool = False,
) -> tuple[set[str], set[str]] | None:
    matched_target = _assignment_lhs_target(left_side)
    if matched_target is None:
        return None
    target, target_start = matched_target
    dereferenced = re.fullmatch(
        r"\(\s*[*&]+\s*(?P<inner>.*?)\s*\)",
        target,
        flags=re.DOTALL,
    )
    target_core = (
        dereferenced.group("inner")
        if dereferenced is not None
        else target
    )
    address_tokens = set(
        token
        for bracket in re.findall(r"\[([^\[\]]+)\]", target_core)
        for token in _source_data_tokens(bracket)
    )
    address_tokens.update(
        token
        for arguments in re.findall(r"\(([^()]*)\)", target_core)
        for token in _source_data_tokens(arguments)
    )
    address_tokens.update(
        token
        for arguments in re.findall(r"<([^<>()]+)>", target_core)
        for token in _source_data_tokens(arguments)
    )
    target_without_accesses = re.sub(
        r"\[[^\[\]]+\]|\([^()]*\)|<[^<>()]+>",
        "",
        target_core,
    )
    produced_tokens = _source_data_tokens(target_without_accesses)
    if not produced_tokens:
        return None
    if (
        dereferenced is not None
        or any(
            operator in target_core
            for operator in (".", "->", "[", "(")
        )
    ):
        base_match = re.match(
            r"\s*[A-Za-z_][A-Za-z0-9_]*",
            target_core,
        )
        if base_match is not None:
            address_tokens.add(base_match.group().strip())
    prefix = left_side[:target_start]
    if strict and _mask_source_non_code(prefix).strip().strip("()"):
        return None
    address_tokens.update(
        token
        for group in re.findall(r"\(([^()]*)\)", prefix)
        for token in _source_data_tokens(group)
    )
    return produced_tokens, address_tokens


def _assignment_lhs_target(
    left_side: str,
) -> tuple[str, int] | None:
    structural_left_side = _mask_source_non_code(left_side)
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    lvalue_core = (
        identifier
        + rf"(?:\s*(?:(?:\.|->)\s*{identifier}"
        r"(?:\s*<[^<>()]+>)?"
        r"|\[[^\[\]]+\]|\([^()]*\)))*"
    )
    target_match = re.search(
        rf"(?P<target>(?:\(\s*[*&]+\s*{lvalue_core}\s*\)"
        rf"|{lvalue_core}))\s*$",
        structural_left_side,
    )
    if target_match is None:
        return None
    return (
        left_side[
            target_match.start("target"):target_match.end("target")
        ],
        target_match.start("target"),
    )


def _publishing_lhs_path(
    publishing_assignment: SourceLineageStep,
) -> tuple[str, list[_SourceFieldComponent]] | None:
    assignments = _assignment_operator_matches(
        publishing_assignment.source_excerpt
    )
    if not assignments:
        return None
    matched_target = _assignment_lhs_target(
        publishing_assignment.source_excerpt[:assignments[0].start()]
    )
    if matched_target is None:
        return None
    target, _target_start = matched_target
    structural_target = re.sub(
        r"<[^<>()]+>",
        "",
        _mask_source_non_code(target),
    )
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    components: list[_SourceFieldComponent] = []
    for match in re.finditer(
        rf"(?P<name>{identifier})"
        r"(?P<accesses>(?:\s*(?:\[[^\[\]]+\]|\([^()]*\)))*)",
        structural_target,
    ):
        accesses = match.group("accesses")
        index_expressions = tuple(
            (bracket or call).strip()
            for bracket, call in re.findall(
                r"\[([^\[\]]+)\]|\(([^()]*)\)",
                accesses,
            )
        )
        components.append(
            _SourceFieldComponent(
                name=match.group("name"),
                index_expressions=index_expressions,
            )
        )
    if components and components[0].name == "this":
        components = components[1:]
    if len(components) < 2:
        return None
    return components[0].name, components[1:]


def _designated_initializer_lhs_path(
    publishing_assignment: SourceLineageStep,
) -> list[_SourceFieldComponent] | None:
    assignments = _assignment_operator_matches(
        publishing_assignment.source_excerpt
    )
    if len(assignments) != 1:
        return None
    structural_left_side = _mask_source_non_code(
        publishing_assignment.source_excerpt[
            :assignments[0].start()
        ]
    )
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    designator = (
        rf"\.\s*{identifier}"
        r"(?:\s*(?:\[[^\[\]]+\]|\([^()]*\)))*"
    )
    if re.fullmatch(
        rf"\s*(?:{designator})+\s*",
        structural_left_side,
    ) is None:
        return None
    components: list[_SourceFieldComponent] = []
    for match in re.finditer(
        rf"\.\s*(?P<name>{identifier})"
        r"(?P<accesses>(?:\s*(?:\[[^\[\]]+\]|\([^()]*\)))*)",
        structural_left_side,
    ):
        components.append(
            _SourceFieldComponent(
                name=match.group("name"),
                index_expressions=tuple(
                    (bracket or call).strip()
                    for bracket, call in re.findall(
                        r"\[([^\[\]]+)\]|\(([^()]*)\)",
                        match.group("accesses"),
                    )
                ),
            )
        )
    return components or None


def _publishing_lhs_binding(
    publishing_assignment: SourceLineageStep,
) -> tuple[str, list[str]] | None:
    binding = _publishing_lhs_path(publishing_assignment)
    if binding is None:
        return None
    return binding[0], [component.name for component in binding[1]]


def _publishing_lhs_message_identifier(
    publishing_assignment: SourceLineageStep,
    *,
    logged_field: str,
) -> str | None:
    binding = _publishing_lhs_binding(publishing_assignment)
    if binding is None or binding[1][-1] != logged_field:
        return None
    return binding[0]


def _cpp_literal_index(expression: str) -> int | None:
    normalized = expression.strip().replace("'", "")
    match = re.fullmatch(
        r"(?P<value>(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|[0-9]+))"
        r"[uUlL]*",
        normalized,
    )
    if match is None:
        return None
    value = match.group("value")
    base = (
        16
        if value.lower().startswith("0x")
        else 2
        if value.lower().startswith("0b")
        else 10
    )
    return int(value, base)


def _lineage_has_connected_data_flow(
    lineage: list[SourceLineageStep],
) -> bool:
    for upstream, downstream in zip(lineage, lineage[1:]):
        output_identifiers = _source_connection_tokens(
            _grounded_step_tokens(
                upstream,
                upstream.output_or_effect,
            )
        )
        input_identifiers = _source_connection_tokens(
            _grounded_step_tokens(
                downstream,
                downstream.input_or_state,
            )
        )
        if not output_identifiers or not input_identifiers:
            return False
        if not output_identifiers.intersection(input_identifiers):
            return False
    return True


def _source_connection_tokens(tokens: set[str]) -> set[str]:
    return {
        token
        for token in tokens
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token)
    }


def _runtime_requirement_id(
    candidate_id: str,
    condition: str,
) -> str:
    normalized = _normalize_evidence_text(condition)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{candidate_id}:runtime:{digest}"


def _candidate_runtime_conditions(
    candidate: CandidateMechanism,
) -> list[str]:
    conditions = [
        *candidate.runtime_conditions,
        *(
            condition
            for step in candidate.upstream_assignment_path
            for condition in step.runtime_conditions
        ),
    ]
    return list(
        dict.fromkeys(
            condition.strip()
            for condition in conditions
            if condition.strip()
        )
    )


def _candidate_runtime_requirement_ids(
    candidate: CandidateMechanism,
) -> list[str]:
    return [
        _runtime_requirement_id(candidate.candidate_id, condition)
        for condition in _candidate_runtime_conditions(candidate)
    ]


def _normalize_evidence_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _text_contains_focus(text: str, focus: str) -> bool:
    expected = str(focus or "").strip()
    if not expected:
        return False
    return re.search(
        rf"(?<![A-Za-z0-9_.]){re.escape(expected)}"
        r"(?![A-Za-z0-9_.])",
        str(text or ""),
    ) is not None


def _logged_value_path_components(
    logged_value: str,
) -> tuple[str, list[_LoggedFieldComponent]] | None:
    normalized = str(logged_value or "").strip()
    parsed = parse_signal_reference(normalized)
    if parsed is None:
        return None
    topic, _instance, field_path = parsed
    parts = field_path.split(".")
    identifier = r"[A-Za-z_][A-Za-z0-9_]*"
    fields: list[_LoggedFieldComponent] = []
    for part in parts:
        match = re.fullmatch(
            rf"(?P<field>{identifier})"
            r"(?P<indices>(?:\[[0-9]+\])*)",
            part,
        )
        if match is None:
            return None
        fields.append(
            _LoggedFieldComponent(
                name=match.group("field"),
                indices=tuple(
                    int(index)
                    for index in re.findall(
                        r"\[([0-9]+)\]",
                        match.group("indices"),
                    )
                ),
            )
        )
    return topic, fields


def _logged_value_path(
    logged_value: str,
) -> tuple[str, list[str]] | None:
    parsed = _logged_value_path_components(logged_value)
    if parsed is None:
        return None
    return parsed[0], [component.name for component in parsed[1]]


def _logged_value_field_identifier(logged_value: str) -> str | None:
    parsed = _logged_value_path(logged_value)
    return parsed[1][-1] if parsed is not None else None


def _logged_value_topic_identifier(logged_value: str) -> str | None:
    parsed = _logged_value_path(logged_value)
    return parsed[0] if parsed is not None else None


def _receipt_json_marker(name: str, value: Any) -> str:
    return (
        f"{name}_json="
        + json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    )


def _model_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    data = getattr(value, "__dict__", None)
    return data if isinstance(data, dict) else {}


def _extract_search_queries(values: list[Any]) -> list[str]:
    queries: set[str] = set()

    def walk(value: Any) -> None:
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"query", "search_query"} and isinstance(child, str):
                    queries.add(" ".join(child.split()))
                walk(child)
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                walk(child)
            return
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                walk(model_dump(mode="json"))
                return
            except Exception:
                pass
        data = getattr(value, "__dict__", None)
        if isinstance(data, dict):
            walk(data)

    walk(values)
    return sorted(queries)


def _extract_urls(raw_responses: list[Any]) -> list[str]:
    urls: set[str] = set()

    def walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value.startswith(("https://", "http://")):
                urls.add(value)
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if (
                    key == "url"
                    and isinstance(child, str)
                    and child.startswith(("https://", "http://"))
                ):
                    urls.add(child)
                walk(child)
            return
        if isinstance(value, (list, tuple, set)):
            for child in value:
                walk(child)
            return
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                walk(model_dump(mode="json"))
                return
            except Exception:
                pass
        data = getattr(value, "__dict__", None)
        if isinstance(data, dict):
            walk(data)

    walk(raw_responses)
    return sorted(urls)
