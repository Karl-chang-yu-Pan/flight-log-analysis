from __future__ import annotations

"""

Architecture goal:
1. Prepass ULog to get PX4 version/git hash, vehicle_type, airframe/control-surface context,
   parameters, topic catalog, timeline, and mission summary.
2. Normalize the user question into source-search intent.
3. Retrieve reusable PX4 source-code mechanisms from the mechanism cache using only:
      px4_git_hash/version + vehicle_type + airframe/control-surface context + question intent.
   Validate cached mechanisms against the current PX4 source footprint. If no valid
   cache hit exists, run bounded source search and the mechanism resolver agent.
   Do NOT use parameter values, full topic catalog, or detailed log samples here.
4. Use parameters/timeline/mission/topic availability to eliminate impossible mechanisms.
5. Use deterministic log-signature evaluation to verify surviving mechanisms.
6. Write a report from verified mechanism results only.

This is intentionally a skeleton. The important part is the data flow and context boundaries.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field
from agents import Agent, Runner, RunContextWrapper, function_tool

from px4_source import checkout_px4_source_revision, read_source_file, search_source
from mission_parser import parse_mission_file as parse_mission_file_impl
from ulog_control_surface import infer_control_surface as infer_control_surface_impl
from ulog_inventory import parse_ulog_inventory as parse_ulog_inventory_impl
from ulog_plots import generate_signal_plot as generate_signal_plot_impl
from ulog_timeline import build_basic_timeline as build_basic_timeline_impl
from ulog_signature_evaluator import evaluate_log_signature as evaluate_log_signature_impl

from run_audit_log import (
    AgentRunAuditHooks,
    DEFAULT_DEV_LOG_ROOT,
    DeveloperAuditLogger,
    log_run_items,
)

from mechanism_cache import (
    MechanismCacheConfig,
    MechanismCacheWriter,
    MechanismRecord,
    MechanismRetriever,
    MechanismRetrievalResult,
    MechanismSourceValidation,
    MechanismSourceValidator,
)


# ============================================================
# 1. Runtime context
# ============================================================

@dataclass
class FlightLogContext:
    log_path: Path
    mission_path: Optional[Path]
    source_path: Optional[Path]
    output_dir: Path


# ============================================================
# 2. Core data models
# ============================================================

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
        "custom",
    ]
    window: Optional[str] = None
    signal: Optional[str] = None
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


# ============================================================
# 3. Prepass wrappers
# ============================================================

def parse_ulog_inventory(log_path: Path, source_path: Optional[Path] = None) -> dict:
    return parse_ulog_inventory_impl(log_path, source_path)


def build_basic_timeline(log_path: Path) -> list[dict]:
    return build_basic_timeline_impl(log_path)


def infer_control_surface(log_path: Path, source_path: Optional[Path]) -> dict:
    return infer_control_surface_impl(log_path, source_path)


def parse_mission_file(mission_path: Optional[Path]) -> Optional[dict]:
    return parse_mission_file_impl(mission_path)


# ============================================================
# 4. Small agents
# ============================================================

question_intent_agent = Agent(
    name="Question Intent Normalizer",
    instructions="""
Convert the user's natural-language PX4 flight-log question into a source-search intent.

Input is only:
- user question
- minimal airframe context

Output must include concise intent and source search queries.
Do not use parameter values, topic lists, or log evidence to verify anything.
Do not draft hypotheses. Do not assign confidence.
""",
    tools=[],
    output_type=QuestionIntent,
)


mechanism_resolver_agent = Agent(
    name="PX4 Mechanism Resolver",
    instructions="""
Resolve reusable PX4 source-code mechanisms from bounded source-search evidence.

Important separation:
- You are only extracting source-level PX4 behavior that could explain the
  normalized question intent for this PX4 source version and vehicle/control domain.
- Do not use parameter values or detailed log evidence to decide whether the
  mechanism happened in the flight.
- Emit cacheable mechanism candidates. For every candidate, include source_refs,
  mechanism summary, source-level gates, required parameters/signals, expected
  logged signature, exclusion checks, numeric checks, and useful plot requests.
- Do not assign confidence that the mechanism happened in the flight.
- The output may be written to the mechanism cache; avoid flight-specific language
  such as "this log shows" or "this aircraft did".
""",
    tools=[],
    output_type=MechanismCandidateSet,
)

# Backward-compatible name for older code/tests.
mechanism_candidate_agent = mechanism_resolver_agent


final_report_agent = Agent(
    name="Verified Mechanism Report Writer",
    instructions="""
Write the final report only from VerifiedMechanismResult objects.

Rules:
- Do not introduce new mechanisms.
- Do not claim a mechanism happened unless applicability and log evaluation support it.
- Separate source mechanism, applicability filtering, numeric log verification,
  contradictions, and confidence.
- Confidence cannot exceed evaluation.confidence_ceiling.
- If applicability is false, confidence must be low or unresolved.
- If evaluation is unresolved or contradicted, confidence must be low or unresolved.
- In report applicability, relevant_parameters must be a list of {name, value}
  objects using string values, not a JSON object/map.
""",
    tools=[],
    output_type=FlightLogReport,
)


# ============================================================
# 5. Optional low-level tools, not exposed to broad agents
# ============================================================

@function_tool
def generate_signal_plot(
    ctx: RunContextWrapper[FlightLogContext],
    title: str,
    start_s: float,
    end_s: float,
    signals: list[str],
    purpose: str,
    plot_type: str = "timeseries",
    bins: int = 50,
    overlays: Optional[list[PlotOverlay]] = None,
) -> dict:
    return generate_signal_plot_impl(
        ctx.context.log_path,
        ctx.context.output_dir,
        title,
        start_s,
        end_s,
        signals,
        purpose,
        plot_type=plot_type,
        bins=bins,
        overlays=[o.model_dump(exclude_none=True) for o in overlays] if overlays else None,
    )


# ============================================================
# 6. Main runner
# ============================================================

async def analyze_flight_log(
    log_path: str,
    user_question: str,
    mission_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_dir: str = "outputs/run_001",
    dev_log_root: str = str(DEFAULT_DEV_LOG_ROOT),
    dev_run_id: Optional[str] = None,
    max_candidates: int = 5,
    mechanism_cache_dir: str = ".flightlog_cache/mechanisms",
    force_mechanism_refresh: bool = False,
) -> FlightLogReport:
    log_path_obj = Path(log_path)
    mission_path_obj = Path(mission_path) if mission_path else None
    source_path_obj = Path(source_path) if source_path else None
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    audit_logger = DeveloperAuditLogger(Path(dev_log_root), run_id=dev_run_id)
    report_path = output_dir_obj / "report.json"

    audit_logger.save_metadata(
        {
            "runner_version": "v3_mechanism_first",
            "log_path": str(log_path_obj),
            "mission_path": str(mission_path_obj) if mission_path_obj else None,
            "source_path": str(source_path_obj) if source_path_obj else None,
            "output_dir": str(output_dir_obj),
            "report_path": str(report_path),
            "mechanism_cache_dir": mechanism_cache_dir,
            "force_mechanism_refresh": force_mechanism_refresh,
        }
    )

    ctx = FlightLogContext(
        log_path=log_path_obj,
        mission_path=mission_path_obj,
        source_path=source_path_obj,
        output_dir=output_dir_obj,
    )

    try:
        # ------------------------------------------------------------
        # Stage 1: deterministic prepass
        # ------------------------------------------------------------
        inventory = _audit_sync_call(
            audit_logger,
            "prepass",
            "parse_ulog_inventory",
            parse_ulog_inventory,
            {"log_path": str(log_path_obj)},
            log_path_obj,
        )
        timeline = _audit_sync_call(
            audit_logger,
            "prepass",
            "build_basic_timeline",
            build_basic_timeline,
            {"log_path": str(log_path_obj)},
            log_path_obj,
        )
        control_surface = _audit_sync_call(
            audit_logger,
            "prepass",
            "infer_control_surface",
            infer_control_surface,
            {"log_path": str(log_path_obj), "source_path": str(source_path_obj) if source_path_obj else None},
            log_path_obj,
            source_path_obj,
        )
        mission = _audit_sync_call(
            audit_logger,
            "prepass",
            "parse_mission_file",
            parse_mission_file,
            {"mission_path": str(mission_path_obj) if mission_path_obj else None},
            mission_path_obj,
        )

        # Only this compact object may enter mechanism discovery.
        airframe_context = build_airframe_context(inventory, control_surface)

        # Optional: checkout exact PX4 revision before source search.
        if source_path_obj is not None and airframe_context.px4_git_hash:
            _audit_sync_call(
                audit_logger,
                "source",
                "checkout_px4_source_revision",
                checkout_px4_source_revision,
                {"revision": airframe_context.px4_git_hash},
                source_path_obj,
                airframe_context.px4_git_hash,
            )

        # ------------------------------------------------------------
        # Stage 2: normalize question into source-search intent
        # ------------------------------------------------------------
        question_intent = await _run_agent(
            audit_logger,
            "question_intent",
            question_intent_agent,
            {
                "user_question": user_question,
                "airframe_context": airframe_context.model_dump(),
            },
            ctx,
            max_turns=2,
        )

        source_search_context = SourceSearchContext(
            airframe=airframe_context,
            question_intent=question_intent,
        )

        # ------------------------------------------------------------
        # Stage 3: retrieve and validate cached source mechanisms
        # ------------------------------------------------------------
        mechanism_cache_config = MechanismCacheConfig(cache_root=Path(mechanism_cache_dir))
        mechanism_cache_summary: dict[str, Any] = {
            "cache_dir": str(mechanism_cache_config.cache_root),
            "force_mechanism_refresh": force_mechanism_refresh,
            "retrieval": None,
            "source_validations": [],
            "cache_hit_candidate_names": [],
            "written_records": [],
        }

        cached_records: list[MechanismRecord] = []
        cache_retrieval = MechanismRetrievalResult()
        if not force_mechanism_refresh:
            cache_retrieval = _audit_sync_call(
                audit_logger,
                "mechanism_cache",
                "retrieve_mechanisms",
                retrieve_cached_mechanisms,
                {
                    "mechanism_cache_dir": mechanism_cache_dir,
                    "source_search_context": source_search_context.model_dump(),
                    "max_records": max_candidates,
                },
                mechanism_cache_config,
                source_search_context,
                max_candidates,
            )
            mechanism_cache_summary["retrieval"] = cache_retrieval.model_dump()

            for record in cache_retrieval.records:
                validation = _audit_sync_call(
                    audit_logger,
                    "mechanism_cache",
                    f"validate_mechanism_source:{record.mechanism_id}",
                    validate_cached_mechanism_source,
                    {
                        "mechanism_id": record.mechanism_id,
                        "source_path": str(source_path_obj) if source_path_obj else None,
                        "current_git_hash": airframe_context.px4_git_hash,
                    },
                    source_path_obj,
                    airframe_context.px4_git_hash,
                    record,
                )
                mechanism_cache_summary["source_validations"].append(validation.model_dump())
                if validation.usable:
                    cached_records.append(record)

        cached_candidates = mechanism_records_to_candidates(cached_records, max_candidates)
        mechanism_cache_summary["cache_hit_candidate_names"] = [c.name for c in cached_candidates]

        if cached_candidates:
            source_evidence: Optional[SourceEvidenceBundle] = None
            candidate_set = MechanismCandidateSet(
                candidates=cached_candidates,
                rejected_source_paths=[],
                unresolved_questions=[
                    f"Used {len(cached_candidates)} source-validated mechanism cache hit(s); source resolver agent was skipped."
                ],
            )
        else:
            # ------------------------------------------------------------
            # Stage 4A: deterministic bounded source search on cache miss
            # ------------------------------------------------------------
            source_evidence = _audit_sync_call(
                audit_logger,
                "source",
                "bounded_source_search",
                bounded_source_search,
                {"source_search_context": source_search_context.model_dump()},
                source_path_obj,
                source_search_context,
            )

            # ------------------------------------------------------------
            # Stage 4B: resolve source mechanisms, then write cache records
            # ------------------------------------------------------------
            candidate_set = await _run_agent(
                audit_logger,
                "resolve_mechanisms",
                mechanism_resolver_agent,
                {
                    "source_search_context": source_search_context.model_dump(),
                    "source_evidence": source_evidence.model_dump(),
                },
                ctx,
                max_turns=3,
            )

            written_records = _audit_sync_call(
                audit_logger,
                "mechanism_cache",
                "write_resolved_mechanisms",
                write_resolved_mechanisms_to_cache,
                {
                    "mechanism_cache_dir": mechanism_cache_dir,
                    "candidate_count": len(candidate_set.candidates),
                },
                mechanism_cache_config,
                candidate_set,
                airframe_context,
                question_intent,
                source_path_obj,
                source_evidence,
            )
            mechanism_cache_summary["written_records"] = written_records

        candidates = candidate_set.candidates[:max_candidates]

        # ------------------------------------------------------------
        # Stage 5: deterministic applicability + log verification
        # ------------------------------------------------------------
        verified_results: list[VerifiedMechanismResult] = []
        for candidate in candidates:
            applicability = _audit_sync_call(
                audit_logger,
                "applicability",
                f"evaluate_applicability:{candidate.name}",
                evaluate_candidate_applicability,
                {"candidate": candidate.model_dump()},
                candidate,
                inventory,
                timeline,
                mission,
            )

            if not applicability.applicable:
                evaluation = SignatureEvaluation(
                    candidate_name=candidate.name,
                    verdict="unresolved",
                    confidence_ceiling="low",
                    evidence=[],
                    contradictions=[f"Mechanism excluded before log verification: {applicability.excluded_by}"],
                    warnings=["Log verification skipped because applicability failed."],
                )
            else:
                evaluation = _audit_sync_call(
                    audit_logger,
                    "verification",
                    f"evaluate_signature:{candidate.name}",
                    evaluate_candidate_log_signature,
                    {
                        "candidate": candidate.model_dump(),
                        "applicability": applicability.model_dump(),
                    },
                    ctx,
                    candidate,
                    applicability,
                )

            verified_results.append(
                VerifiedMechanismResult(
                    candidate=candidate,
                    applicability=applicability,
                    evaluation=evaluation,
                    final_confidence=derive_confidence(applicability, evaluation),
                )
            )

        # ------------------------------------------------------------
        # Stage 6: final report, then deterministic validation/plots
        # ------------------------------------------------------------
        report = await _run_agent(
            audit_logger,
            "final_report",
            final_report_agent,
            {
                "airframe_context": airframe_context.model_dump(),
                "question_intent": question_intent.model_dump(),
                "verified_mechanism_results": [r.model_dump() for r in verified_results],
                "excluded_source_paths": candidate_set.rejected_source_paths,
                "unresolved_source_questions": candidate_set.unresolved_questions,
                "mechanism_cache": mechanism_cache_summary,
            },
            ctx,
            max_turns=4,
        )

        report = generate_report_plots(report, ctx, audit_logger)
        validation = validate_report(report)
        audit_logger.log_event("validation.finished", output=validation.model_dump())

        if not validation.passed:
            # Prefer deterministic downgrade instead of another LLM repair loop.
            report = enforce_validation_downgrades(report, validation)
            validation = validate_report(report)
            audit_logger.log_event("validation_after_downgrade.finished", output=validation.model_dump())

        save_report(report, report_path)
        audit_logger.log_event(
            "run.finished",
            output={
                "report_path": str(report_path),
                "validation_passed": validation.passed,
                "dev_log_dir": str(audit_logger.run_dir),
            },
        )
        return report

    except Exception as exc:
        audit_logger.log_event("run.failed", error=repr(exc))
        raise




# ============================================================
# 7. Deterministic stage implementations
# ============================================================

def build_airframe_context(inventory: dict, control_surface: dict) -> AirframeContext:
    params = inventory.get("parameters") or inventory.get("important_parameters") or {}

    px4_git_hash = (
        inventory.get("git_hash")
        or inventory.get("px4_git_hash")
        or inventory.get("firmware_git_hash")
    )
    px4_version = inventory.get("firmware_version") or inventory.get("px4_version")
    px4_tag = inventory.get("px4_tag") or inventory.get("git_tag")

    sys_autostart = _maybe_int(params.get("SYS_AUTOSTART"))
    vehicle_type = infer_vehicle_type_string(inventory, params, control_surface)

    return AirframeContext(
        px4_git_hash=px4_git_hash,
        px4_version=px4_version,
        px4_tag=px4_tag,
        vehicle_type=vehicle_type,
        sys_autostart=sys_autostart,
        airframe_name=str(control_surface.get("airframe") or control_surface.get("airframe_name") or ""),
        control_surface_summary=summarize_control_surface(control_surface),
    )


def infer_vehicle_type_string(inventory: dict, params: dict, control_surface: dict) -> str:
    # Replace this with your actual inventory/parameter conventions.
    vehicle_type = str(control_surface.get("vehicle_type") or inventory.get("vehicle_type") or "unknown").lower()
    vt_type = params.get("VT_TYPE")

    if "vtol" in vehicle_type or vt_type is not None:
        vt_type_i = _maybe_int(vt_type)
        if vt_type_i == 1:
            return "vtol_tailsitter"
        if vt_type_i == 2:
            return "vtol_standard"
        if vt_type_i == 3:
            return "vtol_tiltrotor"
        return "vtol_unknown_subtype"

    if "fixed" in vehicle_type or "fw" in vehicle_type:
        return "fixed_wing"
    if "multi" in vehicle_type or "mc" in vehicle_type:
        return "multicopter"
    return vehicle_type or "unknown"


def summarize_control_surface(control_surface: dict) -> str:
    # Keep this compact. No full parameter dump.
    if not control_surface:
        return "unknown"
    keys = [
        "vehicle_type",
        "assumed_actuator_mapping",
        "control_surfaces",
        "confidence",
        "warning",
    ]
    compact = {k: control_surface.get(k) for k in keys if k in control_surface}
    return json.dumps(compact, separators=(",", ":"), default=str)[:2000]


def retrieve_cached_mechanisms(
    cache_config: MechanismCacheConfig,
    source_search_context: SourceSearchContext,
    max_records: int,
) -> MechanismRetrievalResult:
    retriever = MechanismRetriever(cache_config)
    return retriever.retrieve(source_search_context, max_records=max_records)


def validate_cached_mechanism_source(
    source_path: Optional[Path],
    current_git_hash: Optional[str],
    record: MechanismRecord,
) -> MechanismSourceValidation:
    validator = MechanismSourceValidator(source_path, current_git_hash=current_git_hash)
    return validator.validate_record(record)


def mechanism_records_to_candidates(
    records: list[MechanismRecord],
    max_candidates: int,
) -> list[MechanismCandidate]:
    candidates: list[MechanismCandidate] = []
    for record in records[:max_candidates]:
        try:
            candidate = MechanismCandidate.model_validate(record.candidate_payload)
        except Exception:
            continue
        candidates.append(candidate)
    return candidates


def write_resolved_mechanisms_to_cache(
    cache_config: MechanismCacheConfig,
    candidate_set: MechanismCandidateSet,
    airframe_context: AirframeContext,
    question_intent: QuestionIntent,
    source_path: Optional[Path],
    source_evidence: SourceEvidenceBundle,
) -> list[dict[str, Any]]:
    writer = MechanismCacheWriter(cache_config)
    written: list[dict[str, Any]] = []
    for candidate in candidate_set.candidates:
        record = writer.write_candidate(
            candidate_payload=candidate.model_dump(),
            airframe_context=airframe_context.model_dump(),
            question_intent=question_intent.model_dump(),
            source_path=source_path,
            source_evidence=source_evidence.model_dump(),
        )
        written.append({
            "mechanism_id": record.mechanism_id,
            "name": record.name,
            "vehicle_control_domain": record.vehicle_control_domain,
            "px4_git_hash": record.source_identity.px4_git_hash,
            "source_ref_count": len(record.source_refs),
        })
    return written


def bounded_source_search(
    source_path: Optional[Path],
    search_context: SourceSearchContext,
    max_hits_total: int = 40,
    max_hits_per_query: int = 8,
    max_snippet_chars: int = 1600,
) -> SourceEvidenceBundle:
    if source_path is None:
        return SourceEvidenceBundle(
            search_context=search_context,
            hits=[],
            warnings=["No PX4 source path provided."],
        )

    hits: list[SourceHit] = []
    warnings: list[str] = []

    queries = dedupe_keep_order(search_context.question_intent.source_queries)
    for query in queries:
        if len(hits) >= max_hits_total:
            break

        raw_results = search_source(source_path, query, max_results=max_hits_per_query)
        for item in _flatten_source_results(query, raw_results):
            if len(hits) >= max_hits_total:
                break
            item.snippet = item.snippet[:max_snippet_chars]
            hits.append(item)

    # Optional: read only bounded snippets from likely files. Never read whole files.
    read_snippets: list[CodeRef] = []
    for file in search_context.question_intent.likely_source_files[:6]:
        try:
            result = read_source_file(source_path, file, 1, 220)
            read_snippets.append(
                CodeRef(
                    file=file,
                    start_line=1,
                    end_line=220,
                    snippet=json.dumps(result, default=str)[:max_snippet_chars],
                    explanation="Bounded top-of-file/context read from likely source file.",
                )
            )
        except Exception as exc:
            warnings.append(f"Failed to read likely source file {file}: {exc!r}")

    return SourceEvidenceBundle(
        search_context=search_context,
        hits=hits,
        read_snippets=read_snippets,
        warnings=warnings,
    )


def evaluate_candidate_applicability(
    candidate: MechanismCandidate,
    inventory: dict,
    timeline: list[dict],
    mission: Optional[dict],
) -> ApplicabilityResult:
    """
    Use actual parameters/timeline/mission/topic availability to eliminate mechanisms.
    This is where parameters and topics enter the workflow.
    """
    params = inventory.get("parameters") or inventory.get("important_parameters") or {}
    topic_fields = inventory.get("topic_fields") or {}
    available_topics = set(inventory.get("available_topics") or topic_fields.keys())

    relevant_parameters = {
        name: params.get(name)
        for name in candidate.required_parameters
        if name in params
    }

    supported: list[str] = []
    excluded: list[str] = []
    unresolved: list[str] = []

    for param_name in candidate.required_parameters:
        if param_name in params:
            supported.append(f"Parameter present: {param_name}={params.get(param_name)}")
        else:
            unresolved.append(f"Required/candidate parameter not found in log: {param_name}")

    available_required_signals, missing_required_signals = check_required_signals(
        candidate.required_signals,
        available_topics,
        topic_fields,
    )

    if missing_required_signals:
        unresolved.append(f"Missing required signals: {missing_required_signals}")

    # TODO: implement candidate.mode_state_gates against timeline.
    candidate_windows = derive_candidate_windows(candidate, timeline, mission)
    if not candidate_windows:
        unresolved.append("No candidate verification window could be derived from timeline/mission.")

    # A candidate is applicable unless a hard exclusion is found.
    # Missing data makes it unresolved, not necessarily excluded.
    applicable = len(excluded) == 0

    return ApplicabilityResult(
        candidate_name=candidate.name,
        applicable=applicable,
        supported_conditions=supported,
        excluded_by=excluded,
        unresolved_conditions=unresolved,
        relevant_parameters=relevant_parameters,
        candidate_windows=candidate_windows,
        available_required_signals=available_required_signals,
        missing_required_signals=missing_required_signals,
    )


def evaluate_candidate_log_signature(
    ctx: FlightLogContext,
    candidate: MechanismCandidate,
    applicability: ApplicabilityResult,
) -> SignatureEvaluation:
    raw = evaluate_log_signature_impl(
        ctx.log_path,
        candidate.name,
        [_model_to_dict(x) for x in candidate.expected_logged_signature],
        [_model_to_dict(x) for x in applicability.candidate_windows],
        candidate.required_signals,
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


def derive_candidate_windows(
    candidate: MechanismCandidate,
    timeline: list[dict],
    mission: Optional[dict],
) -> list[WindowSpec]:
    # Skeleton: replace with real timeline interpretation.
    # For now, use candidate plot windows if provided.
    windows: list[WindowSpec] = []
    for plot in candidate.plot_requests:
        if plot.start_s is not None and plot.end_s is not None:
            windows.append(
                WindowSpec(
                    name=plot.title.lower().replace(" ", "_"),
                    start_s=float(plot.start_s),
                    end_s=float(plot.end_s),
                    reason=plot.purpose,
                )
            )
    return windows


def check_required_signals(
    required_signals: list[str],
    available_topics: set[str],
    topic_fields: dict[str, Any],
) -> tuple[list[str], list[str]]:
    available: list[str] = []
    missing: list[str] = []

    for signal in required_signals:
        topic = signal.split(".", 1)[0]
        if topic in available_topics:
            available.append(signal)
        else:
            missing.append(signal)

    return available, missing


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
    # Conservative fallback: if validation reports structural evidence errors,
    # downgrade affected hypotheses instead of launching another repair agent.
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


def generate_report_plots(
    report: FlightLogReport,
    ctx: FlightLogContext,
    audit_logger: Optional[DeveloperAuditLogger],
) -> FlightLogReport:
    for hyp in report.ranked_hypotheses:
        for plot in hyp.plots:
            if plot.start_s is None or plot.end_s is None or not plot.signals:
                continue
            result = _audit_sync_call(
                audit_logger,
                "postprocess_plot",
                "generate_signal_plot",
                generate_signal_plot_impl,
                {
                    "title": plot.title,
                    "start_s": plot.start_s,
                    "end_s": plot.end_s,
                    "signals": plot.signals,
                },
                ctx.log_path,
                ctx.output_dir,
                plot.title,
                float(plot.start_s),
                float(plot.end_s),
                plot.signals,
                plot.purpose,
                plot_type=plot.plot_type,
                bins=plot.bins,
                overlays=[o.model_dump(exclude_none=True) for o in plot.overlays],
            )
            _apply_plot_result(plot, result)
    return report


# ============================================================
# 8. Utility helpers
# ============================================================

async def _run_agent(
    audit_logger: Optional[DeveloperAuditLogger],
    stage_name: str,
    agent: Agent,
    payload: dict[str, Any],
    ctx: FlightLogContext,
    max_turns: int,
) -> Any:
    started_at = time.perf_counter()
    if audit_logger is not None:
        audit_logger.log_event(f"agent.{stage_name}.started", input=payload)

    result = await Runner.run(
        agent,
        input=json.dumps(payload, separators=(",", ":"), default=str),
        context=ctx,
        max_turns=max_turns,
        hooks=AgentRunAuditHooks(audit_logger) if audit_logger is not None else None,
    )

    if audit_logger is not None:
        log_run_items(audit_logger, getattr(result, "new_items", []) or [])
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        audit_logger.log_event(
            f"agent.{stage_name}.finished",
            output=_safe_model_dump(result.final_output),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            usage=usage,
        )

    return result.final_output


def _audit_sync_call(
    audit_logger: Optional[DeveloperAuditLogger],
    event_prefix: str,
    name: str,
    func: Any,
    input_payload: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if audit_logger is None:
        return func(*args, **kwargs)

    started_at = time.perf_counter()
    audit_logger.log_event(f"{event_prefix}.started", name=name, input=input_payload)
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        audit_logger.log_event(
            f"{event_prefix}.failed",
            name=name,
            error=repr(exc),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
        raise

    audit_logger.log_event(
        f"{event_prefix}.finished",
        name=name,
        output=_safe_model_dump(result),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
    )
    return result


def save_report(report: FlightLogReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _flatten_source_results(query: str, raw_results: Any) -> list[SourceHit]:
    """
    Adapter around your existing px4_source.search_source return format.
    Fill this in once you standardize search_source output.
    """
    hits: list[SourceHit] = []

    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                file = str(item.get("file") or item.get("path") or "unknown")
                line = _maybe_int(item.get("line") or item.get("line_number"))
                snippet = str(item.get("snippet") or item.get("text") or item.get("match") or item)
            else:
                file = "unknown"
                line = None
                snippet = str(item)
            hits.append(SourceHit(query=query, file=file, line=line, snippet=snippet))
    else:
        hits.append(SourceHit(query=query, file="unknown", line=None, snippet=str(raw_results)))

    return hits


def _model_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if v is not None}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if v is not None}
    return dict(value)


def _safe_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_safe_model_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_model_dump(v) for k, v in value.items()}
    return value


def _apply_plot_result(plot: PlotRef, result: dict[str, Any]) -> None:
    for key in ("title", "path", "purpose", "signals", "plot_type", "overlays", "missing_signals", "warnings"):
        if key in result:
            setattr(plot, key, result[key])


def _extract_list(raw: dict[str, Any], keys: list[str]) -> list[Any]:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            return [value]
    return []


def _maybe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


# ============================================================
# 9. Example
# ============================================================

if __name__ == "__main__":
    report = asyncio.run(
        analyze_flight_log(
            log_path="logs/test.ulg",
            mission_path="missions/test.plan",
            source_path="PX4-Autopilot",
            output_dir="outputs/test_case_001",
            user_question="Why did RTL climb higher than expected? Do not provide suggestions.",
        )
    )
    print(report.model_dump_json(indent=2))
