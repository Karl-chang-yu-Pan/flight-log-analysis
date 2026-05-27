from __future__ import annotations

"""

Architecture goal:
1. Prepass ULog to get PX4 version/git hash, vehicle_type, airframe/control-surface context,
   parameters, topic catalog, timeline, and mission summary.
2. Normalize the user question into source-search intent.
3. Retrieve reusable PX4 source-code mechanisms from the mechanism cache using only:
      px4_git_hash/version + vehicle_type + airframe/control-surface context + question intent.
   Validate cached mechanisms against the current PX4 source footprint. If no valid
   cache hit exists, run iterative source-mechanism discovery. The source resolver may
   use static parameter values and topic/field inventory to narrow source branches,
   but it must not use dynamic time-series samples or final log evidence.
4. Use parameters/timeline/mission/topic availability to eliminate impossible mechanisms.
5. Use deterministic log-signature evaluation to verify surviving mechanisms.
6. Write a report from verified mechanism results only.

This is intentionally a skeleton. The important part is the data flow and context boundaries.
"""

import asyncio
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents import Agent, Runner, RunContextWrapper, function_tool

from flight_log_agent.analysis.airframe_context import build_airframe_context
from flight_log_agent.analysis.applicability import evaluate_candidate_applicability
from flight_log_agent.px4.source import checkout_px4_source_revision
from flight_log_agent.mission.parser import parse_mission_file as parse_mission_file_impl
from flight_log_agent.analysis.report_postprocess import generate_report_plots as generate_report_plots_impl
from flight_log_agent.analysis.report_validation import enforce_validation_downgrades, validate_report

try:
    import pydantic as _pydantic

    _loaded_runner_models = sys.modules.get("flight_log_agent.models")
    _loaded_question_intent = getattr(_loaded_runner_models, "QuestionIntent", None)
    if (
        _loaded_runner_models is not None
        and hasattr(_pydantic, "__version__")
        and _loaded_question_intent is not None
        and not hasattr(_loaded_question_intent, "__pydantic_core_schema__")
    ):
        importlib.reload(_loaded_runner_models)
except Exception:
    pass

from flight_log_agent.models import (
    AirframeContext,
    ApplicabilityReport,
    ApplicabilityResult,
    CodeRef,
    ExpectedSignatureItem,
    FlightLogReport,
    HypothesisReportItem,
    MechanismCandidate,
    MechanismCandidateSet,
    ParameterValue,
    PlotOverlay,
    PlotRef,
    QuestionIntent,
    RelationshipCheckSpec,
    SignatureEvaluation,
    SourceEvidenceBundle,
    SourceSearchContext,
    ValidationResult,
    VerifiedMechanismResult,
    WindowSpec,
)
from flight_log_agent.analysis.signature_verification import derive_confidence
from flight_log_agent.analysis.signature_verification import evaluate_candidate_log_signature as evaluate_candidate_log_signature_impl
from flight_log_agent.ulog.control_surface import infer_control_surface as infer_control_surface_impl
from flight_log_agent.ulog.inventory import parse_ulog_inventory as parse_ulog_inventory_impl
from flight_log_agent.ulog.plots import generate_signal_plot as generate_signal_plot_impl
from flight_log_agent.ulog.timeline import build_basic_timeline as build_basic_timeline_impl
from flight_log_agent.px4.source_mechanism_models import (
    ParameterRequirement,
    SourceDiscoveryDecision,
    SourceDiscoveryIterationPacket,
    SourceDiscoveryLogContext,
    SourceFieldRef,
    SourceMechanismCandidate,
    SourceMechanismCandidateSet,
)
from flight_log_agent.px4.source_mechanism_resolver import (
    SourceMechanismResolver,
    build_source_discovery_log_context,
)

from flight_log_agent.audit import (
    AgentRunAuditHooks,
    DEFAULT_DEV_LOG_ROOT,
    DeveloperAuditLogger,
    log_run_items,
)

from flight_log_agent.px4.mechanism_cache import (
    MechanismCacheConfig,
    MechanismCacheWriter,
    MechanismRecord,
    MechanismRetriever,
    MechanismRetrievalResult,
    MechanismSourceValidation,
    MechanismSourceValidator,
)


DEFAULT_PX4_SOURCE_PATH = Path(__file__).resolve().parents[1] / "ref" / "PX4-Autopilot"


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
# Imported from flight_log_agent.models to keep runner_core.py focused on orchestration while
# preserving the public runner.<ModelName> API used by tests and callers.

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


def resolve_source_path(source_path: Optional[str | Path]) -> Optional[Path]:
    if source_path:
        return Path(source_path)
    if DEFAULT_PX4_SOURCE_PATH.exists():
        return DEFAULT_PX4_SOURCE_PATH
    return None


# ============================================================
# 4. Small agents
# ============================================================

question_intent_agent = Agent(
    name="Question Intent Normalizer",
    model="gpt-5.4-nano",
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


source_discovery_agent = Agent(
    name="PX4 Source Mechanism Discovery Decision",
    model="gpt-5.5",
    instructions="""
Guide iterative PX4 source-mechanism discovery from compact profiler output.

Input is a SourceDiscoveryIterationPacket containing:
- current user question and active expansion queries
- source files already visited and newly profiled
- deterministic source facts: related files, parameters, uORB topics, fields,
  function calls, branch conditions, and parameter predicates
- parameter feasibility results only for parameters discovered from source
- topic fields only for topics discovered from source

Rules:
- Do not perform final dynamic log verification.
- Do not claim a mechanism happened in the flight.
- Do not use timestamps, plots, signal comparisons, distance calculations, or
  final root-cause confidence.
- Decide which source files are mechanism-relevant, what expansion queries to
  run next, and whether the source chain is complete enough.
- Candidate drafts must be source-level mechanisms plus verification plans.
- Use required_log_evidence and expected_log_signature to request later
  MechanismVerifier checks; do not compute those checks yourself.
- If the current source evidence is insufficient, return expansion_queries and
  leave candidate_drafts empty.
""",
    tools=[],
    output_type=SourceDiscoveryDecision,
)


final_report_agent = Agent(
    name="Verified Mechanism Report Writer",
    model="gpt-5.4",
    instructions="""
Write the final report only from compact VerifiedMechanismResult objects.

Rules:
- The runner deterministically overwrites airframe_summary and
  question_intent_summary after this step; set them to concise placeholders.
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
    overlays: Optional[str] = None,
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
        overlays=_normalize_plot_overlays(overlays),
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
    source_path_obj = resolve_source_path(source_path)
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

        cached_candidates = [
            sanitize_mechanism_candidate_contract(candidate)
            for candidate in mechanism_records_to_candidates(cached_records, max_candidates)
        ]
        mechanism_cache_summary["cache_hit_candidate_names"] = [c.name for c in cached_candidates]

        if cached_candidates:
            source_evidence = empty_source_discovery_evidence(source_search_context)
            candidate_set = MechanismCandidateSet(
                candidates=cached_candidates,
                rejected_source_paths=[],
                unresolved_questions=[
                    f"Used {len(cached_candidates)} source-validated mechanism cache hit(s); source resolver agent was skipped."
                ],
            )
        else:
            # ------------------------------------------------------------
            # Stage 4: iterative source-mechanism discovery on cache miss
            # ------------------------------------------------------------
            source_discovery_log_context = build_source_discovery_log_context(
                inventory,
                airframe_context,
                mode_state_constraints=compact_timeline_constraints(timeline),
            )
            async def decide_source_discovery(packet: SourceDiscoveryIterationPacket) -> SourceDiscoveryDecision:
                return await _run_agent(
                    audit_logger,
                    "source_discovery_decision",
                    source_discovery_agent,
                    packet.model_dump(),
                    ctx,
                    max_turns=2,
                )

            source_candidate_set = await _audit_async_call(
                audit_logger,
                "source",
                "discover_source_mechanisms",
                discover_source_mechanisms,
                {
                    "source_path": str(source_path_obj) if source_path_obj else None,
                    "question_intent": question_intent.model_dump(),
                    "static_log_context": source_discovery_log_context.model_dump(),
                    "max_candidates": max_candidates,
                },
                source_path_obj,
                question_intent,
                source_discovery_log_context,
                max_candidates,
                decide_source_discovery,
            )
            candidate_set = source_mechanisms_to_candidates(source_candidate_set)
            source_evidence = empty_source_discovery_evidence(
                source_search_context,
                warnings=[
                    "Mechanisms were discovered by the iterative source resolver; "
                    "bounded one-shot source evidence is not used."
                ],
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

        candidate_set.candidates = [
            sanitize_mechanism_candidate_contract(candidate)
            for candidate in candidate_set.candidates
        ]
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
            build_final_report_input(verified_results),
            ctx,
            max_turns=4,
        )

        apply_deterministic_report_summaries(report, airframe_context, question_intent)
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

def retrieve_cached_mechanisms(
    cache_config: MechanismCacheConfig,
    source_search_context: SourceSearchContext,
    max_records: int,
) -> MechanismRetrievalResult:
    retriever = MechanismRetriever(cache_config)
    return retriever.retrieve(source_search_context, max_records=max_records)


async def discover_source_mechanisms(
    source_path: Optional[Path],
    question_intent: QuestionIntent,
    log_context: SourceDiscoveryLogContext,
    max_candidates: int,
    decide,
) -> SourceMechanismCandidateSet:
    if source_path is None:
        return SourceMechanismCandidateSet(
            candidates=[],
            expansion_queries=list(question_intent.source_queries),
            unresolved_questions=["PX4 source path is unavailable for source-mechanism discovery."],
        )

    resolver = SourceMechanismResolver(source_path)
    return await resolver.discover(
        question_intent.original_question,
        log_context,
        seed_queries=[
            question_intent.original_question,
            question_intent.problem_domain,
            question_intent.concise_intent,
            *question_intent.source_queries,
            *question_intent.likely_modules,
            *question_intent.likely_source_files,
        ],
        decide=decide,
        max_total_files=max(max_candidates * 8, 8),
    )


def source_mechanisms_to_candidates(
    source_candidate_set: SourceMechanismCandidateSet,
) -> MechanismCandidateSet:
    return MechanismCandidateSet(
        candidates=[
            source_mechanism_to_candidate(candidate)
            for candidate in source_candidate_set.candidates
        ],
        rejected_source_paths=[],
        unresolved_questions=list(source_candidate_set.unresolved_questions),
    )


def source_mechanism_to_candidate(source_candidate: SourceMechanismCandidate) -> MechanismCandidate:
    required_parameters = dedupe_keep_order([
        requirement.name
        for requirement in getattr(source_candidate, "controlling_parameters", []) or []
        if requirement.name and requirement.name != "unknown"
    ])
    required_signals = source_candidate_required_signals(source_candidate)
    parameter_checks = source_candidate_parameter_checks(source_candidate)
    return sanitize_mechanism_candidate_contract(
        MechanismCandidate(
            name=source_candidate.title,
            summary=source_candidate.source_mechanism,
            source_refs=list(source_candidate.source_chain),
            vehicle_type_gates=[],
            airframe_gates=[],
            mode_state_gates=list(source_candidate.branch_conditions),
            parameter_gates=[
                requirement.effect
                for requirement in getattr(source_candidate, "controlling_parameters", []) or []
            ],
            required_parameters=required_parameters,
            required_signals=required_signals,
            expected_logged_signature=[
                ExpectedSignatureItem(
                    name=f"source_signature_{index + 1}",
                    description=description,
                    signal=signal if signal in required_signals else None,
                    expected_behavior=description,
                )
                for index, (description, signal) in enumerate(
                    signature_descriptions_with_signals(source_candidate, required_signals)
                )
            ],
            exclusion_checks=[
                *parameter_checks,
                *[
                    RelationshipCheckSpec(type="custom", description=check)
                    for check in getattr(source_candidate, "contradiction_checks", []) or []
                ],
            ],
            numeric_checks=[
                *source_candidate_signal_presence_checks(source_candidate, required_signals),
            ],
            plot_requests=[],
        )
    )


def source_candidate_parameter_checks(source_candidate: SourceMechanismCandidate) -> list[RelationshipCheckSpec]:
    checks: list[RelationshipCheckSpec] = []
    for requirement in getattr(source_candidate, "controlling_parameters", []) or []:
        if not requirement.name or requirement.name == "unknown" or requirement.role != "branch_selector":
            continue
        parsed = parse_parameter_predicate(requirement.source_predicate or "", requirement.name)
        kwargs: dict[str, Any] = {
            "type": "branch_parameter_satisfied",
            "parameter": requirement.name,
            "source_predicate": requirement.source_predicate,
            "description": requirement.effect,
            "supports": requirement.effect,
            "contradicts": f"{requirement.name} does not satisfy source branch predicate.",
        }
        if parsed is not None:
            _, op, value = parsed
            kwargs["op"] = op
            kwargs["value"] = value
        checks.append(RelationshipCheckSpec(**kwargs))
    return checks


def source_candidate_signal_presence_checks(
    source_candidate: SourceMechanismCandidate,
    required_signals: list[str],
) -> list[RelationshipCheckSpec]:
    signals = list(required_signals)
    for evidence in getattr(source_candidate, "required_log_evidence", []) or []:
        signal = extract_signal_reference(evidence)
        if signal:
            signals.append(signal)
    return [
        RelationshipCheckSpec(
            type="topic_field_present",
            signal=signal,
            supports=f"Required source signal is logged: {signal}.",
            contradicts=f"Required source signal is not logged: {signal}.",
        )
        for signal in dedupe_keep_order(signals)
    ]


def source_candidate_required_signals(source_candidate: SourceMechanismCandidate) -> list[str]:
    signals = []
    for field in getattr(source_candidate, "relevant_fields", []) or []:
        if field.topic and field.field:
            signals.append(f"{field.topic}.{field.field}")
    for topic_ref in (
        list(getattr(source_candidate, "published_topics", []) or [])
        + list(getattr(source_candidate, "subscribed_topics", []) or [])
    ):
        if topic_ref.topic and topic_ref.field:
            signals.append(f"{topic_ref.topic}.{topic_ref.field}")
    return dedupe_keep_order(signals)


def signature_descriptions_with_signals(
    source_candidate: SourceMechanismCandidate,
    required_signals: list[str],
) -> list[tuple[str, Optional[str]]]:
    descriptions = list(getattr(source_candidate, "expected_log_signature", []) or [])
    if not descriptions:
        descriptions = list(getattr(source_candidate, "required_log_evidence", []) or [])

    pairs: list[tuple[str, Optional[str]]] = []
    for description in descriptions:
        signal = next(
            (candidate_signal for candidate_signal in required_signals if candidate_signal in description),
            None,
        )
        pairs.append((description, signal))
    return pairs


def parse_parameter_predicate(predicate: str, parameter: str) -> Optional[tuple[str, str, Any]]:
    if not predicate:
        return None
    pattern = re.compile(
        r"(?P<left>[A-Za-z_][A-Za-z0-9_.:]*\s*(?:\.\s*get\s*\(\s*\))?)\s*"
        r"(?P<op>>=|<=|==|!=|>|<)\s*"
        r"(?P<right>-?[A-Za-z_][A-Za-z0-9_:]*|-?\d+(?:\.\d+)?|true|false)"
    )
    for match in pattern.finditer(predicate):
        left = match.group("left").replace(" ", "")
        if parameter and parameter not in left and ".get()" not in left:
            continue
        return parameter, match.group("op"), parse_predicate_literal(match.group("right"))
    return None


def parse_predicate_literal(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer() and "." not in value:
        return int(number)
    return number


def extract_signal_reference(text: str) -> Optional[str]:
    match = re.search(r"\b([a-z][a-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b", text)
    return match.group(1) if match else None


def empty_source_discovery_evidence(
    source_search_context: SourceSearchContext,
    warnings: Optional[list[str]] = None,
) -> SourceEvidenceBundle:
    return SourceEvidenceBundle(
        search_context=source_search_context,
        hits=[],
        read_snippets=[],
        warnings=warnings or [],
    )


def compact_timeline_constraints(timeline: list[dict[str, Any]], max_events: int = 80) -> dict[str, Any]:
    events = []
    observed_values: dict[str, list[Any]] = {}
    for event in timeline:
        topic = event.get("topic")
        field = event.get("field")
        if not topic or not field:
            continue

        value = event.get("value")
        key = f"{topic}.{field}"
        values = observed_values.setdefault(key, [])
        if value not in values:
            values.append(value)

        if len(events) < max_events:
            events.append({
                "time_s": event.get("time_s"),
                "event": event.get("event"),
                "topic": topic,
                "field": field,
                "value": value,
            })

    return {
        "observed_values": observed_values,
        "events": events,
        "omitted_event_count": max(len([
            event for event in timeline
            if event.get("topic") and event.get("field")
        ]) - len(events), 0),
    }


def build_final_report_input(
    verified_results: list[VerifiedMechanismResult],
) -> dict[str, Any]:
    return {
        "verified_mechanism_results": [
            compact_verified_mechanism_result(result)
            for result in verified_results
        ],
    }


def compact_verified_mechanism_result(result: VerifiedMechanismResult) -> dict[str, Any]:
    candidate = result.candidate
    evaluation = result.evaluation
    return {
        "candidate": {
            "name": candidate.name,
            "summary": candidate.summary,
            "source_refs": _safe_model_dump(candidate.source_refs),
            "required_parameters": list(candidate.required_parameters),
            "required_signals": list(candidate.required_signals),
            "expected_logged_signature": _safe_model_dump(candidate.expected_logged_signature),
            "exclusion_checks": _safe_model_dump(candidate.exclusion_checks),
            "numeric_checks": _safe_model_dump(candidate.numeric_checks),
            "plot_requests": _safe_model_dump(candidate.plot_requests),
        },
        "applicability": _safe_model_dump(result.applicability),
        "evaluation": {
            "candidate_name": evaluation.candidate_name,
            "verdict": evaluation.verdict,
            "confidence_ceiling": evaluation.confidence_ceiling,
            "evidence": list(evaluation.evidence),
            "contradictions": list(evaluation.contradictions),
            "check_results": _safe_model_dump(evaluation.check_results),
            "warnings": list(evaluation.warnings),
        },
        "final_confidence": result.final_confidence,
    }


def apply_deterministic_report_summaries(
    report: FlightLogReport,
    airframe_context: AirframeContext,
    question_intent: QuestionIntent,
) -> FlightLogReport:
    report.airframe_summary = build_airframe_summary(airframe_context)
    report.question_intent_summary = build_question_intent_summary(question_intent)
    return report


def build_airframe_summary(airframe_context: AirframeContext) -> str:
    parts = []
    px4_version = getattr(airframe_context, "px4_version", None)
    px4_git_hash = getattr(airframe_context, "px4_git_hash", None)
    vehicle_type = getattr(airframe_context, "vehicle_type", None)
    sys_autostart = getattr(airframe_context, "sys_autostart", None)
    airframe_name = getattr(airframe_context, "airframe_name", None)

    if px4_version:
        parts.append(f"PX4 {px4_version}")
    if px4_git_hash:
        parts.append(f"git {str(px4_git_hash)[:12]}")
    if vehicle_type and vehicle_type != "unknown":
        parts.append(f"vehicle_type={vehicle_type}")
    if sys_autostart is not None:
        parts.append(f"SYS_AUTOSTART={sys_autostart}")
    if airframe_name:
        parts.append(f"airframe={airframe_name}")
    return "; ".join(parts) if parts else "Airframe context unavailable."


def build_question_intent_summary(question_intent: QuestionIntent) -> str:
    parts = []
    problem_domain = getattr(question_intent, "problem_domain", None)
    concise_intent = getattr(question_intent, "concise_intent", None)
    original_question = getattr(question_intent, "original_question", None)

    if problem_domain:
        parts.append(str(problem_domain))
    if concise_intent:
        parts.append(str(concise_intent))
    if not parts and original_question:
        parts.append(str(original_question))
    return ": ".join(parts) if parts else "Question intent unavailable."


def sanitize_mechanism_candidate_contract(candidate: MechanismCandidate) -> MechanismCandidate:
    """
    Enforce the source-candidate schema boundary before deterministic checks.

    The resolver sometimes treats "things needed for verification" as signals.
    Only ULog topic.field references belong in required_signals; bare PX4-style
    identifiers are parameters, and expressions are not log signals.
    """
    required_parameters = list(candidate.required_parameters or [])
    required_signals: list[str] = []

    for signal in candidate.required_signals or []:
        signal_name = str(signal).strip()
        if is_logged_signal_reference(signal_name):
            required_signals.append(signal_name)
        elif is_px4_parameter_name(signal_name):
            required_parameters.append(signal_name)

    candidate.required_parameters = dedupe_keep_order(required_parameters)
    candidate.required_signals = dedupe_keep_order(required_signals)

    for item in candidate.expected_logged_signature or []:
        signal = str(getattr(item, "signal", "") or "").strip()
        if signal and not is_logged_signal_reference(signal):
            item.signal = None

    return candidate


def is_logged_signal_reference(value: str) -> bool:
    signal_part = r"[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?"
    return bool(re.fullmatch(rf"[a-z][a-z0-9_]*\.{signal_part}(?:\.{signal_part})*", value))


def is_px4_parameter_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", value)) and "_" in value


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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


def evaluate_candidate_log_signature(
    ctx: FlightLogContext,
    candidate: MechanismCandidate,
    applicability: ApplicabilityResult,
) -> SignatureEvaluation:
    return evaluate_candidate_log_signature_impl(ctx.log_path, candidate, applicability)


def generate_report_plots(
    report: FlightLogReport,
    ctx: FlightLogContext,
    audit_logger: Optional[DeveloperAuditLogger],
) -> FlightLogReport:
    return generate_report_plots_impl(
        report,
        ctx.log_path,
        ctx.output_dir,
        audit_logger,
        audit_call=_audit_sync_call,
        plot_generator=generate_signal_plot_impl,
    )


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


async def _audit_async_call(
    audit_logger: Optional[DeveloperAuditLogger],
    event_prefix: str,
    name: str,
    func: Any,
    input_payload: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    if audit_logger is None:
        return await func(*args, **kwargs)

    started_at = time.perf_counter()
    audit_logger.log_event(f"{event_prefix}.started", name=name, input=input_payload)
    try:
        result = await func(*args, **kwargs)
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


def _normalize_plot_overlays(overlays: Any) -> Optional[list[dict[str, Any]]]:
    if not overlays:
        return None
    if isinstance(overlays, str):
        try:
            overlays = json.loads(overlays)
        except json.JSONDecodeError:
            return None
    if not isinstance(overlays, list):
        return None

    normalized: list[dict[str, Any]] = []
    for overlay in overlays:
        if hasattr(overlay, "model_dump"):
            normalized.append(overlay.model_dump(exclude_none=True))
        elif isinstance(overlay, dict):
            normalized.append({key: value for key, value in overlay.items() if value is not None})
    return normalized or None


def _safe_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_safe_model_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_model_dump(v) for k, v in value.items()}
    return value


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
