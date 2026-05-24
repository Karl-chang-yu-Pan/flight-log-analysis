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
from flight_log_agent.analysis.source_evidence import bounded_source_search
from flight_log_agent.ulog.control_surface import infer_control_surface as infer_control_surface_impl
from flight_log_agent.ulog.inventory import parse_ulog_inventory as parse_ulog_inventory_impl
from flight_log_agent.ulog.plots import generate_signal_plot as generate_signal_plot_impl
from flight_log_agent.ulog.timeline import build_basic_timeline as build_basic_timeline_impl

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


mechanism_resolver_agent = Agent(
    name="PX4 Mechanism Resolver",
    model="gpt-5.5",
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
- Put PX4 parameters only in required_parameters. Put only logged ULog signals
  in required_signals, using exact topic.field names. Do not put parameter names,
  mission-file values, expressions, or source-code variables in required_signals.
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
                build_mechanism_resolver_input(
                    source_search_context,
                    source_evidence,
                    inventory,
                    timeline,
                    mission,
                ),
                ctx,
                max_turns=3,
            )
            candidate_set.candidates = [
                sanitize_mechanism_candidate_contract(candidate)
                for candidate in candidate_set.candidates
            ]

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


def build_mechanism_resolver_input(
    source_search_context: SourceSearchContext,
    source_evidence: SourceEvidenceBundle,
    inventory: Optional[dict[str, Any]] = None,
    timeline: Optional[list[dict[str, Any]]] = None,
    mission: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "mechanism_discovery_context": build_mechanism_discovery_context(
            source_search_context,
            inventory or {},
            timeline or [],
            mission,
        ),
        "source_evidence": compact_source_evidence(source_evidence),
    }


def build_mechanism_discovery_context(
    source_search_context: SourceSearchContext,
    inventory: dict[str, Any],
    timeline: list[dict[str, Any]],
    mission: Optional[dict[str, Any]],
) -> dict[str, Any]:
    airframe = source_search_context.airframe
    intent = source_search_context.question_intent
    return {
        "source_identity": {
            "px4_git_hash": getattr(airframe, "px4_git_hash", None),
            "px4_version": getattr(airframe, "px4_version", None),
            "px4_tag": getattr(airframe, "px4_tag", None),
        },
        "vehicle": {
            "vehicle_type": getattr(airframe, "vehicle_type", None),
            "sys_autostart": getattr(airframe, "sys_autostart", None),
            "airframe_name": getattr(airframe, "airframe_name", None),
            "control_surface_summary": getattr(airframe, "control_surface_summary", None),
        },
        "question_intent": compact_question_intent(intent),
        "source_domain_constraints": {
            "mode_state_timeline": compact_timeline_constraints(timeline),
            "parameter_gates": compact_parameter_gates(inventory, intent, airframe),
            "mission": compact_mission_context(mission),
        },
        "available_log_interfaces": compact_log_interfaces(inventory),
    }


def compact_question_intent(question_intent: QuestionIntent) -> dict[str, Any]:
    return {
        "original_question": getattr(question_intent, "original_question", None),
        "problem_domain": getattr(question_intent, "problem_domain", None),
        "concise_intent": getattr(question_intent, "concise_intent", None),
        "source_queries": list(getattr(question_intent, "source_queries", []) or []),
        "likely_modules": list(getattr(question_intent, "likely_modules", []) or []),
        "likely_source_files": list(getattr(question_intent, "likely_source_files", []) or []),
    }


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


def compact_parameter_gates(
    inventory: dict[str, Any],
    question_intent: QuestionIntent,
    airframe_context: AirframeContext,
) -> dict[str, Any]:
    parameters = inventory.get("parameters") or {}
    requested = set(_parameter_names_from_question_intent(question_intent))
    for name in ("SYS_AUTOSTART", "VT_TYPE"):
        if name in parameters:
            requested.add(name)

    if getattr(airframe_context, "sys_autostart", None) is not None:
        requested.add("SYS_AUTOSTART")

    values = {
        name: parameters[name]
        for name in sorted(requested)
        if name in parameters
    }
    if "SYS_AUTOSTART" not in values and getattr(airframe_context, "sys_autostart", None) is not None:
        values["SYS_AUTOSTART"] = getattr(airframe_context, "sys_autostart")

    return {
        "values": values,
        "matched_parameter_names": sorted(requested),
        "available_parameter_count": len(parameters),
    }


def compact_log_interfaces(inventory: dict[str, Any], max_fields_per_topic: int = 80) -> dict[str, Any]:
    available_topics = sorted(str(topic) for topic in (inventory.get("available_topics") or []))
    topic_fields = inventory.get("topic_fields") or {}
    compact_fields = {}
    for topic in available_topics:
        fields = topic_fields.get(topic)
        if not fields:
            continue
        compact_fields[topic] = [str(field) for field in list(fields)[:max_fields_per_topic]]

    return {
        "available_topics": available_topics,
        "topic_fields": compact_fields,
        "topic_count": len(available_topics),
    }


def compact_mission_context(mission: Optional[dict[str, Any]], max_items: int = 20) -> Optional[dict[str, Any]]:
    if mission is None:
        return None

    items = mission.get("items") or []
    altitudes = [
        item.get("altitude")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("altitude"), (int, float))
    ]
    return {
        "has_mission": True,
        "format": mission.get("format"),
        "planned_home_position": mission.get("planned_home_position"),
        "vehicle_type": mission.get("vehicle_type"),
        "item_count": len(items),
        "command_names": dedupe_keep_order([
            str(item.get("command_name"))
            for item in items
            if isinstance(item, dict) and item.get("command_name")
        ]),
        "frame_names": dedupe_keep_order([
            str(item.get("frame_name"))
            for item in items
            if isinstance(item, dict) and item.get("frame_name")
        ]),
        "altitude_range": {
            "min": min(altitudes),
            "max": max(altitudes),
        } if altitudes else None,
        "items": [
            {
                "sequence": item.get("sequence"),
                "command_name": item.get("command_name"),
                "frame_name": item.get("frame_name"),
                "altitude": item.get("altitude"),
            }
            for item in items[:max_items]
            if isinstance(item, dict)
        ],
        "omitted_item_count": max(len(items) - max_items, 0),
        "warnings": list(mission.get("warnings") or []),
    }


def compact_source_evidence(source_evidence: SourceEvidenceBundle) -> dict[str, Any]:
    evidence = source_evidence.model_dump()
    return {
        "hits": evidence.get("hits", []),
        "read_snippets": evidence.get("read_snippets", []),
        "warnings": evidence.get("warnings", []),
    }


def _parameter_names_from_question_intent(question_intent: QuestionIntent) -> list[str]:
    parts = [
        getattr(question_intent, "original_question", None),
        getattr(question_intent, "problem_domain", None),
        getattr(question_intent, "concise_intent", None),
        *(getattr(question_intent, "source_queries", []) or []),
        *(getattr(question_intent, "likely_modules", []) or []),
        *(getattr(question_intent, "likely_source_files", []) or []),
        *(getattr(question_intent, "notes", []) or []),
    ]
    text = " ".join(str(part) for part in parts if part)
    return dedupe_keep_order([
        name for name in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text)
        if is_px4_parameter_name(name)
    ])


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
