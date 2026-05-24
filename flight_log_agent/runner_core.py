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
