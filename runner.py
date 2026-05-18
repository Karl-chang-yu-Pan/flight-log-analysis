from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal

try:
    from pydantic import BaseModel, Field
except ImportError:
    from pydantic import BaseModel

    def Field(default: Any = None, *, default_factory: Any = None, **_: Any) -> Any:
        return default_factory() if default_factory is not None else default

from agents import Agent, Runner, function_tool, RunContextWrapper, WebSearchTool

from px4_source import checkout_px4_source_revision, read_source_file, search_source
from mission_parser import parse_mission_file as parse_mission_file_impl
from ulog_control_surface import infer_control_surface as infer_control_surface_impl
from ulog_inventory import parse_ulog_inventory as parse_ulog_inventory_impl
from ulog_metrics import compute_log_metrics as compute_log_metrics_impl
from ulog_signature_evaluator import evaluate_log_signature as evaluate_log_signature_impl
from ulog_plots import generate_signal_plot as generate_signal_plot_impl
from ulog_timeline import build_basic_timeline as build_basic_timeline_impl
from run_audit_log import (
    AgentRunAuditHooks,
    DEFAULT_DEV_LOG_ROOT,
    DeveloperAuditLogger,
    log_run_items,
)


# ============================================================
# 1. Context
# ============================================================

@dataclass
class FlightLogContext:
    log_path: Path
    mission_path: Optional[Path]
    source_path: Optional[Path]
    output_dir: Path


# ============================================================
# 2. Shared structured models
# ============================================================

class PlotOverlay(BaseModel):
    start_s: float
    end_s: Optional[float] = None
    label: Optional[str] = None
    color: Optional[str] = None
    alpha: Optional[float] = None
    kind: Optional[str] = None
    source: Optional[str] = None
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


class CodeRef(BaseModel):
    file: str
    function: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    snippet: Optional[str] = None
    explanation: str


class WindowSpec(BaseModel):
    name: str
    start_s: float
    end_s: float
    reason: Optional[str] = None


class ExpectedSignatureItem(BaseModel):
    name: str
    description: str
    signal: Optional[str] = None
    expected_behavior: Optional[str] = None


class DerivedSignalSpec(BaseModel):
    """
    Declarative derived signal. The evaluator implementation should support a
    restricted expression grammar only. Do not eval arbitrary Python.
    """
    name: str
    expr: str
    description: Optional[str] = None
    unit: Optional[str] = None


class EventSpec(BaseModel):
    name: str
    type: Literal[
        "state_transition",
        "threshold_crossing",
        "step_change",
        "local_extreme",
        "time_marker",
    ]
    signal: Optional[str] = None
    from_value: Optional[float | int | str | bool] = None
    to_value: Optional[float | int | str | bool] = None
    threshold: Optional[float] = None
    direction: Optional[Literal["above", "below", "rising", "falling", "any"]] = None
    time_s: Optional[float] = None
    description: Optional[str] = None


class RelationshipCheckSpec(BaseModel):
    """
    Generic check language used by the deterministic log evaluator.
    Keep this declarative; runner/tools perform the actual calculation.
    """
    type: Literal[
        "compare",
        "tracking_error",
        "setpoint_actual_separation",
        "before_after_delta",
        "state_conditioned_mean",
        "threshold_fraction",
        "saturation",
        "event_alignment",
        "rate_of_change",
        "correlation",
        "lagged_correlation",
        "monotonic_change",
        "missing_signal",
        "custom",
    ]
    window: Optional[str] = None
    signal: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None
    actual: Optional[str] = None
    setpoint: Optional[str] = None
    condition: Optional[str] = None
    baseline_condition: Optional[str] = None
    event: Optional[str] = None
    metric: Optional[str] = None
    op: Optional[Literal[">", ">=", "<", "<=", "==", "!=", "between", "outside"]] = None
    value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    max_error: Optional[float] = None
    min_delta: Optional[float] = None
    supports: Optional[str] = None
    contradicts: Optional[str] = None
    description: Optional[str] = None


class LogSignatureSpec(BaseModel):
    mechanism_title: str
    expected_signature: list[ExpectedSignatureItem]
    candidate_windows: list[WindowSpec]
    required_signals: list[str]
    derived_signals: list[DerivedSignalSpec] = Field(default_factory=list)
    events: list[EventSpec] = Field(default_factory=list)
    supporting_checks: list[RelationshipCheckSpec] = Field(default_factory=list)
    exclusion_checks: list[RelationshipCheckSpec] = Field(default_factory=list)
    numeric_checks: list[RelationshipCheckSpec] = Field(default_factory=list)
    plot_requests: list[PlotRef] = Field(default_factory=list)


class SourceMechanism(BaseModel):
    mechanism_confirmed: bool
    mechanism_name: str
    summary: str
    source_refs: list[CodeRef]
    parameters_used: list[str] = Field(default_factory=list)
    state_gates: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    expected_logged_signature_hint: list[ExpectedSignatureItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "unresolved"] = "unresolved"


class HypothesisDraft(BaseModel):
    title: str
    suspected_mechanism: str
    why_plausible: str
    required_source_queries: list[str]
    likely_source_files: list[str] = Field(default_factory=list)
    required_signals: list[str]
    candidate_windows: list[WindowSpec]
    plausible_alternatives_to_exclude: list[str]


class HypothesisDraftSet(BaseModel):
    hypotheses: list[HypothesisDraft]


class SignatureEvaluation(BaseModel):
    mechanism_title: str
    verdict: Literal["supported", "contradicted", "mixed", "unresolved"]
    confidence_ceiling: Literal["high", "medium", "low", "unresolved"]
    evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_required_signals: list[str] = Field(default_factory=list)
    check_results: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class VerifiedHypothesisPackage(BaseModel):
    draft: HypothesisDraft
    source_mechanism: SourceMechanism
    signature_spec: LogSignatureSpec
    evaluation: SignatureEvaluation


class Hypothesis(BaseModel):
    title: str
    known_px4_mechanism: str = ""
    mechanism: str
    expected_logged_signature: list[ExpectedSignatureItem] = Field(default_factory=list)
    exclusion_checks: list[RelationshipCheckSpec] = Field(default_factory=list)
    numeric_checks: list[RelationshipCheckSpec] = Field(default_factory=list)
    evidence: list[str]
    contradicting_evidence: list[str]
    confidence: Literal["high", "medium", "low", "unresolved"]
    plots: list[PlotRef]
    code_references: list[CodeRef]
    verifier_verdict: Literal["supported", "contradicted", "mixed", "unresolved"] = "unresolved"
    source_confirmed: bool = False
    unresolved: list[str] = Field(default_factory=list)


class FlightLogReport(BaseModel):
    assumption_header: str
    log_inventory_summary: str
    timeline_summary: str
    relevant_windows: list[str]
    ranked_hypotheses: list[Hypothesis]
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
# 3. Deterministic pre-pass functions
# ============================================================

def parse_ulog_inventory(log_path: Path) -> dict:
    return parse_ulog_inventory_impl(log_path)


def build_basic_timeline(log_path: Path) -> list[dict]:
    return build_basic_timeline_impl(log_path)


def infer_control_surface(log_path: Path, source_path: Optional[Path]) -> dict:
    return infer_control_surface_impl(log_path, source_path)


def infer_control_mapping(log_path: Path, source_path: Optional[Path]) -> dict:
    return infer_control_surface(log_path, source_path)


def parse_mission_file(mission_path: Optional[Path]) -> Optional[dict]:
    return parse_mission_file_impl(mission_path)


# ============================================================
# 4. Tools available to agents
# ============================================================

@function_tool
def search_px4_source(
    ctx: RunContextWrapper[FlightLogContext],
    query: str,
    max_results: int = 8,
) -> list[dict]:
    source_path = ctx.context.source_path
    if source_path is None:
        return [{"error": "No PX4 source path provided."}]
    return search_source(source_path, query, max_results=max_results)


@function_tool
def checkout_px4_source(
    ctx: RunContextWrapper[FlightLogContext],
    revision: str,
) -> dict:
    source_path = ctx.context.source_path
    if source_path is None:
        return {"error": "No PX4 source path provided."}
    return checkout_px4_source_revision(source_path, revision)


@function_tool
def read_px4_source_file(
    ctx: RunContextWrapper[FlightLogContext],
    relative_path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> dict:
    source_path = ctx.context.source_path
    if source_path is None:
        return {"error": "No PX4 source path provided."}
    return read_source_file(source_path, relative_path, start_line, end_line)


@function_tool
def resolve_px4_mechanism(
    ctx: RunContextWrapper[FlightLogContext],
    hypothesis_title: str,
    suspected_mechanism: str,
    source_queries: list[str],
    likely_source_files: Optional[list[str]] = None,
    max_results_per_query: int = 8,
) -> dict:
    """
    Source evidence gathering helper. It does not decide final confidence.
    The mechanism_resolver_agent must convert this raw evidence into a
    SourceMechanism with concrete files/functions/conditions.
    """
    source_path = ctx.context.source_path
    if source_path is None:
        return {"error": "No PX4 source path provided."}

    evidence: dict[str, Any] = {
        "hypothesis_title": hypothesis_title,
        "suspected_mechanism": suspected_mechanism,
        "search_results": [],
        "file_reads": [],
    }

    for query in source_queries:
        evidence["search_results"].append(
            {
                "query": query,
                "results": search_source(source_path, query, max_results=max_results_per_query),
            }
        )

    for relative_path in likely_source_files or []:
        try:
            evidence["file_reads"].append(
                read_source_file(source_path, relative_path, 1, None)
            )
        except Exception as exc:
            evidence["file_reads"].append(
                {"file": relative_path, "error": repr(exc)}
            )

    return evidence


@function_tool
def compute_log_metrics(
    ctx: RunContextWrapper[FlightLogContext],
    start_s: float,
    end_s: float,
    signals: list[str],
) -> dict:
    return compute_log_metrics_impl(ctx.context.log_path, start_s, end_s, signals)


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
        overlays=[overlay.model_dump(exclude_none=True) for overlay in overlays] if overlays else None,
    )


@function_tool
def evaluate_log_signature(
    ctx: RunContextWrapper[FlightLogContext],
    mechanism_title: str,
    expected_signature: list[ExpectedSignatureItem],
    candidate_windows: list[WindowSpec],
    required_signals: list[str],
    derived_signals: list[DerivedSignalSpec],
    events: list[EventSpec],
    supporting_checks: list[RelationshipCheckSpec],
    exclusion_checks: list[RelationshipCheckSpec],
    numeric_checks: list[RelationshipCheckSpec],
) -> dict:
    """
    General deterministic log relationship evaluator.

    Recommended implementation direction:
    - Support a restricted derived-signal expression grammar.
    - Evaluate signal relationships, event alignment, tracking error,
      saturation, before/after deltas, state-conditioned means, etc.
    - Return support/contradiction/missing-signal details per check.

    Backed by ulog_signature_evaluator, which evaluates generic relationship
    checks and returns normalized support, contradiction, and missing-signal
    details.
    """
    return evaluate_log_signature_impl(
        ctx.context.log_path,
        mechanism_title,
        [_model_to_dict(item) for item in expected_signature],
        [_window_to_legacy_dict(window) for window in candidate_windows],
        required_signals,
        [_model_to_dict(signal) for signal in derived_signals],
        [_model_to_dict(event) for event in events],
        [_relationship_to_legacy_check(check) for check in supporting_checks],
        [_relationship_to_legacy_check(check) for check in exclusion_checks],
        [_relationship_to_legacy_check(check) for check in numeric_checks],
    )


# ============================================================
# 5. Agents for staged workflow
# ============================================================

hypothesis_drafter_agent = Agent(
    name="PX4 Hypothesis Drafter",
    instructions="""
You draft candidate hypotheses only. Do not assign confidence.

For each hypothesis, provide:
- one specific suspected PX4 mechanism, not a bundle of alternatives
- why it is plausible from inventory/timeline/user question
- source queries/files needed to confirm the mechanism
- required log signals
- candidate windows
- plausible alternatives that must be excluded

Do not write final evidence. Do not claim the mechanism is confirmed.
""",
    tools=[WebSearchTool()],
    output_type=HypothesisDraftSet,
)


mechanism_resolver_agent = Agent(
    name="PX4 Mechanism Resolver",
    instructions="""
Resolve one hypothesis against PX4 source code.

Rules:
- Use local PX4 source tools. Web search is only for discovery, never final code truth.
- Return mechanism_confirmed=false if you cannot identify a concrete code path.
- A confirmed mechanism needs concrete source refs: file, function when possible,
  line range when possible, and a short snippet when possible.
- Identify parameters, state gates, and branch conditions that select this behavior.
- Convert source behavior into hints for expected logged signatures.
- Do not assign flight-log confidence. Only resolve source mechanism confidence.
""",
    tools=[
        WebSearchTool(),
        checkout_px4_source,
        search_px4_source,
        read_px4_source_file,
        resolve_px4_mechanism,
    ],
    output_type=SourceMechanism,
)


signature_builder_agent = Agent(
    name="PX4 Log Signature Builder",
    instructions="""
Build a deterministic log-verification spec from a source-resolved mechanism.

Output must include:
- expected logged signature
- candidate windows
- required signals
- optional derived signals using a restricted expression-style description
- events if useful
- supporting relationship checks
- exclusion checks for plausible alternate causes
- numeric checks
- plot requests that visualize only relevant signals/windows

Do not assign confidence. Do not write final report text.
Make checks generic and declarative so evaluate_log_signature can execute them.
""",
    tools=[compute_log_metrics],
    output_type=LogSignatureSpec,
)


final_report_agent = Agent(
    name="PX4 Final Report Writer",
    instructions="""
Write the final report only from verified hypothesis packages.

Rules:
- Keep the exact chain visible: known PX4 mechanism, expected logged signature,
  exclusion checks, numeric checks, contradictions, confidence.
- Do not introduce new hypotheses that were not verified.
- Confidence cannot exceed evaluation.confidence_ceiling.
- If source_mechanism.mechanism_confirmed is false, confidence must be low or unresolved.
- If required signals are missing for a key check, confidence must be low or unresolved.
- If evaluation verdict is contradicted, do not present the hypothesis as likely.
- Separate observed log evidence, source-code inference, parameter inference, and unknowns.
- Do not provide tuning/fix suggestions unless suggestions_requested is true.
""",
    tools=[],
    output_type=FlightLogReport,
)


report_repair_agent = Agent(
    name="PX4 Report Repairer",
    instructions="""
Repair a report so it satisfies validation issues.

Rules:
- Do not add unsupported evidence.
- Downgrade confidence rather than inventing support.
- Preserve the verified packages as the only source of truth.
- Explicitly mark unresolved mechanisms/checks as unresolved.
""",
    tools=[],
    output_type=FlightLogReport,
)


# ============================================================
# 6. Main V2 runner
# ============================================================

async def analyze_flight_log_v2(
    log_path: str,
    user_question: str,
    mission_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_dir: str = "outputs/run_001",
    dev_log_root: str = str(DEFAULT_DEV_LOG_ROOT),
    dev_run_id: Optional[str] = None,
    max_hypotheses: int = 4,
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
            "runner_version": "v2_staged_verification",
            "log_path": str(log_path_obj),
            "mission_path": str(mission_path_obj) if mission_path_obj else None,
            "source_path": str(source_path_obj) if source_path_obj else None,
            "output_dir": str(output_dir_obj),
            "report_path": str(report_path),
        }
    )
    audit_logger.log_event(
        "run.started",
        input={
            "log_path": str(log_path_obj),
            "user_question": user_question,
            "mission_path": str(mission_path_obj) if mission_path_obj else None,
            "source_path": str(source_path_obj) if source_path_obj else None,
            "output_dir": str(output_dir_obj),
            "max_hypotheses": max_hypotheses,
        },
    )

    try:
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
        assumptions = _audit_sync_call(
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

        ctx = FlightLogContext(
            log_path=log_path_obj,
            mission_path=mission_path_obj,
            source_path=source_path_obj,
            output_dir=output_dir_obj,
        )

        base_payload = {
            "user_question": user_question,
            "log_inventory": inventory,
            "flight_timeline": timeline,
            "detected_assumptions": assumptions,
            "mission_summary": mission,
            "source_path": str(source_path_obj) if source_path_obj else None,
            "suggestions_requested": _suggestions_requested(user_question),
        }

        drafts = await _run_agent(
            audit_logger,
            "draft_hypotheses",
            hypothesis_drafter_agent,
            base_payload,
            ctx,
            max_turns=8,
        )
        draft_list = drafts.hypotheses[:max_hypotheses]

        verified_packages: list[VerifiedHypothesisPackage] = []

        for index, draft in enumerate(draft_list, start=1):
            stage_payload = {
                **base_payload,
                "hypothesis_index": index,
                "hypothesis_draft": draft.model_dump(),
            }

            source_mechanism = await _run_agent(
                audit_logger,
                f"resolve_mechanism_{index}",
                mechanism_resolver_agent,
                stage_payload,
                ctx,
                max_turns=12,
            )

            signature_spec = await _run_agent(
                audit_logger,
                f"build_signature_{index}",
                signature_builder_agent,
                {
                    **stage_payload,
                    "source_mechanism": source_mechanism.model_dump(),
                },
                ctx,
                max_turns=8,
            )

            evaluation = _audit_sync_call(
                audit_logger,
                "verification",
                "evaluate_log_signature",
                _evaluate_signature_for_runner,
                {
                    "mechanism_title": signature_spec.mechanism_title,
                    "required_signals": signature_spec.required_signals,
                    "candidate_windows": [w.model_dump() for w in signature_spec.candidate_windows],
                    "supporting_checks": [c.model_dump(exclude_none=True) for c in signature_spec.supporting_checks],
                    "exclusion_checks": [c.model_dump(exclude_none=True) for c in signature_spec.exclusion_checks],
                    "numeric_checks": [c.model_dump(exclude_none=True) for c in signature_spec.numeric_checks],
                },
                ctx,
                signature_spec,
            )

            verified_packages.append(
                VerifiedHypothesisPackage(
                    draft=draft,
                    source_mechanism=source_mechanism,
                    signature_spec=signature_spec,
                    evaluation=evaluation,
                )
            )

        report = await _run_agent(
            audit_logger,
            "final_report",
            final_report_agent,
            {
                **base_payload,
                "verified_hypothesis_packages": [pkg.model_dump() for pkg in verified_packages],
            },
            ctx,
            max_turns=8,
        )

        report = generate_report_plots(report, ctx, audit_logger=audit_logger)
        validation = validate_report(report)
        audit_logger.log_event("validation.finished", output=validation.model_dump())

        if not validation.passed:
            report = await _run_agent(
                audit_logger,
                "repair_report",
                report_repair_agent,
                {
                    **base_payload,
                    "report": report.model_dump(),
                    "validation": validation.model_dump(),
                    "verified_hypothesis_packages": [pkg.model_dump() for pkg in verified_packages],
                },
                ctx,
                max_turns=6,
            )
            report = generate_report_plots(report, ctx, audit_logger=audit_logger)
            validation = validate_report(report)
            audit_logger.log_event("validation_after_repair.finished", output=validation.model_dump())

        save_report(report, report_path)
        audit_logger.log_event(
            "run.finished",
            output={
                "report_path": str(report_path),
                "dev_log_dir": str(audit_logger.run_dir),
                "validation_passed": validation.passed,
            },
        )
        return report

    except Exception as exc:
        audit_logger.log_event("run.failed", error=repr(exc))
        raise


analyze_flight_log = analyze_flight_log_v2


# ============================================================
# 7. Verification, validation, and plot materialization
# ============================================================

def _evaluate_signature_for_runner(
    ctx: FlightLogContext,
    spec: LogSignatureSpec,
) -> SignatureEvaluation:
    raw = evaluate_log_signature_impl(
        ctx.log_path,
        spec.mechanism_title,
        [_model_to_dict(item) for item in spec.expected_signature],
        [_window_to_legacy_dict(window) for window in spec.candidate_windows],
        spec.required_signals,
        [_model_to_dict(signal) for signal in spec.derived_signals],
        [_model_to_dict(event) for event in spec.events],
        [_relationship_to_legacy_check(check) for check in spec.supporting_checks],
        [_relationship_to_legacy_check(check) for check in spec.exclusion_checks],
        [_relationship_to_legacy_check(check) for check in spec.numeric_checks],
    )
    return _normalize_evaluation_result(
        mechanism_title=spec.mechanism_title,
        raw=raw,
        derived_signals=spec.derived_signals,
        events=spec.events,
    )


def _normalize_evaluation_result(
    mechanism_title: str,
    raw: dict[str, Any],
    derived_signals: list[DerivedSignalSpec],
    events: list[EventSpec],
) -> SignatureEvaluation:
    missing = _extract_list(raw, ["missing_required_signals", "missing_signals"])
    evidence = _extract_list(raw, ["evidence", "supporting_evidence", "supports"])
    contradictions = _extract_list(raw, ["contradictions", "contradicting_evidence", "contradicts"])
    warnings = _extract_list(raw, ["warnings"])

    verdict_raw = str(raw.get("verdict") or raw.get("status") or "").lower()
    if "support" in verdict_raw:
        verdict: Literal["supported", "contradicted", "mixed", "unresolved"] = "supported"
    elif "contrad" in verdict_raw:
        verdict = "contradicted"
    elif "mixed" in verdict_raw:
        verdict = "mixed"
    else:
        if contradictions and evidence:
            verdict = "mixed"
        elif contradictions:
            verdict = "contradicted"
        elif evidence and not missing:
            verdict = "supported"
        else:
            verdict = "unresolved"

    if missing:
        ceiling: Literal["high", "medium", "low", "unresolved"] = "low"
    elif verdict == "supported":
        ceiling = "high"
    elif verdict == "mixed":
        ceiling = "medium"
    elif verdict == "contradicted":
        ceiling = "low"
    else:
        ceiling = "unresolved"

    if derived_signals:
        warnings.append(
            "Derived signals were declared; make sure the evaluator implementation supports the restricted expression grammar."
        )
    if events:
        warnings.append(
            "Events were declared; make sure event detection results are represented in check_results."
        )

    check_results = raw.get("check_results")
    if not isinstance(check_results, list):
        check_results = raw.get("checks") if isinstance(raw.get("checks"), list) else []

    return SignatureEvaluation(
        mechanism_title=mechanism_title,
        verdict=verdict,
        confidence_ceiling=ceiling,
        evidence=[str(x) for x in evidence],
        contradictions=[str(x) for x in contradictions],
        missing_required_signals=[str(x) for x in missing],
        check_results=check_results,
        warnings=[str(x) for x in warnings],
        raw=raw,
    )


def validate_report(report: FlightLogReport) -> ValidationResult:
    issues: list[ValidationIssue] = []

    for i, hyp in enumerate(report.ranked_hypotheses):
        path = f"ranked_hypotheses[{i}]"

        if hyp.confidence in ("high", "medium") and not hyp.source_confirmed:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"{path}.confidence",
                    message="Confidence is medium/high but source mechanism is not confirmed.",
                )
            )

        if hyp.source_confirmed and not hyp.code_references:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"{path}.code_references",
                    message="Source-confirmed hypothesis has no code references.",
                )
            )

        for j, ref in enumerate(hyp.code_references):
            if hyp.source_confirmed and not ref.snippet and ref.start_line is None:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        path=f"{path}.code_references[{j}]",
                        message="Code reference lacks both snippet and line range.",
                    )
                )

        if hyp.confidence in ("high", "medium") and hyp.verifier_verdict in ("unresolved", "contradicted"):
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"{path}.verifier_verdict",
                    message="Confidence is medium/high but verifier verdict is unresolved or contradicted.",
                )
            )

        if hyp.confidence == "high" and hyp.contradicting_evidence:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path=f"{path}.contradicting_evidence",
                    message="High confidence hypothesis still has contradicting evidence.",
                )
            )

        if not hyp.expected_logged_signature:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"{path}.expected_logged_signature",
                    message="Hypothesis has no expected logged signature.",
                )
            )

        if not hyp.numeric_checks:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"{path}.numeric_checks",
                    message="Hypothesis has no numeric checks.",
                )
            )

        for j, plot in enumerate(hyp.plots):
            if plot.missing_signals and hyp.confidence in ("high", "medium"):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        path=f"{path}.plots[{j}].missing_signals",
                        message="Plot has missing signals while hypothesis confidence is medium/high.",
                    )
                )

    passed = not any(issue.severity == "error" for issue in issues)
    return ValidationResult(passed=passed, issues=issues)


def generate_report_plots(
    report: FlightLogReport,
    ctx: FlightLogContext,
    audit_logger: Optional[DeveloperAuditLogger] = None,
) -> FlightLogReport:
    hypotheses = getattr(report, "ranked_hypotheses", None)
    if not hypotheses:
        return report

    for hypothesis in hypotheses:
        generated_count = 0

        plots = getattr(hypothesis, "plots", None) or []
        for plot in plots:
            spec = _plot_generation_spec(plot)
            if spec is None:
                continue

            result = _audit_sync_call(
                audit_logger,
                "postprocess_plot",
                "generate_signal_plot",
                generate_signal_plot_impl,
                {
                    "log_path": str(ctx.log_path),
                    "output_dir": str(ctx.output_dir),
                    **spec,
                },
                ctx.log_path,
                ctx.output_dir,
                spec["title"],
                spec["start_s"],
                spec["end_s"],
                spec["signals"],
                spec["purpose"],
                plot_type=spec["plot_type"],
                bins=spec["bins"],
                overlays=spec["overlays"],
            )
            _apply_plot_result(plot, result)
            generated_count += 1

        if generated_count == 0 and not _has_existing_plot_file(plots):
            title = getattr(hypothesis, "title", "untitled hypothesis")
            _append_unconfirmed(
                report,
                f"Plot generation was not attempted for hypothesis '{title}' because no complete plot spec was returned.",
            )

    return report


# ============================================================
# 8. Helpers
# ============================================================

async def _run_agent(
    audit_logger: Optional[DeveloperAuditLogger],
    stage_name: str,
    agent: Agent,
    payload: dict[str, Any],
    ctx: FlightLogContext,
    max_turns: int,
    max_attempts: int = 5,
) -> Any:
    started_at = time.perf_counter()
    if audit_logger is not None:
        audit_logger.log_event(f"agent.{stage_name}.started", input=payload)

    result = None
    serialized_input = json.dumps(payload, indent=2, default=str)
    for attempt in range(1, max_attempts + 1):
        try:
            result = await Runner.run(
                agent,
                input=serialized_input,
                context=ctx,
                max_turns=max_turns,
                hooks=AgentRunAuditHooks(audit_logger) if audit_logger is not None else None,
            )
            break
        except Exception as exc:
            if attempt >= max_attempts or not _is_rate_limit_error(exc):
                if audit_logger is not None:
                    audit_logger.log_event(
                        f"agent.{stage_name}.failed",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error=repr(exc),
                        duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
                    )
                raise

            delay_s = _rate_limit_retry_delay_s(exc, attempt)
            if audit_logger is not None:
                audit_logger.log_event(
                    f"agent.{stage_name}.retrying",
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    max_attempts=max_attempts,
                    delay_s=round(delay_s, 3),
                    error=repr(exc),
                )
            await asyncio.sleep(delay_s)

    if result is None:
        raise RuntimeError(f"agent.{stage_name} did not return a result")

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


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "rate limit",
            "rate_limit",
            "tokens per min",
            "tpm",
            "429",
            "too many requests",
        )
    )


def _rate_limit_retry_delay_s(exc: Exception, attempt: int) -> float:
    suggested_delay = _parse_retry_delay_s(str(exc))
    if suggested_delay is not None:
        return max(0.0, suggested_delay)

    base_delay = min(8.0, 0.5 * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, min(0.25, base_delay * 0.25))
    return base_delay + jitter


def _parse_retry_delay_s(message: str) -> Optional[float]:
    match = re.search(r"try again in\s+([0-9]*\.?[0-9]+)\s*(ms|s|sec|secs|second|seconds)\b", message, re.IGNORECASE)
    if match is None:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ms":
        return value / 1000.0
    return value


def save_report(report: FlightLogReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(report, "model_dump_json"):
        report_json = report.model_dump_json(indent=2)
    else:
        report_json = json.dumps(report, indent=2, default=str)
    path.write_text(report_json, encoding="utf-8")


def _plot_generation_spec(plot: Any) -> Optional[dict[str, Any]]:
    start_s = getattr(plot, "start_s", None)
    end_s = getattr(plot, "end_s", None)
    signals = getattr(plot, "signals", None) or []
    if start_s is None or end_s is None or not signals:
        return None
    return {
        "title": getattr(plot, "title", "Flight Log Plot"),
        "purpose": getattr(plot, "purpose", ""),
        "start_s": float(start_s),
        "end_s": float(end_s),
        "signals": list(signals),
        "plot_type": getattr(plot, "plot_type", "timeseries") or "timeseries",
        "bins": int(getattr(plot, "bins", 50) or 50),
        "overlays": [_model_to_dict(overlay) for overlay in (getattr(plot, "overlays", None) or [])],
    }


def _apply_plot_result(plot: Any, result: dict[str, Any]) -> None:
    for key in (
        "title",
        "path",
        "purpose",
        "signals",
        "plot_type",
        "overlays",
        "missing_signals",
        "warnings",
    ):
        if key in result:
            setattr(plot, key, result[key])


def _has_existing_plot_file(plots: list[Any]) -> bool:
    return any(getattr(plot, "path", "") and Path(getattr(plot, "path")).is_file() for plot in plots)


def _append_unconfirmed(report: Any, message: str) -> None:
    unconfirmed = getattr(report, "unconfirmed", None)
    if unconfirmed is None:
        return
    if message not in unconfirmed:
        unconfirmed.append(message)


def _model_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if item is not None}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if item is not None}
    return dict(value)


def _window_to_legacy_dict(window: WindowSpec) -> dict[str, Any]:
    return {
        "name": window.name,
        "start_s": window.start_s,
        "end_s": window.end_s,
    }


def _relationship_to_legacy_check(check: RelationshipCheckSpec) -> dict[str, Any]:
    data = _model_to_dict(check)
    if "from_value" in data:
        data["from"] = data.pop("from_value")
    if "to_value" in data:
        data["to"] = data.pop("to_value")
    return data


def _extract_list(raw: dict[str, Any], keys: list[str]) -> list[Any]:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            return [value]
    return []


def _safe_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_safe_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _safe_model_dump(item) for key, item in value.items()}
    return value


def _suggestions_requested(user_question: str) -> bool:
    q = user_question.lower()
    return any(token in q for token in ("suggest", "fix", "tune", "recommend", "what should i change"))


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
            user_question=(
                "Why did the aircraft start loitering before reaching "
                "the next waypoint? Do not provide suggestions."
            ),
        )
    )
    print(report.model_dump_json(indent=2))
