from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List

from pydantic import BaseModel
from agents import Agent, Runner, function_tool, RunContextWrapper, WebSearchTool

from px4_source import checkout_px4_source_revision, read_source_file, search_source
from mission_parser import parse_mission_file as parse_mission_file_impl
from ulog_control_surface import infer_control_surface as infer_control_surface_impl
from ulog_inventory import parse_ulog_inventory as parse_ulog_inventory_impl
from ulog_metrics import compute_log_metrics as compute_log_metrics_impl
from ulog_plots import generate_signal_plot as generate_signal_plot_impl
from ulog_timeline import build_basic_timeline as build_basic_timeline_impl
from run_audit_log import (
    AgentRunAuditHooks,
    DEFAULT_DEV_LOG_ROOT,
    DeveloperAuditLogger,
    log_run_items,
)


# ============================================================
# 1. Run context: local data/tools can access this
# ============================================================

@dataclass
class FlightLogContext:
    log_path: Path
    mission_path: Optional[Path]
    source_path: Optional[Path]
    output_dir: Path


# ============================================================
# 2. Structured report output
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
    path: str
    purpose: str
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    signals: Optional[List[str]] = None
    plot_type: str = "timeseries"
    bins: int = 50
    overlays: Optional[List[PlotOverlay]] = None
    missing_signals: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class CodeRef(BaseModel):
    file: str
    function: Optional[str] = None
    snippet: Optional[str] = None
    explanation: str


class Hypothesis(BaseModel):
    title: str
    mechanism: str
    evidence: List[str]
    contradicting_evidence: List[str]
    confidence: str
    plots: List[PlotRef]
    code_references: List[CodeRef]


class FlightLogReport(BaseModel):
    assumption_header: str
    log_inventory_summary: str
    timeline_summary: str
    relevant_windows: List[str]
    ranked_hypotheses: List[Hypothesis]
    confirmed: List[str]
    unconfirmed: List[str]
    final_summary: str


# ============================================================
# 3. Deterministic pre-pass functions
#    These are normal Python, not agent tools.
# ============================================================

def parse_ulog_inventory(log_path: Path) -> dict:
    """
    Extract:
    - firmware version / git hash
    - parameters
    - available uORB topics
    - warnings/errors
    - duration
    """
    return parse_ulog_inventory_impl(log_path)


def build_basic_timeline(log_path: Path) -> list[dict]:
    """
    Build from vehicle_status, vehicle_type, nav_state,
    arming_state, vtol_vehicle_status, mission_result.
    """
    return build_basic_timeline_impl(log_path)


def infer_control_surface(log_path: Path, source_path: Optional[Path]) -> dict:
    """
    Infer actuator/control-surface assumptions from logged PX4 parameters.
    """
    return infer_control_surface_impl(log_path, source_path)


def infer_control_mapping(log_path: Path, source_path: Optional[Path]) -> dict:
    """
    Backward-compatible name for control-surface inference.
    """
    return infer_control_surface(log_path, source_path)


def parse_mission_file(mission_path: Optional[Path]) -> Optional[dict]:
    """
    Parse .plan or mission file if provided.
    """
    return parse_mission_file_impl(mission_path)


# ============================================================
# 4. Agent tools
# ============================================================

@function_tool
def search_px4_source(
    ctx: RunContextWrapper[FlightLogContext],
    query: str,
    max_results: int = 8,
) -> list[dict]:
    """
    Search local PX4 source code with ripgrep.
    """
    source_path = ctx.context.source_path

    if source_path is None:
        return [{"error": "No PX4 source path provided."}]

    return search_source(source_path, query, max_results=max_results)


@function_tool
def checkout_px4_source(
    ctx: RunContextWrapper[FlightLogContext],
    revision: str,
) -> dict:
    """
    Checkout local PX4 source code to a git hash, tag, or branch.

    Refuses to checkout if the PX4 tree has local changes.
    """
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
    """
    Read a line range from a file inside the local PX4 source tree.
    """
    source_path = ctx.context.source_path

    if source_path is None:
        return {"error": "No PX4 source path provided."}

    return read_source_file(source_path, relative_path, start_line, end_line)


@function_tool
def compute_log_metrics(
    ctx: RunContextWrapper[FlightLogContext],
    start_s: float,
    end_s: float,
    signals: list[str],
) -> dict:
    """
    Compute numeric metrics for selected signals.
    """
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
    """
    Generate a hypothesis-specific plot.
    """
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


# ============================================================
# 5. Single V1 agent
# ============================================================

flight_log_agent = Agent(
    name="PX4 Flight Log Analyst V1",
    instructions="""
You are a generic PX4 flight-log investigation agent.

You receive:
- pre-parsed log inventory
- detected vehicle/control assumptions
- broad flight timeline
- optional mission summary
- user question
- access to tools for local PX4 source search, metrics, plots, and web search

Workflow:
1. Start from the provided log inventory and timeline.
2. Show detected vehicle/control-surface assumptions at the top.
3. Use log_inventory.topic_fields when choosing plot signals. Prefer exact
   logged fields from topic_fields. The runner prepares Flight Review-style
   derived vehicle_attitude roll/pitch/yaw fields from quaternions before
   plotting.
4. Use web search only for culprit discovery:
   PX4 docs, forum posts, GitHub issues, parameter concepts, known mechanisms.
5. If a relevant PX4 git hash, tag, or branch is known, checkout the local
   PX4 source tree before source-code investigation.
6. Use local PX4 source search for exact code behavior.
7. Select relevant analysis windows.
8. Generate ranked hypotheses.
9. For each hypothesis, include:
   - mechanism
   - evidence from log
   - contradicting evidence
   - at least one relevant plot request in plots
   - relevant code path
   - confidence
10. Do not provide parameter tuning, code-change, or flight-test suggestions
   unless the user explicitly asks for suggestions or fixes.

Plot requirements:
- For every hypothesis, include at least one plots entry with signals and a
  time window directly related to that hypothesis. Treat plots entries as
  structured generation requests: fill title, purpose, start_s, end_s, signals,
  plot_type, bins, and overlays. Set path to an empty string; the runner will
  generate the PNG after your final report and fill in the actual path,
  missing_signals, and warnings.
- Decide which overlays are useful for each hypothesis plot. Relevant overlays
  can include mode changes, mission item changes, VTOL state changes,
  parameter-change evidence if available, user/control input changes, failsafe
  changes, or command/state transitions.
- Derive overlay times from the provided flight_timeline or from
  compute_log_metrics transitions on discrete signals. Use vertical markers for
  point changes and shaded spans for state intervals.
- Keep overlays selective: include context that explains or challenges the
  hypothesis, and avoid unrelated clutter.
- In the plot purpose, state why the chosen signals and overlays are relevant.
- If a plot cannot be generated because the required signals are missing, state
  the attempted signals, window, and missing-signal reason in the hypothesis.

Always separate:
- observed from log
- inferred from parameters
- inferred from source code
- inferred from web/docs
- unknown

For actuator/control-surface mapping:
- state what was inferred
- state evidence
- state confidence
- warn that physical wiring may differ

For mission acceptance:
- do not assume NAV_ACC_RAD alone explains fixed-wing waypoint switching
- consider NPFG switch distance, ground speed, altitude acceptance,
  mission item logic, and source code
""",
    tools=[
        WebSearchTool(),
        search_px4_source,
        checkout_px4_source,
        read_px4_source_file,
        compute_log_metrics,
        generate_signal_plot,
    ],
    output_type=FlightLogReport,
)


# ============================================================
# 6. V1 runner
# ============================================================

async def analyze_flight_log_v1(
    log_path: str,
    user_question: str,
    mission_path: Optional[str] = None,
    source_path: Optional[str] = None,
    output_dir: str = "outputs/run_001",
    dev_log_root: str = str(DEFAULT_DEV_LOG_ROOT),
) -> FlightLogReport:

    log_path = Path(log_path)
    mission_path_obj = Path(mission_path) if mission_path else None
    source_path_obj = Path(source_path) if source_path else None
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    audit_logger = DeveloperAuditLogger(Path(dev_log_root))
    report_path = output_dir_obj / "report.json"
    audit_logger.save_metadata(
        {
            "log_path": str(log_path),
            "mission_path": str(mission_path_obj) if mission_path_obj else None,
            "source_path": str(source_path_obj) if source_path_obj else None,
            "output_dir": str(output_dir_obj),
            "report_path": str(report_path),
        }
    )
    audit_logger.log_event(
        "run.started",
        input={
            "log_path": log_path,
            "user_question": user_question,
            "mission_path": mission_path_obj,
            "source_path": source_path_obj,
            "output_dir": output_dir_obj,
        },
    )

    try:
        # Deterministic pre-pass
        inventory = _audit_sync_call(
            audit_logger,
            "prepass",
            "parse_ulog_inventory",
            parse_ulog_inventory,
            {"log_path": log_path},
            log_path,
        )
        timeline = _audit_sync_call(
            audit_logger,
            "prepass",
            "build_basic_timeline",
            build_basic_timeline,
            {"log_path": log_path},
            log_path,
        )
        assumptions = _audit_sync_call(
            audit_logger,
            "prepass",
            "infer_control_surface",
            infer_control_surface,
            {"log_path": log_path, "source_path": source_path_obj},
            log_path,
            source_path_obj,
        )
        mission = _audit_sync_call(
            audit_logger,
            "prepass",
            "parse_mission_file",
            parse_mission_file,
            {"mission_path": mission_path_obj},
            mission_path_obj,
        )
    except Exception as exc:
        audit_logger.log_event("run.failed", error=repr(exc))
        raise

    ctx = FlightLogContext(
        log_path=log_path,
        mission_path=mission_path_obj,
        source_path=source_path_obj,
        output_dir=output_dir_obj,
    )

    agent_input = {
        "user_question": user_question,
        "log_inventory": inventory,
        "flight_timeline": timeline,
        "detected_assumptions": assumptions,
        "mission_summary": mission,
        "source_path": str(source_path_obj) if source_path_obj else None,
        "suggestions_requested": (
            "suggest" in user_question.lower()
            or "fix" in user_question.lower()
            or "tune" in user_question.lower()
        ),
    }

    try:
        result = await Runner.run(
            flight_log_agent,
            input=json.dumps(agent_input, indent=2),
            context=ctx,
            max_turns=20,
            hooks=AgentRunAuditHooks(audit_logger),
        )
        log_run_items(audit_logger, getattr(result, "new_items", []) or [])
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        audit_logger.save_usage(usage)

        report = generate_report_plots(result.final_output, ctx, audit_logger=audit_logger)
        save_report(report, report_path)
        audit_logger.log_event(
            "run.finished",
            output={"report_path": report_path, "dev_log_dir": audit_logger.run_dir},
            usage=usage,
        )
        return report
    except Exception as exc:
        audit_logger.log_event("run.failed", error=repr(exc))
        raise


def save_report(report: FlightLogReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(report, "model_dump_json"):
        report_json = report.model_dump_json(indent=2)
    else:
        report_json = json.dumps(report, indent=2)

    path.write_text(report_json, encoding="utf-8")


def generate_report_plots(
    report: FlightLogReport,
    ctx: FlightLogContext,
    audit_logger: Optional[DeveloperAuditLogger] = None,
) -> FlightLogReport:
    """
    Materialize plot requests embedded in the structured report.

    The model chooses hypothesis-specific plot specs; the runner writes the
    files so plot artifacts do not depend on whether the model remembered to
    call a tool during the run.
    """
    hypotheses = getattr(report, "ranked_hypotheses", None)
    if not hypotheses:
        return report

    for hypothesis in hypotheses:
        plots = getattr(hypothesis, "plots", None) or []
        generated_count = 0

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
                    "log_path": ctx.log_path,
                    "output_dir": ctx.output_dir,
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
                (
                    f"Plot generation was not attempted for hypothesis "
                    f"'{title}' because no complete plot spec was returned."
                ),
            )

    return report


def _audit_sync_call(
    audit_logger: Optional[DeveloperAuditLogger],
    event_prefix: str,
    name: str,
    func: Any,
    input_payload: dict,
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
        output=result,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
    )
    return result


def _plot_generation_spec(plot: Any) -> Optional[dict]:
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


def _apply_plot_result(plot: Any, result: dict) -> None:
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
    for plot in plots:
        path = getattr(plot, "path", "")
        if path and Path(path).is_file():
            return True

    return False


def _append_unconfirmed(report: Any, message: str) -> None:
    unconfirmed = getattr(report, "unconfirmed", None)
    if unconfirmed is None:
        return

    if message not in unconfirmed:
        unconfirmed.append(message)


def _model_to_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if item is not None}

    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)

    return {
        key: getattr(value, key)
        for key in ("start_s", "end_s", "label", "color", "alpha", "kind", "source", "ymin", "ymax")
        if getattr(value, key, None) is not None
    }


# ============================================================
# 7. Example
# ============================================================

if __name__ == "__main__":
    report = asyncio.run(
        analyze_flight_log_v1(
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
