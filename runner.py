from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel
from agents import Agent, Runner, function_tool, RunContextWrapper, WebSearchTool

from px4_source import checkout_px4_source_revision, read_source_file, search_source
from mission_parser import parse_mission_file as parse_mission_file_impl
from ulog_control_surface import infer_control_surface as infer_control_surface_impl
from ulog_inventory import parse_ulog_inventory as parse_ulog_inventory_impl
from ulog_metrics import compute_log_metrics as compute_log_metrics_impl
from ulog_plots import generate_signal_plot as generate_signal_plot_impl
from ulog_timeline import build_basic_timeline as build_basic_timeline_impl


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

class PlotRef(BaseModel):
    title: str
    path: str
    purpose: str


class PlotOverlay(BaseModel):
    start_s: float
    end_s: Optional[float] = None
    label: Optional[str] = None
    color: Optional[str] = None
    alpha: Optional[float] = None


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
3. Use web search only for culprit discovery:
   PX4 docs, forum posts, GitHub issues, parameter concepts, known mechanisms.
4. If a relevant PX4 git hash, tag, or branch is known, checkout the local
   PX4 source tree before source-code investigation.
5. Use local PX4 source search for exact code behavior.
6. Select relevant analysis windows.
7. Generate ranked hypotheses.
8. For each hypothesis, include:
   - mechanism
   - evidence from log
   - contradicting evidence
   - relevant plot
   - relevant code path
   - confidence
9. Do not provide parameter tuning, code-change, or flight-test suggestions
   unless the user explicitly asks for suggestions or fixes.

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
) -> FlightLogReport:

    log_path = Path(log_path)
    mission_path_obj = Path(mission_path) if mission_path else None
    source_path_obj = Path(source_path) if source_path else None
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    # Deterministic pre-pass
    inventory = parse_ulog_inventory(log_path)
    timeline = build_basic_timeline(log_path)
    assumptions = infer_control_surface(log_path, source_path_obj)
    mission = parse_mission_file(mission_path_obj)

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

    result = await Runner.run(
        flight_log_agent,
        input=json.dumps(agent_input, indent=2),
        context=ctx,
        max_turns=20,
    )

    return result.final_output


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
