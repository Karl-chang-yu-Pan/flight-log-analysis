from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel
from agents import Agent, Runner, function_tool, RunContextWrapper, WebSearchTool


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
    TODO:
    Implement with pyulog.

    Should extract:
    - firmware version / git hash
    - parameters
    - available uORB topics
    - warnings/errors
    - duration
    """
    return {
        "firmware_version": "TODO_FROM_LOG",
        "git_hash": None,
        "duration_s": None,
        "important_parameters": {},
        "available_topics": [],
        "warnings": [],
        "missing_topics": [],
    }


def build_basic_timeline(log_path: Path) -> list[dict]:
    """
    TODO:
    Build from vehicle_status, vehicle_type, nav_state,
    arming_state, vtol_vehicle_status, mission_result.
    """
    return []


def infer_control_mapping(log_path: Path, source_path: Optional[Path]) -> dict:
    """
    TODO:
    Infer from CA_* params, PWM_MAIN_FUNCx / AUX_FUNCx,
    actuator_motors, actuator_servos, airframe config.
    """
    return {
        "vehicle_type": "unknown",
        "assumed_actuator_mapping": {},
        "evidence": [],
        "confidence": "low",
        "warning": "Mapping is not confirmed yet.",
    }


def parse_mission_file(mission_path: Optional[Path]) -> Optional[dict]:
    """
    TODO:
    Parse .plan or mission file if provided.
    """
    if mission_path is None:
        return None

    return {
        "mission_file": str(mission_path),
        "items": [],
    }


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

    cmd = [
        "rg",
        "-n",
        "--context", "3",
        query,
        str(source_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as e:
        return [{"error": str(e)}]

    lines = result.stdout.splitlines()[:max_results * 8]

    return [{
        "query": query,
        "matches": lines,
    }]


@function_tool
def compute_log_metrics(
    ctx: RunContextWrapper[FlightLogContext],
    start_s: float,
    end_s: float,
    signals: list[str],
) -> dict:
    """
    Compute numeric metrics for selected signals.

    TODO:
    Implement actual signal extraction from pyulog/pandas cache.
    """
    return {
        "window_s": [start_s, end_s],
        "signals": signals,
        "metrics": {
            "TODO": "replace with real metrics",
        },
    }


@function_tool
def generate_signal_plot(
    ctx: RunContextWrapper[FlightLogContext],
    title: str,
    start_s: float,
    end_s: float,
    signals: list[str],
    purpose: str,
) -> dict:
    """
    Generate a hypothesis-specific plot.

    TODO:
    Implement actual matplotlib plotting.
    """
    plots_dir = ctx.context.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    plot_path = plots_dir / f"{safe_title}.png"

    # TODO: actual plotting code here

    return {
        "title": title,
        "path": str(plot_path),
        "purpose": purpose,
        "window_s": [start_s, end_s],
        "signals": signals,
    }


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
4. Use local PX4 source search for exact code behavior.
5. Select relevant analysis windows.
6. Generate ranked hypotheses.
7. For each hypothesis, include:
   - mechanism
   - evidence from log
   - contradicting evidence
   - relevant plot
   - relevant code path
   - confidence
8. Do not provide parameter tuning, code-change, or flight-test suggestions
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
    assumptions = infer_control_mapping(log_path, source_path_obj)
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
