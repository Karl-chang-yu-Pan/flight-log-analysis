from __future__ import annotations

from typing import Any, Optional

from flight_log_agent.models import ApplicabilityResult, MechanismCandidate, WindowSpec


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

    candidate_windows = derive_candidate_windows(candidate, timeline, mission)
    if not candidate_windows:
        unresolved.append("No candidate verification window could be derived from timeline/mission.")

    return ApplicabilityResult(
        candidate_name=candidate.name,
        applicable=len(excluded) == 0,
        supported_conditions=supported,
        excluded_by=excluded,
        unresolved_conditions=unresolved,
        relevant_parameters=relevant_parameters,
        candidate_windows=candidate_windows,
        available_required_signals=available_required_signals,
        missing_required_signals=missing_required_signals,
    )


def derive_candidate_windows(
    candidate: MechanismCandidate,
    timeline: list[dict],
    mission: Optional[dict],
) -> list[WindowSpec]:
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
