from __future__ import annotations

from typing import Any, Optional

from flight_log_agent.models import ApplicabilityResult, MechanismCandidate, WindowSpec
from flight_log_agent.px4.msg_schema import load_px4_msg_schema


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
    params = inventory.get("parameters") or {}
    topic_fields = inventory.get("topic_fields") or {}
    schema_topic_fields = load_px4_msg_schema(inventory.get("source_path"))
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
        schema_topic_fields,
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
    if windows:
        return windows

    span = timeline_time_span(timeline)
    if span is None:
        return windows

    start_s, end_s = span
    for name in check_window_names(candidate):
        windows.append(
            WindowSpec(
                name=name,
                start_s=start_s,
                end_s=end_s,
                reason=f"Fallback verification window from full timeline span for check window '{name}'.",
            )
        )
    return windows


def check_window_names(candidate: MechanismCandidate) -> list[str]:
    names = []
    for check in list(candidate.numeric_checks or []) + list(candidate.exclusion_checks or []):
        window = getattr(check, "window", None)
        if window and window not in names:
            names.append(str(window))
    return names


def timeline_time_span(timeline: list[dict]) -> Optional[tuple[float, float]]:
    times = []
    for event in timeline or []:
        time_s = event.get("time_s") if isinstance(event, dict) else None
        if isinstance(time_s, (int, float)):
            times.append(float(time_s))
    if not times:
        return None
    start_s = min(times)
    end_s = max(times)
    if end_s < start_s:
        return None
    if end_s == start_s:
        end_s = start_s + 0.001
    return start_s, end_s


def check_required_signals(
    required_signals: list[str],
    available_topics: set[str],
    topic_fields: dict[str, Any],
    schema_topic_fields: Optional[dict[str, list[str]]] = None,
) -> tuple[list[str], list[str]]:
    available: list[str] = []
    missing: list[str] = []

    for signal in required_signals:
        if "." not in signal:
            if signal in available_topics:
                available.append(signal)
            else:
                missing.append(signal)
            continue

        topic, field = signal.split(".", 1)
        fields = topic_fields.get(topic)
        schema_fields = (schema_topic_fields or {}).get(topic)
        field_known = (
            topic not in topic_fields
            or field in (fields or [])
            or field in (schema_fields or [])
        )
        if topic in available_topics and field_known:
            available.append(signal)
        else:
            missing.append(signal)

    return available, missing
