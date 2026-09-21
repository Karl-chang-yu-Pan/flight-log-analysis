"""Pure temporal diagnostic-window derivation and candidate selection.

W3A shared capability (ADR-0004, temporal_causal_qualification_spec):
deterministic transition-event matching over logged samples, diagnostic
window construction, and temporal eligibility/discrimination over
represented candidates. No I/O, no LLM, no source interpretation, no
proof/confidence semantics. All inputs are plain data; all outputs are
plain data consumed by the report-adjacent selector in dag_pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from flight_log_agent.utils import json_safe_value as _json_safe_value


def _norm_logged_value(value: Any) -> Any:
    """Normalize a logged value for exact transition comparison.

    Numpy scalars unwrap via ``json_safe_value``; numbers compare by
    float value so spec ``15`` matches logged ``15.0``; booleans and
    strings keep strict identity.
    """
    cleaned = _json_safe_value(value)
    if isinstance(cleaned, bool):
        return cleaned
    if isinstance(cleaned, (int, float)):
        return float(cleaned)
    return cleaned


def _spec_value(value: Any) -> Any:
    """Normalize a spec from/to value with the same rules as samples."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _logged_values_equal(first: Any, second: Any) -> bool:
    """NaN-aware equality for the any-change transition check.

    Python's ``nan != nan`` would otherwise invent a transition for
    repeated missing/non-finite observations. NaN therefore equals
    NaN here (no change); every other value keeps default equality,
    so infinities (``inf == inf``) and all existing coercions are
    untouched. Concrete from/to spec matching is intentionally not
    routed through this helper: NaN must never satisfy a requested
    spec value, and that path already fails closed via ``!=``.
    """
    if (
        isinstance(first, float)
        and isinstance(second, float)
        and math.isnan(first)
        and math.isnan(second)
    ):
        return True
    return bool(first == second)


def derive_transition_events(
    spec: Any,
    samples: Mapping[str, Sequence[tuple[float, Any]]],
) -> list[dict[str, Any]]:
    """Match a transition-event specification against logged samples.

    Returns every matching event as ``{"time", "from_value",
    "to_value"}`` in ascending timestamp order, or ``[]``. An event
    is a consecutive sample pair whose previous value satisfies
    ``from_value`` (or any value when absent) and whose next value
    satisfies ``to_value`` (or any *change* when absent).
    Deterministic and source-order independent.
    """
    signal = str(getattr(spec, "transition_signal", None) or "")
    series = samples.get(signal) if isinstance(samples, Mapping) else None
    if not signal or not series:
        return []
    ordered = sorted(
        ((float(time), _norm_logged_value(value)) for time, value in series),
        key=lambda item: item[0],
    )
    want_from = _spec_value(getattr(spec, "from_value", None))
    want_to = _spec_value(getattr(spec, "to_value", None))
    events: list[dict[str, Any]] = []
    for index in range(1, len(ordered)):
        previous_time, previous_value = ordered[index - 1]
        time, value = ordered[index]
        if want_from is not None and previous_value != want_from:
            continue
        if want_to is not None:
            if value != want_to:
                continue
        elif _logged_values_equal(value, previous_value):
            continue
        events.append(
            {"time": time, "from_value": previous_value, "to_value": value}
        )
    return events


def select_transition_event(
    events: Sequence[Mapping[str, Any]],
    event_selection: Optional[str],
) -> Optional[dict[str, Any]]:
    """Choose one derived event per explicit seed disambiguation.

    Zero events → None. One event → that event. Multiple events →
    ``"first"``/``"last"`` chooses explicitly; without selection the
    result is None (unresolved, never a silent default).
    """
    ordered = sorted(
        (dict(event) for event in events or ()),
        key=lambda event: float(event.get("time", 0.0)),
    )
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    if event_selection == "first":
        return ordered[0]
    if event_selection == "last":
        return ordered[-1]
    return None


def _first_sample_at_or_after(
    samples: Mapping[str, Sequence[tuple[float, Any]]],
    signal: str,
    event_time: float,
) -> Optional[float]:
    """First prepared sample timestamp of ``signal`` at/after the event.

    Closed comparison on exact sample timestamps; no epsilon widening.
    """
    series = samples.get(signal) if isinstance(samples, Mapping) else None
    if not series:
        return None
    after = [
        float(time) for time, _value in series if float(time) >= event_time
    ]
    return min(after) if after else None


def derive_diagnostic_windows(
    spec: Any,
    event: Optional[Mapping[str, Any]],
    samples: Mapping[str, Sequence[tuple[float, Any]]],
) -> list[tuple[float, float]]:
    """Build concrete diagnostic windows from a matched event.

    Duration forms span the event time; ``first_sample_of`` spans
    event time through the first target sample at/after the event
    (a degenerate point window stays valid for selection overlap).
    No event, no target samples, or no sample at/after the event →
    ``[]`` (fail closed downstream).
    """
    if not isinstance(event, Mapping):
        return []
    event_time = float(event.get("time", 0.0))
    relation = str(getattr(spec, "relation", None) or "")
    duration = getattr(spec, "duration_s", None)
    target = getattr(spec, "first_sample_of", None)
    if target:
        first = _first_sample_at_or_after(samples, str(target), event_time)
        if first is None:
            return []
        return [(event_time, first)]
    if duration is None or not float(duration) > 0:
        return []
    span = float(duration)
    if relation == "after":
        return [(event_time, event_time + span)]
    if relation == "before":
        return [(event_time - span, event_time)]
    if relation == "around":
        return [(event_time - span, event_time + span)]
    return []


def _intersect_window_lists(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """Closed-interval intersections, deterministically ordered."""
    intersections: list[tuple[float, float]] = []
    for first_start, first_end in first:
        for second_start, second_end in second:
            start = max(float(first_start), float(second_start))
            end = min(float(first_end), float(second_end))
            if start <= end:
                intersections.append((start, end))
    return sorted(intersections)


def intersect_diagnostic_windows(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    """Intersect two diagnostic window lists (e.g. predicate scope
    with transition-derived windows). Public for stage composition;
    inputs stay unmerged, output is deterministically ordered."""
    return _intersect_window_lists(first, second)


def control_gate_windows_for(dag: Any, op_id: Any) -> list[tuple[str, list]]:
    """Gating-branch (verdict, windows) pairs for one operation.

    Reads incoming control edges into branch vertices, ordered by
    branch id for determinism. Non-branch or missing sources are
    skipped; an operation with no gates yields ``[]``.
    """
    vertices = list((getattr(dag, "vertices", None) if dag is not None else None) or ())
    edges = list((getattr(dag, "edges", None) if dag is not None else None) or ())
    by_id = {getattr(item, "id", None): item for item in vertices}
    gates: list[tuple[str, list]] = []
    for edge in edges:
        if getattr(edge, "target_id", None) != op_id:
            continue
        if getattr(edge, "kind", None) != "control":
            continue
        branch = by_id.get(getattr(edge, "source_id", None))
        if branch is None or getattr(branch, "kind", None) != "branch":
            continue
        verdict = str(getattr(branch, "feasibility_verdict", None) or "unknown")
        windows = [
            (float(start), float(end))
            for start, end in (getattr(branch, "active_windows", None) or ())
        ]
        gates.append((str(getattr(branch, "id", None) or ""), verdict, windows))
    gates.sort(key=lambda item: item[0])
    return [(verdict, windows) for _, verdict, windows in gates]


def _windows_overlap(
    domain: Sequence[Sequence[float]],
    diagnostic: Sequence[Sequence[float]],
) -> bool:
    """Closed-interval overlap between a writer domain and a window."""
    return bool(_intersect_window_lists(domain, diagnostic))


def writer_domain(
    *,
    replay_domain: Optional[Sequence[Sequence[float]]] = None,
    branch_gates: Sequence[tuple[str, Sequence[Sequence[float]]]] = (),
    scope_windows: Optional[Sequence[Sequence[float]]] = None,
) -> list[tuple[float, float]]:
    """Compose one writer's temporal domain from existing structures.

    Priority: replay-result ``active_windows`` for the writer op when
    available (already comparison-intersected by replay
    construction); otherwise gating control-branch windows
    intersected with the scope span — ``always_true`` contributes
    the scope span, ``always_false`` empties the domain, and
    ``unknown`` branches without windows constrain nothing (missing
    feasibility information never excludes). No new window model.
    """
    if replay_domain is not None:
        return sorted(
            (float(start), float(end)) for start, end in replay_domain
        )
    domain = [
        (float(start), float(end)) for start, end in (scope_windows or ())
    ]
    for verdict, windows in branch_gates:
        if verdict == "always_false":
            return []
        if verdict == "always_true":
            continue
        if list(windows or ()):
            domain = _intersect_window_lists(domain, list(windows))
            if not domain:
                return []
    return domain


def temporally_eligible(
    domain: Sequence[Sequence[float]],
    diagnostic_windows: Sequence[Sequence[float]],
) -> bool:
    """Whether a writer domain overlaps the diagnostic window.

    Overlap means the candidate could be relevant during the
    questioned interval; it never means unique causation.
    """
    return _windows_overlap(domain, diagnostic_windows)


def discriminate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    diagnostic_windows: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Narrow temporally eligible candidates toward unique selection.

    Returns ``{"eligible": [...], "unique": key | None,
    "unresolved": str | None}`` with deterministically ordered keys.
    Unique selection needs real discrimination: a sole overlapping
    domain, or exactly one eligible candidate with in-window replay
    support (a caller-supplied flag from already-linked per-writer
    replay results — never invented here). Ties and empty sets fail
    closed with an unresolved note; source order never decides.
    """
    diagnostic = [
        (float(start), float(end)) for start, end in diagnostic_windows
    ]
    eligible = sorted(
        str(candidate.get("key"))
        for candidate in candidates or ()
        if _windows_overlap(candidate.get("domain") or (), diagnostic)
    )
    if len(eligible) == 1:
        return {"eligible": eligible, "unique": eligible[0], "unresolved": None}
    if not eligible:
        return {
            "eligible": [],
            "unique": None,
            "unresolved": (
                "no represented writer domain overlaps the diagnostic "
                "window; no temporally-selected causal claim"
            ),
        }
    supported = sorted(
        str(candidate.get("key"))
        for candidate in candidates or ()
        if str(candidate.get("key")) in eligible
        and candidate.get("replay_match") is True
    )
    if len(supported) == 1:
        return {
            "eligible": eligible,
            "unique": supported[0],
            "unresolved": None,
        }
    return {
        "eligible": eligible,
        "unique": None,
        "unresolved": (
            "multiple temporally eligible writers remain "
            "indistinguishable in the diagnostic window: "
            + ", ".join(eligible)
        ),
    }
