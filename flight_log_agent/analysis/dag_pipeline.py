"""Flag-gated DAG discovery pipeline (#73 Stage 4).

Replaces runner Stages 3–5 when enabled: Layer 4 intent cache instead of
``mechanism_cache`` retrieval, :func:`discover_with_judge` instead of the
SourceDiscoveryDecision loop, and DAG ``evaluate_feasibility`` over ULog
parameters/samples instead of the verification-plan checks. The final
report is constructed deterministically from the verdict and the
annotated DAG — the legacy report agent keeps its VerifiedMechanismResult
contract untouched until the retirement stage rewrites it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

from flight_log_agent.analysis.log_evidence import ULogEvidenceIndex
from flight_log_agent.analysis.mechanism_dag import (
    MechanismDAG,
    evaluate_feasibility,
    layer2_cache_path,
    layer3_cache_path,
    write_dag_to_cache,
)
from flight_log_agent.analysis.mechanism_discovery import DiscoveryResult
from flight_log_agent.analysis.mechanism_judge import (
    DiscoverySeeds,
    JudgedDiscovery,
    SEEDER_ADAPTER_VERSION,
    discover_with_judge,
    render_discovery_compact,
)
from flight_log_agent.models import (
    ApplicabilityReport,
    CodeRef,
    ExpectedSignatureItem,
    FlightLogReport,
    HypothesisReportItem,
    ParameterValue,
    RelationshipCheckSpec,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.symbols import parse_signal_reference


# ---------------------------------------------------------------------------
# Layer 4: question → seeds/terminal cache (the mechanism_cache fold)
# ---------------------------------------------------------------------------


def _question_slug(question: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    stem = "_".join(tokens)[:60] or "question"
    digest = hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:12]
    return f"{stem}.{digest}"


def seeder_fingerprint() -> str:
    """Content fingerprint of what produces a Layer 4 entry: the seeder's
    instructions and output schema. Prompt edits invalidate cached seeds
    automatically — no manual cache clearing."""
    from flight_log_agent.analysis.mechanism_judge import seeder_agent

    import json as _json

    payload = SEEDER_ADAPTER_VERSION + seeder_agent.instructions + _json.dumps(
        DiscoverySeeds.model_json_schema(), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


_PRUNED_INTENT_ROOTS: set[tuple[str, str]] = set()


def _prune_stale_intent(cache_root: Path, current: str) -> None:
    key = (str(cache_root), current)
    if key in _PRUNED_INTENT_ROOTS:
        return
    _PRUNED_INTENT_ROOTS.add(key)
    intent_dir = Path(cache_root) / "intent"
    if not intent_dir.is_dir():
        return
    import shutil

    for entry in intent_dir.iterdir():
        if entry.is_dir() and entry.name != current:
            shutil.rmtree(entry, ignore_errors=True)


def layer4_cache_path(cache_root: Union[str, Path], question: str) -> Path:
    """Layer 4 path: ``{cache_root}/intent/{seeder_fp}/{slug}.json``."""
    return (
        Path(cache_root)
        / "intent"
        / seeder_fingerprint()
        / f"{_question_slug(question)}.json"
    )


def write_seeds_to_cache(seeds: DiscoverySeeds, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(seeds.model_dump_json(), encoding="utf-8")
    tmp.replace(path)


def read_seeds_from_cache(path: Path) -> Optional[DiscoverySeeds]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return DiscoverySeeds.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


@dataclass
class DagStageResult:
    judged: JudgedDiscovery
    annotated_dag: Optional[MechanismDAG]
    render: dict[str, Any]
    layer4_hit: bool
    report: FlightLogReport
    replay: Optional[dict[str, Any]] = None


def _signal_samples_for_dag(
    dag: MechanismDAG, log_path: Path
) -> dict[str, list[tuple[float, Any]]]:
    """Load ULog samples for every logged-signal evidence leaf in ``dag``."""
    references = sorted(
        {
            str(v.signal_name)
            for v in dag.vertices
            if v.kind == "evidence" and v.sub_kind == "logged_signal" and v.signal_name
        }
    )
    if not references:
        return {}
    index = ULogEvidenceIndex.from_path(log_path, references)
    samples: dict[str, list[tuple[float, Any]]] = {}
    for reference in references:
        resolution = index.resolve_signal(reference)
        if resolution.status == "observed" and resolution.series is not None:
            samples[reference] = [
                (sample.time_s, sample.value)
                for sample in resolution.series.samples
            ]
    return samples


def resolve_questioned_signal(
    hint: str,
    logged_set: set[str],
    dags: list[Optional[MechanismDAG]],
) -> tuple[Optional[str], Optional[str], list[str]]:
    """Resolve the seeder's signal hint against the log catalogue and the
    discovered slices — never by string fuzz.

    Exact topic/field identity wins; an omitted topic instance resolves only
    when one observed placement matches. Otherwise the hint's topic is
    intersected with the fields the candidate DAGs actually
    connect to (their logged-evidence leaves): the publish-site facts in
    the slice carry the true field name even when the hinted field was
    renamed across versions. Unique intersection resolves; anything else
    returns an honest error plus the candidates.
    """
    hint = str(hint or "").strip()
    if not hint:
        return None, "empty signal hint", []
    if hint in logged_set:
        return hint, None, []
    parsed_hint = parse_signal_reference(hint)
    if parsed_hint is None:
        return None, f"signal hint {hint!r} is not a canonical signal reference", []
    topic, requested_instance, field = parsed_hint

    def compatible_placement(signal: str, *, require_field: bool) -> bool:
        parsed = parse_signal_reference(signal)
        if parsed is None:
            return False
        candidate_topic, candidate_instance, candidate_field = parsed
        return (
            candidate_topic == topic
            and (not require_field or candidate_field == field)
            and (
                requested_instance is None
                or candidate_instance == requested_instance
            )
        )

    exact_placements = sorted(
        signal
        for signal in logged_set
        if compatible_placement(signal, require_field=True)
    )
    if len(exact_placements) == 1:
        return exact_placements[0], None, []
    if len(exact_placements) > 1:
        return (
            None,
            f"hint {hint!r} is ambiguous across observed topic instances",
            exact_placements,
        )

    connected = {
        str(v.signal_name)
        for dag in dags
        if dag is not None
        for v in dag.vertices
        if v.kind == "evidence"
        and v.sub_kind == "logged_signal"
        and (v.metadata or {}).get("observation") == "observed"
        and compatible_placement(str(v.signal_name or ""), require_field=False)
    }
    if len(connected) == 1:
        return next(iter(connected)), None, []
    if connected:
        return None, f"hint {hint!r} is ambiguous in the slice", sorted(connected)
    # Slice carries no leaf for the hinted topic. Substring containment
    # against the schema is name guessing, not resolution — fields on
    # the topic are returned only as CANDIDATES for the judge, never
    # auto-picked, however few there are.
    candidates = sorted(
        signal
        for signal in logged_set
        if compatible_placement(signal, require_field=False)
    )
    if candidates:
        return None, f"hint {hint!r} did not resolve exactly", candidates
    return None, f"signal hint {hint!r} did not resolve", []


# The five distinct replay states (core correctness invariant):
# ``matched`` is supporting evidence and ``mismatched`` contradictory
# evidence ONLY when replay completeness (writer reachability domains,
# ordering, retained state, alignment, coverage) is trustworthy.
# ``partial`` and ``unevaluable`` are unresolved evidence;
# ``not_attempted`` is neutral.
ReplayStatus = Literal[
    "not_attempted", "unevaluable", "partial", "matched", "mismatched"
]


def _intersect_windows(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    for first_start, first_end in first:
        for second_start, second_end in second:
            start = max(first_start, second_start)
            end = min(first_end, second_end)
            if start <= end:
                intersections.append((start, end))
    return _merge_windows(intersections)


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(windows):
        if end < start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _window_duration(windows: list[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in _merge_windows(windows))


def _replay_tolerance(
    samples: list[tuple[float, Any]],
    policy: Optional[Any],
) -> float:
    """Derive a numerical tolerance from data scale and declared policy."""
    values = [
        float(value)
        for _timestamp, value in samples
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not values:
        return 1e-9
    if policy is not None and hasattr(policy, "model_dump"):
        policy = policy.model_dump()
    if str((policy or {}).get("method") or "") == "discrete_hold":
        return 1e-9
    ordered = sorted(abs(value) for value in values)
    scale = ordered[len(ordered) // 2]
    sorted_values = sorted(values)
    median_value = sorted_values[len(sorted_values) // 2]
    deviations = sorted(abs(value - median_value) for value in values)
    robust_variation = deviations[len(deviations) // 2]
    return max(1e-6, 1e-3 * max(scale, robust_variation))


def replay_terminal_expressions(
    annotated: MechanismDAG,
    log_path: Path,
    parameter_values: dict[str, Any],
    logged_set: set[str],
    observed_hint: Optional[str] = None,
    signal_policies: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Numerically compare the mechanism against the log: ground each
    terminal write's expression through the graph, evaluate it over the
    log, and measure how long it matches the observed signal within
    tolerance.

    Fully structural: grounding walks DAG edges, the observed signal must
    have exact terminal-publication provenance, and each writer is compared
    only inside the intersection of its gating branch windows. A definitive
    match/mismatch is emitted only when exact writer domains are non-
    overlapping and cover the observed output domain.
    """
    from flight_log_agent.analysis.mechanism_dag import (
        _evaluate_predicate_intervals,
        _predicate_policies_complete,
        ground_expression_via_edges,
    )

    def not_attempted(reason: str) -> dict[str, Any]:
        return {"status": "not_attempted", "complete": False, "reason": reason}

    terminal_ops = [
        v
        for v in annotated.vertices
        if v.kind == "operation" and (v.metadata or {}).get("is_terminal")
    ]
    if not terminal_ops:
        return not_attempted("no terminal writes in the graph")
    published = {
        str((op.metadata or {}).get("logged_signal") or "")
        for op in terminal_ops
        if (op.metadata or {}).get("logged_signal")
    }
    terminal = str(annotated.terminal or "")
    if len(published) == 1:
        observed = next(iter(published))
    elif terminal in logged_set:
        observed = terminal
    elif observed_hint and observed_hint in published:
        observed = observed_hint
    else:
        return not_attempted(
            "observed output lacks unique exact terminal-publication provenance"
        )
    if observed not in logged_set:
        return not_attempted(f"terminal output {observed!r} is not in the observed catalogue")
    index = ULogEvidenceIndex.from_path(log_path, [observed])
    resolution = index.resolve_signal(observed)
    if resolution.status != "observed" or resolution.series is None:
        return not_attempted(f"{observed} not observed in this log")
    observed_samples = [(s.time_s, s.value) for s in resolution.series.samples]
    if len(observed_samples) < 2:
        return not_attempted(f"{observed} has too few samples")
    samples = dict(_signal_samples_for_dag(annotated, log_path))
    samples[observed] = observed_samples
    observed_policy = (signal_policies or {}).get(observed)
    tolerance = _replay_tolerance(observed_samples, observed_policy)
    observed_span = (observed_samples[0][0], observed_samples[-1][0])
    span_duration = observed_span[1] - observed_span[0]

    vertices = {vertex.id: vertex for vertex in annotated.vertices}
    controls_by_op: dict[str, list[str]] = {}
    for edge in annotated.edges:
        if edge.kind == "control":
            controls_by_op.setdefault(edge.target_id, []).append(edge.source_id)

    results: list[dict[str, Any]] = []
    writer_domains: list[tuple[str, list[tuple[float, float]]]] = []
    complete = True
    for op in terminal_ops:
        reachability = (op.metadata or {}).get("reachability") or {}
        domain = [observed_span]
        writer_complete = bool(reachability.get("exact", False))
        for branch_id in controls_by_op.get(op.id, []):
            branch = vertices.get(branch_id)
            if branch is None or branch.kind != "branch":
                writer_complete = False
                continue
            if branch.feasibility_verdict == "always_false":
                domain = []
            elif branch.feasibility_verdict == "always_true":
                branch_domain = [observed_span]
            elif branch.active_windows:
                branch_domain = list(branch.active_windows)
                policies_used = (branch.metadata or {}).get("sampling_policies") or {}
                if not policies_used or "unknown" in policies_used.values():
                    writer_complete = False
            else:
                branch_domain = []
                writer_complete = False
            if domain:
                domain = _intersect_windows(domain, branch_domain)

        grounded = ground_expression_via_edges(
            str(op.expression or ""), op.id, annotated
        )
        if not grounded:
            results.append(
                {
                    "operation_id": op.id,
                    "expression": op.expression,
                    "grounded": None,
                    "evaluable": False,
                    "active_windows": domain,
                }
            )
            complete = False
            continue
        match_windows = _evaluate_predicate_intervals(
            f"abs(({grounded}) - ({observed})) <= {tolerance}",
            parameter_values,
            {},
            samples,
            signal_policies,
        )
        if match_windows is None:
            results.append(
                {
                    "operation_id": op.id,
                    "expression": op.expression,
                    "grounded": grounded,
                    "evaluable": False,
                    "active_windows": domain,
                }
            )
            complete = False
            continue
        if not _predicate_policies_complete(
            f"abs(({grounded}) - ({observed})) <= {tolerance}",
            samples,
            signal_policies or {},
        ):
            writer_complete = False
        active_duration = _window_duration(domain)
        matched_duration = _window_duration(_intersect_windows(match_windows, domain))
        results.append(
            {
                "operation_id": op.id,
                "expression": op.expression,
                "grounded": grounded,
                "evaluable": True,
                "active_windows": domain,
                "active_duration": round(active_duration, 6),
                "matched_duration": round(matched_duration, 6),
                "match_fraction": (
                    round(matched_duration / active_duration, 3)
                    if active_duration
                    else None
                ),
            }
        )
        writer_domains.append((op.id, domain))
        complete = complete and writer_complete

    merged_domain = _merge_windows(
        [window for _operation_id, domain in writer_domains for window in domain]
    )
    summed_writer_duration = sum(
        _window_duration(domain) for _operation_id, domain in writer_domains
    )
    covered_duration = _window_duration(merged_domain)
    non_overlapping = abs(summed_writer_duration - covered_duration) <= 1e-6
    covers_output = (
        len(merged_domain) == 1
        and merged_domain[0][0] <= observed_span[0]
        and merged_domain[0][1] >= observed_span[1]
    )
    complete = complete and non_overlapping and covers_output and span_duration > 0

    if not results:
        status: ReplayStatus = "unevaluable"
        reason = "no terminal expression grounded through the graph"
    elif not any(r.get("evaluable") for r in results):
        status = "unevaluable"
        reason = "no grounded expression was evaluable over the log"
    elif not complete:
        status = "partial"
        reason = (
            "writer reachability, policy, non-overlap, or output-domain "
            "coverage is incomplete"
        )
    else:
        matched_duration = sum(
            float(result.get("matched_duration") or 0.0)
            for result in results
            if result.get("evaluable")
        )
        match_fraction = matched_duration / covered_duration if covered_duration else 0.0
        status = "matched" if match_fraction >= 0.95 else "mismatched"
        reason = f"complete piecewise replay match fraction {match_fraction:.3f}"
    return {
        "status": status,
        "complete": complete,
        "reason": reason,
        "observed": observed,
        "tolerance": round(tolerance, 6),
        "results": results,
    }


def build_report_from_dag(
    question: str,
    judged: JudgedDiscovery,
    annotated_dag: Optional[MechanismDAG],
    replay: Optional[dict[str, Any]] = None,
) -> FlightLogReport:
    """Deterministic FlightLogReport from the verdict + annotated DAG.

    Complete deterministic replay can establish ``high`` confidence;
    partial numeric replay can establish at most ``medium``. Structural
    evidence alone remains low. Branch feasibility maps to applicability
    (``always_true`` -> supported, ``always_false`` -> excluded,
    ``unknown`` -> unresolved).
    """
    verdict = judged.verdict
    dag = annotated_dag or (judged.selected.dag if judged.selected else None)
    if dag is None or not dag.vertices:
        validation = (
            getattr(judged.selected, "terminal_validation", None)
            if judged.selected is not None
            else None
        )
        reason = str(
            getattr(validation, "reason", "")
            or verdict.reasoning
            or "no validated mechanism DAG was produced"
        )
        return FlightLogReport(
            airframe_summary="",
            question_intent_summary=question,
            ranked_hypotheses=[],
            excluded_mechanisms=[],
            confirmed=[],
            unconfirmed=[],
            final_summary=f"Mechanism discovery could not produce a report: {reason}",
        )

    supported: list[str] = []
    excluded: list[str] = []
    unresolved_conditions: list[str] = []
    parameters: list[ParameterValue] = []
    source_refs: list[CodeRef] = []
    signature: list[ExpectedSignatureItem] = []

    for vertex in dag.vertices if dag else []:
        if vertex.kind == "branch":
            predicate = vertex.predicate_lowered or vertex.predicate_raw or ""
            if vertex.feasibility_verdict == "always_true":
                supported.append(predicate)
            elif vertex.feasibility_verdict == "always_false":
                excluded.append(predicate)
            else:
                unresolved_conditions.append(predicate)
        elif vertex.kind == "evidence" and vertex.sub_kind == "parameter":
            value = (vertex.metadata or {}).get("value")
            parameters.append(
                ParameterValue(name=str(vertex.signal_name), value=str(value))
            )
        elif vertex.kind == "evidence" and vertex.sub_kind == "logged_signal":
            if (vertex.metadata or {}).get("observation") == "observed":
                signature.append(
                    ExpectedSignatureItem(
                        name=str(vertex.signal_name),
                        description="observed signal grounding the mechanism slice",
                        signal=str(vertex.signal_name),
                    )
                )
        elif vertex.kind == "operation" and vertex.file and vertex.metadata.get("is_terminal"):
            logged_output = str((vertex.metadata or {}).get("logged_signal") or "")
            if (
                logged_output
                and (vertex.metadata or {}).get("logged_observation") == "observed"
                and all(item.signal != logged_output for item in signature)
            ):
                signature.append(
                    ExpectedSignatureItem(
                        name=logged_output,
                        description="observed output written by the terminal operation",
                        signal=logged_output,
                    )
                )
            source_refs.append(
                CodeRef(
                    file=str(vertex.file),
                    start_line=vertex.line,
                    end_line=vertex.line,
                    snippet=vertex.snippet,
                    explanation=f"terminal write: {vertex.variable} <- {vertex.expression}",
                )
            )

    if not signature:
        return FlightLogReport(
            airframe_summary="",
            question_intent_summary=question,
            ranked_hypotheses=[],
            excluded_mechanisms=[],
            confirmed=[],
            unconfirmed=[],
            final_summary=(
                "Mechanism discovery produced source structure but no observed "
                "signal grounding for this flight."
            ),
        )

    # Cross-check the judge's claimed mechanism against flight-data
    # feasibility: a sufficient verdict must name explaining branches,
    # and at least one must exist in the DAG without being feasibility-
    # dead. Otherwise the confirmation is downgraded — the mechanism may
    # be real code, but nothing shows it fired in THIS flight.
    unresolved_evidence = list(dag.unresolved_symbols) if dag else []
    branches_verified = False
    if verdict.sufficient and verdict.explaining_branches and dag is not None:
        dag_branches = {
            vertex.id: vertex
            for vertex in dag.vertices
            if vertex.kind == "branch"
        }
        branches_verified = all(
            branch_id in dag_branches
            and dag_branches[branch_id].feasibility_verdict != "always_false"
            for branch_id in verdict.explaining_branches
        )
        if not branches_verified:
            unresolved_evidence.insert(
                0,
                "judge-named explaining branch ID(s) absent from the DAG or "
                "feasibility-dead: " + "; ".join(verdict.explaining_branches[:3]),
            )
    elif verdict.sufficient:
        unresolved_evidence.insert(
            0, "judge confirmed the mechanism without naming an explaining branch"
        )

    replay_status = str((replay or {}).get("status") or "not_attempted")
    replay_mismatch = replay_status == "mismatched"
    has_numeric_replay = any(
        result.get("evaluable") for result in (replay or {}).get("results", [])
    )
    if replay_mismatch:
        confidence = "unresolved"
    elif (
        verdict.sufficient
        and branches_verified
        and replay
        and replay.get("status") == "matched"
    ):
        # Structure confirmed AND a COMPLETE replay numerically
        # reproduces the observed signal — the honest "high". A
        # ``partial`` replay is unresolved evidence and never upgrades
        # this path to high.
        confidence = "high"
    elif verdict.sufficient and branches_verified and has_numeric_replay:
        confidence = "medium"
    elif verdict.sufficient:
        confidence = "low"
    else:
        confidence = "unresolved"

    replay_checks = [
        RelationshipCheckSpec(
            type="derived_expression",
            actual=str(replay.get("observed") or ""),
            expression=str(result.get("grounded") or ""),
            metric="match_fraction",
            value=result.get("match_fraction"),
            description=f"DAG replay status: {replay_status}",
        )
        for result in (replay or {}).get("results", [])
        if result.get("evaluable")
    ]

    if dag is not None:
        for vertex in dag.vertices:
            if (
                vertex.kind == "evidence"
                and vertex.sub_kind == "logged_signal"
                and (vertex.metadata or {}).get("observation") != "observed"
            ):
                unresolved_evidence.append(
                    f"source-proven but unobserved signal: {vertex.signal_name}"
                )

    hypothesis = HypothesisReportItem(
        title=f"Mechanism slice for {verdict.selected_terminal or 'unknown terminal'}",
        known_px4_mechanism=verdict.selected_terminal or "",
        mechanism=verdict.reasoning or "",
        source_refs=source_refs[:8],
        expected_logged_signature=signature[:12],
        applicability=ApplicabilityReport(
            applicable=verdict.sufficient and branches_verified and not replay_mismatch,
            supported_conditions=supported[:12],
            excluded_by=excluded[:12],
            unresolved_conditions=unresolved_conditions[:12],
            relevant_parameters=parameters[:12],
        ),
        evidence=[
            f"mechanism DAG: {len(dag.vertices)} vertices / {len(dag.edges)} edges"
            if dag
            else "no DAG was produced",
            *(
                [
                    "expression replay ["
                    + str(replay.get("status"))
                    + "] vs "
                    + str(replay.get("observed"))
                    + ": "
                    + "; ".join(
                        f"{r.get('grounded', '')[:60]} match={r.get('match_fraction')}"
                        for r in replay.get("results", [])
                        if r.get("evaluable")
                    )
                ]
                if replay and replay.get("status") not in (None, "not_attempted")
                else []
            ),
        ],
        contradicting_evidence=(
            ["complete deterministic DAG replay mismatched the observed terminal"]
            if replay_mismatch
            else []
        ),
        unresolved_evidence=unresolved_evidence[:12],
        exclusion_checks=[],
        numeric_checks=replay_checks[:12],
        confidence=confidence,
    )

    confirmed = (
        verdict.sufficient
        and branches_verified
        and not replay_mismatch
        and confidence in {"high", "medium"}
    )
    return FlightLogReport(
        airframe_summary="",
        question_intent_summary=question,
        ranked_hypotheses=[hypothesis],
        excluded_mechanisms=[],
        confirmed=[hypothesis.title] if confirmed else [],
        unconfirmed=[] if confirmed else [hypothesis.title],
        final_summary=verdict.reasoning or "",
    )


async def run_dag_discovery_stage(
    profiler: MechanismSourceProfiler,
    cache_root: Union[str, Path],
    question: str,
    source_hash: str,
    log_path: Path,
    *,
    inventory: Optional[dict[str, Any]] = None,
    ulog_hash: Optional[str] = None,
    run_agent: Any = None,
    context: Optional[dict[str, Any]] = None,
    signal_policies: Optional[dict[str, Any]] = None,
    **discovery_kwargs: Any,
) -> DagStageResult:
    """Layer 4 lookup → discover_with_judge → feasibility → Layer 2/3/4.

    ``ulog_hash`` keys the Layer 3 entry; defaults to a digest of the log
    file name + size so distinct logs don't collide.
    """
    cache_root = Path(cache_root)
    _prune_stale_intent(cache_root, seeder_fingerprint())
    seeds_path = layer4_cache_path(cache_root, question)
    cached_seeds = read_seeds_from_cache(seeds_path)

    parameter_values = dict((inventory or {}).get("parameters") or {})

    def annotate(result: DiscoveryResult) -> Optional[MechanismDAG]:
        if result.dag is None:
            return None
        samples = _signal_samples_for_dag(result.dag, log_path)
        return evaluate_feasibility(
            result.dag,
            parameter_values=parameter_values,
            signal_samples=samples,
            signal_policies=signal_policies,
        )

    logged_set = {str(s) for s in (discovery_kwargs.get("logged_signals") or ())}

    def condition_windows(condition: Any, candidates: Any = None) -> Optional[dict[str, Any]]:
        """Evaluate the questioned comparison over the log: resolve the
        signal hint via the schema and the discovered slices, the
        reference against ULog parameters, then compute the intervals
        where the condition held."""
        from flight_log_agent.analysis.mechanism_dag import (
            _evaluate_predicate_intervals,
            _signal_policy,
        )

        dags = [
            r.dag for r in (candidates or {}).values() if r is not None
        ] if isinstance(candidates, dict) else []
        signal, error, options = resolve_questioned_signal(
            condition.signal_hint, logged_set, dags
        )
        if signal is None:
            return {"error": error, "candidates": options, "windows": None}
        policy = _signal_policy(signal, signal_policies or {})
        if policy is None:
            return {
                "error": f"no schema-derived signal policy for {signal}",
                "windows": None,
            }
        if policy.get("confidence") == "low" and not policy.get("unit"):
            return {
                "error": f"units for {signal} are not derivable from schema metadata",
                "windows": None,
            }
        expected_unit = str(policy.get("unit") or "unitless")
        stated_unit = str(condition.units or "").strip()
        stated_frame = str(condition.frame or "").strip()
        if not stated_unit or not stated_frame:
            return {
                "error": "questioned condition lacks explicit units or frame",
                "windows": None,
            }
        if expected_unit and stated_unit.lower() != expected_unit.lower():
            return {
                "error": (
                    f"questioned condition unit {stated_unit!r} is incompatible "
                    f"with schema unit {expected_unit!r}"
                ),
                "windows": None,
            }
        reference: Any = parameter_values.get(str(condition.reference).upper())
        if reference is None:
            try:
                reference = float(condition.reference)
            except (TypeError, ValueError):
                return {"error": f"reference {condition.reference!r} did not resolve", "windows": None}
        index = ULogEvidenceIndex.from_path(log_path, [signal])
        resolution = index.resolve_signal(signal)
        if resolution.status != "observed" or resolution.series is None:
            return {"error": f"{signal} not observed in log", "windows": None}
        samples = {
            signal: [(s.time_s, s.value) for s in resolution.series.samples]
        }
        windows = _evaluate_predicate_intervals(
            f"{signal} {condition.op} {reference}",
            {},
            {},
            samples,
            signal_policies,
        )
        return {
            "signal": signal,
            "op": condition.op,
            "reference": reference,
            "units": stated_unit,
            "frame": stated_frame,
            "assumptions": list(condition.assumptions),
            "windows": windows,
        }

    judged = await discover_with_judge(
        profiler,
        cache_root,
        question,
        source_hash,
        run_agent=run_agent,
        context=context,
        seeds_override=cached_seeds,
        annotate=annotate,
        condition_windows=condition_windows,
        parameter_values=parameter_values,
        inventory=inventory,
        **discovery_kwargs,
    )

    annotated: Optional[MechanismDAG] = judged.selected_annotated
    selected: Optional[DiscoveryResult] = judged.selected
    if selected is not None and selected.dag is not None and selected.dag.vertices:
        terminal = selected.dag.terminal
        write_dag_to_cache(
            selected.dag, layer2_cache_path(cache_root, source_hash, terminal)
        )
        if annotated is None:
            annotated = annotate(selected)
        if ulog_hash is None:
            stat = log_path.stat()
            ulog_hash = hashlib.sha256(
                f"{log_path.name}:{stat.st_size}".encode("utf-8")
            ).hexdigest()[:16]
        write_dag_to_cache(
            annotated,
            layer3_cache_path(cache_root, source_hash, ulog_hash, terminal),
        )
        # Seeds are cached only when they EARNED it — a seeder sample
        # whose discovery produced an empty selected slice must not
        # become the question's replayed answer; the next run retries
        # the seeder instead.
        if cached_seeds is None:
            write_seeds_to_cache(judged.seeds, seeds_path)

    replay: Optional[dict[str, Any]] = None
    if annotated is not None:
        replay = replay_terminal_expressions(
            annotated,
            log_path,
            parameter_values,
            logged_set,
            observed_hint=(
                judged.seeds.questioned_condition.signal_hint
                if judged.seeds.questioned_condition
                else None
            ),
            signal_policies=signal_policies,
        )

    report = build_report_from_dag(question, judged, annotated, replay=replay)
    return DagStageResult(
        judged=judged,
        annotated_dag=annotated,
        render=render_discovery_compact(selected) if selected else {},
        layer4_hit=cached_seeds is not None,
        report=report,
        replay=replay,
    )
