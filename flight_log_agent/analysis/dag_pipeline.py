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
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler


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

    payload = seeder_agent.instructions + _json.dumps(
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

    Exact schema match wins. Otherwise the hint's TOPIC (which must
    exist) is intersected with the fields the candidate DAGs actually
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
    topic = hint.split(".", 1)[0]
    connected = {
        str(v.signal_name)
        for dag in dags
        if dag is not None
        for v in dag.vertices
        if v.kind == "evidence"
        and v.sub_kind == "logged_signal"
        and str(v.signal_name or "").startswith(topic + ".")
    }
    if len(connected) == 1:
        return next(iter(connected)), None, []
    if connected:
        return None, f"hint {hint!r} is ambiguous in the slice", sorted(connected)
    # Slice carries no leaf for the hinted topic. Substring containment
    # against the schema is name guessing, not resolution — fields on
    # the topic are returned only as CANDIDATES for the judge, never
    # auto-picked, however few there are.
    candidates = sorted(s for s in logged_set if s.startswith(topic + "."))
    if candidates:
        return None, f"hint {hint!r} did not resolve exactly", candidates
    return None, f"signal hint {hint!r} did not resolve", []


# The five distinct replay states (core correctness invariant):
# ``matched`` is supporting evidence and ``mismatched`` contradictory
# evidence ONLY when replay completeness (writer reachability domains,
# ordering, retained state, alignment, coverage) is trustworthy —
# neither is reachable from the current whole-log any-writer comparison,
# which caps at ``partial``. ``partial`` and ``unevaluable`` are
# unresolved evidence; ``not_attempted`` is neutral.
ReplayStatus = Literal[
    "not_attempted", "unevaluable", "partial", "matched", "mismatched"
]


def replay_terminal_expressions(
    annotated: MechanismDAG,
    log_path: Path,
    parameter_values: dict[str, Any],
    logged_set: set[str],
    observed_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Numerically compare the mechanism against the log: ground each
    terminal write's expression through the graph, evaluate it over the
    log, and measure how long it matches the observed signal within
    tolerance.

    Fully structural — grounding walks the DAG, the observed signal
    resolves from the logged catalogue (hint first, then the terminal),
    and the match check reuses the interval evaluator as the predicate
    ``abs(grounded - observed) <= tol``. The result's ``status`` is one
    of :data:`ReplayStatus`; because writers are compared over the WHOLE
    log without their reachability domains or ordering, the strongest
    status this implementation can honestly emit is ``partial``.
    """
    from flight_log_agent.analysis.mechanism_dag import (
        _evaluate_predicate_intervals,
        ground_expression_via_edges,
    )

    def not_attempted(reason: str) -> dict[str, Any]:
        return {"status": "not_attempted", "complete": False, "reason": reason}

    def resolve(hint: str) -> Optional[str]:
        # Exact catalogue membership only — suffix uniqueness is name
        # guessing; an unresolved observed signal skips replay rather
        # than comparing against a guessed one.
        return hint if hint and hint in logged_set else None

    observed = resolve(observed_hint or "") or resolve(str(annotated.terminal or ""))
    if observed is None:
        return not_attempted("observed signal did not resolve exactly")
    terminal_ops = [
        v
        for v in annotated.vertices
        if v.kind == "operation" and (v.metadata or {}).get("is_terminal")
    ]
    if not terminal_ops:
        return not_attempted("no terminal writes in the graph")
    index = ULogEvidenceIndex.from_path(log_path, [observed])
    resolution = index.resolve_signal(observed)
    if resolution.status != "observed" or resolution.series is None:
        return not_attempted(f"{observed} not observed in this log")
    observed_samples = [(s.time_s, s.value) for s in resolution.series.samples]
    if len(observed_samples) < 2:
        return not_attempted(f"{observed} has too few samples")
    samples = dict(_signal_samples_for_dag(annotated, log_path))
    samples[observed] = observed_samples
    magnitudes = [
        abs(float(v)) for _, v in observed_samples if isinstance(v, (int, float))
    ]
    tolerance = 0.05 * (sum(magnitudes) / len(magnitudes)) if magnitudes else 1e-3
    span = observed_samples[-1][0] - observed_samples[0][0]

    results: list[dict[str, Any]] = []
    for op in terminal_ops:
        grounded = ground_expression_via_edges(
            str(op.expression or ""), op.id, annotated
        )
        if not grounded:
            continue
        windows = _evaluate_predicate_intervals(
            f"abs(({grounded}) - ({observed})) <= {tolerance}",
            parameter_values,
            {},
            samples,
        )
        if windows is None:
            results.append(
                {"expression": op.expression, "grounded": grounded, "evaluable": False}
            )
            continue
        matched = sum(end - start for start, end in windows)
        results.append(
            {
                "expression": op.expression,
                "grounded": grounded,
                "evaluable": True,
                "match_fraction": round(matched / span, 3) if span else 0.0,
            }
        )
    if not results:
        status: ReplayStatus = "unevaluable"
        reason = "no terminal expression grounded through the graph"
    elif not any(r.get("evaluable") for r in results):
        status = "unevaluable"
        reason = "no grounded expression was evaluable over the log"
    else:
        # Writers were compared over the whole log without reachability
        # domains, ordering, or coverage criteria — the comparison is
        # incomplete by construction, so the evidence stays unresolved.
        status = "partial"
        reason = (
            "writers compared over the whole log without reachability "
            "domains, ordering, or coverage"
        )
    return {
        "status": status,
        "complete": False,
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

    Confidence is capped at ``medium``: this path runs no legacy numeric
    checks, so ``high`` would overclaim. Branch feasibility maps to the
    applicability report (``always_true`` → supported, ``always_false``
    → excluded, ``unknown`` → unresolved).
    """
    verdict = judged.verdict
    dag = annotated_dag or (judged.selected.dag if judged.selected else None)

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
            signature.append(
                ExpectedSignatureItem(
                    name=str(vertex.signal_name),
                    description="logged signal grounding the mechanism slice",
                    signal=str(vertex.signal_name),
                )
            )
        elif vertex.kind == "operation" and vertex.file and vertex.metadata.get("is_terminal"):
            source_refs.append(
                CodeRef(
                    file=str(vertex.file),
                    start_line=vertex.line,
                    end_line=vertex.line,
                    snippet=vertex.snippet,
                    explanation=f"terminal write: {vertex.variable} <- {vertex.expression}",
                )
            )

    # Cross-check the judge's claimed mechanism against flight-data
    # feasibility: a sufficient verdict must name explaining branches,
    # and at least one must exist in the DAG without being feasibility-
    # dead. Otherwise the confirmation is downgraded — the mechanism may
    # be real code, but nothing shows it fired in THIS flight.
    unresolved_evidence = list(dag.unresolved_symbols) if dag else []
    branches_verified = False
    if verdict.sufficient and verdict.explaining_branches and dag is not None:
        dag_predicates = {
            (v.predicate_raw or "", v.feasibility_verdict or "unknown")
            for v in dag.vertices
            if v.kind == "branch"
        } | {
            (v.predicate_lowered or "", v.feasibility_verdict or "unknown")
            for v in dag.vertices
            if v.kind == "branch"
        }
        for named in verdict.explaining_branches:
            # The judge copies entries from the rendering verbatim — strip
            # the trailing feasibility tag and the truncation ellipsis so
            # long predicates still match (prefix containment).
            needle = " ".join(str(named).split())
            needle = re.sub(r"\s*\[[^\]]*\]\s*$", "", needle).rstrip("… ").strip()
            for predicate, feasibility in dag_predicates:
                haystack = " ".join(predicate.split())
                if not needle or not haystack:
                    continue
                if feasibility == "always_false":
                    continue
                if (
                    needle in haystack
                    or haystack in needle
                    or haystack.startswith(needle)
                ):
                    branches_verified = True
                    break
            if branches_verified:
                break
        if not branches_verified:
            unresolved_evidence.insert(
                0,
                "judge-named explaining branch(es) absent from the DAG or "
                "feasibility-dead: " + "; ".join(verdict.explaining_branches[:3]),
            )
    elif verdict.sufficient:
        unresolved_evidence.insert(
            0, "judge confirmed the mechanism without naming an explaining branch"
        )

    if (
        verdict.sufficient
        and branches_verified
        and replay
        and replay.get("status") == "matched"
    ):
        # Structure confirmed AND a COMPLETE replay numerically
        # reproduces the observed signal — the honest "high". A
        # ``partial`` replay is unresolved evidence and never upgrades;
        # the current whole-log comparison caps at partial, so this
        # branch stays unreachable until replay completeness lands.
        confidence = "high"
    elif verdict.sufficient and branches_verified:
        confidence = "medium"
    elif verdict.sufficient:
        confidence = "low"
    else:
        confidence = "unresolved"

    hypothesis = HypothesisReportItem(
        title=f"Mechanism slice for {verdict.selected_terminal or 'unknown terminal'}",
        known_px4_mechanism=verdict.selected_terminal or "",
        mechanism=verdict.reasoning or "",
        source_refs=source_refs[:8],
        expected_logged_signature=signature[:12],
        applicability=ApplicabilityReport(
            applicable=verdict.sufficient,
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
        contradicting_evidence=[],
        unresolved_evidence=unresolved_evidence[:12],
        exclusion_checks=[],
        numeric_checks=[],
        confidence=confidence,
    )

    confirmed = verdict.sufficient and branches_verified
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
        )

    logged_set = {str(s) for s in (discovery_kwargs.get("logged_signals") or ())}

    def condition_windows(condition: Any, candidates: Any = None) -> Optional[dict[str, Any]]:
        """Evaluate the questioned comparison over the log: resolve the
        signal hint via the schema and the discovered slices, the
        reference against ULog parameters, then compute the intervals
        where the condition held."""
        from flight_log_agent.analysis.mechanism_dag import (
            _evaluate_predicate_intervals,
        )

        dags = [
            r.dag for r in (candidates or {}).values() if r is not None
        ] if isinstance(candidates, dict) else []
        signal, error, options = resolve_questioned_signal(
            condition.signal_hint, logged_set, dags
        )
        if signal is None:
            return {"error": error, "candidates": options, "windows": None}
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
            f"{signal} {condition.op} {reference}", {}, {}, samples
        )
        return {
            "signal": signal,
            "op": condition.op,
            "reference": reference,
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
