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
from typing import Any, Optional, Union

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


def layer4_cache_path(cache_root: Union[str, Path], question: str) -> Path:
    """Layer 4 path: ``{cache_root}/intent/{slug}.{digest}.json``."""
    return Path(cache_root) / "intent" / f"{_question_slug(question)}.json"


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


def build_report_from_dag(
    question: str,
    judged: JudgedDiscovery,
    annotated_dag: Optional[MechanismDAG],
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
            needle = " ".join(str(named).split())
            for predicate, feasibility in dag_predicates:
                haystack = " ".join(predicate.split())
                if not needle or not haystack:
                    continue
                if (needle in haystack or haystack in needle) and feasibility != "always_false":
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

    if verdict.sufficient and branches_verified:
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
    seeds_path = layer4_cache_path(cache_root, question)
    cached_seeds = read_seeds_from_cache(seeds_path)

    parameter_values = dict((inventory or {}).get("parameters") or {})

    judged = await discover_with_judge(
        profiler,
        cache_root,
        question,
        source_hash,
        run_agent=run_agent,
        context=context,
        seeds_override=cached_seeds,
        parameter_values=parameter_values,
        inventory=inventory,
        **discovery_kwargs,
    )

    annotated: Optional[MechanismDAG] = None
    selected: Optional[DiscoveryResult] = judged.selected
    if selected is not None and selected.dag is not None:
        terminal = selected.dag.terminal
        write_dag_to_cache(
            selected.dag, layer2_cache_path(cache_root, source_hash, terminal)
        )
        samples = _signal_samples_for_dag(selected.dag, log_path)
        annotated = evaluate_feasibility(
            selected.dag,
            parameter_values=parameter_values,
            signal_samples=samples,
        )
        if ulog_hash is None:
            stat = log_path.stat()
            ulog_hash = hashlib.sha256(
                f"{log_path.name}:{stat.st_size}".encode("utf-8")
            ).hexdigest()[:16]
        write_dag_to_cache(
            annotated,
            layer3_cache_path(cache_root, source_hash, ulog_hash, terminal),
        )
        if cached_seeds is None:
            write_seeds_to_cache(judged.seeds, seeds_path)

    report = build_report_from_dag(question, judged, annotated)
    return DagStageResult(
        judged=judged,
        annotated_dag=annotated,
        render=render_discovery_compact(selected) if selected else {},
        layer4_hit=cached_seeds is not None,
        report=report,
    )
