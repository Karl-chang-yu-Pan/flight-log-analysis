"""Flag-gated DAG discovery pipeline (#73 Stage 4).

Replaces runner Stages 3–5 when enabled: fresh DAG discovery and judging
instead of mechanism-cache retrieval and the SourceDiscoveryDecision loop,
then DAG ``evaluate_feasibility`` over ULog parameters/samples instead of the
verification-plan checks. The final report is constructed deterministically
from the verdict and annotated DAG. Persistent DAG caches are deliberately
dormant until construction semantics are accepted.
"""

from __future__ import annotations

import hashlib
import re
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from flight_log_agent.analysis.dag_checkpoint import assess_checkpoint
from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round
from flight_log_agent.analysis.dag_replay import (
    EvaluationScope,
    observed_checkpoint_roots,
    replay_dag_roots,
)

from flight_log_agent.analysis.dag_value import DAGValueProgram
from flight_log_agent.analysis.log_evidence import ULogEvidenceIndex
from flight_log_agent.analysis.mechanism_dag import (
    MechanismDAG,
    PreparedSignalSeries,
    evaluate_feasibility,
    prepare_signal_series,
    prune_infeasible_operations,
    sample_prepared_signal,
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
from flight_log_agent.px4.msg_schema import canonicalize_unit
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
    checkpoint_rounds: list[dict[str, Any]] = field(default_factory=list)


def _signal_samples_for_dag(
    dag: MechanismDAG, log_path: Path, *, additional_signals: tuple[str, ...] = ()
) -> dict[str, list[tuple[float, Any]]]:
    """Load graph evidence and any independent checkpoint observations together."""
    references = sorted(
        set(additional_signals) | {
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


@dataclass(frozen=True)
class _QuestionedUnits:
    signal: str
    reference: str


def _parse_questioned_units(value: Any) -> tuple[Optional[_QuestionedUnits], Optional[str]]:
    """Parse the condition's explicit left/right unit contract.

    A single unit applies to both sides for backward compatibility. A typed
    condition may instead use ``signal: <unit>; reference: <unit>``. Both
    labels are required in the typed form so a partially specified expression
    never inherits a unit silently.
    """
    text = str(value or "").strip()
    direct = canonicalize_unit(text)
    if direct is not None:
        return _QuestionedUnits(signal=direct, reference=direct), None

    matches = {
        side.lower(): unit.strip()
        for side, unit in re.findall(
            r"\b(signal|reference)\s*:\s*([^;,]+)", text, flags=re.IGNORECASE
        )
    }
    if set(matches) != {"signal", "reference"}:
        return None, (
            "questioned condition units must be one supported unit or "
            "'signal: <unit>; reference: <unit>'"
        )
    signal_unit = canonicalize_unit(matches["signal"])
    reference_unit = canonicalize_unit(matches["reference"])
    if signal_unit is None or reference_unit is None:
        return None, f"questioned condition contains unsupported units {text!r}"
    return _QuestionedUnits(signal=signal_unit, reference=reference_unit), None


def _resolve_question_expression(
    expression: Any,
    *,
    logged_set: set[str],
    parameter_values: dict[str, Any],
    dags: list[Optional[MechanismDAG]],
    allow_slice_resolution: bool,
) -> tuple[Optional[str], list[str], Optional[str], list[str]]:
    """Resolve every operand in a questioned expression without fuzzy names.

    Dotted/indexed operands must resolve to an observed signal, while bare
    operands may resolve to an exact ULog parameter name. The primary signal
    hint retains the existing slice-proven rename fallback, but operands in a
    compound expression require exact topic/field identity (with at most one
    observed instance). Returns a rewritten expression containing canonical
    observed signal placements.
    """
    from flight_log_agent.analysis.source_expression import (
        alias_dotted_names,
        normalize_source_expression,
        source_expression_names,
    )

    normalized = normalize_source_expression(str(expression or "").strip())
    if not normalized:
        return None, [], "empty questioned expression", []
    names = source_expression_names(normalized)
    if not names:
        try:
            float(normalized)
        except (TypeError, ValueError):
            return None, [], f"questioned expression {expression!r} is invalid", []
        return normalized, [], None, []

    resolved_signals: dict[str, str] = {}
    options: list[str] = []
    single_primary = allow_slice_resolution and len(names) == 1 and normalized == names[0]
    for name in names:
        is_parameter = name in parameter_values
        signal, signal_error, candidates = resolve_questioned_signal(
            name,
            logged_set,
            dags if single_primary else [],
        )
        if signal is not None and is_parameter:
            return None, [], f"operand {name!r} is ambiguous between signal and parameter", []
        if signal is not None:
            resolved_signals[name] = signal
            continue
        if is_parameter:
            continue
        options.extend(candidates)
        detail = signal_error or "did not resolve"
        return None, [], f"operand {name!r} did not resolve: {detail}", sorted(set(options))

    rewritten, alias_to_name = alias_dotted_names(normalized, resolved_signals)
    for alias, name in alias_to_name.items():
        rewritten = rewritten.replace(alias, resolved_signals[name])
    return rewritten, list(dict.fromkeys(resolved_signals.values())), None, []


def _validate_question_expression_units(
    signals: list[str],
    stated_unit: str,
    signal_policies: dict[str, Any],
) -> Optional[str]:
    """Validate observed operands against one explicitly typed expression.

    Parameters and numeric literals have no unit metadata in ULog, so their
    type comes from the explicit side contract. Every observed signal operand
    must independently agree with that contract; missing metadata fails
    closed rather than turning an untyped value into evidence.
    """
    from flight_log_agent.analysis.mechanism_dag import _signal_policy

    for signal in signals:
        policy = _signal_policy(signal, signal_policies)
        if policy is None:
            return f"no schema-derived signal policy for {signal}"
        raw_unit = str(policy.get("unit") or "")
        derived = canonicalize_unit(raw_unit)
        if derived is None:
            # Preserve the established treatment of schema-classified
            # booleans/enums/discrete states: a high-confidence policy with no
            # physical unit is explicitly compatible with ``unitless``. A
            # low-confidence untyped scalar remains unresolved.
            if stated_unit == "unitless" and policy.get("confidence", "high") != "low":
                continue
            return f"units for {signal} are not derivable from schema metadata"
        if derived != stated_unit:
            return (
                f"questioned condition unit {stated_unit!r} is incompatible "
                f"with schema unit {derived!r} for {signal}"
            )
    return None


def evaluate_questioned_condition_windows(
    condition: Any,
    *,
    candidates: Any,
    logged_set: set[str],
    log_path: Path,
    parameter_values: dict[str, Any],
    signal_policies: dict[str, Any],
    candidate_dags: Optional[list[MechanismDAG]] = None,
) -> dict[str, Any]:
    """Evaluate a typed, multi-signal questioned comparison over a ULog."""
    from flight_log_agent.analysis.mechanism_dag import _evaluate_predicate_intervals

    dags = list(candidate_dags or [])
    if candidate_dags is None and isinstance(candidates, dict):
        dags = [
            result.dag for result in candidates.values()
            if result is not None and getattr(result, "dag", None) is not None
        ]
    units, unit_error = _parse_questioned_units(condition.units)
    stated_frame = str(condition.frame or "").strip()
    if unit_error:
        return {"error": unit_error, "windows": None}
    if not stated_frame:
        return {"error": "questioned condition lacks an explicit frame", "windows": None}
    assert units is not None

    left, left_signals, error, options = _resolve_question_expression(
        condition.signal_hint,
        logged_set=logged_set,
        parameter_values=parameter_values,
        dags=dags,
        allow_slice_resolution=True,
    )
    if left is None:
        return {"error": error, "candidates": options, "windows": None}
    right, right_signals, error, options = _resolve_question_expression(
        condition.reference,
        logged_set=logged_set,
        parameter_values=parameter_values,
        dags=[],
        allow_slice_resolution=False,
    )
    if right is None:
        return {"error": error, "candidates": options, "windows": None}

    error = _validate_question_expression_units(
        left_signals, units.signal, signal_policies
    ) or _validate_question_expression_units(
        right_signals, units.reference, signal_policies
    )
    if error:
        return {"error": error, "windows": None}
    if units.signal != units.reference:
        return {
            "error": (
                f"questioned comparison has incompatible side units "
                f"{units.signal!r} and {units.reference!r}"
            ),
            "windows": None,
        }

    references = list(dict.fromkeys([*left_signals, *right_signals]))
    if not references:
        return {"error": "questioned comparison contains no observed signal", "windows": None}
    index = ULogEvidenceIndex.from_path(log_path, references)
    samples: dict[str, list[tuple[float, Any]]] = {}
    for signal in references:
        resolution = index.resolve_signal(signal)
        if resolution.status != "observed" or resolution.series is None:
            return {"error": f"{signal} not observed in log", "windows": None}
        samples[signal] = [
            (sample.time_s, sample.value) for sample in resolution.series.samples
        ]

    predicate = f"({left}) {condition.op} ({right})"
    windows = _evaluate_predicate_intervals(
        predicate,
        parameter_values,
        {},
        samples,
        signal_policies,
    )
    if windows is None:
        return {"error": "questioned comparison was not evaluable", "windows": None}
    return {
        "signal": left,
        "op": condition.op,
        "reference": right,
        "units": str(condition.units or "").strip(),
        "resolved_units": {
            "signal": units.signal,
            "reference": units.reference,
        },
        "frame": stated_frame,
        "assumptions": list(condition.assumptions),
        "windows": windows,
    }


def replay_terminal_expressions(
    annotated: MechanismDAG,
    log_path: Path,
    parameter_values: dict[str, Any],
    logged_set: set[str],
    observed_hint: Optional[str] = None,
    signal_policies: Optional[dict[str, Any]] = None,
    signal_samples: Optional[dict[str, list[tuple[float, Any]]]] = None,
    prepared_signal_series: Optional[dict[str, PreparedSignalSeries]] = None,
) -> dict[str, Any]:
    """Numerically compare terminal writes with their observed output.

    Producer values and writer selection are evaluated directly from DAG
    edges. Source text is used only for the local operators within one
    operation; no recursively substituted expression participates in the
    result. Each writer is compared only inside its gating branch windows.
    A definitive match/mismatch requires exact, non-overlapping writer
    domains that cover the observed output domain.
    """

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
        str((op.metadata or {}).get("external_target_signal") or "")
        for op in terminal_ops
        if (op.metadata or {}).get("external_target_signal")
    }
    if len(published) == 1:
        observed = next(iter(published))
    elif observed_hint and observed_hint in published:
        observed = observed_hint
    else:
        return not_attempted(
            "observed output lacks unique exact terminal-publication provenance"
        )
    if observed not in logged_set:
        return not_attempted(f"terminal output {observed!r} is not in the observed catalogue")
    samples = (
        dict(signal_samples)
        if signal_samples is not None
        else dict(_signal_samples_for_dag(annotated, log_path))
    )
    observed_samples = samples.get(observed)
    if observed_samples is None:
        index = ULogEvidenceIndex.from_path(log_path, [observed])
        resolution = index.resolve_signal(observed)
        if resolution.status != "observed" or resolution.series is None:
            return not_attempted(f"{observed} not observed in this log")
        observed_samples = [(s.time_s, s.value) for s in resolution.series.samples]
        samples[observed] = observed_samples
    if len(observed_samples) < 2:
        return not_attempted(f"{observed} has too few samples")
    prepared_series = dict(prepared_signal_series or {})
    missing_samples = {
        signal: series
        for signal, series in samples.items()
        if signal not in prepared_series
    }
    if missing_samples:
        prepared_series.update(
            prepare_signal_series(missing_samples, signal_policies)
        )
    return replay_dag_roots(
        annotated,
        [op.id for op in terminal_ops],
        observed,
        parameter_values=parameter_values,
        signal_samples=samples,
        signal_policies=signal_policies,
        prepared_signal_series=prepared_series,
    )


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
            logged_output = str(
                (vertex.metadata or {}).get("external_target_signal")
                or (vertex.metadata or {}).get("logged_signal")
                or ""
            )
            output_observation = str(
                (vertex.metadata or {}).get("external_target_observation")
                or (vertex.metadata or {}).get("logged_observation")
                or ""
            )
            if (
                logged_output
                and output_observation == "observed"
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
    checkpoint_unresolved = bool(
        replay and "authorizes_discovery_stop" in replay
        and not replay["authorizes_discovery_stop"]
    )
    if checkpoint_unresolved:
        unresolved_evidence.insert(0, "source-backed checkpoint remains unresolved")
        unresolved_evidence.extend(
            str(requirement.get("reason") or requirement.get("kind"))
            for requirement in replay.get("analysis_requirements", [])
        )
    has_numeric_replay = any(
        result.get("evaluable") for result in (replay or {}).get("results", [])
    )
    if replay_mismatch or checkpoint_unresolved:
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
            expression=str(
                result.get("grounded") or result.get("expression") or ""
            ),
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
            applicable=verdict.sufficient and branches_verified and not replay_mismatch and not checkpoint_unresolved,
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
                        f"{str(r.get('grounded') or '')[:60]} match={r.get('match_fraction')}"
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


def _proof_snapshot_args(proof_snapshot: Any) -> dict:
    """Map an immutable proof snapshot onto T6B round inputs (P3).

    Pure value mapping, no session access: `None` yields no arguments
    (legacy omission preserved exactly), otherwise the snapshot's
    exact version plus its deterministic certificate/proof tuples.
    """
    if proof_snapshot is None:
        return {}
    return {
        "proof_version": proof_snapshot.version,
        "coverage_certificates": tuple(
            proof_snapshot.certificates or ()),
        "applicability_proofs": tuple(
            proof_snapshot.applicability_proofs or ()),
    }


def _construction_evaluator_for(control_round: Any) -> Any:
    """Adapt the checkpoint control round to the construction seam.

    Forwards the optional proof snapshot unchanged so construction
    rounds observe the same current proof as checkpoint rounds.
    Extracted (rather than inline lambda) so the forwarding contract
    is directly testable.
    """

    def evaluate(dag: Any, index: Any, proof_snapshot: Any = None) -> Any:
        return control_round(
            dag, index, during_construction=True,
            proof_snapshot=proof_snapshot)

    return evaluate


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
    checkpoint_diagnostics: bool = False,
    checkpoint_observer: Optional[Callable[[dict[str, Any]], None]] = None,
    checkpoint_discovery: bool = False,
    **discovery_kwargs: Any,
) -> DagStageResult:
    """Run fresh DAG discovery, feasibility, replay, and report construction.

    DAG cache use is intentionally dormant while constructor semantics are
    still being validated. The Layer 1-4 cache helpers remain in the codebase
    for later reactivation, but this production path neither reads, writes,
    prunes, nor reports a cache hit. Re-enable cache call sites only after the
    DAG correctness acceptance criteria are satisfied.

    ``ulog_hash`` and ``cache_root`` remain in the stable call contract for
    that future reactivation and for the discovery API, respectively.

    Opt-in checkpoint diagnostics replay source-proven publication roots each
    round. They never alter expansion, judge input, or report confirmation.
    ``checkpoint_discovery`` instead makes that assessment control expansion:
    evaluate ready dependencies, request exact missing source, and stop verified
    or explicitly unresolved. A local match is not the judge's question verdict.
    Only compact results survive a round; graph programs stay run-local.
    """
    cache_root = Path(cache_root)
    cached_seeds = None

    parameter_values = dict((inventory or {}).get("parameters") or {})
    logged_set = {str(s) for s in (discovery_kwargs.get("logged_signals") or ())}
    checkpoint_rounds: list[dict[str, Any]] = []
    scope: Optional[EvaluationScope] = None
    questioned_condition: Any = None

    def prepare_scope(seeds: DiscoverySeeds) -> None:
        nonlocal scope, questioned_condition
        questioned_condition = seeds.questioned_condition
        scope = None
        if questioned_condition is not None:
            scope = EvaluationScope.from_result(evaluate_questioned_condition_windows(
                questioned_condition,
                candidates=None,
                logged_set=logged_set,
                log_path=log_path,
                parameter_values=parameter_values,
                signal_policies=signal_policies or {},
            ))
    signal_data: dict[
        int,
        tuple[
            dict[str, list[tuple[float, Any]]],
            dict[str, PreparedSignalSeries],
        ],
    ] = {}
    # One live annotation, not one per round. ``annotate_dag`` is the fixpoint's
    # round annotator, and every round hands it a NEW graph, so caching by graph
    # retained a full annotated copy, a value program and a set of ULog sample
    # series for EVERY round of EVERY candidate — while only the newest is ever
    # read (a round's annotation is consumed to pick that round's frontier).
    # The entry holds the graph object itself rather than its ``id()``: an id is
    # reused once its object is freed, so an id-keyed cache could return another
    # graph's feasibility.
    annotation: dict[str, Any] = {}

    def annotate_dag(dag: MechanismDAG) -> MechanismDAG:
        if annotation.get("dag") is dag:
            return annotation["annotated"]
        annotation.clear()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        if checkpoint_diagnostics:
            samples = _signal_samples_for_dag(
                dag, log_path,
                additional_signals=tuple(
                    signal for signal in observed_checkpoint_roots(dag)
                    if signal in logged_set
                ),
            )
        else:
            samples = _signal_samples_for_dag(dag, log_path)
        prepared_series = prepare_signal_series(samples, signal_policies)
        program = DAGValueProgram(dag)
        session = program.bind(
            parameter_values=parameter_values,
            sample_resolver=lambda signal, timestamp: sample_prepared_signal(
                prepared_series, signal, timestamp
            ),
        )
        full_annotation = evaluate_feasibility(
            dag,
            parameter_values=parameter_values,
            signal_samples=samples,
            signal_policies=signal_policies,
            prepared_signal_series=prepared_series,
            value_program=program,
            value_session=session,
            prune_dead=not checkpoint_diagnostics,
        )
        annotated = (
            prune_infeasible_operations(full_annotation)
            if checkpoint_diagnostics else full_annotation
        )
        annotation.update(
            {
                "dag": dag,
                "annotated": annotated,
                "samples": samples,
                "prepared": prepared_series,
                **({
                    "full_annotation": full_annotation,
                    "program": program,
                    "session": session,
                } if checkpoint_diagnostics else {}),
                "feasibility_wall_s": time.perf_counter() - wall_started,
                "feasibility_cpu_s": time.process_time() - cpu_started,
            }
        )
        return annotated

    def annotate(result: DiscoveryResult) -> Optional[MechanismDAG]:
        if result.dag is None:
            return None
        if result.checkpoint is not None:
            return result.checkpoint.annotated
        annotated = annotate_dag(result.dag)
        # Retained per CANDIDATE (not per round) because replay needs the
        # selected candidate's series after the verdict.
        signal_data[id(result)] = (annotation["samples"], annotation["prepared"])
        return annotated

    discovery_kwargs.setdefault("round_annotator", annotate_dag)

    previous_observer = discovery_kwargs.get("round_observer")

    def observe_round(dag: MechanismDAG, _feasible: MechanismDAG, index: int) -> None:
        nonlocal scope
        annotate_dag(dag)
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        if scope is not None and scope.windows is None and questioned_condition is not None:
            scope = EvaluationScope.from_result(evaluate_questioned_condition_windows(
                questioned_condition,
                candidates=None,
                candidate_dags=[dag],
                logged_set=logged_set,
                log_path=log_path,
                parameter_values=parameter_values,
                signal_policies=signal_policies or {},
            ))
        checkpoints = {
            signal: assess_checkpoint(
                annotation["full_annotation"], roots, signal,
                source_dag=dag,
                observed_signals=logged_set,
                parameter_values=parameter_values,
                signal_samples=annotation["samples"],
                signal_policies=signal_policies,
                prepared_signal_series=annotation["prepared"],
                value_program=annotation["program"],
                value_session=annotation["session"],
                scope=scope,
            )
            for signal, roots in observed_checkpoint_roots(dag).items()
        }
        terminal_checkpoint = None
        if not checkpoints:
            terminal_checkpoint = assess_checkpoint(
                annotation["full_annotation"],
                [vertex.id for vertex in dag.vertices if vertex.metadata.get("is_terminal")],
                None, source_dag=dag, observed_signals=logged_set,
                parameter_values=parameter_values, signal_samples=annotation["samples"],
                signal_policies=signal_policies, prepared_signal_series=annotation["prepared"],
                value_program=annotation["program"], value_session=annotation["session"], scope=scope,
            )
        summary = {
            "diagnostic_only": True,
            "round_index": index,
            "dag_id": dag.dag_id,
            "terminal": dag.terminal,
            "scope": scope.as_payload() if scope is not None else None,
            "checkpoints": checkpoints,
            "terminal_checkpoint": terminal_checkpoint,
            "unresolved_references": [
                reference.model_dump(mode="json") for reference in dag.unresolved_references
            ],
            "resources": {
                "feasibility_wall_s": annotation["feasibility_wall_s"],
                "feasibility_cpu_s": annotation["feasibility_cpu_s"],
                "checkpoint_wall_s": time.perf_counter() - wall_started,
                "checkpoint_cpu_s": time.process_time() - cpu_started,
                # Linux process high-water mark, not memory allocated by this round.
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        }
        checkpoint_rounds.append(summary)
        if checkpoint_observer is not None:
            checkpoint_observer(summary)
        if previous_observer is not None:
            previous_observer(dag, _feasible, index)

    if checkpoint_diagnostics and not checkpoint_discovery:
        discovery_kwargs["round_observer"] = observe_round

    if checkpoint_discovery:
        def control_round(dag: MechanismDAG, index: int, *, during_construction: bool = False,
                          proof_snapshot: Any = None):
            nonlocal scope
            wall_started, cpu_started = time.perf_counter(), time.process_time()
            target = None
            if questioned_condition is not None:
                scope = EvaluationScope.from_result(evaluate_questioned_condition_windows(
                    questioned_condition, candidates=None, candidate_dags=[dag], logged_set=logged_set,
                    log_path=log_path, parameter_values=parameter_values, signal_policies=signal_policies or {},
                ))
                target, _error, _candidates = resolve_questioned_signal(
                    questioned_condition.signal_hint, logged_set, [dag],
                )
                # An unresolved/multi-signal target cannot silently become a
                # terminal-only claim and authorize early verification.
                target = target or ""
            result = evaluate_checkpoint_round(
                dag, parameter_values=parameter_values, observed_signals=logged_set,
                signal_policies=signal_policies or {}, scope=scope, question_target=target,
                load_samples=lambda view, observed: _signal_samples_for_dag(view, log_path, additional_signals=observed),
                **_proof_snapshot_args(proof_snapshot),
            )
            summary = {
                **result.summary, "diagnostic_only": False, "round_index": index,
                "phase": "construction" if during_construction else "round_complete",
                "dag_id": dag.dag_id, "terminal": dag.terminal,
                "scope": scope.as_payload() if scope is not None else None,
                "unresolved_references": [r.model_dump(mode="json") for r in dag.unresolved_references],
                "resources": {"checkpoint_wall_s": time.perf_counter() - wall_started,
                              "checkpoint_cpu_s": time.process_time() - cpu_started,
                              "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
            }
            result.summary = summary
            # Keep proof payloads for completed rounds. Intermediate snapshots
            # are transient; retaining every frontier/preflight would multiply
            # memory by the number of dependency waves.
            event = summary
            if during_construction:
                event = {key: summary[key] for key in (
                    "phase", "diagnostic_only", "round_index", "dag_id", "terminal",
                    "scope", "resources", "action", "pending_construction_count",
                    "dynamic_gate_count",
                )}
                demand = summary.get("next_analysis") or {}
                event["next_analysis"] = {
                    "kind": demand.get("kind"),
                    "operation_ids": demand.get("operation_ids", []),
                    "guard_ids": demand.get("guard_ids", []),
                    "source_request_count": len(demand.get("source_requests", [])),
                }
            checkpoint_rounds.append(event)
            if checkpoint_observer is not None:
                checkpoint_observer(event)
            return result
        discovery_kwargs["checkpoint_evaluator"] = control_round
        discovery_kwargs["construction_evaluator"] = (
            _construction_evaluator_for(control_round))

    def condition_windows(condition: Any, candidates: Any = None) -> Optional[dict[str, Any]]:
        return evaluate_questioned_condition_windows(
            condition,
            candidates=candidates,
            logged_set=logged_set,
            log_path=log_path,
            parameter_values=parameter_values,
            signal_policies=signal_policies or {},
        )

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
        on_seeds=prepare_scope if checkpoint_diagnostics or checkpoint_discovery else None,
        parameter_values=parameter_values,
        inventory=inventory,
        **discovery_kwargs,
    )

    annotated: Optional[MechanismDAG] = judged.selected_annotated
    selected: Optional[DiscoveryResult] = judged.selected
    if selected is not None and selected.dag is not None and selected.dag.vertices:
        if annotated is None:
            annotated = annotate(selected)

    replay: Optional[dict[str, Any]] = None
    if selected is not None and selected.checkpoint is not None:
        replay = selected.checkpoint.summary.get("selected_checkpoint") or {
            "status": "not_attempted", "complete": False, "reason": selected.checkpoint.summary["reason"],
        }
    elif annotated is not None:
        selected_signal_data = signal_data.get(id(selected))
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
            signal_samples=(
                selected_signal_data[0] if selected_signal_data is not None else None
            ),
            prepared_signal_series=(
                selected_signal_data[1] if selected_signal_data is not None else None
            ),
        )

    report = build_report_from_dag(question, judged, annotated, replay=replay)
    return DagStageResult(
        judged=judged,
        annotated_dag=annotated,
        render=render_discovery_compact(selected) if selected else {},
        layer4_hit=False,
        report=report,
        replay=replay,
        checkpoint_rounds=checkpoint_rounds,
    )
