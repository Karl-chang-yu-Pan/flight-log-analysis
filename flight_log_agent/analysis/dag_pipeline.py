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
    _source_site_order,
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


def evaluate_transition_windows(
    transition: Any,
    *,
    logged_set: set[str],
    log_path: Path,
    signal_policies: dict[str, Any],
) -> dict[str, Any]:
    """Derive diagnostic windows from a generic transition-event intent.

    Loads the transition signal (plus the first-sample target signal
    when requested) through the same evidence index as questioned
    conditions, then applies the pure temporal-selection derivation:
    match transition event(s), disambiguate explicitly, and build
    event-relative windows. Returns ``{"windows": [...]}`` on
    success or ``{"windows": None, "error": ...}`` with the exact
    failure; callers fail closed on ``windows is None``. No
    mechanism-specific signals, values, or timestamps appear here.
    """
    from flight_log_agent.analysis.temporal_selection import (
        derive_diagnostic_windows,
        derive_transition_events,
        select_transition_event,
    )

    signal = str(getattr(transition, "transition_signal", None) or "")
    target = str(getattr(transition, "first_sample_of", None) or "")
    if not signal:
        return {"error": "transition specification names no signal", "windows": None}
    references = [signal] + ([target] if target and target != signal else [])
    # Signal identity is resolved by the evidence index, which accepts
    # both bare topic.field and instanced topic[i].field spellings;
    # no separate logged-set spelling gate is applied here.
    index = ULogEvidenceIndex.from_path(log_path, references)
    samples: dict[str, list[tuple[float, Any]]] = {}
    for name in references:
        resolution = index.resolve_signal(name)
        if resolution.status != "observed" or resolution.series is None:
            return {"error": f"{name} not observed in log", "windows": None}
        samples[name] = [
            (sample.time_s, sample.value) for sample in resolution.series.samples
        ]
    events = derive_transition_events(transition, samples)
    event = select_transition_event(
        events, getattr(transition, "event_selection", None)
    )
    if event is None:
        if not events:
            return {"error": "no logged transition matches the specification", "windows": None}
        return {
            "error": "multiple logged transitions match without explicit selection",
            "windows": None,
        }
    windows = derive_diagnostic_windows(transition, event, samples)
    if not windows:
        return {"error": "no diagnostic window follows the matched transition", "windows": None}
    return {
        "signal": target or signal,
        "transition_signal": signal,
        "event": event,
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
    scope: Optional[EvaluationScope] = None,
) -> dict[str, Any]:
    """Numerically compare terminal writes with their observed output.

    Producer values and writer selection are evaluated directly from DAG
    edges. Source text is used only for the local operators within one
    operation; no recursively substituted expression participates in the
    result. Each writer is compared only inside its gating branch windows.
    A definitive match/mismatch requires exact, non-overlapping writer
    domains that cover the observed output domain. A supplied scope
    restricts the comparison domain; absent scope keeps full-domain
    behavior exactly.
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
        scope=scope,
    )


_CANDIDATE_REF_PREFIX = (
    "candidate terminal write excluded by assumed feasibility condition: "
)
_HELPER_REF_PREFIX = "upstream helper contribution: "


def _data_ancestors(vertices: Any, edges: Any, roots: Any) -> set:
    """Backward data-edge ancestry of roots (edge-native, no new types)."""
    by_id = {}
    for vertex in vertices or ():
        vid = getattr(vertex, "id", None)
        if vid is not None:
            by_id[vid] = vertex
    incoming: dict[Any, list] = {}
    for edge in edges or ():
        if getattr(edge, "kind", None) != "data":
            continue
        incoming.setdefault(edge.target_id, []).append(edge.source_id)
    seen = set(roots or ())
    stack = list(seen)
    while stack:
        for source_id in incoming.get(stack.pop(), ()):
            if source_id not in seen:
                seen.add(source_id)
                stack.append(source_id)
    return seen


def _qualified_helper_entries(dag: Any) -> list:
    """Helper-return vertices supplying terminal value through a
    call/data relationship, with stable identities.

    A helper qualifies iff it is data-reachable from a terminal
    operation AND at least one outgoing data edge carries a
    ``call:`` / ``call-result:`` role into that reachable set.
    Control-only reachability never qualifies. Returns
    ``(vertex, identity)`` pairs deterministically ordered by
    identity; dedupes repeated instances by identity here.
    """
    vertices = list((dag.vertices if dag is not None else None) or ())
    edges = list((dag.edges if dag is not None else None) or ())
    terminal_ids = [
        vertex.id for vertex in vertices
        if getattr(vertex, "kind", None) == "operation"
        and isinstance(getattr(vertex, "metadata", None), dict)
        and vertex.metadata.get("is_terminal")
    ]
    ancestors = _data_ancestors(vertices, edges, terminal_ids)
    by_identity: dict[tuple, list] = {}
    for vertex in vertices:
        identity = _helper_representative_identity(vertex, dag)
        if identity is None:
            continue
        if vertex.id not in ancestors:
            continue
        flows_to_terminal = False
        for edge in edges:
            if (getattr(edge, "source_id", None) != vertex.id
                    or getattr(edge, "kind", None) != "data"):
                continue
            role = str(getattr(edge, "role", None) or "")
            if not (role.startswith("call:")
                    or role.startswith("call-result:")):
                continue
            if edge.target_id in ancestors:
                flows_to_terminal = True
                break
        if not flows_to_terminal:
            continue
        by_identity.setdefault(identity, []).append(vertex)
    chosen = []
    for identity in sorted(by_identity):
        vertex = min(
            by_identity[identity],
            key=lambda v: (str(getattr(v, "file", None) or ""),
                           getattr(v, "line", None)
                           if isinstance(getattr(v, "line", None), int)
                           else -1),
        )
        chosen.append((vertex, identity))
    return chosen


def _is_constant_writer_expression(expression: Any) -> bool:
    """Whether a writer expression is a bare constant (initializer-shaped).

    Decimal numerics (C++ float/int suffixes tolerated), quoted
    strings, and booleans count; anything naming a value or applying
    an operator does not. Structural heuristic only — the anchor
    role additionally requires a runtime cowriter for the same
    symbol (see _order_source_ref_entries), never this alone.
    """
    text = str(expression or "").strip()
    while len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if text.lower() in ("true", "false"):
        return True
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return True
    try:
        float(text.rstrip("fFlLuU"))
    except ValueError:
        return False
    return True


def _helper_representative_identity(vertex: Any, dag: Any = None) -> Optional[tuple]:
    """Stable helper identity for report dedup: (helper callable,
    stable caller SOURCE call site, result-path discriminator).

    Callable comes from the structured ``target_identity``
    declaration (``{callable}:return`` suffix stripped), never
    from diagnostic provenance strings. Call site is the §3 stable
    caller SOURCE site (never an ``invocation_`` instance hash).
    The discriminator distinguishes represented result paths
    (``__return__`` vs ``__return__.{path}``); single-return
    helpers yield ``""``. Returns None when no stable identity
    exists — callers omit rather than invent. Never vertex IDs,
    insertion order, or call-instance scopes.
    """
    metadata = getattr(vertex, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    if metadata.get("synthetic_helper_return_binding") is not True:
        return None
    target_identity = metadata.get("target_identity")
    declaration_id = (
        target_identity.get("declaration_id")
        if isinstance(target_identity, dict) else None)
    if not declaration_id or not str(declaration_id).endswith(":return"):
        return None
    helper_callable = str(declaration_id)[:-len(":return")]
    if not helper_callable:
        return None
    variable = str(getattr(vertex, "variable", None) or "")
    if variable == "__return__":
        discriminator = ""
    elif variable.startswith("__return__."):
        discriminator = variable[len("__return__."):]
    elif variable.startswith("__return__"):
        discriminator = variable[len("__return__"):]
    else:
        return None
    call_site = _stable_caller_source_site(vertex, dag)
    if not call_site:
        return None
    return (helper_callable, call_site, discriminator)


_INVOCATION_SITE_PREFIX = "invocation_"


def _is_stable_source_site(value: Any) -> bool:
    """Whether a call-site value identifies a source statement.

    ``invocation_<hash>`` values identify one runtime/graph call
    instance (nested/contextualized callers); they must never
    define report identity.
    """
    text = str(value or "").strip()
    return bool(text) and not text.startswith(_INVOCATION_SITE_PREFIX)


def _call_text_names(call_text: Any, name: str) -> bool:
    """Whether source call text invokes helper ``name``."""
    text = str(call_text or "")
    if not text or not name:
        return False
    return re.search(r"\b" + re.escape(name) + r"\s*\(", text) is not None


def _match_call_result_site(
    results: Any, *, wanted_path: str, wanted_name: str
) -> Optional[str]:
    """Stable ``call_source_site_id`` for the entry matching a helper edge.

    Match by result path first, then helper short name, then the
    single unambiguous entry. Ambiguity (or no stable entry) yields
    None — never a guess.
    """
    stable: list[dict[str, Any]] = []
    for raw in results or ():
        entry = raw if isinstance(raw, dict) else (
            raw.model_dump(exclude_none=True)
            if hasattr(raw, "model_dump") else None)
        if not isinstance(entry, dict):
            continue
        if _is_stable_source_site(entry.get("call_source_site_id")):
            stable.append(entry)
    if not stable:
        return None
    pathed = [entry for entry in stable
              if str(entry.get("result_path") or "") == wanted_path]
    if len(pathed) == 1:
        return str(pathed[0].get("call_source_site_id"))
    pool = pathed if len(pathed) > 1 else stable
    if wanted_name:
        named = [entry for entry in pool
                 if _call_text_names(entry.get("text"), wanted_name)]
        if len(named) == 1:
            return str(named[0].get("call_source_site_id"))
        if named:
            return None
    if len(stable) == 1:
        return str(stable[0].get("call_source_site_id"))
    return None


def _helper_call_edges(vertex: Any, dag: Any) -> list:
    """Outgoing ``call:``/``call-result:`` data edges of one vertex."""
    edges = list((getattr(dag, "edges", None) if dag is not None else None) or ())
    vertex_id = getattr(vertex, "id", None)
    matched = [
        edge for edge in edges
        if getattr(edge, "source_id", None) == vertex_id
        and getattr(edge, "kind", None) == "data"
        and (str(getattr(edge, "role", None) or "").startswith("call:")
             or str(getattr(edge, "role", None) or "").startswith("call-result:"))
    ]
    matched.sort(key=lambda edge: str(getattr(edge, "id", None) or ""))
    return matched


def _stable_caller_source_site(vertex: Any, dag: Any = None) -> Optional[str]:
    """Stable caller SOURCE call-site identity for one helper vertex.

    Priority per the amended spec: consumer-side
    ``call_results[].call_source_site_id`` matching the helper edge,
    then raw ``call_site_id``, edge ``via``, and
    ``source_call_roles`` keys — each only when source-stable.
    Returns None when no stable site exists: callers omit rather
    than invent (STOP I).
    """
    metadata = getattr(vertex, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    vertices = list((getattr(dag, "vertices", None) if dag is not None else None) or ())
    by_id = {getattr(item, "id", None): item for item in vertices}
    call_edges = _helper_call_edges(vertex, dag)
    for edge in call_edges:
        target = by_id.get(getattr(edge, "target_id", None))
        target_metadata = getattr(target, "metadata", None)
        if not isinstance(target_metadata, dict):
            continue
        expression_ref = target_metadata.get("source_expression_ref") or {}
        if not isinstance(expression_ref, dict):
            continue
        role = str(getattr(edge, "role", None) or "")
        if role.startswith("call-result:"):
            wanted_path, wanted_name = role[len("call-result:"):], ""
        else:
            wanted_path, wanted_name = "", role[len("call:"):] if role.startswith("call:") else ""
        site = _match_call_result_site(
            expression_ref.get("call_results"),
            wanted_path=wanted_path, wanted_name=wanted_name)
        if site is not None:
            return site
    raw_site = metadata.get("call_site_id")
    if _is_stable_source_site(raw_site):
        return str(raw_site)
    for edge in call_edges:
        if _is_stable_source_site(getattr(edge, "via", None)):
            return str(getattr(edge, "via"))
    for edge in call_edges:
        target = by_id.get(getattr(edge, "target_id", None))
        target_metadata = getattr(target, "metadata", None)
        if not isinstance(target_metadata, dict):
            continue
        roles = target_metadata.get("source_call_roles") or {}
        if not isinstance(roles, dict):
            continue
        for key in sorted(str(item) for item in roles):
            if _is_stable_source_site(key):
                return key
    return None


def _normalize_writer_expression(source: Any) -> str:
    """Deterministic spelling for writer-identity comparison.

    Prefers the already-lowered expression form when present,
    then collapses whitespace and strips a trailing semicolon.
    Parentheses and operator spellings are left to the lowered
    form; nothing here merges semantically different expressions.
    """
    mapping = source if isinstance(source, dict) else {}
    text = mapping.get("lowered_expression") or mapping.get("expression") or ""
    text = " ".join(str(text).split())
    return text[:-1].strip() if text.endswith(";") else text


def _writer_scope_identity(source: Any) -> str:
    """Most stable available scope for a writer, preference-ordered.

    Declaration identity first, then source-site identity, then
    callable scope, then target scope. Empty string when nothing
    is recorded; callers must treat it as absent, not as a scope.
    """
    mapping = source if isinstance(source, dict) else {}
    target_identity = mapping.get("target_identity")
    if isinstance(target_identity, dict):
        declaration_id = target_identity.get("declaration_id")
        if declaration_id:
            return f"decl:{declaration_id}"
    source_site_id = mapping.get("source_site_id")
    if source_site_id:
        return f"site:{source_site_id}"
    for key in ("function", "callable", "callable_id"):
        value = mapping.get(key)
        if value:
            return f"call:{value}"
    target_scope = mapping.get("target_scope")
    if isinstance(target_scope, dict):
        scoped = (str(target_scope.get("file") or ""),
                  str(target_scope.get("callable") or ""))
        if any(scoped):
            return f"scope:{scoped[0]}:{scoped[1]}"
    return ""


def _source_writer_identity(source: Any) -> tuple:
    """Stable source-writer identity for dedup/ordering.

    Collapses repeat graph instances of one source writer while
    keeping genuinely different assignments distinct. Instance-only
    metadata (call-instance scopes, source order, edge roles) never
    participates; neither do generated vertex/op IDs or graph
    insertion order. All components tolerate absence.
    """
    mapping = source if isinstance(source, dict) else {}
    line = mapping.get("line")
    end_line = mapping.get("end_line")
    call_site = mapping.get("call_site_id")
    return (
        str(mapping.get("file") or ""),
        line if isinstance(line, int) else None,
        end_line if isinstance(end_line, int) else None,
        str(mapping.get("variable") or "").strip(),
        _normalize_writer_expression(mapping),
        _writer_scope_identity(mapping),
        str(call_site) if call_site else "",
    )


def _entry_source(entry: Any) -> dict[str, Any]:
    """Flattened identity view for one terminal-write entry.

    Vertex-level fields win; missing provenance degrades through
    the same fallback chain as retained records, so both sides
    compare on equal terms.
    """
    mapping = entry if isinstance(entry, dict) else {}
    metadata = mapping.get("metadata")
    merged: dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}
    for key in ("file", "line", "end_line", "variable", "expression",
                "lowered_expression"):
        value = mapping.get(key)
        if value is not None:
            merged[key] = value
    return merged


def _order_rank(source: Any) -> tuple:
    """Deterministic source-order rank: recorded emission order,
    then source-site order, then unordered. Total and type-safe."""
    mapping = source if isinstance(source, dict) else {}
    order = mapping.get("source_order")
    if isinstance(order, int):
        return (0, order)
    site_order = _source_site_order(mapping.get("source_site_id"))
    if site_order is not None:
        return (1, site_order)
    return (2, 0)


def _order_key(source: Any, file: str, line: Any) -> tuple:
    """Total deterministic order key: emission/site order first,
    then stable location and identity tail. Every element is
    mutually comparable; no insertion order, no vertex IDs."""
    line_key = f"{line:012d}" if isinstance(line, int) else ""
    identity = _source_writer_identity(source)
    tail = []
    for part in identity[2:]:
        if part is None:
            tail.append("")
        elif isinstance(part, int):
            tail.append(f"{part:012d}")
        else:
            tail.append(str(part))
    return (
        _order_rank(source) + (file, line_key) + tuple(tail))


def _retained_candidate_ref(record: dict[str, Any]) -> Optional[CodeRef]:
    """Honestly-marked CodeRef for one retained assumed-pruned
    terminal writer, or None when the record cannot identify the
    writer. Candidate wording replaces (never extends) ordinary
    terminal-write wording."""
    file = str(record.get("file") or "")
    if not file:
        return None
    variable = str(record.get("variable") or "")
    expression = str(record.get("expression") or "")
    line = record.get("line")
    location = f"{file}:{line}" if isinstance(line, int) else file
    end_line = record.get("end_line")
    return CodeRef(
        file=file,
        function=(str(record.get("callable") or "") or None),
        start_line=line if isinstance(line, int) else None,
        end_line=end_line if isinstance(end_line, int) else None,
        snippet=None,
        explanation=(
            f"{_CANDIDATE_REF_PREFIX}{variable} <- "
            f"{expression} [{location}]"
        ),
    )


def _helper_candidate_ref(vertex: Any) -> Optional[CodeRef]:
    """Honestly-marked CodeRef for one qualified helper
    representative, or None when it has no reportable file.
    Wording is its own semantic class, distinct from terminal
    writes and assumed candidates alike."""
    file = str(getattr(vertex, "file", None) or "")
    if not file:
        return None
    variable = str(getattr(vertex, "variable", None) or "")
    expression = str(getattr(vertex, "expression", None) or "")
    line = getattr(vertex, "line", None)
    location = f"{file}:{line}" if isinstance(line, int) else file
    return CodeRef(
        file=file,
        start_line=line if isinstance(line, int) else None,
        end_line=line if isinstance(line, int) else None,
        snippet=getattr(vertex, "snippet", None),
        explanation=(
            f"{_HELPER_REF_PREFIX}{variable} <- "
            f"{expression} [{location}]"
        ),
    )


def _order_source_ref_entries(
    entries: list[dict[str, Any]],
    retained: Any,
    selected_terminal: str,
    dag: Any = None,
    temporal_helper_keep: Optional[frozenset] = None,
) -> list[CodeRef]:
    """Order surviving terminal refs with retained assumed-pruned
    candidates per the causal-evidence contract: runtime
    computation writers before declaration/storage anchors;
    surviving derived/proven writers before assumed candidates;
    deterministic source order as tiebreak. Both surviving entries
    and retained records dedup by stable source-writer identity
    (surviving wins cross-dedup ties). When ``temporal_helper_keep``
    is not None, helper representatives outside the keep set (chosen
    by report-adjacent temporal selection) are omitted; None keeps
    existing behavior exactly.
    """
    unique: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for entry in entries:
        key = _source_writer_identity(_entry_source(entry))
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    runtime_vars = {
        str(entry.get("variable") or "") for entry in unique
        if not _is_constant_writer_expression(entry.get("expression"))
    }
    runtime_vars.update(
        str(record.get("variable") or "") for record in (retained or ())
        if isinstance(record, dict) and record.get("assumed", False)
        and str(record.get("file") or "")
        and (not selected_terminal
             or str(record.get("variable") or "") == selected_terminal)
    )
    surviving_keys = {
        _source_writer_identity(_entry_source(entry))
        for entry in unique
    }
    ranked: list[tuple[tuple, CodeRef]] = []
    used_locations: set[tuple] = set()
    for entry in unique:
        source = _entry_source(entry)
        anchor = (
            _is_constant_writer_expression(entry.get("expression"))
            and str(entry.get("variable") or "") in runtime_vars
        )
        if isinstance(entry.get("line"), int):
            used_locations.add(
                (str(entry.get("file") or ""), entry.get("line")))
        ranked.append((
            (1 if anchor else 0, 0,
             *_order_key(source, str(entry.get("file") or ""),
                         entry.get("line"))),
            entry["ref"],
        ))
    for record in (retained or ()):
        if not isinstance(record, dict):
            continue
        if not record.get("assumed", False):
            continue
        if not str(record.get("file") or ""):
            continue
        if (selected_terminal
                and str(record.get("variable") or "") != selected_terminal):
            continue
        key = _source_writer_identity(record)
        if key in surviving_keys:
            continue
        surviving_keys.add(key)
        ref = _retained_candidate_ref(record)
        if ref is None:
            continue
        if isinstance(record.get("line"), int):
            used_locations.add(
                (str(record.get("file") or ""), record.get("line")))
        ranked.append((
            (0, 1,
             *_order_key(record, str(record.get("file") or ""),
                         record.get("line"))),
            ref,
        ))
    for vertex, _identity in _qualified_helper_entries(dag):
        # Physical-statement overlap: one source line yields one ref
        # even when branch duplication gives the same call several
        # opaque per-instance site ids. Terminal wording wins ties.
        if temporal_helper_keep is not None and _identity not in temporal_helper_keep:
            continue
        location = (str(getattr(vertex, "file", None) or ""),
                    getattr(vertex, "line", None))
        if location in used_locations:
            continue
        used_locations.add(location)
        ref = _helper_candidate_ref(vertex)
        if ref is None:
            continue
        ranked.append((
            (0, 2,
             *_order_key(
                 {"file": getattr(vertex, "file", None),
                  "line": getattr(vertex, "line", None),
                  "variable": getattr(vertex, "variable", None),
                  "expression": getattr(vertex, "expression", None),
                  "metadata": (getattr(vertex, "metadata", None)
                               if isinstance(
                                   getattr(vertex, "metadata", None), dict)
                               else {})},
                 str(getattr(vertex, "file", None) or ""),
                 getattr(vertex, "line", None))),
            ref,
        ))
    ranked.sort(key=lambda item: item[0])
    return [ref for _, ref in ranked]


def _temporal_report_selection(
    dag: Any,
    scope: Optional[EvaluationScope],
    replay: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Report-adjacent temporal selection over represented candidates.

    Returns None when temporal selection is inactive (no scope), in
    which case callers keep existing behavior exactly. Otherwise
    returns ``eligible_terminal_ids`` / ``eligible_helper_identities``
    keep-sets (helpers never participate in uniqueness: they explain
    terminal values per WS2 tiers rather than rivaling writers),
    ``unique_terminal_id`` (or None), and deterministic ``notes`` for
    the existing unresolved-evidence path. A scope without diagnostic
    windows fails closed to notes with no filtering, preserving
    broader evidence.
    """
    from flight_log_agent.analysis.temporal_selection import (
        control_gate_windows_for,
        discriminate_candidates,
        temporally_eligible,
        writer_domain,
    )

    if scope is None:
        return None
    windows = [
        (float(start), float(end))
        for start, end in (getattr(scope, "windows", None) or ())
    ]
    if not windows:
        reason = str(getattr(scope, "error", None) or "")
        return {
            "eligible_terminal_ids": frozenset(),
            "eligible_helper_identities": frozenset(),
            "unique_terminal_id": None,
            "filter_active": False,
            "notes": [
                "temporal diagnostic window unavailable"
                + (f": {reason}" if reason else "")
                + "; no temporally-selected causal claim",
            ],
        }
    replay_domains: dict[str, list] = {}
    replay_supported: dict[str, bool] = {}
    for result in ((replay or {}).get("results", None) or ()):
        if not isinstance(result, dict):
            continue
        op_id = result.get("operation_id")
        if not op_id:
            continue
        domain = [
            (float(start), float(end))
            for start, end in (result.get("active_windows") or ())
        ]
        replay_domains[str(op_id)] = domain
        fraction = result.get("match_fraction")
        replay_supported[str(op_id)] = bool(
            result.get("evaluable")
            and fraction is not None
            and float(fraction) >= 0.95
            and temporally_eligible(domain, windows)
        )
    # 0.95 mirrors the replay matched rule in dag_replay: a complete
    # piecewise replay counts as matched at that fraction. Only
    # already-linked per-writer results participate; nothing here
    # invents numeric-check ↔ operation linkage.
    vertices = list((getattr(dag, "vertices", None) if dag is not None else None) or ())

    def domain_of(vertex_id: str) -> list:
        gates = control_gate_windows_for(dag, vertex_id)
        return writer_domain(
            replay_domain=replay_domains.get(vertex_id),
            branch_gates=gates,
            scope_windows=windows,
        )

    runtime_ids: list[str] = []
    anchor_ids: list[str] = []
    for vertex in vertices:
        if getattr(vertex, "kind", None) != "operation":
            continue
        if not getattr(vertex, "file", None):
            continue
        if not isinstance(getattr(vertex, "metadata", None), dict):
            continue
        if not vertex.metadata.get("is_terminal"):
            continue
        if not temporally_eligible(domain_of(vertex.id), windows):
            continue
        if _is_constant_writer_expression(
            str(getattr(vertex, "expression", None) or "")
        ):
            # Declaration/storage anchors pass through eligibility;
            # they support rather than rival runtime writers and
            # never join the uniqueness universe.
            anchor_ids.append(vertex.id)
        else:
            runtime_ids.append(vertex.id)
    eligible_helper_identities: set = set()
    for vertex, identity in _qualified_helper_entries(dag):
        if temporally_eligible(domain_of(vertex.id), windows):
            eligible_helper_identities.add(identity)
    notes: list[str] = []
    unique_terminal_id = None
    eligible_terminal_ids = set(anchor_ids)
    if runtime_ids:
        outcome = discriminate_candidates(
            [
                {
                    "key": vertex_id,
                    "domain": domain_of(vertex_id),
                    "replay_match": (
                        replay_supported[vertex_id]
                        if vertex_id in replay_supported
                        else None
                    ),
                }
                for vertex_id in runtime_ids
            ],
            windows,
        )
        eligible_terminal_ids = set(outcome["eligible"]) | set(anchor_ids)
        unique_terminal_id = outcome["unique"]
        if outcome["unresolved"] is not None:
            if not outcome["eligible"] and eligible_helper_identities:
                notes.append(
                    "no terminal writer overlaps the diagnostic "
                    "window; retained helper evidence is not a "
                    "selected causal writer"
                )
            else:
                notes.append(outcome["unresolved"])
    return {
        "eligible_terminal_ids": frozenset(eligible_terminal_ids),
        "eligible_helper_identities": frozenset(eligible_helper_identities),
        "unique_terminal_id": unique_terminal_id,
        "filter_active": True,
        "notes": notes,
    }


def build_report_from_dag(
    question: str,
    judged: JudgedDiscovery,
    annotated_dag: Optional[MechanismDAG],
    replay: Optional[dict[str, Any]] = None,
    scope: Optional[EvaluationScope] = None,
) -> FlightLogReport:
    """Deterministic FlightLogReport from the verdict + annotated DAG.

    Complete deterministic replay can establish ``high`` confidence;
    partial numeric replay can establish at most ``medium``. Structural
    evidence alone remains low. Branch feasibility maps to applicability
    (``always_true`` -> supported, ``always_false`` -> excluded,
    ``unknown`` -> unresolved). A supplied temporal scope pre-filters
    source candidates by diagnostic-window eligibility before the
    existing normalization; absent scope keeps behavior exactly.
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
    # Candidate terminal-write entries collected alongside surviving
    # refs: each maps a CodeRef to the role/assumed/order facts the
    # Workstream A selection below needs. CodeRefs alone cannot carry
    # variable identity or assumed status.
    terminal_write_entries: list[dict[str, Any]] = []
    temporal = _temporal_report_selection(dag, scope, replay)

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
            if temporal is not None and temporal["filter_active"]:
                if vertex.id not in temporal["eligible_terminal_ids"]:
                    continue
                if (
                    temporal["unique_terminal_id"] is not None
                    and vertex.id != temporal["unique_terminal_id"]
                    and not _is_constant_writer_expression(
                        str(vertex.expression or "")
                    )
                ):
                    continue
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
            terminal_write_entries.append({
                "ref": CodeRef(
                    file=str(vertex.file),
                    start_line=vertex.line,
                    end_line=vertex.line,
                    snippet=vertex.snippet,
                    explanation=f"terminal write: {vertex.variable} <- {vertex.expression}",
                ),
                "variable": str(vertex.variable or ""),
                "expression": str(vertex.expression or ""),
                "lowered_expression": getattr(vertex, "lowered_expression", None),
                "file": str(vertex.file or ""),
                "line": vertex.line,
                "end_line": getattr(vertex, "end_line", None),
                "metadata": dict(vertex.metadata or {}),
                "assumed": False,
            })

    source_refs.extend(
        _order_source_ref_entries(
            terminal_write_entries,
            (getattr(dag, "assumed_pruned_provenance", None) or ()
             if dag is not None else ()),
            str((verdict.selected_terminal if verdict is not None else "") or ""),
            dag=dag,
            temporal_helper_keep=(
                temporal["eligible_helper_identities"]
                if temporal is not None and temporal["filter_active"]
                else None
            ),
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
    if temporal is not None:
        unresolved_evidence.extend(temporal["notes"])
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
                # Diagnostic observer path: keep replay disabled so
                # observer summaries retain their previous no-replay
                # behavior regardless of requirement mix.
                attempt_replay=False,
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
                # Diagnostic observer path: frozen without replay, as above.
                attempt_replay=False,
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
    temporal_scope: Optional[EvaluationScope] = None
    transition = (
        judged.seeds.questioned_condition.transition
        if judged.seeds.questioned_condition is not None
        else None
    )
    if transition is not None:
        from flight_log_agent.analysis.temporal_selection import (
            intersect_diagnostic_windows,
        )

        # Temporal seeds only: derive transition-relative diagnostic
        # windows from logged observations. Absent spec leaves replay
        # and report paths exactly as before.
        evaluated = evaluate_transition_windows(
            transition,
            logged_set=logged_set,
            log_path=log_path,
            signal_policies=signal_policies or {},
        )
        if evaluated.get("windows"):
            windows = evaluated["windows"]
            if scope is not None and scope.windows:
                windows = intersect_diagnostic_windows(
                    scope.windows, windows
                )
            temporal_scope = EvaluationScope(
                windows=tuple(windows),
                signal=str(evaluated.get("signal") or ""),
            )
        else:
            temporal_scope = EvaluationScope(
                windows=None,
                signal="",
                error=evaluated.get("error")
                or "transition window unresolved",
            )
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
            scope=(
                temporal_scope
                if temporal_scope is not None and temporal_scope.windows
                else None
            ),
        )

    report = build_report_from_dag(
        question, judged, annotated, replay=replay, scope=temporal_scope
    )
    return DagStageResult(
        judged=judged,
        annotated_dag=annotated,
        render=render_discovery_compact(selected) if selected else {},
        layer4_hit=False,
        report=report,
        replay=replay,
        checkpoint_rounds=checkpoint_rounds,
    )
