"""Judge-LLM checkpoints for mechanism discovery (#73 Stage 3).

Exactly two LLM touchpoints, neither inside the fixpoint loop:

* **Seeder** — user question → grep-able search seeds + candidate
  terminal symbols. Runs once, before discovery.
* **Judge** — compact rendering of the discovered DAG(s) → sufficiency
  verdict, terminal selection, and unresolved diagnostics. Runs after
  fixpoint convergence; only a validated replacement terminal can trigger
  one additional render and judgment.

The judge consumes :func:`render_discovery_compact` — profile *facts*,
not raw source snippets. Everything else in this module is deterministic
and unit-testable; the LLM runner is injectable so tests stub it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Union,
)

from agents import Agent
from pydantic import BaseModel, Field

from flight_log_agent.expression_math import is_safe_math_function_name

from flight_log_agent.analysis.mechanism_discovery import (
    DAGInputs,
    DiscoveryResult,
    SourceSurvey,
    TerminalValidation,
    discover_mechanism_dag,
    survey_for_queries,
    survey_source_files,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.utils import dedupe_keep_order


SEEDER_ADAPTER_VERSION = "question-intent-authoritative-v1"


# ---------------------------------------------------------------------------
# Compact DAG rendering (the judge's input)
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = 240) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _capped(items: list[Any], cap: int) -> list[Any]:
    if len(items) <= cap:
        return items
    return items[:cap] + [f"… +{len(items) - cap} more"]


def _render_site_key(vertex: Any) -> tuple[Any, ...]:
    """Identity of the SOURCE SITE a vertex stands for.

    The DAG keeps one vertex per call instance, so a single source site becomes
    many vertices (a helper body is instantiated per call site). The judge
    reasons about source mechanism, not call instances, and must name one
    stable branch id — so the rendering projects instances onto their site.
    Deliberately excludes the per-instance scope that distinguishes them.
    """
    if vertex.kind == "operation":
        return (
            "op",
            str(vertex.file or ""),
            int(vertex.line or 0),
            str(vertex.variable or ""),
            str(vertex.expression or ""),
        )
    if vertex.kind == "branch":
        predicate = str(vertex.predicate_raw or vertex.predicate_lowered or "")
        return (
            "br",
            str(vertex.file or ""),
            int(vertex.line or 0),
            " ".join(predicate.split()),
        )
    return ("ev", str(vertex.sub_kind or ""), str(vertex.signal_name or ""))


def _merge_windows(
    windows: Iterable[Sequence[float]],
) -> list[list[float]]:
    """Union of active intervals across the instances of one source site."""
    ordered = sorted(
        (float(window[0]), float(window[1]))
        for window in windows
        if window is not None and len(tuple(window)) == 2
    )
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _join_verdicts(verdicts: Sequence[str]) -> str:
    """Feasibility of a source site across its instances.

    Dead only when EVERY instance is dead, always-true only when every instance
    is — otherwise the site fired on some path and stays ``unknown``. This keeps
    the report's cross-check honest: a site is never presented as dead while one
    of its instances was alive.
    """
    present = {str(value or "unknown") for value in verdicts} or {"unknown"}
    if present == {"always_false"}:
        return "always_false"
    if present == {"always_true"}:
        return "always_true"
    return "unknown"


def render_discovery_compact(
    result: DiscoveryResult,
    *,
    dag: Optional[Any] = None,
    # Site grouping makes the rendering the size of the mechanism, so these
    # valves no longer fire on a real slice (a measured real-source mechanism
    # renders ~440 operation sites). They stay as a guard against a
    # pathological graph, well above any legitimate slice.
    max_operations: int = 4000,
    max_branches: int = 4000,
    max_evidence: int = 4000,
    max_unresolved: int = 4000,
) -> dict[str, Any]:
    """JSON-able compact view of one discovery result.

    Operations render with stable IDs and explicit adjacency; evidence
    groups by kind; the per-round trace shows how the fixpoint converged
    so the judge can see whether gaps were shrinking or churning.
    ``dag`` (optional) substitutes a feasibility-annotated graph for
    ``result.dag`` — branches then carry flight-data verdicts and active
    windows.

    The ``max_*`` limits are SAFETY VALVES against pathological graphs,
    not curation — the judge must see the whole slice (a starved render
    measurably produced wrong-shaped verdicts). Truncation is always
    marked with an explicit ``… +N more`` tail.

    Two projections keep the rendering the size of the MECHANISM rather than
    of the traversal, without dropping anything the judge is asked to name:

    * vertices are grouped by SOURCE SITE (:func:`_render_site_key`) — the DAG
      instantiates a helper body once per call site, so one source branch
      otherwise arrives as many ids and the judge cannot name "the" explaining
      branch. Each entry keeps a real DAG vertex id (so the report's
      cross-check still resolves it) and lists its sibling ``instances``.
    * adjacency is serialized once, in the direction the judge reasons
      (``inputs``/``controls``); the forward ``feeds`` of an operation is the
      exact inverse of its consumers' ``inputs`` and is therefore redundant.

    The DAG itself keeps full per-instance fidelity — this is a projection for
    the judge, not a change to the graph the value engine evaluates.
    """
    dag = dag if dag is not None else result.dag
    validation = getattr(result, "terminal_validation", None)
    scoped = {vertex.id for vertex in dag.vertices} if dag is not None else set()
    site_groups: dict[tuple[Any, ...], list[Any]] = {}
    for vertex in dag.vertices if dag else []:
        if vertex.id in scoped:
            site_groups.setdefault(_render_site_key(vertex), []).append(vertex)
    representative: dict[tuple[Any, ...], Any] = {}
    for key, members in site_groups.items():
        if key[0] == "br":
            # A site is only dead when every instance is: prefer a live
            # instance so the report's feasibility cross-check is not
            # downgraded by an arbitrary dead sibling.
            representative[key] = min(
                members,
                key=lambda item: (
                    item.feasibility_verdict == "always_false",
                    -len(item.active_windows or ()),
                    item.id,
                ),
            )
        else:
            representative[key] = min(members, key=lambda item: item.id)
    rep_id: dict[str, str] = {}
    for key, members in site_groups.items():
        for member in members:
            rep_id[member.id] = representative[key].id
    rendered_ids = {vertex.id for vertex in representative.values()}

    # Branch ids stay the real DAG ids: the judge copies them into
    # explaining_branches and the report resolves them against the graph.
    # Operations and evidence are referenced only from adjacency, so they carry
    # short render-local ids — the same graph, a cheaper spelling.
    render_id: dict[str, str] = {}
    operation_index = evidence_index = 0
    for vertex in dag.vertices if dag else []:
        if vertex.id not in rendered_ids:
            continue
        if vertex.kind == "branch":
            render_id[vertex.id] = vertex.id
        elif vertex.kind == "operation":
            operation_index += 1
            render_id[vertex.id] = f"o{operation_index}"
        else:
            evidence_index += 1
            render_id[vertex.id] = f"e{evidence_index}"

    def _ref(vertex_id: str) -> str:
        target = rep_id.get(vertex_id, vertex_id)
        return render_id.get(target, target)

    def _instances(key: tuple[Any, ...]) -> dict[str, Any]:
        members = site_groups.get(key, [])
        if len(members) <= 1:
            return {}
        # The count discloses that the site is instantiated per call site; the
        # instance ids themselves are recoverable from the graph by site and are
        # never named by the judge.
        return {"instances": len(members)}
    operations: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    helper_returns: list[str] = []
    call_heads: set[str] = set()
    evidence: dict[str, list[dict[str, Any]]] = {}
    incoming_data: dict[str, list[dict[str, str]]] = {}
    incoming_controls: dict[str, list[str]] = {}
    incoming_selections: dict[str, list[str]] = {}
    control_outgoing: dict[str, list[str]] = {}
    selection_outgoing: dict[str, list[str]] = {}
    seen_edges: set[tuple[str, str, str, str]] = set()
    for edge in dag.edges if dag else []:
        if edge.source_id not in rep_id or edge.target_id not in rep_id:
            continue
        source = _ref(edge.source_id)
        target = _ref(edge.target_id)
        role = str(edge.role or "")
        signature = (source, target, edge.kind, role)
        if signature in seen_edges:
            continue
        seen_edges.add(signature)
        if edge.kind == "data":
            incoming_data.setdefault(target, []).append(
                {"source_id": source, "role": role}
            )
        elif edge.kind == "control":
            incoming_controls.setdefault(target, []).append(source)
            control_outgoing.setdefault(source, []).append(target)
        elif edge.kind == "selection":
            incoming_selections.setdefault(target, []).append(source)
            selection_outgoing.setdefault(source, []).append(target)

    for vertex in dag.vertices if dag else []:
        if vertex.id not in rendered_ids:
            continue
        if vertex.kind == "operation":
            local_id = render_id[vertex.id]
            entry = {
                "id": local_id,
                "target": str(vertex.variable or ""),
                "expression": _truncate(vertex.expression or ""),
                "file": vertex.file,
                "line": vertex.line,
                "inputs": incoming_data.get(local_id, []),
                "controls": incoming_controls.get(local_id, []),
                "value_selectors": incoming_selections.get(local_id, []),
                # Only whether the write's reachability is fully known; the
                # gating predicates themselves are the `controls` edges.
                "reachability_exact": bool(
                    ((vertex.metadata or {}).get("reachability") or {}).get(
                        "exact", False
                    )
                ),
                **_instances(_render_site_key(vertex)),
            }
            if vertex.provenance and vertex.provenance.startswith("helper_return"):
                helper_returns.append(str(vertex.variable))
            for head in re.findall(
                r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(", str(vertex.expression or "")
            ):
                if not is_safe_math_function_name(head):
                    call_heads.add(head)
            operations.append(entry)
        elif vertex.kind == "branch":
            predicate = vertex.predicate_lowered or vertex.predicate_raw or ""
            site_key = _render_site_key(vertex)
            members = site_groups.get(site_key, [vertex])
            branches.append(
                {
                    "id": vertex.id,
                    "predicate": _truncate(predicate),
                    # Aggregated over the site's instances: dead only when every
                    # instance is dead, windows are the union of when any
                    # instance was active.
                    "feasibility": _join_verdicts(
                        [item.feasibility_verdict for item in members]
                    ),
                    "active_windows": _merge_windows(
                        window
                        for item in members
                        for window in (item.active_windows or ())
                    ),
                    "evaluation_domain": (vertex.metadata or {}).get("evaluation_domain"),
                    "gates": sorted(set(control_outgoing.get(vertex.id, []))),
                    "selects_values_for": sorted(
                        set(selection_outgoing.get(vertex.id, []))
                    ),
                    "inputs": incoming_data.get(vertex.id, []),
                    **_instances(site_key),
                }
            )
        elif vertex.kind == "evidence":
            kind = str(vertex.sub_kind or "other")
            metadata = vertex.metadata or {}
            evidence.setdefault(kind, []).append(
                {
                    "id": render_id[vertex.id],
                    "signal": str(vertex.signal_name or ""),
                    "value": metadata.get("value"),
                    "observation": metadata.get("observation"),
                    "boundary": metadata.get("boundary"),
                    **_instances(_render_site_key(vertex)),
                }
            )

    return {
        "terminal": dag.terminal if dag else None,
        **(
            {
                "terminal_validation": {
                    "status": validation.status,
                    "reason": validation.reason,
                }
            }
            if validation is not None and validation.status != "valid"
            else {}
        ),
        "vertices": len(dag.vertices) if dag else 0,
        "edges": len(dag.edges) if dag else 0,
        "operations": _capped(operations, max_operations),
        "branches": _capped(branches, max_branches),
        "helper_subgraphs": sorted(set(helper_returns)),
        # Call heads in operation expressions with no expanded subgraph —
        # what the judge can name in expand_calls.
        "unexpanded_calls": _capped(
            sorted(
                call_heads
                - {name.split("::")[-2] for name in helper_returns if "::" in name}
            ),
            max_unresolved,
        ),
        "evidence": {
            kind: _capped(sorted(items, key=lambda item: str(item.get("id") or "")), max_evidence)
            for kind, items in sorted(evidence.items())
        },
        "unresolved_symbols": _capped(
            list(dag.unresolved_symbols) if dag else [], max_unresolved
        ),
        "files_loaded": result.files_loaded,
        "rounds": [
            {
                "index": r.index,
                "new_files": r.new_files,
                "vertices": r.vertices,
                "edges": r.edges,
                "unresolved": len(r.unresolved_symbols),
            }
            for r in result.rounds
        ],
    }


# ---------------------------------------------------------------------------
# Agent output models
# ---------------------------------------------------------------------------


class TerminalCandidate(BaseModel):
    """One candidate terminal: the C++ symbol that holds the questioned
    quantity, plus the file expected to define/write it."""

    terminal: str
    terminal_file: Optional[str] = None
    source_target_id: Optional[str] = None
    reason: str = ""


class QuestionedCondition(BaseModel):
    """The comparison the question asserts, when it asserts one —
    evaluated deterministically over the log so the judge can compare
    deviation windows against branch active-windows. ``signal_hint`` and
    ``reference`` may be arithmetic expressions over exact logged signal
    references, ULog parameters, and numeric literals."""

    signal_hint: str
    op: Literal[">", ">=", "<", "<=", "==", "!="]
    reference: str
    units: str
    frame: str
    assumptions: list[str] = Field(default_factory=list)


class DiscoverySeeds(BaseModel):
    """Seeder output: grep-able queries + candidate terminals."""

    seeds: list[str]
    candidate_terminals: list[TerminalCandidate]
    questioned_condition: Optional[QuestionedCondition] = None
    notes: list[str] = Field(default_factory=list)


class DiscoveryVerdict(BaseModel):
    """Judge output over the compact DAG rendering(s)."""

    sufficient: bool
    selected_terminal: str
    essential_gaps: list[str] = Field(default_factory=list)
    expand_calls: list[str] = Field(default_factory=list)
    next_terminals: list[TerminalCandidate] = Field(default_factory=list)
    explaining_branches: list[str] = Field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


seeder_agent = Agent(
    name="Mechanism Discovery Seeder",
    model="gpt-5.5",
    instructions="""
Convert an authoritative normalized PX4 question intent into inputs for
deterministic source discovery. Do not reinterpret or broaden the intent.

Output:
- seeds: concrete grep-able queries for the PX4 source tree — exact
  parameter names, class or method names, uORB topic or field names
  drawn from the question and context. Prefer specific over generic
  terms.
- candidate_terminals: the C++ symbols (class members or locals) that
  HOLD the quantity the question asks about, each with the source file
  expected to write it. Order by likelihood; keep the list concise because
  each accepted terminal increases the downstream LLM payload and API cost.

source_survey is the authority on what exists. Its anchors are the
identifiers the question names AND the pinned tree confirms (declared
parameters, observed signals, defined callables). Under each real file
it lists the writes those anchors reach through source dataflow, each
with the assignment expression, the writing function, and
reaches_anchor_in_hops (0 = the write itself reads an anchor — the
decision site; 1 = one dataflow step away). When source_survey lists
any write target:

- every candidate_terminals entry MUST be a symbol the survey lists,
  spelled exactly as the survey spells it;
- source_target_id MUST be copied exactly from that same surveyed write;
- terminal_file MUST be the surveyed file that writes that symbol;
- never propose a symbol or a file path the survey does not show,
  however familiar it seems — recalled names and module layouts often
  belong to a different release than the pinned tree, and a terminal
  that does not exist yields no mechanism at all;
- read the expressions: a hop-0 write whose expression combines the
  anchored parameter with other quantities is usually the decision
  site the question is about.

rejected_terminals (when present) maps earlier proposals to the reason
deterministic validation refused them. Do not repeat those symbols;
choose different survey targets.

Each terminal must be the bare variable name exactly as it appears on
the LEFT side of its assignment in source — never class-qualified
(``ClassName::member``) and never type-qualified
(``struct_type_s::field``).

Prefer the variable written at the DECISION SITE — where the questioned
quantity is computed or adapted — over the published topic field that
merely logs it; list the published field as a secondary candidate.

If the question asserts a comparison (a quantity above/below/equal to a
reference), fill questioned_condition with the LOGGED signal that records
the quantity (topic.field), the comparison operator, and the reference. The
reference may be a parameter, numeric literal, or an arithmetic expression
composed only from logged topic.field signals and parameters justified by the
question/context. State both side units as
"signal: <unit>; reference: <unit>" and state the coordinate/reference frame;
use "unitless" or "not_applicable" only when those concepts genuinely do not
apply. Record any assumption explicitly. It is evaluated over the log to find
when the questioned behavior actually occurred.

Do not use log data, do not verify anything, do not draft hypotheses.
""",
    tools=[],
    output_type=DiscoverySeeds,
)


judge_agent = Agent(
    name="Mechanism DAG Judge",
    model="gpt-5.5",
    instructions="""
Judge whether a discovered mechanism DAG grounds the user's question.

Input: the question plus one compact DAG rendering per candidate
terminal — operations (target <- expression @ file:line), branch
predicates, evidence leaves (logged signals, parameters, constants),
unresolved symbols, and the per-round discovery trace.

Decide from these facts only:
- sufficient: does the selected DAG connect the terminal to logged
  signals / parameters / constants well enough to answer the question?
- selected_terminal: which candidate terminal matches the question.
  Copy the exact key from the candidates object; scoped same-named locals
  have distinct keys.
- essential_gaps: unresolved symbols that MUST be resolved to answer
  (they carry the questioned quantity or gate it). Everything else —
  bookkeeping counters, foreign-module noise, display-only values — is
  ignorable and must not be listed.
- expand_calls: entries from the rendering's unexpanded_calls whose
  body must be inlined to answer (the function computing or adapting
  the questioned quantity).
- next_terminals: when NO given candidate holds the questioned quantity
  at its decision site, name a better bare variable (with file and surveyed
  source_target_id when available) from the rendering's operations.

When sufficient is false, use essential_gaps and expand_calls as diagnostics
over the completed fixed-point graph. Fill next_terminals only when a shown
source write proves that the selected terminal is wrong; one validated
replacement terminal may be followed up.

Branch entries contain a stable id, predicate, flight-data feasibility,
and the exact active_windows. always_false means the predicate never held
in THIS flight — treat that path as inactive and do NOT demand its
grounding; always_true held throughout. For questions about
behavior that occurs only sometimes, prefer the branch whose active
windows can explain WHEN it occurred.

questioned_windows (when present) gives the time intervals where the
QUESTIONED condition itself held in the log. The explaining branch's
active windows should overlap them; a branch active only outside them
cannot be the answer.

When sufficient is true you MUST fill explaining_branches with the stable
branch id(s), copied exactly from the rendering's branches, whose taking
explains the questioned behavior. They are cross-checked against
flight-data feasibility — a mechanism whose explaining branch never
fired cannot be the answer.

essential_gaps and expand_calls entries are BARE symbol or function
names copied from unresolved_symbols / unexpanded_calls / operation
expressions — never sentences; they are report diagnostics, not file-
admission instructions. empty_candidates lists terminals whose slice found nothing:
never select those; if no shown candidate holds the questioned quantity,
propose a replacement in next_terminals taken from the operations of a
non-empty rendering.

rejected_terminals maps candidates that deterministic validation
refused to slice to the reason: no write target in the loaded source,
write targets ambiguous across modules, or absent from the declared
file. Never select them either. When proposing next_terminals, always
include terminal_file (the file whose operations show the write) so
validation can scope the slice.

next_terminals entries must name a symbol that the rendering's
operations show being written, or a write target listed in
source_survey when one is given. A symbol that appears in neither does
not exist in the pinned source and yields no mechanism.

Never request raw source; never speculate beyond the rendering.
""",
    tools=[],
    output_type=DiscoveryVerdict,
)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


AgentRunnerFn = Callable[[Agent, dict[str, Any]], Awaitable[Any]]


async def _default_run_agent(agent: Agent, payload: dict[str, Any]) -> Any:
    from agents import Runner

    result = await Runner.run(
        agent,
        input=json.dumps(payload, separators=(",", ":"), default=str),
    )
    return result.final_output


@dataclass
class JudgedDiscovery:
    seeds: DiscoverySeeds
    verdict: DiscoveryVerdict
    results: dict[str, DiscoveryResult] = field(default_factory=dict)
    selected: Optional[DiscoveryResult] = None
    selected_annotated: Optional[Any] = None
    bonus_round_used: bool = False
    reseeded: bool = False


async def discover_with_judge(
    profiler: MechanismSourceProfiler,
    cache_root: Union[str, Path],
    question: str,
    source_hash: str,
    *,
    run_agent: Optional[AgentRunnerFn] = None,
    context: Optional[dict[str, Any]] = None,
    max_terminals: int = 2,
    max_terminal_attempts: int = 4,
    survey_max_files: int = 8,
    seeds_override: Optional[DiscoverySeeds] = None,
    annotate: Optional[Callable[[DiscoveryResult], Any]] = None,
    condition_windows: Optional[Callable[["QuestionedCondition"], Any]] = None,
    **discovery_kwargs: Any,
) -> JudgedDiscovery:
    """Seeder → deterministic fixpoint per candidate terminal → judge.

    ``seeds_override`` (e.g. a Layer 4 cache hit for a previously judged
    question) skips the seeder call entirely; the judge still runs.

    ``annotate`` (optional) maps a DiscoveryResult to a feasibility-
    annotated DAG; when given, the judge sees annotated renders (branch
    verdicts + active windows) so it can dismiss paths dead in THIS
    flight instead of demanding their static grounding. A replacement-
    terminal DAG is re-annotated and judged once more so the verdict is never
    stale. Terminal and retry limits deliberately bound LLM payload/API cost;
    they are not deterministic-expansion limits.

    Deterministic source expansion reaches its own fixed point before the
    judge runs. Judge-provided textual gaps do not steer file admission; only
    a source-validated replacement terminal can trigger another complete
    slice.

    ``discovery_kwargs`` pass through to
    :func:`~flight_log_agent.analysis.mechanism_discovery.discover_mechanism_dag`
    (``source_root``, ``inventory``, ``logged_signals``, and compatibility
    arguments for the retired deterministic-expansion limits).
    """
    runner = run_agent or _default_run_agent

    context_dict = dict(context or {})
    intent = dict(context_dict.get("question_intent") or {})

    intent_seeds = [
        *(str(value) for value in (intent.get("source_queries") or []) if value),
        *(str(value) for value in (intent.get("likely_modules") or []) if value),
        *(str(value) for value in (intent.get("likely_source_files") or []) if value),
    ]

    # Deterministic source survey BEFORE the seeder: the question intent
    # names modules and files from model recall, which can describe a
    # different release than the pinned tree. Surveying the tree turns
    # terminal selection from recall into choice — a symbol absent from
    # the survey cannot be proposed, so a stale module path or renamed
    # member never reaches discovery. The survey anchors on identifiers
    # the question names AND the tree confirms (declared parameters,
    # observed signals, defined callables), so the menu holds the writes
    # those anchors actually reach rather than every symbol in a file.
    survey_texts = [
        question,
        *(str(value) for value in intent.values() if isinstance(value, str)),
        *(
            str(item)
            for value in intent.values()
            if isinstance(value, list)
            for item in value
        ),
    ]
    observed_signals = discovery_kwargs.get("logged_signals") or ()

    survey = SourceSurvey()
    survey_files: list[str] = []
    if intent_seeds:
        survey, survey_files = survey_for_queries(
            profiler,
            intent_seeds,
            texts=survey_texts,
            observed_signals=observed_signals,
            max_files=survey_max_files,
            source_hash=source_hash,
        )

    if seeds_override is None:
        authoritative_intent = {
            key: value
            for key, value in intent.items()
            if key != "original_question"
        } or {"concise_intent": question}
        seeds = await runner(
            seeder_agent,
            {
                "question_intent": authoritative_intent,
                "airframe": context_dict.get("airframe") or {},
                "source_survey": survey.as_payload(),
            },
        )
    else:
        seeds = seeds_override

    seeds = seeds.model_copy(
        update={"seeds": dedupe_keep_order([*intent_seeds, *seeds.seeds])}
    )

    annotated_by_terminal: dict[str, Any] = {}

    def _render(terminal: str, result: DiscoveryResult) -> dict[str, Any]:
        annotated = annotate(result) if annotate is not None else None
        annotated_by_terminal[terminal] = annotated
        return render_discovery_compact(result, dag=annotated)

    def _rejected(pool: dict[str, DiscoveryResult]) -> dict[str, str]:
        """Terminals whose deterministic validation refused to build,
        mapped to the reason — the judge must know WHY a candidate is
        empty to propose a usable replacement."""
        out: dict[str, str] = {}
        for terminal, result in pool.items():
            validation = getattr(result, "terminal_validation", None)
            if validation is not None and validation.status != "valid":
                out[terminal] = validation.reason or validation.status
        return out

    def _resolved_survey_target(
        candidate: TerminalCandidate,
    ) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[str]]:
        matches = survey.targets_for(
            candidate.terminal,
            source_target_id=str(candidate.source_target_id or ""),
            file=candidate.terminal_file,
        )
        if not matches:
            matches = survey.targets_for(
                candidate.terminal,
                source_target_id=str(candidate.source_target_id or ""),
            )
        identities = {target.source_target_id for _file, target in matches}
        if len(identities) != 1:
            return candidate.terminal_file, None, None
        preferred = next(
            (
                (file, target)
                for file, target in matches
                if file == candidate.terminal_file
            ),
            matches[0],
        )
        return preferred[0], preferred[1].identity, preferred[1].source_target_id

    def _candidate_key(candidate: TerminalCandidate) -> str:
        _file, _identity, source_target_id = _resolved_survey_target(candidate)
        same_spelling = {
            target.source_target_id
            for _file, target in survey.targets_for(candidate.terminal)
        }
        if source_target_id and len(same_spelling) > 1:
            return f"{candidate.terminal}@{source_target_id}"
        return candidate.terminal

    def _slice(candidate: TerminalCandidate) -> DiscoveryResult:
        terminal_file, terminal_identity, source_target_id = _resolved_survey_target(
            candidate
        )
        if candidate.source_target_id and source_target_id is None:
            return DiscoveryResult(
                dag=None,
                inputs=DAGInputs(),
                files_loaded=[],
                rounds=[],
                terminal_validation=TerminalValidation(
                    terminal=candidate.terminal,
                    status="absent_in_scope",
                    reason="source_target_id is not present in the source survey",
                ),
            )
        return discover_mechanism_dag(
            profiler,
            cache_root,
            seeds.seeds,
            candidate.terminal,
            source_hash,
            terminal_file=terminal_file,
            terminal_identity=terminal_identity,
            preranked_files=survey_files or None,
            **discovery_kwargs,
        )

    def _is_valid(result: DiscoveryResult) -> bool:
        return result.dag is not None and bool(result.dag.vertices)

    results: dict[str, DiscoveryResult] = {}

    def _slice_candidates(candidates: Sequence[TerminalCandidate]) -> int:
        """Slice candidates under explicit LLM/API-cost safeguards.

        ``max_terminals`` and ``max_terminal_attempts`` are not correctness
        claims and can hide a later valid terminal. They are retained while
        each additional rendered DAG materially increases paid model input.
        Revisit them if terminal rejection or selection evidence shows that
        this cost guard is affecting answers.
        """
        valid = sum(1 for result in results.values() if _is_valid(result))
        attempts = 0
        for candidate in candidates:
            if valid >= max_terminals or attempts >= max_terminal_attempts:
                break
            candidate_key = _candidate_key(candidate)
            if not candidate.terminal or candidate_key in results:
                continue
            attempts += 1
            result = _slice(candidate)
            results[candidate_key] = result
            if _is_valid(result):
                valid += 1
        return valid

    # The questioned quantity's LOGGED signal (topic.field) is the terminal
    # that ties a mechanism to the observation. The seeder may anchor on
    # decision-site internals and demote it, so force an observable terminal to
    # the front of the candidate list for the judge.
    if seeds.questioned_condition and seeds.questioned_condition.signal_hint:
        hint = str(seeds.questioned_condition.signal_hint)
        existing_keys = {
            _candidate_key(candidate) for candidate in seeds.candidate_terminals
        }
        added: list[TerminalCandidate] = []
        # Precise: a surveyed write that publishes the questioned signal.
        for target_file, target in survey.published_writes_for(hint):
            candidate = TerminalCandidate(
                terminal=target.symbol,
                terminal_file=target_file,
                source_target_id=target.source_target_id,
                reason="write that publishes the questioned logged signal",
            )
            key = _candidate_key(candidate)
            if key not in existing_keys:
                existing_keys.add(key)
                added.append(candidate)
        # Fallback: the logged signal itself, anchored to a module the seeder
        # already targeted; discover_mechanism_dag resolves the write site.
        if not added and "." in hint:
            hint_files = [
                candidate.terminal_file
                for candidate in seeds.candidate_terminals
                if candidate.terminal_file
            ] or [
                str(value)
                for value in (intent.get("likely_source_files") or [])
                if value
            ]
            candidate = TerminalCandidate(
                terminal=hint,
                terminal_file=hint_files[0] if hint_files else "",
                source_target_id="",
                reason="logged signal of the questioned quantity",
            )
            if hint_files and _candidate_key(candidate) not in existing_keys:
                added.append(candidate)
        if added:
            seeds = seeds.model_copy(
                update={
                    "candidate_terminals": [*added, *seeds.candidate_terminals]
                }
            )

    valid_count = _slice_candidates(seeds.candidate_terminals)

    # Rejection recovery: every proposal failed deterministic validation,
    # so there is no graph for the judge to re-terminal from. Re-seed ONCE
    # with the rejection reasons plus the survey, then slice again. The
    # seeder can only answer with symbols the tree actually writes.
    reseeded = False
    if valid_count == 0 and seeds_override is None and results:
        reseeded = True
        if not survey.files:
            # No intent queries seeded a survey, but the failed slices did
            # load files — survey those rather than re-reading the tree.
            loaded = dedupe_keep_order(
                [file for result in results.values() for file in result.files_loaded]
            )
            survey = survey_source_files(
                profiler,
                loaded,
                texts=[*survey_texts, *seeds.seeds],
                observed_signals=observed_signals,
                source_hash=source_hash,
            )
        retry = await runner(
            seeder_agent,
            {
                "question_intent": {
                    key: value
                    for key, value in intent.items()
                    if key != "original_question"
                }
                or {"concise_intent": question},
                "airframe": context_dict.get("airframe") or {},
                "source_survey": survey.as_payload(),
                "rejected_terminals": _rejected(results),
                "note": (
                    "every proposed terminal failed deterministic validation "
                    "against the pinned source; propose terminals only from "
                    "source_survey write targets"
                ),
            },
        )
        seeds = seeds.model_copy(
            update={
                "seeds": dedupe_keep_order([*seeds.seeds, *retry.seeds]),
                "candidate_terminals": [
                    *seeds.candidate_terminals,
                    *retry.candidate_terminals,
                ],
                "questioned_condition": (
                    seeds.questioned_condition or retry.questioned_condition
                ),
            }
        )
        _slice_candidates(retry.candidate_terminals)

    # A candidate whose slice found nothing can't ground anything —
    # keep it out of the judge's choices so an authoritative-sounding
    # name doesn't outrank a smaller-but-real slice. If everything is
    # empty the judge sees it all and must re-terminal.
    non_empty = {
        terminal: result
        for terminal, result in results.items()
        if _is_valid(result)
    }
    judged_candidates = non_empty or results

    deviation: Any = None
    if seeds.questioned_condition is not None and condition_windows is not None:
        deviation = condition_windows(seeds.questioned_condition, judged_candidates)

    verdict = await runner(
        judge_agent,
        {
            "question": question,
            "questioned_windows": deviation,
            "candidates": {
                terminal: _render(terminal, result)
                for terminal, result in judged_candidates.items()
            },
            "empty_candidates": sorted(set(results) - set(judged_candidates)),
            "rejected_terminals": _rejected(results),
            **(
                {"source_survey": survey.as_payload()}
                if not non_empty and survey.files
                else {}
            ),
        },
    )

    selected = results.get(verdict.selected_terminal)
    bonus_round_used = False
    if not verdict.sufficient and verdict.next_terminals:
        # One replacement terminal is another API-cost guard: the new slice
        # itself is deterministic and complete, but every re-judge sends a
        # full graph payload. It is not a source-expansion depth limit.
        target = verdict.next_terminals[0]
        bonus_terminal = target.terminal
        bonus_file, bonus_identity, _bonus_source_target_id = _resolved_survey_target(
            target
        )
        bonus_key = _candidate_key(target)
        if bonus_terminal:
            selected = discover_mechanism_dag(
                profiler,
                cache_root,
                seeds.seeds,
                bonus_terminal,
                source_hash,
                terminal_file=bonus_file,
                terminal_identity=bonus_identity,
                **discovery_kwargs,
            )
            results[bonus_key] = selected
            bonus_round_used = True
            # Re-judge once so the verdict describes the replacement graph.
            # Further replacements are not acted on because each additional
            # full render incurs model input/API cost.
            verdict = await runner(
                judge_agent,
                {
                    "question": question,
                    "questioned_windows": deviation,
                    "candidates": {bonus_key: _render(bonus_key, selected)},
                    "empty_candidates": [],
                    "rejected_terminals": _rejected({bonus_key: selected}),
                    "note": "post-follow-up render; no further discovery rounds remain",
                },
            )

    selected_annotated = annotated_by_terminal.get(verdict.selected_terminal)
    return JudgedDiscovery(
        seeds=seeds,
        verdict=verdict,
        results=results,
        selected=results.get(verdict.selected_terminal, selected),
        selected_annotated=selected_annotated,
        bonus_round_used=bonus_round_used,
        reseeded=reseeded,
    )
