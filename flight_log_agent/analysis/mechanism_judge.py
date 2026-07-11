"""Judge-LLM checkpoints for mechanism discovery (#73 Stage 3).

Exactly two LLM touchpoints, neither inside the fixpoint loop:

* **Seeder** — user question → grep-able search seeds + candidate
  terminal symbols. Runs once, before discovery.
* **Judge** — compact rendering of the discovered DAG(s) → sufficiency
  verdict, terminal selection, and triage of unresolved symbols into
  essential (worth ONE more targeted discovery round) vs ignorable.
  Runs once at fixpoint convergence; grants at most one bonus round.

The judge consumes :func:`render_discovery_compact` — profile *facts*,
not raw source snippets. Everything else in this module is deterministic
and unit-testable; the LLM runner is injectable so tests stub it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

from agents import Agent
from pydantic import BaseModel, Field

from flight_log_agent.expression_math import is_safe_math_function_name

from flight_log_agent.analysis.mechanism_discovery import (
    DiscoveryResult,
    discover_mechanism_dag,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler


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


def render_discovery_compact(
    result: DiscoveryResult,
    *,
    dag: Optional[Any] = None,
    max_operations: int = 400,
    max_branches: int = 400,
    max_evidence: int = 400,
    max_unresolved: int = 400,
) -> dict[str, Any]:
    """JSON-able compact view of one discovery result.

    Operations render as ``target <- expression @ file:line``; evidence
    groups by kind; the per-round trace shows how the fixpoint converged
    so the judge can see whether gaps were shrinking or churning.
    ``dag`` (optional) substitutes a feasibility-annotated graph for
    ``result.dag`` — branches then carry flight-data verdicts and active
    windows.

    The ``max_*`` limits are SAFETY VALVES against pathological graphs,
    not curation — the judge must see the whole slice (a starved render
    measurably produced wrong-shaped verdicts). Truncation is always
    marked with an explicit ``… +N more`` tail.
    """
    dag = dag if dag is not None else result.dag
    operations: list[str] = []
    branches: list[str] = []
    helper_returns: list[str] = []
    call_heads: set[str] = set()
    evidence: dict[str, list[str]] = {}

    for vertex in dag.vertices if dag else []:
        if vertex.kind == "operation":
            location = f"{vertex.file}:{vertex.line}" if vertex.file else "?"
            entry = f"{vertex.variable} <- {_truncate(vertex.expression or '')} @ {location}"
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
            tag = vertex.feasibility_verdict or "unknown"
            if vertex.active_windows:
                first = vertex.active_windows[0][0]
                last = vertex.active_windows[-1][1]
                tag += f"; active {len(vertex.active_windows)}w {first:.1f}-{last:.1f}s"
            branches.append(f"{_truncate(predicate)} [{tag}]")
        elif vertex.kind == "evidence":
            kind = str(vertex.sub_kind or "other")
            label = str(vertex.signal_name or "")
            value = (vertex.metadata or {}).get("value")
            if value is not None:
                label = f"{label}={value}"
            evidence.setdefault(kind, []).append(label)

    return {
        "terminal": dag.terminal if dag else None,
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
            kind: _capped(sorted(set(items)), max_evidence)
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
    reason: str = ""


class QuestionedCondition(BaseModel):
    """The comparison the question asserts, when it asserts one —
    evaluated deterministically over the log so the judge can compare
    deviation windows against branch active-windows."""

    signal_hint: str
    op: str
    reference: str


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
Convert a PX4 flight-log question into inputs for deterministic source
discovery.

Output:
- seeds: concrete grep-able queries for the PX4 source tree — exact
  parameter names, class or method names, uORB topic or field names
  drawn from the question and context. Prefer specific over generic
  terms.
- candidate_terminals: the C++ symbols (class members or locals) that
  HOLD the quantity the question asks about, each with the source file
  expected to write it. Order by likelihood; at most three.

Each terminal must be the bare variable name exactly as it appears on
the LEFT side of its assignment in source — never class-qualified
(``ClassName::member``) and never type-qualified
(``struct_type_s::field``).

Prefer the variable written at the DECISION SITE — where the questioned
quantity is computed or adapted — over the published topic field that
merely logs it; list the published field as a secondary candidate.

If the question asserts a comparison (a quantity above/below/equal to a
parameter or value), fill questioned_condition with the LOGGED signal
that records the quantity (topic.field), the comparison operator, and
the reference (a parameter name or number). It is evaluated over the
log to find when the questioned behavior actually occurred.

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
- essential_gaps: unresolved symbols that MUST be resolved to answer
  (they carry the questioned quantity or gate it). Everything else —
  bookkeeping counters, foreign-module noise, display-only values — is
  ignorable and must not be listed.
- expand_calls: entries from the rendering's unexpanded_calls whose
  body must be inlined to answer (the function computing or adapting
  the questioned quantity).
- next_terminals: when NO given candidate holds the questioned quantity
  at its decision site, name a better bare variable (with file) from
  the rendering's operations.

When sufficient is false you MUST fill at least one of essential_gaps,
expand_calls, or next_terminals — or leave all empty only if no further
discovery could possibly help. One follow-up round is granted at most.

Branch entries end with a flight-data feasibility tag: [always_false]
means the predicate never held in THIS flight — treat that path as
inactive and do NOT demand its grounding; [always_true] held
throughout; "active Nw A-Bs" lists when it held. For questions about
behavior that occurs only sometimes, prefer the branch whose active
windows can explain WHEN it occurred.

questioned_windows (when present) gives the time intervals where the
QUESTIONED condition itself held in the log. The explaining branch's
active windows should overlap them; a branch active only outside them
cannot be the answer.

When sufficient is true you MUST fill explaining_branches with the
branch predicate string(s), copied verbatim from the rendering's
branches, whose taking explains the questioned behavior. They are
cross-checked against flight-data feasibility — a mechanism whose
explaining branch never fired cannot be the answer.

essential_gaps and expand_calls entries are BARE symbol or function
names copied from unresolved_symbols / unexpanded_calls / operation
expressions — never sentences; they are used verbatim as source-search
queries. empty_candidates lists terminals whose slice found nothing:
never select those; if no shown candidate holds the questioned quantity,
propose a replacement in next_terminals taken from the operations of a
non-empty rendering.

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


async def discover_with_judge(
    profiler: MechanismSourceProfiler,
    cache_root: Union[str, Path],
    question: str,
    source_hash: str,
    *,
    run_agent: Optional[AgentRunnerFn] = None,
    context: Optional[dict[str, Any]] = None,
    max_terminals: int = 2,
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
    flight instead of demanding their static grounding. After a bonus
    round the improved DAG is re-annotated and judged ONCE more — the
    verdict is never left stale relative to the graph it describes.
    Worst case: seeder + judge + bonus re-judge = 3 LLM calls.

    The judge may grant at most ONE bonus discovery round: when the
    verdict is insufficient and names ``essential_gaps``, discovery for
    the selected terminal reruns once with those gaps added as seeds
    (their definition search then pulls the missing files). The LLM is
    never consulted inside the loop.

    ``discovery_kwargs`` pass through to
    :func:`~flight_log_agent.analysis.mechanism_discovery.discover_mechanism_dag`
    (``source_root``, ``inventory``, ``logged_signals``, budgets, …).
    """
    runner = run_agent or _default_run_agent

    seeds = seeds_override or await runner(
        seeder_agent,
        {"question": question, "context": context or {}},
    )

    results: dict[str, DiscoveryResult] = {}
    for candidate in seeds.candidate_terminals[:max_terminals]:
        results[candidate.terminal] = discover_mechanism_dag(
            profiler,
            cache_root,
            seeds.seeds,
            candidate.terminal,
            source_hash,
            terminal_file=candidate.terminal_file,
            **discovery_kwargs,
        )

    # A candidate whose slice found nothing can't ground anything —
    # keep it out of the judge's choices so an authoritative-sounding
    # name doesn't outrank a smaller-but-real slice. If everything is
    # empty the judge sees it all and must re-terminal.
    non_empty = {
        terminal: result
        for terminal, result in results.items()
        if result.dag is not None and result.dag.vertices
    }
    judged_candidates = non_empty or results

    annotated_by_terminal: dict[str, Any] = {}

    def _render(terminal: str, result: DiscoveryResult) -> dict[str, Any]:
        annotated = annotate(result) if annotate is not None else None
        annotated_by_terminal[terminal] = annotated
        return render_discovery_compact(result, dag=annotated)

    deviation: Any = None
    if seeds.questioned_condition is not None and condition_windows is not None:
        deviation = condition_windows(seeds.questioned_condition)

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
        },
    )

    selected = results.get(verdict.selected_terminal)
    bonus_round_used = False
    steer = verdict.essential_gaps or verdict.expand_calls or verdict.next_terminals
    if not verdict.sufficient and steer:
        # The judge steers exactly one follow-up: re-terminal when it
        # named a better decision-site variable, else re-slice the
        # selected terminal with gaps/calls as extra seeds. The rerun
        # gets a larger budget — the flat one suits single-module
        # slices but starves multi-module chains.
        if verdict.next_terminals:
            target = verdict.next_terminals[0]
            bonus_terminal = target.terminal
            bonus_file = target.terminal_file
        else:
            bonus_terminal = verdict.selected_terminal
            bonus_file = next(
                (
                    c.terminal_file
                    for c in seeds.candidate_terminals
                    if c.terminal == verdict.selected_terminal
                ),
                None,
            )
        # Gap entries are used verbatim as search queries — a prose
        # sentence greps nothing. Keep only symbol-shaped entries.
        symbol_gaps = [
            gap
            for gap in [*verdict.essential_gaps, *verdict.expand_calls]
            if gap and " " not in gap.strip() and len(gap) < 60
        ]
        if bonus_terminal:
            bonus_kwargs = dict(discovery_kwargs)
            bonus_kwargs["max_rounds"] = bonus_kwargs.get("max_rounds", 3) + 2
            bonus_kwargs["max_files_total"] = (
                bonus_kwargs.get("max_files_total", 24) + 12
            )
            selected = discover_mechanism_dag(
                profiler,
                cache_root,
                [*seeds.seeds, *symbol_gaps],
                bonus_terminal,
                source_hash,
                terminal_file=bonus_file,
                **bonus_kwargs,
            )
            results[bonus_terminal] = selected
            bonus_round_used = True
            # Re-judge ONCE over the improved graph — otherwise the
            # verdict (and the report built from it) describes the
            # pre-bonus DAG. The re-judge's own steering fields are not
            # acted on; one follow-up round is the hard cap.
            verdict = await runner(
                judge_agent,
                {
                    "question": question,
                    "questioned_windows": deviation,
                    "candidates": {bonus_terminal: _render(bonus_terminal, selected)},
                    "empty_candidates": [],
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
    )
