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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

from agents import Agent
from pydantic import BaseModel, Field

from flight_log_agent.analysis.mechanism_discovery import (
    DiscoveryResult,
    discover_mechanism_dag,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler


# ---------------------------------------------------------------------------
# Compact DAG rendering (the judge's input)
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = 90) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _capped(items: list[Any], cap: int) -> list[Any]:
    if len(items) <= cap:
        return items
    return items[:cap] + [f"… +{len(items) - cap} more"]


def render_discovery_compact(
    result: DiscoveryResult,
    *,
    max_operations: int = 60,
    max_branches: int = 25,
    max_evidence: int = 30,
    max_unresolved: int = 25,
) -> dict[str, Any]:
    """JSON-able compact view of one discovery result.

    Operations render as ``target <- expression @ file:line``; evidence
    groups by kind; the per-round trace shows how the fixpoint converged
    so the judge can see whether gaps were shrinking or churning.
    """
    dag = result.dag
    operations: list[str] = []
    branches: list[str] = []
    helper_returns: list[str] = []
    evidence: dict[str, list[str]] = {}

    for vertex in dag.vertices if dag else []:
        if vertex.kind == "operation":
            location = f"{vertex.file}:{vertex.line}" if vertex.file else "?"
            entry = f"{vertex.variable} <- {_truncate(vertex.expression or '')} @ {location}"
            if vertex.provenance and vertex.provenance.startswith("helper_return"):
                helper_returns.append(str(vertex.variable))
            operations.append(entry)
        elif vertex.kind == "branch":
            predicate = vertex.predicate_lowered or vertex.predicate_raw or ""
            branches.append(
                f"{_truncate(predicate)} [{vertex.feasibility_verdict or 'unknown'}]"
            )
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


class DiscoverySeeds(BaseModel):
    """Seeder output: grep-able queries + candidate terminals."""

    seeds: list[str]
    candidate_terminals: list[TerminalCandidate]
    notes: list[str] = Field(default_factory=list)


class DiscoveryVerdict(BaseModel):
    """Judge output over the compact DAG rendering(s)."""

    sufficient: bool
    selected_terminal: str
    essential_gaps: list[str] = Field(default_factory=list)
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
- seeds: concrete grep-able queries for the PX4 source tree — parameter
  names (RTL_RETURN_ALT), class/method names (RTL::find_RTL_destination),
  uORB topic or field names. Prefer specific over generic terms.
- candidate_terminals: the C++ symbols (class members or locals) that
  HOLD the quantity the question asks about, each with the source file
  expected to write it. Order by likelihood; at most three.

Each terminal must be the bare variable name exactly as it appears on
the LEFT side of its assignment in source — ``_destination.alt``,
``_rtl_alt`` — never class-qualified (``RTL::_destination.alt``) and
never type-qualified (``mission_item_s::altitude``).

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
  ignorable and must not be listed. An empty list means no further
  discovery is worthwhile even if sufficient is false.

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
    **discovery_kwargs: Any,
) -> JudgedDiscovery:
    """Seeder → deterministic fixpoint per candidate terminal → judge.

    ``seeds_override`` (e.g. a Layer 4 cache hit for a previously judged
    question) skips the seeder call entirely; the judge still runs.

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

    verdict = await runner(
        judge_agent,
        {
            "question": question,
            "candidates": {
                terminal: render_discovery_compact(result)
                for terminal, result in results.items()
            },
        },
    )

    selected = results.get(verdict.selected_terminal)
    bonus_round_used = False
    if (
        not verdict.sufficient
        and verdict.essential_gaps
        and verdict.selected_terminal in results
    ):
        candidate = next(
            (
                c
                for c in seeds.candidate_terminals
                if c.terminal == verdict.selected_terminal
            ),
            None,
        )
        selected = discover_mechanism_dag(
            profiler,
            cache_root,
            [*seeds.seeds, *verdict.essential_gaps],
            verdict.selected_terminal,
            source_hash,
            terminal_file=candidate.terminal_file if candidate else None,
            **discovery_kwargs,
        )
        results[verdict.selected_terminal] = selected
        bonus_round_used = True

    return JudgedDiscovery(
        seeds=seeds,
        verdict=verdict,
        results=results,
        selected=selected,
        bonus_round_used=bonus_round_used,
    )
