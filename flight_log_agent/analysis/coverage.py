"""Structured search-attempt evidence for source expansion (T1).

Observation only: recording what source search attempted, under which
strategy, against which domain, and with what outcome. Nothing here
certifies writer coverage, retires requests, changes scheduling, or
authorizes discovery stop. Those belong to later workstream tickets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# Strategy identifiers mirror resolver dispatch branches. The `-query`
# strategies are heuristic recall aids; the file-enumeration strategies
# examine a closed enumerated set as constructed (which is not the same
# as proving tree-wide completeness).
STRATEGY_STORAGE_OWNER_FILES = "storage-owner-files"
STRATEGY_STORAGE_DECLARATION_FILES = "storage-declaration-files"
STRATEGY_STORAGE_ASSIGNMENT_QUERY = "storage-assignment-query"
STRATEGY_STORAGE_INTERNAL_ONLY = "storage-internal-only"
STRATEGY_CALLABLE_OWNER_FILES = "callable-owner-files"
STRATEGY_CALLABLE_DEFINITION_SEARCH = "callable-definition-search"
STRATEGY_CALLABLE_QUERY_GROUP = "callable-query-group"
STRATEGY_QUERY_GROUP_ASSIGNMENT = "query-group-assignment"
STRATEGY_QUERY_GROUP_BARE = "query-group-bare"
STRATEGY_QUERY_GROUP = "query-group"
STRATEGY_LOCAL_SHORTCIRCUIT = "local-shortcircuit"
STRATEGY_UNIQUE_ENTITY_FILTER = "unique-entity-filter"

# Completeness classes describe what the strategy examined, not what the
# tree contains. "complete" means closed over its enumerated set as
# constructed; only later derivation may judge sufficiency.
CLASS_COMPLETE = "complete"
CLASS_HEURISTIC = "heuristic"
CLASS_UNAVAILABLE = "unavailable"

# Admission outcomes. Every branch that previously produced bare []/"" maps
# to one of these; none of them certifies coverage.
OUTCOME_ADMITTED = "admitted"
OUTCOME_REJECTED_DECLARATION = "rejected-declaration"
OUTCOME_REJECTED_OWNER = "rejected-owner"
OUTCOME_NON_WRITER = "non-writer"
OUTCOME_AMBIGUOUS = "ambiguous"
OUTCOME_INAPPLICABLE = "inapplicable"
OUTCOME_NO_QUERY = "no-query-issued"
OUTCOME_UNAVAILABLE = "unavailable"
OUTCOME_NO_CANDIDATE_COMPLETE = "no-candidate-complete-domain"
OUTCOME_NO_CANDIDATE_PARTIAL = "no-candidate-partial-domain"


@dataclass
class SearchAttempt:
    """One strategy execution against one obligation.

    `examined_domain` records files actually opened and admission-checked
    plus the search configuration in force; profiler-internal query
    normalization is opaque to this layer and recorded only by mode.
    """

    obligation_key: tuple
    scheduling_key: tuple
    strategy: str
    strategy_class: str
    dispatch: Optional[str] = None
    flags: dict = field(default_factory=dict)
    queries_issued: tuple = ()
    intended_boundary: str = ""
    examined_domain: dict = field(default_factory=dict)
    universe_ref: dict = field(default_factory=dict)
    outcome: str = OUTCOME_UNAVAILABLE
    details: dict = field(default_factory=dict)


@dataclass
class CoverageEvidence:
    """Attempt records for one resolution, without any authority."""

    obligation_key: tuple
    scheduling_key: tuple
    attempts: list = field(default_factory=list)
    universe_ref: dict = field(default_factory=dict)


def stage_id(strategy: str, group_index: Optional[int] = None) -> str:
    """Stable identifier for one resolver stage.

    Scheduling only: identifies what was attempted so a later retry can
    skip completed stages. Carries no coverage meaning.
    """
    if group_index is None:
        return str(strategy)
    return f"{strategy}:{group_index}"


def attempt_stage(strategy: str, details: Optional[dict] = None) -> str:
    """Canonical stage id for one recorded attempt or record site.

    The single mapping used by the recorder, the static plan, and tests:
    strategies whose name contains "group" are per query-group stages and
    take their index from `details`; all others are unindexed. Do not
    reimplement this inline.
    """
    details = details or {}
    group_index = (
        details.get("group_index")
        if "group" in str(strategy)
        else None
    )
    return stage_id(strategy, group_index)


@dataclass
class CoverageSearchState:
    """Session-owned scheduling progress for one search universe version.

    Tracks which stages completed per obligation and which visit keys were
    exhausted, scoped to the current universe version. Advancing the version
    (new source admitted) prunes old progress and reopens eligibility
    without discarding recorded evidence (evidence lives in sinks, not here).

    `visited` is the version-scoped suppression set owned by the discovery
    session: a request resolved under this version is not resolved again
    until the version advances. Per-obligation stage progress
    (`record_stages`/`eligible`) is reserved for later tickets and stays
    inert until a consumer wires it in.

    Scheduling only: never certifies coverage, applicability, or stop.
    """

    version: int = 0
    _completed: dict = field(default_factory=dict, repr=False)
    _exhausted: set = field(default_factory=set, repr=False)
    visited: set = field(default_factory=set, repr=False)

    def mark_visited(self, resolution_key: Any) -> None:
        self.visited.add(resolution_key)

    def was_visited(self, resolution_key: Any) -> bool:
        return resolution_key in self.visited

    def record_stages(self, obligation_key: Any, stages: Any) -> None:
        entry = self._completed.setdefault(obligation_key, set())
        for stage in stages or ():
            entry.add(stage)
        # An empty stage set still marks the obligation seen, so empty-plan
        # kinds resolve once and then settle.

    def mark_exhausted(self, visit_key: Any) -> None:
        self._exhausted.add(visit_key)

    def exhausted_keys(self) -> set:
        return set(self._exhausted)

    def advance_version(self) -> int:
        """Open a new search universe version, pruning prior suppression."""
        self.version += 1
        self._completed.clear()
        self._exhausted.clear()
        self.visited.clear()
        return self.version

    def eligible(self, obligation_key: Any, planned: Any) -> bool:
        """Whether an obligation deserves another resolution attempt."""
        if obligation_key not in self._completed:
            # Unseen keys are always eligible once, including empty plans.
            return True
        recorded = self._completed[obligation_key]
        if set(planned or ()) <= set(recorded):
            return False
        if self._exhausted:
            # Same-version exhaustion suppresses retry of seen obligations
            # until the universe version advances.
            return False
        return True
