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
