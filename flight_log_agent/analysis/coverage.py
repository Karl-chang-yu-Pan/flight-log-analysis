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


# --- T3: pure coverage-certificate derivation (unconsumed data) ---
#
# Certificates are DATA ONLY. Derivation is a pure function of an explicit
# obligation, its recorded evidence, and the current search-universe
# version. It never touches scheduling state (visited/exhausted/eligible),
# checkpoint state, or stop authority, and nothing in production consumes
# its output yet. See docs/writer_coverage_stop_authority_spec.md.

# Certifying closure assumptions. A certificate must say WHY its boundary
# is closed; these are explicit vocabulary, never silent optimism.
#
# file-linkage-closed: linkage proves one file is the complete boundary,
# so examining that file's verdicts accounts for every location.
# writer-syntax-enumerated: the examined census covers identifier-mention
# bodies plus global/member initializers as extracted; anything outside
# the extractor's writer forms (macros, aliases, unrepresented syntax) is
# an explicit assumption, not a proven absence.
ASSUMPTION_FILE_LINKAGE_CLOSED = "file-linkage-closed"
ASSUMPTION_WRITER_SYNTAX_ENUMERATED = "writer-syntax-enumerated"

# Explicit refusal reasons. Deterministic strings for TDD/debugging.
REFUSAL_HEURISTIC_STRATEGY = "heuristic-strategy"
REFUSAL_UNSUPPORTED_CLOSURE = "unsupported-closure"
REFUSAL_STALE_VERSION = "stale-version"
REFUSAL_VERSION_UNKNOWN = "version-unknown"
REFUSAL_UNAVAILABLE_SEARCH = "unavailable-search"
REFUSAL_AMBIGUOUS_CANDIDATES = "ambiguous-candidates"
REFUSAL_PARTIAL_DOMAIN = "partial-domain"
REFUSAL_MISSING_REQUIRED_ATTEMPT = "missing-required-attempt"
REFUSAL_UNEXAMINED_BOUNDARY_MEMBER = "unexamined-boundary-member"
REFUSAL_IDENTITY_MISMATCH = "identity-mismatch"

# Strategies whose evidence can support a certificate. This follows the
# spec taxonomy (§2), deliberately NOT the code-level completeness
# classes: owner-lineage enumeration is code-complete over its
# constructed set but heuristic for certification.
_CERTIFIABLE_STRATEGIES = frozenset({STRATEGY_STORAGE_INTERNAL_ONLY})

# Outcomes that describe a failed/incomplete search rather than an
# accounted-for domain. Checked before strategy class so a catastrophic
# outcome is reported precisely even on a heuristic path.
_OUTCOME_REFUSALS = {
    OUTCOME_UNAVAILABLE: REFUSAL_UNAVAILABLE_SEARCH,
    OUTCOME_AMBIGUOUS: REFUSAL_AMBIGUOUS_CANDIDATES,
    OUTCOME_NO_CANDIDATE_PARTIAL: REFUSAL_PARTIAL_DOMAIN,
}

# Outcomes a certifying attempt may carry: an admitted writer, or a
# completely examined domain with no candidate. In the latter case
# absence is proven per file by absence verdicts (see
# _ABSENCE_VERDICTS), never by the outcome alone.
_ACCOUNTED_OUTCOMES = frozenset(
    {OUTCOME_ADMITTED, OUTCOME_NO_CANDIDATE_COMPLETE})

# File verdicts that prove absence inside a proven-closed file:
# `index-miss` (the admission census over identifier-mention bodies,
# calls, and initializers holds no same-declaration form) and
# `no-declaration-match` (full exact matching over the file's facts
# found no same-declaration writer). Both rest on the extractor's
# writer-form census, carried explicitly as
# ASSUMPTION_WRITER_SYNTAX_ENUMERATED.
_ABSENCE_VERDICTS = frozenset({"index-miss", "no-declaration-match"})


@dataclass(frozen=True)
class WriterCoverageCertificate:
    """Positive proof that one obligation's writers are accounted for.

    Binds the obligation (semantic + scheduling keys, declaration
    identity with a spelling label, receiver context) to the exact
    examined boundary, the search-universe version, the explicit closure
    assumptions, and the admitted writer identities. Immutable value;
    carries no authority by itself.
    """

    obligation_key: tuple
    scheduling_key: tuple
    declaration: tuple
    receiver_context: tuple
    strategy: str
    boundary: tuple
    examined: tuple
    version: int
    assumptions: tuple
    writers: tuple


@dataclass
class CoverageDerivationResult:
    """A certificate or an explicit refusal. `refusal` is "" on success."""

    certificate: Optional[WriterCoverageCertificate] = None
    refusal: str = ""
    refusal_detail: dict = field(default_factory=dict)


def _obligation_identity_fields(obligation: Any) -> Optional[tuple]:
    """Structure-free recomputation of a semantic obligation key.

    Mirrors only the branches of obligation-key derivation that need no
    source structure (proven storage writers, identity-carrying symbol
    references). Returns None when the key needs structural context;
    callers then rely on the scheduling-key binding alone.
    """
    identity = getattr(obligation, "identity", None)
    if identity is None:
        return None
    kind = getattr(obligation, "kind", "")
    if kind in {"member_writers", "storage_writers"}:
        if not bool(getattr(identity, "declaration_proven", False)):
            return None
        return (
            kind,
            getattr(identity, "kind", ""),
            getattr(identity, "declaration_id", ""),
        )
    if kind == "symbol" and hasattr(identity, "storage_key"):
        try:
            return (kind, identity.storage_key())
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def _declares_internal_boundary(obligation: Any, evidence: Any) -> bool:
    """Whether the obligation claims an internal-linkage closed boundary.

    Mechanical marker only: a proven global declaration identity minted
    with the internal-linkage prefix, either on the obligation or in the
    evidence's obligation key. No source lookup.
    """
    identity = getattr(obligation, "identity", None)
    declaration_id = str(getattr(identity, "declaration_id", "") or "")
    if declaration_id.startswith("global:internal:"):
        return True
    key = tuple(getattr(evidence, "obligation_key", None) or ())
    return (
        len(key) == 3
        and key[0] in {"member_writers", "storage_writers"}
        and isinstance(key[2], str)
        and key[2].startswith("global:internal:")
    )


def _refuse(code: str, **detail: Any) -> CoverageDerivationResult:
    return CoverageDerivationResult(
        certificate=None, refusal=code, refusal_detail=dict(detail))


def derive_writer_coverage_certificate(
    obligation: Any,
    evidence: Any,
    current_version: int,
) -> CoverageDerivationResult:
    """Derive a coverage certificate purely, or refuse explicitly.

    Inputs are an explicit obligation (reference with semantic identity),
    its recorded `CoverageEvidence`, and the current search-universe
    version. Reads nothing else: no scheduling state, no checkpoint
    state, no source structure. Returns a certificate or a deterministic
    refusal reason; never None, never a side effect.
    """
    attempts = list(getattr(evidence, "attempts", None) or [])

    # Version validity: every attempt and the evidence object must carry
    # the current version stamp. Unstamped (T1-shape) evidence cannot
    # certify at any version.
    stamps = [a.universe_ref.get("search_version") for a in attempts]
    stamps.append((getattr(evidence, "universe_ref", None) or {}).get(
        "search_version"))
    if any(stamp is None for stamp in stamps):
        return _refuse(REFUSAL_VERSION_UNKNOWN,
                       current_version=current_version)
    if any(stamp != current_version for stamp in stamps):
        return _refuse(REFUSAL_STALE_VERSION,
                       current_version=current_version)

    # Obligation binding: the evidence must belong to this obligation.
    # Scheduling-key equality always applies; the semantic key is
    # additionally recomputed wherever no structure is needed for it.
    try:
        scheduling_match = (
            tuple(evidence.scheduling_key)
            == tuple(obligation.visit_key())
        )
    except (AttributeError, TypeError, ValueError):
        scheduling_match = False
    if not scheduling_match:
        return _refuse(REFUSAL_IDENTITY_MISMATCH, reason="scheduling-key")
    expected_key = _obligation_identity_fields(obligation)
    if expected_key is not None and (
            tuple(evidence.obligation_key) != tuple(expected_key)):
        return _refuse(REFUSAL_IDENTITY_MISMATCH, reason="obligation-key")

    if not attempts:
        return _refuse(REFUSAL_MISSING_REQUIRED_ATTEMPT, reason="no-attempts")

    # Catastrophic outcomes first, so they report precisely regardless of
    # the strategy that recorded them.
    for attempt in attempts:
        outcome_refusal = _OUTCOME_REFUSALS.get(attempt.outcome)
        if outcome_refusal is not None:
            return _refuse(outcome_refusal, strategy=attempt.strategy)

    # Strategy certifiability follows the spec taxonomy (§2).
    certifying = [attempt for attempt in attempts
                  if attempt.strategy in _CERTIFIABLE_STRATEGIES]
    if not certifying:
        strategies = {attempt.strategy for attempt in attempts}
        if STRATEGY_LOCAL_SHORTCIRCUIT in strategies:
            # Same-callable local enumeration: the spec lists it as
            # certifiable only under preconditions (loaded body, syntax
            # enumeration) that current evidence cannot mechanically
            # establish, and recording an unexamined domain would violate
            # evidence honesty. Non-certifiable in T3.
            return _refuse(REFUSAL_UNSUPPORTED_CLOSURE,
                           strategies=sorted(strategies))
        if _declares_internal_boundary(obligation, evidence):
            # An internal-closed obligation whose completing strategy
            # never ran: heuristic attempts alone cannot close it.
            return _refuse(REFUSAL_MISSING_REQUIRED_ATTEMPT,
                           strategies=sorted(strategies))
        return _refuse(REFUSAL_HEURISTIC_STRATEGY,
                       strategies=sorted(strategies))

    # Complete accounting over the certifying attempts: every examined
    # file needs an explicit verdict. `exact:<site>` admits a writer of
    # the requested declaration; the absence verdicts prove no writer
    # form in that file (see _ABSENCE_VERDICTS). Anything else leaves a
    # boundary member unaccounted for.
    if any(attempt.strategy_class != CLASS_COMPLETE
           for attempt in certifying):
        return _refuse(REFUSAL_PARTIAL_DOMAIN, reason="strategy-class")
    examined: list = []
    verdicts: dict = {}
    for attempt in certifying:
        if attempt.outcome not in _ACCOUNTED_OUTCOMES:
            return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                           strategy=attempt.strategy,
                           outcome=attempt.outcome)
        for examined_file in (attempt.examined_domain.get("files") or ()):
            if examined_file not in examined:
                examined.append(examined_file)
        verdicts.update((attempt.details or {}).get("file_verdicts") or {})
    if not examined:
        return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                       reason="empty-examined-domain")
    writers: list = []
    for examined_file in examined:
        verdict = verdicts.get(examined_file, "")
        if verdict.startswith("exact:"):
            writers.append(verdict[len("exact:"):])
        elif verdict in _ABSENCE_VERDICTS:
            continue
        elif "multi-match" in verdict:
            return _refuse(REFUSAL_AMBIGUOUS_CANDIDATES,
                           examined_file=examined_file)
        else:
            return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                           examined_file=examined_file,
                           verdict=verdict)

    identity = getattr(obligation, "identity", None)
    declaration = (
        (str(getattr(identity, "kind", "")),
         str(getattr(identity, "declaration_id", "")),
         str(getattr(obligation, "symbol", "")))
        if identity is not None
        else (str(getattr(obligation, "kind", "")),
              "",
              str(getattr(obligation, "symbol", "")))
    )
    receiver_context = (
        str(getattr(obligation, "receiver", "") or ""),
        str(getattr(obligation, "receiver_type", "") or ""),
        str(getattr(obligation, "class_owner", "") or ""),
    )
    return CoverageDerivationResult(
        certificate=WriterCoverageCertificate(
            obligation_key=tuple(evidence.obligation_key),
            scheduling_key=tuple(evidence.scheduling_key),
            declaration=declaration,
            receiver_context=receiver_context,
            strategy=STRATEGY_STORAGE_INTERNAL_ONLY,
            boundary=tuple(examined),
            examined=tuple(examined),
            version=current_version,
            assumptions=(ASSUMPTION_FILE_LINKAGE_CLOSED,
                         ASSUMPTION_WRITER_SYNTAX_ENUMERATED),
            writers=tuple(sorted(set(writers))),
        ),
        refusal="",
        refusal_detail={},
    )
