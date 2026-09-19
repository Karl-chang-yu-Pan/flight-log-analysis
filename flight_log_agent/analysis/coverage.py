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


@dataclass(frozen=True)
class WriterCoverageRetirement:
    """Proof-retirement of one exact search obligation under one version.

    Records that a current T3 certificate proved the obligation's closed
    writer boundary complete, and retains the exhaustive supported writer
    provenance downstream logic still needs. Immutable value; scheduling
    suppression only, never stop authority. Keyed by (version,
    resolution key) so old versions go stale without clearing and one
    obligation's retirement can never affect another.
    """

    resolution_key: tuple
    scheduling_key: tuple
    obligation_key: tuple
    declaration: tuple
    version: int
    boundary: tuple
    assumptions: tuple
    writers: tuple
    reason: str = "proof-retired"


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
    # Proof retirements keyed by (version, resolution key). Version
    # scoping (not clearing) invalidates old retirements: advancing the
    # version simply stops matching them. Per-key storage keeps
    # unrelated obligations independent by construction — unlike the
    # inert global `_exhausted`/`eligible()` scheduling helpers above,
    # which proof logic must never consume.
    retired: dict = field(default_factory=dict, repr=False)

    def mark_visited(self, resolution_key: Any) -> None:
        self.visited.add(resolution_key)

    def was_visited(self, resolution_key: Any) -> bool:
        return resolution_key in self.visited

    def is_proof_retired(self, resolution_key: Any) -> bool:
        """Whether this obligation retired under the current version."""
        try:
            key = (self.version, tuple(resolution_key))
        except TypeError:
            return False
        return key in self.retired

    def proof_retirement(self, resolution_key: Any) -> Optional[
            WriterCoverageRetirement]:
        """The current-version retirement record, if one exists."""
        try:
            key = (self.version, tuple(resolution_key))
        except TypeError:
            return None
        return self.retired.get(key)

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


def apply_writer_coverage_retirement(
    search_state: CoverageSearchState,
    certificate: Any,
    current_version: int,
) -> bool:
    """Retire one obligation on current proof, or refuse explicitly.

    Consumes a T3 certificate's bound fields without re-deriving
    coverage: the certificate must be current (its version equals the
    supplied current version) and well-formed (non-empty obligation and
    boundary identity). Records an immutable retirement under
    (version, resolution key) and returns True. Stale or malformed
    certificates record nothing and return False. Never touches
    visited/exhausted/completion state, scheduling, or stop authority.
    """
    if certificate is None:
        return False
    if getattr(certificate, "version", None) != current_version:
        return False
    obligation_key = tuple(getattr(certificate, "obligation_key", None)
                           or ())
    if not obligation_key:
        return False
    boundary = tuple(getattr(certificate, "boundary", None) or ())
    if not boundary:
        return False
    try:
        record_key = (current_version, tuple(obligation_key))
    except TypeError:
        return False
    search_state.retired[record_key] = WriterCoverageRetirement(
        resolution_key=tuple(obligation_key),
        scheduling_key=tuple(
            getattr(certificate, "scheduling_key", None) or ()),
        obligation_key=tuple(obligation_key),
        declaration=tuple(getattr(certificate, "declaration", None) or ()),
        version=current_version,
        boundary=boundary,
        assumptions=tuple(
            getattr(certificate, "assumptions", None) or ()),
        writers=tuple(getattr(certificate, "writers", None) or ()),
    )
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
    # file needs an explicit verdict backed by an exhaustive writer
    # census. `exact:<site>` admits a writer of the requested
    # declaration; the absence verdicts prove no writer form in that
    # file (see _ABSENCE_VERDICTS). The census (all proven
    # same-declaration sites per file, recorded at resolve time from
    # full file facts) is what makes the writer set exhaustive: verdicts
    # alone record only the first admitted site per file. Anything else
    # leaves a boundary member unaccounted for.
    if any(attempt.strategy_class != CLASS_COMPLETE
           for attempt in certifying):
        return _refuse(REFUSAL_PARTIAL_DOMAIN, reason="strategy-class")
    examined: list = []
    verdicts: dict = {}
    censuses: dict = {}
    for attempt in certifying:
        if attempt.outcome not in _ACCOUNTED_OUTCOMES:
            return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                           strategy=attempt.strategy,
                           outcome=attempt.outcome)
        for examined_file in (attempt.examined_domain.get("files") or ()):
            if examined_file not in examined:
                examined.append(examined_file)
        verdicts.update((attempt.details or {}).get("file_verdicts") or {})
        censuses.update(
            (attempt.details or {}).get("writer_census") or {})
    if not examined:
        return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                       reason="empty-examined-domain")
    writers: list = []
    for examined_file in examined:
        verdict = verdicts.get(examined_file, "")
        if examined_file not in censuses:
            # Evidence predating writer-census recording cannot prove
            # exhaustiveness: refuse rather than fall back to the
            # first-match verdict alone.
            return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                           examined_file=examined_file,
                           reason="missing-writer-census")
        census = [str(site) for site in censuses[examined_file]]
        if verdict.startswith("exact:"):
            site = verdict[len("exact:"):]
            if site not in census:
                return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                               examined_file=examined_file,
                               reason="verdict-census-mismatch")
        elif verdict in _ABSENCE_VERDICTS:
            if census:
                return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                               examined_file=examined_file,
                               reason="verdict-census-mismatch")
        elif "multi-match" in verdict:
            return _refuse(REFUSAL_AMBIGUOUS_CANDIDATES,
                           examined_file=examined_file)
        else:
            return _refuse(REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
                           examined_file=examined_file,
                           verdict=verdict)
        writers.extend(census)

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


# --- T5: positive per-use applicability proofs (no stop activation) ---
#
# Applicability answers, for one concrete use, whether a current proven
# writer-coverage set positively applies to that use. It is downstream of
# coverage, per use (one origin vertex plus one operand — never a whole
# declaration), positive-evidence-only, and fail-closed. Like T3
# certificates it is unconsumed data: derivation is pure, reads no
# scheduling/checkpoint state, and changes nothing downstream.
#
# The single supported basis is one exactly ordered producer: the use's
# dataflow shows exactly one operation producer for the operand, that
# producer is exactly the covered writer, and neither use nor producer
# carries control/conditional uncertainty. Ordering here is dataflow
# role-edge order (producer value flows into the use operand) — never
# source text order, declaration order, or search result order, which
# prove nothing about runtime applicability. Guard-specific reasoning
# and transfer-execution engines are refused: no exact guard-to-use
# binding machinery exists, and transfer vertices qualify only through
# this same structural rule where exact and unconditional.

# The one positive basis T5 can prove with existing graph facts.
APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER = "single-exact-producer"

# Explicit refusal reasons. Deterministic strings for TDD/debugging.
APPLICABILITY_NO_POSITIVE_BASIS = "no-positive-basis"
APPLICABILITY_CONDITIONAL_WRITER = "conditional-writer-present"
APPLICABILITY_UNRESOLVED_CONTROL = "unresolved-control"
APPLICABILITY_WRITER_SET_MISMATCH = "writer-set-mismatch"
APPLICABILITY_STALE_COVERAGE = "stale-coverage"
APPLICABILITY_USE_IDENTITY_MISMATCH = "use-identity-mismatch"
APPLICABILITY_NO_COVERED_PRODUCER = "no-covered-producer"


@dataclass(frozen=True)
class WriterApplicabilityProof:
    """Positive proof that a covered writer set applies to one use.

    Binds the concrete use (origin vertex plus operand), the scheduling
    and semantic obligation identity, the covered writer set, the basis,
    the exact supporting graph facts (use, producer, and edge IDs), the
    proof version, and the receiver context. Immutable value; carries no
    authority by itself and never implies discovery stop.
    """

    use_key: tuple
    scheduling_key: tuple
    obligation_key: tuple
    declaration: tuple
    writers: tuple
    producer_vertex: str
    basis: str
    supporting_facts: tuple
    version: int
    receiver_context: tuple


@dataclass
class ApplicabilityDerivationResult:
    """A proof or an explicit refusal. `refusal` is "" on success."""

    proof: Optional[WriterApplicabilityProof] = None
    refusal: str = ""
    refusal_detail: dict = field(default_factory=dict)


def _refuse_applicability(code: str, **detail: Any
                          ) -> ApplicabilityDerivationResult:
    return ApplicabilityDerivationResult(
        proof=None, refusal=code, refusal_detail=dict(detail))


def reference_concrete_uses(
    reference: Any,
) -> tuple[list[tuple[str, str]], str]:
    """Effective concrete uses of one unresolved obligation.

    Returns `(uses, provenance)` where provenance is one of:
    - `"exact"`: `origin_uses` present — authoritative; legacy
      origins/operands arrays are ignored, however noisy;
    - `"entailed"`: legacy single-origin/single-operand shape
      mechanically entails exactly one pair;
    - `"compatibility"`: ambiguous legacy fields — conservative
      cartesian fallback, fail-closed (extra pairs only add proof
      requirements, never omit a real use).

    The cartesian path exists only for references predating exact
    provenance and is removed when the old DAG/snapshot format
    retires; no format version exists today, so the condition is the
    empty-pair state itself. Shared pure helper so T5 membership and
    T6B enumeration cannot implement divergent fallback semantics.
    Order-preserving, deduplicated.
    """
    pairs: list[tuple[str, str]] = []
    for pair in list(getattr(reference, "origin_uses", None) or ()):
        try:
            origin, operand = pair
        except (TypeError, ValueError):
            continue
        if (origin and operand
                and (origin, operand) not in pairs):
            pairs.append((origin, operand))
    if pairs:
        return pairs, "exact"
    origins = [item for item in list(
        getattr(reference, "origin_vertex_ids", None) or ()) if item]
    operands = [item for item in list(
        getattr(reference, "origin_operands", None) or ()) if item]
    if len(origins) == 1 and len(operands) == 1:
        return [(origins[0], operands[0])], "entailed"
    uses: list[tuple[str, str]] = []
    for origin in origins:
        for operand in operands:
            if (origin, operand) not in uses:
                uses.append((origin, operand))
    return uses, "compatibility"


def _reachability_exact(vertex: Any) -> Optional[bool]:
    """The builder's exactness verdict for one vertex, if recorded."""
    metadata = getattr(vertex, "metadata", None) or {}
    reachability = metadata.get("reachability") or {}
    if not isinstance(reachability, dict):
        return None
    exact = reachability.get("exact")
    conditions = reachability.get("all_of") or []
    if exact is True and not conditions:
        return True
    if exact is False or conditions:
        return False
    return None


def derive_writer_applicability(
    reference: Any,
    origin_vertex_id: str,
    operand: str,
    certificate: Any,
    dag: Any,
    current_version: int,
    *,
    conditional_writer_ids: Any = (),
) -> ApplicabilityDerivationResult:
    """Derive per-use applicability purely, or refuse explicitly.

    Inputs are the obligation reference carrying the concrete use, the
    use's origin vertex and operand, the T3 coverage certificate, the
    mechanism DAG (read-only: vertices and data/control edges), the
    current proof version, and the caller-scoped conditional writer IDs
    (operations whose execution the evaluation left unresolved, e.g. the
    union over the terminal's local equation checks). Reads nothing
    else: no scheduling state, no checkpoint state, no source. Returns
    a proof or a deterministic refusal reason; never None, never a side
    effect, never stop authority.
    """
    use_key = (origin_vertex_id, operand)
    # Concrete-use membership over the effective use view: exact pairs
    # when provenance exists, else the bounded legacy compatibility
    # (single entailed pair or conservative cartesian). A requested use
    # outside that view — e.g. a Cartesian-invented cross-pair — is not
    # a use of this obligation. Empty views constrain nothing, as
    # before: a use without recorded provenance cannot be refused here.
    concrete_uses, _use_provenance = reference_concrete_uses(reference)
    if concrete_uses and (origin_vertex_id, operand) not in [
            (str(use_origin), str(use_operand))
            for use_origin, use_operand in concrete_uses]:
        return _refuse_applicability(
            APPLICABILITY_USE_IDENTITY_MISMATCH, reason="concrete-use",
            use_key=use_key)
    # Semantic obligation binding: the certificate's proven obligation
    # must equal the use's obligation recomputed without structure.
    # Scheduling keys deliberately do NOT bind here — visit identity
    # carries use-site context (consumer callable), so two concrete uses
    # of one declaration have different visits by design while sharing
    # one coverage certificate. Coverage may be shared; applicability
    # may not. Cross-declaration reuse still refuses: declaration ids
    # differ.
    try:
        use_visit = tuple(reference.visit_key())
    except (AttributeError, TypeError, ValueError):
        return _refuse_applicability(
            APPLICABILITY_USE_IDENTITY_MISMATCH, reason="scheduling-key",
            use_key=use_key)
    expected_key = _obligation_identity_fields(reference)
    if expected_key is None or (
            tuple(getattr(certificate, "obligation_key", None) or ())
            != tuple(expected_key)):
        return _refuse_applicability(
            APPLICABILITY_USE_IDENTITY_MISMATCH, reason="obligation-key",
            use_key=use_key)
    certificate_receiver = tuple(
        getattr(certificate, "receiver_context", None) or ())
    reference_receiver = (
        str(getattr(reference, "receiver", "") or ""),
        str(getattr(reference, "receiver_type", "") or ""),
        str(getattr(reference, "class_owner", "") or ""),
    )
    if certificate_receiver != reference_receiver:
        return _refuse_applicability(
            APPLICABILITY_USE_IDENTITY_MISMATCH, reason="receiver-context",
            use_key=use_key)
    if getattr(certificate, "version", None) != current_version:
        return _refuse_applicability(
            APPLICABILITY_STALE_COVERAGE, current_version=current_version,
            use_key=use_key)
    covered = [str(writer) for writer in
               (getattr(certificate, "writers", None) or ())]
    if not covered:
        # An absence certificate proves no writer exists in the closed
        # boundary; a use that needs a value therefore has nothing whose
        # applicability could be proven.
        return _refuse_applicability(
            APPLICABILITY_NO_COVERED_PRODUCER, use_key=use_key)

    vertices = {vertex.id: vertex
                for vertex in (getattr(dag, "vertices", None) or ())}
    use = vertices.get(origin_vertex_id)
    if use is None or getattr(use, "kind", "") != "operation":
        return _refuse_applicability(
            APPLICABILITY_NO_POSITIVE_BASIS, reason="use-unknown",
            use_key=use_key)
    role_edges = [
        edge for edge in (getattr(dag, "edges", None) or ())
        if getattr(edge, "target_id", "") == origin_vertex_id
        and getattr(edge, "kind", "") == "data"
        and getattr(edge, "role", "") == operand
    ]
    if any(getattr(vertices.get(edge.source_id), "kind", "")
           == "evidence" for edge in role_edges):
        # An unresolved placeholder feeds this operand: an unaccounted
        # alternative writer may govern the use.
        return _refuse_applicability(
            APPLICABILITY_NO_POSITIVE_BASIS, reason="competitor-unknown",
            use_key=use_key)
    producers = [
        vertices[edge.source_id] for edge in role_edges
        if edge.source_id in vertices
        and getattr(vertices[edge.source_id], "kind", "") == "operation"
        and edge.source_id != origin_vertex_id
    ]
    if not producers:
        return _refuse_applicability(
            APPLICABILITY_NO_POSITIVE_BASIS, reason="no-producer",
            use_key=use_key)
    if len(producers) != 1:
        # Several structural producers with unknown runtime order:
        # source text order across invocations proves nothing.
        return _refuse_applicability(
            APPLICABILITY_NO_POSITIVE_BASIS, reason="multiple-producers",
            use_key=use_key)
    producer = producers[0]
    producer_site = str(
        (getattr(producer, "metadata", None) or {}).get(
            "source_site_id") or "")
    if set([producer_site]) != set(covered):
        # The use's flow shows a different writer set than the covered
        # set, in either direction: a competitor is unaccounted for, or
        # the covered writer does not feed this use.
        return _refuse_applicability(
            APPLICABILITY_WRITER_SET_MISMATCH,
            structural=[producer_site], covered=sorted(set(covered)),
            use_key=use_key)
    conditional = set(conditional_writer_ids or ())
    if producer.id in conditional or origin_vertex_id in conditional:
        return _refuse_applicability(
            APPLICABILITY_CONDITIONAL_WRITER,
            use_key=use_key)
    for vertex in (use, producer):
        if any(getattr(edge, "kind", "") == "control"
               and getattr(edge, "target_id", "") == vertex.id
               for edge in (getattr(dag, "edges", None) or ())):
            # Control-gated use or producer (loops, branches, guarded
            # regions, including statically-always-taken gates, which T5
            # does not attempt to prove taken): ordering and invocation
            # need history reasoning T5 does not perform.
            return _refuse_applicability(
                APPLICABILITY_UNRESOLVED_CONTROL,
                vertex_id=vertex.id, use_key=use_key)
        if _reachability_exact(vertex) is not True:
            return _refuse_applicability(
                APPLICABILITY_UNRESOLVED_CONTROL,
                reason="reachability-unproven", vertex_id=vertex.id,
                use_key=use_key)
    return ApplicabilityDerivationResult(
        proof=WriterApplicabilityProof(
            use_key=use_key,
            scheduling_key=use_visit,
            obligation_key=tuple(certificate.obligation_key),
            declaration=tuple(certificate.declaration),
            writers=tuple(sorted(set(covered))),
            producer_vertex=producer.id,
            basis=APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER,
            supporting_facts=tuple(
                sorted({"use:" + origin_vertex_id,
                        "producer:" + producer.id}
                       | {"edge:" + edge.id for edge in role_edges
                          if getattr(edge, "id", "")})),
            version=current_version,
            receiver_context=tuple(certificate.receiver_context),
        ),
        refusal="",
        refusal_detail={},
    )


# --- P2A: session proof store (derived state, never authority) ---
#
# `CoverageProofStore` is session-owned derived proof state for future
# P2B/P2C/P3 consumption. It stores/indexes T3 certificates, T5
# proofs, and per-key evidence fingerprints; it derives nothing,
# retires nothing, and authorizes nothing. It is deliberately separate
# from `CoverageSearchState` (scheduling/version/visited/retirement):
# the two synchronize only through the search-universe version, which
# remains owned by `CoverageSearchState`. All authority reads pass an
# explicit version, so stale proof can never be current by
# construction — there is no "current version" state to go stale.


def evidence_fingerprint(attempts: Any) -> tuple:
    """Deterministic digest of ordered search attempts.

    Derived only from stable coverage-relevant fields (strategy,
    outcome, examined files, per-file verdicts/census, universe
    version, obligation/scheduling identity). Never object identity,
    timestamps, or iteration order beyond the recorded attempt order.
    P2B compares fingerprints to detect materially changed evidence
    without re-deriving on identical input.
    """
    digest: list = []
    for attempt in attempts or ():
        details = getattr(attempt, "details", None) or {}
        examined = getattr(attempt, "examined_domain", None) or {}
        files = tuple(sorted(
            str(item) for item in (examined.get("files") or ())))
        verdicts = tuple(sorted(
            (str(name), str(verdict))
            for name, verdict in (
                (details.get("file_verdicts") or {}).items())))
        census = tuple(sorted(
            (str(name), tuple(sorted(
                str(site) for site in (sites or ()))))
            for name, sites in (
                (details.get("writer_census") or {}).items())))
        universe = (getattr(attempt, "universe_ref", None) or {}).get(
            "search_version")
        digest.append((
            str(getattr(attempt, "strategy", "") or ""),
            str(getattr(attempt, "strategy_class", "") or ""),
            str(getattr(attempt, "outcome", "") or ""),
            files,
            verdicts,
            census,
            universe,
            repr(tuple(getattr(attempt, "obligation_key", None) or ())),
            repr(tuple(getattr(attempt, "scheduling_key", None) or ())),
        ))
    return tuple(digest)


@dataclass(frozen=True)
class ProofSnapshot:
    """Immutable current-version proof view for future checkpoint use.

    Carries only derived values (never verification flags): the exact
    version plus deterministic tuples of that version's certificates
    and applicability proofs. Field names project directly onto the
    T6B round inputs (`proof_version`, `coverage_certificates`,
    `applicability_proofs`). P3 threads snapshots; nothing mutates
    through them.
    """

    version: Any
    certificates: tuple = ()
    applicability_proofs: tuple = ()


@dataclass
class CoverageProofStore:
    """Session-scoped store of derived proof state.

    Certificates keyed by (version, scheduling, obligation);
    applicability proofs by (version, use, scheduling); evidence
    fingerprints by (version, scheduling). Raw attempts stay in the
    P0 session sink (canonical history); only fingerprints live here.
    Identical re-insertion is idempotent; a differing value under an
    occupied key is refused with the existing entry retained
    (fail-closed, never silently merged). Reads return tuples in
    repr-key order. Pruning is explicit and caller-driven; the store
    never decides when the universe changes.
    """

    _certificates: dict = field(default_factory=dict, repr=False)
    _proofs: dict = field(default_factory=dict, repr=False)
    _evidence: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def _sorted_entries(entries: Any) -> tuple:
        return tuple(
            value for _key, value in sorted(
                entries, key=lambda item: repr(item[0])))

    @staticmethod
    def _certificate_key(certificate: Any) -> Optional[tuple]:
        """Storage key, or None when the object cannot bind proof."""
        try:
            version = getattr(certificate, "version", None)
            scheduling = tuple(
                getattr(certificate, "scheduling_key", None) or ())
            obligation = tuple(
                getattr(certificate, "obligation_key", None) or ())
        except TypeError:
            return None
        if version is None or not scheduling or not obligation:
            return None
        return (version, scheduling, obligation)

    @staticmethod
    def _proof_key(proof: Any) -> Optional[tuple]:
        """Storage key, or None when the object cannot bind proof."""
        try:
            version = getattr(proof, "version", None)
            use = tuple(getattr(proof, "use_key", None) or ())
            scheduling = tuple(
                getattr(proof, "scheduling_key", None) or ())
        except TypeError:
            return None
        if version is None or not use or not scheduling:
            return None
        return (version, use, scheduling)

    def store_certificate(self, certificate: Any) -> bool:
        """Store one certificate, or refuse without touching state."""
        key = self._certificate_key(certificate)
        if key is None:
            return False
        existing = self._certificates.get(key)
        if existing is not None:
            return bool(existing == certificate)
        self._certificates[key] = certificate
        return True

    def replace_certificate(self, certificate: Any) -> bool:
        """Supersede one stored certificate with a newer value.

        Explicit replacement route for later derivation lifecycle
        (newer evidence re-derives, then replaces): overwrites exactly
        the key derived from the replacement object itself, so an
        object can never migrate between identities. Refuses when no
        entry occupies that key or the replacement cannot bind proof.
        Contains no lifecycle policy — it never decides WHEN newer
        evidence justifies replacement.
        """
        key = self._certificate_key(certificate)
        if key is None or key not in self._certificates:
            return False
        self._certificates[key] = certificate
        return True

    def discard_certificate(
        self, version: Any, scheduling_key: Any, obligation_key: Any,
    ) -> bool:
        """Remove one stored certificate by exact key, if present.

        Mechanism only, for derivation lifecycle (changed evidence
        whose rederivation refuses must not leave the superseded
        positive value current). Removes exactly
        (version, scheduling, obligation); sibling keys and versions
        are untouched. No fingerprint comparison, no derivation, no
        policy — the caller decides when invalidation is justified.
        """
        try:
            key = (version, tuple(scheduling_key or ()),
                   tuple(obligation_key or ()))
        except TypeError:
            return False
        if key not in self._certificates:
            return False
        del self._certificates[key]
        return True

    def certificates_for(self, version: Any) -> tuple:
        """All certificates stored under one version, deterministically."""
        return self._sorted_entries(
            (key, certificate)
            for key, certificate in self._certificates.items()
            if key[0] == version)

    def certificates_for_visit(
        self, version: Any, scheduling_key: Any,
    ) -> tuple:
        """Certificates bound to one scheduling visit under one version."""
        try:
            wanted = tuple(scheduling_key or ())
        except TypeError:
            return ()
        return self._sorted_entries(
            (key, certificate)
            for key, certificate in self._certificates.items()
            if key[0] == version and key[1] == wanted)

    def certificates_for_obligation(
        self, version: Any, obligation_key: Any,
    ) -> tuple:
        """Certificates bound to one obligation under one version."""
        try:
            wanted = tuple(obligation_key or ())
        except TypeError:
            return ()
        return self._sorted_entries(
            (key, certificate)
            for key, certificate in self._certificates.items()
            if key[0] == version and key[2] == wanted)

    def store_proof(self, proof: Any) -> bool:
        """Store one applicability proof, or refuse without touching."""
        key = self._proof_key(proof)
        if key is None:
            return False
        existing = self._proofs.get(key)
        if existing is not None:
            return bool(existing == proof)
        self._proofs[key] = proof
        return True

    def replace_proof(self, proof: Any) -> bool:
        """Supersede one stored proof with a newer value.

        Same explicit-replacement contract as `replace_certificate`:
        overwrites exactly the key derived from the replacement proof
        itself, refuses absent or unbindable keys, decides no lifecycle
        policy.
        """
        key = self._proof_key(proof)
        if key is None or key not in self._proofs:
            return False
        self._proofs[key] = proof
        return True

    def discard_proof(
        self, version: Any, use_key: Any, scheduling_key: Any,
    ) -> bool:
        """Remove one stored applicability proof by exact key.

        Mechanism only, mirroring `discard_certificate`: removes
        exactly (version, use, scheduling) when present, otherwise
        reports absence without mutation. No policy of any kind.
        """
        try:
            key = (version, tuple(use_key or ()),
                   tuple(scheduling_key or ()))
        except TypeError:
            return False
        if key not in self._proofs:
            return False
        del self._proofs[key]
        return True

    def proofs_for(self, version: Any) -> tuple:
        """All applicability proofs stored under one version."""
        return self._sorted_entries(
            (key, proof)
            for key, proof in self._proofs.items()
            if key[0] == version)

    def proof_for(
        self, version: Any, use_key: Any, scheduling_key: Any,
    ) -> Optional[Any]:
        """The proof for one exact (use, visit) under one version."""
        try:
            key = (version, tuple(use_key or ()),
                   tuple(scheduling_key or ()))
        except TypeError:
            return None
        return self._proofs.get(key)

    def note_evidence(
        self, version: Any, scheduling_key: Any, attempts: Any,
    ) -> bool:
        """Record an evidence fingerprint; True when it changed.

        P2B passes the session sink's current attempts for one key;
        identical restatement is not a change. Raw attempts stay in
        the sink; only the deterministic fingerprint lives here.
        """
        try:
            key = (version, tuple(scheduling_key or ()))
        except TypeError:
            return False
        fingerprint = evidence_fingerprint(attempts)
        if self._evidence.get(key) == fingerprint:
            return False
        self._evidence[key] = fingerprint
        return True

    def prune_older_than(self, version: Any) -> None:
        """Drop entries below one version (caller owns version truth)."""
        self._certificates = {
            key: value for key, value in self._certificates.items()
            if not key[0] < version}
        self._proofs = {
            key: value for key, value in self._proofs.items()
            if not key[0] < version}
        self._evidence = {
            key: value for key, value in self._evidence.items()
            if not key[0] < version}

    def snapshot_for(self, version: Any) -> ProofSnapshot:
        """Immutable deterministic view of one version's proof."""
        return ProofSnapshot(
            version=version,
            certificates=self.certificates_for(version),
            applicability_proofs=self.proofs_for(version),
        )
