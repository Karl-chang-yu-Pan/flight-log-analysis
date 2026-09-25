"""Deterministic bounded discovery-frontier machinery (offline).

Implements ``docs/discovery_frontier_bounded_input_spec.md`` without
any model calls: canonical in-memory evidence store with lifecycle
metadata, deterministic reactivation, relevance closure, prefetch,
bounded frontier + carry-forward projection, bounded identity lookup,
fallback accounting, and packet-mass measurement.

The canonical store is the source of truth. The carry-forward summary
is a decision-oriented projection of it, never a second store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Optional

# ----------------------------------------------------------------------
# Slice 1 — lifecycle vocabulary and canonical identity (spec §7, §10)
# ----------------------------------------------------------------------

ACTIVE = "active"
RESOLVED = "resolved"
SUPERSEDED = "superseded"
CONTRADICTED = "contradicted"
RETAINED_SUMMARY = "retained_summary"
IRRELEVANT = "irrelevant"

LIFECYCLE_STATES = frozenset({
    ACTIVE, RESOLVED, SUPERSEDED, CONTRADICTED, RETAINED_SUMMARY, IRRELEVANT,
})

# Content keys that may legitimately change a semantic decision
# (spec §10D). Everything else is serialization noise.
_MEANINGFUL_KEYS = frozenset({
    "actual_value",
    "gate_outcome",
    "semantic_effect",
    "window_applicability",
})


def make_identity(*parts: Any, window: Any = None) -> tuple:
    """Build a stable deterministic identity from caller-supplied parts.

    Parts keep their given order; the optional ``window`` qualifier is
    appended as a distinct segment so the same source under a different
    questioned window never collides (spec §10D).
    """
    identity = tuple(str(part) for part in parts)
    if window is not None:
        identity = identity + ("window", str(window))
    return identity


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return repr(value)


def meaningful_fingerprint(content: Mapping[str, Any]) -> tuple:
    """Project content onto decision-relevant fields, order-insensitively.

    Only ``actual_value``, ``gate_outcome``, ``semantic_effect``, and
    ``window_applicability`` participate; key order, whitespace,
    counter values, and round numbers never affect the result, so two
    extractions with equal identity and equal fingerprint are
    interchangeable by construction (spec §10D).
    """
    items = (
        (str(key), _normalize_scalar(content[key]))
        for key in sorted(content.keys())
        if str(key) in _MEANINGFUL_KEYS
    )
    return tuple(items)


@dataclass
class EvidenceRecord:
    """One canonical evidence item with lifecycle metadata (spec §10D)."""

    identity: Hashable
    kind: str
    lifecycle: str = ACTIVE
    first_seen_round: int = 0
    last_changed_round: int = 0
    value_revision: int = 0
    window_identity: Optional[str] = None
    fingerprint: tuple = ()
    prior_verdicts: tuple = ()
    content: Mapping[str, Any] = field(default_factory=dict)


class CanonicalStore:
    """Deterministic in-memory source of truth for one discovery run.

    Holds every observed evidence record by stable identity, including
    superseded predecessors. Never deletes: inactivation only moves
    records between lifecycle states.
    """

    def __init__(self) -> None:
        self._current: dict[Hashable, EvidenceRecord] = {}
        self._retired: dict[Hashable, list[EvidenceRecord]] = {}

    def get(self, identity: Hashable) -> Optional[EvidenceRecord]:
        """The current record for one identity, if ever observed."""
        return self._current.get(identity)

    def retired_for(self, identity: Hashable) -> list[EvidenceRecord]:
        """Superseded predecessors retained under one identity."""
        return list(self._retired.get(identity, ()))

    def observe(
        self,
        identity: Hashable,
        *,
        kind: str,
        content: Mapping[str, Any],
        round_no: int,
        window_identity: Optional[str] = None,
    ) -> EvidenceRecord:
        """Record one extraction round for one identity.

        New identities enter ACTIVE at revision zero. Re-observation
        with an unchanged meaningful fingerprint keeps the record (and
        its revision) untouched. A changed fingerprint retires the
        predecessor as SUPERSEDED and opens a new ACTIVE record at
        revision +1 carrying the predecessor's verdict history (spec
        §10, §10A, §10D).
        """
        fingerprint = meaningful_fingerprint(content)
        previous = self._current.get(identity)
        if previous is None:
            record = EvidenceRecord(
                identity=identity,
                kind=kind,
                lifecycle=ACTIVE,
                first_seen_round=round_no,
                last_changed_round=round_no,
                value_revision=0,
                window_identity=window_identity,
                fingerprint=fingerprint,
                content=dict(content),
            )
            self._current[identity] = record
            return record
        if previous.fingerprint == fingerprint:
            return previous
        self._retired.setdefault(identity, []).append(previous)
        previous.lifecycle = SUPERSEDED
        record = EvidenceRecord(
            identity=identity,
            kind=kind,
            lifecycle=ACTIVE,
            first_seen_round=previous.first_seen_round,
            last_changed_round=round_no,
            value_revision=previous.value_revision + 1,
            window_identity=(
                window_identity
                if window_identity is not None
                else previous.window_identity
            ),
            fingerprint=fingerprint,
            prior_verdicts=tuple(previous.prior_verdicts),
            content=dict(content),
        )
        self._current[identity] = record
        return record

    def resolve(
        self, identity: Hashable, *, outcome: str, reason: str
    ) -> Optional[EvidenceRecord]:
        """Mark one item admitted/rejected; it constrains via verdict
        records and never equals deletion (spec §10)."""
        record = self._current.get(identity)
        if record is None:
            return None
        record.lifecycle = RESOLVED
        record.prior_verdicts = tuple(record.prior_verdicts) + (
            (str(outcome), str(reason)),
        )
        return record

    def refute(
        self, identity: Hashable, *, reason: str
    ) -> Optional[EvidenceRecord]:
        """Mark one item refuted; it stays visible while referenced
        (spec §10)."""
        record = self._current.get(identity)
        if record is None:
            return None
        record.lifecycle = CONTRADICTED
        record.prior_verdicts = tuple(record.prior_verdicts) + (
            ("refuted", str(reason)),
        )
        return record

    def summarize(self, identity: Hashable) -> Optional[EvidenceRecord]:
        """Fold one unchanged item into the carry-forward projection by
        identity (spec §10). Content stays canonical and retrievable."""
        record = self._current.get(identity)
        if record is None:
            return None
        if record.lifecycle == ACTIVE:
            record.lifecycle = RETAINED_SUMMARY
        return record

    def all_current(self) -> list[EvidenceRecord]:
        """Every current record, deterministically ordered by identity."""
        return [
            self._current[identity]
            for identity in sorted(self._current.keys(), key=repr)
        ]


# ----------------------------------------------------------------------
# Slice 2 — reactivation, referenced semantics, irrelevance, relevance
# closure, prefetch (spec §10A, §10B, §10C, §27 Mode A)
# ----------------------------------------------------------------------

# Lifecycle states eligible for deterministic reactivation. CONTRADICTED
# is excluded: refuted items stay visible with their refutation and
# reactivation must never clear it (spec §10A).
_REACTIVATABLE = frozenset({RESOLVED, SUPERSEDED, IRRELEVANT, RETAINED_SUMMARY})


def is_referenced(
    identity: Hashable,
    *,
    support_sets: Any = (),
    claim_deps: Any = (),
    rival_cites: Any = (),
    contradiction_supports: Any = (),
    requirement_points: Any = (),
) -> bool:
    """Machine-checkable reference test (spec §10C).

    True iff the identity appears in any support set, open-claim
    dependency, rival-exclusion cite, contradiction support, or
    parameter-requirement pointer. Never prose, never similarity.
    """
    for group in (
        support_sets, claim_deps, rival_cites,
        contradiction_supports, requirement_points,
    ):
        for identities in group or ():
            if identity in set(identities):
                return True
    return False


def reactivation_triggers(
    record: EvidenceRecord,
    *,
    new_hypotheses: Any = (),
    new_candidates: Any = (),
    new_contradictions: Any = (),
    changed_requirements: Any = (),
    new_windows: Any = (),
    new_relations: Any = (),
) -> list[str]:
    """Deterministic §10A trigger names fired by one retained record.

    Each trigger is an identity intersection over the round's new
    material. A changed requirement fires only when its recorded
    change flag is set (gate/value change on an overlapping site).
    """
    fired: list[str] = []
    identity = record.identity
    if is_referenced(identity, support_sets=new_hypotheses):
        fired.append("hypothesis-reference")
    if is_referenced(identity, support_sets=new_candidates):
        fired.append("candidate-relation")
    if is_referenced(identity, support_sets=new_contradictions):
        fired.append("contradiction-touch")
    for requirement_identity, changed in changed_requirements or ():
        if changed and requirement_identity == identity:
            fired.append("requirement-intersection")
            break
    if (
        record.window_identity is not None
        and any(str(window) != str(record.window_identity)
                for window in (new_windows or ()))
    ):
        fired.append("window-change")
    for first, second in new_relations or ():
        if identity == first or identity == second:
            fired.append("source-relation")
            break
    return fired


def evaluate_reactivation(store: CanonicalStore, *, round_no: int, **new: Any) -> list[EvidenceRecord]:
    """Re-enter every triggered retained item as ACTIVE (spec §10A).

    Evaluated deterministically per round BEFORE frontier
    construction. Fired items keep their full prior verdict history;
    they never re-enter blank. Returns fired records in deterministic
    identity order.
    """
    _ = round_no
    fired: list[EvidenceRecord] = []
    for record in store.all_current():
        if record.lifecycle not in _REACTIVATABLE:
            continue
        if reactivation_triggers(record, **new):
            record.lifecycle = ACTIVE
            fired.append(record)
    return fired


def mark_irrelevant(
    record: EvidenceRecord,
    *,
    superseded_by_newer: bool,
    referenced: bool,
    rounds_unchanged: int,
) -> bool:
    """Classify one item IRRELEVANT only under ALL §10B conditions.

    Requires supersession by a newer extraction of the same identity,
    no reference from any open claim, candidate, contradiction, or
    requirement, and at least one full unchanged round. Otherwise the
    record is untouched (it stays summarized, never deleted).
    IRRELEVANT means dropped-from-packet-with-count, retained in the
    canonical store under identity.
    """
    if (
        superseded_by_newer
        and not referenced
        and rounds_unchanged >= 1
    ):
        record.lifecycle = IRRELEVANT
        return True
    return False


def relevance_closure(
    *,
    active_candidates: Any = (),
    open_claims: Any = (),
    new_contradictions: Any = (),
    new_branches: Any = (),
    reactivated: Any = (),
    unresolved_requirements: Any = (),
) -> frozenset:
    """Deterministic Mode-A relevance closure (spec §27).

    Unions the spec-required inputs by identity. Callers prefetch
    every retained member before frontier construction.
    """
    closure: set = set()
    for group in (
        active_candidates, open_claims, new_contradictions,
        new_branches, reactivated, unresolved_requirements,
    ):
        for identity in group or ():
            closure.add(identity)
    return frozenset(closure)


def prefetch(
    store: CanonicalStore, closure: frozenset
) -> tuple[dict[Hashable, EvidenceRecord], list[Hashable]]:
    """Retrieve canonical records for closure members (spec §27).

    Returns ``(found, missing)`` with missing identities in
    deterministic order. Unknown identities are reported, never
    invented, so the LLM is never required to discover omitted
    evidence on its own.
    """
    found: dict[Hashable, EvidenceRecord] = {}
    missing: list[Hashable] = []
    for identity in sorted(closure, key=repr):
        record = store.get(identity)
        if record is None:
            missing.append(identity)
        else:
            found[identity] = record
    return found, missing


def closed_prefetch(
    store: CanonicalStore,
    *,
    active_candidates: Any = (),
    open_claims: Any = (),
    new_contradictions: Any = (),
    new_branches: Any = (),
    reactivated: Any = (),
    unresolved_requirements: Any = (),
) -> dict[Hashable, EvidenceRecord]:
    """Relevance closure plus fail-closed prefetch (spec §13, §27).

    Computes the Mode-A closure and retrieves every member. Any
    identity missing from the canonical store is a bounded-packet
    invariant failure: raise instead of silently dropping an open
    question from the decision input.
    """
    closure = relevance_closure(
        active_candidates=active_candidates,
        open_claims=open_claims,
        new_contradictions=new_contradictions,
        new_branches=new_branches,
        reactivated=reactivated,
        unresolved_requirements=unresolved_requirements,
    )
    found, missing = prefetch(store, closure)
    if missing:
        raise BoundedPacketError("invariant-violation")
    return found


# ----------------------------------------------------------------------
# Slice 3 — bounded frontier, carry-forward projection, round-trip
# invariant (spec §8, §9, §13, §19)
# ----------------------------------------------------------------------

import json as _json


def partition_delta(
    prior_identities: Any,
    current_identities: Any,
    *,
    changed: Any = (),
) -> tuple[set, set, set]:
    """Split current identities into (new, changed, unchanged).

    New items never appeared in identity terms in any prior round;
    changed items were sent before with different content; the three
    sets are disjoint and cover current exactly (spec §8).
    """
    prior = set(prior_identities or ())
    current = set(current_identities or ())
    changed_set = set(changed or ()) & current & prior
    new = current - prior
    unchanged = current & prior - changed_set
    return new, changed_set, unchanged


def _identity_sort_key(identity: Hashable) -> str:
    return repr(identity)


@dataclass
class DiscoveryFrontier:
    """Per-round bounded decision input (spec §8).

    Category payloads map stable identities to caller-supplied
    content dicts. Only new/changed/prefetched material belongs
    here; unchanged corpus lives in the summary by identity.
    """

    new_candidates: Mapping[Hashable, Any] = field(default_factory=dict)
    changed_assignments: Mapping[Hashable, Any] = field(default_factory=dict)
    changed_calls: Mapping[Hashable, Any] = field(default_factory=dict)
    changed_branches: Mapping[Hashable, Any] = field(default_factory=dict)
    new_helper_relationships: Mapping[Hashable, Any] = field(default_factory=dict)
    new_requirements: Mapping[Hashable, Any] = field(default_factory=dict)
    new_verification_candidates: Mapping[Hashable, Any] = field(default_factory=dict)
    unresolved_questions: Mapping[Hashable, Any] = field(default_factory=dict)
    contradictions: Mapping[Hashable, Any] = field(default_factory=dict)
    work_state: Mapping[str, Any] = field(default_factory=dict)

    def to_packet_dict(self) -> dict[str, Any]:
        """Deterministic packet mapping with identity-sorted sections."""
        payload: dict[str, Any] = {}
        for name in (
            "new_candidates", "changed_assignments", "changed_calls",
            "changed_branches", "new_helper_relationships",
            "new_requirements", "new_verification_candidates",
            "unresolved_questions", "contradictions",
        ):
            section = getattr(self, name) or {}
            payload[name] = {
                repr(identity): section[identity]
                for identity in sorted(section.keys(), key=_identity_sort_key)
            }
        payload["work_state"] = dict(self.work_state or {})
        return payload


@dataclass
class CarryForwardSummary:
    """Deterministic projection of canonical state (spec §9).

    Consumed, never modified, by the model. Fresh objects per build:
    mutating a summary cannot affect the canonical store.
    """

    coverage_map: dict[str, Any] = field(default_factory=dict)
    resolved_claims: list = field(default_factory=list)
    open_claims: list = field(default_factory=list)
    contradiction_ledger: list = field(default_factory=list)
    candidate_standings: list = field(default_factory=list)
    gate_tally: dict[str, Any] = field(default_factory=dict)

    def to_packet_dict(self) -> dict[str, Any]:
        """Deterministic packet mapping."""
        return {
            "coverage_map": dict(self.coverage_map or {}),
            "resolved_claims": list(self.resolved_claims or []),
            "open_claims": list(self.open_claims or []),
            "contradiction_ledger": list(self.contradiction_ledger or []),
            "candidate_standings": list(self.candidate_standings or []),
            "gate_tally": dict(self.gate_tally or {}),
        }


def build_summary(
    store: CanonicalStore,
    *,
    coverage_map: Mapping[str, Any],
    gate_tally: Mapping[str, Any],
    candidate_standings: Any,
    round_no: int = 0,
) -> CarryForwardSummary:
    """Project canonical lifecycles into summary sections (spec §9).

    RESOLVED records feed resolved claims with verdict history;
    ACTIVE/RETAINED_SUMMARY records feed open claims by identity;
    CONTRADICTED records feed the ledger with refutations and window
    identity preserved. IRRELEVANT items contribute counts only,
    never content.
    """
    resolved: list = []
    open_claims: list = []
    ledger: list = []
    irrelevant_count = 0
    for record in store.all_current():
        identity = list(record.identity) if isinstance(record.identity, tuple) else record.identity
        if record.lifecycle == RESOLVED:
            resolved.append({
                "identity": identity,
                "verdicts": [list(verdict) for verdict in record.prior_verdicts],
            })
        elif record.lifecycle == CONTRADICTED:
            ledger.append({
                "identity": identity,
                "refutations": [list(verdict) for verdict in record.prior_verdicts],
                "window_identity": record.window_identity,
            })
        elif record.lifecycle == IRRELEVANT:
            irrelevant_count += 1
        else:
            open_claims.append({
                "identity": identity,
                "lifecycle": record.lifecycle,
                "value_revision": record.value_revision,
                "rounds_waiting": max(0, round_no - record.first_seen_round),
            })
    summary = CarryForwardSummary(
        coverage_map=dict(coverage_map or {}),
        resolved_claims=resolved,
        open_claims=open_claims,
        contradiction_ledger=ledger,
        candidate_standings=list(candidate_standings or []),
        gate_tally=dict(gate_tally or {}),
    )
    if irrelevant_count:
        summary.coverage_map = dict(summary.coverage_map)
        summary.coverage_map["irrelevant_dropped_count"] = irrelevant_count
    return summary


def check_round_trip(
    *,
    full_identities: Any,
    frontier_identities: Any,
    summary_identities: Any,
    verdict_identities: Any,
) -> tuple[bool, dict[str, list]]:
    """Assert frontier + summary + verdicts cover exactly the full
    evidence set (spec §10, §13, §19).

    Returns ``(ok, {"missing": [...], "extra": [...]})`` with
    deterministic ordering. Missing items were dropped without
    license; extra items are unaccounted for.
    """
    full = set(full_identities or ())
    covered = (
        set(frontier_identities or ())
        | set(summary_identities or ())
        | set(verdict_identities or ())
    )
    missing = sorted(full - covered, key=_identity_sort_key)
    extra = sorted(covered - full, key=_identity_sort_key)
    return (not missing and not extra, {"missing": missing, "extra": extra})


def serialize_packet(packet: Mapping[str, Any]) -> str:
    """Deterministic JSON serialization for byte-identical repeated
    builds (spec §19). No tokenizer involvement."""
    return _json.dumps(packet, sort_keys=True, separators=(",", ":"), default=repr)


# ----------------------------------------------------------------------
# Slice 4 — bounded Mode-B lookup and traversal ownership (spec §15, §27)
# ----------------------------------------------------------------------

LOOKUP_COMPLETE = "lookup_complete"
LOOKUP_BUDGET_EXHAUSTED = "lookup_budget_exhausted"
LOOKUP_UNAVAILABLE = "lookup_unavailable"
FRONTIER_TOO_LARGE = "frontier_too_large"


@dataclass
class LookupBudget:
    """Declared Mode-B bounds for one discovery round (spec §27).

    The default maximum is exactly 1 lookup round. The configured
    value is echoed in every result so a raise can never be silent;
    changing it requires explicit evidence and spec review. A
    nonpositive round budget is a programming error.
    """

    max_rounds: int = 1
    max_identities: int = 8
    max_bytes: int = 65536

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("lookup max_rounds must be at least 1")
        if self.max_identities < 1:
            raise ValueError("lookup max_identities must be at least 1")
        if self.max_bytes < 1:
            raise ValueError("lookup max_bytes must be at least 1")


@dataclass
class LookupUsage:
    """Cumulative lookup accounting for one discovery round.

    Counts rounds, requests, and result bytes for packet-mass and
    cost accounting (spec §17, §27). Each lookup is a separate model
    invocation and counts as one.
    """

    lookup_request_count: int = 0
    lookup_round_count: int = 0
    lookup_result_bytes: int = 0
    lookup_budget_exhausted: bool = False

    def mass_fragment(
        self, *, prefetched_item_count: int, prefetched_bytes: int
    ) -> dict[str, Any]:
        """Exactly the committed §27 mass fields for P0 measurement."""
        return {
            "prefetched_item_count": int(prefetched_item_count),
            "prefetched_bytes": int(prefetched_bytes),
            "lookup_request_count": self.lookup_request_count,
            "lookup_result_bytes": self.lookup_result_bytes,
            "lookup_round_count": self.lookup_round_count,
            "lookup_budget_exhausted": self.lookup_budget_exhausted,
        }


@dataclass
class LookupResult:
    """One bounded lookup outcome (spec §27)."""

    status: str
    retrieved: dict = field(default_factory=dict)
    dropped_unknown_count: int = 0
    round_count: int = 0
    result_bytes: int = 0
    max_rounds: int = 1


def lookup(
    store: CanonicalStore,
    identities: Any,
    *,
    budget: LookupBudget,
    usage: LookupUsage,
) -> LookupResult:
    """Retrieve already-known canonical evidence by identity (spec §27).

    Mode B covers only the narrow case where a semantic judgment
    exposes a need for identity-addressable evidence that
    deterministic prefetch could not identify in advance. Unknown
    identities are dropped with a count; all-unknown requests yield
    LOOKUP_UNAVAILABLE. Exceeding any declared bound yields
    LOOKUP_BUDGET_EXHAUSTED without retrieving anything. This
    function accepts no traversal state and performs no source
    search: lookup is not traversal.
    """
    requested = list(identities or ())
    usage.lookup_request_count += 1
    if usage.lookup_round_count >= budget.max_rounds:
        usage.lookup_budget_exhausted = True
        return LookupResult(
            status=LOOKUP_BUDGET_EXHAUSTED, max_rounds=budget.max_rounds,
            round_count=usage.lookup_round_count,
        )
    if len(requested) > budget.max_identities:
        usage.lookup_budget_exhausted = True
        return LookupResult(
            status=LOOKUP_BUDGET_EXHAUSTED, max_rounds=budget.max_rounds,
            round_count=usage.lookup_round_count,
        )
    retrieved: dict[Hashable, Any] = {}
    dropped = 0
    for identity in requested:
        record = store.get(identity)
        if record is None:
            dropped += 1
        else:
            retrieved[identity] = dict(record.content)
    result_bytes = len(serialize_packet(
        {repr(identity): content for identity, content in retrieved.items()}
    ).encode("utf-8"))
    if result_bytes > budget.max_bytes:
        usage.lookup_budget_exhausted = True
        return LookupResult(
            status=LOOKUP_BUDGET_EXHAUSTED, max_rounds=budget.max_rounds,
            round_count=usage.lookup_round_count,
            dropped_unknown_count=dropped,
        )
    usage.lookup_round_count += 1
    usage.lookup_result_bytes += result_bytes
    if not retrieved:
        return LookupResult(
            status=LOOKUP_UNAVAILABLE,
            dropped_unknown_count=dropped,
            round_count=usage.lookup_round_count,
            result_bytes=result_bytes,
            max_rounds=budget.max_rounds,
        )
    return LookupResult(
        status=LOOKUP_COMPLETE,
        retrieved=retrieved,
        dropped_unknown_count=dropped,
        round_count=usage.lookup_round_count,
        result_bytes=result_bytes,
        max_rounds=budget.max_rounds,
    )


def check_frontier_budget(serialized: str, *, max_bytes: int) -> Optional[str]:
    """Report FRONTIER_TOO_LARGE for over-budget frontiers (spec §16).

    Returns None when the frontier fits. Never truncates: callers
    fail closed through paging first, then the unavailable result.
    """
    if len(serialized.encode("utf-8")) > max_bytes:
        return FRONTIER_TOO_LARGE
    return None


# ----------------------------------------------------------------------
# Slice 5a — resolver-category identities, requirement grouping,
# decision validation, packet mass, P0 fragment (spec §11, §12, §15,
# §17, §18)
# ----------------------------------------------------------------------

_GATE_OUTCOMES = (
    "satisfied", "contradicted", "verification_required", "unknown",
)


def _field(record: Any, name: str, default: Any = None) -> Any:
    value = getattr(record, name, default)
    if isinstance(value, dict):
        return value
    return value


def identity_for_assignment(ref: Any) -> tuple:
    """Stable identity reusing target/file/line (spec §11)."""
    return (
        "assignment",
        str(_field(ref, "target", "")),
        str(_field(ref, "file", "")),
        str(_field(ref, "line", "")),
    )


def identity_for_call(ref: Any) -> tuple:
    """Stable identity reusing name/site/order (spec §11)."""
    return (
        "call",
        str(_field(ref, "name", "")),
        str(_field(ref, "file", "")),
        str(_field(ref, "line", "")),
        str(_field(ref, "source_site_id", "") or ""),
        str(_field(ref, "source_order", "") if _field(ref, "source_order", None) is not None else ""),
    )


def identity_for_helper(ref: Any) -> tuple:
    """Stable identity reusing name/file/callable scope (spec §11)."""
    return (
        "helper",
        str(_field(ref, "name", "")),
        str(_field(ref, "file", "")),
        str(_field(ref, "line", "")),
        str(_field(ref, "callable_id", "") or ""),
    )


def identity_for_branch(ref: Any) -> tuple:
    """Stable identity reusing file/line/site/order, never bare
    condition text alone (spec §11)."""
    return (
        "branch",
        str(_field(ref, "file", "")),
        str(_field(ref, "line", "")),
        str(_field(ref, "source_site_id", "") or ""),
        str(_field(ref, "source_order", "") if _field(ref, "source_order", None) is not None else ""),
    )


def identity_for_requirement(req: Any) -> tuple:
    """Variant-level identity: name + predicate + site + role + gate
    outcome (spec §10D, §12). A gate/role change is a distinct
    variant; an actual-value change bumps the revision instead."""
    return (
        "requirement",
        str(_field(req, "name", "")),
        str(_field(req, "source_predicate", "") or ""),
        str(_field(req, "source_file", "") or ""),
        str(_field(req, "source_line", "") if _field(req, "source_line", None) is not None else ""),
        str(_field(req, "role", "") or ""),
        str(_field(req, "gate_result", "") or ""),
    )


def content_for_requirement(req: Any) -> dict[str, Any]:
    """Meaningful requirement content for revision tracking."""
    return {
        "actual_value": _field(req, "actual_value", None),
        "gate_outcome": _field(req, "gate_result", None),
        "semantic_effect": _field(req, "effect", None),
    }


def content_for_assignment(ref: Any) -> dict[str, Any]:
    """Meaningful assignment content for change detection."""
    return {"semantic_effect": _field(ref, "expression", None)}


def content_for_call(ref: Any) -> dict[str, Any]:
    """Meaningful call content for change detection."""
    return {"semantic_effect": (
        str(_field(ref, "name", "")),
        str(_field(ref, "receiver", "") or ""),
        str(_field(ref, "resolved_callable_id", "") or ""),
    )}


def content_for_helper(ref: Any) -> dict[str, Any]:
    """Meaningful helper content for change detection."""
    return {"semantic_effect": _field(ref, "return_expression", None)}


def content_for_branch(ref: Any) -> dict[str, Any]:
    """Meaningful branch content for change detection."""
    return {"semantic_effect": _field(ref, "condition", None)}


def group_requirements(requirements: Any) -> dict[str, Any]:
    """Group requirement variants under parameter name for
    display/counting only, retaining every variant explicitly with
    per-parameter gate-outcome rollups (spec §12).

    Returns ``{"groups": {name: [variant, ...]}, "rollup": {name:
    {"counts": {...}, "sites": [...]}}}`` with deterministic name
    order. Never collapses variants to scalars, never truncates.
    """
    groups: dict[str, list] = {}
    for requirement in requirements or ():
        name = str(_field(requirement, "name", "") or "")
        groups.setdefault(name, []).append(requirement)
    ordered = {name: groups[name] for name in sorted(groups.keys())}
    rollup: dict[str, Any] = {}
    for name, variants in ordered.items():
        counts = {outcome: 0 for outcome in _GATE_OUTCOMES}
        sites: list[str] = []
        for variant in variants:
            outcome = str(_field(variant, "gate_result", "") or "")
            if outcome in counts:
                counts[outcome] += 1
            site = (
                f"{_field(variant, 'source_file', '') or ''}"
                f":{_field(variant, 'source_line', '') if _field(variant, 'source_line', None) is not None else ''}"
            )
            if site not in sites:
                sites.append(site)
        rollup[name] = {"counts": counts, "sites": sites}
    return {"groups": ordered, "rollup": rollup}


def validate_decision_files(
    relevant_files: Any, *, eligible_files: Any
) -> tuple[list[str], int]:
    """Keep only deterministically eligible LLM-suggested files.

    Returns ``(kept, dropped_count)`` preserving suggestion order.
    Unknown identities drop with a count; they never become
    traversal commands here (spec §15).
    """
    eligible = set(eligible_files or ())
    kept = [path for path in (relevant_files or ()) if path in eligible]
    return kept, len(list(relevant_files or ())) - len(kept)


def measure_packet_mass(
    *,
    sections: Mapping[str, str],
    section_item_counts: Mapping[str, int],
    new_item_count: int,
    summary_bytes: int,
    frontier_bytes: int,
    full_equivalent_bytes: int,
    fallback_used: bool = False,
    fallback_reason: str = "",
    fallback_count: int = 0,
    fallback_bytes: int = 0,
) -> dict[str, Any]:
    """Deterministic pre-call packet-mass measurement (spec §17).

    Sizes come from the actual serialized payload sections; no
    tokenizer, no estimates. Includes the exact §16 fallback
    accounting names so fallback runs can never read as optimized
    success.
    """
    total_bytes = sum(len(text.encode("utf-8")) for text in sections.values())
    total_chars = sum(len(text) for text in sections.values())
    return {
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "section_bytes": {
            name: len(text.encode("utf-8")) for name, text in sections.items()
        },
        "section_chars": {
            name: len(text) for name, text in sections.items()
        },
        "section_item_counts": dict(section_item_counts or {}),
        "new_item_count": int(new_item_count),
        "summary_bytes": int(summary_bytes),
        "frontier_bytes": int(frontier_bytes),
        "retransmission_avoided_bytes": max(
            0, int(full_equivalent_bytes) - total_bytes
        ),
        "full_packet_fallback_used": bool(fallback_used),
        "full_packet_fallback_reason": str(fallback_reason),
        "full_packet_fallback_count": int(fallback_count),
        "full_packet_fallback_bytes": int(fallback_bytes),
    }


def p0_mass_fragment(mass: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap mass observations in the readiness artifact shape.

    Follows the version/status/thresholds/basis contract with
    thresholds null: measurement only, never a GREEN gate (spec §18).
    """
    return {
        "version": 1,
        "status": "measurement_only",
        "thresholds": None,
        "basis": dict(mass or {}),
    }


def process_lookup_requests(
    requests: Any,
    *,
    store: CanonicalStore,
    budget: LookupBudget,
    usage: LookupUsage,
) -> LookupResult:
    """Validate advisory lookup requests before retrieval (spec §27).

    Request strings match canonical identities by exact ``repr``;
    anything else drops with a count and never reaches retrieval.
    Validated identities retrieve in a single bounded round.
    """
    by_repr = {repr(identity): identity for identity in store._current.keys()}
    validated: list[Hashable] = []
    dropped = 0
    for request in requests or ():
        identity = by_repr.get(str(request))
        if identity is None:
            dropped += 1
        elif identity not in validated:
            validated.append(identity)
    if not validated:
        usage.lookup_request_count += 1
        return LookupResult(
            status=LOOKUP_UNAVAILABLE,
            dropped_unknown_count=dropped,
            round_count=usage.lookup_round_count,
            max_rounds=budget.max_rounds,
        )
    result = lookup(store, validated, budget=budget, usage=usage)
    result.dropped_unknown_count += dropped
    return result


@dataclass
class DiscoveryRoundState:
    """Per-run bounded-frontier bookkeeping for the resolver loop.

    Owns the canonical store, the set of already-sent identities
    (novelty baseline), prior log-context deltas, fallback counters,
    and mass records. Traversal state (visited files, queries,
    exhaustion) is NOT owned here; the resolver loop keeps it.
    """

    store: CanonicalStore = field(default_factory=CanonicalStore)
    prior_sent_identities: set = field(default_factory=set)
    prior_parameter_names: set = field(default_factory=set)
    prior_topics: set = field(default_factory=set)
    prior_draft_titles: list = field(default_factory=list)
    all_observed_identities: set = field(default_factory=set)
    round_no: int = 0
    fallback_count: int = 0
    fallback_bytes_total: int = 0
    mass_log: list = field(default_factory=list)


class BoundedPacketError(Exception):
    """Bounded packet construction failed; the caller must fail
    closed to the current full packet for the round (spec §16)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def identity_for_predicate(ref: Any) -> tuple:
    """Stable predicate identity reusing name/predicate/site (spec §11)."""
    return (
        "predicate",
        str(_field(ref, "name", "") or ""),
        str(_field(ref, "predicate", "") or ""),
        str(_field(ref, "file", "") or ""),
        str(_field(ref, "line", "") if _field(ref, "line", None) is not None else ""),
    )


def identity_for_topic(ref: Any) -> tuple:
    """Stable topic identity reusing topic/file/line (spec §11)."""
    return (
        "topic",
        str(_field(ref, "topic", "") or ""),
        str(_field(ref, "file", "") or ""),
        str(_field(ref, "line", "") if _field(ref, "line", None) is not None else ""),
    )


def identity_for_field(ref: Any) -> tuple:
    """Stable field identity reusing field/file/line (spec §11)."""
    return (
        "field",
        str(_field(ref, "field", "") or ""),
        str(_field(ref, "file", "") or ""),
        str(_field(ref, "line", "") if _field(ref, "line", None) is not None else ""),
    )


def identity_for_parameter_ref(ref: Any) -> tuple:
    """Stable parameter-reference identity reusing name/site (spec §11)."""
    return (
        "paramref",
        str(_field(ref, "name", "") or ""),
        str(_field(ref, "file", "") or ""),
        str(_field(ref, "line", "") if _field(ref, "line", None) is not None else ""),
    )


def identity_for_file(path: str) -> tuple:
    """Stable file identity reusing the relative path (spec §11)."""
    return ("file", str(path))


def identity_for_verification(name: str, file: str, line: Any) -> tuple:
    """Stable verification-candidate identity (spec §11)."""
    return (
        "verification",
        str(name or ""),
        str(file or ""),
        str(line) if line is not None else "",
    )


def requirement_base_key(identity: Hashable) -> tuple:
    """Variant-independent base: name + predicate + site (spec §10D).

    Lets a newly contradicted variant touch the retained
    predecessor of the same base for reactivation.
    """
    parts = tuple(identity) if isinstance(identity, tuple) else (identity,)
    if parts[:1] == ("requirement",) and len(parts) >= 5:
        return parts[:5]
    return parts


def _dump_content(ref: Any) -> dict[str, Any]:
    """Full model dump when available, else scalar attributes.

    The fingerprint still selects only decision-relevant keys, so
    richer content never changes revision semantics; it only makes
    prefetched full text complete.
    """
    dumped = getattr(ref, "model_dump", None)
    if callable(dumped):
        try:
            value = dumped()
            if isinstance(value, dict):
                return value
        except (TypeError, ValueError):
            pass
    scalars: dict[str, Any] = {}
    try:
        attributes = sorted(vars(ref).keys())
    except TypeError:
        attributes = ()
    for key in attributes:
        value = getattr(ref, key, None)
        if value is None or isinstance(value, (str, int, float, bool)):
            scalars[str(key)] = value
    scalars.setdefault("semantic_effect", repr(ref))
    return scalars


def observe_discovery_round(
    state: DiscoveryRoundState,
    *,
    round_no: int,
    assignments: Any = (),
    calls: Any = (),
    helpers: Any = (),
    branches: Any = (),
    predicates: Any = (),
    topics: Any = (),
    fields: Any = (),
    parameter_refs: Any = (),
    requirements: Any = (),
    files: Any = (),
) -> dict[str, Any]:
    """Observe one extraction round into the canonical store.

    Maps every ref to its stable identity, tracks new vs changed
    (revision-bumped) identities, resolves requirement gate
    outcomes, and summarizes quiet ACTIVE records with no open
    question attached. Returns per-category identity maps plus
    ``new_all``, ``changed_all``, ``open_requirements``,
    ``contradicted``, ``changed_requirements``, and ``gate_tally``.
    """
    store = state.store
    by_category: dict[str, dict] = {
        "assignment": {}, "call": {}, "helper": {}, "branch": {},
        "predicate": {}, "topic": {}, "field": {}, "paramref": {},
        "requirement": {}, "file": {},
    }

    def track(category: str, identity: Hashable, ref: Any) -> None:
        by_category[category][identity] = ref
        state.all_observed_identities.add(identity)

    for path in files or ():
        identity = identity_for_file(str(path))
        store.observe(identity, kind="file",
                      content={"semantic_effect": str(path)},
                      round_no=round_no)
        track("file", identity, str(path))

    for ref in assignments or ():
        identity = identity_for_assignment(ref)
        store.observe(identity, kind="assignment",
                      content=content_for_assignment(ref), round_no=round_no)
        track("assignment", identity, ref)
    for ref in calls or ():
        identity = identity_for_call(ref)
        store.observe(identity, kind="call",
                      content=content_for_call(ref), round_no=round_no)
        track("call", identity, ref)
    for ref in helpers or ():
        identity = identity_for_helper(ref)
        store.observe(identity, kind="helper",
                      content=content_for_helper(ref), round_no=round_no)
        track("helper", identity, ref)
    for ref in branches or ():
        identity = identity_for_branch(ref)
        store.observe(identity, kind="branch",
                      content=content_for_branch(ref), round_no=round_no)
        track("branch", identity, ref)
    for ref in predicates or ():
        identity = identity_for_predicate(ref)
        store.observe(identity, kind="predicate",
                      content={"semantic_effect": _field(ref, "predicate", None)},
                      round_no=round_no)
        track("predicate", identity, ref)
    for ref in topics or ():
        identity = identity_for_topic(ref)
        store.observe(identity, kind="topic",
                      content={"semantic_effect": (
                          str(_field(ref, "topic", "") or ""),
                          str(_field(ref, "direction", "") or ""),
                      )}, round_no=round_no)
        track("topic", identity, ref)
    for ref in fields or ():
        identity = identity_for_field(ref)
        store.observe(identity, kind="field",
                      content={"semantic_effect": str(_field(ref, "field", "") or "")},
                      round_no=round_no)
        track("field", identity, ref)
    for ref in parameter_refs or ():
        identity = identity_for_parameter_ref(ref)
        store.observe(identity, kind="paramref",
                      content={"semantic_effect": str(_field(ref, "name", "") or "")},
                      round_no=round_no)
        track("paramref", identity, ref)

    open_requirements: list = []
    contradicted: list = []
    changed_requirements: list = []
    for req in requirements or ():
        identity = identity_for_requirement(req)
        previous = store.get(identity)
        record = store.observe(
            identity, kind="requirement",
            content=_dump_content(req), round_no=round_no)
        track("requirement", identity, req)
        gate = str(_field(req, "gate_result", "") or "")
        if previous is not None and record.value_revision > previous.value_revision:
            changed_requirements.append((identity, True))
        if record.lifecycle == ACTIVE:
            if gate == "satisfied":
                store.resolve(identity, outcome="admitted",
                              reason="gate-satisfied")
            elif gate == "contradicted":
                store.refute(identity, reason="gate-contradicted")
        if gate in {"verification_required", "unknown"}:
            open_requirements.append(identity)
        if gate == "contradicted":
            contradicted.append(identity)

    open_set = set(open_requirements)
    for record in store.all_current():
        if (
            record.lifecycle == ACTIVE
            and record.last_changed_round < round_no
            and record.identity not in open_set
        ):
            store.summarize(record.identity)

    gate_tally = {outcome: 0 for outcome in _GATE_OUTCOMES}
    for req in requirements or ():
        outcome = str(_field(req, "gate_result", "") or "")
        if outcome in gate_tally:
            gate_tally[outcome] += 1

    current_all = set()
    for mapping in by_category.values():
        current_all.update(mapping.keys())
    new_all = {identity for identity in current_all
               if identity not in state.prior_sent_identities}
    changed_all = {
        identity for identity in current_all
        if identity in state.prior_sent_identities
        and (store.get(identity) is not None
             and store.get(identity).last_changed_round == round_no)
    }
    return {
        "by_category": by_category,
        "new_all": new_all,
        "changed_all": changed_all,
        "open_requirements": open_requirements,
        "contradicted": contradicted,
        "changed_requirements": changed_requirements,
        "gate_tally": gate_tally,
    }
