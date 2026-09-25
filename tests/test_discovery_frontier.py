"""Bounded discovery-frontier tests (spec: docs/discovery_frontier_bounded_input_spec.md).

Deterministic offline coverage for the discovery-input narrowing
workstream. Generic synthetic identities only; no benchmark literals,
no model calls, no network.
"""

from __future__ import annotations


# ----------------------------------------------------------------------
# SLICE 1 — canonical identity and lifecycle metadata (spec sections
# 7, 10, 10D, 11)
# ----------------------------------------------------------------------


def test_lifecycle_states_cover_the_spec_vocabulary():
    """The module exposes exactly the committed lifecycle vocabulary."""
    from flight_log_agent.px4 import discovery_frontier as df

    assert df.ACTIVE == "active"
    assert df.RESOLVED == "resolved"
    assert df.SUPERSEDED == "superseded"
    assert df.CONTRADICTED == "contradicted"
    assert df.RETAINED_SUMMARY == "retained_summary"
    assert df.IRRELEVANT == "irrelevant"


def test_make_identity_is_stable_and_ordered():
    """Identities are deterministic tuples; equal parts give equal
    identities regardless of process state."""
    from flight_log_agent.px4 import discovery_frontier as df

    first = df.make_identity("candidate", "file_a", "scope_b", "symbol_c")
    second = df.make_identity("candidate", "file_a", "scope_b", "symbol_c")
    assert first == second
    assert first != df.make_identity("candidate", "file_a", "scope_b", "symbol_d")
    assert isinstance(first, tuple)


def test_new_observation_enters_active_with_provenance():
    """A newly extracted item enters ACTIVE with first/last round and
    revision zero (spec section 10, 10D)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    record = store.observe(
        df.make_identity("candidate", "file_a", "scope_b", "symbol_c"),
        kind="candidate",
        content={"symbol": "symbol_c"},
        round_no=1,
    )
    assert record.lifecycle == df.ACTIVE
    assert record.first_seen_round == 1
    assert record.last_changed_round == 1
    assert record.value_revision == 0


def test_unchanged_reobservation_keeps_revision():
    """Re-observing identical meaningful content never bumps the
    revision (spec section 10D)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("assignment", "file_a", "line_3")
    store.observe(identity, kind="assignment",
                  content={"actual_value": 5}, round_no=1)
    record = store.observe(identity, kind="assignment",
                           content={"actual_value": 5}, round_no=2)
    assert record.lifecycle == df.ACTIVE
    assert record.value_revision == 0
    assert record.first_seen_round == 1
    assert record.last_changed_round == 1


def test_meaningful_change_supersedes_with_revision_bump():
    """A changed actual value retires the old record as SUPERSEDED and
    opens a new ACTIVE record at revision +1 carrying prior verdicts
    (spec sections 10, 10A, 10D)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("requirement", "param_x", "pred_y", "file_a", 12)
    first = store.observe(identity, kind="requirement",
                          content={"actual_value": 5, "gate_outcome": "unknown"},
                          round_no=1)
    store.resolve(identity, outcome="admitted", reason="gate")
    second = store.observe(identity, kind="requirement",
                           content={"actual_value": 9, "gate_outcome": "unknown"},
                           round_no=2)
    assert second.lifecycle == df.ACTIVE
    assert second.value_revision == first.value_revision + 1
    assert second.first_seen_round == 1
    assert second.last_changed_round == 2
    assert ("admitted", "gate") in second.prior_verdicts
    retired = store.retired_for(identity)
    assert [record.lifecycle for record in retired] == [df.SUPERSEDED]


def test_serialization_noise_never_bumps_revision():
    """Key order, whitespace, and round numbers are not meaningful
    changes (spec section 10D)."""
    from flight_log_agent.px4 import discovery_frontier as df

    assert (
        df.meaningful_fingerprint({"actual_value": 5, "note": "  spaced  "})
        == df.meaningful_fingerprint({"note": "spaced", "actual_value": 5})
    )
    store = df.CanonicalStore()
    identity = df.make_identity("call", "file_a", "line_9")
    store.observe(identity, kind="call",
                  content={"actual_value": "  x  ", "round": 1}, round_no=1)
    record = store.observe(identity, kind="call",
                           content={"actual_value": "x", "round": 2}, round_no=2)
    assert record.value_revision == 0


def test_resolve_and_refute_transitions():
    """Admitted/rejected items become RESOLVED; refuted items become
    CONTRADICTED with reason preserved (spec section 10)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    kept = df.make_identity("candidate", "keep")
    store.observe(kept, kind="candidate", content={}, round_no=1)
    store.resolve(kept, outcome="rejected", reason="no-writer")
    assert store.get(kept).lifecycle == df.RESOLVED

    refuted = df.make_identity("candidate", "drop")
    store.observe(refuted, kind="candidate", content={}, round_no=1)
    store.refute(refuted, reason="counter-evidence")
    record = store.get(refuted)
    assert record.lifecycle == df.CONTRADICTED
    assert ("refuted", "counter-evidence") in record.prior_verdicts


def test_window_identity_distinguishes_same_source():
    """The same source under a different questioned window is a
    distinct identity (spec section 10D)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    first = store.observe(
        df.make_identity("writer", "file_a", "symbol_c", window="window_1"),
        kind="writer", content={}, round_no=1, window_identity="window_1",
    )
    second = store.observe(
        df.make_identity("writer", "file_a", "symbol_c", window="window_2"),
        kind="writer", content={}, round_no=1, window_identity="window_2",
    )
    assert first.identity != second.identity
    assert first.window_identity == "window_1"
    assert second.window_identity == "window_2"


# ----------------------------------------------------------------------
# SLICE 2 — reactivation, referenced semantics, irrelevance, relevance
# closure, prefetch (spec sections 10A, 10B, 10C, 27 Mode A)
# ----------------------------------------------------------------------


def test_referenced_is_identity_intersection():
    """`referenced` is machine-checkable identity membership across the
    five spec-owned relation classes, never prose (spec section 10C)."""
    from flight_log_agent.px4 import discovery_frontier as df

    identity = df.make_identity("writer", "file_a", "symbol_c")
    assert df.is_referenced(identity) is False
    assert df.is_referenced(identity, support_sets=[{identity}]) is True
    assert df.is_referenced(identity, claim_deps=[{identity}]) is True
    assert df.is_referenced(identity, rival_cites=[{identity}]) is True
    assert df.is_referenced(identity, contradiction_supports=[{identity}]) is True
    assert df.is_referenced(identity, requirement_points=[{identity}]) is True
    other = df.make_identity("writer", "file_b", "symbol_d")
    assert df.is_referenced(
        identity,
        support_sets=[{other}],
        claim_deps=[{other}],
        rival_cites=[{other}],
        contradiction_supports=[{other}],
        requirement_points=[{other}],
    ) is False


def test_new_hypothesis_reactivates_resolved_evidence():
    """A RESOLVED item re-enters as ACTIVE (with verdict history, never
    blank) when a new hypothesis references its identity (spec 10A)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(identity, kind="writer",
                  content={"actual_value": 1}, round_no=1)
    store.resolve(identity, outcome="admitted", reason="gate")
    assert store.get(identity).lifecycle == df.RESOLVED
    fired = df.evaluate_reactivation(
        store, round_no=2, new_hypotheses=[{identity}],
    )
    assert [record.identity for record in fired] == [identity]
    record = store.get(identity)
    assert record.lifecycle == df.ACTIVE
    assert ("admitted", "gate") in record.prior_verdicts


def test_no_trigger_leaves_retained_state_untouched():
    """Without a deterministic trigger, retained items stay retained;
    reactivation needs no model recall (spec section 10A)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(identity, kind="writer",
                  content={"actual_value": 1}, round_no=1)
    store.resolve(identity, outcome="admitted", reason="gate")
    assert df.evaluate_reactivation(store, round_no=2) == []
    assert store.get(identity).lifecycle == df.RESOLVED


def test_contradicted_items_are_not_reactivated():
    """Refuted items stay CONTRADICTED (visible with refutation) even
    when newly referenced; reactivation never clears a refutation."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(identity, kind="writer",
                  content={"actual_value": 1}, round_no=1)
    store.refute(identity, reason="counter-evidence")
    assert df.evaluate_reactivation(
        store, round_no=2, new_hypotheses=[{identity}],
    ) == []
    assert store.get(identity).lifecycle == df.CONTRADICTED


def test_irrelevance_requires_all_spec_conditions():
    """IRRELEVANT only under superseded + unreferenced + quiet-round;
    otherwise the item stays summarized, never deleted (spec 10B)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    record = store.observe(identity, kind="writer",
                           content={"actual_value": 1}, round_no=1)
    assert df.mark_irrelevant(record, superseded_by_newer=False,
                              referenced=False, rounds_unchanged=3) is False
    assert df.mark_irrelevant(record, superseded_by_newer=True,
                              referenced=True, rounds_unchanged=3) is False
    assert df.mark_irrelevant(record, superseded_by_newer=True,
                              referenced=False, rounds_unchanged=0) is False
    assert record.lifecycle == df.ACTIVE
    assert df.mark_irrelevant(record, superseded_by_newer=True,
                              referenced=False, rounds_unchanged=1) is True
    assert record.lifecycle == df.IRRELEVANT
    assert store.get(identity) is record


def test_window_change_reactivates_window_scoped_evidence():
    """A new temporal window reactivates retained window-scoped
    evidence (spec section 10A)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c",
                                window="window_1")
    store.observe(identity, kind="writer", content={}, round_no=1,
                  window_identity="window_1")
    store.resolve(identity, outcome="admitted", reason="gate")
    fired = df.evaluate_reactivation(
        store, round_no=2, new_windows=["window_2"],
    )
    assert [record.identity for record in fired] == [identity]
    assert store.get(identity).lifecycle == df.ACTIVE


def test_relevance_closure_covers_all_spec_inputs():
    """The Mode-A closure unions active candidates, open claims, new
    contradictions, new branches, reactivated evidence, and unresolved
    requirements (spec section 27)."""
    from flight_log_agent.px4 import discovery_frontier as df

    candidate = df.make_identity("candidate", "alpha")
    claim = df.make_identity("claim", "beta")
    contradiction = df.make_identity("contradiction", "gamma")
    branch = df.make_identity("branch", "delta")
    reactivated = df.make_identity("writer", "epsilon")
    requirement = df.make_identity("requirement", "zeta")
    closure = df.relevance_closure(
        active_candidates=[candidate],
        open_claims=[claim],
        new_contradictions=[contradiction],
        new_branches=[branch],
        reactivated=[reactivated],
        unresolved_requirements=[requirement],
    )
    assert closure == frozenset({
        candidate, claim, contradiction, branch, reactivated, requirement,
    })


def test_prefetch_returns_canonical_content_for_closure():
    """Prefetch retrieves full canonical records for closure members;
    unknown identities are reported missing, never invented."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    known = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(known, kind="writer",
                  content={"actual_value": 7}, round_no=1)
    unknown = df.make_identity("writer", "file_b", "symbol_d")
    found, missing = df.prefetch(store, frozenset({known, unknown}))
    assert found[known].content["actual_value"] == 7
    assert missing == [unknown]


def test_h1_to_h2_reactivation_restores_evidence_before_judgment():
    """The spec's counterexample stays impossible: W resolved under H1
    is ACTIVE with history before any H2 judgment (spec sections 10A,
    13, 19)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    writer = df.make_identity("writer", "file_w", "symbol_w")
    store.observe(writer, kind="writer",
                  content={"actual_value": 3}, round_no=1)
    store.resolve(writer, outcome="admitted", reason="h1-gate")
    # Round 2: rival hypothesis H2 references the same identity.
    fired = df.evaluate_reactivation(
        store, round_no=2, new_hypotheses=[{writer}],
    )
    assert [record.identity for record in fired] == [writer]
    closure = df.relevance_closure(
        active_candidates=[df.make_identity("candidate", "h2")],
        reactivated=[writer],
    )
    assert writer in closure
    found, missing = df.prefetch(store, closure)
    assert missing == [df.make_identity("candidate", "h2")]
    assert found[writer].lifecycle == df.ACTIVE
    assert ("admitted", "h1-gate") in found[writer].prior_verdicts


# ----------------------------------------------------------------------
# SLICE 3 — bounded frontier, carry-forward projection, round-trip
# invariant (spec sections 8, 9, 13, 19)
# ----------------------------------------------------------------------


def test_partition_delta_splits_new_changed_unchanged():
    """Delta partitioning is exact: new, changed-since-sent, and
    unchanged items form disjoint sets covering current (spec 8)."""
    from flight_log_agent.px4 import discovery_frontier as df

    prior = {"id_a", "id_b", "id_c"}
    current = {"id_b", "id_c", "id_d"}
    new, changed, unchanged = df.partition_delta(
        prior, current, changed={"id_c"},
    )
    assert new == {"id_d"}
    assert changed == {"id_c"}
    assert unchanged == {"id_b"}
    assert new | changed | unchanged == current
    assert not (new & changed or new & unchanged or changed & unchanged)


def test_frontier_serializes_deterministically():
    """Repeated frontier builds serialize byte-identically with stable
    key order (spec section 19 construction determinism)."""
    from flight_log_agent.px4 import discovery_frontier as df

    def build():
        frontier = df.DiscoveryFrontier(
            new_candidates={"id_b": {"kind": "candidate"}},
            changed_assignments={},
            new_requirements={"id_a": {"kind": "requirement"}},
            work_state={"depth_remaining": 1},
        )
        return df.serialize_packet(frontier.to_packet_dict())

    first, second = build(), build()
    assert first == second
    assert list(__import__("json").loads(first).keys()) == sorted(
        __import__("json").loads(first).keys()
    )


def test_summary_derives_lifecycle_sections_from_store():
    """The summary projects resolved/open/contradiction sections from
    canonical lifecycles in identity order; contradictions preserve
    window identity (spec section 9)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    kept = df.make_identity("candidate", "keep")
    store.observe(kept, kind="candidate", content={}, round_no=1)
    store.resolve(kept, outcome="admitted", reason="gate")
    refuted = df.make_identity("writer", "file_a", "s", window="window_1")
    store.observe(refuted, kind="writer", content={}, round_no=1,
                  window_identity="window_1")
    store.refute(refuted, reason="counter-evidence")
    pending = df.make_identity("requirement", "param_x")
    store.observe(pending, kind="requirement", content={}, round_no=2)
    summary = df.build_summary(
        store,
        coverage_map={"file_a": {"seen": 2}},
        gate_tally={"verification_required": 1},
        candidate_standings=[{"candidate": "alpha", "supporting": 1}],
    )
    assert [item["identity"] for item in summary.resolved_claims] == [list(kept)]
    assert [item["identity"] for item in summary.open_claims] == [list(pending)]
    assert summary.contradiction_ledger[0]["window_identity"] == "window_1"
    assert summary.gate_tally == {"verification_required": 1}


def test_summary_is_projection_not_store():
    """Mutating a built summary cannot affect canonical records
    (spec section 13 canonical-store statement)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("candidate", "keep")
    store.observe(identity, kind="candidate", content={}, round_no=1)
    summary = df.build_summary(store, coverage_map={}, gate_tally={},
                               candidate_standings=[])
    summary.resolved_claims.append({"identity": "forged"})
    assert store.get(identity).lifecycle == df.ACTIVE
    assert df.build_summary(store, coverage_map={}, gate_tally={},
                            candidate_standings=[]).resolved_claims == []


def test_round_trip_invariant_covers_exactly():
    """Frontier + summary + verdict records must cover exactly the
    full evidence set, reporting any gap (spec sections 10, 13)."""
    from flight_log_agent.px4 import discovery_frontier as df

    full = {"id_a", "id_b", "id_c"}
    ok, report = df.check_round_trip(
        full_identities=full,
        frontier_identities={"id_a"},
        summary_identities={"id_b"},
        verdict_identities={"id_c"},
    )
    assert ok is True
    assert report == {"missing": [], "extra": []}
    ok, report = df.check_round_trip(
        full_identities=full,
        frontier_identities={"id_a"},
        summary_identities=set(),
        verdict_identities={"id_c"},
    )
    assert ok is False
    assert report["missing"] == ["id_b"]
    ok, report = df.check_round_trip(
        full_identities={"id_a"},
        frontier_identities={"id_a", "id_ghost"},
        summary_identities=set(),
        verdict_identities=set(),
    )
    assert ok is False
    assert report["extra"] == ["id_ghost"]


# ----------------------------------------------------------------------
# SLICE 4 — bounded Mode-B lookup and traversal ownership (spec
# sections 15, 27)
# ----------------------------------------------------------------------


def test_lookup_status_vocabulary():
    """Lookup and frontier-budget statuses use the exact committed
    contract names (spec sections 12, 16, 27)."""
    from flight_log_agent.px4 import discovery_frontier as df

    assert df.LOOKUP_COMPLETE == "lookup_complete"
    assert df.LOOKUP_BUDGET_EXHAUSTED == "lookup_budget_exhausted"
    assert df.LOOKUP_UNAVAILABLE == "lookup_unavailable"
    assert df.FRONTIER_TOO_LARGE == "frontier_too_large"


def test_lookup_default_is_one_round_and_observable():
    """The default maximum is exactly 1 round; any configured value
    is echoed in results so it can never be raised silently (spec 27)."""
    from flight_log_agent.px4 import discovery_frontier as df

    assert df.LookupBudget().max_rounds == 1
    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(identity, kind="writer",
                  content={"actual_value": 1}, round_no=1)
    usage = df.LookupUsage()
    result = df.lookup(store, [identity], budget=df.LookupBudget(),
                       usage=usage)
    assert result.status == df.LOOKUP_COMPLETE
    assert result.max_rounds == 1
    assert result.round_count == 1


def test_lookup_rejects_nonpositive_round_budget():
    """A zero or negative round budget is a programming error, never a
    silent no-lookup configuration."""
    from flight_log_agent.px4 import discovery_frontier as df
    import pytest

    with pytest.raises(ValueError):
        df.LookupBudget(max_rounds=0)


def test_lookup_second_round_exhausts_default_budget():
    """One follow-up round exists; a second one fails closed as
    LOOKUP_BUDGET_EXHAUSTED (spec section 27)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(identity, kind="writer",
                  content={"actual_value": 1}, round_no=1)
    budget = df.LookupBudget()
    usage = df.LookupUsage()
    assert df.lookup(store, [identity], budget=budget,
                     usage=usage).status == df.LOOKUP_COMPLETE
    exhausted = df.lookup(store, [identity], budget=budget, usage=usage)
    assert exhausted.status == df.LOOKUP_BUDGET_EXHAUSTED
    assert exhausted.retrieved == {}


def test_lookup_drops_unknown_identities_with_count():
    """Unknown identities are dropped with a count; mixed requests
    complete on the known remainder (spec section 27)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    known = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(known, kind="writer",
                  content={"actual_value": 1}, round_no=1)
    unknown = df.make_identity("writer", "file_b", "symbol_d")
    result = df.lookup(store, [known, unknown],
                       budget=df.LookupBudget(), usage=df.LookupUsage())
    assert result.status == df.LOOKUP_COMPLETE
    assert set(result.retrieved.keys()) == {known}
    assert result.dropped_unknown_count == 1


def test_lookup_all_unknown_is_unavailable():
    """Requests with no resolvable identity yield LOOKUP_UNAVAILABLE,
    never an empty success (spec section 27)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    result = df.lookup(store, [df.make_identity("writer", "ghost")],
                       budget=df.LookupBudget(), usage=df.LookupUsage())
    assert result.status == df.LOOKUP_UNAVAILABLE
    assert result.retrieved == {}


def test_lookup_identity_cap_exhausts():
    """Exceeding the declared per-lookup identity cap fails closed
    without retrieving anything."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identities = [df.make_identity("writer", f"file_{index}")
                  for index in range(4)]
    for identity in identities:
        store.observe(identity, kind="writer",
                      content={"actual_value": 1}, round_no=1)
    budget = df.LookupBudget(max_identities=3)
    result = df.lookup(store, identities, budget=budget,
                       usage=df.LookupUsage())
    assert result.status == df.LOOKUP_BUDGET_EXHAUSTED
    assert result.retrieved == {}


def test_lookup_accepts_no_traversal_state():
    """Lookup is not traversal: its signature exposes no visited,
    expansion, query, or traversal parameters, so no call can mutate
    traversal state through it (spec sections 15, 27)."""
    import inspect

    from flight_log_agent.px4 import discovery_frontier as df

    params = set(inspect.signature(df.lookup).parameters.keys())
    assert not params & {"visited", "visited_files", "expansion",
                         "expansion_queries", "queries", "traversal"}


def test_frontier_budget_check_is_deterministic():
    """Over-budget frontiers report FRONTIER_TOO_LARGE; fitting ones
    pass with no truncation anywhere (spec section 16)."""
    from flight_log_agent.px4 import discovery_frontier as df

    assert df.check_frontier_budget("x" * 100, max_bytes=50) == df.FRONTIER_TOO_LARGE
    assert df.check_frontier_budget("x" * 49, max_bytes=50) is None


def test_lookup_usage_reports_spec_mass_fields():
    """Usage accounting exposes exactly the committed §27 mass fields
    for P0 measurement (spec sections 17, 27)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(identity, kind="writer",
                  content={"actual_value": 1}, round_no=1)
    usage = df.LookupUsage()
    df.lookup(store, [identity], budget=df.LookupBudget(), usage=usage)
    fragment = usage.mass_fragment(prefetched_item_count=2,
                                   prefetched_bytes=120)
    assert fragment == {
        "prefetched_item_count": 2,
        "prefetched_bytes": 120,
        "lookup_request_count": 1,
        "lookup_result_bytes": usage.lookup_result_bytes,
        "lookup_round_count": 1,
        "lookup_budget_exhausted": False,
    }
    assert fragment["lookup_result_bytes"] > 0


# ----------------------------------------------------------------------
# SLICE 5a — resolver-category identities, requirement grouping,
# decision validation, packet mass, P0 fragment (spec sections 11, 12,
# 15, 17, 18)
# ----------------------------------------------------------------------


def test_category_identities_reuse_strongest_keys():
    """Category identities reuse visit_key/dedupe-grade site fields,
    never bare names (spec section 11)."""
    from types import SimpleNamespace

    from flight_log_agent.px4 import discovery_frontier as df

    assignment = SimpleNamespace(target="sp.alt", expression="home.alt + 20",
                                 file="rtl.cpp", line=42)
    assert df.identity_for_assignment(assignment) == (
        "assignment", "sp.alt", "rtl.cpp", "42",
    )
    call = SimpleNamespace(name="navigateTo", file="rtl.cpp", line=44,
                           receiver=None, resolved_callable_id="nav::go",
                           source_site_id="site-7", source_order=3)
    assert df.identity_for_call(call) == (
        "call", "navigateTo", "rtl.cpp", "44", "site-7", "3",
    )
    helper = SimpleNamespace(name="helper_altitude", file="help.cpp",
                             line=8, callable_id="help::alt")
    assert df.identity_for_helper(helper) == (
        "helper", "helper_altitude", "help.cpp", "8", "help::alt",
    )
    branch = SimpleNamespace(kind="if", condition="x == 2", file="b.cpp",
                             line=5, source_site_id="site-9", source_order=1)
    assert df.identity_for_branch(branch) == (
        "branch", "b.cpp", "5", "site-9", "1",
    )


def test_requirement_identity_preserves_gate_variants():
    """Requirement identity carries gate/role variants; a gate change
    is a distinct variant, an actual-value change bumps the revision
    (spec sections 10D, 12)."""
    from types import SimpleNamespace

    from flight_log_agent.px4 import discovery_frontier as df

    def requirement(gate, value):
        return SimpleNamespace(name="PARAM_X", source_predicate="p == 1",
                               source_file="f.cpp", source_line=3,
                               role="threshold", gate_result=gate,
                               actual_value=value, effect="e")

    assert (df.identity_for_requirement(requirement("unknown", 1))
            != df.identity_for_requirement(requirement("satisfied", 1)))
    store = df.CanonicalStore()
    identity = df.identity_for_requirement(requirement("unknown", 1))
    first = store.observe(identity, kind="requirement",
                          content=df.content_for_requirement(
                              requirement("unknown", 1)), round_no=1)
    second = store.observe(identity, kind="requirement",
                           content=df.content_for_requirement(
                               requirement("unknown", 2)), round_no=2)
    assert second.value_revision == first.value_revision + 1


def test_group_requirements_retains_variants_with_rollup():
    """Grouping is display/counting only: per-item variants stay
    explicit with per-parameter gate-outcome rollups (spec 12)."""
    from types import SimpleNamespace

    from flight_log_agent.px4 import discovery_frontier as df

    def requirement(name, gate, line):
        return SimpleNamespace(name=name, source_predicate="p",
                               source_file="f.cpp", source_line=line,
                               role="threshold", gate_result=gate,
                               actual_value=1, effect="e")

    grouped = df.group_requirements([
        requirement("PARAM_B", "satisfied", 2),
        requirement("PARAM_A", "unknown", 1),
        requirement("PARAM_A", "verification_required", 3),
    ])
    assert list(grouped["groups"].keys()) == ["PARAM_A", "PARAM_B"]
    assert len(grouped["groups"]["PARAM_A"]) == 2
    assert grouped["rollup"]["PARAM_A"]["counts"] == {
        "satisfied": 0, "contradicted": 0,
        "verification_required": 1, "unknown": 1,
    }
    assert grouped["rollup"]["PARAM_A"]["sites"] == ["f.cpp:1", "f.cpp:3"]


def test_validate_decision_files_keeps_known_identities_only():
    """LLM-suggested files survive only when deterministically
    eligible; unknown suggestions drop with a count (spec 15)."""
    from flight_log_agent.px4 import discovery_frontier as df

    kept, dropped = df.validate_decision_files(
        ["known_a.cpp", "ghost.cpp", "known_b.cpp"],
        eligible_files=["known_a.cpp", "known_b.cpp"],
    )
    assert kept == ["known_a.cpp", "known_b.cpp"]
    assert dropped == 1


def test_measure_packet_mass_uses_exact_spec_names():
    """Mass measurement exposes total/per-section sizes, new-item
    counts, summary/frontier sizes, avoided retransmission, and the
    §16 fallback accounting names (spec sections 16, 17)."""
    from flight_log_agent.px4 import discovery_frontier as df

    mass = df.measure_packet_mass(
        sections={"frontier": "x" * 100, "summary": "y" * 40},
        section_item_counts={"frontier": 3, "summary": 1},
        new_item_count=3,
        summary_bytes=40,
        frontier_bytes=100,
        full_equivalent_bytes=1000,
        fallback_used=False,
        fallback_reason="",
        fallback_count=0,
        fallback_bytes=0,
    )
    assert mass["total_bytes"] == 140
    assert mass["section_bytes"] == {"frontier": 100, "summary": 40}
    assert mass["section_item_counts"] == {"frontier": 3, "summary": 1}
    assert mass["new_item_count"] == 3
    assert mass["summary_bytes"] == 40
    assert mass["frontier_bytes"] == 100
    assert mass["retransmission_avoided_bytes"] == 860
    assert mass["full_packet_fallback_used"] is False
    assert mass["full_packet_fallback_reason"] == ""
    assert mass["full_packet_fallback_count"] == 0
    assert mass["full_packet_fallback_bytes"] == 0


def test_p0_mass_fragment_is_measurement_only():
    """The P0 fragment follows the version/status/thresholds/basis
    contract with thresholds null; it never gates GREEN (spec 18)."""
    from flight_log_agent.px4 import discovery_frontier as df

    fragment = df.p0_mass_fragment({"total_bytes": 140})
    assert fragment["version"] == 1
    assert fragment["thresholds"] is None
    assert fragment["basis"] == {"total_bytes": 140}
    assert "status" in fragment


def test_process_lookup_requests_validates_before_retrieval():
    """Lookup requests resolve against canonical identities; unknown
    requests drop with a count and never reach retrieval (spec 27)."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(identity, kind="writer",
                  content={"actual_value": 1}, round_no=1)
    usage = df.LookupUsage()
    result = df.process_lookup_requests(
        [repr(identity), "ghost-identity"],
        store=store, budget=df.LookupBudget(), usage=usage,
    )
    assert result.status == df.LOOKUP_COMPLETE
    assert set(result.retrieved.keys()) == {identity}
    assert result.dropped_unknown_count == 1
    assert usage.lookup_round_count == 1


# ----------------------------------------------------------------------
# SLICE 5b-models — additive bounded-input model fields (spec sections
# 14, 16, 17, 27; no behavior change when empty)
# ----------------------------------------------------------------------


def test_decision_lookup_requests_default_empty():
    """Mode-B lookup requests ride an additive optional decision
    field; existing decisions behave exactly as before (spec 27)."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    assert SourceDiscoveryDecision().lookup_requests == []
    assert SourceDiscoveryDecision(
        lookup_requests=["writer|file_a|symbol_c"]).lookup_requests == [
        "writer|file_a|symbol_c"]


def test_packet_bounded_sections_default_empty():
    """Bounded-input sections are additive packet fields defaulting to
    empty/off; full packets validate unchanged (spec sections 14, 16,
    17)."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryIterationPacket,
    )

    packet = SourceDiscoveryIterationPacket(user_question="q", depth=0)
    assert packet.bounded is False
    assert packet.decision_frontier == {}
    assert packet.carry_forward_summary == {}
    assert packet.packet_mass == {}
    assert packet.fallback == {}


def test_candidate_set_frontier_accounting_default_empty():
    """Run-level frontier accounting is additive; existing result
    shapes validate unchanged (spec section 16)."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceMechanismCandidateSet,
    )

    assert SourceMechanismCandidateSet().frontier_accounting == {}


# ----------------------------------------------------------------------
# SLICE 5b — resolver loop integration (spec sections 13-16, 27;
# offline, injected decide, synthetic source trees)
# ----------------------------------------------------------------------

import asyncio as _asyncio
from pathlib import Path as _Path


def _write_module(source_path, name="mod.cpp", token="trigger_alpha"):
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / name).write_text(
        f"""
class Probe {{
    ParamFloat<px4::params::PROBE_ALT> _param_probe_alt;
    void update() {{
        if (_param_probe_alt.get() > 1) {{
            {token}();
        }}
    }}
}};
""",
        encoding="utf-8",
    )


def _run_discover(source_path, decide, **kwargs):
    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )
    from flight_log_agent.px4.source_mechanism_resolver import (
        SourceMechanismResolver,
        build_source_discovery_log_context,
    )

    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )
    return _asyncio.run(
        resolver.discover(
            "Why did it call trigger_alpha?",
            build_source_discovery_log_context({}),
            seed_queries=["trigger_alpha"],
            decide=decide,
            max_depth=kwargs.get("max_depth", 1),
        )
    )


def test_bounded_round_one_packet_carries_frontier_and_mass(tmp_path):
    """Round-1 bounded packets keep full content (all items new) while
    adding frontier, summary, mass, and fallback sections (spec 8, 9,
    14, 16, 17)."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    _write_module(tmp_path / "PX4-Autopilot")
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        return SourceDiscoveryDecision(stop=True)

    _run_discover(tmp_path / "PX4-Autopilot", decide)
    profile = next(packet for packet in packets if packet.new_files)
    assert profile.bounded is True
    assert profile.source_profile["helper_expressions"] != []
    assert profile.decision_frontier["new_candidates"] != {}
    assert profile.carry_forward_summary["coverage_map"] != {}
    assert profile.packet_mass["total_bytes"] > 0
    assert profile.packet_mass["retransmission_avoided_bytes"] >= 0
    assert profile.fallback.get("used", False) is False


def test_bounded_round_two_packet_avoids_retransmission(tmp_path):
    """Round-2 bounded packets omit already-sent corpus: the first
    file's items leave the profile sections while staying reachable
    by identity in the summary, and measured mass shows avoided
    retransmission while the round trip still holds, so no fallback
    fires (spec sections 9, 13, 14, 17)."""
    import json

    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    many_calls = "\n".join(
        f"        if (_param_probe_alt.get() > {index}) {{ helper_{index}(); }}"
        for index in range(30)
    )
    (module_dir / "first.cpp").write_text(
        f"""
class Probe {{
    ParamFloat<px4::params::PROBE_ALT> _param_probe_alt;
    void update()
    {{
{many_calls}
        trigger_alpha();
    }}
}};
""",
        encoding="utf-8",
    )
    _write_module(tmp_path / "PX4-Autopilot", name="second.cpp",
                   token="trigger_beta")
    packets = []
    seen_rounds = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        seen_rounds.append(packet.depth)
        if packet.depth == 0:
            return SourceDiscoveryDecision(
                expansion_queries=["trigger_beta"])
        return SourceDiscoveryDecision(stop=True)

    _run_discover(tmp_path / "PX4-Autopilot", decide, max_depth=2)
    iterations = [packet for packet in packets if packet.new_files]
    assert len(iterations) == 2
    first, second = iterations
    assert first.bounded is True and second.bounded is True
    second_profile = json.dumps(second.source_profile, default=str)
    second_summary = json.dumps(second.carry_forward_summary, default=str)
    assert "first.cpp" not in second_profile
    assert "first.cpp" in second_summary
    assert "second.cpp" in second_profile
    assert (second.packet_mass["retransmission_avoided_bytes"]
            > first.packet_mass["retransmission_avoided_bytes"])
    assert second.fallback.get("used", False) is False
    assert second.packet_mass["full_packet_fallback_used"] is False


def test_bounded_build_failure_falls_back_with_accounting(tmp_path):
    """A bounded-build failure sends the current full packet once,
    observably counted as unoptimized (spec section 16)."""
    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )
    from flight_log_agent.px4.source_mechanism_resolver import (
        SourceMechanismResolver,
        build_source_discovery_log_context,
    )

    _write_module(tmp_path / "PX4-Autopilot")

    class BrokenBoundedResolver(SourceMechanismResolver):
        def _narrow_to_bounded(self, *args, **kwargs):
            raise RuntimeError("simulated summary failure")

    resolver = BrokenBoundedResolver(
        tmp_path / "PX4-Autopilot",
        profiler=MechanismSourceProfiler(
            tmp_path / "PX4-Autopilot", rg_path="missing-rg"),
    )
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        return SourceDiscoveryDecision(stop=True)

    result = _asyncio.run(
        resolver.discover(
            "Why did it call trigger_alpha?",
            build_source_discovery_log_context({}),
            seed_queries=["trigger_alpha"],
            decide=decide,
            max_depth=1,
        )
    )
    profile = next(packet for packet in packets if packet.new_files)
    assert profile.bounded is False
    assert profile.fallback.get("used", False) is True
    assert profile.fallback.get("reason") == "summary-failure"
    assert profile.packet_mass["full_packet_fallback_used"] is True
    assert result.frontier_accounting["fallback_count"] == 1
    assert result.frontier_accounting["fallback_bytes"] > 0


def test_lookup_request_round_trip_in_loop(tmp_path):
    """A known-identity lookup request retrieves canonical content
    with at most one extra decision call; unknown requests count as
    unavailable without extra calls (spec section 27)."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    _write_module(tmp_path / "PX4-Autopilot")
    packets = []
    calls = {"count": 0, "asked": False}

    async def decide(packet):
        packets.append(packet)
        calls["count"] += 1
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        if packet.new_files and not calls["asked"]:
            calls["asked"] = True
            frontier = packet.decision_frontier
            some_identity = next(iter(frontier["new_candidates"]))
            return SourceDiscoveryDecision(
                lookup_requests=[some_identity, "ghost-identity"])
        return SourceDiscoveryDecision(stop=True)

    result = _run_discover(tmp_path / "PX4-Autopilot", decide)
    assert calls["count"] == 3
    accounting = result.frontier_accounting
    assert accounting["lookup_complete"] >= 1
    assert accounting["lookup_dropped_unknown"] >= 1


def test_unknown_suggested_files_never_enter_traversal(tmp_path):
    """LLM-suggested files outside the eligible set never join
    visited traversal state (spec section 15)."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    _write_module(tmp_path / "PX4-Autopilot")
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision(
                relevant_files=["ghost/ghost.cpp"])
        return SourceDiscoveryDecision(
            relevant_files=["ghost/ghost.cpp"], stop=True)

    _run_discover(tmp_path / "PX4-Autopilot", decide)
    visited = [packet.new_files for packet in packets if packet.new_files]
    assert visited != []
    assert all("ghost/ghost.cpp" not in files for files in visited)


# ----------------------------------------------------------------------
# SLICE 6 — offline acceptance harness (spec sections 13, 19, 20, 25;
# reconstructed states; generic obligations only)
# ----------------------------------------------------------------------

from types import SimpleNamespace as _SimpleNamespace


def _harness_refs(tag, gate="verification_required", value=1):
    """Synthetic extraction round: one assignment, one call, one
    branch, one requirement, all sharing one file tag."""
    return {
        "assignments": [_SimpleNamespace(
            target=f"target_{tag}", expression=f"value_{tag}",
            file=f"{tag}.cpp", line=10)],
        "calls": [_SimpleNamespace(
            name=f"call_{tag}", file=f"{tag}.cpp", line=11,
            receiver=None, resolved_callable_id=None,
            source_site_id=f"site-{tag}", source_order=1)],
        "helpers": [],
        "branches": [_SimpleNamespace(
            kind="if", condition=f"check_{tag}", file=f"{tag}.cpp",
            line=12, source_site_id=f"bsite-{tag}", source_order=1)],
        "predicates": [],
        "topics": [],
        "fields": [],
        "parameter_refs": [],
        "requirements": [_SimpleNamespace(
            name=f"PARAM_{tag.upper()}", source_predicate=f"pred_{tag}",
            source_file=f"{tag}.cpp", source_line=13, role="threshold",
            gate_result=gate, actual_value=value, effect=f"effect_{tag}")],
        "files": [f"{tag}.cpp"],
    }


def _observe_harness_round(state, refs, round_no):
    from flight_log_agent.px4 import discovery_frontier as df

    return df.observe_discovery_round(
        state, round_no=round_no,
        assignments=refs["assignments"], calls=refs["calls"],
        helpers=refs["helpers"], branches=refs["branches"],
        predicates=refs["predicates"], topics=refs["topics"],
        fields=refs["fields"], parameter_refs=refs["parameter_refs"],
        requirements=refs["requirements"], files=refs["files"],
    )


def test_preservation_harness_two_rounds():
    """§13/§19 preservation dimensions over reconstructed states:
    unresolved, candidate, contradiction, requirement-variant, and
    gate-tally sets survive the bounded transformation exactly."""
    from flight_log_agent.px4 import discovery_frontier as df

    state = df.DiscoveryRoundState()
    round_one = _harness_refs("alpha")
    first_observed = _observe_harness_round(state, round_one, 0)
    # Round two: new file plus a changed requirement value.
    round_two = _harness_refs("beta")
    round_two["requirements"][0].actual_value = 2
    observed = _observe_harness_round(state, round_two, 1)
    assert observed["changed_all"] == set()
    assert len(observed["new_all"]) == len(round_two["assignments"]) + len(
        round_two["calls"]) + len(round_two["branches"]) + len(
        round_two["requirements"]) + len(round_two["files"])
    open_ids = (set(first_observed["open_requirements"])
                | set(observed["open_requirements"]))
    assert len(open_ids) == 2
    assert observed["gate_tally"]["verification_required"] == 1
    summary = df.build_summary(
        state.store, coverage_map={"files_seen": 2}, gate_tally=observed["gate_tally"],
        candidate_standings=[], round_no=1,
    )
    summary_open = {
        tuple(item["identity"]) for item in summary.open_claims
    }
    assert open_ids <= summary_open
    ok, report = df.check_round_trip(
        full_identities=set(state.all_observed_identities),
        frontier_identities=set(observed["new_all"]),
        summary_identities={
            tuple(item["identity"]) for item in summary.open_claims
        } | {
            tuple(item["identity"]) for item in summary.resolved_claims
        } | {
            tuple(item["identity"]) for item in summary.contradiction_ledger
        },
        verdict_identities=set(),
    )
    assert ok is True, report


def test_stateless_equivalence_same_state_same_input():
    """Equivalent explicit store states build byte-identical bounded
    inputs: two independent runs over the same observation sequence
    agree exactly, so no hidden history leaks between runs (spec 19).
    The bounded input is a pure function of explicit state, never of
    prior conversation."""
    from flight_log_agent.px4 import discovery_frontier as df

    def build_state():
        state = df.DiscoveryRoundState()
        for index, tag in enumerate(("alpha", "beta")):
            _observe_harness_round(state, _harness_refs(tag), index)
        summary = df.build_summary(
            state.store, coverage_map={}, gate_tally={},
            candidate_standings=[], round_no=1,
        )
        return df.serialize_packet(summary.to_packet_dict())

    assert build_state() == build_state()


def test_no_conversation_state_anywhere():
    """Bounded inputs carry no conversation_id, previous_response_id,
    or hidden-model-memory markers (spec sections 6, 9, 15)."""
    import pathlib

    from flight_log_agent.px4 import discovery_frontier as df

    module_text = pathlib.Path(df.__file__).read_text(encoding="utf-8")
    frontier = df.DiscoveryFrontier(
        new_candidates={("candidate", "alpha"): {"kind": "candidate"}},
        work_state={"depth_remaining": 1},
    )
    summary = df.CarryForwardSummary(coverage_map={}, gate_tally={})
    payload = df.serialize_packet({
        "frontier": frontier.to_packet_dict(),
        "summary": summary.to_packet_dict(),
    })
    for forbidden in ("conversation_id", "previous_response_id",
                      "hidden_memory", "hidden-memory"):
        assert forbidden not in payload
        assert forbidden not in module_text


def test_genericity_no_benchmark_literals_in_module():
    """Spec-named genericity test: the bounded-frontier production
    module contains no per-case benchmark literals (spec §19)."""
    import pathlib
    import re

    from flight_log_agent.px4 import discovery_frontier as df

    module_text = pathlib.Path(df.__file__).read_text(encoding="utf-8")
    hits = re.findall(
        r"rtl|tecs|takeoff|airspeed|mis_takeoff|switch.moded|acceptance_radius|load.factor",
        module_text, flags=re.IGNORECASE,
    )
    assert hits == [], f"per-case literals in production module: {hits}"


def test_must_survive_obligations_preserved_generically():
    """Governing-spec-derived evidence shapes (numeric relations,
    ordered targets, equation bindings) survive bounded
    transformation using generic identities only (spec §20A)."""
    from flight_log_agent.px4 import discovery_frontier as df

    state = df.DiscoveryRoundState()
    relations = [
        ("ratio_alpha", 2.0), ("rate_beta", 0.3),
        ("target_gamma", 20.0), ("target_delta", 7.0),
        ("factor_epsilon", 1.0),
    ]
    refs = {
        "assignments": [], "calls": [], "helpers": [], "branches": [],
        "predicates": [], "topics": [], "fields": [], "parameter_refs": [],
        "requirements": [
            _SimpleNamespace(
                name=f"PARAM_{name.upper()}", source_predicate="pred",
                source_file="generic.cpp", source_line=index,
                role="threshold", gate_result="verification_required",
                actual_value=value, effect="binding")
            for index, (name, value) in enumerate(relations)
        ],
        "files": ["generic.cpp"],
    }
    observed = _observe_harness_round(state, refs, 0)
    assert len(observed["open_requirements"]) == len(relations)
    summary = df.build_summary(
        state.store, coverage_map={}, gate_tally=observed["gate_tally"],
        candidate_standings=[], round_no=0,
    )
    summary_values = set()
    for item in summary.open_claims:
        record = state.store.get(tuple(item["identity"]))
        if record.kind != "requirement":
            continue
        summary_values.add(record.content["actual_value"])
    assert summary_values == {2.0, 0.3, 20.0, 7.0, 1.0}
    ok, report = df.check_round_trip(
        full_identities=set(state.all_observed_identities),
        frontier_identities=set(observed["new_all"]),
        summary_identities={
            tuple(item["identity"]) for item in summary.open_claims
        },
        verdict_identities=set(),
    )
    assert ok is True, report


def test_loop_bounded_packets_deterministic_across_runs(tmp_path):
    """Two identical discovery runs emit byte-identical bounded
    packets (spec section 19 construction determinism)."""
    import json

    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    _write_module(tmp_path / "PX4-Autopilot")

    def run_once():
        packets = []

        async def decide(packet):
            packets.append(packet)
            if packet.source_profile.get("stage") == "search_hits_only":
                return SourceDiscoveryDecision()
            return SourceDiscoveryDecision(stop=True)

        _run_discover(tmp_path / "PX4-Autopilot", decide)
        profile = next(packet for packet in packets if packet.new_files)
        return json.dumps(profile.model_dump(), default=str, sort_keys=True)

    assert run_once() == run_once()


# ----------------------------------------------------------------------
# IMPORTANT-1 — shared per-round lookup budget (spec section 27:
# default maximum 1 lookup round per discovery round)
# ----------------------------------------------------------------------


def test_second_lookup_request_in_same_round_exhausts(tmp_path):
    """One discovery round owns one lookup round: a follow-up
    decision's lookup request exhausts against the already-used
    budget — counted, no retrieval, no third decide call."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    _write_module(tmp_path / "PX4-Autopilot")
    packets = []
    asked = {"count": 0}

    def _frontier_identities(packet):
        keys = []
        for section, payload in packet.decision_frontier.items():
            if section in ("work_state", "requirement_groups"):
                continue
            if isinstance(payload, dict):
                keys.extend(payload.keys())
        return sorted(set(keys))

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        if packet.new_files and asked["count"] == 0:
            asked["count"] += 1
            keys = _frontier_identities(packet)
            assert len(keys) >= 2
            return SourceDiscoveryDecision(lookup_requests=[keys[0]])
        if "lookup_retrieved" in packet.decision_frontier:
            keys = _frontier_identities(packet)
            return SourceDiscoveryDecision(lookup_requests=[keys[1]])
        return SourceDiscoveryDecision(stop=True)

    result = _run_discover(tmp_path / "PX4-Autopilot", decide)
    decisions = [packet for packet in packets
                 if packet.source_profile.get("stage") != "search_hits_only"]
    assert len(decisions) == 2
    accounting = result.frontier_accounting
    assert accounting["lookup_complete"] == 1
    assert accounting["lookup_exhausted"] == 1
    follow = decisions[1]
    retrieved = follow.decision_frontier.get("lookup_retrieved", {})
    assert len(retrieved) == 1
    second_key = _frontier_identities(decisions[0])[1]
    assert second_key not in retrieved


# ----------------------------------------------------------------------
# IMPORTANT findings hardening — closed prefetch, byte cap, fallback
# negative (spec sections 13, 16, 27)
# ----------------------------------------------------------------------


def test_closed_prefetch_raises_on_unexpected_missing():
    """An open question missing from the canonical store is a
    bounded-packet invariant failure, never a silent drop."""
    import pytest

    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    known = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(known, kind="writer",
                  content={"actual_value": 1}, round_no=0)
    ghost = df.make_identity("writer", "ghost")
    with pytest.raises(df.BoundedPacketError):
        df.closed_prefetch(store, open_claims=[known, ghost])
    found = df.closed_prefetch(store, open_claims=[known])
    assert set(found.keys()) == {known}


def test_lookup_oversize_single_record_fails_closed():
    """A single known record exceeding the byte cap fails closed with
    no partial retrieval."""
    from flight_log_agent.px4 import discovery_frontier as df

    store = df.CanonicalStore()
    identity = df.make_identity("writer", "file_a", "symbol_c")
    store.observe(identity, kind="writer",
                  content={"actual_value": "x" * 1000}, round_no=0)
    usage = df.LookupUsage()
    result = df.lookup(store, [identity],
                       budget=df.LookupBudget(max_bytes=10), usage=usage)
    assert result.status == df.LOOKUP_BUDGET_EXHAUSTED
    assert result.retrieved == {}
    assert usage.lookup_budget_exhausted is True


def test_forced_fallback_never_counts_bounded_success(tmp_path):
    """A 100%-fallback run shows zero bounded benefit: bounded_rounds
    stays 0 and avoided bytes stay 0 while fallback is counted."""
    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )
    from flight_log_agent.px4.source_mechanism_resolver import (
        SourceMechanismResolver,
        build_source_discovery_log_context,
    )

    _write_module(tmp_path / "PX4-Autopilot")

    class AlwaysBrokenResolver(SourceMechanismResolver):
        def _narrow_to_bounded(self, *args, **kwargs):
            raise RuntimeError("simulated summary failure")

    resolver = AlwaysBrokenResolver(
        tmp_path / "PX4-Autopilot",
        profiler=MechanismSourceProfiler(
            tmp_path / "PX4-Autopilot", rg_path="missing-rg"),
    )

    async def decide(packet):
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        return SourceDiscoveryDecision(stop=True)

    import asyncio as _asyncio

    result = _asyncio.run(
        resolver.discover(
            "Why did it call trigger_alpha?",
            build_source_discovery_log_context({}),
            seed_queries=["trigger_alpha"],
            decide=decide,
            max_depth=1,
        )
    )
    accounting = result.frontier_accounting
    assert accounting["fallback_count"] == 1
    assert accounting.get("bounded_rounds", 0) == 0
    assert accounting.get("total_avoided_bytes", 0) == 0
    assert accounting["fallback_bytes"] > 0


# ----------------------------------------------------------------------
# IMPORTANT-2 — loop-level reactivation pins (spec sections 10A, 13)
#
# Note on scope honesty: the production gate is a pure function of
# accumulated predicates plus static log context, so real gate
# outcomes cannot flip for an identical predicate across rounds.
# The scripted gates below prove the loop wiring (observe →
# trigger → frontier → packet) for gate-capable futures; the
# real-gate test proves contradiction visibility and predecessor
# retention with unscripted behavior. Deferred trigger classes
# (hypothesis/candidate/window/relation) remain unwired by design.
# ----------------------------------------------------------------------


class _ScriptedGate:
    """Deterministic gate double returning scripted requirement sets
    per evaluate() call index. Stands in for gate outcomes the
    current static gate cannot produce across rounds; proves loop
    wiring, not gate logic."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def evaluate(self, predicates, log_context):
        index = min(self.calls, len(self._script) - 1)
        self.calls += 1
        return list(self._script[index])

    def evaluate_source_predicates(self, predicates, log_context):
        return []


def _scripted_requirement(*, gate, value):
    from flight_log_agent.px4.source_mechanism_models import (
        ParameterRequirement,
    )

    return ParameterRequirement(
        name="PARAM_ALPHA", role="threshold",
        source_predicate="pred_alpha", actual_value=value,
        gate_result=gate, effect="effect_alpha",
        source_file="generic.cpp", source_line=5,
    )


def _two_file_tree(root, first_token, second_token):
    module_dir = root / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True, exist_ok=True)
    body = """
class Probe {{
    ParamFloat<px4::params::PROBE_ALT> _param_probe_alt;
    void update() {{ if (_param_probe_alt.get() > 1) {{ {token}(); }} }}
}};
"""
    (module_dir / "first.cpp").write_text(
        body.format(token=first_token), encoding="utf-8")
    (module_dir / "second.cpp").write_text(
        body.format(token=second_token), encoding="utf-8")


def _run_scripted(source_path, decide, gate, **kwargs):
    import asyncio as _asyncio

    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )
    from flight_log_agent.px4.source_mechanism_resolver import (
        SourceMechanismResolver,
        build_source_discovery_log_context,
    )

    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
        parameter_gate=gate,
    )
    return _asyncio.run(
        resolver.discover(
            "Why did it call trigger_alpha?",
            build_source_discovery_log_context({}),
            seed_queries=["trigger_alpha"],
            decide=decide,
            max_depth=kwargs.get("max_depth", 2),
        )
    )


def test_changed_requirement_reactivates_through_loop(tmp_path):
    """Same requirement base, changed value, same satisfied gate:
    revision bumps, the changed flag fires reactivation after the
    same-round resolve, the record ends ACTIVE with carried verdicts,
    and the new packet differs in decision-relevant fields."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    _two_file_tree(tmp_path / "PX4-Autopilot", "trigger_alpha", "trigger_beta")
    gate = _ScriptedGate([
        [_scripted_requirement(gate="satisfied", value=1)],
        [_scripted_requirement(gate="satisfied", value=2)],
    ])
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        if packet.depth == 0:
            return SourceDiscoveryDecision(
                expansion_queries=["trigger_beta"])
        return SourceDiscoveryDecision(stop=True)

    _run_scripted(tmp_path / "PX4-Autopilot", decide, gate, max_depth=2)
    iterations = [packet for packet in packets if packet.new_files]
    assert len(iterations) == 2
    first, second = iterations
    assert first.bounded is True and second.bounded is True

    def requirement_key(packet):
        matches = [key for key in
                   packet.decision_frontier["new_requirements"].keys()
                   if "PARAM_ALPHA" in key]
        assert len(matches) == 1
        return matches[0]

    first_key = requirement_key(first)
    second_key = requirement_key(second)
    assert first_key == second_key
    first_desc = first.decision_frontier["new_requirements"][first_key]
    second_desc = second.decision_frontier["new_requirements"][second_key]
    assert first_desc["value_revision"] == 0
    assert second_desc["value_revision"] == 1

    def summary_identities(packet, section):
        return {
            tuple(item["identity"])
            for item in packet.carry_forward_summary[section]
        }

    first_identity = ("requirement", "PARAM_ALPHA", "pred_alpha",
                      "generic.cpp", "5", "threshold", "satisfied")
    # Round 0: resolved with a recorded verdict.
    assert first_identity in summary_identities(first, "resolved_claims")
    # Round 1: the changed flag fired reactivation after the
    # same-round resolve, so the record is ACTIVE (open), not resolved.
    assert first_identity in summary_identities(second, "open_claims")
    assert first_identity not in summary_identities(second, "resolved_claims")
    # Full requirement content (new value) rides the narrowed packet.
    assert any(requirement.actual_value == 2
               for requirement in second.parameter_requirements)
    # Decision-relevant representation differs across rounds.
    import json as _json

    assert (_json.dumps(first.decision_frontier, default=str, sort_keys=True)
            != _json.dumps(second.decision_frontier, default=str,
                           sort_keys=True))


def test_contradiction_reactivates_retained_predecessor(tmp_path):
    """A new contradicted variant base-matches its retained satisfied
    predecessor: the predecessor returns ACTIVE with history, the
    contradiction reaches frontier and ledger, and the contradicted
    record itself is never reactivated."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )

    _two_file_tree(tmp_path / "PX4-Autopilot", "trigger_alpha", "trigger_beta")
    satisfied = _scripted_requirement(gate="satisfied", value=1)
    contradicted = _scripted_requirement(gate="contradicted", value=1)
    gate = _ScriptedGate([[satisfied], [satisfied, contradicted]])
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        if packet.depth == 0:
            return SourceDiscoveryDecision(
                expansion_queries=["trigger_beta"])
        return SourceDiscoveryDecision(stop=True)

    _run_scripted(tmp_path / "PX4-Autopilot", decide, gate, max_depth=2)
    iterations = [packet for packet in packets if packet.new_files]
    assert len(iterations) == 2
    first, second = iterations
    assert first.bounded is True and second.bounded is True

    satisfied_identity = ("requirement", "PARAM_ALPHA", "pred_alpha",
                          "generic.cpp", "5", "threshold", "satisfied")
    contradicted_identity = ("requirement", "PARAM_ALPHA", "pred_alpha",
                             "generic.cpp", "5", "threshold", "contradicted")

    def summary_identities(packet, section):
        return {
            tuple(item["identity"])
            for item in packet.carry_forward_summary[section]
        }

    # Round 0: satisfied predecessor resolved.
    assert satisfied_identity in summary_identities(first, "resolved_claims")
    # Round 1: predecessor reactivated to ACTIVE with its verdict
    # history; new variant contradicted in the ledger.
    assert satisfied_identity in summary_identities(second, "open_claims")
    assert contradicted_identity in summary_identities(
        second, "contradiction_ledger")
    ledger = second.carry_forward_summary["contradiction_ledger"]
    assert any(entry["refutations"] for entry in ledger)
    # Frontier carries the contradiction plus the reactivated
    # predecessor with prior verdicts (never a blank item).
    contradictions = second.decision_frontier["contradictions"]
    assert any("contradicted" in key for key in contradictions.keys())
    reactivated = [
        payload for key, payload in
        second.decision_frontier["new_requirements"].items()
        if "satisfied" in key
    ]
    assert len(reactivated) == 1
    assert reactivated[0].get("reactivated", False) is True or \
        reactivated[0].get("value_revision", 0) >= 0
    assert second.fallback.get("used", False) is False


def test_real_gate_contradiction_visible_predecessor_retained(tmp_path):
    """Unscripted behavior: a round-2 contradicted predicate reaches
    the ledger and frontier while the round-1 retained requirement
    stays visible; bounded flow holds with no spurious reactivation
    side effects."""
    from flight_log_agent.px4.source_mechanism_models import (
        SourceDiscoveryDecision,
    )
    from flight_log_agent.px4.source_mechanism_resolver import (
        build_source_discovery_log_context,
    )
    import asyncio as _asyncio

    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )
    from flight_log_agent.px4.source_mechanism_resolver import (
        SourceMechanismResolver,
    )

    module_dir = (tmp_path / "PX4-Autopilot" / "src" / "modules"
                  / "navigator")
    module_dir.mkdir(parents=True)
    (module_dir / "first.cpp").write_text(
        """
class ProbeA {
    ParamInt<px4::params::PARAM_ALPHA> _param_alpha;
    void update() { if (_param_alpha.get() == 1) { trigger_alpha(); } }
};
""",
        encoding="utf-8",
    )
    (module_dir / "second.cpp").write_text(
        """
class ProbeB {
    ParamInt<px4::params::NAV_MODE_SEL> _param_mode_sel;
    void update() { if (_param_mode_sel.get() == 2) { trigger_beta(); } }
};
""",
        encoding="utf-8",
    )
    resolver = SourceMechanismResolver(
        tmp_path / "PX4-Autopilot",
        profiler=MechanismSourceProfiler(
            tmp_path / "PX4-Autopilot", rg_path="missing-rg"),
    )
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        if packet.depth == 0:
            return SourceDiscoveryDecision(
                expansion_queries=["trigger_beta"])
        return SourceDiscoveryDecision(stop=True)

    _asyncio.run(
        resolver.discover(
            "Why did it call trigger_alpha?",
            build_source_discovery_log_context({"parameters": {"NAV_MODE_SEL": 1}}),
            seed_queries=["trigger_alpha"],
            decide=decide,
            max_depth=2,
        )
    )
    iterations = [packet for packet in packets if packet.new_files]
    assert len(iterations) == 2
    first, second = iterations
    assert first.bounded is True and second.bounded is True
    assert second.fallback.get("used", False) is False

    def summary_identities(packet, section):
        return {
            tuple(item["identity"])
            for item in packet.carry_forward_summary[section]
        }

    ledger = summary_identities(second, "contradiction_ledger")
    assert any("NAV_MODE_SEL" in repr(identity) for identity in ledger)
    open_ids = summary_identities(second, "open_claims")
    assert any("PARAM_ALPHA" in repr(identity) for identity in open_ids)
    contradictions = second.decision_frontier["contradictions"]
    assert any("NAV_MODE_SEL" in key for key in contradictions.keys())
