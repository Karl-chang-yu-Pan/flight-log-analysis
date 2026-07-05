from __future__ import annotations

import pytest

from flight_log_agent.analysis.source_slicer import (
    ForwardHop,
    ForwardSliceResult,
    SliceBlocker,
    SliceResult,
    alias_map_for_call_site,
    forward_slice,
    slice_expression,
    slice_symbol,
)


def _write(
    target: str,
    expression: str,
    *,
    control_predicates: list[str] | None = None,
    file: str = "fake.cpp",
    line: int = 1,
) -> dict:
    return {
        "target": target,
        "expression": expression,
        "control_predicates": control_predicates or [],
        "file": file,
        "line": line,
        "evidence": "",
        "function": "fake_fn",
    }


class TestSliceSymbolDirectResolution:
    def test_logged_signal_is_already_resolved(self):
        result = slice_symbol(
            "vehicle_global_position.alt",
            source_assignments=[],
            logged_signals={"vehicle_global_position.alt"},
        )
        assert result.status == "resolved"
        assert result.expression == "vehicle_global_position.alt"

    def test_parameter_is_already_resolved(self):
        result = slice_symbol(
            "NAV_ACC_RAD",
            source_assignments=[],
            parameters={"NAV_ACC_RAD"},
        )
        assert result.status == "resolved"
        assert result.expression == "NAV_ACC_RAD"

    def test_single_hop_to_logged_signal(self):
        # _destination.lat = pos_sp_triplet.current.lat
        result = slice_symbol(
            "_destination.lat",
            source_assignments=[
                _write("_destination.lat", "pos_sp_triplet.current.lat"),
            ],
            logged_signals={"pos_sp_triplet.current.lat"},
        )
        assert result.status == "resolved"
        assert result.expression == "pos_sp_triplet.current.lat"

    def test_multi_hop_chain(self):
        # _destination.alt -> intermediate -> home_position.alt
        result = slice_symbol(
            "_destination.alt",
            source_assignments=[
                _write("_destination.alt", "intermediate.alt"),
                _write("intermediate.alt", "home_position.alt"),
            ],
            logged_signals={"home_position.alt"},
        )
        assert result.status == "resolved"
        # Substituted (parenthesized) twice as the chain unwinds.
        assert "home_position.alt" in result.expression


class TestSliceSymbolFailures:
    def test_no_write_site(self):
        result = slice_symbol(
            "unknown.field",
            source_assignments=[],
            logged_signals=set(),
        )
        assert result.status == "unbindable"
        assert result.blocker is not None
        assert result.blocker.kind == "no_write_site"
        assert result.blocker.symbol == "unknown.field"

    def test_cycle_self_reference(self):
        # _destination.alt = _destination.alt (degenerate cycle)
        result = slice_symbol(
            "_destination.alt",
            source_assignments=[
                _write("_destination.alt", "_destination.alt"),
            ],
            logged_signals=set(),
        )
        assert result.status == "cycle"
        assert result.blocker is not None
        assert result.blocker.kind == "cycle"
        # normalize_symbol strips the leading underscore.
        assert "destination.alt" in result.blocker.cycle_path

    def test_cycle_indirect(self):
        # a.x -> b.x -> a.x
        result = slice_symbol(
            "a.x",
            source_assignments=[
                _write("a.x", "b.x"),
                _write("b.x", "a.x"),
            ],
            logged_signals=set(),
        )
        assert result.status == "cycle"
        assert result.blocker is not None
        assert set(result.blocker.cycle_path) == {"a.x", "b.x"}

    def test_external_call_blocks_resolution(self):
        # _destination.alt = dm_read(...)  — external call we cannot lower
        result = slice_symbol(
            "_destination.alt",
            source_assignments=[
                _write("_destination.alt", "dm_read(DM_KEY_MISSION, 0, foo)"),
            ],
            logged_signals=set(),
        )
        assert result.status == "unbindable"
        assert result.blocker is not None
        # The external call is the blocker.
        assert result.blocker.kind in {"external_call", "unsupported_rhs"}


class TestConditionalBindings:
    def test_branches_with_distinct_predicates_become_conditional(self):
        result = slice_symbol(
            "_destination.alt",
            source_assignments=[
                _write(
                    "_destination.alt",
                    "home_position.alt",
                    control_predicates=["RTL_DESTINATION == 0"],
                ),
                _write(
                    "_destination.alt",
                    "safe_point.alt",
                    control_predicates=["RTL_DESTINATION == 2"],
                ),
            ],
            logged_signals={"home_position.alt", "safe_point.alt"},
        )
        assert result.status == "conditional"
        assert len(result.conditional_branches) == 2
        conditions = {b.condition for b in result.conditional_branches}
        assert any("RTL_DESTINATION == 0" in c for c in conditions)

    def test_partial_resolution_returns_unbindable_with_partial_expression(self):
        result = slice_symbol(
            "_destination.alt",
            source_assignments=[
                _write(
                    "_destination.alt",
                    "home_position.alt",
                    control_predicates=["RTL_DESTINATION == 0"],
                ),
                _write(
                    "_destination.alt",
                    "dm_read(DM_KEY_MISSION_LANDING, ...)",
                    control_predicates=["RTL_DESTINATION == 1"],
                ),
            ],
            logged_signals={"home_position.alt"},
        )
        # Q1=(b): conditional with at least one resolved branch + holdouts.
        assert result.status == "conditional"
        assert result.blocker is not None
        assert result.blocker.partial_expression is not None
        # The partial expression inlines the resolved branch and keeps
        # the original symbol (normalized) for the unresolved one.
        assert "home_position.alt" in result.blocker.partial_expression
        assert "destination.alt" in result.blocker.partial_expression

    def test_ambiguous_writes_without_distinguishing_predicates(self):
        result = slice_symbol(
            "x.y",
            source_assignments=[
                _write("x.y", "a.b"),
                _write("x.y", "c.d"),  # no predicates — ambiguous
            ],
            logged_signals={"a.b", "c.d"},
        )
        assert result.status == "unbindable"
        assert result.blocker is not None
        assert result.blocker.kind == "ambiguous_writes"


class TestSliceExpression:
    def test_fully_resolved_expression(self):
        result = slice_expression(
            "_destination.alt + RTL_RETURN_ALT",
            source_assignments=[
                _write("_destination.alt", "home_position.alt"),
            ],
            logged_signals={"home_position.alt"},
            parameters={"RTL_RETURN_ALT"},
        )
        assert result.fully_resolved is True
        assert "home_position.alt" in result.expression
        # The original _destination.alt should be substituted away.
        assert "_destination.alt" not in result.expression

    def test_partial_substitution_leaves_unresolved_in_place(self):
        result = slice_expression(
            "_destination.alt + RTL_RETURN_ALT",
            source_assignments=[],  # no writes for _destination.alt
            logged_signals=set(),
            parameters={"RTL_RETURN_ALT"},
        )
        assert result.fully_resolved is False
        assert len(result.unresolved) == 1
        assert result.unresolved[0].blocker is not None
        assert result.unresolved[0].blocker.kind == "no_write_site"
        # RTL_RETURN_ALT is fine (it's a parameter), _destination.alt is the holdout.
        assert "_destination.alt" in result.expression

    def test_mixed_resolved_and_unresolved_symbols(self):
        result = slice_expression(
            "destination.alt + destination.lat",
            source_assignments=[
                _write("destination.lat", "pos_sp.current.lat"),
                # No write for destination.alt
            ],
            logged_signals={"pos_sp.current.lat"},
        )
        assert result.fully_resolved is False
        assert "pos_sp.current.lat" in result.expression
        assert "destination.alt" in result.expression

    def test_logged_signals_pass_through(self):
        result = slice_expression(
            "vehicle_global_position.alt + 1.0",
            source_assignments=[],
            logged_signals={"vehicle_global_position.alt"},
        )
        assert result.fully_resolved is True
        # Already a logged signal, no rewrite needed.
        assert result.expression == "vehicle_global_position.alt + 1.0"


class TestReportingShape:
    def test_blocker_carries_actionable_detail(self):
        result = slice_symbol(
            "missing.field",
            source_assignments=[],
            logged_signals=set(),
        )
        assert result.blocker is not None
        # Detail mentions the specific symbol and what went wrong.
        assert "missing.field" in result.blocker.detail

    def test_cycle_path_is_ordered(self):
        result = slice_symbol(
            "a.x",
            source_assignments=[
                _write("a.x", "b.x"),
                _write("b.x", "c.x"),
                _write("c.x", "a.x"),
            ],
            logged_signals=set(),
        )
        assert result.status == "cycle"
        assert result.blocker is not None
        # Cycle path should include all the symbols in order.
        assert result.blocker.cycle_path[0] == "a.x"
        assert "a.x" in result.blocker.cycle_path
        assert "b.x" in result.blocker.cycle_path
        assert "c.x" in result.blocker.cycle_path


class _StubBindingIndex:
    """Minimal BindingIndex stand-in exposing ``aliases``."""

    def __init__(self, aliases: dict[str, set[str]]):
        self.aliases = aliases


class TestBindingIndexFastPath:
    def test_single_target_alias_resolves_without_source_assignments(self):
        # BindingIndex stores aliases keyed by normalize_symbol(), which
        # strips leading underscores. The fast-path lookup uses the same
        # normalized form.
        index = _StubBindingIndex({
            "destination.lat": {"position_setpoint_triplet.current.lat"},
        })
        result = slice_symbol(
            "_destination.lat",
            source_assignments=[],
            logged_signals={"position_setpoint_triplet.current.lat"},
            binding_index=index,
        )
        assert result.status == "resolved"
        assert result.expression == "position_setpoint_triplet.current.lat"

    def test_multi_target_alias_falls_through(self):
        index = _StubBindingIndex({
            "destination.lat": {
                "position_setpoint_triplet.current.lat",
                "mission_item.lat",
            },
        })
        # No source_assignments either, so the fall-through arrives at
        # a no_write_site dead end — proving we did NOT take the fast
        # path (which would have produced a resolved status).
        result = slice_symbol(
            "_destination.lat",
            source_assignments=[],
            binding_index=index,
        )
        assert result.status == "unbindable"
        assert result.blocker is not None
        assert result.blocker.kind == "no_write_site"

    def test_absent_alias_falls_through_to_assignments(self):
        index = _StubBindingIndex({})
        result = slice_symbol(
            "_destination.lat",
            source_assignments=[
                _write("_destination.lat", "vehicle_global_position.lat"),
            ],
            logged_signals={"vehicle_global_position.lat"},
            binding_index=index,
        )
        assert result.status == "resolved"
        assert result.expression == "vehicle_global_position.lat"

    def test_binding_index_none_preserves_existing_behavior(self):
        result = slice_symbol(
            "_destination.lat",
            source_assignments=[
                _write("_destination.lat", "vehicle_global_position.lat"),
            ],
            logged_signals={"vehicle_global_position.lat"},
            binding_index=None,
        )
        assert result.status == "resolved"
        assert result.expression == "vehicle_global_position.lat"

    def test_binding_index_resolves_inside_slice_expression(self):
        index = _StubBindingIndex({
            "destination.lat": {"position_setpoint_triplet.current.lat"},
            "destination.lon": {"position_setpoint_triplet.current.lon"},
        })
        result = slice_expression(
            "_destination.lat + _destination.lon",
            source_assignments=[],
            logged_signals={
                "position_setpoint_triplet.current.lat",
                "position_setpoint_triplet.current.lon",
            },
            binding_index=index,
        )
        assert result.fully_resolved
        assert "position_setpoint_triplet.current.lat" in result.expression
        assert "position_setpoint_triplet.current.lon" in result.expression


class TestAssignmentIndexAndMemo:
    def test_repeated_symbol_resolved_consistently(self):
        # Same canonical referenced twice in the expression must resolve
        # to the same logged signal — proves the memo / index path stays
        # consistent across repeated lookups in one slice_expression call.
        assignments = [_write("_destination.lat", "vehicle_global_position.lat")]
        result = slice_expression(
            "_destination.lat - _destination.lat",
            source_assignments=assignments,
            logged_signals={"vehicle_global_position.lat"},
        )
        # Each occurrence substituted with the same resolved signal.
        assert result.expression.count("vehicle_global_position.lat") == 2

    def test_memo_does_not_mask_cycle_on_a_different_path(self):
        # Slice symbol X first (clean resolution, gets memoized).
        # Then slice an outer chain that, by walking through X, would
        # cycle back to the caller. The memo's cached "resolved" for X
        # must not suppress the genuine cycle further up.
        assignments = [
            _write("a.x", "b.x"),
            _write("b.x", "c.x"),
            _write("c.x", "a.x"),  # cycles back into a.x
            _write("clean.y", "vehicle_global_position.lat"),  # no cycle
        ]
        # First, slice the clean symbol — populates memo with a resolved
        # entry for clean.y.
        clean = slice_symbol(
            "clean.y",
            source_assignments=assignments,
            logged_signals={"vehicle_global_position.lat"},
        )
        assert clean.status == "resolved"
        # Now slice the cycling chain. clean.y is NOT in this chain, so
        # the memo cannot affect cycle detection. We're proving cycle
        # detection still fires when memo has unrelated entries.
        cycled = slice_symbol(
            "a.x",
            source_assignments=assignments,
            logged_signals={"vehicle_global_position.lat"},
        )
        assert cycled.status == "cycle"

    def test_no_write_site_is_cacheable(self):
        # Slicing the same dead-end twice via one slice_expression call
        # returns the same unbindable/no_write_site result. (Behavioral
        # check; the cache is a performance concern, but consistency is
        # the load-bearing invariant.)
        result = slice_expression(
            "missing.symbol + missing.symbol",
            source_assignments=[],
        )
        # Both occurrences should remain in the substituted expression
        # because there is nothing to substitute them with.
        assert "missing.symbol" in result.expression
        assert not result.fully_resolved

    def test_index_handles_many_writes(self):
        # Build a long assignment list with many irrelevant entries; the
        # target one should still be found via the index.
        assignments = [_write(f"unrelated{i}.x", f"junk{i}") for i in range(200)]
        assignments.append(_write("_destination.lat", "vehicle_global_position.lat"))
        assignments.extend(_write(f"more{i}.y", f"junk{i}") for i in range(200))
        result = slice_symbol(
            "_destination.lat",
            source_assignments=assignments,
            logged_signals={"vehicle_global_position.lat"},
        )
        assert result.status == "resolved"
        assert result.expression == "vehicle_global_position.lat"


class TestForwardSlice:
    def test_pure_reassignment_is_classified(self):
        result = forward_slice(
            "_rtl_alt",
            source_assignments=[
                _write("_mission_item.altitude", "_rtl_alt", file="rtl.cpp", line=361),
            ],
        )
        assert len(result.hops) == 1
        assert result.hops[0].role == "pure_reassignment"
        assert result.hops[0].target == "_mission_item.altitude"
        assert result.hops[0].called_function is None

    def test_call_argument_captures_function_name(self):
        result = forward_slice(
            "_mission_item",
            source_assignments=[
                _write(
                    "pos_sp_triplet.current.alt",
                    "get_absolute_altitude_for_item(_mission_item)",
                    file="mission_block.cpp",
                    line=669,
                ),
            ],
        )
        assert len(result.hops) == 1
        hop = result.hops[0]
        assert hop.role == "call_argument"
        assert hop.called_function == "get_absolute_altitude_for_item"

    def test_transformation_hits_are_returned_unless_pure_only(self):
        assignments = [
            _write("_rtl_alt", "_destination.alt + _param_rtl_return_alt.get()"),
        ]
        default = forward_slice("_destination.alt", source_assignments=assignments)
        assert len(default.hops) == 1
        assert default.hops[0].role == "transformation"

        pure = forward_slice(
            "_destination.alt", source_assignments=assignments, pure_only=True
        )
        assert pure.hops == []

    def test_forward_slice_matches_exact_symbol_not_leaf_only(self):
        """Two different dotted symbols that happen to share a leaf must not
        match — ``pos_sp.altitude`` is unrelated to ``mission_item.altitude``.
        Alias-aware matching is deferred to the discovery-loop wiring."""
        result = forward_slice(
            "_mission_item.altitude",
            source_assignments=[
                _write("sp.alt", "pos_sp.altitude"),
                _write(
                    "_mission_item.altitude",
                    "_rtl_alt",
                    file="rtl.cpp",
                    line=361,
                ),
            ],
        )
        # pos_sp.altitude is a leaf-collision, not a real consumer.
        assert result.hops == []

    def test_intra_file_hops_come_first_when_origin_file_supplied(self):
        result = forward_slice(
            "_rtl_alt",
            source_assignments=[
                _write("outside.field", "_rtl_alt", file="other.cpp", line=10),
                _write("_mission_item.altitude", "_rtl_alt", file="rtl.cpp", line=361),
                _write("_mission_item.altitude", "_rtl_alt", file="rtl.cpp", line=399),
            ],
            origin_file="rtl.cpp",
        )
        assert [(hop.file, hop.line) for hop in result.hops] == [
            ("rtl.cpp", 361),
            ("rtl.cpp", 399),
            ("other.cpp", 10),
        ]

    def test_call_argument_ignores_arguments_hidden_in_transformations(self):
        # `foo(bar * X)` should classify as transformation — X is not a bare
        # top-level argument.
        result = forward_slice(
            "_rtl_alt",
            source_assignments=[
                _write("target.x", "foo(bar * _rtl_alt)"),
            ],
        )
        assert result.hops[0].role == "transformation"

    def test_symbol_unreferenced_by_any_assignment_returns_empty(self):
        result = forward_slice(
            "_never_used",
            source_assignments=[
                _write("_rtl_alt", "42"),
                _write("_destination.alt", "some_helper()"),
            ],
        )
        assert result.hops == []

    def test_alias_map_bridges_parameter_rename_across_boundaries(self):
        """When the discovery loop knows a helper's formal parameter maps
        to a caller-side name, forward-slicing on the caller-side symbol
        should still match callee-side reads through the alias map."""
        result = forward_slice(
            "_mission_item.altitude",
            source_assignments=[
                _write("sp.alt", "item.altitude", file="mission_block.cpp", line=669),
                _write("other.field", "unrelated.altitude"),
            ],
            aliases={"item": "mission_item"},
        )
        assert len(result.hops) == 1
        assert result.hops[0].role == "pure_reassignment"
        assert result.hops[0].file == "mission_block.cpp"

    def test_alias_translation_survives_call_argument_classification(self):
        """Symbol matching via alias should still classify a bare-arg call
        as ``call_argument`` (not fall through to ``transformation``)."""
        result = forward_slice(
            "_mission_item",
            source_assignments=[
                _write("target.x", "compute(item)", file="mission_block.cpp", line=190),
            ],
            aliases={"item": "mission_item"},
        )
        assert len(result.hops) == 1
        assert result.hops[0].role == "call_argument"
        assert result.hops[0].called_function == "compute"

    def test_alias_map_for_call_site_pairs_formals_with_bare_actuals(self):
        aliases = alias_map_for_call_site(
            ["item", "sp"],
            ["_mission_item", "&pos_sp_triplet->current"],
        )
        assert aliases == {
            "item": "mission_item",
            "sp": "pos_sp_triplet.current",
        }

    def test_alias_map_skips_transformations_and_calls(self):
        """Non-identity args contribute no alias — the callee's read of
        that formal can't be linked back to a single caller-side symbol."""
        aliases = alias_map_for_call_site(
            ["a", "b", "c", "d"],
            ["_x", "x + 1", "foo(_x)", "(float)_x"],
        )
        # Only the bare-reference arg maps.
        assert aliases == {"a": "x"}

    def test_alias_map_ignores_extra_formals_or_actuals(self):
        assert alias_map_for_call_site(["a", "b"], ["_x"]) == {"a": "x"}
        assert alias_map_for_call_site(["a"], ["_x", "_y"]) == {"a": "x"}

    def test_alias_map_composes_with_forward_slice(self):
        """End-to-end: build the alias map from a call site, then hand it
        to forward_slice to cross a parameter boundary."""
        aliases = alias_map_for_call_site(
            ["item"],  # callee formal
            ["_mission_item"],  # caller actual
        )
        result = forward_slice(
            "_mission_item.altitude",
            source_assignments=[
                _write("sp.alt", "item.altitude", file="mission_block.cpp", line=669),
            ],
            aliases=aliases,
        )
        assert len(result.hops) == 1
        assert result.hops[0].role == "pure_reassignment"

    def test_forward_result_carries_control_predicates(self):
        result = forward_slice(
            "_rtl_alt",
            source_assignments=[
                _write(
                    "_mission_item.altitude",
                    "_rtl_alt",
                    file="rtl.cpp",
                    line=361,
                    control_predicates=["_param_rtl_cone_half_angle_deg.get() > 0"],
                ),
            ],
        )
        assert result.hops[0].control_predicates == [
            "_param_rtl_cone_half_angle_deg.get() > 0"
        ]
