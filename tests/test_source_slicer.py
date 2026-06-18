from __future__ import annotations

import pytest

from flight_log_agent.analysis.source_slicer import (
    SliceBlocker,
    SliceResult,
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
