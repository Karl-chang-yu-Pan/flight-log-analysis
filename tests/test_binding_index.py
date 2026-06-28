from __future__ import annotations

import pytest

from flight_log_agent.analysis.binding_index import BindingIndex
from flight_log_agent.px4.source_mechanism_models import SourceOutputBindingRecord


def _binding(
    *,
    logged_signal: str = "",
    source_symbol: str = "",
    target_symbol: str = "",
    symbol_bindings: dict[str, str] | None = None,
    assignment_path: list[dict] | None = None,
) -> SourceOutputBindingRecord:
    return SourceOutputBindingRecord(
        binding_id=f"binding_{logged_signal}_{source_symbol}".replace(".", "_"),
        logged_signal=logged_signal,
        source_symbol=source_symbol,
        target_symbol=target_symbol,
        symbol_bindings=symbol_bindings or {},
        assignment_path=assignment_path or [],
    )


def _inventory(topic_fields: dict[str, list[str]]) -> dict:
    return {
        "topic_fields": topic_fields,
        "available_topics": list(topic_fields),
        "parameters": {},
    }


class TestIsKnown:
    def test_logged_signal_directly(self):
        index = BindingIndex(_inventory({"vehicle_status": ["nav_state"]}), [])
        assert index.is_known("vehicle_status.nav_state") is True

    def test_alias_via_binding(self):
        bindings = [_binding(
            logged_signal="position_setpoint_triplet.current.cruising_speed",
            source_symbol="sp.cruising_speed",
            target_symbol="pos_sp_triplet.current.cruising_speed",
        )]
        index = BindingIndex(
            _inventory({"position_setpoint_triplet": ["current.cruising_speed"]}),
            bindings,
        )
        assert index.is_known("sp.cruising_speed") is True

    def test_suffix_match(self):
        bindings = [_binding(
            logged_signal="tecs_status.equivalent_airspeed_sp",
            source_symbol="tecs_status.equivalent_airspeed_sp",
            target_symbol="tecs_status.equivalent_airspeed_sp",
        )]
        index = BindingIndex(
            _inventory({"tecs_status": ["equivalent_airspeed_sp"]}),
            bindings,
        )
        assert index.is_known("foo.equivalent_airspeed_sp") is True

    def test_unknown(self):
        index = BindingIndex(_inventory({"vehicle_status": ["nav_state"]}), [])
        assert index.is_known("never_heard_of_this.field") is False


class TestResolve:
    def test_exact_logged_signal(self):
        index = BindingIndex(_inventory({"vehicle_status": ["nav_state"]}), [])
        resolution = index.resolve("vehicle_status.nav_state")
        assert resolution.status == "resolved"
        assert resolution.resolved == "vehicle_status.nav_state"

    def test_single_alias_resolves(self):
        bindings = [_binding(
            logged_signal="position_setpoint_triplet.current.cruising_speed",
            source_symbol="sp.cruising_speed",
            target_symbol="pos_sp_triplet.current.cruising_speed",
        )]
        index = BindingIndex(
            _inventory({"position_setpoint_triplet": ["current.cruising_speed"]}),
            bindings,
        )
        resolution = index.resolve("sp.cruising_speed")
        assert resolution.status == "resolved"
        assert resolution.resolved == "position_setpoint_triplet.current.cruising_speed"

    def test_suffix_collision_is_ambiguous_without_prefer(self):
        # The classic 6/14 cruising_speed case: both previous and current
        # share the .cruising_speed suffix via the previous = current copy.
        bindings = [
            _binding(
                logged_signal="position_setpoint_triplet.current.cruising_speed",
                source_symbol="pos_sp_triplet.current.cruising_speed",
                target_symbol="pos_sp_triplet.current.cruising_speed",
            ),
            _binding(
                logged_signal="position_setpoint_triplet.previous.cruising_speed",
                source_symbol="pos_sp_triplet.previous.cruising_speed",
                target_symbol="pos_sp_triplet.previous.cruising_speed",
            ),
        ]
        index = BindingIndex(
            _inventory({
                "position_setpoint_triplet": [
                    "current.cruising_speed",
                    "previous.cruising_speed",
                ],
            }),
            bindings,
        )
        resolution = index.resolve("position_setpoint.cruising_speed")
        assert resolution.status == "ambiguous"
        assert set(resolution.candidates) == {
            "position_setpoint_triplet.current.cruising_speed",
            "position_setpoint_triplet.previous.cruising_speed",
        }

    def test_prefer_tiebreaker_picks_one_candidate(self):
        bindings = [
            _binding(
                logged_signal="position_setpoint_triplet.current.cruising_speed",
                source_symbol="pos_sp_triplet.current.cruising_speed",
                target_symbol="pos_sp_triplet.current.cruising_speed",
            ),
            _binding(
                logged_signal="position_setpoint_triplet.previous.cruising_speed",
                source_symbol="pos_sp_triplet.previous.cruising_speed",
                target_symbol="pos_sp_triplet.previous.cruising_speed",
            ),
        ]
        index = BindingIndex(
            _inventory({
                "position_setpoint_triplet": [
                    "current.cruising_speed",
                    "previous.cruising_speed",
                ],
            }),
            bindings,
        )
        # The candidate cares about the "current" side (its primary output).
        resolution = index.resolve(
            "position_setpoint.cruising_speed",
            prefer={"position_setpoint_triplet.current.cruising_speed"},
        )
        assert resolution.status == "resolved"
        assert resolution.resolved == "position_setpoint_triplet.current.cruising_speed"

    def test_prefer_does_not_invent_a_candidate(self):
        bindings = [_binding(
            logged_signal="vehicle_status.nav_state",
            source_symbol="vstatus.nav_state",
            target_symbol="vstatus.nav_state",
        )]
        index = BindingIndex(_inventory({"vehicle_status": ["nav_state"]}), bindings)
        # prefer is unrelated to the actual candidates, so resolution stays
        # unambiguous via the original single alias.
        resolution = index.resolve(
            "vstatus.nav_state",
            prefer={"unrelated.signal"},
        )
        assert resolution.status == "resolved"
        assert resolution.resolved == "vehicle_status.nav_state"

    def test_schema_only_signal_is_unresolved_with_reason(self):
        # The schema knows about a topic that isn't in the log.
        index = BindingIndex(
            _inventory({"vehicle_status": ["nav_state"]}),  # no battery_status
            [],
        )
        # Pretend schema has battery_status.voltage_v even though the log doesn't.
        index.schema_signals.add("battery_status.voltage_v")
        resolution = index.resolve("battery_status.voltage_v")
        assert resolution.status == "unresolved"
        assert "not present in the log" in (resolution.reason or "")

    def test_no_match_at_all(self):
        index = BindingIndex(_inventory({"vehicle_status": ["nav_state"]}), [])
        resolution = index.resolve("totally.unknown")
        assert resolution.status == "unresolved"
        assert "no deterministic logged output-binding match" in (resolution.reason or "")


class TestCanonicalize:
    def test_returns_canonical_when_unique(self):
        bindings = [_binding(
            logged_signal="vehicle_status.nav_state",
            source_symbol="vstatus.nav_state",
            target_symbol="vstatus.nav_state",
        )]
        index = BindingIndex(_inventory({"vehicle_status": ["nav_state"]}), bindings)
        assert index.canonicalize("vstatus.nav_state") == "vehicle_status.nav_state"

    def test_returns_input_on_ambiguity(self):
        bindings = [
            _binding(
                logged_signal="position_setpoint_triplet.current.cruising_speed",
                source_symbol="pos_sp_triplet.current.cruising_speed",
                target_symbol="pos_sp_triplet.current.cruising_speed",
            ),
            _binding(
                logged_signal="position_setpoint_triplet.previous.cruising_speed",
                source_symbol="pos_sp_triplet.previous.cruising_speed",
                target_symbol="pos_sp_triplet.previous.cruising_speed",
            ),
        ]
        index = BindingIndex(
            _inventory({
                "position_setpoint_triplet": [
                    "current.cruising_speed",
                    "previous.cruising_speed",
                ],
            }),
            bindings,
        )
        # No prefer → ambiguous → input passes through.
        assert index.canonicalize("position_setpoint.cruising_speed") == "position_setpoint.cruising_speed"

    def test_returns_canonical_when_prefer_disambiguates(self):
        bindings = [
            _binding(
                logged_signal="position_setpoint_triplet.current.cruising_speed",
                source_symbol="pos_sp_triplet.current.cruising_speed",
                target_symbol="pos_sp_triplet.current.cruising_speed",
            ),
            _binding(
                logged_signal="position_setpoint_triplet.previous.cruising_speed",
                source_symbol="pos_sp_triplet.previous.cruising_speed",
                target_symbol="pos_sp_triplet.previous.cruising_speed",
            ),
        ]
        index = BindingIndex(
            _inventory({
                "position_setpoint_triplet": [
                    "current.cruising_speed",
                    "previous.cruising_speed",
                ],
            }),
            bindings,
        )
        canonical = index.canonicalize(
            "position_setpoint.cruising_speed",
            prefer={"position_setpoint_triplet.current.cruising_speed"},
        )
        assert canonical == "position_setpoint_triplet.current.cruising_speed"

    def test_empty_input_passes_through(self):
        index = BindingIndex(_inventory({}), [])
        assert index.canonicalize("") == ""
        assert index.canonicalize(None) is None


class TestSliceForTerminal:
    def test_returns_terminal_and_upstream_signals(self):
        # tecs_status.equivalent_airspeed_sp ← pos_sp_triplet.current.cruising_speed
        bindings = [
            _binding(
                logged_signal="tecs_status.equivalent_airspeed_sp",
                source_symbol="position_setpoint_triplet.current.cruising_speed",
                target_symbol="tecs_status.equivalent_airspeed_sp",
            ),
            _binding(
                logged_signal="position_setpoint_triplet.current.cruising_speed",
                source_symbol="vehicle_command.param2",
                target_symbol="position_setpoint_triplet.current.cruising_speed",
            ),
            _binding(
                # An unrelated "previous" binding that should NOT appear in
                # the slice for the equivalent_airspeed_sp terminal.
                logged_signal="position_setpoint_triplet.previous.cruising_speed",
                source_symbol="position_setpoint_triplet.current.cruising_speed",
                target_symbol="position_setpoint_triplet.previous.cruising_speed",
            ),
        ]
        index = BindingIndex(
            _inventory({
                "tecs_status": ["equivalent_airspeed_sp"],
                "position_setpoint_triplet": [
                    "current.cruising_speed",
                    "previous.cruising_speed",
                ],
                "vehicle_command": ["param2"],
            }),
            bindings,
        )

        slice_signals = index.slice_for_terminal("tecs_status.equivalent_airspeed_sp")

        assert "tecs_status.equivalent_airspeed_sp" in slice_signals
        assert "position_setpoint_triplet.current.cruising_speed" in slice_signals
        # The "previous" binding is NOT upstream of the chosen terminal.
        assert "position_setpoint_triplet.previous.cruising_speed" not in slice_signals

    def test_empty_terminal_returns_empty_set(self):
        index = BindingIndex(_inventory({}), [])
        assert index.slice_for_terminal("") == set()

    def test_unknown_terminal_returns_empty_set(self):
        index = BindingIndex(_inventory({"vehicle_status": ["nav_state"]}), [])
        assert index.slice_for_terminal("totally.unknown") == set()


class TestBindingsReaching:
    def test_returns_bindings_in_terminal_chain(self):
        bindings = [
            _binding(
                logged_signal="tecs_status.equivalent_airspeed_sp",
                source_symbol="position_setpoint_triplet.current.cruising_speed",
                target_symbol="tecs_status.equivalent_airspeed_sp",
            ),
            _binding(
                logged_signal="position_setpoint_triplet.current.cruising_speed",
                source_symbol="vehicle_command.param2",
                target_symbol="position_setpoint_triplet.current.cruising_speed",
            ),
            _binding(
                logged_signal="vehicle_global_position.alt",
                source_symbol="unrelated_field",
                target_symbol="vehicle_global_position.alt",
            ),
        ]
        index = BindingIndex(
            _inventory({
                "tecs_status": ["equivalent_airspeed_sp"],
                "position_setpoint_triplet": ["current.cruising_speed"],
                "vehicle_command": ["param2"],
                "vehicle_global_position": ["alt"],
            }),
            bindings,
        )

        reaching = index.bindings_reaching("tecs_status.equivalent_airspeed_sp")
        logged_signals = {b.get("logged_signal") for b in reaching}
        assert "tecs_status.equivalent_airspeed_sp" in logged_signals
        assert "position_setpoint_triplet.current.cruising_speed" in logged_signals
        # Unrelated binding NOT in the slice.
        assert "vehicle_global_position.alt" not in logged_signals

    def test_empty_terminal_returns_empty_list(self):
        index = BindingIndex(_inventory({}), [])
        assert index.bindings_reaching("") == []


def _helper(
    *,
    name: str,
    lowered: str | None = None,
    helper_calls: list[str] | None = None,
    parameters: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "lowered_return_expression": lowered,
        "helper_calls": helper_calls or [],
        "parameters": parameters or [],
    }


def _assignment(target: str, expression: str) -> dict:
    return {
        "target": target,
        "expression": expression,
        "control_predicates": [],
        "file": "fake.cpp",
        "line": 1,
    }


class TestHelperResolutions:
    def test_leaf_helper_resolves_to_logged_signal(self):
        inv = _inventory({"vehicle_global_position": ["alt"]})
        index = BindingIndex(
            inv,
            [],
            helper_expressions=[
                _helper(name="get_global_alt", lowered="vehicle_global_position.alt"),
            ],
        )
        result = index.helper_resolutions["get_global_alt"]
        assert result.fully_resolved is True
        assert "vehicle_global_position.alt" in result.expression

    def test_nested_helper_inlines_resolved_inner(self):
        # outer() calls inner(); both must resolve and outer's resolution
        # must reference the logged signal, not inner's literal call.
        inv = _inventory({"vehicle_global_position": ["alt"]})
        helpers = [
            _helper(name="inner", lowered="vehicle_global_position.alt"),
            _helper(name="outer", lowered="inner()", helper_calls=["inner"]),
        ]
        index = BindingIndex(inv, [], helper_expressions=helpers)
        outer_result = index.helper_resolutions["outer"]
        assert outer_result.fully_resolved is True
        assert "vehicle_global_position.alt" in outer_result.expression
        # No literal call survives in the resolved form.
        assert "inner(" not in outer_result.expression

    def test_cycle_does_not_hang_and_records_partial(self):
        # a() calls b(); b() calls a(). The materializer should still
        # produce entries for both (with blockers / partial expressions),
        # not hang.
        inv = _inventory({"vehicle_global_position": ["alt"]})
        helpers = [
            _helper(name="a", lowered="b()", helper_calls=["b"]),
            _helper(name="b", lowered="a()", helper_calls=["a"]),
        ]
        index = BindingIndex(inv, [], helper_expressions=helpers)
        assert "a" in index.helper_resolutions
        assert "b" in index.helper_resolutions

    def test_external_call_in_body_passes_through(self):
        # External (non-registered) calls are not lowered; they pass
        # through unchanged. The verifier downstream is responsible for
        # treating un-substituted calls as evaluation failures.
        inv = _inventory({"vehicle_global_position": ["alt"]})
        helpers = [
            _helper(
                name="needs_external",
                lowered="some_external_thing(vehicle_global_position.alt)",
                helper_calls=["some_external_thing"],
            ),
        ]
        index = BindingIndex(inv, [], helper_expressions=helpers)
        result = index.helper_resolutions["needs_external"]
        # The resolution still carries the substituted form with the
        # external call literal preserved.
        assert "some_external_thing" in result.expression
        assert "vehicle_global_position.alt" in result.expression

    def test_short_name_lookup_works(self):
        inv = _inventory({"vehicle_global_position": ["alt"]})
        helpers = [
            _helper(name="Navigator::get_alt", lowered="vehicle_global_position.alt"),
        ]
        index = BindingIndex(inv, [], helper_expressions=helpers)
        # Both fully-qualified and short forms map to the same resolution.
        assert "Navigator::get_alt" in index.helper_resolutions
        assert "get_alt" in index.helper_resolutions
        assert (
            index.helper_resolutions["get_alt"].expression
            == index.helper_resolutions["Navigator::get_alt"].expression
        )

    def test_no_helpers_means_empty_table(self):
        # Default kwargs: nothing materialized, no surprises.
        index = BindingIndex(_inventory({"vehicle_status": ["nav_state"]}), [])
        assert index.helper_resolutions == {}
        assert index.assignment_resolutions == {}


class TestAssignmentResolutions:
    def test_simple_assignment_resolves_to_logged(self):
        inv = _inventory({"vehicle_global_position": ["alt"]})
        index = BindingIndex(
            inv,
            [],
            source_assignments=[
                _assignment("_destination.alt", "vehicle_global_position.alt"),
            ],
        )
        # Note: normalize_symbol strips leading underscore.
        result = index.assignment_resolutions["destination.alt"]
        assert result.fully_resolved is True
        assert "vehicle_global_position.alt" in result.expression

    def test_assignment_with_dead_end_records_blocker(self):
        inv = _inventory({"vehicle_global_position": ["alt"]})
        index = BindingIndex(
            inv,
            [],
            source_assignments=[
                _assignment("destination.alt", "some_unknown.x + vehicle_global_position.alt"),
            ],
        )
        result = index.assignment_resolutions["destination.alt"]
        assert result.fully_resolved is False
        # The resolved branch still got substituted; the dead end stays
        # as a blocker.
        assert "vehicle_global_position.alt" in result.expression
