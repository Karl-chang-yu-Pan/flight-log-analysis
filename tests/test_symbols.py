from __future__ import annotations

import pytest

from flight_log_agent.symbols import (
    is_signal_reference,
    looks_like_enum_constant,
    looks_like_signal_reference,
    normalize_symbol,
    parse_signal_reference,
    parse_simple_signal,
)


class TestNormalizeSymbol:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("_pos_sp_triplet.current.alt", "pos_sp_triplet.current.alt"),
            ("vstatus->vehicle_type", "vstatus.vehicle_type"),
            ("vehicle_status_s::VEHICLE_TYPE_ROTARY_WING", "vehicle_status_s.VEHICLE_TYPE_ROTARY_WING"),
            ("position_setpoint_triplet.current.alt", "position_setpoint_triplet.current.alt"),
            ("output[0]", "output"),
            ("actuator_outputs.output[15]", "actuator_outputs.output"),
            ("&pos_sp", "pos_sp"),
            ("*foo", "foo"),
            ("  vehicle_status.nav_state  ", "vehicle_status.nav_state"),
            ("_param_rtl_return_alt", "param_rtl_return_alt"),
            ("a._b._c", "a._b._c"),
        ],
    )
    def test_canonical_forms(self, raw, expected):
        assert normalize_symbol(raw) == expected

    @pytest.mark.parametrize("raw", ["", None])
    def test_empty_inputs(self, raw):
        assert normalize_symbol(raw) == ""

    def test_strips_only_one_leading_underscore(self):
        assert normalize_symbol("__pos_sp") == "_pos_sp"

    def test_preserves_inner_dotted_underscores(self):
        assert normalize_symbol("topic._inner.field") == "topic._inner.field"


class TestIsSignalReference:
    @pytest.mark.parametrize(
        "value",
        [
            "vehicle_status.nav_state",
            "position_setpoint_triplet.current.alt",
            "actuator_outputs.output[3]",
            "sensor_accel[0].x",
            "tecs_status.equivalent_airspeed_sp",
        ],
    )
    def test_accepts_canonical_logged_signals(self, value):
        assert is_signal_reference(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "vehicle_status_s.VEHICLE_TYPE_ROTARY_WING",  # C++ struct type, not a topic
            "position_setpoint_s.alt",
            "_pos_sp_triplet.current.alt",                # leading underscore (pre-norm form)
            "vstatus->vehicle_type",                       # not yet normalized
            "q_gimbal(0)",                                 # paren element access
            "_param_rtl_return_alt.get()",                 # method call
            "VEHICLE_TYPE_AUTO",                           # no topic+field
            "foo",                                         # no dot
            "",
        ],
    )
    def test_rejects_non_canonical(self, value):
        assert is_signal_reference(value) is False

    def test_rejects_none(self):
        assert is_signal_reference(None) is False


class TestLooksLikeSignalReference:
    @pytest.mark.parametrize(
        "value",
        [
            "_pos_sp_triplet.current.alt",
            "vstatus->vehicle_type",
            "position_setpoint_triplet.current.alt",
            "_destination.alt",
        ],
    )
    def test_accepts_pre_normalization_forms(self, value):
        assert looks_like_signal_reference(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "vehicle_status_s.X",            # struct
            "vehicle_status_s->X",           # struct via pointer
            "foo",                           # no dot or arrow
            "q_gimbal(0)",                   # paren
            "VEHICLE_TYPE_AUTO",             # no separator
            "",
        ],
    )
    def test_rejects(self, value):
        assert looks_like_signal_reference(value) is False


class TestParseSignalReference:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("vehicle_status.nav_state", ("vehicle_status", None, "nav_state")),
            ("sensor_accel[0].x", ("sensor_accel", 0, "x")),
            ("position_setpoint_triplet.current.alt", ("position_setpoint_triplet", None, "current.alt")),
            ("actuator_outputs.output[3]", ("actuator_outputs", None, "output[3]")),
            ("vehicle_status[1].nav_state", ("vehicle_status", 1, "nav_state")),
        ],
    )
    def test_parses_canonical_forms(self, value, expected):
        assert parse_signal_reference(value) == expected

    @pytest.mark.parametrize("value", ["vehicle_status_s.X", "foo", "", "Uppercase.field", "vstatus->vehicle_type"])
    def test_rejects_invalid(self, value):
        assert parse_signal_reference(value) is None

    def test_round_trip_with_format(self):
        # Format used by analysis/log_evidence.format_signal: "topic[multi_id].field"
        formatted = "sensor_accel[2].temperature"
        parsed = parse_signal_reference(formatted)
        assert parsed == ("sensor_accel", 2, "temperature")


class TestParseSimpleSignal:
    def test_topic_dot_field(self):
        assert parse_simple_signal("vehicle_status.nav_state") == ("vehicle_status", "nav_state")

    def test_topic_dot_nested_field(self):
        # Nested field stays as-is; the split is on the first dot only.
        assert parse_simple_signal("position_setpoint_triplet.current.alt") == (
            "position_setpoint_triplet",
            "current.alt",
        )

    def test_no_dot_returns_none(self):
        assert parse_simple_signal("topic") is None

    def test_empty_topic_or_field_returns_none(self):
        assert parse_simple_signal(".field") is None
        assert parse_simple_signal("topic.") is None

    def test_non_string_returns_none(self):
        assert parse_simple_signal(None) is None  # type: ignore[arg-type]
        assert parse_simple_signal(123) is None  # type: ignore[arg-type]


class TestLooksLikeEnumConstant:
    @pytest.mark.parametrize(
        "value",
        [
            "vehicle_status.VEHICLE_TYPE_ROTARY_WING",
            "navigator.RTL_STATE_NONE",
            "vehicle_status_s.VEHICLE_TYPE_AUTO",
        ],
    )
    def test_accepts_uppercase_leaf(self, value):
        assert looks_like_enum_constant(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "vehicle_status.nav_state",          # lowercase leaf
            "VEHICLE_TYPE_ROTARY_WING",          # no dot
            "vehicle_status.Mixed_Case",         # mixed leaf
            "",
        ],
    )
    def test_rejects(self, value):
        assert looks_like_enum_constant(value) is False


class TestExactSymbol:
    """Exact identity (the DAG-path form): only syntactic variance
    collapses; indices, instances, and the member underscore distinguish."""

    def test_preserves_identity_markers(self):
        from flight_log_agent.symbols import exact_symbol

        assert exact_symbol("state.q[0]") == "state.q[0]"
        assert exact_symbol("state.q[0]") != exact_symbol("state.q[1]")
        assert exact_symbol("sensor[0].value") != exact_symbol("sensor[1].value")
        assert exact_symbol("_x") == "_x"
        assert exact_symbol("_x") != exact_symbol("x")

    def test_collapses_only_syntax(self):
        from flight_log_agent.symbols import exact_symbol

        assert exact_symbol(" &_dest->alt ") == "_dest.alt"
        assert exact_symbol("Class::member") == "Class.member"
        assert exact_symbol("*ptr") == "ptr"

    def test_empty(self):
        from flight_log_agent.symbols import exact_symbol

        assert exact_symbol("") == ""
        assert exact_symbol(None) == ""


class TestSymbolIndexCompatibility:
    def test_shape_and_compatibility(self):
        from flight_log_agent.symbols import (
            strip_symbol_indices,
            symbol_indices_compatible,
        )

        assert strip_symbol_indices("state.q[0]") == "state.q"
        assert strip_symbol_indices("sensor[1].value") == "sensor.value"

        # index-free covers any index (whole-object write / read)
        assert symbol_indices_compatible("q", "q[0]")
        assert symbol_indices_compatible("state.q[2]", "state.q")
        # explicit indices never fuse
        assert not symbol_indices_compatible("state.q[0]", "state.q[1]")
        assert not symbol_indices_compatible("sensor[0].value", "sensor[1].value")
        assert symbol_indices_compatible("sensor[1].value", "sensor[1].value")
