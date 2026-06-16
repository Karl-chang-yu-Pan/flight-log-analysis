from __future__ import annotations

import pytest

from flight_log_agent.symbols import (
    is_signal_reference,
    looks_like_enum_constant,
    looks_like_signal_reference,
    normalize_symbol,
    parse_signal_reference,
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
