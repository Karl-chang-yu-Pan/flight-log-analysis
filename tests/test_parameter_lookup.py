from __future__ import annotations

import math

import pytest

from flight_log_agent.analysis.parameter_lookup import (
    CXX_STDLIB_CONSTANTS,
    PX4_PARAM_NAME_LIMIT,
    filter_present_parameters,
    get_parameter,
    has_parameter,
    is_px4_parameter_name,
    lookup_cxx_constant,
    resolve_numeric_value,
)


class TestIsPx4ParameterName:
    @pytest.mark.parametrize(
        "name",
        ["RTL_RETURN_ALT", "SYS_AUTOSTART", "MC_PITCH_P", "FW_AIRSPD_TRIM"],
    )
    def test_accepts_real_px4_params(self, name):
        assert is_px4_parameter_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "rtl_return_alt",         # lowercase
            "RtlReturnAlt",           # mixed case
            "RTL",                    # no underscore
            "_RTL_TYPE",              # leading underscore
            "FLT_EPSILON_TOO_LONG_NAME",  # > 16 chars
            "",
        ],
    )
    def test_rejects_non_param_names(self, name):
        assert is_px4_parameter_name(name) is False

    def test_enforces_16_char_limit(self):
        assert PX4_PARAM_NAME_LIMIT == 16
        assert is_px4_parameter_name("A" * 17) is False
        assert is_px4_parameter_name("A_" + "B" * 14) is True   # exactly 16


class TestGetParameter:
    def test_returns_native_int_in_auto_mode(self):
        inv = {"parameters": {"SYS_AUTOSTART": 13000, "RTL_TYPE": 0}}
        value, reason = get_parameter(inv, "SYS_AUTOSTART")
        assert value == 13000
        assert isinstance(value, int)
        assert reason is None

    def test_returns_native_float_in_auto_mode(self):
        inv = {"parameters": {"FW_AIRSPD_TRIM": 21.0}}
        value, reason = get_parameter(inv, "FW_AIRSPD_TRIM")
        assert value == 21.0
        assert isinstance(value, float)
        assert reason is None

    def test_does_not_promote_int_to_float_in_auto_mode(self):
        inv = {"parameters": {"RTL_TYPE": 0}}
        value, _ = get_parameter(inv, "RTL_TYPE")
        assert isinstance(value, int)
        assert not isinstance(value, float)

    def test_float_kind_coerces_int_to_float(self):
        inv = {"parameters": {"RTL_TYPE": 2}}
        value, reason = get_parameter(inv, "RTL_TYPE", kind="float")
        assert value == 2.0
        assert isinstance(value, float)
        assert reason is None

    def test_int_kind_truncates_float(self):
        inv = {"parameters": {"FW_AIRSPD_TRIM": 21.7}}
        value, reason = get_parameter(inv, "FW_AIRSPD_TRIM", kind="int")
        assert value == 21
        assert isinstance(value, int)
        assert reason is None

    def test_bool_kind_treats_nonzero_as_true(self):
        inv = {"parameters": {"MAV_USEHILGPS": 1, "MAV_PROTO_VER": 0}}
        assert get_parameter(inv, "MAV_USEHILGPS", kind="bool") == (True, None)
        assert get_parameter(inv, "MAV_PROTO_VER", kind="bool") == (False, None)

    def test_missing_parameter_returns_actionable_reason(self):
        inv = {"parameters": {"RTL_RETURN_ALT": 30.0}}
        value, reason = get_parameter(inv, "RTL_TYPE")
        assert value is None
        assert reason is not None
        assert "RTL_TYPE" in reason
        assert "not set in this log" in reason

    def test_rejects_non_px4_parameter_shape(self):
        inv = {"parameters": {"FLT_EPSILON": 1e-7}}  # would be stored erroneously
        value, reason = get_parameter(inv, "flt_epsilon")
        assert value is None
        assert reason is not None
        assert "not a PX4 parameter name" in reason

    def test_empty_name_rejected(self):
        value, reason = get_parameter({}, "")
        assert value is None
        assert reason == "parameter name is empty"

    def test_inventory_can_be_parameters_dict_directly(self):
        # Some callers pass the parameters dict, not the full inventory.
        value, reason = get_parameter({"RTL_RETURN_ALT": 30.0}, "RTL_RETURN_ALT")
        assert value == 30.0
        assert reason is None

    def test_non_numeric_value_with_float_kind_reports_reason(self):
        inv = {"parameters": {"RTL_RETURN_ALT": "not a number"}}
        value, reason = get_parameter(inv, "RTL_RETURN_ALT", kind="float")
        assert value is None
        assert reason is not None
        assert "not numeric" in reason

    def test_nan_is_rejected_in_float_kind(self):
        inv = {"parameters": {"RTL_RETURN_ALT": math.nan}}
        value, reason = get_parameter(inv, "RTL_RETURN_ALT", kind="float")
        assert value is None
        assert reason is not None


class TestHasAndFilter:
    def test_has_parameter(self):
        inv = {"parameters": {"RTL_RETURN_ALT": 30.0}}
        assert has_parameter(inv, "RTL_RETURN_ALT") is True
        assert has_parameter(inv, "RTL_TYPE") is False

    def test_filter_present_parameters_preserves_order(self):
        inv = {"parameters": {"RTL_RETURN_ALT": 30.0, "SYS_AUTOSTART": 13000}}
        names = ["RTL_TYPE", "RTL_RETURN_ALT", "MC_PITCH_P", "SYS_AUTOSTART"]
        assert filter_present_parameters(inv, names) == ["RTL_RETURN_ALT", "SYS_AUTOSTART"]


class TestResolveNumericValue:
    def test_passes_through_numeric_literal(self):
        v, missing = resolve_numeric_value(21.0, {})
        assert v == 21.0 and missing is None

    def test_parses_numeric_string(self):
        v, missing = resolve_numeric_value("21.0", {})
        assert v == 21.0 and missing is None

    def test_looks_up_px4_parameter_name(self):
        inv = {"parameters": {"FW_AIRSPD_TRIM": 21.0}}
        v, missing = resolve_numeric_value("FW_AIRSPD_TRIM", inv)
        assert v == 21.0 and missing is None

    def test_missing_parameter_returns_reason(self):
        inv = {"parameters": {}}
        v, missing = resolve_numeric_value("FW_AIRSPD_TRIM", inv)
        assert v is None and missing is not None

    def test_unknown_identifier_returns_reason(self):
        v, missing = resolve_numeric_value("not_a_param", {})
        assert v is None and missing is not None

    def test_none_passes_through(self):
        v, missing = resolve_numeric_value(None, {})
        assert v is None and missing is None


# Const-expression evaluator tests live in test_safe_eval.py.


class TestLookupCxxConstant:
    def test_flt_epsilon_is_single_precision(self):
        # FLT_EPSILON is the IEEE 754 single-precision epsilon, ~1.19e-7.
        v = lookup_cxx_constant("FLT_EPSILON")
        assert v is not None
        assert 1e-7 < v < 2e-7

    def test_dbl_epsilon_distinct_from_flt(self):
        # DBL_EPSILON is the double-precision epsilon, ~2.22e-16.
        assert lookup_cxx_constant("DBL_EPSILON") < lookup_cxx_constant("FLT_EPSILON")

    def test_m_pi(self):
        v = lookup_cxx_constant("M_PI")
        assert v is not None
        assert 3.14 < v < 3.15

    def test_unknown_returns_none(self):
        assert lookup_cxx_constant("NOT_A_CONST") is None
        assert lookup_cxx_constant("") is None
        assert lookup_cxx_constant(None) is None  # type: ignore[arg-type]


class TestResolveNumericValueStdlibFallback:
    def test_flt_epsilon_resolved_via_stdlib(self):
        v, missing = resolve_numeric_value("FLT_EPSILON", {})
        assert v is not None and missing is None
        assert 1e-7 < v < 2e-7

    def test_param_wins_over_stdlib_when_present(self):
        # If a PX4 parameter shares a name with a stdlib constant (an
        # edge case but worth pinning), the parameter value wins.
        inv = {"parameters": {"M_PI": 999.0}}
        v, missing = resolve_numeric_value("M_PI", inv)
        assert v == 999.0 and missing is None
