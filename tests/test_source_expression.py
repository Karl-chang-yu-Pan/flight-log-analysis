import pytest

from flight_log_agent.analysis.source_expression import (
    SourceExpressionError,
    alias_dotted_names,
    evaluate_source_expression,
    normalize_source_expression,
    source_expression_names,
)


def test_source_expression_evaluates_arithmetic_safe_math_and_finite_check():
    result = evaluate_source_expression(
        "PX4_ISFINITE(vehicle.alt) ? max(vehicle.alt + RTL_RETURN_ALT, 20.f) : 0.f",
        {"vehicle.alt": 12.0, "RTL_RETURN_ALT": 10.0},
    )

    assert result == 22.0


def test_source_expression_evaluates_bitmask_branch_predicate():
    assert evaluate_source_expression("(MODE & 2) != 0 && enabled", {"MODE": 3, "enabled": True}) is True
    assert source_expression_names("(MODE & 2) != 0 && enabled") == ["MODE", "enabled"]


def test_source_expression_rejects_unsupported_helper_calls():
    with pytest.raises(SourceExpressionError, match="unsupported function"):
        evaluate_source_expression("stateful_helper(value)", {"value": 1.0})


def test_alias_dotted_names_substitutes_longest_first():
    expr, aliases = alias_dotted_names(
        "tecs_status.true_airspeed_sp - tecs_status.equivalent_airspeed_sp",
        ["tecs_status.true_airspeed_sp", "tecs_status.equivalent_airspeed_sp"],
    )
    assert "tecs_status" not in expr
    assert set(aliases.values()) == {"tecs_status.true_airspeed_sp", "tecs_status.equivalent_airspeed_sp"}


def test_alias_dotted_names_skips_unmentioned_names():
    expr, aliases = alias_dotted_names("a + b", ["unused.topic"])
    assert expr == "a + b"
    assert aliases == {}


def test_alias_dotted_names_handles_bracketed_index_as_identifier():
    expr, aliases = alias_dotted_names(
        "vehicle_attitude.q[0] * vehicle_attitude.q[0]",
        ["vehicle_attitude.q[0]"],
    )
    assert "vehicle_attitude.q[0]" not in expr
    assert list(aliases.values()) == ["vehicle_attitude.q[0]"]


def test_normalize_source_expression_does_not_guess_call_as_indexing():
    assert normalize_source_expression("q(0) + lookup(1)") == "q(0) + lookup(1)"
    assert normalize_source_expression("isfinite(x)") == "isfinite(x)"
    assert normalize_source_expression("_param_x.get()") == "_param_x.get()"


def test_source_expression_evaluates_dotted_env_keys():
    # End-to-end: a dotted env key (e.g. tecs_status.true_airspeed_sp) is
    # consumed by evaluate_source_expression via the alias rewrite, without
    # requiring ast.Attribute support in the evaluator itself.
    result = evaluate_source_expression(
        "tecs_status.true_airspeed_sp - tecs_status.equivalent_airspeed_sp",
        {
            "tecs_status.true_airspeed_sp": 25.0,
            "tecs_status.equivalent_airspeed_sp": 20.0,
        },
    )
    assert result == 5.0


def test_source_expression_evaluates_bracket_index_as_identifier():
    result = evaluate_source_expression(
        "vehicle_attitude.q[0] + vehicle_attitude.q[1]",
        {"vehicle_attitude.q[0]": 1.0, "vehicle_attitude.q[1]": 0.5},
    )
    assert result == 1.5


def test_source_expression_names_preserve_constant_indices():
    assert source_expression_names("state.q[0] + state.q[1]") == [
        "state.q[0]",
        "state.q[1]",
    ]
