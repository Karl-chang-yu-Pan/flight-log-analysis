import pytest

from flight_log_agent.analysis.source_expression import (
    SourceExpressionError,
    evaluate_source_expression,
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
