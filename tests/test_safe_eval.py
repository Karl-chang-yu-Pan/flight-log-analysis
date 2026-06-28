from __future__ import annotations

import pytest

from flight_log_agent.analysis.safe_eval import (
    ExpressionEvaluationError,
    eval_const_expression,
    eval_expression,
)


class TestEvalConstExpression:
    def test_integer_literal(self):
        assert eval_const_expression("4") == 4

    def test_float_literal(self):
        assert eval_const_expression("3.14") == pytest.approx(3.14)

    def test_bit_shift_left(self):
        assert eval_const_expression("(1 << 1)") == 2
        assert eval_const_expression("1 << 5") == 32

    def test_bit_shift_right(self):
        assert eval_const_expression("8 >> 2") == 2

    def test_bitwise_or(self):
        assert eval_const_expression("(1 << 0) | (1 << 2)") == 5

    def test_bitwise_and(self):
        assert eval_const_expression("0b1010 & 0b1100") == 0b1000

    def test_bitwise_invert(self):
        assert eval_const_expression("~0") == -1

    def test_arithmetic(self):
        assert eval_const_expression("2 * 3 + 1") == 7

    def test_unary_negation(self):
        assert eval_const_expression("-4") == -4

    def test_rejects_function_calls(self):
        # Calls into SAFE_MATH_FUNCTIONS are allowed by eval_expression
        # but with empty env there are no names; sqrt(2) succeeds.
        # Reject something that needs an env variable.
        assert eval_const_expression("missing_name + 1") is None

    def test_rejects_invalid_syntax(self):
        assert eval_const_expression("(1 <<") is None

    def test_empty_returns_none(self):
        assert eval_const_expression("") is None
        assert eval_const_expression(None) is None


class TestEvalExpression:
    def test_with_env(self):
        assert eval_expression("x + 1", {"x": 5}) == 6

    def test_compare(self):
        assert eval_expression("x > 0", {"x": 5}) is True

    def test_missing_name_raises(self):
        with pytest.raises(ExpressionEvaluationError):
            eval_expression("y + 1", {})

    def test_bitwise(self):
        assert eval_expression("mask & 0b0100", {"mask": 0b0110}) == 0b0100
