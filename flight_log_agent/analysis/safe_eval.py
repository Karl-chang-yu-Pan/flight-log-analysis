"""Shared safe AST expression evaluator.

Single owner of the evaluator that previously lived inline in
``ulog.signature_evaluator``. Used at runtime for derived_expression
checks (with an ``env`` of bound signal values) AND at compile time
for constant resolution of PX4 enum entries / ``#define`` macros
(with an empty ``env``).

Restricted node set: constants, names (env lookup), unary, binary
arithmetic, bitwise / shift, boolean, ternary, function calls into
:data:`SAFE_MATH_FUNCTIONS`, and comparisons. Anything else raises
:class:`ExpressionEvaluationError`.
"""

from __future__ import annotations

import ast
from typing import Any, Optional, Union

from flight_log_agent.expression_math import SAFE_MATH_FUNCTIONS
from flight_log_agent.utils import compare as _compare
from flight_log_agent.utils import json_safe_value as _json_safe_value
from flight_log_agent.utils import safe_float as _number


ALLOWED_EXPRESSION_FUNCTIONS = SAFE_MATH_FUNCTIONS


class ExpressionEvaluationError(ValueError):
    """Raised when the safe evaluator refuses or fails to evaluate."""


def eval_expression(expression: str, env: Optional[dict[str, Any]] = None) -> Any:
    """Parse and evaluate ``expression`` against ``env``.

    Raises :class:`ExpressionEvaluationError` on parse errors,
    unsupported syntax, or missing names. Returns the evaluated value
    (number, bool, or whatever an env entry holds).
    """
    tree = _parse_expression(expression)
    return _eval_expression_node(tree, env or {})


def eval_const_expression(expression: str) -> Optional[Union[int, float]]:
    """Evaluate a constant arithmetic expression to a number.

    Convenience wrapper for the compile-time constant case: parses
    ``expression`` against an empty env, returns the result if numeric,
    or ``None`` on any failure. Used by the predicate parser to fold
    enum entries / ``#define`` macros like ``(1 << 1)`` into ``2``.
    """
    if not isinstance(expression, str) or not expression.strip():
        return None
    try:
        result = eval_expression(expression, {})
    except (ExpressionEvaluationError, TypeError, ValueError, ZeroDivisionError):
        return None
    if isinstance(result, bool):
        return int(result)
    if isinstance(result, (int, float)):
        return result
    return None


def _parse_expression(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionEvaluationError("invalid expression syntax") from exc
    return tree


def _eval_expression_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_expression_node(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ExpressionEvaluationError("string literals are not supported")
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExpressionEvaluationError(f"missing runtime variable {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _numeric_expression_value(_eval_expression_node(node.operand, env))
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_eval_expression_node(node.operand, env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return ~int(_numeric_expression_value(_eval_expression_node(node.operand, env)))
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return all(bool(_eval_expression_node(value, env)) for value in node.values)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return any(bool(_eval_expression_node(value, env)) for value in node.values)
    if isinstance(node, ast.IfExp):
        branch = node.body if bool(_eval_expression_node(node.test, env)) else node.orelse
        return _eval_expression_node(branch, env)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
        left = _numeric_expression_value(_eval_expression_node(node.left, env))
        right = _numeric_expression_value(_eval_expression_node(node.right, env))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ExpressionEvaluationError("division by zero")
        if isinstance(node.op, ast.Mod):
            return left % right
        return left / right
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
    ):
        left = int(_numeric_expression_value(_eval_expression_node(node.left, env)))
        right = int(_numeric_expression_value(_eval_expression_node(node.right, env)))
        if isinstance(node.op, ast.BitAnd):
            return left & right
        if isinstance(node.op, ast.BitOr):
            return left | right
        if isinstance(node.op, ast.BitXor):
            return left ^ right
        if isinstance(node.op, ast.LShift):
            return left << right
        return left >> right
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_EXPRESSION_FUNCTIONS:
            raise ExpressionEvaluationError("unsupported function")
        if node.keywords:
            raise ExpressionEvaluationError("keyword arguments are not supported")
        args = [_numeric_expression_value(_eval_expression_node(arg, env)) for arg in node.args]
        if not args:
            raise ExpressionEvaluationError("function requires at least one argument")
        return ALLOWED_EXPRESSION_FUNCTIONS[node.func.id](*args)
    if isinstance(node, ast.Compare):
        left = _eval_expression_node(node.left, env)
        for operator, comparator in zip(node.ops, node.comparators):
            right = _eval_expression_node(comparator, env)
            op = _comparison_operator(operator)
            if not _compare_literal(left, op, right):
                return False
            left = right
        return True
    raise ExpressionEvaluationError("unsupported expression syntax")


def _comparison_operator(operator: ast.cmpop) -> str:
    if isinstance(operator, ast.Gt):
        return ">"
    if isinstance(operator, ast.GtE):
        return ">="
    if isinstance(operator, ast.Lt):
        return "<"
    if isinstance(operator, ast.LtE):
        return "<="
    if isinstance(operator, ast.Eq):
        return "=="
    if isinstance(operator, ast.NotEq):
        return "!="
    raise ExpressionEvaluationError("unsupported comparison operator")


def _numeric_expression_value(value: Any) -> float:
    number = _number(value)
    if number is None:
        raise ExpressionEvaluationError(f"non-numeric expression value: {value}")
    return number


def _compare_literal(actual: Any, op: str, expected: Any, *, tolerance: Optional[float] = None) -> bool:
    actual_number = _number(actual)
    expected_number = _number(expected)
    if actual_number is not None and expected_number is not None:
        if op == "==" and tolerance is not None:
            return abs(actual_number - expected_number) <= tolerance
        if op == "!=" and tolerance is not None:
            return abs(actual_number - expected_number) > tolerance
        return _compare(actual_number, op, expected_number)

    if op == "==":
        return _normalized_literal(actual) == _normalized_literal(expected)
    if op == "!=":
        return _normalized_literal(actual) != _normalized_literal(expected)
    raise ValueError(f"operator {op} requires numeric values")


def _normalized_literal(value: Any) -> Any:
    value = _json_safe_value(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value.strip()
    return value
