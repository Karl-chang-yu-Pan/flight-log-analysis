from __future__ import annotations

import ast
import math
import re
from typing import Any

from flight_log_agent.expression_math import SAFE_MATH_FUNCTIONS, normalize_expression_function_names


class SourceExpressionError(ValueError):
    pass


def normalize_source_expression(expression: str) -> str:
    normalized = normalize_expression_function_names(str(expression or "").strip())
    normalized = normalized.replace("PX4_ISFINITE", "isfinite")
    normalized = normalized.replace("&&", " and ").replace("||", " or ")
    normalized = re.sub(r"(?<![=!<>])!(?!=)", " not ", normalized)
    normalized = re.sub(r"\btrue\b", "True", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"(?P<number>\d+)\.[fF]\b", r"\g<number>.0", normalized)
    normalized = re.sub(r"(?<=\d)[fF]\b", "", normalized)
    normalized = normalize_simple_ternary(normalized)
    return normalized.strip()


def normalize_simple_ternary(expression: str) -> str:
    question = top_level_operator_index(expression, "?")
    if question < 0:
        return expression
    colon = top_level_operator_index(expression, ":", start=question + 1)
    if colon < 0:
        return expression
    condition = expression[:question].strip()
    when_true = expression[question + 1:colon].strip()
    when_false = expression[colon + 1:].strip()
    return f"({when_true} if {condition} else {when_false})"


def top_level_operator_index(expression: str, operator: str, *, start: int = 0) -> int:
    depth = 0
    for index in range(start, len(expression)):
        char = expression[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == operator and depth == 0:
            return index
    return -1


def source_expression_names(expression: str) -> list[str]:
    try:
        tree = ast.parse(normalize_source_expression(expression), mode="eval")
    except SyntaxError:
        return []
    names: list[tuple[int, int, str]] = []
    attribute_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = attribute_name(node)
            if name:
                names.append((node.lineno, node.col_offset, name))
                attribute_roots.add(name.split(".", 1)[0])
    function_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    names.extend(
        (node.lineno, node.col_offset, node.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id not in function_names
        and node.id not in attribute_roots
    )
    return list(dict.fromkeys(name for _, _, name in sorted(names)))


def evaluate_source_expression(expression: str, env: dict[str, Any]) -> Any:
    normalized = normalize_source_expression(expression)
    bound_env: dict[str, Any] = {}
    for index, name in enumerate(sorted(env, key=len, reverse=True)):
        alias = f"__value_{index}"
        normalized = re.sub(
            rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_.])",
            alias,
            normalized,
        )
        bound_env[alias] = env[name]
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise SourceExpressionError("invalid expression syntax") from exc
    return evaluate_node(tree, bound_env)


def attribute_name(node: ast.Attribute) -> str | None:
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return None
    parts.append(value.id)
    return ".".join(reversed(parts))


def evaluate_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return evaluate_node(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise SourceExpressionError(f"missing runtime variable {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)):
        value = evaluate_node(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not bool(value)
        if isinstance(node.op, ast.Invert):
            return ~int(numeric(value))
        return numeric(value) if isinstance(node.op, ast.UAdd) else -numeric(value)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return all(bool(evaluate_node(value, env)) for value in node.values)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return any(bool(evaluate_node(value, env)) for value in node.values)
    if isinstance(node, ast.IfExp):
        selected = node.body if bool(evaluate_node(node.test, env)) else node.orelse
        return evaluate_node(selected, env)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor),
    ):
        left = numeric(evaluate_node(node.left, env))
        right = numeric(evaluate_node(node.right, env))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise SourceExpressionError("division by zero")
            return left / right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise SourceExpressionError("modulo by zero")
            return left % right
        if isinstance(node.op, ast.BitAnd):
            return int(left) & int(right)
        if isinstance(node.op, ast.BitOr):
            return int(left) | int(right)
        return int(left) ^ int(right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in allowed_functions():
            raise SourceExpressionError("unsupported function")
        if node.keywords:
            raise SourceExpressionError("keyword arguments are not supported")
        return allowed_functions()[node.func.id](*(evaluate_node(arg, env) for arg in node.args))
    if isinstance(node, ast.Compare):
        left = evaluate_node(node.left, env)
        for operator, comparator in zip(node.ops, node.comparators):
            right = evaluate_node(comparator, env)
            if not compare(left, operator, right):
                return False
            left = right
        return True
    raise SourceExpressionError("unsupported expression syntax")


def allowed_functions() -> dict[str, Any]:
    return {**SAFE_MATH_FUNCTIONS, "isfinite": lambda value: math.isfinite(float(value))}


def compare(left: Any, operator: ast.cmpop, right: Any) -> bool:
    if isinstance(operator, ast.Eq):
        return left == right
    if isinstance(operator, ast.NotEq):
        return left != right
    if isinstance(operator, ast.Gt):
        return left > right
    if isinstance(operator, ast.GtE):
        return left >= right
    if isinstance(operator, ast.Lt):
        return left < right
    if isinstance(operator, ast.LtE):
        return left <= right
    raise SourceExpressionError("unsupported comparison operator")


def numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    raise SourceExpressionError(f"non-numeric expression value: {value}")
