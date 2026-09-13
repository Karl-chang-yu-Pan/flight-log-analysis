from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cache
import math
import re
import io
import tokenize
from collections.abc import Callable, Mapping
from typing import Any, Iterable

from flight_log_agent.expression_math import SAFE_MATH_FUNCTIONS, normalize_expression_function_names


ALLOWED_EXPRESSION_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.IfExp,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Call,
    ast.Compare,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
)


class SourceExpressionError(ValueError):
    pass


def alias_dotted_names(expression: str, names: Iterable[str]) -> tuple[str, dict[str, str]]:
    """Replace occurrences of ``names`` in ``expression`` with safe flat aliases.

    Returns ``(rewritten_expression, alias_to_name)``. Names are processed
    longest-first so that ``topic.field.subfield`` is rewritten before
    ``topic.field``. Empty names and names that don't appear are skipped.

    This is the single place where dotted (``topic.field``) and bracketed
    (``q[0]``) references are folded into flat identifiers before AST parsing,
    so downstream validators and evaluators can rely on a plain ``Name``
    instead of handling ``Attribute`` / ``Subscript`` directly.
    """
    rewritten = expression
    alias_to_name: dict[str, str] = {}
    sorted_names = sorted(
        {name for name in names if name}, key=lambda name: (-len(name), name)
    )
    for index, name in enumerate(sorted_names):
        alias = f"__alias_{index}"
        new_rewritten = re.sub(
            rf"(?<![A-Za-z0-9_.]){re.escape(name)}(?![A-Za-z0-9_.])",
            alias,
            rewritten,
        )
        if new_rewritten != rewritten:
            alias_to_name[alias] = name
            rewritten = new_rewritten
    return rewritten, alias_to_name


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
    return list(_source_expression_names_cached(str(expression or "")))


@cache
def _source_expression_names_cached(expression: str) -> tuple[str, ...]:
    """Parse one immutable source expression once within this process."""
    normalized = normalize_source_expression(expression)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return ()

    parents: dict[ast.AST, ast.AST] = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def reference_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = reference_name(node.value)
            return f"{base}.{node.attr}" if base else None
        if isinstance(node, ast.Subscript):
            base = reference_name(node.value)
            if not base:
                return None
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                return f"{base}[{node.slice.value}]"
            # A dynamic index reads the aggregate value; the index expression
            # is walked independently and remains another dependency.
            return base
        return None

    def is_reference_child(node: ast.AST) -> bool:
        parent = parents.get(node)
        return bool(
            (isinstance(parent, ast.Attribute) and parent.value is node)
            or (isinstance(parent, ast.Subscript) and parent.value is node)
        )

    names: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            continue
        if is_reference_child(node):
            continue
        name = reference_name(node)
        if name:
            names.append(
                (getattr(node, "lineno", 0), getattr(node, "col_offset", 0), name)
            )
    function_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return tuple(
        dict.fromkeys(
            name for _, _, name in sorted(names) if name not in function_names
        )
    )


@dataclass(frozen=True)
class CompiledSourceExpression:
    """One normalized local expression with stable operand aliases."""

    source: str
    normalized: str
    tree: ast.Expression
    alias_to_operand: tuple[tuple[str, str], ...]

    @property
    def operands(self) -> tuple[str, ...]:
        return tuple(operand for _alias, operand in self.alias_to_operand)

    def evaluate(self, resolve_operand: Callable[[str], Any]) -> Any:
        return evaluate_node(
            self.tree,
            _LazyOperandEnvironment(dict(self.alias_to_operand), resolve_operand),
        )


class _LazyOperandEnvironment(Mapping[str, Any]):
    """Resolve an operand only if short-circuit evaluation reaches it."""

    def __init__(
        self,
        aliases: dict[str, str],
        resolver: Callable[[str], Any],
    ) -> None:
        self._aliases = aliases
        self._resolver = resolver
        self._values: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self._aliases:
            raise KeyError(key)
        if key not in self._values:
            self._values[key] = self._resolver(self._aliases[key])
        return self._values[key]

    def __iter__(self):
        return iter(self._aliases)

    def __len__(self) -> int:
        return len(self._aliases)


def compile_source_expression(
    expression: str,
    operand_names: Iterable[str],
    *,
    occurrence_operands: Iterable[tuple[str, str]] = (),
) -> CompiledSourceExpression:
    """Compile one source expression against parser/DAG-proven operands."""
    normalized = normalize_source_expression(expression)
    occurrences = list(occurrence_operands)
    # A graph-bound C++ call may contain syntax Python cannot parse (such as
    # an address argument). Tokenize these exact, parser-proven occurrences
    # before AST parsing; never reinterpret the address as a numeric value.
    opaque: dict[tuple[tuple[int, str], ...], deque[tuple[str, str]]] = defaultdict(deque)
    remaining = []
    occurrence_aliases: dict[str, str] = {}

    def tokens(text: str) -> list[tuple[int, str]]:
        try:
            return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline)
                    if t.type not in {tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}]
        except (tokenize.TokenError, IndentationError) as exc:
            raise SourceExpressionError("invalid parser-proven call operand") from exc

    for index, (operand_id, operand_expression) in enumerate(occurrences):
        spelling = normalize_source_expression(operand_expression)
        try:
            ast.parse(spelling, mode="eval")
        except SyntaxError:
            opaque[tuple(tokens(spelling))].append((f"__source_opaque_call_{index}", operand_id))
        else:
            remaining.append((operand_id, operand_expression))
    if opaque:
        source_tokens = tokens(normalized)
        lowered = []
        position = 0
        patterns = sorted(opaque, key=len, reverse=True)
        while position < len(source_tokens):
            match = next((pattern for pattern in patterns if pattern and opaque[pattern]
                          and tuple(source_tokens[position:position + len(pattern)]) == pattern
                          and (position == 0 or source_tokens[position - 1][1] != ".")), None)
            if match is None:
                lowered.append(source_tokens[position])
                position += 1
            else:
                token, operand_id = opaque[match].popleft()
                lowered.append((tokenize.NAME, token))
                occurrence_aliases[token] = operand_id
                position += len(match)
        normalized = tokenize.untokenize(lowered).strip()
    try:
        source_tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise SourceExpressionError("invalid expression syntax") from exc

    occurrence_queues: dict[str, deque[tuple[str, str]]] = defaultdict(deque)
    for index, (operand_id, operand_expression) in enumerate(
        remaining
    ):
        try:
            operand_tree = ast.parse(
                normalize_source_expression(operand_expression), mode="eval"
            )
        except SyntaxError as exc:
            raise SourceExpressionError(
                "invalid parser-proven call operand"
            ) from exc
        token = f"__source_call_operand_{index}"
        occurrence_queues[
            ast.dump(operand_tree.body, include_attributes=False)
        ].append((token, operand_id))

    class ReplaceOccurrenceOperands(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            queue = occurrence_queues.get(
                ast.dump(node, include_attributes=False)
            )
            if queue:
                token, operand_id = queue.popleft()
                occurrence_aliases[token] = operand_id
                return ast.copy_location(ast.Name(id=token, ctx=ast.Load()), node)
            return super().visit(node)

    transformed = ReplaceOccurrenceOperands().visit(source_tree)
    ast.fix_missing_locations(transformed)
    transformed_text = ast.unparse(transformed)
    rewritten, alias_to_name = alias_dotted_names(
        transformed_text, operand_names
    )
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        raise SourceExpressionError("invalid expression syntax") from exc
    aliases = {**occurrence_aliases, **alias_to_name}
    return CompiledSourceExpression(
        source=str(expression or ""),
        normalized=normalized,
        tree=tree,
        alias_to_operand=tuple(aliases.items()),
    )


def evaluate_source_expression(expression: str, env: dict[str, Any]) -> Any:
    compiled = compile_source_expression(expression, env.keys())
    return compiled.evaluate(lambda operand: env[operand])


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


def evaluate_node(node: ast.AST, env: Mapping[str, Any]) -> Any:
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
