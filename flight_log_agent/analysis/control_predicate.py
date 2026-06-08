from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from flight_log_agent.analysis.source_expression import (
    allowed_functions,
    normalize_source_expression,
    source_expression_names,
)
from flight_log_agent.px4.msg_schema import is_valid_topic_field, normalize_px4_enum_value
from flight_log_agent.source_path import resolve_source_path


class LoweredControlPredicate(BaseModel):
    raw: str
    status: Literal["log_verifiable", "internal_unlogged", "unsupported"]
    expression: Optional[str] = None
    variables: dict[str, str] = Field(default_factory=dict)
    unresolved_symbols: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


def lower_control_predicates(
    predicates: list[str],
    symbol_bindings: dict[str, str],
    *,
    source_path: str | Path | None = None,
) -> list[LoweredControlPredicate]:
    return [
        lower_control_predicate(predicate, symbol_bindings, source_path=source_path)
        for predicate in predicates
        if predicate
    ]


def lower_control_predicate(
    predicate: str,
    symbol_bindings: dict[str, str],
    *,
    source_path: str | Path | None = None,
) -> LoweredControlPredicate:
    raw = str(predicate or "").strip()
    if not raw:
        return LoweredControlPredicate(raw=raw, status="unsupported", reason="empty predicate")

    expression = raw.replace("::", "__")
    expression = replace_enum_constants(expression, symbol_bindings.values(), source_path)
    expression = normalize_source_expression(expression)

    variables: dict[str, str] = {}
    used_names: set[str] = set()
    for source, signal in sorted(symbol_bindings.items(), key=lambda item: len(item[0]), reverse=True):
        if not source or source not in expression:
            continue
        name = unique_variable_name(signal, used_names)
        expression = replace_source_symbol(expression, source, name)
        variables[name] = signal

    for token in sorted(set(dotted_symbols(expression)), key=len, reverse=True):
        if not is_valid_topic_field(token, source_path):
            continue
        name = unique_variable_name(token, used_names)
        expression = replace_source_symbol(expression, token, name)
        variables[name] = token

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        call = first_call_name(expression)
        return LoweredControlPredicate(
            raw=raw,
            status="unsupported",
            expression=expression,
            variables=variables,
            unresolved_symbols=unresolved_symbols(expression, variables),
            reason=f"unsupported call {call}" if call else "predicate is not valid evaluator syntax",
        )

    unsupported_call = first_unsupported_call(tree)
    if unsupported_call:
        return LoweredControlPredicate(
            raw=raw,
            status="unsupported",
            expression=expression,
            variables=variables,
            unresolved_symbols=unresolved_symbols(expression, variables),
            reason=f"unsupported call {unsupported_call}",
        )

    unresolved = unresolved_symbols(expression, variables)
    if unresolved:
        return LoweredControlPredicate(
            raw=raw,
            status="internal_unlogged",
            expression=expression,
            variables=variables,
            unresolved_symbols=unresolved,
            reason=f"unresolved source symbols: {unresolved}",
        )

    return LoweredControlPredicate(
        raw=raw,
        status="log_verifiable",
        expression=expression,
        variables=variables,
    )


def replace_enum_constants(expression: str, signals: Any, source_path: str | Path | None) -> str:
    lowered = expression
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:__|::)[A-Z][A-Z0-9_]*\b", lowered))
    tokens.update(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", lowered))
    for token in sorted(tokens, key=len, reverse=True):
        enum_token = token.replace("__", "::")
        value = enum_value(enum_token, signals, source_path)
        if isinstance(value, int):
            lowered = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", str(value), lowered)
    return lowered


def enum_value(token: str, signals: Any, source_path: str | Path | None) -> Any:
    for signal in signals:
        value = normalize_px4_enum_value(str(signal), token, source_path)
        if isinstance(value, int):
            return value
    value = global_constant_value(token, source_path)
    if isinstance(value, int):
        return value
    return token


def global_constant_value(token: str, source_path: str | Path | None) -> Any:
    source_path = resolve_source_path(source_path)
    if source_path is None:
        return token
    root = Path(source_path)
    msg_dir = root / "msg"
    if not msg_dir.exists():
        return token
    constant = token.rsplit("::", 1)[-1]
    pattern = re.compile(
        rf"^\s*[A-Za-z][A-Za-z0-9_]*(?:\[[0-9]*\])?\s+{re.escape(constant)}\s*=\s*(?P<value>-?(?:0x[0-9A-Fa-f]+|\d+))\b"
    )
    values = []
    for path in msg_dir.glob("*.msg"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                values.append(int(match.group("value"), 0))
    unique_values = set(values)
    return values[0] if len(unique_values) == 1 else token


def replace_source_symbol(expression: str, symbol: str, replacement: str) -> str:
    return re.sub(
        rf"(?<![A-Za-z0-9_\.]){re.escape(symbol)}(?![A-Za-z0-9_\.])",
        replacement,
        expression,
    )


def unique_variable_name(signal: str, used_names: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", signal).strip("_") or "value"
    if base[0].isdigit():
        base = f"v_{base}"
    name = base
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


def dotted_symbols(expression: str) -> list[str]:
    return re.findall(
        r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b",
        expression,
    )


def unresolved_symbols(expression: str, variables: dict[str, str]) -> list[str]:
    variable_names = set(variables)
    names = [
        name
        for name in source_expression_names(expression)
        if name not in variable_names
        and name not in {"True", "False"}
    ]
    return list(dict.fromkeys(names))


def first_unsupported_call(tree: ast.AST) -> Optional[str]:
    allowed = set(allowed_functions())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in allowed:
            return getattr(node.func, "id", None) or ast.dump(node.func)
    return None


def first_call_name(expression: str) -> Optional[str]:
    match = re.search(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", expression)
    return match.group("name") if match else None
