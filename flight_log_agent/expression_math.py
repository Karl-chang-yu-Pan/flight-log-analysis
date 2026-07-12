from __future__ import annotations

import math
import re
from typing import Callable


def constrain(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


SAFE_MATH_FUNCTIONS: dict[str, Callable] = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_") and callable(getattr(math, name))
}
SAFE_MATH_FUNCTIONS.update(
    {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "constrain": constrain,
    }
)


def canonical_math_function_name(name: str) -> str:
    short_name = name
    if "::" in name:
        namespace, candidate = name.rsplit("::", 1)
        if namespace in {"math", "std"}:
            short_name = candidate
    if short_name.startswith("PX4_"):
        # PX4 macro spellings of stdlib math (``PX4_ISFINITE``) — the
        # same shape-aliasing as the namespace and ``f``-suffix rules.
        lowered = short_name[4:].lower()
        if lowered in SAFE_MATH_FUNCTIONS:
            return lowered
    if short_name in {"fmin", "fminf"}:
        return "min"
    if short_name in {"fmax", "fmaxf"}:
        return "max"
    if short_name.endswith("f") and short_name[:-1] in SAFE_MATH_FUNCTIONS:
        return short_name[:-1]
    return short_name


def is_safe_math_function_name(name: str) -> bool:
    return canonical_math_function_name(name) in SAFE_MATH_FUNCTIONS


def normalize_expression_function_names(expression: str) -> str:
    return re.sub(
        r"(?<![A-Za-z0-9_:])(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*)\s*(?=\()",
        lambda match: canonical_math_function_name(match.group("name"))
        if is_safe_math_function_name(match.group("name"))
        else match.group("name"),
        expression,
    )
