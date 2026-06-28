"""Source-derived helper expression substitution.

The PX4 source profiler emits ``HelperExpressionRef`` records for every
helper function it can lower: each record carries the helper's parameter
list, its ``lowered_return_expression`` (for single-return helpers), or
``branches`` (a list of ``{condition, expression}`` pairs the profiler
already normalized into safe-expression form). Helpers whose bodies the
profiler refuses (state mutation, control flow, pointer outputs) carry
a non-empty ``unresolved_reason``.

This module turns those records into runtime substitutions:

- :func:`substitute_helpers` rewrites every ``helper_name(args...)`` call
  in an expression by inlining the lowered body, with the helper's formal
  parameters replaced by the actual call arguments.
- :func:`lower_helper_call` is the single-call entry point. For branched
  helpers it evaluates each ``condition`` against the caller-provided
  environment and picks the first match. For single-return helpers it
  inlines ``lowered_return_expression`` directly.

The lowering is fully generic: there are no hardcoded helper names,
formulas, or domain-specific shortcuts. If the profiler can lower a
helper from PX4 source the substitution works automatically; if it
cannot, :func:`lower_helper_call` returns ``None`` and the caller falls
back to whatever pre-helper behaviour it already had.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from flight_log_agent.analysis.source_expression import (
    SourceExpressionError,
    evaluate_source_expression,
    normalize_source_expression,
)


class HelperResolutionError(ValueError):
    pass


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NAMESPACE_SEPARATOR_RE = re.compile(r"::|\.")


def _short_name(name: str) -> str:
    """Return the last component of a possibly namespaced helper name."""
    return _NAMESPACE_SEPARATOR_RE.split(name.strip())[-1] if name else ""


class HelperRegistry:
    """Index of :class:`HelperExpressionRef`-shaped dicts by helper name.

    Helpers can be referenced either by their fully-qualified name
    (``Navigator::get_acceptance_radius``) or by their short name
    (``get_acceptance_radius``). Both forms resolve to the same record.
    When two refs share a short name they remain accessible by their
    fully-qualified names; the short-name lookup returns ``None`` so the
    caller can decide whether to be more specific.
    """

    def __init__(self, refs: Iterable[dict[str, Any]] = ()) -> None:
        self._by_full_name: dict[str, dict[str, Any]] = {}
        self._by_short_name: dict[str, list[dict[str, Any]]] = {}
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            name = str(ref.get("name") or "").strip()
            if not name:
                continue
            self._by_full_name[name] = ref
            short = _short_name(name)
            if short and short != name:
                self._by_short_name.setdefault(short, []).append(ref)

    @classmethod
    def from_index(
        cls,
        binding_index: Any,
        helper_expressions: Iterable[Any] = (),
    ) -> "HelperRegistry":
        """Build a registry whose helper bodies are the index-resolved forms.

        For each helper, if ``binding_index.helper_resolutions`` has an
        :class:`ExpressionSliceResult` for it, replace the helper's
        ``lowered_return_expression`` with the resolved (terminal-form)
        expression. ``substitute_helpers`` therefore inlines that form
        in one pass instead of re-expanding nested bodies.
        """
        resolutions = getattr(binding_index, "helper_resolutions", {}) or {}
        records: list[dict[str, Any]] = []
        for helper in helper_expressions:
            if isinstance(helper, dict):
                record = dict(helper)
            elif hasattr(helper, "model_dump"):
                record = helper.model_dump(exclude_none=True)
            else:
                record = dict(vars(helper))
            name = str(record.get("name") or "")
            resolution = resolutions.get(name)
            if resolution is None and "::" in name:
                resolution = resolutions.get(name.split("::")[-1])
            resolved_expression = (
                getattr(resolution, "expression", None) if resolution is not None else None
            )
            if isinstance(resolved_expression, str) and resolved_expression:
                record["lowered_return_expression"] = resolved_expression
            records.append(record)
        return cls(records)

    def get(self, name: str) -> Optional[dict[str, Any]]:
        if not isinstance(name, str):
            return None
        cleaned = name.strip()
        if not cleaned:
            return None
        if cleaned in self._by_full_name:
            return self._by_full_name[cleaned]
        candidates = self._by_short_name.get(_short_name(cleaned), [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None


def lower_helper_call(
    name: str,
    args: list[str],
    *,
    registry: HelperRegistry,
    env: dict[str, Any],
) -> Optional[str]:
    """Return the lowered expression for a helper call, or ``None``.

    The result is a string in safe-expression form with the helper's
    formal parameters textually substituted with ``args``. For branched
    helpers the condition is evaluated against ``env`` (parameters and
    bound signal values); the first matching branch's expression is
    used. Helpers that the profiler marked unresolved, or branched
    helpers whose conditions all fail, return ``None``.
    """
    ref = registry.get(name)
    if ref is None or ref.get("unresolved_reason"):
        return None

    parameters = [str(p) for p in (ref.get("parameters") or []) if p]
    if len(args) != len(parameters):
        return None

    branches = ref.get("branches") or []
    if branches:
        for branch in branches:
            condition = str(branch.get("condition") or "").strip()
            expression = str(branch.get("expression") or "").strip()
            if not expression:
                continue
            if not condition or condition == "default":
                return _substitute_parameters(expression, parameters, args)
            if _condition_holds(condition, env):
                return _substitute_parameters(expression, parameters, args)
        return None

    body = ref.get("lowered_return_expression") or ref.get("return_expression")
    if not body:
        return None
    return _substitute_parameters(str(body), parameters, args)


def substitute_helpers(
    expression: str,
    *,
    registry: HelperRegistry,
    env: dict[str, Any],
    max_passes: int = 4,
) -> str:
    """Rewrite all helper calls in ``expression`` using ``registry``.

    Substitution runs iteratively (bounded by ``max_passes``) so that a
    helper whose body itself contains another helper call resolves in
    successive passes. The output expression is returned even if some
    helper calls could not be lowered; the caller is responsible for
    deciding what to do with un-substituted calls (typically: fail the
    check with the existing "cannot evaluate helper X" message).
    """
    if not isinstance(expression, str) or not expression:
        return expression

    current = expression
    for _ in range(max(1, max_passes)):
        rewritten, changed = _substitute_helpers_once(current, registry, env)
        if not changed:
            return rewritten
        current = rewritten
    return current


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _substitute_helpers_once(
    expression: str,
    registry: HelperRegistry,
    env: dict[str, Any],
) -> tuple[str, bool]:
    result_parts: list[str] = []
    cursor = 0
    changed = False
    for match in _iter_function_calls(expression):
        name = match["name"]
        start = match["start"]
        end_args = match["end"]
        if name not in registry:
            continue
        args = _split_top_level_arguments(expression[match["args_start"]:match["args_end"]])
        lowered = lower_helper_call(name, args, registry=registry, env=env)
        if lowered is None:
            continue
        result_parts.append(expression[cursor:start])
        result_parts.append(f"({lowered})")
        cursor = end_args
        changed = True
    if not changed:
        return expression, False
    result_parts.append(expression[cursor:])
    return "".join(result_parts), True


def _iter_function_calls(expression: str) -> Iterable[dict[str, Any]]:
    """Yield ``{name, start, end, args_start, args_end}`` for each call.

    Each yielded entry matches ``IDENTIFIER(...)`` where the identifier
    may contain ``::`` (PX4 namespace), ``.`` (member access) or ``->``
    (pointer member access) prefixes. Nested parens inside the args are
    tracked so the args_end is the matching close paren.
    """
    pattern = re.compile(
        r"(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*(?:(?:::|->|\.)\s*))*[A-Za-z_][A-Za-z0-9_]*)\s*\("
    )
    for match in pattern.finditer(expression):
        args_start = match.end()
        args_end = _find_matching_paren(expression, args_start - 1)
        if args_end is None:
            continue
        yield {
            "name": re.sub(r"\s+", "", match.group("name")),
            "start": match.start(),
            "end": args_end + 1,
            "args_start": args_start,
            "args_end": args_end,
        }


def _find_matching_paren(expression: str, open_paren_index: int) -> Optional[int]:
    if open_paren_index >= len(expression) or expression[open_paren_index] != "(":
        return None
    depth = 1
    for i in range(open_paren_index + 1, len(expression)):
        char = expression[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _split_top_level_arguments(args: str) -> list[str]:
    if not args.strip():
        return []
    pieces: list[str] = []
    depth = 0
    last = 0
    for i, char in enumerate(args):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append(args[last:i].strip())
            last = i + 1
    pieces.append(args[last:].strip())
    return [piece for piece in pieces if piece]


def _substitute_parameters(
    body: str,
    parameters: list[str],
    args: list[str],
) -> str:
    """Replace formal-parameter identifiers in ``body`` with actual args.

    Parameters are processed longest-first so a parameter named ``lat``
    isn't substituted inside one named ``latitude``. Each argument is
    parenthesized to preserve operator precedence in the surrounding
    expression.
    """
    if not parameters:
        return body
    substituted = body
    indexed = sorted(
        enumerate(parameters),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for index, parameter in indexed:
        if not parameter:
            continue
        replacement = f"({args[index]})"
        substituted = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(parameter)}(?![A-Za-z0-9_])",
            replacement,
            substituted,
        )
    return substituted


def _condition_holds(condition: str, env: dict[str, Any]) -> bool:
    """Return True if ``condition`` evaluates truthy against ``env``.

    Conditions that fail to parse or reference variables missing from
    ``env`` count as not-holding rather than raising — this lets the
    caller try the next branch.
    """
    try:
        value = evaluate_source_expression(normalize_source_expression(condition), env)
    except (SourceExpressionError, KeyError, TypeError, ValueError):
        return False
    return bool(value)
