"""Backward source-symbol slicer.

Given an unbound dotted symbol like ``_destination.lat`` and the source
assignments the profiler already emits (``target = expression``), slice
backward through writes until reaching a logged signal, a parameter, or
a dead end. The slicer is the complement to :class:`HelperRegistry`:
helper substitution rewrites helper calls inside an expression; this
slicer resolves the still-unbound symbols that remain *inside* a
substituted expression.

The shape mirrors :func:`verification_graph.backward_binding_slice`'s
frontier/seen pattern, plus per-call ``_path`` tracking so dependency
cycles (``A`` depends on ``B`` which depends on ``A``) are detected and
reported rather than silently terminating via visitation.

Partial-substitution policy (per the design discussion): when a symbol
has multiple write sites and only *some* resolve to logged signals or
parameters, the slicer returns ``unbindable`` *with* a partial expression
that inlines the resolved branches and leaves the unresolved ones as
the original symbol. That gives the caller both a more-useful expression
*and* a precise list of holdouts.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Literal, Optional

from pydantic import BaseModel, Field

from flight_log_agent.analysis.source_expression import source_expression_names
from flight_log_agent.symbols import normalize_symbol


BlockerKind = Literal[
    "cycle",
    "no_write_site",
    "external_call",
    "ambiguous_writes",
    "unsupported_rhs",
]


class SliceBlocker(BaseModel):
    kind: BlockerKind
    symbol: str
    detail: str
    cycle_path: list[str] = Field(default_factory=list)
    # Partial expression assembled before the slicer gave up; useful for
    # the caller's "we got this far" diagnostic.
    partial_expression: Optional[str] = None


class ConditionalSlice(BaseModel):
    condition: str
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    branch_result: "SliceResult"


class SliceResult(BaseModel):
    status: Literal["resolved", "conditional", "cycle", "unbindable"]
    expression: Optional[str] = None
    conditional_branches: list[ConditionalSlice] = Field(default_factory=list)
    blocker: Optional[SliceBlocker] = None
    trace: list[str] = Field(default_factory=list)


ConditionalSlice.model_rebuild()


class ExpressionSliceResult(BaseModel):
    expression: str
    unresolved: list[SliceResult] = Field(default_factory=list)
    fully_resolved: bool


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def slice_symbol(
    symbol: str,
    *,
    source_assignments: Iterable[Any],
    logged_signals: Iterable[str] = (),
    parameters: Iterable[str] = (),
    binding_index: Optional[Any] = None,
    _path: tuple[str, ...] = (),
) -> SliceResult:
    """Backward-slice ``symbol`` to logged signals / parameters / dead end.

    When ``binding_index`` is provided (typically the
    :class:`BindingIndex` already built by ``compile_verification_plan``),
    symbols whose alias resolves to a single logged signal are returned
    immediately without walking ``source_assignments``. Multi-target and
    absent symbols fall through to the assignment walk.
    """
    logged_set = _frozen(logged_signals)
    parameter_set = _frozen(parameters)
    assignments = list(source_assignments)
    index = _build_assignment_index(assignments)
    memo: dict[str, SliceResult] = {}
    return _slice_symbol(
        symbol,
        index=index,
        logged_set=logged_set,
        parameter_set=parameter_set,
        path=_path,
        memo=memo,
        binding_index=binding_index,
    )


def slice_expression(
    expression: str,
    *,
    source_assignments: Iterable[Any],
    logged_signals: Iterable[str] = (),
    parameters: Iterable[str] = (),
    binding_index: Optional[Any] = None,
) -> ExpressionSliceResult:
    """Walk every unbound dotted symbol in ``expression`` and substitute.

    ``binding_index`` short-circuits symbols already mapped by the
    resolver-produced :class:`BindingIndex`, eliminating the redundant
    backward walk for the common case where a symbol has a single
    unambiguous logged-signal alias.
    """
    logged_set = _frozen(logged_signals)
    parameter_set = _frozen(parameters)
    assignments = list(source_assignments)
    index = _build_assignment_index(assignments)
    memo: dict[str, SliceResult] = {}

    names = _expression_symbols(expression)
    unresolved: list[SliceResult] = []
    substituted = expression

    for name in names:
        canonical = normalize_symbol(name)
        if not canonical:
            continue
        if canonical in logged_set or canonical in parameter_set:
            continue
        if "." not in canonical and "[" not in canonical:
            # Non-dotted bare identifiers (locals, formal params, math
            # function names) are out of scope for the slicer.
            continue
        result = _slice_symbol(
            canonical,
            index=index,
            logged_set=logged_set,
            parameter_set=parameter_set,
            path=(),
            memo=memo,
            binding_index=binding_index,
        )
        if result.status == "resolved" and result.expression is not None:
            substituted = _substitute_symbol(substituted, name, result.expression)
            continue
        if result.status == "conditional":
            ternary = _conditional_to_ternary(result.conditional_branches, fallback=name)
            if ternary is not None:
                substituted = _substitute_symbol(substituted, name, ternary)
                if not _all_branches_resolved(result.conditional_branches):
                    unresolved.append(result)
                continue
        if result.status == "unbindable" and result.blocker and result.blocker.partial_expression:
            substituted = _substitute_symbol(substituted, name, result.blocker.partial_expression)
        unresolved.append(result)

    return ExpressionSliceResult(
        expression=substituted,
        unresolved=unresolved,
        fully_resolved=not unresolved,
    )


# ----------------------------------------------------------------------
# Core slicing logic
# ----------------------------------------------------------------------


def _slice_symbol(
    symbol: str,
    *,
    index: dict[str, list[Any]],
    logged_set: frozenset[str],
    parameter_set: frozenset[str],
    path: tuple[str, ...],
    memo: dict[str, SliceResult],
    binding_index: Optional[Any] = None,
) -> SliceResult:
    canonical = normalize_symbol(symbol)
    if not canonical:
        return _unbindable(symbol, "no_write_site", "empty symbol", path)

    if canonical in logged_set or canonical in parameter_set:
        return SliceResult(
            status="resolved",
            expression=canonical,
            trace=list(path) + [canonical],
        )

    if canonical in path:
        cycle_path = list(path) + [canonical]
        return SliceResult(
            status="cycle",
            blocker=SliceBlocker(
                kind="cycle",
                symbol=canonical,
                detail=" -> ".join(cycle_path),
                cycle_path=cycle_path,
            ),
            trace=cycle_path,
        )

    cached = memo.get(canonical)
    if cached is not None:
        return cached

    binding_result = _resolve_via_binding_index(canonical, binding_index, path)
    if binding_result is not None:
        memo[canonical] = binding_result
        return binding_result

    assignment_result = _resolve_via_assignment_resolutions(canonical, binding_index, path)
    if assignment_result is not None:
        memo[canonical] = assignment_result
        return assignment_result

    writes = _find_writes(canonical, index)
    if not writes:
        result = _unbindable(canonical, "no_write_site", f"no assignment writes to {canonical}", path)
        memo[canonical] = result
        return result

    new_path = path + (canonical,)

    # Single write — no conditional, just slice the RHS.
    if len(writes) == 1:
        return _slice_single_write(
            canonical,
            writes[0],
            index=index,
            logged_set=logged_set,
            parameter_set=parameter_set,
            path=new_path,
            memo=memo,
            binding_index=binding_index,
        )

    # Multiple writes — see whether they're distinguishable via control
    # predicates. If yes, return a conditional with one branch per write;
    # if no, ambiguous.
    distinguishable = _writes_have_distinct_predicates(writes)
    if not distinguishable:
        return SliceResult(
            status="unbindable",
            blocker=SliceBlocker(
                kind="ambiguous_writes",
                symbol=canonical,
                detail=(
                    f"{len(writes)} write sites for {canonical} with overlapping or no "
                    "control predicates: " + "; ".join(_describe_write(w) for w in writes)
                ),
            ),
            trace=list(new_path),
        )

    branches: list[ConditionalSlice] = []
    for write in writes:
        condition = _condition_from_write(write)
        sub_result = _slice_expression_string(
            _get(write, "expression"),
            index=index,
            logged_set=logged_set,
            parameter_set=parameter_set,
            path=new_path,
            memo=memo,
            binding_index=binding_index,
        )
        branches.append(
            ConditionalSlice(
                condition=condition,
                source_file=_get(write, "file"),
                source_line=_get(write, "line"),
                branch_result=sub_result,
            )
        )

    if all(branch.branch_result.status == "resolved" for branch in branches):
        return SliceResult(
            status="conditional",
            conditional_branches=branches,
            trace=list(new_path),
        )

    # Partial-substitution case: at least one branch failed. We still
    # return the conditional structure so the caller can inline the
    # resolved ones and report holdouts.
    if any(branch.branch_result.status == "resolved" for branch in branches):
        return SliceResult(
            status="conditional",
            conditional_branches=branches,
            blocker=SliceBlocker(
                kind="ambiguous_writes",
                symbol=canonical,
                detail=(
                    "some branches resolved, others did not: "
                    + "; ".join(_describe_branch_status(b) for b in branches)
                ),
                partial_expression=_conditional_to_ternary(branches, fallback=canonical),
            ),
            trace=list(new_path),
        )

    # No branch resolved.
    return SliceResult(
        status="unbindable",
        blocker=SliceBlocker(
            kind="ambiguous_writes",
            symbol=canonical,
            detail="no branch resolved; "
            + "; ".join(_describe_branch_status(b) for b in branches),
        ),
        trace=list(new_path),
        conditional_branches=branches,
    )


def _slice_single_write(
    symbol: str,
    write: Any,
    *,
    index: dict[str, list[Any]],
    logged_set: frozenset[str],
    parameter_set: frozenset[str],
    path: tuple[str, ...],
    memo: dict[str, SliceResult],
    binding_index: Optional[Any] = None,
) -> SliceResult:
    rhs = _get(write, "expression") or ""
    sub_result = _slice_expression_string(
        rhs,
        index=index,
        logged_set=logged_set,
        parameter_set=parameter_set,
        path=path,
        memo=memo,
        binding_index=binding_index,
    )
    if _get(write, "control_predicates"):
        condition = _condition_from_write(write)
        branches = [
            ConditionalSlice(
                condition=condition,
                source_file=_get(write, "file"),
                source_line=_get(write, "line"),
                branch_result=sub_result,
            )
        ]
        if sub_result.status == "resolved":
            return SliceResult(
                status="conditional",
                conditional_branches=branches,
                trace=list(path),
            )
        return SliceResult(
            status="unbindable",
            blocker=SliceBlocker(
                kind="ambiguous_writes",
                symbol=symbol,
                detail=f"only branch ({condition}) failed to resolve",
                partial_expression=_conditional_to_ternary(branches, fallback=symbol),
            ),
            conditional_branches=branches,
            trace=list(path),
        )
    return sub_result


def _slice_expression_string(
    expression: str,
    *,
    index: dict[str, list[Any]],
    logged_set: frozenset[str],
    parameter_set: frozenset[str],
    path: tuple[str, ...],
    memo: dict[str, SliceResult],
    binding_index: Optional[Any] = None,
) -> SliceResult:
    """Slice every dotted symbol in ``expression``; substitute resolved ones."""
    if not expression or not isinstance(expression, str):
        return SliceResult(status="unbindable", blocker=SliceBlocker(
            kind="unsupported_rhs",
            symbol=expression or "",
            detail="empty RHS expression",
        ), trace=list(path))

    if _expression_has_unsupported_call(expression):
        return SliceResult(
            status="unbindable",
            blocker=SliceBlocker(
                kind="external_call",
                symbol=expression,
                detail=f"expression contains an external call: {expression}",
            ),
            trace=list(path),
        )

    names = _expression_symbols(expression)
    substituted = expression
    partials: list[str] = []
    for name in names:
        canonical = normalize_symbol(name)
        if not canonical:
            continue
        if canonical in logged_set or canonical in parameter_set:
            continue
        if "." not in canonical and "[" not in canonical:
            continue
        sub = _slice_symbol(
            canonical,
            index=index,
            logged_set=logged_set,
            parameter_set=parameter_set,
            path=path,
            memo=memo,
            binding_index=binding_index,
        )
        if sub.status == "resolved" and sub.expression is not None:
            substituted = _substitute_symbol(substituted, name, sub.expression)
            continue
        if sub.status == "conditional":
            ternary = _conditional_to_ternary(sub.conditional_branches, fallback=name)
            if ternary is not None:
                substituted = _substitute_symbol(substituted, name, ternary)
                if not _all_branches_resolved(sub.conditional_branches):
                    partials.append(_describe_slice_blocker(sub))
                continue
        if sub.status == "cycle":
            return SliceResult(
                status="cycle",
                blocker=sub.blocker,
                trace=sub.trace,
            )
        if sub.blocker and sub.blocker.partial_expression:
            substituted = _substitute_symbol(substituted, name, sub.blocker.partial_expression)
        partials.append(_describe_slice_blocker(sub))

    if not partials and _all_symbols_resolved(substituted, logged_set, parameter_set):
        return SliceResult(
            status="resolved",
            expression=substituted,
            trace=list(path),
        )

    return SliceResult(
        status="unbindable",
        blocker=SliceBlocker(
            kind="unsupported_rhs",
            symbol=expression,
            detail="; ".join(partials) if partials else f"could not resolve all symbols in {expression}",
            partial_expression=substituted,
        ),
        trace=list(path),
    )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


# Identifiers we never try to slice — these are the safe-math /
# language-level names the expression evaluator already knows.
_BUILTIN_NAMES: frozenset[str] = frozenset({
    "True", "False",
    "abs", "min", "max", "round", "constrain", "isfinite",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sqrt", "fabs", "fmin", "fmax", "exp", "log", "pow",
})

_FUNCTION_CALL_RE = re.compile(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _unbindable(
    symbol: str,
    kind: BlockerKind,
    detail: str,
    path: tuple[str, ...],
) -> SliceResult:
    return SliceResult(
        status="unbindable",
        blocker=SliceBlocker(kind=kind, symbol=symbol, detail=detail),
        trace=list(path) + [symbol],
    )


def _frozen(values: Iterable[str]) -> frozenset[str]:
    return frozenset(normalize_symbol(v) for v in values if v)


def _resolve_via_binding_index(
    canonical: str,
    binding_index: Optional[Any],
    path: tuple[str, ...],
) -> Optional[SliceResult]:
    """Return a resolved SliceResult when ``binding_index`` has a single alias.

    Reuses the resolver's already-completed backward slice so the slicer
    does not redo the work for symbols the resolver materialized as
    output bindings. Multi-target aliases and absent symbols fall back to
    the assignment walk.
    """
    if binding_index is None:
        return None
    aliases = getattr(binding_index, "aliases", None)
    if not isinstance(aliases, dict):
        return None
    targets = aliases.get(canonical)
    if not targets or len(targets) != 1:
        return None
    logged = next(iter(targets))
    if not isinstance(logged, str) or not logged:
        return None
    return SliceResult(
        status="resolved",
        expression=logged,
        trace=list(path) + [canonical, logged],
    )


def _resolve_via_assignment_resolutions(
    canonical: str,
    binding_index: Optional[Any],
    path: tuple[str, ...],
) -> Optional[SliceResult]:
    """Return a SliceResult derived from a pre-materialized assignment resolution.

    BindingIndex materializes each assignment's RHS into an
    ExpressionSliceResult at construction time. When the canonical
    symbol matches a materialized target, return that result as a single
    SliceResult so the recursive walk does not redo the work.
    """
    if binding_index is None:
        return None
    lookup = getattr(binding_index, "assignment_resolution_for", None)
    pre = lookup(canonical) if callable(lookup) else None
    if pre is None:
        return None
    expression = getattr(pre, "expression", None)
    if not isinstance(expression, str) or not expression:
        return None
    fully = bool(getattr(pre, "fully_resolved", False))
    if fully:
        return SliceResult(
            status="resolved",
            expression=expression,
            trace=list(path) + [canonical],
        )
    return SliceResult(
        status="unbindable",
        blocker=SliceBlocker(
            kind="unsupported_rhs",
            symbol=canonical,
            detail=f"partial pre-materialized resolution for {canonical}",
            partial_expression=expression,
        ),
        trace=list(path) + [canonical],
    )


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _build_assignment_index(assignments: list[Any]) -> dict[str, list[Any]]:
    """Group assignments by normalized ``target`` for O(1) lookup."""
    index: dict[str, list[Any]] = {}
    for assignment in assignments:
        target = _get(assignment, "target")
        if not target:
            continue
        key = normalize_symbol(str(target))
        if not key:
            continue
        index.setdefault(key, []).append(assignment)
    return index


def _find_writes(symbol: str, index: dict[str, list[Any]]) -> list[Any]:
    return index.get(symbol, [])


def _writes_have_distinct_predicates(writes: list[Any]) -> bool:
    seen: set[tuple[str, ...]] = set()
    for write in writes:
        predicates = tuple(_get(write, "control_predicates") or [])
        if predicates in seen:
            return False
        if not predicates and len(writes) > 1:
            # An unconditional write among multiple writes makes them
            # ambiguous — we cannot reliably pick when this one wins.
            # (A single unconditional write is handled before this
            # function is called.)
            return False
        seen.add(predicates)
    return True


def _condition_from_write(write: Any) -> str:
    predicates = _get(write, "control_predicates") or []
    if not predicates:
        return "default"
    return " and ".join(f"({p})" for p in predicates)


def _expression_symbols(expression: str) -> list[str]:
    """The *longest* dotted symbols in ``expression``.

    ``source_expression_names`` walks every attribute node, which means a
    reference like ``topic.sub.field`` emits both ``topic.sub.field`` and
    its prefix ``topic.sub``. For slicing we only want the leaf —
    resolving the longer path implicitly resolves the prefix, and trying
    to slice the prefix on its own would dead-end at no_write_site.
    """
    names = source_expression_names(expression)
    longest = []
    sorted_names = sorted(set(names), key=len, reverse=True)
    for name in sorted_names:
        if any(other.startswith(name + ".") for other in longest):
            continue
        longest.append(name)
    # Preserve original ordering for downstream substitution stability.
    return [name for name in names if name in set(longest)]


def _substitute_symbol(expression: str, symbol: str, replacement: str) -> str:
    """Replace bare occurrences of ``symbol`` with ``(replacement)``."""
    escaped = re.escape(symbol)
    pattern = re.compile(rf"(?<![A-Za-z0-9_.]){escaped}(?![A-Za-z0-9_.])")
    return pattern.sub(f"({replacement})", expression)


def _conditional_to_ternary(
    branches: list[ConditionalSlice],
    *,
    fallback: str,
) -> Optional[str]:
    """Build a nested ternary from ``branches``.

    Resolved branches contribute their expression. Unresolved branches
    contribute the original symbol (``fallback``) so the resulting
    expression is still well-formed; the caller is expected to surface
    those holdouts separately.
    """
    if not branches:
        return None
    pieces: list[tuple[str, str]] = []
    default_expression: Optional[str] = None
    for branch in branches:
        result = branch.branch_result
        if result.status == "resolved" and result.expression is not None:
            value = result.expression
        elif result.blocker and result.blocker.partial_expression:
            value = result.blocker.partial_expression
        else:
            value = fallback
        if branch.condition == "default":
            default_expression = value
        else:
            pieces.append((branch.condition, value))
    if not pieces and default_expression is None:
        return None
    expression = default_expression if default_expression is not None else fallback
    for condition, value in reversed(pieces):
        expression = f"({value} if {condition} else {expression})"
    return expression


def _all_branches_resolved(branches: list[ConditionalSlice]) -> bool:
    return all(branch.branch_result.status == "resolved" for branch in branches)


def _all_symbols_resolved(
    expression: str,
    logged_set: frozenset[str],
    parameter_set: frozenset[str],
) -> bool:
    for name in _expression_symbols(expression):
        canonical = normalize_symbol(name)
        if not canonical:
            continue
        if "." not in canonical and "[" not in canonical:
            continue
        if canonical in logged_set or canonical in parameter_set:
            continue
        return False
    return True


def _expression_has_unsupported_call(expression: str) -> bool:
    """Detect non-safe-math function calls in ``expression``."""
    for match in _FUNCTION_CALL_RE.finditer(expression):
        name = match.group("name")
        if name in _BUILTIN_NAMES:
            continue
        # Anything else is an external call we cannot lower further.
        return True
    return False


def _describe_write(write: Any) -> str:
    file = _get(write, "file") or "?"
    line = _get(write, "line") or "?"
    expression = _get(write, "expression") or "?"
    return f"{file}:{line} = {expression}"


def _describe_branch_status(branch: ConditionalSlice) -> str:
    result = branch.branch_result
    if result.status == "resolved":
        return f"if {branch.condition}: {result.expression} [resolved]"
    blocker = result.blocker
    if blocker is not None:
        return f"if {branch.condition}: [{blocker.kind}: {blocker.detail}]"
    return f"if {branch.condition}: [{result.status}]"


def _describe_slice_blocker(result: SliceResult) -> str:
    if result.blocker is None:
        return result.status
    return f"{result.blocker.symbol}: {result.blocker.kind} ({result.blocker.detail})"


def _expression_symbols_iter(expression: str) -> Iterator[str]:
    yield from _expression_symbols(expression)
