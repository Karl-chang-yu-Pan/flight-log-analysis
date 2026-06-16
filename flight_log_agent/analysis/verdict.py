"""Centralized verdict and confidence-ceiling rules.

Single owner of the supported/contradicted/mixed/unresolved/excluded reduction
and the verdict -> ceiling table. Replaces seven near-duplicate aggregators
that previously lived in ``signature_verification.py`` and
``ulog/signature_evaluator.py``, each of which had its own slight variant of
the same rule.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Optional


Verdict = Literal["supported", "contradicted", "mixed", "unresolved", "excluded"]
Ceiling = Literal["high", "medium", "low", "unresolved"]
Level = Literal["check_list", "mechanism", "graph"]


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------


def verdict_from_counts(
    *,
    supported: int,
    contradicted: int,
    unresolved: int = 0,
) -> Verdict:
    """Apply the shared "any + any = mixed" reduction rule.

    Returns:
      - "mixed" when both supported and contradicted are present
      - "supported" or "contradicted" when only one of them is
      - "unresolved" otherwise
    """
    if supported and contradicted:
        return "mixed"
    if supported:
        return "supported"
    if contradicted:
        return "contradicted"
    return "unresolved"


_CEILING_RANK: dict[str, int] = {"unresolved": 0, "low": 1, "medium": 2, "high": 3}


def ceiling_for(
    verdict: Verdict,
    *,
    has_unresolved_defining: bool = False,
    missing_required_signals: bool = False,
    max_ceiling: Optional[Ceiling] = None,
) -> Ceiling:
    """Map a verdict (plus context) to a confidence ceiling.

    - ``missing_required_signals`` forces ``"low"`` regardless of verdict.
    - "supported" defaults to ``"high"`` but caps at ``"medium"`` when there
      are unresolved defining checks, and further caps at ``max_ceiling``
      when provided (used by the graph-merge path so a graph-promoted
      verdict can't claim more confidence than the underlying evaluation).
    - "mixed" caps at "medium", "contradicted" / "excluded" at "low".
    - "unresolved" stays "unresolved".
    """
    if missing_required_signals:
        return "low"
    if verdict == "supported":
        result: Ceiling = "medium" if has_unresolved_defining else "high"
    elif verdict == "mixed":
        result = "medium"
    elif verdict in ("contradicted", "excluded"):
        result = "low"
    else:
        result = "unresolved"
    if max_ceiling and _CEILING_RANK.get(max_ceiling, 0) < _CEILING_RANK.get(result, 0):
        return max_ceiling
    return result


def combine_verdicts(*verdicts: str) -> Verdict:
    """Combine multiple top-level verdicts into one.

    Used when an evaluation has both a flat-plan verdict and a graph verdict
    that need to be reconciled into a single mechanism-level verdict.
    """
    distinct = {v for v in verdicts if v}
    if not distinct:
        return "unresolved"
    if distinct == {"unresolved"}:
        return "unresolved"
    distinct.discard("unresolved")
    if "mixed" in distinct:
        return "mixed"
    if len(distinct) == 1:
        return distinct.pop()  # type: ignore[return-value]
    if {"supported", "contradicted"}.issubset(distinct):
        return "mixed"
    return "mixed"


# ----------------------------------------------------------------------
# Level dispatcher
# ----------------------------------------------------------------------


def aggregate(items: Iterable[Any], *, level: Level) -> Verdict:
    """Reduce ``items`` to a single verdict using level-specific rules.

    - ``level="check_list"``: ``items`` are check-result dicts carrying
      ``role`` and ``status`` fields. Mirrors the previous
      ``verdict_from_role_results`` behaviour: any failed branch-applicability
      check makes the mechanism excluded; any unresolved defining check makes
      it unresolved.
    - ``level="mechanism"``: ``items`` are branch-result dicts with
      ``verdict`` and ``has_mechanism_defining_checks`` fields. Excluded
      branches and branches without defining checks are filtered out before
      reduction.
    - ``level="graph"``: ``items`` are graph result objects with a
      ``verdict`` attribute (or dicts with that key). Pure
      ``supported`` / ``contradicted`` / mixed reduction.
    """
    items = list(items)
    if level == "check_list":
        return _from_check_list(items)
    if level == "mechanism":
        return _from_mechanism(items)
    if level == "graph":
        return _from_graph(items)
    raise ValueError(f"unknown aggregation level: {level!r}")


def branch_result(
    branch: Any,
    window_results: list[dict[str, Any]],
    unresolved_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the standard branch-result dict using the shared verdict rule.

    Replaces the previous ``aggregate_branch_results``. The unusual return
    shape (dict, not str) is preserved so callers see no schema change.
    """
    verdicts = [str(window.get("verdict") or "") for window in window_results]

    has_blocked_dependency = bool(getattr(branch, "unresolved_dependencies", None)) or any(
        check.get("role") == "mechanism_defining" for check in unresolved_checks
    )

    if has_blocked_dependency:
        verdict: Verdict = "unresolved"
    elif "supported" in verdicts and any(v in ("contradicted", "mixed") for v in verdicts):
        verdict = "mixed"
    elif "supported" in verdicts:
        verdict = "supported"
    elif "mixed" in verdicts:
        verdict = "mixed"
    elif verdicts and all(v == "excluded" for v in verdicts):
        verdict = "excluded"
    elif "contradicted" in verdicts:
        verdict = "contradicted"
    else:
        verdict = "unresolved"

    return {
        "branch_id": branch.branch_id,
        "name": branch.name,
        "verdict": verdict,
        "has_mechanism_defining_checks": any(
            planned.role == "mechanism_defining" for planned in branch.checks
        ),
        "unresolved_dependencies": list(branch.unresolved_dependencies),
        "window_results": window_results,
    }


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _from_check_list(results: list[dict[str, Any]]) -> Verdict:
    applicability = [r for r in results if r.get("role") == "branch_applicability"]
    if any(r.get("status") == "failed" for r in applicability):
        return "excluded"
    if any(r.get("status") == "unresolved" for r in applicability):
        return "unresolved"

    defining = [r for r in results if r.get("role") == "mechanism_defining"]
    if not defining or any(r.get("status") == "unresolved" for r in defining):
        return "unresolved"

    passed = sum(1 for r in defining if r.get("status") == "passed")
    failed = sum(1 for r in defining if r.get("status") == "failed")
    verdict = verdict_from_counts(supported=passed, contradicted=failed)
    if verdict == "unresolved" and all(r.get("status") == "passed" for r in defining):
        return "supported"
    return verdict


def _from_mechanism(branch_results: list[dict[str, Any]]) -> Verdict:
    verdicts = [
        r["verdict"]
        for r in branch_results
        if r["verdict"] != "excluded" and r.get("has_mechanism_defining_checks")
    ]
    if "supported" in verdicts and any(v in ("contradicted", "mixed") for v in verdicts):
        return "mixed"
    if "supported" in verdicts:
        return "supported"
    if "mixed" in verdicts:
        return "mixed"
    if verdicts and all(v == "contradicted" for v in verdicts):
        return "contradicted"
    return "unresolved"


def _from_graph(results: list[Any]) -> Verdict:
    verdicts = {_extract_verdict(r) for r in results}
    if verdicts == {"supported"}:
        return "supported"
    if verdicts == {"contradicted"}:
        return "contradicted"
    return "mixed"


def _extract_verdict(item: Any) -> str:
    if hasattr(item, "verdict"):
        return str(getattr(item, "verdict") or "")
    if isinstance(item, dict):
        return str(item.get("verdict") or "")
    return ""
