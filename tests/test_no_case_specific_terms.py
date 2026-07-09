"""Guard: case-specific PX4 terms must not appear in behavior-affecting
positions of production code.

Rule (user, 2026-07-09): mechanism-specific identifiers are allowed only
in test scripts, docstrings, and comments — never where they can affect
results (agent instruction strings, search queries, table keys, code).
Comments never reach the AST; docstrings are excluded explicitly; every
other string literal in ``flight_log_agent/`` is checked.

PX4 *framework* vocabulary (uORB macros, ``px4::params``,
``DEFINE_PARAMETERS``) is generic extraction machinery and deliberately
not listed here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "flight_log_agent"

CASE_TERMS = re.compile(
    r"RTL_[A-Z]|FW_[A-Z]|WV_[A-Z]|_rtl_alt|find_RTL|calc_CAS|CAS_scale"
    r"|[Ww]eather[Vv]ane|cone_half_angle|DO_JUMP|read_mission_item"
    r"|adapt_airspeed_setpoint|airspeed_demand|get_distance_to_next_waypoint"
    r"|_destination\.alt|lateral_accel"
)

# flight_review-mirrored plot definitions are per-topic by design
# (see memory: flight-review-alignment); their tables track upstream.
EXEMPT = {"ulog/plots.py", "ulog/interactive_plots.py"}


def _docstring_constants(tree: ast.AST) -> set[int]:
    """ids of Constant nodes that are docstrings."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def test_no_case_specific_terms_outside_docstrings():
    violations: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel in EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_constants(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                match = CASE_TERMS.search(node.value)
                if match:
                    snippet = node.value.strip().replace("\n", " ")[:80]
                    violations.append(
                        f"{rel}:{node.lineno}: {match.group(0)!r} in {snippet!r}"
                    )
    assert not violations, (
        "case-specific terms in behavior-affecting string literals "
        "(allowed only in tests/docstrings/comments):\n  "
        + "\n  ".join(violations)
    )
