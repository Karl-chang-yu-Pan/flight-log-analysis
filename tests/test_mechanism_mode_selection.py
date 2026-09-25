"""Stage-2 S1: DAG-only routing, no-source handling, deprecation, bakeoff.

Normal runtime routes mechanism analysis through DAG only. Old
legacy selectors route to DAG with a deprecation audit event.
Missing/invalid source yields a deterministic model-free
unresolved result. Explicit legacy bakeoff fails explicitly.
No provider calls; pure orchestration + structural contracts.
"""

from __future__ import annotations


def _selector(**kwargs):
    from flight_log_agent.runner_core import deprecated_legacy_selector

    return deprecated_legacy_selector(**kwargs)


def test_no_legacy_selector_by_default():
    assert _selector(dag_discovery=None, env_value="") is None


def test_explicit_false_requests_deprecation():
    result = _selector(dag_discovery=False, env_value="")
    assert result is not None
    assert result["requested_source"] == "parameter"
    assert result["requested_value"] == "False"
    assert result["effective_route"] == "dag"
    assert result["reason"] == "legacy_removed"


def test_explicit_true_requests_no_deprecation():
    assert _selector(dag_discovery=True, env_value="") is None


def test_off_env_values_request_deprecation():
    for env_value in ("0", "false", "no", "off", "legacy",
                      "FALSE", " Off "):
        result = _selector(dag_discovery=None, env_value=env_value)
        assert result is not None
        assert result["requested_source"] == "environment"
        assert result["effective_route"] == "dag"
        assert result["reason"] == "legacy_removed"


def test_on_env_values_request_no_deprecation():
    for env_value in ("1", "true", "yes", "on", "dag", "banana"):
        assert _selector(dag_discovery=None, env_value=env_value) is None


def test_param_false_wins_over_env_on():
    result = _selector(dag_discovery=False, env_value="1")
    assert result is not None
    assert result["requested_source"] == "parameter"


def test_gate_has_no_legacy_opt_in_branch():
    """Structural pin: the DAG gate is snapshot-conditional only;
    the old opt-in resolver and off-value set are gone."""
    import ast
    from pathlib import Path

    tree = ast.parse(
        (Path(__file__).resolve().parent.parent
         / "flight_log_agent" / "runner_core.py").read_text(
            encoding="utf-8")
    )
    analyze = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "analyze_flight_log"
    )
    dumped = ast.dump(analyze)
    assert "resolve_mechanism_path" not in dumped
    assert "_DAG_EXPLICIT_OFF" not in dumped
    assert "mechanism_selection.deprecated_legacy_requested" in dumped


def test_dag_branch_returns_before_legacy_stages():
    """Structural pin retained from retirement: the DAG-enabled
    branch returns before any legacy stage code, so DAG failure
    cannot fall through into legacy discovery."""
    import ast
    from pathlib import Path

    tree = ast.parse(
        (Path(__file__).resolve().parent.parent
         / "flight_log_agent" / "runner_core.py").read_text(
            encoding="utf-8")
    )
    analyze = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "analyze_flight_log"
    )
    dag_branches = [
        node for node in ast.walk(analyze)
        if isinstance(node, ast.If)
        and "dag_discovery_enabled" in ast.dump(node.test)
    ]
    assert dag_branches, "DAG gate must remain a single explicit branch"
    assert len(dag_branches) == 1
    returns = [node for node in ast.walk(dag_branches[0])
               if isinstance(node, ast.Return)]
    assert returns, "DAG branch must return before legacy stages run"
    legacy_defs = [
        node for node in ast.walk(analyze)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "decide_source_discovery"
    ]
    assert legacy_defs == [], "legacy decide callback must not exist"


def test_no_source_discovery_is_model_free_unresolved():
    """Missing source resolves deterministically without any
    provider call: no resolver construction anywhere in the
    runner, and the exact unresolved contract wording is
    preserved."""
    runner_text = (
        _Path(__file__).resolve().parent.parent
        / "flight_log_agent" / "runner_core.py"
    ).read_text(encoding="utf-8")
    assert "SourceMechanismResolver(" not in runner_text
    assert (
        "Exact PX4 source snapshot is unavailable for "
        "source-mechanism discovery." in runner_text
    )


# ----------------------------------------------------------------------
# S2 — runner legacy provider path is gone
# ----------------------------------------------------------------------

import ast as _ast
from pathlib import Path as _Path


def _runner_tree():
    return _ast.parse(
        (_Path(__file__).resolve().parent.parent
         / "flight_log_agent" / "runner_core.py").read_text(
            encoding="utf-8")
    )


def test_no_legacy_provider_symbols_in_runner():
    """The legacy mechanism-discovery provider path no longer
    exists in production: no agent, no decide closure, no
    discover/convert wrappers, no runner-local canonicalizer."""
    tree = _runner_tree()
    names = {
        node.id for node in _ast.walk(tree)
        if isinstance(node, _ast.Name)
    }
    for symbol in (
        "source_discovery_agent",
        "decide_source_discovery",
        "discover_source_mechanisms",
        "source_mechanisms_to_candidates",
        "source_mechanism_to_candidate",
        "canonicalize_mechanism_candidate_signals",
    ):
        assert symbol not in names, f"legacy provider symbol remains: {symbol}"
    defined = {
        node.name for node in _ast.walk(tree)
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                             _ast.ClassDef))
    }
    assert "SignalCanonicalizer" not in defined


def test_legacy_branch_references_no_decide_call():
    """No decide/model callback construction remains anywhere in
    the runner module."""
    for node in _ast.walk(_runner_tree()):
        if isinstance(node, _ast.Name) and node.id == "decide_source_discovery":
            raise AssertionError("legacy decide callback reference remains")


# ----------------------------------------------------------------------
# S4 — legacy subsystem physically absent
# ----------------------------------------------------------------------


def test_legacy_subsystem_modules_absent():
    """Stage-2 S4: the legacy mechanism-discovery implementation
    does not exist in production — no resolver, no frontier
    machinery, no legacy schema module."""
    import importlib.util

    for module in (
        "flight_log_agent.px4.source_mechanism_resolver",
        "flight_log_agent.px4.discovery_frontier",
        "flight_log_agent.px4.source_mechanism_models",
    ):
        assert importlib.util.find_spec(module) is None, module
