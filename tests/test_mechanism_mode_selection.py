"""Mechanism-path mode selection tests (DAG-primary retirement).

Pin the normal/default runtime routing for mechanism analysis:
DAG by default, legacy only via explicit opt-in or when no pinned
source snapshot exists for the DAG stage. No provider calls;
pure orchestration contract.
"""

from __future__ import annotations


def _resolve(**kwargs):
    from flight_log_agent.runner_core import resolve_mechanism_path

    return resolve_mechanism_path(**kwargs)


def test_default_with_source_selects_dag():
    """Normal/default execution with a pinned source snapshot runs
    the DAG path, never legacy discovery."""
    assert _resolve(dag_discovery=None, env_value="") == "dag"


def test_explicit_true_selects_dag():
    assert _resolve(dag_discovery=True, env_value="") == "dag"


def test_explicit_false_preserves_legacy_opt_in():
    """Explicit False keeps the legacy path for manual
    rollback/debugging and explicitly-invoked comparison."""
    assert _resolve(dag_discovery=False, env_value="") == "legacy"


def test_explicit_off_env_values_select_legacy():
    for env_value in ("0", "false", "no", "off", "legacy",
                      "FALSE", " Off "):
        assert _resolve(dag_discovery=None, env_value=env_value) == "legacy"


def test_explicit_on_env_values_select_dag():
    for env_value in ("1", "true", "yes", "on", "dag"):
        assert _resolve(dag_discovery=None, env_value=env_value) == "dag"


def test_param_overrides_env():
    """Explicit param wins over env in both directions."""
    assert _resolve(dag_discovery=True, env_value="0") == "dag"
    assert _resolve(dag_discovery=False, env_value="1") == "legacy"


def test_source_gating_lives_at_call_site():
    """Source-snapshot gating stays in the analyze_flight_log gate
    (`if dag_discovery_enabled and source_snapshot is not None`),
    pinned by the P0 migration shape test — not in this pure
    flag-routing contract. Without a snapshot the legacy branch
    handles the run (model-free in practice)."""
    from pathlib import Path

    runner = (Path(__file__).resolve().parent.parent
              / "flight_log_agent" / "runner_core.py").read_text(
        encoding="utf-8")
    assert ("if dag_discovery_enabled and source_snapshot is not None"
            in runner)


def test_dag_branch_returns_before_legacy_stages():
    """Structural pin: the DAG-enabled branch of analyze_flight_log
    returns before any legacy stage code, so DAG failure cannot fall
    through into legacy discovery (mirrors the repo's T3-style
    AST-isolation test convention)."""
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
    assert len(legacy_defs) == 1
    assert legacy_defs[0].lineno > dag_branches[0].lineno
    assert dag_branches[0].lineno < legacy_defs[0].lineno
    for node in ast.walk(dag_branches[0]):
        assert not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "decide_source_discovery"
        ), "legacy decide callback must not live inside the DAG branch"
