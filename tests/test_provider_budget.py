"""Cumulative pre-call provider budget guard tests (offline only).

Every test uses fake runners, fake usage objects, and fake clocks.
No provider calls, no credentials, no network.
"""

from __future__ import annotations

from types import SimpleNamespace


def _usage(**overrides):
    fields = {
        "requests": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _budget(**overrides):
    from flight_log_agent.provider_budget import ProviderBudget

    return ProviderBudget(**overrides)


def _usage_state(**kwargs):
    from flight_log_agent.provider_budget import ProviderBudgetUsage

    return ProviderBudgetUsage(**kwargs)


def test_call_count_guard_blocks_second_call():
    """max_provider_calls=1: first call allowed, second blocked
    BEFORE any provider invocation; abort names the dimension."""
    from flight_log_agent.provider_budget import BudgetExceeded

    budget = _budget(max_provider_calls=1)
    state = _usage_state()
    invocations = []

    def fake_invoke():
        invocations.append(1)
        state.record_call("intent", "gpt-5.5", _usage())

    state.check_or_raise(budget, role="intent")
    fake_invoke()
    try:
        state.check_or_raise(budget, role="seeder")
    except BudgetExceeded as exc:
        assert exc.dimension == "provider_calls"
        assert exc.limit == 1
        assert exc.observed == 1
    else:
        raise AssertionError("second call was not blocked")
    assert len(invocations) == 1


def test_zero_call_budget_blocks_first_call():
    """max_provider_calls=0 blocks immediately (off-by-one pin)."""
    from flight_log_agent.provider_budget import BudgetExceeded

    budget = _budget(max_provider_calls=0)
    state = _usage_state()
    try:
        state.check_or_raise(budget, role="intent")
    except BudgetExceeded as exc:
        assert exc.dimension == "provider_calls"
        assert exc.observed == 0
    else:
        raise AssertionError("first call was not blocked")


def test_token_guard_spans_roles_without_reset():
    """One accumulator across intent→seeder→judge: threshold reached
    after seeder blocks the judge role; totals carry both calls."""
    from flight_log_agent.provider_budget import BudgetExceeded

    budget = _budget(max_total_input_tokens=250)
    state = _usage_state()
    state.check_or_raise(budget, role="intent")
    state.record_call("intent", "gpt-5.5", _usage(input_tokens=100))
    state.check_or_raise(budget, role="seeder")
    state.record_call("seeder", "gpt-5.5", _usage(input_tokens=200))
    assert state.total_input_tokens == 300
    try:
        state.check_or_raise(budget, role="judge")
    except BudgetExceeded as exc:
        assert exc.dimension == "input_tokens"
        assert exc.observed == 300
    else:
        raise AssertionError("judge call was not blocked")


def test_output_token_guard_boundary():
    """output == limit trips (>= semantics, pinned explicitly)."""
    from flight_log_agent.provider_budget import BudgetExceeded

    budget = _budget(max_total_output_tokens=20)
    state = _usage_state()
    state.record_call("intent", "gpt-5.5", _usage(output_tokens=20))
    try:
        state.check_or_raise(budget, role="seeder")
    except BudgetExceeded as exc:
        assert exc.dimension == "output_tokens"
    else:
        raise AssertionError("boundary did not trip")


def test_wall_guard_uses_monotonic_elapsed():
    """Fake clock: elapsed >= max blocks without sleeping."""
    from flight_log_agent.provider_budget import BudgetExceeded

    budget = _budget(max_wall_seconds=60.0)
    state = _usage_state(run_start_monotonic=1000.0)
    state.check_or_raise(budget, role="intent", now=1010.0)
    try:
        state.check_or_raise(budget, role="seeder", now=1060.0)
    except BudgetExceeded as exc:
        assert exc.dimension == "wall_seconds"
    else:
        raise AssertionError("wall guard did not trip")


def test_cost_guard_with_user_prices():
    """User-supplied prices make completed cumulative cost
    enforceable; no baked-in pricing exists."""
    from flight_log_agent.provider_budget import BudgetExceeded, ModelPrices

    prices = {"gpt-5.5": ModelPrices(input_usd_per_token=0.01,
                                     output_usd_per_token=0.02)}
    budget = _budget(max_total_cost_usd=2.0, model_prices=prices)
    state = _usage_state()
    state.record_call("intent", "gpt-5.5",
                      _usage(input_tokens=100, output_tokens=20),
                      prices=prices)
    # cost = 100*0.01 + 20*0.02 = 1.40 < 2.0: allowed
    state.check_or_raise(budget, role="seeder")
    state.record_call("seeder", "gpt-5.5",
                      _usage(input_tokens=100, output_tokens=20),
                      prices=prices)
    # cost = 2.80 >= 2.0: next call blocked
    try:
        state.check_or_raise(budget, role="judge")
    except BudgetExceeded as exc:
        assert exc.dimension == "cost_usd"
    else:
        raise AssertionError("cost guard did not trip")


def test_unknown_model_cost_is_zero_and_visible():
    """Models without price entries contribute zero cost but keep
    their identity recorded (auditable gap, never silent)."""
    from flight_log_agent.provider_budget import ProviderBudgetUsage

    state = ProviderBudgetUsage()
    state.record_call("intent", "future-model-9", _usage(),
                      prices={})
    assert state.total_cost_usd == 0.0
    assert state.calls[0].model == "future-model-9"


def test_no_limit_configuration_changes_nothing():
    """All-None budget never raises, regardless of usage volume."""
    budget = _budget()
    assert budget.enabled is False
    state = _usage_state()
    for _ in range(5):
        state.record_call("intent", "gpt-5.5",
                          _usage(input_tokens=10 ** 9))
    state.check_or_raise(budget, role="judge")


def test_abort_carries_operational_context():
    """The abort exposes dimension, limit, observed totals, and
    counts — no secrets, everything needed to report."""
    from flight_log_agent.provider_budget import BudgetExceeded

    budget = _budget(max_provider_calls=1)
    state = _usage_state()
    state.record_call("intent", "gpt-5.5", _usage())
    try:
        state.check_or_raise(budget, role="judge")
    except BudgetExceeded as exc:
        assert exc.provider_calls == 1
        assert exc.input_tokens == 100
        assert exc.output_tokens == 20
        assert exc.role == "judge"
        message = repr(exc)
        assert "provider_calls" in message
        assert "role=judge" in message
    else:
        raise AssertionError("expected abort")


def test_completed_calls_persist_and_blocked_call_absent():
    """Completed fake calls stay in the trail; the blocked call
    never appears as completed usage."""
    from flight_log_agent.provider_budget import BudgetExceeded

    budget = _budget(max_provider_calls=1)
    state = _usage_state()
    state.check_or_raise(budget, role="intent")
    state.record_call("intent", "gpt-5.5", _usage())
    try:
        state.check_or_raise(budget, role="seeder")
    except BudgetExceeded:
        pass
    assert [call.role for call in state.calls] == ["intent"]
    assert len(state.calls) == 1


def test_roles_attributed_per_call():
    """Cumulative budget is global; observability keeps per-role
    attribution for every completed call."""
    state = _usage_state()
    state.record_call("intent", "gpt-5.5", _usage())
    state.record_call("seeder", "gpt-5.5", _usage())
    assert [(call.role, call.model) for call in state.calls] == [
        ("intent", "gpt-5.5"), ("seeder", "gpt-5.5")]


def test_fake_orchestration_abort_is_not_success():
    """Simulated intent→seeder→judge→report pipeline where the
    guard trips before the report: no report object is built and
    the abort stays operationally distinct from success and
    from evidence-unresolved outcomes."""

    from flight_log_agent.provider_budget import BudgetExceeded

    budget = _budget(max_provider_calls=2)
    state = _usage_state()
    built_report = False

    def guarded_role(role):
        state.check_or_raise(budget, role=role)
        state.record_call(role, "gpt-5.5", _usage())

    guarded_role("intent")
    guarded_role("seeder")
    try:
        guarded_role("judge")
    except BudgetExceeded as exc:
        abort = exc
    else:
        raise AssertionError("pipeline should have aborted")
    assert built_report is False
    assert abort.dimension == "provider_calls"
    assert not isinstance(abort, LookupError)


# ----------------------------------------------------------------------
# Wiring: guard inside the shared run-agent seam (offline, SDK faked)
# ----------------------------------------------------------------------

import asyncio as _asyncio
from types import SimpleNamespace as _SimpleNamespace


def _fake_runner(monkeypatch, calls, usage=None):
    from flight_log_agent import runner_core as _runner

    async def fake_run(agent, input, context, max_turns, hooks=None):
        calls.append(getattr(agent, "name", "agent"))
        return _SimpleNamespace(
            final_output={"ok": True},
            context_wrapper=_SimpleNamespace(
                usage=usage if usage is not None else _usage()),
            new_items=[],
        )

    monkeypatch.setattr(_runner.Runner, "run", fake_run)


def _agent(name="role-agent", model="gpt-5.5"):
    return _SimpleNamespace(name=name, model=model)


def test_run_agent_enforces_shared_budget(monkeypatch):
    """Two guarded _run_agent calls share one budget: the second
    trips pre-invocation, the fake provider runs once, abort is
    explicit with role attribution."""
    from flight_log_agent import runner_core as _runner
    from flight_log_agent.provider_budget import (
        BudgetExceeded, ProviderBudget, ProviderBudgetUsage,
    )

    calls: list = []
    _fake_runner(monkeypatch, calls)
    budget = ProviderBudget(max_provider_calls=1)
    usage = ProviderBudgetUsage()
    first = _asyncio.run(_runner._run_agent(
        None, "intent", _agent(), {"q": "x"}, None, 2,
        budget=budget, budget_usage=usage, role="intent"))
    assert first == {"ok": True}
    try:
        _asyncio.run(_runner._run_agent(
            None, "seeder", _agent(), {"q": "x"}, None, 2,
            budget=budget, budget_usage=usage, role="seeder"))
    except BudgetExceeded as exc:
        assert exc.dimension == "provider_calls"
        assert exc.role == "seeder"
    else:
        raise AssertionError("second guarded call was not blocked")
    assert calls == ["role-agent"]
    assert [call.role for call in usage.calls] == ["intent"]


def test_run_agent_without_budget_unchanged(monkeypatch):
    """No budget configured: existing behavior exactly (no guard,
    no accumulation, no audit requirement)."""
    from flight_log_agent import runner_core as _runner

    calls: list = []
    _fake_runner(monkeypatch, calls)
    first = _asyncio.run(_runner._run_agent(
        None, "intent", _agent(), {"q": "x"}, None, 2))
    second = _asyncio.run(_runner._run_agent(
        None, "seeder", _agent(), {"q": "x"}, None, 2))
    assert (first, second) == ({"ok": True}, {"ok": True})
    assert calls == ["role-agent", "role-agent"]


def test_run_agent_abort_emits_audit_event(tmp_path, monkeypatch):
    """The blocked call is observable: one explicit abort event,
    no completed-usage event for the blocked call."""
    import json as _json

    from flight_log_agent import runner_core as _runner
    from flight_log_agent.audit import DeveloperAuditLogger
    from flight_log_agent.provider_budget import (
        ProviderBudget, ProviderBudgetUsage,
    )

    calls: list = []
    _fake_runner(monkeypatch, calls)
    logger = DeveloperAuditLogger(tmp_path / "devlogs")
    budget = ProviderBudget(max_provider_calls=0)
    usage = ProviderBudgetUsage()
    try:
        _asyncio.run(_runner._run_agent(
            logger, "intent", _agent(), {"q": "x"}, None, 2,
            budget=budget, budget_usage=usage, role="intent"))
    except Exception:
        pass
    else:
        raise AssertionError("expected abort")
    events = [ _json.loads(line) for line in
        (logger.events_path.read_text(encoding="utf-8").splitlines()) ]
    aborts = [event for event in events
              if event.get("event") == "run.budget_guard_aborted"]
    assert len(aborts) == 1
    assert aborts[0]["role"] == "intent"
    finished = [event for event in events
                if event.get("event", "").endswith(".finished")]
    assert finished == []
    assert calls == []


# ----------------------------------------------------------------------
# e2e_suite calibration plumbing (offline: pure helpers only)
# ----------------------------------------------------------------------

import importlib.util as _importlib_util


def _e2e_suite():
    path = __import__("pathlib").Path(__file__).resolve().parent.parent / "scripts" / "e2e_suite.py"
    spec = _importlib_util.spec_from_file_location("e2e_suite_under_test", path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2e_budget_json_empty_without_flags():
    """No guard flags → empty transport (existing behavior)."""
    import argparse as _argparse

    e2e = _e2e_suite()
    args = _argparse.Namespace(
        max_provider_calls=None, max_input_tokens=None,
        max_output_tokens=None, max_cost_usd=None,
        max_wall_seconds=None, model_prices="")
    assert e2e.build_provider_budget_json(args) == ""


def test_e2e_budget_json_carries_configured_limits():
    """Configured flags serialize to the child transport payload."""
    import argparse as _argparse
    import json as _json

    e2e = _e2e_suite()
    args = _argparse.Namespace(
        max_provider_calls=6, max_input_tokens=None,
        max_output_tokens=None, max_cost_usd=2.5,
        max_wall_seconds=900.0, model_prices="")
    payload = _json.loads(e2e.build_provider_budget_json(args))
    assert payload["max_provider_calls"] == 6
    assert payload["max_total_cost_usd"] == 2.5
    assert payload["max_wall_seconds"] == 900.0
    assert payload["max_total_input_tokens"] is None


def test_e2e_abort_fragment_marks_budget_abort_only():
    """Abort marking is machine-detectable and exclusive to
    budget aborts; ordinary failures gain no flag."""
    e2e = _e2e_suite()
    from flight_log_agent.provider_budget import BudgetExceeded

    abort = BudgetExceeded(
        dimension="provider_calls", limit=6, observed=6,
        role="judge", provider_calls=6, input_tokens=1,
        output_tokens=1, cost_usd=0.0, elapsed_s=1.0)
    assert e2e._abort_status_fragment(abort) == {
        "aborted_by_budget_guard": True,
        "abort_dimension": "provider_calls"}
    assert e2e._abort_status_fragment(ValueError("boom")) == {}
