"""Resolution of model provider/transport/effort configuration.

Seam: resolve_model_runtime maps explicit arguments plus environment to a
ModelRuntime. Tests assert externally visible resolution outcomes, never SDK
internals.
"""

from __future__ import annotations

import pytest

from flight_log_agent.analysis import model_provider as mp


def test_default_resolution_preserves_current_behavior(monkeypatch):
    monkeypatch.delenv("FLIGHT_LOG_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("FLIGHT_LOG_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    runtime = mp.resolve_model_runtime(model="gpt-5.6")
    assert runtime.provider == "openai"
    assert runtime.model == "gpt-5.6"
    assert runtime.reasoning_effort == "high"
    assert runtime.transport == "responses"
    assert runtime.run_config is None
    assert runtime.web_search_supported is True


def test_explicit_openai_provider_matches_default_path():
    runtime = mp.resolve_model_runtime(
        model="gpt-5.6", model_provider="openai"
    )
    assert runtime.provider == "openai"
    assert runtime.run_config is None
    assert runtime.web_search_supported is True


def test_openrouter_requires_api_key():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        mp.resolve_model_runtime(
            model="openai/gpt-5-nano", model_provider="openrouter"
        )


def test_openrouter_chat_transport_disables_hosted_search():
    runtime = mp.resolve_model_runtime(
        model="openai/gpt-5-nano",
        model_provider="openrouter",
        openrouter_api_key="or-test-key",
    )
    assert runtime.provider == "openrouter"
    assert runtime.transport == "chat"
    assert runtime.run_config is not None
    assert runtime.web_search_supported is False


def test_openrouter_responses_transport_keeps_hosted_search():
    runtime = mp.resolve_model_runtime(
        model="openai/gpt-5-nano",
        model_provider="openrouter",
        model_transport="responses",
        openrouter_api_key="or-test-key",
    )
    assert runtime.transport == "responses"
    assert runtime.run_config is not None
    assert runtime.web_search_supported is True


def test_openai_chat_transport_forces_hosted_search_off():
    runtime = mp.resolve_model_runtime(
        model="gpt-5.6", model_provider="openai", model_transport="chat"
    )
    assert runtime.transport == "chat"
    assert runtime.run_config is None
    assert runtime.web_search_supported is False


def test_unknown_provider_is_local_config_error():
    with pytest.raises(ValueError, match="model_provider"):
        mp.resolve_model_runtime(model="m", model_provider="nope")


def test_unknown_transport_is_local_config_error():
    with pytest.raises(ValueError, match="model_transport"):
        mp.resolve_model_runtime(model="m", model_transport="nope")


def test_empty_model_is_local_config_error():
    with pytest.raises(ValueError, match="model"):
        mp.resolve_model_runtime(model="")


def test_invalid_effort_is_local_config_error():
    with pytest.raises(ValueError, match="reasoning_effort"):
        mp.resolve_model_runtime(model="m", reasoning_effort="ultra")


def test_effort_precedence_explicit_over_env(monkeypatch):
    monkeypatch.setenv("FLIGHT_LOG_REASONING_EFFORT", "low")
    runtime = mp.resolve_model_runtime(model="m", reasoning_effort="medium")
    assert runtime.reasoning_effort == "medium"


def test_effort_env_over_default(monkeypatch):
    monkeypatch.setenv("FLIGHT_LOG_REASONING_EFFORT", "low")
    runtime = mp.resolve_model_runtime(model="m")
    assert runtime.reasoning_effort == "low"


def test_provider_env_selects_openrouter(monkeypatch):
    monkeypatch.setenv("FLIGHT_LOG_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-env-key")
    runtime = mp.resolve_model_runtime(model="openai/gpt-5-nano")
    assert runtime.provider == "openrouter"
    assert runtime.run_config is not None


def test_model_id_passes_through_verbatim():
    runtime = mp.resolve_model_runtime(
        model="typesafe/jev-router",
        model_provider="openrouter",
        openrouter_api_key="or-test-key",
    )
    assert runtime.model == "typesafe/jev-router"


def test_served_model_unavailable_when_result_exposes_nothing():
    assert mp.served_model_from_result(object()) == "unavailable"


def test_served_model_prefers_exposed_raw_response_value():
    class Raw:
        model = "served-model-id"

    class Result:
        raw_responses = [Raw()]

    assert mp.served_model_from_result(Result()) == "served-model-id"


def test_served_model_ignores_empty_exposed_values():
    class Raw:
        model = ""

    class Result:
        raw_responses = [Raw()]

    assert mp.served_model_from_result(Result()) == "unavailable"


def _chat_completion_payload(
    *,
    response_id="gen-1",
    model="served-a",
    extra=None,
    cost=0.001,
):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if cost is not None:
        usage["cost"] = cost
    payload = {
        "id": response_id,
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "created": 0,
        "model": model,
        "object": "chat.completion",
        "usage": usage,
    }
    if extra is not None:
        payload["openrouter_metadata"] = extra
    return payload


def _mock_transport(payloads, seen, fail_after=None):
    import httpx

    calls = {"n": 0}

    def handler(request):
        seen.append(
            {
                "url": str(request.url),
                "metadata_header": request.headers.get("X-OpenRouter-Metadata"),
            }
        )
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            raise httpx.ConnectError("boom", request=request)
        payload = payloads[min(calls["n"] - 1, len(payloads) - 1)]
        return httpx.Response(200, json=payload, request=request)

    return httpx.MockTransport(handler)


def test_openrouter_hook_records_metadata_and_opt_in_header():
    import asyncio

    import httpx
    from agents import Usage
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

    from flight_log_agent.analysis import model_provider as mp

    seen = []
    transport = _mock_transport(
        [
            _chat_completion_payload(
                extra={"requested": "openai/gpt-5-nano", "strategy": "direct"}
            )
        ],
        seen,
    )
    sink = []
    provider, aclose = mp.build_openrouter_provider(
        "or-test-key", sink.append, use_responses=False, http_transport=transport
    )
    try:
        model = provider.get_model("openai/gpt-5-nano")
        assert isinstance(model, OpenAIChatCompletionsModel)
        response = asyncio.run(
            model.get_response(
                "system", "hi", ModelSettings(), [], None, [], ModelTracing.DISABLED
            )
        )
        assert response.usage.requests == 1
    finally:
        asyncio.run(aclose())
    assert seen[0]["metadata_header"] == "enabled"
    assert seen[0]["url"].startswith("https://openrouter.ai/api/v1/")
    assert len(sink) == 1
    event = sink[0]
    assert event["generation_id"] == "gen-1"
    assert event["served_model"] == "served-a"
    assert event["charged_cost"] == 0.001
    assert event["routing"] == {"requested": "openai/gpt-5-nano", "strategy": "direct"}
    assert event["served_reasoning_effort"] == "unavailable"
    assert not any(k.startswith("requested_") for k in event)
    for forbidden in ("prompt", "content", "messages", "api_key", "key"):
        assert forbidden not in event


def test_openrouter_hook_marks_missing_fields_unavailable():
    import asyncio

    import httpx
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

    from flight_log_agent.analysis import model_provider as mp

    seen = []
    transport = _mock_transport(
        [
            _chat_completion_payload(response_id="gen-1", model="served-a", cost=None),
            _chat_completion_payload(response_id="gen-2", model="served-b", cost=None),
        ],
        seen,
    )
    sink = []
    provider, aclose = mp.build_openrouter_provider(
        "or-test-key", sink.append, use_responses=False, http_transport=transport
    )
    try:
        model = provider.get_model("openai/gpt-5-nano")
        settings = ModelSettings()
        for _ in range(2):
            asyncio.run(
                model.get_response(
                    "system", "hi", settings, [], None, [], ModelTracing.DISABLED
                )
            )
    finally:
        asyncio.run(aclose())
    assert [e["served_model"] for e in sink] == ["served-a", "served-b"]
    assert all(e["charged_cost"] == "unavailable" for e in sink)
    assert all(e["routing"] == "unavailable" for e in sink)
    assert all(e["generation_id"] in ("gen-1", "gen-2") for e in sink)


def test_openrouter_hook_records_survive_terminal_failure():
    import asyncio

    import httpx
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing

    from flight_log_agent.analysis import model_provider as mp

    seen = []
    transport = _mock_transport(
        [_chat_completion_payload()], seen, fail_after=1
    )
    sink = []
    provider, aclose = mp.build_openrouter_provider(
        "or-test-key", sink.append, use_responses=False, http_transport=transport
    )
    try:
        model = provider.get_model("openai/gpt-5-nano")
        settings = ModelSettings()
        asyncio.run(
            model.get_response(
                "system", "hi", settings, [], None, [], ModelTracing.DISABLED
            )
        )
        with __import__("pytest").raises(Exception):
            asyncio.run(
                model.get_response(
                    "system", "hi", settings, [], None, [], ModelTracing.DISABLED
                )
            )
    finally:
        asyncio.run(aclose())
    assert len(sink) == 1
    assert sink[0]["generation_id"] == "gen-1"


def test_transmitted_request_carries_strict_report_contract():
    """Regression: our requests are well-formed, so schema-ignoring
    failures are provider-side. Asserts against the transmitted body."""
    import asyncio
    import json

    import httpx
    from agents.agent_output import AgentOutputSchema
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing
    from openai.types.shared.reasoning import Reasoning

    from flight_log_agent.models import FlightLogReport

    from flight_log_agent.analysis import model_provider as mp

    seen = []

    def handler(request):
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json=_chat_completion_payload(),
            request=request,
        )

    runtime = mp.resolve_model_runtime(
        model="typesafe/jev-router",
        model_provider="openrouter",
        model_transport="chat",
        reasoning_effort="high",
        openrouter_api_key="repro-key",
        http_transport=httpx.MockTransport(handler),
    )
    model = runtime.run_config.model_provider.get_model("typesafe/jev-router")
    schema = AgentOutputSchema(output_type=FlightLogReport)
    asyncio.run(
        model.get_response(
            "system",
            "investigate",
            ModelSettings(reasoning=Reasoning(effort="high")),
            [],
            schema,
            [],
            ModelTracing.DISABLED,
        )
    )
    assert len(seen) == 1
    body = seen[0]
    assert body["model"] == "typesafe/jev-router"
    assert body["reasoning_effort"] == "high"
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert (
        response_format["json_schema"]["schema"]["title"] == "FlightLogReport"
    )
    assert "provider" not in body


def test_router_cost_tier_unset_by_default(monkeypatch):
    monkeypatch.delenv("FLIGHT_LOG_ROUTER_COST_TIER", raising=False)
    runtime = mp.resolve_model_runtime(model="m")
    assert runtime.router_cost_tier is None


def test_router_cost_tier_explicit_over_env(monkeypatch):
    from flight_log_agent.analysis import model_provider as mpm

    monkeypatch.setenv("FLIGHT_LOG_ROUTER_COST_TIER", "low")
    assert (
        mpm.resolve_model_runtime(model="m", router_cost_tier="high").router_cost_tier
        == "high"
    )
    assert (
        mpm.resolve_model_runtime(model="m").router_cost_tier == "low"
    )


def test_router_cost_tier_rejects_unknown_value():
    import pytest

    with pytest.raises(ValueError, match="router_cost_tier"):
        mp.resolve_model_runtime(model="m", router_cost_tier="ultra")


def test_router_plugins_body_absent_without_tier():
    from flight_log_agent.analysis import model_provider as mpm

    assert mpm.router_plugins_body(None) is None


def test_router_plugins_body_matches_documented_shape():
    from flight_log_agent.analysis import model_provider as mpm

    assert mpm.router_plugins_body("low") == {
        "plugins": [{"id": "jev-router", "cost_tier": "low"}]
    }
