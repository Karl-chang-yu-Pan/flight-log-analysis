"""Model provider, transport, and reasoning-effort resolution.

Single seam mapping explicit configuration plus environment to a
:class:`ModelRuntime`. The analysis loop, tools, and report contract never
inspect these details directly.

Precedence:
- model: passed through verbatim by the caller (the analyzer keeps its
  existing explicit-argument > ``OPENAI_MODEL`` > ``"gpt-5.6"`` rule).
- provider: explicit argument > ``FLIGHT_LOG_MODEL_PROVIDER`` > ``"auto"``
  (``"auto"`` behaves as ``"openai"``).
- transport: explicit argument > ``"auto"`` (``"auto"`` means the SDK
  default ``"responses"`` for OpenAI and ``"chat"`` for OpenRouter).
- reasoning effort: explicit argument > ``FLIGHT_LOG_REASONING_EFFORT`` >
  ``"high"``.
- OpenRouter key: explicit argument > ``OPENROUTER_API_KEY``.

Local configuration problems raise :class:`ValueError` before any model
request. Remote problems (authentication, unknown model, unreachable
endpoint) surface from the provider at run time through the existing
failure path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from agents import OpenAIProvider, RunConfig
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_METADATA_HEADER = "X-OpenRouter-Metadata"
OPENROUTER_METADATA_OPT_IN = "enabled"

PROVIDERS = ("auto", "openai", "openrouter")
TRANSPORTS = ("auto", "responses", "chat")
REASONING_EFFORTS = ("low", "medium", "high")
ROUTER_COST_TIERS = ("low", "medium", "high")
JEV_ROUTER_PLUGIN_ID = "jev-router"

UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ModelRuntime:
    """Resolved model configuration for one analysis run."""

    provider: str
    model: str
    reasoning_effort: str
    transport: str
    run_config: Optional[RunConfig]
    web_search_supported: bool
    aclose: Optional[Callable[[], Awaitable[None]]] = None
    """Release run-scoped client resources; None when nothing was built."""
    router_cost_tier: Optional[str] = None
    """Router cost-tier policy; None keeps provider defaults."""


def router_plugins_body(router_cost_tier: Optional[str]) -> Optional[dict[str, Any]]:
    """Documented router plugin fragment, or None when unset.

    Shape follows the provider's router documentation verbatim; no other
    provider preferences are added, leaving the candidate pool
    unrestricted.
    """
    if router_cost_tier is None:
        return None
    return {
        "plugins": [
            {"id": JEV_ROUTER_PLUGIN_ID, "cost_tier": router_cost_tier}
        ]
    }


def _provider_response_record(payload: Any, headers: Any) -> dict[str, Any]:
    """Observed OpenRouter response metadata; never inferred or enriched."""
    body = payload if isinstance(payload, dict) else {}
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    cost = usage.get("cost")
    header_id = None
    try:
        header_id = headers.get("X-Generation-Id") if headers is not None else None
    except Exception:
        header_id = None
    generation_id = body.get("id") or header_id
    metadata = body.get("openrouter_metadata")
    return {
        "generation_id": (
            generation_id.strip()
            if isinstance(generation_id, str) and generation_id.strip()
            else UNAVAILABLE
        ),
        "served_model": (
            body["model"].strip()
            if isinstance(body.get("model"), str) and body["model"].strip()
            else UNAVAILABLE
        ),
        "served_provider": (
            body["provider"].strip()
            if isinstance(body.get("provider"), str) and body["provider"].strip()
            else UNAVAILABLE
        ),
        "charged_cost": (
            cost if isinstance(cost, (int, float)) else UNAVAILABLE
        ),
        "routing": metadata if isinstance(metadata, dict) else UNAVAILABLE,
        "served_reasoning_effort": UNAVAILABLE,
    }


def build_openrouter_provider(
    api_key: str,
    response_sink: Callable[[dict[str, Any]], None],
    *,
    use_responses: bool,
    http_transport: Any = None,
) -> tuple[OpenAIProvider, Callable[[], Awaitable[None]]]:
    """OpenRouter provider capturing raw response metadata pre-normalization.

    Builds a run-scoped client whose HTTP response hook records generation
    ID, served model, charged cost, and router metadata into audit events
    before the SDK normalizes responses (which discards those fields). Only
    metadata fields are recorded: never prompts, response content, or
    credentials. Non-2xx responses are skipped; unparseable bodies record
    unavailable fields without failing the request. Retry and timeout
    behavior keeps the client defaults.
    """
    metadata_header = {OPENROUTER_METADATA_HEADER: OPENROUTER_METADATA_OPT_IN}

    async def _record_response(response: Any) -> None:
        try:
            url = str(getattr(getattr(response, "request", None), "url", ""))
            if not url.startswith(OPENROUTER_BASE_URL):
                return
            if not 200 <= int(getattr(response, "status_code", 0)) < 300:
                return
            read = getattr(response, "aread", None)
            if callable(read):
                await read()
            try:
                payload = response.json()
            except Exception:
                payload = None
            response_sink(
                _provider_response_record(
                    payload, getattr(response, "headers", None)
                )
            )
        except Exception:
            response_sink(_provider_response_record(None, None))

    http_client_kwargs: dict[str, Any] = {
        "event_hooks": {"response": [_record_response]}
    }
    if http_transport is not None:
        http_client_kwargs["transport"] = http_transport
    http_client = DefaultAsyncHttpxClient(**http_client_kwargs)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        http_client=http_client,
        default_headers=metadata_header,
    )
    provider = OpenAIProvider(openai_client=client, use_responses=use_responses)

    async def _aclose() -> None:
        await client.close()

    return provider, _aclose


def resolve_model_runtime(
    *,
    model: str,
    model_provider: Optional[str] = None,
    model_transport: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    response_sink: Optional[Callable[[dict[str, Any]], None]] = None,
    http_transport: Any = None,
    router_cost_tier: Optional[str] = None,
) -> ModelRuntime:
    """Resolve provider/transport/effort without making any model request."""
    if not model or not str(model).strip():
        raise ValueError("model must be a non-empty model ID")

    provider = (model_provider or os.environ.get("FLIGHT_LOG_MODEL_PROVIDER") or "auto").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown model_provider {provider!r}; expected one of {PROVIDERS}"
        )
    if provider == "auto":
        provider = "openai"

    transport = (model_transport or "auto").strip().lower()
    if transport not in TRANSPORTS:
        raise ValueError(
            f"unknown model_transport {transport!r}; expected one of {TRANSPORTS}"
        )

    effort = (
        reasoning_effort
        or os.environ.get("FLIGHT_LOG_REASONING_EFFORT")
        or "high"
    ).strip().lower()
    if effort not in REASONING_EFFORTS:
        raise ValueError(
            f"unknown reasoning_effort {effort!r}; expected one of {REASONING_EFFORTS}"
        )

    tier_raw = (
        router_cost_tier or os.environ.get("FLIGHT_LOG_ROUTER_COST_TIER") or None
    )
    tier = tier_raw.strip().lower() if tier_raw else None
    if tier is not None and tier not in ROUTER_COST_TIERS:
        raise ValueError(
            f"unknown router_cost_tier {tier!r}; expected one of {ROUTER_COST_TIERS}"
        )

    if provider == "openai":
        # Today's default path: no RunConfig, so Runner.run behaves exactly
        # as before (SDK default provider and Responses transport).
        resolved_transport = transport if transport != "auto" else "responses"
        return ModelRuntime(
            provider="openai",
            model=model,
            reasoning_effort=effort,
            transport=resolved_transport,
            run_config=None,
            # Hosted web search is a Responses-only tool on every provider.
            web_search_supported=(resolved_transport == "responses"),
            router_cost_tier=tier,
        )

    key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "model_provider 'openrouter' requires an API key: pass "
            "openrouter_api_key or set OPENROUTER_API_KEY"
        )
    resolved_transport = transport if transport != "auto" else "chat"
    provider_instance, aclose = build_openrouter_provider(
        key,
        response_sink or (lambda _fields: None),
        use_responses=(resolved_transport == "responses"),
        http_transport=http_transport,
    )
    return ModelRuntime(
        provider="openrouter",
        model=model,
        reasoning_effort=effort,
        transport=resolved_transport,
        run_config=RunConfig(model_provider=provider_instance),
        # Hosted web search is a Responses-only tool; the chat-completions
        # path cannot serve it.
        web_search_supported=(resolved_transport == "responses"),
        aclose=aclose,
        router_cost_tier=tier,
    )


def served_model_from_result(result: Any) -> str:
    """Best-effort served-model identity; never inferred.

    Returns the model ID exposed by the run result, or ``"unavailable"``
    when the result exposes nothing usable. Per-request served values are
    not exposed by the SDK and are therefore always unavailable.
    """
    direct = getattr(result, "model", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    raw_responses: Any = getattr(result, "raw_responses", None)
    candidates = (
        raw_responses if isinstance(raw_responses, (list, tuple)) else ()
    )
    for response in candidates:
        exposed = getattr(response, "model", None)
        if isinstance(exposed, str) and exposed.strip():
            return exposed.strip()
    return UNAVAILABLE
