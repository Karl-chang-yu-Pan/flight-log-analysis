"""Cumulative pre-call provider budget guard (offline-safe).

Enforces configured call/token/cost/wall limits BEFORE the next
provider invocation, so a run can never start work it is not
allowed to finish paying for. All limits are optional; anything
left as None is disabled. No baked-in model pricing exists:
cost enforcement requires user-supplied prices.

Completed calls only are counted: a failed invocation records
nothing, which keeps the ``max N calls`` boundary crisp (N
completed calls allowed, call N+1 blocked). Token/cost limits
are necessarily evaluated on completed usage — this guard
stops before the next call, it never kills a request in
flight. Boundary semantics are ``>=`` everywhere and pinned
by tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ModelPrices:
    """User-supplied USD-per-token prices for one model string."""

    input_usd_per_token: float = 0.0
    output_usd_per_token: float = 0.0
    cached_input_usd_per_token: float = 0.0


@dataclass(frozen=True)
class ProviderBudget:
    """Optional per-run limits. None disables that dimension."""

    max_provider_calls: Optional[int] = None
    max_total_input_tokens: Optional[int] = None
    max_total_output_tokens: Optional[int] = None
    max_total_cost_usd: Optional[float] = None
    max_wall_seconds: Optional[float] = None
    model_prices: Optional[Mapping[str, ModelPrices]] = None

    @property
    def enabled(self) -> bool:
        """Whether any dimension is configured at all."""
        return any(
            limit is not None
            for limit in (
                self.max_provider_calls,
                self.max_total_input_tokens,
                self.max_total_output_tokens,
                self.max_total_cost_usd,
                self.max_wall_seconds,
            )
        )


@dataclass
class ProviderCallRecord:
    """One completed provider invocation, for accounting only."""

    role: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0


class BudgetExceeded(Exception):
    """A configured budget tripped before the next provider call.

    Operational abort only: carries the tripped dimension, the
    configured limit, and the observed cumulative state. Never
    authorizes, proves, or classifies anything diagnostic, and
    carries no secrets.
    """

    def __init__(
        self,
        *,
        dimension: str,
        limit: Any,
        observed: Any,
        role: str,
        provider_calls: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        elapsed_s: float,
    ) -> None:
        super().__init__(
            f"provider budget exceeded: {dimension} observed "
            f"{observed} reached limit {limit} "
            f"(role={role}, calls={provider_calls})"
        )
        self.dimension = dimension
        self.limit = limit
        self.observed = observed
        self.role = role
        self.provider_calls = provider_calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.elapsed_s = elapsed_s


@dataclass
class ProviderBudgetUsage:
    """Cumulative per-run budget state. Operational only.

    One instance per analysis run, threaded explicitly through
    provider call sites (never module-global, never persisted
    as semantic state, never read by proof/authority logic).
    """

    calls: list[ProviderCallRecord] = field(default_factory=list)
    run_start_monotonic: float = field(default_factory=time.monotonic)

    @property
    def provider_calls(self) -> int:
        """Completed provider invocations so far."""
        return len(self.calls)

    @property
    def total_input_tokens(self) -> int:
        """Billed input tokens include cached tokens (they are
        billed input); cached volume stays separately visible
        on each record."""
        return sum(
            call.input_tokens + call.cached_input_tokens
            for call in self.calls
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.calls)

    def _cost_for(
        self,
        model: str,
        usage: Any,
        prices: Optional[Mapping[str, ModelPrices]],
    ) -> float:
        """Price one completed call. Models without a price entry
        contribute zero cost but keep their identity recorded, so
        the gap is auditable rather than silent."""
        if not prices:
            return 0.0
        price = prices.get(str(model or ""))
        if price is None:
            return 0.0
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cached_tokens = int(getattr(usage, "cached_input_tokens", 0) or 0)
        return (
            input_tokens * price.input_usd_per_token
            + output_tokens * price.output_usd_per_token
            + cached_tokens * price.cached_input_usd_per_token
        )

    def record_call(
        self,
        role: str,
        model: str,
        usage: Any,
        *,
        duration_ms: float = 0.0,
        prices: Optional[Mapping[str, ModelPrices]] = None,
    ) -> ProviderCallRecord:
        """Append one completed call's provider-reported usage."""
        record = ProviderCallRecord(
            role=str(role),
            model=str(model or ""),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cached_input_tokens=int(
                getattr(usage, "cached_input_tokens", 0) or 0),
            cost_usd=self._cost_for(model, usage, prices),
            duration_ms=float(duration_ms or 0.0),
        )
        self.calls.append(record)
        return record

    def elapsed_s(self, now: Optional[float] = None) -> float:
        """Monotonic seconds since run start (never wall-clock)."""
        current = time.monotonic() if now is None else now
        return max(0.0, current - self.run_start_monotonic)

    def check_or_raise(
        self,
        budget: ProviderBudget,
        *,
        role: str,
        now: Optional[float] = None,
    ) -> None:
        """Enforce every configured dimension before the next call.

        Raises BudgetExceeded on the first tripped dimension in a
        fixed order (calls, input, output, cost, wall) so the
        reported dimension is deterministic. Returns silently when
        nothing trips, including when the budget has no limits.
        """
        role = str(role)
        elapsed = self.elapsed_s(now)
        calls = self.provider_calls
        input_tokens = self.total_input_tokens
        output_tokens = self.total_output_tokens
        cost = self.total_cost_usd

        def abort(dimension: str, limit: Any, observed: Any) -> None:
            raise BudgetExceeded(
                dimension=dimension,
                limit=limit,
                observed=observed,
                role=role,
                provider_calls=calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                elapsed_s=elapsed,
            )

        if (budget.max_provider_calls is not None
                and calls >= budget.max_provider_calls):
            abort("provider_calls", budget.max_provider_calls, calls)
        if (budget.max_total_input_tokens is not None
                and input_tokens >= budget.max_total_input_tokens):
            abort("input_tokens", budget.max_total_input_tokens,
                  input_tokens)
        if (budget.max_total_output_tokens is not None
                and output_tokens >= budget.max_total_output_tokens):
            abort("output_tokens", budget.max_total_output_tokens,
                  output_tokens)
        if (budget.max_total_cost_usd is not None
                and cost >= budget.max_total_cost_usd):
            abort("cost_usd", budget.max_total_cost_usd, cost)
        if (budget.max_wall_seconds is not None
                and elapsed >= budget.max_wall_seconds):
            abort("wall_seconds", budget.max_wall_seconds, elapsed)
