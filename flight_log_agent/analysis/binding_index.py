"""Unified signal binding index.

Single owner of the alias + suffix + slice logic that previously lived
duplicated across ``runner_core.SignalCanonicalizer``,
``verification_plan.SignalResolver``, and
``verification_graph.source_signal_bindings``.

The index keeps every alias even when multiple logged signals share it, so
ambiguity is reported instead of silently dropped. Callers that know which
terminal output they care about can pass ``prefer=`` (typically the backward
binding slice of the candidate's primary output) to tiebreak when several
candidates share a suffix — which is the case that produced the
``position_setpoint.cruising_speed`` ambiguity in the 6/14 reports.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Optional

from flight_log_agent.models import VerificationSignalResolution
from flight_log_agent.px4.msg_schema import load_px4_msg_schema
from flight_log_agent.px4.source_snapshot import source_from_inventory
from flight_log_agent.symbols import normalize_symbol


def _get(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def assignment_path_symbols(path: list[dict[str, Any]] | None) -> list[str]:
    """Return dotted symbol references found in an assignment-path list.

    Mirrors the policy from ``verification_plan.assignment_path_symbols`` so
    consumers keep getting the same set of alias seeds.
    """
    import re

    values: list[str] = []
    for step in path or []:
        if not isinstance(step, dict):
            continue
        for key in ("source", "target", "source_symbol", "target_symbol", "expression"):
            value = step.get(key)
            if isinstance(value, str) and re.fullmatch(
                r"[_A-Za-z][_A-Za-z0-9]*(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)+",
                value,
            ):
                values.append(value)
    return values


class BindingIndex:
    """Resolve symbol references against logged signals and output bindings.

    Construction is cheap relative to the cost of building the alias map.
    Instances are intended to be created once per candidate evaluation
    (since the inputs — inventory + output bindings — do not change inside
    one evaluation pass).
    """

    def __init__(
        self,
        inventory: dict[str, Any],
        output_bindings: Iterable[Any],
    ) -> None:
        schema = load_px4_msg_schema(source_from_inventory(inventory))
        topic_fields = inventory.get("topic_fields") or {}
        available_topics = set(inventory.get("available_topics") or topic_fields)

        self.logged_signals: set[str] = {
            f"{topic}.{field}"
            for topic, fields in topic_fields.items()
            for field in fields
        }
        self.schema_signals: set[str] = {
            f"{topic}.{field}"
            for topic, fields in schema.items()
            for field in fields
        }

        aliases: dict[str, set[str]] = {}
        binding_suffix_aliases: dict[str, set[str]] = {}
        unavailable_aliases: dict[str, set[str]] = {}
        symbol_bindings: dict[str, str] = {}

        for signal in self.logged_signals:
            aliases.setdefault(normalize_symbol(signal), set()).add(signal)

        bindings: list[dict[str, Any]] = []
        for binding in output_bindings:
            binding_dict = self._binding_as_dict(binding)
            bindings.append(binding_dict)
            logged_signal = str(binding_dict.get("logged_signal") or "")
            if not logged_signal:
                continue

            topic = logged_signal.split(".", 1)[0]
            binding_is_logged = logged_signal in self.logged_signals or (
                topic in available_topics and topic not in topic_fields
            )
            target_aliases = aliases if binding_is_logged else unavailable_aliases

            for value in (
                logged_signal,
                binding_dict.get("source_symbol"),
                binding_dict.get("target_symbol"),
                *assignment_path_symbols(binding_dict.get("assignment_path")),
            ):
                normalized = normalize_symbol(str(value or ""))
                if not normalized:
                    continue
                target_aliases.setdefault(normalized, set()).add(logged_signal)

            # Suffix aliases only come from the logged_signal itself.
            # Building them from source / target / assignment-path symbols
            # would map unrelated dependencies (e.g. a terminal whose source
            # symbol happens to end in `.cruising_speed`) onto the same
            # suffix as the actual `.cruising_speed` logged signal, which
            # creates spurious ambiguity at resolve time.
            logged_parts = normalize_symbol(logged_signal).split(".")
            for index in range(1, len(logged_parts)):
                suffix = ".".join(logged_parts[index:])
                binding_suffix_aliases.setdefault(suffix, set()).add(logged_signal)

            # Per-binding source_symbol → logged_signal map (legacy
            # source_signal_bindings consumers expect a flat str → str dict
            # for control-predicate lowering and graph walks).
            normalized_logged = normalize_symbol(logged_signal)
            for source, logged in (binding_dict.get("symbol_bindings") or {}).items():
                raw_source = str(source or "").strip()
                normalized_source = normalize_symbol(raw_source)
                normalized_target = normalize_symbol(str(logged or ""))
                if raw_source and normalized_target:
                    symbol_bindings[raw_source] = normalized_target
                if normalized_source and normalized_target:
                    symbol_bindings[normalized_source] = normalized_target
            for symbol in (
                str(binding_dict.get("target_symbol") or ""),
                normalize_symbol(str(binding_dict.get("target_symbol") or "")),
            ):
                if symbol and normalized_logged:
                    symbol_bindings[symbol] = normalized_logged

        self.aliases = aliases
        self.binding_suffix_aliases = binding_suffix_aliases
        self.unavailable_aliases = unavailable_aliases
        self.symbol_bindings = symbol_bindings
        self._bindings = bindings

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def is_known(self, reference: str) -> bool:
        """Whether ``reference`` could plausibly refer to a logged signal."""
        normalized = normalize_symbol(reference)
        suffix = normalized.split(".", 1)[1] if "." in normalized else ""
        return bool(
            normalized in self.logged_signals
            or normalized in self.schema_signals
            or normalized in self.aliases
            or normalized in self.unavailable_aliases
            or suffix in self.binding_suffix_aliases
        )

    def resolve(
        self,
        reference: str,
        *,
        prefer: Iterable[str] = (),
    ) -> VerificationSignalResolution:
        """Resolve ``reference`` to a logged signal.

        ``prefer`` is a set of logged signals (typically from
        :meth:`slice_for_terminal`) used as a tiebreaker when the reference
        matches multiple candidates by suffix or alias. When the prefer set
        narrows the candidates to exactly one, the resolution is
        ``"resolved"`` instead of ``"ambiguous"``.
        """
        normalized = normalize_symbol(reference)
        candidates: set[str] = set(self.aliases.get(normalized, set()))
        if normalized in self.logged_signals:
            candidates.add(normalized)
        if not candidates and "." in normalized:
            suffix = normalized.split(".", 1)[1]
            candidates.update(self.binding_suffix_aliases.get(suffix, set()))

        prefer_set = {p for p in prefer if p}
        if len(candidates) > 1 and prefer_set:
            biased = candidates & prefer_set
            if biased:
                candidates = biased

        ordered = sorted(candidates)
        if len(ordered) == 1:
            return VerificationSignalResolution(
                original=reference,
                status="resolved",
                resolved=ordered[0],
                candidates=ordered,
            )
        if len(ordered) > 1:
            return VerificationSignalResolution(
                original=reference,
                status="ambiguous",
                candidates=ordered,
                reason="multiple deterministic logged-signal matches",
            )
        unavailable = sorted(self.unavailable_aliases.get(normalized, set()))
        if normalized in self.schema_signals or unavailable:
            return VerificationSignalResolution(
                original=reference,
                status="unresolved",
                candidates=unavailable or [normalized],
                reason="deterministic source/schema match is not present in the log",
            )
        return VerificationSignalResolution(
            original=reference,
            status="unresolved",
            reason="no deterministic logged output-binding match",
        )

    def canonicalize(
        self,
        reference: Optional[str],
        *,
        prefer: Iterable[str] = (),
    ) -> Optional[str]:
        """Return the canonical logged signal for ``reference``.

        Returns the input unchanged when no unique resolution is available;
        this matches the previous ``SignalCanonicalizer.canonicalize``
        contract used by ``runner_core``.
        """
        if not reference:
            return reference
        resolution = self.resolve(reference, prefer=prefer)
        if resolution.status == "resolved" and resolution.resolved:
            return resolution.resolved
        return reference

    def slice_for_terminal(self, terminal: str) -> set[str]:
        """Return the logged signals reachable backward from ``terminal``.

        Mirrors ``verification_graph.backward_binding_slice`` but returns
        the *set of logged signals* covered by the slice, suitable for use
        as ``prefer=`` in :meth:`resolve`.
        """
        if not terminal:
            return set()
        terminal_norm = normalize_symbol(terminal)
        by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for binding in self._bindings:
            logged_signal = normalize_symbol(str(binding.get("logged_signal") or ""))
            target_symbol = normalize_symbol(str(binding.get("target_symbol") or ""))
            if logged_signal:
                by_output[logged_signal].append(binding)
            if target_symbol:
                by_target[target_symbol].append(binding)

        seen: set[int] = set()
        signals: set[str] = set()
        frontier = deque(by_output.get(terminal_norm, []))
        while frontier:
            binding = frontier.popleft()
            key = id(binding)
            if key in seen:
                continue
            seen.add(key)
            logged_signal = str(binding.get("logged_signal") or "")
            if logged_signal:
                signals.add(logged_signal)
            for symbol in self._source_expression_symbols(
                str(binding.get("source_symbol") or "")
            ):
                frontier.extend(by_target.get(symbol, []))
        return signals

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    @staticmethod
    def _binding_as_dict(binding: Any) -> dict[str, Any]:
        if isinstance(binding, dict):
            return binding
        if hasattr(binding, "model_dump"):
            return binding.model_dump(exclude_none=True)
        return dict(vars(binding))

    @staticmethod
    def _source_expression_symbols(expression: str) -> list[str]:
        # Lazy import to avoid a circular dependency with verification_graph.
        from flight_log_agent.analysis.source_expression import source_expression_names

        return [normalize_symbol(name) for name in source_expression_names(expression)]
