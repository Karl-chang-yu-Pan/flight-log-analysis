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


def logged_signals_from_inventory(inventory: dict[str, Any]) -> set[str]:
    """Set of ``topic.field`` references that are actually logged.

    Single owner of the derivation that previously lived inline in
    ``BindingIndex.__init__`` and in
    ``verification_plan._logged_signals_from_inventory``.
    """
    out: set[str] = set()
    for topic, fields in (inventory.get("topic_fields") or {}).items():
        if not isinstance(topic, str):
            continue
        for field in fields or []:
            if isinstance(field, str) and field:
                out.add(f"{topic}.{field}")
    return out


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
        *,
        helper_expressions: Iterable[Any] = (),
        source_assignments: Iterable[Any] = (),
    ) -> None:
        schema = load_px4_msg_schema(source_from_inventory(inventory))
        topic_fields = inventory.get("topic_fields") or {}
        available_topics = set(inventory.get("available_topics") or topic_fields)

        self.logged_signals: set[str] = logged_signals_from_inventory(inventory)
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

        # Backward-slice indexes — built once so per-terminal walks
        # (slice_for_terminal, bindings_reaching) and downstream consumers
        # (verification_graph.backward_binding_slice) share one index set
        # instead of rebuilding defaultdicts on each call.
        self._by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for binding in bindings:
            logged_signal = normalize_symbol(str(binding.get("logged_signal") or ""))
            target_symbol = normalize_symbol(str(binding.get("target_symbol") or ""))
            if logged_signal:
                self._by_output[logged_signal].append(binding)
            if target_symbol:
                self._by_target[target_symbol].append(binding)

        # Resolved-chain tables: materialize each helper's lowered return
        # expression and each assignment's RHS once, against this index.
        # Downstream consumers (substitute_helpers, slicer) then look up
        # the resolved form in O(1) instead of re-walking the chain per
        # check / per candidate.
        self.helper_resolutions: dict[str, Any] = {}
        self.assignment_resolutions: dict[str, Any] = {}
        parameter_set = set((inventory.get("parameters") or {}).keys())
        helpers_list = list(helper_expressions)
        assignments_list = [
            self._assignment_as_dict(assignment) for assignment in source_assignments
        ]
        if helpers_list:
            self._materialize_helper_resolutions(
                helpers_list, assignments_list, parameter_set
            )
        if assignments_list:
            self._materialize_assignment_resolutions(
                assignments_list, parameter_set
            )

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

    def assignment_resolution_for(self, name: str) -> Any:
        """Return the materialized resolution for ``name``, or None.

        Single lookup used by both the slicer's fast path and the
        predicate parser's symbolic-constant fallback. Wraps the
        ``assignment_resolutions`` dict so callers do not duplicate the
        ``normalize_symbol`` + ``.get`` pair.
        """
        if not isinstance(name, str) or not name:
            return None
        return self.assignment_resolutions.get(normalize_symbol(name))

    def slice_for_terminal(self, terminal: str) -> set[str]:
        """Return the logged signals reachable backward from ``terminal``.

        Delegates the walk to :meth:`bindings_reaching` and projects
        each visited binding's ``logged_signal``. Suitable for use as
        ``prefer=`` in :meth:`resolve`.
        """
        if not terminal:
            return set()
        signals: set[str] = set()
        for binding in self.bindings_reaching(terminal):
            logged_signal = str(binding.get("logged_signal") or "")
            if logged_signal:
                signals.add(logged_signal)
        return signals

    def bindings_reaching(self, terminal: str) -> list[dict[str, Any]]:
        """Return the binding records that contribute to ``terminal``.

        Single owner of the backward walk over ``output_bindings``: the
        ``by_output`` / ``by_target`` indexes are built once at
        construction time and reused. Replaces the standalone
        ``verification_graph.backward_binding_slice`` so the two
        previously-parallel implementations cannot drift.
        """
        if not terminal:
            return []
        terminal_norm = normalize_symbol(terminal)
        seen: set[int] = set()
        selected: list[dict[str, Any]] = []
        frontier = deque(self._by_output.get(terminal_norm, []))
        while frontier:
            binding = frontier.popleft()
            key = id(binding)
            if key in seen:
                continue
            seen.add(key)
            selected.append(binding)
            for symbol in self._source_expression_symbols(
                str(binding.get("source_symbol") or "")
            ):
                frontier.extend(self._by_target.get(symbol, []))
        return selected

    def bindings_writing_prefix(self, prefix: str) -> list[dict[str, Any]]:
        """Return every binding whose target begins with ``{prefix}.``.

        Callers use this to walk struct field writes when a bare struct
        root like ``_mission_item`` appears as an unresolved symbol —
        no direct ``_mission_item = X`` write exists, but many
        ``_mission_item.altitude = ...`` writes do.
        """
        prefix_norm = normalize_symbol(prefix)
        if not prefix_norm:
            return []
        needle = f"{prefix_norm}."
        matches: list[dict[str, Any]] = []
        for target_key, bindings in self._by_target.items():
            if target_key.startswith(needle):
                matches.extend(bindings)
        return matches

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
    def _assignment_as_dict(assignment: Any) -> dict[str, Any]:
        if isinstance(assignment, dict):
            return dict(assignment)  # shallow copy so callers don't mutate
        if hasattr(assignment, "model_dump"):
            return assignment.model_dump(exclude_none=True)
        return dict(vars(assignment))

    @staticmethod
    def _source_expression_symbols(expression: str) -> list[str]:
        # Lazy import to avoid a circular dependency with verification_graph.
        from flight_log_agent.analysis.source_expression import source_expression_names

        return [normalize_symbol(name) for name in source_expression_names(expression)]

    # ------------------------------------------------------------
    # Chain-resolution materialization (Flavor 0 Phase A)
    # ------------------------------------------------------------

    def _materialize_helper_resolutions(
        self,
        helpers: list[Any],
        assignments: list[dict[str, Any]],
        parameter_set: set[str],
    ) -> None:
        """Resolve each helper's lowered body against the index once.

        Runs in iterative passes: each pass resolves helpers whose
        ``helper_calls`` are either external (not in the registry) or
        already resolved in a previous pass, so deeply nested helpers
        inline their callees' resolved forms instead of the unbounded
        original bodies. After 8 passes (a safety bound), any remaining
        helpers are resolved with whatever state is available — this is
        the cycle / over-dependency fallback.
        """
        # Lazy imports — keeps the module-level import graph acyclic.
        from flight_log_agent.analysis.helper_resolution import (
            HelperRegistry,
            substitute_helpers,
        )
        from flight_log_agent.analysis.source_slicer import slice_expression

        working = [self._helper_as_dict(h) for h in helpers]
        # Only consider entries with a name and a body to lower.
        candidates = [
            h for h in working
            if h.get("name") and (
                h.get("lowered_return_expression") or h.get("return_expression")
            )
        ]
        registered_names: set[str] = set()
        for helper in working:
            name = helper.get("name")
            if not name:
                continue
            registered_names.add(name)
            short = name.split("::")[-1]
            if short:
                registered_names.add(short)

        resolved: set[str] = set()

        def resolve_one(helper: dict[str, Any]) -> None:
            name = helper["name"]
            body = helper.get("lowered_return_expression") or helper.get("return_expression")
            registry = HelperRegistry(working)
            substituted = substitute_helpers(str(body), registry=registry, env={})
            result = slice_expression(
                substituted,
                source_assignments=assignments,
                logged_signals=self.logged_signals,
                parameters=parameter_set,
                binding_index=self,
            )
            self.helper_resolutions[name] = result
            short = name.split("::")[-1]
            if short and short != name and short not in self.helper_resolutions:
                self.helper_resolutions[short] = result
            # Update working entry so subsequent passes substitute the
            # resolved form (terminal-level expression) instead of the
            # original nested body.
            helper["lowered_return_expression"] = result.expression
            resolved.add(name)
            if short:
                resolved.add(short)

        for _pass in range(8):
            progress = False
            for helper in candidates:
                name = helper["name"]
                if name in resolved:
                    continue
                callees = helper.get("helper_calls") or []
                deps_ready = True
                for callee in callees:
                    short = str(callee).split("::")[-1]
                    in_registry = (callee in registered_names) or (short in registered_names)
                    if not in_registry:
                        continue  # external call — leave it for the slicer to flag
                    if (callee not in resolved) and (short not in resolved):
                        deps_ready = False
                        break
                if not deps_ready:
                    continue
                resolve_one(helper)
                progress = True
            if not progress:
                break

        # Cycle / over-dependency fallback: any helper not yet resolved
        # is resolved now without waiting for dependency readiness. The
        # result may carry blockers, which is the correct partial form.
        for helper in candidates:
            if helper["name"] in resolved:
                continue
            resolve_one(helper)

    def _materialize_assignment_resolutions(
        self,
        assignments: list[dict[str, Any]],
        parameter_set: set[str],
    ) -> None:
        """Resolve unconditional single-write targets' RHS once.

        Skipped:
        - Multi-write targets — the slicer builds a conditional from the
          per-write control_predicates.
        - Single-write targets whose only write carries control_predicates
          — the slicer wraps them in a ternary with the original symbol
          as the fallback, which the materialized resolution would erase.
        """
        from flight_log_agent.analysis.source_slicer import slice_expression

        writes_per_target: dict[str, int] = {}
        for assignment in assignments:
            target = assignment.get("target")
            if not target:
                continue
            normalized_target = normalize_symbol(str(target))
            if not normalized_target:
                continue
            writes_per_target[normalized_target] = writes_per_target.get(normalized_target, 0) + 1

        for assignment in assignments:
            target = assignment.get("target")
            if not target:
                continue
            normalized_target = normalize_symbol(str(target))
            if not normalized_target or normalized_target in self.assignment_resolutions:
                continue
            if writes_per_target.get(normalized_target, 0) != 1:
                continue  # multi-write — let the slicer build a conditional
            if assignment.get("control_predicates"):
                continue  # gated single write — slicer's ternary preserves the fallback
            rhs = assignment.get("expression")
            if not rhs or not isinstance(rhs, str):
                continue
            result = slice_expression(
                rhs,
                source_assignments=assignments,
                logged_signals=self.logged_signals,
                parameters=parameter_set,
                binding_index=self,
            )
            self.assignment_resolutions[normalized_target] = result

    @staticmethod
    def _helper_as_dict(helper: Any) -> dict[str, Any]:
        if isinstance(helper, dict):
            return dict(helper)
        if hasattr(helper, "model_dump"):
            return helper.model_dump(exclude_none=True)
        return dict(vars(helper))
