"""Compiled graph-native value evaluation for mechanism DAGs.

One :class:`DAGValueProgram` owns immutable graph topology and compiled local
expressions. A bound :class:`DAGValueSession` owns run-specific parameters and
memoized timestamp values. Producer expressions are never spliced into source
text; every non-local value still comes through a DAG edge.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Callable, Literal, Optional

from flight_log_agent.analysis.source_expression import (
    CompiledSourceExpression,
    SourceExpressionError,
    compile_source_expression,
)


ValueStatus = Literal["value", "inactive", "unresolved"]
ActivityStatus = Literal["active", "inactive", "unknown"]
SampleResolver = Callable[[str, float], Optional[Any]]
_PARAMETER_ACCESSOR = re.compile(
    r"^_param_(?P<name>[A-Za-z0-9_]+)\.get(?:\(\))?$"
)


@dataclass(frozen=True)
class DAGValueResult:
    status: ValueStatus
    value: Any = None
    reason: str = ""


@dataclass(frozen=True)
class _CompiledVertex:
    expression: Optional[CompiledSourceExpression]
    compile_error: str
    producers_by_operand: tuple[tuple[str, tuple[str, ...]], ...]
    exact: bool


class DAGValueProgram:
    """Immutable topology and local-expression program for one DAG."""

    def __init__(self, dag: Any) -> None:
        vertices = {vertex.id: vertex for vertex in dag.vertices}
        data: dict[str, list[Any]] = defaultdict(list)
        controls: dict[str, list[str]] = defaultdict(list)
        selections: dict[str, list[str]] = defaultdict(list)
        dependents: dict[str, list[str]] = defaultdict(list)
        for edge in dag.edges:
            dependents[edge.source_id].append(edge.target_id)
            if edge.kind == "data":
                data[edge.target_id].append(edge)
            elif edge.kind == "control":
                controls[edge.target_id].append(edge.source_id)
            elif edge.kind == "selection":
                selections[edge.target_id].append(edge.source_id)

        self.vertices = MappingProxyType(vertices)
        self.data_by_target = MappingProxyType(
            {key: tuple(value) for key, value in data.items()}
        )
        self.controls_by_target = MappingProxyType(
            {key: tuple(dict.fromkeys(value)) for key, value in controls.items()}
        )
        self.selections_by_target = MappingProxyType(
            {key: tuple(dict.fromkeys(value)) for key, value in selections.items()}
        )
        self._logged_signals = MappingProxyType(
            self._derive_leaf_dependencies(
                vertices, dependents, observed_only=True
            )
        )
        self._observable_inputs = MappingProxyType(
            self._derive_leaf_dependencies(
                vertices, dependents, observed_only=True
            )
        )

        compiled: dict[str, _CompiledVertex] = {}
        for vertex in dag.vertices:
            if vertex.kind not in {"operation", "branch"}:
                continue
            expression = self._vertex_expression(vertex)
            by_operand: dict[str, list[str]] = defaultdict(list)
            call_occurrences: list[tuple[str, str]] = []
            call_roles = dict(
                (vertex.metadata or {}).get("source_call_roles") or {}
            )
            for edge in self.data_by_target.get(vertex.id, ()):
                if edge.via and edge.via in call_roles and edge.role.startswith(
                    ("call:", "call-result:")
                ):
                    operand_id = f"call-site:{edge.via}"
                    call_role = call_roles[edge.via] or {}
                    call_expression = str(
                        call_role.get("result")
                        or call_role.get("call")
                        or ""
                    )
                    call_expression = self._expression_spelling(
                        call_expression
                    )
                    if call_expression and not any(
                        existing_id == operand_id
                        for existing_id, _expression in call_occurrences
                    ):
                        call_occurrences.append(
                            (operand_id, call_expression)
                        )
                    by_operand[operand_id].append(edge.source_id)
                    continue
                role = self._source_role(vertex, edge)
                if role:
                    spelling = self._expression_spelling(role)
                    if not spelling.endswith("()") and f"{spelling}()" in expression:
                        spelling = f"{spelling}()"
                    by_operand[spelling].append(edge.source_id)
            operands = tuple(
                (role, tuple(dict.fromkeys(producer_ids)))
                for role, producer_ids in by_operand.items()
            )
            expression_exact = bool(
                (vertex.metadata or {}).get("expression_inputs_exact", False)
            )
            try:
                if not expression_exact:
                    raise SourceExpressionError(
                        "source expression dependencies are not parser-exact"
                    )
                local_expression = compile_source_expression(
                    expression,
                    (
                        role
                        for role, _producer_ids in operands
                        if not role.startswith("call-site:")
                    ),
                    occurrence_operands=call_occurrences,
                )
                compile_error = ""
            except SourceExpressionError as exc:
                local_expression = None
                compile_error = str(exc)
            compiled[vertex.id] = _CompiledVertex(
                expression=local_expression,
                compile_error=compile_error,
                producers_by_operand=operands,
                exact=expression_exact,
            )
        self.compiled_vertices = MappingProxyType(compiled)

    def bind(
        self,
        *,
        parameter_values: Optional[dict[str, Any]] = None,
        enum_values: Optional[dict[str, Any]] = None,
        sample_resolver: Optional[SampleResolver] = None,
    ) -> "DAGValueSession":
        return DAGValueSession(
            self,
            parameter_values=parameter_values,
            enum_values=enum_values,
            sample_resolver=sample_resolver,
        )

    def logged_signals_for(self, vertex_id: str) -> tuple[str, ...]:
        return self._logged_signals.get(vertex_id, ())

    def observable_inputs_for(self, vertex_id: str) -> tuple[str, ...]:
        """Named evidence leaves that runtime data may bind exactly."""
        return self._observable_inputs.get(vertex_id, ())

    @staticmethod
    def _derive_leaf_dependencies(
        vertices: dict[str, Any],
        dependents: dict[str, list[str]],
        *,
        observed_only: bool,
    ) -> dict[str, tuple[str, ...]]:
        signals: dict[str, set[str]] = {vertex_id: set() for vertex_id in vertices}
        pending: deque[str] = deque()
        for vertex_id, vertex in vertices.items():
            observed = (
                vertex.sub_kind == "logged_signal"
                and str((vertex.metadata or {}).get("observation") or "observed")
                == "observed"
            )
            if (
                vertex.kind == "evidence"
                and vertex.signal_name
                and (observed if observed_only else True)
            ):
                signals[vertex_id].add(str(vertex.signal_name))
                pending.append(vertex_id)
        while pending:
            source_id = pending.popleft()
            source_signals = signals[source_id]
            for target_id in dependents.get(source_id, ()):
                before = len(signals[target_id])
                signals[target_id].update(source_signals)
                if len(signals[target_id]) != before:
                    pending.append(target_id)
        return {
            vertex_id: tuple(sorted(values))
            for vertex_id, values in signals.items()
        }

    @classmethod
    def _vertex_expression(cls, vertex: Any) -> str:
        metadata = vertex.metadata or {}
        expression_ref = metadata.get("source_expression_ref") or {}
        expression = str(
            expression_ref.get("lowered_text")
            or expression_ref.get("text")
            or vertex.lowered_expression
            or metadata.get("source_expression")
            or (
                vertex.predicate_raw or vertex.predicate_lowered or ""
                if vertex.kind == "branch"
                else vertex.expression or vertex.lowered_expression or ""
            )
        )
        return cls._expression_spelling(expression)

    @staticmethod
    def _expression_spelling(value: str) -> str:
        return str(value or "").replace("->", ".").replace("::", ".")

    @staticmethod
    def _source_role(vertex: Any, edge: Any) -> str:
        role = str(edge.role or "")
        call_roles = dict((vertex.metadata or {}).get("source_call_roles") or {})
        if edge.via and role.startswith("call:"):
            return str((call_roles.get(edge.via) or {}).get("call") or role)
        if edge.via and role.startswith("call-result:"):
            return str((call_roles.get(edge.via) or {}).get("result") or role)
        if role.startswith("branch:"):
            return role.removeprefix("branch:")
        return role


class DAGValueSession:
    """Run-specific memoization over one immutable DAG value program."""

    def __init__(
        self,
        program: DAGValueProgram,
        *,
        parameter_values: Optional[dict[str, Any]] = None,
        enum_values: Optional[dict[str, Any]] = None,
        sample_resolver: Optional[SampleResolver] = None,
    ) -> None:
        self.program = program
        self.parameters = MappingProxyType(
            {
                str(name).upper(): value
                for name, value in (parameter_values or {}).items()
            }
        )
        self.enums = MappingProxyType(dict(enum_values or {}))
        self.sample_resolver = sample_resolver
        self._value_cache: dict[
            tuple[str, Optional[float]], DAGValueResult
        ] = {}
        self._sample_cache: dict[tuple[str, float], Optional[Any]] = {}

    def evaluate(
        self, vertex_id: str, timestamp: Optional[float]
    ) -> DAGValueResult:
        return self._evaluate_vertex(vertex_id, timestamp, frozenset())

    def evaluate_many(
        self, vertex_ids: list[str] | tuple[str, ...], timestamp: Optional[float]
    ) -> dict[str, DAGValueResult]:
        return {
            vertex_id: self.evaluate(vertex_id, timestamp)
            for vertex_id in dict.fromkeys(vertex_ids)
        }

    def release_timestamp_values(self) -> None:
        """End one shared timestamp batch; retain only static evaluation results.

        This evaluator has no persistent state interpreter. Dynamic entries
        memoize pure evaluation, not vehicle state or signal history.
        """
        self._value_cache = {key: value for key, value in self._value_cache.items() if key[1] is None}
        self._sample_cache.clear()

    def _evaluate_vertex(
        self,
        vertex_id: str,
        timestamp: Optional[float],
        active: frozenset[str],
    ) -> DAGValueResult:
        cache_key = (vertex_id, timestamp)
        cached = self._value_cache.get(cache_key)
        if cached is not None:
            return cached
        if vertex_id in active:
            return DAGValueResult("unresolved", reason="cyclic value dependency")
        vertex = self.program.vertices.get(vertex_id)
        if vertex is None:
            return DAGValueResult("unresolved", reason="missing DAG vertex")
        next_active = active | {vertex_id}

        if vertex.kind == "evidence":
            result = self._evaluate_evidence(vertex, timestamp, next_active)
        elif vertex.kind == "operation":
            activity = self._operation_activity(vertex_id, timestamp, next_active)
            if activity == "inactive":
                result = DAGValueResult("inactive", reason="writer is inactive")
            elif activity == "unknown":
                result = DAGValueResult(
                    "unresolved", reason="writer reachability is unresolved"
                )
            else:
                result = self._evaluate_expression_vertex(
                    vertex, timestamp, next_active
                )
        elif vertex.kind == "branch":
            result = self._evaluate_branch(vertex, timestamp, next_active)
        else:
            result = DAGValueResult("unresolved", reason="unsupported vertex kind")
        self._value_cache[cache_key] = result
        return result

    def _evaluate_evidence(
        self,
        vertex: Any,
        timestamp: Optional[float],
        active: frozenset[str],
    ) -> DAGValueResult:
        metadata = vertex.metadata or {}
        if vertex.sub_kind == "logged_signal" and vertex.signal_name:
            if timestamp is None or self.sample_resolver is None:
                return DAGValueResult(
                    "unresolved", reason="logged value requires a timestamp"
                )
            signal = str(vertex.signal_name)
            sample_key = (signal, timestamp)
            if sample_key not in self._sample_cache:
                self._sample_cache[sample_key] = self.sample_resolver(
                    signal, timestamp
                )
            value = self._sample_cache[sample_key]
            if value is None:
                return DAGValueResult(
                    "unresolved", reason=f"{signal} is not evaluable"
                )
            return DAGValueResult("value", value=value)
        if metadata.get("value") is not None:
            return DAGValueResult("value", value=metadata["value"])
        name = str(vertex.signal_name or "")
        if vertex.sub_kind == "parameter" and name.upper() in self.parameters:
            return DAGValueResult("value", value=self.parameters[name.upper()])
        if name in self.enums:
            return DAGValueResult("value", value=self.enums[name])
        parameter_match = _PARAMETER_ACCESSOR.fullmatch(name)
        if parameter_match:
            parameter_name = parameter_match.group("name").upper()
            if parameter_name in self.parameters:
                return DAGValueResult(
                    "value", value=self.parameters[parameter_name]
                )
        if vertex.sub_kind == "helper_parameter":
            incoming = self.program.data_by_target.get(vertex.id, ())
            if incoming:
                return self._select_producer(
                    [edge.source_id for edge in incoming], timestamp, active
                )
        return DAGValueResult(
            "unresolved", reason=f"unresolved evidence {name or vertex.id}"
        )

    def _evaluate_branch(
        self,
        vertex: Any,
        timestamp: Optional[float],
        active: frozenset[str],
    ) -> DAGValueResult:
        if vertex.feasibility_verdict == "always_true":
            return DAGValueResult("value", value=True)
        if vertex.feasibility_verdict == "always_false":
            return DAGValueResult("value", value=False)
        if timestamp is not None and vertex.active_windows:
            if any(start <= timestamp <= end for start, end in vertex.active_windows):
                return DAGValueResult("value", value=True)
            domain = (vertex.metadata or {}).get("evaluation_domain") or []
            if len(domain) == 2 and float(domain[0]) <= timestamp <= float(domain[1]):
                return DAGValueResult("value", value=False)
        return self._evaluate_expression_vertex(vertex, timestamp, active)

    def _operation_activity(
        self,
        vertex_id: str,
        timestamp: Optional[float],
        active: frozenset[str],
    ) -> ActivityStatus:
        metadata = self.program.vertices[vertex_id].metadata or {}
        if (
            metadata.get("synthetic_boundary_transfer")
            and metadata.get("boundary_direction") == "subscribe"
        ):
            # An input boundary transfer reads the member's value from the
            # logged topic. The logged signal already observes that value over
            # the whole timeline (hold-last between samples), so the source-code
            # control flow around the copy statement does not gate the
            # observation — the log records what the member held regardless.
            return "active"
        unknown = not bool(
            (metadata.get("reachability") or {}).get("exact", False)
        )
        for branch_id in self.program.controls_by_target.get(vertex_id, ()):
            result = self._evaluate_vertex(branch_id, timestamp, active)
            if result.status != "value":
                unknown = True
            elif not bool(result.value):
                return "inactive"
        return "unknown" if unknown else "active"

    def _evaluate_expression_vertex(
        self,
        vertex: Any,
        timestamp: Optional[float],
        active: frozenset[str],
    ) -> DAGValueResult:
        local = self.program.compiled_vertices.get(vertex.id)
        if local is None or local.expression is None or not local.exact:
            return DAGValueResult(
                "unresolved",
                reason=(local.compile_error if local is not None else "vertex has no expression"),
            )
        producers = dict(local.producers_by_operand)

        def resolve_operand(role: str) -> Any:
            producer_ids = list(producers.get(role, ()))
            if not producer_ids:
                raise SourceExpressionError(f"missing DAG operand {role}")
            selected = self._select_producer(producer_ids, timestamp, active)
            if selected.status != "value":
                raise SourceExpressionError(
                    f"{role}: {selected.reason or selected.status}"
                )
            return selected.value

        try:
            return DAGValueResult("value", value=local.expression.evaluate(resolve_operand))
        except (SourceExpressionError, TypeError, ValueError, ZeroDivisionError) as exc:
            return DAGValueResult("unresolved", reason=str(exc))

    def _select_producer(
        self,
        producer_ids: list[str],
        timestamp: Optional[float],
        active: frozenset[str],
    ) -> DAGValueResult:
        candidates: list[tuple[str, DAGValueResult]] = []
        for producer_id in dict.fromkeys(producer_ids):
            if producer_id not in self.program.vertices:
                continue
            candidates.append(
                (
                    producer_id,
                    self._evaluate_vertex(producer_id, timestamp, active),
                )
            )

        active_values = [item for item in candidates if item[1].status == "value"]
        unknown_ids = [
            producer_id
            for producer_id, result in candidates
            if result.status == "unresolved"
        ]
        if not active_values:
            if unknown_ids:
                return DAGValueResult(
                    "unresolved", reason="all reaching producers are unresolved"
                )
            return DAGValueResult("inactive", reason="no reaching producer is active")
        if len(active_values) == 1 and not unknown_ids:
            return active_values[0][1]

        selected = self._latest_ordered_producer(active_values, unknown_ids)
        if selected is not None:
            return selected
        values = {repr(item[1].value) for item in active_values}
        if len(values) == 1 and not unknown_ids:
            return active_values[0][1]
        return DAGValueResult(
            "unresolved", reason="multiple reaching producers remain possible"
        )

    def _latest_ordered_producer(
        self,
        active_values: list[tuple[str, DAGValueResult]],
        unknown_ids: list[str],
    ) -> Optional[DAGValueResult]:
        all_ids = [item[0] for item in active_values] + unknown_ids
        operations = [self.program.vertices[vertex_id] for vertex_id in all_ids]
        if not operations or any(vertex.kind != "operation" for vertex in operations):
            return None
        scopes = {
            (
                str(((vertex.metadata or {}).get("target_scope") or {}).get("file") or ""),
                str(((vertex.metadata or {}).get("target_scope") or {}).get("callable") or ""),
            )
            for vertex in operations
        }
        if len(scopes) != 1 or any(vertex.line is None for vertex in operations):
            return None
        latest_active = max(
            active_values,
            key=lambda item: self._producer_order(item[0]),
        )
        latest_order = self._producer_order(latest_active[0])
        if any(
            self._producer_order(vertex_id) > latest_order
            for vertex_id in unknown_ids
        ):
            return None
        return latest_active[1]

    def _producer_order(self, vertex_id: str) -> tuple[int, int]:
        vertex = self.program.vertices[vertex_id]
        metadata = vertex.metadata or {}
        target_scope = metadata.get("target_scope") or {}
        return (
            int(target_scope.get("line") or vertex.line or 0),
            int(target_scope.get("source_order") or metadata.get("source_order") or 0),
        )


class DAGValuePlan:
    """Compatibility wrapper for callers that evaluate one root at a time."""

    def __init__(self, dag: Any, root_id: str) -> None:
        self.root_id = root_id
        self.program = DAGValueProgram(dag)

    @property
    def logged_signals(self) -> tuple[str, ...]:
        return self.program.logged_signals_for(self.root_id)

    def evaluate(
        self,
        timestamp: Optional[float],
        *,
        parameter_values: Optional[dict[str, Any]] = None,
        enum_values: Optional[dict[str, Any]] = None,
        sample_resolver: Optional[SampleResolver] = None,
    ) -> DAGValueResult:
        session = self.program.bind(
            parameter_values=parameter_values,
            enum_values=enum_values,
            sample_resolver=sample_resolver,
        )
        return session.evaluate(self.root_id, timestamp)
