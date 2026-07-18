"""Graph-native value evaluation for mechanism DAGs.

Storage identity and provenance come exclusively from DAG edges. Source text
is parsed only within one operation to apply its local operators to values
already selected from those edges; producer expressions are never spliced
into consumer strings.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from flight_log_agent.analysis.source_expression import (
    SourceExpressionError,
    evaluate_source_expression,
)


ValueStatus = Literal["value", "inactive", "unresolved"]
ActivityStatus = Literal["active", "inactive", "unknown"]
SampleResolver = Callable[[str, float], Optional[Any]]


@dataclass(frozen=True)
class DAGValueResult:
    status: ValueStatus
    value: Any = None
    reason: str = ""


class DAGValuePlan:
    """Compiled adjacency for one value-producing DAG vertex."""

    def __init__(self, dag: Any, root_id: str) -> None:
        self.root_id = root_id
        self.vertices = {vertex.id: vertex for vertex in dag.vertices}
        self.data_by_target: dict[str, list[Any]] = defaultdict(list)
        self.controls_by_target: dict[str, list[str]] = defaultdict(list)
        self.selections_by_target: dict[str, list[str]] = defaultdict(list)
        for edge in dag.edges:
            if edge.kind == "data":
                self.data_by_target[edge.target_id].append(edge)
            elif edge.kind == "control":
                self.controls_by_target[edge.target_id].append(edge.source_id)
            elif edge.kind == "selection":
                self.selections_by_target[edge.target_id].append(edge.source_id)

    @property
    def logged_signals(self) -> tuple[str, ...]:
        """Observed leaves reachable through data and control adjacency."""
        signals: list[str] = []
        seen: set[str] = set()
        pending = [self.root_id]
        while pending:
            vertex_id = pending.pop()
            if vertex_id in seen:
                continue
            seen.add(vertex_id)
            vertex = self.vertices.get(vertex_id)
            if (
                vertex is not None
                and vertex.kind == "evidence"
                and vertex.sub_kind == "logged_signal"
                and vertex.signal_name
                and str((vertex.metadata or {}).get("observation") or "observed")
                == "observed"
            ):
                signals.append(str(vertex.signal_name))
            pending.extend(
                edge.source_id for edge in self.data_by_target.get(vertex_id, ())
            )
            pending.extend(self.controls_by_target.get(vertex_id, ()))
            pending.extend(self.selections_by_target.get(vertex_id, ()))
        return tuple(dict.fromkeys(signals))

    def evaluate(
        self,
        timestamp: Optional[float],
        *,
        parameter_values: Optional[dict[str, Any]] = None,
        enum_values: Optional[dict[str, Any]] = None,
        sample_resolver: Optional[SampleResolver] = None,
    ) -> DAGValueResult:
        parameters = {
            str(name).upper(): value
            for name, value in (parameter_values or {}).items()
        }
        enums = dict(enum_values or {})
        cache: dict[tuple[str, Optional[float]], DAGValueResult] = {}
        return self._evaluate_vertex(
            self.root_id,
            timestamp,
            parameters,
            enums,
            sample_resolver,
            cache,
            frozenset(),
        )

    def _evaluate_vertex(
        self,
        vertex_id: str,
        timestamp: Optional[float],
        parameters: dict[str, Any],
        enums: dict[str, Any],
        sample_resolver: Optional[SampleResolver],
        cache: dict[tuple[str, Optional[float]], DAGValueResult],
        active: frozenset[str],
    ) -> DAGValueResult:
        cache_key = (vertex_id, timestamp)
        if cache_key in cache:
            return cache[cache_key]
        if vertex_id in active:
            return DAGValueResult("unresolved", reason="cyclic value dependency")
        vertex = self.vertices.get(vertex_id)
        if vertex is None:
            return DAGValueResult("unresolved", reason="missing DAG vertex")
        next_active = active | {vertex_id}

        if vertex.kind == "evidence":
            result = self._evaluate_evidence(
                vertex, timestamp, parameters, enums, sample_resolver
            )
        elif vertex.kind == "operation":
            activity = self._operation_activity(
                vertex_id,
                timestamp,
                parameters,
                enums,
                sample_resolver,
                cache,
                next_active,
            )
            if activity == "inactive":
                result = DAGValueResult("inactive", reason="writer is inactive")
            elif activity == "unknown":
                result = DAGValueResult(
                    "unresolved", reason="writer reachability is unresolved"
                )
            else:
                result = self._evaluate_expression_vertex(
                    vertex,
                    str(vertex.expression or ""),
                    timestamp,
                    parameters,
                    enums,
                    sample_resolver,
                    cache,
                    next_active,
                )
        elif vertex.kind == "branch":
            result = self._evaluate_branch(
                vertex,
                timestamp,
                parameters,
                enums,
                sample_resolver,
                cache,
                next_active,
            )
        else:
            result = DAGValueResult("unresolved", reason="unsupported vertex kind")
        cache[cache_key] = result
        return result

    @staticmethod
    def _evaluate_evidence(
        vertex: Any,
        timestamp: Optional[float],
        parameters: dict[str, Any],
        enums: dict[str, Any],
        sample_resolver: Optional[SampleResolver],
    ) -> DAGValueResult:
        metadata = vertex.metadata or {}
        if vertex.sub_kind == "logged_signal" and vertex.signal_name:
            if timestamp is None or sample_resolver is None:
                return DAGValueResult(
                    "unresolved", reason="logged value requires a timestamp"
                )
            value = sample_resolver(str(vertex.signal_name), timestamp)
            if value is None:
                return DAGValueResult(
                    "unresolved", reason=f"{vertex.signal_name} is not evaluable"
                )
            return DAGValueResult("value", value=value)
        if metadata.get("value") is not None:
            return DAGValueResult("value", value=metadata["value"])
        name = str(vertex.signal_name or "")
        if vertex.sub_kind == "parameter" and name.upper() in parameters:
            return DAGValueResult("value", value=parameters[name.upper()])
        if name in enums:
            return DAGValueResult("value", value=enums[name])
        return DAGValueResult(
            "unresolved", reason=f"unresolved evidence {name or vertex.id}"
        )

    def _evaluate_branch(
        self,
        vertex: Any,
        timestamp: Optional[float],
        parameters: dict[str, Any],
        enums: dict[str, Any],
        sample_resolver: Optional[SampleResolver],
        cache: dict[tuple[str, Optional[float]], DAGValueResult],
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
        return self._evaluate_expression_vertex(
            vertex,
            str(vertex.predicate_raw or vertex.predicate_lowered or ""),
            timestamp,
            parameters,
            enums,
            sample_resolver,
            cache,
            active,
        )

    def _operation_activity(
        self,
        vertex_id: str,
        timestamp: Optional[float],
        parameters: dict[str, Any],
        enums: dict[str, Any],
        sample_resolver: Optional[SampleResolver],
        cache: dict[tuple[str, Optional[float]], DAGValueResult],
        active: frozenset[str],
    ) -> ActivityStatus:
        unknown = False
        for branch_id in self.controls_by_target.get(vertex_id, ()):
            result = self._evaluate_vertex(
                branch_id,
                timestamp,
                parameters,
                enums,
                sample_resolver,
                cache,
                active,
            )
            if result.status != "value":
                unknown = True
            elif not bool(result.value):
                return "inactive"
        return "unknown" if unknown else "active"

    def _evaluate_expression_vertex(
        self,
        vertex: Any,
        expression: str,
        timestamp: Optional[float],
        parameters: dict[str, Any],
        enums: dict[str, Any],
        sample_resolver: Optional[SampleResolver],
        cache: dict[tuple[str, Optional[float]], DAGValueResult],
        active: frozenset[str],
    ) -> DAGValueResult:
        if not expression.strip():
            return DAGValueResult("unresolved", reason="vertex has no expression")
        by_role: dict[str, list[str]] = defaultdict(list)
        for edge in self.data_by_target.get(vertex.id, ()):
            role = self._source_role(vertex, edge)
            if role:
                by_role[role].append(edge.source_id)

        env: dict[str, Any] = {}
        unresolved: list[str] = []
        for role, producer_ids in by_role.items():
            selected = self._select_producer(
                producer_ids,
                timestamp,
                parameters,
                enums,
                sample_resolver,
                cache,
                active,
            )
            if selected.status == "value":
                env[self._expression_spelling(role)] = selected.value
            elif selected.status == "unresolved":
                unresolved.append(f"{role}: {selected.reason}")

        try:
            value = evaluate_source_expression(
                expression.replace("->", ".").replace("::", "."), env
            )
        except (SourceExpressionError, TypeError, ValueError, ZeroDivisionError) as exc:
            reason = str(exc)
            if unresolved:
                reason = f"{reason}; " + "; ".join(unresolved)
            return DAGValueResult("unresolved", reason=reason)
        return DAGValueResult("value", value=value)

    def _select_producer(
        self,
        producer_ids: list[str],
        timestamp: Optional[float],
        parameters: dict[str, Any],
        enums: dict[str, Any],
        sample_resolver: Optional[SampleResolver],
        cache: dict[tuple[str, Optional[float]], DAGValueResult],
        active: frozenset[str],
    ) -> DAGValueResult:
        candidates: list[tuple[str, ActivityStatus, DAGValueResult]] = []
        for producer_id in dict.fromkeys(producer_ids):
            producer = self.vertices.get(producer_id)
            if producer is None:
                continue
            activity: ActivityStatus = "active"
            if producer.kind == "operation":
                activity = self._operation_activity(
                    producer_id,
                    timestamp,
                    parameters,
                    enums,
                    sample_resolver,
                    cache,
                    active,
                )
            value = (
                self._evaluate_vertex(
                    producer_id,
                    timestamp,
                    parameters,
                    enums,
                    sample_resolver,
                    cache,
                    active,
                )
                if activity == "active"
                else DAGValueResult(
                    "inactive" if activity == "inactive" else "unresolved",
                    reason=f"producer is {activity}",
                )
            )
            candidates.append((producer_id, activity, value))

        active_values = [item for item in candidates if item[2].status == "value"]
        unknown_ids = [item[0] for item in candidates if item[1] == "unknown"]
        if not active_values:
            if unknown_ids:
                return DAGValueResult(
                    "unresolved", reason="all reaching producers are unresolved"
                )
            return DAGValueResult("inactive", reason="no reaching producer is active")
        if len(active_values) == 1 and not unknown_ids:
            return active_values[0][2]

        selected = self._latest_ordered_producer(active_values, unknown_ids)
        if selected is not None:
            return selected
        values = {repr(item[2].value) for item in active_values}
        if len(values) == 1 and not unknown_ids:
            return active_values[0][2]
        return DAGValueResult(
            "unresolved", reason="multiple reaching producers remain possible"
        )

    def _latest_ordered_producer(
        self,
        active_values: list[tuple[str, ActivityStatus, DAGValueResult]],
        unknown_ids: list[str],
    ) -> Optional[DAGValueResult]:
        all_ids = [item[0] for item in active_values] + unknown_ids
        operations = [self.vertices[vertex_id] for vertex_id in all_ids]
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
        latest_active = max(active_values, key=lambda item: int(self.vertices[item[0]].line or 0))
        latest_line = int(self.vertices[latest_active[0]].line or 0)
        if any(int(self.vertices[vertex_id].line or 0) > latest_line for vertex_id in unknown_ids):
            return None
        return latest_active[2]

    @staticmethod
    def _expression_spelling(role: str) -> str:
        return role.replace("->", ".").replace("::", ".")

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
