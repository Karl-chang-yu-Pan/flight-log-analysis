"""Source-linked observations and explicitly conditional local equation checks.

Publication witnesses point from a source value to its recorded observation.
They are not subscription edges and never replace the mechanism's producers.
Local checks use the existing compiled expressions, not reconstructed formulas.
"""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from typing import Any, Optional

from flight_log_agent.analysis.dag_replay import EvaluationScope, _replay_tolerance
from flight_log_agent.analysis.dag_value import DAGValueProgram
from flight_log_agent.analysis.source_expression import SourceExpressionError


def observation_correspondences(dag: Any, program: DAGValueProgram) -> list[dict[str, Any]]:
    """Validate copy paths using graph edges; never invert an arithmetic write."""
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for witness in dag.observation_witnesses:
        publication = program.vertices.get(witness["publication_id"])
        if (publication is None or not witness.get("publication_site")
                or publication.metadata.get("external_target_signal") != witness["signal"]
                or publication.metadata.get("boundary_direction") != "publish"
                or not publication.metadata.get("synthetic_boundary_transfer")):
            continue
        vertex_id = publication.id
        path: list[str] = []
        while vertex_id not in path:
            path.append(vertex_id)
            local = program.compiled_vertices.get(vertex_id)
            if local is None or not local.exact or local.expression is None:
                break
            metadata = program.vertices[vertex_id].metadata
            if not metadata.get("source_expression_ref", {}).get("direct_storage"):
                break
            node = local.expression.tree.body
            if not isinstance(node, ast.Name):
                break
            operand = dict(local.expression.alias_to_operand).get(node.id)
            producers = dict(local.producers_by_operand).get(operand, ())
            if len(producers) != 1:
                break
            vertex_id = producers[0]
            if vertex_id not in program.vertices or vertex_id in path:
                break
            key = (vertex_id, witness["signal"], witness["publication_site"])
            result[key] = {
                **witness, "value_id": vertex_id, "copy_vertex_ids": list(path),
                "value_file": program.vertices[vertex_id].file,
                "value_line": program.vertices[vertex_id].line,
                "relation": "source_value_observed_by_publication",
            }
    return list(result.values())


def evaluate_local_observed_equations(
    dag: Any,
    program: DAGValueProgram,
    *,
    signal_samples: dict[str, list[tuple[float, Any]]],
    parameter_values: dict[str, Any],
    signal_policies: dict[str, Any],
    scope: Optional[EvaluationScope],
    relevant_ids: set[str],
) -> list[dict[str, Any]]:
    """Check equations given source-linked values sampled in the same packet.

    A successful check does not prove that the writer ran, that upstream state
    was initialized correctly, or that no alternative writer exists. Cross-
    publication interpolation and ambiguous correspondences stay unresolved.
    The normal DAG value session and discovery verification are untouched.
    """
    correspondences = observation_correspondences(dag, program)
    if not correspondences:
        return []
    by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in correspondences:
        by_value[item["value_id"]].append(item)
    session = program.bind(parameter_values=parameter_values)
    needed_signals = {item["signal"] for item in correspondences}
    series = {signal: dict(signal_samples.get(signal, ())) for signal in needed_signals}
    duplicate_times = {signal for signal in needed_signals
                       if len(series[signal]) != len(signal_samples.get(signal, ()))}
    checks: list[dict[str, Any]] = []
    pending = set(dag.pending_construction)

    for output in correspondences:
        root = output["value_id"]
        local = program.compiled_vertices.get(root)
        if root not in relevant_ids or local is None or not local.exact:
            continue
        vertex = program.vertices[root]
        direct_copy = vertex.metadata.get("source_expression_ref", {}).get("direct_storage")
        # Observed intermediate copies are input cut points, not requests to
        # reconstruct their histories. A pending terminal copy may still need
        # one value step to expose the actual equation behind its output.
        if direct_copy and not vertex.metadata.get("is_terminal"):
            continue
        if root not in pending and local.expression is not None and isinstance(local.expression.tree.body, ast.Name):
            continue
        used: dict[str, dict[str, Any]] = {}
        errors: set[str] = set()
        comparisons: list[tuple[float, float, float]] = []
        needs: dict[tuple[str, str, str], dict[str, Any]] = {}
        conditional_writers: set[str] = set()

        def require(kind: str, vertex_id: str, operand: str = "", producers: tuple[str, ...] = ()) -> None:
            # Frontier provenance, not a new symbol search or control-ancestor
            # walk, identifies source needed by this particular value operand.
            vertex = program.vertices.get(vertex_id)
            expression_ref = (vertex.metadata.get("source_expression_ref") or {}) if vertex else {}
            symbols = {operand} if operand else set(expression_ref.get("input_symbols") or ())
            call_sites = {item["call_source_site_id"] for item in expression_ref.get("call_results", ())
                          if item.get("call_source_site_id")}
            references = [r.model_dump(mode="json") for r in dag.unresolved_references
                          if vertex_id in r.origin_vertex_ids
                          and (r.source_site_id in call_sites if r.kind == "callable"
                               else bool(symbols.intersection(r.origin_operands)))]
            needs[(kind, vertex_id, operand)] = {
                "kind": kind, "vertex_id": vertex_id, "operand": operand,
                "producer_ids": list(producers), "source_requests": references,
            }
        samples = signal_samples.get(output["signal"], [])
        valid_scope = scope is None or bool(scope.windows and not scope.error and not scope.assumptions)
        if not valid_scope:
            errors.add("comparison scope is unresolved or assumption-dependent")
            samples = []
        elif scope is not None:
            samples = [(t, v) for t, v in samples if any(a <= t <= b for a, b in scope.windows)]
        if not samples:
            errors.add("no output samples in the comparison scope")
        if output["signal"] in duplicate_times:
            errors.add("publication has duplicate sample timestamps")

        def value(vertex_id: str, timestamp: float, visiting: frozenset[str],
                  consumer: str, operand: str) -> Any:
            if vertex_id == root or vertex_id in visiting:
                raise SourceExpressionError("circular local equation")
            observations = by_value.get(vertex_id, [])
            aligned = [item for item in observations
                       if item["signal"] != output["signal"]
                       and item["publication_site"] == output["publication_site"]
                       and item["file"] == output["file"]
                       and item["callable"] == output["callable"]
                       and item["signal"].split(".", 1)[0] == output["signal"].split(".", 1)[0]]
            identities = {(item["signal"], item["publication_id"]) for item in aligned}
            if len(identities) == 1:
                observation = aligned[0]
                if observation["signal"] in duplicate_times:
                    raise SourceExpressionError("publication has duplicate sample timestamps")
                measured = series.get(observation["signal"], {}).get(timestamp)
                if measured is None:
                    raise SourceExpressionError("input lacks an exactly aligned publication sample")
                used[vertex_id] = observation
                return measured
            if observations:
                raise SourceExpressionError("observation is circular, ambiguous, or requires cross-publication alignment")
            if vertex_id in pending:
                require("construction", vertex_id)
                raise SourceExpressionError("local value dependencies have not been materialized")
            static = session.evaluate(vertex_id, None)
            if static.status == "value":
                return static.value
            producer = program.vertices.get(vertex_id)
            reachability = producer.metadata.get("reachability", {}) if producer else {}
            if producer is None or producer.kind != "operation" or not reachability.get("exact"):
                require("source_linkage", consumer, operand, (vertex_id,))
                raise SourceExpressionError("input has no aligned observation or exact source calculation")
            return equation(vertex_id, timestamp, visiting | {vertex_id})

        def equation(vertex_id: str, timestamp: float, visiting: frozenset[str]) -> Any:
            vertex = program.vertices.get(vertex_id)
            if vertex and vertex.metadata.get("reachability", {}).get("all_of"):
                conditional_writers.add(vertex_id)
            if vertex_id in pending:
                require("construction", vertex_id)
                raise SourceExpressionError("local value dependencies have not been materialized")
            compiled = program.compiled_vertices.get(vertex_id)
            if compiled is None or compiled.expression is None or not compiled.exact:
                require("expression", vertex_id)
                raise SourceExpressionError("local source expression is not evaluable")
            operands = dict(compiled.producers_by_operand)
            operand_failed = False

            def resolve(role: str) -> Any:
                nonlocal operand_failed
                producers = operands.get(role, ())
                if len(producers) != 1:
                    require("writer_coverage" if producers else "source_linkage", vertex_id, role, producers)
                    raise SourceExpressionError("local operand has missing or alternative writers")
                try:
                    return value(producers[0], timestamp, visiting, vertex_id, role)
                except SourceExpressionError:
                    operand_failed = True
                    raise

            try:
                return compiled.expression.evaluate(resolve)
            except SourceExpressionError:
                if not needs and not operand_failed:
                    require("expression", vertex_id)
                raise

        for timestamp, observed in samples:
            try:
                predicted = float(equation(root, timestamp, frozenset({root})))
                measured = float(observed)
                if not math.isfinite(predicted) or not math.isfinite(measured):
                    raise SourceExpressionError("non-finite local comparison")
                comparisons.append((timestamp, predicted, measured))
            except (SourceExpressionError, ValueError, TypeError, ArithmeticError) as exc:
                errors.add(str(exc))
                if needs:
                    # Structural prerequisites cannot change at later sample
                    # timestamps. Let construction satisfy them before replay.
                    break
        tolerance = _replay_tolerance(samples, signal_policies.get(output["signal"]))
        matched = sum(abs(predicted - measured) <= tolerance for _, predicted, measured in comparisons)
        complete = bool(samples) and len(comparisons) == len(samples) and not errors
        checks.append({
            "root_vertex_id": root, "observation": output,
            "verification_scope": "local_equation_given_observations",
            "status": ("matched" if matched == len(comparisons) else "mismatched") if complete else "unevaluable",
            "sample_count": len(comparisons), "requested_sample_count": len(samples),
            "matched_sample_count": matched, "tolerance": tolerance,
            "max_absolute_error": max((abs(p - y) for _, p, y in comparisons), default=None),
            "observed_inputs": list(used.values()), "requirements": sorted(errors),
            "input_requirements": list(needs.values()),
            "conditional_writer_ids": sorted(conditional_writers),
            "scope": scope.as_payload() if scope is not None else None,
            "authorizes_discovery_stop": False, "applicability_verified": False,
            "writer_coverage_verified": False, "upstream_obligations_retained": True,
            "conditions": ["source value-to-publication timing and representation",
                           "selected writer applicability and alternative-writer coverage",
                           "observed intermediate state initialization and history"],
        })
    return checks
