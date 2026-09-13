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
from flight_log_agent.analysis.dag_value import DAGValueContext, DAGValueProgram, DAGValueResult
from flight_log_agent.analysis.mechanism_dag import prepare_signal_series, sample_prepared_signal


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
    prepared_signal_series: Optional[dict[str, Any]] = None,
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
    prepared = (prepared_signal_series if prepared_signal_series is not None
                else prepare_signal_series(signal_samples, signal_policies))
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
            operand = dict(local.expression.alias_to_operand).get(local.expression.tree.body.id)
            # A compiled call-result operand forwards a value, not completion
            # of its dependencies. Let the shared evaluator demand that work;
            # do not propagate the output observation into its own producer.
            call_results = vertex.metadata.get("source_expression_ref", {}).get("call_results", [])
            if not call_results or not operand or not operand.startswith("call-site:"):
                continue
        used: dict[str, dict[str, Any]] = {}
        errors: set[str] = set()
        comparisons: list[tuple[float, float, float]] = []
        needs: dict[tuple[str, str, str], dict[str, Any]] = {}
        conditional_writers: set[str] = set()

        def require(kind: str, vertex_id: str, operand: str = "", producers: tuple[str, ...] = (), reason: str = "") -> None:
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
                "producer_ids": list(producers), "source_requests": references, "reason": reason,
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

        def observation(vertex_id: str, timestamp: Optional[float]) -> Optional[DAGValueResult]:
            if vertex_id == root:
                return None
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
                    return DAGValueResult("unresolved", reason="publication has duplicate sample timestamps")
                measured = series.get(observation["signal"], {}).get(timestamp)
                if measured is None:
                    return DAGValueResult("unresolved", reason="input lacks an exactly aligned publication sample")
                used[vertex_id] = observation
                return DAGValueResult("value", value=measured, observed_vertex_ids=frozenset({vertex_id}))
            if observations:
                return DAGValueResult("unresolved", reason="observation is circular, ambiguous, or requires cross-publication alignment")
            vertex = program.vertices[vertex_id]
            if vertex.sub_kind == "logged_signal":
                series_policy = prepared.get(vertex.signal_name)
                if series_policy is None or series_policy.policy is None:
                    return DAGValueResult("unresolved", reason="input samples or sampling policy are unavailable")
            return None

        session = program.bind(
            parameter_values=parameter_values,
            sample_resolver=lambda signal, timestamp: sample_prepared_signal(prepared, signal, timestamp),
            context=DAGValueContext(
                observation_resolver=observation, pending_vertices=frozenset(pending),
                forbidden_signals=frozenset({output["signal"]}), conditional_equations=True,
            ),
        )
        observed_ids: set[str] = set()

        for timestamp, observed in samples:
            evaluated = session.evaluate(root, timestamp)
            observed_ids.update(evaluated.observed_vertex_ids)
            conditional_writers.update(evaluated.conditional_writer_ids)
            for issue in evaluated.issues:
                require(issue.kind, issue.vertex_id, issue.operand, issue.producer_ids, issue.reason)
            session.release_timestamp_values()
            if evaluated.status != "value":
                errors.add(evaluated.reason or evaluated.status)
                if any(need["kind"] in {"construction", "expression", "source_linkage"} for need in needs.values()):
                    break
                continue
            try:
                predicted = float(evaluated.value)
                measured = float(observed)
                if not math.isfinite(predicted) or not math.isfinite(measured):
                    raise ValueError("non-finite local comparison")
                comparisons.append((timestamp, predicted, measured))
            except (ValueError, TypeError, ArithmeticError) as exc:
                errors.add(str(exc))
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
            "observed_inputs": [item for vertex_id, item in used.items() if vertex_id in observed_ids],
            "requirements": sorted(errors),
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
