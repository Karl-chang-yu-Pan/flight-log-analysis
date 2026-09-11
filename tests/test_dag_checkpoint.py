from __future__ import annotations

from copy import deepcopy

import pytest

from flight_log_agent.analysis.dag_checkpoint import assess_checkpoint
from flight_log_agent.analysis.dag_replay import EvaluationScope, observed_checkpoint_roots
from flight_log_agent.analysis.mechanism_dag import (
    DAGEdge, DAGVertex, build_mechanism_dag, evaluate_feasibility,
)
from flight_log_agent.analysis.source_expansion import UnresolvedSourceReference


def expression(text, *inputs):
    return {"text": text, "lowered_text": text, "input_symbols": list(inputs),
            "input_identities": {}, "call_results": [], "exact": True}


@pytest.fixture
def checkpoint():
    bindings = [
        {"target_symbol": "sample", "source_symbol": "measurement.value",
         "external_source_signal": True, "synthetic_boundary_transfer": True,
         "boundary_direction": "subscribe",
         "expression_ref": expression("measurement.value", "measurement.value"),
         "assignment_path": [{"file": "sample.cpp", "line": 2}],
         "function": "Controller::step"},
        {"target_symbol": "command.value", "source_symbol": "sample * 2.0",
         "external_target_signal": True, "synthetic_boundary_transfer": True,
         "boundary_direction": "publish",
         "expression_ref": expression("sample * 2.0", "sample"),
         "assignment_path": [{"file": "sample.cpp", "line": 3}],
         "function": "Controller::step"},
    ]
    samples = {"measurement.value": [(0.0, 3.0), (10.0, 3.0)],
               "command.value": [(0.0, 6.0), (10.0, 6.0)]}
    dag = build_mechanism_dag(bindings, "command.value", logged_signals=set(samples))
    return dag, samples, {signal: {"method": "linear"} for signal in samples}


def assess(checkpoint, **kwargs):
    dag, samples, policies = checkpoint
    kwargs.setdefault("observed_signals", set(samples))
    kwargs.setdefault("signal_samples", samples)
    kwargs.setdefault("signal_policies", policies)
    return assess_checkpoint(dag, observed_checkpoint_roots(dag)["command.value"],
                             "command.value", **kwargs)


def kinds(result):
    return {item["kind"] for item in result["analysis_requirements"]}


def test_source_backed_checkpoint_verifies_known_graph_without_authorizing_stop(checkpoint):
    dag, _, _ = checkpoint
    before = deepcopy(dag.model_dump())
    result = assess(checkpoint)
    assert result["status"] == "matched", result
    assert result["complete"] is True
    assert result["analysis_requirements"] == []
    assert result["source_requests"] == []
    assert result["verification_scope"] == "known_graph"
    assert result["authorizes_discovery_stop"] is False
    assert dag.model_dump() == before


def test_mismatch_is_not_missing_evidence(checkpoint):
    checkpoint[1]["command.value"] = [(0.0, 9.0), (10.0, 9.0)]
    result = assess(checkpoint)
    assert result["status"] == "mismatched"
    assert result["complete"] is True


def test_unrelated_opaque_branch_does_not_block_checkpoint(checkpoint):
    dag, _, _ = checkpoint
    dag.vertices.append(DAGVertex(id="foreign", kind="evidence", sub_kind="opaque_symbol",
                                  signal_name="sample", file="other.cpp"))
    dag.unresolved_references.append(UnresolvedSourceReference(
        symbol="sample", file="other.cpp", callable_id="other", origin_vertex_ids=["foreign"],
    ))
    result = assess(checkpoint)
    assert result["status"] == "matched"
    assert "foreign" not in result["dependency_vertex_ids"]
    assert result["source_requests"] == []


@pytest.mark.parametrize("change", ["no_boundary", "type_only", "unobserved", "no_transfer"])
def test_observation_requires_positive_source_provenance(checkpoint, change):
    dag, _, _ = checkpoint
    leaf = next(v for v in dag.vertices if v.sub_kind == "logged_signal")
    if change == "no_boundary":
        leaf.metadata.pop("boundary")
    elif change == "type_only":
        leaf.metadata["grounded_via"] = "declared_type"
    elif change == "unobserved":
        leaf.metadata["observation"] = "unobserved"
    else:
        transfer = next(v for v in dag.vertices if v.metadata.get("boundary_direction") == "subscribe")
        transfer.metadata.pop("synthetic_boundary_transfer")
    result = assess(checkpoint)
    assert result["status"] == "not_attempted"
    assert "observation_binding" in kinds(result)


def test_no_missing_data_requirement_becomes_a_source_search(checkpoint):
    result = assess(checkpoint, observed_signals={"command.value"})
    assert "observation_data" in kinds(result)
    assert result["source_requests"] == []
    assert result["complete"] is False


def test_missing_local_flow_preserves_declaration_and_does_not_guess_search(checkpoint):
    dag, _, _ = checkpoint
    leaf = next(v for v in dag.vertices if v.sub_kind == "logged_signal")
    leaf.sub_kind = "opaque_symbol"
    leaf.metadata = {"source_identity": {
        "kind": "local", "symbol": "snapshot.control.rate", "root": "snapshot",
        "file": "sample.cpp", "callable_id": "Controller::step",
        "declaration_id": "local:declaration", "declaration_proven": True,
    }}
    result = assess(checkpoint)
    requirement = next(r for r in result["analysis_requirements"] if r["vertex_id"] == leaf.id)
    assert requirement["kind"] == "source_linkage"
    assert requirement["source_identity"]["declaration_id"] == "local:declaration"
    assert result["source_requests"] == []


def test_exact_frontier_request_survives_annotation_and_is_checkpoint_scoped(checkpoint):
    dag, _, _ = checkpoint
    root = observed_checkpoint_roots(dag)["command.value"][0]
    ref = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root], origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "Controller:stored",
                  "declaration_proven": True},
    )
    dag.unresolved_references.append(ref)
    annotated = evaluate_feasibility(dag, prune_dead=False)
    assert annotated.unresolved_references == [ref]
    assert "unresolved_references" not in annotated.model_dump()
    result = assess((annotated, checkpoint[1], checkpoint[2]))
    assert result["source_requests"] == [ref.model_dump(mode="json")]
    assert "source_lookup" in kinds(result)
    assert result["complete"] is False


def add_gate(dag, target, *, assumed=False):
    branch = DAGVertex(
        id="gate", kind="branch", predicate_raw="False", feasibility_verdict="always_false",
        metadata={"source_expression_ref": expression("False"), "expression_inputs_exact": True,
                  "static_evaluation": {"status": "value", "assumed": assumed}},
    )
    dag.vertices.append(branch)
    dag.edges.append(DAGEdge(id="gate-edge", source_id=branch.id, target_id=target, kind="control"))
    return branch


@pytest.mark.parametrize("assumed", [False, True])
def test_only_proven_inactive_writers_can_discharge_missing_inputs(checkpoint, assumed):
    dag, _, _ = checkpoint
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    alternative = root.model_copy(deep=True, update={"id": "alternative", "expression": "missing"})
    alternative.metadata["source_expression_ref"] = expression("missing", "missing")
    dag.vertices.extend([alternative, DAGVertex(id="missing", kind="evidence", sub_kind="opaque_symbol",
                                              signal_name="missing")])
    dag.edges.append(DAGEdge(id="missing-edge", source_id="missing", target_id="alternative",
                            kind="data", role="missing"))
    add_gate(dag, "alternative", assumed=assumed)
    result = assess(checkpoint)
    if assumed:
        assert "missing" in result["dependency_vertex_ids"]
        assert "control_flow" in kinds(result)
        assert result["complete"] is False
    else:
        assert "missing" not in result["dependency_vertex_ids"]
        assert result["inactive_writer_ids"] == ["alternative"]
        assert len(result["root_vertex_ids"]) == 2
        assert result["status"] == "matched", result


def test_assumed_gate_is_checked_transitively(checkpoint):
    dag, _, _ = checkpoint
    transfer = next(v for v in dag.vertices if v.metadata.get("boundary_direction") == "subscribe")
    branch = add_gate(dag, transfer.id, assumed=True)
    branch.feasibility_verdict = "always_true"
    result = assess(checkpoint)
    assert "control_flow" in kinds(result)
    assert any(r["vertex_id"] == "gate" for r in result["analysis_requirements"])


def test_unlinked_guard_operand_is_reported_even_when_parser_omitted_it(checkpoint):
    dag, _, _ = checkpoint
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    branch = add_gate(dag, root.id)
    branch.feasibility_verdict = "unknown"
    branch.metadata = {"source_expression_ref": expression("not running"), "expression_inputs_exact": True}
    result = assess(checkpoint)
    assert any(r["kind"] == "source_linkage" and r.get("operand") == "running"
               for r in result["analysis_requirements"])
    assert result["source_requests"] == []


def test_missing_observable_terminal_binding_is_explicit(checkpoint):
    dag, samples, policies = checkpoint
    roots = observed_checkpoint_roots(dag)["command.value"]
    result = assess_checkpoint(dag, roots, None, signal_samples=samples, signal_policies=policies)
    assert "observation_binding" in kinds(result)
    assert result["observed"] is None


def test_assumptions_and_missing_policy_cannot_be_verified(checkpoint):
    result = assess(checkpoint, scope=EvaluationScope(((0, 10),), assumptions=("reference frame assumed",)))
    assert "evaluation_scope" in kinds(result)
    assert result["complete"] is False
    result = assess(checkpoint, signal_policies={})
    assert "sampling_policy" in kinds(result)


def test_requirement_ids_and_requests_are_order_independent(checkpoint):
    dag, _, _ = checkpoint
    leaf = next(v for v in dag.vertices if v.sub_kind == "logged_signal")
    leaf.sub_kind = "opaque_symbol"
    first = assess(checkpoint)
    dag.vertices.reverse()
    dag.edges.reverse()
    second = assess(checkpoint)
    assert first["checkpoint_id"] == second["checkpoint_id"]
    assert first["analysis_requirements"] == second["analysis_requirements"]


def test_source_request_for_one_operand_does_not_hide_another_missing_operand(checkpoint):
    dag, _, _ = checkpoint
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    root.expression = "sample + absent"
    root.metadata["source_expression_ref"] = expression("sample + absent", "sample", "absent")
    dag.vertices.append(DAGVertex(id="absent", kind="evidence", sub_kind="opaque_symbol", signal_name="absent"))
    dag.edges.append(DAGEdge(id="absent-edge", source_id="absent", target_id=root.id, kind="data", role="absent"))
    dag.unresolved_references.append(UnresolvedSourceReference(
        symbol="sample", origin_vertex_ids=[root.id], origin_operands=["sample"],
    ))
    result = assess(checkpoint)
    assert any(r["kind"] == "source_linkage" and r["vertex_id"] == "absent"
               for r in result["analysis_requirements"])


def test_program_from_another_graph_cannot_certify_checkpoint(checkpoint):
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    dag, _, _ = checkpoint
    unrelated = dag.model_copy(deep=True)
    root = next(v for v in unrelated.vertices if v.metadata.get("is_terminal"))
    root.metadata["source_expression_ref"] = expression("sample * 3", "sample")
    with pytest.raises(ValueError, match="source graph"):
        assess(checkpoint, value_program=DAGValueProgram(unrelated))


def test_checkpoint_identity_survives_discovery_of_another_writer(checkpoint):
    dag, _, _ = checkpoint
    first = assess(checkpoint)
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    dag.vertices.append(root.model_copy(deep=True, update={"id": "new-writer"}))
    second = assess(checkpoint)
    assert first["checkpoint_id"] == second["checkpoint_id"]
    assert "new-writer" in second["root_vertex_ids"]
    assert second["complete"] is False


def test_changed_edges_cannot_reuse_an_old_program(checkpoint):
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    dag, _, _ = checkpoint
    program = DAGValueProgram(dag.model_copy(deep=True))
    dag.edges[0].role = "different_operand"
    with pytest.raises(ValueError, match="source graph"):
        assess(checkpoint, value_program=program)


def test_nan_source_constant_is_not_mistaken_for_a_stale_program(checkpoint):
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    dag = checkpoint[0]
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    root.metadata["value"] = float("nan")
    program_graph = dag.model_copy(deep=True)
    next(v for v in program_graph.vertices if v.id == root.id).metadata["value"] = float("nan")
    result = assess(checkpoint, value_program=DAGValueProgram(program_graph))
    assert result["status"] == "matched"


def test_false_input_copy_gate_cannot_certify_latest_topic_value(checkpoint):
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    dag, _, _ = checkpoint
    transfer = next(v for v in dag.vertices if v.metadata.get("boundary_direction") == "subscribe")
    add_gate(dag, transfer.id)
    result = assess(checkpoint, value_program=DAGValueProgram(dag))
    assert result["status"] == "not_attempted"
    assert "state_alignment" in kinds(result)
    assert transfer.id not in result["inactive_writer_ids"]


def control(checkpoint, **kwargs):
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round

    dag, samples, policies = checkpoint
    return evaluate_checkpoint_round(
        dag, parameter_values={}, observed_signals=set(samples), signal_policies=policies,
        load_samples=lambda _view, _observed: samples, **kwargs,
    )


def test_checkpoint_controller_stops_on_verified_terminal(checkpoint):
    result = control(checkpoint)
    assert result.action == "verified"
    assert result.summary["selected_checkpoint"]["authorizes_discovery_stop"] is True
    assert result.references == []


def test_checkpoint_controller_does_not_verify_a_different_question(checkpoint):
    result = control(checkpoint, question_target="different.output")
    assert result.action == "unresolved"
    assert result.summary["selected_checkpoint"]["authorizes_discovery_stop"] is False


def test_checkpoint_controller_keeps_numerical_contradiction(checkpoint):
    checkpoint[1]["command.value"] = [(0.0, 100.0), (10.0, 100.0)]
    result = control(checkpoint)
    assert result.action == "unresolved"
    assert result.summary["selected_checkpoint"]["status"] == "mismatched"


def test_checkpoint_controller_evaluates_a_ready_logged_gate(checkpoint):
    dag = checkpoint[0]
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    transfer = next(v for v in dag.vertices if v.metadata.get("boundary_direction") == "subscribe")
    gate = add_gate(dag, root.id)
    gate.predicate_raw = "sample > 0"
    gate.metadata = {"source_expression_ref": expression("sample > 0", "sample"), "expression_inputs_exact": True}
    gate.feasibility_verdict = "unknown"
    dag.edges.append(DAGEdge(id="gate-input", source_id=transfer.id, target_id=gate.id, kind="data", role="sample"))
    result = control(checkpoint)
    assert result.summary["dynamic_gate_count"] == 1
    assert result.action == "verified", result.summary


def test_checkpoint_controller_does_not_schedule_missing_gate_inputs(checkpoint, monkeypatch):
    from flight_log_agent.analysis.dag_value import DAGValueSession

    dag = checkpoint[0]
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    gate = add_gate(dag, root.id)
    gate.predicate_raw = "missing"
    gate.metadata = {"source_expression_ref": expression("missing"), "expression_inputs_exact": True}
    gate.feasibility_verdict = "unknown"
    original = DAGValueSession.evaluate

    def static_only(self, vertex_id, timestamp):
        assert timestamp is None, "unlinked gate must not consume a flight timeline"
        return original(self, vertex_id, timestamp)
    monkeypatch.setattr(DAGValueSession, "evaluate", static_only)
    result = control(checkpoint)
    assert result.action == "unresolved"
    assert result.summary["dynamic_gate_count"] == 0


def test_checkpoint_controller_returns_only_relevant_exact_requests(checkpoint):
    dag = checkpoint[0]
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    needed = UnresolvedSourceReference(symbol="reader", kind="callable", origin_vertex_ids=[root.id])
    foreign = UnresolvedSourceReference(symbol="reader", kind="callable", file="other.cpp", origin_vertex_ids=["other"])
    dag.unresolved_references.extend([needed, foreign])
    result = control(checkpoint)
    assert result.action == "continue"
    assert result.references == [needed]


def test_shared_source_request_blocks_every_gate_consumer(checkpoint):
    dag = checkpoint[0]
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    gate = add_gate(dag, root.id)
    gate.predicate_raw = "True"
    gate.metadata = {"source_expression_ref": expression("True"), "expression_inputs_exact": True}
    dag.unresolved_references.append(UnresolvedSourceReference(
        symbol="reader", kind="callable", origin_vertex_ids=[root.id, gate.id],
    ))
    result = control(checkpoint)
    assert result.summary["dynamic_gate_count"] == 0
    assert result.action == "continue"


def test_streamed_feasibility_matches_retained_results(checkpoint):
    from flight_log_agent.analysis.dag_value import DAGValueProgram
    from flight_log_agent.analysis.mechanism_dag import prepare_signal_series, sample_prepared_signal

    dag, samples, policies = checkpoint
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    transfer = next(v for v in dag.vertices if v.metadata.get("boundary_direction") == "subscribe")
    gate = add_gate(dag, root.id)
    gate.predicate_raw = "sample > 0"
    gate.metadata = {"source_expression_ref": expression("sample > 0", "sample"), "expression_inputs_exact": True}
    dag.edges.append(DAGEdge(id="input", source_id=transfer.id, target_id=gate.id, kind="data", role="sample"))
    prepared = prepare_signal_series(samples, policies)
    program = DAGValueProgram(dag)
    session = program.bind(sample_resolver=lambda s, t: sample_prepared_signal(prepared, s, t))
    kwargs = dict(signal_samples=samples, signal_policies=policies, prepared_signal_series=prepared,
                  value_program=program, prune_dead=False)
    expected = evaluate_feasibility(dag, **kwargs)
    actual = evaluate_feasibility(dag, value_session=session, stream_timestamps=True, **kwargs)
    assert expected.model_dump() == actual.model_dump()
    assert all(timestamp is None for _vertex, timestamp in session._value_cache)
    assert session._sample_cache == {}


def test_pending_value_work_cannot_verify_even_when_old_edges_match(checkpoint):
    dag = checkpoint[0]
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    dag.pending_construction = [root.id]
    result = control(checkpoint)
    assert result.action == "continue"
    assert result.summary["selected_checkpoint"]["status"] == "not_attempted"
    assert "construction" in kinds(result.summary["selected_checkpoint"])
    assert "pending_construction" not in dag.model_dump()
    assert root.id in result.construction.materialize


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_branch_exactness_does_not_depend_on_consumer_arrival(staged, reverse):
    bindings = []
    for line, exact in [(2, True), (3, False)]:
        binding = {"target_symbol": "output", "source_symbol": str(line),
                   "assignment_path": [{"file": "sample.cpp", "line": line}],
                   "function": "run", "control_predicates": ["flag"],
                   "control_predicate_lines": [1],
                   "control_predicate_site_ids": ["sample.cpp:1:if"]}
        if exact:
            binding["control_expression_refs"] = [expression("flag", "flag")]
        bindings.append(binding)
    dag = build_mechanism_dag(
        list(reversed(bindings)) if reverse else bindings, "output",
        construction_checkpoint=(lambda _: set()) if staged else None,
    )
    branches = [v for v in dag.vertices if v.kind == "branch"]
    assert len(branches) == 1
    assert branches[0].metadata["expression_inputs_exact"] is True
    assert branches[0].metadata["source_expression_ref"]["input_symbols"] == ["flag"]
    assert len([e for e in dag.edges if e.target_id == branches[0].id and e.kind == "data"]) == 1


def test_explicit_empty_demand_preserves_unexpanded_work():
    from flight_log_agent.analysis.mechanism_dag import ConstructionDemand

    dag = build_mechanism_dag(
        [{"target_symbol": "output", "source_symbol": "missing",
          "expression_ref": expression("missing", "missing"),
          "assignment_path": [{"file": "sample.cpp", "line": 1}], "function": "run"}],
        "output", construction_checkpoint=lambda _: ConstructionDemand(),
    )
    assert dag.pending_construction
    assert not any(v.signal_name == "missing" for v in dag.vertices)


@pytest.mark.parametrize("local_matches", [False, True])
def test_intermediate_calculation_cannot_replace_final_question(checkpoint, local_matches):
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round

    dag, samples, policies = checkpoint
    intermediate = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    intermediate.metadata.pop("is_terminal")
    metadata = {**intermediate.metadata, "is_terminal": True,
                "external_target_signal": "final.value",
                "source_expression_ref": expression("command.value * 0", "command.value")}
    final = intermediate.model_copy(update={
        "id": "final", "variable": "final.value", "expression": "command.value * 0",
        "lowered_expression": None, "metadata": metadata,
    })
    dag.vertices.append(final)
    dag.edges.append(DAGEdge(id="to-final", source_id=intermediate.id, target_id=final.id,
                             kind="data", role="command.value"))
    samples["final.value"] = [(0., 0.), (10., 0.)]
    policies["final.value"] = {"method": "linear"}
    if local_matches:
        dag.pending_construction = [final.id]
    else:
        samples["command.value"] = [(0., 9.), (10., 9.)]
    result = evaluate_checkpoint_round(
        dag, parameter_values={}, observed_signals=set(samples), signal_policies=policies,
        load_samples=lambda *_: samples, question_target="final.value",
    )
    local = result.summary["intermediate_checkpoints"]["command.value"]
    assert local["status"] == ("matched" if local_matches else "mismatched")
    assert local["authorizes_discovery_stop"] is False
    assert result.summary["selected_checkpoint"]["observed"] == "final.value"
    assert result.action != "verified"
    if local_matches:
        assert final.id in result.construction.materialize


def test_pending_conditional_subscription_is_not_discharged(checkpoint):
    dag = checkpoint[0]
    transfer = next(v for v in dag.vertices if v.metadata.get("boundary_direction") == "subscribe")
    add_gate(dag, transfer.id)
    dag.pending_construction = [transfer.id]
    result = control(checkpoint)
    selected = result.summary["selected_checkpoint"]
    assert result.action == "continue"
    assert transfer.id not in selected["inactive_writer_ids"]
    assert "construction" in kinds(selected)


def test_terminal_views_preserve_pending_work_and_writer_requirements(checkpoint):
    from flight_log_agent.analysis.mechanism_dag import split_by_terminal

    dag = checkpoint[0]
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    other = root.model_copy(update={"id": "other-root"})
    dag.vertices.append(other)
    dag.pending_construction = [root.id]
    request = UnresolvedSourceReference(symbol="writer", kind="callable", origin_vertex_ids=[root.id])
    dag.unresolved_references.append(request)
    selected = next(view for view in split_by_terminal(dag) if root.id in {v.id for v in view.vertices})
    assert selected.pending_construction == [root.id]
    assert selected.unresolved_references == [request]


@pytest.mark.parametrize("gate_value", ["False", "True", "unavailable"])
@pytest.mark.parametrize("explicit_demands", [False, True])
def test_construction_resumes_guard_inputs_before_guarded_values(gate_value, explicit_demands):
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round
    from flight_log_agent.analysis.mechanism_dag import DAGConstructionSession

    def binding(target, value, line, inputs=(), controls=()):
        return {"target_symbol": target, "source_symbol": value,
                "expression_ref": expression(value, *inputs),
                "assignment_path": [{"file": "sample.cpp", "line": line}],
                "function": "run", "control_predicates": list(controls),
                "control_expression_refs": [expression(p, p) for p in controls]}

    bindings = [binding("seed", gate_value, 1, () if gate_value != "unavailable" else (gate_value,)),
                binding("gate", "seed", 2, ("seed",)),
                binding("output", "missing_value", 3, ("missing_value",), ("gate",))]
    snapshots = []

    def checkpoint(dag):
        snapshots.append(dag)
        result = evaluate_checkpoint_round(
            dag, parameter_values={}, observed_signals=set(), signal_policies={},
            load_samples=lambda *_: {},
        )
        assert result.action != "verified"
        if explicit_demands:
            return result.construction
        return set(result.summary["selected_checkpoint"]["inactive_writer_ids"])

    session = DAGConstructionSession()
    staged = build_mechanism_dag(bindings, "output", construction_checkpoint=checkpoint,
                                 construction_session=session)
    if explicit_demands and gate_value == "unavailable":
        assert staged.pending_construction
        assert not any(v.signal_name == "missing_value" for v in staged.vertices)
        # The host has searched this exact guard frontier and found no new
        # facts. That permits further work, not an inactivity certificate.
        session.builder.exhausted_source_requests.update(
            reference.visit_key() for reference in staged.unresolved_references
        )
        staged = session.builder.build(construction_checkpoint=checkpoint)
    assert len(snapshots) >= 2
    assert snapshots[0].pending_construction
    assert not any(v.signal_name == "missing_value" for v in snapshots[0].vertices)
    gate = next(v for v in staged.vertices if v.variable == "gate")
    assert all(not (v.sub_kind == "opaque_symbol" and v.signal_name == "gate") for v in staged.vertices)
    assert any(e.source_id == gate.id and e.kind == "data" for e in staged.edges)
    if gate_value == "False":
        assert staged.pending_construction
        assert not any(v.signal_name == "missing_value" for v in staged.vertices)
    else:
        assert staged.pending_construction == []
        eager = build_mechanism_dag(bindings, "output")
        assert {v.id: v.model_dump() for v in staged.vertices} == {v.id: v.model_dump() for v in eager.vertices}
        assert {e.id: e.model_dump() for e in staged.edges} == {e.id: e.model_dump() for e in eager.edges}
        assert staged.unresolved_symbols == eager.unresolved_symbols


def test_construction_revisits_discharge_after_new_definitions():
    def binding(target, value, line, *inputs):
        return {"target_symbol": target, "source_symbol": value,
                "expression_ref": expression(value, *inputs),
                "assignment_path": [{"file": "sample.cpp", "line": line}], "function": "run"}

    bindings = [binding("left", "missing", 1, "missing"),
                binding("step", "1", 2), binding("right", "step", 3, "step"),
                binding("output", "left + right", 4, "left", "right")]
    discharged = []

    def checkpoint(dag):
        left = next((v for v in dag.vertices if v.variable == "left"), None)
        if left is not None and not discharged:
            discharged.append(left.id)
            return {left.id}
        return set()

    dag = build_mechanism_dag(bindings, "output", construction_checkpoint=checkpoint)
    assert discharged
    assert dag.pending_construction == []
    assert "missing" in dag.unresolved_symbols
