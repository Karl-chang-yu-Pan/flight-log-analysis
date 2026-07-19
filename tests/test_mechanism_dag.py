from __future__ import annotations

import pytest

import flight_log_agent.analysis.dag_value as dag_value_module
from flight_log_agent.analysis.dag_value import DAGValueProgram, DAGValueSession
from flight_log_agent.analysis.mechanism_dag import (
    DAGEdge,
    DAGVertex,
    MechanismDAG,
    build_mechanism_dag,
    evaluate_feasibility,
    layer2_cache_path,
    layer3_cache_path,
    prepare_signal_series,
    read_dag_from_cache,
    split_by_terminal,
    write_dag_to_cache,
)
from flight_log_agent.analysis.source_expansion import SourceStructureIndex
from flight_log_agent.ulog.inventory import observed_signals_from_inventory


def _fake_binding(
    *,
    binding_id: str,
    target: str,
    expression: str,
    file: str,
    line: int,
    logged_signal: str | None = None,
    control_predicates: list[str] | None = None,
    function: str | None = None,
    declaration_kind: str | None = None,
) -> dict:
    """A binding with ``logged_signal`` defaulted to the target — the
    BindingIndex backward walk starts from ``logged_signal``, so tests
    seed it explicitly. Pass ``logged_signal=""`` to exercise the scoped
    target-writer path (logged-output writers bypass visibility)."""
    return {
        "binding_id": binding_id,
        "source_symbol": expression,
        "target_symbol": target,
        "logged_signal": target if logged_signal is None else logged_signal,
        "assignment_path": [{"file": file, "line": line, "expression": expression}],
        "control_predicates": control_predicates or [],
        "function": function or "",
        "declaration_kind": declaration_kind or "",
    }


def _fake_inventory(topics: dict[str, list[str]] | None = None) -> dict:
    return {
        "topic_fields": topics or {},
        "available_topics": list((topics or {}).keys()),
    }


def _boundary(
    source_symbol: str,
    topic: str,
    *,
    file: str = "",
    direction: str = "subscribe",
    instance: int | None = None,
) -> dict:
    return {
        "source_symbol": source_symbol,
        "topic": topic,
        "direction": direction,
        "file": file,
        "function": "",
        "callable_id": "",
        "instance": instance,
        "provenance": "test source boundary",
    }


def _fake_helper(
    *,
    name: str,
    file: str,
    line: int,
    evidence: str,
    assignments: dict[str, str],
    return_expression: str,
    branches: list[dict] | None = None,
) -> dict:
    return {
        "name": name,
        "file": file,
        "line": line,
        "evidence": evidence,
        "assignments": assignments,
        "return_expression": return_expression,
        "lowered_return_expression": return_expression,
        "branches": branches or [],
        "helper_calls": [],
        "call_resolutions": [],
        "statements": [],
        "symbol_bindings": {},
        "parameters": [],
    }


def _owned_bindings(
    bindings: list[dict],
    *,
    bases: dict[str, set[str]] | None = None,
    members: dict[tuple[str, str], dict] | None = None,
) -> tuple[list[dict], SourceStructureIndex]:
    structure = SourceStructureIndex(
        direct_bases=bases or {},
        members=members or {},
    )
    return structure.enrich_bindings(bindings), structure


def test_backward_slice_emits_vertices_for_reaching_bindings():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="max(global_position.alt, _destination.alt + _param_rtl_return_alt.get())",
            file="src/modules/navigator/rtl.cpp",
            line=248,
        ),
        _fake_binding(
            binding_id="b2",
            target="_destination.alt",
            expression="home_landing_position.alt",
            file="src/modules/navigator/rtl.cpp",
            line=127,
        ),
    ]

    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        logged_signals={"vehicle_global_position.alt"},
        parameter_names={"RTL_RETURN_ALT"},
    )

    kinds = {v.kind for v in dag.vertices}
    assert "operation" in kinds
    assert "evidence" in kinds

    operations = [v for v in dag.vertices if v.kind == "operation"]
    op_targets = {v.variable for v in operations}
    assert "_rtl_alt" in op_targets
    assert "_destination.alt" in op_targets


def test_branch_deduplicates_by_source_site():
    """Two writes gated by the SAME if-statement share one branch vertex
    — the profiler-emitted site line, not the predicate text, is the
    identity. Without site info the fallback keys on each write's own
    line: conservatively split, never falsely merged."""
    predicate = "_param_rtl_type.get() != RTL_TYPE_HOME_OR_RALLY"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_destination.alt",
            expression="mission_landing_alt",
            file="src/modules/navigator/rtl.cpp",
            line=163,
            control_predicates=[predicate],
        ),
        _fake_binding(
            binding_id="b2",
            target="_destination.alt",
            expression="mission_landing_alt",
            file="src/modules/navigator/rtl.cpp",
            line=172,
            control_predicates=[predicate],
        ),
    ]
    bindings[0]["control_predicate_lines"] = [160]
    bindings[1]["control_predicate_lines"] = [160]

    dag = build_mechanism_dag(bindings, "_destination.alt")

    branches = [v for v in dag.vertices if v.kind == "branch"]
    assert len(branches) == 1
    assert branches[0].predicate_raw == predicate
    assert branches[0].line == 160

    without_sites = build_mechanism_dag(
        [{**bindings[0], "control_predicate_lines": []},
         {**bindings[1], "control_predicate_lines": []}],
        "_destination.alt",
    )
    assert len([v for v in without_sites.vertices if v.kind == "branch"]) == 2


def test_two_writes_at_different_lines_produce_two_operation_vertices():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_destination.alt",
            expression="mission_landing_alt",
            file="src/modules/navigator/rtl.cpp",
            line=163,
        ),
        _fake_binding(
            binding_id="b2",
            target="_destination.alt",
            expression="closest_safe_point.alt",
            file="src/modules/navigator/rtl.cpp",
            line=224,
        ),
    ]

    dag = build_mechanism_dag(bindings, "_destination.alt")

    operations = [v for v in dag.vertices if v.kind == "operation" and v.variable == "_destination.alt"]
    assert len(operations) == 2
    assert {v.line for v in operations} == {163, 224}


def test_evidence_leaf_classification_covers_logged_and_parameter():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="max(gpos_alt, _destination_alt + _param_rtl_return_alt.get())",
            file="src/modules/navigator/rtl.cpp",
            line=248,
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        logged_signals={"gpos_alt"},
        parameter_names={"RTL_RETURN_ALT"},
    )

    evidence_by_kind = {v.sub_kind: v for v in dag.vertices if v.kind == "evidence"}
    assert "logged_signal" in evidence_by_kind
    assert evidence_by_kind["logged_signal"].signal_name == "gpos_alt"
    assert "parameter" in evidence_by_kind
    assert evidence_by_kind["parameter"].signal_name == "RTL_RETURN_ALT"


def test_unresolved_symbol_becomes_opaque_evidence():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="magic_helper(some_unresolved_thing)",
            file="src/modules/navigator/rtl.cpp",
            line=245,
        ),
    ]

    dag = build_mechanism_dag(bindings, "_rtl_alt")

    opaque = [v for v in dag.vertices if v.kind == "evidence" and v.sub_kind == "opaque_symbol"]
    assert opaque, "expected at least one opaque_symbol leaf for an unresolved reference"
    assert "some_unresolved_thing" in dag.unresolved_symbols


def test_helper_body_preserves_intermediates_per_call_site():
    helper = _fake_helper(
        name="RTL::calc_cone_alt",
        file="src/modules/navigator/rtl.cpp",
        line=696,
        evidence="float RTL::calc_cone_alt(float half_angle) {",
        assignments={
            "destination_dist": "get_distance_to_next_waypoint(_destination.lat, gpos.lat)",
            "return_altitude_amsl": "_destination.alt + _param_rtl_return_alt.get()",
        },
        return_expression="max(return_altitude_amsl, gpos.alt)",
    )
    bindings = [
        # Common downstream terminal that reaches both callers.
        _fake_binding(
            binding_id="downstream",
            target="combined_alt",
            expression="max(_rtl_alt, _alt_snapshot)",
            file="src/modules/navigator/rtl.cpp",
            line=999,
        ),
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="calc_cone_alt(_param_rtl_cone_half_angle_deg.get())",
            file="src/modules/navigator/rtl.cpp",
            line=245,
        ),
        _fake_binding(
            binding_id="b2",
            target="_alt_snapshot",
            expression="calc_cone_alt(_param_rtl_cone_half_angle_deg.get())",
            file="src/modules/navigator/rtl.cpp",
            line=311,
        ),
    ]

    dag = build_mechanism_dag(
        bindings,
        "combined_alt",
        helper_expressions=[helper],
        parameter_names={"RTL_CONE_HALF_ANGLE_DEG", "RTL_RETURN_ALT"},
    )

    intermediates = {
        v.variable: v
        for v in dag.vertices
        if v.kind == "operation" and v.provenance and v.provenance.startswith("helper_body")
    }
    assert "destination_dist" in intermediates
    assert "return_altitude_amsl" in intermediates

    helper_terminals = [
        v
        for v in dag.vertices
        if v.provenance and v.provenance.startswith("helper_return")
    ]
    assert len(helper_terminals) == 2
    assert len(
        {
            terminal.metadata.get("call_site_id")
            for terminal in helper_terminals
        }
    ) == 2

    caller_op_ids = {
        v.id for v in dag.vertices if v.variable in {"_rtl_alt", "_alt_snapshot"}
    }
    via_edges = [
        edge
        for edge in dag.edges
        if edge.kind == "data"
        and edge.target_id in caller_op_ids
        and edge.source_id in {terminal.id for terminal in helper_terminals}
    ]
    assert len(via_edges) == 2
    assert len({edge.source_id for edge in via_edges}) == 2


def test_dag_terminal_is_recorded():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="src/modules/navigator/rtl.cpp",
            line=248,
        ),
    ]

    dag = build_mechanism_dag(bindings, "_rtl_alt")
    assert dag.terminal == "_rtl_alt"
    assert dag.dag_id.startswith("dag_")


def test_snippet_embedding_reads_from_source_root(tmp_path):
    file_rel = "src/modules/navigator/rtl.cpp"
    file_path = tmp_path / file_rel
    file_path.parent.mkdir(parents=True)
    file_path.write_text("\n".join([f"line {i}" for i in range(1, 21)]), encoding="utf-8")

    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file=file_rel,
            line=10,
        ),
    ]

    dag = build_mechanism_dag(bindings, "_rtl_alt", source_root=tmp_path, snippet_context_lines=2)
    op = next(v for v in dag.vertices if v.kind == "operation")
    assert op.snippet is not None
    assert "line 10" in op.snippet
    assert "line 8" in op.snippet
    assert "line 12" in op.snippet


def test_feasibility_marks_always_true_for_satisfied_predicate():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="src/modules/navigator/rtl.cpp",
            line=245,
            control_predicates=["_param_rtl_cone_half_angle_deg.get() > 0"],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    reduced = evaluate_feasibility(
        dag,
        parameter_values={"RTL_CONE_HALF_ANGLE_DEG": 45},
        prune_dead=False,
    )
    branch = next(v for v in reduced.vertices if v.kind == "branch")
    assert branch.feasibility_verdict == "always_true"


def test_feasibility_reuses_prepared_signal_series(monkeypatch):
    from flight_log_agent.analysis import mechanism_dag as mechanism_dag_module

    bindings = [
        _fake_binding(
            binding_id="b1",
            target="output",
            expression="1",
            file="src/modules/example/ctrl.cpp",
            line=10,
            control_predicates=["topic.flag > 0"],
        ),
        _fake_binding(
            binding_id="b2",
            target="output",
            expression="2",
            file="src/modules/example/ctrl.cpp",
            line=20,
            control_predicates=["topic.flag <= 0"],
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "output",
        logged_signals={"topic.flag"},
    )
    samples = {"topic.flag": [(2.0, 1), (0.0, 0), (1.0, 1)]}
    policies = {"topic.flag": {"method": "discrete_hold"}}
    prepared = prepare_signal_series(samples, policies)
    assert prepared["topic.flag"].times == (0.0, 1.0, 2.0)

    def unexpected_prepare(*_args, **_kwargs):
        raise AssertionError("prepared DAG signals must not be sorted again")

    monkeypatch.setattr(
        mechanism_dag_module, "prepare_signal_series", unexpected_prepare
    )
    reduced = evaluate_feasibility(
        dag,
        signal_samples=samples,
        signal_policies=policies,
        prepared_signal_series=prepared,
        prune_dead=False,
    )
    assert len([vertex for vertex in reduced.vertices if vertex.kind == "branch"]) == 2


def test_feasibility_evaluates_internal_predicate_from_dag_edges():
    bindings = [
        _fake_binding(
            binding_id="state",
            target="internal_state",
            expression="topic.flag",
            file="src/modules/example/ctrl.cpp",
            line=10,
            logged_signal="",
            function="Control::run",
        ),
        _fake_binding(
            binding_id="terminal",
            target="output",
            expression="topic.value",
            file="src/modules/example/ctrl.cpp",
            line=20,
            logged_signal="output.value",
            control_predicates=["internal_state > 0"],
            function="Control::run",
        ),
    ]
    samples = {
        "topic.flag": [(0.0, 0), (5.0, 1), (10.0, 1)],
        "topic.value": [(0.0, 2.0), (10.0, 2.0)],
        "output.value": [(0.0, 2.0), (10.0, 2.0)],
    }
    policies = {
        "topic.flag": {"method": "discrete_hold"},
        "topic.value": {"method": "linear"},
        "output.value": {"method": "linear"},
    }
    dag = build_mechanism_dag(
        bindings,
        "output",
        logged_signals=set(samples),
    )

    annotated = evaluate_feasibility(
        dag,
        signal_samples=samples,
        signal_policies=policies,
        prune_dead=False,
    )

    branch = next(vertex for vertex in annotated.vertices if vertex.kind == "branch")
    assert branch.active_windows == [(5.0, 10.0)]
    assert branch.feasibility_verdict == "unknown"
    assert branch.metadata["evaluation_mode"] == "dag_value_plan"
    assert branch.metadata["sampling_policies"] == {
        "topic.flag": "discrete_hold"
    }


def test_feasibility_prunes_always_false_gated_operation():
    predicate = "_param_rtl_cone_half_angle_deg.get() > 0"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="cone_result",
            file="src/modules/navigator/rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
        _fake_binding(
            binding_id="b2",
            target="_rtl_alt",
            expression="max(gpos.alt, _destination.alt + _param_rtl_return_alt.get())",
            file="src/modules/navigator/rtl.cpp",
            line=248,
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    reduced = evaluate_feasibility(
        dag,
        parameter_values={"RTL_CONE_HALF_ANGLE_DEG": 0},   # cone disabled
        prune_dead=True,
    )
    remaining_ops = [v for v in reduced.vertices if v.kind == "operation" and v.variable == "_rtl_alt"]
    assert len(remaining_ops) == 1, "cone-gated write should be pruned when RTL_CONE_HALF_ANGLE_DEG=0"
    assert "max" in (remaining_ops[0].expression or "")

    # And the always_false branch should be gone.
    assert not any(v.kind == "branch" for v in reduced.vertices)


def test_feasibility_leaves_unknown_when_predicate_references_unresolved_signal():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="src/modules/navigator/rtl.cpp",
            line=245,
            control_predicates=["vehicle_status.vehicle_type == 1"],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    reduced = evaluate_feasibility(
        dag,
        parameter_values={},
        prune_dead=False,
    )
    branch = next(v for v in reduced.vertices if v.kind == "branch")
    assert branch.feasibility_verdict == "unknown"


def test_feasibility_reduces_compound_predicate_with_enum_value():
    predicate = "_param_rtl_type.get() != RTL_TYPE_HOME_OR_RALLY"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_destination.alt",
            expression="mission_landing_alt",
            file="src/modules/navigator/rtl.cpp",
            line=163,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_destination.alt")

    reduced = evaluate_feasibility(
        dag,
        parameter_values={"RTL_TYPE": 1},
        enum_values={"RTL_TYPE_HOME_OR_RALLY": 0},
        prune_dead=False,
    )
    branch = next(v for v in reduced.vertices if v.kind == "branch")
    assert branch.feasibility_verdict == "always_true"


def test_feasibility_preserves_always_true_gate_without_pruning():
    predicate = "_param_rtl_return_alt.get() > 0"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="_destination.alt + _param_rtl_return_alt.get()",
            file="src/modules/navigator/rtl.cpp",
            line=248,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    reduced = evaluate_feasibility(
        dag,
        parameter_values={"RTL_RETURN_ALT": 10},
        prune_dead=True,
    )
    # Always-true branches survive (they gate live operations); only
    # always_false subgraphs get pruned.
    branch = next(v for v in reduced.vertices if v.kind == "branch")
    assert branch.feasibility_verdict == "always_true"
    op = next(v for v in reduced.vertices if v.kind == "operation" and v.variable == "_rtl_alt")
    assert op is not None


def test_signal_samples_compute_active_windows_for_partial_true_predicate():
    predicate = "nav_state == 5"  # 5 = AUTO_RTL in this fake enum
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="src/modules/navigator/rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    reduced = evaluate_feasibility(
        dag,
        signal_samples={
            "nav_state": [
                (0.0, 1),   # MANUAL
                (5.0, 5),   # AUTO_RTL starts
                (12.0, 3),  # AUTO_LOITER
                (20.0, 5),  # AUTO_RTL again
                (25.0, 1),  # back to MANUAL
            ],
        },
        prune_dead=False,
    )
    branch = next(v for v in reduced.vertices if v.kind == "branch")
    assert branch.active_windows == [(5.0, 12.0), (20.0, 25.0)]
    assert branch.feasibility_verdict == "unknown"  # partial


def test_signal_samples_mark_always_true_when_predicate_covers_whole_span():
    predicate = "vehicle_type == 1"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="src/modules/navigator/rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    reduced = evaluate_feasibility(
        dag,
        signal_samples={
            "vehicle_type": [(0.0, 1), (10.0, 1), (20.0, 1)],
        },
        signal_policies={"vehicle_type": {"method": "discrete_hold"}},
        prune_dead=False,
    )
    branch = next(v for v in reduced.vertices if v.kind == "branch")
    assert branch.feasibility_verdict == "always_true"


def test_signal_samples_prune_branch_when_predicate_never_true():
    predicate = "vehicle_type == 2"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="cone_result",
            file="src/modules/navigator/rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
        _fake_binding(
            binding_id="b2",
            target="_rtl_alt",
            expression="max(gpos.alt, _destination.alt)",
            file="src/modules/navigator/rtl.cpp",
            line=248,
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    reduced = evaluate_feasibility(
        dag,
        signal_samples={
            "vehicle_type": [(0.0, 1), (10.0, 1), (20.0, 1)],  # never 2
        },
        signal_policies={"vehicle_type": {"method": "discrete_hold"}},
        prune_dead=True,
    )
    # Cone-gated write pruned; else write survives.
    remaining_ops = [v for v in reduced.vertices if v.kind == "operation" and v.variable == "_rtl_alt"]
    assert len(remaining_ops) == 1
    assert remaining_ops[0].line == 248


def test_signal_samples_combine_with_parameter_substitution():
    predicate = "_param_rtl_cone_half_angle_deg.get() > 0 && vehicle_type == 1"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="cone_result",
            file="src/modules/navigator/rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    reduced = evaluate_feasibility(
        dag,
        parameter_values={"RTL_CONE_HALF_ANGLE_DEG": 45},
        signal_samples={
            "vehicle_type": [
                (0.0, 1),   # ROTARY_WING
                (10.0, 2),  # FIXED_WING
                (20.0, 1),  # back to ROTARY_WING
            ],
        },
        prune_dead=False,
    )
    branch = next(v for v in reduced.vertices if v.kind == "branch")
    assert branch.active_windows == [(0.0, 10.0), (20.0, 20.0)]


def test_split_by_terminal_returns_singleton_when_one_terminal_op():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="src/modules/navigator/rtl.cpp",
            line=245,
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    subgraphs = split_by_terminal(dag)
    assert len(subgraphs) == 1
    assert subgraphs[0] is dag


def test_split_by_terminal_produces_one_subgraph_per_terminal_op():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="cone_helper()",
            file="src/modules/navigator/rtl.cpp",
            line=245,
        ),
        _fake_binding(
            binding_id="b2",
            target="_rtl_alt",
            expression="max(gpos_alt, _destination_alt)",
            file="src/modules/navigator/rtl.cpp",
            line=248,
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    # Both operations write to the terminal → each is is_terminal.
    terminal_ops = [v for v in dag.vertices if v.kind == "operation" and v.metadata.get("is_terminal")]
    assert len(terminal_ops) == 2

    subgraphs = split_by_terminal(dag)
    assert len(subgraphs) == 2
    for subgraph in subgraphs:
        terminal_count = sum(
            1 for v in subgraph.vertices
            if v.kind == "operation" and v.metadata.get("is_terminal")
        )
        assert terminal_count == 1


def test_layer2_cache_roundtrip(tmp_path):
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="src/modules/navigator/rtl.cpp",
            line=245,
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    path = layer2_cache_path(tmp_path, "abcdef", "position_setpoint_triplet.current.alt")
    assert not path.exists()
    write_dag_to_cache(dag, path)
    assert path.exists()
    assert path.name == "position_setpoint_triplet__current__alt.json"

    restored = read_dag_from_cache(path)
    assert restored is not None
    assert restored.dag_id == dag.dag_id
    assert restored.terminal == dag.terminal
    assert len(restored.vertices) == len(dag.vertices)


def test_layer3_cache_path_composes_with_ulog_hash(tmp_path):
    path = layer3_cache_path(tmp_path, "abcdef", "0123456", "_mission_item.altitude")
    assert path.parent.name == "0123456"
    assert path.parent.parent.name == "abcdef"
    assert path.name == "_mission_item__altitude.json"


def test_read_dag_from_cache_returns_none_on_missing_or_corrupt(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    assert read_dag_from_cache(missing) is None

    corrupt = tmp_path / "bad.json"
    corrupt.write_text("not valid json {", encoding="utf-8")
    assert read_dag_from_cache(corrupt) is None


def test_write_dag_to_cache_replaces_prior_entry_atomically(tmp_path):
    bindings1 = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
        ),
    ]
    bindings2 = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="99",
            file="rtl.cpp",
            line=1,
        ),
    ]
    path = layer2_cache_path(tmp_path, "hash", "_rtl_alt")

    write_dag_to_cache(build_mechanism_dag(bindings1, "_rtl_alt"), path)
    write_dag_to_cache(build_mechanism_dag(bindings2, "_rtl_alt"), path)

    restored = read_dag_from_cache(path)
    assert restored is not None
    # Latest write wins; expression should be "99".
    op = next(v for v in restored.vertices if v.kind == "operation")
    assert op.expression == "99"


def test_helper_chain_source_form_resolves_to_logged_evidence():
    """A branch predicate reading a helper-chain source form like
    ``_navigator.get_vstatus().vehicle_type`` should emit
    ``evidence:logged_signal`` with the canonical logged name — resolved
    graph-natively via the helper's ``return_type`` and the PX4 msg schema."""
    helper = _fake_helper(
        name="Navigator::get_vstatus",
        file="navigator.cpp",
        line=100,
        evidence="vehicle_status_s * Navigator::get_vstatus()",
        assignments={},
        return_expression="_vehicle_status",
    )
    helper["return_type"] = "vehicle_status_s *"
    predicate = "_navigator.get_vstatus().vehicle_type == 1"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        inventory=_fake_inventory({"vehicle_status": ["vehicle_type"]}),
        helper_expressions=[helper],
        boundary_bindings=[
            _boundary("_vehicle_status", "vehicle_status", file="navigator.cpp")
        ],
    )

    logged = next(
        (v for v in dag.vertices
         if v.kind == "evidence" and v.sub_kind == "logged_signal"
         and v.signal_name == "vehicle_status.vehicle_type"),
        None,
    )
    assert logged is not None
    assert logged.metadata.get("source_form")
    assert logged.metadata.get("derivation") == "source_boundary"


def test_source_enum_resolution_stores_value_on_constant_vertex():
    """A predicate mentioning an enum defined in source (numeric value
    resolved) should emit ``evidence:constant`` with the value in metadata."""
    predicate = "x == RTL_TYPE_HOME_OR_RALLY"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
    ]
    # RTL_TYPE_HOME_OR_RALLY is defined in source as a numeric constant.
    # The DAG resolves it natively from the source binding
    # (target=NAME, expression=<numeric>) — not an injected
    # assignment_resolutions entry, which is what previously masked the
    # SliceResult type mismatch.
    bindings.append(
        _fake_binding(
            binding_id="const",
            target="RTL_TYPE_HOME_OR_RALLY",
            expression="0",
            file="rtl.cpp",
            line=1,
            declaration_kind="enum",
        )
    )

    dag = build_mechanism_dag(bindings, "_rtl_alt")

    const_vertex = next(
        (v for v in dag.vertices
         if v.kind == "evidence" and v.sub_kind == "constant"
         and v.signal_name == "RTL_TYPE_HOME_OR_RALLY"),
        None,
    )
    assert const_vertex is not None
    assert const_vertex.metadata.get("value") == 0
    assert const_vertex.metadata.get("source") == "enum"


def test_cxx_stdlib_constant_stored_on_constant_vertex():
    """A predicate mentioning ``FLT_EPSILON`` should emit an ``evidence:constant``
    vertex with the standard IEEE 754 value from ``CXX_STDLIB_CONSTANTS``."""
    predicate = "cruising_speed > FLT_EPSILON"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
    ]

    dag = build_mechanism_dag(bindings, "_rtl_alt")

    const_vertex = next(
        (v for v in dag.vertices
         if v.kind == "evidence" and v.sub_kind == "constant"
         and v.signal_name == "FLT_EPSILON"),
        None,
    )
    assert const_vertex is not None
    assert const_vertex.metadata.get("value") is not None
    assert const_vertex.metadata.get("source") == "cxx_stdlib"


def test_helper_call_arguments_wire_into_formal_parameter_vertices():
    """Caller's actual argument should be connected to the helper's
    formal-parameter vertex — the graph flows through parameter ports
    instead of leaving the formal name unresolved."""
    helper = _fake_helper(
        name="scale_alt",
        file="rtl.cpp",
        line=800,
        evidence="float scale_alt(float base) {",
        assignments={},
        return_expression="base * 2",
    )
    helper["parameters"] = ["base"]

    bindings = [
        _fake_binding(
            binding_id="caller",
            target="_rtl_alt",
            expression="scale_alt(_destination.alt)",
            file="rtl.cpp",
            line=500,
        ),
        _fake_binding(
            binding_id="alt_write",
            target="_destination.alt",
            expression="home.alt",
            file="rtl.cpp",
            line=200,
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        helper_expressions=[helper],
    )

    # Formal-parameter vertex emitted?
    formal = next(
        (v for v in dag.vertices if v.kind == "evidence" and v.sub_kind == "helper_parameter"),
        None,
    )
    assert formal is not None, "helper_parameter vertex missing"
    assert formal.signal_name == "base"

    # A data edge should feed the argument into the formal-parameter vertex.
    incoming = [e for e in dag.edges if e.target_id == formal.id and e.kind == "data"]
    assert incoming, "no data edge feeding the helper formal parameter"
    # The role should reference the formal name so LLM presentation stays useful.
    assert any(e.role == "arg:base" for e in incoming)


def test_struct_root_does_not_guess_which_fields_a_helper_reads():
    """An aggregate argument stays distinct from element writes until the
    called helper body proves which fields it consumes."""
    bindings = [
        # Terminal: mission_item_altitude_amsl = get_absolute_altitude_for_item(_mission_item)
        _fake_binding(
            binding_id="terminal",
            target="mission_item_altitude_amsl",
            expression="get_absolute_altitude_for_item(_mission_item)",
            file="mission_block.cpp",
            line=190,
            function="MissionBlock::set_item",
        ),
        # Two field writes on _mission_item — the walk would normally
        # never see these because nothing writes _mission_item bare.
        _fake_binding(
            binding_id="field_altitude",
            target="_mission_item.altitude",
            expression="_rtl_alt",
            file="rtl.cpp",
            line=361,
            function="MissionBlock::update_item",
        ),
        _fake_binding(
            binding_id="field_lat",
            target="_mission_item.lat",
            expression="_destination.lat",
            file="rtl.cpp",
            line=360,
            function="MissionBlock::update_item",
        ),
        # An upstream write for _rtl_alt so the walk continues past the
        # field expansion.
        _fake_binding(
            binding_id="rtl_alt_write",
            target="_rtl_alt",
            expression="max(gpos.alt, _destination.alt + _param_rtl_return_alt.get())",
            file="rtl.cpp",
            line=248,
            function="MissionBlock::update_item",
        ),
    ]
    bindings, structure = _owned_bindings(
        bindings,
        bases={"MissionBlock": set()},
        members={
            ("MissionBlock", "_mission_item"): {"type": "mission_item_s"},
            ("MissionBlock", "_rtl_alt"): {"type": "float"},
            ("MissionBlock", "_destination"): {"type": "Position"},
        },
    )
    dag = build_mechanism_dag(
        bindings,
        "mission_item_altitude_amsl",
        source_structure=structure,
    )

    variables = {v.variable for v in dag.vertices if v.kind == "operation"}
    # No blanket aggregate-to-element fan-out: the helper is unresolved, so
    # its actual field dependencies remain unresolved too.
    assert "mission_item_altitude_amsl" in variables
    assert "_mission_item.altitude" not in variables
    assert "_mission_item.lat" not in variables
    assert "_mission_item" in dag.unresolved_symbols


def test_dag_without_source_root_omits_snippets():
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="src/modules/navigator/rtl.cpp",
            line=10,
        ),
    ]

    dag = build_mechanism_dag(bindings, "_rtl_alt")
    assert all(v.snippet is None for v in dag.vertices)


def test_helper_pointer_output_writes_emit_ops_at_call_site():
    """When a helper record carries ``pointer_output_writes``, the DAG
    builder should emit an operation vertex per write at the call site,
    with the caller's actual arg substituted for the pointer formal."""
    helper = _fake_helper(
        name="RTL::compute_setpoint",
        file="rtl.cpp",
        line=100,
        evidence="void RTL::compute_setpoint(position_setpoint_s *sp)",
        assignments={},
        return_expression="",
    )
    helper["parameters"] = ["sp"]
    helper["assignments"] = {"sp.alt": "_rtl_alt"}
    helper["pointer_output_writes"] = [
        {"param": "sp", "field": "alt", "expression": "_rtl_alt"},
    ]
    # Caller writes _navigator.get_position_setpoint_triplet().current via
    # the helper call; the DAG should emit an op targeting
    # <arg>.alt = _rtl_alt at the caller site.
    bindings = [
        _fake_binding(
            binding_id="caller",
            target="_rtl_alt",
            expression="compute_setpoint(&_pos_sp)",
            file="rtl.cpp",
            line=500,
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt", helper_expressions=[helper])

    ops = [v for v in dag.vertices if v.kind == "operation"]
    variables = {v.variable for v in ops}
    assert "_pos_sp.alt" in variables
    pointer_op = next(v for v in ops if v.variable == "_pos_sp.alt")
    assert pointer_op.expression == "sp.alt"
    assert (pointer_op.provenance or "").startswith("pointer_output:")
    helper_write = next(v for v in ops if v.variable == "sp.alt")
    assert any(
        edge.source_id == helper_write.id and edge.target_id == pointer_op.id
        for edge in dag.edges
    )


def test_short_call_statement_name_is_not_filtered_by_spelling():
    helper = _fake_helper(
        name="gen",
        file="helper.cpp",
        line=10,
        evidence="void gen(float value)",
        assignments={},
        return_expression="",
    )
    helper["parameters"] = ["value"]
    helper["callable_id"] = "helper.cpp:10:gen:value"
    bindings = [
        _fake_binding(
            binding_id="state",
            target="state",
            expression="value",
            file="helper.cpp",
            line=12,
            function="gen",
        )
    ]
    bindings[0]["callable_id"] = helper["callable_id"]
    calls = [
        {
            "name": "gen",
            "args": ["input"],
            "file": "caller.cpp",
            "line": 20,
            "function": "run",
            "callable_id": "caller.cpp:20:run:",
        }
    ]

    dag = build_mechanism_dag(
        bindings,
        "state",
        helper_expressions=[helper],
        call_statements=calls,
    )

    assert any(
        vertex.kind == "operation" and vertex.variable == "value"
        for vertex in dag.vertices
    )


def test_exact_local_terminal_is_selected_before_call_context_cloning():
    helper_callable = "helper.cpp:10:gen:value"
    caller_callable = "caller.cpp:1:run:"
    helper = _fake_helper(
        name="gen",
        file="helper.cpp",
        line=10,
        evidence="void gen(float value)",
        assignments={"state": "value"},
        return_expression="",
    )
    helper.update(
        {
            "callable_id": helper_callable,
            "parameters": ["value"],
        }
    )
    writer = _fake_binding(
        binding_id="state",
        target="state",
        expression="value",
        file="helper.cpp",
        line=12,
        logged_signal="",
        function="gen",
    )
    writer["callable_id"] = helper_callable
    writer["function_parameters"] = ["value"]
    structure = SourceStructureIndex(
        callables_by_id={
            helper_callable: {
                "callable_id": helper_callable,
                "name": "gen",
                "parameters": ["value"],
            },
            caller_callable: {
                "callable_id": caller_callable,
                "name": "run",
                "parameters": [],
            },
        }
    )
    enriched_writer = structure.enrich_bindings([writer])[0]
    call = structure.enrich_calls(
        [
            {
                "name": "gen",
                "args": ["input"],
                "file": "caller.cpp",
                "line": 20,
                "function": "run",
                "callable_id": caller_callable,
            }
        ]
    )[0]
    terminal_identity = structure.symbol_identity(
        "state",
        file="helper.cpp",
        callable_id=helper_callable,
        function_name="gen",
        function_parameters=["value"],
    )

    dag = build_mechanism_dag(
        [enriched_writer],
        "state",
        terminal_file="helper.cpp",
        terminal_identity=terminal_identity,
        helper_expressions=[helper],
        call_statements=[call],
        source_structure=structure,
    )

    state = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation" and vertex.variable == "state"
    )
    assert state.metadata.get("is_terminal") is True
    assert state.metadata.get("call_instance_scope")
    assert any(
        vertex.kind == "operation"
        and vertex.variable == "value"
        and vertex.expression == "input"
        for vertex in dag.vertices
    )


def test_statement_call_instances_do_not_share_formal_actuals():
    helper_callable = "helper.cpp:10:fill:value,sp"
    helper = _fake_helper(
        name="fill",
        file="helper.cpp",
        line=10,
        evidence="void fill(float value, output_s *sp)",
        assignments={"sp.alt": "value"},
        return_expression="",
    )
    helper.update(
        {
            "callable_id": helper_callable,
            "parameters": ["value", "sp"],
            "pointer_output_writes": [
                {"param": "sp", "field": "alt", "expression": "value"}
            ],
        }
    )
    writer = _fake_binding(
        binding_id="writer",
        target="sp.alt",
        expression="value",
        file="helper.cpp",
        line=12,
        logged_signal="",
        function="fill",
    )
    writer["callable_id"] = helper_callable
    writer["function_parameters"] = ["value", "sp"]
    calls = [
        {
            "name": "fill",
            "args": ["first", "&out_a"],
            "file": "caller.cpp",
            "line": 20,
            "function": "run",
            "callable_id": "caller.cpp:1:run:",
        },
        {
            "name": "fill",
            "args": ["second", "&out_b"],
            "file": "caller.cpp",
            "line": 30,
            "function": "run",
            "callable_id": "caller.cpp:1:run:",
        },
    ]

    expected = {"out_a.alt": "first", "out_b.alt": "second"}
    for terminal, actual in expected.items():
        dag = build_mechanism_dag(
            [writer],
            terminal,
            terminal_file="caller.cpp",
            helper_expressions=[helper],
            call_statements=calls,
        )
        formal_ops = [
            vertex
            for vertex in dag.vertices
            if vertex.kind == "operation" and vertex.variable == "value"
        ]
        assert [(vertex.expression, vertex.line) for vertex in formal_ops] == [
            (actual, 20 if actual == "first" else 30)
        ]


def test_statement_call_rebinds_structured_formal_to_caller_storage():
    helper_callable = "helper.cpp:10:fill:item,sp"
    helper = _fake_helper(
        name="fill",
        file="helper.cpp",
        line=10,
        evidence="void fill(const item_s &item, output_s *sp)",
        assignments={"sp.alt": "item.altitude"},
        return_expression="",
    )
    helper.update(
        {
            "callable_id": helper_callable,
            "parameters": ["item", "sp"],
            "pointer_output_writes": [
                {
                    "param": "sp",
                    "field": "alt",
                    "expression": "item.altitude",
                }
            ],
        }
    )
    writer = _fake_binding(
        binding_id="writer",
        target="sp.alt",
        expression="item.altitude",
        file="helper.cpp",
        line=12,
        logged_signal="",
        function="fill",
    )
    writer["callable_id"] = helper_callable
    writer["function_parameters"] = ["item", "sp"]
    call = {
        "name": "fill",
        "args": ["source_item", "&out"],
        "file": "caller.cpp",
        "line": 20,
        "function": "run",
        "callable_id": "caller.cpp:1:run:",
    }

    dag = build_mechanism_dag(
        [writer],
        "out.alt",
        terminal_file="caller.cpp",
        helper_expressions=[helper],
        call_statements=[call],
    )

    assert "item.altitude" not in dag.unresolved_symbols
    assert "source_item.altitude" in dag.unresolved_symbols


def test_direct_reference_output_does_not_create_actual_formal_cycle():
    helper_callable = "helper.cpp:10:fill:value,out"
    helper = _fake_helper(
        name="fill",
        file="helper.cpp",
        line=10,
        evidence="void fill(float value, float &out)",
        assignments={"out": "value * 2"},
        return_expression="",
    )
    helper.update(
        {
            "callable_id": helper_callable,
            "parameters": ["value", "out"],
            "pointer_output_writes": [
                {
                    "param": "out",
                    "field": "",
                    "expression": "value * 2",
                }
            ],
        }
    )
    writer = _fake_binding(
        binding_id="writer",
        target="out",
        expression="value * 2",
        file="helper.cpp",
        line=12,
        logged_signal="",
        function="fill",
    )
    writer["callable_id"] = helper_callable
    writer["function_parameters"] = ["value", "out"]
    call = {
        "name": "fill",
        "args": ["input", "result"],
        "file": "caller.cpp",
        "line": 20,
        "function": "run",
        "callable_id": "caller.cpp:1:run:",
    }

    dag = build_mechanism_dag(
        [writer],
        "result",
        terminal_file="caller.cpp",
        helper_expressions=[helper],
        call_statements=[call],
    )

    operations = [vertex for vertex in dag.vertices if vertex.kind == "operation"]
    assert any(
        vertex.variable == "result" and vertex.expression == "out"
        for vertex in operations
    )
    assert any(
        vertex.variable == "out" and vertex.expression == "value * 2"
        for vertex in operations
    )
    assert not any(
        vertex.variable == "out" and vertex.expression == "result"
        for vertex in operations
    )


def test_expression_call_instances_have_private_formals_and_returns():
    helper = _fake_helper(
        name="scale",
        file="helper.cpp",
        line=10,
        evidence="float scale(float value)",
        assignments={},
        return_expression="value * 2",
    )
    helper.update(
        {
            "callable_id": "helper.cpp:10:scale:value",
            "parameters": ["value"],
        }
    )
    bindings = [
        _fake_binding(
            binding_id="first",
            target="output",
            expression="scale(first)",
            file="caller.cpp",
            line=20,
            logged_signal="",
            control_predicates=["mode == 0"],
            function="run",
        ),
        _fake_binding(
            binding_id="second",
            target="output",
            expression="scale(second)",
            file="caller.cpp",
            line=30,
            logged_signal="",
            control_predicates=["mode != 0"],
            function="run",
        ),
    ]
    for binding in bindings:
        binding["callable_id"] = "caller.cpp:1:run:"

    dag = build_mechanism_dag(
        bindings,
        "output",
        terminal_file="caller.cpp",
        helper_expressions=[helper],
    )

    returns = [
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation" and "__return__" in str(vertex.variable)
    ]
    formals = [
        vertex
        for vertex in dag.vertices
        if vertex.kind == "evidence" and vertex.sub_kind == "helper_parameter"
    ]
    assert len(returns) == 2
    assert len(formals) == 2
    assert len({vertex.metadata["call_site_id"] for vertex in returns}) == 2
    incoming = {
        formal.id: {
            edge.source_id for edge in dag.edges if edge.target_id == formal.id
        }
        for formal in formals
    }
    evidence = {
        vertex.signal_name: vertex.id
        for vertex in dag.vertices
        if vertex.kind == "evidence" and vertex.sub_kind == "opaque_symbol"
    }
    assert sorted(
        source in predecessors
        for source in (evidence["first"], evidence["second"])
        for predecessors in incoming.values()
    ) == [False, False, True, True]


def test_unrelated_call_does_not_probe_helper_provider():
    probes: list[str] = []

    def provider(name, reference=None):
        probes.append(name)
        return []

    binding = _fake_binding(
        binding_id="answer",
        target="answer",
        expression="input_value",
        file="main.cpp",
        line=10,
        logged_signal="",
        function="run",
    )
    binding["callable_id"] = "main.cpp:1:run:"
    dag = build_mechanism_dag(
        [binding],
        "answer",
        terminal_file="main.cpp",
        call_statements=[
            {
                "name": "unrelated",
                "args": ["value"],
                "file": "other.cpp",
                "line": 20,
                "function": "tick",
                "callable_id": "other.cpp:1:tick:",
            }
        ],
        helper_body_provider=provider,
    )

    assert probes == []
    assert [
        (vertex.variable, vertex.expression)
        for vertex in dag.vertices
        if vertex.kind == "operation"
    ] == [("answer", "input_value")]


def test_loaded_read_only_call_does_not_become_a_writer():
    helper = _fake_helper(
        name="observe",
        file="main.cpp",
        line=20,
        evidence="float observe(float value)",
        assignments={},
        return_expression="value",
    )
    helper.update(
        {
            "callable_id": "main.cpp:20:observe:value",
            "parameters": ["value"],
        }
    )
    binding = _fake_binding(
        binding_id="answer",
        target="answer",
        expression="input_value",
        file="main.cpp",
        line=10,
        logged_signal="",
        function="run",
    )
    binding["callable_id"] = "main.cpp:1:run:"

    dag = build_mechanism_dag(
        [binding],
        "answer",
        terminal_file="main.cpp",
        helper_expressions=[helper],
        call_statements=[
            {
                "name": "observe",
                "args": ["answer"],
                "file": "main.cpp",
                "line": 30,
                "function": "run",
                "callable_id": "main.cpp:1:run:",
            }
        ],
    )

    operations = [
        (vertex.variable, vertex.expression)
        for vertex in dag.vertices
        if vertex.kind == "operation"
    ]
    assert operations == [("answer", "input_value")]
    assert "observe" not in dag.unresolved_symbols


def test_unloaded_possible_call_effect_is_a_typed_gap_not_inline_fetch():
    probes: list[str] = []

    def provider(name, reference=None):
        probes.append(name)
        return []

    dag = build_mechanism_dag(
        [],
        "out.alt",
        terminal_file="main.cpp",
        call_statements=[
            {
                "name": "fill",
                "args": ["source", "&out"],
                "file": "main.cpp",
                "line": 30,
                "function": "run",
                "callable_id": "main.cpp:1:run:",
                "evidence": "fill(source, &out);",
            }
        ],
        helper_body_provider=provider,
    )

    assert probes == []
    assert "fill" in dag.unresolved_symbols
    reference = next(
        item for item in dag.unresolved_references if item.symbol == "fill"
    )
    assert reference.kind == "callable"
    assert reference.callable_id == "main.cpp:1:run:"
    assert reference.argument_count == 2


def test_receiver_call_result_does_not_admit_other_receiver_methods():
    probes: list[str] = []

    def provider(name, reference=None):
        probes.append(name)
        return []

    binding = _fake_binding(
        binding_id="answer",
        target="answer",
        expression="client.get_value().field",
        file="main.cpp",
        line=10,
        logged_signal="",
        function="run",
    )
    binding["callable_id"] = "main.cpp:1:run:"
    dag = build_mechanism_dag(
        [binding],
        "answer",
        terminal_file="main.cpp",
        call_statements=[
            {
                "name": "get_value",
                "receiver": "client",
                "args": [],
                "file": "main.cpp",
                "line": 10,
                "function": "run",
                "callable_id": "main.cpp:1:run:",
            },
            {
                "name": "unrelated_mutation",
                "receiver": "client",
                "args": [],
                "file": "main.cpp",
                "line": 20,
                "function": "run",
                "callable_id": "main.cpp:1:run:",
            },
        ],
        helper_body_provider=provider,
    )

    assert probes == ["get_value"]
    assert "unrelated_mutation" not in dag.unresolved_symbols
    assert any(
        item.kind == "callable" and item.symbol == "client.get_value"
        for item in dag.unresolved_references
    )
    assert not any(
        item.kind == "symbol" and item.symbol == "client.get_value"
        for item in dag.unresolved_references
    )


def test_source_boundary_receiver_does_not_expand_framework_method():
    probes: list[str] = []

    def provider(name, reference=None):
        probes.append(name)
        return []

    binding = _fake_binding(
        binding_id="answer",
        target="answer",
        expression="_channel.update(&sample)",
        file="main.cpp",
        line=10,
        logged_signal="",
        function="Controller::run",
    )
    structure = SourceStructureIndex(
        direct_bases={"Controller": set(), "Channel": set()},
        declared_classes={"Controller", "Channel"},
        members={
            ("Controller", "_channel"): {
                "name": "_channel",
                "owner": "Controller",
                "type": "Channel",
            }
        },
    )
    binding = structure.enrich_bindings([binding])[0]
    dag = build_mechanism_dag(
        [binding],
        "answer",
        terminal_file="main.cpp",
        call_statements=structure.enrich_calls(
            [
                {
                    "name": "update",
                    "receiver": "_channel",
                    "args": ["&sample"],
                    "file": "main.cpp",
                    "line": 10,
                    "function": "Controller::run",
                    "callable_id": "main.cpp:1:Controller::run:",
                }
            ]
        ),
        boundary_bindings=[
            {
                "source_symbol": "_channel",
                "topic": "sample_stream",
                "direction": "subscribe",
                "source_owner": "Controller",
                "endpoint_kind": "member",
                "file": "main.hpp",
            }
        ],
        source_structure=structure,
        helper_body_provider=provider,
    )

    assert probes == []
    assert any(
        item.kind == "callable"
        and item.symbol == "_channel.update"
        and item.receiver == "_channel"
        for item in dag.unresolved_references
    )


def test_unresolved_receiver_calls_share_one_provider_lookup_identity():
    probes: list[str] = []

    def provider(name, reference=None):
        probes.append(name)
        return []

    binding = _fake_binding(
        binding_id="answer",
        target="answer",
        expression="client.compute(left) + client.compute(right)",
        file="main.cpp",
        line=10,
        logged_signal="",
        function="Controller::run",
    )
    structure = SourceStructureIndex(
        direct_bases={"Controller": set(), "Client": set()},
        declared_classes={"Controller", "Client"},
        members={
            ("Controller", "client"): {
                "name": "client",
                "owner": "Controller",
                "type": "Client",
            }
        },
    )
    binding = structure.enrich_bindings([binding])[0]
    build_mechanism_dag(
        [binding],
        "answer",
        terminal_file="main.cpp",
        source_structure=structure,
        helper_body_provider=provider,
    )

    assert probes == ["compute"]


def test_helper_body_provider_lazily_supplies_missing_helper():
    """When the initial helper set does not contain a called helper name,
    the DAG builder should invoke ``helper_body_provider`` to fetch it.
    Cross-file callee then materializes and its return feeds the caller."""
    provided_helper = _fake_helper(
        name="geo::haversine_distance",
        file="lib/geo/geo.cpp",
        line=42,
        evidence="float haversine_distance(...)",
        assignments={},
        return_expression="R * atan2(sqrt(a), sqrt(1 - a))",
    )
    fetches: list[str] = []

    def provider(name: str):
        fetches.append(name)
        if name == "haversine_distance":
            return provided_helper
        return None

    bindings = [
        _fake_binding(
            binding_id="caller",
            target="dist_squared",
            expression="haversine_distance(a, b)",
            file="rtl.cpp",
            line=300,
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "dist_squared",
        helper_expressions=[],
        helper_body_provider=provider,
    )

    assert "haversine_distance" in fetches
    return_ops = [
        v for v in dag.vertices
        if v.kind == "operation" and v.sub_kind == "helper_call"
    ]
    assert return_ops, "helper subgraph terminal not emitted"


def test_helper_body_provider_probes_each_name_at_most_once():
    calls: list[str] = []

    def provider(name: str):
        calls.append(name)
        return None

    bindings = [
        _fake_binding(
            binding_id="b1",
            target="x",
            expression="mystery(a) + mystery(b) + mystery(c)",
            file="rtl.cpp",
            line=10,
        ),
    ]
    build_mechanism_dag(bindings, "x", helper_body_provider=provider)
    assert calls.count("mystery") == 1


def test_parameter_predicate_metadata_surfaces_on_branch_vertex():
    """A branch predicate matching a ``ParameterPredicateRef`` should carry
    the operator + compared_value on the branch vertex metadata so
    downstream evaluators don't re-run ``parse_parameter_predicate``."""
    predicate = "_param_rtl_type.get() != RTL_TYPE_HOME_OR_RALLY"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
    ]
    parameter_predicate = {
        "name": "RTL_TYPE",
        "predicate": predicate,
        "file": "rtl.cpp",
        "line": 245,
        "evidence": predicate,
        "member": None,
        "operator": "!=",
        "compared_value": "RTL_TYPE_HOME_OR_RALLY",
    }
    dag = build_mechanism_dag(
        bindings, "_rtl_alt", parameter_predicates=[parameter_predicate]
    )

    branch = next(v for v in dag.vertices if v.kind == "branch")
    assert branch.metadata.get("parameter") == "RTL_TYPE"
    assert branch.metadata.get("operator") == "!="
    assert branch.metadata.get("compared_value") == "RTL_TYPE_HOME_OR_RALLY"


def test_parameter_values_populate_resolved_value_on_evidence_constant():
    """A bare PX4-parameter-shaped name mentioned in source should carry
    the runtime value on the constant vertex metadata when the parameter
    is present in ``parameter_values``."""
    predicate = "cruising_speed > FW_AIRSPD_TRIM"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        parameter_values={"FW_AIRSPD_TRIM": 15.0},
    )
    const_vertex = next(
        v for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "constant"
        and v.signal_name == "FW_AIRSPD_TRIM"
    )
    assert const_vertex.metadata.get("value") == 15.0
    assert const_vertex.metadata.get("source") == "parameter"


def test_symbol_bindings_flat_dict_no_longer_consumed():
    """The DAG builder no longer reads ``binding_index.symbol_bindings``.
    Even a well-formed entry present in the flat dict must not produce an
    ``evidence:logged_signal`` vertex — the only paths are graph-native
    (helper chain, struct variable)."""
    predicate = "_navigator.get_vstatus().vehicle_type == 1"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
            control_predicates=[predicate],
        ),
    ]
    # The DAG never consulted a flat symbol_bindings table (it's gone from
    # the builder entirely now), so there is nothing to inject: with no
    # helper return_type and no struct-variable info, the chain simply
    # doesn't resolve to a logged signal.
    dag = build_mechanism_dag(bindings, "_rtl_alt")

    phantom = [
        v for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "logged_signal"
        and v.signal_name == "vehicle_status.vehicle_type"
    ]
    assert not phantom, "flat symbol_bindings should no longer produce evidence vertices"


def test_predicate_lowering_substitutes_enum_and_helper_chain_derivation():
    """Branch ``predicate_lowered`` should reflect enum resolution AND
    graph-native helper-chain substitution — no flat symbol_bindings
    entry required."""
    helper = _fake_helper(
        name="Navigator::get_vstatus",
        file="navigator.cpp",
        line=100,
        evidence="vehicle_status_s * Navigator::get_vstatus()",
        assignments={},
        return_expression="_vehicle_status",
    )
    helper["return_type"] = "vehicle_status_s *"
    predicate = "_navigator.get_vstatus().vehicle_type == VEHICLE_TYPE_ROTARY_WING"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
            control_predicates=[predicate],
        ),
    ]
    # Enum-shaped constant → resolved value, from a real source binding.
    bindings.append(
        _fake_binding(
            binding_id="const",
            target="VEHICLE_TYPE_ROTARY_WING",
            expression="1",
            file="rtl.cpp",
            line=1,
            declaration_kind="enum",
        )
    )

    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        inventory=_fake_inventory({"vehicle_status": ["vehicle_type"]}),
        helper_expressions=[helper],
        boundary_bindings=[
            _boundary("_vehicle_status", "vehicle_status", file="navigator.cpp")
        ],
    )

    branch = next(v for v in dag.vertices if v.kind == "branch")
    assert "vehicle_status.vehicle_type" in (branch.predicate_lowered or "")
    assert "VEHICLE_TYPE_ROTARY_WING" not in (branch.predicate_lowered or "")
    variables = branch.metadata.get("variables") or {}
    assert any(
        "get_vstatus" in key and value == "vehicle_status.vehicle_type"
        for key, value in variables.items()
    )


def test_derive_pointer_output_bindings_shared_by_profiler_and_dag():
    """Both the profiler's flatten and the DAG builder's graph-native
    emission route through :func:`derive_pointer_output_bindings`, so
    every call site (however discovered) produces the same target/expression
    entries. Test the shared function directly."""
    from flight_log_agent.analysis.mechanism_dag import derive_pointer_output_bindings

    pointer_writes = [
        {"param": "sp", "field": "alt", "expression": "_rtl_alt"},
        {"param": "sp", "field": "lat", "expression": "_destination.lat"},
    ]
    pointer_params = {"sp": 1}
    call_args = ["item", "&triplet.current"]
    bindings = derive_pointer_output_bindings(pointer_writes, pointer_params, call_args)

    targets = {(b["target"], b["expression"]) for b in bindings}
    assert ("triplet.current.alt", "_rtl_alt") in targets
    assert ("triplet.current.lat", "_destination.lat") in targets


def test_derive_pointer_output_bindings_skips_missing_arg_position():
    from flight_log_agent.analysis.mechanism_dag import derive_pointer_output_bindings

    pointer_writes = [{"param": "sp", "field": "alt", "expression": "_rtl_alt"}]
    pointer_params = {"sp": 3}  # Position exceeds call_args length.
    call_args = ["item"]
    assert derive_pointer_output_bindings(pointer_writes, pointer_params, call_args) == []


def test_helper_chain_resolves_via_return_type_and_schema():
    """A chain like ``_navigator.get_vstatus().vehicle_type`` should
    resolve to ``vehicle_status.vehicle_type`` purely from the helper's
    return_type + PX4 msg schema — without any entry in the flat
    symbol_bindings dict."""
    helper = _fake_helper(
        name="Navigator::get_vstatus",
        file="navigator.cpp",
        line=100,
        evidence="vehicle_status_s *Navigator::get_vstatus()",
        assignments={},
        return_expression="_vehicle_status",
    )
    helper["return_type"] = "vehicle_status_s *"
    predicate = "_navigator.get_vstatus().vehicle_type == 1"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
            control_predicates=[predicate],
        ),
    ]
    # The chain resolves purely from graph derivation (helper return_type
    # + schema lookup); the DAG has no flat symbol_bindings table at all.
    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        inventory=_fake_inventory({"vehicle_status": ["vehicle_type"]}),
        helper_expressions=[helper],
        boundary_bindings=[
            _boundary("_vehicle_status", "vehicle_status", file="navigator.cpp")
        ],
    )

    branch = next(v for v in dag.vertices if v.kind == "branch")
    assert "vehicle_status.vehicle_type" in (branch.predicate_lowered or "")
    variables = branch.metadata.get("variables") or {}
    # Each substitution keys on the original chain-with-field text.
    assert any(
        "get_vstatus" in key and value == "vehicle_status.vehicle_type"
        for key, value in variables.items()
    )


def test_helper_chain_skipped_when_return_type_not_struct():
    """A helper returning a scalar (``float``) should not produce any
    graph-derived source→logged binding — the derivation returns None
    and the predicate stays unlowered for that chain."""
    helper = _fake_helper(
        name="RTL::calc_alt",
        file="rtl.cpp",
        line=100,
        evidence="float RTL::calc_alt()",
        assignments={},
        return_expression="42.0",
    )
    helper["return_type"] = "float"
    predicate = "calc_alt().value > 0"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt", helper_expressions=[helper])

    branch = next(v for v in dag.vertices if v.kind == "branch")
    # No substitution should have occurred — the predicate text stays as-is.
    assert "calc_alt" in (branch.predicate_lowered or "")


def test_helper_chain_skipped_when_topic_not_in_schema():
    """A helper returning ``mystery_topic_s`` where ``mystery_topic`` is
    not in the PX4 schema/logged catalogue should not silently produce
    a phantom binding — the derivation returns None."""
    helper = _fake_helper(
        name="Mystery::get_thing",
        file="mystery.cpp",
        line=1,
        evidence="mystery_topic_s *Mystery::get_thing()",
        assignments={},
        return_expression="_thing",
    )
    helper["return_type"] = "mystery_topic_s *"
    predicate = "_navigator.get_thing().value > 0"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt", helper_expressions=[helper])

    branch = next(v for v in dag.vertices if v.kind == "branch")
    # Nothing was substituted; the branch predicate keeps its source form.
    assert "get_thing" in (branch.predicate_lowered or "")


def test_derive_topic_from_return_type_variants():
    """The topic derivation should handle pointer, reference, namespaced,
    and bare-struct return types — and return None for scalars."""
    from flight_log_agent.analysis.mechanism_dag import _derive_topic_from_return_type

    assert _derive_topic_from_return_type("vehicle_status_s *") == "vehicle_status"
    assert _derive_topic_from_return_type("vehicle_status_s&") == "vehicle_status"
    assert _derive_topic_from_return_type("vehicle_status_s") == "vehicle_status"
    assert _derive_topic_from_return_type("px4::vehicle_status_s *") == "vehicle_status"
    assert _derive_topic_from_return_type("float") is None
    assert _derive_topic_from_return_type("") is None
    assert _derive_topic_from_return_type(None) is None


def test_struct_var_field_resolves_via_source_boundary():
    """A copied local resolves only through its source-proven subscription."""
    helper = _fake_helper(
        name="Example::check",
        file="example.cpp",
        line=100,
        evidence="float Example::check()",
        assignments={},
        return_expression="1.0",
    )
    helper["struct_variables"] = {"vstatus": "vehicle_status_s"}
    predicate = "vstatus.vehicle_type == 1"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        inventory=_fake_inventory({"vehicle_status": ["vehicle_type"]}),
        helper_expressions=[helper],
        boundary_bindings=[
            _boundary("vstatus", "vehicle_status", file="rtl.cpp")
        ],
    )

    branch = next(v for v in dag.vertices if v.kind == "branch")
    assert "vehicle_status.vehicle_type" in (branch.predicate_lowered or "")


def test_struct_type_without_source_boundary_does_not_become_evidence():
    """A ``*_s`` declaration proves schema shape, not uORB provenance."""
    predicate = "vstatus.vehicle_type == 1"
    binding = _fake_binding(
        binding_id="b1",
        target="_rtl_alt",
        expression="42",
        file="rtl.cpp",
        line=1,
        control_predicates=[predicate],
    )
    # Carry the struct-variable map on the binding dict as if it came
    # from a SourceAssignmentRef.
    binding["struct_variables"] = {"vstatus": "vehicle_status_s"}
    dag = build_mechanism_dag(
        [binding],
        "_rtl_alt",
        inventory=_fake_inventory({"vehicle_status": ["vehicle_type"]}),
    )

    branch = next(v for v in dag.vertices if v.kind == "branch")
    assert "vehicle_status.vehicle_type" not in (branch.predicate_lowered or "")
    assert not [
        vertex
        for vertex in dag.vertices
        if vertex.kind == "evidence" and vertex.sub_kind == "logged_signal"
    ]


def test_struct_var_field_skipped_when_topic_unknown():
    """When ``var.field`` derives a topic that isn't in the trusted signal
    catalogue, the predicate should stay unlowered — no phantom binding."""
    helper = _fake_helper(
        name="Example::check",
        file="example.cpp",
        line=100,
        evidence="float Example::check()",
        assignments={},
        return_expression="1.0",
    )
    helper["struct_variables"] = {"mystery": "mystery_topic_s"}
    predicate = "mystery.value == 1"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=1,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt", helper_expressions=[helper])

    branch = next(v for v in dag.vertices if v.kind == "branch")
    assert "mystery" in (branch.predicate_lowered or "")


def test_parameter_alias_resolves_member_differing_from_param_name():
    """A PX4 param whose member name differs from the param name
    (`_param_rtl_cone_half_angle_deg` ↔ RTL_CONE_ANG) resolves only via the
    parameter_aliases map (the DEFINE_PARAMETERS member→name mapping), not
    the `_param_<snake>→UPPER` heuristic."""
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="_param_rtl_cone_half_angle_deg.get()",
            file="rtl.cpp",
            line=245,
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        parameter_names={"RTL_CONE_ANG"},
        parameter_aliases={"_param_rtl_cone_half_angle_deg": "RTL_CONE_ANG"},
    )
    params = {
        v.signal_name for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "parameter"
    }
    assert "RTL_CONE_ANG" in params


def test_parameter_alias_not_resolved_without_map():
    """Without the alias map, the same member is NOT resolvable (the
    heuristic would produce RTL_CONE_HALF_ANGLE_DEG, which isn't the param)."""
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="_param_rtl_cone_half_angle_deg.get()",
            file="rtl.cpp",
            line=245,
        ),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt", parameter_names={"RTL_CONE_ANG"})
    params = {
        v.signal_name for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "parameter"
    }
    assert "RTL_CONE_ANG" not in params


def test_cpp_predicate_symbols_are_extracted_for_alias_and_chain():
    """A C++ branch predicate with `&&`, `->`, `::` and a cast must still
    yield its symbols: the aliased parameter resolves and the accessor
    chain lowers — previously the `&&` made ast extraction return nothing,
    so no evidence edges at all."""
    predicate = (
        "_param_rtl_cone_half_angle_deg.get() > 0 "
        "&& _navigator->get_vstatus()->vehicle_type "
        "== vehicle_status_s::VEHICLE_TYPE_ROTARY_WING"
    )
    helper = _fake_helper(
        name="Navigator::get_vstatus",
        file="navigator.h",
        line=10,
        evidence="vehicle_status_s *Navigator::get_vstatus()",
        assignments={},
        return_expression="_vstatus",
    )
    helper["return_type"] = "vehicle_status_s"
    bindings = [
        _fake_binding(
            binding_id="b1",
            target="_rtl_alt",
            expression="42",
            file="rtl.cpp",
            line=245,
            control_predicates=[predicate],
        ),
    ]
    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        inventory=_fake_inventory({"vehicle_status": ["vehicle_type"]}),
        helper_expressions=[helper],
        parameter_names={"RTL_CONE_ANG"},
        parameter_aliases={"_param_rtl_cone_half_angle_deg": "RTL_CONE_ANG"},
        boundary_bindings=[
            _boundary("_vstatus", "vehicle_status", file="navigator.h")
        ],
    )
    params = {v.signal_name for v in dag.vertices
              if v.kind == "evidence" and v.sub_kind == "parameter"}
    logged = {v.signal_name for v in dag.vertices
              if v.kind == "evidence" and v.sub_kind == "logged_signal"}
    assert "RTL_CONE_ANG" in params
    assert "vehicle_status.vehicle_type" in logged


def test_helper_branch_selects_value_without_gating_return_reachability():
    """Alternative helper return values are selection, not conjunction."""
    helper = _fake_helper(
        name="RTL::calc",
        file="rtl.cpp",
        line=10,
        evidence="float RTL::calc()",
        assignments={},
        return_expression="fallback_alt",
        branches=[{"condition": "cond_flag > 0", "expression": "early_alt"}],
    )
    bindings = [
        _fake_binding(
            binding_id="caller",
            target="_rtl_alt",
            expression="calc()",
            file="rtl.cpp",
            line=1,
        ),
        # producers for the fallback and the branch's alternate value
        _fake_binding(binding_id="fb", target="fallback_alt", expression="home.alt",
                      file="rtl.cpp", line=2),
        _fake_binding(binding_id="ea", target="early_alt", expression="mission.alt",
                      file="rtl.cpp", line=3),
    ]
    dag = build_mechanism_dag(bindings, "_rtl_alt", helper_expressions=[helper])

    branch = next((v for v in dag.vertices if v.kind == "branch"), None)
    assert branch is not None, "no branch vertex emitted from helper.branches"
    selection_out = [
        e for e in dag.edges if e.source_id == branch.id and e.kind == "selection"
    ]
    assert selection_out
    assert not [
        edge
        for edge in dag.edges
        if edge.source_id == branch.id and edge.kind == "control"
    ]

    # the branch's alternate return value's producer feeds the return op
    return_op = next(v for v in dag.vertices if v.kind == "operation"
                     and "__return__" in str(v.variable))
    assert any(e.target_id == return_op.id for e in selection_out)


def test_helper_branches_on_same_line_keep_parser_source_identity():
    helper = _fake_helper(
        name="Control::select",
        file="control.cpp",
        line=10,
        evidence="float Control::select()",
        assignments={},
        return_expression="fallback",
        branches=[
            {
                "condition": "valid",
                "expression": "first",
                "file": "control.cpp",
                "line": 20,
                "source_site_id": "control.cpp:100:120:return_statement",
            },
            {
                "condition": "valid",
                "expression": "second",
                "file": "control.cpp",
                "line": 20,
                "source_site_id": "control.cpp:140:160:return_statement",
            },
        ],
    )
    dag = build_mechanism_dag(
        [
            _fake_binding(
                binding_id="caller",
                target="output",
                expression="select()",
                file="control.cpp",
                line=1,
            )
        ],
        "output",
        helper_expressions=[helper],
    )

    branches = [vertex for vertex in dag.vertices if vertex.kind == "branch"]
    assert len(branches) == 2
    assert len({vertex.id for vertex in branches}) == 2


def test_shared_producer_vertex_reused_across_two_consumers():
    """A symbol written once but read by two distinct consumers must resolve
    to a SINGLE producer operation vertex (graph-native reuse), with one data
    edge from that shared producer into each consumer — not a duplicated
    producer per consumer (the flattened-BindingIndex failure mode)."""
    bindings = [
        _fake_binding(binding_id="s", target="shared", expression="gpos.alt",
                      file="rtl.cpp", line=1),
        _fake_binding(binding_id="a", target="consumer_a", expression="shared * 2",
                      file="rtl.cpp", line=2),
        _fake_binding(binding_id="b", target="consumer_b", expression="shared + 3",
                      file="rtl.cpp", line=3),
        _fake_binding(binding_id="t", target="terminal",
                      expression="consumer_a + consumer_b",
                      file="rtl.cpp", line=4),
    ]
    dag = build_mechanism_dag(bindings, "terminal")

    shared_ops = [v for v in dag.vertices
                  if v.kind == "operation" and v.variable == "shared"]
    assert len(shared_ops) == 1, "shared producer duplicated instead of reused"
    shared_id = shared_ops[0].id

    consumer_ids = {v.id for v in dag.vertices
                    if v.kind == "operation" and v.variable in {"consumer_a", "consumer_b"}}
    fanout = [e for e in dag.edges
              if e.source_id == shared_id and e.target_id in consumer_ids and e.kind == "data"]
    assert len(fanout) == 2, "shared producer must fan out to both consumers"


def test_cpp_qualified_expression_wires_inner_logged_signal_not_dropped():
    """An operation whose RHS is a ``::``-qualified call must still wire the
    logged signal referenced inside it. Regression for the weathervane
    ``R_yaw = matrix::Eulerf(0, 0, -vehicle_local_position.heading)`` case:
    edge-wiring extracted symbols from the raw ``::`` expression, which
    source_expression_names returns [] for, so the valid logged signal was
    silently dropped — not wired, not even marked unresolved."""
    bindings = [
        _fake_binding(
            binding_id="ry",
            target="R_yaw",
            expression="matrix::Eulerf(0.0f, 0.0f, -vehicle_local_position.heading)",
            file="wv.cpp",
            line=1,
        ),
    ]
    dag = build_mechanism_dag(
        bindings, "R_yaw", logged_signals={"vehicle_local_position.heading"}
    )

    ev = [
        v for v in dag.vertices
        if v.kind == "evidence"
        and v.sub_kind == "logged_signal"
        and v.signal_name == "vehicle_local_position.heading"
    ]
    assert ev, "logged signal inside a :: expression was dropped from wiring"
    ryaw = next(v for v in dag.vertices if v.variable == "R_yaw")
    assert any(
        e.source_id == ev[0].id and e.target_id == ryaw.id for e in dag.edges
    ), "logged signal not connected to the operation that references it"


def test_member_and_local_spellings_are_distinct_identities():
    """NPFG regression shape, root-caused: ``_lateral_accel`` (one
    module's member) and ``lateral_accel`` (another module's local) are
    DIFFERENT variables and must never fuse — exact identity keeps them
    apart even with no ``terminal_file`` scoping at all."""
    bindings = [
        _fake_binding(binding_id="l1", target="_lateral_accel",
                      expression="_K_L1 * ground_speed", file="l1.cpp", line=10),
        _fake_binding(binding_id="npfg", target="lateral_accel",
                      expression="lateralAccel(air_vel, airspeed)", file="npfg.cpp", line=20),
    ]

    unscoped = build_mechanism_dag(bindings, "_lateral_accel")
    sliced = {v.file for v in unscoped.vertices if v.kind == "operation"
              and v.variable in {"_lateral_accel", "lateral_accel"}}
    assert sliced == {"l1.cpp"}, "member and local spellings fused"


def test_terminal_file_scopes_writers_to_named_module():
    """Two modules writing the SAME spelling: ``terminal_file`` keeps
    only that file's writers; a hint naming a writer-less file falls
    back instead of emptying the slice."""
    bindings = [
        _fake_binding(binding_id="l1", target="_lateral_accel",
                      expression="_K_L1 * ground_speed", file="l1.cpp", line=10),
        _fake_binding(binding_id="npfg", target="_lateral_accel",
                      expression="lateralAccel(air_vel, airspeed)", file="npfg.cpp", line=20),
    ]

    scoped = build_mechanism_dag(bindings, "_lateral_accel", terminal_file="l1.cpp")
    terminal_ops = [v for v in scoped.vertices if v.kind == "operation"
                    and v.variable == "_lateral_accel"]
    assert {v.file for v in terminal_ops} == {"l1.cpp"}

    # Hint naming a file with no writers must fall back, not empty the slice.
    fallback = build_mechanism_dag(bindings, "_lateral_accel", terminal_file="other.cpp")
    assert any(v.kind == "operation" for v in fallback.vertices)


def test_symbol_producer_prefers_same_file_writer():
    """A consumer reading a symbol written in its own file AND in another
    module links to the same-file producer, not blindly the last one."""
    bindings = [
        # other-module writer indexed AFTER the same-file one would win
        # under pure producers[-1]; order the same-file writer first so a
        # last-wins bug is caught.
        _fake_binding(binding_id="own", target="shared_gain",
                      expression="own_input * 2", file="a.cpp", line=5),
        _fake_binding(binding_id="foreign", target="shared_gain",
                      expression="foreign_input * 3", file="b.cpp", line=7),
        _fake_binding(binding_id="consumer", target="output",
                      expression="shared_gain + 1", file="a.cpp", line=9),
    ]
    dag = build_mechanism_dag(bindings, "output")

    consumer = next(v for v in dag.vertices if v.variable == "output")
    producers = {v.id: v for v in dag.vertices if v.variable == "shared_gain"}
    incoming = [e for e in dag.edges if e.target_id == consumer.id
                and e.source_id in producers]
    assert len(incoming) == 1
    assert producers[incoming[0].source_id].file == "a.cpp"


def test_short_helper_names_do_not_claim_calls():
    """A helper literally named ``get`` (extracted from some loaded file)
    must not claim every ``.get()`` call in the slice, and stdlib-shaped
    short heads must not materialize helper subgraphs."""
    helper = _fake_helper(
        name="get",
        file="unrelated.hpp",
        line=5,
        evidence="uint8_t UserModeIntention::get()",
        assignments={},
        return_expression="_intended_mode",
    )
    bindings = [
        _fake_binding(binding_id="b1", target="_out",
                      expression="_param_thing.get() + max(a_value, b_value)",
                      file="rtl.cpp", line=1),
    ]
    dag = build_mechanism_dag(bindings, "_out", helper_expressions=[helper])

    assert not [v for v in dag.vertices
                if v.provenance and v.provenance.startswith("helper_return")], \
        "short-named helper was materialized"


def test_bare_short_helper_still_expands():
    """The dotted-short guard must not block a short helper called BARE
    (implicit this) — only ``.name(`` / ``->name(`` accessor heads."""
    helper = _fake_helper(
        name="gen",
        file="rtl.cpp",
        line=5,
        evidence="float Rtl::gen()",
        assignments={},
        return_expression="base_value * 2.0",
    )
    bindings = [
        _fake_binding(binding_id="b1", target="_out", expression="gen() + 1.0",
                      file="rtl.cpp", line=1),
    ]
    dag = build_mechanism_dag(bindings, "_out", helper_expressions=[helper])

    assert [v for v in dag.vertices
            if v.provenance and v.provenance.startswith("helper_return")], \
        "bare short helper was not expanded"


def test_local_symbols_never_fuse_across_files():
    """The ATT-fusion shape: two modules both have a local named ``spd``.
    The walk from module A's terminal must resolve ``spd`` only within
    A's file+function — B's same-named local is a different variable and
    its subtree (sensor_b) must not enter the slice."""
    bindings = [
        _fake_binding(binding_id="wa", target="spd", expression="sensor_a",
                      file="a.cpp", line=1, function="Alpha::run",
                      logged_signal=""),
        _fake_binding(binding_id="t", target="_out_a", expression="spd + 1.0",
                      file="a.cpp", line=2, function="Alpha::run"),
        _fake_binding(binding_id="wb", target="spd", expression="sensor_b",
                      file="b.cpp", line=3, function="Beta::run",
                      logged_signal=""),
    ]
    dag = build_mechanism_dag(bindings, "_out_a", terminal_file="a.cpp")

    spd_files = {v.file for v in dag.vertices
                 if v.kind == "operation" and v.variable == "spd"}
    assert spd_files == {"a.cpp"}
    assert not any("sensor_b" in str(v.signal_name or "") for v in dag.vertices)


def test_member_symbols_do_not_widen_to_sibling_class():
    """Same-spelled sibling members never connect through file proximity."""
    bindings = [
        _fake_binding(binding_id="t", target="_out_a", expression="_shared_member + 1.0",
                      file="a.cpp", line=1, function="Alpha::run"),
        _fake_binding(binding_id="w", target="_shared_member", expression="sensor_b",
                      file="b.cpp", line=2, function="Beta::update",
                      logged_signal=""),
    ]
    bindings, structure = _owned_bindings(
        bindings,
        bases={"Alpha": {"Base"}, "Beta": {"Base"}, "Base": set()},
        members={("Base", "_shared_member"): {"type": "float"}},
    )
    dag = build_mechanism_dag(
        bindings,
        "_out_a",
        terminal_file="a.cpp",
        source_structure=structure,
    )

    ops = {v.variable for v in dag.vertices if v.kind == "operation"}
    assert "_shared_member" not in ops
    assert "_shared_member" in dag.unresolved_symbols


def test_member_symbols_use_declared_owner_not_file_family():
    bindings = [
        _fake_binding(binding_id="t", target="_out_a", expression="_gain + 1.0",
                      file="src/modules/a/a.cpp", line=1, function="Alpha::run"),
        _fake_binding(binding_id="near", target="_gain", expression="near_source",
                      file="src/modules/a/a.hpp", line=2, function="Alpha::init",
                      logged_signal=""),
        _fake_binding(binding_id="far", target="_gain", expression="far_source",
                      file="src/modules/b/b.cpp", line=3, function="Beta::init",
                      logged_signal=""),
    ]
    bindings, structure = _owned_bindings(
        bindings,
        bases={"Alpha": set(), "Beta": set()},
        members={
            ("Alpha", "_gain"): {"type": "float"},
            ("Beta", "_gain"): {"type": "float"},
        },
    )
    dag = build_mechanism_dag(
        bindings,
        "_out_a",
        terminal_file="src/modules/a/a.cpp",
        source_structure=structure,
    )

    gain_files = {v.file for v in dag.vertices
                  if v.kind == "operation" and v.variable == "_gain"}
    assert gain_files == {"src/modules/a/a.hpp"}


def test_trailing_underscore_members_resolve_across_class_files():
    """PX4 library classes mark members with a TRAILING underscore
    (``airspeed_ref_``); a consumer in the class header must reach
    writers in the class cpp — member family rule, not local rule."""
    bindings = [
        _fake_binding(binding_id="t", target="_out", expression="getRef()",
                      file="src/lib/npfg/npfg.hpp", line=1, function="Npfg::getRef"),
        _fake_binding(binding_id="r", target="ref_out", expression="airspeed_ref_",
                      file="src/lib/npfg/npfg.hpp", line=2, function="Npfg::getRef",
                      logged_signal=""),
        _fake_binding(binding_id="w", target="airspeed_ref_", expression="wind_speed + margin",
                      file="src/lib/npfg/npfg.cpp", line=3, function="Npfg::guide",
                      logged_signal=""),
    ]
    bindings, structure = _owned_bindings(
        bindings,
        bases={"Npfg": set()},
        members={("Npfg", "airspeed_ref_"): {"type": "float"}},
    )
    dag = build_mechanism_dag(
        bindings,
        "ref_out",
        terminal_file="src/lib/npfg/npfg.hpp",
        source_structure=structure,
    )

    ops = {v.variable for v in dag.vertices if v.kind == "operation"}
    assert "airspeed_ref_" in ops


def test_internal_state_branch_grounds_through_graph_and_gets_windows():
    """A branch predicate over internal state (no direct logged/param
    reference) grounds through the graph — the state's producer chain
    ends at a logged signal — and evaluates to active windows."""
    bindings = [
        _fake_binding(binding_id="t", target="_out", expression="val + 1.0",
                      file="a.cpp", line=1, function="A::run",
                      control_predicates=["_flare_states.flaring"]),
        _fake_binding(binding_id="s", target="_flare_states.flaring",
                      expression="vehicle_land_detected.flaring_flag",
                      file="a.cpp", line=2, function="A::poll",
                      logged_signal=""),
    ]
    bindings, structure = _owned_bindings(
        bindings,
        bases={"A": set()},
        members={("A", "_flare_states"): {"type": "FlareStates"}},
    )
    dag = build_mechanism_dag(
        bindings, "_out",
        logged_signals={"vehicle_land_detected.flaring_flag"},
        source_structure=structure,
    )
    annotated = evaluate_feasibility(
        dag,
        signal_samples={
            "vehicle_land_detected.flaring_flag": [
                (0.0, 0), (10.0, 1), (20.0, 0), (30.0, 0),
            ]
        },
        prune_dead=False,
    )
    branch = next(v for v in annotated.vertices if v.kind == "branch")
    assert branch.active_windows, "grounded internal-state predicate produced no windows"
    assert branch.active_windows[0][0] == 10.0


def test_grounding_substitutes_constant_values():
    """Grounding a predicate whose producer is a value-carrying constant
    leaf exercises the literal-rendering path (live RTL crashed on it)."""
    bindings = [
        _fake_binding(binding_id="t", target="_out", expression="val + 1.0",
                      file="a.cpp", line=1, function="A::run",
                      control_predicates=["_mode_state == MODE_ON"]),
        _fake_binding(binding_id="m", target="_mode_state",
                      expression="vehicle_status.nav_mode",
                      file="a.cpp", line=2, function="A::poll",
                      logged_signal=""),
        _fake_binding(binding_id="c", target="MODE_ON", expression="3",
                      file="a.cpp", line=3, function="",
                      logged_signal="", declaration_kind="enum"),
    ]
    bindings, structure = _owned_bindings(
        bindings,
        bases={"A": set()},
        members={("A", "_mode_state"): {"type": "int"}},
    )
    dag = build_mechanism_dag(
        bindings,
        "_out",
        logged_signals={"vehicle_status.nav_mode"},
        source_structure=structure,
    )
    annotated = evaluate_feasibility(
        dag,
        signal_samples={"vehicle_status.nav_mode": [
            (0.0, 0), (10.0, 3), (20.0, 0), (30.0, 0)]},
        prune_dead=False,
    )
    branch = next(v for v in annotated.vertices if v.kind == "branch")
    assert branch.active_windows
    assert branch.active_windows[0][0] == 10.0


def test_logged_inputs_are_leaves_not_publisher_tunnels():
    """A logged input the mechanism reads must terminate as evidence —
    its PUBLISHER (another module across the uORB boundary) must not be
    walked. Only the terminal enters source through its publishers."""
    bindings = [
        # terminal enters source via its publisher (logged_signal set)
        # mechanism reads a logged input topic field
        _fake_binding(binding_id="g", target="gate_state",
                      expression="vehicle_status.vehicle_type",
                      file="fw.cpp", line=1, function="Fw::run",
                      logged_signal=""),
        _fake_binding(binding_id="t", target="out_field",
                      expression="gate_state + 1.0",
                      file="fw.cpp", line=2, function="Fw::run",
                      logged_signal="fw_status.out_field"),
        # the FOREIGN publisher of that input topic — must stay out
        _fake_binding(binding_id="pub", target="vehicle_status.vehicle_type",
                      expression="commander_internal_state",
                      file="commander.cpp", line=3, function="Commander::run",
                      logged_signal="vehicle_status.vehicle_type"),
    ]
    dag = build_mechanism_dag(
        bindings, "fw_status.out_field",
        logged_signals={"vehicle_status.vehicle_type", "fw_status.out_field"},
    )

    files = {v.file for v in dag.vertices if v.kind == "operation" and v.file}
    assert "commander.cpp" not in files, "publisher tunneled through a logged leaf"
    leaves = {v.signal_name for v in dag.vertices
              if v.kind == "evidence" and v.sub_kind == "logged_signal"}
    assert "vehicle_status.vehicle_type" in leaves
    assert not any("commander_internal_state" in str(v.signal_name or "")
                   for v in dag.vertices)


def test_helper_pick_prefers_caller_module_and_skips_foreign_ambiguity():
    """A bare call resolves to the helper in the CALLER's module when
    several classes define the name; all-foreign ambiguity stays opaque
    (sorted-first used to materialize another module's subgraph)."""
    ours = _fake_helper(
        name="Mission::set_index", file="src/modules/navigator/mission.cpp",
        line=5, evidence="int Mission::set_index(int i)",
        assignments={}, return_expression="nav_internal * 2",
    )
    foreign = _fake_helper(
        name="MavlinkMissionManager::set_index",
        file="src/modules/mavlink/mavlink_mission.cpp",
        line=9, evidence="int MavlinkMissionManager::set_index(int i)",
        assignments={}, return_expression="mav_internal * 3",
    )
    bindings = [
        _fake_binding(binding_id="t", target="_out", expression="set_index(2)",
                      file="src/modules/navigator/rtl.cpp", line=1,
                      function="RTL::run"),
    ]
    structure = SourceStructureIndex(
        direct_bases={
            "RTL": {"Mission"},
            "Mission": set(),
            "MavlinkMissionManager": set(),
        }
    )
    dag = build_mechanism_dag(
        structure.enrich_bindings(bindings),
        "_out",
        helper_expressions=[foreign, ours],
        source_structure=structure,
    )

    bodies = {v.file for v in dag.vertices
              if v.provenance and v.provenance.startswith("helper_return")}
    assert bodies == {"src/modules/navigator/mission.cpp"}

    # all-foreign ambiguity: caller in a third module → opaque, no subgraph
    bindings2 = [
        _fake_binding(binding_id="t", target="_out", expression="set_index(2)",
                      file="src/modules/commander/Commander.cpp", line=1,
                      function="Commander::run"),
    ]
    dag2 = build_mechanism_dag(bindings2, "_out",
                               helper_expressions=[foreign, ours])
    assert not [v for v in dag2.vertices
                if v.provenance and v.provenance.startswith("helper_return")]


def test_helper_pick_qualifies_nested_receiver_type_from_lexical_owner():
    helper = _fake_helper(
        name="RTL::RTLPosition::set",
        file="src/modules/navigator/rtl.h",
        line=20,
        evidence="void RTL::RTLPosition::set(float value)",
        assignments={"alt": "value"},
        return_expression="",
    )
    helper["parameters"] = ["value"]
    binding = _fake_binding(
        binding_id="terminal",
        target="_out",
        expression="_destination.set(input_value)",
        file="src/modules/navigator/rtl.cpp",
        line=10,
        function="RTL::run",
    )
    structure = SourceStructureIndex(
        direct_bases={"RTL": set(), "RTL::RTLPosition": set()},
        declared_classes={"RTL", "RTL::RTLPosition"},
        members={
            ("RTL", "_destination"): {
                "name": "_destination",
                "owner": "RTL",
                "type": "RTLPosition",
            }
        },
    )

    dag = build_mechanism_dag(
        structure.enrich_bindings([binding]),
        "_out",
        helper_expressions=[helper],
        source_structure=structure,
    )

    assert structure.member_receiver_type("RTL", "_destination") == "RTL::RTLPosition"
    assert any(
        vertex.kind == "operation"
        and vertex.variable == "alt"
        and vertex.file == "src/modules/navigator/rtl.h"
        for vertex in dag.vertices
    )
    assert not any(
        reference.kind == "callable" and reference.symbol == "set"
        for reference in dag.unresolved_references
    )


def test_schema_enum_resolves_scoped_constant_and_evaluates():
    """``struct_s::NAME`` resolves from the schema enum registry SCOPED
    by its own message — value-carrying leaf, evaluable branch."""
    bindings = [
        _fake_binding(binding_id="t", target="_out", expression="val + 1.0",
                      file="a.cpp", line=1, function="A::run",
                      control_predicates=[
                          "_type_state == position_setpoint_s::SETPOINT_TYPE_LAND"]),
        _fake_binding(binding_id="m", target="_type_state",
                      expression="position_setpoint.type",
                      file="a.cpp", line=2, function="A::poll",
                      logged_signal=""),
    ]
    bindings, structure = _owned_bindings(
        bindings,
        bases={"A": set()},
        members={("A", "_type_state"): {"type": "int"}},
    )
    dag = build_mechanism_dag(
        bindings, "_out",
        logged_signals={"position_setpoint.type"},
        enum_registry={"position_setpoint": {"SETPOINT_TYPE_LAND": 3}},
        source_structure=structure,
    )
    leaf = next(v for v in dag.vertices if v.kind == "evidence"
                and v.sub_kind == "constant"
                and "SETPOINT_TYPE_LAND" in str(v.signal_name))
    assert leaf.metadata.get("value") == 3

    annotated = evaluate_feasibility(
        dag,
        signal_samples={"position_setpoint.type": [
            (0.0, 0), (10.0, 3), (20.0, 0), (30.0, 0)]},
        signal_policies={
            "position_setpoint.type": {"method": "discrete_hold"}
        },
        prune_dead=False,
    )
    branch = next(v for v in annotated.vertices if v.kind == "branch")
    assert branch.active_windows
    assert branch.active_windows[0][0] == 10.0


def test_px4_macro_predicates_evaluate():
    """PX4 macro spellings (PX4_ISFINITE) and float suffixes must not
    poison predicate evaluation — the interval evaluator routes through
    the centralized expression lowering."""
    bindings = [
        _fake_binding(binding_id="t", target="_out", expression="v + 1.0",
                      file="a.cpp", line=1, function="A::run",
                      control_predicates=[
                          "PX4_ISFINITE(wind_speed.value) && wind_speed.value > 0.5f"]),
    ]
    dag = build_mechanism_dag(bindings, "_out",
                              logged_signals={"wind_speed.value"})
    annotated = evaluate_feasibility(
        dag,
        signal_samples={"wind_speed.value": [
            (0.0, 0.0), (10.0, 2.0), (20.0, 0.1)]},
        signal_policies={"wind_speed.value": {"method": "linear"}},
        prune_dead=False,
    )
    branch = next(v for v in annotated.vertices if v.kind == "branch")
    assert branch.active_windows == [(10.0, 20.0)]


def test_dotted_formal_rebinds_to_actual_nested_placement():
    """``formal.field`` follows the formal's unique simple rebinding to
    the actual's logged placement instead of degrading to the nested
    message name (position_setpoint.type vs the logged
    position_setpoint_triplet.current.type)."""
    bindings = [
        _fake_binding(binding_id="rebind", target="pos_sp_curr",
                      expression="_pos_sp_triplet.current",
                      file="fw.cpp", line=1, function="Fw::run",
                      logged_signal=""),
        _fake_binding(binding_id="t", target="_out",
                      expression="pos_sp_curr.type + 1",
                      file="fw.cpp", line=2, function="Fw::run"),
    ]
    bindings[1]["struct_variables"] = {
        "_pos_sp_triplet": "position_setpoint_triplet_s"
    }
    dag = build_mechanism_dag(
        bindings, "_out",
        logged_signals={"position_setpoint_triplet.current.type"},
        boundary_bindings=[
            _boundary("_pos_sp_triplet", "position_setpoint_triplet", file="fw.cpp")
        ],
    )
    leaves = {v.signal_name for v in dag.vertices
              if v.kind == "evidence" and v.sub_kind == "logged_signal"}
    assert "position_setpoint_triplet.current.type" in leaves


def test_rebinding_survives_passthrough_and_cycles():
    """Pass-through forwarding (formal bound to a same-named actual) is
    ignored as identity, and alias cycles must not recurse forever."""
    bindings = [
        # identity pass-through from a forwarding call
        _fake_binding(binding_id="fwd", target="pos_sp_curr",
                      expression="pos_sp_curr", file="fw.cpp", line=1,
                      function="Fw::inner",
                      logged_signal=""),
        # the real rebinding
        _fake_binding(binding_id="rebind", target="pos_sp_curr",
                      expression="_trip.current",
                      file="fw.cpp", line=2, function="Fw::run",
                      logged_signal=""),
        _fake_binding(binding_id="t", target="_out",
                      expression="pos_sp_curr.type + 1",
                      file="fw.cpp", line=3, function="Fw::run"),
        # alias cycle: a <-> b
        _fake_binding(binding_id="c1", target="alias_a",
                      expression="alias_b", file="fw.cpp", line=4,
                      function="Fw::run", logged_signal=""),
        _fake_binding(binding_id="c2", target="alias_b",
                      expression="alias_a", file="fw.cpp", line=5,
                      function="Fw::run", logged_signal=""),
        _fake_binding(binding_id="t2", target="_out2",
                      expression="alias_a.field + 1",
                      file="fw.cpp", line=6, function="Fw::run"),
    ]
    bindings[2]["struct_variables"] = {"_trip": "position_setpoint_triplet_s"}
    dag = build_mechanism_dag(
        bindings, "_out",
        logged_signals={"position_setpoint_triplet.current.type"},
        boundary_bindings=[
            _boundary("_trip", "position_setpoint_triplet", file="fw.cpp")
        ],
    )
    leaves = {v.signal_name for v in dag.vertices
              if v.kind == "evidence" and v.sub_kind == "logged_signal"}
    assert "position_setpoint_triplet.current.type" in leaves

    dag2 = build_mechanism_dag(bindings, "_out2")  # must terminate
    assert dag2.vertices


def test_array_indices_are_distinct_terminal_identities():
    """``q[0]`` and ``q[1]`` are different values: slicing from one index
    takes only that element's writer; an index-free aggregate reference
    must not collect element writers by fuzzy shape."""
    bindings = [
        _fake_binding(binding_id="e0", target="q[0]",
                      expression="a0 * 2", file="att.cpp", line=5,
                      logged_signal=""),
        _fake_binding(binding_id="e1", target="q[1]",
                      expression="a1 * 3", file="att.cpp", line=6,
                      logged_signal=""),
    ]

    element = build_mechanism_dag(bindings, "q[0]", terminal_file="att.cpp")
    ops = {v.variable for v in element.vertices if v.kind == "operation"}
    assert "q[0]" in ops
    assert "q[1]" not in ops, "sibling index fused into the slice"

    whole = build_mechanism_dag(
        [
            *bindings,
            _fake_binding(binding_id="use", target="out",
                          expression="q * 2", file="att.cpp", line=9,
                          logged_signal=""),
        ],
        "out",
        terminal_file="att.cpp",
    )
    ops = {v.variable for v in whole.vertices if v.kind == "operation"}
    assert "q[0]" not in ops and "q[1]" not in ops
    assert "q" in whole.unresolved_symbols


def test_member_copy_grounds_only_through_source_boundary():
    """Member spelling alone does not prove a topic placement."""
    bindings = [
        _fake_binding(binding_id="b", target="out",
                      expression="_vehicle_status.nav_state + 1",
                      file="mod.cpp", line=4, logged_signal=""),
    ]
    dag = build_mechanism_dag(
        bindings, "out", terminal_file="mod.cpp",
        logged_signals={"vehicle_status.nav_state"},
        boundary_bindings=[
            _boundary("_vehicle_status", "vehicle_status", file="mod.cpp")
        ],
    )
    leaves = {
        v.signal_name for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "logged_signal"
    }
    assert "vehicle_status.nav_state" in leaves
    assert dag.unresolved_symbols == []


def test_indexed_catalogue_entry_does_not_ground_aggregate_reference():
    """An observed array element is not evidence for the aggregate."""
    bindings = [
        _fake_binding(binding_id="b", target="out",
                      expression="state.q + 1", file="mod.cpp", line=4,
                      logged_signal=""),
    ]
    dag = build_mechanism_dag(
        bindings, "out", terminal_file="mod.cpp",
        logged_signals={"state.q[0]"},
    )
    leaves = {
        v.signal_name for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "logged_signal"
    }
    assert "state.q" not in leaves
    assert dag.unresolved_symbols == ["state.q"]


def test_branch_identity_is_the_source_site():
    """Identical predicate text at two source sites is TWO branches;
    the same site shared by several gated operations is ONE branch."""
    bindings = [
        _fake_binding(binding_id="a", target="out_a",
                      expression="in_a + 1", file="mod.cpp", line=12,
                      logged_signal="",
                      control_predicates=["mode == 2"]),
        _fake_binding(binding_id="b", target="out_b",
                      expression="in_b + 2", file="mod.cpp", line=13,
                      logged_signal="",
                      control_predicates=["mode == 2"]),
        _fake_binding(binding_id="c", target="out_c",
                      expression="out_a + out_b", file="mod.cpp", line=40,
                      logged_signal="",
                      control_predicates=["mode == 2"]),
    ]
    # a and b share one if (site 10); c sits under a DIFFERENT if with
    # the same text (site 38).
    bindings[0]["control_predicate_lines"] = [10]
    bindings[1]["control_predicate_lines"] = [10]
    bindings[2]["control_predicate_lines"] = [38]

    dag = build_mechanism_dag(bindings, "out_c", terminal_file="mod.cpp")
    branches = [v for v in dag.vertices if v.kind == "branch"]
    assert len(branches) == 2, "same-text branches at distinct sites must not fuse"
    assert {v.line for v in branches} == {10, 38}


def test_logged_signal_leaves_carry_observation_status():
    """Declared and observed are independent: a schema-derived placement
    is a source-proven boundary, but only presence in THIS flight's log
    marks the leaf observed."""
    binding = {
        "target_symbol": "out",
        "source_symbol": "data.field_x + 1",
        "function": "Gate::update",
        "assignment_path": [
            {"file": "mod.cpp", "line": 4, "expression": "data.field_x + 1"}
        ],
        "logged_signal": "",
        "control_predicates": [],
        "struct_variables": {"data": "gate_status_s"},
    }

    declared_only = build_mechanism_dag(
        [dict(binding)], "out", terminal_file="mod.cpp",
        schema_signals={"gate_status.field_x"},
        boundary_bindings=[_boundary("data", "gate_status", file="mod.cpp")],
    )
    leaf = next(
        v for v in declared_only.vertices
        if v.kind == "evidence" and v.sub_kind == "logged_signal"
        and v.signal_name == "gate_status.field_x"
    )
    assert leaf.metadata.get("observation") == "unobserved"
    assert leaf.metadata.get("declaration_status") == "valid"
    assert leaf.metadata.get("boundary_status") == "source_proven"

    observed = build_mechanism_dag(
        [dict(binding)], "out", terminal_file="mod.cpp",
        schema_signals={"gate_status.field_x"},
        logged_signals={"gate_status.field_x"},
        boundary_bindings=[_boundary("data", "gate_status", file="mod.cpp")],
    )
    leaf = next(
        v for v in observed.vertices
        if v.kind == "evidence" and v.sub_kind == "logged_signal"
        and v.signal_name == "gate_status.field_x"
    )
    assert leaf.metadata.get("observation") == "observed"


def test_inventory_observation_preserves_multi_instance_identity():
    inventory = {
        "topic_fields": {"sensor": ["value"]},
        "topic_instances": {
            "sensor": [
                {"multi_id": 0, "fields": ["timestamp", "value"]},
                {"multi_id": 1, "fields": ["timestamp", "value"]},
            ]
        },
    }
    dag = build_mechanism_dag(
        [_fake_binding(
            binding_id="b", target="out", expression="sensor[0].value",
            file="mod.cpp", line=1,
        )],
        "out",
        inventory=inventory,
    )
    leaves = {
        (v.signal_name, v.metadata.get("observation"))
        for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "logged_signal"
    }
    assert ("sensor[0].value", "observed") in leaves
    assert ("sensor[1].value", "observed") not in leaves


def test_observed_inventory_preserves_a_lone_nonzero_instance():
    observed = observed_signals_from_inventory(
        {
            "topic_instances": {
                "sensor": [
                    {"multi_id": 3, "fields": ["timestamp", "value"]}
                ]
            }
        }
    )
    assert observed == {"sensor[3].value"}


def test_unqualified_signal_resolves_only_one_observed_instance():
    binding = _fake_binding(
        binding_id="b",
        target="out",
        expression="sensor.value",
        file="mod.cpp",
        line=1,
        logged_signal="",
    )
    unique = build_mechanism_dag(
        [binding], "out", logged_signals={"sensor[3].value"}
    )
    assert any(
        vertex.kind == "evidence"
        and vertex.signal_name == "sensor[3].value"
        and vertex.metadata.get("observation") == "observed"
        for vertex in unique.vertices
    )

    ambiguous = build_mechanism_dag(
        [binding],
        "out",
        logged_signals={"sensor[0].value", "sensor[3].value"},
    )
    assert "sensor.value" in ambiguous.unresolved_symbols


def test_explicit_sibling_index_is_not_observed_by_shape():
    dag = build_mechanism_dag(
        [_fake_binding(
            binding_id="b", target="out", expression="state.q[0]",
            file="mod.cpp", line=1,
        )],
        "out",
        logged_signals={"state.q[1]"},
    )
    assert not any(
        v.kind == "evidence" and v.sub_kind == "logged_signal"
        for v in dag.vertices
    )
    assert "state.q[0]" in dag.unresolved_symbols


def test_operation_carries_composed_reachability():
    """Invariant: every operation carries its complete reachability —
    the conjunction of governing predicates plus an explicit exactness
    flag when an unmodeled construct made the formula incomplete."""
    exact = _fake_binding(binding_id="a", target="out_a",
                          expression="in_a + 1", file="mod.cpp", line=12,
                          logged_signal="",
                          control_predicates=["a > 0", "b > 0"])
    inexact = {**_fake_binding(binding_id="b", target="out_b",
                               expression="out_a + 2", file="mod.cpp", line=30,
                               logged_signal=""),
               "reachability_exact": False}

    dag = build_mechanism_dag([exact, inexact], "out_b", terminal_file="mod.cpp")
    ops = {v.variable: v for v in dag.vertices if v.kind == "operation"}

    assert ops["out_a"].metadata["reachability"] == {
        "all_of": ["a > 0", "b > 0"], "exact": True,
    }
    assert ops["out_b"].metadata["reachability"] == {"all_of": [], "exact": False}


def test_wiring_respects_callable_scope_for_locals():
    """Two functions' same-named locals are different variables in
    WIRING too: the consumer links to its own function's producer, not
    the file's last one."""
    bindings = [
        _fake_binding(binding_id="own", target="gain",
                      expression="f_in * 1", file="a.cpp", line=5,
                      logged_signal="", function="C::f"),
        _fake_binding(binding_id="foreign", target="gain",
                      expression="g_in * 2", file="a.cpp", line=20,
                      logged_signal="", function="C::g"),
        _fake_binding(binding_id="consumer", target="out",
                      expression="gain + 1", file="a.cpp", line=7,
                      logged_signal="", function="C::f"),
    ]
    dag = build_mechanism_dag(bindings, "out", terminal_file="a.cpp")

    consumer = next(v for v in dag.vertices if v.variable == "out")
    linked = {
        e.source_id for e in dag.edges
        if e.target_id == consumer.id and e.kind == "data"
    }
    producers = {v.id: v for v in dag.vertices if v.variable == "gain"}
    linked_lines = {producers[p].line for p in linked if p in producers}
    assert linked_lines == {5}, "consumer wired to another function's local"


def test_failed_rebinding_falls_back_to_original_resolution():
    """A unique simple rebinder whose rewritten form resolves to nothing
    must not hijack resolution: the ORIGINAL dotted symbol continues its
    own sequence (here: struct-variable derivation to a schema-proven
    placement)."""
    bindings = [
        {
            "target_symbol": "out",
            "source_symbol": "data.alt + 1",
            "function": "C::run",
            "assignment_path": [
                {"file": "mod.cpp", "line": 9, "expression": "data.alt + 1"}
            ],
            "logged_signal": "",
            "control_predicates": [],
            "struct_variables": {"data": "gate_status_s"},
        },
        # unique simple writer of the root: rebinding rewrites
        # data.alt -> other_thing.alt, which resolves to NOTHING
        {
            "target_symbol": "data",
            "source_symbol": "other_thing",
            "function": "C::run",
            "assignment_path": [
                {"file": "mod.cpp", "line": 5, "expression": "other_thing"}
            ],
            "logged_signal": "",
            "control_predicates": [],
            "struct_variables": {},
        },
    ]
    dag = build_mechanism_dag(
        bindings, "out", terminal_file="mod.cpp",
        schema_signals={"gate_status.alt"},
        boundary_bindings=[_boundary("data", "gate_status", file="mod.cpp")],
    )
    leaves = {
        v.signal_name for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "logged_signal"
    }
    assert "gate_status.alt" in leaves, "failed rewrite hijacked resolution"
    assert "other_thing.alt" not in dag.unresolved_symbols


def test_aggregate_writer_can_reach_indexed_read_directionally():
    bindings = [
        _fake_binding(
            binding_id="aggregate", target="state.q", expression="source_q",
            file="att.cpp", line=1, logged_signal="", function="Att::run",
        ),
        _fake_binding(
            binding_id="consumer", target="out", expression="state.q[0]",
            file="att.cpp", line=2, logged_signal="", function="Att::run",
        ),
    ]
    dag = build_mechanism_dag(bindings, "out", terminal_file="att.cpp")
    aggregate = next(
        vertex for vertex in dag.vertices
        if vertex.kind == "operation" and vertex.variable == "state.q"
    )
    consumer = next(
        vertex for vertex in dag.vertices
        if vertex.kind == "operation" and vertex.variable == "out"
    )
    assert any(
        edge.source_id == aggregate.id and edge.target_id == consumer.id
        for edge in dag.edges
    )


def test_future_local_write_does_not_feed_earlier_read():
    bindings = [
        _fake_binding(
            binding_id="consumer", target="out", expression="value + 1",
            file="scope.cpp", line=10, logged_signal="", function="A::run",
        ),
        _fake_binding(
            binding_id="future", target="value", expression="99",
            file="scope.cpp", line=20, logged_signal="", function="A::run",
        ),
    ]
    dag = build_mechanism_dag(bindings, "out", terminal_file="scope.cpp")
    consumer = next(vertex for vertex in dag.vertices if vertex.variable == "out")
    assert not [vertex for vertex in dag.vertices if vertex.variable == "value"]
    incoming = [edge for edge in dag.edges if edge.target_id == consumer.id]
    assert incoming
    source = next(vertex for vertex in dag.vertices if vertex.id == incoming[0].source_id)
    assert source.kind == "evidence" and source.sub_kind == "opaque_symbol"


def test_conditional_reaching_writers_are_all_retained():
    bindings = [
        _fake_binding(
            binding_id="base", target="value", expression="0",
            file="scope.cpp", line=1, logged_signal="", function="A::run",
        ),
        _fake_binding(
            binding_id="a", target="value", expression="1",
            file="scope.cpp", line=2, logged_signal="", function="A::run",
            control_predicates=["mode == 1"],
        ),
        _fake_binding(
            binding_id="b", target="value", expression="2",
            file="scope.cpp", line=3, logged_signal="", function="A::run",
            control_predicates=["mode == 2"],
        ),
        _fake_binding(
            binding_id="consumer", target="out", expression="value",
            file="scope.cpp", line=4, logged_signal="", function="A::run",
        ),
    ]
    dag = build_mechanism_dag(bindings, "out", terminal_file="scope.cpp")
    consumer = next(vertex for vertex in dag.vertices if vertex.variable == "out")
    producers = {
        vertex.id
        for vertex in dag.vertices
        if vertex.kind == "operation" and vertex.variable == "value"
    }
    wired = {
        edge.source_id
        for edge in dag.edges
        if edge.kind == "data" and edge.target_id == consumer.id
    }
    assert producers <= wired


def test_false_conjunct_prunes_operation_even_when_other_gate_is_true():
    binding = _fake_binding(
        binding_id="gated", target="out", expression="1",
        file="gate.cpp", line=3, control_predicates=["A", "B"],
    )
    dag = build_mechanism_dag([binding], "out")
    reduced = evaluate_feasibility(
        dag, enum_values={"A": 0, "B": 1}, prune_dead=True,
    )
    assert not [vertex for vertex in reduced.vertices if vertex.kind == "operation"]


def test_unrelated_signal_span_does_not_change_branch_classification():
    binding = _fake_binding(
        binding_id="gated", target="out", expression="1",
        file="gate.cpp", line=3, control_predicates=["mode == 1"],
    )
    dag = build_mechanism_dag([binding], "out")
    reduced = evaluate_feasibility(
        dag,
        signal_samples={
            "mode": [(10.0, 1), (20.0, 1)],
            "unrelated": [(0.0, 0), (100.0, 0)],
        },
        signal_policies={
            "mode": {"method": "discrete_hold"},
            "unrelated": {"method": "discrete_hold"},
        },
        prune_dead=False,
    )
    branch = next(vertex for vertex in reduced.vertices if vertex.kind == "branch")
    assert branch.feasibility_verdict == "always_true"
    assert branch.metadata["evaluation_domain"] == [10.0, 20.0]


def test_missing_signal_policy_keeps_dynamic_verdict_unknown():
    binding = _fake_binding(
        binding_id="gated", target="out", expression="1",
        file="gate.cpp", line=3, control_predicates=["value > 0"],
    )
    dag = build_mechanism_dag([binding], "out")
    reduced = evaluate_feasibility(
        dag,
        signal_samples={"value": [(0.0, 1.0), (10.0, 1.0)]},
        prune_dead=False,
    )
    branch = next(vertex for vertex in reduced.vertices if vertex.kind == "branch")
    assert branch.active_windows == [(0.0, 10.0)]
    assert branch.feasibility_verdict == "unknown"
    assert branch.metadata["sampling_policies"] == {"value": "unknown"}


def _shared_value_program_dag() -> MechanismDAG:
    vertices = [
        DAGVertex(
            id="signal",
            kind="evidence",
            sub_kind="logged_signal",
            signal_name="sensor.value",
        ),
        DAGVertex(
            id="enabled",
            kind="evidence",
            sub_kind="constant",
            signal_name="ENABLED",
            metadata={"value": True},
        ),
        DAGVertex(
            id="gate",
            kind="branch",
            predicate_raw="ENABLED",
            feasibility_verdict="unknown",
        ),
        DAGVertex(
            id="shared",
            kind="operation",
            variable="shared",
            expression="sensor.value + 1",
            file="value.cpp",
            line=2,
            metadata={"target_scope": {"file": "value.cpp", "callable": "run"}},
        ),
        DAGVertex(
            id="positive",
            kind="branch",
            predicate_raw="shared > 0",
            feasibility_verdict="unknown",
        ),
        DAGVertex(
            id="bounded",
            kind="branch",
            predicate_raw="shared < 10",
            feasibility_verdict="unknown",
        ),
    ]
    edges = [
        DAGEdge(
            id="enabled-gate", source_id="enabled", target_id="gate",
            kind="data", role="ENABLED",
        ),
        DAGEdge(
            id="gate-shared", source_id="gate", target_id="shared",
            kind="control",
        ),
        DAGEdge(
            id="signal-shared", source_id="signal", target_id="shared",
            kind="data", role="sensor.value",
        ),
        DAGEdge(
            id="shared-positive", source_id="shared", target_id="positive",
            kind="data", role="shared",
        ),
        DAGEdge(
            id="shared-bounded", source_id="shared", target_id="bounded",
            kind="data", role="shared",
        ),
    ]
    return MechanismDAG(
        dag_id="shared-program",
        terminal="shared",
        vertices=vertices,
        edges=edges,
    )


def test_dag_value_program_compiles_each_expression_once(monkeypatch):
    compile_calls = []
    original = dag_value_module.compile_source_expression

    def counted_compile(expression, operand_names):
        compile_calls.append(expression)
        return original(expression, operand_names)

    monkeypatch.setattr(
        dag_value_module, "compile_source_expression", counted_compile
    )
    program = DAGValueProgram(_shared_value_program_dag())
    session = program.bind(sample_resolver=lambda _signal, _timestamp: 2.0)
    session.evaluate_many(("positive", "bounded"), 0.0)
    session.evaluate_many(("positive", "bounded"), 1.0)

    assert len(compile_calls) == 4


def test_dag_value_session_normalizes_parameters_once():
    class CountingParameters(dict):
        calls = 0

        def items(self):
            self.calls += 1
            return super().items()

    dag = build_mechanism_dag(
        [
            _fake_binding(
                binding_id="parameter-gate",
                target="out",
                expression="1",
                file="parameter.cpp",
                line=2,
                control_predicates=["_param_limit.get() > 0"],
            )
        ],
        "out",
    )
    branch = next(vertex for vertex in dag.vertices if vertex.kind == "branch")
    parameters = CountingParameters({"LIMIT": 2})
    session = DAGValueProgram(dag).bind(parameter_values=parameters)

    assert session.evaluate(branch.id, None).value is True
    assert session.evaluate(branch.id, None).value is True
    assert parameters.calls == 1


def test_dag_value_session_shares_vertex_activity_and_sample_results(monkeypatch):
    expression_calls = []
    activity_calls = []
    sample_calls = []
    original_expression = DAGValueSession._evaluate_expression_vertex
    original_activity = DAGValueSession._operation_activity

    def counted_expression(self, vertex, timestamp, active):
        expression_calls.append((vertex.id, timestamp))
        return original_expression(self, vertex, timestamp, active)

    def counted_activity(self, vertex_id, timestamp, active):
        activity_calls.append((vertex_id, timestamp))
        return original_activity(self, vertex_id, timestamp, active)

    monkeypatch.setattr(
        DAGValueSession, "_evaluate_expression_vertex", counted_expression
    )
    monkeypatch.setattr(DAGValueSession, "_operation_activity", counted_activity)
    program = DAGValueProgram(_shared_value_program_dag())
    session = program.bind(
        sample_resolver=lambda signal, timestamp: (
            sample_calls.append((signal, timestamp)) or 2.0
        )
    )

    results = session.evaluate_many(("positive", "bounded"), 4.0)

    assert all(result.status == "value" for result in results.values())
    assert max(expression_calls.count(item) for item in set(expression_calls)) == 1
    assert activity_calls.count(("shared", 4.0)) == 1
    assert sample_calls == [("sensor.value", 4.0)]


def test_feasibility_rejects_a_session_from_another_program():
    dag = _shared_value_program_dag()
    program = DAGValueProgram(dag)
    other_program = DAGValueProgram(dag)

    with pytest.raises(
        ValueError, match="value_session must be bound to value_program"
    ):
        evaluate_feasibility(
            dag,
            value_program=program,
            value_session=other_program.bind(),
        )


def test_dag_value_program_preserves_call_result_operand_identity():
    dag = MechanismDAG(
        dag_id="call-result",
        terminal="output",
        vertices=[
            DAGVertex(
                id="value", kind="evidence", sub_kind="constant",
                signal_name="reader result", metadata={"value": 4.0},
            ),
            DAGVertex(
                id="output", kind="operation", variable="output",
                expression="reader().value + 1",
                metadata={
                    "source_call_roles": {
                        "call-site": {
                            "call": "reader()",
                            "result": "reader().value",
                        }
                    }
                },
            ),
        ],
        edges=[
            DAGEdge(
                id="call-edge", source_id="value", target_id="output",
                kind="data", role="call-result:value", via="call-site",
            )
        ],
    )

    result = DAGValueProgram(dag).bind().evaluate("output", None)

    assert result.status == "value"
    assert result.value == 5.0


def test_implicit_enum_chain_is_available_to_static_feasibility():
    bindings = [
        _fake_binding(
            binding_id="enum-a", target="MODE_A", expression="0",
            file="mode.h", line=1, logged_signal="", declaration_kind="enum",
        ),
        _fake_binding(
            binding_id="enum-b", target="MODE_B", expression="MODE_A + 1",
            file="mode.h", line=2, logged_signal="", declaration_kind="enum",
        ),
        _fake_binding(
            binding_id="enum-c", target="MODE_C", expression="MODE_B + 1",
            file="mode.h", line=3, logged_signal="", declaration_kind="enum",
        ),
        _fake_binding(
            binding_id="terminal", target="out", expression="1",
            file="mode.cpp", line=4, control_predicates=["mode == MODE_C"],
        ),
    ]
    dag = build_mechanism_dag(bindings, "out")

    annotated = evaluate_feasibility(
        dag, enum_values={"mode": 2}, prune_dead=False
    )

    branch = next(vertex for vertex in annotated.vertices if vertex.kind == "branch")
    assert branch.feasibility_verdict == "always_true"


def test_static_short_circuit_does_not_request_dynamic_operand():
    dag = MechanismDAG(
        dag_id="short-circuit",
        terminal="gate",
        vertices=[
            DAGVertex(
                id="disabled", kind="evidence", sub_kind="constant",
                signal_name="DISABLED", metadata={"value": False},
            ),
            DAGVertex(
                id="signal", kind="evidence", sub_kind="logged_signal",
                signal_name="sensor.value",
            ),
            DAGVertex(
                id="gate", kind="branch",
                predicate_raw="DISABLED && sensor.value > 0",
                feasibility_verdict="unknown",
            ),
        ],
        edges=[
            DAGEdge(
                id="disabled-edge", source_id="disabled", target_id="gate",
                kind="data", role="DISABLED",
            ),
            DAGEdge(
                id="signal-edge", source_id="signal", target_id="gate",
                kind="data", role="sensor.value",
            ),
        ],
    )
    sample_calls = []
    session = DAGValueProgram(dag).bind(
        sample_resolver=lambda signal, timestamp: sample_calls.append(
            (signal, timestamp)
        )
    )

    result = session.evaluate("gate", None)

    assert result.status == "value"
    assert result.value is False
    assert sample_calls == []


def test_static_unevaluable_branch_schedules_no_timestamps(monkeypatch):
    dag = build_mechanism_dag(
        [
            _fake_binding(
                binding_id="unsupported-gate",
                target="out",
                expression="1",
                file="static.cpp",
                line=2,
                control_predicates=["unknown_value > 0"],
            )
        ],
        "out",
    )
    timestamps = []
    original = DAGValueSession.evaluate_many

    def counted(self, vertex_ids, timestamp):
        timestamps.append(timestamp)
        return original(self, vertex_ids, timestamp)

    monkeypatch.setattr(DAGValueSession, "evaluate_many", counted)
    annotated = evaluate_feasibility(
        dag,
        signal_samples={"unrelated": [(0.0, 1), (1.0, 2)]},
        signal_policies={"unrelated": {"method": "linear"}},
        prune_dead=False,
    )

    branch = next(vertex for vertex in annotated.vertices if vertex.kind == "branch")
    assert branch.feasibility_verdict == "unknown"
    assert timestamps == [None]
