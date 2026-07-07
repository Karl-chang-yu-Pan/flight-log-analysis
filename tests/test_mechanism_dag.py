from __future__ import annotations

from flight_log_agent.analysis.mechanism_dag import (
    build_mechanism_dag,
    evaluate_feasibility,
    layer2_cache_path,
    layer3_cache_path,
    read_dag_from_cache,
    split_by_terminal,
    write_dag_to_cache,
)


def _fake_binding(
    *,
    binding_id: str,
    target: str,
    expression: str,
    file: str,
    line: int,
    logged_signal: str | None = None,
    control_predicates: list[str] | None = None,
) -> dict:
    """A binding with ``logged_signal`` defaulted to the target — the
    BindingIndex backward walk starts from ``logged_signal``, so tests
    seed it explicitly."""
    return {
        "binding_id": binding_id,
        "source_symbol": expression,
        "target_symbol": target,
        "logged_signal": target if logged_signal is None else logged_signal,
        "assignment_path": [{"file": file, "line": line, "expression": expression}],
        "control_predicates": control_predicates or [],
    }


def _fake_inventory(topics: dict[str, list[str]] | None = None) -> dict:
    return {
        "topic_fields": topics or {},
        "available_topics": list((topics or {}).keys()),
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


def test_branch_deduplicates_by_predicate():
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

    dag = build_mechanism_dag(bindings, "_destination.alt")

    branches = [v for v in dag.vertices if v.kind == "branch"]
    assert len(branches) == 1
    assert branches[0].predicate_raw == predicate


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


def test_helper_body_preserves_intermediates_and_reuses_subgraph():
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
    assert len(helper_terminals) == 1, "helper subgraph should be reused across callers"

    caller_op_ids = {
        v.id for v in dag.vertices if v.variable in {"_rtl_alt", "_alt_snapshot"}
    }
    via_edges = [e for e in dag.edges if e.via == "calc_cone_alt" and e.target_id in caller_op_ids]
    assert len(via_edges) == 2


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
    )

    logged = next(
        (v for v in dag.vertices
         if v.kind == "evidence" and v.sub_kind == "logged_signal"
         and v.signal_name == "vehicle_status.vehicle_type"),
        None,
    )
    assert logged is not None
    assert logged.metadata.get("source_form")
    assert logged.metadata.get("derivation") == "helper_return_type"


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


def test_struct_root_expansion_walks_through_bare_struct_argument():
    """Backward walk landing on a bare ``_mission_item`` (no direct writer,
    only field writes) should expand into ``_mission_item.*`` writes so
    the chain closes at the field level."""
    bindings = [
        # Terminal: mission_item_altitude_amsl = get_absolute_altitude_for_item(_mission_item)
        _fake_binding(
            binding_id="terminal",
            target="mission_item_altitude_amsl",
            expression="get_absolute_altitude_for_item(_mission_item)",
            file="mission_block.cpp",
            line=190,
        ),
        # Two field writes on _mission_item — the walk would normally
        # never see these because nothing writes _mission_item bare.
        _fake_binding(
            binding_id="field_altitude",
            target="_mission_item.altitude",
            expression="_rtl_alt",
            file="rtl.cpp",
            line=361,
        ),
        _fake_binding(
            binding_id="field_lat",
            target="_mission_item.lat",
            expression="_destination.lat",
            file="rtl.cpp",
            line=360,
        ),
        # An upstream write for _rtl_alt so the walk continues past the
        # field expansion.
        _fake_binding(
            binding_id="rtl_alt_write",
            target="_rtl_alt",
            expression="max(gpos.alt, _destination.alt + _param_rtl_return_alt.get())",
            file="rtl.cpp",
            line=248,
        ),
    ]
    dag = build_mechanism_dag(bindings, "mission_item_altitude_amsl")

    variables = {v.variable for v in dag.vertices if v.kind == "operation"}
    # All three field-level writes + the upstream _rtl_alt should be reachable.
    assert "mission_item_altitude_amsl" in variables
    assert "_mission_item.altitude" in variables
    assert "_mission_item.lat" in variables
    assert "_rtl_alt" in variables


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
    assert pointer_op.expression == "_rtl_alt"
    assert (pointer_op.provenance or "").startswith("pointer_output:")


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
        )
    )

    dag = build_mechanism_dag(
        bindings,
        "_rtl_alt",
        inventory=_fake_inventory({"vehicle_status": ["vehicle_type"]}),
        helper_expressions=[helper],
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


def test_struct_var_field_resolves_via_return_type_convention():
    """A ``vstatus.vehicle_type`` predicate should resolve graph-natively
    when ``vstatus`` is struct-typed — same ``foo_s`` → topic convention
    as helper return types, no flat symbol_bindings entry required."""
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
    )

    branch = next(v for v in dag.vertices if v.kind == "branch")
    assert "vehicle_status.vehicle_type" in (branch.predicate_lowered or "")


def test_struct_var_from_source_assignment_reaches_dag():
    """Struct-variable maps arriving via SourceAssignmentRef.struct_variables
    (not just helper records) should also be aggregated by the DAG builder."""
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
    assert "vehicle_status.vehicle_type" in (branch.predicate_lowered or "")


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
    )
    params = {v.signal_name for v in dag.vertices
              if v.kind == "evidence" and v.sub_kind == "parameter"}
    logged = {v.signal_name for v in dag.vertices
              if v.kind == "evidence" and v.sub_kind == "logged_signal"}
    assert "RTL_CONE_ANG" in params
    assert "vehicle_status.vehicle_type" in logged
