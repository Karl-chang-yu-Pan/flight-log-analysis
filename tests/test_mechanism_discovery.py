from __future__ import annotations

import pytest

from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag
from flight_log_agent.analysis.mechanism_discovery import (
    binding_from_assignment,
    dag_inputs_from_facts,
    load_facts,
)
from flight_log_agent.analysis.source_expansion import (
    SourceExpansionResolver,
    SourceStructureIndex,
    SourceSymbolIdentity,
    UnresolvedSourceReference,
)
from flight_log_agent.px4.mechanism_source_profiler import (
    FunctionCallRef,
    MechanismSourceProfiler,
    ParameterRef,
    SourceAssignmentRef,
)
from flight_log_agent.px4.source_facts_cache import SourceFileFacts
from flight_log_agent.symbols import exact_symbol


def _assignment(**overrides) -> SourceAssignmentRef:
    base = dict(
        target="_rtl_alt",
        expression="_destination_alt + _param_rtl_return_alt.get()",
        file="src/modules/navigator/rtl.cpp",
        line=248,
        evidence="_rtl_alt = _destination_alt + _param_rtl_return_alt.get();",
    )
    base.update(overrides)
    return SourceAssignmentRef(**base)


def test_binding_from_assignment_maps_all_fields():
    ref = _assignment(
        target_topic="rtl_status",
        target_field="rtl_alt",
        control_predicates=["_param_rtl_type.get() == 1"],
        struct_variables={"gpos": "vehicle_global_position_s"},
    )
    binding = binding_from_assignment(ref)

    assert binding["target_symbol"] == "_rtl_alt"
    assert binding["source_symbol"] == "_destination_alt + _param_rtl_return_alt.get()"
    assert binding["declared_signal"] == "rtl_status.rtl_alt"
    assert binding["logged_signal"] == ""
    assert binding["control_predicates"] == ["_param_rtl_type.get() == 1"]
    assert binding["struct_variables"] == {"gpos": "vehicle_global_position_s"}
    assert binding["assignment_path"] == [
        {
            "file": "src/modules/navigator/rtl.cpp",
            "line": 248,
            "expression": "_destination_alt + _param_rtl_return_alt.get()",
        }
    ]


def test_binding_from_assignment_without_topic_leaves_logged_signal_empty():
    binding = binding_from_assignment(_assignment())
    assert binding["logged_signal"] == ""
    assert binding["target_symbol"] == "_rtl_alt"


def test_dag_inputs_from_facts_aggregates_and_dedupes():
    shared_assignment = _assignment()
    facts_a = SourceFileFacts(
        file="a.cpp",
        source_hash="h",
        source_assignments=[shared_assignment],
        referenced_parameters=[
            ParameterRef(
                name="RTL_RETURN_ALT",
                file="a.cpp",
                line=10,
                evidence="e",
                access_pattern="member",
                member="_param_rtl_return_alt",
            )
        ],
    )
    facts_b = SourceFileFacts(
        file="b.cpp",
        source_hash="h",
        # identical assignment appears again (shared header case) plus one new
        source_assignments=[
            shared_assignment,
            _assignment(target="_destination_alt", expression="gpos_alt", line=127),
        ],
        referenced_parameters=[
            # same member seen again — alias must not flap
            ParameterRef(
                name="RTL_RETURN_ALT",
                file="b.cpp",
                line=20,
                evidence="e",
                access_pattern="member",
                member="_param_rtl_return_alt",
            ),
            # name without member still lands in parameter_names
            ParameterRef(
                name="RTL_TYPE",
                file="b.cpp",
                line=21,
                evidence="e",
                access_pattern="direct",
            ),
        ],
    )

    inputs = dag_inputs_from_facts([facts_a, facts_b])

    targets = [b["target_symbol"] for b in inputs.bindings]
    assert targets.count("_rtl_alt") == 1, "identical assignment must dedupe"
    assert "_destination_alt" in targets
    assert inputs.parameter_aliases == {"_param_rtl_return_alt": "RTL_RETURN_ALT"}
    assert inputs.parameter_names == {"RTL_RETURN_ALT", "RTL_TYPE"}


def test_dag_inputs_preserve_zero_argument_calls():
    facts = SourceFileFacts(
        file="control.cpp",
        source_hash="h",
        function_calls=[
            FunctionCallRef(
                name="update",
                args=[],
                file="control.cpp",
                line=12,
                evidence="controller.update();",
                receiver="controller",
                function="run",
                callable_id="control.cpp:1:run:",
                source_site_id="control.cpp:12:update",
            )
        ],
    )

    inputs = dag_inputs_from_facts([facts])

    assert len(inputs.call_statements) == 1
    assert inputs.call_statements[0]["args"] == []


def test_load_facts_extracts_fresh_without_populating_layer1(tmp_path):
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    file_rel = "src/modules/example/helper.cpp"
    file_abs = tmp_path / "PX4-Autopilot" / file_rel
    file_abs.write_text("void foo() { _x = 1; }\n", encoding="utf-8")

    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")
    cache_root = tmp_path / "cache"

    # duplicate path in the request loads once
    facts = load_facts(profiler, cache_root, [file_rel, file_rel], "hash")
    assert len(facts) == 1
    assert {a.target for a in facts[0].source_assignments} == {"_x"}
    assert not cache_root.exists()

    # A fresh profiler sees the rewrite even under the same source hash: the
    # production discovery path must not consume a persistent Layer 1 entry.
    file_abs.write_text("void bar() { _y = 2; }\n", encoding="utf-8")
    fresh_profiler = MechanismSourceProfiler(
        tmp_path / "PX4-Autopilot", rg_path="missing-rg"
    )
    again = load_facts(fresh_profiler, cache_root, [file_rel], "hash")
    assert {a.target for a in again[0].source_assignments} == {"_y"}
    assert not cache_root.exists()


def test_facts_to_dag_end_to_end(tmp_path):
    """The full Stage 1 seam: profiler extraction → SourceFileFacts →
    dag_inputs_from_facts → build_mechanism_dag on a real (mini) source
    tree, asserting the DAG grounds the terminal in its operations,
    branch, and logged evidence."""
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.cpp").write_text(
        """
void Rtl::pick()
{
    if (_param_rtl_type.get() == 1) {
        _rtl_alt = _destination_alt + 10.0f;
    }
    _destination_alt = gpos_alt;
}
""",
        encoding="utf-8",
    )
    (module_dir / "rtl.h").write_text(
        """
class Rtl
{
    float _rtl_alt;
    float _destination_alt;
};
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/rtl.cpp", "src/modules/example/rtl.h"],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    dag = build_mechanism_dag(
        inputs.bindings,
        "_rtl_alt",
        helper_expressions=inputs.helper_expressions,
        parameter_predicates=inputs.parameter_predicates,
        parameter_names=inputs.parameter_names,
        parameter_aliases=inputs.parameter_aliases,
        logged_signals={"gpos_alt"},
        source_structure=inputs.structure,
    )

    op_targets = {v.variable for v in dag.vertices if v.kind == "operation"}
    assert "_rtl_alt" in op_targets
    assert "_destination_alt" in op_targets

    branches = [v for v in dag.vertices if v.kind == "branch"]
    assert any("_param_rtl_type" in (b.predicate_raw or "") for b in branches)

    logged = {
        v.signal_name
        for v in dag.vertices
        if v.kind == "evidence" and v.sub_kind == "logged_signal"
    }
    assert "gpos_alt" in logged


@pytest.mark.parametrize(
    "copy_statement",
    [
        "_status_sub.copy(&_status);",
        "orb_copy(ORB_ID(vehicle_status), _status_handle, &_status);",
    ],
)
def test_class_owned_copy_grounds_a_sibling_method(tmp_path, copy_statement):
    profiler = _mini_tree(tmp_path, {
        "src/modules/example/reader.cpp": """
class Reader {
    vehicle_status_s _status{};
    uORB::Subscription _status_sub{ORB_ID(vehicle_status)};
    void poll();
    void calculate();
};

void Reader::poll()
{
    COPY_STATEMENT
}

void Reader::calculate()
{
    output = _status.nav_state + 1;
}
""".replace("COPY_STATEMENT", copy_statement),
    })
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/reader.cpp"],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)
    copied = next(
        boundary
        for boundary in inputs.boundary_bindings
        if boundary["source_symbol"] == "_status"
    )
    assert copied["source_owner"] == "Reader"

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file="src/modules/example/reader.cpp",
        boundary_bindings=inputs.boundary_bindings,
        call_statements=inputs.call_statements,
        logged_signals={"vehicle_status.nav_state"},
    )
    leaves = {
        vertex.signal_name
        for vertex in dag.vertices
        if vertex.kind == "evidence" and vertex.sub_kind == "logged_signal"
    }
    assert "vehicle_status.nav_state" in leaves
    assert "_status.nav_state" not in dag.unresolved_symbols


def test_method_local_copy_does_not_ground_same_named_local_in_sibling_method(tmp_path):
    profiler = _mini_tree(tmp_path, {
        "src/modules/example/reader.cpp": """
class Reader {
    uORB::Subscription _status_sub{ORB_ID(vehicle_status)};
    void poll();
    void calculate();
};

void Reader::poll()
{
    vehicle_status_s status{};
    _status_sub.copy(&status);
}

void Reader::calculate()
{
    vehicle_status_s status{};
    output = status.nav_state + 1;
}
""",
    })
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/reader.cpp"],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)
    copied = next(
        boundary
        for boundary in inputs.boundary_bindings
        if boundary["source_symbol"] == "status"
    )
    assert copied["source_owner"] == ""

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file="src/modules/example/reader.cpp",
        boundary_bindings=inputs.boundary_bindings,
        call_statements=inputs.call_statements,
        logged_signals={"vehicle_status.nav_state"},
    )
    leaves = {
        vertex.signal_name
        for vertex in dag.vertices
        if vertex.kind == "evidence" and vertex.sub_kind == "logged_signal"
    }
    assert "vehicle_status.nav_state" not in leaves
    assert "status.nav_state" in dag.unresolved_symbols


def _mini_tree(tmp_path, files: dict[str, str], *, backend: str = "legacy"):
    root = tmp_path / "PX4-Autopilot"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend=backend,
    )


def test_fixpoint_resolves_symbol_across_files_in_second_round(tmp_path):
    """Round 0 loads only the seeded file and leaves ``_dest_val``
    unresolved; the gap's definition search pulls the second file in
    round 1 and the slice completes — the DAG's own gaps drive discovery."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    _final_out = _dest_val + 1.0f;
}
""",
        "src/modules/example/rtl.h": """
class Rtl
{
    float _final_out;
    float _dest_val;
};
""",
        "src/modules/example/dest.cpp": """
void Rtl::update()
{
    _dest_val = gspeed;
}
""",
    })

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=["pick_altitude"],
        terminal="_final_out",
        source_hash="hash",
        logged_signals={"gspeed"},
    )

    assert len(result.rounds) == 2
    assert set(result.rounds[0].new_files) == {
        "src/modules/example/rtl.cpp",
        "src/modules/example/rtl.h",
    }
    assert "_dest_val" in result.rounds[0].unresolved_symbols
    assert "src/modules/example/dest.cpp" in result.rounds[1].new_files

    op_targets = {v.variable for v in result.dag.vertices if v.kind == "operation"}
    assert {"_final_out", "_dest_val"} <= op_targets
    logged = {v.signal_name for v in result.dag.vertices
              if v.kind == "evidence" and v.sub_kind == "logged_signal"}
    assert "gspeed" in logged
    assert result.dag.unresolved_symbols == []


def test_fixpoint_ignores_legacy_round_budget(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    _final_out = _dest_val + 1.0f;
}
""",
        "src/modules/example/rtl.h": """
class Rtl
{
    float _final_out;
    float _dest_val;
};
""",
        "src/modules/example/dest.cpp": """
void Rtl::update()
{
    _dest_val = gspeed;
}
""",
    })

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=["pick_altitude"],
        terminal="_final_out",
        source_hash="hash",
        max_rounds=1,
        logged_signals={"gspeed"},
    )

    assert len(result.rounds) == 2
    assert result.dag.unresolved_symbols == []


def test_fixpoint_provider_expands_cross_file_helper(tmp_path):
    """A helper defined in a file discovery never seeded still expands:
    the on-demand provider finds ``Class::name(`` within the round, and
    the fetched file's facts join the next round."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    _alt_out = calc_gain(base_in);
}
""",
        "src/lib/gain/gain.cpp": """
float Rtl::calc_gain(float base_in)
{
    return base_in * 2.0f;
}
""",
    })

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=["pick_altitude"],
        terminal="_alt_out",
        source_hash="hash",
    )

    helper_returns = [v for v in result.dag.vertices
                      if v.provenance and v.provenance.startswith("helper_return")]
    assert helper_returns, "cross-file helper did not expand via provider"
    assert "src/lib/gain/gain.cpp" in result.files_loaded


def test_fixpoint_bootstraps_from_terminal_without_seeds(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    _lone_terminal = 42.0f;
}
""",
    })

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=[],
        terminal="_lone_terminal",
        source_hash="hash",
    )

    op_targets = {v.variable for v in result.dag.vertices if v.kind == "operation"}
    assert "_lone_terminal" in op_targets


def test_gap_search_keeps_every_exact_definition(tmp_path):
    """Hit count cannot reject exact definitions; provenance resolves them."""
    from flight_log_agent.analysis.mechanism_discovery import _gap_definition_files

    files = {
        f"src/modules/junk{i}/mod{i}.cpp": f"void f{i}() {{ scale = {i}.0f; }}\n"
        for i in range(12)
    }
    files["src/modules/example/dest.cpp"] = "void g() { _dest_val = gspeed; }\n"
    profiler = _mini_tree(tmp_path, files)

    out = _gap_definition_files(profiler, ["scale", "_dest_val"])

    assert "src/modules/example/dest.cpp" in out
    assert len([path for path in out if "junk" in path]) == 12


def test_gap_search_ignores_legacy_file_cap(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import _gap_definition_files

    files = {
        f"src/modules/example/w{i}.cpp": f"void f{i}() {{ _multi_writer = {i}.0f; }}\n"
        for i in range(4)
    }
    profiler = _mini_tree(tmp_path, files)

    out = _gap_definition_files(profiler, ["_multi_writer"], max_files_per_gap=2)

    assert len(out) == 4


def test_gap_search_does_not_reject_exact_definition_by_path_score(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import _gap_definition_files

    profiler = _mini_tree(tmp_path, {
        "test/catch2/catch.hpp": "void f() { _only_in_test = 1; }\n",
    })

    assert _gap_definition_files(profiler, ["_only_in_test"]) == [
        "test/catch2/catch.hpp"
    ]


def test_provider_admits_exact_callable_independent_of_path_score(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import make_helper_body_provider

    profiler = _mini_tree(tmp_path, {
        "test/catch2/catch.hpp": "float Rtl::junk_helper(float x) { return x; }\n",
    })
    fetched: list[str] = []
    provider = make_helper_body_provider(profiler, fetched)
    reference = UnresolvedSourceReference(
        symbol="junk_helper",
        kind="callable",
        class_owner="Rtl",
        argument_count=1,
    )

    assert len(provider("junk_helper", reference)) == 1
    assert fetched == ["test/catch2/catch.hpp"]


def test_provider_resolves_definition_despite_many_callers(tmp_path):
    """A real helper is *called* from many files (geo.cpp's
    get_distance_to_next_waypoint). The provider must still resolve its
    definition — the definition file matches both the ``::name(`` and
    bare queries and outranks bare-call hits, and extraction filters to
    the requested name — rather than treating caller count as ambiguity."""
    from flight_log_agent.analysis.mechanism_discovery import make_helper_body_provider

    files = {
        f"src/modules/caller{i}/mod{i}.cpp": f"void f{i}() {{ _x{i} = calc_gain({i}.0f); }}\n"
        for i in range(11)
    }
    files["src/lib/gain/gain.cpp"] = "float Rtl::calc_gain(float x) { return x * 2.0f; }\n"
    profiler = _mini_tree(tmp_path, files)
    fetched: list[str] = []
    provider = make_helper_body_provider(profiler, fetched)
    reference = UnresolvedSourceReference(
        symbol="calc_gain",
        kind="callable",
        class_owner="Rtl",
        argument_count=1,
    )

    found = provider("calc_gain", reference)
    assert found and found[0].name.endswith("calc_gain")
    assert "src/lib/gain/gain.cpp" in fetched


def test_provider_keeps_multiple_exact_callable_definitions_unresolved(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import make_helper_body_provider

    profiler = _mini_tree(
        tmp_path,
        {
            "platforms/first/clock.cpp": "float platform_clock() { return 1.0f; }",
            "platforms/second/clock.cpp": "float platform_clock() { return 2.0f; }",
        },
    )
    fetched: list[str] = []
    provider = make_helper_body_provider(profiler, fetched)

    assert provider("platform_clock") == []
    assert fetched == []


def test_qualified_terminal_is_stripped_to_bare_member(tmp_path):
    """Seeder-style qualified terminals (``Rtl::_final_out``) must slice
    the same DAG as the bare member — a qualified form matches no writer
    and previously produced an empty DAG."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    _final_out = gspeed + 1.0f;
}
""",
    })

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=["pick_altitude"],
        terminal="Rtl::_final_out",
        source_hash="hash",
        logged_signals={"gspeed"},
    )

    op_targets = {v.variable for v in result.dag.vertices if v.kind == "operation"}
    assert "_final_out" in op_targets


def test_call_statement_argument_flow_crosses_object_boundary(tmp_path):
    """The airspeed-residual shape: a caller local flows into a member
    object's state only through a bare call statement's argument. The
    synthesized formal<-actual binding lets the backward slice cross
    caller -> callee: _speed_state <- speed_sp(formal) <- target_speed
    <- adapt_speed() and its gating branch."""
    from flight_log_agent.analysis.mechanism_discovery import (
        dag_inputs_from_facts,
        load_facts,
    )

    profiler = _mini_tree(tmp_path, {
        "src/modules/fw/fw.h": """
class Tecs;
class Fw {
    Tecs _tecs;
    void control();
    float adapt_speed(float base_speed);
};
""",
        "src/modules/fw/fw.cpp": """
void Fw::control()
{
    float target_speed = adapt_speed(base_speed);
    _tecs.update(target_speed);
}

float Fw::adapt_speed(float base_speed)
{
    if (_param_gnd_min.get() > base_speed) {
        return _param_gnd_min.get();
    }
    return base_speed;
}
""",
        "src/lib/tecs/tecs.h": """
class Tecs {
    float _speed_state;
    void update(float speed_sp);
};
""",
        "src/lib/tecs/tecs.cpp": """
void Tecs::update(float speed_sp)
{
    _speed_state = speed_sp;
}
""",
    })
    facts = load_facts(
        profiler, tmp_path / "cache",
        [
            "src/modules/fw/fw.cpp",
            "src/modules/fw/fw.h",
            "src/lib/tecs/tecs.cpp",
            "src/lib/tecs/tecs.h",
        ],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    dag = build_mechanism_dag(
        inputs.bindings,
        "_speed_state",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
    )

    op_targets = {v.variable for v in dag.vertices if v.kind == "operation"}
    assert "_speed_state" in op_targets
    assert "speed_sp" in op_targets, "synthesized formal<-actual hop missing"
    assert "target_speed" in op_targets, "caller local not reached"
    branches = [v.predicate_raw or "" for v in dag.vertices if v.kind == "branch"]
    assert any("_param_gnd_min" in branch for branch in branches), (
        "adaptation branch not reached through the argument hop"
    )


def test_tree_sitter_pointer_output_crosses_inherited_helper_boundary(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/mode/mode.h": """
struct mission_item_s;
struct position_setpoint_triplet_s;
class Base;
class Mode : public Base {
public:
    void run();
    position_setpoint_triplet_s *get_triplet();
private:
    mission_item_s _item;
};
""",
            "src/modules/mode/mode.cpp": """
void Mode::run()
{
    position_setpoint_triplet_s *triplet = get_triplet();
    fill(_item, &triplet->current);
}
""",
            "src/modules/mode/base.h": """
struct mission_item_s;
struct position_setpoint_s;
class Base {
protected:
    void fill(const mission_item_s &item, position_setpoint_s *sp);
};
""",
            "src/modules/mode/base.cpp": """
void Base::fill(const mission_item_s &item, position_setpoint_s *sp)
{
    if (item.valid) {
        sp->alt = item.altitude;
    } else {
        sp->alt = 0.0f;
    }
}
""",
        },
        backend="tree_sitter",
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        [
            "src/modules/mode/mode.h",
            "src/modules/mode/mode.cpp",
            "src/modules/mode/base.h",
            "src/modules/mode/base.cpp",
        ],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    dag = build_mechanism_dag(
        inputs.bindings,
        "get_triplet().current.alt",
        terminal_file="src/modules/mode/mode.cpp",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
    )

    operations = [vertex for vertex in dag.vertices if vertex.kind == "operation"]
    effect = next(
        vertex
        for vertex in operations
        if exact_symbol(vertex.variable or "")
        == exact_symbol("get_triplet().current.alt")
        and vertex.expression == "sp.alt"
    )
    helper_writers = [
        vertex
        for vertex in operations
        if exact_symbol(vertex.variable or "") == exact_symbol("sp.alt")
    ]
    assert len(helper_writers) == 2
    assert {vertex.id for vertex in helper_writers} <= {
        edge.source_id for edge in dag.edges if edge.target_id == effect.id
    }
    assert any(
        vertex.kind == "branch" and "item.valid" in (vertex.predicate_raw or "")
        for vertex in dag.vertices
    )


def test_tree_sitter_expansion_admission_uses_selected_backend(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/candidate.cpp": """
void Controller::run()
{
    output = input;
}
""",
        },
        backend="tree_sitter",
    )

    def fail_legacy_scan(*args, **kwargs):
        raise AssertionError("Tree-sitter admission must not invoke the legacy scanner")

    profiler.extract_source_assignments_from_source = fail_legacy_scan  # type: ignore[assignment]
    resolver = SourceExpansionResolver(profiler, "hash")
    candidates = resolver.resolve(
        UnresolvedSourceReference(symbol="output"),
        SourceStructureIndex(),
    )

    assert len(candidates) == 1
    assert candidates[0].facts.parser_backend == "tree_sitter"


def _vt_binding(
    target: str,
    file: str,
    logged: str = "",
    function: str = "C::f",
) -> dict:
    return {
        "target_symbol": target,
        "source_symbol": "input_val + 1.0f",
        "function": function,
        "assignment_path": [{"file": file, "line": 1, "expression": "input_val + 1.0f"}],
        "logged_signal": logged,
        "control_predicates": [],
        "struct_variables": {},
    }


def test_validate_terminal_statuses():
    """Terminal validation resolves against actual write targets and the
    observed catalogue — qualified spellings canonicalize, absence and
    observed-output membership are explicit."""
    from flight_log_agent.analysis.mechanism_discovery import validate_terminal

    bindings = [_vt_binding("_final_out", "src/modules/example/rtl.cpp")]

    valid = validate_terminal("_final_out", bindings, [])
    assert valid.status == "valid"
    assert valid.write_files == ["src/modules/example/rtl.cpp"]

    qualified = validate_terminal("Rtl::_final_out", bindings, [])
    assert qualified.status == "valid" and qualified.terminal == "_final_out"

    absent = validate_terminal("_ghost_var", bindings, [])
    assert absent.status == "absent" and "no write target" in absent.reason

    observed = validate_terminal("topic_a.alt", [], ["topic_a.alt"])
    assert observed.status == "valid" and observed.logged is True


def test_validate_terminal_scoping_rules():
    """Without ownership metadata, same names in different files remain
    ambiguous; an explicit file or its companion can scope the terminal."""
    from flight_log_agent.analysis.mechanism_discovery import validate_terminal

    cross = [
        _vt_binding("dist", "src/modules/aaa/alpha.cpp"),
        _vt_binding("dist", "src/modules/bbb/beta.cpp"),
    ]
    ambiguous = validate_terminal("dist", cross, [])
    assert ambiguous.status == "ambiguous"
    assert "alpha.cpp" in ambiguous.reason and "beta.cpp" in ambiguous.reason

    scoped = validate_terminal(
        "dist", cross, [], terminal_file="src/modules/aaa/alpha.cpp"
    )
    assert scoped.status == "valid"
    assert scoped.resolved_file == "src/modules/aaa/alpha.cpp"

    twin = validate_terminal(
        "dist", cross, [], terminal_file="src/modules/aaa/alpha.hpp"
    )
    assert twin.status == "valid"
    assert twin.resolved_file == "src/modules/aaa/alpha.cpp"

    outside = validate_terminal(
        "dist", cross, [], terminal_file="src/modules/ccc/gamma.cpp"
    )
    assert outside.status == "absent_in_scope"

    member_same_module = [
        _vt_binding("_alt", "src/modules/aaa/alpha.cpp"),
        _vt_binding("_alt", "src/modules/aaa/base.cpp"),
    ]
    assert validate_terminal("_alt", member_same_module, []).status == "ambiguous"

    member_cross_module = [
        _vt_binding("_alt", "src/modules/aaa/alpha.cpp"),
        _vt_binding("_alt", "src/modules/bbb/beta.cpp"),
    ]
    assert validate_terminal("_alt", member_cross_module, []).status == "ambiguous"

    # Directory proximity is not ownership evidence.
    inherited = validate_terminal(
        "_alt", member_cross_module, [], terminal_file="src/modules/aaa/other.cpp"
    )
    assert inherited.status == "absent_in_scope"


def test_discovery_rejects_ambiguous_terminal_without_declared_file(tmp_path):
    """Two modules writing the same bare name are different variables:
    with no declared terminal file nothing is built (slicing would fuse
    them); the declared file scopes the slice to one module."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/aaa/alpha.cpp": """
void Alpha::run()
{
    shared_out = alpha_in + 1.0f;
}
""",
        "src/modules/bbb/beta.cpp": """
void Beta::run()
{
    shared_out = beta_in + 2.0f;
}
""",
    })

    rejected = discover_mechanism_dag(
        profiler, tmp_path / "cache",
        seeds=["shared_out"], terminal="shared_out", source_hash="hash",
    )
    assert rejected.dag is None
    assert rejected.terminal_validation.status == "ambiguous"

    scoped = discover_mechanism_dag(
        profiler, tmp_path / "cache",
        seeds=["shared_out"], terminal="shared_out", source_hash="hash",
        terminal_file="src/modules/bbb/beta.cpp",
    )
    assert scoped.terminal_validation.status == "valid"
    terminal_ops = [
        v for v in scoped.dag.vertices
        if v.kind == "operation" and v.variable == "shared_out"
    ]
    assert terminal_ops
    assert all(v.file == "src/modules/bbb/beta.cpp" for v in terminal_ops)
    # The declared file is loaded first — validation decides with it in
    # evidence even when the ranked seed search would defer it.
    assert scoped.rounds[0].new_files[0] == "src/modules/bbb/beta.cpp"


def test_discovery_reports_absent_terminal(tmp_path):
    """A terminal with no write target anywhere in the loaded facts
    builds nothing and carries the structured reason."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/aaa/alpha.cpp": """
void Alpha::run()
{
    real_out = alpha_in + 1.0f;
}
""",
    })

    result = discover_mechanism_dag(
        profiler, tmp_path / "cache",
        seeds=["alpha_in"], terminal="_ghost_out", source_hash="hash",
    )
    assert result.dag is None
    assert result.terminal_validation.status == "absent"
    assert "no write target" in result.terminal_validation.reason


def test_expansion_does_not_search_for_callable_local_producers(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/ctrl.cpp": """
void Ctrl::run()
{
    output = local_value;
}
""",
        },
    )

    def fail_search(*args, **kwargs):
        raise AssertionError("a callable-local reference cannot resolve tree-wide")

    profiler.search_related_source_files = fail_search  # type: ignore[assignment]
    identity = SourceSymbolIdentity(
        kind="local",
        symbol="local_value",
        root="local_value",
        file="src/modules/example/ctrl.cpp",
        callable_id="src/modules/example/ctrl.cpp:2:Ctrl::run:",
        class_owner="Ctrl",
    )
    reference = UnresolvedSourceReference(
        symbol="local_value",
        file=identity.file,
        callable_id=identity.callable_id,
        class_owner="Ctrl",
        identity=identity,
    )

    resolver = SourceExpansionResolver(profiler, "hash")
    assert resolver.resolve(reference, SourceStructureIndex()) == []


def test_expansion_does_not_bind_method_without_receiver_type(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/unrelated/other.cpp": """
float Other::get()
{
    return 1.0f;
}
""",
        },
    )
    reference = UnresolvedSourceReference(
        symbol="get",
        kind="callable",
        file="src/modules/example/ctrl.cpp",
        callable_id="src/modules/example/ctrl.cpp:2:Ctrl::run:",
        class_owner="Ctrl",
        receiver="_param_value",
        argument_count=0,
    )

    resolver = SourceExpansionResolver(profiler, "hash")
    assert resolver.resolve(reference, SourceStructureIndex()) == []


def test_discovery_checks_all_exact_terminal_writers_before_slicing(tmp_path):
    """Unbounded terminal retrieval exposes ambiguity before graph build."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/aaa/alpha.cpp": """
void Alpha::run_alpha_marker()
{
    shared_out = helper_in + 1.0f;
}
""",
        "src/modules/bbb/beta.cpp": """
void Beta::run()
{
    helper_in = 3.0f;
    shared_out = beta_src + 2.0f;
}
""",
    })

    result = discover_mechanism_dag(
        profiler, tmp_path / "cache",
        seeds=["run_alpha_marker"], terminal="shared_out", source_hash="hash",
        max_files_per_round=1,
    )
    assert set(result.rounds[0].new_files) == {
        "src/modules/aaa/alpha.cpp",
        "src/modules/bbb/beta.cpp",
    }
    assert result.dag is None
    assert result.terminal_validation.status == "ambiguous"


def test_qualifier_scopes_terminal_to_class_family():
    """A qualifier resolves through declared class ownership, not filenames."""
    from flight_log_agent.analysis.mechanism_discovery import validate_terminal

    structure = SourceStructureIndex(
        direct_bases={"Alpha": set(), "Beta": set()},
        members={
            ("Alpha", "shared_out"): {"type": "float"},
            ("Beta", "shared_out"): {"type": "float"},
        },
    )
    cross = structure.enrich_bindings([
        _vt_binding(
            "shared_out", "src/modules/aaa/alpha.cpp", function="Alpha::run"
        ),
        _vt_binding(
            "shared_out", "src/modules/bbb/beta.cpp", function="Beta::run"
        ),
    ])

    qualified = validate_terminal(
        "Alpha::shared_out", cross, [], source_structure=structure
    )
    assert qualified.status == "valid"
    assert qualified.resolved_file == "src/modules/aaa/alpha.cpp"
    assert qualified.terminal == "shared_out"

    snake_structure = SourceStructureIndex(
        direct_bases={"MissionBlock": set(), "Thing": set()},
        members={
            ("MissionBlock", "dist"): {"type": "float"},
            ("Thing", "dist"): {"type": "float"},
        },
    )
    snake_bindings = snake_structure.enrich_bindings([
        _vt_binding(
            "dist",
            "src/modules/nav/mission_block.cpp",
            function="MissionBlock::run",
        ),
        _vt_binding(
            "dist", "src/modules/other/thing.cpp", function="Thing::run"
        ),
    ])
    snake = validate_terminal(
        "MissionBlock::dist",
        snake_bindings,
        [],
        source_structure=snake_structure,
    )
    assert snake.status == "valid"
    assert snake.resolved_file == "src/modules/nav/mission_block.cpp"

    unknown = validate_terminal(
        "Gamma::shared_out", cross, [], source_structure=structure
    )
    assert unknown.status == "ambiguous"


def test_binding_carries_branch_sites_and_reachability():
    ref = _assignment(
        control_predicates=["a > 0"],
        control_predicate_lines=[42],
        reachability_exact=False,
    )
    binding = binding_from_assignment(ref)
    assert binding["control_predicates"] == ["a > 0"]
    assert binding["control_predicate_lines"] == [42]
    assert binding["reachability_exact"] is False


def test_publication_proof_is_scoped_to_the_calling_callable(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/example.cpp": """
uORB::Publication<alpha_s> _pub{ORB_ID(alpha)};

void Example::publish_value()
{
    alpha_s msg{};
    msg.value = input_value;
    _pub.publish(msg);
}

void Example::calculate_only()
{
    alpha_s msg{};
    msg.value = unrelated_value;
}
""",
        },
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/example.cpp"],
        "hash",
    )

    inputs = dag_inputs_from_facts(facts)
    by_expression = {
        binding["source_symbol"]: binding for binding in inputs.bindings
    }
    assert by_expression["input_value"]["logged_signal"] == "alpha.value"
    assert by_expression["unrelated_value"]["logged_signal"] == ""


def test_survey_anchors_on_declared_parameter_and_selects_its_writes(tmp_path):
    """The survey is ANCHORED, not a dump: a parameter the question names
    resolves through DEFINE_PARAMETERS to its member, and the writes that
    read that member are the decision sites. Unrelated writes in the same
    file stay out."""
    from flight_log_agent.analysis.mechanism_discovery import survey_source_files

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/ctrl.cpp": """
class Ctrl
{
    DEFINE_PARAMETERS(
        (ParamFloat<px4::params::EXA_TRIM_SPD>) _param_exa_trim_spd
    )
};

void Ctrl::update()
{
    speed_sp = _param_exa_trim_spd.get() + margin;
    unrelated_out = counter + 1.0f;
    logger_value = unrelated_out * 2.0f;
}
""",
    })

    survey = survey_source_files(
        profiler,
        ["src/modules/example/ctrl.cpp"],
        texts=["why is the speed above EXA_TRIM_SPD sometimes?"],
    )

    assert survey.anchors.parameters == {"EXA_TRIM_SPD"}
    assert survey.anchors.parameter_members == {"_param_exa_trim_spd"}

    targets = {t.symbol: t for entry in survey.files for t in entry.targets}
    # hop 0: the write that reads the anchored parameter
    assert targets["speed_sp"].distance == 0
    assert "_param_exa_trim_spd" in targets["speed_sp"].expression
    assert targets["speed_sp"].function == "Ctrl::update"
    # hop 1: producer of an input the anchored write reads
    assert targets["margin"].distance == 1 if "margin" in targets else True
    # a write unreachable from the anchor is NOT surveyed
    assert "unrelated_out" not in targets
    assert "logger_value" not in targets


def test_survey_anchors_on_observed_signal_and_defined_callable(tmp_path):
    """Signals anchor only when the flight recorded them; callables only
    when the tree defines them. A name the tree does not confirm anchors
    nothing."""
    from flight_log_agent.analysis.mechanism_discovery import survey_source_files

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/ctrl.cpp": """
void Ctrl::adapt_value()
{
    adapted = topic_a.field_x * 2.0f;
}

void Ctrl::other()
{
    elsewhere = 5.0f;
}
""",
    })

    signal_anchored = survey_source_files(
        profiler,
        ["src/modules/example/ctrl.cpp"],
        texts=["why does topic_a.field_x drive the output?"],
        observed_signals={"topic_a.field_x"},
    )
    assert signal_anchored.anchors.signals == {"topic_a.field_x"}
    assert "adapted" in signal_anchored.symbols()
    assert "elsewhere" not in signal_anchored.symbols()

    # the same signal name, NOT observed in this flight, anchors nothing
    unobserved = survey_source_files(
        profiler,
        ["src/modules/example/ctrl.cpp"],
        texts=["why does topic_a.field_x drive the output?"],
        observed_signals=set(),
    )
    assert unobserved.anchors.signals == set()

    callable_anchored = survey_source_files(
        profiler,
        ["src/modules/example/ctrl.cpp"],
        texts=["explain adapt_value"],
    )
    assert callable_anchored.anchors.callables == {"adapt_value"}
    assert "adapted" in callable_anchored.symbols()
    assert "elsewhere" not in callable_anchored.symbols()


def test_survey_without_anchors_reports_every_target(tmp_path):
    """No anchor the tree confirms: report the ranked files' write targets
    in full — honest breadth, never a silent cut."""
    from flight_log_agent.analysis.mechanism_discovery import survey_source_files

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/ctrl.cpp": """
void Ctrl::run()
{
    first_out = a + 1.0f;
    second_out = b + 2.0f;
}
""",
    })

    survey = survey_source_files(
        profiler, ["src/modules/example/ctrl.cpp"], texts=["nothing recognizable"]
    )

    assert not survey.anchors
    assert survey.symbols() == {"first_out", "second_out"}


def test_survey_payload_carries_expressions_and_hops(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import survey_source_files

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/ctrl.cpp": """
class Ctrl
{
    DEFINE_PARAMETERS(
        (ParamFloat<px4::params::EXA_TRIM_SPD>) _param_exa_trim_spd
    )
};

void Ctrl::update()
{
    status_s status;
    status.speed_sp = _param_exa_trim_spd.get();
    orb_publish(ORB_ID(status), _pub, &status);
}
""",
    })

    survey = survey_source_files(
        profiler,
        ["src/modules/example/ctrl.cpp"],
        texts=["EXA_TRIM_SPD"],
    )
    payload = survey.as_payload()

    assert payload["anchors"]["parameters"] == ["EXA_TRIM_SPD"]
    entry = payload["files"][0]
    assert entry["file"] == "src/modules/example/ctrl.cpp"
    target = next(t for t in entry["write_targets"] if t["symbol"] == "status.speed_sp")
    assert target["reaches_anchor_in_hops"] == 0
    assert "_param_exa_trim_spd" in target["expression"]
    assert target["published_signal"] == "status.speed_sp"


def test_survey_preserves_same_named_locals_in_distinct_callables(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import survey_source_files

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/ctrl.cpp": """
void Ctrl::first()
{
    value = first_input;
}

void Ctrl::second()
{
    value = second_input;
}
""",
    })

    survey = survey_source_files(
        profiler,
        ["src/modules/example/ctrl.cpp"],
        texts=["no source anchor"],
        source_hash="hash",
    )

    targets = [
        target
        for entry in survey.files
        for target in entry.targets
        if target.symbol == "value"
    ]
    assert len(targets) == 2
    assert len({target.source_target_id for target in targets}) == 2
    assert {target.identity["kind"] for target in targets} == {"local"}
    assert len({target.identity["callable_id"] for target in targets}) == 2


def test_terminal_identity_separates_locals_but_unifies_class_members():
    from flight_log_agent.analysis.mechanism_discovery import validate_terminal

    local_structure = SourceStructureIndex()
    local_bindings = local_structure.enrich_bindings([
        _vt_binding("value", "src/modules/example/ctrl.cpp", function="Ctrl::first"),
        _vt_binding("value", "src/modules/example/ctrl.cpp", function="Ctrl::second"),
    ])
    assert validate_terminal(
        "value",
        local_bindings,
        [],
        terminal_file="src/modules/example/ctrl.cpp",
        source_structure=local_structure,
    ).status == "ambiguous"

    first_identity = local_bindings[0]["target_identity"]
    scoped = validate_terminal(
        "value",
        local_bindings,
        [],
        terminal_file="src/modules/example/ctrl.cpp",
        source_structure=local_structure,
        terminal_identity=first_identity,
    )
    assert scoped.status == "valid"
    dag = build_mechanism_dag(
        local_bindings,
        "value",
        terminal_file="src/modules/example/ctrl.cpp",
        terminal_identity=scoped.resolved_identity,
        source_structure=local_structure,
    )
    terminal_ops = [
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation" and vertex.metadata.get("is_terminal")
    ]
    assert len(terminal_ops) == 1
    assert terminal_ops[0].metadata["function"] == "Ctrl::first"

    member_structure = SourceStructureIndex(
        direct_bases={"Ctrl": set()},
        members={("Ctrl", "value"): {"name": "value", "owner": "Ctrl"}},
    )
    member_bindings = member_structure.enrich_bindings([
        _vt_binding("value", "src/modules/example/ctrl.cpp", function="Ctrl::first"),
        _vt_binding("value", "src/modules/example/ctrl.cpp", function="Ctrl::second"),
    ])
    member = validate_terminal(
        "value", member_bindings, [], source_structure=member_structure
    )
    assert member.status == "valid"
    assert member.resolved_identity["kind"] == "member"

    base_writer = SourceSymbolIdentity(
        kind="member",
        symbol="value",
        root="value",
        class_owner="Base",
        declaring_class="Base",
    )
    derived_writer = SourceSymbolIdentity(
        kind="member",
        symbol="value",
        root="value",
        class_owner="Derived",
        declaring_class="Base",
    )
    assert SourceStructureIndex.storage_compatible(base_writer, derived_writer)



def test_discovery_accepts_preranked_files(tmp_path):
    """A declared terminal file can reuse its survey-ranked candidate set."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/ctrl.cpp": """
void Ctrl::run()
{
    speed_sp = gspeed + 1.0f;
}
""",
    })

    def _fail(*args, **kwargs):
        raise AssertionError("search must not run when files are preranked")

    profiler.search_related_source_files = _fail  # type: ignore[assignment]

    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=[], terminal="speed_sp",
        source_hash="hash", logged_signals={"gspeed"},
        preranked_files=["src/modules/example/ctrl.cpp"],
        terminal_file="src/modules/example/ctrl.cpp",
    )

    assert result.terminal_validation.status == "valid"
    assert {v.variable for v in result.dag.vertices if v.kind == "operation"} == {"speed_sp"}
