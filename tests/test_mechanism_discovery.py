from __future__ import annotations

from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag
from flight_log_agent.analysis.mechanism_discovery import (
    binding_from_assignment,
    dag_inputs_from_facts,
    load_facts,
)
from flight_log_agent.px4.mechanism_source_profiler import (
    MechanismSourceProfiler,
    ParameterRef,
    SourceAssignmentRef,
)
from flight_log_agent.px4.source_facts_cache import SourceFileFacts


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
    assert binding["logged_signal"] == "rtl_status.rtl_alt"
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


def test_load_facts_populates_layer1_and_reuses_it(tmp_path):
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

    # rewrite on disk; same hash must serve the cached extraction
    file_abs.write_text("void bar() { _y = 2; }\n", encoding="utf-8")
    again = load_facts(profiler, cache_root, [file_rel], "hash")
    assert {a.target for a in again[0].source_assignments} == {"_x"}


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

    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/rtl.cpp"],
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


def _mini_tree(tmp_path, files: dict[str, str]):
    root = tmp_path / "PX4-Autopilot"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return MechanismSourceProfiler(root, rg_path="missing-rg")


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
    assert result.rounds[0].new_files == ["src/modules/example/rtl.cpp"]
    assert "_dest_val" in result.rounds[0].unresolved_symbols
    assert "src/modules/example/dest.cpp" in result.rounds[1].new_files

    op_targets = {v.variable for v in result.dag.vertices if v.kind == "operation"}
    assert {"_final_out", "_dest_val"} <= op_targets
    logged = {v.signal_name for v in result.dag.vertices
              if v.kind == "evidence" and v.sub_kind == "logged_signal"}
    assert "gspeed" in logged
    assert result.dag.unresolved_symbols == []


def test_fixpoint_stops_at_round_budget(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    _final_out = _dest_val + 1.0f;
}
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
    )

    assert len(result.rounds) == 1
    assert "_dest_val" in result.dag.unresolved_symbols


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


def test_gap_search_drops_ungreppable_queries(tmp_path):
    """A gap whose definition query matches more files than the threshold
    is generic noise ('scale =' matches half the tree) — it must load
    nothing, while a specific gap still resolves to its defining file."""
    from flight_log_agent.analysis.mechanism_discovery import _gap_definition_files

    files = {
        f"src/modules/junk{i}/mod{i}.cpp": f"void f{i}() {{ scale = {i}.0f; }}\n"
        for i in range(12)
    }
    files["src/modules/example/dest.cpp"] = "void g() { _dest_val = gspeed; }\n"
    profiler = _mini_tree(tmp_path, files)

    out = _gap_definition_files(profiler, ["scale", "_dest_val"])

    assert out == ["src/modules/example/dest.cpp"]


def test_gap_search_caps_files_per_gap(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import _gap_definition_files

    files = {
        f"src/modules/example/w{i}.cpp": f"void f{i}() {{ _multi_writer = {i}.0f; }}\n"
        for i in range(4)
    }
    profiler = _mini_tree(tmp_path, files)

    out = _gap_definition_files(profiler, ["_multi_writer"], max_files_per_gap=2)

    assert len(out) == 2


def test_gap_search_never_loads_negatively_scored_files(tmp_path):
    """A gap whose only definition lives in a test/vendored path (negative
    path boost) must load nothing rather than the junk file."""
    from flight_log_agent.analysis.mechanism_discovery import _gap_definition_files

    profiler = _mini_tree(tmp_path, {
        "test/catch2/catch.hpp": "void f() { _only_in_test = 1; }\n",
    })

    assert _gap_definition_files(profiler, ["_only_in_test"]) == []


def test_provider_never_extracts_from_negatively_scored_files(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import make_helper_body_provider

    profiler = _mini_tree(tmp_path, {
        "test/catch2/catch.hpp": "float Rtl::junk_helper(float x) { return x; }\n",
    })
    fetched: list[str] = []
    provider = make_helper_body_provider(profiler, fetched)

    assert provider("junk_helper") == []
    assert fetched == []


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

    found = provider("calc_gain")
    assert found and found[0].name.endswith("calc_gain")
    assert "src/lib/gain/gain.cpp" in fetched
