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
        "src/lib/tecs/tecs.cpp": """
void Tecs::update(float speed_sp)
{
    _speed_state = speed_sp;
}
""",
    })
    facts = load_facts(
        profiler, tmp_path / "cache",
        ["src/modules/fw/fw.cpp", "src/lib/tecs/tecs.cpp"], "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    dag = build_mechanism_dag(
        inputs.bindings,
        "_speed_state",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
    )

    op_targets = {v.variable for v in dag.vertices if v.kind == "operation"}
    assert "_speed_state" in op_targets
    assert "speed_sp" in op_targets, "synthesized formal<-actual hop missing"
    assert "target_speed" in op_targets, "caller local not reached"
    branches = [v.predicate_raw or "" for v in dag.vertices if v.kind == "branch"]
    assert any("_param_gnd_min" in b for b in branches), \
        "adaptation branch not reached through the argument hop"


def _vt_binding(target: str, file: str, logged: str = "") -> dict:
    return {
        "target_symbol": target,
        "source_symbol": "input_val + 1.0f",
        "function": "C::f",
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
    """Scope rules mirror the walk's visibility conventions: locals are
    ambiguous across file families, members across module directories;
    a declared terminal file selects, including through its header twin."""
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
    assert validate_terminal("_alt", member_same_module, []).status == "valid"

    member_cross_module = [
        _vt_binding("_alt", "src/modules/aaa/alpha.cpp"),
        _vt_binding("_alt", "src/modules/bbb/beta.cpp"),
    ]
    assert validate_terminal("_alt", member_cross_module, []).status == "ambiguous"

    # A member reaches its module's other families through the declared
    # file's directory (inheritance widening), where a local cannot.
    inherited = validate_terminal(
        "_alt", member_cross_module, [], terminal_file="src/modules/aaa/other.cpp"
    )
    assert inherited.status == "valid"
    assert inherited.resolved_file == "src/modules/aaa/alpha.cpp"


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


def test_discovery_rejects_late_cross_module_ambiguity(tmp_path):
    """An unscoped terminal that looks unique in round 0 but gains a
    foreign-module writer from a later gap-search round is genuinely
    ambiguous — the slice built before the evidence arrived is discarded,
    not kept. (A verdict scoped by a declared/resolved file stays locked;
    only the unscoped case re-checks.)"""
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
    assert result.rounds[0].new_files == ["src/modules/aaa/alpha.cpp"]
    assert result.dag is None
    assert result.terminal_validation.status == "ambiguous"


def test_qualifier_scopes_terminal_to_class_family():
    """A ``Class::member`` qualifier is scope evidence, not spelling
    noise: it selects the write file whose family matches the class
    name; an unresolvable qualifier falls through to the unqualified
    rules."""
    from flight_log_agent.analysis.mechanism_discovery import validate_terminal

    cross = [
        _vt_binding("shared_out", "src/modules/aaa/alpha.cpp"),
        _vt_binding("shared_out", "src/modules/bbb/beta.cpp"),
    ]

    qualified = validate_terminal("Alpha::shared_out", cross, [])
    assert qualified.status == "valid"
    assert qualified.resolved_file == "src/modules/aaa/alpha.cpp"
    assert qualified.terminal == "shared_out"

    snake = validate_terminal("MissionBlock::dist", [
        _vt_binding("dist", "src/modules/nav/mission_block.cpp"),
        _vt_binding("dist", "src/modules/other/thing.cpp"),
    ], [])
    assert snake.status == "valid"
    assert snake.resolved_file == "src/modules/nav/mission_block.cpp"

    unknown = validate_terminal("Gamma::shared_out", cross, [])
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
