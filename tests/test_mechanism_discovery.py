from __future__ import annotations

import pytest

from flight_log_agent.analysis.dag_value import DAGValueContext, DAGValueProgram, DAGValueResult
from flight_log_agent.analysis.mechanism_dag import (
    build_mechanism_dag,
    evaluate_dag_vertex_series,
    evaluate_feasibility,
)
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


@pytest.fixture(params=("legacy", "tree_sitter"), ids=("legacy", "tree-sitter"))
def source_backend(request):
    return request.param


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


def _source_contract_dag(tmp_path, backend, source, terminal="output"):
    source_file = "src/modules/example/contract.cpp"
    profiler = _mini_tree(tmp_path, {source_file: source}, backend=backend)
    inputs = dag_inputs_from_facts(load_facts(profiler, tmp_path / "cache", [source_file], "hash"))
    dag = build_mechanism_dag(
        inputs.bindings, terminal, terminal_file=source_file,
        call_statements=inputs.call_statements, helper_expressions=inputs.helper_expressions,
        parameter_bindings=inputs.parameter_bindings, boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )
    return inputs, dag


def _receiver_observation_session(dag, receiver_samples, *, parameters=None):
    """Fixture evidence explicitly records receiver storage, not topic presence.

    This permits a conditional numerical check without claiming that the
    subscription returned success or that its execution history is known.
    """
    program = DAGValueProgram(dag)
    def observe(vertex_id, timestamp):
        vertex = program.vertices[vertex_id]
        samples = receiver_samples.get(vertex.variable, {})
        if vertex.metadata.get("synthetic_boundary_transfer") and timestamp in samples:
            return DAGValueResult("value", value=samples[timestamp],
                                  observed_vertex_ids=frozenset({vertex_id}))
        return None
    return program.bind(parameter_values=parameters, context=DAGValueContext(observation_resolver=observe))


@pytest.mark.parametrize("source_backend", [
    pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason="legacy lacks exact expression-effect reachability; retire after contract parity")),
    "tree_sitter",
])
@pytest.mark.parametrize("expression,status", [
    ("enabled && (output = 7)", "inactive"),
    ("enabled || (output = 7)", "value"),
    ("enabled ? (output = 7) : 0", "inactive"),
    ("enabled ? 0 : (output = 7)", "value"),
    ("enabled || (true && (output = 7))", "value"),
])
def test_expression_effects_use_source_execution_conditions(tmp_path, source_backend, expression, status):
    _, dag = _source_contract_dag(tmp_path, source_backend, """
class Probe { int output; void run() {
    bool enabled = false;
    EFFECT;
} };
""".replace("EFFECT", expression))
    root = next(v for v in dag.vertices if v.variable == "output")
    result = DAGValueProgram(dag).bind().evaluate(root.id, None)
    assert result.status == status, result
    if status == "value":
        assert result.value == 7
    assert any(e.kind == "control" and e.target_id == root.id for e in dag.edges)


@pytest.mark.parametrize("source_backend", [
    pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason="legacy lacks authoritative member declarations for coverage; retire after contract parity")),
    "tree_sitter",
])
def test_resolved_parameter_does_not_discharge_mutable_writer_coverage(tmp_path, source_backend):
    _, dag = _source_contract_dag(tmp_path, source_backend, """
class Probe {
    DEFINE_PARAMETERS((ParamInt<px4::params::SYNTHETIC_MODE>) _setting)
    int mutable_value;
    int output;
    void other() { mutable_value = 2; }
    void run() { output = _setting.get() + mutable_value; }
};
""")
    assert any(v.sub_kind == "parameter" and v.signal_name == "SYNTHETIC_MODE" for v in dag.vertices)
    requests = [r for r in dag.unresolved_references if r.kind == "storage_writers"]
    assert not any(r.symbol == "_setting" for r in requests)
    assert any(r.symbol == "mutable_value" for r in requests)


@pytest.mark.parametrize("source_backend", [
    pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason="legacy lacks exact call-result predicates; retire after contract parity")),
    "tree_sitter",
])
def test_early_return_preserves_helper_and_parameter_operands(tmp_path, source_backend):
    _, dag = _source_contract_dag(tmp_path, source_backend, """
class Probe {
    DEFINE_PARAMETERS((ParamInt<px4::params::SYNTHETIC_MODE>) _setting)
    int output;
    bool ready() { return true; }
    void run() {
        if (_setting.get() != 2 || !ready()) return;
        output = 7;
    }
};
""")
    root = next(v for v in dag.vertices if v.variable == "output")
    program = DAGValueProgram(dag)
    assert program.bind(parameter_values={"SYNTHETIC_MODE": 2}).evaluate(root.id, None).value == 7
    assert program.bind(parameter_values={"SYNTHETIC_MODE": 1}).evaluate(root.id, None).status == "inactive"
    assert any(e.role.startswith("call:") for e in dag.edges)


@pytest.mark.parametrize("source_backend", [
    pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason="legacy lacks exact reference-result projections; retire after contract parity")),
    "tree_sitter",
])
@pytest.mark.parametrize("initializer", ["= getStatus()", "{getStatus()}"])
def test_reference_initialization_projects_nested_call_result(tmp_path, source_backend, initializer):
    _, dag = _source_contract_dag(tmp_path, source_backend, """
struct Control { float rate; };
struct Status { Control control; };
class Probe {
    Status state;
    float output;
    const Status &getStatus() { return state; }
    void run() {
        state.control.rate = 4;
        const Status &alias INITIALIZER;
        output = alias.control.rate;
    }
};
""".replace("INITIALIZER", initializer))
    root = next(v for v in dag.vertices if v.variable == "output")
    result = DAGValueProgram(dag).bind().evaluate(root.id, None)
    assert result.status == "value" and result.value == 4, result
    assert any(e.role == "call-result:control.rate" for e in dag.edges)


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
    assert {
        (item["member"], item["name"])
        for item in inputs.parameter_bindings
    } == {("_param_rtl_return_alt", "RTL_RETURN_ALT")}
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


def test_facts_to_dag_end_to_end(tmp_path, source_backend):
    """The full Stage 1 seam: profiler extraction → SourceFileFacts →
    dag_inputs_from_facts → build_mechanism_dag on a real (mini) source
    tree, asserting the DAG grounds the terminal in its operations,
    branch, and logged evidence."""
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.cpp").write_text(
        """
#include "rtl.h"

void Rtl::pick()
{
    gpos_s gpos_data{};
    orb_copy(ORB_ID(gpos), subscription, &gpos_data);
    if (_param_rtl_type.get() == 1) {
        _rtl_alt = _destination_alt + 10.0f;
    }
    _destination_alt = gpos_data.alt;
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

    profiler = MechanismSourceProfiler(
        tmp_path / "PX4-Autopilot",
        rg_path="missing-rg",
        source_parser_backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/rtl.cpp", "src/modules/example/rtl.h"],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)
    assert all(
        call["name"].rsplit("::", 1)[-1] != "ORB_ID"
        for call in inputs.call_statements
    )
    destination_write = next(
        binding
        for binding in inputs.bindings
        if binding["target_symbol"] == "_destination_alt"
    )
    assert destination_write["target_identity"]["kind"] == "member"
    assert destination_write["target_identity"]["declaration_proven"] is True

    dag = build_mechanism_dag(
        inputs.bindings,
        "_rtl_alt",
        helper_expressions=inputs.helper_expressions,
        parameter_predicates=inputs.parameter_predicates,
        parameter_names=inputs.parameter_names,
        parameter_bindings=inputs.parameter_bindings,
        logged_signals={"gpos.alt"},
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
    assert "gpos.alt" in logged
    assert "ORB_ID" not in dag.unresolved_symbols


@pytest.mark.parametrize(
    "copy_statement",
    [
        "_status_sub.copy(&_status);",
        "orb_copy(ORB_ID(vehicle_status), _status_handle, &_status);",
    ],
)
def test_class_owned_copy_grounds_a_sibling_method(
    tmp_path, copy_statement, source_backend
):
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
    }, backend=source_backend)
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


def test_method_local_copy_does_not_ground_same_named_local_in_sibling_method(
    tmp_path, source_backend
):
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
    }, backend=source_backend)
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


def test_class_owned_payload_publication_is_an_explicit_operation(
    tmp_path, source_backend
):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/publisher.cpp": """
class Publisher {
    uORB::Publication<status_s> _publisher{ORB_ID(status)};
    status_s _status{};

    void calculate(float input)
    {
        _status.value = input;
    }

    void Run()
    {
        _publisher.publish(_status);
    }
};
""",
        },
        backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/publisher.cpp"],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    write = next(
        item
        for item in inputs.bindings
        if item["target_symbol"] == "_status.value"
    )
    assert write["logged_signal"] == ""
    publication = next(
        item
        for item in inputs.bindings
        if item.get("synthetic_boundary_transfer")
        and item.get("boundary_direction") == "publish"
    )
    assert publication["target_symbol"] == "status"
    assert publication["source_symbol"] == "_status"

    dag = build_mechanism_dag(
        inputs.bindings,
        "status.value",
        terminal_file="src/modules/example/publisher.cpp",
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )
    targets = {
        vertex.variable for vertex in dag.vertices if vertex.kind == "operation"
    }
    assert {"status.value", "_status.value"} <= targets


@pytest.mark.parametrize("condition", ["CALL", "false && CALL", "true || CALL", "false && (unknown && CALL)"])
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("api", ["object", "c_api"])
@pytest.mark.parametrize("source_backend", [
    pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason="legacy lacks exact predicate operands; retire after source-to-evaluation parity")),
    "tree_sitter",
])
def test_boundary_call_in_predicate_is_not_a_helper_gap(tmp_path, source_backend, condition, staged, api):
    call_text = "_status_sub.update(&_status)" if api == "object" else "orb_copy(ORB_ID(status), fd, &_status)"
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/reader.cpp": """
struct status_s { float value; };

class Reader {
    uORB::Subscription _status_sub{ORB_ID(status)};
    status_s _status{};
    float output{};

    void Run()
    {
        if (_status_sub.update(&_status)) {
            output = _status.value;
        }
    }
};
""".replace("if (_status_sub.update(&_status))", "if (" + condition.replace("CALL", call_text) + ")"),
        },
        backend=source_backend,
    )
    inputs = dag_inputs_from_facts(
        load_facts(
            profiler,
            tmp_path / "cache",
            ["src/modules/example/reader.cpp"],
            "hash",
        )
    )

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file="src/modules/example/reader.cpp",
        logged_signals={"status.value"},
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
        construction_checkpoint=(lambda _: set()) if staged else None,
    )

    assert any(
        vertex.kind == "operation"
        and (vertex.metadata or {}).get("synthetic_boundary_transfer")
        for vertex in dag.vertices
    )
    assert not any(
        reference.kind == "callable"
        and reference.symbol.rsplit(".", 1)[-1] in {"update", "orb_copy"}
        for reference in dag.unresolved_references
    )

    from flight_log_agent.analysis.dag_value import DAGValueContext, DAGValueResult

    branch = next(v for v in dag.vertices if v.kind == "branch" and call_text in (v.predicate_raw or ""))
    program = DAGValueProgram(dag)
    assert not program.compiled_vertices[branch.id].compile_error
    results = [v for v in dag.vertices if v.sub_kind == "boundary_result"]
    assert len(results) == 1
    assert any(e.source_id == results[0].id and e.target_id == branch.id and e.role.startswith("call:") for e in dag.edges)
    evaluated = program.bind().evaluate(branch.id, 0.)
    if condition == "CALL":
        assert evaluated.status == "unresolved"
        assert any(i.kind == "boundary_result" for i in evaluated.issues)
        for value in (False, True):
            session = program.bind(context=DAGValueContext(observation_resolver=lambda vertex_id, _t: (
                DAGValueResult("value", value=value)
                if program.vertices[vertex_id].sub_kind == "runtime_obligation"
                and program.vertices[vertex_id].metadata.get("requirement") == "boundary_result" else None
            )))
            assert session.evaluate(branch.id, 0.).value is value
    else:
        assert evaluated.status == "value"
        assert evaluated.value is (condition == "true || CALL")
        assert program.bind().evaluate(results[0].id, 0.).status == "inactive"
    # Knowing the returned Boolean does not fabricate a successful payload copy.
    transfer = next(v for v in dag.vertices if v.sub_kind == "boundary_transfer")
    copies = [v for v in dag.vertices if v.metadata.get("synthetic_boundary_transfer")]
    assert copies
    for copy in copies:
        assert copy.metadata.get("boundary_transfer_event_id") == transfer.id
        assert any(e.source_id == transfer.id and e.target_id == copy.id and e.kind == "control" for e in dag.edges)
    if condition == "CALL":
        assert program.bind().evaluate(transfer.id, 0.).status == "unresolved"
    else:
        assert program.bind().evaluate(transfer.id, 0.).value is False


def test_c_api_copy_field_grounds_branch_and_feasibility(
    tmp_path, source_backend
):
    """A copied field used only by a branch remains flight-grounded."""
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/reader.cpp": """
struct topic_a_s { float value; };

class Reader {
    int _subscription{};
    float output{};

    void Run()
    {
        topic_a_s sample{};
        orb_copy(ORB_ID(topic_a), _subscription, &sample);

        if (sample.value > 0.5f) {
            output = 1.0f;
        }
    }
};
""",
        },
        backend=source_backend,
    )
    inputs = dag_inputs_from_facts(
        load_facts(
            profiler,
            tmp_path / "cache",
            ["src/modules/example/reader.cpp"],
            "hash",
        )
    )

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file="src/modules/example/reader.cpp",
        logged_signals={"topic_a.value"},
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )

    branch = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "branch"
        and "sample.value" in str(vertex.predicate_raw or "")
    )
    logged = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal"
        and vertex.signal_name == "topic_a.value"
    )
    transfer = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and vertex.variable == "sample.value"
        and (vertex.metadata or {}).get("synthetic_boundary_transfer")
    )
    assert any(
        edge.source_id == logged.id and edge.target_id == transfer.id
        for edge in dag.edges
    )
    assert any(
        edge.source_id == transfer.id and edge.target_id == branch.id
        for edge in dag.edges
    )
    assert not any(
        vertex.kind == "evidence"
        and vertex.sub_kind == "opaque_symbol"
        and vertex.signal_name == "sample.value"
        for vertex in dag.vertices
    )

    annotated = evaluate_feasibility(
        dag,
        signal_samples={
            "topic_a.value": [(0.0, 0.0), (10.0, 1.0), (20.0, 0.0)]
        },
        signal_policies={"topic_a.value": {"method": "discrete_hold"}},
        prune_dead=False,
    )
    evaluated_branch = next(
        vertex for vertex in annotated.vertices if vertex.id == branch.id
    )
    assert evaluated_branch.feasibility_verdict == "unknown"
    if source_backend == "legacy":
        assert evaluated_branch.active_windows == []
        assert evaluated_branch.metadata["static_evaluation"] == {
            "status": "unresolved",
            "reason": "source expression dependencies are not parser-exact",
        }
        return
    assert evaluated_branch.active_windows == []
    assert evaluated_branch.metadata["evaluation_failures"]
    observed_receiver = evaluate_feasibility(
        dag, signal_samples={"topic_a.value": [(0., 0.), (10., 1.), (20., 0.)]},
        signal_policies={"topic_a.value": {"method": "discrete_hold"}},
        value_session=_receiver_observation_session(dag, {"sample.value": {0.: 0., 10.: 1., 20.: 0.}}),
        prune_dead=False,
    )
    assert next(v for v in observed_receiver.vertices if v.id == branch.id).active_windows == [(10., 20.)]


def test_control_flow_governed_boundary_transfer_stays_grounded(
    tmp_path, source_backend
):
    """A source transfer remains linked, but topic samples do not prove receiver state."""
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/reader.cpp": """
class Reader {
    vehicle_status_s _status{};
    uORB::Subscription _status_sub{ORB_ID(vehicle_status)};
    bool _enabled{};
    float output{};

    void Run()
    {
        if (_enabled) {
            _status_sub.update(&_status);
        }

        if (_status.nav_state > 0.5f) {
            output = 1.0f;
        }
    }
};
""",
        },
        backend=source_backend,
    )
    inputs = dag_inputs_from_facts(
        load_facts(
            profiler,
            tmp_path / "cache",
            ["src/modules/example/reader.cpp"],
            "hash",
        )
    )
    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file="src/modules/example/reader.cpp",
        logged_signals={"vehicle_status.nav_state"},
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )
    branch = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "branch"
        and "nav_state" in str(vertex.predicate_raw or "")
    )
    annotated = evaluate_feasibility(
        dag,
        signal_samples={
            "vehicle_status.nav_state": [(0.0, 0.0), (10.0, 1.0), (20.0, 0.0)]
        },
        signal_policies={"vehicle_status.nav_state": {"method": "discrete_hold"}},
        prune_dead=False,
    )
    evaluated = next(
        vertex for vertex in annotated.vertices if vertex.id == branch.id
    )
    if source_backend == "legacy":
        assert evaluated.active_windows == []
        return
    assert evaluated.active_windows == []
    assert evaluated.feasibility_verdict == "unknown"
    observed_receiver = evaluate_feasibility(
        dag, signal_samples={"vehicle_status.nav_state": [(0., 0.), (10., 1.), (20., 0.)]},
        signal_policies={"vehicle_status.nav_state": {"method": "discrete_hold"}},
        value_session=_receiver_observation_session(dag, {"_status.nav_state": {0.: 0., 10.: 1., 20.: 0.}}),
        prune_dead=False,
    )
    assert next(v for v in observed_receiver.vertices if v.id == branch.id).active_windows == [(10., 20.)]


def test_conditional_publication_operations_remain_explicit(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/publisher.cpp": """
class Publisher {
    uORB::Publication<status_s> _publisher;
    status_s _status{};
    float output{};

public:
    Publisher(bool primary) :
        _publisher(primary ? ORB_ID(status) : ORB_ID(virtual_status))
    {}

    void calculate(float input)
    {
        _status.value = input;
    }

    void Run()
    {
        _publisher.publish(_status);
    }

    void review()
    {
        output = _status.value;
    }
};
""",
        },
        backend="tree_sitter",
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/publisher.cpp"],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    write = next(
        item
        for item in inputs.bindings
        if item["target_symbol"] == "_status.value"
    )
    assert write["logged_signal"] == ""
    publications = [
        item
        for item in inputs.bindings
        if item.get("synthetic_boundary_transfer")
        and item.get("boundary_direction") == "publish"
    ]
    assert {
        (item["target_symbol"], tuple(item["control_predicates"]))
        for item in publications
    } == {
        ("status", ("primary",)),
        ("virtual_status", ("!(primary)",)),
    }

    selected = build_mechanism_dag(
        inputs.bindings,
        "status.value",
        terminal_file="src/modules/example/publisher.cpp",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )
    selected_operations = [
        vertex
        for vertex in selected.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("synthetic_boundary_transfer")
    ]
    assert [vertex.variable for vertex in selected_operations] == ["status.value"]
    assert any(
        branch.kind == "branch" and branch.predicate_raw == "primary"
        for branch in selected.vertices
    )

    alternate = build_mechanism_dag(
        inputs.bindings,
        "virtual_status.value",
        terminal_file="src/modules/example/publisher.cpp",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )
    assert any(
        branch.kind == "branch" and branch.predicate_raw == "!(primary)"
        for branch in alternate.vertices
    )


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


def test_tree_sitter_source_order_selects_prior_same_line_writer(tmp_path):
    source_file = "src/modules/example/order.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: "float output; void run() { float value = 1; output = value; value = 2; }",
        },
        backend="tree_sitter",
    )
    inputs = dag_inputs_from_facts(
        load_facts(profiler, tmp_path / "cache", [source_file], "hash")
    )
    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        source_structure=inputs.structure,
    )
    terminal = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("is_terminal")
    )
    value_writers = sorted(
        (
            vertex
            for vertex in dag.vertices
            if vertex.kind == "operation" and vertex.variable == "value"
        ),
        key=lambda vertex: int(
            ((vertex.metadata or {}).get("target_scope") or {}).get(
                "source_order"
            )
            or 0
        ),
    )

    result = DAGValueProgram(dag).bind().evaluate(terminal.id, None)

    assert len(value_writers) == 1
    assert value_writers[0].expression == "1"
    assert result.status == "value"
    assert result.value == 1.0


def test_source_defined_max_is_not_treated_as_evaluator_intrinsic(tmp_path):
    source_file = "src/modules/example/source_max.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
float max(float left, float right) { return left - right; }
float output;
void run() { output = max(5.0f, 2.0f); }
""",
        },
        backend="tree_sitter",
    )
    inputs = dag_inputs_from_facts(
        load_facts(profiler, tmp_path / "cache", [source_file], "hash")
    )
    call = next(item for item in inputs.call_statements if item["name"] == "max")
    assert call.get("resolved_callable_id")
    assert not call.get("evaluation_intrinsic")

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
    )
    terminal = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("is_terminal")
    )
    result = DAGValueProgram(dag).bind().evaluate(terminal.id, None)

    assert result.status == "value"
    assert result.value == 3.0


def test_no_argument_callee_write_retains_caller_reachability(tmp_path):
    source_file = "src/modules/example/caller_gate.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
class Control {
    float output{};
    void assign() { output = 2.0f; }
    void run(bool enabled) { if (enabled) { assign(); } }
};
""",
        },
        backend="tree_sitter",
    )
    inputs = dag_inputs_from_facts(
        load_facts(profiler, tmp_path / "cache", [source_file], "hash")
    )
    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
    )
    terminal = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("is_terminal")
        and vertex.expression == "2.0"
    )
    controls = [
        edge.source_id
        for edge in dag.edges
        if edge.kind == "control" and edge.target_id == terminal.id
    ]

    assert controls
    assert any(
        vertex.id in controls and vertex.predicate_raw == "enabled"
        for vertex in dag.vertices
        if vertex.kind == "branch"
    )


def test_authoritative_constants_do_not_resolve_by_global_uniqueness(tmp_path):
    source_file = "src/modules/example/constants.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
class First { enum Mode { ACTIVE = 1 }; };
class Second { enum Mode { ACTIVE = 2 }; };
float output;
void run() { output = ACTIVE; }
""",
        },
        backend="tree_sitter",
    )
    inputs = dag_inputs_from_facts(
        load_facts(profiler, tmp_path / "cache", [source_file], "hash")
    )
    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        source_structure=inputs.structure,
    )

    assert not any(
        vertex.kind == "evidence"
        and vertex.sub_kind == "constant"
        and vertex.signal_name == "ACTIVE"
        for vertex in dag.vertices
    )
    assert "ACTIVE" in dag.unresolved_symbols


def test_fixpoint_resolves_symbol_across_files_in_second_round(tmp_path):
    """Round 0 loads only the seeded file and leaves ``_dest_val``
    unresolved; the gap's definition search pulls the second file in
    round 1 and the slice completes — the DAG's own gaps drive discovery."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
#include "rtl.h"

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
#include "rtl.h"

void Rtl::update()
{
    speed_s speed_data{};
    orb_copy(ORB_ID(speed), subscription, &speed_data);
    _dest_val = speed_data.value;
}
""",
    })

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=["pick_altitude"],
        terminal="_final_out",
        source_hash="hash",
        terminal_file="src/modules/example/rtl.cpp",
        logged_signals={"speed.value"},
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
    assert "speed.value" in logged
    assert result.dag.unresolved_symbols == []


def test_frontier_pruning_uses_vertex_origin_not_shared_source_line():
    from flight_log_agent.analysis.mechanism_dag import (
        DAGEdge,
        DAGVertex,
        MechanismDAG,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        _reachable_frontier_references,
    )
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )

    terminal = DAGVertex(
        id="terminal",
        kind="operation",
        variable="output",
        metadata={"is_terminal": True},
    )
    dead = DAGVertex(
        id="dead",
        kind="operation",
        file="same.cpp",
        line=10,
        variable="dead_value",
    )
    live = DAGVertex(
        id="live",
        kind="operation",
        file="same.cpp",
        line=10,
        variable="live_value",
    )
    full = MechanismDAG(
        dag_id="full",
        terminal="output",
        vertices=[terminal, dead, live],
        edges=[
            DAGEdge(
                id="dead-edge",
                source_id="dead",
                target_id="terminal",
                kind="data",
            ),
            DAGEdge(
                id="live-edge",
                source_id="live",
                target_id="terminal",
                kind="data",
            ),
        ],
    )
    feasible = MechanismDAG(
        dag_id="feasible",
        terminal="output",
        vertices=[terminal, live],
        edges=[
            DAGEdge(
                id="live-edge",
                source_id="live",
                target_id="terminal",
                kind="data",
            ),
        ],
    )
    references = [
        UnresolvedSourceReference(
            symbol="dead_gap",
            file="same.cpp",
            line=10,
            origin_vertex_ids=["dead"],
        ),
        UnresolvedSourceReference(
            symbol="live_gap",
            file="same.cpp",
            line=10,
            origin_vertex_ids=["live"],
        ),
        UnresolvedSourceReference(
            symbol="unplaced_gap",
            file="same.cpp",
            line=10,
        ),
    ]

    kept = _reachable_frontier_references(references, full, feasible)

    assert [reference.symbol for reference in kept] == [
        "live_gap",
        "unplaced_gap",
    ]


def test_fixpoint_ignores_legacy_round_budget(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
#include "rtl.h"

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
#include "rtl.h"

void Rtl::update()
{
    speed_s speed_data{};
    orb_copy(ORB_ID(speed), subscription, &speed_data);
    _dest_val = speed_data.value;
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
        logged_signals={"speed.value"},
    )

    assert len(result.rounds) == 2
    assert result.dag.unresolved_symbols == []


@pytest.mark.parametrize("staged", [False, True])
def test_fixpoint_provider_expands_cross_file_helper(tmp_path, staged):
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

    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round

    def checkpoint(dag, index):
        return evaluate_checkpoint_round(
            dag, parameter_values={}, observed_signals=set(), signal_policies={},
            load_samples=lambda *_: {},
        )

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=["pick_altitude"],
        terminal="_alt_out",
        source_hash="hash",
        construction_evaluator=checkpoint if staged else None,
        checkpoint_evaluator=checkpoint if staged else None,
    )

    helper_returns = [v for v in result.dag.vertices
                      if v.provenance and v.provenance.startswith("helper_return")]
    assert helper_returns, "cross-file helper did not expand via provider"
    assert "src/lib/gain/gain.cpp" in result.files_loaded
    fresh = build_mechanism_dag(
        result.inputs.bindings, "_alt_out",
        terminal_file=result.terminal_validation.resolved_file,
        terminal_identity=result.terminal_validation.resolved_identity,
        helper_expressions=result.inputs.helper_expressions,
        call_statements=result.inputs.call_statements,
        boundary_bindings=result.inputs.boundary_bindings,
        source_structure=result.inputs.structure,
    )
    assert {v.id: v.model_dump() for v in result.dag.vertices} == {
        v.id: v.model_dump() for v in fresh.vertices
    }
    assert {e.id: e.model_dump() for e in result.dag.edges} == {
        e.id: e.model_dump() for e in fresh.edges
    }


def test_receiver_getter_reads_state_written_by_prior_call_site(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/control.cpp": """
class State {
public:
    void set(float value) { _value = value; }
    float get() const { return _value; }
private:
    float _value{};
};

class Control {
public:
    void run(float input, float later)
    {
        state.set(input);
        output = state.get();
        state.set(later);
    }
private:
    State state{};
    float output{};
};
""",
        },
        backend="tree_sitter",
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/example/control.cpp"],
        "source-hash",
    )
    inputs = dag_inputs_from_facts(facts)
    output_binding = next(
        binding
        for binding in inputs.bindings
        if binding["target_symbol"] == "output"
        and binding["function"] == "Control::run"
    )
    setter_calls = [
        call for call in inputs.call_statements if call["name"] == "set"
    ]
    assert [call["receiver_access"] for call in setter_calls] == [
        "value",
        "value",
    ]

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
        terminal_file="src/modules/example/control.cpp",
        terminal_identity=output_binding["target_identity"],
    )

    output_vertex = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and vertex.variable == "output"
        and vertex.expression == "state.get()"
    )
    predecessors: dict[str, set[str]] = {}
    for edge in dag.edges:
        if edge.kind == "data":
            predecessors.setdefault(edge.target_id, set()).add(edge.source_id)
    reachable = {output_vertex.id}
    pending = [output_vertex.id]
    while pending:
        current = pending.pop()
        for predecessor in predecessors.get(current, set()):
            if predecessor not in reachable:
                reachable.add(predecessor)
                pending.append(predecessor)

    assert any(
        vertex.id in reachable
        and vertex.kind == "operation"
        and vertex.variable == "state._value"
        and vertex.expression == "_value"
        for vertex in dag.vertices
    )
    member_output_sites = {
        str((vertex.metadata or {}).get("call_site_id") or "")
        for vertex in dag.vertices
        if (vertex.metadata or {}).get("synthetic_member_output_binding")
    }
    assert member_output_sites == {setter_calls[0]["source_site_id"]}
    assert any(
        vertex.id in reachable
        and vertex.kind == "operation"
        and vertex.variable == "_value"
        and vertex.expression == "value"
        for vertex in dag.vertices
    )


def test_fixpoint_does_not_expand_reference_from_proven_dead_branch(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/control.cpp": """
extern float missing_value;
class Control {
    float output;
    void run()
    {
        if (false) {
            output = missing_value;
        } else {
            output = 1.0f;
        }
    }
};
""",
            "src/modules/example/missing.cpp": "float missing_value = 4.0f;",
        },
        backend="tree_sitter",
    )

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        [],
        "output",
        "source-hash",
        terminal_file="src/modules/example/control.cpp",
    )

    assert result.dag is not None
    assert "src/modules/example/missing.cpp" not in result.files_loaded


def test_fixpoint_keeps_reference_from_unknown_branch(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/control.cpp": """
extern float missing_value;
class Control {
    float output;
    void run()
    {
        if (external_gate) {
            output = missing_value;
        } else {
            output = 1.0f;
        }
    }
};
""",
            "src/modules/example/missing.cpp": "float missing_value = 4.0f;",
        },
        backend="tree_sitter",
    )

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        [],
        "output",
        "source-hash",
        terminal_file="src/modules/example/control.cpp",
    )

    assert result.dag is not None
    assert "src/modules/example/missing.cpp" in result.files_loaded


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
        "triplet.current.alt",
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
        == exact_symbol("triplet.current.alt")
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


def test_expression_dependencies_are_backend_interchangeable_in_dag(
    tmp_path, source_backend
):
    source_file = "src/modules/example/dependencies.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
static constexpr float B = 2.0f;

void Example::update()
{
    vehicle_state_s vehicle_state{};
    orb_copy(ORB_ID(vehicle_state), subscription, &vehicle_state);
    float var = B;
    var *= std::max(static_cast<float>(B), vehicle_state.mut);
    output = var;
}
""",
        },
        backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        [source_file],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        logged_signals={"vehicle_state.mut"},
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
    )

    var_operations = [
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation" and vertex.variable == "var"
    ]
    initial = next(
        vertex
        for vertex in var_operations
        if (vertex.metadata or {}).get("assignment_operator") == "="
    )
    compound = next(
        vertex
        for vertex in var_operations
        if (vertex.metadata or {}).get("assignment_operator") == "*="
    )
    assert any(
        edge.source_id == initial.id
        and edge.target_id == compound.id
        and edge.kind == "data"
        and edge.role == "var"
        for edge in dag.edges
    )
    assert any(
        vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal"
        and vertex.signal_name == "vehicle_state.mut"
        for vertex in dag.vertices
    )
    assert any(
        vertex.kind == "evidence"
        and vertex.sub_kind == "constant"
        and vertex.signal_name == "B"
        and (vertex.metadata or {}).get("value") == 2.0
        for vertex in dag.vertices
    )
    assert not {
        "max",
        "static_cast",
        "float",
        "B",
        "var",
        "vehicle_state.mut",
    } & set(dag.unresolved_symbols)


def test_helper_call_instance_reaches_member_writer_in_sibling_method(
    tmp_path, source_backend
):
    source_file = "src/modules/example/member_call.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
class Control {
    float state;
    float output;

    float read_state() const
    {
        return state;
    }

    void update(float input)
    {
        state = input;
    }

    void run()
    {
        output = read_state();
    }
};
""",
        },
        backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        [source_file],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
    )

    assert any(
        vertex.kind == "operation" and vertex.variable == "state"
        for vertex in dag.vertices
    )
    assert "state" not in dag.unresolved_symbols


def test_terminal_formal_rebinding_crosses_nested_caller_paths(
    tmp_path, source_backend
):
    source_file = "src/modules/example/nested_calls.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
struct point_s { float value; };
struct triplet_s { point_s current; };

class Control {
    uORB::Subscription _triplet_sub{ORB_ID(triplet)};
    triplet_s _triplet{};

    void poll()
    {
        _triplet_sub.copy(&_triplet);
    }

    void inner(const point_s &setpoint)
    {
        float target = setpoint.value;
        consume(target);
    }

    void outer(const point_s &incoming)
    {
        const point_s &forwarded = incoming;
        inner(forwarded);
    }

    void Run()
    {
        outer(_triplet.current);
    }
};
""",
        },
        backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        [source_file],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)
    inner = next(
        item
        for item in inputs.structure.callables_by_id.values()
        if item["name"] == "Control::inner"
    )
    terminal_identity = inputs.structure.symbol_identity(
        "target",
        file=source_file,
        callable_id=inner["callable_id"],
        function_name=inner["name"],
        function_parameters=inner["parameters"],
    )

    dag = build_mechanism_dag(
        inputs.bindings,
        "target",
        terminal_file=source_file,
        terminal_identity=terminal_identity,
        logged_signals={"triplet.current.value"},
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )

    assert any(
        vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal"
        and vertex.signal_name == "triplet.current.value"
        for vertex in dag.vertices
    )
    assert not {
        "setpoint.value",
        "forwarded.value",
        "incoming.value",
        "_triplet.current.value",
    } & set(dag.unresolved_symbols)


def test_terminal_formal_field_rebinding_crosses_gated_value_copy(
    tmp_path, source_backend
):
    """Call gates control reachability, not formal-to-actual identity."""
    source_file = "src/modules/example/gated_calls.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
struct point_s { float latitude; float speed; };
struct triplet_s { point_s current; };

class Control {
    uORB::Subscription _triplet_sub{ORB_ID(triplet)};
    triplet_s _triplet{};

    void poll() { _triplet_sub.copy(&_triplet); }
    void adjust(point_s &setpoint) { setpoint.latitude = 1.0f; }

    void inner(const point_s &setpoint)
    {
        float target = setpoint.speed;
        consume(target);
    }

    void outer(const point_s &incoming)
    {
        point_s current = incoming;
        adjust(current);

        if (enabled) {
            inner(current);
        }
    }

    void Run()
    {
        if (automatic) {
            outer(_triplet.current);
        }
    }
};
""",
        },
        backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        [source_file],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)
    inner = next(
        item
        for item in inputs.structure.callables_by_id.values()
        if item["name"] == "Control::inner"
    )
    terminal_identity = inputs.structure.symbol_identity(
        "target",
        file=source_file,
        callable_id=inner["callable_id"],
        function_name=inner["name"],
        function_parameters=inner["parameters"],
    )

    dag = build_mechanism_dag(
        inputs.bindings,
        "target",
        terminal_file=source_file,
        terminal_identity=terminal_identity,
        logged_signals={"triplet.current.speed"},
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )

    assert any(
        vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal"
        and vertex.signal_name == "triplet.current.speed"
        for vertex in dag.vertices
    )
    assert not {
        "setpoint.speed",
        "current.speed",
        "incoming.speed",
        "_triplet.current.speed",
    } & set(dag.unresolved_symbols)
    gated_terminal = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("is_terminal")
    )
    assert any(
        edge.kind == "control" and edge.target_id == gated_terminal.id
        for edge in dag.edges
    )


def test_plain_self_read_assignment_reaches_caller_actual(
    tmp_path, source_backend
):
    source_file = "src/modules/example/self_read_call.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
class Control {
    float source;

    float adapt(float value)
    {
        return value * 2.0f;
    }

    void consume(float floor, float &setpoint)
    {
        setpoint = max(setpoint, floor);
    }

    void Run()
    {
        float candidate = adapt(source);
        consume(1.0f, candidate);
    }
};
""",
        },
        backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        [source_file],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)
    consume = next(
        item
        for item in inputs.structure.callables_by_id.values()
        if item["name"] == "Control::consume"
    )
    terminal_identity = inputs.structure.symbol_identity(
        "setpoint",
        file=source_file,
        callable_id=consume["callable_id"],
        function_name=consume["name"],
        function_parameters=consume["parameters"],
    )

    dag = build_mechanism_dag(
        inputs.bindings,
        "setpoint",
        terminal_file=source_file,
        terminal_identity=terminal_identity,
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
    )

    operations = [
        vertex for vertex in dag.vertices if vertex.kind == "operation"
    ]
    terminal = next(
        vertex
        for vertex in operations
        if vertex.variable == "setpoint"
        and vertex.expression == "max(setpoint, floor)"
    )
    formal_bindings = [
        vertex
        for vertex in operations
        if vertex.variable == "setpoint"
        and vertex.expression == "candidate"
        and (vertex.metadata or {}).get("synthetic_call_binding")
    ]
    assert formal_bindings, [
        (
            vertex.variable,
            vertex.expression,
            (vertex.metadata or {}).get("target_scope"),
            (vertex.metadata or {}).get("site_scope"),
        )
        for vertex in operations
    ]
    formal_binding = formal_bindings[0]
    candidate = next(
        vertex
        for vertex in operations
        if vertex.variable == "candidate"
        and vertex.expression == "adapt(source)"
    )
    assert any(
        edge.source_id == formal_binding.id
        and edge.target_id == terminal.id
        and edge.role == "setpoint"
        for edge in dag.edges
    )
    assert any(
        edge.source_id == candidate.id
        and edge.target_id == formal_binding.id
        and edge.role == "candidate"
        for edge in dag.edges
    )


@pytest.mark.parametrize("staged", [False, True])
def test_nested_helper_return_dataflow_is_backend_interchangeable(
    tmp_path, source_backend, staged
):
    source_file = "src/modules/example/nested_returns.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
float inner(float value)
{
    return value * 2.0f;
}

float outer(float value)
{
    return inner(value);
}

void run()
{
    output = outer(input);
}
""",
        },
        backend=source_backend,
    )
    inputs = dag_inputs_from_facts(
        load_facts(
            profiler,
            tmp_path / "cache",
            [source_file],
            "hash",
        )
    )

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
        construction_checkpoint=(lambda _: set()) if staged else None,
    )

    outer_returns = [
        vertex
        for vertex in dag.vertices
        if str(vertex.provenance or "").startswith("helper_return:outer@")
    ]
    inner_returns = [
        vertex
        for vertex in dag.vertices
        if str(vertex.provenance or "").startswith("helper_return:inner@")
    ]
    assert len(outer_returns) == 1
    assert len(inner_returns) == 1
    assert any(
        edge.source_id == inner_returns[0].id
        and edge.target_id == outer_returns[0].id
        and edge.role == "call:inner"
        for edge in dag.edges
    )


@pytest.mark.parametrize("staged", [False, True])
def test_source_grounded_helper_condition_is_backend_interchangeable(
    tmp_path, source_backend, staged
):
    """Source-resolved calls, parameters, enums, and logged gates compose.

    Both parser backends must carry the same source mechanism through the
    shared DAG consumer even though they represent helper internals
    differently.
    """
    source_file = "src/modules/example/acceptance.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
struct status_s {
    int vehicle_type;
};

struct controller_status_s {
    bool valid;
    float radius;
};

class Control {
    uORB::Subscription _vehicle_status_sub{ORB_ID(status)};
    uORB::Subscription _destination_sub{ORB_ID(destination)};
    status_s vehicle_status{};
    destination_s destination_state{};
    controller_status_s controller_status{};
    float floor{};

    DEFINE_PARAMETERS(
        (ParamFloat<px4::params::ACCEPT_RADIUS>) _param_accept_radius
    )

    float default_radius();
    float acceptance_radius();
    void poll();
    void run();
};

void Control::poll()
{
    _vehicle_status_sub.copy(&vehicle_status);
    _destination_sub.copy(&destination_state);
}

float Control::default_radius()
{
    return _param_accept_radius.get();
}

float Control::acceptance_radius()
{
    float selected = default_radius();

    if (vehicle_status.vehicle_type != status_s::ROTARY
        && controller_status.valid) {
        selected = std::max(selected, controller_status.radius);
    }

    return selected;
}

void Control::run()
{
    floor = destination_state.value + 2.0f * acceptance_radius();
}
""",
        },
        backend=source_backend,
    )
    inputs = dag_inputs_from_facts(
        load_facts(profiler, tmp_path / "cache", [source_file], "hash")
    )

    dag = build_mechanism_dag(
        inputs.bindings,
        "floor",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        parameter_names=inputs.parameter_names,
        parameter_bindings=inputs.parameter_bindings,
        logged_signals={"destination.value", "status.vehicle_type"},
        enum_registry={"status": {"ROTARY": 1}},
        source_structure=inputs.structure,
        construction_checkpoint=(lambda _: set()) if staged else None,
    )

    receiver_samples = {"destination_state.value": {0.: 100., 1.: 100.},
                        "vehicle_status.vehicle_type": {0.: 1, 1.: 1}}
    annotated = evaluate_feasibility(
        dag,
        parameter_values={"ACCEPT_RADIUS": 10.0},
        value_session=_receiver_observation_session(dag, receiver_samples, parameters={"ACCEPT_RADIUS": 10.}),
        signal_samples={
            "destination.value": [(0.0, 100.0), (1.0, 100.0)],
            "status.vehicle_type": [(0.0, 1), (1.0, 1)],
        },
        signal_policies={
            "destination.value": {"method": "linear"},
            "status.vehicle_type": {"method": "discrete_hold"},
        },
        prune_dead=False,
    )

    override_branches = [
        vertex
        for vertex in annotated.vertices
        if vertex.kind == "branch"
        and "vehicle_status" in str(vertex.predicate_raw or "")
    ]
    if source_backend == "tree_sitter":
        # Replacement-only topology assertion: the legacy extractor embeds
        # this selection in its flattened helper expression. The shared
        # numerical contract below remains identical for both backends.
        assert override_branches
        assert all(
            vertex.feasibility_verdict == "always_false"
            for vertex in override_branches
        )

    terminal = next(
        vertex
        for vertex in annotated.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("is_terminal")
        and "acceptance_radius" in str(vertex.expression or "")
    )
    reconstructed = evaluate_dag_vertex_series(
        annotated,
        terminal.id,
        parameter_values={"ACCEPT_RADIUS": 10.0},
        value_session=_receiver_observation_session(annotated, receiver_samples, parameters={"ACCEPT_RADIUS": 10.}),
        signal_samples={
            "destination.value": [(0.0, 100.0), (1.0, 100.0)],
            "status.vehicle_type": [(0.0, 1), (1.0, 1)],
        },
        signal_policies={
            "destination.value": {"method": "linear"},
            "status.vehicle_type": {"method": "discrete_hold"},
        },
        timestamps=[0.0, 1.0],
        evaluation_windows=[(0.0, 1.0)],
    )
    if source_backend == "legacy":
        # The shared consumer now rejects non-parser-exact dependencies. This
        # pins the explicit retirement gap instead of letting the legacy
        # backend numerically guess through its flattened helper expression.
        assert reconstructed.complete is False
        assert "not parser-exact" in reconstructed.reason
        return
    assert reconstructed.complete is True
    assert set(reconstructed.referenced_signals) == {
        "destination.value",
        "status.vehicle_type",
    }
    assert [value for _timestamp, value in reconstructed.samples] == [120.0, 120.0]


def test_tree_sitter_exact_callable_survives_without_owner_rederivation(tmp_path):
    source_file = "src/modules/example/exact_call.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
class Service {
public:
    float read() { return input; }
    float input{};
};

class Control {
    Service service{};
    float output{};
    void run() { output = service.read(); }
};
""",
        },
        backend="tree_sitter",
    )
    inputs = dag_inputs_from_facts(
        load_facts(profiler, tmp_path / "cache", [source_file], "hash")
    )
    call = next(item for item in inputs.call_statements if item["name"] == "read")
    assert call.get("resolved_callable_id")

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        logged_signals={"input"},
        # The exact callable carried by the call fact is authoritative. The
        # graph must not need the receiver owner's declaration a second time.
        source_structure=SourceStructureIndex(),
    )

    helper_return = next(
        vertex
        for vertex in dag.vertices
        if str(vertex.provenance or "").startswith("helper_return:read@")
    )
    terminal = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("is_terminal")
        and "service.read" in str(vertex.expression or "")
    )
    assert any(
        edge.source_id == helper_return.id
        and edge.target_id == terminal.id
        and edge.role == "call:read"
        for edge in dag.edges
    )


def test_terminal_formal_rebinding_preserves_diamond_caller_paths(
    tmp_path, source_backend
):
    source_file = "src/modules/example/diamond_calls.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
struct point_s { float value; };
struct pair_s { point_s first; point_s second; };

class Control {
    uORB::Subscription _pair_sub{ORB_ID(pair)};
    pair_s _pair{};

    void poll() { _pair_sub.copy(&_pair); }
    void inner(const point_s &setpoint) { float target = setpoint.value; }
    void left(const point_s &value) { inner(value); }
    void right(const point_s &value) { inner(value); }
    void Run()
    {
        left(_pair.first);
        right(_pair.second);
    }
};
""",
        },
        backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        [source_file],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)
    inner = next(
        item
        for item in inputs.structure.callables_by_id.values()
        if item["name"] == "Control::inner"
    )
    terminal_identity = inputs.structure.symbol_identity(
        "target",
        file=source_file,
        callable_id=inner["callable_id"],
        function_name=inner["name"],
        function_parameters=inner["parameters"],
    )

    dag = build_mechanism_dag(
        inputs.bindings,
        "target",
        terminal_file=source_file,
        terminal_identity=terminal_identity,
        logged_signals={"pair.first.value", "pair.second.value"},
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )

    observed = {
        vertex.signal_name
        for vertex in dag.vertices
        if vertex.kind == "evidence" and vertex.sub_kind == "logged_signal"
    }
    assert {"pair.first.value", "pair.second.value"} <= observed


def test_tree_sitter_default_argument_expands_and_binds_in_dag(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
                "src/modules/caller.cpp": """
#include <src/lib/numeric.hpp>

float input;
float output;
void run()
{
    output = adjust(input);
}
""",
            "src/lib/numeric.hpp": """
float adjust(float value, bool enabled = false);
""",
                "src/lib/numeric.cpp": """
#include "numeric.hpp"

float adjust(float value, bool enabled)
{
    return enabled ? value * 2.0f : value;
}
""",
        },
        backend="tree_sitter",
    )
    caller_facts = load_facts(
        profiler,
        tmp_path / "cache",
        ["src/modules/caller.cpp"],
        "hash",
    )
    caller_inputs = dag_inputs_from_facts(caller_facts)
    reference = UnresolvedSourceReference(
        symbol="adjust",
        kind="callable",
        file="src/modules/caller.cpp",
        callable_id=next(
            item.callable_id
            for item in caller_facts[0].callables
            if item.name == "run"
        ),
        argument_count=1,
    )

    candidates = SourceExpansionResolver(profiler, "hash").resolve(
        reference,
        caller_inputs.structure,
    )

    assert [item.file for item in candidates] == ["src/lib/numeric.cpp"]
    helper = next(
        item
        for item in candidates[0].facts.helper_expressions
        if item.name == "adjust"
    )
    assert helper.parameter_defaults == [None, "false"]

    inputs = dag_inputs_from_facts([*caller_facts, candidates[0].facts])
    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file="src/modules/caller.cpp",
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        source_structure=inputs.structure,
    )

    assert "adjust" not in dag.unresolved_symbols
    assert any(
        vertex.kind == "operation"
        and vertex.variable == "enabled"
        and vertex.expression == "false"
        for vertex in dag.vertices
    )


@pytest.mark.parametrize("initializer", ["= *_navigator->get_global_position()", "{*_navigator->get_global_position()}"])
@pytest.mark.parametrize("source_backend", [
    pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason="legacy lacks call-result reference projection; retire after contract parity")),
    "tree_sitter",
])
def test_dereferenced_reference_alias_reaches_source_proven_log_boundary(tmp_path, source_backend, initializer):
    source_file = "src/modules/mode/mode.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
struct vehicle_global_position_s { float alt; };

class Navigator {
public:
    void update();
    vehicle_global_position_s *get_global_position();
private:
    uORB::Subscription _global_position_sub{ORB_ID(vehicle_global_position)};
    vehicle_global_position_s _global_position{};
};

void Navigator::update()
{
    _global_position_sub.copy(&_global_position);
}

vehicle_global_position_s *Navigator::get_global_position()
{
    return &_global_position;
}

class Mode {
public:
    void run();
private:
    Navigator *_navigator;
    float output;
};

void Mode::run()
{
    const vehicle_global_position_s &global_position = *_navigator->get_global_position();
    output = global_position.alt;
}
""".replace("= *_navigator->get_global_position()", initializer),
        },
        backend=source_backend,
    )
    facts = load_facts(
        profiler,
        tmp_path / "cache",
        [source_file],
        "hash",
    )
    inputs = dag_inputs_from_facts(facts)

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        inventory={
            "topic_fields": {"vehicle_global_position": ["alt"]},
            "available_topics": ["vehicle_global_position"],
        },
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )

    assert any(
        vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal"
        and vertex.signal_name == "vehicle_global_position.alt"
        for vertex in dag.vertices
    )
    assert "global_position.alt" not in dag.unresolved_symbols


@pytest.mark.parametrize("staged", [False, True])
def test_helper_return_consumes_aliased_call_result_without_flattening(tmp_path, staged):
    source_file = "src/modules/mode/mode.cpp"
    profiler = _mini_tree(
        tmp_path,
        {
            source_file: """
struct vehicle_global_position_s { float alt; };

class Navigator {
public:
    void update();
    const vehicle_global_position_s *get_global_position() const;
private:
    uORB::Subscription _global_position_sub{ORB_ID(vehicle_global_position)};
    vehicle_global_position_s _global_position{};
};

void Navigator::update()
{
    _global_position_sub.copy(&_global_position);
}

const vehicle_global_position_s *Navigator::get_global_position() const
{
    return &_global_position;
}

class Mode {
public:
    float calculate() const;
    void run();
private:
    Navigator *_navigator;
    float output;
};

float Mode::calculate() const
{
    const vehicle_global_position_s &gpos =
        *_navigator->get_global_position();
    const float floor = 100.0f;
    return std::max(floor, gpos.alt);
}

void Mode::run()
{
    output = calculate();
}
""",
        },
        backend="tree_sitter",
    )
    facts = load_facts(profiler, tmp_path / "cache", [source_file], "hash")
    inputs = dag_inputs_from_facts(facts)

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file=source_file,
        inventory={
            "topic_fields": {"vehicle_global_position": ["alt"]},
            "available_topics": ["vehicle_global_position"],
        },
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
        construction_checkpoint=(lambda _: set()) if staged else None,
    )

    helper_return = next(
        vertex
        for vertex in dag.vertices
        if str(vertex.provenance or "").startswith("helper_return:calculate@")
    )
    assert helper_return.expression == "max(floor, gpos.alt)"
    assert not any(
        str(vertex.provenance or "").startswith("helper_body")
        for vertex in dag.vertices
    )
    logged_alt = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal"
        and vertex.signal_name == "vehicle_global_position.alt"
    )
    getter_projection = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and vertex.variable == "__return__.alt"
        and str(vertex.provenance or "").startswith(
            "helper_return:get_global_position@"
        )
    )
    boundary_transfer = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and vertex.variable == "_global_position.alt"
        and (vertex.metadata or {}).get("synthetic_boundary_transfer")
        and (vertex.metadata or {}).get("boundary_direction") == "subscribe"
    )
    alias_projection = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation" and vertex.variable == "gpos.alt"
    )
    assert any(
        edge.source_id == logged_alt.id
        and edge.target_id == boundary_transfer.id
        and edge.kind == "data"
        for edge in dag.edges
    )
    assert any(
        edge.source_id == boundary_transfer.id
        and edge.target_id == getter_projection.id
        and edge.kind == "data"
        for edge in dag.edges
    )
    assert any(
        edge.source_id == getter_projection.id
        and edge.target_id == alias_projection.id
        and edge.kind == "data"
        for edge in dag.edges
    )
    assert any(
        edge.source_id == alias_projection.id
        and edge.target_id == helper_return.id
        and edge.kind == "data"
        for edge in dag.edges
    )
    assert not any(
        edge.source_id == logged_alt.id
        and edge.target_id == helper_return.id
        for edge in dag.edges
    )
    assert "gpos.alt" not in dag.unresolved_symbols
    terminal = next(
        vertex
        for vertex in dag.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("is_terminal")
    )
    reconstructed = evaluate_dag_vertex_series(
        dag,
        terminal.id,
        signal_samples={
            "vehicle_global_position.alt": [(0.0, 125.0)],
        },
        signal_policies={
            "vehicle_global_position.alt": {"method": "linear"},
        },
        timestamps=[0.0],
        evaluation_windows=[(0.0, 0.0)],
    )
    assert reconstructed.complete is False
    reconstructed = evaluate_dag_vertex_series(
        dag, terminal.id,
        signal_samples={"vehicle_global_position.alt": [(0., 125.)]},
        signal_policies={"vehicle_global_position.alt": {"method": "linear"}},
        timestamps=[0.], evaluation_windows=[(0., 0.)],
        value_session=_receiver_observation_session(dag, {"_global_position.alt": {0.: 125.}}),
    )
    assert reconstructed.complete is True
    assert reconstructed.samples == ((0.0, 125.0),)


def test_split_file_call_result_discovers_class_owned_runtime_writer(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/mode/navigator.h": """
struct position_s { float alt; };
class Navigator {
public:
    position_s *get_position() { return &_position; }
    void update();
private:
    uORB::Subscription _position_sub{ORB_ID(position)};
    position_s _position{};
};
""",
            "src/modules/mode/mode.cpp": """
#include "navigator.h"
class Mode {
    Navigator *_navigator;
    float output;
    void run();
};
void Mode::run()
{
    const position_s &position = *_navigator->get_position();
    output = position.alt;
}
""",
            "src/modules/mode/navigator_main.cpp": """
#include "navigator.h"
void Navigator::update()
{
    _position_sub.copy(&_position);
}
""",
            "src/modules/other/other.cpp": """
class Other {
    float _position;
    void update() { _position = 42.0f; }
};
""",
        },
        backend="tree_sitter",
    )
    queries: list[str] = []
    original_search = profiler.search_related_source_files

    def record_search(values, *args, **kwargs):
        queries.extend([values] if isinstance(values, str) else values)
        return original_search(values, *args, **kwargs)

    profiler.search_related_source_files = record_search  # type: ignore[assignment]

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        [],
        "output",
        "hash",
        terminal_file="src/modules/mode/mode.cpp",
        inventory={
            "topic_fields": {"position": ["alt"]},
            "available_topics": ["position"],
        },
        logged_signals={"position.alt"},
    )

    assert result.dag is not None
    assert "src/modules/mode/navigator.h" in result.files_loaded
    assert "src/modules/mode/navigator_main.cpp" in result.files_loaded
    assert "src/modules/other/other.cpp" not in result.files_loaded
    assert "alt" not in queries
    assert any(
        vertex.kind == "evidence"
        and vertex.sub_kind == "logged_signal"
        and vertex.signal_name == "position.alt"
        for vertex in result.dag.vertices
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


def test_rejected_admission_facts_are_not_retained(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/candidate.cpp": """
void Candidate::run()
{
    consume(output);
}
""",
        },
        backend="tree_sitter",
    )
    resolver = SourceExpansionResolver(profiler, "hash")

    candidates = resolver.resolve(
        UnresolvedSourceReference(symbol="output"),
        SourceStructureIndex(),
    )

    assert candidates == []
    assert resolver._facts == {}
    assert not hasattr(resolver, "_candidate_facts")


def test_candidate_file_builds_one_compact_admission_index_per_run(
    tmp_path, monkeypatch
):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/candidate.cpp": """
void Candidate::run()
{
    output = input;
}
""",
        },
        backend="tree_sitter",
    )
    from flight_log_agent.px4.tree_sitter_source import TreeSitterSourceExtractor

    calls = 0
    original = TreeSitterSourceExtractor.extract_admission

    def count_admission(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        TreeSitterSourceExtractor,
        "extract_admission",
        count_admission,
    )
    resolver = SourceExpansionResolver(profiler, "hash")
    structure = SourceStructureIndex()

    assert resolver.resolve(
        UnresolvedSourceReference(symbol="output"), structure
    )
    assert resolver.resolve(
        UnresolvedSourceReference(symbol="input"), structure
    ) == []
    assert calls == 1


def _vt_binding(
    target: str,
    file: str,
    logged: str = "",
    function: str = "C::f",
    published: bool = False,
) -> dict:
    return {
        "target_symbol": target,
        "source_symbol": "input_val + 1.0f",
        "function": function,
        "assignment_path": [{"file": file, "line": 1, "expression": "input_val + 1.0f"}],
        "logged_signal": logged,
        "external_target_signal": published,
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
    assert observed.status == "absent" and observed.logged is True
    assert "no source writer" in observed.reason

    unique_instance = validate_terminal(
        "topic_a.alt",
        [_vt_binding("topic_a", "src/modules/aaa/publisher.cpp", published=True)],
        ["topic_a[0].alt"],
    )
    assert unique_instance.status == "valid"
    assert unique_instance.logged is True

    ambiguous_instance = validate_terminal(
        "topic_a.alt",
        [_vt_binding("topic_a", "src/modules/aaa/publisher.cpp", published=True)],
        ["topic_a[0].alt", "topic_a[1].alt"],
    )
    assert ambiguous_instance.status == "ambiguous"
    assert "multiple observed placements" in ambiguous_instance.reason

    ambiguous_publishers = validate_terminal(
        "topic_a.alt",
        [
            _vt_binding(
                "topic_a", "src/modules/aaa/publisher.cpp", published=True
            ),
            _vt_binding(
                "topic_a", "src/modules/bbb/publisher.cpp", published=True
            ),
        ],
        ["topic_a.alt"],
    )
    assert ambiguous_publishers.status == "ambiguous"


def test_discovery_finds_logged_terminal_through_publish_operation(
    tmp_path, source_backend
):
    """ULog membership locates candidates but never replaces source flow."""
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/publisher.cpp": """
struct status_s { float value; };

class Publisher {
    uORB::Publication<status_s> _status_pub{ORB_ID(status)};
    status_s _status{};
    void run(float input);
};

void Publisher::run(float input)
{
    _status.value = input;
    _status_pub.publish(_status);
}
""",
        },
        backend=source_backend,
    )

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=[],
        terminal="status.value",
        source_hash="hash",
        logged_signals={"status.value"},
    )

    assert result.dag is not None
    assert result.terminal_validation.status == "valid"
    transfers = [
        vertex
        for vertex in result.dag.vertices
        if vertex.kind == "operation"
        and (vertex.metadata or {}).get("synthetic_boundary_transfer")
    ]
    assert transfers
    terminal_transfer = next(
        vertex for vertex in transfers if vertex.variable == "status.value"
    )
    status_write = next(
        vertex
        for vertex in result.dag.vertices
        if vertex.kind == "operation"
        and vertex.variable == "_status.value"
        and vertex.expression == "input"
    )
    assert any(
        edge.source_id == status_write.id
        and edge.target_id == terminal_transfer.id
        and edge.kind == "data"
        for edge in result.dag.edges
    )


def test_discovery_rejects_ambiguous_logged_publishers(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    files = {
        f"src/modules/{owner.lower()}/publisher.cpp": f"""
struct status_s {{ float value; }};
class {owner} {{
    uORB::Publication<status_s> pub{{ORB_ID(status)}};
    status_s payload{{}};
    void run(float input) {{ payload.value = input; pub.publish(payload); }}
}};
"""
        for owner in ("Alpha", "Beta")
    }
    profiler = _mini_tree(tmp_path, files, backend="tree_sitter")

    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=[],
        terminal="status.value",
        source_hash="hash",
        logged_signals={"status.value"},
    )

    assert result.dag is None
    assert result.terminal_validation.status == "ambiguous"


def test_validate_terminal_scoping_rules():
    """Without ownership metadata, same names in different files remain
    ambiguous; an explicit file or source-proven include can scope it."""
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
    assert twin.status == "absent_in_scope"

    twin = validate_terminal(
        "dist",
        cross,
        [],
        terminal_file="src/modules/aaa/alpha.hpp",
        source_structure=SourceStructureIndex(
            includes={
                "src/modules/aaa/alpha.cpp": {
                    "src/modules/aaa/alpha.hpp"
                }
            }
        ),
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


def test_typed_receiver_expansion_searches_qualified_owner_only(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/services/widget_impl.cpp": """
float Widget::update(float value)
{
    return value + 1.0f;
}
""",
            "src/unrelated/other.cpp": """
float Other::update(float value)
{
    return value - 1.0f;
}
""",
        },
        backend="tree_sitter",
    )
    structure = SourceStructureIndex(
        direct_bases={"Caller": set(), "Widget": set(), "Other": set()},
        declared_classes={"Caller", "Widget", "Other"},
        members={
            ("Caller", "_service"): {
                "name": "_service",
                "owner": "Caller",
                "type": "Widget",
            }
        },
    )
    reference = UnresolvedSourceReference(
        symbol="update",
        kind="callable",
        file="src/controllers/caller.cpp",
        callable_id="src/controllers/caller.cpp:1:Caller::run:",
        class_owner="Caller",
        receiver="_service",
        argument_count=1,
    )
    queries: list[str] = []
    original_search = profiler.search_related_source_files

    def record_search(values, *args, **kwargs):
        queries.extend([values] if isinstance(values, str) else values)
        return original_search(values, *args, **kwargs)

    profiler.search_related_source_files = record_search  # type: ignore[assignment]
    candidates = SourceExpansionResolver(profiler, "hash").resolve(
        reference, structure
    )

    assert [item.file for item in candidates] == ["src/services/widget_impl.cpp"]
    assert "Widget::update(" in queries
    assert "update(" not in queries


def test_inherited_inline_callable_resolves_without_class_named_filename(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "platform/common/runtime_contract.hpp": """
template<typename T>
class BaseControl
{
public:
    bool halted() const { return stop_requested; }
    bool stop_requested{false};
};
""",
            "src/controllers/derived.hpp": """
class DerivedControl : public BaseControl<DerivedControl> {};
""",
        },
        backend="tree_sitter",
    )
    structure = SourceStructureIndex(
        direct_bases={
            "DerivedControl": {"BaseControl<DerivedControl>"},
            "BaseControl": set(),
        },
        declared_classes={"DerivedControl", "BaseControl"},
        class_files={
            "DerivedControl": {"src/controllers/derived.hpp"},
        },
    )
    reference = UnresolvedSourceReference(
        symbol="halted",
        kind="callable",
        file="src/controllers/derived.cpp",
        callable_id="src/controllers/derived.cpp:1:DerivedControl::run:",
        class_owner="DerivedControl",
        argument_count=0,
    )

    candidates = SourceExpansionResolver(profiler, "hash").resolve(
        reference, structure
    )

    assert [item.file for item in candidates] == [
        "platform/common/runtime_contract.hpp"
    ]
    assert candidates[0].matched_identity.endswith("BaseControl::halted:")


def test_explicitly_qualified_namespace_function_is_not_treated_as_class_member(
    tmp_path,
):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/lib/numeric_ops.cpp": """
namespace numeric {
float reshape(float value) { return value + 2.0f; }
}
""",
        },
        backend="tree_sitter",
    )
    reference = UnresolvedSourceReference(
        symbol="numeric::reshape",
        kind="callable",
        file="src/controllers/caller.cpp",
        callable_id="src/controllers/caller.cpp:1:Caller::run:",
        class_owner="Caller",
        argument_count=1,
    )

    candidates = SourceExpansionResolver(profiler, "hash").resolve(
        reference,
        SourceStructureIndex(direct_bases={"Caller": set()}),
    )

    assert [item.file for item in candidates] == ["src/lib/numeric_ops.cpp"]


def test_unqualified_call_rejects_foreign_member_but_keeps_free_function(tmp_path):
    profiler = _mini_tree(
        tmp_path,
        {
            "src/unrelated/other.cpp": """
float Other::transform(float value) { return value; }
""",
            "src/lib/free_transform.cpp": """
float reshape(float value) { return value + 2.0f; }
""",
        },
        backend="tree_sitter",
    )
    structure = SourceStructureIndex(
        direct_bases={"Caller": set(), "Other": set()},
        declared_classes={"Caller", "Other"},
    )
    resolver = SourceExpansionResolver(profiler, "hash")

    foreign = resolver.resolve(
        UnresolvedSourceReference(
            symbol="transform",
            kind="callable",
            file="src/controllers/caller.cpp",
            callable_id="src/controllers/caller.cpp:1:Caller::run:",
            class_owner="Caller",
            argument_count=1,
        ),
        structure,
    )
    free = resolver.resolve(
        UnresolvedSourceReference(
            symbol="reshape",
            kind="callable",
            file="src/controllers/caller.cpp",
            callable_id="src/controllers/caller.cpp:1:Caller::run:",
            class_owner="Caller",
            argument_count=1,
        ),
        structure,
    )

    assert foreign == []
    assert [item.file for item in free] == ["src/lib/free_transform.cpp"]


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
    assert by_expression["input_value"]["logged_signal"] == ""
    assert by_expression["unrelated_value"]["logged_signal"] == ""
    publications = [
        binding
        for binding in inputs.bindings
        if binding.get("synthetic_boundary_transfer")
        and binding.get("boundary_direction") == "publish"
    ]
    assert len(publications) == 1
    assert publications[0]["target_symbol"] == "alpha"
    assert publications[0]["source_symbol"] == "msg"
    assert "Example::publish_value" in str(publications[0]["callable_id"])

    dag = build_mechanism_dag(
        inputs.bindings,
        "alpha.value",
        terminal_file="src/modules/example/example.cpp",
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )
    expressions = {
        vertex.expression
        for vertex in dag.vertices
        if vertex.kind == "operation"
    }
    assert "input_value" in expressions
    assert "unrelated_value" not in expressions


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
    assert "published_signal" not in target


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
    speed_s speed_data{};
    orb_copy(ORB_ID(speed), subscription, &speed_data);
    speed_sp = speed_data.value + 1.0f;
}
""",
    })

    def _fail(*args, **kwargs):
        raise AssertionError("search must not run when files are preranked")

    profiler.search_related_source_files = _fail  # type: ignore[assignment]

    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=[], terminal="speed_sp",
        source_hash="hash", logged_signals={"speed.value"},
        preranked_files=["src/modules/example/ctrl.cpp"],
        terminal_file="src/modules/example/ctrl.cpp",
    )

    assert result.terminal_validation.status == "valid"
    assert {
        v.variable for v in result.dag.vertices if v.kind == "operation"
    } == {"speed_sp", "speed_data.value"}


def test_internal_global_gap_never_falls_back_to_repository_search(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(
        tmp_path,
        {
            "src/modules/example/ctrl.cpp": """
struct Queue { float size; };
static Queue work_queue;
static float output;

void run()
{
    output = work_queue.size;
}
""",
        },
        backend="tree_sitter",
    )

    def fail_search(*args, **kwargs):
        raise AssertionError("proven internal storage must not use source search")

    profiler.search_related_source_files = fail_search  # type: ignore[assignment]
    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=[],
        terminal="output",
        source_hash="hash",
        preranked_files=["src/modules/example/ctrl.cpp"],
        terminal_file="src/modules/example/ctrl.cpp",
    )

    references = [
        item
        for item in result.dag.unresolved_references
        if item.kind == "storage_writers"
    ]
    declaration_ids = {
        item.identity.declaration_id
        for item in references
        if item.identity is not None
    }
    assert len(references) == len(declaration_ids)
    assert "work_queue.size" in result.dag.unresolved_symbols


@pytest.mark.parametrize("action", ["verified", "unresolved"])
def test_checkpoint_stop_precedes_whole_graph_annotation_and_expansion(tmp_path, monkeypatch, action):
    from flight_log_agent.analysis.checkpoint_discovery import CheckpointRound
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag
    from flight_log_agent.analysis.source_expansion import SourceExpansionResolver

    profiler = _mini_tree(tmp_path, {
        "sample.cpp": "void run() { float output = missing; }",
    }, backend="tree_sitter")
    calls = []

    def unexpected(*args, **kwargs):
        raise AssertionError("checkpoint stop must precede annotation and source search")

    def checkpoint(dag, index):
        calls.append(index)
        assert dag.vertices
        return CheckpointRound(action, dag, [], {"action": action})

    monkeypatch.setattr(SourceExpansionResolver, "resolve", unexpected)
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=[], terminal="output", source_hash="hash",
        preranked_files=["sample.cpp"], terminal_file="sample.cpp",
        round_annotator=unexpected, checkpoint_evaluator=checkpoint,
    )
    assert calls == [0]
    assert result.stop_reason == f"checkpoint_{action}"
    assert result.files_loaded == ["sample.cpp"]


def test_checkpoint_continues_only_its_requested_source_frontier(tmp_path, monkeypatch):
    from flight_log_agent.analysis.checkpoint_discovery import CheckpointRound
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "sample.cpp": "extern float wanted; extern float unrelated; void run() { float output = wanted + unrelated; }",
        "wanted.cpp": "float wanted = 3;",
        "unrelated.cpp": "float unrelated = 9;",
    }, backend="tree_sitter")
    original = SourceExpansionResolver.resolve
    original_with_evidence = SourceExpansionResolver.resolve_with_evidence
    searched = []

    def resolve(self, reference, structure):
        searched.append(reference.symbol)
        assert reference.symbol == "wanted"
        return original(self, reference, structure)

    def resolve_with_evidence(self, reference, structure, sink,
                              skip_stages=None, universe_version=None):
        searched.append(reference.symbol)
        assert reference.symbol == "wanted"
        return original_with_evidence(
            self, reference, structure, sink, skip_stages=skip_stages,
            universe_version=universe_version)

    def checkpoint(dag, index):
        references = [r for r in dag.unresolved_references if r.symbol == "wanted"]
        assert references
        return CheckpointRound("continue", dag, references, {"action": "continue"})

    monkeypatch.setattr(SourceExpansionResolver, "resolve", resolve)
    monkeypatch.setattr(
        SourceExpansionResolver, "resolve_with_evidence",
        resolve_with_evidence)
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=[], terminal="output", source_hash="hash",
        preranked_files=["sample.cpp"], terminal_file="sample.cpp",
        checkpoint_evaluator=checkpoint,
    )
    assert searched
    assert result.files_loaded == ["sample.cpp", "wanted.cpp"]
    assert len(result.rounds) == 2
    assert result.stop_reason == "checkpoint_unresolved"
    assert result.checkpoint.action == "unresolved"


@pytest.mark.parametrize("guard", ["false", "true", "missing"])
def test_staged_construction_gates_loaded_helper_materialization(tmp_path, guard):
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round

    profiler = _mini_tree(tmp_path, {"sample.cpp": """
float calculate(float input) { return input * 2.0f; }
void run() {
    bool gate = GUARD;
    float output = 0;
    if (gate) { output = calculate(7.0f); }
}
""".replace("GUARD", guard)}, backend="tree_sitter")
    inputs = dag_inputs_from_facts(load_facts(profiler, tmp_path / "cache", ["sample.cpp"], "hash"))
    snapshots = []

    def checkpoint(dag):
        snapshots.append(len(dag.vertices))
        result = evaluate_checkpoint_round(
            dag, parameter_values={}, observed_signals=set(), signal_policies={},
            load_samples=lambda *_: {},
        )
        return set(result.summary["selected_checkpoint"]["inactive_writer_ids"])

    dag = build_mechanism_dag(
        inputs.bindings, "output", terminal_file="sample.cpp",
        helper_expressions=inputs.helper_expressions, call_statements=inputs.call_statements,
        source_structure=inputs.structure, construction_checkpoint=checkpoint,
    )
    assert snapshots
    returns = [v for v in dag.vertices if str(v.provenance or "").startswith("helper_return:calculate@")]
    assert bool(returns) == (guard != "false")


@pytest.mark.parametrize("guard", ["false", "true"])
def test_guard_source_is_discovered_before_guarded_value(tmp_path, monkeypatch, guard):
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round
    from flight_log_agent.analysis.mechanism_dag import _DAGBuilder
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "policy.hpp": "class Policy { public: bool allow(); float value(); void run(); float output; };",
        "run.cpp": '#include "policy.hpp"\nvoid Policy::run() { if (allow()) { output = value(); } }',
        "guard.cpp": '#include "policy.hpp"\nbool Policy::allow() { return ' + guard + '; }',
        "value.cpp": '#include "policy.hpp"\nfloat Policy::value() { return 7.f; }',
    }, backend="tree_sitter")
    builders = []
    original = _DAGBuilder.__init__

    def initialize(self, **kwargs):
        builders.append(self)
        original(self, **kwargs)

    monkeypatch.setattr(_DAGBuilder, "__init__", initialize)
    decisions = []

    def checkpoint(dag, index):
        result = evaluate_checkpoint_round(
            dag, parameter_values={}, observed_signals=set(), signal_policies={},
            load_samples=lambda *_: {},
        )
        decisions.append((index, result.summary["next_analysis"]))
        return result

    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=[], terminal="output", terminal_file="run.cpp",
        source_hash="hash", construction_evaluator=checkpoint, checkpoint_evaluator=checkpoint,
    )
    assert len(builders) == 1
    assert "guard.cpp" in result.files_loaded
    assert ("value.cpp" in result.files_loaded) == (guard == "true")
    guard_round = next(r.index for r in result.rounds if "guard.cpp" in r.new_files)
    if guard == "true":
        value_round = next(r.index for r in result.rounds if "value.cpp" in r.new_files)
        assert value_round > guard_round
    assert any(item["kind"] == "guard_source" for _, item in decisions)
    assert result.checkpoint.action != "verified"


def test_source_admission_revalidates_late_caller_without_stale_producers(tmp_path):
    from flight_log_agent.analysis.mechanism_dag import DAGConstructionSession

    profiler = _mini_tree(tmp_path, {
        "worker.hpp": "class Worker { public: void compute(float input); float output; };",
        "worker.cpp": '#include "worker.hpp"\nvoid Worker::compute(float input) { output = input * 2.f; }',
        "caller.cpp": '#include "worker.hpp"\nvoid invoke(Worker &worker) { worker.compute(3.f); }',
    }, backend="tree_sitter")
    session = DAGConstructionSession()

    def build(files, retained):
        inputs = dag_inputs_from_facts(load_facts(profiler, tmp_path / "cache", files, "hash"))
        return build_mechanism_dag(
            inputs.bindings, "output", terminal_file="worker.cpp",
            helper_expressions=inputs.helper_expressions, call_statements=inputs.call_statements,
            source_structure=inputs.structure, boundary_bindings=inputs.boundary_bindings,
            construction_checkpoint=lambda _: set(), construction_session=retained,
        )

    before = build(["worker.hpp", "worker.cpp"], session)
    builder = session.builder
    after = build(["worker.hpp", "worker.cpp", "caller.cpp"], session)
    fresh = build(["worker.hpp", "worker.cpp", "caller.cpp"], None)
    assert session.builder is builder
    assert {v.id for v in before.vertices if v.metadata.get("is_terminal")} != {
        v.id for v in after.vertices if v.metadata.get("is_terminal")
    }
    assert {v.id: v.model_dump() for v in after.vertices} == {v.id: v.model_dump() for v in fresh.vertices}
    assert {e.id: e.model_dump() for e in after.edges} == {e.id: e.model_dump() for e in fresh.edges}


def _published_observation_probe(tmp_path, backend="tree_sitter", *, result_expression="measured * 2.f",
                                 output_copy="result", observed_instances=(0,), guard=None,
                                 construction_checkpoint=None, calculation="", helper_source=""):
    source = """
class Probe {
    uORB::Publication<diagnostic_s> pub{ORB_ID(diagnostic)};
    void run(float input) {
        float measured = input;
        float result = measured * 2.f;
        diagnostic_s packet{};
        packet.input = measured;
        packet.result = result;
        pub.publish(packet);
    }
};
""".replace("measured * 2.f", result_expression).replace("packet.result = result;", f"packet.result = {output_copy};")
    source = source.replace("class Probe {", "class Probe {\n" + helper_source)
    source = source.replace("float result =", calculation + "\n        float result =")
    if guard is not None:
        source = source.replace("void run(float input) {", f"void run(float input) {{ if ({guard}) {{")
        source = source.replace("pub.publish(packet);", "pub.publish(packet); }")
    profiler = _mini_tree(tmp_path, {"sample.cpp": source}, backend=backend)
    inputs = dag_inputs_from_facts(load_facts(profiler, tmp_path / "cache", ["sample.cpp"], "hash"))
    samples = {f"diagnostic[{instance}].{field}": [(0., value), (1., value)]
               for instance in observed_instances for field, value in [("input", 3.), ("result", 6.)]}
    dag = build_mechanism_dag(
        inputs.bindings, "result", terminal_file="sample.cpp", source_structure=inputs.structure,
        call_statements=inputs.call_statements, boundary_bindings=inputs.boundary_bindings,
        helper_expressions=inputs.helper_expressions, logged_signals=set(samples),
        construction_checkpoint=(lambda dag: construction_checkpoint(dag, samples))
        if construction_checkpoint is not None else lambda _: set(),
    )
    return dag, samples


@pytest.mark.parametrize("guard", ["permit()", "false"])
def test_local_input_demand_resumes_without_resolving_guard(tmp_path, guard):
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round

    snapshots = []

    def checkpoint(dag, samples):
        result = evaluate_checkpoint_round(
            dag, parameter_values={}, observed_signals=set(samples),
            signal_policies={s: {"method": "linear"} for s in samples},
            load_samples=lambda *_: samples,
        )
        snapshots.append((dag, result))
        return result.construction

    dag, _ = _published_observation_probe(tmp_path, guard=guard, construction_checkpoint=checkpoint)
    root = next(v.id for v in dag.vertices if v.variable == "result")
    if guard == "false":
        assert all(root not in result.construction.materialize for _, result in snapshots)
    else:
        assert any(result.summary["next_analysis"]["kind"] == "local_calculation_construction"
                   and root in result.construction.materialize for _, result in snapshots)
        assert any(check["root_vertex_id"] == root and check["status"] == "matched"
                   for _, result in snapshots for check in result.summary["local_equation_checks"])
        matched_dag, _ = next((snapshot, result) for snapshot, result in snapshots
                              if any(c["root_vertex_id"] == root and c["status"] == "matched"
                                     for c in result.summary["local_equation_checks"]))
        measured = next(v.id for v in matched_dag.vertices if v.variable == "measured")
        assert measured in matched_dag.pending_construction
        assert any(r.symbol == "permit" for r in dag.unresolved_references)
    assert all(result.action != "verified" for _, result in snapshots)
    assert all(not check["applicability_verified"] and not check["authorizes_discovery_stop"]
               for _, result in snapshots for check in result.summary["local_equation_checks"])


@pytest.mark.parametrize("forwarding", [False, True])
@pytest.mark.parametrize("source_backend", [
    pytest.param("legacy", marks=pytest.mark.xfail(
        strict=True, reason="legacy lacks exact local-equation operands; retire after staged contract parity")),
    "tree_sitter",
])
def test_local_equation_discovers_its_helper_before_unknown_guard(tmp_path, forwarding, source_backend):
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path, {
        "probe.hpp": """class Probe {
            uORB::Publication<diagnostic_s> pub{ORB_ID(diagnostic)};
            bool permit(); float gain(); float calculated(float measured); void run(float input);
        };""",
        "run.cpp": """#include "probe.hpp"
        void Probe::run(float input) {
            if (permit()) {
                float measured = input;
                float result = measured * gain();
                diagnostic_s packet{};
                packet.input = measured;
                packet.result = result;
                pub.publish(packet);
            }
        }""".replace("measured * gain()", "calculated(measured)" if forwarding else "measured * gain()"),
        "gain.cpp": '#include "probe.hpp"\nfloat Probe::gain() { return 2.f; }',
        "calculated.cpp": '#include "probe.hpp"\nfloat Probe::calculated(float measured) { return measured * 2.f; }',
    }, backend=source_backend)
    samples = {"diagnostic[0].input": [(0., 3.), (1., 3.)],
               "diagnostic[0].result": [(0., 6.), (1., 6.)]}
    decisions = []

    def checkpoint(dag, index):
        result = evaluate_checkpoint_round(
            dag, parameter_values={}, observed_signals=set(samples),
            signal_policies={s: {"method": "linear"} for s in samples},
            load_samples=lambda *_: samples,
        )
        decisions.append(result)
        return result

    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=[], terminal="result", terminal_file="run.cpp",
        source_hash="hash", logged_signals=set(samples),
        construction_evaluator=checkpoint, checkpoint_evaluator=checkpoint,
    )
    source_decisions = [d for d in decisions if d.references]
    assert source_decisions[0].summary["next_analysis"]["kind"] == "local_calculation_source", [
        (d.summary["next_analysis"]["kind"], [
            (c["requirements"], [(n["kind"], n["operand"], [r["symbol"] for r in n["source_requests"]])
                                    for n in c["input_requirements"]])
            for c in d.summary["local_equation_checks"]]) for d in decisions
    ]
    assert {r.symbol for r in source_decisions[0].references} == {"calculated" if forwarding else "gain"}
    assert ("calculated.cpp" if forwarding else "gain.cpp") in result.files_loaded
    if forwarding:
        assert any(d.summary["next_analysis"]["kind"] == "local_calculation_construction"
                   and any(v.id in d.construction.materialize
                           and v.variable.startswith("__return__")
                           for v in d.annotated.vertices)
                   for d in decisions)
        assert all(not c["authorizes_discovery_stop"] for d in decisions
                   for c in d.summary["local_equation_checks"])
    assert any(c["status"] == "matched" for d in decisions for c in d.summary["local_equation_checks"])
    assert result.checkpoint.action != "verified"


@pytest.mark.parametrize("predicate,status", [("measured > 0", "matched"), ("permit()", "unevaluable")])
def test_local_equation_selects_source_writers_without_guessing(tmp_path, predicate, status):
    from flight_log_agent.analysis.dag_observation import evaluate_local_observed_equations
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    dag, _ = _published_observation_probe(
        tmp_path, result_expression="measured * factor",
        calculation=f"float factor; if ({predicate}) {{ factor = 2.f; }} else {{ factor = 3.f; }}",
    )
    samples = {"diagnostic[0].input": [(0., 3.), (1., -3.)],
               "diagnostic[0].result": [(0., 6.), (1., -9.)]}
    checks = evaluate_local_observed_equations(
        dag, DAGValueProgram(dag), signal_samples=samples, parameter_values={},
        signal_policies={s: {"method": "linear"} for s in samples}, scope=None,
        relevant_ids={v.id for v in dag.vertices},
    )
    root = next(v.id for v in dag.vertices if v.variable == "result")
    check = next(c for c in checks if c["root_vertex_id"] == root)
    assert check["status"] == status, check
    if status == "matched":
        assert check["sample_count"] == 2
        assert not check["input_requirements"]
    else:
        assert any(n["kind"] == "writer_coverage" for n in check["input_requirements"])
    assert not check["authorizes_discovery_stop"]


@pytest.mark.parametrize("backend", [
    pytest.param("legacy", marks=pytest.mark.xfail(strict=True, reason="legacy assignments lack exact expression/copy metadata")),
    "tree_sitter",
])
def test_published_intermediate_observes_source_value_without_subscription(tmp_path, backend):
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round
    from flight_log_agent.analysis.dag_observation import observation_correspondences
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    dag, samples = _published_observation_probe(tmp_path, backend)
    correspondence = observation_correspondences(dag, DAGValueProgram(dag))
    result_vertex = next(v for v in dag.vertices if v.variable == "result")
    assert any(c["value_id"] == result_vertex.id and c["signal"] == "diagnostic[0].result"
               for c in correspondence)
    assert not any(v.metadata.get("boundary_direction") == "subscribe" for v in dag.vertices)
    decision = evaluate_checkpoint_round(
        dag, parameter_values={}, observed_signals=set(samples),
        signal_policies={s: {"method": "linear"} for s in samples}, load_samples=lambda *_: samples,
    )
    check = next(c for c in decision.summary["local_equation_checks"] if c["root_vertex_id"] == result_vertex.id)
    assert check["status"] == "matched", check
    assert check["verification_scope"] == "local_equation_given_observations"
    assert check["authorizes_discovery_stop"] is False
    assert check["upstream_obligations_retained"] is True
    assert decision.action != "verified"
    assert "observation_witnesses" not in dag.model_dump()


@pytest.mark.parametrize("copy", ["static_cast<int>(result)", "result * 3.f"])
def test_observation_does_not_invert_transformation(tmp_path, copy):
    dag, _ = _published_observation_probe(tmp_path, output_copy=copy)
    assert not any(w["signal"].endswith(".result") for w in dag.observation_witnesses)


def test_observation_does_not_guess_topic_instance(tmp_path):
    dag, _ = _published_observation_probe(tmp_path, observed_instances=(0, 1))
    assert dag.observation_witnesses == []


@pytest.mark.parametrize("problem", [
    "unaligned", "duplicate", "outside_scope", "mismatch", "circular",
    "different_publication", "different_callable", "alternative_writer", "cycle",
])
@pytest.mark.parametrize("forwarding", [False, True])
def test_local_observation_scope_and_alignment(tmp_path, problem, forwarding):
    from flight_log_agent.analysis.dag_observation import evaluate_local_observed_equations
    from flight_log_agent.analysis.dag_replay import EvaluationScope
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    dag, samples = _published_observation_probe(
        tmp_path, result_expression="calculate(measured)" if forwarding else "measured * 2.f",
        helper_source="float calculate(float value) { return value * 2.f; }" if forwarding else "",
    )
    if problem == "unaligned":
        samples["diagnostic[0].input"] = [(0.1, 3.), (1.1, 3.)]
    elif problem == "duplicate":
        samples["diagnostic[0].input"].append((1., 4.))
    elif problem in {"mismatch", "outside_scope"}:
        samples["diagnostic[0].result"][0] = (0., 99.)
    elif problem == "circular":
        dag.observation_witnesses = [w for w in dag.observation_witnesses if w["signal"].endswith(".result")]
    elif problem in {"different_publication", "different_callable"}:
        witness = next(w for w in dag.observation_witnesses if w["signal"].endswith(".input"))
        key = "publication_site" if problem == "different_publication" else "callable"
        witness[key] += ":different"
    elif problem == "alternative_writer":
        root = next(v.id for v in dag.vertices if v.variable == "result")
        edge = next(e for e in dag.edges if e.target_id == root and e.kind == "data"
                    and (not forwarding or e.role.startswith("call:")))
        original = next(v for v in dag.vertices if v.id == edge.source_id)
        alternate = original.model_copy(update={"id": original.id + "_alternative"})
        dag.vertices.append(alternate)
        if forwarding:
            dag.pending_construction.append(alternate.id)
        dag.edges.append(edge.model_copy(update={"id": edge.id + "_alternative", "source_id": alternate.id}))
    elif problem == "cycle":
        root = next(v.id for v in dag.vertices if v.variable == "result")
        edge = next(e for e in dag.edges if e.target_id == root and e.kind == "data"
                    and (not forwarding or e.role.startswith("call:")))
        edge.source_id = root
    program = DAGValueProgram(dag)
    result = evaluate_local_observed_equations(
        dag, program, signal_samples=samples, parameter_values={},
        signal_policies={s: {"method": "linear"} for s in samples},
        scope=EvaluationScope(windows=((1., 1.),)) if problem == "outside_scope" else None,
        relevant_ids={v.id for v in dag.vertices},
    )
    root = next(v.id for v in dag.vertices if v.variable == "result")
    check = next(c for c in result if c["root_vertex_id"] == root)
    assert check["status"] == ("matched" if problem == "outside_scope" else "mismatched" if problem == "mismatch" else "unevaluable")
    assert check["authorizes_discovery_stop"] is False
    if problem in {"unaligned", "duplicate", "different_publication", "different_callable"}:
        assert check["input_requirements"] == []
    if problem == "circular":
        assert not any(n["source_requests"] for n in check["input_requirements"])


_UNRESOLVED_MEMBER_SOURCE = """
class Cfg {
public:
    float factor{};
};
class Probe {
    uORB::Publication<diagnostic_s> pub{ORB_ID(diagnostic)};
    Cfg config{};
    void run(float input) {
        float measured = input;
        float result = measured * config.factor;
        diagnostic_s packet{};
        packet.input = measured;
        packet.result = result;
        pub.publish(packet);
    }
};
"""


def _unresolved_member_dag(tmp_path, source):
    profiler = _mini_tree(tmp_path, {"sample.cpp": source}, backend="tree_sitter")
    inputs = dag_inputs_from_facts(
        load_facts(profiler, tmp_path / "cache", ["sample.cpp"], "hash"))
    samples = {"diagnostic[0].input": [(0., 3.), (1., 3.)],
               "diagnostic[0].result": [(0., 6.), (1., 6.)]}
    dag = build_mechanism_dag(
        inputs.bindings, "result", terminal_file="sample.cpp",
        source_structure=inputs.structure,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        helper_expressions=inputs.helper_expressions,
        logged_signals=set(samples),
        construction_checkpoint=lambda _: set(),
    )
    return dag, samples


def _local_checks_for(dag, samples):
    from flight_log_agent.analysis.dag_observation import evaluate_local_observed_equations
    from flight_log_agent.analysis.dag_value import DAGValueProgram
    return evaluate_local_observed_equations(
        dag, DAGValueProgram(dag), signal_samples=samples, parameter_values={},
        signal_policies={s: {"method": "linear"} for s in samples}, scope=None,
        relevant_ids={v.id for v in dag.vertices},
    )


def test_opaque_leaf_need_links_proven_declaration_request(tmp_path):
    """An opaque-leaf local need associates the unresolved request sharing
    its proven declaration — the same storage obligation, not a spelling
    coincidence. The context-free ordinary path already rejects the leaf;
    this pins the requirement linkage that must carry the exact request."""
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    dag, samples = _unresolved_member_dag(tmp_path, _UNRESOLVED_MEMBER_SOURCE)
    request = next(r for r in dag.unresolved_references
                   if r.kind == "storage_writers" and r.symbol == "config")
    assert request.identity is not None
    assert request.identity.declaration_proven is True
    assert request.identity.symbol == "config.factor"
    leaf = next(v for v in dag.vertices
                if v.kind == "evidence" and v.sub_kind == "opaque_symbol"
                and v.signal_name == "config.factor")
    leaf_identity = (leaf.metadata or {}).get("source_identity") or {}
    assert leaf_identity.get("declaration_proven") is True
    assert (leaf_identity.get("declaration_id")
            == request.identity.declaration_id)

    checks = _local_checks_for(dag, samples)
    check = next(c for c in checks
                 if c["root_vertex_id"] == next(
                     v.id for v in dag.vertices
                     if v.kind == "operation" and v.variable == "result"))
    need = next(n for n in check["input_requirements"]
                if n["vertex_id"] == leaf.id)
    assert any(s["symbol"] == request.symbol
               and (s.get("identity") or {}).get("declaration_id")
               == request.identity.declaration_id
               for s in need["source_requests"])


def test_opaque_leaf_need_ignores_unrelated_declaration(tmp_path):
    """Same member spelling under an unrelated declaration must not satisfy
    an opaque-leaf need: association is by proven declaration, never by
    spelling. Each need links only its own declaration's request."""
    source = _UNRESOLVED_MEMBER_SOURCE.replace(
        "    Cfg config{};",
        "    Cfg config{};\n    Other other{};").replace(
        "class Probe {",
        "class Other {\npublic:\n    float factor{};\n};\nclass Probe {").replace(
        "measured * config.factor",
        "measured * config.factor + other.factor")
    dag, samples = _unresolved_member_dag(tmp_path, source)
    by_declaration = {}
    for reference in dag.unresolved_references:
        if reference.kind != "storage_writers" or reference.identity is None:
            continue
        assert reference.identity.declaration_proven is True
        by_declaration.setdefault(reference.identity.declaration_id, reference)
    assert len(by_declaration) == 2

    checks = _local_checks_for(dag, samples)
    linked = [n for c in checks for n in c["input_requirements"]
              if n["source_requests"]]
    assert linked, "expected declaration-linked needs"
    for need in linked:
        leaf = next(v for v in dag.vertices if v.id == need["vertex_id"])
        leaf_declaration = ((leaf.metadata or {}).get("source_identity") or {}
                            ).get("declaration_id")
        assert leaf_declaration, "need must come from a proven leaf"
        for attached in need["source_requests"]:
            assert ((attached.get("identity") or {}).get("declaration_id")
                    == leaf_declaration)


def test_declaration_linked_need_drives_local_calculation_scheduling(tmp_path):
    """A Step-D declaration-linked need must reach checkpoint discovery
    scheduling as local-calculation source work: linkage alone is not the
    goal, selection of the same semantic obligation is. Scheduling changes
    nothing about evidence, coverage, or stop authority."""
    from flight_log_agent.analysis.checkpoint_discovery import evaluate_checkpoint_round

    source = _UNRESOLVED_MEMBER_SOURCE
    dag, samples = _unresolved_member_dag(tmp_path, source)
    expected = set()
    for reference in dag.unresolved_references:
        if reference.kind != "storage_writers" or reference.identity is None:
            continue
        assert reference.identity.declaration_proven is True
        expected.add(reference.identity.declaration_id)
    assert len(expected) == 1

    decision = evaluate_checkpoint_round(
        dag, parameter_values={}, observed_signals=set(samples),
        signal_policies={s: {"method": "linear"} for s in samples},
        load_samples=lambda *_: samples,
    )
    assert decision.summary["next_analysis"]["kind"] == "local_calculation_source"
    selected = {r.identity.declaration_id for r in decision.references
                if r.identity is not None}
    assert expected <= selected
    assert selected <= expected, "no unrelated declaration may be scheduled"
    assert decision.action != "verified"
    assert all(not c["authorizes_discovery_stop"]
               for c in decision.summary["local_equation_checks"])
    assert all(c["status"] != "matched"
               for c in decision.summary["local_equation_checks"])


@pytest.mark.parametrize("name,formula,rows", [
    ("rtl_floor", "std::max(a, b * 2.f)", [(10., 10., 0., 20.)]),
    ("airspeed_bank", "a * sqrtf(1.f / cosf(b))", [(19., 0.8726646259971648, 0., 23.698446)]),
    ("tecs_reference", "(a - b) / 5.f + 0.3f * c", [(28.537527, 28.537385560152085, 3.3737552, 1.0121266)]),
    ("takeoff_stage", "a ? b : c", [(1., 20., 7., 20.), (0., 20., 7., 7.)]),
])
def test_archived_calculation_shapes_use_source_and_observed_intermediates(tmp_path, name, formula, rows):
    """Local acceptance shapes, not a substitute for the four real-source logs.

    The equations are extracted from this source fixture, not passed directly
    to replay. The numerical targets reflect the archived investigations.
    Applicability, state history, and alternative-writer proofs remain open.
    """
    from flight_log_agent.analysis.dag_observation import evaluate_local_observed_equations
    from flight_log_agent.analysis.dag_replay import EvaluationScope
    from flight_log_agent.analysis.dag_value import DAGValueProgram

    source = """
class Instrument {
    uORB::Publication<diagnostic_s> pub{ORB_ID(diagnostic)};
    void run(float x, float y, float z) {
        float a = x;
        float b = y;
        float c = z;
        float result = FORMULA;
        diagnostic_s packet{};
        packet.a = a;
        packet.b = b;
        packet.c = c;
        packet.result = result;
        pub.publish(packet);
    }
};
""".replace("FORMULA", formula)
    profiler = _mini_tree(tmp_path, {"sample.cpp": source}, backend="tree_sitter")
    inputs = dag_inputs_from_facts(load_facts(profiler, tmp_path / "cache", ["sample.cpp"], "hash"))
    samples = {f"diagnostic[0].{field}": [(float(i), row[column]) for i, row in enumerate(rows)]
               for column, field in enumerate(("a", "b", "c", "result"))}
    dag = build_mechanism_dag(
        inputs.bindings, "result", terminal_file="sample.cpp", source_structure=inputs.structure,
        call_statements=inputs.call_statements, boundary_bindings=inputs.boundary_bindings,
        helper_expressions=inputs.helper_expressions, logged_signals=set(samples),
        construction_checkpoint=lambda _: set(),
    )
    program = DAGValueProgram(dag)
    checks = evaluate_local_observed_equations(
        dag, program, signal_samples=samples, parameter_values={},
        signal_policies={s: {"method": "linear"} for s in samples},
        scope=EvaluationScope(windows=((0., float(len(rows) - 1)),)),
        relevant_ids={v.id for v in dag.vertices},
    )
    root = next(v.id for v in dag.vertices if v.variable == "result")
    check = next(c for c in checks if c["root_vertex_id"] == root)
    assert check["status"] == "matched", (name, check)
    assert check["sample_count"] == len(rows)
    assert check["observed_inputs"]
    assert check["applicability_verified"] is False
    assert check["writer_coverage_verified"] is False
    assert check["upstream_obligations_retained"] is True
    assert not check["authorizes_discovery_stop"]


# --- T1: structured search evidence (observation only, no authority) ---

def _evidence_setup(tmp_path, files, backend="tree_sitter"):
    profiler = _mini_tree(tmp_path, files, backend=backend)
    file_list = list(files)
    facts = load_facts(profiler, tmp_path / "cache", file_list, "hash")
    inputs = dag_inputs_from_facts(facts)
    return profiler, inputs


def _resolve_with_evidence(profiler, reference, structure):
    from flight_log_agent.analysis.coverage import CoverageEvidence
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    candidates, evidence = resolver.resolve_with_evidence(
        reference, structure, sink)
    assert isinstance(evidence, CoverageEvidence)
    assert evidence.attempts == sink
    return candidates, evidence


def test_evidence_records_admitted_storage_writer(tmp_path):
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/widget.cpp": (
            "struct Widget { float level; void fill() { level = 1; } };"
        ),
    })
    assert inputs.structure.authoritative_declarations
    identity = inputs.structure.symbol_identity(
        "level", file="src/lib/widget.cpp",
        callable_id="Widget::fill", function_name="Widget::fill")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="level", kind="storage_writers", file="src/lib/widget.cpp",
        callable_id="Widget::fill", identity=identity)
    candidates, evidence = _resolve_with_evidence(
        profiler, reference, inputs.structure)
    assert [item.file for item in candidates] == ["src/lib/widget.cpp"]
    assert len(evidence.attempts) >= 1
    admitted = [a for a in evidence.attempts if a.outcome == "admitted"]
    assert len(admitted) == 1
    assert "src/lib/widget.cpp" in admitted[0].examined_domain["files"]
    assert admitted[0].universe_ref["authoritative_declarations"] is True


def test_evidence_distinguishes_ambiguous_from_empty(tmp_path):
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/over.cpp": "void tune(int x) {} void tune(float x) {}",
    })
    overloaded = UnresolvedSourceReference(
        symbol="tune", kind="callable", file="src/lib/over.cpp",
        argument_count=1)
    candidates, evidence = _resolve_with_evidence(
        profiler, overloaded, inputs.structure)
    assert candidates == []
    ambiguous = [a for a in evidence.attempts if a.outcome == "ambiguous"]
    assert len(ambiguous) >= 1
    assert ambiguous[0].details.get("colliding_identities"), (
        "ambiguity must retain the colliding set")

    ghost = UnresolvedSourceReference(
        symbol="NoSuchEntity", kind="class", file="src/lib/over.cpp")
    ghost_candidates, ghost_evidence = _resolve_with_evidence(
        profiler, ghost, inputs.structure)
    assert ghost_candidates == []
    ghost_outcomes = {a.outcome for a in ghost_evidence.attempts}
    assert "ambiguous" not in ghost_outcomes
    assert ghost_outcomes <= {"no-candidate-complete-domain",
                              "no-candidate-partial-domain", "unavailable",
                              "non-writer"}


def test_evidence_records_inapplicable_local(tmp_path):
    from flight_log_agent.analysis.source_expansion import SourceSymbolIdentity
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/local.cpp": "void run() { float temp = 1; sink(temp); }",
    })
    reference = UnresolvedSourceReference(
        symbol="temp", kind="symbol", file="src/lib/local.cpp",
        callable_id="run",
        identity=SourceSymbolIdentity(
            kind="local", symbol="temp", root="temp",
            file="src/lib/local.cpp", callable_id="run",
            declaration_id="run:parameter:0", declaration_proven=True))
    candidates, evidence = _resolve_with_evidence(
        profiler, reference, inputs.structure)
    assert candidates == []
    assert any(a.outcome == "inapplicable" for a in evidence.attempts)


def test_evidence_records_unavailable_receiver(tmp_path):
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/user.cpp": "void run() { helper.adjust(1); }",
    })
    reference = UnresolvedSourceReference(
        symbol="adjust", kind="callable", file="src/lib/user.cpp",
        callable_id="run", receiver="helper", receiver_type="",
        argument_count=1)
    candidates, evidence = _resolve_with_evidence(
        profiler, reference, inputs.structure)
    assert candidates == []
    assert any(a.outcome == "unavailable" for a in evidence.attempts)


def test_evidence_records_no_query_issued(tmp_path):
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/empty.cpp": "void run() {}",
    })
    reference = UnresolvedSourceReference(symbol="", file="src/lib/empty.cpp")
    candidates, evidence = _resolve_with_evidence(
        profiler, reference, inputs.structure)
    assert candidates == []
    assert any(a.outcome == "no-query-issued" for a in evidence.attempts)


def test_heuristic_discovery_is_not_identity_proof(tmp_path):
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/split.cpp": "float output;\nvoid run() {\noutput\n= 1;\n}",
    })
    reference = UnresolvedSourceReference(
        symbol="output", file="src/lib/split.cpp")
    candidates, evidence = _resolve_with_evidence(
        profiler, reference, inputs.structure)
    assert [item.file for item in candidates] == ["src/lib/split.cpp"]
    bare_hits = [a for a in evidence.attempts
                 if a.outcome == "admitted" and "bare" in a.strategy]
    assert len(bare_hits) == 1, (
        "the bare-name fallback group must be labeled heuristic, "
        "not an exact-shaped strategy")
    assert bare_hits[0].strategy_class == "heuristic"


def test_helper_provider_records_evidence(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import (
        make_helper_body_provider,
    )
    profiler, _inputs = _evidence_setup(tmp_path, {
        "src/lib/numeric.hpp": "float adjust(float v, bool scale = false);",
        "src/lib/numeric.cpp": (
            '#include "numeric.hpp"\n'
            "float adjust(float v, bool scale) { return v; }"),
        "src/modules/caller.cpp": (
            '#include <src/lib/numeric.hpp>\n'
            "void run() { float x = adjust(1); }"),
    })
    sink: list = []
    provider = make_helper_body_provider(
        profiler, [], resolver=SourceExpansionResolver(profiler, "hash"),
        attempt_sink=sink)
    caller_facts = load_facts(
        profiler, tmp_path / "cache", ["src/modules/caller.cpp"], "hash")
    reference = UnresolvedSourceReference(
        symbol="adjust", kind="callable", file="src/modules/caller.cpp",
        callable_id=next(
            item.callable_id
            for item in caller_facts[0].callables
            if item.name == "run"
        ),
        argument_count=1)
    helpers = provider("adjust", reference)
    assert helpers, "expected the helper definition to be found"
    assert len(sink) >= 1, "helper-driven resolution must record evidence"
    assert all(hasattr(a, "outcome") for a in sink)


def test_empty_examined_set_never_means_rejected(tmp_path):
    """Zero examined candidates must fall through to the domain-class rule,
    never to a rejection outcome: absence of examination is not rejection."""
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/empty.cpp": "void run() {}",
    })
    reference = UnresolvedSourceReference(
        symbol="NoSuchCallable", kind="callable",
        file="src/lib/empty.cpp", argument_count=0)
    candidates, evidence = _resolve_with_evidence(
        profiler, reference, inputs.structure)
    assert candidates == []
    assert evidence.attempts, "expected recorded attempts"
    forbidden = {"non-writer", "rejected-declaration", "rejected-owner",
                 "ambiguous"}
    for attempt in evidence.attempts:
        assert attempt.outcome not in forbidden, (
            attempt.strategy, attempt.outcome)
    assert {a.outcome for a in evidence.attempts} <= {
        "no-candidate-complete-domain", "no-candidate-partial-domain",
        "unavailable", "no-query-issued"}


def test_issued_queries_recorded_pre_normalization(tmp_path):
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/plain.cpp": "float output; void run() { output = 2; }",
    })
    reference = UnresolvedSourceReference(
        symbol="output", file="src/lib/plain.cpp")
    _candidates, evidence = _resolve_with_evidence(
        profiler, reference, inputs.structure)
    issued = [q for a in evidence.attempts for q in a.queries_issued]
    assert "output =" in issued or "output=" in issued
    assert all(isinstance(q, str) for q in issued)


def test_evidence_path_matches_legacy_results(tmp_path):
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/widget.cpp": (
            "struct Widget { float level; void fill() { level = 1; } };"
        ),
        "src/lib/over.cpp": "void tune(int x) {} void tune(float x) {}",
    })
    resolver = SourceExpansionResolver(profiler, "hash")
    identity = inputs.structure.symbol_identity(
        "level", file="src/lib/widget.cpp",
        callable_id="Widget::fill", function_name="Widget::fill")
    references = [
        UnresolvedSourceReference(symbol="level", kind="storage_writers",
                                  file="src/lib/widget.cpp",
                                  callable_id="Widget::fill",
                                  identity=identity),
        UnresolvedSourceReference(symbol="tune", kind="callable",
                                  file="src/lib/over.cpp", argument_count=1),
        UnresolvedSourceReference(symbol="NoSuch", kind="class",
                                  file="src/lib/over.cpp"),
        UnresolvedSourceReference(symbol="missing", kind="symbol",
                                  file="src/lib/widget.cpp"),
    ]
    for reference in references:
        legacy = [(c.file, c.matched_kind)
                  for c in resolver.resolve(reference, inputs.structure)]
        sink: list = []
        candidates, evidence = resolver.resolve_with_evidence(
            reference, inputs.structure, sink)
        assert [(c.file, c.matched_kind) for c in candidates] == legacy
        if legacy:
            assert any(a.outcome == "admitted" for a in evidence.attempts)


def test_compatible_clause_mirror_matches_compatible():
    from flight_log_agent.analysis.source_expansion import SourceSymbolIdentity

    structure = SourceStructureIndex(
        direct_bases={"Derived": {"Base"}},
        members={("Base", "value"): {"owner": "Base", "name": "value"}},
    )
    structure.authoritative_declarations = True

    def identity(**fields):
        base = {"kind": "member", "symbol": "value", "root": "value",
                "file": "a.cpp", "callable_id": "A::run",
                "class_owner": "Base", "declaring_class": "Base",
                "declaration_id": "a.cpp:1:member:value",
                "declaration_proven": True}
        base.update(fields)
        return SourceSymbolIdentity(**base)

    pairs = [
        # (reference overrides, producer overrides, expected clause or None)
        ({}, {}, None),
        ({"symbol": "other"}, {}, "spelling"),
        ({"declaration_proven": False}, {}, "proven"),
        ({"kind": "local", "callable_id": "A::run",
          "declaration_id": "A::run:parameter:0"},
         {"kind": "local", "callable_id": "A::run",
          "declaration_id": "A::run:parameter:0"}, None),
        ({"kind": "local", "callable_id": "A::run",
          "declaration_id": "x"},
         {"kind": "local", "callable_id": "B::run",
          "declaration_id": "x"}, "local-scope"),
        ({"declaring_class": "Other"}, {}, "declaration-id"),
        ({"class_owner": "Derived"}, {}, None),
        ({"class_owner": "Unrelated"}, {}, "owner-lineage"),
        ({"kind": "global"}, {"kind": "member"}, "kind"),
    ]
    for ref_fields, prod_fields, expected in pairs:
        reference = identity(**ref_fields)
        producer = identity(**prod_fields)
        assert (structure.compatible(reference, producer)
                == (SourceExpansionResolver._incompatible_clause(
                    reference, producer, structure) is None))
        assert (SourceExpansionResolver._incompatible_clause(
            reference, producer, structure) == expected)


# --- T2: search-universe version, retry, invalidation (scheduling only) ---

def _derived_gain_fixture(tmp_path):
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _mini_tree(tmp_path, {
        "src/main.cpp": """
struct Cfg { float gain; };
float tweak(float v);
struct Ctrl {
    Cfg cfg;
    float out;
    void run() { out = cfg.gain * tweak(1.0f); }
};
""",
        "src/help.cpp": """
float tweak(float v) { return v; }
struct Derived : Cfg {};
void Derived::apply(float v) { gain = v; }
void Cfg::other() {}
""",
    }, backend="tree_sitter")
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["run"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp")
    assert {f for _, files in [(r.index, r.new_files) for r in result.rounds]
            for f in files} >= {"src/main.cpp", "src/help.cpp"}
    return result


def test_cross_version_retry_reopens_suppressed_request(tmp_path, monkeypatch):
    """Scheduling-only retry: a request suppressed as visited must be
    re-examined after new source extends the searchable universe.

    This asserts scheduling (re-examination, evidence preserved,
    fail-closed retention), not admission: admitting help.cpp's `gain`
    field writer for the `cfg` struct obligation needs member-path
    semantic matching, which is deferred to a later ticket. Here the
    obligation must stay open and be retried, never dropped or falsely
    satisfied."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    # Pin the production invalidation: count discovery-loop resolutions of
    # the cfg obligation. Round 0 resolves it once and marks it visited;
    # the post-extension round must resolve it AGAIN. Without version
    # invalidation (visited.clear on universe extension) the second call
    # never happens, so this count fails.
    resolve_calls: list = []
    original_resolve = SourceExpansionResolver.resolve
    original_with_evidence = SourceExpansionResolver.resolve_with_evidence

    def counting_resolve(self, reference, structure):
        if (reference.kind == "storage_writers"
                and reference.symbol == "cfg"):
            resolve_calls.append(reference.visit_key())
        return original_resolve(self, reference, structure)

    def counting_resolve_with_evidence(self, reference, structure, sink,
                                       skip_stages=None,
                                       universe_version=None):
        if (reference.kind == "storage_writers"
                and reference.symbol == "cfg"):
            resolve_calls.append(reference.visit_key())
        return original_with_evidence(
            self, reference, structure, sink, skip_stages=skip_stages,
            universe_version=universe_version)

    monkeypatch.setattr(
        SourceExpansionResolver, "resolve", counting_resolve)
    monkeypatch.setattr(
        SourceExpansionResolver, "resolve_with_evidence",
        counting_resolve_with_evidence)
    result = _derived_gain_fixture(tmp_path)
    assert result.dag is not None
    assert len(resolve_calls) == 2, (
        "expected the cfg obligation to be resolved once per universe "
        f"version (round 0 + post-extension retry), got {resolve_calls}")
    # The universe extended across versions: both files loaded.
    assert len(result.rounds) >= 2
    assert result.rounds[1].new_files == ["src/help.cpp"]
    # The frontier request was re-examined after extension, not
    # permanently suppressed by same-version visited state.
    assert "cfg.gain" in result.rounds[1].unresolved_symbols
    # Fail-closed: the struct obligation is retained, not dropped.
    assert [r for r in result.dag.unresolved_references
            if r.kind == "storage_writers" and r.symbol == "cfg"], (
        "expected the cfg storage obligation to be retained open")
    # Resolver-level re-eligibility: resolving twice records evidence
    # both times with identical candidates (retry corrupts nothing).
    from flight_log_agent.analysis.mechanism_discovery import load_facts
    from flight_log_agent.analysis.mechanism_discovery import (
        dag_inputs_from_facts,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    reference = next(
        r for r in result.dag.unresolved_references
        if r.kind == "storage_writers" and r.symbol == "cfg")
    profiler = _mini_tree(tmp_path, {
        "src/main.cpp": """
struct Cfg { float gain; };
float tweak(float v);
struct Ctrl {
    Cfg cfg;
    float out;
    void run() { out = cfg.gain * tweak(1.0f); }
};
""",
        "src/help.cpp": """
float tweak(float v) { return v; }
struct Derived : Cfg {};
void Derived::apply(float v) { gain = v; }
void Cfg::other() {}
""",
    }, backend="tree_sitter")
    facts = load_facts(
        profiler, tmp_path / "cache",
        ["src/main.cpp", "src/help.cpp"], "hash")
    expanded = dag_inputs_from_facts(facts)
    resolver = SourceExpansionResolver(profiler, "hash")
    first_sink: list = []
    first_candidates, first_evidence = resolver.resolve_with_evidence(
        reference, expanded.structure, first_sink)
    second_sink: list = []
    second_candidates, second_evidence = resolver.resolve_with_evidence(
        reference, expanded.structure, second_sink)
    assert [(c.file, c.matched_kind) for c in first_candidates] == [
        (c.file, c.matched_kind) for c in second_candidates]
    assert first_evidence.attempts, "expected recorded attempts"
    assert second_evidence.attempts, "expected retry to re-record attempts"


def test_stage_cursor_resumes_after_ambiguity(tmp_path, monkeypatch):
    """Record suppression after ambiguity: revisiting with already-recorded
    stages suppressed records only the remaining stages and then settles,
    while the underlying search still executes authoritatively and
    candidates stay identical.

    `skip_stages` suppresses duplicate SearchAttempt records only; it never
    skips search execution. A suppressed record means "already recorded",
    never "this stage did not run"."""
    from flight_log_agent.analysis.coverage import attempt_stage
    from flight_log_agent.analysis.source_expansion import planned_stages
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/over.cpp": "void tune(int x) {} void tune(float x) {}",
    })
    reference = UnresolvedSourceReference(
        symbol="tune", kind="callable", file="src/lib/over.cpp",
        argument_count=1)
    resolver = SourceExpansionResolver(profiler, "hash")
    planned = planned_stages(reference, inputs.structure)
    assert planned, "expected a non-empty stage plan"
    first_sink: list = []
    first_candidates, first_evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, first_sink)
    assert first_candidates == []
    done = {attempt_stage(a.strategy, a.details)
            for a in first_evidence.attempts}
    assert done, "expected recorded stages from the first pass"
    remaining = [stage for stage in planned if stage not in done]
    assert remaining, "expected untried stages after ambiguity"
    # The suppressed pass still executes the underlying search: the
    # query-group search runs even though its record is suppressed.
    searches: list = []
    original_search = profiler.search_related_source_files

    def counting_search(queries, **kwargs):
        searches.append(tuple(queries))
        return original_search(queries, **kwargs)

    monkeypatch.setattr(
        profiler, "search_related_source_files", counting_search)
    second_sink: list = []
    second_candidates, second_evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, second_sink,
        skip_stages=frozenset(done))
    assert searches, (
        "suppressed stages must still execute their underlying search")
    assert [(c.file, c.matched_kind) for c in second_candidates] == [
        (c.file, c.matched_kind) for c in first_candidates], (
        "record suppression must not change candidates")
    rerun_stages = {attempt_stage(a.strategy, a.details)
                    for a in second_evidence.attempts}
    assert rerun_stages.isdisjoint(done), (
        "already-recorded stages must not be recorded again")
    assert rerun_stages <= set(remaining)
    third_sink: list = []
    third_candidates, third_evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, third_sink,
        skip_stages=frozenset(done | rerun_stages))
    assert third_candidates == []
    assert third_evidence.attempts == []


def test_version_bump_clears_suppression(tmp_path):
    """Visited and exhausted suppression is scoped to a search-universe
    version: advancing the version reopens eligibility and prunes old
    progress, while repeat work in an unchanged version settles."""
    from flight_log_agent.analysis.coverage import CoverageSearchState
    state = CoverageSearchState()
    assert state.version == 0
    state.record_stages("obligation-a", {"owner-files"})
    state.mark_exhausted("visit-key-a")
    assert not state.eligible("obligation-a", ["owner-files", "query-group"])
    state.advance_version()
    assert state.version == 1
    assert state.eligible("obligation-a", ["owner-files", "query-group"])
    assert state.exhausted_keys() == set()
    state.record_stages("obligation-a", {"owner-files", "query-group"})
    assert not state.eligible("obligation-a", ["owner-files", "query-group"])
    # Empty-plan kinds still resolve once: unseen keys are always eligible.
    assert state.eligible("fresh-obligation", [])


def test_planned_stages_matches_internal_linkage_path(tmp_path):
    """`planned_stages()` must describe the resolver's actual stages: an
    internal-linkage global is searched via the internal-only stage, never
    via declaration-files or assignment-query stages."""
    from flight_log_agent.analysis.coverage import (
        attempt_stage,
        stage_id,
        CoverageSearchState,
    )
    from flight_log_agent.analysis.source_expansion import (
        planned_stages,
        STRATEGY_STORAGE_INTERNAL_ONLY,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/g.cpp": (
            "static float kgain = 1.0f;\nvoid run() { kgain = 2.0f; }\n"
        ),
    })
    identity = inputs.structure.symbol_identity(
        "kgain", file="src/lib/g.cpp",
        callable_id="run", function_name="run")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/lib/g.cpp",
        callable_id="run", identity=identity)
    planned = planned_stages(reference, inputs.structure)
    assert planned == [stage_id(STRATEGY_STORAGE_INTERNAL_ONLY)], (
        f"internal-linkage plan must be exactly the internal-only stage: {planned}")
    # Cross-fidelity: every stage the resolver can record here is planned.
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    candidates, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink)
    assert [item.file for item in candidates] == ["src/lib/g.cpp"]
    recorded = {attempt_stage(a.strategy, a.details)
                for a in evidence.attempts}
    assert recorded <= set(planned), (
        f"resolver recorded unplanned stages: {recorded - set(planned)}")
    # The plan is progress state, not proof: recording it settles nothing
    # beyond scheduling eligibility.
    state = CoverageSearchState()
    assert state.eligible(reference.visit_key(), planned)


def test_certificate_closed_empty_positive(tmp_path):
    """A proven-closed internal boundary, fully examined with zero writers
    present, certifies absence within that boundary (empty writer set)."""
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/e.cpp": "static float kzero;\nvoid run() { (void)0; }\n",
    })
    identity = inputs.structure.symbol_identity(
        "kzero", file="src/lib/e.cpp",
        callable_id="run", function_name="run")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="kzero", kind="storage_writers", file="src/lib/e.cpp",
        callable_id="run", identity=identity)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    assert evidence.attempts, "expected recorded attempts"
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.refusal == "", f"unexpected refusal: {result.refusal}"
    cert = result.certificate
    assert cert is not None
    assert cert.boundary == ("src/lib/e.cpp",)
    assert cert.writers == ()
    assert cert.version == 0


def test_certificate_refuses_heuristic_search(tmp_path):
    """Heuristic search never certifies: neither one exact admitted writer
    nor a zero-result heuristic search produces a certificate. Exact
    admission is precision, not completeness."""
    from flight_log_agent.analysis.coverage import (
        REFUSAL_HEURISTIC_STRATEGY,
        derive_writer_coverage_certificate,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/plain.cpp": "float output; void run() { output = 2; }",
        "src/lib/empty.cpp": "void run() {}",
    })
    resolver = SourceExpansionResolver(profiler, "hash")
    # One exact writer via an assignment-shape heuristic strategy.
    admitted_ref = UnresolvedSourceReference(
        symbol="output", file="src/lib/plain.cpp")
    sink: list = []
    candidates, admitted_evidence = resolver.resolve_with_evidence(
        admitted_ref, inputs.structure, sink, universe_version=0)
    assert [item.file for item in candidates] == ["src/lib/plain.cpp"]
    admitted_result = derive_writer_coverage_certificate(
        admitted_ref, admitted_evidence, 0)
    assert admitted_result.certificate is None
    assert admitted_result.refusal == REFUSAL_HEURISTIC_STRATEGY
    # Zero-result heuristic search: no query can even be issued.
    empty_ref = UnresolvedSourceReference(
        symbol="", file="src/lib/empty.cpp")
    empty_sink: list = []
    _, empty_evidence = resolver.resolve_with_evidence(
        empty_ref, inputs.structure, empty_sink, universe_version=0)
    empty_result = derive_writer_coverage_certificate(
        empty_ref, empty_evidence, 0)
    assert empty_result.certificate is None
    assert empty_result.refusal == REFUSAL_HEURISTIC_STRATEGY


def test_certificate_refuses_unexamined_boundary_member(tmp_path):
    """One admitted writer plus a boundary member with no admission
    verdict does NOT certify: every examined file must be accounted for."""
    from copy import deepcopy
    from flight_log_agent.analysis.coverage import (
        REFUSAL_UNEXAMINED_BOUNDARY_MEMBER,
        derive_writer_coverage_certificate,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    assert len(evidence.attempts) == 1
    tampered = deepcopy(evidence)
    only = tampered.attempts[0]
    only.examined_domain["files"] = (
        tuple(only.examined_domain["files"]) + ("src/lib/other.cpp",))
    result = derive_writer_coverage_certificate(reference, tampered, 0)
    assert result.certificate is None
    assert result.refusal == REFUSAL_UNEXAMINED_BOUNDARY_MEMBER


def test_certificate_refuses_ambiguity(tmp_path):
    """Ambiguous colliding identity blocks certification, reported
    precisely even when the strategy is otherwise certifiable."""
    from copy import deepcopy
    from flight_log_agent.analysis.coverage import (
        REFUSAL_AMBIGUOUS_CANDIDATES,
        derive_writer_coverage_certificate,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    tampered = deepcopy(evidence)
    only = tampered.attempts[0]
    only.outcome = "ambiguous"
    only.details["colliding_identities"] = ["id-a", "id-b"]
    result = derive_writer_coverage_certificate(reference, tampered, 0)
    assert result.certificate is None
    assert result.refusal == REFUSAL_AMBIGUOUS_CANDIDATES
    # Natural ambiguity (overload set) also refuses, never certifies.
    over_profiler, over_inputs = _evidence_setup(tmp_path, {
        "src/lib/over.cpp": "void tune(int x) {} void tune(float x) {}",
    })
    over_ref = UnresolvedSourceReference(
        symbol="tune", kind="callable", file="src/lib/over.cpp",
        argument_count=1)
    over_resolver = SourceExpansionResolver(over_profiler, "hash")
    over_sink: list = []
    _, over_evidence = over_resolver.resolve_with_evidence(
        over_ref, over_inputs.structure, over_sink, universe_version=0)
    assert any(a.outcome == "ambiguous" for a in over_evidence.attempts)
    over_result = derive_writer_coverage_certificate(
        over_ref, over_evidence, 0)
    assert over_result.certificate is None
    # The evidence holds several independent blockers (partial domain on
    # one stage, ambiguity on another); any explicit refusal is correct,
    # and the rule order is attempt-record order.
    assert over_result.refusal != ""


def test_certificate_refuses_unavailable_search(tmp_path):
    """Search that cannot meaningfully run (unproven identity) blocks
    certification: unexamined is not absent."""
    from flight_log_agent.analysis.coverage import (
        REFUSAL_UNAVAILABLE_SEARCH,
        derive_writer_coverage_certificate,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/user.cpp": "void run() { helper.adjust(1); }",
    })
    reference = UnresolvedSourceReference(
        symbol="adjust", kind="callable", file="src/lib/user.cpp",
        callable_id="run", receiver="helper", receiver_type="",
        argument_count=1)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    assert evidence.attempts
    assert all(a.outcome == "unavailable" for a in evidence.attempts)
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.certificate is None
    assert result.refusal == REFUSAL_UNAVAILABLE_SEARCH


def test_certificate_refuses_partial_domain(tmp_path):
    """A partial-domain result (heuristic search over an incompletely
    enumerated hit set) blocks certification even with zero candidates."""
    from flight_log_agent.analysis.coverage import (
        REFUSAL_PARTIAL_DOMAIN,
        derive_writer_coverage_certificate,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/empty.cpp": "void run() {}",
    })
    reference = UnresolvedSourceReference(
        symbol="NoSuchEntity", kind="class", file="src/lib/empty.cpp")
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    candidates, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    assert candidates == []
    assert any(
        a.outcome == "no-candidate-partial-domain"
        for a in evidence.attempts), (
        "expected a partial-domain outcome to refuse on")
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.certificate is None
    assert result.refusal == REFUSAL_PARTIAL_DOMAIN


def test_certificate_refuses_stale_and_unknown_versions(tmp_path):
    """Certificates are valid only for the stamped current version:
    newer current versions refuse as stale, and unstamped (T1-shape)
    evidence refuses as version-unknown at any version."""
    from flight_log_agent.analysis.coverage import (
        REFUSAL_STALE_VERSION,
        REFUSAL_VERSION_UNKNOWN,
        derive_writer_coverage_certificate,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    stale = derive_writer_coverage_certificate(reference, evidence, 1)
    assert stale.certificate is None
    assert stale.refusal == REFUSAL_STALE_VERSION
    unstamped_sink: list = []
    _, unstamped = resolver.resolve_with_evidence(
        reference, inputs.structure, unstamped_sink)
    unknown = derive_writer_coverage_certificate(reference, unstamped, 0)
    assert unknown.certificate is None
    assert unknown.refusal == REFUSAL_VERSION_UNKNOWN


def test_certificate_cross_declaration_isolation(tmp_path):
    """Evidence for declaration A cannot satisfy obligation B, even with
    identical spelling: identity mismatch refuses."""
    from flight_log_agent.analysis.coverage import (
        REFUSAL_IDENTITY_MISMATCH,
        derive_writer_coverage_certificate,
    )
    profiler, inputs, reference_a = _internal_gain_fixture(tmp_path)
    other_profiler, other_inputs = _evidence_setup(tmp_path, {
        "src/lib/h.cpp": (
            "static float kgain = 9.0f;\nvoid other() { kgain = 3.0f; }\n"
        ),
    })
    other_identity = other_inputs.structure.symbol_identity(
        "kgain", file="src/lib/h.cpp",
        callable_id="other", function_name="other")
    assert other_identity.declaration_proven
    assert (other_identity.declaration_id
            != reference_a.identity.declaration_id)
    reference_b = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/lib/h.cpp",
        callable_id="other", identity=other_identity)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence_a = resolver.resolve_with_evidence(
        reference_a, inputs.structure, sink, universe_version=0)
    result = derive_writer_coverage_certificate(reference_b, evidence_a, 0)
    assert result.certificate is None
    assert result.refusal == REFUSAL_IDENTITY_MISMATCH


def test_certificate_ignores_unrelated_heuristic_evidence(tmp_path):
    """Relevant exact evidence plus unrelated heuristic evidence must not
    broaden the certificate's boundary: the boundary stays exactly the
    closed examined set."""
    from copy import deepcopy
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    plain_profiler, plain_inputs = _evidence_setup(tmp_path, {
        "src/lib/plain.cpp": "float output; void run() { output = 2; }",
    })
    plain_ref = UnresolvedSourceReference(
        symbol="output", file="src/lib/plain.cpp")
    plain_resolver = SourceExpansionResolver(plain_profiler, "hash")
    plain_sink: list = []
    _, plain_evidence = plain_resolver.resolve_with_evidence(
        plain_ref, plain_inputs.structure, plain_sink, universe_version=0)
    assert any(a.outcome == "admitted" for a in plain_evidence.attempts)
    merged = deepcopy(evidence)
    merged.attempts.extend(deepcopy(plain_evidence.attempts))
    result = derive_writer_coverage_certificate(reference, merged, 0)
    assert result.refusal == "", f"unexpected refusal: {result.refusal}"
    assert result.certificate is not None
    assert result.certificate.boundary == ("src/lib/g.cpp",)
    assert len(result.certificate.writers) == 2, (
        "merged heuristic evidence must not narrow the exhaustive census")


def test_certificate_ignores_scheduling_state(tmp_path):
    """D1 independence: derivation reads obligation + evidence + version
    only. Visited/exhausted/completed scheduling state — however set —
    must not change the derivation result."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.source_expansion import planned_stages
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    baseline = derive_writer_coverage_certificate(reference, evidence, 0)
    assert baseline.certificate is not None
    polluted = CoverageSearchState()
    polluted.mark_visited(
        resolver.resolution_key(reference, inputs.structure))
    polluted.mark_exhausted(reference.visit_key())
    polluted.record_stages(
        reference.visit_key(), planned_stages(reference, inputs.structure))
    polluted_result = derive_writer_coverage_certificate(
        reference, evidence, 0)
    assert polluted_result == baseline
    # Polluted scheduling state must not flip a refusal into success
    # either: heuristic evidence still refuses identically.
    plain_profiler, plain_inputs = _evidence_setup(tmp_path, {
        "src/lib/plain.cpp": "float output; void run() { output = 2; }",
    })
    plain_ref = UnresolvedSourceReference(
        symbol="output", file="src/lib/plain.cpp")
    plain_resolver = SourceExpansionResolver(plain_profiler, "hash")
    plain_sink: list = []
    _, plain_evidence = plain_resolver.resolve_with_evidence(
        plain_ref, plain_inputs.structure, plain_sink, universe_version=0)
    plain_baseline = derive_writer_coverage_certificate(
        plain_ref, plain_evidence, 0)
    assert plain_baseline.certificate is None
    polluted.mark_visited(
        plain_resolver.resolution_key(plain_ref, plain_inputs.structure))
    assert (derive_writer_coverage_certificate(
        plain_ref, plain_evidence, 0) == plain_baseline)


def test_certificate_refuses_same_callable_shortcircuit(tmp_path):
    """Same-callable local enumeration cannot certify in T3: the
    short-circuit records no examined domain, and recording one would
    violate evidence honesty (only examined counts) while opening the
    file would change search behavior. Refusal, not silent admission."""
    from flight_log_agent.analysis.coverage import (
        REFUSAL_UNSUPPORTED_CLOSURE,
        STRATEGY_LOCAL_SHORTCIRCUIT,
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceSymbolIdentity,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/local.cpp": "void run() { float temp = 1; sink(temp); }",
    })
    reference = UnresolvedSourceReference(
        symbol="temp", kind="symbol", file="src/lib/local.cpp",
        callable_id="run",
        identity=SourceSymbolIdentity(
            kind="local", symbol="temp", root="temp",
            file="src/lib/local.cpp", callable_id="run",
            declaration_id="run:parameter:0", declaration_proven=True))
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    assert evidence.attempts
    assert {a.strategy for a in evidence.attempts} == {
        STRATEGY_LOCAL_SHORTCIRCUIT}
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.certificate is None
    assert result.refusal == REFUSAL_UNSUPPORTED_CLOSURE


def test_evidence_carries_search_version_when_provided(tmp_path):
    """Version stamping is observational only: attempts record the universe
    version they ran under, and omitting the version keeps the exact T1
    evidence shape."""
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/plain.cpp": "float output; void run() { output = 2; }",
    })
    reference = UnresolvedSourceReference(
        symbol="output", file="src/lib/plain.cpp")
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    candidates, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=3)
    assert [item.file for item in candidates] == ["src/lib/plain.cpp"]
    assert evidence.universe_ref.get("search_version") == 3
    assert sink, "expected recorded attempts"
    assert all(a.universe_ref.get("search_version") == 3 for a in sink)
    legacy_sink: list = []
    _, legacy_evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, legacy_sink)
    assert "search_version" not in legacy_evidence.universe_ref, (
        "omitted version must preserve the exact T1 evidence shape")


# --- T3: pure coverage-certificate derivation (unconsumed data) ---

def _internal_gain_fixture(tmp_path):
    """One internal-linkage global with one exact writer in its file."""
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/g.cpp": (
            "static float kgain = 1.0f;\nvoid run() { kgain = 2.0f; }\n"
        ),
    })
    identity = inputs.structure.symbol_identity(
        "kgain", file="src/lib/g.cpp",
        callable_id="run", function_name="run")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/lib/g.cpp",
        callable_id="run", identity=identity)
    return profiler, inputs, reference


def test_certificate_internal_linkage_positive(tmp_path):
    """A single-file internal-linkage boundary, fully examined with its
    exact writer admitted, certifies: boundary, writers, version, and
    explicit closure assumptions are all bound."""
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.refusal == "", f"unexpected refusal: {result.refusal}"
    cert = result.certificate
    assert cert is not None
    assert cert.version == 0
    assert cert.strategy == "storage-internal-only"
    assert cert.boundary == ("src/lib/g.cpp",)
    assert cert.examined == ("src/lib/g.cpp",)
    assert sorted(cert.writers) == [
        "src/lib/g.cpp:13:25:init_declarator",
        "src/lib/g.cpp:40:52:assignment_expression",
    ]
    # Both same-declaration writers are enumerated: the initializer and
    # the function-body assignment. Writer-list exhaustiveness for the
    # closed boundary is now mechanical, not first-match.
    from flight_log_agent.analysis.mechanism_discovery import load_facts
    check_facts = load_facts(
        profiler, tmp_path / "cache_b", ["src/lib/g.cpp"], "hash")
    assert sum(1 for item in check_facts[0].source_assignments
               if item.target == "kgain") == 2
    assert "file-linkage-closed" in cert.assumptions
    assert cert.obligation_key == evidence.obligation_key
    assert cert.scheduling_key == evidence.scheduling_key


# --- T4: record-only checkpoint threading (diagnostic-only) ---

def _internal_h_certificate(tmp_path):
    """Second same-spelling internal global under another declaration."""
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/h.cpp": (
            "static float kgain = 9.0f;\nvoid other() { kgain = 3.0f; }\n"
        ),
    })
    identity = inputs.structure.symbol_identity(
        "kgain", file="src/lib/h.cpp",
        callable_id="other", function_name="other")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/lib/h.cpp",
        callable_id="other", identity=identity)
    return profiler, inputs, reference


def _derive_for(profiler, inputs, reference, version=0):
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate,
    )
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=version)
    result = derive_writer_coverage_certificate(
        reference, evidence, version)
    assert result.certificate is not None, result.refusal
    return result.certificate


def test_proof_observation_covers_matching_certificate(tmp_path):
    """H: a current certificate for the exact relevant obligation is
    observed as covered, with its semantic obligation key retained."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    cert = _derive_for(profiler, inputs, reference, version=0)
    key = reference.visit_key()
    observation = observe_checkpoint_coverage(
        [reference], certificates=[cert], proof_version=0,
        searchable_keys={key})
    assert observation.relevant_obligation_keys == (key,)
    assert observation.covered_obligation_keys == (key,)
    assert observation.covered_semantic_keys == (cert.obligation_key,)
    assert observation.uncovered_obligation_keys == ()
    assert observation.stale_certificate_keys == ()
    assert observation.proof_version == 0
    assert observation.relevant_empty is False
    assert observation.scope_degenerate is False


def test_proof_observation_isolates_declarations(tmp_path):
    """B: same spelling, different proven declaration — certificate A
    covers only A; B stays uncovered."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    profiler, inputs, reference_a = _internal_gain_fixture(tmp_path)
    other_profiler, other_inputs, reference_b = _internal_h_certificate(
        tmp_path)
    assert (reference_b.identity.declaration_id
            != reference_a.identity.declaration_id)
    cert_a = _derive_for(profiler, inputs, reference_a, version=0)
    key_a, key_b = reference_a.visit_key(), reference_b.visit_key()
    assert key_a != key_b
    observation = observe_checkpoint_coverage(
        [reference_a, reference_b], certificates=[cert_a], proof_version=0,
        searchable_keys={key_a, key_b})
    assert set(observation.relevant_obligation_keys) == {key_a, key_b}
    assert observation.covered_obligation_keys == (key_a,)
    assert observation.uncovered_obligation_keys == (key_b,)


def test_proof_observation_excludes_unrelated_certificate(tmp_path):
    """C: a certificate outside the relevance scope is excluded — not
    covered, not stale, not associated."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    other_profiler, other_inputs, other_ref = _internal_h_certificate(
        tmp_path)
    foreign = _derive_for(
        other_profiler, other_inputs, other_ref, version=0)
    key = reference.visit_key()
    observation = observe_checkpoint_coverage(
        [reference], certificates=[foreign], proof_version=0,
        searchable_keys={key})
    assert observation.covered_obligation_keys == ()
    assert observation.uncovered_obligation_keys == (key,)
    assert observation.stale_certificate_keys == ()
    assert foreign.scheduling_key not in (
        observation.covered_obligation_keys)


def test_proof_observation_ignores_stale_certificate(tmp_path):
    """D: a version-N certificate under current version N+1 records no
    current coverage — the obligation stays uncovered and the stale key
    is reported diagnostically."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    cert = _derive_for(profiler, inputs, reference, version=0)
    key = reference.visit_key()
    observation = observe_checkpoint_coverage(
        [reference], certificates=[cert], proof_version=1,
        searchable_keys={key})
    assert observation.covered_obligation_keys == ()
    assert observation.uncovered_obligation_keys == (key,)
    assert observation.stale_certificate_keys == (key,)


def test_proof_observation_keeps_filtered_relevant(tmp_path):
    """E: relevance is computed before scheduler filtering — a filtered
    obligation stays relevant (and uncovered without a certificate, or
    covered with one). Filtering never creates proof."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    key = reference.visit_key()
    uncovered = observe_checkpoint_coverage(
        [reference], certificates=[], proof_version=0, searchable_keys=set())
    assert uncovered.relevant_obligation_keys == (key,)
    assert uncovered.filtered_obligation_keys == (key,)
    assert uncovered.uncovered_obligation_keys == (key,)
    assert uncovered.covered_obligation_keys == ()
    cert = _derive_for(profiler, inputs, reference, version=0)
    covered = observe_checkpoint_coverage(
        [reference], certificates=[cert], proof_version=0,
        searchable_keys=set())
    assert covered.relevant_obligation_keys == (key,)
    assert covered.filtered_obligation_keys == (key,)
    assert covered.covered_obligation_keys == (key,)
    assert covered.uncovered_obligation_keys == ()


def test_proof_observation_marks_vacuity(tmp_path):
    """G: diagnostics distinguish genuinely-no-obligations from a
    degenerate scope; neither shape certifies anything by itself."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    empty = observe_checkpoint_coverage(
        [], certificates=[], proof_version=0, scope_degenerate=False)
    assert empty.relevant_obligation_keys == ()
    assert empty.covered_obligation_keys == ()
    assert empty.relevant_empty is True
    assert empty.scope_degenerate is False
    degenerate = observe_checkpoint_coverage(
        [], certificates=[], proof_version=0, scope_degenerate=True)
    assert degenerate.relevant_empty is True
    assert degenerate.scope_degenerate is True
    assert degenerate.covered_obligation_keys == ()


def test_proof_observation_marks_vacuity(tmp_path):
    """G: diagnostics distinguish genuinely-no-obligations from a
    degenerate scope; neither shape certifies anything by itself."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    empty = observe_checkpoint_coverage(
        [], certificates=[], proof_version=0, scope_degenerate=False)
    assert empty.relevant_obligation_keys == ()
    assert empty.covered_obligation_keys == ()
    assert empty.relevant_empty is True
    assert empty.scope_degenerate is False
    degenerate = observe_checkpoint_coverage(
        [], certificates=[], proof_version=0, scope_degenerate=True)
    assert degenerate.relevant_empty is True
    assert degenerate.scope_degenerate is True
    assert degenerate.covered_obligation_keys == ()


def _round_dag_with_reference(tmp_path):
    """Hand-built DAG mirroring the checkpoint-controller fixture, with
    one real proven internal reference attached at the terminal root."""
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    def expression(text, *inputs):
        return {"text": text, "lowered_text": text,
                "input_symbols": list(inputs),
                "input_identities": {}, "call_results": [], "exact": True}

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
    policies = {signal: {"method": "linear"} for signal in samples}
    dag = build_mechanism_dag(bindings, "command.value",
                              logged_signals=set(samples))
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    _, _, reference = _internal_gain_fixture(tmp_path)
    attached = reference.model_copy(
        update={"origin_vertex_ids": [root.id]})
    assert attached.visit_key() == reference.visit_key()
    dag.unresolved_references.append(attached)
    return dag, samples, policies, attached


def _run_round(dag, samples, policies, **kwargs):
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_checkpoint_round,
    )
    return evaluate_checkpoint_round(
        dag, parameter_values={}, observed_signals=set(samples),
        signal_policies=policies,
        load_samples=lambda _view, _observed: samples, **kwargs)


def _behavior_surface(result):
    selected = result.summary.get("selected_checkpoint") or {}
    return {
        "action": result.action,
        "references": sorted(r.visit_key() for r in result.references),
        "next_analysis": {
            key: value for key, value in
            (result.summary.get("next_analysis") or {}).items()
            if key != "source_requests"
        },
        "next_requests": sorted(
            raw.get("symbol", "") + ":" + raw.get("kind", "")
            for raw in (result.summary.get("next_analysis") or {}).get(
                "source_requests", [])),
        "stop": selected.get("authorizes_discovery_stop", False),
        "writer_flag": selected.get("writer_coverage_verified", False),
        "applicability_flag": selected.get(
            "applicability_verified", False),
        "summary_keys": sorted(result.summary.keys()),
    }


def test_round_observes_certificate_without_behavior_change(tmp_path):
    """A: the round observes a current matching certificate
    diagnostically; scheduling, requirements, flags, and DAG mutation
    behavior are identical with and without threading. Stop behavior
    now follows Gate-B discharge plus replay: the valid shape verifies
    while observation stays record-identical otherwise."""
    from copy import deepcopy
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    cert = _derive_for(profiler, inputs, reference, version=0)
    dag, samples, policies, attached = _round_dag_with_reference(tmp_path)
    key = attached.visit_key()
    assert key == cert.scheduling_key
    before = deepcopy(dag.model_dump())
    plain = _run_round(dag, samples, policies)
    assert plain.proof_observation is not None
    assert plain.proof_observation.relevant_obligation_keys == (key,)
    assert plain.proof_observation.covered_obligation_keys == ()
    threaded = _run_round(
        dag, samples, policies, coverage_certificates=[cert],
        proof_version=0)
    assert threaded.proof_observation.covered_obligation_keys == (key,)
    assert (threaded.proof_observation.covered_semantic_keys
            == (cert.obligation_key,))
    assert threaded.proof_observation.uncovered_obligation_keys == ()
    assert dag.model_dump() == before
    plain_surface = _behavior_surface(plain)
    threaded_surface = _behavior_surface(threaded)
    for field in ("references", "next_analysis", "next_requests",
                  "writer_flag", "applicability_flag", "summary_keys"):
        assert threaded_surface[field] == plain_surface[field]
    assert threaded_surface["action"] == "verified"
    assert threaded_surface["stop"] is True
    assert plain_surface["action"] != "verified"
    assert plain_surface["stop"] is False


def test_round_keeps_exhausted_relevant_without_covering(tmp_path):
    """F: an exhausted relevant obligation stays relevant (and uncovered
    without a current certificate) while existing scheduler filtering
    still drops it from the search queue; stop stays false."""
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    cert = _derive_for(profiler, inputs, reference, version=0)
    dag, samples, policies, attached = _round_dag_with_reference(tmp_path)
    key = attached.visit_key()
    dag.exhausted_source_requests.add(key)
    uncovered = _run_round(dag, samples, policies)
    assert key in uncovered.proof_observation.relevant_obligation_keys
    assert key in uncovered.proof_observation.filtered_obligation_keys
    assert key in uncovered.proof_observation.uncovered_obligation_keys
    assert key not in [r.visit_key() for r in uncovered.references]
    covered = _run_round(
        dag, samples, policies, coverage_certificates=[cert],
        proof_version=0)
    assert key in covered.proof_observation.relevant_obligation_keys
    assert key in covered.proof_observation.filtered_obligation_keys
    assert key in covered.proof_observation.covered_obligation_keys
    # Exhaustion stays scheduling-only: with genuine coverage the
    # covered round verifies through proof (not through the queue),
    # while the uncovered round still cannot.
    assert covered.action == "verified"
    assert covered.summary.get("selected_checkpoint", {}).get(
        "authorizes_discovery_stop", False) is True
    assert uncovered.action != "verified"


def test_round_differential_across_proof_shapes(tmp_path):
    """I: across valid / stale / unrelated / uncovered proof
    threading, observation distinguishes the shapes while behavior
    follows proof truth: only the valid shape verifies (through
    coverage, never through queue/scheduling state); the other three
    remain behaviorally identical in refusal."""
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    cert = _derive_for(profiler, inputs, reference, version=0)
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    cert = _derive_for(profiler, inputs, reference, version=0)
    other_profiler, other_inputs, other_ref = _internal_h_certificate(
        tmp_path)
    foreign = _derive_for(
        other_profiler, other_inputs, other_ref, version=0)
    observations = {}
    surfaces = {}
    scenarios = {}
    dag, samples, policies, attached = _round_dag_with_reference(tmp_path)
    scenarios["uncovered"] = {}
    scenarios["valid"] = {"coverage_certificates": [cert],
                          "proof_version": 0}
    scenarios["stale"] = {"coverage_certificates": [cert],
                          "proof_version": 1}
    scenarios["unrelated"] = {"coverage_certificates": [foreign],
                              "proof_version": 0}
    key = attached.visit_key()
    for name, kwargs in scenarios.items():
        result = _run_round(dag, samples, policies, **kwargs)
        observations[name] = result.proof_observation
        surfaces[name] = _behavior_surface(result)
    assert observations["valid"].covered_obligation_keys == (key,)
    assert observations["stale"].covered_obligation_keys == ()
    assert observations["stale"].stale_certificate_keys == (key,)
    assert observations["unrelated"].covered_obligation_keys == ()
    assert observations["unrelated"].stale_certificate_keys == ()
    assert observations["uncovered"].covered_obligation_keys == ()
    assert surfaces["valid"]["action"] == "verified"
    assert surfaces["valid"]["stop"] is True
    assert (surfaces["stale"] == surfaces["unrelated"]
            == surfaces["uncovered"])
    assert surfaces["stale"]["action"] != "verified"
    assert surfaces["stale"]["stop"] is False
    # Vacuous scope: no references at all, foreign proof threaded.
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    def expression(text, *inputs):
        return {"text": text, "lowered_text": text,
                "input_symbols": list(inputs),
                "input_identities": {}, "call_results": [], "exact": True}

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
    policies = {signal: {"method": "linear"} for signal in samples}
    bare = build_mechanism_dag(bindings, "command.value",
                               logged_signals=set(samples))
    assert bare.unresolved_references == []
    plain_vacuous = _run_round(bare, samples, policies)
    threaded_vacuous = _run_round(
        bare, samples, policies, coverage_certificates=[foreign],
        proof_version=0)
    assert plain_vacuous.proof_observation.relevant_empty is True
    assert threaded_vacuous.proof_observation.relevant_empty is True
    assert threaded_vacuous.proof_observation.covered_obligation_keys == ()
    assert (_behavior_surface(threaded_vacuous)
            == _behavior_surface(plain_vacuous))


# --- T5: positive per-use applicability without stop activation ---

def _single_writer_use_tree(tmp_path):
    """Internal global with exactly one writer (its initializer) plus one
    straight-line consumer; the consumer's operand is fed by exactly
    that writer vertex."""
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _mini_tree(tmp_path, {
        "src/main.cpp": (
            "static float kgain = 1.0f;\n"
            "float out;\n"
            "void use() { out = kgain * 3.0f; }\n"
        ),
    }, backend="tree_sitter")
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["use"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp")
    assert result.dag is not None
    return profiler, result


def _use_producer_edge(dag, use_id, operand):
    return [edge for edge in dag.edges
            if edge.target_id == use_id and edge.kind == "data"
            and edge.role == operand]


def test_applicability_single_exact_producer_positive(tmp_path):
    """A: one covered writer exactly ordered to one unconditional use,
    with no competitor and no uncertainty, proves applicability with an
    explicit positive basis."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER,
        derive_writer_applicability,
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
        UnresolvedSourceReference,
    )
    profiler, result = _single_writer_use_tree(tmp_path)
    dag, structure = result.dag, result.inputs.structure
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    operand = "kgain"
    feeding = _use_producer_edge(dag, use.id, operand)
    assert len(feeding) == 1, (
        "fixture must feed the operand from exactly one writer")
    producer = next(vertex for vertex in dag.vertices
                    if vertex.id == feeding[0].source_id)
    assert producer.kind == "operation"
    assert not [edge for edge in dag.edges
                if edge.target_id in {use.id, producer.id}
                and edge.kind == "control"], (
        "fixture use and producer must be unconditional")
    # The live frontier reference for this obligation carries the use
    # as its origin: the honest use+obligation bundle, not hand-built.
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    assert use.id in reference.origin_vertex_ids
    assert operand in reference.origin_operands
    assert reference.identity is not None
    assert reference.identity.declaration_proven
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, structure, sink, universe_version=0)
    coverage = derive_writer_coverage_certificate(reference, evidence, 0)
    assert coverage.certificate is not None, coverage.refusal
    cert = coverage.certificate
    assert list(cert.writers) == [
        producer.metadata["source_site_id"]]
    outcome = derive_writer_applicability(
        reference, use.id, operand, cert, dag, 0,
        conditional_writer_ids=())
    assert outcome.refusal == "", f"unexpected refusal: {outcome.refusal}"
    proof = outcome.proof
    assert proof is not None
    assert proof.basis == APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER
    assert proof.use_key == (use.id, operand)
    assert proof.producer_vertex == producer.id
    assert proof.version == 0
    assert proof.scheduling_key == reference.visit_key()


def _two_writer_use_tree(tmp_path):
    """Same declaration written twice (initializer plus assignment): the
    internal certificate lists the first admitted site while the use's
    flow shows two structural producers with unknown runtime order."""
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _mini_tree(tmp_path, {
        "src/main.cpp": (
            "static float kgain = 1.0f;\n"
            "void run() { kgain = 2.0f; }\n"
            "float out;\n"
            "void use() { out = kgain * 3.0f; }\n"
        ),
    }, backend="tree_sitter")
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["use"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp")
    assert result.dag is not None
    return profiler, result


def _derive_use_certificate(profiler, structure, reference, version=0):
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, structure, sink, universe_version=version)
    result = derive_writer_coverage_certificate(
        reference, evidence, version)
    assert result.certificate is not None, result.refusal
    return result.certificate


def test_applicability_coverage_alone_is_insufficient(tmp_path):
    """B: a valid, now-exhaustive certificate still proves nothing when
    the use's flow shows an unresolved placeholder instead of a wired
    producer — the builder itself wires no producer when two
    same-declaration writers compete, so coverage alone cannot pick a
    winner."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_NO_POSITIVE_BASIS,
        derive_writer_applicability,
    )
    profiler, result = _two_writer_use_tree(tmp_path)
    dag, structure = result.dag, result.inputs.structure
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    operand = "kgain"
    feeding = _use_producer_edge(dag, use.id, operand)
    assert len(feeding) == 1
    feeder = next(vertex for vertex in dag.vertices
                  if vertex.id == feeding[0].source_id)
    assert feeder.kind == "evidence", (
        "ambiguous same-declaration writers must leave an unresolved "
        "placeholder, never a guessed producer")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, structure, reference, version=0)
    assert len(cert.writers) == 2, (
        "exhaustive certificate must list both real writers")
    outcome = derive_writer_applicability(
        reference, use.id, operand, cert, dag, 0,
        conditional_writer_ids=())
    assert outcome.proof is None
    assert outcome.refusal == APPLICABILITY_NO_POSITIVE_BASIS


def test_applicability_guarded_use_refuses(tmp_path):
    """B/C: a control-gated use refuses even with a valid single-writer
    certificate — and an empty conditional set changes nothing, because
    absence of the marker is not positive evidence."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_UNRESOLVED_CONTROL,
        derive_writer_applicability,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _mini_tree(tmp_path, {
        "src/main.cpp": (
            "static float kgain = 1.0f;\n"
            "float out2;\n"
            "int cond;\n"
            "void use2() { if (cond) { out2 = kgain * 3.0f; } }\n"
        ),
    }, backend="tree_sitter")
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["use2"], terminal="out2",
        source_hash="hash", terminal_file="src/main.cpp")
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out2" and vertex.kind == "operation")
    operand = "kgain"
    assert any(edge.kind == "control" and edge.target_id == use.id
               for edge in dag.edges), "fixture use must be control-gated"
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, result.inputs.structure, reference, version=0)
    assert len(cert.writers) == 1
    outcome = derive_writer_applicability(
        reference, use.id, operand, cert, dag, 0,
        conditional_writer_ids=())
    assert outcome.proof is None
    assert outcome.refusal == APPLICABILITY_UNRESOLVED_CONTROL


def test_applicability_conditional_writer_blocks(tmp_path):
    """D: the otherwise-positive case refuses once the covered producer
    is marked conditional by evaluation."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_CONDITIONAL_WRITER,
        derive_writer_applicability,
    )
    profiler, result = _single_writer_use_tree(tmp_path)
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    operand = "kgain"
    producer = next(vertex for vertex in dag.vertices
                    if vertex.id == _use_producer_edge(
                        dag, use.id, operand)[0].source_id)
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, result.inputs.structure, reference, version=0)
    outcome = derive_writer_applicability(
        reference, use.id, operand, cert, dag, 0,
        conditional_writer_ids={producer.id})
    assert outcome.proof is None
    assert outcome.refusal == APPLICABILITY_CONDITIONAL_WRITER


def _split_use_tree(tmp_path):
    """One declaration, two consumers: a straight-line use and a
    branch-guarded use of the same internal global."""
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    files = {
        "src/main.cpp": (
            "static float kgain = 1.0f;\n"
            "float out_a;\n"
            "float out_b;\n"
            "int cond;\n"
            "void usea() { out_a = kgain * 3.0f; }\n"
            "void useb() { if (cond) { out_b = kgain * 3.0f; } }\n"
        ),
    }
    profiler_a = _mini_tree(tmp_path, files, backend="tree_sitter")
    result_a = discover_mechanism_dag(
        profiler_a, tmp_path / "cache_a", seeds=["usea"],
        terminal="out_a", source_hash="hash",
        terminal_file="src/main.cpp")
    profiler_b = _mini_tree(tmp_path, files, backend="tree_sitter")
    result_b = discover_mechanism_dag(
        profiler_b, tmp_path / "cache_b", seeds=["useb"],
        terminal="out_b", source_hash="hash",
        terminal_file="src/main.cpp")
    assert result_a.dag is not None and result_b.dag is not None
    return (profiler_a, result_a), (profiler_b, result_b)


def test_applicability_splits_shared_declaration_by_use(tmp_path):
    """E: two uses share one declaration and one coverage certificate —
    the straight-line use proves applicable while the guarded use does
    not. Coverage may be shared; applicability may not."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER,
        APPLICABILITY_UNRESOLVED_CONTROL,
        derive_writer_applicability,
    )
    (profiler_a, result_a), (_profiler_b, result_b) = _split_use_tree(
        tmp_path)
    dag_a, dag_b = result_a.dag, result_b.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    use_b = next(vertex for vertex in dag_b.vertices
                 if vertex.variable == "out_b"
                 and vertex.kind == "operation")
    assert use_a.id != use_b.id
    assert any(edge.kind == "control" and edge.target_id == use_b.id
               for edge in dag_b.edges)
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    reference_b = next(item for item in dag_b.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    # Same semantic obligation across scheduling identities: visit keys
    # carry use-site context (consumer callable) and therefore differ,
    # while one shared certificate covers the declaration for both uses.
    # Per-use splitting starts here.
    assert reference_a.visit_key() != reference_b.visit_key()
    assert (use_a.id, "kgain") != (use_b.id, "kgain")
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    positive = derive_writer_applicability(
        reference_a, use_a.id, "kgain", cert, dag_a, 0,
        conditional_writer_ids=())
    assert positive.refusal == "", positive.refusal
    assert positive.proof is not None
    assert positive.proof.basis == APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER
    refused = derive_writer_applicability(
        reference_b, use_b.id, "kgain", cert, dag_b, 0,
        conditional_writer_ids=())
    assert refused.proof is None
    assert refused.refusal == APPLICABILITY_UNRESOLVED_CONTROL


def test_applicability_history_tainted_producer_refuses(tmp_path):
    """F: a loop-carried producer refuses even with a well-formed
    certificate: its control edge and inexact reachability mark ordering
    and invocation as requiring history reasoning T5 does not perform.
    The staged certificate is counterfactual hardening — current
    admission cannot cover loop-body static writes, so no natural
    certificate exists here; the co-asserted natural derivation refuses
    without ever certifying."""
    from dataclasses import replace
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_UNRESOLVED_CONTROL,
        WriterCoverageCertificate,
        derive_writer_applicability,
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler = _mini_tree(tmp_path, {
        "src/main.cpp": (
            "float acc;\n"
            "float out;\n"
            "void run() {\n"
            "  for (int i = 0; i < 3; i++) { acc = acc + i; }\n"
            "  out = acc * 2.0f;\n"
            "}\n"
        ),
    }, backend="tree_sitter")
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["run"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp")
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    producer = next(vertex for vertex in dag.vertices
                    if vertex.id == _use_producer_edge(
                        dag, use.id, "acc")[0].source_id)
    assert any(edge.kind == "control" and edge.target_id == producer.id
               for edge in dag.edges), (
        "loop-carried producer must be control-gated")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "acc"
                     and "acc" in item.origin_operands)
    # Natural coverage is unavailable for this shape, so no valid
    # certificate can exist to consume here.
    natural_resolver = SourceExpansionResolver(profiler, "hash")
    natural_sink: list = []
    _, natural_evidence = natural_resolver.resolve_with_evidence(
        reference, result.inputs.structure, natural_sink,
        universe_version=0)
    natural = derive_writer_coverage_certificate(
        reference, natural_evidence, 0)
    assert natural.certificate is None
    staged = WriterCoverageCertificate(
        obligation_key=("storage_writers", "global",
                        "global:external:acc"),
        scheduling_key=tuple(reference.visit_key()),
        declaration=("global", "", "acc"),
        receiver_context=("", "", ""),
        strategy="storage-internal-only",
        boundary=("src/main.cpp",),
        examined=("src/main.cpp",),
        version=0,
        assumptions=("file-linkage-closed",
                     "writer-syntax-enumerated"),
        writers=(producer.metadata["source_site_id"],),
    )
    outcome = derive_writer_applicability(
        reference, use.id, "acc", staged, dag, 0,
        conditional_writer_ids=())
    assert outcome.proof is None
    assert outcome.refusal == APPLICABILITY_UNRESOLVED_CONTROL


def test_applicability_use_mismatch_refuses(tmp_path):
    """G: a certificate for obligation A cannot prove applicability for
    an unrelated obligation B, even with identical spelling."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_USE_IDENTITY_MISMATCH,
        derive_writer_applicability,
    )
    profiler, result = _single_writer_use_tree(tmp_path)
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, result.inputs.structure, reference, version=0)
    _other_profiler, other_inputs = _evidence_setup(tmp_path, {
        "src/lib/h.cpp": (
            "static float kgain = 9.0f;\nvoid other() { kgain = 3.0f; }\n"
        ),
    })
    other_identity = other_inputs.structure.symbol_identity(
        "kgain", file="src/lib/h.cpp",
        callable_id="other", function_name="other")
    other_ref = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/lib/h.cpp",
        callable_id="other", identity=other_identity)
    assert other_ref.visit_key() != reference.visit_key()
    outcome = derive_writer_applicability(
        other_ref, use.id, "kgain", cert, dag, 0,
        conditional_writer_ids=())
    assert outcome.proof is None
    assert outcome.refusal == APPLICABILITY_USE_IDENTITY_MISMATCH


def test_applicability_stale_coverage_refuses(tmp_path):
    """H: a version-0 certificate cannot prove applicability under
    current version 1."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_STALE_COVERAGE,
        derive_writer_applicability,
    )
    profiler, result = _single_writer_use_tree(tmp_path)
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, result.inputs.structure, reference, version=0)
    outcome = derive_writer_applicability(
        reference, use.id, "kgain", cert, dag, 1,
        conditional_writer_ids=())
    assert outcome.proof is None
    assert outcome.refusal == APPLICABILITY_STALE_COVERAGE


def test_applicability_writer_set_mismatch_refuses(tmp_path):
    """I: applicability evidence for one writer set cannot be reused
    when the use's flow shows another — and an absence certificate
    proves nothing for a use that needs a value."""
    from dataclasses import replace
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_NO_COVERED_PRODUCER,
        APPLICABILITY_WRITER_SET_MISMATCH,
        derive_writer_applicability,
    )
    profiler, result = _single_writer_use_tree(tmp_path)
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, result.inputs.structure, reference, version=0)
    foreign = replace(
        cert, writers=("elsewhere.cpp:1:2:assignment_expression",))
    mismatch = derive_writer_applicability(
        reference, use.id, "kgain", foreign, dag, 0,
        conditional_writer_ids=())
    assert mismatch.proof is None
    assert mismatch.refusal == APPLICABILITY_WRITER_SET_MISMATCH
    absent = replace(cert, writers=())
    no_producer = derive_writer_applicability(
        reference, use.id, "kgain", absent, dag, 0,
        conditional_writer_ids=())
    assert no_producer.proof is None
    assert no_producer.refusal == APPLICABILITY_NO_COVERED_PRODUCER


def test_applicability_ignores_scheduling_state(tmp_path):
    """J: derivation reads use + certificate + graph facts only.
    Polluting visited/exhausted/completed scheduler state and the DAG's
    own exhausted set must leave the result identical."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        derive_writer_applicability,
    )
    profiler, result = _single_writer_use_tree(tmp_path)
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, result.inputs.structure, reference, version=0)
    baseline = derive_writer_applicability(
        reference, use.id, "kgain", cert, dag, 0,
        conditional_writer_ids=())
    assert baseline.proof is not None
    polluted = CoverageSearchState()
    polluted.mark_visited(reference.visit_key())
    polluted.mark_exhausted(reference.visit_key())
    polluted.record_stages(reference.visit_key(), {"owner-files"})
    polluted.advance_version()
    dag.exhausted_source_requests.add(reference.visit_key())
    assert (derive_writer_applicability(
        reference, use.id, "kgain", cert, dag, 0,
        conditional_writer_ids=()) == baseline)


def test_applicability_leaves_stop_unchanged(tmp_path):
    """K: a positive applicability proof changes nothing downstream —
    the checkpoint round surface is identical before and after
    derivation runs, and stop authorization is untouched."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_checkpoint_round,
    )
    from flight_log_agent.analysis.coverage import derive_writer_applicability
    profiler, result = _single_writer_use_tree(tmp_path)
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out" and vertex.kind == "operation")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, result.inputs.structure, reference, version=0)

    def surface(round_result):
        selected = round_result.summary.get("selected_checkpoint") or {}
        return {
            "action": round_result.action,
            "references": sorted(
                item.visit_key() for item in round_result.references),
            "next_analysis": round_result.summary.get("next_analysis"),
            "stop": selected.get("authorizes_discovery_stop", False),
            "writer_flag": selected.get(
                "writer_coverage_verified", False),
            "applicability_flag": selected.get(
                "applicability_verified", False),
        }

    def run_round():
        return evaluate_checkpoint_round(
            dag, parameter_values={}, observed_signals=set(),
            signal_policies={},
            load_samples=lambda _view, _observed: {})

    before = surface(run_round())
    positive = derive_writer_applicability(
        reference, use.id, "kgain", cert, dag, 0,
        conditional_writer_ids=())
    assert positive.proof is not None
    assert surface(run_round()) == before


# --- Pre-T6A: internal-linkage global admission identity correction ---

def _static_gain_update_tree(tmp_path):
    """File-static global written only by a function body: the admission
    census must mint the same proven identity as full extraction."""
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/sg.cpp": "static int gain;\nvoid update() { gain = 5; }\n",
    })
    identity = inputs.structure.symbol_identity(
        "gain", file="src/lib/sg.cpp",
        callable_id=next(key for key, item in
                         inputs.structure.callables_by_id.items()
                         if item.get("name") == "update"),
        function_name="update")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="gain", kind="storage_writers", file="src/lib/sg.cpp",
        callable_id=identity.callable_id, identity=identity)
    return profiler, inputs, reference


def test_admission_identity_matches_full_extraction(tmp_path):
    """RED1: the admission-time declaration identity for a file-static
    global must equal the full-extraction identity — same proven
    declaration entity, including the translation-unit component."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs, reference = _static_gain_update_tree(tmp_path)
    facts = load_facts(
        profiler, tmp_path / "cache", ["src/lib/sg.cpp"], "hash")
    full_identity = facts[0].source_assignments[0].target_identity
    assert full_identity.declaration_proven
    assert full_identity.kind == "global"
    resolver = SourceExpansionResolver(profiler, "hash")
    admission_index = resolver._admission_index_for("src/lib/sg.cpp")
    admission_identity = next(
        item.identity for item in admission_index.assignments
        if item.target == "gain")
    assert admission_identity is not None
    assert admission_identity.declaration_proven
    assert admission_identity.kind == "global"
    assert (admission_identity.declaration_id
            == full_identity.declaration_id)
    assert (admission_identity.declaration_id
            == "global:internal:src/lib/sg.cpp:gain")


def test_function_body_static_writer_is_admitted(tmp_path):
    """RED2: the function-body write must index-match as a writer of the
    exact internal-linkage global — admitted on identity, never via a
    bare-name fallback."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs, reference = _static_gain_update_tree(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    candidates, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    assert [item.file for item in candidates] == ["src/lib/sg.cpp"]
    assert [attempt.strategy for attempt in evidence.attempts] == [
        "storage-internal-only"]
    assert all(attempt.outcome == "admitted"
               for attempt in evidence.attempts)


def test_no_false_closed_empty_for_function_body_writer(tmp_path):
    """RED3 (T6A safety): where the only writer is the function-body
    write, the T3 certificate must list that real writer — never an
    empty writer set produced by an identity mismatch hiding it."""
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs, reference = _static_gain_update_tree(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.refusal == "", result.refusal
    assert result.certificate is not None
    assert result.certificate.writers != (), (
        "closed-empty must not certify while a real writer exists")


# --- Pre-T6A: exhaustive writer census for closed boundaries ---

def _two_writer_gain_tree(tmp_path):
    """One closed file with two proven same-declaration writers: the
    initializer and the function-body assignment."""
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/tw.cpp": (
            "static int gain = 1;\nvoid update() { gain = 5; }\n"
        ),
    })
    identity = inputs.structure.symbol_identity(
        "gain", file="src/lib/tw.cpp",
        callable_id=next(key for key, item in
                         inputs.structure.callables_by_id.items()
                         if item.get("name") == "update"),
        function_name="update")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="gain", kind="storage_writers", file="src/lib/tw.cpp",
        callable_id=identity.callable_id, identity=identity)
    return profiler, inputs, reference


def test_writer_census_lists_all_same_declaration_sites(tmp_path):
    """RED1: the recorded census for a closed file must enumerate every
    proven same-declaration writer site — initializer and function-body
    assignment alike — not just the first exact match."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs, reference = _two_writer_gain_tree(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    assert len(evidence.attempts) == 1
    census = dict(evidence.attempts[0].details.get("writer_census") or {})
    assert set(census.get("src/lib/tw.cpp", ())) == {
        "src/lib/tw.cpp:11:19:init_declarator",
        "src/lib/tw.cpp:37:45:assignment_expression",
    }


def test_certificate_writers_are_exhaustive_for_closed_file(tmp_path):
    """RED2 (T6A safety contract): certificate.writers must equal all
    proven same-declaration writer sites inside the closed file — the
    set T6A must preserve after retiring the obligation."""
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs, reference = _two_writer_gain_tree(tmp_path)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.refusal == "", result.refusal
    assert result.certificate is not None
    assert set(result.certificate.writers) == {
        "src/lib/tw.cpp:11:19:init_declarator",
        "src/lib/tw.cpp:37:45:assignment_expression",
    }
    # Legacy admission still selects the file once, first match first.
    assert [item.file for item in
            resolver.resolve(reference, inputs.structure)] == [
        "src/lib/tw.cpp"]


def test_writer_census_empty_for_writerless_closed_file(tmp_path):
    """RED3: a supported census over a closed file with zero writers is
    empty — and must not manufacture writers. Guards the census change
    against over-collection."""
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/e.cpp": "static float kzero;\nvoid run() { (void)0; }\n",
    })
    identity = inputs.structure.symbol_identity(
        "kzero", file="src/lib/e.cpp",
        callable_id=next(key for key, item in
                         inputs.structure.callables_by_id.items()
                         if item.get("name") == "run"),
        function_name="run")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="kzero", kind="storage_writers", file="src/lib/e.cpp",
        callable_id=identity.callable_id, identity=identity)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    census = dict(evidence.attempts[0].details.get("writer_census") or {})
    assert census == {"src/lib/e.cpp": []}, census
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.refusal == "", result.refusal
    assert result.certificate is not None
    assert result.certificate.writers == ()


def test_writer_census_excludes_unrelated_writers(tmp_path):
    """RED4: one file holding writes to two distinct static globals —
    the census for `gain` must include only proven `gain` writers, never
    a file-wide indiscriminate collection."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/two.cpp": (
            "static int gain;\nstatic int other;\n"
            "void update() { gain = 5; other = 7; }\n"
        ),
    })
    identity = inputs.structure.symbol_identity(
        "gain", file="src/lib/two.cpp",
        callable_id=next(key for key, item in
                         inputs.structure.callables_by_id.items()
                         if item.get("name") == "update"),
        function_name="update")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="gain", kind="storage_writers", file="src/lib/two.cpp",
        callable_id=identity.callable_id, identity=identity)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    candidates, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    assert [item.file for item in candidates] == ["src/lib/two.cpp"]
    census = dict(evidence.attempts[0].details.get("writer_census") or {})
    assert set(census) == {"src/lib/two.cpp"}
    assert len(census["src/lib/two.cpp"]) == 1
    assert "other" not in str(census["src/lib/two.cpp"])


def test_writer_census_isolates_same_spelling_declarations(tmp_path):
    """RED5: same spelling under two proven declarations in different
    files — each obligation's census contains only its own file's
    writers. No spelling equality anywhere."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/g.cpp": "static int gain = 1;\nvoid run() { gain = 2; }\n",
        "src/lib/h.cpp": "static int gain = 9;\nvoid other() { gain = 3; }\n",
    })
    resolver = SourceExpansionResolver(profiler, "hash")
    seen: dict = {}
    for filename, callable_name in (("src/lib/g.cpp", "run"),
                                    ("src/lib/h.cpp", "other")):
        identity = inputs.structure.symbol_identity(
            "gain", file=filename,
            callable_id=next(key for key, item in
                             inputs.structure.callables_by_id.items()
                             if item.get("name") == callable_name),
            function_name=callable_name)
        assert identity.declaration_proven
        reference = UnresolvedSourceReference(
            symbol="gain", kind="storage_writers", file=filename,
            callable_id=identity.callable_id, identity=identity)
        sink: list = []
        candidates, evidence = resolver.resolve_with_evidence(
            reference, inputs.structure, sink, universe_version=0)
        assert [item.file for item in candidates] == [filename]
        census = dict(
            evidence.attempts[0].details.get("writer_census") or {})
        assert set(census) == {filename}, census
        seen[filename] = census[filename]
    assert seen["src/lib/g.cpp"] != seen["src/lib/h.cpp"]
    assert all("h.cpp" not in site for site in seen["src/lib/g.cpp"])
    assert all("g.cpp" not in site for site in seen["src/lib/h.cpp"])


# --- T6A: safe retirement of covered source-search obligations ---

def _retirement_cert(obligation_key, version=0, writers=("w:1",)):
    """Hand-built certificate with controlled identity: isolates state
    semantics from derivation for D1 unit tests."""
    from flight_log_agent.analysis.coverage import WriterCoverageCertificate
    scheduling = ("sched",) + tuple(obligation_key[1:])
    return WriterCoverageCertificate(
        obligation_key=tuple(obligation_key),
        scheduling_key=tuple(scheduling),
        declaration=(obligation_key[1], obligation_key[2], "gain"),
        receiver_context=("", "", ""),
        strategy="storage-internal-only",
        boundary=("src/lib/sg.cpp",),
        examined=("src/lib/sg.cpp",),
        version=version,
        assumptions=("file-linkage-closed",
                     "writer-syntax-enumerated"),
        writers=tuple(writers),
    )


_KEY_A = ("storage_writers", "global", "global:internal:a.cpp:gain")
_KEY_B = ("storage_writers", "global", "global:internal:b.cpp:gain")


def test_retirement_is_per_obligation_and_version_scoped(tmp_path):
    """D1: retiring A never affects B; retirement lives under its
    version; re-applying is idempotent (single record)."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
    )
    _ = tmp_path
    state = CoverageSearchState()
    assert state.version == 0
    assert apply_writer_coverage_retirement(
        state, _retirement_cert(_KEY_A), 0) is True
    assert state.is_proof_retired(_KEY_A) is True
    assert state.is_proof_retired(_KEY_B) is False
    record = state.proof_retirement(_KEY_A)
    assert record is not None
    assert record.resolution_key == _KEY_A
    assert record.version == 0
    assert record.writers == ("w:1",)
    assert state.proof_retirement(_KEY_B) is None
    assert apply_writer_coverage_retirement(
        state, _retirement_cert(_KEY_A), 0) is True
    assert len(state.retired) == 1
    state.advance_version()
    assert state.version == 1
    assert state.is_proof_retired(_KEY_A) is False
    assert state.proof_retirement(_KEY_A) is None


def test_retirement_ignores_scheduler_exhaustion_state(tmp_path):
    """D1/K-unit: visited/exhausted/completed scheduler state neither
    creates retirement without a certificate nor disturbs retirement
    with one. Proof retirement and scheduler exhaustion stay distinct."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
    )
    _ = tmp_path
    state = CoverageSearchState()
    state.mark_visited(_KEY_A)
    state.mark_exhausted("visit-key-a")
    state.mark_exhausted("visit-key-b")
    state.record_stages("obligation-a", {"owner-files"})
    assert state.is_proof_retired(_KEY_A) is False
    assert state.proof_retirement(_KEY_A) is None
    assert apply_writer_coverage_retirement(
        state, _retirement_cert(_KEY_A), 0) is True
    assert state.is_proof_retired(_KEY_A) is True
    assert state.is_proof_retired(_KEY_B) is False


def test_retirement_refuses_stale_and_malformed(tmp_path):
    """D/J-unit: stale-version and malformed certificates retire
    nothing and record nothing."""
    from dataclasses import replace
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
    )
    _ = tmp_path
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(
        state, _retirement_cert(_KEY_A, version=0), 1) is False
    assert state.is_proof_retired(_KEY_A) is False
    assert apply_writer_coverage_retirement(state, None, 0) is False
    empty_key = replace(_retirement_cert(_KEY_A), obligation_key=())
    assert apply_writer_coverage_retirement(
        state, empty_key, 0) is False
    empty_boundary = replace(_retirement_cert(_KEY_A), boundary=())
    assert apply_writer_coverage_retirement(
        state, empty_boundary, 0) is False
    assert state.retired == {}


def _retire_tree(tmp_path):
    """Single-file tree: two certifiable storage obligations (gain and
    bias), both schedulable in one discovery run."""
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    files = {
        "src/main.cpp": (
            "static int gain = 1;\n"
            "static int bias = 10;\n"
            "void update() { gain = 5; bias = 7; }\n"
            "float out;\n"
            "void run() { out = gain * bias; }\n"
        ),
    }
    profiler = _mini_tree(tmp_path, files, backend="tree_sitter")
    return profiler, files


def _spy_resolution_keys(monkeypatch):
    """Count discovery-loop resolutions keyed by resolution key.

    Intercepts both the legacy and evidence-capable resolver entry
    points (P0 routes frontier scheduling through
    `resolve_with_evidence`, which records the same search): every
    resolution is counted once with the identical key tuple, so
    pinned scheduling behavior is observed unchanged."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    calls: list = []
    original = SourceExpansionResolver.resolve
    original_with_evidence = SourceExpansionResolver.resolve_with_evidence

    def spy(self, reference, structure):
        calls.append((reference.kind, reference.symbol,
                      self.resolution_key(reference, structure)))
        return original(self, reference, structure)

    def evidence_spy(self, reference, structure, sink, skip_stages=None,
                     universe_version=None):
        calls.append((reference.kind, reference.symbol,
                      self.resolution_key(reference, structure)))
        return original_with_evidence(
            self, reference, structure, sink, skip_stages=skip_stages,
            universe_version=universe_version)

    monkeypatch.setattr(SourceExpansionResolver, "resolve", spy)
    monkeypatch.setattr(
        SourceExpansionResolver, "resolve_with_evidence", evidence_spy)
    return calls


def _run_retire_tree(tmp_path, monkeypatch, **discover_kwargs):
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler, files = _retire_tree(tmp_path)
    calls = _spy_resolution_keys(monkeypatch)
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["run"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp",
        **discover_kwargs)
    assert result.dag is not None
    return profiler, files, calls, result


def _gain_key_and_cert(tmp_path, profiler, files, loop_gain_key):
    """Real T3 certificate for the tree's gain obligation, pinned to the
    exact resolution key the discovery loop computed (spy-captured)."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
        UnresolvedSourceReference,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        dag_inputs_from_facts, load_facts,
    )
    facts = load_facts(
        profiler, tmp_path / "cache", list(files), "hash")
    structure = dag_inputs_from_facts(facts).structure
    callable_id = next(
        key for key, item in structure.callables_by_id.items()
        if item.get("name") == "run")
    identity = structure.symbol_identity(
        "gain", file="src/main.cpp", callable_id=callable_id,
        function_name="run")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="gain", kind="storage_writers", file="src/main.cpp",
        callable_id=callable_id, identity=identity)
    cert = _derive_use_certificate(
        profiler, structure, reference, version=0)
    assert tuple(cert.obligation_key) == tuple(loop_gain_key), (
        "certificate obligation must equal the loop resolution key")
    return reference, cert


def test_retired_obligation_leaves_scheduling(tmp_path, monkeypatch):
    """A: a current-version retired obligation is omitted from
    source-search scheduling while its provenance record is retained;
    DAG, files, rounds, and stop are otherwise identical."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
    )
    profiler, files, calls, result = _run_retire_tree(tmp_path, monkeypatch)
    gain_key = next(key for _kind, symbol, key in calls
                    if symbol == "gain")
    assert any(symbol == "bias" for _kind, symbol, _key in calls)
    _, cert = _gain_key_and_cert(tmp_path, profiler, files, gain_key)
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(state, cert, 0) is True
    profiler2, _files2, calls2, result2 = _run_retire_tree(
        tmp_path, monkeypatch, search_state=state)
    assert [symbol for _kind, symbol, _key in calls2
            if symbol == "gain"] == []
    assert any(symbol == "bias" for _kind, symbol, _key in calls2)
    record = state.proof_retirement(gain_key)
    assert record is not None
    assert record.writers == tuple(cert.writers) != ()
    assert record.version == 0
    assert result2.files_loaded == result.files_loaded
    assert result2.rounds == result.rounds
    assert result2.dag.model_dump() == result.dag.model_dump()
    assert result2.stop_reason == result.stop_reason


def test_retirement_preserves_unrelated_scheduling(tmp_path, monkeypatch):
    """B: retiring gain leaves the independent bias obligation fully
    schedulable — per-obligation independence at the loop level."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
    )
    profiler, files, calls, result = _run_retire_tree(
        tmp_path, monkeypatch)
    gain_key = next(key for _kind, symbol, key in calls
                    if symbol == "gain")
    _, cert = _gain_key_and_cert(tmp_path, profiler, files, gain_key)
    bias_key = next(key for _kind, symbol, key in calls
                    if symbol == "bias")
    assert bias_key != gain_key
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(state, cert, 0) is True
    assert state.is_proof_retired(bias_key) is False
    assert state.proof_retirement(bias_key) is None
    _p2, _f2, calls2, _r2 = _run_retire_tree(
        tmp_path, monkeypatch, search_state=state)
    assert any(symbol == "bias" for _kind, symbol, _key in calls2)
    assert not any(symbol == "gain" for _kind, symbol, _key in calls2)

def test_retirement_isolates_same_spelling_obligations(tmp_path,
                                                        monkeypatch):
    """C: same spelling `run` demanded through two receivers of
    unrelated classes resolves under distinct scheduling keys in one
    version; retiring the A key never suppresses the B key."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        WriterCoverageCertificate,
        apply_writer_coverage_retirement,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    files = {
        "src/main.cpp": (
            "struct A { float run(); };\n"
            "struct B { float run(); };\n"
            "A a;\n"
            "B b;\n"
            "float out;\n"
            "void go() { out = a.run() + b.run(); }\n"
        ),
    }
    profiler = _mini_tree(tmp_path, files, backend="tree_sitter")
    calls = _spy_resolution_keys(monkeypatch)
    first = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["go"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp")
    assert first.dag is not None
    run_keys = {(symbol, key) for _kind, symbol, key in calls
                if symbol in {"run", "a.run", "b.run"}}
    assert len({key for _symbol, key in run_keys}) >= 2, (
        f"expected two distinct scheduling keys, got {run_keys}")
    retired_key = sorted(key for _symbol, key in run_keys)[0]
    other_keys = {key for _symbol, key in run_keys} - {retired_key}
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(state, WriterCoverageCertificate(
        obligation_key=tuple(retired_key),
        scheduling_key=("sched", "run-a"),
        declaration=("callable", "run-a", "run"),
        receiver_context=("", "", ""),
        strategy="callable-owner-files",
        boundary=("src/main.cpp",),
        examined=("src/main.cpp",),
        version=0,
        assumptions=(),
        writers=("src/main.cpp:0:0:run-a"),
    ), 0) is True
    profiler2 = _mini_tree(tmp_path, files, backend="tree_sitter")
    second = discover_mechanism_dag(
        profiler2, tmp_path / "cache", seeds=["go"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp",
        search_state=state)
    assert second.dag is not None
    assert second.dag.model_dump() == first.dag.model_dump()
    # Loop-level proof: the retired key was never marked visited (the
    # loop schedules only what it marks), while the sibling key was.
    # NOTE: helper-body loading during graph construction resolves
    # callable symbols through a separate construction path that
    # intentionally bypasses both visited and retired suppression —
    # suppressing it would degrade the built DAG. Retirement governs
    # frontier source-search scheduling only.
    assert retired_key not in state.visited
    assert other_keys <= state.visited
    assert second.stop_reason == first.stop_reason


def _two_file_gain_tree(tmp_path):
    """Two-file tree where the gain storage obligation is first
    suppressed, then must retry after the universe extends via an
    independent callable path."""
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    files = {
        "src/main.cpp": (
            "struct Cfg { float gain; };\n"
            "float tweak(float v);\n"
            "struct Ctrl {\n"
            "    Cfg cfg;\n"
            "    float out;\n"
            "    void run() { out = cfg.gain * tweak(1.0f); }\n"
            "};\n"
        ),
        "src/help.cpp": (
            "float tweak(float v) { return v; }\n"
            "struct Derived : Cfg {};\n"
            "void Derived::apply(float v) { gain = v; }\n"
            "void Cfg::other() {}\n"
        ),
    }
    profiler = _mini_tree(tmp_path, files, backend="tree_sitter")
    return profiler, files


def _discover_gain_tree(tmp_path, monkeypatch, files, **kwargs):
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler = _mini_tree(tmp_path, files, backend="tree_sitter")
    calls: list = []
    original = SourceExpansionResolver.resolve
    original_with_evidence = SourceExpansionResolver.resolve_with_evidence

    def spy(self, reference, structure):
        calls.append((reference.kind, reference.symbol,
                      self.resolution_key(reference, structure)))
        return original(self, reference, structure)

    def evidence_spy(self, reference, structure, sink, skip_stages=None,
                     universe_version=None):
        calls.append((reference.kind, reference.symbol,
                      self.resolution_key(reference, structure)))
        return original_with_evidence(
            self, reference, structure, sink, skip_stages=skip_stages,
            universe_version=universe_version)

    monkeypatch.setattr(SourceExpansionResolver, "resolve", spy)
    monkeypatch.setattr(
        SourceExpansionResolver, "resolve_with_evidence", evidence_spy)
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["run"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp", **kwargs)
    assert result.dag is not None
    return calls, result


def test_version_advance_reactivates_retirement(tmp_path, monkeypatch):
    """E: retirement is version-scoped, not sticky. Pre-retired at
    version 0, the obligation is suppressed in round 0; after new
    source extends the universe (version 1), the stale retirement no
    longer suppresses it and it resolves exactly once.

    Exactly-one-total is airtight here: round 0 must suppress (record
    is current), and round 1 cannot suppress (visited was cleared by
    the advance and no version-1 retirement exists)."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        WriterCoverageCertificate,
        apply_writer_coverage_retirement,
    )
    profiler, files = _two_file_gain_tree(tmp_path)
    calls, _first = _discover_gain_tree(tmp_path, monkeypatch, files)
    cfg_key = next(key for _kind, symbol, key in calls if symbol == "cfg")
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(state, WriterCoverageCertificate(
        obligation_key=tuple(cfg_key),
        scheduling_key=("sched", "cfg"),
        declaration=("member", "cfg", "cfg"),
        receiver_context=("", "", ""),
        strategy="storage-owner-files",
        boundary=("src/main.cpp",),
        examined=("src/main.cpp",),
        version=0,
        assumptions=(),
        writers=("src/main.cpp:0:0:cfg",),
    ), 0) is True
    calls2, second = _discover_gain_tree(
        tmp_path, monkeypatch, files, search_state=state)
    cfg_calls2 = [key for _kind, symbol, key in calls2
                  if symbol == "cfg"]
    assert len(cfg_calls2) == 1
    assert {f for _, files in [(r.index, r.new_files)
                               for r in second.rounds]
            for f in files} >= {"src/main.cpp", "src/help.cpp"}
    assert len(second.rounds) >= 2
    assert state.version == 1
    assert state.proof_retirement(cfg_key) is None


def test_retired_obligation_stays_stable(tmp_path, monkeypatch):
    """F: within one immutable universe version, a pre-retired
    obligation never re-enters scheduling across repeated runs sharing
    the session state: no resolve calls, exactly one record, normal
    completion for everything else."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
    )
    profiler, files = _retire_tree(tmp_path)
    _, _, calls, _first = _run_retire_tree(tmp_path, monkeypatch)
    gain_key = next(key for _kind, symbol, key in calls
                    if symbol == "gain")
    _, cert = _gain_key_and_cert(tmp_path, profiler, files, gain_key)
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(state, cert, 0) is True
    first_calls, first_result = _spy_run(tmp_path, monkeypatch, state)
    assert not any(symbol == "gain"
                   for _kind, symbol, _key in first_calls)
    assert any(symbol == "bias" for _kind, symbol, _key in first_calls)
    assert first_result.stop_reason == "frontier_exhausted"
    # A second run sharing the session state stays stable: gain is
    # still retired (record intact), bias is now retired too — P2D
    # derived its genuine certificate in the first run — and neither
    # re-resolves. Suppression reasons may upgrade to proof, but
    # scheduling stays settled.
    second_calls, second_result = _spy_run(tmp_path, monkeypatch, state)
    assert not any(symbol == "gain"
                   for _kind, symbol, _key in second_calls)
    assert not any(symbol == "bias"
                   for _kind, symbol, _key in second_calls)
    assert second_result.stop_reason == "frontier_exhausted"
    assert len(state.retired) == 2
    assert state.proof_retirement(gain_key) is not None


def _spy_run(tmp_path, monkeypatch, state):
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler, files = _retire_tree(tmp_path)
    calls = _spy_resolution_keys(monkeypatch)
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["run"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp",
        search_state=state)
    assert result.dag is not None
    return calls, result


def _closed_empty_consumer_tree(tmp_path):
    """Writerless internal global with a straight-line consumer: the
    obligation stays open (evidence placeholder) while T3 certifies
    genuine absence."""
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    files = {
        "src/main.cpp": (
            "static float kzero;\n"
            "float out;\n"
            "void use() { out = kzero * 3.0f; }\n"
        ),
    }
    profiler = _mini_tree(tmp_path, files, backend="tree_sitter")
    return profiler, files


def test_closed_empty_certificate_retires_without_invention(tmp_path,
                                                            monkeypatch):
    """G: a valid closed-empty certificate retires its exact obligation —
    retained writer tuple stays empty, nothing is invented, and the
    discovery outcome is otherwise identical."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        dag_inputs_from_facts,
        discover_mechanism_dag,
        load_facts,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
        UnresolvedSourceReference,
    )
    profiler, files = _closed_empty_consumer_tree(tmp_path)
    calls: list = []
    original = SourceExpansionResolver.resolve
    original_with_evidence = SourceExpansionResolver.resolve_with_evidence

    def spy(self, reference, structure):
        calls.append((reference.kind, reference.symbol,
                      self.resolution_key(reference, structure)))
        return original(self, reference, structure)

    def evidence_spy(self, reference, structure, sink, skip_stages=None,
                     universe_version=None):
        calls.append((reference.kind, reference.symbol,
                      self.resolution_key(reference, structure)))
        return original_with_evidence(
            self, reference, structure, sink, skip_stages=skip_stages,
            universe_version=universe_version)

    monkeypatch.setattr(SourceExpansionResolver, "resolve", spy)
    monkeypatch.setattr(
        SourceExpansionResolver, "resolve_with_evidence", evidence_spy)
    baseline = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["use"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp")
    assert baseline.dag is not None
    key = next(key for _kind, symbol, key in calls if symbol == "kzero")
    facts = load_facts(
        profiler, tmp_path / "cache", list(files), "hash")
    structure = dag_inputs_from_facts(facts).structure
    callable_id = next(
        key for key, item in structure.callables_by_id.items()
        if item.get("name") == "use")
    identity = structure.symbol_identity(
        "kzero", file="src/main.cpp", callable_id=callable_id,
        function_name="use")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="kzero", kind="storage_writers", file="src/main.cpp",
        callable_id=callable_id, identity=identity)
    cert = _derive_use_certificate(
        profiler, structure, reference, version=0)
    assert cert.writers == ()
    assert tuple(cert.obligation_key) == tuple(key)
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(state, cert, 0) is True
    assert state.proof_retirement(key) is not None
    assert state.proof_retirement(key).writers == ()
    profiler2, _files2 = _closed_empty_consumer_tree(tmp_path)
    calls2: list = []

    def spy2(self, reference, structure):
        calls2.append((reference.kind, reference.symbol,
                       self.resolution_key(reference, structure)))
        return original(self, reference, structure)

    def evidence_spy2(self, reference, structure, sink, skip_stages=None,
                      universe_version=None):
        calls2.append((reference.kind, reference.symbol,
                       self.resolution_key(reference, structure)))
        return original_with_evidence(
            self, reference, structure, sink, skip_stages=skip_stages,
            universe_version=universe_version)

    monkeypatch.setattr(SourceExpansionResolver, "resolve", spy2)
    monkeypatch.setattr(
        SourceExpansionResolver, "resolve_with_evidence", evidence_spy2)
    retired = discover_mechanism_dag(
        profiler2, tmp_path / "cache", seeds=["use"], terminal="out",
        source_hash="hash", terminal_file="src/main.cpp",
        search_state=state)
    assert retired.dag is not None
    assert not any(symbol == "kzero" for _kind, symbol, _key in calls2)
    assert retired.stop_reason == baseline.stop_reason
    assert retired.dag.model_dump() == baseline.dag.model_dump()


def _guarded_gain_tree(tmp_path, terminal="out_a", seeds=("usea",)):
    """Internal global with one writer, consumed once straight and once
    behind a branch: coverage is shared, applicability splits."""
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    files = {
        "src/main.cpp": (
            "static float kgain = 1.0f;\n"
            "float out_a;\n"
            "float out_b;\n"
            "int cond;\n"
            "void usea() { out_a = kgain * 3.0f; }\n"
            "void useb() { if (cond) { out_b = kgain * 3.0f; } }\n"
        ),
    }
    profiler = _mini_tree(tmp_path, files, backend="tree_sitter")
    result = discover_mechanism_dag(
        profiler, tmp_path / f"cache_{terminal}", seeds=list(seeds),
        terminal=terminal, source_hash="hash",
        terminal_file="src/main.cpp")
    assert result.dag is not None
    return profiler, files, result


def test_coverage_retires_while_applicability_refuses(tmp_path,
                                                      monkeypatch):
    """H: valid coverage retires source search while T5 still refuses
    applicability for the guarded use — separation pinned end to end."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
        derive_writer_applicability,
    )
    profiler, files, result = _guarded_gain_tree(
        tmp_path, terminal="out_b", seeds=("useb",))
    dag = result.dag
    use = next(vertex for vertex in dag.vertices
               if vertex.variable == "out_b"
               and vertex.kind == "operation")
    assert any(edge.kind == "control" and edge.target_id == use.id
               for edge in dag.edges), (
        "guarded use must be control-gated")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler, result.inputs.structure, reference, version=0)
    guarded_outcome = derive_writer_applicability(
        reference, use.id, "kgain", cert, dag, 0,
        conditional_writer_ids=())
    assert guarded_outcome.proof is None
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(state, cert, 0) is True
    key = tuple(cert.obligation_key)
    calls, retired = _discover_guarded_tree_at_version(
        tmp_path, monkeypatch, state)
    assert not any(symbol == "kgain" for _kind, symbol, _key in calls)
    assert retired.stop_reason == result.stop_reason
    assert derive_writer_applicability(
        reference, use.id, "kgain", cert, dag, 0,
        conditional_writer_ids=()) == guarded_outcome


def _discover_guarded_tree_at_version(tmp_path, monkeypatch, state):
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler, _files = _guarded_gain_tree(
        tmp_path, terminal="out_b", seeds=("useb",))[:2]
    calls = _spy_resolution_keys(monkeypatch)
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache_out_b", seeds=["useb"],
        terminal="out_b", source_hash="hash",
        terminal_file="src/main.cpp", search_state=state)
    assert result.dag is not None
    return calls, result


def test_missing_certificate_retires_nothing(tmp_path, monkeypatch):
    """I: without proof nothing is fabricated — and P2D only retires
    genuinely derived current certificates while scheduling proceeds
    normally (no hand proof was supplied here)."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
    )
    assert apply_writer_coverage_retirement(
        CoverageSearchState(), None, 0) is False
    state = CoverageSearchState()
    calls, result = _spy_run(tmp_path, monkeypatch, state)
    assert any(symbol == "gain" for _kind, symbol, _key in calls)
    for (version, _obligation), record in state.retired.items():
        assert version == state.version
        assert tuple(record.obligation_key)
        assert tuple(record.boundary)
        assert record.version == state.version


def test_malformed_certificate_leaves_scheduling_unchanged(tmp_path,
                                                           monkeypatch):
    """J: malformed certificates refuse without recording, and the
    subsequent discovery schedules exactly as the unretired baseline."""
    from dataclasses import replace
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
    )
    profiler, files, calls, result = _run_retire_tree(
        tmp_path, monkeypatch)
    gain_key = next(key for _kind, symbol, key in calls
                    if symbol == "gain")
    _, cert = _gain_key_and_cert(tmp_path, profiler, files, gain_key)
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(
        state, replace(cert, obligation_key=()), 0) is False
    assert apply_writer_coverage_retirement(
        state, replace(cert, boundary=()), 0) is False
    assert apply_writer_coverage_retirement(
        state, replace(cert, version=99), 0) is False
    assert state.retired == {}
    calls2, result2 = _spy_run(tmp_path, monkeypatch, state)
    assert any(symbol == "gain" for _kind, symbol, _key in calls2)
    assert result2.stop_reason == result.stop_reason


def test_visited_suppression_has_no_retirement_record(tmp_path,
                                                      monkeypatch):
    """K: the scheduling distinguisher — a visited-suppressed
    obligation resolves zero times with NO retirement record, while a
    proof-retired obligation resolves zero times WITH one."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        WriterCoverageCertificate,
        apply_writer_coverage_retirement,
    )
    profiler, files, calls, _result = _run_retire_tree(
        tmp_path, monkeypatch)
    gain_key = next(key for _kind, symbol, key in calls
                    if symbol == "gain")
    bias_key = next(key for _kind, symbol, key in calls
                    if symbol == "bias")
    state = CoverageSearchState()
    state.mark_visited(gain_key)
    assert apply_writer_coverage_retirement(state, WriterCoverageCertificate(
        obligation_key=tuple(bias_key),
        scheduling_key=("sched", "bias"),
        declaration=("global", "bias", "bias"),
        receiver_context=("", "", ""),
        strategy="storage-internal-only",
        boundary=("src/main.cpp",),
        examined=("src/main.cpp",),
        version=0,
        assumptions=(),
        writers=("src/main.cpp:0:0:bias",),
    ), 0) is True
    calls2, _result2 = _spy_run(tmp_path, monkeypatch, state)
    resolved2 = {key for _kind, _symbol, key in calls2}
    assert gain_key not in resolved2
    assert bias_key not in resolved2
    assert state.proof_retirement(gain_key) is None
    assert state.proof_retirement(bias_key) is not None
    assert state.proof_retirement(bias_key).writers == (
        "src/main.cpp:0:0:bias",)


def test_multi_writer_provenance_is_complete(tmp_path):
    """L: retirement of a multi-writer certificate retains the complete
    writer tuple for later diagnostic lookup — no first-writer
    collapse."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        apply_writer_coverage_retirement,
        derive_writer_coverage_certificate,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/tw.cpp": (
            "static int gain = 1;\nvoid update() { gain = 5; }\n"
        ),
    })
    identity = inputs.structure.symbol_identity(
        "gain", file="src/lib/tw.cpp",
        callable_id=next(key for key, item in
                         inputs.structure.callables_by_id.items()
                         if item.get("name") == "update"),
        function_name="update")
    reference = UnresolvedSourceReference(
        symbol="gain", kind="storage_writers", file="src/lib/tw.cpp",
        callable_id=identity.callable_id, identity=identity)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    result = derive_writer_coverage_certificate(reference, evidence, 0)
    assert result.certificate is not None, result.refusal
    assert len(result.certificate.writers) == 2
    state = CoverageSearchState()
    assert apply_writer_coverage_retirement(
        state, result.certificate, 0) is True
    record = state.proof_retirement(result.certificate.obligation_key)
    assert record is not None
    assert record.writers == tuple(result.certificate.writers)
    assert len(record.writers) == 2


def test_retirement_behavioral_differential(tmp_path, monkeypatch):
    """Differential matrix: plain / valid-retired / stale-attempted /
    unrelated-retired. Only valid retirement moves gain scheduling;
    stop, files, rounds, and DAG stay identical everywhere."""
    from flight_log_agent.analysis.coverage import (
        CoverageSearchState,
        WriterCoverageCertificate,
        apply_writer_coverage_retirement,
    )
    profiler, files, calls, baseline = _run_retire_tree(
        tmp_path, monkeypatch)
    gain_key = next(key for _kind, symbol, key in calls
                    if symbol == "gain")
    bias_key = next(key for _kind, symbol, key in calls
                    if symbol == "bias")
    _, cert = _gain_key_and_cert(tmp_path, profiler, files, gain_key)

    def surface(result, run_calls):
        return {
            "stop": result.stop_reason,
            "files": list(result.files_loaded),
            "rounds": [(item.index, list(item.new_files))
                       for item in result.rounds],
            "dag": result.dag.model_dump(),
            "gain_calls": sum(1 for _kind, symbol, _key in run_calls
                              if symbol == "gain"),
            "bias_calls": sum(1 for _kind, symbol, _key in run_calls
                              if symbol == "bias"),
        }

    scenarios = {}
    plain_state = CoverageSearchState()
    _, plain_result = _spy_run(tmp_path, monkeypatch, plain_state)
    scenarios["plain"] = (plain_result, None)
    valid_state = CoverageSearchState()
    assert apply_writer_coverage_retirement(valid_state, cert, 0) is True
    _, valid_result = _spy_run(tmp_path, monkeypatch, valid_state)
    scenarios["valid"] = (valid_result, None)
    stale_state = CoverageSearchState()
    assert apply_writer_coverage_retirement(stale_state, cert, 1) is False
    _, stale_result = _spy_run(tmp_path, monkeypatch, stale_state)
    scenarios["stale"] = (stale_result, None)
    unrelated_state = CoverageSearchState()
    assert apply_writer_coverage_retirement(unrelated_state,
                                            WriterCoverageCertificate(
        obligation_key=tuple(bias_key),
        scheduling_key=("sched", "bias"),
        declaration=("global", "bias", "bias"),
        receiver_context=("", "", ""),
        strategy="storage-internal-only",
        boundary=("src/main.cpp",),
        examined=("src/main.cpp",),
        version=0,
        assumptions=(),
        writers=("src/main.cpp:0:0:bias",),
    ), 0) is True
    _, unrelated_result = _spy_run(tmp_path, monkeypatch, unrelated_state)
    scenarios["unrelated"] = (unrelated_result, None)
    assert baseline.dag.model_dump() == scenarios["plain"][0].dag.model_dump()
    for name in ("valid", "stale", "unrelated"):
        result = scenarios[name][0]
        assert result.stop_reason == baseline.stop_reason, name
        assert result.files_loaded == baseline.files_loaded, name
        assert result.dag.model_dump() == baseline.dag.model_dump(), name
    # Re-run with spies for call counts (states are single-use per run
    # above only for results; counts come from dedicated runs below).
    counts = {}
    for name, maker in (
            ("plain", CoverageSearchState),
            ("stale", CoverageSearchState)):
        state = maker()
        run_calls, _ = _spy_run(tmp_path, monkeypatch, state)
        counts[name] = sum(1 for _kind, symbol, _key in run_calls
                           if symbol == "gain")
    valid_state2 = CoverageSearchState()
    assert apply_writer_coverage_retirement(valid_state2, cert, 0) is True
    valid_calls, _ = _spy_run(tmp_path, monkeypatch, valid_state2)
    counts["valid"] = sum(1 for _kind, symbol, _key in valid_calls
                          if symbol == "gain")
    unrelated_state2 = CoverageSearchState()
    assert apply_writer_coverage_retirement(unrelated_state2,
                                            WriterCoverageCertificate(
        obligation_key=tuple(bias_key),
        scheduling_key=("sched", "bias"),
        declaration=("global", "bias", "bias"),
        receiver_context=("", "", ""),
        strategy="storage-internal-only",
        boundary=("src/main.cpp",),
        examined=("src/main.cpp",),
        version=0,
        assumptions=(),
        writers=("src/main.cpp:0:0:bias",),
    ), 0) is True
    unrelated_calls, _ = _spy_run(tmp_path, monkeypatch, unrelated_state2)
    counts["unrelated"] = sum(1 for _kind, symbol, _key in unrelated_calls
                              if symbol == "gain")
    assert counts["plain"] >= 1
    assert counts["valid"] == 0
    assert counts["stale"] >= 1
    assert counts["unrelated"] >= 1


def _opaque_member_obligation(tmp_path):
    """Opaque-leaf member fixture plus its proven config obligation,
    unevaluated: callers detach origins then run the round."""
    dag, samples = _unresolved_member_dag(tmp_path, _UNRESOLVED_MEMBER_SOURCE)
    reference = next(r for r in dag.unresolved_references
                     if r.kind == "storage_writers" and r.symbol == "config")
    return dag, samples, reference


def _run_proof_round(dag, samples, **kwargs):
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_checkpoint_round,
    )
    return evaluate_checkpoint_round(
        dag, parameter_values={}, observed_signals=set(samples),
        signal_policies={s: {"method": "linear"} for s in samples},
        load_samples=lambda *_: samples, **kwargs)


def test_local_only_obligation_enters_proof_relevance(tmp_path):
    """A (T4 F1): an obligation attached to local scheduling through
    exact declaration identity — while outside the ordinary checkpoint
    closure — must appear in proof relevance. It is scheduled
    (local-calculation source work) yet pre-fix invisible to proof."""
    dag, samples, reference = _opaque_member_obligation(tmp_path)
    reference.origin_vertex_ids = ["elsewhere-outside-closure"]
    result = _run_proof_round(dag, samples)
    key = reference.visit_key()
    assert result.summary["next_analysis"]["kind"] == (
        "local_calculation_source")
    assert key in [item.visit_key() for item in result.references], (
        "local-scheduled obligation must exist in round references")
    observation = result.proof_observation
    assert observation is not None
    assert key in observation.relevant_obligation_keys, (
        "local-only obligation missing from proof relevance")
    assert key in observation.uncovered_obligation_keys


def test_origin_less_local_obligation_enters_proof_relevance(tmp_path):
    """B (T4 F1): an origin-less reference attached to local scheduling
    through exact declaration identity must appear in proof relevance.
    Assess records a source-linkage requirement for it (pre-existing
    behavior, unchanged); relevance must still include it."""
    dag, samples, reference = _opaque_member_obligation(tmp_path)
    reference.origin_vertex_ids = []
    result = _run_proof_round(dag, samples)
    key = reference.visit_key()
    assert result.summary["next_analysis"]["kind"] == (
        "local_calculation_source")
    assert key in [item.visit_key() for item in result.references]
    kinds = {item["kind"] for item in
             result.summary.get("selected_checkpoint", {}).get(
                 "analysis_requirements", [])}
    assert "source_linkage" in kinds, (
        "origin-less references keep their pre-existing requirement")
    observation = result.proof_observation
    assert observation is not None
    assert key in observation.relevant_obligation_keys, (
        "origin-less local obligation missing from proof relevance")
    assert key in observation.uncovered_obligation_keys


def test_ordinary_and_local_duplicate_dedupes(tmp_path):
    """E (T4 F1): one obligation reachable through both ordinary
    checkpoint references and local needs appears exactly once in
    relevance, with correct covered/uncovered accounting."""
    dag, samples, reference = _opaque_member_obligation(tmp_path)
    result = _run_proof_round(dag, samples)
    key = reference.visit_key()
    observation = result.proof_observation
    assert observation is not None
    relevant = list(observation.relevant_obligation_keys)
    assert relevant.count(key) == 1
    assert key in observation.uncovered_obligation_keys
    assert key not in observation.covered_obligation_keys


def test_local_matching_certificate_covers(tmp_path):
    """C (T4 F1): a current certificate whose scheduling key matches a
    local-sourced obligation covers it — association is scheduling-key
    equality over a real derived certificate, not origin matching."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    cert = _derive_for(profiler, inputs, reference, version=0)
    local = reference.model_copy(update={"origin_vertex_ids": []})
    assert local.visit_key() == reference.visit_key()
    key = local.visit_key()
    observation = observe_checkpoint_coverage(
        [local], certificates=[cert], proof_version=0,
        searchable_keys={key})
    assert observation.relevant_obligation_keys == (key,)
    assert observation.covered_obligation_keys == (key,)
    assert observation.covered_semantic_keys == (cert.obligation_key,)
    assert observation.uncovered_obligation_keys == ()


def test_local_cross_declaration_isolation(tmp_path):
    """D (T4 F1): same-spelling local obligations under different proven
    declarations stay isolated — a certificate for one never covers the
    other."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    profiler, inputs, reference_a = _internal_gain_fixture(tmp_path)
    _other_profiler, _other_inputs, reference_b = _internal_h_certificate(
        tmp_path)
    assert (reference_b.identity.declaration_id
            != reference_a.identity.declaration_id)
    cert_a = _derive_for(profiler, inputs, reference_a, version=0)
    local_a = reference_a.model_copy(update={"origin_vertex_ids": []})
    local_b = reference_b.model_copy(update={"origin_vertex_ids": []})
    key_a, key_b = local_a.visit_key(), local_b.visit_key()
    assert key_a != key_b
    observation = observe_checkpoint_coverage(
        [local_a, local_b], certificates=[cert_a], proof_version=0,
        searchable_keys={key_a, key_b})
    assert set(observation.relevant_obligation_keys) == {key_a, key_b}
    assert observation.covered_obligation_keys == (key_a,)
    assert observation.uncovered_obligation_keys == (key_b,)


def test_local_filtered_union_semantics(tmp_path):
    """Filtered means omitted from every current scheduling pool:
    a local obligation scheduled through the local path is relevant
    but not filtered; one scheduled nowhere is both relevant and
    filtered."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        observe_checkpoint_coverage,
    )
    profiler, inputs, reference = _internal_gain_fixture(tmp_path)
    local = reference.model_copy(update={"origin_vertex_ids": []})
    key = local.visit_key()
    scheduled = observe_checkpoint_coverage(
        [local], certificates=[], proof_version=0,
        searchable_keys={key})
    assert scheduled.filtered_obligation_keys == ()
    assert scheduled.uncovered_obligation_keys == (key,)
    unscheduled = observe_checkpoint_coverage(
        [local], certificates=[], proof_version=0,
        searchable_keys=set())
    assert unscheduled.filtered_obligation_keys == (key,)
    assert unscheduled.uncovered_obligation_keys == (key,)


def test_exhausted_local_obligation_stays_relevant(tmp_path):
    """F-analog (T4 F1): scheduler suppression (exhausted, the only
    suppression a checkpoint round can see — T6A retirement is
    invisible here by design) never erases relevance. The obligation
    stays relevant, filtered, and uncovered."""
    dag, samples, reference = _opaque_member_obligation(tmp_path)
    reference.origin_vertex_ids = ["elsewhere-outside-closure"]
    key = reference.visit_key()
    dag.exhausted_source_requests.add(key)
    result = _run_proof_round(dag, samples)
    observation = result.proof_observation
    assert observation is not None
    assert key in observation.relevant_obligation_keys
    assert key in observation.filtered_obligation_keys
    assert key in observation.uncovered_obligation_keys
    assert key not in [item.visit_key() for item in result.references]
    assert result.summary.get("selected_checkpoint", {}).get(
        "authorizes_discovery_stop", False) is False


def test_vacuity_flip_keeps_behavior(tmp_path):
    """G-integration (T4 F1): local-only relevance turns a previously
    false-empty relevance set non-empty diagnostically, while action,
    next analysis, requirements, flags, and stop stay exactly as the
    round computes them."""
    dag, samples, reference = _opaque_member_obligation(tmp_path)
    reference.origin_vertex_ids = ["elsewhere-outside-closure"]
    result = _run_proof_round(dag, samples)
    observation = result.proof_observation
    assert observation is not None
    assert observation.relevant_empty is False
    assert observation.scope_degenerate is False
    assert observation.covered_obligation_keys == ()
    assert result.summary["next_analysis"]["kind"] == (
        "local_calculation_source")
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("authorizes_discovery_stop", False) is False
    assert selected.get("writer_coverage_verified", False) is False
    assert selected.get("applicability_verified", False) is False


# --- T6B: checkpoint-owned discovery stop authority ---

def _proof_observation(relevant=(), covered=(), proof_version=0,
                       degenerate=False):
    """Hand-built observation for authority-logic pins: the T4 helper
    contract makes these shapes producible, so the truth table tests
    the conjunction, not the plumbing."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        CheckpointProofObservation,
    )
    relevant = tuple(relevant)
    covered = tuple(k for k in covered if k in set(relevant))
    uncovered = tuple(k for k in relevant if k not in set(covered))
    return CheckpointProofObservation(
        relevant_obligation_keys=relevant,
        covered_obligation_keys=covered,
        uncovered_obligation_keys=uncovered,
        filtered_obligation_keys=(),
        stale_certificate_keys=(),
        covered_semantic_keys=tuple(("sem", k) for k in covered),
        proof_version=proof_version,
        scope_degenerate=bool(degenerate),
        relevant_empty=not relevant,
    )


def _authority(**overrides):
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_proof_authority,
    )
    arguments = dict(
        observation=_proof_observation(),
        certificates=(),
        applicability_proofs=(),
        proof_version=0,
        legacy_verified=False,
        applicability_uses=(),
    )
    arguments.update(overrides)
    return evaluate_proof_authority(**arguments)


def _proof_pair(use, visit, semantic, writers=("w:1",), version=0):
    """Matching hand-built certificate + applicability proof with fully
    consistent keys: the mechanism under test is combination logic, and
    frozen value construction keeps every binding explicit."""
    from flight_log_agent.analysis.coverage import (
        WriterApplicabilityProof,
        WriterCoverageCertificate,
    )
    certificate = WriterCoverageCertificate(
        obligation_key=tuple(semantic),
        scheduling_key=tuple(visit),
        declaration=("global", semantic[2] if len(semantic) > 2 else "",
                     "signal"),
        receiver_context=("", "", ""),
        strategy="storage-internal-only",
        boundary=("src/lib/sg.cpp",),
        examined=("src/lib/sg.cpp",),
        version=version,
        assumptions=("file-linkage-closed",
                     "writer-syntax-enumerated"),
        writers=tuple(writers),
    )
    proof = WriterApplicabilityProof(
        use_key=tuple(use),
        scheduling_key=tuple(visit),
        obligation_key=tuple(semantic),
        declaration=("global", semantic[2] if len(semantic) > 2 else "",
                     "signal"),
        writers=tuple(writers),
        producer_vertex="op-producer",
        basis="single-exact-producer",
        supporting_facts=("use:op", "producer:op-producer"),
        version=version,
        receiver_context=("", "", ""),
    )
    return certificate, proof


def test_proof_authority_full_conjunction_authorizes():
    """A-logic: legacy verified + non-empty relevance + full coverage +
    full applicability + non-degenerate scope authorizes stop. This pins
    the gate LOGIC with explicit inputs; see the report on why no
    natural single-round fixture satisfies it (STOP-B analysis)."""
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[proof], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.legacy_verified is True
    assert authority.coverage_ok is True
    assert authority.applicability_ok is True
    assert authority.non_vacuous_ok is True
    assert authority.writer_coverage_verified is True
    assert authority.applicability_verified is True
    assert authority.authorizes_stop is True


def test_proof_authority_missing_applicability_vetoes():
    """B-logic: full coverage with no applicability proof for the
    required use vetoes stop while coverage stays true."""
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, _proof = _proof_pair(use, key, semantic)
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.coverage_ok is True
    assert authority.writer_coverage_verified is True
    assert authority.applicability_ok is False
    assert authority.applicability_verified is False
    assert authority.authorizes_stop is False


def test_proof_authority_incomplete_coverage_vetoes():
    """C-logic: applicability proofs cannot rescue missing coverage."""
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[])
    certificate, proof = _proof_pair(use, key, semantic)
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[proof], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.coverage_ok is False
    assert authority.writer_coverage_verified is False
    assert authority.authorizes_stop is False


def test_proof_authority_inactive_defers_to_legacy():
    """Omitted proof version with empty relevance (the default
    observation here) follows legacy exactly via the §9
    genuinely-no-work path — proving nothing. Non-empty relevance
    with omitted proof vetoes instead (see omitted_proof_vetoes)."""
    for legacy in (False, True):
        authority = _authority(proof_version=None,
                               legacy_verified=legacy)
        assert authority.gate_active is False
        assert authority.writer_coverage_verified is False
        assert authority.applicability_verified is False
        assert authority.authorizes_stop is legacy


def test_proof_authority_vacuous_preserves_legacy():
    """Vacuous relevance (nothing to prove) with a non-degenerate scope
    follows legacy: True stays True (T4 G-shape compatibility)."""
    observation = _proof_observation(relevant=[])
    authority = _authority(observation=observation, legacy_verified=True)
    assert authority.coverage_ok is True
    assert authority.non_vacuous_ok is True
    assert authority.writer_coverage_verified is False
    assert authority.authorizes_stop is True


def test_proof_authority_degenerate_vetoes():
    """Degenerate scope authorizes nothing, even with legacy verified
    and vacuous relevance: fail closed on unknowable scope."""
    observation = _proof_observation(relevant=[], degenerate=True)
    authority = _authority(observation=observation, legacy_verified=True)
    assert authority.non_vacuous_ok is False
    assert authority.authorizes_stop is False
    assert authority.writer_coverage_verified is False


def test_round_degenerate_scope_vetoes_despite_proof_inputs():
    """N: a rootless degenerate round cannot authorize stop even with
    threaded proof state — non-vacuity fails closed."""
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    def expression(text, *inputs):
        return {"text": text, "lowered_text": text,
                "input_symbols": list(inputs),
                "input_identities": {}, "call_results": [], "exact": True}

    bindings = [
        {"target_symbol": "sample", "source_symbol": "measurement.value",
         "external_source_signal": True, "synthetic_boundary_transfer": True,
         "boundary_direction": "subscribe",
         "expression_ref": expression("measurement.value", "measurement.value"),
         "assignment_path": [{"file": "s.cpp", "line": 1}],
         "function": "f"},
        {"target_symbol": "orphan", "source_symbol": "sample",
         "expression_ref": expression("sample", "sample"),
         "assignment_path": [{"file": "s.cpp", "line": 2}],
         "function": "f"},
    ]
    dag = build_mechanism_dag(bindings, "missing-terminal",
                              logged_signals=set())
    result = _run_proof_round(dag, {}, proof_version=0)
    assert result.proof_observation is not None
    assert result.proof_observation.scope_degenerate is True
    assert result.proof_authority is not None
    assert result.proof_authority.gate_active is True
    assert result.proof_authority.non_vacuous_ok is False
    assert result.proof_authority.authorizes_stop is False
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("authorizes_discovery_stop", False) is False


def test_round_conflicting_proofs_listed_without_authority(tmp_path):
    """Two rival VALID bindings for one scheduled use are both recorded
    as conflicting and authorize nothing. (Finding 6: each rival
    matches its own current certificate for the same visit with a
    different writer set — genuine ambiguity. A rival matching no
    current certificate would be ignored instead.)"""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    key = reference.visit_key()
    use = (root.id, "sample")
    semantic = ("storage_writers", "member", "decl:x")
    certificate, proof = _t6b_hand_proof(use, key, semantic)
    rival_certificate, rival = _t6b_hand_proof(
        use, key, semantic, writers=("other-site",))
    from dataclasses import replace
    rival = replace(rival, basis="guard-exact-producer")
    result = _run_t6b_round(
        dag, samples, policies,
        coverage_certificates=[certificate, rival_certificate],
        proof_version=0, applicability_proofs=[proof, rival])
    assert result.proof_authority is not None
    assert result.proof_authority.conflicting_applicability_uses == (
        tuple(use),)
    assert result.proof_authority.applicability_ok is False
    assert result.proof_authority.authorizes_stop is False


def test_round_local_obligation_uncovered_vetoes_silently(tmp_path):
    """E-authority: an F1 local-only obligation with no certificate is
    uncovered; authority records it without disturbing the round."""
    dag, samples, reference = _opaque_member_obligation(tmp_path)
    reference.origin_vertex_ids = ["elsewhere-outside-closure"]
    result = _run_proof_round(dag, samples, proof_version=0)
    key = reference.visit_key()
    assert result.proof_authority is not None
    assert result.proof_authority.gate_active is True
    assert key in result.proof_authority.uncovered_relevant
    assert result.proof_authority.coverage_ok is False
    assert result.proof_authority.authorizes_stop is False
    assert result.action != "verified"


def test_round_exhausted_covered_obligation_ignores_queue(tmp_path):
    """Q-analog: an exhausted (unscheduled) obligation with a current
    certificate is still covered — authority follows proof, and with
    replay now attempting, the proof-adjusted round verifies. Queue
    state itself authorizes nothing either way."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    key = reference.visit_key()
    dag.exhausted_source_requests.add(key)
    semantic = ("storage_writers", "member", "decl:x")
    use = (root.id, "sample")
    certificate, proof = _t6b_hand_proof(use, key, semantic)
    result = _run_t6b_round(
        dag, samples, policies, coverage_certificates=[certificate],
        proof_version=0, applicability_proofs=[proof])
    assert key not in [item.visit_key() for item in result.references]
    assert key in result.proof_observation.relevant_obligation_keys
    assert key in result.proof_observation.covered_obligation_keys
    assert result.proof_authority is not None
    assert result.proof_authority.coverage_ok is True
    assert result.proof_authority.authorizes_stop is False
    selected = result.summary.get("selected_checkpoint") or {}
    # Coverage (not queue state) authorizes: the exhausted
    # obligation verifies through proof now that replay runs.
    assert result.action == "verified"
    assert selected.get("authorizes_discovery_stop", False) is True


def test_proof_authority_stale_applicability_refuses():
    """H: coverage current but the only applicability proof is old —
    applicability stays false and stop stays false."""
    from dataclasses import replace
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    stale_proof = replace(proof, version=99)
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[stale_proof], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.coverage_ok is True
    assert authority.writer_coverage_verified is True
    assert authority.applicability_ok is False
    assert authority.missing_applicability_uses == (tuple(use),)
    assert authority.authorizes_stop is False


def test_proof_authority_writer_set_mismatch_refuses():
    """I: the applicability proof binds a different writer set than the
    current certificate — no valid pair exists, so the use is unproven."""
    from dataclasses import replace
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    skewed = replace(proof, writers=("other-site",))
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[skewed], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.applicability_ok is False
    assert authority.missing_applicability_uses == (tuple(use),)
    assert authority.authorizes_stop is False


def test_proof_authority_conflicting_proofs_fail_closed():
    """Two distinct VALID bindings for one scheduled use conflict:
    each proof matches its own current certificate for the same visit
    with a different writer set, so neither is trusted, while
    byte-identical duplicates merge silently. (Finding 6: conflict is
    scoped to otherwise-binding proofs; basis labels are not
    compared. A rival that matches no current certificate is simply
    ignored — see irrelevant_rival_ignored.)"""
    from dataclasses import replace
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    rival_certificate, rival = _proof_pair(
        use, key, semantic, writers=("other-site",))
    rival = replace(rival, basis="guard-exact-producer")
    conflicted = _authority(
        observation=observation,
        certificates=[certificate, rival_certificate],
        applicability_proofs=[proof, rival], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert conflicted.conflicting_applicability_uses == (tuple(use),)
    assert conflicted.applicability_ok is False
    assert conflicted.authorizes_stop is False
    assert conflicted.conflicting_applicability_uses == (tuple(use),)
    assert conflicted.applicability_ok is False
    assert conflicted.authorizes_stop is False
    duet = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[proof, proof], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert duet.conflicting_applicability_uses == ()
    assert duet.applicability_ok is True
    assert duet.authorizes_stop is True


def test_proof_authority_split_use_isolation():
    """F: one shared certificate, two uses — the proven use does not
    carry the unproven one. Applicability stays per-use."""
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use_a = ("op-a", "signal-a")
    use_b = ("op-b", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof_a = _proof_pair(use_a, key, semantic)
    split = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[proof_a], legacy_verified=True,
        applicability_uses=[(use_a, key), (use_b, key)])
    assert split.applicability_ok is False
    assert split.missing_applicability_uses == (tuple(use_b),)
    assert split.authorizes_stop is False
    whole = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[
            proof_a,
            _proof_pair(use_b, key, semantic)[1],
        ],
        legacy_verified=True,
        applicability_uses=[(use_a, key), (use_b, key)])
    assert whole.applicability_ok is True
    assert whole.authorizes_stop is True


def test_proof_authority_multi_obligation_conjunction():
    """Two covered obligations, each with its own use: every use must
    prove — no any() semantics over uses or obligations."""
    key_a = ("storage_writers", "global", "decl-a")
    key_b = ("storage_writers", "global", "decl-b")
    semantic_a = ("storage_writers", "global", "decl-a")
    semantic_b = ("storage_writers", "global", "decl-b")
    use_a = ("op-a", "signal-a")
    use_b = ("op-b", "signal-b")
    observation = _proof_observation(
        relevant=[key_a, key_b], covered=[key_a, key_b])
    certificate_a, proof_a = _proof_pair(use_a, key_a, semantic_a)
    certificate_b, _proof_b = _proof_pair(use_b, key_b, semantic_b)
    half = _authority(
        observation=observation,
        certificates=[certificate_a, certificate_b],
        applicability_proofs=[proof_a], legacy_verified=True,
        applicability_uses=[(use_a, key_a), (use_b, key_b)])
    assert half.coverage_ok is True
    assert half.applicability_ok is False
    assert half.missing_applicability_uses == (tuple(use_b),)
    assert half.authorizes_stop is False


def test_collect_applicability_uses():
    """Use assembly: reference origins × operands plus exact local-need
    pairs, deduplicated, skipping origin-less/operand-less entries."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        collect_applicability_uses,
    )
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="gain", kind="storage_writers",
        origin_vertex_ids=["op-a", "op-b"],
        origin_operands=["signal-a", ""],
        identity={"kind": "global", "symbol": "gain", "root": "gain",
                  "file": "s.cpp", "declaration_id": "global:decl",
                  "declaration_proven": True})
    assert collect_applicability_uses([reference], []) == (
        ((("op-a", "signal-a"), reference.visit_key()),
         (("op-b", "signal-a"), reference.visit_key())))
    needs = [{"vertex_id": "op-need", "operand": "signal-n",
              "source_requests": []}]
    assert collect_applicability_uses([], needs) == ()
    assert collect_applicability_uses([], []) == ()


def _t6b_round_dag():
    """Hand DAG mirroring the checkpoint-controller fixture shape, with
    samples that verify the terminal when nothing is unresolved."""
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    def expression(text, *inputs):
        return {"text": text, "lowered_text": text,
                "input_symbols": list(inputs),
                "input_identities": {}, "call_results": [], "exact": True}

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
    policies = {signal: {"method": "linear"} for signal in samples}
    dag = build_mechanism_dag(bindings, "command.value",
                              logged_signals=set(samples))
    return dag, samples, policies


def _run_t6b_round(dag, samples, policies, **kwargs):
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_checkpoint_round,
    )
    return evaluate_checkpoint_round(
        dag, parameter_values={}, observed_signals=set(samples),
        signal_policies=policies,
        load_samples=lambda _view, _observed: samples, **kwargs)


def _t6b_hand_proof(use, visit, semantic, writers=("w:1",), version=0):
    """Mechanism-test proof/cert pair with fully controlled keys (the
    combination logic is under test, not derivation)."""
    from flight_log_agent.analysis.coverage import (
        WriterApplicabilityProof,
        WriterCoverageCertificate,
    )
    certificate = WriterCoverageCertificate(
        obligation_key=tuple(semantic),
        scheduling_key=tuple(visit),
        declaration=("global", semantic[2] if len(semantic) > 2 else "",
                     "signal"),
        receiver_context=("", "", ""),
        strategy="storage-internal-only",
        boundary=("src/lib/sg.cpp",),
        examined=("src/lib/sg.cpp",),
        version=version,
        assumptions=("file-linkage-closed",
                     "writer-syntax-enumerated"),
        writers=tuple(writers),
    )
    proof = WriterApplicabilityProof(
        use_key=tuple(use),
        scheduling_key=tuple(visit),
        obligation_key=tuple(semantic),
        declaration=("global", semantic[2] if len(semantic) > 2 else "",
                     "signal"),
        writers=tuple(writers),
        producer_vertex="op-producer",
        basis="single-exact-producer",
        supporting_facts=("use:op",),
        version=version,
        receiver_context=("", "", ""),
    )
    return certificate, proof


def _t6b_round_dag():
    """Hand DAG mirroring the checkpoint-controller shape for authority
    round tests (no mini-tree needed; obligation refs are appended)."""
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    def expression(text, *inputs):
        return {"text": text, "lowered_text": text,
                "input_symbols": list(inputs),
                "input_identities": {}, "call_results": [], "exact": True}

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
    policies = {signal: {"method": "linear"} for signal in samples}
    dag = build_mechanism_dag(bindings, "command.value",
                              logged_signals=set(samples))
    return dag, samples, policies


def _run_t6b_round(dag, samples, policies, **kwargs):
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_checkpoint_round,
    )
    return evaluate_checkpoint_round(
        dag, parameter_values={}, observed_signals=set(samples),
        signal_policies=policies,
        load_samples=lambda _view, _observed: samples, **kwargs)


def test_proof_discharged_legacy_dirt_still_records_legacy_false():
    """O (Gate-B evolution): raw legacy stays False for a covered
    source obligation, while proof-adjusted final authority verifies —
    raw legacy_verified is diagnostic, round verified is decisive."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    key = reference.visit_key()
    semantic = ("storage_writers", "member", "decl:x")
    use = (root.id, "sample")
    certificate, proof = _t6b_hand_proof(use, key, semantic)
    result = _run_t6b_round(
        dag, samples, policies, coverage_certificates=[certificate],
        proof_version=0, applicability_proofs=[proof])
    assert result.action == "verified"
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("authorizes_discovery_stop", False) is True
    authority = result.proof_authority
    assert authority is not None
    assert authority.legacy_verified is False
    assert authority.coverage_ok is True
    assert authority.applicability_ok is True
    assert authority.writer_coverage_verified is True
    assert authority.applicability_verified is True
    assert authority.authorizes_stop is False


def test_round_pending_construction_blocks_with_full_proof():
    """P: an unresolved construction requirement blocks stop even when
    coverage and applicability both prove — flags never imply stop."""
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    dag.pending_construction = [root.id]
    result = _run_t6b_round(
        dag, samples, policies)
    assert result.action != "verified"
    assert result.proof_authority is not None
    assert result.proof_authority.legacy_verified is False
    assert result.proof_authority.authorizes_stop is False


def test_round_exhausted_without_proof_stays_unresolved():
    """K: scheduler exhaustion without proof changes nothing about
    authority — and the gate stays disengaged without threaded proof."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    dag.exhausted_source_requests.add(reference.visit_key())
    result = _run_t6b_round(dag, samples, policies)
    assert result.action != "verified"
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("authorizes_discovery_stop", False) is False
    assert result.proof_authority is not None
    assert result.proof_authority.gate_active is False
    assert result.proof_observation is not None
    assert (reference.visit_key() in
            result.proof_observation.relevant_obligation_keys)


def test_round_stale_certificate_covers_nothing():
    """G-stale at round level: a stale threaded certificate leaves the
    obligation uncovered; authority follows the current (empty) proof."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    key = reference.visit_key()
    certificate, _proof = _t6b_hand_proof(
        (root.id, "sample"), key,
        ("storage_writers", "member", "decl:x"), version=99)
    result = _run_t6b_round(
        dag, samples, policies, coverage_certificates=[certificate],
        proof_version=0)
    assert result.proof_observation is not None
    assert key in result.proof_observation.relevant_obligation_keys
    assert key in result.proof_observation.uncovered_obligation_keys
    assert result.proof_authority is not None
    assert result.proof_authority.coverage_ok is False
    assert result.proof_authority.authorizes_stop is False
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("authorizes_discovery_stop", False) is False


def test_round_vacuous_threaded_preserves_legacy_stop():
    """M-vacuous at round level: threaded proof on a clean verified
    terminal changes nothing — stop follows legacy, flags stay false."""
    dag, samples, policies = _t6b_round_dag()
    plain = _run_t6b_round(dag, samples, policies)
    plain_selected = plain.summary.get("selected_checkpoint") or {}
    assert plain.action == "verified"
    assert plain_selected.get("authorizes_discovery_stop") is True
    certificate, _proof = _t6b_hand_proof(
        ("op-nowhere", "signal-nowhere"),
        ("storage_writers", "global", "decl-foreign"),
        ("storage_writers", "global", "decl-foreign"), version=0)
    threaded = _run_t6b_round(
        dag, samples, policies, coverage_certificates=[certificate],
        proof_version=0)
    assert threaded.action == "verified"
    threaded_selected = threaded.summary.get("selected_checkpoint") or {}
    assert threaded_selected.get("authorizes_discovery_stop") is True
    assert threaded_selected.get("writer_coverage_verified", False) is False
    assert threaded_selected.get("applicability_verified", False) is False
    assert threaded.proof_observation is not None
    assert threaded.proof_observation.relevant_empty is True
    assert threaded.proof_authority is not None
    assert threaded.proof_authority.gate_active is True
    assert threaded.proof_authority.writer_coverage_verified is False
    assert threaded.proof_authority.applicability_verified is False
    assert threaded.proof_authority.authorizes_stop is True


def test_proof_authority_partial_coverage_blocks_one_uncovered():
    """D: two relevant obligations, one covered — coverage stays false
    even though most proof is present."""
    key_a = ("storage_writers", "global", "decl-a")
    key_b = ("storage_writers", "global", "decl-b")
    semantic_a = ("storage_writers", "global", "decl-a")
    use_a = ("op-a", "signal-a")
    observation = _proof_observation(
        relevant=[key_a, key_b], covered=[key_a])
    certificate_a, proof_a = _proof_pair(use_a, key_a, semantic_a)
    authority = _authority(
        observation=observation,
        certificates=[certificate_a],
        applicability_proofs=[proof_a], legacy_verified=True,
        applicability_uses=[(use_a, key_a)])
    assert authority.coverage_ok is False
    assert authority.uncovered_relevant == (key_b,)
    assert authority.authorizes_stop is False


def test_proof_authority_cross_declaration_proof_rejected():
    """J: a proof bound to another declaration cannot satisfy this use,
    even when its writer set and version look plausible."""
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, _proof = _proof_pair(use, key, semantic)
    foreign = _proof_pair(use, ("storage_writers", "global", "decl-b"),
                          ("storage_writers", "global", "decl-b"))[1]
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[foreign], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.applicability_ok is False
    assert authority.missing_applicability_uses == (tuple(use),)
    assert authority.authorizes_stop is False


def test_proof_authority_vacuous_use_needs_no_proof():
    """Covered obligations with no concrete uses require no
    applicability proof: nothing consumes the value, so nothing must
    govern it. Flags stay false while the internal ok stays true."""
    key = ("storage_writers", "global", "decl-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    authority = _authority(
        observation=observation, legacy_verified=True,
        applicability_uses=[])
    assert authority.applicability_ok is True
    assert authority.applicability_verified is False
    assert authority.authorizes_stop is True


def test_proof_authority_omitted_proof_vetoes_nonempty_relevance():
    """R1/Finding 1: non-empty relevance + omitted proof inputs vetoes
    stop even when legacy verifies — omitted proof is required proof
    absent, never a disengaged gate."""
    key = ("storage_writers", "global", "decl-a")
    observation = _proof_observation(relevant=[key], covered=[])
    for legacy in (False, True):
        authority = _authority(
            observation=observation, proof_version=None,
            legacy_verified=legacy,
            applicability_uses=[(("op-use", "signal-a"), key)])
        assert authority.writer_coverage_verified is False
        assert authority.applicability_verified is False
        assert authority.authorizes_stop is False


def test_proof_authority_omitted_proof_preserves_empty_nondegenerate_path():
    """R2/Finding 1 §9 exception: empty relevance + non-degenerate
    scope follows legacy; proof flags stay non-authoritative."""
    observation = _proof_observation(relevant=[])
    authority = _authority(
        observation=observation, proof_version=None, legacy_verified=True)
    assert authority.writer_coverage_verified is False
    assert authority.applicability_verified is False
    assert authority.authorizes_stop is True
    denied = _authority(
        observation=observation, proof_version=None,
        legacy_verified=False)
    assert denied.authorizes_stop is False


def test_proof_authority_omitted_proof_degenerate_vetoes():
    """R3/Finding 1: empty relevance + degenerate scope vetoes even
    with legacy verified and proof omitted."""
    observation = _proof_observation(relevant=[], degenerate=True)
    authority = _authority(
        observation=observation, proof_version=None, legacy_verified=True)
    assert authority.authorizes_stop is False


def test_proof_authority_foreign_scheduling_proof_rejected():
    """R4/Finding 5: same use/obligation/writers/version but a
    DIFFERENT proof scheduling key does not satisfy the required
    visit — applicability is per call scope."""
    from dataclasses import replace
    key = ("storage_writers", "global", "decl-a")
    foreign_visit = ("storage_writers", "global", "decl-foreign")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    foreign = replace(proof, scheduling_key=tuple(foreign_visit))
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[foreign], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.applicability_ok is False
    assert authority.missing_applicability_uses == (tuple(use),)
    assert authority.authorizes_stop is False


def test_proof_authority_exact_scheduling_proof_accepted():
    """R5/Finding 5 control: the exact scheduling context qualifies —
    same bindings as R4 except the visit matches."""
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[proof], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.applicability_ok is True
    assert authority.authorizes_stop is True


def test_proof_authority_irrelevant_rival_ignored():
    """R6/Finding 6: a rival sharing only the use key but mismatching
    the current obligation/writer set is ignored — the valid exact
    proof still satisfies applicability with no conflict."""
    from dataclasses import replace
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    rival = replace(
        proof,
        obligation_key=("storage_writers", "global", "decl-b"),
        writers=("other-site",),
        basis="guard-exact-producer")
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[proof, rival], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.conflicting_applicability_uses == ()
    assert authority.applicability_ok is True
    assert authority.authorizes_stop is True


def test_proof_authority_dual_positive_basis_corroborates():
    """R7/Finding 6: two otherwise-valid proofs for the SAME exact
    current claim differing only in positive basis corroborate the
    claim rather than conflict."""
    from dataclasses import replace
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    twin = replace(proof, basis="guard-exact-producer")
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[proof, twin], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.conflicting_applicability_uses == ()
    assert authority.applicability_ok is True
    assert authority.authorizes_stop is True


def test_proof_authority_invalid_bindings_still_veto():
    """R8/conflict-relaxation guard: two invalid proofs (wrong writer
    set; stale version) still leave the use unproven — relaxation
    never manufactures applicability."""
    from dataclasses import replace
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-use", "signal-a")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate, proof = _proof_pair(use, key, semantic)
    skewed = replace(proof, writers=("other-site",))
    stale = replace(proof, version=99)
    authority = _authority(
        observation=observation, certificates=[certificate],
        applicability_proofs=[skewed, stale], legacy_verified=True,
        applicability_uses=[(use, key)])
    assert authority.missing_applicability_uses == (tuple(use),)
    assert authority.applicability_ok is False
    assert authority.authorizes_stop is False


# --- P0: production evidence source ---

def _p0_tree(tmp_path):
    return _mini_tree(tmp_path, {
        "src/modules/example/rtl.cpp": """
#include "rtl.h"

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
#include "rtl.h"

void Rtl::update()
{
    speed_s speed_data{};
    orb_copy(ORB_ID(speed), subscription, &speed_data);
    _dest_val = speed_data.value;
}
""",
    })


def _p0_discover(profiler, tmp_path, **kwargs):
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    return discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=["pick_altitude"],
        terminal="_final_out",
        source_hash="hash",
        terminal_file="src/modules/example/rtl.cpp",
        **kwargs)


def test_production_frontier_resolve_retains_evidence(tmp_path):
    """P0 RED: a natural writer obligation reaching the production
    frontier retains version-stamped search attempts while discovery
    behavior stays identical with and without the sink."""
    profiler = _p0_tree(tmp_path)
    sink: list = []
    result = _p0_discover(profiler, tmp_path, attempt_sink=sink)
    plain = _p0_discover(profiler, tmp_path)
    assert sink, "production frontier resolution retained no evidence"
    assert all(
        attempt.universe_ref.get("search_version") is not None
        for attempt in sink
    )
    assert {a.universe_ref["search_version"] for a in sink} == {0, 1}
    assert result.files_loaded == plain.files_loaded
    assert result.stop_reason == plain.stop_reason
    assert result.dag is not None and plain.dag is not None
    assert [v.id for v in result.dag.vertices] == [
        v.id for v in plain.dag.vertices]
    assert [e.id for e in result.dag.edges] == [
        e.id for e in plain.dag.edges]


def _p0_candidate_dump(candidates):
    import dataclasses
    dumped = []
    for item in candidates:
        as_dict = (
            dataclasses.asdict(item)
            if dataclasses.is_dataclass(item)
            else item.model_dump(mode="json")
        )
        dumped.append(as_dict)
    return dumped


def test_evidence_path_preserves_resolver_semantics(tmp_path):
    """P0 pins B–F: evidence-enabled resolution returns semantically
    identical candidates/admission for success, miss, ambiguity, and
    unavailable shapes (T1 fixtures)."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
        UnresolvedSourceReference,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/widget.cpp": (
            "struct Widget { float level; void fill() { level = 1; } };"
        ),
        "src/lib/over.cpp": "void tune(int x) {} void tune(float x) {}",
        "src/lib/user.cpp": "void run() { helper.adjust(1); }",
    })
    resolver = SourceExpansionResolver(profiler, "hash")
    identity = inputs.structure.symbol_identity(
        "level", file="src/lib/widget.cpp",
        callable_id="Widget::fill", function_name="Widget::fill")
    references = [
        UnresolvedSourceReference(
            symbol="level", kind="storage_writers",
            file="src/lib/widget.cpp", callable_id="Widget::fill",
            identity=identity),
        UnresolvedSourceReference(
            symbol="tune", kind="callable", file="src/lib/over.cpp",
            argument_count=1),
        UnresolvedSourceReference(
            symbol="NoSuchEntity", kind="class",
            file="src/lib/over.cpp"),
        UnresolvedSourceReference(
            symbol="adjust", kind="callable", file="src/lib/user.cpp",
            callable_id="run", receiver="helper", receiver_type="",
            argument_count=1),
    ]
    for reference in references:
        legacy = resolver.resolve(reference, inputs.structure)
        sink: list = []
        evidenced, evidence = resolver.resolve_with_evidence(
            reference, inputs.structure, sink)
        assert _p0_candidate_dump(evidenced) == _p0_candidate_dump(legacy)
        assert evidence.attempts == sink
        assert sink, "evidence path recorded no attempt"


def test_evidence_version_stamp_and_session_isolation(tmp_path):
    """P0 pins G–I: explicit version stamps, per-session sink
    isolation, and deterministic same-version repeats."""
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
        UnresolvedSourceReference,
    )
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/widget.cpp": (
            "struct Widget { float level; void fill() { level = 1; } };"
        ),
    })
    identity = inputs.structure.symbol_identity(
        "level", file="src/lib/widget.cpp",
        callable_id="Widget::fill", function_name="Widget::fill")
    reference = UnresolvedSourceReference(
        symbol="level", kind="storage_writers",
        file="src/lib/widget.cpp", callable_id="Widget::fill",
        identity=identity)
    resolver = SourceExpansionResolver(profiler, "hash")
    stamped: list = []
    _, stamped_evidence = resolver.resolve_with_evidence(
        reference, inputs.structure, stamped, universe_version=3)
    assert stamped
    assert all(
        attempt.universe_ref.get("search_version") == 3
        for attempt in stamped)
    assert stamped_evidence.universe_ref.get("search_version") == 3
    unstamped: list = []
    resolver.resolve_with_evidence(
        reference, inputs.structure, unstamped)
    assert all(
        "search_version" not in attempt.universe_ref
        for attempt in unstamped)

    tree_profiler = _p0_tree(tmp_path)
    sink_a: list = []
    sink_b: list = []
    _p0_discover(tree_profiler, tmp_path, attempt_sink=sink_a)
    _p0_discover(tree_profiler, tmp_path, attempt_sink=sink_b)
    assert sink_a and sink_b
    assert sink_a is not sink_b
    assert not any(a is b for a in sink_a for b in sink_b)
    assert len(sink_a) == len(sink_b)

    from flight_log_agent.analysis.coverage import CoverageSearchState
    shared_state = CoverageSearchState()
    replay_sink: list = []
    _p0_discover(tree_profiler, tmp_path, attempt_sink=replay_sink,
                 search_state=shared_state)
    first_pass = len(replay_sink)
    assert first_pass > 0
    _p0_discover(tree_profiler, tmp_path, attempt_sink=replay_sink,
                 search_state=shared_state)
    assert len(replay_sink) == first_pass


# --- P1B: exact-pair consumers + bounded fallback ---

def _p1b_rigged_pairs(reference, use_a, use_b):
    """Crafted {(A,x),(B,y)} shape over noisy legacy arrays, through
    validation-bypassing copy (production merge always normalizes)."""
    return reference.model_copy(update={
        "origin_vertex_ids": [use_a, use_b],
        "origin_operands": ["kgain", "other"],
        "origin_uses": [(use_a, "kgain"), (use_b, "other")],
    })


def test_t5_rejects_cartesian_invented_pair(tmp_path):
    """P1B-A: (A,y) shares sides with known pairs but is not an exact
    use — T5 must refuse on pair membership, not later stages."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_USE_IDENTITY_MISMATCH,
        derive_writer_applicability,
    )
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    rigged = _p1b_rigged_pairs(reference_a, use_a.id, "op-foreign")
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    outcome = derive_writer_applicability(
        rigged, use_a.id, "other", cert, dag_a, 0,
        conditional_writer_ids=())
    assert outcome.proof is None
    assert outcome.refusal == APPLICABILITY_USE_IDENTITY_MISMATCH


def test_t5_accepts_real_exact_pair(tmp_path):
    """P1B-B: the exact pair (A,x) still proceeds through normal T5
    qualification on the same rigged reference."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER,
        derive_writer_applicability,
    )
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    rigged = _p1b_rigged_pairs(reference_a, use_a.id, "op-foreign")
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    positive = derive_writer_applicability(
        rigged, use_a.id, "kgain", cert, dag_a, 0,
        conditional_writer_ids=())
    assert positive.refusal == "", positive.refusal_detail
    assert positive.proof is not None
    assert positive.proof.basis == APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER


def test_t6b_enumerates_exact_pairs_only():
    """P1B-C/D/G: exact pairs are authoritative — two pairs out, no
    Cartesian cross-pairs, duplicates collapsed, noisy legacy arrays
    ignored."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        collect_applicability_uses,
    )
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/main.cpp",
        callable_id="usea", origin_vertex_ids=["op-a", "op-b"],
        origin_operands=["kgain", "other"],
        origin_uses=[("op-a", "kgain"), ("op-b", "other"),
                     ("op-a", "kgain")])
    visit = reference.visit_key()
    assert collect_applicability_uses([reference], []) == (
        ((("op-a", "kgain"), visit), (("op-b", "other"), visit)))


def test_concrete_use_provenance_labels():
    """P1B-E/F: single-single legacy entails one pair; ambiguous
    legacy stays compatibility-cartesian (fail-closed); exact wins."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    from flight_log_agent.analysis.coverage import reference_concrete_uses
    single = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/main.cpp",
        callable_id="usea", origin_vertex_ids=["op-a"],
        origin_operands=["kgain"])
    assert reference_concrete_uses(single) == (
        [("op-a", "kgain")], "entailed")
    ambiguous = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/main.cpp",
        callable_id="usea", origin_vertex_ids=["op-a", "op-b"],
        origin_operands=["kgain", "other"])
    uses, provenance = reference_concrete_uses(ambiguous)
    assert provenance == "compatibility"
    assert uses == [("op-a", "kgain"), ("op-a", "other"),
                    ("op-b", "kgain"), ("op-b", "other")]
    exact = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/main.cpp",
        callable_id="usea", origin_vertex_ids=["op-a", "op-b"],
        origin_operands=["kgain", "other"],
        origin_uses=[("op-a", "kgain"), ("op-b", "other")])
    assert reference_concrete_uses(exact) == (
        [("op-a", "kgain"), ("op-b", "other")], "exact")


def test_operand_less_reference_yields_no_use():
    """P1B-H: origins without operands (and no pairs) fabricate no
    concrete applicability use."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        collect_applicability_uses,
    )
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/main.cpp",
        callable_id="usea", origin_vertex_ids=["op-a"],
        origin_operands=[])
    assert collect_applicability_uses([reference], []) == ()


def test_exact_pairs_remove_fabricated_liveness_veto(tmp_path):
    """P1B liveness: with exact pairs, proving the two REAL uses
    authorizes — the Cartesian-fabricated vetoes are gone while every
    real requirement remains."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/main.cpp",
        callable_id="usea", origin_vertex_ids=["op-a", "op-b"],
        origin_operands=["kgain", "other"],
        origin_uses=[("op-a", "kgain"), ("op-b", "other")])
    key = reference.visit_key()
    semantic = ("storage_writers", "global", "decl-kgain")
    use_a, use_b = ("op-a", "kgain"), ("op-b", "other")
    observation = _proof_observation(relevant=[key], covered=[key])
    certificate_a, proof_a = _proof_pair(use_a, key, semantic)
    _certificate_b, proof_b = _proof_pair(use_b, key, semantic)
    from flight_log_agent.analysis.checkpoint_discovery import (
        collect_applicability_uses,
    )
    uses = collect_applicability_uses([reference], [])
    assert [use for use, _visit in uses] == [use_a, use_b]
    authority = _authority(
        observation=observation,
        certificates=[certificate_a],
        applicability_proofs=[proof_a, proof_b], legacy_verified=True,
        applicability_uses=uses)
    assert authority.missing_applicability_uses == ()
    assert authority.applicability_ok is True
    assert authority.authorizes_stop is True


def test_exact_pairs_preserve_declaration_isolation(tmp_path):
    """P1B-J: pair-aware membership does not weaken cross-declaration
    isolation — a foreign declaration's use still refuses."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_USE_IDENTITY_MISMATCH,
        derive_writer_applicability,
    )
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    foreign = reference_a.model_copy(update={
        "origin_uses": [(use_a.id, "kgain")],
    })
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    outcome = derive_writer_applicability(
        foreign, "op-foreign", "kgain", cert, dag_a, 0,
        conditional_writer_ids=())
    assert outcome.proof is None
    assert outcome.refusal == APPLICABILITY_USE_IDENTITY_MISMATCH


# --- P2A: session proof-store model ---

def _p2a_store():
    from flight_log_agent.analysis.coverage import CoverageProofStore
    return CoverageProofStore()


def test_proof_store_starts_empty():
    """P2A-A: a fresh store holds nothing and exposes no authority."""
    store = _p2a_store()
    assert store.certificates_for(0) == ()
    assert store.proofs_for(0) == ()
    for name in ("coverage_verified", "applicability_verified",
                 "authorizes_stop", "verified", "complete",
                 "safe_to_stop", "visited", "retired", "exhausted",
                 "eligible", "mark_visited", "mark_exhausted"):
        assert not hasattr(store, name), name


def test_proof_store_certificate_isolation():
    """P2A-B: certificates retrieve by exact (version, scheduling)
    and (version, obligation) — unrelated keys miss."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    certificate, _proof = _proof_pair(("op-a", "signal-a"), key, semantic)
    assert store.store_certificate(certificate) is True
    assert store.certificates_for(0) == (certificate,)
    assert store.certificates_for_obligation(0, semantic) == (certificate,)
    assert store.certificates_for_visit(0, key) == (certificate,)
    assert store.certificates_for(1) == ()
    other_visit = ("storage_writers", "global", "decl-b")
    assert store.certificates_for_visit(0, other_visit) == ()
    assert store.certificates_for_obligation(
        0, ("storage_writers", "global", "decl-b")) == ()
    assert _proof_pair(("op-a", "signal-a"), other_visit,
                       semantic)[0] not in store.certificates_for(0)


def test_proof_store_proof_isolation():
    """P2A-C: proofs retrieve by exact (version, use, visit)."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    _certificate, proof = _proof_pair(use, key, semantic)
    assert store.store_proof(proof) is True
    assert store.proofs_for(0) == (proof,)
    assert store.proof_for(0, use, key) == proof
    assert store.proof_for(0, ("op-b", "signal-a"), key) is None
    assert store.proof_for(0, use, ("storage_writers", "x", "y")) is None
    assert store.proof_for(1, use, key) is None


def test_proof_store_idempotent_insert():
    """P2A-D: identical re-insertion is a no-op, not a duplicate."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    certificate, proof = _proof_pair(("op-a", "signal-a"), key, semantic)
    assert store.store_certificate(certificate) is True
    assert store.store_certificate(certificate) is True
    assert store.store_proof(proof) is True
    assert store.store_proof(proof) is True
    assert store.certificates_for(0) == (certificate,)
    assert store.proofs_for(0) == (proof,)


def test_proof_store_conflicting_replacement_fail_closed():
    """P2A-E: same key with incompatible data is refused; the
    existing entry is retained, never silently merged."""
    from dataclasses import replace
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    certificate, proof = _proof_pair(use, key, semantic)
    assert store.store_certificate(certificate) is True
    assert store.store_proof(proof) is True
    rival_certificate = replace(certificate, writers=("other-site",))
    rival_proof = replace(proof, writers=("other-site",))
    assert store.store_certificate(rival_certificate) is False
    assert store.store_proof(rival_proof) is False
    assert store.certificates_for(0) == (certificate,)
    assert store.proofs_for(0) == (proof,)


def test_proof_store_version_isolation_and_prune():
    """P2A-F/G: v0 proof is non-current under v1; pruning drops stale
    versions while keeping current proof queryable."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    old_certificate, old_proof = _proof_pair(use, key, semantic, version=0)
    new_certificate, new_proof = _proof_pair(use, key, semantic, version=1)
    assert store.store_certificate(old_certificate) is True
    assert store.store_proof(old_proof) is True
    assert store.store_certificate(new_certificate) is True
    assert store.store_proof(new_proof) is True
    assert store.certificates_for(1) == (new_certificate,)
    assert store.proofs_for(1) == (new_proof,)
    assert old_certificate not in store.certificates_for(1)
    assert old_proof not in store.proofs_for(1)
    store.prune_older_than(1)
    assert store.certificates_for(0) == ()
    assert store.proofs_for(0) == ()
    assert store.certificates_for(1) == (new_certificate,)
    assert store.proofs_for(1) == (new_proof,)


def test_proof_store_deterministic_order():
    """P2A-H: insertion order never leaks into read order."""
    store = _p2a_store()
    key_a = ("storage_writers", "global", "decl-a")
    key_b = ("storage_writers", "global", "decl-b")
    certificate_a, _ = _proof_pair(
        ("op-a", "signal-a"), key_a, key_a)
    certificate_b, _ = _proof_pair(
        ("op-b", "signal-b"), key_b, key_b)
    assert store.store_certificate(certificate_b) is True
    assert store.store_certificate(certificate_a) is True
    assert store.certificates_for(0) == (certificate_a, certificate_b)


def test_proof_store_immutable_reads():
    """P2A-I: reads are tuples; mutating attempts cannot reach the
    store, and snapshots detach from later inserts."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    certificate, proof = _proof_pair(("op-a", "signal-a"), key, semantic)
    assert store.store_certificate(certificate) is True
    assert store.store_proof(proof) is True
    certificates = store.certificates_for(0)
    try:
        certificates[0] = certificate
        mutated = True
    except TypeError:
        mutated = False
    assert mutated is False
    snapshot = store.snapshot_for(0)
    other, other_proof = _proof_pair(
        ("op-b", "signal-b"),
        ("storage_writers", "global", "decl-b"),
        ("storage_writers", "global", "decl-b"))
    assert store.store_certificate(other) is True
    assert store.store_proof(other_proof) is True
    assert snapshot.certificates == (certificate,)
    assert snapshot.applicability_proofs == (proof,)


def _p2a_attempt(**overrides):
    from flight_log_agent.analysis.coverage import SearchAttempt
    fields = dict(
        obligation_key=("storage_writers", "global", "decl-a"),
        scheduling_key=("storage_writers", "global", "decl-a"),
        strategy="storage-internal-only",
        strategy_class="complete",
        examined_domain={"files": ("src/lib/sg.cpp",)},
        universe_ref={"search_version": 0},
        outcome="admitted",
        details={"file_verdicts": {"src/lib/sg.cpp": "exact:op"},
                 "writer_census": {"src/lib/sg.cpp": ["op"]}},
    )
    fields.update(overrides)
    return SearchAttempt(**fields)


def test_evidence_fingerprint_contract():
    """P2A-J: identical ordered evidence fingerprints equal;
    material changes differ; object identity never matters."""
    from flight_log_agent.analysis.coverage import evidence_fingerprint
    first = [_p2a_attempt(), _p2a_attempt(strategy="storage-owner-files",
                                         outcome="non-writer")]
    twin = [_p2a_attempt(), _p2a_attempt(strategy="storage-owner-files",
                                        outcome="non-writer")]
    assert first[0] is not twin[0]
    assert evidence_fingerprint(first) == evidence_fingerprint(twin)
    changed = [_p2a_attempt(outcome="non-writer"), twin[1]]
    assert evidence_fingerprint(changed) != evidence_fingerprint(first)
    reordered = [twin[1], twin[0]]
    assert evidence_fingerprint(reordered) != evidence_fingerprint(first)


def test_proof_store_evidence_state():
    """P2A-J store side: per-key fingerprint notes change exactly
    once; identical restatement is not a change."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    attempts = [_p2a_attempt()]
    assert store.note_evidence(0, key, attempts) is True
    assert store.note_evidence(0, key, [_p2a_attempt()]) is False
    assert store.note_evidence(0, key, [_p2a_attempt(outcome="x")]) is True
    assert store.note_evidence(1, key, attempts) is True


def test_proof_store_cross_declaration_isolation():
    """Same spelling under different declarations stays isolated
    through obligation-keyed indexing."""
    store = _p2a_store()
    key_a = ("storage_writers", "global", "decl-a")
    key_b = ("storage_writers", "global", "decl-b")
    certificate_a, _ = _proof_pair(("op-a", "s"), key_a, key_a)
    certificate_b, _ = _proof_pair(("op-a", "s"), key_b, key_b)
    assert store.store_certificate(certificate_a) is True
    assert store.store_certificate(certificate_b) is True
    assert store.certificates_for_obligation(0, key_a) == (certificate_a,)
    assert store.certificates_for_obligation(0, key_b) == (certificate_b,)


def test_proof_store_writer_set_distinction():
    """Same use/visit with different writer sets are not equivalent:
    the rival is refused and the original retained."""
    from dataclasses import replace
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    _certificate, proof = _proof_pair(use, key, semantic)
    assert store.store_proof(proof) is True
    assert store.store_proof(replace(proof, writers=("other",))) is False
    assert store.proofs_for(0) == (proof,)
    assert store.proof_for(0, use, key) == proof


def test_proof_store_snapshot_contract():
    """Current-version snapshot: only that version, deterministic
    immutable tuples, frozen value, no verification flags."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    certificate, proof = _proof_pair(use, key, semantic, version=1)
    stale_certificate, stale_proof = _proof_pair(use, key, semantic,
                                                 version=0)
    assert store.store_certificate(certificate) is True
    assert store.store_proof(proof) is True
    assert store.store_certificate(stale_certificate) is True
    assert store.store_proof(stale_proof) is True
    snapshot = store.snapshot_for(1)
    assert snapshot.version == 1
    assert snapshot.certificates == (certificate,)
    assert snapshot.applicability_proofs == (proof,)
    assert isinstance(snapshot.certificates, tuple)
    assert isinstance(snapshot.applicability_proofs, tuple)
    for name in ("coverage_verified", "applicability_verified",
                 "authorizes_stop"):
        assert not hasattr(snapshot, name), name
    import dataclasses
    try:
        snapshot.version = 99
        frozen = False
    except dataclasses.FrozenInstanceError:
        frozen = True
    assert frozen is True


# --- P2B: incremental T3 derivation from retained evidence ---

def _p2b_kzero_tree(tmp_path):
    files = {
        "src/main.cpp": (
            "static float kzero;\n"
            "float out;\n"
            "void use() { out = kzero * 3.0f; }\n"
        ),
    }
    return _mini_tree(tmp_path, files, backend="tree_sitter")


def _p2b_discover_kzero(tmp_path, **kwargs):
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    return discover_mechanism_dag(
        _p2b_kzero_tree(tmp_path),
        tmp_path / "cache",
        seeds=["use"],
        terminal="out",
        source_hash="hash",
        terminal_file="src/main.cpp",
        **kwargs)


def _p2b_spy(monkeypatch):
    import flight_log_agent.analysis.mechanism_discovery as discovery
    calls: list = []
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate as real_derive,
    )

    def spy(obligation, evidence, version):
        calls.append(evidence.obligation_key)
        return real_derive(obligation, evidence, version)

    monkeypatch.setattr(
        discovery, "derive_writer_coverage_certificate", spy)
    return calls


def test_p2b_production_evidence_derives_certificate(tmp_path):
    """P2B-A/H: a natural certifiable obligation flows production
    resolve → retained attempt → stored v0 certificate with exact
    identity, writers, and assumptions."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    sink: list = []
    store = CoverageProofStore()
    result = _p2b_discover_kzero(
        tmp_path, attempt_sink=sink, proof_store=store)
    assert result.dag is not None
    assert sink, "no production evidence retained"
    certificates = store.certificates_for(0)
    assert len(certificates) == 1
    certificate = certificates[0]
    reference = next(
        item for item in result.dag.unresolved_references
        if item.symbol == "kzero" and item.kind == "storage_writers")
    assert tuple(certificate.obligation_key) == (
        reference.kind, reference.identity.kind,
        reference.identity.declaration_id)
    assert tuple(certificate.scheduling_key) == reference.visit_key()
    assert certificate.version == 0
    assert tuple(certificate.writers) == ()
    assert set(certificate.assumptions) == {
        "file-linkage-closed", "writer-syntax-enumerated"}
    assert any(str(item).endswith("src/main.cpp")
               for item in certificate.boundary)


def test_p2b_no_evidence_no_derivation(tmp_path, monkeypatch):
    """P2B-B: without retained evidence T3 is never invoked and
    nothing is stored."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_coverage_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    calls = _p2b_spy(monkeypatch)
    store = CoverageProofStore()
    derive_current_coverage_proofs([], 0, [], store)
    assert calls == []
    assert store.certificates_for(0) == ()


def _p2b_internal_fixture(tmp_path, files=None):
    """One certifiable internal-linkage writer obligation with real
    resolver-produced evidence (storage-internal-only, admitted)."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
        UnresolvedSourceReference,
    )
    profiler, inputs = _evidence_setup(tmp_path, files or {
        "src/lib/g.cpp": (
            "static float kgain = 1.0f;\n"
            "void usea() { kgain = 3.0f; }\n"
        ),
    })
    identity = inputs.structure.symbol_identity(
        "kgain", file="src/lib/g.cpp",
        callable_id="usea", function_name="usea")
    assert identity.declaration_proven
    reference = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers",
        file="src/lib/g.cpp", callable_id="usea",
        identity=identity)
    resolver = SourceExpansionResolver(profiler, "hash")
    sink: list = []
    _, envelope = resolver.resolve_with_evidence(
        reference, inputs.structure, sink, universe_version=0)
    return reference, envelope


def test_p2b_unchanged_evidence_reuses(tmp_path, monkeypatch):
    """P2B-C: identical material evidence re-runs derive nothing —
    the existing certificate is reused (T3 call count stable)."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_coverage_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    reference, envelope = _p2b_internal_fixture(tmp_path)
    calls = _p2b_spy(monkeypatch)
    store = CoverageProofStore()
    derive_current_coverage_proofs([reference], 0, [envelope], store)
    assert len(calls) == 1
    first = store.certificates_for(0)
    assert len(first) == 1
    derive_current_coverage_proofs([reference], 0, [envelope], store)
    assert len(calls) == 1
    assert store.certificates_for(0) == first


def test_p2b_changed_evidence_rederives_and_replaces(tmp_path,
                                                    monkeypatch):
    """P2B-D: materially changed same-version evidence rederives and
    explicitly replaces the superseded certificate."""
    import dataclasses
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_coverage_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    reference, envelope = _p2b_internal_fixture(tmp_path)
    calls = _p2b_spy(monkeypatch)
    store = CoverageProofStore()
    derive_current_coverage_proofs([reference], 0, [envelope], store)
    assert len(calls) == 1
    old = store.certificates_for(0)
    assert len(old) == 1
    site = next(iter(old[0].writers))
    grown = dataclasses.replace(
        envelope.attempts[0],
        details={**(envelope.attempts[0].details or {}),
                 "writer_census": {"src/lib/g.cpp": [site, site + "#2"]}})
    grown_envelope = dataclasses.replace(
        envelope, attempts=(grown,))
    derive_current_coverage_proofs(
        [reference], 0, [envelope, grown_envelope], store)
    assert len(calls) == 2
    current = store.certificates_for(0)
    assert len(current) == 1
    assert set(current[0].writers) == {site, site + "#2"}


def test_p2b_changed_evidence_refusal_invalidates(tmp_path, monkeypatch):
    """P2B-E (tripwire): changed evidence whose rederivation REFUSES
    must not leave the old certificate current for the changed state."""
    import dataclasses
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_coverage_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    reference, envelope = _p2b_internal_fixture(tmp_path)
    _calls = _p2b_spy(monkeypatch)
    store = CoverageProofStore()
    derive_current_coverage_proofs([reference], 0, [envelope], store)
    assert len(store.certificates_for(0)) == 1
    ambiguous = dataclasses.replace(
        envelope.attempts[0], outcome="ambiguous",
        details={**(envelope.attempts[0].details or {}),
                 "colliding_identities": ["decl-a", "decl-b"]})
    poisoned = dataclasses.replace(envelope, attempts=(ambiguous,))
    derive_current_coverage_proofs(
        [reference], 0, [envelope, poisoned], store)
    assert store.certificates_for_obligation(
        0, tuple(envelope.obligation_key)) == ()
    assert store.note_evidence(
        0, tuple(envelope.scheduling_key),
        list(envelope.attempts) + [ambiguous]) is False
    before = len(_calls)
    derive_current_coverage_proofs(
        [reference], 0, [envelope, poisoned], store)
    assert len(_calls) == before
    assert store.certificates_for_obligation(
        0, tuple(envelope.obligation_key)) == ()


def test_p2b_unrelated_obligation_isolated(tmp_path, monkeypatch):
    """P2B-F: changed evidence for A rederives A only; B is derived
    once and never touched again."""
    import dataclasses
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_coverage_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/two.cpp": (
            "static float ka = 1.0f;\n"
            "static float kb = 2.0f;\n"
            "void usea() { ka = 3.0f; kb = 4.0f; }\n"
        ),
    })
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
        UnresolvedSourceReference,
    )
    refs = []
    envelopes = []
    for symbol in ("ka", "kb"):
        identity = inputs.structure.symbol_identity(
            symbol, file="src/lib/two.cpp",
            callable_id="usea", function_name="usea")
        assert identity.declaration_proven
        reference = UnresolvedSourceReference(
            symbol=symbol, kind="storage_writers",
            file="src/lib/two.cpp", callable_id="usea",
            identity=identity)
        resolver = SourceExpansionResolver(profiler, "hash")
        sink: list = []
        _, envelope = resolver.resolve_with_evidence(
            reference, inputs.structure, sink, universe_version=0)
        refs.append(reference)
        envelopes.append(envelope)
    by_obligation = {}
    real_derive_calls: list = []
    import flight_log_agent.analysis.mechanism_discovery as discovery
    from flight_log_agent.analysis.coverage import (
        derive_writer_coverage_certificate as real_derive,
    )

    def counting(obligation, evidence, version):
        by_obligation.setdefault(
            tuple(evidence.obligation_key), 0)
        by_obligation[tuple(evidence.obligation_key)] += 1
        real_derive_calls.append(evidence.obligation_key)
        return real_derive(obligation, evidence, version)

    monkeypatch.setattr(discovery, "derive_writer_coverage_certificate",
                        counting)
    store = CoverageProofStore()
    derive_current_coverage_proofs(refs, 0, envelopes, store)
    assert len(store.certificates_for(0)) == 2
    assert all(count == 1 for count in by_obligation.values())
    site = next(iter(store.certificates_for_obligation(
        0, tuple(envelopes[0].obligation_key))[0].writers))
    grown = dataclasses.replace(
        envelopes[0].attempts[0],
        details={**(envelopes[0].attempts[0].details or {}),
                 "writer_census": {"src/lib/two.cpp": [site, site + "#2"]}})
    grown_envelope = dataclasses.replace(envelopes[0], attempts=(grown,))
    derive_current_coverage_proofs(
        refs, 0, [envelopes[1], grown_envelope], store)
    assert by_obligation[tuple(envelopes[0].obligation_key)] == 2
    assert by_obligation[tuple(envelopes[1].obligation_key)] == 1
    assert len(store.certificates_for(0)) == 2


def test_p2b_version_extension_invalidates(tmp_path, monkeypatch):
    """P2B-G: v0 proof is never current under v1; v1 evidence derives
    a fresh v1 certificate."""
    import dataclasses
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_coverage_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    reference, envelope = _p2b_internal_fixture(tmp_path)
    calls = _p2b_spy(monkeypatch)
    store = CoverageProofStore()
    derive_current_coverage_proofs([reference], 0, [envelope], store)
    assert len(store.certificates_for(0)) == 1
    assert store.certificates_for(1) == ()
    stamped_attempts = tuple(
        dataclasses.replace(
            attempt,
            universe_ref={**(attempt.universe_ref or {}),
                          "search_version": 1})
        for attempt in envelope.attempts)
    v1_envelope = dataclasses.replace(
        envelope, attempts=stamped_attempts,
        universe_ref={**envelope.universe_ref, "search_version": 1})
    derive_current_coverage_proofs([reference], 1, [v1_envelope], store)
    assert len(calls) == 2
    assert len(store.certificates_for(1)) == 1
    assert store.certificates_for(1)[0].version == 1
    assert (store.certificates_for(1)[0]
            not in store.certificates_for(0))


def test_p2b_unsupported_class_stores_nothing(tmp_path):
    """P2B-I: heuristic-only evidence is retained but derives no
    certificate and no fake proof."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    sink: list = []
    store = CoverageProofStore()
    result = _p0_discover(
        _mini_tree(tmp_path, {
            "src/modules/example/rtl.cpp": """
#include "rtl.h"
void Rtl::pick_altitude() { _final_out = _dest_val + 1.0f; }
""",
            "src/modules/example/rtl.h": """
class Rtl { float _final_out; float _dest_val; };
""",
            "src/modules/example/dest.cpp": """
#include "rtl.h"
void Rtl::update() {
    speed_s speed_data{};
    orb_copy(ORB_ID(speed), subscription, &speed_data);
    _dest_val = speed_data.value;
}
""",
        }),
        tmp_path, attempt_sink=sink, proof_store=store)
    assert result.dag is not None
    assert sink, "expected retained frontier evidence"
    assert store.certificates_for(0) == ()
    assert store.certificates_for(1) == ()


def test_p2b_same_spelling_isolation(tmp_path):
    """P2B-J: same spelling under different proven declarations
    derives separately bound certificates."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_coverage_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    profiler, inputs = _evidence_setup(tmp_path, {
        "src/lib/a.cpp": "static float kgain = 1.0f;\n",
        "src/lib/b.cpp": "static float kgain = 2.0f;\n",
    })
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
        UnresolvedSourceReference,
    )
    refs = []
    envelopes = []
    for path in ("src/lib/a.cpp", "src/lib/b.cpp"):
        identity = inputs.structure.symbol_identity(
            "kgain", file=path, callable_id="", function_name="")
        assert identity.declaration_proven, path
        reference = UnresolvedSourceReference(
            symbol="kgain", kind="storage_writers", file=path,
            identity=identity)
        resolver = SourceExpansionResolver(profiler, "hash")
        sink: list = []
        _, envelope = resolver.resolve_with_evidence(
            reference, inputs.structure, sink, universe_version=0)
        refs.append(reference)
        envelopes.append(envelope)
    assert refs[0].visit_key() != refs[1].visit_key()
    store = CoverageProofStore()
    derive_current_coverage_proofs(refs, 0, envelopes, store)
    certificates = store.certificates_for(0)
    assert len(certificates) == 2
    for certificate, envelope in zip(
            sorted(certificates,
                   key=lambda item: repr(item.obligation_key)),
            sorted(envelopes,
                   key=lambda item: repr(item.obligation_key))):
        assert tuple(certificate.obligation_key) == tuple(
            envelope.obligation_key)
        assert tuple(certificate.scheduling_key) == tuple(
            envelope.scheduling_key)


def test_p2b_session_isolation(tmp_path):
    """P2B-K: independent sessions never share proof-store state."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    store_a = CoverageProofStore()
    store_b = CoverageProofStore()
    first = _p2b_discover_kzero(
        tmp_path, proof_store=store_a)
    second = _p2b_discover_kzero(
        tmp_path, proof_store=store_b)
    assert first.dag is not None and second.dag is not None
    assert len(store_a.certificates_for(0)) == 1
    assert len(store_b.certificates_for(0)) == 1
    assert store_a is not store_b
    import dataclasses
    lone = store_a.certificates_for(0)[0]
    assert store_a.replace_certificate(dataclasses.replace(
        lone, writers=("elsewhere",))) is True
    assert (store_b.certificates_for(0)[0].writers
            == lone.writers)


def test_p2b_checkpoint_unaffected(tmp_path):
    """P2B-L: certificates in the session store never reach the
    checkpoint in P2B — discovery behavior identical with/without."""
    result = _p2b_discover_kzero(tmp_path)
    from flight_log_agent.analysis.coverage import CoverageProofStore
    store = CoverageProofStore()
    sink: list = []
    threaded = _p2b_discover_kzero(
        tmp_path, attempt_sink=sink, proof_store=store)
    assert threaded.files_loaded == result.files_loaded
    assert threaded.stop_reason == result.stop_reason
    assert threaded.dag is not None and result.dag is not None
    assert [v.id for v in threaded.dag.vertices] == [
        v.id for v in result.dag.vertices]
    assert len(store.certificates_for(0)) == 1


def test_p2b_production_prunes_on_extension(tmp_path):
    """P2B prune: a seeded stale v0 certificate cannot survive the
    genuine v0→v1 extension of a production session."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    profiler = _p0_tree(tmp_path)
    _certificate, _proof = _proof_pair(
        ("op-a", "signal-a"),
        ("storage_writers", "global", "decl-seed"),
        ("storage_writers", "global", "decl-seed"), version=0)
    store = CoverageProofStore()
    assert store.store_certificate(_certificate) is True
    assert len(store.certificates_for(0)) == 1
    result = _p0_discover(
        profiler, tmp_path, proof_store=store)
    assert result.dag is not None
    assert store.certificates_for(0) == ()


def test_proof_store_certificate_replacement():
    """Review-1A: ordinary conflicting insert refuses, but explicit
    replacement supersedes the same logical key — exactly one entry
    remains and unrelated keys/versions are untouched."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    old_certificate, _proof = _proof_pair(use, key, semantic, version=0)
    new_certificate = old_certificate
    import dataclasses as _dataclasses
    new_certificate = _dataclasses.replace(
        old_certificate, writers=("w:2",),
        assumptions=("file-linkage-closed",
                     "writer-syntax-enumerated", "extra-assumption"))
    other_certificate, _ = _proof_pair(
        use, ("storage_writers", "global", "decl-b"),
        ("storage_writers", "global", "decl-b"), version=0)
    assert store.replace_certificate(new_certificate) is False
    assert store.store_certificate(old_certificate) is True
    assert store.store_certificate(other_certificate) is True
    assert store.store_certificate(new_certificate) is False
    assert store.certificates_for_obligation(0, semantic) == (
        old_certificate,)
    assert store.replace_certificate(new_certificate) is True
    assert store.certificates_for_obligation(0, semantic) == (
        new_certificate,)
    assert store.certificates_for(0) == (new_certificate,
                                         other_certificate)
    assert store.replace_certificate(old_certificate) is True
    assert store.certificates_for_obligation(0, semantic) == (
        old_certificate,)


def test_proof_store_proof_replacement():
    """Review-1B: same supersede pattern for applicability proofs,
    keyed by exact (version, use, visit)."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    _certificate, old_proof = _proof_pair(use, key, semantic, version=0)
    import dataclasses as _dataclasses
    new_proof = _dataclasses.replace(
        old_proof, writers=("w:1", "w:2"), basis="refined-basis",
        supporting_facts=("use:op-a", "producer:op-second"))
    other_use = ("op-b", "signal-b")
    _other_certificate, other_proof = _proof_pair(
        other_use, key, semantic, version=0)
    assert store.replace_proof(new_proof) is False
    assert store.store_proof(old_proof) is True
    assert store.store_proof(other_proof) is True
    assert store.store_proof(new_proof) is False
    assert store.proof_for(0, use, key) == old_proof
    assert store.replace_proof(new_proof) is True
    assert store.proof_for(0, use, key) == new_proof
    assert store.proof_for(0, other_use, key) == other_proof
    assert store.proofs_for(0) == (new_proof, other_proof)


def test_evidence_fingerprint_dict_order_canonical():
    """Review-3: identical semantic evidence with different nested
    dict insertion orders fingerprints equally; attempt order stays
    significant."""
    from flight_log_agent.analysis.coverage import evidence_fingerprint
    first = [_p2a_attempt(details={
        "file_verdicts": {"b.cpp": "exact:y", "a.cpp": "exact:x"},
        "writer_census": {"b.cpp": ["y2", "y1"], "a.cpp": ["x"]}})]
    second = [_p2a_attempt(details={
        "file_verdicts": {"a.cpp": "exact:x", "b.cpp": "exact:y"},
        "writer_census": {"a.cpp": ["x"], "b.cpp": ["y1", "y2"]}})]
    assert evidence_fingerprint(first) == evidence_fingerprint(second)


def test_proof_store_discard_certificate():
    """Gate-B A/C: exact-key discard removes one certificate, leaves
    siblings across obligation/scheduling/version untouched, and
    reports absence without mutation."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    certificate, _proof = _proof_pair(use, key, semantic, version=0)
    sibling_obligation, _ = _proof_pair(
        use, key, ("storage_writers", "global", "decl-b"), version=0)
    sibling_visit, _ = _proof_pair(
        use, ("storage_writers", "global", "decl-v"), semantic, version=0)
    sibling_version, _ = _proof_pair(use, key, semantic, version=1)
    for entry in (certificate, sibling_obligation, sibling_visit,
                  sibling_version):
        assert store.store_certificate(entry) is True
    assert store.discard_certificate(0, key, semantic) is True
    assert store.certificates_for_obligation(0, semantic) == (
        sibling_visit,)
    assert store.certificates_for_visit(0, key) == (sibling_obligation,)
    assert store.certificates_for_obligation(
        0, ("storage_writers", "global", "decl-b")) == (sibling_obligation,)
    assert store.certificates_for_visit(
        0, ("storage_writers", "global", "decl-v")) == (sibling_visit,)
    assert store.certificates_for(1) == (sibling_version,)
    assert store.discard_certificate(0, key, semantic) is False
    assert store.discard_certificate(0, key, ("missing",)) is False


def test_proof_store_discard_proof():
    """Gate-B D/E: exact-key discard removes one applicability proof;
    other uses, visits, and versions survive; absence reports False."""
    store = _p2a_store()
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    use = ("op-a", "signal-a")
    _certificate, proof = _proof_pair(use, key, semantic, version=0)
    _c2, other_use = _proof_pair(("op-b", "signal-b"), key, semantic,
                                 version=0)
    _c3, other_visit = _proof_pair(
        use, ("storage_writers", "global", "decl-v"), semantic, version=0)
    _c4, other_version = _proof_pair(use, key, semantic, version=1)
    for entry in (proof, other_use, other_visit, other_version):
        assert store.store_proof(entry) is True
    assert store.discard_proof(0, use, key) is True
    assert store.proof_for(0, use, key) is None
    assert store.proof_for(0, ("op-b", "signal-b"), key) == other_use
    assert store.proof_for(
        0, use, ("storage_writers", "global", "decl-v")) == other_visit
    assert store.proof_for(1, use, key) == other_version
    assert store.discard_proof(0, use, key) is False
    assert store.discard_proof(0, ("missing",), key) is False


# --- P2C: incremental T5 applicability derivation ---

def _p2c_continue_evaluator(dag, index):
    """Test checkpoint evaluator exposing empty conditional context
    on a branch-free fixture (emptiness established by inspection)."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        CheckpointRound,
    )
    return CheckpointRound(
        "continue", dag, list(dag.unresolved_references),
        {"action": "continue", "local_equation_checks": []})


def _p2c_t5_spy(monkeypatch):
    import flight_log_agent.analysis.mechanism_discovery as discovery
    calls: list = []
    from flight_log_agent.analysis.coverage import (
        derive_writer_applicability as real_derive,
    )

    def spy(reference, origin_vertex_id, operand, certificate, dag,
            version, conditional_writer_ids=()):
        calls.append((origin_vertex_id, operand,
                      tuple(getattr(certificate, "writers", None) or ())))
        return real_derive(reference, origin_vertex_id, operand,
                           certificate, dag, version,
                           conditional_writer_ids=conditional_writer_ids)

    monkeypatch.setattr(discovery, "derive_writer_applicability", spy)
    return calls


def test_p2c_production_chain_derives_proof(tmp_path):
    """P2C-A: natural obligation → retained evidence → real T3 cert →
    exact use → pure T5 proof stored with full payload."""
    from flight_log_agent.analysis.coverage import (
        APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER,
        CoverageProofStore,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    assert result_a.dag is not None
    store = CoverageProofStore()
    result = discover_mechanism_dag(
        profiler_a, tmp_path / "cache_a", seeds=["usea"],
        terminal="out_a", source_hash="hash",
        terminal_file="src/main.cpp", proof_store=store,
        checkpoint_evaluator=_p2c_continue_evaluator)
    assert result.dag is not None
    dag = result.dag
    use_a = next(vertex for vertex in dag.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference = next(item for item in dag.unresolved_references
                     if item.symbol == "kgain"
                     and item.kind == "storage_writers")
    visit = reference.visit_key()
    certificates = store.certificates_for(0)
    assert len(certificates) == 1
    proof = store.proof_for(0, (use_a.id, "kgain"), visit)
    assert proof is not None
    assert proof.version == 0
    assert tuple(proof.use_key) == (use_a.id, "kgain")
    assert tuple(proof.scheduling_key) == tuple(visit)
    assert tuple(proof.obligation_key) == tuple(
        certificates[0].obligation_key)
    assert set(proof.writers) == set(certificates[0].writers)
    assert proof.producer_vertex
    assert proof.basis == APPLICABILITY_BASIS_SINGLE_EXACT_PRODUCER
    assert tuple(proof.supporting_facts)
    assert tuple(proof.receiver_context) == tuple(
        certificates[0].receiver_context)


def test_p2c_no_certificate_no_derivation(tmp_path, monkeypatch):
    """P2C-B: exact use without a current certificate never invokes
    T5 and stores nothing."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    ( _profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    calls = _p2c_t5_spy(monkeypatch)
    store = CoverageProofStore()
    derive_current_applicability_proofs(
        result_a.dag.unresolved_references, 0, result_a.dag, store, set())
    assert calls == []
    assert store.proofs_for(0) == ()


def test_p2c_fabricated_pair_never_derived(tmp_path, monkeypatch):
    """P2C-D: T5 is never invoked with a Cartesian-invented pair, even
    when both sides appear in legacy arrays."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    rigged = reference_a.model_copy(update={
        "origin_vertex_ids": [use_a.id, "op-foreign"],
        "origin_operands": ["kgain", "other"],
        "origin_uses": [(use_a.id, "kgain")],
    })
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(cert) is True
    calls = _p2c_t5_spy(monkeypatch)
    derive_current_applicability_proofs(
        [rigged], 0, dag_a, store, set())
    assert (use_a.id, "other") not in [
        (origin, operand) for origin, operand, _writers in calls]
    assert (use_a.id, "kgain") in [
        (origin, operand) for origin, operand, _writers in calls]


def test_p2c_shared_coverage_per_use_proofs(tmp_path, monkeypatch):
    """P2C-E: one shared certificate, two exact uses → independent
    proof attempts per use, no certificate duplication."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    rigged = reference_a.model_copy(update={
        "origin_vertex_ids": [use_a.id, "op-second"],
        "origin_operands": ["kgain", "kgain"],
        "origin_uses": [(use_a.id, "kgain"), ("op-second", "kgain")],
    })
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(cert) is True
    calls = _p2c_t5_spy(monkeypatch)
    derive_current_applicability_proofs(
        [rigged], 0, dag_a, store, set())
    attempted = {(origin, operand) for origin, operand, _ in calls}
    assert (use_a.id, "kgain") in attempted
    assert ("op-second", "kgain") in attempted
    assert len(store.certificates_for(0)) == 1


def test_p2c_unchanged_proof_reused(tmp_path, monkeypatch):
    """P2C-F: identical current inputs rerun T5 zero times."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(cert) is True
    calls = _p2c_t5_spy(monkeypatch)
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, store, set())
    first_count = len(calls)
    assert first_count > 0
    first_proofs = store.proofs_for(0)
    assert len(first_proofs) == len(
        {tuple(proof.use_key) for proof in first_proofs})
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, store, set())
    assert len(calls) == first_count
    assert store.proofs_for(0) == first_proofs


def test_p2c_certificate_change_revalidates(tmp_path, monkeypatch):
    """P2C-G: same-version certificate replacement with a new writer
    set invalidates the old proof unless T5 re-proves the new set."""
    from dataclasses import replace
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    visit = reference_a.visit_key()
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(cert) is True
    calls = _p2c_t5_spy(monkeypatch)
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, store, set())
    old_proof = store.proof_for(0, (use_a.id, "kgain"), visit)
    assert old_proof is not None
    grown = replace(cert, writers=tuple(cert.writers) + ("extra-site",))
    assert store.replace_certificate(grown) is True
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, store, set())
    current = store.proof_for(0, (use_a.id, "kgain"), visit)
    assert current is None or set(current.writers) == set(grown.writers)
    assert current != old_proof or current is None
    assert len(calls) > 1


def test_p2c_conditional_control_mismatch_refuse(tmp_path, monkeypatch):
    """P2C-H/I/J/K: conditional membership, guarded control, and
    writer mismatch each refuse without storing positive proof."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, result_b) = _split_use_tree(
        tmp_path)
    dag_a, dag_b = result_a.dag, result_b.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    use_b = next(vertex for vertex in dag_b.vertices
                 if vertex.variable == "out_b"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    # Conditional producer blocks.
    blocked = CoverageProofStore()
    assert blocked.store_certificate(cert) is True
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, blocked, {use_a.id})
    assert blocked.proofs_for(0) == ()
    # Guarded use refuses via control.
    guarded = CoverageProofStore()
    assert guarded.store_certificate(cert) is True
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    guarded_ref = UnresolvedSourceReference.model_validate(
        reference_a.model_dump())
    calls = _p2c_t5_spy(monkeypatch)
    derive_current_applicability_proofs(
        [guarded_ref], 0, dag_b, guarded, set())
    assert guarded.proof_for(
        0, (use_b.id, "kgain"), guarded_ref.visit_key()) is None
    # Writer-mismatched certificate cannot prove.
    from dataclasses import replace
    skewed_store = CoverageProofStore()
    skewed = replace(cert, writers=("elsewhere",))
    assert skewed_store.store_certificate(skewed) is True
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, skewed_store, set())
    assert skewed_store.proof_for(
        0, (use_a.id, "kgain"), reference_a.visit_key()) is None
    assert calls is not None


def test_p2c_cross_declaration_isolated(tmp_path, monkeypatch):
    """P2C-L: a proof derived for declaration A never satisfies a
    same-spelling declaration B use."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(cert) is True
    _p2c_t5_spy(monkeypatch)
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, store, set())
    assert store.proof_for(
        0, (use_a.id, "kgain"), reference_a.visit_key()) is not None
    _other_profiler, other_inputs = _evidence_setup(tmp_path, {
        "src/lib/h.cpp": (
            "static float kgain = 9.0f;\nvoid other() { kgain = 3.0f; }\n"
        ),
    })
    other_identity = other_inputs.structure.symbol_identity(
        "kgain", file="src/lib/h.cpp",
        callable_id="other", function_name="other")
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    other_ref = UnresolvedSourceReference(
        symbol="kgain", kind="storage_writers", file="src/lib/h.cpp",
        callable_id="other", identity=other_identity,
        origin_vertex_ids=[use_a.id], origin_operands=["kgain"],
        origin_uses=[(use_a.id, "kgain")])
    assert other_ref.visit_key() != reference_a.visit_key()
    derive_current_applicability_proofs(
        [other_ref], 0, dag_a, store, set())
    assert store.proof_for(
        0, (use_a.id, "kgain"), other_ref.visit_key()) is None


def test_p2c_version_extension_starts_fresh(tmp_path, monkeypatch):
    """P2C-M: v0 proofs never appear under v1; v1 needs fresh certs."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(cert) is True
    calls = _p2c_t5_spy(monkeypatch)
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, store, set())
    assert store.proof_for(
        0, (use_a.id, "kgain"), reference_a.visit_key()) is not None
    derive_current_applicability_proofs(
        [reference_a], 1, dag_a, store, set())
    assert store.proofs_for(1) == ()
    assert store.proof_for(
        1, (use_a.id, "kgain"), reference_a.visit_key()) is None
    assert len(calls) >= 1


def test_p2c_removed_use_discarded(tmp_path, monkeypatch):
    """P2C-N: a proof whose use leaves the effective set is
    explicitly discarded, not left reusable."""
    from flight_log_agent.analysis.mechanism_discovery import (
        derive_current_applicability_proofs,
    )
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    dag_a = result_a.dag
    use_a = next(vertex for vertex in dag_a.vertices
                 if vertex.variable == "out_a"
                 and vertex.kind == "operation")
    reference_a = next(item for item in dag_a.unresolved_references
                       if item.symbol == "kgain"
                       and item.kind == "storage_writers")
    cert = _derive_use_certificate(
        profiler_a, result_a.inputs.structure, reference_a, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(cert) is True
    _p2c_t5_spy(monkeypatch)
    derive_current_applicability_proofs(
        [reference_a], 0, dag_a, store, set())
    assert store.proof_for(
        0, (use_a.id, "kgain"), reference_a.visit_key()) is not None
    narrowed = reference_a.model_copy(update={
        "origin_vertex_ids": ["op-elsewhere"],
        "origin_operands": ["kgain"],
        "origin_uses": [("op-elsewhere", "kgain")],
    })
    assert narrowed.visit_key() == reference_a.visit_key()
    derive_current_applicability_proofs(
        [narrowed], 0, dag_a, store, set())
    assert store.proof_for(
        0, (use_a.id, "kgain"), reference_a.visit_key()) is None


def test_p2c_skips_without_conditional_context(tmp_path, monkeypatch):
    """Conditional gating: unknown conditional context (no
    checkpoint) means no T5 invocation at all — fail-closed."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    store = CoverageProofStore()
    calls = _p2c_t5_spy(monkeypatch)
    result = _p2c_discover_no_evaluator(
        profiler_a, tmp_path, store)
    assert result.dag is not None
    assert len(store.certificates_for(0)) == 1
    assert calls == []
    assert store.proofs_for(0) == ()


def _p2c_discover_no_evaluator(profiler, tmp_path, store):
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    return discover_mechanism_dag(
        profiler, tmp_path / "cache_a", seeds=["usea"],
        terminal="out_a", source_hash="hash",
        terminal_file="src/main.cpp", proof_store=store)


def test_p2c_checkpoint_unaffected(tmp_path):
    """P2C-O: stored applicability proofs never reach the checkpoint
    in P2C — discovery behavior identical with/without the store."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    (profiler_a, _result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    plain = discover_mechanism_dag(
        profiler_a, tmp_path / "cache_a", seeds=["usea"],
        terminal="out_a", source_hash="hash",
        terminal_file="src/main.cpp",
        checkpoint_evaluator=_p2c_continue_evaluator)
    store = CoverageProofStore()
    threaded = discover_mechanism_dag(
        profiler_a, tmp_path / "cache_a", seeds=["usea"],
        terminal="out_a", source_hash="hash",
        terminal_file="src/main.cpp", proof_store=store,
        checkpoint_evaluator=_p2c_continue_evaluator)
    assert threaded.files_loaded == plain.files_loaded
    assert threaded.stop_reason == plain.stop_reason
    assert threaded.dag is not None and plain.dag is not None
    assert [v.id for v in threaded.dag.vertices] == [
        v.id for v in plain.dag.vertices]


# --- P2D: production T6A retirement application ---

def _p2d_resolve_spy(monkeypatch):
    """Count frontier/helper resolutions by symbol across both
    resolver entry points (P0 routes frontier through evidence)."""
    from flight_log_agent.analysis.source_expansion import (
        SourceExpansionResolver,
    )
    calls: list = []
    original = SourceExpansionResolver.resolve
    original_with_evidence = SourceExpansionResolver.resolve_with_evidence

    def spy(self, reference, structure):
        calls.append(reference.symbol)
        return original(self, reference, structure)

    def evidence_spy(self, reference, structure, sink, skip_stages=None,
                     universe_version=None):
        calls.append(reference.symbol)
        return original_with_evidence(
            self, reference, structure, sink, skip_stages=skip_stages,
            universe_version=universe_version)

    monkeypatch.setattr(SourceExpansionResolver, "resolve", spy)
    monkeypatch.setattr(
        SourceExpansionResolver, "resolve_with_evidence", evidence_spy)
    return calls


def test_p2d_current_certificate_retires_obligation(tmp_path):
    """P2D-A/G: honest P0→P2B chain stores a (closed-empty) cert and
    P2D retires the exact obligation with full provenance."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    from flight_log_agent.analysis.coverage import CoverageSearchState
    store = CoverageProofStore()
    state = CoverageSearchState()
    result = _p2b_discover_kzero(
        tmp_path, proof_store=store, search_state=state)
    assert result.dag is not None
    certificates = store.certificates_for(0)
    assert len(certificates) == 1
    certificate = certificates[0]
    assert tuple(certificate.writers) == ()
    reference = next(
        item for item in result.dag.unresolved_references
        if item.symbol == "kzero" and item.kind == "storage_writers")
    key = tuple(certificate.obligation_key)
    assert state.is_proof_retired(key) is True
    record = state.proof_retirement(key)
    assert record is not None
    assert record.version == 0
    assert tuple(record.scheduling_key) == reference.visit_key()
    assert tuple(record.obligation_key) == tuple(
        certificate.obligation_key)
    assert tuple(record.declaration) == tuple(certificate.declaration)
    assert tuple(record.writers) == ()
    assert tuple(record.boundary) == tuple(certificate.boundary)
    assert tuple(record.assumptions) == tuple(certificate.assumptions)


def test_p2d_retirement_suppresses_next_scheduling(tmp_path, monkeypatch):
    """P2D-B: a retired obligation is skipped by the real frontier
    filter on the next session pass, while the DAG still builds."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    from flight_log_agent.analysis.coverage import CoverageSearchState
    store = CoverageProofStore()
    state = CoverageSearchState()
    calls = _p2d_resolve_spy(monkeypatch)
    first = _p2b_discover_kzero(
        tmp_path, proof_store=store, search_state=state)
    assert first.dag is not None
    assert "kzero" in calls
    state.visited.clear()
    del calls[:]
    second = _p2b_discover_kzero(
        tmp_path, proof_store=store, search_state=state)
    assert second.dag is not None
    assert "kzero" not in calls


def test_p2d_no_certificate_no_retirement(tmp_path):
    """P2D-C: heuristic-only evidence derives nothing, retires
    nothing — retirement store stays empty."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        apply_current_coverage_retirements,
    )
    store = CoverageProofStore()
    state = CoverageSearchState()
    assert apply_current_coverage_retirements(store, state, 0) == 0
    assert state.retired == {}


def test_p2d_stale_certificate_no_retirement(tmp_path):
    """P2D-D: a v0 certificate never retires under current v1."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        apply_current_coverage_retirements,
    )
    key = ("storage_writers", "global", "decl-a")
    semantic = ("storage_writers", "global", "decl-a")
    certificate, _proof = _proof_pair(("op-a", "s"), key, semantic,
                                      version=0)
    store = CoverageProofStore()
    assert store.store_certificate(certificate) is True
    state = CoverageSearchState()
    state.version = 1
    assert apply_current_coverage_retirements(store, state, 1) == 0
    assert state.is_proof_retired(semantic) is False
    assert state.retired == {}


def test_p2d_unrelated_and_spelling_isolation(tmp_path):
    """P2D-E/F: cert A retires only A — unrelated obligations and
    same-spelling distinct declarations are unaffected."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        apply_current_coverage_retirements,
    )
    key_a = ("storage_writers", "global", "decl-a")
    certificate_a, _ = _proof_pair(("op-a", "s"), key_a, key_a,
                                   version=0)
    store = CoverageProofStore()
    assert store.store_certificate(certificate_a) is True
    state = CoverageSearchState()
    assert apply_current_coverage_retirements(store, state, 0) == 1
    assert state.is_proof_retired(key_a) is True
    assert state.is_proof_retired(
        ("storage_writers", "global", "decl-b")) is False
    assert state.is_proof_retired(
        ("storage_writers", "member", "decl-a")) is False


def test_p2d_nonempty_writer_provenance_retained(tmp_path):
    """P2D-H: a non-empty writer set retires with provenance exactly
    equal to the certificate's writers."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        apply_current_coverage_retirements,
    )
    key = ("storage_writers", "global", "decl-a")
    writers = ("src/a.cpp:1:w1", "src/a.cpp:2:w2")
    certificate, _ = _proof_pair(("op-a", "s"), key, key,
                                 writers=writers, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(certificate) is True
    state = CoverageSearchState()
    assert apply_current_coverage_retirements(store, state, 0) == 1
    record = state.proof_retirement(key)
    assert record is not None
    assert tuple(record.writers) == writers


def test_p2d_retirement_idempotent(tmp_path):
    """P2D-I: repeating the same current certificate leaves one
    identical retirement record, not duplicates."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        apply_current_coverage_retirements,
    )
    key = ("storage_writers", "global", "decl-a")
    certificate, _ = _proof_pair(("op-a", "s"), key, key, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(certificate) is True
    state = CoverageSearchState()
    assert apply_current_coverage_retirements(store, state, 0) == 1
    first = state.proof_retirement(key)
    assert apply_current_coverage_retirements(store, state, 0) == 1
    assert state.proof_retirement(key) == first
    assert len(state.retired) == 1


def test_p2d_retirement_needs_no_applicability(tmp_path):
    """P2D-J: retirement applies with zero applicability proofs —
    coverage retirement is independent of applicability authority."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    store = CoverageProofStore()
    state = CoverageSearchState()
    result = _p2b_discover_kzero(
        tmp_path, proof_store=store, search_state=state)
    assert result.dag is not None
    assert store.proofs_for(0) == ()
    key = next(
        tuple(certificate.obligation_key)
        for certificate in store.certificates_for(0))
    assert state.is_proof_retired(key) is True


def test_p2d_version_extension_reactivates(tmp_path):
    """P2D-K: v0 retirement is inert under v1 through real
    search-state version semantics."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        apply_current_coverage_retirements,
    )
    key = ("storage_writers", "global", "decl-a")
    certificate, _ = _proof_pair(("op-a", "s"), key, key, version=0)
    store = CoverageProofStore()
    assert store.store_certificate(certificate) is True
    state = CoverageSearchState()
    assert apply_current_coverage_retirements(store, state, 0) == 1
    assert state.is_proof_retired(key) is True
    assert state.advance_version() == 1
    assert state.is_proof_retired(key) is False
    assert apply_current_coverage_retirements(store, state, 1) == 0
    assert state.is_proof_retired(key) is False


def test_p2d_checkpoint_unaffected(tmp_path):
    """P2D-L: retirement changes scheduling only — discovery
    behavior identical with/without the proof store."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    profiler = _p0_tree(tmp_path)
    plain = _p0_discover(profiler, tmp_path)
    store = CoverageProofStore()
    from flight_log_agent.analysis.coverage import CoverageSearchState
    threaded = _p0_discover(
        profiler, tmp_path, proof_store=store,
        search_state=CoverageSearchState())
    assert threaded.files_loaded == plain.files_loaded
    assert threaded.stop_reason == plain.stop_reason
    assert threaded.dag is not None and plain.dag is not None
    assert [v.id for v in threaded.dag.vertices] == [
        v.id for v in plain.dag.vertices]


# --- P3: immutable proof snapshot + checkpoint threading ---

def _p3_threading_evaluator(seen_rounds, seen_snapshots):
    """Real checkpoint round accepting an optional proof snapshot
    exactly like the production control round: snapshot values flow
    into the existing T6B inputs, omission preserves legacy call."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_checkpoint_round,
    )

    def evaluator(dag, index, proof_snapshot=None):
        seen_snapshots.append(proof_snapshot)
        kwargs = {}
        if proof_snapshot is not None:
            kwargs = {
                "coverage_certificates": tuple(
                    proof_snapshot.certificates or ()),
                "applicability_proofs": tuple(
                    proof_snapshot.applicability_proofs or ()),
                "proof_version": proof_snapshot.version,
            }
        result = evaluate_checkpoint_round(
            dag, parameter_values={}, observed_signals=set(),
            signal_policies={},
            load_samples=lambda _view, _observed: {},
            **kwargs)
        seen_rounds.append(result)
        return result

    return evaluator


def test_p3_stored_proof_reaches_checkpoint(tmp_path):
    """P3-A/C/G/J/O: P0 evidence → P2B cert → threaded snapshot →
    real checkpoint observes coverage (flags per T6B rules, stop
    still false on dirty legacy) — with retirement active, which
    removes scheduling but not relevance or proof."""
    from flight_log_agent.analysis.coverage import (
        CoverageProofStore,
        CoverageSearchState,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _p2b_kzero_tree(tmp_path)
    kwargs = dict(
        seeds=["use"], terminal="out", source_hash="hash",
        terminal_file="src/main.cpp")
    store = CoverageProofStore()
    state = CoverageSearchState()
    first = discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state, **kwargs)
    assert first.dag is not None
    assert len(store.certificates_for(0)) == 1
    certificate = store.certificates_for(0)[0]
    seen_rounds: list = []
    seen_snapshots: list = []
    second = discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state,
        checkpoint_evaluator=_p3_threading_evaluator(
            seen_rounds, seen_snapshots),
        **kwargs)
    assert second.dag is not None
    assert seen_rounds, "checkpoint never evaluated"
    assert seen_snapshots and seen_snapshots[0] is not None
    snapshot = seen_snapshots[0]
    assert snapshot.version == 0
    assert snapshot.version == state.version
    assert snapshot.certificates == store.certificates_for(0)
    key = certificate.scheduling_key
    first_round = seen_rounds[0]
    observation = first_round.proof_observation
    assert observation is not None
    assert observation.proof_version == 0
    assert tuple(observation.relevant_obligation_keys) == (tuple(key),)
    assert tuple(observation.covered_obligation_keys) == (tuple(key),)
    authority = first_round.proof_authority
    assert authority is not None
    assert authority.gate_active is True
    assert authority.writer_coverage_verified is True
    assert authority.applicability_verified is False
    assert authority.authorizes_stop is False
    assert first_round.action != "verified"


def test_p3_stable_version_proof_persists_across_rounds(tmp_path):
    """P3-C: the same current proof is visible to consecutive
    checkpoints while the version is stable."""
    from flight_log_agent.analysis.coverage import (
        CoverageProofStore,
        CoverageSearchState,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _p2b_kzero_tree(tmp_path)
    kwargs = dict(
        seeds=["use"], terminal="out", source_hash="hash",
        terminal_file="src/main.cpp")
    store = CoverageProofStore()
    state = CoverageSearchState()
    discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state, **kwargs)
    seen_rounds: list = []
    seen_snapshots: list = []
    evaluator = _p3_threading_evaluator(seen_rounds, seen_snapshots)
    discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state, checkpoint_evaluator=evaluator, **kwargs)
    discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state, checkpoint_evaluator=evaluator, **kwargs)
    assert len(seen_rounds) >= 2
    covered = [
        tuple(round_.proof_observation.covered_obligation_keys)
        for round_ in seen_rounds]
    assert covered
    assert all(entry == covered[0] for entry in covered)
    assert all(entry != () for entry in covered)
    assert all(round_.proof_observation.proof_version == 0
              for round_ in seen_rounds)


def test_p3_version_extension_invalidates_snapshot(tmp_path):
    """P3-D/L: v0 proof accumulated before extension never enters a
    v1 snapshot; the v1 checkpoint sees no v0 content."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _p2b_kzero_tree(tmp_path)
    kwargs = dict(
        seeds=["use"], terminal="out", source_hash="hash",
        terminal_file="src/main.cpp")
    store = CoverageProofStore()
    state = CoverageSearchState()
    discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state, **kwargs)
    assert len(store.certificates_for(0)) == 1
    seen_rounds: list = []
    seen_snapshots: list = []
    profiler_p0 = _p0_tree(tmp_path)
    result = discover_mechanism_dag(
        profiler_p0, tmp_path / "cache", seeds=["pick_altitude"],
        terminal="_final_out", source_hash="hash",
        terminal_file="src/modules/example/rtl.cpp",
        proof_store=store, search_state=state,
        checkpoint_evaluator=_p3_threading_evaluator(
            seen_rounds, seen_snapshots))
    assert result.dag is not None
    assert state.version == 1
    assert store.certificates_for(0) == ()
    v1_snapshots = [snapshot for snapshot in seen_snapshots
                    if snapshot is not None and snapshot.version == 1]
    assert v1_snapshots, "no v1 snapshot reached checkpoint"
    for snapshot in v1_snapshots:
        assert snapshot.certificates == ()
        assert snapshot.applicability_proofs == ()
    v1_rounds = [
        round_ for round_, snapshot in zip(seen_rounds, seen_snapshots)
        if snapshot is not None and snapshot.version == 1]
    assert v1_rounds
    for round_ in v1_rounds:
        assert round_.proof_authority is not None
        assert not round_.proof_observation.covered_obligation_keys


def test_p3_empty_snapshot_stays_fail_closed(tmp_path):
    """P3-E: an explicit empty current snapshot keeps non-empty
    relevance fail-closed while marking the gate engaged."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _p0_tree(tmp_path)
    store = CoverageProofStore()
    state = CoverageSearchState()
    seen_rounds: list = []
    seen_snapshots: list = []
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["pick_altitude"],
        terminal="_final_out", source_hash="hash",
        terminal_file="src/modules/example/rtl.cpp",
        proof_store=store, search_state=state,
        checkpoint_evaluator=_p3_threading_evaluator(
            seen_rounds, seen_snapshots))
    assert result.dag is not None
    assert seen_rounds
    assert seen_snapshots and seen_snapshots[0] is not None
    assert seen_snapshots[0].certificates == ()
    assert seen_snapshots[0].applicability_proofs == ()
    assert seen_snapshots[0].version == 0
    for round_ in seen_rounds:
        assert round_.proof_authority is not None
        assert round_.proof_authority.gate_active is True
        if round_.proof_observation.relevant_obligation_keys:
            assert round_.proof_authority.authorizes_stop is False


def test_p3_snapshot_immutable_and_store_untouched(tmp_path):
    """P3-F/I: snapshots detach from later store mutation and
    checkpoint evaluation never mutates the store."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _p2b_kzero_tree(tmp_path)
    kwargs = dict(
        seeds=["use"], terminal="out", source_hash="hash",
        terminal_file="src/main.cpp")
    store = CoverageProofStore()
    state = CoverageSearchState()
    discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state, **kwargs)
    before = (store.certificates_for(0), store.proofs_for(0))
    seen_rounds: list = []
    seen_snapshots: list = []
    discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state,
        checkpoint_evaluator=_p3_threading_evaluator(
            seen_rounds, seen_snapshots),
        **kwargs)
    assert seen_snapshots and seen_snapshots[0] is not None
    assert (store.certificates_for(0), store.proofs_for(0)) == before
    extra, _proof = _proof_pair(
        ("op-x", "s"), ("storage_writers", "global", "decl-x"),
        ("storage_writers", "global", "decl-x"), version=0)
    assert store.store_certificate(extra) is True
    assert seen_snapshots[0].certificates == before[0]
    assert extra not in seen_snapshots[0].certificates


def test_p3_proof_without_certificate_stays_fail_closed(tmp_path):
    """P3-K: an applicability proof with no current certificate
    cannot satisfy coverage — T6B stays fail-closed."""
    from flight_log_agent.analysis.coverage import CoverageProofStore
    from flight_log_agent.analysis.coverage import CoverageSearchState
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _p2b_kzero_tree(tmp_path)
    kwargs = dict(
        seeds=["use"], terminal="out", source_hash="hash",
        terminal_file="src/main.cpp")
    store = CoverageProofStore()
    state = CoverageSearchState()
    key = ("storage_writers", "global", "decl-seed")
    _certificate, proof = _proof_pair(
        ("op-a", "signal-a"), key,
        ("storage_writers", "global", "decl-seed"), version=0)
    assert store.store_proof(proof) is True
    seen_rounds: list = []
    seen_snapshots: list = []
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", proof_store=store,
        search_state=state,
        checkpoint_evaluator=_p3_threading_evaluator(
            seen_rounds, seen_snapshots),
        **kwargs)
    assert result.dag is not None
    assert seen_rounds
    seeded_visit = proof.scheduling_key
    for round_ in seen_rounds:
        assert round_.proof_authority is not None
        assert round_.proof_authority.authorizes_stop is False
        # The seeded foreign proof's visit never enters relevance, so
        # it can satisfy nothing; any coverage present comes only
        # from genuinely derived in-session certificates.
        assert tuple(seeded_visit) not in {
            tuple(item)
            for item in round_.proof_observation.relevant_obligation_keys
        }


def test_p3_construction_evaluator_receives_snapshot(tmp_path):
    """P3 construction site: a proof-aware construction evaluator
    observes the current-version snapshot through real discovery."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        CheckpointRound,
    )
    from flight_log_agent.analysis.coverage import (
        CoverageProofStore,
        CoverageSearchState,
    )
    from flight_log_agent.analysis.mechanism_dag import ConstructionDemand
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    profiler = _mini_tree(tmp_path, {
        "policy.hpp": "class Policy { public: bool allow(); float value(); void run(); float output; };",
        "run.cpp": '#include "policy.hpp"\nvoid Policy::run() { if (allow()) { output = value(); } }',
        "guard.cpp": '#include "policy.hpp"\nbool Policy::allow() { return true; }',
        "value.cpp": '#include "policy.hpp"\nfloat Policy::value() { return 7.f; }',
    }, backend="tree_sitter")
    seen = []

    def construction_evaluator(dag, index, proof_snapshot=None):
        seen.append((proof_snapshot, state.version))
        return CheckpointRound(
            "continue", dag, [], {"action": "continue"},
            construction=ConstructionDemand())

    store = CoverageProofStore()
    state = CoverageSearchState()
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=[], terminal="output",
        terminal_file="run.cpp", source_hash="hash",
        proof_store=store, search_state=state,
        construction_evaluator=construction_evaluator)
    assert result.dag is not None
    assert seen, "construction evaluation never ran"
    assert all(snapshot is not None for snapshot, _ in seen)
    assert all(snapshot.version == version_at_call
              for snapshot, version_at_call in seen)


@pytest.mark.xfail(
    strict=True,
    reason="Gate B: covered relevant obligations still dirty legacy "
           "source_requests (relevance and source-requests derive from "
           "the same origin sets), so no non-vacuous positive stop is "
           "reachable yet; see P4 STOP report",
)
def test_p4_honest_production_positive_stop(tmp_path):
    """P4 E2E (known gap): natural obligation → P0 evidence → P2B
    cert → exact use → P2C proof → threaded snapshot → real round
    must eventually authorize a non-vacuous stop. Today full proof
    (coverage AND applicability, non-empty relevance) still yields
    stop False because the proven obligation itself keeps legacy
    source_requests dirty. XPASS means the gap closed: update this
    test and the P4 report instead of weakening anything."""
    from flight_log_agent.analysis.coverage import (
        CoverageProofStore,
        CoverageSearchState,
    )
    from flight_log_agent.analysis.mechanism_discovery import (
        discover_mechanism_dag,
    )
    (profiler_a, _result_a), (_profiler_b, _result_b) = _split_use_tree(
        tmp_path)
    store = CoverageProofStore()
    state = CoverageSearchState()
    kwargs = dict(
        seeds=["usea"], terminal="out_a", source_hash="hash",
        terminal_file="src/main.cpp")
    seen_rounds: list = []
    seen_snapshots: list = []
    discover_mechanism_dag(
        profiler_a, tmp_path / "cache_a", proof_store=store,
        search_state=state,
        checkpoint_evaluator=_p3_threading_evaluator(
            seen_rounds, seen_snapshots),
        **kwargs)
    discover_mechanism_dag(
        profiler_a, tmp_path / "cache_a", proof_store=store,
        search_state=state,
        checkpoint_evaluator=_p3_threading_evaluator(
            seen_rounds, seen_snapshots),
        **kwargs)
    assert seen_rounds, "no checkpoint evaluated"
    final = seen_rounds[-1]
    authority = final.proof_authority
    observation = final.proof_observation
    assert len(observation.relevant_obligation_keys) >= 1
    assert authority.writer_coverage_verified is True
    assert authority.applicability_verified is True
    assert final.action == "verified"
    selected = final.summary.get("selected_checkpoint") or {}
    assert selected.get("authorizes_discovery_stop") is True


# --- Gate-B outstanding-source-work discharge ---

def _gateb_covered_round(proof_version=0, extra_references=(),
                         certificates=(), proofs=(), certificate_version=None):
    """Clean _t6b terminal plus appended writer obligations with
    caller-supplied proof threaded, returning the round and keys."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    base = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(base)
    for reference in extra_references:
        dag.unresolved_references.append(reference)
    key = base.visit_key()
    semantic = ("storage_writers", "member", "decl:x")
    use = (root.id, "sample")
    own_certificate, own_proof = _t6b_hand_proof(
        use, key, semantic,
        version=proof_version if certificate_version is None
        else certificate_version)
    result = _run_t6b_round(
        dag, samples, policies,
        coverage_certificates=[own_certificate, *certificates],
        proof_version=proof_version,
        applicability_proofs=[own_proof, *proofs])
    return result, base, key, semantic, use


def _gateb_bare_round(proof_version=0, with_certificate=True,
                      with_proof=True):
    """Covered-round fixture with proof threading toggles exposed."""
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    key = reference.visit_key()
    semantic = ("storage_writers", "member", "decl:x")
    use = (root.id, "sample")
    certificate, proof = _t6b_hand_proof(use, key, semantic)
    result = _run_t6b_round(
        dag, samples, policies,
        coverage_certificates=[certificate] if with_certificate else [],
        proof_version=proof_version,
        applicability_proofs=[proof] if with_proof else [])
    return result, reference, key, semantic, use


def test_discharged_source_request_keys_default_empty():
    """G1: fresh proof authority carries no discharged visits."""
    authority = _authority()
    assert authority.discharged_source_request_keys == ()


def test_covered_source_request_discharged_and_verified():
    """Gate-B A: covered visit leaves raw diagnostics intact but no
    longer vetoes — full proof authorizes stop."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    result, _reference, key, _semantic, _use = _gateb_covered_round()
    assert result.action == "verified"
    assert result.proof_authority.discharged_source_request_keys == (
        tuple(key),)
    selected = result.summary.get("selected_checkpoint") or {}
    assert any(
        UnresolvedSourceReference.model_validate(raw).visit_key()
        == tuple(key)
        for raw in selected.get("source_requests", []))
    assert result.proof_authority.legacy_verified is False
    assert selected.get("authorizes_discovery_stop") is True


def test_paired_source_lookup_discharged_together():
    """Gate-B B: the source_lookup requirement paired to a covered
    visit is satisfied together with its source request."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    result, _reference, key, _semantic, _use = _gateb_covered_round()
    assert result.action == "verified"
    selected = result.summary.get("selected_checkpoint") or {}
    assert any(
        requirement.get("kind") == "source_lookup"
        for requirement in selected.get("analysis_requirements", []))
    assert result.proof_authority.discharged_source_request_keys == (
        tuple(key),)


def test_uncovered_request_stays_outstanding():
    """Gate-B C: a second uncovered obligation is not discharged and
    still vetoes, while the covered visit stays discharged."""
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    other = UnresolvedSourceReference(
        symbol="other", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "other", "root": "other",
                  "class_owner": "Controller", "declaration_id": "decl:y",
                  "declaration_proven": True})
    result, _reference, key, _semantic, _use = _gateb_covered_round(
        extra_references=[other])
    assert result.proof_authority.discharged_source_request_keys == (
        tuple(key),)
    assert other.visit_key() not in (
        result.proof_authority.discharged_source_request_keys)
    assert result.action != "verified"
    assert result.proof_authority.coverage_ok is False


def test_undischarged_without_coverage():
    """Gate-B D: no accepted coverage means no discharge."""
    result, _reference, _key, _semantic, _use = _gateb_bare_round(
        with_certificate=False, with_proof=False)
    assert result.proof_authority.discharged_source_request_keys == ()
    assert result.action != "verified"


def test_covered_missing_applicability_still_vetoes():
    """Gate-B E: source-work discharge never bypasses T5 — covered
    but unproven use keeps stop false while discharge is recorded."""
    result, _reference, key, _semantic, _use = _gateb_bare_round(
        with_certificate=True, with_proof=False)
    assert result.proof_authority.writer_coverage_verified is True
    assert result.proof_authority.applicability_verified is False
    assert result.proof_authority.authorizes_stop is False
    assert result.proof_authority.discharged_source_request_keys == (
        tuple(key),)
    assert result.action != "verified"


def test_stale_coverage_discharges_nothing():
    """Gate-B F: stale-version certificates discharge nothing."""
    result, _reference, _key, _semantic, _use = _gateb_covered_round(
        certificate_version=99)
    assert result.proof_authority.discharged_source_request_keys == ()
    assert result.proof_authority.coverage_ok is False
    assert result.action != "verified"


def test_no_proof_version_no_discharge():
    """Gate-B G: disengaged proof gate discharges nothing."""
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    result = _run_t6b_round(dag, samples, policies)
    assert result.proof_authority.discharged_source_request_keys == ()
    assert result.action != "verified"


def test_exhausted_without_coverage_no_discharge():
    """Gate-B H: exhausted scheduling state without accepted coverage
    discharges nothing and stays fail-closed."""
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    dag.exhausted_source_requests.add(reference.visit_key())
    result = _run_t6b_round(dag, samples, policies, proof_version=0)
    assert result.proof_authority.discharged_source_request_keys == ()
    assert result.action != "verified"


def test_exhausted_with_coverage_discharges_normally():
    """Exhaustion is scheduling-only: with accepted current coverage
    the exhausted obligation discharges exactly like its unexhausted
    twin — exhaustion neither grants nor blocks discharge."""
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )

    def make_reference():
        reference = UnresolvedSourceReference(
            symbol="stored", kind="storage_writers", file="sample.cpp",
            callable_id="Controller::step",
            origin_vertex_ids=[root.id], origin_operands=["sample"],
            identity={"kind": "member", "symbol": "stored",
                      "root": "stored", "class_owner": "Controller",
                      "declaration_id": "decl:x",
                      "declaration_proven": True})
        dag.unresolved_references.append(reference)
        return reference

    reference = make_reference()
    key = reference.visit_key()
    semantic = ("storage_writers", "member", "decl:x")
    use = (root.id, "sample")
    certificate, _proof = _t6b_hand_proof(use, key, semantic)
    plain = _run_t6b_round(
        dag, samples, policies, coverage_certificates=[certificate],
        proof_version=0)
    assert plain.proof_authority.discharged_source_request_keys == (
        tuple(key),)
    dag.exhausted_source_requests.add(key)
    exhausted = _run_t6b_round(
        dag, samples, policies, coverage_certificates=[certificate],
        proof_version=0)
    assert exhausted.proof_authority.discharged_source_request_keys == (
        tuple(key),)
    assert (exhausted.proof_authority.coverage_ok
            == plain.proof_authority.coverage_ok)
    selected = exhausted.summary.get("selected_checkpoint") or {}
    assert any(
        UnresolvedSourceReference.model_validate(raw).visit_key()
        == tuple(key)
        for raw in selected.get("source_requests", []))


def test_partition_outstanding_source_work_unit():
    """Gate-B I/J/K unit pins: malformed entries stay outstanding,
    visits partition independently, spelling never cross-discharges."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        partition_outstanding_source_work,
    )
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )

    def raw(**fields):
        base = dict(
            symbol="stored", kind="storage_writers", file="sample.cpp",
            callable_id="Controller::step", origin_vertex_ids=["op"],
            origin_operands=["sample"],
            identity={"kind": "member", "symbol": "stored",
                      "root": "stored", "class_owner": "Controller",
                      "declaration_id": "decl:x",
                      "declaration_proven": True})
        base.update(fields)
        return UnresolvedSourceReference(**base).model_dump(mode="json")

    def requirement(raw_request):
        return {"kind": "source_lookup", "reason": "r",
                "source_reference": raw_request}

    covered_a = UnresolvedSourceReference.model_validate(
        raw()).visit_key()
    outstanding, requirements, discharged = (
        partition_outstanding_source_work(
            [raw(), {"symbol": [], "kind": 42},
             {"kind": "source_lookup"}],
            [requirement(raw()),
             {"kind": "construction", "reason": "c"},
             {"kind": "source_lookup", "reason": "unbound"}],
            {tuple(covered_a)}))
    assert len(outstanding) == 2
    assert discharged == (tuple(covered_a),)
    assert [item.get("kind") for item in requirements] == [
        "construction", "source_lookup"]
    other = raw(identity={"kind": "member", "symbol": "stored",
                          "root": "stored", "class_owner": "Controller",
                          "declaration_id": "decl:other",
                          "declaration_proven": True})
    outstanding, _requirements, discharged = (
        partition_outstanding_source_work(
            [raw(), other], [], {tuple(covered_a)}))
    assert len(outstanding) == 1
    assert discharged == (tuple(covered_a),)


def test_raw_diagnostics_preserved_after_discharge():
    """Gate-B M: raw source_requests, raw requirements, and raw
    legacy_verified keep their structural values after discount."""
    result, _reference, key, _semantic, _use = _gateb_covered_round()
    selected = result.summary.get("selected_checkpoint") or {}
    assert len(selected.get("source_requests", [])) >= 1
    assert any(
        requirement.get("kind") == "source_lookup"
        for requirement in selected.get("analysis_requirements", []))
    assert result.proof_authority.legacy_verified is False
    assert result.proof_authority.discharged_source_request_keys == (
        tuple(key),)


def test_replay_attempts_despite_discharged_requirements():
    """Replay gate (R1): a fully replay-capable graph attempts replay
    even while raw source_lookup requirements exist. Gate eligibility
    is asserted here, not numeric success — matched/partial outcome
    remains replay's own verdict."""
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    result = _run_t6b_round(dag, samples, policies, proof_version=0)
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("status") != "not_attempted"


def _t6b_covered_round_no_proof():
    """_t6b terminal plus one covered-but-unproven writer obligation."""
    dag, samples, policies = _t6b_round_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    key = reference.visit_key()
    semantic = ("storage_writers", "member", "decl:x")
    use = (root.id, "sample")
    certificate, _proof = _t6b_hand_proof(use, key, semantic)
    return dag, samples, policies, reference, key, semantic, use, certificate


def test_uncovered_matched_replay_still_vetoes():
    """R1-H: replay may now run (and match) beside an uncovered
    obligation — final authority still vetoes via coverage."""
    dag, samples, policies, reference, key, _semantic, _use, _cert = (
        _t6b_covered_round_no_proof())
    result = _run_t6b_round(dag, samples, policies, proof_version=0)
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("status") == "matched"
    assert result.proof_authority.coverage_ok is False
    assert result.proof_authority.authorizes_stop is False
    assert result.action != "verified"


def test_missing_applicability_vetoes_despite_replay():
    """R1-I: replay matched + coverage valid, but applicability
    missing → final authority false."""
    dag, samples, policies, reference, key, semantic, use, certificate = (
        _t6b_covered_round_no_proof())
    result = _run_t6b_round(
        dag, samples, policies, coverage_certificates=[certificate],
        proof_version=0)
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("status") == "matched"
    assert result.proof_authority.writer_coverage_verified is True
    assert result.proof_authority.applicability_verified is False
    assert result.proof_authority.authorizes_stop is False
    assert result.action != "verified"


def test_mismatch_vetoes_despite_full_proof():
    """R1-J: replay mismatch vetoes even with valid coverage and
    applicability proofs threaded."""
    dag, samples, policies, reference, key, semantic, use = (
        _t6b_covered_round_no_proof()[:7])
    certificate, proof = _t6b_hand_proof(use, key, semantic)
    bad_samples = dict(samples)
    bad_samples["command.value"] = [(0.0, 9.0), (10.0, 9.0)]
    result = _run_t6b_round(
        dag, bad_samples, policies, coverage_certificates=[certificate],
        proof_version=0, applicability_proofs=[proof])
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("status") == "mismatched"
    assert result.proof_authority.authorizes_stop is False
    assert result.action != "verified"


def _t6b_unevaluable_dag():
    """_t6b shape whose terminal equation references an unknown
    input: replay attempts but cannot evaluate."""
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    def expression(text, *inputs):
        return {"text": text, "lowered_text": text,
                "input_symbols": list(inputs),
                "input_identities": {}, "call_results": [], "exact": True}

    bindings = [
        {"target_symbol": "sample", "source_symbol": "measurement.value",
         "external_source_signal": True, "synthetic_boundary_transfer": True,
         "boundary_direction": "subscribe",
         "expression_ref": expression("measurement.value", "measurement.value"),
         "assignment_path": [{"file": "sample.cpp", "line": 2}],
         "function": "Controller::step"},
        {"target_symbol": "command.value", "source_symbol": "sample * mystery",
         "external_target_signal": True, "synthetic_boundary_transfer": True,
         "boundary_direction": "publish",
         "expression_ref": expression("sample * mystery", "sample", "mystery"),
         "assignment_path": [{"file": "sample.cpp", "line": 3}],
         "function": "Controller::step"},
    ]
    samples = {"measurement.value": [(0.0, 3.0), (10.0, 3.0)],
               "command.value": [(0.0, 6.0), (10.0, 6.0)]}
    policies = {signal: {"method": "linear"} for signal in samples}
    dag = build_mechanism_dag(bindings, "command.value",
                              logged_signals=set(samples))
    return dag, samples, policies


def test_missing_input_replay_incomplete_not_matched():
    """R1-K: source_lookup-only requirements permit the attempt, but
    missing numeric input still reports incomplete — never matched."""
    dag, samples, policies = _t6b_unevaluable_dag()
    root = next(v for v in dag.vertices if v.metadata.get("is_terminal"))
    from flight_log_agent.analysis.source_expansion import (
        UnresolvedSourceReference,
    )
    reference = UnresolvedSourceReference(
        symbol="stored", kind="storage_writers", file="sample.cpp",
        callable_id="Controller::step", origin_vertex_ids=[root.id],
        origin_operands=["sample"],
        identity={"kind": "member", "symbol": "stored", "root": "stored",
                  "class_owner": "Controller", "declaration_id": "decl:x",
                  "declaration_proven": True})
    dag.unresolved_references.append(reference)
    result = _run_t6b_round(dag, samples, policies, proof_version=0)
    selected = result.summary.get("selected_checkpoint") or {}
    assert selected.get("status") != "not_attempted"
    assert selected.get("status") != "matched"
    assert selected.get("complete") is False
    assert "numerical_replay" in {
        item.get("kind") for item in selected.get("analysis_requirements", [])}
    assert result.action != "verified"
