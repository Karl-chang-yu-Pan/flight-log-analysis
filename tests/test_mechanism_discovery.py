from __future__ import annotations

import pytest

from flight_log_agent.analysis.dag_value import DAGValueProgram
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


def test_boundary_call_in_predicate_is_not_a_helper_gap(tmp_path, source_backend):
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
        logged_signals={"status.value"},
        helper_expressions=inputs.helper_expressions,
        call_statements=inputs.call_statements,
        boundary_bindings=inputs.boundary_bindings,
        source_structure=inputs.structure,
    )

    assert any(
        vertex.kind == "operation"
        and (vertex.metadata or {}).get("synthetic_boundary_transfer")
        for vertex in dag.vertices
    )
    assert not any(
        reference.kind == "callable"
        and reference.symbol.rsplit(".", 1)[-1] == "update"
        for reference in dag.unresolved_references
    )


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
    assert evaluated_branch.active_windows == [(10.0, 20.0)]


def test_control_flow_governed_boundary_transfer_stays_grounded(
    tmp_path, source_backend
):
    """A subscription copy inside control flow still grounds a gating branch.

    The logged topic observes the member's value over the whole timeline, so
    the source-code guard around the copy does not gate the observation.
    """
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
    assert evaluated.active_windows == [(10.0, 20.0)]


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


def test_nested_helper_return_dataflow_is_backend_interchangeable(
    tmp_path, source_backend
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


def test_source_grounded_helper_condition_is_backend_interchangeable(
    tmp_path, source_backend
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
    )

    annotated = evaluate_feasibility(
        dag,
        parameter_values={"ACCEPT_RADIUS": 10.0},
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


def test_dereferenced_reference_alias_reaches_source_proven_log_boundary(tmp_path):
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
""",
        },
        backend="tree_sitter",
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


def test_helper_return_consumes_aliased_call_result_without_flattening(tmp_path):
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
