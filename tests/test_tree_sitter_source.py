from __future__ import annotations

import pytest

from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag
from flight_log_agent.analysis.mechanism_discovery import dag_inputs_from_facts
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.px4.source_facts_cache import extract_facts_for_file
from flight_log_agent.px4.tree_sitter_source import TreeSitterSourceExtractor


def _facts(tmp_path, source: str, *, backend: str = "tree_sitter"):
    root = tmp_path / "PX4-Autopilot"
    source_file = root / "src" / "modules" / "example" / "example.cpp"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(source, encoding="utf-8")
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend=backend,
    )
    return extract_facts_for_file(
        profiler,
        "src/modules/example/example.cpp",
        "source-hash",
    )


def test_admission_parses_one_file_without_retaining_rejected_ast(tmp_path):
    root = tmp_path / "PX4-Autopilot"
    source_file = root / "src" / "modules" / "example" / "candidate.cpp"
    header_file = source_file.with_suffix(".hpp")
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        "void Candidate::run() { consume(output); }",
        encoding="utf-8",
    )
    header_file.write_text(
        "class Candidate { void write() { output = input; } };",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )
    extractor = TreeSitterSourceExtractor(profiler)

    facts = extractor.extract_admission(
        "src/modules/example/candidate.cpp",
        "source-hash",
        kind="symbol",
        symbol="output",
    )

    assert facts.source_assignments == []
    assert [item.name for item in facts.callables] == ["Candidate::run"]
    assert extractor._units == {}
    assert profiler._text_cache == {}


@pytest.mark.parametrize("initializer", ["= *_navigator->get_position()", "{*_navigator->get_position()}"])
def test_reference_alias_preserves_source_variable_and_resolved_callee(tmp_path, initializer):
    root = tmp_path / "PX4-Autopilot"
    module = root / "src" / "modules" / "example"
    module.mkdir(parents=True)
    (module / "navigator.h").write_text(
        """
struct Position { float alt; };
class Navigator {
public:
    Position *get_position() { return &_position; }
private:
    Position _position{};
};
""",
        encoding="utf-8",
    )
    (module / "mode.cpp").write_text(
        """
#include "navigator.h"
class Mode {
    Navigator *_navigator;
    float output;
    void run();
};
void Mode::run()
{
    const Position &first INITIALIZER;
    const Position &position = first;
    output = position.alt;
}
""".replace("INITIALIZER", initializer),
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )

    facts = extract_facts_for_file(
        profiler,
        "src/modules/example/mode.cpp",
        "source-hash",
    )

    getter_call = next(item for item in facts.function_calls if item.name == "get_position")
    output = next(item for item in facts.source_assignments if item.target == "output")
    assert getter_call.receiver_type == "Navigator"
    assert getter_call.receiver_access == "pointer"
    assert getter_call.resolved_callable_id
    assert getter_call.resolved_callable_file == "src/modules/example/navigator.h"
    assert output.expression_ref is not None
    assert output.expression_ref.lowered_text == "position.alt"
    assert output.expression_ref.input_symbols == ["position.alt"]
    assert output.expression_ref.call_results == []


def test_reference_list_binding_does_not_make_value_construction_an_alias(tmp_path):
    facts = _facts(tmp_path, """
struct Status {};
class Source { public: const Status &get(); };
void run(Source &object) {
    const Status &reference{object.get()};
    Status value{object.get()};
}
""")
    assignments = {a.target: a for a in facts.source_assignments}
    assert assignments["reference"].expression_ref.direct_call_result is not None
    assert assignments["value"].expression_ref.direct_call_result is None


def test_class_logical_operator_does_not_invent_short_circuit_controls(tmp_path):
    facts = _facts(tmp_path, """
struct Flag { bool operator&&(bool value); };
bool effect();
void run() {
    Flag flag;
    (flag && effect());
}
""")
    call = next(c for c in facts.function_calls if c.name == "effect")
    assert not call.control_predicates
    assert call.reachability_exact is False


def test_branch_condition_ref_preserves_projected_call_result(tmp_path):
    facts = _facts(
        tmp_path,
        """
struct Position { float alt; };
class Control {
    Position *get_position();
    float output;
    void run();
};
void Control::run()
{
    if (get_position()->alt > 0.0f) {
        output = 1.0f;
    }
}
""",
    )

    branch = next(item for item in facts.branch_conditions if "get_position" in item.condition)
    assert branch.condition_ref is not None
    assert branch.condition_ref.input_symbols == []
    assert len(branch.condition_ref.call_results) == 1
    assert branch.condition_ref.call_results[0].result_path == "alt"


def test_untyped_receiver_does_not_resolve_unrelated_bare_method(tmp_path):
    facts = _facts(
        tmp_path,
        """
class Unrelated {
public:
    float getState() const { return state; }
    float state;
};

class Caller {
    ExternalType service;
    float run() { return service.getState(); }
};
""",
    )

    call = next(item for item in facts.function_calls if item.name == "getState")
    assert call.receiver == "service"
    assert call.receiver_type is None
    assert call.resolved_callable_id is None
    assert call.resolved_callable_file is None


def test_nested_enum_entries_preserve_source_qualification_scopes(tmp_path):
    facts = _facts(
        tmp_path,
        """
namespace nav {
class Control {
public:
    enum Mode { MODE_IDLE = 0, MODE_ACTIVE };
};
}
""",
    )

    active = next(
        item for item in facts.source_assignments if item.target == "MODE_ACTIVE"
    )
    assert active.constant_scopes == [
        "nav::Control",
        "Mode",
        "nav::Control::Mode",
    ]
    assert active.target_identity is not None
    assert active.target_identity.class_owner == "nav::Control"
    assert active.expression == "MODE_IDLE + 1"


def test_direct_call_projection_is_not_a_storage_symbol(tmp_path):
    facts = _facts(
        tmp_path,
        """
struct Result { float values[2]; };
class Provider { public: Result read(); };
class Consumer {
    Provider _provider;
    float output;
    void run() { output = _provider.read().values[1]; }
};
""",
    )

    output = next(item for item in facts.source_assignments if item.target == "output")
    assert output.expression_ref is not None
    assert "_provider.read().values[1]" not in output.expression_ref.input_symbols
    assert "values" not in output.expression_ref.input_symbols
    assert output.expression_ref.call_results[0].result_path == "values[1]"


def test_tree_sitter_extracts_assignment_heap_base_and_c_api_boundaries(tmp_path):
    facts = _facts(
        tmp_path,
        """
class CallbackReader : public uORB::SubscriptionCallbackWorkItem
{
public:
    CallbackReader() :
        uORB::SubscriptionCallbackWorkItem(this, ORB_ID(callback_topic))
    {}
};

class Reader
{
    uORB::Subscription *_heap_sub;
    uORB::Subscription _value_sub;

    void configure()
    {
        _heap_sub = new uORB::Subscription(ORB_ID(heap_topic));
        _value_sub = uORB::Subscription{ORB_ID(value_topic)};
    }

    void copy_c_api()
    {
        c_api_topic_s sample{};
        orb_copy(ORB_ID(c_api_topic), handle, &sample);
    }
};
""",
    )

    by_topic = {item.topic: item for item in facts.subscribed_topics}
    assert by_topic["heap_topic"].variable == "_heap_sub"
    assert by_topic["heap_topic"].endpoint_kind == "member"
    assert by_topic["heap_topic"].variable_owner == "Reader"
    assert by_topic["value_topic"].variable == "_value_sub"
    assert by_topic["callback_topic"].variable == "this"
    assert by_topic["callback_topic"].endpoint_kind == "base"
    assert by_topic["callback_topic"].variable_owner == "CallbackReader"
    assert by_topic["c_api_topic"].variable == "sample"
    assert by_topic["c_api_topic"].api == "orb_copy"


def test_tree_sitter_qualifies_namespace_ownership_and_bases(tmp_path):
    facts = _facts(
        tmp_path,
        """
namespace first {
class Control
{
    int value;
public:
    void update() { value = 1; }
};
}

namespace second {
class Control
{
    int value;
public:
    void update() { value = 2; }
};
}

namespace nav {
class Base { protected: int inherited; };
class Derived : public Base { public: void update(); };
void Derived::update() { inherited = 3; }
}
""",
    )

    classes = {item.name: item for item in facts.classes}
    assert {"first::Control", "second::Control", "nav::Base", "nav::Derived"} <= set(classes)
    assert classes["nav::Derived"].bases == ["nav::Base"]

    callables = {item.name: item for item in facts.callables}
    assert callables["first::Control::update"].owner == "first::Control"
    assert callables["second::Control::update"].owner == "second::Control"
    assert callables["nav::Derived::update"].owner == "nav::Derived"

    inputs = dag_inputs_from_facts([facts])
    values = [
        binding["target_identity"]
        for binding in inputs.bindings
        if binding["target_symbol"] == "value"
    ]
    assert {identity["declaring_class"] for identity in values} == {
        "first::Control",
        "second::Control",
    }
    inherited = next(
        binding["target_identity"]
        for binding in inputs.bindings
        if binding["target_symbol"] == "inherited"
    )
    assert inherited["class_owner"] == "nav::Derived"
    assert inherited["declaring_class"] == "nav::Base"


def test_declaration_identity_distinguishes_shadowed_local_from_member(tmp_path):
    facts = _facts(
        tmp_path,
        """
class Control
{
    int value;
    int inner_output;
    int outer_output;

    void run()
    {
        value = 1;
        {
            int value = 2;
            inner_output = value;
        }
        outer_output = value;
    }
};
""",
    )

    inputs = dag_inputs_from_facts([facts])
    member_write = next(
        binding
        for binding in inputs.bindings
        if binding["target_symbol"] == "value"
        and binding["target_identity"]["kind"] == "member"
    )
    local_write = next(
        binding
        for binding in inputs.bindings
        if binding["target_symbol"] == "value"
        and binding["target_identity"]["kind"] == "local"
    )
    inner_read = next(
        binding
        for binding in inputs.bindings
        if binding["target_symbol"] == "inner_output"
    )["reference_identities"]["value"]
    outer_read = next(
        binding
        for binding in inputs.bindings
        if binding["target_symbol"] == "outer_output"
    )["reference_identities"]["value"]

    assert member_write["target_identity"]["declaration_proven"] is True
    assert local_write["target_identity"]["declaration_proven"] is True
    assert inner_read["declaration_id"] == local_write["target_identity"]["declaration_id"]
    assert outer_read["declaration_id"] == member_write["target_identity"]["declaration_id"]


def test_file_static_global_projection_resolves_to_declared_storage(tmp_path):
    facts = _facts(
        tmp_path,
        """
struct Queue { int size; };
static Queue work_queue;
static int output;

void write_queue()
{
    work_queue.size = 3;
}

void read_queue()
{
    output = work_queue.size;
}
""",
    )

    declarations = {
        item.name: item
        for item in facts.declarations
        if item.identity.kind == "global"
    }
    assert declarations["work_queue"].linkage == "internal"
    inputs = dag_inputs_from_facts([facts])
    writer = next(
        item for item in inputs.bindings if item["target_symbol"] == "work_queue.size"
    )
    reader = next(
        item for item in inputs.bindings if item["target_symbol"] == "output"
    )
    read_identity = reader["reference_identities"]["work_queue.size"]

    assert writer["target_identity"]["kind"] == "global"
    assert read_identity["kind"] == "global"
    assert read_identity["declaration_id"] == declarations[
        "work_queue"
    ].identity.declaration_id
    assert writer["target_identity"]["declaration_id"] == read_identity[
        "declaration_id"
    ]


def test_update_expressions_emit_declared_storage_writes(tmp_path):
    facts = _facts(
        tmp_path,
        """
struct Queue { int size; };
static Queue work_queue;

void mutate_queue()
{
    ++work_queue.size;
    work_queue.size--;
}
""",
    )

    declaration = next(
        item for item in facts.declarations if item.name == "work_queue"
    )
    updates = [
        item
        for item in facts.source_assignments
        if item.target == "work_queue.size"
    ]

    assert [item.assignment_operator for item in updates] == ["+=", "-="]
    assert all(
        item.target_identity.declaration_id
        == declaration.identity.declaration_id
        for item in updates
    )
    assert all(
        item.expression_ref.input_identities[
            "work_queue.size"
        ].declaration_id
        == declaration.identity.declaration_id
        for item in updates
    )


def test_file_static_globals_with_same_name_have_distinct_identity(tmp_path):
    root = tmp_path / "PX4-Autopilot"
    files = [
        "src/modules/example/first.cpp",
        "src/modules/example/second.cpp",
    ]
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("static int state;", encoding="utf-8")
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )
    facts = [
        extract_facts_for_file(profiler, relative, "source-hash")
        for relative in files
    ]

    identities = {
        entry.declarations[0].identity.declaration_id for entry in facts
    }
    assert len(identities) == 2


def test_header_static_global_identity_is_translation_unit_scoped(tmp_path):
    root = tmp_path / "PX4-Autopilot"
    module = root / "src" / "modules" / "example"
    module.mkdir(parents=True)
    (module / "shared.h").write_text(
        "static int state;",
        encoding="utf-8",
    )
    files = ["first.cpp", "second.cpp"]
    for name in files:
        (module / name).write_text(
            '#include "shared.h"\nstatic int output;\n'
            "void run() { output = state; }",
            encoding="utf-8",
        )
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )

    identities = set()
    for name in files:
        facts = extract_facts_for_file(
            profiler,
            f"src/modules/example/{name}",
            "source-hash",
        )
        inputs = dag_inputs_from_facts([facts])
        reader = next(
            item for item in inputs.bindings if item["target_symbol"] == "output"
        )
        identities.add(
            reader["reference_identities"]["state"]["declaration_id"]
        )

    assert len(identities) == 2
    assert all("shared.h" not in identity for identity in identities)


def test_out_of_class_static_member_initializer_uses_member_identity(tmp_path):
    facts = _facts(
        tmp_path,
        """
class Control
{
    static int value;
};

int Control::value = 4;
""",
    )

    assignment = next(
        item
        for item in facts.source_assignments
        if item.declaration_kind == "global"
    )
    member = next(item for item in facts.members if item.name == "value")

    assert assignment.target == "Control.value"
    assert assignment.target_identity.kind == "member"
    assert assignment.target_identity.declaration_id == (
        f"{member.file}:{member.line}:member:{member.name}"
    )


def test_declarator_identity_handles_symbolic_arrays_and_function_pointers(tmp_path):
    facts = _facts(
        tmp_path,
        """
static int values[COUNT];
static int (*callback)(int);
int *function();

void run()
{
    int local_values[LIMIT];
    int (*handler)(int);
    consume(values, callback, local_values, handler);
}
""",
    )

    declarations = {item.name: item for item in facts.declarations}
    assert {"values", "callback", "local_values", "handler"} <= declarations.keys()
    assert "COUNT" not in declarations
    assert "LIMIT" not in declarations
    assert "function" not in declarations
    assert declarations["callback"].identity.kind == "global"
    assert declarations["handler"].identity.kind == "local"


def test_companion_member_declaration_survives_primary_fact_aggregation(tmp_path):
    root = tmp_path / "PX4-Autopilot"
    module = root / "src" / "modules" / "example"
    module.mkdir(parents=True)
    (module / "control.hpp").write_text(
        "class Control { int value; public: void set(); int get(); };",
        encoding="utf-8",
    )
    (module / "control.cpp").write_text(
        '#include "control.hpp"\n'
        "void Control::set() { value = 3; }\n"
        "int Control::get() { return value; }\n",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )
    facts = extract_facts_for_file(
        profiler,
        "src/modules/example/control.cpp",
        "source-hash",
    )

    # The Layer-1 payload remains primary-file scoped, but identities retain
    # the declaration proven from its parsed companion header.
    assert facts.members == []
    inputs = dag_inputs_from_facts([facts])
    write = next(
        binding for binding in inputs.bindings if binding["target_symbol"] == "value"
    )
    assert write["target_identity"]["kind"] == "member"
    assert write["target_identity"]["declaring_class"] == "Control"
    assert write["target_identity"]["declaration_proven"] is True


def test_missing_declaration_stays_unknown_instead_of_becoming_local(tmp_path):
    facts = _facts(
        tmp_path,
        """
void Control::run()
{
    value = input;
    output = value;
}
""",
    )
    inputs = dag_inputs_from_facts([facts])
    identities = {
        binding["target_symbol"]: binding["target_identity"]
        for binding in inputs.bindings
    }

    assert identities["value"]["kind"] == "unknown"
    assert identities["output"]["kind"] == "unknown"
    assert identities["value"]["declaration_proven"] is False

    dag = build_mechanism_dag(
        inputs.bindings,
        "output",
        terminal_file="src/modules/example/example.cpp",
        terminal_identity=identities["output"],
        source_structure=inputs.structure,
    )
    assert not any(
        vertex.kind == "operation" and vertex.variable == "value"
        for vertex in dag.vertices
    )
    assert dag.vertices == []


def test_anonymous_namespace_ownership_is_source_unit_scoped(tmp_path):
    root = tmp_path / "PX4-Autopilot"
    files = [
        "src/modules/example/first.cpp",
        "src/modules/example/second.cpp",
    ]
    for index, relative in enumerate(files):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"namespace {{ class Control {{ int value; void run() {{ value = {index}; }} }}; }}",
            encoding="utf-8",
        )
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )
    facts = [
        extract_facts_for_file(profiler, relative, "source-hash")
        for relative in files
    ]

    owners = {entry.classes[0].name for entry in facts}
    assert len(owners) == 2
    assert all("(anonymous@" in owner for owner in owners)
    inputs = dag_inputs_from_facts(facts)
    identities = [
        binding["target_identity"]
        for binding in inputs.bindings
        if binding["target_symbol"] == "value"
    ]
    assert len({identity["declaring_class"] for identity in identities}) == 2


def test_tree_sitter_switch_reachability_is_exact_and_site_scoped(tmp_path):
    facts = _facts(
        tmp_path,
        """
void Control::run()
{
    switch (mode) {
    case MODE_A:
        output = first;
        break;
    case MODE_B:
        if (valid) {
            output = second;
        } else {
            output = third;
        }
        break;
    default:
        output = fallback;
    }
}
""",
    )

    writes = [item for item in facts.source_assignments if item.target == "output"]
    assert len(writes) == 4
    assert all(item.reachability_exact for item in writes)
    assert any("mode == MODE_A" in " ".join(item.control_predicates) for item in writes)
    assert any(
        "mode == MODE_B" in " ".join(item.control_predicates)
        and "valid" in " ".join(item.control_predicates)
        for item in writes
    )
    assert all(
        len(item.control_predicate_site_ids) == len(item.control_predicates)
        for item in writes
    )


def test_switch_exit_gates_later_case_statements_and_stops_unreachable_walk(tmp_path):
    facts = _facts(
        tmp_path,
        """
void Control::run()
{
    switch (mode) {
    case MODE_A: {
        if (stop) {
            break;
        }
        output = active;
        break;
        unreachable = value;
    }
    default:
        output = fallback;
    }
}
""",
    )

    active = next(
        item
        for item in facts.source_assignments
        if item.target == "output" and item.expression == "active"
    )
    predicate = " ".join(active.control_predicates)
    assert "mode == MODE_A" in predicate
    assert "!(stop)" in predicate
    assert active.reachability_exact is True
    assert not any(
        item.target == "unreachable" for item in facts.source_assignments
    )


def test_switch_continue_gates_rest_of_case_without_claiming_loop_exactness(tmp_path):
    facts = _facts(
        tmp_path,
        """
void Control::run()
{
    while (running) {
        switch (mode) {
        case MODE_A:
            if (retry) continue;
            output = active;
            break;
        default:
            break;
        }
    }
}
""",
    )

    active = next(
        item
        for item in facts.source_assignments
        if item.target == "output"
    )
    assert "!(retry)" in " ".join(active.control_predicates)
    assert active.reachability_exact is False


def test_tree_sitter_helper_return_paths_keep_distinct_source_sites(tmp_path):
    facts = _facts(
        tmp_path,
        """
float choose(int value)
{
    if (value > 0) return 1.f;
    if (value < 0) return -1.f;
    return 0.f;
}
""",
    )

    helper = next(item for item in facts.helper_expressions if item.name == "choose")
    assert helper.unresolved_reason is None
    assert len(helper.branches) == 3
    assert [site.expression for site in helper.return_sites] == [
        "1.f",
        "-1.f",
        "0.f",
    ]
    assert all(site.expression_ref is not None for site in helper.return_sites)
    assert len({item["source_site_id"] for item in helper.branches}) == 3
    assert len({item["line"] for item in helper.branches}) == 3


def test_owner_scoped_boundary_join_does_not_cross_same_named_members(tmp_path):
    facts = _facts(
        tmp_path,
        """
class First
{
    uORB::Subscription _sub{ORB_ID(first_topic)};
    void read() { first_topic_s first{}; _sub.copy(&first); }
};

class Second
{
    uORB::Subscription _sub{ORB_ID(second_topic)};
    void read() { second_topic_s second{}; _sub.copy(&second); }
};
""",
    )

    inputs = dag_inputs_from_facts([facts])
    placements = {
        (item["source_symbol"], item["topic"])
        for item in inputs.boundary_bindings
        if item["source_symbol"] in {"first", "second"}
    }
    assert placements == {("first", "first_topic"), ("second", "second_topic")}


def test_compare_mode_returns_tree_sitter_facts_without_union(tmp_path):
    source = """
void Control::run()
{
    value = input;
}
"""
    tree_facts = _facts(tmp_path, source, backend="tree_sitter")
    compare_facts = _facts(tmp_path, source, backend="compare")

    assert compare_facts.parser_backend == "compare:tree_sitter"
    assert compare_facts.source_assignments == tree_facts.source_assignments
    comparison = compare_facts.parse_diagnostics["legacy_comparison"]
    assert "source_assignments" in comparison
    assert comparison["source_assignments"]["tree_sitter_count"] == len(
        tree_facts.source_assignments
    )


def test_lambda_is_extracted_as_scoped_helper_not_parent_assignment(tmp_path):
    facts = _facts(
        tmp_path,
        """
float Control::run(float input)
{
    const float scale = 2.f;
    const auto apply = [scale](float value) {
        return scale * value;
    };
    return apply(input);
}
""",
    )

    assert not any(item.target == "apply" for item in facts.source_assignments)
    helper = next(
        item for item in facts.helper_expressions if item.name.endswith("::apply")
    )
    assert helper.owner == "Control"
    assert helper.parameters == ["scale", "value"]
    assert "scale * value" in str(helper.lowered_return_expression)
    call = next(item for item in facts.function_calls if item.name == "apply")
    assert call.args == ["scale", "input"]


def test_lambda_default_and_initializer_captures_are_source_bound(tmp_path):
    facts = _facts(
        tmp_path,
        """
float Control::run(float input)
{
    const float scale = 2.f;
    const float offset = 1.f;
    const auto apply = [&, bias = offset](float value) {
        const float local = value + bias;
        return scale * local;
    };
    return apply(input);
}
""",
    )

    helper = next(
        item for item in facts.helper_expressions if item.name.endswith("::apply")
    )
    assert helper.unresolved_reason is None
    assert helper.parameters == ["bias", "scale", "value"]
    call = next(item for item in facts.function_calls if item.name == "apply")
    assert call.args == ["offset", "scale", "input"]


def test_lambda_local_shadow_only_applies_in_its_lexical_scope(tmp_path):
    facts = _facts(
        tmp_path,
        """
float Control::run(float input)
{
    const float scale = 2.f;
    const auto apply = [&](float value) {
        float result = scale * value;
        {
            const float scale = 3.f;
            result += scale;
        }
        return result;
    };
    return apply(input);
}
""",
    )

    helper = next(
        item for item in facts.helper_expressions if item.name.endswith("::apply")
    )
    assert helper.parameters == ["scale", "value"]
    call = next(item for item in facts.function_calls if item.name == "apply")
    assert call.args == ["scale", "input"]


def test_storage_aliases_preserve_pointer_and_reference_targets(tmp_path):
    facts = _facts(
        tmp_path,
        """
void Control::run(float input, float alternate, float yaw)
{
    position_setpoint_triplet_s *triplet = get_triplet();
    const position_setpoint_triplet_s &snapshot = *get_triplet();
    float observed_altitude = snapshot.current.alt;
    triplet->current.alt = input;
    {
        position_setpoint_s &current = triplet->next;
        current.alt = alternate;
        float &altitude = current.alt;
        altitude = input;
    }
    triplet->current.yaw = yaw;
}
""",
    )

    writes = {
        (item.expression, item.target.replace("->", "."))
        for item in facts.source_assignments
    }
    assert ("input", "get_triplet().current.alt") in writes
    assert ("snapshot.current.alt", "observed_altitude") in writes
    assert ("alternate", "get_triplet().next.alt") in writes
    assert ("input", "get_triplet().next.alt") in writes
    assert ("yaw", "get_triplet().current.yaw") in writes


def test_included_declaration_defaults_and_call_comments_are_structural(tmp_path):
    root = tmp_path / "PX4-Autopilot"
    source_file = root / "src" / "modules" / "example" / "control.cpp"
    header_file = source_file.with_suffix(".hpp")
    source_file.parent.mkdir(parents=True, exist_ok=True)
    header_file.write_text(
        """
class Control {
public:
    float adapt(float value, bool enabled = false);
    void run(float input, bool enabled);
};
""",
        encoding="utf-8",
    )
    source_file.write_text(
        """
#include "control.hpp"

float Control::adapt(float value, bool enabled)
{
    return enabled ? value * 2.0f : value;
}

void Control::run(float input, bool enabled)
{
    float output = adapt(input);
    consume(input,
            // explanation between arguments
            output,
            /* block comment is trivia too */ enabled);
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )
    facts = extract_facts_for_file(
        profiler,
        "src/modules/example/control.cpp",
        "source-hash",
    )

    helper = next(item for item in facts.helper_expressions if item.name.endswith("::adapt"))
    callable_ref = next(item for item in facts.callables if item.name.endswith("::adapt"))
    assert helper.parameter_defaults == [None, "false"]
    assert callable_ref.parameter_defaults == [None, "false"]
    assert next(item for item in facts.function_calls if item.name == "adapt").args == ["input"]
    assert next(item for item in facts.function_calls if item.name == "consume").args == [
        "input",
        "output",
        "enabled",
    ]


def test_member_initializers_and_value_receivers_are_source_facts(tmp_path):
    facts = _facts(
        tmp_path,
        """
class VectorState {
public:
    float norm() const;
};

class Control {
    bool valid{false};
    VectorState velocity{};
    float output{};

    void run()
    {
        output = velocity.norm();
    }
};
""",
    )

    initial = next(
        item
        for item in facts.source_assignments
        if item.target == "valid" and item.declaration_kind == "member_initializer"
    )
    assert initial.owner == "Control"
    assert initial.expression == "false"
    write = next(
        item
        for item in facts.source_assignments
        if item.target == "output" and item.function == "Control::run"
    )
    assert write.expression_ref is not None
    assert write.expression_ref.exact is True
    assert write.expression_ref.input_symbols == ["velocity"]


def test_conditional_endpoint_alternatives_keep_one_storage_identity(tmp_path):
    facts = _facts(
        tmp_path,
        """
class Publisher {
    uORB::Publication<message_s> endpoint;

public:
    Publisher(bool use_primary) :
        endpoint(use_primary ? ORB_ID(primary_topic) : ORB_ID(secondary_topic))
    {}
};
""",
    )

    endpoints = sorted(facts.published_topics, key=lambda item: item.topic)
    assert [(item.variable, item.topic) for item in endpoints] == [
        ("endpoint", "primary_topic"),
        ("endpoint", "secondary_topic"),
    ]
    by_topic = {item.topic: item for item in endpoints}
    assert by_topic["primary_topic"].control_predicates == ["use_primary"]
    assert by_topic["secondary_topic"].control_predicates == ["!(use_primary)"]
    assert all(
        item.control_expression_refs
        and item.control_expression_refs[0].input_symbols == ["use_primary"]
        for item in endpoints
    )


def test_storage_aliases_do_not_leak_between_switch_cases(tmp_path):
    facts = _facts(
        tmp_path,
        """
void Control::run(int mode, float input, float alternate)
{
    state_s first{};
    state_s second{};
    state_s *selected = &first;
    switch (mode) {
    case 0:
        selected = &second;
        selected->value = input;
        break;
    case 1:
        selected->value = alternate;
        break;
    }
}
""",
    )

    writes = {
        (item.expression, item.target.replace("->", "."))
        for item in facts.source_assignments
    }
    assert ("input", "second.value") in writes
    assert ("alternate", "first.value") in writes


def test_switch_fallthrough_alias_target_is_explicitly_non_exact(tmp_path):
    facts = _facts(
        tmp_path,
        """
void Control::run(int mode, float input)
{
    state_s first{};
    state_s second{};
    state_s *selected = &first;
    switch (mode) {
    case 0:
        selected = &second;
    case 1:
        selected->value = input;
        break;
    }
}
""",
    )

    write = next(
        item
        for item in facts.source_assignments
        if item.expression == "input" and item.target.replace("->", ".").endswith(".value")
    )
    assert write.reachability_exact is False


def test_helper_metadata_uses_resolved_pointer_output_storage(tmp_path):
    facts = _facts(
        tmp_path,
        """
void fill(output_s *sp, float value)
{
    output_s *local = sp;
    local->alt = value;
}
""",
    )

    helper = next(item for item in facts.helper_expressions if item.name == "fill")
    assert helper.assignments["sp.alt"] == "value"
    assert helper.pointer_output_writes == [
        {"param": "sp", "field": "alt", "expression": "value"}
    ]


def test_helper_metadata_derives_reference_output_writes(tmp_path):
    facts = _facts(
        tmp_path,
        """
void fill(float value, output_s &out, float &direct)
{
    out.alt = value;
    direct = value * 2.0f;
}
""",
    )

    helper = next(item for item in facts.helper_expressions if item.name == "fill")
    assert helper.pointer_output_writes == [
        {"param": "out", "field": "alt", "expression": "value"},
        {"param": "direct", "field": "", "expression": "value * 2.0"},
    ]


def test_source_defined_attribute_macro_is_projected_without_moving_sites(
    tmp_path,
):
    root = tmp_path / "PX4-Autopilot"
    module = root / "src" / "modules" / "example"
    module.mkdir(parents=True)
    (module / "attributes.h").write_text(
        '#define PUBLIC_ENTRY __attribute__((visibility("default")))\n',
        encoding="utf-8",
    )
    source = """#include "attributes.h"
extern "C" PUBLIC_ENTRY int example_main(int input)
{
    output = input;
    return output;
}
"""
    (module / "example.cpp").write_text(source, encoding="utf-8")
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )

    facts = extract_facts_for_file(
        profiler,
        "src/modules/example/example.cpp",
        "source-hash",
    )

    assert facts.parse_diagnostics == {"has_error": False, "error_nodes": []}
    assert any(item.name == "example_main" for item in facts.callables)
    assignment = next(
        item for item in facts.source_assignments if item.target == "output"
    )
    assert assignment.line == 4
    assert f":{source.encode().index(b'output')}:" in str(
        assignment.source_site_id
    )


def test_source_defined_declaration_macro_projects_parameter_members(tmp_path):
    root = tmp_path / "PX4-Autopilot"
    module = root / "src" / "modules" / "example"
    module.mkdir(parents=True)
    (module / "members.h").write_text(
        "#define DECLARE_GROUP(...) FOR_EACH_FIELD(__VA_ARGS__)\n",
        encoding="utf-8",
    )
    (module / "example.cpp").write_text(
        """#include "members.h"
class Controller {
    DECLARE_GROUP(
        (ParamFloat<px4::params::GAIN_VALUE>) gain_value,
        // Comments inside the invocation are parser trivia, not arguments.
        (ParamInt<px4::params::SELECT_MODE>) select_mode
    )
};
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )

    facts = extract_facts_for_file(
        profiler,
        "src/modules/example/example.cpp",
        "source-hash",
    )

    assert facts.parse_diagnostics == {"has_error": False, "error_nodes": []}
    assert {(item.owner, item.name) for item in facts.members} >= {
        ("Controller", "gain_value"),
        ("Controller", "select_mode"),
    }
    assert {(item.name, item.member) for item in facts.referenced_parameters} >= {
        ("GAIN_VALUE", "gain_value"),
        ("SELECT_MODE", "select_mode"),
    }


def test_undefined_declaration_shaped_macro_remains_a_parse_error(tmp_path):
    facts = _facts(
        tmp_path,
        """
class Controller {
    UNKNOWN_GROUP((SomeType) value)
};
""",
    )

    assert facts.parse_diagnostics["has_error"] is True
    assert facts.parse_diagnostics["error_nodes"]


def test_macro_that_discards_declaration_arguments_remains_a_parse_error(
    tmp_path,
):
    root = tmp_path / "PX4-Autopilot"
    module = root / "src" / "modules" / "example"
    module.mkdir(parents=True)
    (module / "macros.h").write_text(
        "#define IGNORE_GROUP(...) unrelated_token\n",
        encoding="utf-8",
    )
    (module / "example.cpp").write_text(
        """#include "macros.h"
class Controller {
    IGNORE_GROUP((SomeType) value)
};
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(
        root,
        rg_path="missing-rg",
        source_parser_backend="tree_sitter",
    )

    facts = extract_facts_for_file(
        profiler,
        "src/modules/example/example.cpp",
        "source-hash",
    )

    assert facts.parse_diagnostics["has_error"] is True
    assert facts.parse_diagnostics["error_nodes"]


def test_parse_errors_are_reported_without_legacy_fallback(tmp_path):
    facts = _facts(
        tmp_path,
        """
void broken()
{
    if (condition {
        output = value;
    }
}
""",
    )

    assert facts.parser_backend == "tree_sitter"
    assert facts.parse_diagnostics["has_error"] is True
    assert facts.parse_diagnostics["error_nodes"]
