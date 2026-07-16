from __future__ import annotations

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
    assert ("get_triplet().current.alt", "observed_altitude") in writes
    assert ("alternate", "get_triplet().next.alt") in writes
    assert ("input", "get_triplet().next.alt") in writes
    assert ("yaw", "get_triplet().current.yaw") in writes


def test_companion_declaration_defaults_and_call_comments_are_structural(tmp_path):
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


def test_tree_sitter_literal_for_loop_uses_step_ir(tmp_path):
    facts = _facts(
        tmp_path,
        """
int pick_iter()
{
    for (int i = 2; i < 3; i++) {
        return i;
    }
    return 0;
}
""",
    )

    helper = next(item for item in facts.helper_expressions if item.name == "pick_iter")
    assert helper.unresolved_reason is None
    assert helper.lowered_return_expression == "(2)"


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
