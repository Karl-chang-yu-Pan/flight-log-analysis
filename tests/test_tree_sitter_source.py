from __future__ import annotations

from flight_log_agent.analysis.mechanism_discovery import dag_inputs_from_facts
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.px4.source_facts_cache import extract_facts_for_file


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
