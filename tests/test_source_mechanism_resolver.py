import asyncio
from pathlib import Path
from types import SimpleNamespace

from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.models import RelationshipCheckSpec
from flight_log_agent.px4.source_mechanism_models import (
    SourceBackedParameterPredicate,
    SourceBackedVerificationCheck,
    SourceDiscoveryCandidateDraft,
    SourceDiscoveryDecision,
)
from flight_log_agent.px4.source_mechanism_resolver import (
    SourceMechanismResolver,
    build_source_discovery_log_context,
    helper_expression_verification_candidates,
    lower_source_expression_for_evaluator,
    source_output_binding_candidates,
)


class LargeExtractionProfiler(MechanismSourceProfiler):
    def extract_function_calls_from_source(self, files):
        return [
            SimpleNamespace(
                name=f"call_{index}",
                receiver=None,
                file=str(files[0]),
                line=index,
                evidence="x" * 1000,
            )
            for index in range(200)
        ]

    def extract_branch_conditions_from_source(self, files):
        return [
            SimpleNamespace(
                kind="if",
                condition=f"condition_{index}",
                file=str(files[0]),
                line=index,
                evidence="y" * 1000,
            )
            for index in range(200)
        ]


def test_source_mechanism_resolver_discovers_and_expands_source_path(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.cpp").write_text(
        """
class RtlTest {
    ParamInt<px4::params::VT_TYPE> _param_vt_type;
    ParamFloat<px4::params::RTL_RETURN_ALT> _param_rtl_return_alt;
    uORB::Publication<position_setpoint_triplet_s> _triplet_pub{ORB_ID(position_setpoint_triplet)};
    uORB::Subscription<vehicle_status_s> _vehicle_status_sub{ORB_ID(vehicle_status)};

    void update()
    {
        vehicle_status_s status{};
        position_setpoint_s sp{};
        if (_param_vt_type.get() == 2) {
            sp.alt = _param_rtl_return_alt.get();
        }
        if (status.nav_state == 5) {
            navigateTo(sp);
        }
    }
};
""",
        encoding="utf-8",
    )
    (module_dir / "rtl_helpers.cpp").write_text(
        """
void navigateTo(position_setpoint_s sp)
{
    position_setpoint_triplet_s triplet{};
    triplet.current.alt = sp.alt;
}
""",
        encoding="utf-8",
    )

    log_context = build_source_discovery_log_context(
        {
            "parameters": {
                "VT_TYPE": 2,
                "RTL_RETURN_ALT": 20,
                "UNRELATED": 99,
            },
            "topic_fields": {
                "position_setpoint": ["alt"],
                "position_setpoint_triplet": ["current.alt"],
                "vehicle_status": ["nav_state"],
            },
            "available_topics": [
                "position_setpoint",
                "position_setpoint_triplet",
                "vehicle_status",
            ],
        }
    )
    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )

    result = asyncio.run(
        resolver.discover(
            "Why did RTL use RTL_RETURN_ALT during navigateTo?",
            log_context,
            max_depth=2,
        )
    )

    assert result.candidates
    candidate = result.candidates[0]
    assert "src/modules/navigator/rtl.cpp" in candidate.source_files
    assert "src/modules/navigator/rtl_helpers.cpp" in candidate.source_files
    assert "navigateTo" in result.expansion_queries
    assert {param.name for param in candidate.controlling_parameters} == {
        "RTL_RETURN_ALT",
        "VT_TYPE",
    }
    vt_type = next(param for param in candidate.controlling_parameters if param.name == "VT_TYPE")
    rtl_return_alt = next(param for param in candidate.controlling_parameters if param.name == "RTL_RETURN_ALT")
    assert vt_type.actual_value == 2
    assert vt_type.gate_result == "satisfied"
    assert rtl_return_alt.actual_value == 20
    assert rtl_return_alt.gate_result == "verification_required"
    assert {topic.topic for topic in candidate.published_topics} >= {"position_setpoint_triplet"}
    assert {topic.topic for topic in candidate.subscribed_topics} >= {"vehicle_status"}
    assert any(field.topic == "position_setpoint" and field.field == "alt" for field in candidate.relevant_fields)
    assert any("Fetch time-series for position_setpoint.alt" in item for item in candidate.required_log_evidence)
    assert any("No time-series signal comparison" in note for note in candidate.resolver_notes)


def test_source_mechanism_resolver_follows_generic_helper_call_to_parameter(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "mode.cpp").write_text(
        """
class ModeTest {
    Navigator *_owner;

    void run_vehicle_mode()
    {
        if (_owner->get_acceptance_radius() > 0.0f) {
            do_work();
        }
    }
};
""",
        encoding="utf-8",
    )
    (module_dir / "navigator.h").write_text(
        """
class Navigator {
    ParamFloat<px4::params::NAV_ACC_RAD> _param_nav_acc_rad;
    float get_acceptance_radius();
    float get_default_acceptance_radius();
};
""",
        encoding="utf-8",
    )
    (module_dir / "navigator_main.cpp").write_text(
        """
#include "navigator.h"

float Navigator::get_default_acceptance_radius()
{
    return _param_nav_acc_rad.get();
}

float Navigator::get_acceptance_radius()
{
    return get_default_acceptance_radius();
}
""",
        encoding="utf-8",
    )
    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )

    result = asyncio.run(
        resolver.discover(
            "Why did the selected mode use the helper radius?",
            build_source_discovery_log_context({"parameters": {"NAV_ACC_RAD": 12.5}}),
            seed_queries=["run_vehicle_mode"],
            max_depth=3,
        )
    )

    candidate = result.candidates[0]
    assert "get_acceptance_radius" in result.expansion_queries
    assert "src/modules/navigator/mode.cpp" in candidate.source_files
    assert "src/modules/navigator/navigator_main.cpp" in candidate.source_files
    assert any(requirement.name == "NAV_ACC_RAD" for requirement in candidate.controlling_parameters)
    nav_acc_rad = next(
        requirement for requirement in candidate.controlling_parameters
        if requirement.name == "NAV_ACC_RAD"
    )
    assert nav_acc_rad.actual_value == 12.5
    assert nav_acc_rad.gate_result == "verification_required"


def test_source_mechanism_resolver_profiles_requested_file_line_snippets(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "start.cpp").write_text(
        "void start() { trigger_source_search(); }\n",
        encoding="utf-8",
    )
    deep_lines = [f"// filler {index}" for index in range(1, 260)]
    deep_lines[184] = "float requested_line_185 = helper_value();"
    (module_dir / "deep.cpp").write_text("\n".join(deep_lines), encoding="utf-8")

    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        if packet.new_files == ["src/modules/navigator/start.cpp"]:
            return SourceDiscoveryDecision(
                expansion_queries=[
                    "Profile src/modules/navigator/deep.cpp around lines 180-190 for helper_value()."
                ]
            )
        return SourceDiscoveryDecision(stop=True)

    asyncio.run(
        resolver.discover(
            "Why did it call trigger_source_search?",
            build_source_discovery_log_context({}),
            seed_queries=["trigger_source_search"],
            decide=decide,
            max_depth=2,
        )
    )

    deep_packet = next(
        packet for packet in packets
        if packet.new_files == ["src/modules/navigator/deep.cpp"]
    )
    deep_snippet = next(
        snippet for snippet in deep_packet.source_profile["source_snippets"]
        if snippet["file"] == "src/modules/navigator/deep.cpp"
    )

    assert deep_snippet["start_line"] <= 185 <= deep_snippet["end_line"]
    assert "requested_line_185" in deep_snippet["text"]
    assert "filler 30" not in deep_snippet["text"]


def test_source_mechanism_resolver_includes_helper_expression_ir_in_packet(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "mode.cpp").write_text(
        """
float helper_altitude(float current_alt, float return_alt)
{
    const float candidate = current_alt + return_alt;
    return max(candidate, current_alt);
}

void run_vehicle_mode()
{
    helper_altitude(10.0f, 20.0f);
}
""",
        encoding="utf-8",
    )
    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision(relevant_files=["src/modules/navigator/mode.cpp"])
        return SourceDiscoveryDecision(stop=True)

    asyncio.run(
        resolver.discover(
            "Why did run_vehicle_mode use helper_altitude?",
            build_source_discovery_log_context({}),
            seed_queries=["run_vehicle_mode"],
            decide=decide,
            max_depth=1,
        )
    )

    profile_packet = next(packet for packet in packets if packet.new_files)
    helpers = profile_packet.source_profile["helper_expressions"]
    helper = next(item for item in helpers if item["name"] == "helper_altitude")

    assert helper["assignments"] == {"candidate": "current_alt + return_alt"}
    assert helper["return_expression"] == "max(candidate, current_alt)"
    assert helper["lowered_return_expression"] == "max((current_alt + return_alt), current_alt)"
    assert helper["parameters"] == ["current_alt", "return_alt"]
    verification_candidates = profile_packet.source_profile["expression_verification_candidates"]
    verification_candidate = next(item for item in verification_candidates if item["name"] == "helper_altitude")
    assert verification_candidate["lowered_return_expression"] == "max((current_alt + return_alt), current_alt)"
    assert verification_candidate["output_binding_required"] is True


def test_source_output_binding_candidates_bind_call_arguments_to_logged_output(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "mode.cpp").write_text(
        """
bool convert_item(const mission_item_s &item, position_setpoint_s *sp)
{
    sp->lat = item.lat;
    sp->alt = get_absolute_altitude_for_item(item);
    return true;
}

void publish_setpoint()
{
    position_setpoint_triplet_s *pos_sp_triplet = owner->get_position_setpoint_triplet();
    mission_item_s mission_item{};
    mission_item.lat = destination.lat;
    mission_item.altitude = selected_altitude;
    convert_item(mission_item, &pos_sp_triplet->current);
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(["src/modules/navigator/mode.cpp"])
    calls = profiler.extract_function_calls_from_source(["src/modules/navigator/mode.cpp"])

    bindings = source_output_binding_candidates(assignments, calls)

    assert any(
        binding["target_symbol"] == "pos_sp_triplet.current.lat"
        and binding["source_symbol"] == "mission_item.lat"
        and binding["logged_signal"] == "position_setpoint_triplet.current.lat"
        for binding in bindings
    )
    assert any(
        binding["target_symbol"] == "pos_sp_triplet.current.alt"
        and binding["source_symbol"] == "get_absolute_altitude_for_item(mission_item)"
        and binding["logged_signal"] == "position_setpoint_triplet.current.alt"
        for binding in bindings
    )
    assert any(
        binding["target_symbol"] == "pos_sp_triplet.current.lat"
        and binding["source_symbol"] == "destination.lat"
        and binding["logged_signal"] == "position_setpoint_triplet.current.lat"
        for binding in bindings
    )


def test_lower_source_expression_reuses_px4_enum_registry_for_bound_fields(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    msg_dir = source_path / "msg"
    msg_dir.mkdir(parents=True)
    (msg_dir / "VehicleStatus.msg").write_text(
        """
uint64 timestamp
uint8 vehicle_type
uint8 VEHICLE_TYPE_ROTARY_WING = 1
uint8 VEHICLE_TYPE_FIXED_WING = 2
""",
        encoding="utf-8",
    )

    lowered = lower_source_expression_for_evaluator(
        "_vstatus.vehicle_type != vehicle_status_s::VEHICLE_TYPE_ROTARY_WING",
        {"_vstatus.vehicle_type": "vehicle_status.vehicle_type"},
        source_path=source_path,
    )

    assert lowered["expression"] == "vehicle_status_vehicle_type != 1"
    assert lowered["variables"] == {
        "vehicle_status_vehicle_type": "vehicle_status.vehicle_type",
    }
    assert lowered["unresolved_symbols"] == []


def test_helper_expression_candidates_generate_output_derived_check(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "mode.cpp").write_text(
        """
float compute_alt(float radius)
{
    return 2.0f * radius;
}

bool convert_item(const mission_item_s &item, position_setpoint_s *sp)
{
    sp->alt = item.altitude;
    return true;
}

void publish_setpoint()
{
    position_setpoint_triplet_s *pos_sp_triplet = owner->get_position_setpoint_triplet();
    mission_item_s mission_item{};
    mission_item.altitude = compute_alt(NAV_ACC_RAD);
    convert_item(mission_item, &pos_sp_triplet->current);
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/mode.cpp"],
        helper_names=["compute_alt"],
    )
    assignments = profiler.extract_source_assignments_from_source(["src/modules/navigator/mode.cpp"])
    calls = profiler.extract_function_calls_from_source(["src/modules/navigator/mode.cpp"])

    candidates = helper_expression_verification_candidates(
        helpers,
        source_assignments=assignments,
        function_calls=calls,
        source_path=source_path,
        limit=20,
    )
    checks = [
        check
        for candidate in candidates
        for check in candidate["derived_expression_checks"]
        if check["source_output"] == "position_setpoint_triplet.current.alt"
    ]

    assert any(
        check["expected_expression"] == "(2.0 * NAV_ACC_RAD)"
        for check in checks
    )


def test_source_output_binding_candidates_bind_receiver_method_assignments(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.h").write_text(
        """
struct Destination {
    float alt;
    void set(const home_position_s &home_position)
    {
        alt = home_position.alt;
    }
};
""",
        encoding="utf-8",
    )
    (module_dir / "rtl.cpp").write_text(
        """
#include "rtl.h"

void select_destination()
{
    Destination _destination{};
    home_position_s home_position{};
    _destination.set(home_position);
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(["src/modules/navigator/rtl.cpp"])
    calls = profiler.extract_function_calls_from_source(["src/modules/navigator/rtl.cpp"])
    bindings = source_output_binding_candidates(assignments, calls)

    assert any(
        binding["target_symbol"] == "_destination.alt"
        and binding["source_symbol"] == "home_position.alt"
        for binding in bindings
    )


def test_source_mechanism_resolver_prioritizes_requested_file_over_search_hits(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    for index in range(3):
        (module_dir / f"noise_{index}.cpp").write_text(
            "void noise() { shared_noise_token(); }\n",
            encoding="utf-8",
        )
    (module_dir / "requested.cpp").write_text(
        "\n".join(f"// requested filler {index}" for index in range(1, 80)),
        encoding="utf-8",
    )

    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision()
        return SourceDiscoveryDecision(stop=True)

    asyncio.run(
        resolver.discover(
            "Profile src/modules/navigator/requested.cpp around lines 40-42 for shared_noise_token",
            build_source_discovery_log_context({}),
            decide=decide,
            max_depth=1,
            max_files_per_query=3,
            max_profile_files_per_iteration=1,
        )
    )

    profile_packet = next(packet for packet in packets if packet.new_files)
    assert profile_packet.new_files == ["src/modules/navigator/requested.cpp"]
    requested_snippets = [
        snippet for snippet in profile_packet.source_profile["source_snippets"]
        if snippet["file"] == "src/modules/navigator/requested.cpp"
    ]
    assert requested_snippets
    assert any(snippet["start_line"] <= 40 <= snippet["end_line"] for snippet in requested_snippets)


def test_source_mechanism_resolver_returns_unresolved_when_no_source_matches(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )

    result = asyncio.run(
        resolver.discover(
            "Why did RTL climb?",
            build_source_discovery_log_context({}),
        )
    )

    assert result.candidates == []
    assert result.unresolved_questions == ["No PX4 source files matched the discovery seed queries."]


def test_source_mechanism_resolver_calls_decision_agent_with_compact_packet(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.cpp").write_text(
        """
class RtlTest {
    ParamInt<px4::params::VT_TYPE> _param_vt_type;
    ParamFloat<px4::params::RTL_RETURN_ALT> _param_rtl_return_alt;
    void update()
    {
        position_setpoint_s sp{};
        if (_param_vt_type.get() == 2) {
            sp.alt = _param_rtl_return_alt.get();
        }
    }
};
""",
        encoding="utf-8",
    )
    log_context = build_source_discovery_log_context(
        {
            "parameters": {
                "VT_TYPE": 2,
                "RTL_RETURN_ALT": 20,
                "UNRELATED": 99,
            },
            "topic_fields": {
                "position_setpoint": ["alt"],
                "unrelated_topic": ["value"],
            },
            "available_topics": ["position_setpoint", "unrelated_topic"],
        }
    )
    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision(
                relevant_files=["src/modules/navigator/rtl.cpp"],
                expansion_queries=[],
                stop=False,
                notes=["profile rtl.cpp"],
            )
        return SourceDiscoveryDecision(
            relevant_files=["src/modules/navigator/rtl.cpp"],
            expansion_queries=[],
            stop=True,
            candidate_drafts=[
                SourceDiscoveryCandidateDraft(
                    title="LLM drafted RTL altitude mechanism",
                    source_mechanism="PX4 source gates RTL altitude behavior on VT_TYPE and RTL_RETURN_ALT.",
                    source_files=["src/modules/navigator/rtl.cpp"],
                    controlling_parameter_names=["VT_TYPE", "RTL_RETURN_ALT"],
                    relevant_signals=["position_setpoint.alt"],
                    expected_log_signature=[
                        "Verify later whether position_setpoint.alt follows the source-selected altitude."
                    ],
                    required_log_evidence=["Fetch time-series for position_setpoint.alt."],
                    source_confidence="medium",
                )
            ],
            notes=["source chain complete enough"],
        )

    result = asyncio.run(
        resolver.discover(
            "Why did RTL use RTL_RETURN_ALT?",
            log_context,
            decide=decide,
            max_depth=2,
        )
    )

    assert len(packets) == 2
    search_packet = packets[0]
    assert search_packet.source_profile["stage"] == "search_hits_only"
    assert search_packet.new_files == []
    assert list(search_packet.source_profile.keys()) == ["stage", "related_files"]
    assert search_packet.static_log_context["discovered_parameter_values"] == {}

    profile_packet = packets[1]
    assert profile_packet.user_question == "Why did RTL use RTL_RETURN_ALT?"
    assert profile_packet.depth == 0
    assert profile_packet.new_files == ["src/modules/navigator/rtl.cpp"]
    assert profile_packet.static_log_context["discovered_parameter_values"] == {}
    assert "UNRELATED" not in profile_packet.static_log_context["discovered_parameter_values"]
    assert profile_packet.static_log_context["discovered_topic_fields"] == {
        "position_setpoint": ["alt"],
    }
    assert profile_packet.source_profile["source_snippets"][0]["file"] == "src/modules/navigator/rtl.cpp"
    assert "_param_vt_type.get() == 2" in profile_packet.source_profile["source_snippets"][0]["text"]
    assert profile_packet.source_profile["published_topic_names"] == []
    assert profile_packet.source_profile["subscribed_topic_names"] == []
    assert "unrelated_topic" not in profile_packet.static_log_context["discovered_topic_fields"]
    assert result.candidates[0].title == "LLM drafted RTL altitude mechanism"
    assert result.candidates[0].source_confidence == "medium"
    assert any(
        "No time-series signal comparison" in note
        for note in result.candidates[0].resolver_notes
    )


def test_source_mechanism_resolver_caps_profiled_decision_packet(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.cpp").write_text(
        """
void update()
{
    navigateTo();
}
""",
        encoding="utf-8",
    )
    resolver = SourceMechanismResolver(
        source_path,
        profiler=LargeExtractionProfiler(source_path, rg_path="missing-rg"),
    )
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision(
                relevant_files=["src/modules/navigator/rtl.cpp"],
                stop=False,
            )
        return SourceDiscoveryDecision(stop=True)

    asyncio.run(
        resolver.discover(
            "Why did it call navigateTo?",
            build_source_discovery_log_context({}),
            decide=decide,
            max_depth=1,
        )
    )

    profile_packet = packets[1]
    assert len(profile_packet.source_profile["function_calls"]) == 80
    assert len(profile_packet.source_profile["branch_conditions"]) == 80
    assert all(
        len(item["evidence"]) <= 240
        for item in profile_packet.source_profile["function_calls"]
    )
    assert all(
        len(item["evidence"]) <= 240
        for item in profile_packet.source_profile["branch_conditions"]
    )


def test_source_mechanism_resolver_drops_drafts_with_contradicted_branch_parameters(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.cpp").write_text(
        """
class RtlTest {
    ParamInt<px4::params::VT_TYPE> _param_vt_type;
    void update()
    {
        if (_param_vt_type.get() == 2) {
            navigateTo();
        }
    }
};
""",
        encoding="utf-8",
    )
    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )
    packets = []

    async def decide(packet):
        packets.append(packet)
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision(
                relevant_files=["src/modules/navigator/rtl.cpp"],
                stop=False,
            )
        return SourceDiscoveryDecision(
            relevant_files=["src/modules/navigator/rtl.cpp"],
            stop=True,
            candidate_drafts=[
                SourceDiscoveryCandidateDraft(
                    title="Contradicted VTOL path",
                    source_mechanism="This path requires VT_TYPE == 2.",
                    source_files=["src/modules/navigator/rtl.cpp"],
                    controlling_parameter_names=["VT_TYPE"],
                    source_confidence="medium",
                )
            ],
        )

    result = asyncio.run(
        resolver.discover(
            "Why did it use the VTOL RTL branch?",
            build_source_discovery_log_context({"parameters": {"VT_TYPE": 1}}),
            decide=decide,
            max_depth=1,
        )
    )

    profile_packet = packets[1]
    assert profile_packet.static_log_context["eliminated_parameter_paths"][0]["name"] == "VT_TYPE"
    assert profile_packet.static_log_context["eliminated_parameter_paths"][0]["gate_result"] == "contradicted"
    assert all(candidate.title != "Contradicted VTOL path" for candidate in result.candidates)
    assert all(
        requirement.name != "VT_TYPE"
        for candidate in result.candidates
        for requirement in candidate.controlling_parameters
    )


def test_source_mechanism_resolver_gates_agent_produced_predicates(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.cpp").write_text(
        """
void update()
{
    if (_param_vt_type.get() == 2) {
        navigateTo();
    }
}
""",
        encoding="utf-8",
    )
    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )

    async def decide(packet):
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision(relevant_files=["src/modules/navigator/rtl.cpp"])
        return SourceDiscoveryDecision(
            relevant_files=["src/modules/navigator/rtl.cpp"],
            stop=True,
            candidate_drafts=[
                SourceDiscoveryCandidateDraft(
                    title="Agent interpreted VTOL branch",
                    source_mechanism="Agent says VT_TYPE gates the branch.",
                    source_files=["src/modules/navigator/rtl.cpp"],
                    interpreted_parameter_predicates=[
                        SourceBackedParameterPredicate(
                            name="VT_TYPE",
                            role="branch_selector",
                            predicate="_param_vt_type.get() == 2",
                            operator="==",
                            compared_value=2,
                            effect="VT_TYPE satisfies the VTOL branch.",
                            source_file="src/modules/navigator/rtl.cpp",
                            source_line=4,
                        )
                    ],
                    verification_checks=[
                        SourceBackedVerificationCheck(
                            check=RelationshipCheckSpec(
                                type="branch_parameter_satisfied",
                                parameter="VT_TYPE",
                                op="==",
                                value=2,
                            ),
                            source_file="src/modules/navigator/rtl.cpp",
                            source_line=4,
                        )
                    ],
                )
            ],
        )

    result = asyncio.run(
        resolver.discover(
            "Why did it use the VTOL branch?",
            build_source_discovery_log_context({"parameters": {"VT_TYPE": 2}}),
            seed_queries=["navigateTo"],
            decide=decide,
            max_depth=1,
        )
    )

    candidate = result.candidates[0]
    assert candidate.title == "Agent interpreted VTOL branch"
    assert candidate.controlling_parameters[0].name == "VT_TYPE"
    assert candidate.controlling_parameters[0].gate_result == "satisfied"
    assert candidate.verification_checks[0].check.type == "branch_parameter_satisfied"


def test_source_mechanism_resolver_discards_uncited_agent_facts(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "rtl.cpp").write_text("void update() { navigateTo(); }\n", encoding="utf-8")
    resolver = SourceMechanismResolver(
        source_path,
        profiler=MechanismSourceProfiler(source_path, rg_path="missing-rg"),
    )

    async def decide(packet):
        if packet.source_profile.get("stage") == "search_hits_only":
            return SourceDiscoveryDecision(relevant_files=["src/modules/navigator/rtl.cpp"])
        return SourceDiscoveryDecision(
            relevant_files=["src/modules/navigator/rtl.cpp"],
            stop=True,
            candidate_drafts=[
                SourceDiscoveryCandidateDraft(
                    title="Uncited agent facts",
                    source_mechanism="Agent omitted source refs.",
                    source_files=["src/modules/navigator/rtl.cpp"],
                    interpreted_parameter_predicates=[
                        SourceBackedParameterPredicate(
                            name="VT_TYPE",
                            role="branch_selector",
                            predicate="VT_TYPE == 2",
                            operator="==",
                            compared_value=2,
                        )
                    ],
                    verification_checks=[
                        SourceBackedVerificationCheck(
                            check=RelationshipCheckSpec(type="parameter_equals", parameter="VT_TYPE", value=2)
                        )
                    ],
                )
            ],
        )

    result = asyncio.run(
        resolver.discover(
            "Why did it use the VTOL branch?",
            build_source_discovery_log_context({"parameters": {"VT_TYPE": 2}}),
            seed_queries=["navigateTo"],
            decide=decide,
            max_depth=1,
        )
    )

    candidate = result.candidates[0]
    assert candidate.interpreted_parameter_predicates == []
    assert candidate.verification_checks == []
