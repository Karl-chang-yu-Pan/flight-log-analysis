import asyncio
from pathlib import Path

from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler
from flight_log_agent.px4.source_mechanism_models import (
    SourceDiscoveryCandidateDraft,
    SourceDiscoveryDecision,
)
from flight_log_agent.px4.source_mechanism_resolver import (
    SourceMechanismResolver,
    build_source_discovery_log_context,
)


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

    assert len(packets) == 1
    packet = packets[0]
    assert packet.user_question == "Why did RTL use RTL_RETURN_ALT?"
    assert packet.depth == 0
    assert packet.new_files == ["src/modules/navigator/rtl.cpp"]
    assert packet.static_log_context["discovered_parameter_values"] == {
        "RTL_RETURN_ALT": 20,
        "VT_TYPE": 2,
    }
    assert "UNRELATED" not in packet.static_log_context["discovered_parameter_values"]
    assert packet.static_log_context["discovered_topic_fields"] == {
        "position_setpoint": ["alt"],
    }
    assert "unrelated_topic" not in packet.static_log_context["discovered_topic_fields"]
    assert result.candidates[0].title == "LLM drafted RTL altitude mechanism"
    assert result.candidates[0].source_confidence == "medium"
    assert any(
        "No time-series signal comparison" in note
        for note in result.candidates[0].resolver_notes
    )
