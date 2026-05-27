from pathlib import Path

from flight_log_agent.px4.mechanism_source_profiler import (
    BranchConditionRef,
    FieldRef,
    FunctionCallRef,
    MechanismSourceProfile,
    ParameterPredicateRef,
    MechanismSourceProfiler,
    ParameterRef,
    SourceFileHit,
    SourceMatch,
    TopicRef,
)
from flight_log_agent.px4.source_mechanism_resolver import (
    ParameterFeasibilityGate,
    build_source_discovery_log_context,
)


def test_profiler_models_are_pydantic_serializable():
    profile = MechanismSourceProfile(
        query="RTL altitude",
        source_root="/tmp/PX4-Autopilot",
        related_files=[
            SourceFileHit(
                file="src/modules/navigator/rtl.cpp",
                score=4.0,
                matched_queries=["RTL"],
                matches=[
                    SourceMatch(
                        file="src/modules/navigator/rtl.cpp",
                        line=12,
                        text="RTL_RETURN_ALT",
                        query="RTL",
                    )
                ],
            )
        ],
        published_topics=[
            TopicRef(
                topic="position_setpoint_triplet",
                direction="publish",
                file="src/modules/navigator/rtl.cpp",
                line=20,
                evidence="uORB::Publication<position_setpoint_triplet_s>",
            )
        ],
        subscribed_topics=[],
        unknown_direction_topics=[],
        referenced_parameters=[
            ParameterRef(
                name="RTL_RETURN_ALT",
                file="src/modules/navigator/rtl.cpp",
                line=30,
                evidence="ParamFloat<px4::params::RTL_RETURN_ALT>",
                access_pattern="Param<px4::params::PARAM>",
            )
        ],
        assigned_fields=[
            FieldRef(
                topic="position_setpoint",
                struct="position_setpoint_s",
                variable="sp",
                field="alt",
                file="src/modules/navigator/rtl.cpp",
                line=40,
                evidence="sp.alt = return_alt;",
                assignment_operator="=",
            )
        ],
        read_fields=[
            FieldRef(
                topic="vehicle_status",
                struct="vehicle_status_s",
                variable="status",
                field="nav_state",
                file="src/modules/navigator/rtl.cpp",
                line=45,
                evidence="if (status.nav_state == NAVIGATION_STATE_AUTO_RTL) {",
            )
        ],
        function_calls=[
            FunctionCallRef(
                name="navigateTo",
                file="src/modules/navigator/rtl.cpp",
                line=50,
                evidence="navigateTo(sp);",
            )
        ],
        branch_conditions=[
            BranchConditionRef(
                kind="if",
                condition="_param_rtl_return_alt.get() > 0",
                file="src/modules/navigator/rtl.cpp",
                line=55,
                evidence="if (_param_rtl_return_alt.get() > 0) {",
            )
        ],
        parameter_predicates=[
            ParameterPredicateRef(
                name="RTL_RETURN_ALT",
                member="_param_rtl_return_alt",
                predicate="_param_rtl_return_alt.get() > 0",
                operator=">",
                compared_value="0",
                file="src/modules/navigator/rtl.cpp",
                line=55,
                evidence="if (_param_rtl_return_alt.get() > 0) {",
            )
        ],
    )

    dumped = profile.model_dump()

    assert dumped["related_files"][0]["matches"][0]["query"] == "RTL"
    assert dumped["published_topics"][0]["topic"] == "position_setpoint_triplet"
    assert dumped["referenced_parameters"][0]["name"] == "RTL_RETURN_ALT"
    assert dumped["assigned_fields"][0]["field"] == "alt"
    assert dumped["read_fields"][0]["field"] == "nav_state"
    assert dumped["function_calls"][0]["name"] == "navigateTo"
    assert dumped["branch_conditions"][0]["kind"] == "if"
    assert dumped["parameter_predicates"][0]["operator"] == ">"


def test_profile_mechanism_returns_plain_dict_with_current_extractions(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    source_file = module_dir / "rtl.cpp"
    source_file.write_text(
        """
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>

class RtlTest {
    ParamFloat<px4::params::RTL_RETURN_ALT> _param_rtl_return_alt;
    uORB::Publication<position_setpoint_triplet_s> _triplet_pub{ORB_ID(position_setpoint_triplet)};
    uORB::Subscription<vehicle_status_s> _vehicle_status_sub{ORB_ID(vehicle_status)};

    void update()
    {
        vehicle_status_s status{};
        position_setpoint_s sp{};
        if (_param_rtl_return_alt.get() > 0) {
            sp.alt = _param_rtl_return_alt.get();
        }
        if (status.nav_state == 5) {
            navigateTo(sp);
        }
        orb_copy(ORB_ID(vehicle_status), 0, &status);
        orb_publish(ORB_ID(position_setpoint_triplet), 0, nullptr);
    }
};
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    profile = profiler.profile_mechanism("RTL_RETURN_ALT altitude", max_files=4)

    assert isinstance(profile, dict)
    assert profile["related_files"][0]["file"] == str(Path("src/modules/navigator/rtl.cpp"))
    assert {ref["name"] for ref in profile["referenced_parameters"]} == {"RTL_RETURN_ALT"}
    assert {
        ref["topic"]
        for ref in profile["published_topics"]
    } >= {"position_setpoint_triplet"}
    assert {
        ref["topic"]
        for ref in profile["subscribed_topics"]
    } >= {"vehicle_status"}
    assert any(
        ref["topic"] == "position_setpoint" and ref["field"] == "alt"
        for ref in profile["assigned_fields"]
    )
    assert any(
        ref["topic"] == "vehicle_status" and ref["field"] == "nav_state"
        for ref in profile["read_fields"]
    )
    assert any(ref["name"] == "navigateTo" for ref in profile["function_calls"])
    assert any(
        ref["kind"] == "if" and "_param_rtl_return_alt.get() > 0" in ref["condition"]
        for ref in profile["branch_conditions"]
    )
    assert any(
        ref["name"] == "RTL_RETURN_ALT" and ref["operator"] == ">"
        for ref in profile["parameter_predicates"]
    )


def test_search_ranks_px4_source_tiers(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    files = [
        source_path / "src" / "modules" / "navigator" / "rtl.cpp",
        source_path / "src" / "lib" / "geo" / "geo.cpp",
        source_path / "src" / "drivers" / "uavcan" / "driver.cpp",
        source_path / "src" / "include" / "px4_platform_common" / "module.h",
        source_path / "src" / "systemcmds" / "param" / "param.cpp",
        source_path / "test" / "mavsdk_tests" / "catch2" / "catch.hpp",
    ]
    for index, file in enumerate(files):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"// common_mechanism_token {index}\n", encoding="utf-8")

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    hits = profiler.search_related_source_files("common_mechanism_token", max_files=10)
    ranked_files = [hit.file for hit in hits]

    assert ranked_files.index("src/modules/navigator/rtl.cpp") < ranked_files.index("src/drivers/uavcan/driver.cpp")
    assert ranked_files.index("src/lib/geo/geo.cpp") < ranked_files.index("src/include/px4_platform_common/module.h")
    assert ranked_files.index("src/drivers/uavcan/driver.cpp") < ranked_files.index("src/systemcmds/param/param.cpp")
    assert ranked_files.index("src/systemcmds/param/param.cpp") < ranked_files.index("test/mavsdk_tests/catch2/catch.hpp")


def test_parameter_feasibility_gate_uses_only_discovered_parameters():
    context = build_source_discovery_log_context(
        {
            "parameters": {
                "VT_TYPE": 2,
                "NAV_ACC_RAD": 10,
                "UNRELATED": 1,
            },
            "topic_fields": {"vehicle_status": ["timestamp", "nav_state"]},
            "available_topics": ["vehicle_status"],
        }
    )
    gate = ParameterFeasibilityGate()
    requirements = gate.evaluate(
        [
            ParameterPredicateRef(
                name="VT_TYPE",
                member="_param_vt_type",
                predicate="_param_vt_type.get() == 2",
                operator="==",
                compared_value="2",
                file="src/modules/navigator/rtl.cpp",
                line=12,
                evidence="if (_param_vt_type.get() == 2) {",
            ),
            ParameterPredicateRef(
                name="NAV_ACC_RAD",
                member="_param_nav_acc_rad",
                predicate="_param_nav_acc_rad.get() < distance_to_wp",
                operator="<",
                compared_value="distance_to_wp",
                file="src/modules/navigator/mission.cpp",
                line=30,
                evidence="if (_param_nav_acc_rad.get() < distance_to_wp) {",
            ),
        ],
        context,
    )

    assert requirements[0].name == "VT_TYPE"
    assert requirements[0].actual_value == 2
    assert requirements[0].gate_result == "satisfied"
    assert requirements[1].name == "NAV_ACC_RAD"
    assert requirements[1].role == "threshold"
    assert requirements[1].gate_result == "verification_required"
