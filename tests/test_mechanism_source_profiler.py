from pathlib import Path

from flight_log_agent.px4.mechanism_source_profiler import (
    FieldRef,
    MechanismSourceProfile,
    MechanismSourceProfiler,
    ParameterRef,
    SourceFileHit,
    SourceMatch,
    TopicRef,
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
    )

    dumped = profile.model_dump()

    assert dumped["related_files"][0]["matches"][0]["query"] == "RTL"
    assert dumped["published_topics"][0]["topic"] == "position_setpoint_triplet"
    assert dumped["referenced_parameters"][0]["name"] == "RTL_RETURN_ALT"
    assert dumped["assigned_fields"][0]["field"] == "alt"


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
        sp.alt = _param_rtl_return_alt.get();
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
