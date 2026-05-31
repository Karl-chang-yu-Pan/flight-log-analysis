from pathlib import Path

from flight_log_agent.px4.mechanism_source_profiler import (
    BranchConditionRef,
    FieldRef,
    FunctionCallRef,
    HelperExpressionRef,
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
        helper_expressions=[
            HelperExpressionRef(
                name="computeAltitude",
                file="src/modules/navigator/rtl.cpp",
                line=52,
                evidence="float computeAltitude(float current_alt)",
                parameters=["current_alt"],
                assignments={"candidate": "current_alt + 10.0"},
                return_expression="max(candidate, current_alt)",
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
    assert dumped["helper_expressions"][0]["return_expression"] == "max(candidate, current_alt)"
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


def test_search_filters_broad_prompt_noise_and_downranks_vendor_paths(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    relevant = source_path / "src" / "modules" / "navigator" / "mode.cpp"
    catch_header = source_path / "test" / "mavsdk_tests" / "catch2" / "catch.hpp"
    vendor_header = (
        source_path
        / "src"
        / "drivers"
        / "uavcan"
        / "uavcan_drivers"
        / "stm32h7"
        / "driver"
        / "include"
        / "fdcan.h"
    )
    relevant.parent.mkdir(parents=True)
    catch_header.parent.mkdir(parents=True)
    vendor_header.parent.mkdir(parents=True)
    relevant.write_text(
        "void update_altitude_waypoint() { calculate_setpoint(); }\n",
        encoding="utf-8",
    )
    catch_header.write_text(
        "\n".join(["// around return source including parameter values"] * 200),
        encoding="utf-8",
    )
    vendor_header.write_text(
        "\n".join(["// around selection definitions including source handling"] * 200),
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    hits = profiler.search_related_source_files(
        "Open src/modules/navigator/mode.cpp around return altitude source including parameter values",
        max_files=10,
    )
    ranked_files = [hit.file for hit in hits]

    assert ranked_files[0] == "src/modules/navigator/mode.cpp"
    assert "test/mavsdk_tests/catch2/catch.hpp" not in ranked_files[:3]
    assert "src/drivers/uavcan/uavcan_drivers/stm32h7/driver/include/fdcan.h" not in ranked_files[:3]


def test_helper_expression_translation_extracts_simple_returns(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float helper_altitude(float current_alt, float return_alt)
{
    const float candidate = current_alt + return_alt;
    return max(candidate, current_alt);
}

float branch_altitude(float dist, float min_dist, float return_alt)
{
    if (dist <= min_dist) {
        return min(dist / tanf(1.0f), return_alt);
    } else {
        return return_alt;
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["helper_altitude", "branch_altitude"],
    )
    by_name = {helper.name: helper for helper in helpers}

    assert by_name["helper_altitude"].parameters == ["current_alt", "return_alt"]
    assert by_name["helper_altitude"].assignments == {"candidate": "current_alt + return_alt"}
    assert by_name["helper_altitude"].return_expression == "max(candidate, current_alt)"
    assert by_name["helper_altitude"].unresolved_reason is None
    assert by_name["branch_altitude"].branches == [
        {"condition": "dist <= min_dist", "expression": "min(dist / tan(1.0), return_alt)"},
        {"condition": "!(dist <= min_dist)", "expression": "return_alt"},
    ]


def test_helper_expression_translation_handles_px4_style_multiline_math(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    lib_dir = source_path / "src" / "lib" / "geo"
    lib_dir.mkdir(parents=True)
    (lib_dir / "geo.cpp").write_text(
        """
float get_distance_to_next_waypoint(double lat_now, double lon_now, double lat_next, double lon_next)
{
    const double lat_now_rad = math::radians(lat_now);
    const double lat_next_rad = math::radians(lat_next);
    const double a = sin(lat_now_rad) * cos(
        lat_next_rad);

    return static_cast<float>(CONSTANTS_RADIUS_OF_EARTH * 2.0 * atan2(sqrt(a), sqrt(1.0 - a)));
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/lib/geo/geo.cpp"],
        helper_names=["get_distance_to_next_waypoint"],
    )

    assert len(helpers) == 1
    helper = helpers[0]
    assert helper.name == "get_distance_to_next_waypoint"
    assert helper.parameters == ["lat_now", "lon_now", "lat_next", "lon_next"]
    assert helper.assignments["lat_now_rad"] == "radians(lat_now)"
    assert helper.assignments["a"] == "sin(lat_now_rad) * cos(lat_next_rad)"
    assert helper.return_expression == (
        "CONSTANTS_RADIUS_OF_EARTH * 2.0 * atan2(sqrt(a), sqrt(1.0 - a))"
    )
    assert helper.unresolved_reason is None


def test_helper_expression_translation_lowers_assignment_control_flow(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float helper_value(float distance, float radius, float floor_value, float current_value)
{
    float selected = floor_value;

    if (distance <= radius) {
        selected = distance * 2.0f;
    } else {
        selected = max(selected, radius * 2.0f);
    }

    return max(selected, current_value);
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/example/helpers.cpp"],
        helper_names=["helper_value"],
    )

    helper = helpers[0]
    assert helper.statements[1]["kind"] == "if"
    assert helper.statements[1]["then"][0] == {
        "kind": "assign",
        "target": "selected",
        "expression": "distance * 2.0",
    }
    assert helper.lowered_return_expression == (
        "max(((distance * 2.0 if distance <= radius else max((floor_value), radius * 2.0))), current_value)"
    )
    assert helper.unresolved_reason is None


def test_helper_expression_translation_composes_pure_wrappers(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.h").write_text(
        """
class Example {
    DEFINE_PARAMETERS(
        (ParamFloat<px4::params::NAV_ACC_RAD>) _param_nav_acc_rad
    )
};
""",
        encoding="utf-8",
    )
    (module_dir / "helpers.cpp").write_text(
        """
#include "helpers.h"

float get_default_acceptance_radius()
{
    return _param_nav_acc_rad.get();
}

float get_acceptance_radius(bool rotary_wing, Example *navigator, float controller_radius)
{
    if (rotary_wing) {
        return navigator->get_default_acceptance_radius();
    }

    return controller_radius;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/example/helpers.cpp"],
        helper_names=["get_acceptance_radius"],
    )

    helper = helpers[0]
    assert helper.lowered_return_expression == "((NAV_ACC_RAD) if rotary_wing else controller_radius)"
    assert {
        "call": "navigator.get_default_acceptance_radius()",
        "kind": "translated_pure_helper",
        "expression": "NAV_ACC_RAD",
    } in helper.call_resolutions
    assert helper.unresolved_reason is None


def test_helper_expression_translation_reports_generic_symbol_bindings(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.h").write_text(
        """
class Example {
    DEFINE_PARAMETERS(
        (ParamFloat<px4::params::NAV_ACC_RAD>) _param_nav_acc_rad
    )
};
""",
        encoding="utf-8",
    )
    (module_dir / "helpers.cpp").write_text(
        """
#include "helpers.h"

float helper_value(const vehicle_global_position_s &input)
{
    const vehicle_global_position_s &gpos = input;
    const float selected = _param_nav_acc_rad.get() + gpos.alt;
    return selected;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/example/helpers.cpp"],
        helper_names=["helper_value"],
    )

    helper = helpers[0]
    assert helper.symbol_bindings["_param_nav_acc_rad.get()"] == "NAV_ACC_RAD"
    assert helper.symbol_bindings["gpos.alt"] == "vehicle_global_position.alt"
    assert {
        "call": "_param_nav_acc_rad.get()",
        "kind": "parameter_accessor",
        "parameter": "NAV_ACC_RAD",
    } in helper.call_resolutions


def test_helper_expression_translation_binds_class_member_struct_fields(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.h").write_text(
        """
class Example {
    vehicle_status_s _vstatus{};
};
""",
        encoding="utf-8",
    )
    (module_dir / "helpers.cpp").write_text(
        """
#include "helpers.h"

float helper_radius(float fallback)
{
    if (_vstatus.vehicle_type == vehicle_status_s::VEHICLE_TYPE_ROTARY_WING) {
        return fallback;
    }

    return fallback * 2.0f;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helper = profiler.extract_helper_expressions_from_source(
        ["src/modules/example/helpers.cpp"],
        helper_names=["helper_radius"],
    )[0]

    assert helper.symbol_bindings["_vstatus.vehicle_type"] == "vehicle_status.vehicle_type"


def test_helper_expression_translation_canonicalizes_safe_math_by_rule(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float helper_math(float x, float y)
{
    const float clamped = math::constrain(x, -1.0f, 1.0f);
    return ceilf(std::sqrt(acosf(clamped))) + fminf(y, 5.0f);
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/example/helpers.cpp"],
        helper_names=["helper_math"],
    )

    helper = helpers[0]
    assert helper.assignments["clamped"] == "constrain(x, -1.0, 1.0)"
    assert helper.return_expression == "ceil(sqrt(acos(clamped))) + min(y, 5.0)"
    assert {
        "call": "math::constrain()",
        "kind": "math_function",
        "canonical_name": "constrain",
    } in helper.call_resolutions
    assert {
        "call": "acosf()",
        "kind": "math_function",
        "canonical_name": "acos",
    } in helper.call_resolutions
    assert {
        "call": "std::sqrt()",
        "kind": "math_function",
        "canonical_name": "sqrt",
    } in helper.call_resolutions
    assert {
        "call": "fminf()",
        "kind": "math_function",
        "canonical_name": "min",
    } in helper.call_resolutions


def test_source_assignment_extraction_preserves_rhs_and_function_parameters(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
bool convert_item(const mission_item_s &item, position_setpoint_s *sp)
{
    sp->lat = item.lat;
    sp->alt = get_absolute_altitude_for_item(item);
    return true;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(["src/modules/example/helpers.cpp"])
    by_target = {assignment.target: assignment for assignment in assignments}

    assert by_target["sp.lat"].expression == "item.lat"
    assert by_target["sp.lat"].function == "convert_item"
    assert by_target["sp.lat"].function_parameters == ["item", "sp"]
    assert by_target["sp.alt"].expression == "get_absolute_altitude_for_item(item)"


def test_helper_expression_translation_marks_complex_helpers_unresolved(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
void helper_with_output(float input, float *output)
{
    *output = input;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["helper_with_output"],
    )

    assert helpers[0].name == "helper_with_output"
    assert helpers[0].unresolved_reason == "helper body mutates pointer output"


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
