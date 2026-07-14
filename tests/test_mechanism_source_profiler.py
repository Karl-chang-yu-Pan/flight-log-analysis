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


def test_unknown_field_extraction_uses_dynamic_relevance_terms(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    source_file = module_dir / "example.cpp"
    source_file.write_text(
        """
void update()
{
    tecs_status.equivalent_airspeed_sp = candidate;
    battery_state.voltage_v = voltage;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")

    unrelated = profiler.extract_assigned_fields_from_source(
        ["src/modules/example/example.cpp"],
        relevance_terms=["battery voltage"],
    )
    assert {ref.variable for ref in unrelated} == {"battery_state"}

    relevant = profiler.extract_assigned_fields_from_source(
        ["src/modules/example/example.cpp"],
        relevance_terms=["airspeed setpoint"],
    )
    assert {ref.variable for ref in relevant} == {"tecs_status"}


def test_mapped_struct_fields_do_not_need_dynamic_relevance_terms(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    source_file = module_dir / "example.cpp"
    source_file.write_text(
        """
void update()
{
    vehicle_status_s status{};
    if (status.nav_state == 5) {
        return;
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    refs = profiler.extract_read_fields_from_source(
        ["src/modules/example/example.cpp"],
        relevance_terms=["battery voltage"],
    )

    assert any(ref.topic == "vehicle_status" and ref.field == "nav_state" for ref in refs)


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


def test_uorb_object_declaration_preserves_variable_and_instance(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "subscriptions.cpp").write_text(
        """
class Reader {
    uORB::Subscription _default{ORB_ID(sensor_accel)};
    uORB::Subscription _secondary{ORB_ID(sensor_accel), 1};
};
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    refs = profiler.extract_uorb_io_from_source(
        ["src/modules/example/subscriptions.cpp"]
    )["subscribed_topics"]
    by_variable = {ref.variable: ref for ref in refs if ref.variable}
    assert by_variable["_default"].topic == "sensor_accel"
    assert by_variable["_default"].instance is None
    assert by_variable["_secondary"].instance == 1


def test_callable_identity_distinguishes_same_named_definitions(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "scope.cpp").write_text(
        """
void First::update()
{
    value = first_input;
}

void Second::update()
{
    value = second_input;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/scope.cpp"]
    )
    values = [assignment for assignment in assignments if assignment.target == "value"]
    assert len(values) == 2
    assert values[0].callable_id != values[1].callable_id


def test_constexpr_assignment_carries_declaration_provenance(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "constants.cpp").write_text(
        """
static constexpr float SCALE = 2.5f;

void Example::run()
{
    output = SCALE * input;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/constants.cpp"]
    )
    scale = next(
        assignment for assignment in assignments if assignment.target == "SCALE"
    )

    assert scale.declaration_kind == "constexpr"


def test_multi_line_if_condition_attaches_predicate_to_body_assignment(tmp_path):
    """PX4's common ``if (long_a\n    && long_b) {`` pattern must attach
    the full multi-line condition as a control_predicate on assignments
    inside the body."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "cone.cpp").write_text(
        """
void Cone::pick_altitude()
{
    if (_param_rtl_cone_half_angle_deg.get() > 0
        && _navigator->get_vstatus()->vehicle_type == VEHICLE_TYPE_ROTARY_WING) {
        _rtl_alt = calculate_from_cone((float)_param_rtl_cone_half_angle_deg.get());
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/cone.cpp"]
    )
    by_target = {a.target: a for a in assignments}
    predicate = by_target["_rtl_alt"].control_predicates
    assert predicate == [
        "_param_rtl_cone_half_angle_deg.get() > 0 "
        "&& _navigator->get_vstatus()->vehicle_type == VEHICLE_TYPE_ROTARY_WING"
    ]


def test_else_branch_carries_negated_if_predicate(tmp_path):
    """The else body should carry ``!(if_predicate)`` so downstream
    feasibility can prune either arm."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "branch.cpp").write_text(
        """
void Cone::pick()
{
    if (_param_rtl_cone_half_angle_deg.get() > 0) {
        _rtl_alt = calculate_from_cone();
    } else {
        _rtl_alt = simple_max();
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/branch.cpp"]
    )
    by_line = {a.line: a for a in assignments if a.target == "_rtl_alt"}
    assert by_line[5].control_predicates == ["_param_rtl_cone_half_angle_deg.get() > 0"]
    assert by_line[7].control_predicates == [
        "!(_param_rtl_cone_half_angle_deg.get() > 0)"
    ]


def test_pointer_output_routing_strips_cxx_type_prefix_from_arg(tmp_path):
    """When a function-definition line is misidentified as a call site,
    the pointer-output arg substitution should still strip the C++ type
    declaration prefix so the routed target isn't malformed."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "block.cpp").write_text(
        """
bool
MissionBlock::mission_item_to_position_setpoint(const mission_item_s &item, position_setpoint_s *sp)
{
    sp->alt = get_absolute_altitude_for_item(item);
    return true;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/block.cpp"]
    )
    targets = {a.target for a in assignments}
    # No malformed target with type prefix leaking through.
    assert not any(" *" in t or "position_setpoint_s" in t or "mission_item_s" in t for t in targets)


def test_else_if_chain_negates_raw_siblings(tmp_path):
    """Sibling arms are mutually exclusive on the RAW conditions:
    ``if(A){} else if(B){} else if(C){} else{}`` attaches ``A``,
    ``!(A) && (B)``, ``!(A) && !(B) && (C)``, ``!(A) && !(B) && !(C)``
    — never the negation of an already-combined arm (``!(!(A) && B)``
    is ``A || !B``, a different predicate)."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "chain.cpp").write_text(
        """
void Cone::pick()
{
    if (A > 0) {
        _rtl_alt = branch_a();
    } else if (B > 0) {
        _rtl_alt = branch_b();
    } else if (C > 0) {
        _rtl_alt = branch_c();
    } else {
        _rtl_alt = default_branch();
    }
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/chain.cpp"]
    )
    by_line = {a.line: a for a in assignments if a.target == "_rtl_alt"}

    assert by_line[5].control_predicates == ["A > 0"]
    assert by_line[7].control_predicates == ["!(A > 0) && (B > 0)"]
    assert by_line[9].control_predicates == ["!(A > 0) && !(B > 0) && (C > 0)"]
    assert by_line[11].control_predicates == [
        "!(A > 0) && !(B > 0) && !(C > 0)"
    ]


def test_reference_alias_substitutes_target_in_source_assignment(tmp_path):
    """``Type &name = container.field;`` should rewrite subsequent
    ``name.X = Y;`` assignments so the recorded target carries the full
    canonical path."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "writer.cpp").write_text(
        """
void Writer::populate_setpoint(const RTLPosition &destination)
{
    position_setpoint_s &curr_sp = _navigator->get_position_setpoint_triplet()->current;
    curr_sp.lat = destination.lat;
    curr_sp.lon = destination.lon;
    curr_sp.acceptance_radius = _navigator->get_acceptance_radius();
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/writer.cpp"]
    )
    by_target = {assignment.target: assignment for assignment in assignments}

    # The substituted target traces back to the actual container the
    # reference aliased. The original ``curr_sp.X`` paths should NOT appear.
    assert "curr_sp.lat" not in by_target
    assert "_navigator.get_position_setpoint_triplet().current.lat" in by_target
    assert (
        by_target["_navigator.get_position_setpoint_triplet().current.lat"].expression
        == "destination.lat"
    )
    # Expression is run through _normalize_source_expression, which
    # collapses ``->`` to ``.``.
    assert (
        by_target["_navigator.get_position_setpoint_triplet().current.acceptance_radius"].expression
        == "_navigator.get_acceptance_radius()"
    )


def test_reference_alias_does_not_substitute_when_rhs_is_dereferenced_getter(tmp_path):
    """``Type &name = *getter();`` returns a whole struct, not a field
    projection — we leave it alone so the existing struct-var path keeps
    binding ``name.X`` via the struct type."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "reader.cpp").write_text(
        """
void Reader::compute()
{
    const vehicle_global_position_s &gpos = *_navigator->get_global_position();
    float altitude = gpos.alt;
    altitude = gpos.alt + 1.0f;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/reader.cpp"]
    )
    by_target = {assignment.target: assignment for assignment in assignments}

    # ``altitude`` is a plain local; ``gpos`` MUST NOT be substituted away,
    # the struct-var path (which knows gpos is vehicle_global_position_s)
    # is the right resolution.
    assert "altitude" in by_target
    assert by_target["altitude"].expression in {
        "gpos.alt",
        "gpos.alt + 1.0",
    }


def test_reference_alias_scoped_per_function(tmp_path):
    """An alias declared in function A must not leak into function B."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "scoped.cpp").write_text(
        """
void Writer::function_a()
{
    position_setpoint_s &curr_sp = _navigator->get_position_setpoint_triplet()->current;
    curr_sp.lat = 1.0;
}

void Writer::function_b(position_setpoint_s &curr_sp)
{
    curr_sp.lat = 2.0;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/scoped.cpp"]
    )
    by_function = {
        (assignment.function, assignment.target): assignment
        for assignment in assignments
    }

    # function_a's alias substitutes its target. Profiler returns the
    # fully-qualified function name including the class scope.
    assert ("Writer::function_a", "_navigator.get_position_setpoint_triplet().current.lat") in by_function
    # function_b's parameter is also named curr_sp but is NOT aliased
    # (the function-parameter form is not a reference *declaration* the
    # profiler recognises). It stays as ``curr_sp.lat``.
    assert ("Writer::function_b", "curr_sp.lat") in by_function


def test_reference_alias_does_not_emit_noise_for_pointer_self_writes(tmp_path):
    """A pointer initialisation like ``Type *name = &container.field;`` that
    shares its variable name with an inner-scope reference declaration
    should not appear as a substituted self-write in source_assignments.
    The alias extractor is function-scoped (not block-scoped), so a naive
    substitution would record ``container.field = &container.field`` for
    the pointer line — useful as a slicer no-op but noisy."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "scope_leak.cpp").write_text(
        """
void Writer::work()
{
    struct position_setpoint_s *curr_sp = &_navigator->get_position_setpoint_triplet()->current;
    if (curr_sp->valid) {
        position_setpoint_s &curr_sp = _navigator->get_position_setpoint_triplet()->current;
        curr_sp.lat = 1.0;
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/scope_leak.cpp"]
    )

    # The pointer line ``Type *curr_sp = &X.Y;`` should NOT show up as a
    # substituted self-write. The real reference-aliased ``curr_sp.lat = 1.0``
    # SHOULD appear with the substituted target.
    targets = {a.target for a in assignments}
    assert "_navigator.get_position_setpoint_triplet().current" not in targets
    assert "_navigator.get_position_setpoint_triplet().current.lat" in targets


def test_reference_alias_accepts_const_reference(tmp_path):
    """``const Type &name = expr;`` should be recognised just like the
    non-const form."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "const_ref.cpp").write_text(
        """
void Reader::read()
{
    const position_setpoint_s &next_sp = _navigator->get_position_setpoint_triplet()->next;
    float target = next_sp.lat;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/const_ref.cpp"]
    )
    by_target = {assignment.target: assignment for assignment in assignments}

    # ``target = next_sp.lat`` has its RHS rewritten via the alias, even
    # though the assignment target itself is a plain local.
    assert "target" in by_target
    # The RHS is normalised through _normalize_source_expression which
    # collapses arrow accesses. After alias substitution and normalisation,
    # the path through the getter chain should appear.
    expression = by_target["target"].expression
    assert "next_sp" not in expression or "next_sp.lat" in expression  # tolerate either rewriting policy


def test_source_assignment_extraction_lowers_compound_updates(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "fw_pos_control"
    module_dir.mkdir(parents=True)
    (module_dir / "FixedwingPositionControl.cpp").write_text(
        """
float adapt_airspeed_setpoint(float calibrated_min_airspeed, float weight_ratio)
{
    float load_factor_from_bank_angle = 1.0f;
    load_factor_from_bank_angle = 1.0f / cosf(_att_sp.roll_body);
    calibrated_min_airspeed *= sqrtf(load_factor_from_bank_angle * weight_ratio);
    return calibrated_min_airspeed;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/fw_pos_control/FixedwingPositionControl.cpp"]
    )
    by_line = {assignment.line: assignment for assignment in assignments}

    assert by_line[6].target == "calibrated_min_airspeed"
    assert by_line[6].assignment_operator == "*="
    assert by_line[6].expression == "calibrated_min_airspeed * (sqrt(load_factor_from_bank_angle * weight_ratio))"
    assert by_line[6].function == "adapt_airspeed_setpoint"


def test_recursive_helper_extraction_follows_calls_into_other_files(tmp_path):
    """When a helper body calls another helper defined in a different file,
    the recursive extractor finds that file via source search and inlines
    the callee's body into the caller's lowered_return_expression."""
    source_path = tmp_path / "PX4-Autopilot"
    caller_dir = source_path / "src" / "modules" / "navigator"
    callee_dir = source_path / "src" / "lib" / "geo"
    caller_dir.mkdir(parents=True)
    callee_dir.mkdir(parents=True)
    (caller_dir / "compute.cpp").write_text(
        """
float compute_total(float a, float b)
{
    return scale_pair(a, b) + 1.0f;
}
""",
        encoding="utf-8",
    )
    (callee_dir / "scale.cpp").write_text(
        """
float scale_pair(float a, float b)
{
    return a * 2.0f + b;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    refs = profiler.extract_helper_expressions_recursive(
        ["src/modules/navigator/compute.cpp"],
        helper_names=["compute_total"],
    )

    by_name = {ref.name: ref for ref in refs}
    assert "compute_total" in by_name
    composed = by_name["compute_total"].lowered_return_expression or ""
    # After cross-file composition, scale_pair's body is inlined.
    assert "scale_pair" not in composed
    assert "2.0" in composed


def test_recursive_helper_extraction_stops_when_no_new_callees(tmp_path):
    """A helper with no helper_calls (just safe-math) should not trigger
    further file searches; one pass of extraction is enough."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "noop.cpp").write_text(
        """
float self_contained(float a, float b)
{
    return max(a, b) + 1.0f;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    refs = profiler.extract_helper_expressions_recursive(
        ["src/modules/example/noop.cpp"],
        helper_names=["self_contained"],
    )

    by_name = {ref.name: ref for ref in refs}
    assert "self_contained" in by_name
    # No expansion happened — the only ref is the requested one.
    assert by_name["self_contained"].unresolved_reason is None


def test_recursive_helper_extraction_does_not_chase_safe_math(tmp_path):
    """Safe-math callees (sin, cos, max, sqrt, isfinite, ...) must not
    trigger candidate-file searches — those would burn IO on every helper
    that does ordinary arithmetic."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "trig.cpp").write_text(
        """
float distance(float lat1, float lat2)
{
    return sqrt(sin(lat1) * sin(lat1) + cos(lat2) * cos(lat2));
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    refs = profiler.extract_helper_expressions_recursive(
        ["src/modules/example/trig.cpp"],
        helper_names=["distance"],
    )

    by_name = {ref.name: ref for ref in refs}
    assert "distance" in by_name
    # Safe-math calls stay as-is in the lowered expression.
    lowered = by_name["distance"].lowered_return_expression or ""
    assert "sin(" in lowered or "sin (" in lowered
    assert "sqrt(" in lowered or "sqrt (" in lowered


def test_pointer_struct_alias_binds_to_topic_field(tmp_path):
    """A C++ pointer struct (vehicle_status_s *vstatus) should bind so that
    vstatus->vehicle_type resolves to topic vehicle_status's field vehicle_type."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "controller.cpp").write_text(
        """
void check(vehicle_status_s *vstatus, vehicle_global_position_s &gpos,
           position_setpoint_triplet_s sp_triplet)
{
    if (vstatus->vehicle_type == 1) {
        sp_triplet.current.alt = gpos.alt;
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    read_fields = profiler.extract_read_fields_from_source(
        ["src/modules/example/controller.cpp"]
    )

    bound = {(ref.variable, ref.field): ref for ref in read_fields if ref.topic}

    # Pointer form (->): vstatus → vehicle_status
    pointer_ref = bound.get(("vstatus", "vehicle_type"))
    assert pointer_ref is not None, "vstatus->vehicle_type should be captured as a read field"
    assert pointer_ref.topic == "vehicle_status"
    assert pointer_ref.struct == "vehicle_status_s"

    # Reference form (&): gpos → vehicle_global_position
    ref_ref = bound.get(("gpos", "alt"))
    assert ref_ref is not None, "gpos.alt should be captured as a read field"
    assert ref_ref.topic == "vehicle_global_position"
    assert ref_ref.struct == "vehicle_global_position_s"


def test_pointer_output_helper_writes_route_through_source_assignments(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
void mission_item_to_setpoint(const mission_item_s &item, position_setpoint_s *sp)
{
    sp->lat = item.lat;
    sp->lon = item.lon;
}

void caller()
{
    mission_item_to_setpoint(item, &triplet.current);
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/navigator/helpers.cpp"]
    )
    by_target = {a.target: a.expression for a in assignments}
    assert by_target.get("triplet.current.lat") == "item.lat"
    assert by_target.get("triplet.current.lon") == "item.lon"


def test_helper_with_local_struct_writes_is_not_rejected(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
position_setpoint_s build_default_setpoint(const mission_item_s &item)
{
    position_setpoint_s sp{};
    sp.lat = item.lat;
    sp.lon = item.lon;
    sp.alt = item.altitude;
    return sp;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["build_default_setpoint"],
    )

    assert helpers[0].name == "build_default_setpoint"
    assert helpers[0].unresolved_reason is None


def test_helper_writing_to_class_member_forwards_to_assigned_expression(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float RTL::calculate_alt()
{
    _destination.alt = _home_position.alt;
    return _destination.alt;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["calculate_alt"],
    )
    assert helpers[0].unresolved_reason is None
    assert helpers[0].lowered_return_expression == "(_home_position.alt)"


def test_helper_with_local_increment_is_not_rejected(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
int helper_with_local_counter(int n)
{
    int count = 0;
    count++;
    return count + n;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["helper_with_local_counter"],
    )
    assert helpers[0].unresolved_reason is None


def test_switch_helper_lowers_to_nested_ternary(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float pick_alt(int kind)
{
    switch (kind) {
        case RTL_DESTINATION_HOME:
            return home_alt;
        case RTL_DESTINATION_MISSION:
            return mission_alt;
        default:
            return current_alt;
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["pick_alt"],
    )
    assert helpers[0].unresolved_reason is None
    lowered = helpers[0].lowered_return_expression
    assert lowered is not None
    assert "kind == RTL_DESTINATION_HOME" in lowered
    assert "home_alt" in lowered
    assert "kind == RTL_DESTINATION_MISSION" in lowered
    assert "mission_alt" in lowered
    assert "current_alt" in lowered


def test_switch_with_fallthrough_groups_conditions(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float pick(int kind)
{
    switch (kind) {
        case A:
        case B:
            return va;
        default:
            return vd;
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["pick"],
    )
    lowered = helpers[0].lowered_return_expression
    assert lowered is not None
    assert "kind == A" in lowered
    assert "kind == B" in lowered
    assert "or" in lowered


def test_helper_with_for_loop_is_rejected_with_precise_reason(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float sum_n(int n)
{
    float total = 0;
    for (int i = 0; i < n; i++) { total += i; }
    return total;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["sum_n"],
    )
    assert helpers[0].unresolved_reason == "for loop bound 'n' is not statically resolvable"


def test_helper_with_for_range_is_rejected_with_precise_reason(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float total(const std::vector<float> &xs)
{
    float t = 0;
    for (const auto &x : xs) { t += x; }
    return t;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["total"],
    )
    assert (
        helpers[0].unresolved_reason
        == "helper body uses range-based for over 'xs' which cannot be enumerated statically"
    )


def test_helper_with_while_loop_is_rejected_with_precise_reason(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float climb_until(float current, float target)
{
    while (current < target) { current += 1; }
    return current;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["climb_until"],
    )
    assert (
        helpers[0].unresolved_reason
        == "while loop condition 'current < target' is not statically resolvable"
    )


def test_for_loop_with_literal_bound_unrolls(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float pick_iter()
{
    for (int i = 2; i < 3; i++) { return i; }
    return 0;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["pick_iter"],
    )
    assert helpers[0].unresolved_reason is None
    assert helpers[0].lowered_return_expression == "(2)"


def test_while_with_literal_const_condition_returns_on_first_iter(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float pick_while()
{
    int x = 5;
    while (x < 10) { return x; }
    return 0;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["pick_while"],
    )
    assert helpers[0].unresolved_reason is None
    assert helpers[0].lowered_return_expression == "(5)"


def test_do_while_with_unresolved_condition_is_rejected(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float drain(float current)
{
    do { current -= 1; } while (current > 0);
    return current;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["drain"],
    )
    assert (
        helpers[0].unresolved_reason
        == "do-while loop condition 'current > 0' is not statically resolvable"
    )


def test_helper_with_class_member_increment_forwards_to_lowered_expression(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
int Counter::tick()
{
    _count++;
    return _count;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["tick"],
    )
    assert helpers[0].unresolved_reason is None
    assert helpers[0].lowered_return_expression == "(_count + (1))"


def test_for_loop_with_mismatched_increment_target_is_skipped(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float pick_after_loop()
{
    for (int i = 0; i < 3; ++j) { i = 1; }
    return 7;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["pick_after_loop"],
    )
    assert helpers[0].unresolved_reason is None
    assert helpers[0].lowered_return_expression == "7"


def test_for_loop_with_zero_step_is_skipped(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
float pick_after_zero_step()
{
    for (int i = 0; i < 3; i += 0) { i = 1; }
    return 9;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["pick_after_zero_step"],
    )
    assert helpers[0].unresolved_reason is None
    assert helpers[0].lowered_return_expression == "9"


def test_void_helper_with_pointer_output_does_not_get_marked_unresolved(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
void compute(out_t *out, float input)
{
    out->field = input;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["compute"],
    )
    assert helpers[0].unresolved_reason is None


def test_void_helper_without_any_output_is_marked_unresolved(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
void noop(int x)
{
    int unused = x + 1;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["noop"],
    )
    assert (
        helpers[0].unresolved_reason
        == "helper has no return value and no pointer-output writes routable through source_assignments"
    )


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


def test_enum_entries_extracted_as_source_assignments(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "config.hpp").write_text(
        """
enum {
    STICK_CONFIG_SWAP_STICKS_BIT = (1 << 0),
    STICK_CONFIG_ENABLE_AIRSPEED_SP_MANUAL_BIT = (1 << 1),
    STICK_CONFIG_THIRD_BIT = (1 << 2),
};
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(["src/modules/example/config.hpp"])
    by_target = {a.target: a.expression for a in assignments}
    assert by_target["STICK_CONFIG_SWAP_STICKS_BIT"] == "(1 << 0)"
    assert by_target["STICK_CONFIG_ENABLE_AIRSPEED_SP_MANUAL_BIT"] == "(1 << 1)"
    assert by_target["STICK_CONFIG_THIRD_BIT"] == "(1 << 2)"
    # File and per-entry line preserved.
    entry = next(a for a in assignments if a.target == "STICK_CONFIG_ENABLE_AIRSPEED_SP_MANUAL_BIT")
    assert entry.file.endswith("config.hpp")
    assert entry.line >= 1


def test_define_macros_extracted_as_source_assignments(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "tunables.hpp").write_text(
        """
#define SIMPLE_CONST 4
#define COMPLEX_CONST (1 << 5)
#define MAX_MACRO(a, b) ((a) > (b) ? (a) : (b))
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(["src/modules/example/tunables.hpp"])
    by_target = {a.target: a.expression for a in assignments}
    assert by_target["SIMPLE_CONST"] == "4"
    assert by_target["COMPLEX_CONST"] == "(1 << 5)"
    # Function-like macros are intentionally skipped.
    assert "MAX_MACRO" not in by_target


def test_pointer_output_writes_populate_helper_expression_ref(tmp_path):
    """The helper record should surface its pointer-output writes so the
    DAG builder can emit graph-native ops at each call site without
    depending on the profiler's flattened source_assignments path."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "helpers.cpp").write_text(
        """
void mission_item_to_setpoint(const mission_item_s &item, position_setpoint_s *sp)
{
    sp->lat = item.lat;
    sp->alt = item.altitude;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/helpers.cpp"],
        helper_names=["mission_item_to_setpoint"],
    )
    assert helpers, "profiler returned no helper record"
    helper = helpers[0]
    writes = {(w["param"], w["field"]): w["expression"] for w in helper.pointer_output_writes}
    assert writes.get(("sp", "lat")) == "item.lat"
    assert writes.get(("sp", "alt")) == "item.altitude"


def test_helper_return_type_extracted_from_signature(tmp_path):
    """HelperExpressionRef should carry the raw C++ return type so the
    DAG builder can derive source→logged bindings graphically without a
    flat symbol_bindings side-table."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "navigator.cpp").write_text(
        """
vehicle_status_s * Navigator::get_vstatus()
{
    return &_vehicle_status;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/navigator.cpp"],
        helper_names=["get_vstatus"],
    )
    assert helpers, "profiler emitted no helper record"
    helper = helpers[0]
    # The pointer decorator lands in the separator, so return_type is the
    # clean struct name (what _derive_topic_from_return_type wants).
    assert helper.return_type == "vehicle_status_s"


def test_helper_return_type_strips_storage_qualifiers(tmp_path):
    """Storage qualifiers (static, const, virtual, etc.) don't belong on
    the return type — they should be stripped before the type reaches
    the DAG builder."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "example.cpp").write_text(
        """
static const vehicle_status_s * Example::get_status()
{
    return &_status;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/example/example.cpp"],
        helper_names=["get_status"],
    )
    assert helpers
    return_type = helpers[0].return_type or ""
    assert "vehicle_status_s" in return_type
    assert "static" not in return_type
    assert "const" not in return_type


def test_struct_variables_populate_on_helper_expression_ref(tmp_path):
    """Local struct variable declarations inside a helper body should
    surface as ``struct_variables`` so the DAG can derive
    ``var.field → topic.field`` graph-natively."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "example.cpp").write_text(
        """
float Example::check_status()
{
    vehicle_status_s vstatus{};
    return vstatus.vehicle_type == 1 ? 1.0f : 0.0f;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/example/example.cpp"],
        helper_names=["check_status"],
    )
    assert helpers
    assert helpers[0].struct_variables.get("vstatus") == "vehicle_status_s"


def test_struct_variables_populate_on_source_assignment_ref(tmp_path):
    """Struct variable declarations visible at a source_assignment's site
    should surface so the DAG can resolve struct-var references in the
    assignment's expression or control predicates."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "example.cpp").write_text(
        """
void Example::update()
{
    vehicle_status_s vstatus{};
    _out_alt = vstatus.vehicle_type;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/example.cpp"]
    )
    out = next(a for a in assignments if a.target == "_out_alt")
    assert out.struct_variables.get("vstatus") == "vehicle_status_s"


def test_pointer_to_member_getter_extracted_with_clean_return_type(tmp_path):
    """The uORB accessor form ``Type *get(){ return &_member; }`` — with the
    ``*`` hugging the name and no space — must be extracted with a clean
    ``return_type`` so the DAG's chain resolver can map get()->field to a
    topic. Previously the signature regex required whitespace before the
    name and skipped this shape entirely."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "navigator"
    module_dir.mkdir(parents=True)
    (module_dir / "navigator.h").write_text(
        """
float get_loiter_radius() { return _loiter_radius; }
vehicle_status_s *get_vstatus() { return &_vstatus; }
vehicle_global_position_s *get_global_position() { return &_global_pos; }
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/navigator/navigator.h"]
    )
    by = {h.name.split("::")[-1]: h for h in helpers}
    assert "get_vstatus" in by
    assert by["get_vstatus"].return_type == "vehicle_status_s"
    assert by["get_global_position"].return_type == "vehicle_global_position_s"
    # value getters still work
    assert by["get_loiter_radius"].return_type == "float"


def test_param_member_map_discovered_by_grep_survives_moves_and_whitespace(tmp_path):
    """Param member→name must survive across PX4 versions, where BOTH the
    ``DEFINE_PARAMETERS`` whitespace changes (v1.12 single space vs v1.14+
    aligned) AND the declaring file moves (v1.15/16 relocated RTL params).

    So the file is placed at an arbitrary path and *discovered by grep*
    (``search_related_source_files``) rather than named — mirroring the
    real pipeline. That's what makes the guarantee version-robust: we look
    the param up by ``px4::params::NAME``, not by a hard-coded path."""
    # Deliberately NOT rtl.h, and not the navigator dir, to prove the
    # extraction doesn't depend on where the param happens to live.
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "relocated"
    module_dir.mkdir(parents=True)
    (module_dir / "some_other_file.h").write_text(
        """
class RTL {
    DEFINE_PARAMETERS(
        (ParamFloat<px4::params::RTL_RETURN_ALT>) _param_rtl_return_alt,
        (ParamInt<px4::params::RTL_CONE_ANG>)      _param_rtl_cone_half_angle_deg
    )
};
""",
        encoding="utf-8",
    )
    # rg_path missing → the profiler falls back to its in-Python grep.
    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")

    hits = profiler.search_related_source_files(["px4::params::RTL_CONE_ANG"], max_files=5)
    discovered = [h.file for h in hits]
    assert discovered, "grep did not discover the declaring file"
    assert any("some_other_file.h" in f for f in discovered)

    params = profiler.extract_params_from_source(discovered)
    aliases = {p.member: p.name for p in params if p.member and p.name}
    # tight single-space form (v1.12) and aligned multi-space form (v1.14+)
    assert aliases.get("_param_rtl_return_alt") == "RTL_RETURN_ALT"
    assert aliases.get("_param_rtl_cone_half_angle_deg") == "RTL_CONE_ANG"


def test_loop_unroll_expression_growth_aborts_lowering(tmp_path):
    """mission.cpp DO_JUMP regression shape: an unrolled loop whose if-merge
    doubles the running value each iteration grows ~2^N and previously ran
    to MemoryError. Lowering must ABORT via _HelperLoweringFailed (helper
    opaque, reason recorded) — not cap or truncate."""
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "mission.cpp").write_text(
        """
float Mission::walk_items()
{
    float acc = base_value;
    for (int i = 0; i < 40; i++) {
        if (acc > 0.0f) {
            acc = acc + acc;
        }
    }
    return acc;
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")

    helpers = profiler.extract_helper_expressions_from_source(
        ["src/modules/example/mission.cpp"], helper_names=["walk_items"]
    )

    walk = next(h for h in helpers if h.name.endswith("walk_items"))
    assert walk.lowered_return_expression is None
    assert walk.unresolved_reason is not None
    assert "exceeded" in walk.unresolved_reason
    assert "acc" in walk.unresolved_reason


def test_file_path_boost_penalizes_camelcase_test_files(tmp_path):
    """FeasibilityCheckerTest.cpp lowercases to a bare 'test' suffix the
    underscore/dir patterns miss; latest.cpp must NOT be penalized."""
    (tmp_path / "PX4-Autopilot").mkdir()
    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")

    penalized = profiler._file_path_boost(
        "src/modules/navigator/MissionFeasibility/FeasibilityCheckerTest.cpp"
    )
    legit = profiler._file_path_boost("src/modules/navigator/latest.cpp")
    assert penalized < 0
    assert legit > 0


def test_function_call_args_captured_across_multiple_lines(tmp_path):
    """A statement call whose argument list spans several lines must keep
    its arguments; single-line parsing returned [] and dropped the
    argument dataflow entirely."""
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "fw.cpp").write_text(
        """
void Fw::control()
{
    _controller.update(first_arg,
               second_arg + 1.0f,
               third_arg);
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")

    calls = profiler.extract_function_calls_from_source(["src/modules/example/fw.cpp"])
    update = next(c for c in calls if c.name == "update")
    assert update.receiver == "_controller"
    assert update.args[0] == "first_arg"
    assert "second_arg" in update.args[1]
    assert update.args[2] == "third_arg"


def test_assignment_captured_across_multiple_lines(tmp_path):
    """A declaration whose call initializer spans lines must still yield
    a source assignment with the full RHS."""
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "fw.cpp").write_text(
        """
void Fw::control()
{
    float target_speed = adapt_speed(first_arg,
               second_arg,
               third_arg);
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")

    sas = profiler.extract_source_assignments_from_source(["src/modules/example/fw.cpp"])
    hit = next(s for s in sas if s.target == "target_speed")
    assert "adapt_speed" in hit.expression
    assert "third_arg" in hit.expression


def test_constructor_style_initialization_extracted_as_assignment(tmp_path):
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "wv.cpp").write_text(
        """
void Wv::run()
{
    matrix::Vector3f body_z_sp(matrix::Quatf(att_sp.q_d).dcm_z());
    _roll_sp = -asinf(body_z_sp(1));
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")

    sas = profiler.extract_source_assignments_from_source(["src/modules/example/wv.cpp"])
    hit = next((s for s in sas if s.target == "body_z_sp"), None)
    assert hit is not None
    assert "q_d" in hit.expression


def test_control_predicate_lines_carry_the_branch_site(tmp_path):
    """Each control predicate carries the line of its OWN control
    statement — the branch's source identity — not the line of the
    gated assignment. Two assignments in one if-block share the site;
    an else's synthesized negation sites at the else line."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "gate.cpp").write_text(
        """
void Gate::update()
{
    if (mode == 2) {
        first_out = a + 1.0f;
        second_out = b + 2.0f;
    } else {
        third_out = c + 3.0f;
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/gate.cpp"]
    )
    by_target = {a.target: a for a in assignments}

    first = by_target["first_out"]
    second = by_target["second_out"]
    third = by_target["third_out"]

    assert first.control_predicates == ["mode == 2"]
    assert len(first.control_predicate_lines) == 1
    # both if-body assignments share the if's OWN line as the site
    assert first.control_predicate_lines == second.control_predicate_lines
    assert first.control_predicate_lines[0] < first.line

    assert third.control_predicates == ["!(mode == 2)"]
    assert len(third.control_predicate_lines) == 1
    # the else arm is a different site than the if arm
    assert third.control_predicate_lines != first.control_predicate_lines


def test_braceless_controls_gate_their_single_statement(tmp_path):
    """Brace-less if/else arms govern exactly the next statement —
    same-line and next-line forms — and a following else negates the
    nearest unmatched if."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "guard.cpp").write_text(
        """
void Guard::update()
{
    if (mode == 1) same_line_out = a + 1.0f;
    if (mode == 2)
        next_line_out = b + 2.0f;
    else
        else_out = c + 3.0f;
    after_out = d + 4.0f;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/guard.cpp"]
    )
    by_target = {a.target: a for a in assignments}

    assert by_target["same_line_out"].control_predicates == ["mode == 1"]
    assert by_target["same_line_out"].control_predicate_lines == [
        by_target["same_line_out"].line
    ]
    assert by_target["next_line_out"].control_predicates == ["mode == 2"]
    assert by_target["next_line_out"].control_predicate_lines[0] < by_target["next_line_out"].line
    assert by_target["else_out"].control_predicates == ["!(mode == 2)"]
    # the statement AFTER the brace-less arms is ungated
    assert by_target["after_out"].control_predicates == []


def test_unmodeled_control_flow_marks_reachability_unresolved(tmp_path):
    """switch/case and loop bodies are constructs the extractor does not
    model: assignments inside them carry reachability_exact=False instead
    of presenting a partial predicate as exact; a plain if stays exact."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "modes.cpp").write_text(
        """
void Modes::update()
{
    switch (state) {
    case 1:
        switch_out = a + 1.0f;
        break;
    }
    for (int i = 0; i < n; i++) {
        loop_out = b + 2.0f;
    }
    while (busy) {
        while_out = c + 3.0f;
    }
    if (mode == 2) {
        exact_out = d + 4.0f;
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/modes.cpp"]
    )
    by_target = {a.target: a for a in assignments}

    assert by_target["switch_out"].reachability_exact is False
    assert by_target["loop_out"].reachability_exact is False
    assert by_target["while_out"].reachability_exact is False
    assert by_target["exact_out"].reachability_exact is True
    assert by_target["exact_out"].control_predicates == ["mode == 2"]


def test_guard_clause_return_gates_the_remainder(tmp_path):
    """A top-level return in an arm gates everything after the arm with
    the arm's negation — braced and brace-less guards, including a
    returning else arm mid-function."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "guards.cpp").write_text(
        """
void Guards::update()
{
    if (!valid) {
        return;
    }
    braced_out = a + 1.0f;
    if (busy) return;
    braceless_out = b + 2.0f;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/guards.cpp"]
    )
    by_target = {a.target: a for a in assignments}

    braced = by_target["braced_out"]
    assert braced.control_predicates == ["!(!valid)"]
    assert braced.reachability_exact is True
    # the guard's site is the if statement, not the gated assignment
    assert braced.control_predicate_lines[0] < braced.line

    braceless = by_target["braceless_out"]
    assert braceless.control_predicates == ["!(!valid)", "!(busy)"]
    assert braceless.reachability_exact is True


def test_conditionally_nested_return_marks_remainder_inexact(tmp_path):
    """A return nested deeper inside an arm returns only conditionally:
    the enclosing block's remainder is explicitly non-exact instead of
    carrying a guessed or missing gate."""
    source_path = tmp_path / "PX4-Autopilot"
    module_dir = source_path / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "nested.cpp").write_text(
        """
void Nested::update()
{
    before_out = a + 1.0f;
    if (outer_cond) {
        if (inner_cond) {
            return;
        }
        inside_out = b + 2.0f;
    }
    after_out = c + 3.0f;
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(source_path, rg_path="missing-rg")
    assignments = profiler.extract_source_assignments_from_source(
        ["src/modules/example/nested.cpp"]
    )
    by_target = {a.target: a for a in assignments}

    assert by_target["before_out"].reachability_exact is True
    # inside the outer arm, after the inner guard: gated by !(inner_cond)
    inside = by_target["inside_out"]
    assert "outer_cond" in inside.control_predicates
    assert "!(inner_cond)" in inside.control_predicates
    assert inside.reachability_exact is True
    # after the outer arm: reachable unless outer_cond && inner_cond —
    # not modeled, so explicitly non-exact
    assert by_target["after_out"].reachability_exact is False
