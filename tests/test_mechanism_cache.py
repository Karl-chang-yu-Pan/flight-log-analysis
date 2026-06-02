from pathlib import Path

from flight_log_agent.px4.mechanism_cache import (
    MECHANISM_CACHE_SCHEMA_VERSION,
    MechanismCacheConfig,
    MechanismRecord,
    MechanismRetriever,
    MechanismSourceRef,
    SourceIdentity,
)


def test_mechanism_cache_rejects_unrelated_record_despite_git_and_airframe_match(tmp_path):
    cache_root = tmp_path / "mechanisms"
    _write_record(cache_root, _rtl_record())

    result = MechanismRetriever(MechanismCacheConfig(cache_root=cache_root)).retrieve(
        {
            "airframe": {
                "vehicle_type": "vtol_standard",
                "px4_git_hash": "1dacb4cdef2d7145754fc788fa8dc482eed74b40",
            },
            "question_intent": {
                "original_question": "Why is airspeed setpoint sometimes different from FW_AIRSPD_TRIM?",
                "problem_domain": "PX4 fixed-wing/VTOL airspeed setpoint generation and fixed-wing position control",
                "concise_intent": "Trace FW_AIRSPD_TRIM to the final fixed-wing airspeed setpoint.",
                "source_queries": [
                    "FW_AIRSPD_TRIM airspeed setpoint",
                    "_param_fw_airspd_trim",
                    "position_setpoint_triplet cruising_speed",
                ],
                "likely_modules": ["fw_pos_control_l1", "navigator"],
                "likely_source_files": [
                    "src/modules/fw_pos_control_l1/FixedwingPositionControl.cpp",
                    "src/modules/navigator/mission_block.cpp",
                ],
            },
        },
        max_records=5,
    )

    assert result.records == []
    assert result.rejected_files


def test_mechanism_cache_accepts_record_with_structured_intent_overlap(tmp_path):
    cache_root = tmp_path / "mechanisms"
    _write_record(cache_root, _rtl_record())

    result = MechanismRetriever(MechanismCacheConfig(cache_root=cache_root)).retrieve(
        {
            "airframe": {
                "vehicle_type": "vtol_standard",
                "px4_git_hash": "1dacb4cdef2d7145754fc788fa8dc482eed74b40",
            },
            "question_intent": {
                "original_question": "Why is RTL_RETURN_ALT not the exact RTL altitude?",
                "problem_domain": "PX4 navigator RTL altitude selection",
                "concise_intent": "Trace RTL_RETURN_ALT, RTL_CONE_ANG, and NAV_ACC_RAD in RTL altitude selection.",
                "source_queries": [
                    "RTL_RETURN_ALT RTL_CONE_ANG PX4 navigator return altitude",
                    "NAV_ACC_RAD acceptance radius",
                ],
                "likely_modules": ["navigator"],
                "likely_source_files": ["src/modules/navigator/rtl.cpp"],
            },
        },
        max_records=5,
    )

    assert [record.name for record in result.records] == [
        "RTL cone calculation can raise the return altitude above RTL_RETURN_ALT"
    ]


def _write_record(cache_root: Path, record: MechanismRecord) -> None:
    path = cache_root / "schema_v1" / "test" / f"{record.mechanism_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def _rtl_record() -> MechanismRecord:
    return MechanismRecord(
        schema_version=MECHANISM_CACHE_SCHEMA_VERSION,
        mechanism_id="rtl_cone_altitude",
        name="RTL cone calculation can raise the return altitude above RTL_RETURN_ALT",
        summary=(
            "RTL_CONE_ANG can route RTL altitude through "
            "calculate_return_alt_from_cone_half_angle using RTL_RETURN_ALT and NAV_ACC_RAD."
        ),
        vehicle_control_domain="PX4 navigator RTL altitude selection for standard VTOL",
        source_identity=SourceIdentity(
            px4_git_hash="1dacb4cdef2d7145754fc788fa8dc482eed74b40",
            px4_version="v1.14.3",
        ),
        source_refs=[
            MechanismSourceRef(
                file="src/modules/navigator/rtl.cpp",
                function="RTL::calculate_return_alt_from_cone_half_angle",
                explanation="Computes cone-based RTL altitude.",
            ),
            MechanismSourceRef(
                file="src/modules/navigator/mission_block.cpp",
                function="MissionBlock::mission_item_to_position_setpoint",
                explanation="Publishes mission item altitude.",
            ),
        ],
        question_intents=[
            "PX4 navigator RTL altitude selection",
            "Trace RTL_RETURN_ALT in RTL altitude selection.",
        ],
        source_queries=[
            "RTL_RETURN_ALT RTL_DESCEND_ALT RTL_CONE_ANG PX4 navigator return altitude",
            "NAV_ACC_RAD acceptance radius",
        ],
        search_terms=[
            "RTL_RETURN_ALT",
            "RTL_CONE_ANG",
            "NAV_ACC_RAD",
            "calculate_return_alt_from_cone_half_angle",
        ],
        candidate_payload={
            "name": "RTL cone calculation can raise the return altitude above RTL_RETURN_ALT",
            "summary": "RTL cone altitude source mechanism.",
            "source_refs": [],
            "required_parameters": ["RTL_RETURN_ALT", "RTL_CONE_ANG", "NAV_ACC_RAD"],
            "required_signals": ["position_setpoint_triplet.current.alt"],
            "vehicle_type_gates": [],
            "airframe_gates": [],
            "mode_state_gates": [],
            "parameter_gates": [],
            "source_relevant_fields": [],
            "expected_logged_signature": [],
            "exclusion_checks": [],
            "numeric_checks": [],
            "plot_requests": [],
        },
    )
