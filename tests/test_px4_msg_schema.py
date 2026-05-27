from flight_log_agent.px4.msg_schema import (
    is_valid_topic_field,
    load_px4_msg_schema,
    resolve_topic_field,
)


def test_px4_msg_schema_expands_nested_topics():
    schema = load_px4_msg_schema()

    assert "position_setpoint_triplet" in schema
    assert "current.alt" in schema["position_setpoint_triplet"]
    assert "previous.valid" in schema["position_setpoint_triplet"]
    assert "vehicle_local_position" in schema
    assert "z" in schema["vehicle_local_position"]


def test_px4_msg_schema_validates_and_resolves_nested_field_suffixes():
    assert is_valid_topic_field("position_setpoint_triplet.current.alt")
    assert resolve_topic_field("position_setpoint_triplet", "current.alt") == "position_setpoint_triplet.current.alt"
    assert resolve_topic_field("position_setpoint_triplet", "alt") is None
