from flight_log_agent.analysis.verification_graph import (
    compile_verification_graphs,
    terminal_outputs,
)
from flight_log_agent.models import ExpectedSignatureItem, MechanismCandidate, RelationshipCheckSpec
from flight_log_agent.px4.source_mechanism_models import SourceOutputBindingRecord


def test_terminal_rooted_graph_excludes_unrelated_cruising_speed_check():
    candidate = MechanismCandidate(
        name="Altitude mechanism",
        summary="Computes and publishes an altitude.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
        expected_logged_signature=[
            ExpectedSignatureItem(
                name="altitude",
                description="Published altitude matches the source calculation.",
                signal="position_setpoint_triplet.current.alt",
            )
        ],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="actual",
                expected_expression="selected_altitude",
                variables=[
                    {"name": "actual", "source": "position_setpoint_triplet.current.alt"},
                    {"name": "selected_altitude", "source": "selected_altitude"},
                ],
            ),
            RelationshipCheckSpec(
                type="derived_expression",
                expression="actual",
                expected_expression="mission_cruising_speed",
                variables=[
                    {"name": "actual", "source": "position_setpoint_triplet.current.cruising_speed"},
                    {"name": "mission_cruising_speed", "source": "mission_cruising_speed"},
                ],
            ),
        ],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="altitude",
            source_symbol="selected_altitude",
            target_symbol="pos_sp_triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
        ),
        SourceOutputBindingRecord(
            binding_id="cruising_speed",
            source_symbol="mission_cruising_speed",
            target_symbol="pos_sp_triplet.current.cruising_speed",
            logged_signal="position_setpoint_triplet.current.cruising_speed",
        ),
    ]

    graph = compile_verification_graphs(candidate, bindings)[0]
    graph_text = graph.model_dump_json()

    assert graph.terminal_output == "position_setpoint_triplet.current.alt"
    assert "selected_altitude" in graph_text
    assert "cruising_speed" not in graph_text
    assert graph.validation_errors == []


def test_graph_preserves_specific_current_cruising_speed_primary_source():
    candidate = MechanismCandidate(
        name="Cruising speed mechanism",
        summary="Publishes the current cruising speed.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.cruising_speed"],
        expected_logged_signature=[
            ExpectedSignatureItem(
                name="current_cruising_speed",
                description="Current cruising speed matches the source value.",
                signal="position_setpoint_triplet.current.cruising_speed",
            )
        ],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="actual",
                expected_expression="navigator_cruising_speed",
                variables=[
                    {"name": "actual", "source": "position_setpoint_triplet.current.cruising_speed"},
                    {"name": "navigator_cruising_speed", "source": "navigator_cruising_speed"},
                ],
            )
        ],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="current",
            source_symbol="navigator_cruising_speed",
            target_symbol="pos_sp_triplet.current.cruising_speed",
            logged_signal="position_setpoint_triplet.current.cruising_speed",
        ),
        SourceOutputBindingRecord(
            binding_id="previous",
            source_symbol="previous_cruising_speed",
            target_symbol="pos_sp_triplet.previous.cruising_speed",
            logged_signal="position_setpoint_triplet.previous.cruising_speed",
        ),
        SourceOutputBindingRecord(
            binding_id="next",
            source_symbol="next_cruising_speed",
            target_symbol="pos_sp_triplet.next.cruising_speed",
            logged_signal="position_setpoint_triplet.next.cruising_speed",
        ),
    ]

    graph = compile_verification_graphs(candidate, bindings)[0]
    graph_text = graph.model_dump_json()

    assert graph.terminal_output == "position_setpoint_triplet.current.cruising_speed"
    assert "navigator_cruising_speed" in graph_text
    assert "previous_cruising_speed" not in graph_text
    assert "next_cruising_speed" not in graph_text
    assert graph.validation_errors == []


def test_terminal_outputs_do_not_fall_back_to_descriptions_presence_or_required_signals():
    candidate = MechanismCandidate(
        name="Presence-only candidate",
        summary="Has no primary terminal output.",
        source_refs=[],
        required_signals=["position_setpoint_triplet.current.alt"],
        expected_logged_signature=[
            ExpectedSignatureItem(
                name="descriptive_altitude",
                description="Altitude should match the mechanism.",
                signal="position_setpoint_triplet.current.alt",
            )
        ],
        numeric_checks=[
            RelationshipCheckSpec(
                type="topic_field_present",
                signal="position_setpoint_triplet.current.alt",
            )
        ],
    )

    assert terminal_outputs(candidate) == []


def test_graph_uses_transitive_primary_source_bindings():
    candidate = MechanismCandidate(
        name="Transitive altitude",
        summary="Publishes a value through an intermediate assignment.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
        expected_logged_signature=[
            ExpectedSignatureItem(
                name="altitude",
                description="Published altitude matches the selected value.",
                signal="position_setpoint_triplet.current.alt",
            )
        ],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="publish",
            source_symbol="mission_item.altitude",
            target_symbol="pos_sp_triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
        ),
        SourceOutputBindingRecord(
            binding_id="select",
            source_symbol="selected_altitude",
            target_symbol="mission_item.altitude",
        ),
    ]

    graph = compile_verification_graphs(candidate, bindings)[0]
    graph_text = graph.model_dump_json()

    assert "mission_item.altitude" in graph_text
    assert "selected_altitude" in graph_text
    assert graph.validation_errors == []
