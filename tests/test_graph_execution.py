from types import SimpleNamespace

from flight_log_agent.analysis.graph_execution import execute_verification_graph
from flight_log_agent.analysis.log_evidence import ULogEvidenceIndex
from flight_log_agent.analysis.signature_verification import merge_graph_results
from flight_log_agent.analysis.verification_graph import compile_verification_graphs
from flight_log_agent.models import MechanismBranchGroup, MechanismCandidate, RelationshipCheckSpec, SignatureEvaluation
from flight_log_agent.px4.source_mechanism_models import SourceOutputBindingRecord


def build_graph():
    candidate = MechanismCandidate(
        name="Direct altitude publication",
        summary="Publishes global altitude as the current setpoint altitude.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    binding = SourceOutputBindingRecord(
        binding_id="altitude",
        source_symbol="vehicle_global_position.alt",
        target_symbol="triplet.current.alt",
        logged_signal="position_setpoint_triplet.current.alt",
        symbol_bindings={"vehicle_global_position.alt": "vehicle_global_position.alt"},
    )
    return compile_verification_graphs(candidate, [binding])[0]


def build_index(
    actual_values,
    *,
    parameters=None,
    source_timestamps=None,
    actual_timestamps=None,
    mode_values=None,
):
    data_list = [
        SimpleNamespace(
            name="vehicle_global_position",
            multi_id=0,
            data={"timestamp": source_timestamps or [1_000_000, 2_000_000], "alt": [10.0, 11.0]},
        ),
        SimpleNamespace(
            name="position_setpoint_triplet",
            multi_id=0,
            data={
                "timestamp": actual_timestamps or [1_000_000, 2_000_000],
                "current.alt": actual_values,
            },
        ),
    ]
    if mode_values is not None:
        data_list.append(
            SimpleNamespace(
                name="vehicle_status",
                multi_id=0,
                data={"timestamp": [1_000_000, 2_000_000], "nav_state": mode_values},
            )
        )
    else:
        data_list.append(
            SimpleNamespace(
                name="vehicle_status",
                multi_id=0,
                data={"timestamp": [1_000_000, 2_000_000], "nav_state": [5, 4], "arming_state": [2, 2]},
            )
        )
    data_list.append(
        SimpleNamespace(
            name="vehicle_command",
            multi_id=0,
            data={"timestamp": [1_000_000, 2_000_000], "command": [178, 16], "param1": [0.0, 1.0]},
        )
    )
    return ULogEvidenceIndex(
        SimpleNamespace(
            initial_parameters=parameters or {},
            data_list=data_list,
        )
    )


def test_graph_execution_supports_only_after_terminal_log_comparison_matches():
    result = execute_verification_graph(build_graph(), build_index([10.0, 11.0]))

    assert result.verdict == "supported"
    assert result.unresolved_dependencies == []
    comparison = next(item for item in result.node_results if item.status == "supported")
    assert comparison.value["sample_count"] == 2
    assert comparison.value["max_error"] == 0.0


def test_graph_execution_contradicts_when_terminal_log_comparison_mismatches():
    result = execute_verification_graph(build_graph(), build_index([10.0, 12.0]))

    assert result.verdict == "contradicted"
    comparison = next(item for item in result.node_results if item.status == "contradicted")
    assert comparison.value["max_error"] == 1.0


def test_conclusive_graph_result_changes_unresolved_signature_evaluation():
    graph_result = execute_verification_graph(build_graph(), build_index([10.0, 11.0]))
    evaluation = SignatureEvaluation(
        candidate_name="Direct altitude publication",
        verdict="unresolved",
        confidence_ceiling="unresolved",
    )

    merged = merge_graph_results(evaluation, [graph_result])

    assert merged.verdict == "supported"
    assert merged.confidence_ceiling == "medium"
    assert merged.evidence == [
        "Primary-source reconstruction matched logged terminal output position_setpoint_triplet.current.alt."
    ]
    assert merged.check_results[0]["status"] == "passed"
    assert merged.raw["verification_graphs"][0]["verdict"] == "supported"


def test_graph_execution_reconstructs_stateless_source_expression():
    candidate = MechanismCandidate(
        name="Complex altitude publication",
        summary="Requires arithmetic expression execution.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    binding = SourceOutputBindingRecord(
        binding_id="altitude",
        source_symbol="vehicle_global_position.alt + RTL_RETURN_ALT",
        target_symbol="triplet.current.alt",
        logged_signal="position_setpoint_triplet.current.alt",
        symbol_bindings={"vehicle_global_position.alt": "vehicle_global_position.alt"},
    )
    graph = compile_verification_graphs(candidate, [binding])[0]

    result = execute_verification_graph(
        graph,
        build_index([20.0, 21.0], parameters={"RTL_RETURN_ALT": 10.0}),
    )

    assert result.verdict == "supported"


def test_graph_execution_aligns_reconstruction_to_actual_output_by_prior_sample():
    result = execute_verification_graph(
        build_graph(),
        build_index(
            [10.0, 11.0],
            source_timestamps=[1_000_000, 2_000_000],
            actual_timestamps=[1_500_000, 2_500_000],
        ),
    )

    assert result.verdict == "supported"


def test_graph_execution_compares_constant_source_expression_to_logged_series():
    candidate = MechanismCandidate(
        name="Default cruising speed",
        summary="Publishes the default unset cruising speed.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    binding = SourceOutputBindingRecord(
        binding_id="constant",
        source_symbol="-1.f",
        target_symbol="triplet.current.alt",
        logged_signal="position_setpoint_triplet.current.alt",
    )
    graph = compile_verification_graphs(candidate, [binding])[0]

    result = execute_verification_graph(graph, build_index([-1.0, -1.0]))

    assert result.verdict == "supported"


def test_graph_execution_does_not_duplicate_branch_plan_predicates():
    candidate = MechanismCandidate(
        name="Bitmask branch",
        summary="Publishes altitude while a source branch is active.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
        branch_groups=[
            MechanismBranchGroup(
                name="enabled",
                source_predicates=["MODE & 2 != 0"],
                numeric_checks=[
                    RelationshipCheckSpec(
                        type="derived_expression",
                        actual="position_setpoint_triplet.current.alt",
                        expected_expression="vehicle_global_position.alt",
                    )
                ],
            )
        ],
    )
    graph = compile_verification_graphs(
        candidate,
        [
            SourceOutputBindingRecord(
                binding_id="altitude",
                source_symbol="vehicle_global_position.alt",
                target_symbol="triplet.current.alt",
                logged_signal="position_setpoint_triplet.current.alt",
                symbol_bindings={"vehicle_global_position.alt": "vehicle_global_position.alt"},
            )
        ],
    )[0]

    result = execute_verification_graph(graph, build_index([10.0, 11.0], parameters={"MODE": 2}))

    assert result.verdict == "supported"
    assert result.unresolved_dependencies == []
    assert not any(item.status in {"applicable", "excluded"} for item in result.node_results)


def test_graph_execution_selects_exactly_one_controlled_producer_per_actual_timestamp():
    candidate = MechanismCandidate(
        name="Controlled altitude producers",
        summary="Different source branches publish the same terminal.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="mode-one",
            source_symbol="10.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["vehicle_status.nav_state == 1"],
        ),
        SourceOutputBindingRecord(
            binding_id="mode-two",
            source_symbol="20.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["vehicle_status.nav_state == 2"],
        ),
    ]
    graph = compile_verification_graphs(candidate, bindings)[0]

    result = execute_verification_graph(graph, build_index([10.0, 20.0], mode_values=[1, 2]))

    assert result.verdict == "supported"
    comparison = next(item for item in result.node_results if item.status == "supported")
    assert comparison.value["sample_count"] == 2


def test_graph_execution_keeps_overlapping_controlled_producers_unresolved():
    candidate = MechanismCandidate(
        name="Overlapping altitude producers",
        summary="More than one source branch can appear active.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="positive",
            source_symbol="10.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["vehicle_status.nav_state > 0"],
        ),
        SourceOutputBindingRecord(
            binding_id="one",
            source_symbol="10.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["vehicle_status.nav_state == 1"],
        ),
    ]
    graph = compile_verification_graphs(candidate, bindings)[0]

    result = execute_verification_graph(graph, build_index([10.0, 10.0], mode_values=[1, 1]))

    assert result.verdict == "unresolved"
    assert any("2 active producers" in reason for reason in result.unresolved_dependencies)


def test_graph_execution_keeps_competing_uncontrolled_producer_unresolved():
    candidate = MechanismCandidate(
        name="Incomplete producer controls",
        summary="One competing source assignment has no recoverable control context.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="controlled",
            source_symbol="10.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["vehicle_status.nav_state == 1"],
        ),
        SourceOutputBindingRecord(
            binding_id="uncontrolled",
            source_symbol="20.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
        ),
    ]
    graph = compile_verification_graphs(candidate, bindings)[0]

    result = execute_verification_graph(graph, build_index([10.0, 10.0], mode_values=[1, 1]))

    assert result.verdict == "unresolved"
    assert any("without control predicates" in reason for reason in result.unresolved_dependencies)


def test_graph_execution_uses_lowered_control_predicate_aliases_for_log_evidence(tmp_path):
    msg_dir = tmp_path / "msg"
    msg_dir.mkdir()
    (msg_dir / "VehicleCommand.msg").write_text(
        """
uint64 timestamp
uint16 command
uint16 VEHICLE_CMD_DO_CHANGE_SPEED = 178
""",
        encoding="utf-8",
    )
    candidate = MechanismCandidate(
        name="Command-controlled altitude",
        summary="A source branch is selected by a logged command enum.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="command-field",
            source_symbol="cmd.command",
            target_symbol="cmd.command",
            logged_signal="vehicle_command.command",
        ),
        SourceOutputBindingRecord(
            binding_id="change-speed",
            source_symbol="10.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["cmd.command == vehicle_command_s::VEHICLE_CMD_DO_CHANGE_SPEED"],
            symbol_bindings={"cmd.command": "vehicle_command.command"},
        ),
        SourceOutputBindingRecord(
            binding_id="other",
            source_symbol="20.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["cmd.command != vehicle_command_s::VEHICLE_CMD_DO_CHANGE_SPEED"],
            symbol_bindings={"cmd.command": "vehicle_command.command"},
        ),
    ]
    graph = compile_verification_graphs(candidate, bindings, source_path=str(tmp_path))[0]

    result = execute_verification_graph(graph, build_index([10.0, 20.0]))

    assert result.verdict == "supported"


def test_graph_execution_does_not_invent_common_control_predicate_aliases(tmp_path):
    msg_dir = tmp_path / "msg"
    msg_dir.mkdir()
    (msg_dir / "VehicleStatus.msg").write_text(
        """
uint64 timestamp
uint8 nav_state
uint8 NAVIGATION_STATE_AUTO_RTL = 5
""",
        encoding="utf-8",
    )
    candidate = MechanismCandidate(
        name="Status-controlled altitude",
        summary="A source branch is selected by logged vehicle status.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="rtl",
            source_symbol="10.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["_vstatus.nav_state == vehicle_status_s::NAVIGATION_STATE_AUTO_RTL"],
        ),
        SourceOutputBindingRecord(
            binding_id="not-rtl",
            source_symbol="20.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["_vstatus.nav_state != vehicle_status_s::NAVIGATION_STATE_AUTO_RTL"],
        ),
    ]
    graph = compile_verification_graphs(candidate, bindings, source_path=str(tmp_path))[0]

    result = execute_verification_graph(graph, build_index([10.0, 20.0]))

    assert result.verdict == "unresolved"
    assert any("_vstatus.nav_state" in reason for reason in result.unresolved_dependencies)


def test_graph_execution_uses_source_derived_status_alias_for_control_predicate(tmp_path):
    msg_dir = tmp_path / "msg"
    msg_dir.mkdir()
    (msg_dir / "VehicleStatus.msg").write_text(
        """
uint64 timestamp
uint8 nav_state
uint8 NAVIGATION_STATE_AUTO_RTL = 5
""",
        encoding="utf-8",
    )
    candidate = MechanismCandidate(
        name="Status-controlled altitude",
        summary="A source branch is selected by logged vehicle status.",
        source_refs=[],
        primary_output_signals=["position_setpoint_triplet.current.alt"],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="rtl",
            source_symbol="10.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["_vstatus.nav_state == vehicle_status_s::NAVIGATION_STATE_AUTO_RTL"],
            symbol_bindings={"_vstatus.nav_state": "vehicle_status.nav_state"},
        ),
        SourceOutputBindingRecord(
            binding_id="not-rtl",
            source_symbol="20.f",
            target_symbol="triplet.current.alt",
            logged_signal="position_setpoint_triplet.current.alt",
            control_predicates=["_vstatus.nav_state != vehicle_status_s::NAVIGATION_STATE_AUTO_RTL"],
            symbol_bindings={"_vstatus.nav_state": "vehicle_status.nav_state"},
        ),
    ]
    graph = compile_verification_graphs(candidate, bindings, source_path=str(tmp_path))[0]

    result = execute_verification_graph(graph, build_index([10.0, 20.0]))

    assert result.verdict == "supported"
