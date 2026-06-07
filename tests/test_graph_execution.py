from types import SimpleNamespace

from flight_log_agent.analysis.graph_execution import execute_verification_graph
from flight_log_agent.analysis.log_evidence import ULogEvidenceIndex
from flight_log_agent.analysis.verification_graph import compile_verification_graphs
from flight_log_agent.models import MechanismCandidate
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
    )
    return compile_verification_graphs(candidate, [binding])[0]


def build_index(actual_values):
    return ULogEvidenceIndex(
        SimpleNamespace(
            initial_parameters={},
            data_list=[
                SimpleNamespace(
                    name="vehicle_global_position",
                    multi_id=0,
                    data={"timestamp": [1_000_000, 2_000_000], "alt": [10.0, 11.0]},
                ),
                SimpleNamespace(
                    name="position_setpoint_triplet",
                    multi_id=0,
                    data={"timestamp": [1_000_000, 2_000_000], "current.alt": actual_values},
                ),
            ],
        )
    )


def test_graph_execution_supports_only_after_terminal_log_comparison_matches():
    result = execute_verification_graph(build_graph(), build_index([10.0, 11.0]))

    assert result.verdict == "supported"
    comparison = next(item for item in result.node_results if item.status == "supported")
    assert comparison.value["sample_count"] == 2
    assert comparison.value["max_error"] == 0.0


def test_graph_execution_contradicts_when_terminal_log_comparison_mismatches():
    result = execute_verification_graph(build_graph(), build_index([10.0, 12.0]))

    assert result.verdict == "contradicted"
    comparison = next(item for item in result.node_results if item.status == "contradicted")
    assert comparison.value["max_error"] == 1.0


def test_graph_execution_keeps_complex_source_expression_pending():
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
    )
    graph = compile_verification_graphs(candidate, [binding])[0]

    result = execute_verification_graph(graph, build_index([20.0, 21.0]))

    assert result.verdict == "unresolved"
    assert any("parameter is not present" in item for item in result.unresolved_dependencies)
    assert any("requires expression execution" in item for item in result.unresolved_dependencies)
