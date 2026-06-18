from types import SimpleNamespace

from flight_log_agent.analysis.signature_verification import evaluate_candidate_log_signature
from flight_log_agent.analysis.verification_plan import (
    applicability_from_verification_plan,
    compile_verification_plan,
    resolved_candidate_predicate_signals,
)
from flight_log_agent.models import (
    MechanismBranchGroup,
    MechanismCandidate,
    PlotRef,
    RelationshipCheckSpec,
)
from flight_log_agent.px4.source_mechanism_models import SourceOutputBindingRecord
import flight_log_agent.analysis.log_evidence as log_evidence
import flight_log_agent.ulog.signature_evaluator as signature_evaluator


def test_verification_plan_has_stable_branch_and_check_ids_and_roles():
    candidate = MechanismCandidate(
        name="Airspeed selection",
        summary="Selects and publishes an airspeed setpoint.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(type="topic_field_present", signal="airspeed_validated.true_airspeed_m_s"),
            RelationshipCheckSpec(
                type="tracks_setpoint",
                actual="airspeed_validated.true_airspeed_m_s",
                setpoint="position_setpoint_triplet.current.cruising_speed",
            ),
        ],
        exclusion_checks=[
            RelationshipCheckSpec(type="branch_parameter_satisfied", parameter="FW_ARSP_MODE", op="==", value=1),
        ],
    )
    inventory = {
        "duration_s": 10.0,
        "parameters": {"FW_ARSP_MODE": 1},
        "available_topics": ["airspeed_validated", "position_setpoint_triplet"],
        "topic_fields": {
            "airspeed_validated": ["true_airspeed_m_s"],
            "position_setpoint_triplet": ["current.cruising_speed"],
        },
    }

    first = compile_verification_plan(candidate, inventory, [], None)
    second = compile_verification_plan(candidate, inventory, [], None)

    assert first.mechanism_id == second.mechanism_id
    assert first.branches[0].branch_id == second.branches[0].branch_id
    assert [check.check_id for check in first.branches[0].checks] == [
        check.check_id for check in second.branches[0].checks
    ]
    assert [check.role for check in first.branches[0].checks] == [
        "evidence_availability",
        "mechanism_defining",
        "branch_applicability",
    ]


def test_resolved_candidate_predicate_signals_returns_arbitrary_logged_signal():
    candidate = MechanismCandidate(
        name="Custom branch",
        summary="Uses a source-derived custom branch signal.",
        source_refs=[],
        mode_state_gates=["custom_topic.branch_state == 1"],
    )

    signals = resolved_candidate_predicate_signals(
        [candidate],
        {
            "available_topics": ["custom_topic"],
            "topic_fields": {"custom_topic": ["branch_state"]},
        },
    )

    assert signals == ["custom_topic.branch_state"]


def test_verification_plan_resolves_source_predicate_and_intersects_windows(tmp_path):
    msg_dir = tmp_path / "msg"
    msg_dir.mkdir()
    (msg_dir / "VehicleStatus.msg").write_text(
        """
uint64 timestamp
uint8 nav_state
uint8 NAV_STATE_AUTO_MISSION = 3
uint8 NAV_STATE_POSCTL = 4
""",
        encoding="utf-8",
    )
    candidate = MechanismCandidate(
        name="Mission branch",
        summary="Only applies during mission.",
        source_refs=[],
        plot_requests=[PlotRef(title="Requested", purpose="user window", start_s=2.0, end_s=8.0)],
        branch_groups=[
            MechanismBranchGroup(
                name="mission",
                source_predicates=[
                    "_vehicle_status.nav_state == vehicle_status_s::NAV_STATE_AUTO_MISSION"
                ],
                numeric_checks=[
                    RelationshipCheckSpec(
                        type="threshold",
                        signal="vehicle_local_position.z",
                        metric="mean",
                        op=">=",
                        value=0,
                    )
                ],
            )
        ],
    )
    inventory = {
        "source_path": tmp_path,
        "available_topics": ["vehicle_status", "vehicle_local_position"],
        "topic_fields": {
            "vehicle_status": ["nav_state"],
            "vehicle_local_position": ["z"],
        },
    }
    timeline = [
        {"time_s": 0.0, "topic": "vehicle_status", "field": "nav_state", "value": 4},
        {"time_s": 3.0, "topic": "vehicle_status", "field": "nav_state", "value": 3},
        {"time_s": 7.0, "topic": "vehicle_status", "field": "nav_state", "value": 4},
        {"time_s": 10.0, "topic": "vehicle_status", "field": "nav_state", "value": 4},
    ]

    plan = compile_verification_plan(candidate, inventory, timeline, None)
    mission = next(branch for branch in plan.branches if branch.name == "mission")

    assert mission.resolved_predicates == [
        "vehicle_status.nav_state == vehicle_status_s::NAV_STATE_AUTO_MISSION"
    ]
    assert [(window.start_s, window.end_s) for window in mission.windows] == [(3.0, 7.0)]


def test_verification_plan_preserves_ambiguous_signal_as_unresolved():
    candidate = MechanismCandidate(
        name="Ambiguous output",
        summary="Uses an ambiguous source alias.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(type="threshold", signal="sp.value", metric="mean", op=">=", value=0),
        ],
    )
    bindings = [
        SourceOutputBindingRecord(
            binding_id="one",
            source_symbol="sp.value",
            target_symbol="first.value",
            logged_signal="topic_one.value",
        ),
        SourceOutputBindingRecord(
            binding_id="two",
            source_symbol="sp.value",
            target_symbol="second.value",
            logged_signal="topic_two.value",
        ),
    ]
    plan = compile_verification_plan(
        candidate,
        {
            "duration_s": 1.0,
            "available_topics": ["topic_one", "topic_two"],
            "topic_fields": {"topic_one": ["value"], "topic_two": ["value"]},
        },
        [],
        None,
        bindings,
    )

    check = plan.branches[0].checks[0]
    assert check.executable is False
    assert "ambiguous" in " ".join(check.unresolved_dependencies)
    applicability = applicability_from_verification_plan(candidate, plan, {
        "duration_s": 1.0,
        "available_topics": ["topic_one", "topic_two"],
        "topic_fields": {"topic_one": ["value"], "topic_two": ["value"]},
    })
    assert applicability.missing_required_signals == []


def test_verification_plan_does_not_create_ambiguity_from_field_suffixes():
    candidate = MechanismCandidate(
        name="Exact cruising speed",
        summary="Requires an exact source-to-log binding.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="threshold",
                signal="position_setpoint.cruising_speed",
                metric="mean",
                op=">=",
                value=0,
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "available_topics": ["position_setpoint_triplet"],
        "topic_fields": {
            "position_setpoint_triplet": [
                "previous.cruising_speed",
                "current.cruising_speed",
                "next.cruising_speed",
            ],
        },
    }

    plan = compile_verification_plan(candidate, inventory, [], None)

    check = plan.branches[0].checks[0]
    assert check.executable is False
    assert check.check.signal == "position_setpoint.cruising_speed"
    assert "ambiguous" not in " ".join(check.unresolved_dependencies)


def test_derived_expression_accepts_dotted_variable_via_alias_rewrite():
    """A check using ``tecs_status.true_airspeed_sp`` directly should be
    executable when the variable is declared in canonical dotted form."""
    candidate = MechanismCandidate(
        name="True airspeed scaling",
        summary="EAS-to-TAS scaling check.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="tecs_status.true_airspeed_sp",
                expected_expression="tecs_status.equivalent_airspeed_sp * eas2tas",
                variables=[
                    {"name": "tecs_status.true_airspeed_sp", "source": "tecs_status.true_airspeed_sp"},
                    {"name": "tecs_status.equivalent_airspeed_sp", "source": "tecs_status.equivalent_airspeed_sp"},
                    {"name": "eas2tas", "source": "1.05"},
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "available_topics": ["tecs_status"],
        "topic_fields": {"tecs_status": ["true_airspeed_sp", "equivalent_airspeed_sp"]},
    }

    plan = compile_verification_plan(candidate, inventory, [], None)

    check = plan.branches[0].checks[0]
    assert check.executable is True, check.unresolved_dependencies


def test_derived_expression_rejects_undeclared_dotted_reference():
    """A dotted reference not declared in variables remains an Attribute node
    after alias rewriting, so the AST whitelist correctly rejects it."""
    candidate = MechanismCandidate(
        name="Undeclared attribute",
        summary="Uses tecs_status.true_airspeed_sp without declaring it.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="tecs_status.true_airspeed_sp",
                expected_expression="",
                variables=[],
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "available_topics": ["tecs_status"],
        "topic_fields": {"tecs_status": ["true_airspeed_sp"]},
    }

    plan = compile_verification_plan(candidate, inventory, [], None)

    check = plan.branches[0].checks[0]
    assert check.executable is False
    assert any(
        "Attribute" in dep
        for dep in check.unresolved_dependencies
    ), check.unresolved_dependencies


def test_derived_expression_accepts_quaternion_paren_via_normalize_rewrite():
    """q(0) in the source-style expression should be normalized to q[0] and
    matched against a declared variable using the bracket form."""
    candidate = MechanismCandidate(
        name="Quaternion element check",
        summary="Vector element access via operator().",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="q(0)",
                expected_expression="1.0",
                variables=[
                    {"name": "q[0]", "source": "vehicle_attitude.q[0]"},
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "available_topics": ["vehicle_attitude"],
        "topic_fields": {"vehicle_attitude": ["q[0]"]},
    }

    plan = compile_verification_plan(candidate, inventory, [], None)

    check = plan.branches[0].checks[0]
    assert check.executable is True, check.unresolved_dependencies


def test_signal_resolver_prefers_primary_output_slice_to_break_suffix_collision():
    """The 6/14 cruising_speed gap: two bindings share the .cruising_speed
    suffix (current and previous), so a bare ``position_setpoint.cruising_speed``
    reference is ambiguous. When the candidate declares its primary output
    as tecs_status.equivalent_airspeed_sp and a binding chain shows that
    terminal is reached from current.cruising_speed (not previous), the
    SignalResolver should resolve uniquely to the current side."""
    from flight_log_agent.px4.source_mechanism_models import SourceOutputBindingRecord

    bindings = [
        SourceOutputBindingRecord(
            binding_id="b_terminal",
            logged_signal="tecs_status.equivalent_airspeed_sp",
            source_symbol="position_setpoint_triplet.current.cruising_speed",
            target_symbol="tecs_status.equivalent_airspeed_sp",
        ),
        SourceOutputBindingRecord(
            binding_id="b_current",
            logged_signal="position_setpoint_triplet.current.cruising_speed",
            source_symbol="vehicle_command.param2",
            target_symbol="position_setpoint_triplet.current.cruising_speed",
        ),
        SourceOutputBindingRecord(
            binding_id="b_previous",
            logged_signal="position_setpoint_triplet.previous.cruising_speed",
            source_symbol="position_setpoint_triplet.current.cruising_speed",
            target_symbol="position_setpoint_triplet.previous.cruising_speed",
        ),
    ]
    candidate = MechanismCandidate(
        name="DO_CHANGE_SPEED flow",
        summary="vehicle_command.param2 sets cruising_speed which becomes the equivalent airspeed sp.",
        primary_output_signals=["tecs_status.equivalent_airspeed_sp"],
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="threshold",
                signal="position_setpoint.cruising_speed",
                metric="mean",
                op=">",
                value=0,
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "available_topics": ["tecs_status", "position_setpoint_triplet", "vehicle_command"],
        "topic_fields": {
            "tecs_status": ["equivalent_airspeed_sp"],
            "position_setpoint_triplet": [
                "current.cruising_speed",
                "previous.cruising_speed",
            ],
            "vehicle_command": ["param2"],
        },
    }

    plan = compile_verification_plan(candidate, inventory, [], None, bindings)

    threshold_check = next(
        planned.check
        for planned in plan.branches[0].checks
        if planned.check.type == "threshold"
    )
    assert threshold_check.signal == "position_setpoint_triplet.current.cruising_speed"


def test_verification_plan_accepts_variable_load_context_in_derived_expression():
    candidate = MechanismCandidate(
        name="Derived expression",
        summary="Uses ordinary variable reads.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="actual",
                expected_expression="source_value + 1",
                variables=[
                    {"name": "actual", "source": "topic.actual"},
                    {"name": "source_value", "source": "topic.source_value"},
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "available_topics": ["topic"],
        "topic_fields": {"topic": ["actual", "source_value"]},
    }

    plan = compile_verification_plan(candidate, inventory, [], None)

    assert plan.branches[0].checks[0].executable is True


def test_custom_semantic_check_is_advisory_and_does_not_block_branch_verdict(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={"timestamp": [1_000_000], "z": [10.0]},
                )
            ]

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    candidate = MechanismCandidate(
        name="Advisory custom check",
        summary="Has executable evidence and an advisory semantic note.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="threshold",
                signal="vehicle_local_position.z",
                metric="mean",
                op=">=",
                value=5,
            ),
        ],
        exclusion_checks=[
            RelationshipCheckSpec(type="custom", description="Review this manually."),
        ],
    )
    inventory = {
        "duration_s": 2.0,
        "available_topics": ["vehicle_local_position"],
        "topic_fields": {"vehicle_local_position": ["z"]},
    }
    plan = compile_verification_plan(candidate, inventory, [], None)
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    evaluation = evaluate_candidate_log_signature(tmp_path / "flight.ulg", candidate, applicability, plan)

    custom = next(check for check in plan.branches[0].checks if check.check.type == "custom")
    assert custom.role == "advisory"
    assert custom.executable is False
    assert evaluation.verdict == "supported"


def test_signature_evaluation_executes_terminal_graph_and_uses_its_verdict(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_global_position",
                    multi_id=0,
                    data={"timestamp": [1_000_000, 2_000_000], "alt": [10.0, 11.0]},
                ),
                SimpleNamespace(
                    name="position_setpoint_triplet",
                    multi_id=0,
                    data={"timestamp": [1_000_000, 2_000_000], "current.alt": [10.0, 11.0]},
                ),
            ]

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    monkeypatch.setattr(log_evidence, "ULog", FakeULog)
    candidate = MechanismCandidate(
        name="Direct altitude publication",
        summary="Publishes the current altitude directly.",
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
    inventory = {
        "duration_s": 2.0,
        "available_topics": ["vehicle_global_position", "position_setpoint_triplet"],
        "topic_fields": {
            "vehicle_global_position": ["alt"],
            "position_setpoint_triplet": ["current.alt"],
        },
    }
    plan = compile_verification_plan(candidate, inventory, [], None, [binding])
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    evaluation = evaluate_candidate_log_signature(
        tmp_path / "flight.ulg",
        candidate,
        applicability,
        plan,
        output_bindings=[binding],
    )

    assert evaluation.verdict == "supported"
    assert evaluation.confidence_ceiling == "medium"
    assert evaluation.raw["verification_graphs"][0]["verdict"] == "supported"


def test_helper_substitution_clears_unresolved_when_profiler_lowered_the_body(tmp_path, monkeypatch):
    """A derived_expression check that lists a helper_dependency with an
    LLM-supplied unresolved_reason should be evaluable when the source
    profile actually contains a clean ``lowered_return_expression`` for
    that helper. The verifier substitutes the helper body into the
    expression and proceeds with normal evaluation."""

    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {"NAV_ACC_RAD": 10.0}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_global_position",
                    multi_id=0,
                    data={"timestamp": [1_000_000, 2_000_000], "alt": [20.0, 20.0]},
                ),
            ]

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    monkeypatch.setattr(log_evidence, "ULog", FakeULog)

    candidate = MechanismCandidate(
        name="Helper-lowering smoke",
        summary="Floors altitude at twice the acceptance radius.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="vehicle_global_position.alt",
                expected_expression="2.0 * get_acceptance_radius()",
                op=">=",
                max_error=1.0,
                variables=[
                    {"name": "vehicle_global_position.alt", "source": "vehicle_global_position.alt"},
                ],
                helper_dependencies=[
                    {
                        "name": "get_acceptance_radius",
                        "args": [],
                        "source_file": "src/modules/navigator/navigator_main.cpp",
                        "source_line": 100,
                        # LLM marks this conservatively; the profile knows better.
                        "unresolved_reason": "Later verification needs the exact default branch.",
                    }
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 2.0,
        "parameters": {"NAV_ACC_RAD": 10.0},
        "available_topics": ["vehicle_global_position"],
        "topic_fields": {"vehicle_global_position": ["alt"]},
    }
    # The profiler's lowered form for get_acceptance_radius() on the
    # rotary-wing default branch.
    helper_expressions = [
        {
            "name": "get_acceptance_radius",
            "file": "src/modules/navigator/navigator_main.cpp",
            "line": 100,
            "evidence": "",
            "parameters": [],
            "lowered_return_expression": "NAV_ACC_RAD",
            "branches": [],
            "unresolved_reason": None,
        }
    ]

    plan = compile_verification_plan(
        candidate,
        inventory,
        [],
        None,
        helper_expressions=helper_expressions,
    )
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    evaluation = evaluate_candidate_log_signature(
        tmp_path / "flight.ulg",
        candidate,
        applicability,
        plan,
        helper_expressions=helper_expressions,
    )

    # The check now evaluates instead of returning "cannot evaluate helper".
    assert evaluation.verdict == "supported"
    # No "cannot evaluate helper" message in the check_results.
    helper_failure_messages = [
        result.get("message", "")
        for result in evaluation.check_results
        if "cannot evaluate helper" in str(result.get("message", ""))
    ]
    assert helper_failure_messages == []


def test_helper_substitution_picks_branch_using_inventory_parameter(tmp_path, monkeypatch):
    """A branched helper whose condition references an inventory parameter
    should be resolved at plan-compile time: the verifier evaluates each
    condition against the known parameter values and substitutes the
    matching branch into the expression."""

    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {"NAV_ACC_RAD": 10.0, "RTL_CONE_ANG": 45}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_global_position",
                    multi_id=0,
                    data={"timestamp": [1_000_000, 2_000_000], "alt": [30.0, 30.0]},
                ),
            ]

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    monkeypatch.setattr(log_evidence, "ULog", FakeULog)

    # A helper with two branches. With RTL_CONE_ANG=45, the cone branch is
    # selected and ``acceptance_floor()`` lowers to ``2.0 * NAV_ACC_RAD``.
    helper_expressions = [
        {
            "name": "acceptance_floor",
            "file": "src/modules/navigator/rtl.cpp",
            "line": 720,
            "parameters": [],
            "branches": [
                {"condition": "RTL_CONE_ANG > 0", "expression": "2.0 * NAV_ACC_RAD"},
                {"condition": "default", "expression": "0.0"},
            ],
            "unresolved_reason": None,
        }
    ]

    candidate = MechanismCandidate(
        name="Branched helper smoke",
        summary="Floors altitude at the branch-resolved acceptance floor.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="vehicle_global_position.alt",
                expected_expression="acceptance_floor()",
                op=">=",
                max_error=1.0,
                variables=[
                    {"name": "vehicle_global_position.alt", "source": "vehicle_global_position.alt"},
                ],
                helper_dependencies=[
                    {
                        "name": "acceptance_floor",
                        "args": [],
                        "source_file": "src/modules/navigator/rtl.cpp",
                        "source_line": 720,
                        "unresolved_reason": "Acceptance-radius branch resolution remained unresolved.",
                    }
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 2.0,
        "parameters": {"NAV_ACC_RAD": 10.0, "RTL_CONE_ANG": 45},
        "available_topics": ["vehicle_global_position"],
        "topic_fields": {"vehicle_global_position": ["alt"]},
    }

    plan = compile_verification_plan(
        candidate, inventory, [], None, helper_expressions=helper_expressions,
    )
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    check = plan.branches[0].checks[0]
    assert check.executable is True, check.unresolved_dependencies

    evaluation = evaluate_candidate_log_signature(
        tmp_path / "flight.ulg",
        candidate,
        applicability,
        plan,
        helper_expressions=helper_expressions,
    )
    assert evaluation.verdict == "supported"


def test_slicer_resolves_unbound_symbol_from_source_assignments(tmp_path, monkeypatch):
    """A derived_expression check whose lowered body still references a
    class-member symbol like ``_destination.lat`` should become executable
    when source_assignments trace that symbol back to a logged signal."""

    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {}
            self.data_list = [
                SimpleNamespace(
                    name="position_setpoint_triplet",
                    multi_id=0,
                    data={
                        "timestamp": [1_000_000, 2_000_000],
                        "current.lat": [47.3977, 47.3978],
                    },
                ),
            ]

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    monkeypatch.setattr(log_evidence, "ULog", FakeULog)

    candidate = MechanismCandidate(
        name="Slicer smoke",
        summary="Compares destination latitude against the logged setpoint.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="position_setpoint_triplet.current.lat",
                expected_expression="_destination.lat",
                op="==",
                max_error=0.001,
                variables=[
                    {
                        "name": "position_setpoint_triplet.current.lat",
                        "source": "position_setpoint_triplet.current.lat",
                    },
                ],
                helper_dependencies=[],
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "available_topics": ["position_setpoint_triplet"],
        "topic_fields": {"position_setpoint_triplet": ["current.lat"]},
    }
    # source_assignments dict shape mirrors SourceAssignmentRef.model_dump.
    source_assignments = [
        {
            "target": "_destination.lat",
            "expression": "position_setpoint_triplet.current.lat",
            "control_predicates": [],
            "file": "src/modules/navigator/rtl.cpp",
            "line": 100,
            "evidence": "_destination.lat = position_setpoint_triplet.current.lat;",
            "function": "RTL::find_RTL_destination",
        },
    ]

    plan = compile_verification_plan(
        candidate,
        inventory,
        [],
        None,
        source_assignments=source_assignments,
    )

    check = plan.branches[0].checks[0]
    # The slicer rewrites the expression so the dotted symbol now refers
    # to the actually-logged signal.
    assert "position_setpoint_triplet.current.lat" in check.check.expected_expression
    assert check.executable is True, check.unresolved_dependencies


def test_slicer_attaches_warnings_on_partial_substitution(tmp_path):
    """When the slicer can resolve one branch but another dead-ends in an
    external call, the partial expression is kept executable and the
    unresolved branch is surfaced as a warning."""
    candidate = MechanismCandidate(
        name="Slicer partial smoke",
        summary="Conditional binding resolved on one branch.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="vehicle_global_position.alt",
                expected_expression="_destination.alt",
                variables=[
                    {
                        "name": "vehicle_global_position.alt",
                        "source": "vehicle_global_position.alt",
                    },
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "parameters": {"RTL_DESTINATION": 0},
        "available_topics": ["vehicle_global_position", "home_position"],
        "topic_fields": {
            "vehicle_global_position": ["alt"],
            "home_position": ["alt"],
        },
    }
    source_assignments = [
        {
            "target": "_destination.alt",
            "expression": "home_position.alt",
            "control_predicates": ["RTL_DESTINATION == 0"],
            "file": "src/modules/navigator/rtl.cpp",
            "line": 100,
            "evidence": "",
            "function": "RTL::find_RTL_destination",
        },
        {
            "target": "_destination.alt",
            "expression": "dm_read(DM_KEY_MISSION_LANDING, 0)",
            "control_predicates": ["RTL_DESTINATION == 1"],
            "file": "src/modules/navigator/rtl.cpp",
            "line": 110,
            "evidence": "",
            "function": "RTL::find_RTL_destination",
        },
    ]

    plan = compile_verification_plan(
        candidate, inventory, [], None,
        source_assignments=source_assignments,
    )

    check = plan.branches[0].checks[0]
    # Partial substitution keeps the check executable.
    assert check.executable is True
    # Warning carries actionable detail naming the specific symbol.
    assert any("destination.alt" in w for w in check.warnings), check.warnings
    # The rewritten expression is a ternary that inlines the resolved
    # branch (home_position.alt) under the RTL_DESTINATION == 0 condition.
    assert "home_position.alt" in check.check.expected_expression
    assert "RTL_DESTINATION" in check.check.expected_expression


def test_slicer_extends_variables_with_introduced_symbols(tmp_path):
    """Slicer-introduced logged signals / parameters get appended to the
    check's variables[] so the runtime expression context binds them."""
    candidate = MechanismCandidate(
        name="Variables extension smoke",
        summary="",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="vehicle_global_position.alt",
                expected_expression="_destination.alt",
                variables=[
                    {
                        "name": "vehicle_global_position.alt",
                        "source": "vehicle_global_position.alt",
                    },
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "parameters": {"RTL_DESTINATION": 0},
        "available_topics": ["vehicle_global_position", "home_position"],
        "topic_fields": {
            "vehicle_global_position": ["alt"],
            "home_position": ["alt"],
        },
    }
    source_assignments = [
        {
            "target": "_destination.alt",
            "expression": "home_position.alt",
            "control_predicates": ["RTL_DESTINATION == 0"],
            "file": "x.cpp", "line": 1, "evidence": "", "function": "f",
        },
    ]

    plan = compile_verification_plan(
        candidate, inventory, [], None,
        source_assignments=source_assignments,
    )

    variable_names = {
        (v.get("name") if isinstance(v, dict) else v.name)
        for v in plan.branches[0].checks[0].check.variables
    }
    # Logged signal introduced by the slicer is declared so the validator
    # accepts the rewritten Attribute reference. (Bare parameter names
    # like RTL_DESTINATION don't need variables[] entries — the runtime
    # expression context auto-resolves them from inventory.parameters.)
    assert "home_position.alt" in variable_names
    # The fallback symbol for the partially-resolved branch is also
    # declared so the validator's Attribute check passes.
    assert "_destination.alt" in variable_names or "destination.alt" in variable_names


def test_helper_substitution_does_not_resolve_when_branch_condition_fails(tmp_path):
    """When a branched helper's conditions all fail against the static
    parameter env and there is no default branch, the check stays
    unresolved at compile time."""

    helper_expressions = [
        {
            "name": "f",
            "file": "fake.cpp",
            "line": 1,
            "parameters": [],
            "branches": [
                {"condition": "MISSING_PARAM > 0", "expression": "1.0"},
            ],
            "unresolved_reason": None,
        }
    ]
    candidate = MechanismCandidate(
        name="No matching branch",
        summary="",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                expression="vehicle_global_position.alt",
                expected_expression="f()",
                variables=[
                    {"name": "vehicle_global_position.alt", "source": "vehicle_global_position.alt"},
                ],
                helper_dependencies=[
                    {
                        "name": "f",
                        "args": [],
                        "unresolved_reason": "Branch not statically resolvable.",
                    }
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "parameters": {},
        "available_topics": ["vehicle_global_position"],
        "topic_fields": {"vehicle_global_position": ["alt"]},
    }
    plan = compile_verification_plan(
        candidate, inventory, [], None, helper_expressions=helper_expressions,
    )
    check = plan.branches[0].checks[0]
    assert check.executable is False
    assert any("helper f" in dep for dep in check.unresolved_dependencies), check.unresolved_dependencies


def test_invalid_source_filename_presence_check_is_ignored():
    candidate = MechanismCandidate(
        name="Malformed source check",
        summary="A source filename is not a ULog signal.",
        source_refs=[],
        numeric_checks=[
            RelationshipCheckSpec(type="topic_field_present", signal="rtl.cpp"),
            RelationshipCheckSpec(type="topic_field_present", signal="vehicle_status.nav_state"),
        ],
    )
    inventory = {
        "duration_s": 1.0,
        "available_topics": ["vehicle_status"],
        "topic_fields": {"vehicle_status": ["nav_state"]},
    }

    plan = compile_verification_plan(candidate, inventory, [], None)

    assert [check.check.signal for check in plan.branches[0].checks] == ["vehicle_status.nav_state"]


def test_branch_parameter_fallback_replaces_none_value_and_operator():
    result = signature_evaluator._check_branch_parameter_satisfied(
        {},
        {},
        {"FW_AIRSPD_MAX": 25.0},
        {
            "type": "branch_parameter_satisfied",
            "parameter": "FW_AIRSPD_MAX",
            "source_predicate": "FW_AIRSPD_MAX > 20",
            "op": None,
            "value": None,
        },
    )

    assert result["status"] == "passed"
    assert result["value"]["op"] == ">"
    assert result["value"]["expected"] == 20


def test_presence_check_cannot_support_mechanism_verdict(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={"timestamp": [1_000_000], "z": [10.0]},
                )
            ]

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    candidate = MechanismCandidate(
        name="Presence only",
        summary="Has availability evidence but no defining check.",
        source_refs=[],
        required_signals=["vehicle_local_position.z"],
        numeric_checks=[
            RelationshipCheckSpec(type="topic_field_present", signal="vehicle_local_position.z"),
        ],
    )
    inventory = {
        "duration_s": 2.0,
        "available_topics": ["vehicle_local_position"],
        "topic_fields": {"vehicle_local_position": ["z"]},
    }
    plan = compile_verification_plan(candidate, inventory, [], None)
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    evaluation = evaluate_candidate_log_signature(tmp_path / "flight.ulg", candidate, applicability, plan)

    assert evaluation.verdict == "unresolved"
    assert evaluation.evidence == []
    assert evaluation.check_results[0]["role"] == "evidence_availability"


def test_descriptive_expected_signature_does_not_create_required_unresolved_check(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {}
            self.data_list = []

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    candidate = MechanismCandidate(
        name="Descriptive requirement",
        summary="Has a formula that is not deterministically executable.",
        source_refs=[],
        expected_logged_signature=[
            {
                "name": "semantic_formula",
                "description": "The selected altitude should follow the complete source mechanism.",
            }
        ],
    )
    inventory = {"duration_s": 2.0}
    plan = compile_verification_plan(candidate, inventory, [], None)
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    evaluation = evaluate_candidate_log_signature(tmp_path / "flight.ulg", candidate, applicability, plan)

    assert evaluation.verdict == "unresolved"
    assert plan.branches[0].checks == []


def test_branch_results_are_evaluated_independently_and_aggregated(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={"timestamp": [1_000_000], "z": [10.0]},
                )
            ]

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    candidate = MechanismCandidate(
        name="Independent branches",
        summary="One branch passes and another fails.",
        source_refs=[],
        branch_groups=[
            MechanismBranchGroup(
                name="passing",
                numeric_checks=[
                    RelationshipCheckSpec(
                        type="threshold",
                        signal="vehicle_local_position.z",
                        metric="mean",
                        op=">=",
                        value=5,
                    )
                ],
            ),
            MechanismBranchGroup(
                name="failing",
                numeric_checks=[
                    RelationshipCheckSpec(
                        type="threshold",
                        signal="vehicle_local_position.z",
                        metric="mean",
                        op="<",
                        value=5,
                    )
                ],
            ),
        ],
    )
    inventory = {
        "duration_s": 2.0,
        "available_topics": ["vehicle_local_position"],
        "topic_fields": {"vehicle_local_position": ["z"]},
    }
    plan = compile_verification_plan(candidate, inventory, [], None)
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    evaluation = evaluate_candidate_log_signature(tmp_path / "flight.ulg", candidate, applicability, plan)

    assert evaluation.verdict == "mixed"
    branch_verdicts = {
        result["name"]: result["verdict"]
        for result in evaluation.raw["branch_results"]
    }
    assert branch_verdicts["passing"] == "supported"
    assert branch_verdicts["failing"] == "contradicted"


def test_excluded_branch_defining_check_does_not_leak_into_evidence(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {"BRANCH_MODE": 0}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={"timestamp": [1_000_000], "z": [10.0]},
                )
            ]

    monkeypatch.setattr(signature_evaluator, "ULog", FakeULog)
    candidate = MechanismCandidate(
        name="Excluded branch",
        summary="A defining check passes in a branch whose applicability check fails.",
        source_refs=[],
        branch_groups=[
            MechanismBranchGroup(
                name="disabled",
                numeric_checks=[
                    RelationshipCheckSpec(
                        type="threshold",
                        signal="vehicle_local_position.z",
                        metric="mean",
                        op=">=",
                        value=5,
                        supports="Disabled branch defining check passed.",
                    )
                ],
                exclusion_checks=[
                    RelationshipCheckSpec(
                        type="branch_parameter_satisfied",
                        parameter="BRANCH_MODE",
                        op="==",
                        value=1,
                    )
                ],
            )
        ],
    )
    inventory = {
        "duration_s": 2.0,
        "parameters": {"BRANCH_MODE": 0},
        "available_topics": ["vehicle_local_position"],
        "topic_fields": {"vehicle_local_position": ["z"]},
    }
    plan = compile_verification_plan(candidate, inventory, [], None)
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    evaluation = evaluate_candidate_log_signature(tmp_path / "flight.ulg", candidate, applicability, plan)

    assert evaluation.verdict == "unresolved"
    assert evaluation.evidence == []


def test_all_real_branches_excluded_makes_mechanism_inapplicable():
    candidate = MechanismCandidate(
        name="No applicable branches",
        summary="Every explicit branch is contradicted.",
        source_refs=[],
        branch_groups=[
            MechanismBranchGroup(
                name="disabled",
                source_predicates=["BRANCH_MODE == 1"],
                numeric_checks=[
                    RelationshipCheckSpec(
                        type="threshold",
                        signal="vehicle_local_position.z",
                        metric="mean",
                        op=">=",
                        value=0,
                    )
                ],
            )
        ],
    )
    inventory = {
        "duration_s": 2.0,
        "parameters": {"BRANCH_MODE": 0},
        "available_topics": ["vehicle_local_position"],
        "topic_fields": {"vehicle_local_position": ["z"]},
    }

    plan = compile_verification_plan(candidate, inventory, [], None)
    applicability = applicability_from_verification_plan(candidate, plan, inventory)

    assert len(plan.branches) == 1
    assert plan.branches[0].applicable is False
    assert applicability.applicable is False
