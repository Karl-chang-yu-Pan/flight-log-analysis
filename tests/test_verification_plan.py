from types import SimpleNamespace

from flight_log_agent.analysis.signature_verification import evaluate_candidate_log_signature
from flight_log_agent.analysis.verification_plan import (
    applicability_from_verification_plan,
    compile_verification_plan,
)
from flight_log_agent.models import (
    MechanismBranchGroup,
    MechanismCandidate,
    PlotRef,
    RelationshipCheckSpec,
)
from flight_log_agent.px4.source_mechanism_models import SourceOutputBindingRecord
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
