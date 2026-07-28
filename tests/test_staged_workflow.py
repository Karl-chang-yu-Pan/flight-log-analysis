import json
from pathlib import Path
from types import SimpleNamespace

from agents import ShellResult, ShellTool, Usage
import pytest

import flight_log_agent.analysis.staged_workflow as workflow
from flight_log_agent.analysis.report_validation import (
    enforce_validation_downgrades,
    validate_report,
)
from flight_log_agent.audit import DeveloperAuditLogger
from flight_log_agent.models import (
    ApplicabilityReport,
    CodeRef,
    ExpectedSignatureItem,
    FlightLogReport,
    HypothesisReportItem,
    PlotRef,
    RelationshipCheckSpec,
)

SOURCE_COMMAND = (
    "git -C PX4-Autopilot show SNAPSHOT:src/module.cpp"
)
DECLARATION_COMMAND = (
    "git -C PX4-Autopilot show SNAPSHOT:src/binding.cpp"
)
SCHEMA_COMMAND = (
    "git -C PX4-Autopilot show SNAPSHOT:msg/Output.msg"
)
TARGET_COMMAND = "python check_candidate.py"
REVIEW_COMMAND = "python review_candidate.py"
INITIAL_COMMAND = "python inspect_log.py"
INITIAL_FINDING = "output.value changed"
INITIAL_EVIDENCE = "output.value=3 at 5 s"
INITIAL_TOPIC_FIELDS = ["output.value"]
INITIAL_STDOUT = workflow._receipt_json_marker(
    "observation",
    {
        "observation_index": 0,
        "question_focus": "output.value",
        "finding": INITIAL_FINDING,
        "evidence": INITIAL_EVIDENCE,
        "topic_fields": INITIAL_TOPIC_FIELDS,
        "time_or_window": "5 s",
    },
)
SOURCE_STDOUT = (
    "transformed = input * scale\nmessage.value = transformed\n"
    "published as output.value"
)
DECLARATION_STDOUT = "output_s message{}"
SCHEMA_STDOUT = "float32 value"
TARGET_RESULT = "maximum error 0.01"
TARGET_TOPIC_FIELDS = ["input.value", "output.value"]
DEFAULT_REQUIREMENT_ID = workflow._runtime_requirement_id(
    "c1_1",
    "scale is active",
)
TARGET_STDOUT = workflow._receipt_json_marker(
    "check",
    {
        "check_id": "check_1",
        "candidate_id": "c1_1",
        "evidence_role": "causal_discriminator",
        "window": "4-6 s",
        "topic_fields": TARGET_TOPIC_FIELDS,
        "requirement_ids": [DEFAULT_REQUIREMENT_ID],
        "parameters": [["SCALE", "2"]],
        "result": TARGET_RESULT,
        "assessment": "supports",
    },
)
REVIEW_RESULT = "Independent same-window replay matched."
REVIEW_STDOUT = workflow._receipt_json_marker(
    "reviewed_check",
    {
        "check_id": "check_1",
        "independent_result": REVIEW_RESULT,
        "assessment": "supports",
        "window": "4-6 s",
        "topic_fields": TARGET_TOPIC_FIELDS,
        "requirement_ids": [DEFAULT_REQUIREMENT_ID],
    },
)


def _initial() -> workflow.InitialInspection:
    return workflow.InitialInspection(
        question_is_causal=True,
        question_focus="output.value",
        firmware_and_version="test",
        log_duration_and_timebase="10 s",
        observations=[
            workflow.LogObservation(
                finding=INITIAL_FINDING,
                evidence=INITIAL_EVIDENCE,
                topic_fields=INITIAL_TOPIC_FIELDS,
                time_or_window="5 s",
            )
        ],
        relevant_topic_fields=["output.value", "input.value"],
        likely_windows=["4-6 s"],
        source_search_terms=["output.value"],
        execution_receipts=[
            workflow.ShellExecutionReceipt(
                command=INITIAL_COMMAND,
                exit_code=0,
                stdout=INITIAL_STDOUT,
            )
        ],
    )


def _candidate(
    candidate_id: str = "c1_1",
    title: str = "Upstream transform",
) -> workflow.CandidateMechanism:
    return workflow.CandidateMechanism(
        candidate_id=candidate_id,
        title=title,
        mechanism="An upstream transform changes the value before publication.",
        explained_logged_value="output.value",
        upstream_assignment_path=[
            workflow.SourceLineageStep(
                sequence=1,
                file="src/module.cpp",
                symbol="update",
                lines="1",
                operation="transformed = input * scale",
                source_excerpt="transformed = input * scale",
                input_or_state="input, scale",
                output_or_effect="transformed",
                role="transformation",
                execution_commands=[SOURCE_COMMAND],
            ),
            workflow.SourceLineageStep(
                sequence=2,
                file="src/module.cpp",
                symbol="publish",
                lines="2",
                operation="message.value = transformed",
                source_excerpt="message.value = transformed",
                input_or_state="transformed",
                output_or_effect="message value",
                role="publication",
                execution_commands=[SOURCE_COMMAND],
            ),
        ],
        causal_transformation="transformed = input * scale",
        downstream_propagation=["consumer.value copies output.value"],
        runtime_conditions=["scale is active"],
        alternatives_considered=["consumer creates the value"],
        expected_log_signature=["input * scale matches output.value"],
        contradicting_log_signature=["input * scale does not match output.value"],
        required_topic_fields=["input.value", "output.value"],
        required_parameters=["SCALE"],
        discriminating_calculations=["Replay input * SCALE in the same window"],
    )


def _source(
    candidate_id: str = "c1_1",
    title: str = "Upstream transform",
) -> workflow.SourceInvestigation:
    candidate = _candidate(candidate_id, title)
    return workflow.SourceInvestigation(
        source_available=True,
        snapshot_identity="repo@abc",
        explained_logged_value="output.value",
        publishing_assignment=candidate.upstream_assignment_path[-1],
        publication_binding=workflow.PublicationBinding(
            logged_value="output.value",
            message_identifier="message",
            message_type_identifier="output_s",
            declaration_file="src/binding.cpp",
            declaration_lines="1",
            declaration_excerpt=DECLARATION_STDOUT,
            declaration_execution_commands=[DECLARATION_COMMAND],
            schema_file="msg/Output.msg",
            schema_lines="1",
            schema_excerpt=SCHEMA_STDOUT,
            schema_execution_commands=[SCHEMA_COMMAND],
        ),
        candidates=[candidate],
        execution_receipts=[
            workflow.ShellExecutionReceipt(
                command=SOURCE_COMMAND,
                exit_code=0,
                stdout=SOURCE_STDOUT,
            ),
            workflow.ShellExecutionReceipt(
                command=DECLARATION_COMMAND,
                exit_code=0,
                stdout=DECLARATION_STDOUT,
            ),
            workflow.ShellExecutionReceipt(
                command=SCHEMA_COMMAND,
                exit_code=0,
                stdout=SCHEMA_STDOUT,
            ),
        ],
    )


def _check(
    *,
    candidate_id: str = "c1_1",
    role: str = "causal_discriminator",
    assessment: str = "supports",
    suffix: str = "1",
) -> workflow.EvidenceCheck:
    return workflow.EvidenceCheck(
        check_id=f"check_{suffix}",
        candidate_id=candidate_id,
        evidence_role=role,
        topic_fields=["input.value", "output.value"],
        window="4-6 s",
        source_prediction="output = input * scale",
        method="Replay the source expression on aligned samples",
        result="maximum error 0.01",
        evaluated_requirement_ids=[
            workflow._runtime_requirement_id(
                candidate_id,
                "scale is active",
            )
        ],
        execution_commands=[TARGET_COMMAND],
        assessment=assessment,
    )


def _targeted(
    *,
    candidate_id: str = "c1_1",
    role: str = "causal_discriminator",
) -> workflow.TargetedLogParse:
    targeted = workflow.TargetedLogParse(
        explained_logged_value="output.value",
        event_windows=["4-6 s"],
        parameter_values=[workflow.NamedValue(name="SCALE", value="2")],
        checks=[_check(candidate_id=candidate_id, role=role)],
    )
    targeted.execution_receipts = [
        workflow.ShellExecutionReceipt(
            command=TARGET_COMMAND,
            exit_code=0,
            stdout=workflow._receipt_json_marker(
                "check",
                workflow._targeted_check_receipt_payload(
                    targeted.checks[0],
                    _candidate(candidate_id),
                    targeted,
                ),
            ),
        )
    ]
    return targeted


def _review(
    *,
    candidate_id: str = "c1_1",
    web_queries: list[str] | None = None,
) -> workflow.EvidenceReview:
    review = workflow.EvidenceReview(
        question_is_causal=True,
        explained_logged_value="output.value",
        local_observations_sufficient=not web_queries,
        reviewed_publication_binding=(
            workflow.ReviewedPublicationBinding(
                **_source().publication_binding.model_dump(),
                assessment="verified",
                reasoning=(
                    "The exact declaration and schema bind output.value."
                ),
            )
        ),
        candidate_reviews=[
            workflow.CandidateReview(
                candidate_id=candidate_id,
                verdict="supported",
                confidence="high",
                source_path_complete=True,
                source_verification_commands=[SOURCE_COMMAND],
                reviewed_source_lineage=[
                    workflow.ReviewedSourceLineageStep(
                        sequence=1,
                        file="src/module.cpp",
                        source_excerpt="transformed = input * scale",
                        input_tokens=["input", "scale"],
                        output_tokens=["transformed"],
                        direction_assessment="verified",
                        reasoning=(
                            "The right-hand inputs produce transformed."
                        ),
                    ),
                    workflow.ReviewedSourceLineageStep(
                        sequence=2,
                        file="src/module.cpp",
                        source_excerpt="message.value = transformed",
                        input_tokens=["transformed"],
                        output_tokens=["message", "value"],
                        direction_assessment="verified",
                        reasoning=(
                            "The transformed value is assigned for publication."
                        ),
                    ),
                ],
                reviewed_checks=[
                    workflow.ReviewedEvidenceCheck(
                        check_id="check_1",
                        assessment="supports",
                        independent_result=(
                            REVIEW_RESULT
                        ),
                        execution_commands=[REVIEW_COMMAND],
                    )
                ],
                supporting_evidence=["The replay matches."],
                reasoning="The upstream transform is source ordered and replayed.",
            )
        ],
        preferred_candidate_id=candidate_id,
        requested_state=(
            "needs_web_research" if web_queries else "sufficient"
        ),
        reason="External context is needed." if web_queries else "Evidence is sufficient.",
        web_gap_kind="external_context" if web_queries else "none",
        web_queries=web_queries or [],
        execution_receipts=[
            workflow.ShellExecutionReceipt(
                command=SOURCE_COMMAND,
                exit_code=0,
                stdout=SOURCE_STDOUT,
            ),
            workflow.ShellExecutionReceipt(
                command=REVIEW_COMMAND,
                exit_code=0,
                stdout="",
            ),
            workflow.ShellExecutionReceipt(
                command=DECLARATION_COMMAND,
                exit_code=0,
                stdout=DECLARATION_STDOUT,
            ),
            workflow.ShellExecutionReceipt(
                command=SCHEMA_COMMAND,
                exit_code=0,
                stdout=SCHEMA_STDOUT,
            ),
        ],
    )
    reviewed_check = review.candidate_reviews[0].reviewed_checks[0]
    review.execution_receipts[1].stdout = workflow._receipt_json_marker(
        "reviewed_check",
        workflow._reviewed_check_receipt_payload(
            reviewed_check,
            _check(candidate_id=candidate_id),
        ),
    )
    return review


def _nested_evidence_inputs(
    *,
    focus="position_setpoint_triplet.current.valid",
    root_schema_file="msg/PositionSetpointTriplet.msg",
    nested_schema_file="msg/PositionSetpoint.msg",
    message_type_identifier="position_setpoint_triplet_s",
    root_schema="PositionSetpoint current",
    nested_schema="bool valid",
    publishing_source="message.current.valid = transformed",
):
    root_schema_command = (
        "git -C PX4-Autopilot show "
        f"SNAPSHOT:{root_schema_file}"
    )
    nested_schema_command = (
        "git -C PX4-Autopilot show "
        f"SNAPSHOT:{nested_schema_file}"
    )
    declaration = f"{message_type_identifier} message{{}}"

    initial = _initial()
    initial.question_focus = focus
    observation = initial.observations[0]
    observation.finding = f"{focus} changed"
    observation.evidence = f"{focus}=true at 5 s"
    observation.topic_fields = [focus]
    initial.relevant_topic_fields = [focus, "input.value"]
    initial.source_search_terms = [focus]
    initial.execution_receipts[0].stdout = workflow._receipt_json_marker(
        "observation",
        workflow._observation_receipt_payload(
            0,
            observation,
            question_focus=focus,
        ),
    )

    source = _source()
    source.explained_logged_value = focus
    candidate = source.candidates[0]
    candidate.explained_logged_value = focus
    candidate.required_topic_fields = ["input.value", focus]
    publishing = candidate.upstream_assignment_path[-1]
    publishing.operation = publishing_source
    publishing.source_excerpt = publishing_source
    publishing_binding = workflow._publishing_lhs_binding(publishing)
    assert publishing_binding is not None
    publishing.output_or_effect = " ".join(
        [publishing_binding[0], *publishing_binding[1]]
    )
    source.publishing_assignment = publishing
    binding = source.publication_binding
    assert binding is not None
    binding.logged_value = focus
    binding.message_type_identifier = message_type_identifier
    binding.declaration_excerpt = declaration
    binding.schema_file = root_schema_file
    binding.schema_excerpt = root_schema
    binding.schema_execution_commands = [root_schema_command]
    binding.nested_schema_bindings = [
        workflow.SchemaSourceBinding(
            schema_file=nested_schema_file,
            schema_lines="1",
            schema_excerpt=nested_schema,
            schema_execution_commands=[nested_schema_command],
        )
    ]
    source.execution_receipts[0].stdout = SOURCE_STDOUT.replace(
        "message.value = transformed",
        publishing_source,
    )
    source.execution_receipts[1].stdout = declaration
    source.execution_receipts[2].command = root_schema_command
    source.execution_receipts[2].stdout = root_schema
    source.execution_receipts.append(
        workflow.ShellExecutionReceipt(
            command=nested_schema_command,
            exit_code=0,
            stdout=nested_schema,
        )
    )

    targeted = _targeted()
    targeted.explained_logged_value = focus
    targeted.checks[0].topic_fields = ["input.value", focus]
    targeted.execution_receipts[0].stdout = (
        workflow._receipt_json_marker(
            "check",
            workflow._targeted_check_receipt_payload(
                targeted.checks[0],
                candidate,
                targeted,
            ),
        )
    )

    review = _review()
    review.explained_logged_value = focus
    review.reviewed_publication_binding = (
        workflow.ReviewedPublicationBinding(
            **binding.model_dump(),
            assessment="verified",
            reasoning=(
                "The root and nested schemas bind the flattened field."
            ),
        )
    )
    reviewed_publishing = (
        review.candidate_reviews[0].reviewed_source_lineage[-1]
    )
    reviewed_publishing.source_excerpt = publishing_source
    reviewed_publishing.output_tokens = [
        publishing_binding[0],
        *publishing_binding[1],
    ]
    review.execution_receipts[0].stdout = (
        source.execution_receipts[0].stdout
    )
    review.execution_receipts[2].stdout = declaration
    review.execution_receipts[3].command = root_schema_command
    review.execution_receipts[3].stdout = root_schema
    review.execution_receipts.append(
        workflow.ShellExecutionReceipt(
            command=nested_schema_command,
            exit_code=0,
            stdout=nested_schema,
        )
    )
    reviewed_check = review.candidate_reviews[0].reviewed_checks[0]
    review.execution_receipts[1].stdout = (
        workflow._receipt_json_marker(
            "reviewed_check",
            workflow._reviewed_check_receipt_payload(
                reviewed_check,
                targeted.checks[0],
            ),
        )
    )
    return initial, source, targeted, review


def _designated_initializer_evidence_inputs():
    declaration = "output_s message {\n    .value = transformed,"
    initializer = ".value = transformed,"
    source = _source()
    candidate = source.candidates[0]
    publishing = candidate.upstream_assignment_path[-1]
    publishing.file = "src/binding.cpp"
    publishing.lines = "2"
    publishing.operation = initializer
    publishing.source_excerpt = initializer
    publishing.input_or_state = "transformed"
    publishing.output_or_effect = "value"
    publishing.execution_commands = [DECLARATION_COMMAND]
    source.publishing_assignment = publishing
    binding = source.publication_binding
    assert binding is not None
    binding.declaration_lines = "1-2"
    binding.declaration_excerpt = declaration
    source.execution_receipts[1].stdout = declaration

    review = _review()
    reviewed_binding = review.reviewed_publication_binding
    assert reviewed_binding is not None
    reviewed_binding.declaration_lines = "1-2"
    reviewed_binding.declaration_excerpt = declaration
    candidate_review = review.candidate_reviews[0]
    candidate_review.source_verification_commands.append(
        DECLARATION_COMMAND
    )
    reviewed_publishing = candidate_review.reviewed_source_lineage[-1]
    reviewed_publishing.file = "src/binding.cpp"
    reviewed_publishing.source_excerpt = initializer
    reviewed_publishing.input_tokens = ["transformed"]
    reviewed_publishing.output_tokens = ["value"]
    review.execution_receipts[2].stdout = declaration
    return _initial(), source, _targeted(), review


def _observational_review(
    initial: workflow.InitialInspection,
    *,
    independent_result: str | None = None,
) -> workflow.EvidenceReview:
    result = independent_result or (
        f"Independent observation: {initial.observations[0].evidence}"
    )
    reviewed_observation = workflow.ReviewedObservation(
        observation_index=0,
        assessment="supports",
        independent_result=result,
        execution_commands=[REVIEW_COMMAND],
    )
    return workflow.EvidenceReview(
        question_is_causal=False,
        explained_logged_value=initial.question_focus,
        local_observations_sufficient=True,
        reviewed_observations=[reviewed_observation],
        candidate_reviews=[],
        requested_state="sufficient",
        reason="The requested observation was read directly from the log.",
        execution_receipts=[
            workflow.ShellExecutionReceipt(
                command=REVIEW_COMMAND,
                exit_code=0,
                stdout=workflow._receipt_json_marker(
                    "reviewed_observation",
                    workflow._reviewed_observation_receipt_payload(
                        reviewed_observation,
                        initial.observations[0],
                        question_focus=initial.question_focus,
                    ),
                ),
            )
        ],
    )


def _report(title: str = "Upstream transform") -> FlightLogReport:
    hypothesis = HypothesisReportItem(
        title=title,
        known_px4_mechanism="Source-derived transform",
        mechanism="The upstream transform created the logged deviation.",
        source_refs=[
            CodeRef(
                file="src/module.cpp",
                function="update",
                start_line=10,
                end_line=21,
                snippet="transformed = input * scale",
                explanation="Defines and publishes the transformed value.",
            )
        ],
        expected_logged_signature=[
            ExpectedSignatureItem(
                name="expression replay",
                description="The source expression matches the output.",
                signal="output.value",
                expected_behavior="input * scale",
            )
        ],
        applicability=ApplicabilityReport(
            applicable=True,
            supported_conditions=["scale is active"],
            available_required_signals=["input.value", "output.value"],
        ),
        evidence=["Same-window replay matched."],
        contradicting_evidence=[],
        exclusion_checks=[],
        numeric_checks=[
            RelationshipCheckSpec(
                type="derived_expression",
                window="4-6 s",
                actual="output.value",
                expression="input.value * SCALE",
                supports="Upstream transform",
            )
        ],
        confidence="high",
    )
    return FlightLogReport(
        airframe_summary="test",
        question_intent_summary="explain output.value",
        ranked_hypotheses=[hypothesis],
        excluded_mechanisms=[],
        confirmed=[title],
        unconfirmed=[],
        final_summary="The upstream transform caused the value.",
    )


def test_evidence_gate_rejects_downstream_consistency_as_causal_proof():
    source = _source()
    targeted = _targeted(role="downstream_consistency")
    review = _review(
        web_queries=["PX4 output.value source history"],
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=True,
    )

    assert state.status == "needs_web_research"
    assert state.accepted_candidate_ids == []
    assert state.web_queries == ["PX4 output.value source history"]


def test_evidence_gate_accepts_complete_upstream_discriminator():
    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=True,
    )

    assert state.status == "sufficient"
    assert state.accepted_candidate_ids == ["c1_1"]
    assert state.accepted_candidate_titles == [
        workflow._candidate_public_title(_source().candidates[0])
    ]


def test_source_commands_use_px4_root_relative_submodule_paths():
    command = (
        "git -C PX4-Autopilot/modules/vendor "
        "show SNAPSHOT:src/module.cpp"
    )

    assert workflow._source_command_reads_file(
        command,
        "modules/vendor/src/module.cpp",
    )
    assert not workflow._source_command_reads_file(
        command,
        "src/module.cpp",
    )
    assert workflow._source_command_reads_file(
        SOURCE_COMMAND,
        "src/module.cpp",
    )


def test_evidence_gate_rejects_an_invented_lineage_bridge():
    source = _source()
    lineage = source.candidates[0].upstream_assignment_path
    lineage[0].output_or_effect = "fictional_bridge_value"
    lineage[1].input_or_state = "fictional_bridge_value"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "disconnected or reversed data flow" in issue
        or "operation or data flow is not grounded" in issue
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_a_reversed_source_lineage():
    source = _source()
    producer, publisher = source.candidates[0].upstream_assignment_path
    publisher.sequence = 1
    publisher.role = "transformation"
    publisher.input_or_state = "message.value"
    publisher.output_or_effect = "transformed"
    producer.sequence = 2
    producer.role = "publication"
    producer.input_or_state = "transformed input"
    producer.output_or_effect = "output.value transformed"
    source.candidates[0].upstream_assignment_path = [publisher, producer]
    source.publishing_assignment = producer

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "data-flow direction" in issue
        for issue in state.validation_issues
    )


def test_assignment_direction_rejects_a_mirrored_reversed_review():
    source = _source()
    producer, publisher = source.candidates[0].upstream_assignment_path
    publisher.sequence = 1
    publisher.role = "transformation"
    publisher.input_or_state = "message.value"
    publisher.output_or_effect = "transformed"
    producer.sequence = 2
    producer.role = "publication"
    producer.input_or_state = "transformed input"
    producer.output_or_effect = "output.value transformed"
    source.candidates[0].upstream_assignment_path = [publisher, producer]
    source.publishing_assignment = producer
    review = _review()
    review.candidate_reviews[0].reviewed_source_lineage = [
        workflow.ReviewedSourceLineageStep(
            sequence=1,
            file="src/module.cpp",
            source_excerpt="message.value = transformed",
            input_tokens=["message", "value"],
            output_tokens=["transformed"],
            direction_assessment="verified",
            reasoning="Mirrors the proposed direction.",
        ),
        workflow.ReviewedSourceLineageStep(
            sequence=2,
            file="src/module.cpp",
            source_excerpt="transformed = input * scale",
            input_tokens=["transformed", "input"],
            output_tokens=["transformed"],
            direction_assessment="verified",
            reasoning="Mirrors the proposed direction.",
        ),
    ]

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "reverses the source assignment direction" in issue
        for issue in state.validation_issues
    )


def test_source_grounding_accepts_a_literal_origin():
    step = workflow.SourceLineageStep(
        sequence=1,
        file="src/module.cpp",
        lines="1",
        operation="result = 0",
        source_excerpt="result = 0",
        input_or_state="0",
        output_or_effect="result",
        role="origin",
    )

    assert workflow._source_step_fields_are_grounded(step)


def test_source_grounding_rejects_an_invented_origin_input():
    step = workflow.SourceLineageStep(
        sequence=1,
        file="src/module.cpp",
        lines="1",
        operation="result = 0",
        source_excerpt="result = 0",
        input_or_state="fictional_input",
        output_or_effect="result",
        role="origin",
    )

    assert not workflow._source_step_fields_are_grounded(step)


@pytest.mark.parametrize(
    ("operation", "inputs", "output", "expected"),
    [
        (
            "float transformed = input * scale",
            "input scale",
            "transformed",
            True,
        ),
        (
            "float transformed = input * scale",
            "input scale",
            "float",
            False,
        ),
        (
            "if (enabled) output = input",
            "enabled input",
            "output",
            True,
        ),
        (
            "if (enabled) output = input",
            "input",
            "enabled",
            False,
        ),
        (
            "values[index] = input",
            "index input",
            "values",
            True,
        ),
        (
            "values[index] = input",
            "input",
            "index",
            False,
        ),
        (
            "output = intermediate = input",
            "input",
            "output intermediate",
            True,
        ),
        (
            "output = input; // stale = reversed",
            "input",
            "output",
            True,
        ),
        (
            "output = input; // stale = reversed",
            "output",
            "input",
            False,
        ),
        (
            'output /* stale = other */ = choose("mode=rtl", input)',
            "input",
            "output",
            True,
        ),
        (
            'output = parse("say \\"lhs=rhs\\"", input)',
            "input",
            "output",
            True,
        ),
        (
            "output = '='",
            "'='",
            "output",
            True,
        ),
        (
            'output += choose("x=y", input)',
            "output input",
            "output",
            True,
        ),
        (
            'output = R"tag(raw " quote = hidden)tag"',
            'R"tag(raw " quote = hidden)tag"',
            "output",
            True,
        ),
        (
            "output = 1'000",
            "1'000",
            "output",
            True,
        ),
        (
            "if (threshold < 0x1.f'0p2) output = input",
            "output",
            "input",
            False,
        ),
        (
            "v(0) = input",
            "input",
            "v",
            True,
        ),
        (
            "m(0, 1) = input",
            "input",
            "m",
            True,
        ),
        (
            "A.slice<2,2>(1,1) = C",
            "C",
            "A",
            True,
        ),
        (
            "m33.slice<2,3>(0,0).slice<2,1>(0,2) = rhs",
            "rhs",
            "m33",
            True,
        ),
        (
            "(*parserbuf_index) = 0",
            "0",
            "parserbuf_index",
            True,
        ),
        (
            "v(0) = input",
            "v",
            "input",
            False,
        ),
    ],
)
def test_assignment_direction_uses_lvalue_tokens(
    operation,
    inputs,
    output,
    expected,
):
    step = workflow.SourceLineageStep(
        sequence=1,
        file="src/module.cpp",
        lines="1",
        operation=operation,
        source_excerpt=operation,
        input_or_state=inputs,
        output_or_effect=output,
        role="transformation",
    )

    assert workflow._source_assignment_direction_is_valid(step) is expected


def test_source_mask_preserves_offsets_and_ignores_non_code_assignments():
    operation = (
        'output /* stale = value */ = parse("x=y", input); '
        "// ignored = text"
    )
    masked = workflow._mask_source_non_code(operation)

    assert len(masked) == len(operation)
    assert [
        match.start()
        for match in workflow._assignment_operator_matches(operation)
    ] == [operation.index("=", operation.index("*/") + 2)]


def test_source_mask_handles_raw_strings_and_digit_separators():
    raw_operation = (
        'output = R"tag(raw " quote = hidden)tag"; next = input'
    )
    digit_operation = "output = 1'000; next = input"
    hex_float_operation = (
        "if (threshold < 0x1.f'0p2) output = input"
    )

    assert [
        match.start()
        for match in workflow._assignment_operator_matches(raw_operation)
    ] == [
        raw_operation.index("="),
        raw_operation.rindex("="),
    ]
    assert [
        match.start()
        for match in workflow._assignment_operator_matches(digit_operation)
    ] == [
        digit_operation.index("="),
        digit_operation.rindex("="),
    ]
    assert [
        match.start()
        for match in workflow._assignment_operator_matches(
            hex_float_operation
        )
    ] == [hex_float_operation.index("=")]


def test_source_grounding_ignores_identifiers_inside_log_strings():
    operation = 'PX4_INFO("output = %f", input)'
    step = workflow.SourceLineageStep(
        sequence=1,
        file="src/module.cpp",
        lines="1",
        operation=operation,
        source_excerpt=operation,
        input_or_state="input",
        output_or_effect="output",
        role="transformation",
    )

    assert not workflow._source_step_fields_are_grounded(step)


def test_literal_only_source_steps_cannot_form_a_lineage_connection():
    upstream_text = 'PX4_INFO("shared literal")'
    downstream_text = 'PX4_WARN("shared literal")'
    lineage = [
        workflow.SourceLineageStep(
            sequence=1,
            file="src/module.cpp",
            lines="1",
            operation=upstream_text,
            source_excerpt=upstream_text,
            input_or_state='"shared literal"',
            output_or_effect='"shared literal"',
            role="propagation",
        ),
        workflow.SourceLineageStep(
            sequence=2,
            file="src/module.cpp",
            lines="2",
            operation=downstream_text,
            source_excerpt=downstream_text,
            input_or_state='"shared literal"',
            output_or_effect='"shared literal"',
            role="publication",
        ),
    ]

    assert not workflow._lineage_has_connected_data_flow(lineage)


def test_review_accepts_semantically_identical_quoted_git_command():
    quoted_source_command = (
        "git -C 'PX4-Autopilot' show SNAPSHOT:src/module.cpp"
    )
    review = _review()
    review.candidate_reviews[0].source_verification_commands = [
        quoted_source_command
    ]
    review.execution_receipts[0].command = quoted_source_command

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "sufficient"


def test_evidence_gate_fails_closed_on_cross_stage_candidate_mismatch():
    review = _review(candidate_id="unknown")
    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert "review refers to unknown candidate: unknown" in state.validation_issues


def test_evidence_gate_cannot_relabel_a_causal_question_as_observational():
    review = _review()
    review.question_is_causal = False

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.question_is_causal is True
    assert state.accepted_candidate_ids == []
    assert any(
        "causal classification changed" in issue
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_a_reviewer_invented_check():
    review = _review()
    review.candidate_reviews[0].reviewed_checks[0].check_id = "not_performed"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert any(
        "unknown targeted check not_performed" in issue
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_an_empty_source_lineage():
    source = _source()
    source.candidates[0].upstream_assignment_path = []

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert "candidate c1_1 has an empty source lineage" in state.validation_issues


def test_evidence_gate_rejects_duplicate_check_ids():
    targeted = _targeted()
    targeted.checks.append(_check())

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert "duplicate targeted check id: check_1" in state.validation_issues


def test_evidence_gate_rejects_duplicate_candidate_titles():
    source = _source()
    source.candidates.append(_candidate("c1_2", "Upstream transform"))

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        issue.startswith("duplicate candidate title: Upstream transform")
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_cross_stage_target_drift():
    source = _source()
    source.explained_logged_value = "different.value"
    source.candidates[0].explained_logged_value = "different.value"
    targeted = _targeted()
    targeted.explained_logged_value = "different.value"
    review = _review()
    review.explained_logged_value = "different.value"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert any(
        "changed the analysis target" in issue
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_material_contradiction():
    targeted = _targeted()
    targeted.checks.append(
        _check(assessment="contradicts", suffix="2")
    )
    review = _review()
    review.candidate_reviews[0].contradicting_evidence = [
        "The second same-window replay contradicts the candidate."
    ]

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert any(
        "retains contradicting evidence" in issue
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_missing_required_signal():
    targeted = _targeted()
    targeted.checks[0].topic_fields = ["output.value"]
    targeted.missing_topic_fields = ["input.value"]

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=True,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert any(
        "did not evaluate required topic fields" in issue
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_model_authored_check_without_receipt():
    targeted = _targeted()
    targeted.execution_receipts = []

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert any(
        "cites an unexecuted command" in issue
        for issue in state.validation_issues
    )


def test_invalid_alternative_does_not_erase_supported_candidate():
    source = _source()
    alternative = _candidate("c1_2", "Incomplete alternative")
    alternative.upstream_assignment_path = []
    source.candidates.append(alternative)

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "sufficient"
    assert state.accepted_candidate_ids == ["c1_1"]
    assert "source candidate was not reviewed: c1_2" in state.validation_issues


def test_complete_competing_candidate_must_be_reviewed():
    source = _source()
    source.candidates.append(_candidate("c1_2", "Viable alternative"))

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert "source candidate was not reviewed: c1_2" in state.validation_issues


def test_not_evaluable_check_does_not_satisfy_a_required_signal():
    targeted = _targeted()
    targeted.checks[0].topic_fields = ["output.value"]
    not_evaluable = _check(
        role="observation",
        assessment="not_evaluable",
        suffix="2",
    )
    not_evaluable.topic_fields = ["input.value"]
    targeted.checks.append(not_evaluable)

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert any(
        "did not evaluate required topic fields" in issue
        and "input.value" in issue
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_receipts_with_unrelated_output():
    source = _source()
    source.execution_receipts[0].stdout = "unrelated README text"
    targeted = _targeted()
    targeted.execution_receipts[0].stdout = "hello"
    review = _review()
    review.execution_receipts[-1].stdout = "hello"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert any(
        "excerpt was not returned" in issue
        for issue in state.validation_issues
    )
    assert any(
        "result was not returned" in issue
        for issue in state.validation_issues
    )


def test_evidence_gate_rejects_invented_source_operation_and_line_range():
    source = _source()
    for step in source.candidates[0].upstream_assignment_path:
        step.operation = "="
        step.source_excerpt = "="
        step.lines = "999999"
    source.publishing_assignment = (
        source.candidates[0].upstream_assignment_path[-1]
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "line range does not match" in issue
        for issue in state.validation_issues
    )


def test_runtime_condition_requires_an_output_bound_performed_check():
    targeted = _targeted()
    targeted.checks[0].evaluated_requirement_ids = []

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "did not evaluate required runtime conditions" in issue
        for issue in state.validation_issues
    )


def test_unrelated_review_git_read_cannot_verify_candidate_source():
    unrelated_command = (
        "git -C PX4-Autopilot show SNAPSHOT:README.md"
    )
    review = _review()
    review.candidate_reviews[0].source_verification_commands = [
        unrelated_command
    ]
    review.execution_receipts[0] = workflow.ShellExecutionReceipt(
        command=unrelated_command,
        exit_code=0,
        stdout="unrelated README",
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "did not independently verify lineage step" in issue
        for issue in state.validation_issues
    )


def test_causal_result_must_come_from_a_cited_python_receipt():
    targeted = _targeted()
    rg_command = "rg maximum check_candidate.py"
    targeted.checks[0].execution_commands.append(rg_command)
    requirement_id = targeted.checks[0].evaluated_requirement_ids[0]
    targeted.execution_receipts[0].stdout = (
        "fields=input.value,output.value parameter SCALE=2 "
        f"{requirement_id}"
    )
    targeted.execution_receipts.append(
        workflow.ShellExecutionReceipt(
            command=rg_command,
            exit_code=0,
            stdout="maximum error 0.01",
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "result was not returned" in issue
        for issue in state.validation_issues
    )


def test_targeted_result_cannot_match_a_receipt_value_prefix():
    targeted = _targeted()
    targeted.checks[0].result = "maximum error 0.0"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []


def test_review_result_cannot_match_a_receipt_value_prefix():
    review = _review()
    review.candidate_reviews[0].reviewed_checks[0].independent_result = (
        "Independent same-window replay"
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []


def test_two_receipt_records_cannot_be_recombined_into_one_check():
    targeted = _targeted()
    targeted.checks[0].result = "wrong result"
    expected_payload = workflow._targeted_check_receipt_payload(
        targeted.checks[0],
        _candidate(),
        targeted,
    )
    other_payload = dict(expected_payload)
    other_payload["check_id"] = "other_check"
    original_payload = dict(expected_payload)
    original_payload["result"] = TARGET_RESULT
    targeted.execution_receipts[0].stdout = "\n".join(
        [
            workflow._receipt_json_marker("check", other_payload),
            workflow._receipt_json_marker("check", original_payload),
        ]
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []


@pytest.mark.parametrize("stage", ["targeted", "review"])
def test_performed_window_must_be_output_bound(stage):
    targeted = _targeted()
    review = _review()
    if stage == "targeted":
        targeted.execution_receipts[0].stdout = ""
    else:
        review.execution_receipts[1].stdout = ""

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "result was not returned" in issue
        for issue in state.validation_issues
    )


def test_performed_window_cannot_match_a_receipt_fragment():
    targeted = _targeted()
    targeted.checks[0].window = "4"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"


def test_competing_candidate_cannot_be_contradicted_without_evidence():
    source = _source()
    source.candidates.append(_candidate("c1_2", "Viable alternative"))
    review = _review()
    review.candidate_reviews.append(
        workflow.CandidateReview(
            candidate_id="c1_2",
            verdict="contradicted",
            confidence="medium",
            source_path_complete=True,
            source_verification_commands=[SOURCE_COMMAND],
            reviewed_checks=[],
            contradicting_evidence=[],
            reasoning="The candidate was dismissed without a performed test.",
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "contradicted without output-bound contradiction evidence" in issue
        for issue in state.validation_issues
    )


def test_output_bound_competing_exclusion_reaches_public_report():
    source = _source()
    source.candidates.append(_candidate("c1_2", "Competing transform"))
    targeted = _targeted()
    competing_check = _check(
        candidate_id="c1_2",
        role="contradiction",
        assessment="contradicts",
        suffix="2",
    )
    targeted.checks.append(competing_check)
    competing_requirement = competing_check.evaluated_requirement_ids[0]
    targeted.execution_receipts[0].stdout += (
        "\n"
        + workflow._receipt_json_marker(
            "check",
            workflow._targeted_check_receipt_payload(
                competing_check,
                source.candidates[1],
                targeted,
            ),
        )
    )

    review = _review()
    competing_review = _review(
        candidate_id="c1_2"
    ).candidate_reviews[0]
    competing_review.verdict = "contradicted"
    competing_review.confidence = "medium"
    competing_review.reviewed_checks[0].check_id = "check_2"
    competing_review.reviewed_checks[0].assessment = "contradicts"
    competing_review.reviewed_checks[0].independent_result = (
        "Independent contradiction."
    )
    competing_review.supporting_evidence = []
    competing_review.contradicting_evidence = [
        "The independently replayed check contradicted the candidate."
    ]
    review.candidate_reviews.append(competing_review)
    review.execution_receipts[1].stdout += (
        "\n"
        + workflow._receipt_json_marker(
            "reviewed_check",
            workflow._reviewed_check_receipt_payload(
                competing_review.reviewed_checks[0],
                competing_check,
            ),
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )
    gated = workflow.apply_evidence_state_to_report(
        _report(),
        evidence_state=state,
        review=review,
        source=source,
        targeted=targeted,
        initial=_initial(),
    )

    assert state.status == "sufficient"
    assert state.excluded_candidate_ids == ["c1_2"]
    assert gated.excluded_mechanisms == (
        state.excluded_candidate_titles
    )
    assert "Competing transform" not in gated.excluded_mechanisms


def test_target_and_reviewer_disagreement_cannot_exclude_a_candidate():
    source = _source()
    source.candidates.append(_candidate("c1_2", "Competing transform"))
    targeted = _targeted()
    competing_check = _check(
        candidate_id="c1_2",
        role="contradiction",
        assessment="contradicts",
        suffix="2",
    )
    targeted.checks.append(competing_check)
    targeted.execution_receipts[0].stdout += (
        "\n"
        + workflow._receipt_json_marker(
            "check",
            workflow._targeted_check_receipt_payload(
                competing_check,
                source.candidates[1],
                targeted,
            ),
        )
    )

    review = _review()
    competing_review = _review(candidate_id="c1_2").candidate_reviews[0]
    competing_review.verdict = "contradicted"
    competing_review.confidence = "medium"
    competing_review.supporting_evidence = []
    competing_review.contradicting_evidence = [
        "The target-stage check claimed a contradiction."
    ]
    competing_review.reviewed_checks[0].check_id = "check_2"
    competing_review.reviewed_checks[0].assessment = "supports"
    review.candidate_reviews.append(competing_review)
    review.execution_receipts[1].stdout += (
        "\n"
        + workflow._receipt_json_marker(
            "reviewed_check",
            workflow._reviewed_check_receipt_payload(
                competing_review.reviewed_checks[0],
                competing_check,
            ),
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert "c1_2" not in state.excluded_candidate_ids


def test_stage_artifacts_preserve_receipts_needed_to_replay_the_gate(
    tmp_path,
):
    initial_path = tmp_path / "initial.json"
    source_path = tmp_path / "source.json"
    targeted_path = tmp_path / "targeted.json"
    review_path = tmp_path / "review.json"
    workflow._write_stage_artifact(initial_path, _initial())
    workflow._write_stage_artifact(source_path, _source())
    workflow._write_stage_artifact(targeted_path, _targeted())
    workflow._write_stage_artifact(review_path, _review())

    state = workflow.derive_evidence_state(
        initial=workflow.InitialInspection.model_validate_json(
            initial_path.read_text(encoding="utf-8")
        ),
        source=workflow.SourceInvestigation.model_validate_json(
            source_path.read_text(encoding="utf-8")
        ),
        targeted=workflow.TargetedLogParse.model_validate_json(
            targeted_path.read_text(encoding="utf-8")
        ),
        review=workflow.EvidenceReview.model_validate_json(
            review_path.read_text(encoding="utf-8")
        ),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "sufficient"
    assert state.accepted_candidate_ids == ["c1_1"]


def test_ambiguous_check_cannot_satisfy_a_runtime_requirement():
    targeted = _targeted()
    requirement_id = targeted.checks[0].evaluated_requirement_ids[0]
    targeted.checks[0].evaluated_requirement_ids = []
    ambiguous = _check(
        role="observation",
        assessment="ambiguous",
        suffix="2",
    )
    ambiguous.evaluated_requirement_ids = [requirement_id]
    targeted.checks.append(ambiguous)

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "did not evaluate required runtime conditions" in issue
        for issue in state.validation_issues
    )


def test_unreviewed_check_cannot_satisfy_a_runtime_requirement():
    targeted = _targeted()
    targeted.checks[0].evaluated_requirement_ids = []
    targeted.checks.append(
        _check(
            role="downstream_consistency",
            suffix="2",
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "did not evaluate required runtime conditions" in issue
        for issue in state.validation_issues
    )


def test_source_lineage_rejects_reversed_data_flow():
    source = _source()
    lineage = source.candidates[0].upstream_assignment_path
    lineage.reverse()
    for sequence, step in enumerate(lineage, start=1):
        step.sequence = sequence

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "disconnected or reversed data flow" in issue
        for issue in state.validation_issues
    )


def test_review_must_verify_the_same_snapshot_repository_alias():
    different_alias_command = (
        "git -C PX4-Autopilot/modules/other "
        "show SNAPSHOT:src/module.cpp"
    )
    review = _review()
    review.candidate_reviews[0].source_verification_commands = [
        different_alias_command
    ]
    review.execution_receipts[0] = workflow.ShellExecutionReceipt(
        command=different_alias_command,
        exit_code=0,
        stdout=SOURCE_STDOUT,
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "did not independently verify lineage step" in issue
        for issue in state.validation_issues
    )


def test_one_python_receipt_must_bind_check_result_fields_and_requirements():
    targeted = _targeted()
    result_command = "python print_result.py"
    targeted.checks[0].execution_commands.append(result_command)
    requirement_id = targeted.checks[0].evaluated_requirement_ids[0]
    targeted.execution_receipts[0].stdout = (
        "fields=input.value,output.value parameter SCALE=2 "
        f"{requirement_id}"
    )
    targeted.execution_receipts.append(
        workflow.ShellExecutionReceipt(
            command=result_command,
            exit_code=0,
            stdout="maximum error 0.01",
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "all cited fields and requirements" in issue
        for issue in state.validation_issues
    )


def test_duplicate_targeted_parameter_names_fail_closed():
    source = _source()
    targeted = _targeted()
    targeted.parameter_values.append(
        workflow.NamedValue(name="SCALE", value="999")
    )
    targeted.execution_receipts[0].stdout = (
        workflow._receipt_json_marker(
            "check",
            workflow._targeted_check_receipt_payload(
                targeted.checks[0],
                source.candidates[0],
                targeted,
            ),
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert (
        "targeted log parse contains duplicate parameter value: SCALE"
        in state.validation_issues
    )


def test_incomplete_source_review_cannot_exclude_competitor_with_free_text():
    source = _source()
    source.candidates.append(_candidate("c1_2", "Viable alternative"))
    review = _review()
    review.candidate_reviews.append(
        workflow.CandidateReview(
            candidate_id="c1_2",
            verdict="contradicted",
            confidence="medium",
            source_path_complete=False,
            source_verification_commands=[SOURCE_COMMAND],
            reviewed_checks=[],
            contradicting_evidence=["The source path looked incomplete."],
            reasoning="No output-bound contradiction check was performed.",
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "contradicted without output-bound contradiction evidence" in issue
        for issue in state.validation_issues
    )


def test_source_excerpt_requires_its_exact_line_span():
    source = _source()
    source.candidates[0].upstream_assignment_path[0].lines = "1-999999"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "line range does not match" in issue
        for issue in state.validation_issues
    )


@pytest.mark.parametrize("claimed_lines", ["-1", "line 1", "1 to 1"])
def test_source_line_range_uses_canonical_positive_syntax(claimed_lines):
    source = _source()
    source.candidates[0].upstream_assignment_path[0].lines = claimed_lines

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "line range does not match" in issue
        for issue in state.validation_issues
    )


def test_source_excerpt_cannot_omit_the_rest_of_a_selected_line():
    source = _source()
    step = source.candidates[0].upstream_assignment_path[0]
    step.source_excerpt = "transformed = input"
    step.operation = "transformed = input"
    review = _review()
    reviewed_step = (
        review.candidate_reviews[0].reviewed_source_lineage[0]
    )
    reviewed_step.source_excerpt = step.source_excerpt
    reviewed_step.input_tokens = ["input"]

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "line range does not match" in issue
        for issue in state.validation_issues
    )


def test_source_operation_must_be_the_complete_verified_excerpt():
    source = _source()
    source.candidates[0].upstream_assignment_path[0].operation = "input"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert any(
        "operation or data flow is not grounded" in issue
        for issue in state.validation_issues
    )


def test_publishing_assignment_must_contain_the_logged_field():
    source = _source()
    publishing = source.candidates[0].upstream_assignment_path[-1]
    publishing.operation = "message.foo = transformed"
    publishing.source_excerpt = "message.foo = transformed"
    publishing.output_or_effect = "message foo"
    source.publishing_assignment = publishing
    source.execution_receipts[0].stdout = SOURCE_STDOUT.replace(
        "message.value = transformed",
        "message.foo = transformed",
    )
    review = _review()
    reviewed = review.candidate_reviews[0].reviewed_source_lineage[-1]
    reviewed.source_excerpt = "message.foo = transformed"
    reviewed.output_tokens = ["message", "foo"]
    review.execution_receipts[0].stdout = (
        review.execution_receipts[0].stdout.replace(
            "message.value = transformed",
            "message.foo = transformed",
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "does not contain the source field" in issue
        for issue in state.validation_issues
    )


def test_publishing_assignment_must_match_the_logged_topic_schema():
    source = _source()
    publishing = source.candidates[0].upstream_assignment_path[-1]
    publishing.operation = "unrelated.value = transformed"
    publishing.source_excerpt = "unrelated.value = transformed"
    publishing.output_or_effect = "unrelated value"
    source.publishing_assignment = publishing
    source.execution_receipts[0].stdout = SOURCE_STDOUT.replace(
        "message.value = transformed",
        "unrelated.value = transformed",
    )
    binding = source.publication_binding
    assert binding is not None
    binding.message_identifier = "unrelated"
    binding.message_type_identifier = "unrelated_s"
    binding.declaration_excerpt = "unrelated_s unrelated{}"
    source.execution_receipts[1].stdout = binding.declaration_excerpt

    review = _review()
    reviewed_step = (
        review.candidate_reviews[0].reviewed_source_lineage[-1]
    )
    reviewed_step.source_excerpt = publishing.source_excerpt
    reviewed_step.output_tokens = ["unrelated", "value"]
    review.execution_receipts[0].stdout = (
        review.execution_receipts[0].stdout.replace(
            "message.value = transformed",
            "unrelated.value = transformed",
        )
    )
    reviewed_binding = review.reviewed_publication_binding
    assert reviewed_binding is not None
    reviewed_binding.message_identifier = "unrelated"
    reviewed_binding.message_type_identifier = "unrelated_s"
    reviewed_binding.declaration_excerpt = "unrelated_s unrelated{}"
    review.execution_receipts[2].stdout = (
        reviewed_binding.declaration_excerpt
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "message type and schema" in issue
        for issue in state.validation_issues
    )


def test_nested_logged_field_reaches_sufficient_with_schema_chain():
    initial, source, targeted, review = _nested_evidence_inputs()

    state = workflow.derive_evidence_state(
        initial=initial,
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "sufficient"
    assert state.accepted_candidate_ids == ["c1_1"]


def test_indexed_nested_logged_field_reaches_sufficient():
    initial, source, targeted, review = _nested_evidence_inputs(
        focus="esc_status[0].esc[3].esc_rpm",
        root_schema_file="msg/EscStatus.msg",
        nested_schema_file="msg/EscReport.msg",
        message_type_identifier="esc_status_s",
        root_schema="EscReport[8] esc",
        nested_schema="int32 esc_rpm",
        publishing_source="message.esc[i].esc_rpm = transformed",
    )

    state = workflow.derive_evidence_state(
        initial=initial,
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "sufficient"
    assert state.accepted_candidate_ids == ["c1_1"]


def test_nested_logged_field_requires_every_intermediate_schema():
    initial, source, targeted, review = _nested_evidence_inputs()
    binding = source.publication_binding
    reviewed_binding = review.reviewed_publication_binding
    assert binding is not None
    assert reviewed_binding is not None
    binding.nested_schema_bindings = []
    reviewed_binding.nested_schema_bindings = []

    state = workflow.derive_evidence_state(
        initial=initial,
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "message type and schema" in issue
        for issue in state.validation_issues
    )


def test_nested_logged_field_requires_independent_schema_receipt():
    initial, source, targeted, review = _nested_evidence_inputs()
    review.execution_receipts = review.execution_receipts[:-1]

    state = workflow.derive_evidence_state(
        initial=initial,
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "independent review did not verify" in issue
        for issue in state.validation_issues
    )


@pytest.mark.parametrize(
    ("focus", "publishing_source", "schema_excerpt"),
    [
        (
            "output.value[0]",
            "message.value[0] = transformed",
            "float32 value",
        ),
        (
            "output.value",
            "message.value = transformed",
            "float32[2] value",
        ),
        (
            "output.value[2]",
            "message.value[i] = transformed",
            "float32[2] value",
        ),
        (
            "output.value[1]",
            "message.value[0] = transformed",
            "float32[2] value",
        ),
    ],
)
def test_publication_binding_rejects_array_shape_or_index_mismatch(
    focus,
    publishing_source,
    schema_excerpt,
):
    source = _source()
    binding = source.publication_binding
    assert binding is not None
    binding.logged_value = focus
    binding.schema_excerpt = schema_excerpt
    source.execution_receipts[2].stdout = schema_excerpt
    publishing = source.publishing_assignment
    assert publishing is not None
    publishing.source_excerpt = publishing_source

    assert not workflow._publication_binding_is_valid(
        binding,
        publishing_assignment=publishing,
        expected_logged_value=focus,
        receipts=source.execution_receipts,
        successful_commands=workflow._successful_execution_commands(
            source.execution_receipts
        ),
    )


def test_publication_binding_rejects_nested_message_as_logged_leaf():
    _initial_state, source, _targeted_state, _review_state = (
        _nested_evidence_inputs()
    )
    binding = source.publication_binding
    publishing = source.publishing_assignment
    assert binding is not None
    assert publishing is not None
    focus = "position_setpoint_triplet.current"
    binding.logged_value = focus
    binding.nested_schema_bindings = []
    publishing.source_excerpt = "message.current = transformed"

    assert not workflow._publication_binding_is_valid(
        binding,
        publishing_assignment=publishing,
        expected_logged_value=focus,
        receipts=source.execution_receipts,
        successful_commands=workflow._successful_execution_commands(
            source.execution_receipts
        ),
    )


def test_designated_initializer_reaches_sufficient_when_enclosed():
    initial, source, targeted, review = (
        _designated_initializer_evidence_inputs()
    )

    state = workflow.derive_evidence_state(
        initial=initial,
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "sufficient"
    assert state.accepted_candidate_ids == ["c1_1"]


def test_designated_initializer_requires_enclosing_declaration_excerpt():
    _initial_state, source, _targeted_state, _review_state = (
        _designated_initializer_evidence_inputs()
    )
    binding = source.publication_binding
    publishing = source.publishing_assignment
    assert binding is not None
    assert publishing is not None
    binding.declaration_lines = "1"
    binding.declaration_excerpt = "output_s message {"
    source.execution_receipts[1].stdout = binding.declaration_excerpt

    assert not workflow._publication_binding_is_valid(
        binding,
        publishing_assignment=publishing,
        expected_logged_value="output.value",
        receipts=source.execution_receipts,
        successful_commands=workflow._successful_execution_commands(
            source.execution_receipts
        ),
    )


def test_nested_logged_field_schema_type_must_select_next_schema():
    initial, source, targeted, review = _nested_evidence_inputs()
    binding = source.publication_binding
    reviewed_binding = review.reviewed_publication_binding
    assert binding is not None
    assert reviewed_binding is not None
    binding.schema_excerpt = "Unrelated current"
    reviewed_binding.schema_excerpt = "Unrelated current"
    source.execution_receipts[2].stdout = binding.schema_excerpt
    review.execution_receipts[3].stdout = binding.schema_excerpt

    state = workflow.derive_evidence_state(
        initial=initial,
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "message type and schema" in issue
        for issue in state.validation_issues
    )


@pytest.mark.parametrize(
    ("logged_value", "expected"),
    [
        (
            "position_setpoint_triplet.current.valid",
            (
                "position_setpoint_triplet",
                ["current", "valid"],
            ),
        ),
        (
            "esc_status.esc[0].timestamp",
            ("esc_status", ["esc", "timestamp"]),
        ),
        (
            "esc_status[0].esc[3].esc_rpm",
            ("esc_status", ["esc", "esc_rpm"]),
        ),
    ],
)
def test_logged_value_path_preserves_nested_fields_and_removes_indices(
    logged_value,
    expected,
):
    assert workflow._logged_value_path(logged_value) == expected


@pytest.mark.parametrize(
    "logged_value",
    [
        "output",
        "output.",
        "output.values[index]",
        "output..value",
    ],
)
def test_logged_value_path_rejects_non_ulog_field_syntax(logged_value):
    assert workflow._logged_value_path(logged_value) is None


@pytest.mark.parametrize(
    ("source_excerpt", "expected"),
    [
        (
            "this->_message.current.valid = transformed",
            ("_message", ["current", "valid"]),
        ),
        (
            "_esc_status.esc[i].timestamp = transformed",
            ("_esc_status", ["esc", "timestamp"]),
        ),
    ],
)
def test_publishing_lhs_binding_preserves_nested_field_path(
    source_excerpt,
    expected,
):
    publishing = workflow.SourceLineageStep(
        sequence=1,
        file="src/module.cpp",
        lines="1",
        operation=source_excerpt,
        source_excerpt=source_excerpt,
        input_or_state="transformed",
        output_or_effect="message",
        role="publication",
    )

    assert workflow._publishing_lhs_binding(publishing) == expected


def test_schema_topics_override_the_default_filename_topic():
    schema_text = "# TOPICS unrelated\nfloat32 value"
    source = _source()
    binding = source.publication_binding
    assert binding is not None
    binding.schema_lines = "2"
    source.execution_receipts[2].stdout = schema_text
    review = _review()
    reviewed_binding = review.reviewed_publication_binding
    assert reviewed_binding is not None
    reviewed_binding.schema_lines = "2"
    review.execution_receipts[3].stdout = schema_text

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "message type and schema" in issue
        for issue in state.validation_issues
    )


def test_schema_topic_directive_must_match_px4_generator_syntax():
    unrelated_schema_command = (
        "git -C PX4-Autopilot show SNAPSHOT:msg/Unrelated.msg"
    )
    schema_text = " # TOPICS output\nfloat32 value"
    source = _source()
    binding = source.publication_binding
    assert binding is not None
    binding.message_type_identifier = "unrelated_s"
    binding.declaration_excerpt = "unrelated_s message{}"
    binding.schema_file = "msg/Unrelated.msg"
    binding.schema_lines = "2"
    binding.schema_execution_commands = [unrelated_schema_command]
    source.execution_receipts[1].stdout = binding.declaration_excerpt
    source.execution_receipts[2].command = unrelated_schema_command
    source.execution_receipts[2].stdout = schema_text

    review = _review()
    reviewed_binding = review.reviewed_publication_binding
    assert reviewed_binding is not None
    reviewed_binding.message_type_identifier = "unrelated_s"
    reviewed_binding.declaration_excerpt = "unrelated_s message{}"
    reviewed_binding.schema_file = "msg/Unrelated.msg"
    reviewed_binding.schema_lines = "2"
    reviewed_binding.schema_execution_commands = [
        unrelated_schema_command
    ]
    review.execution_receipts[2].stdout = (
        reviewed_binding.declaration_excerpt
    )
    review.execution_receipts[3].command = unrelated_schema_command
    review.execution_receipts[3].stdout = schema_text

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert workflow._schema_declared_topic_names(schema_text) == set()
    assert state.status == "unresolved"
    assert any(
        "message type and schema" in issue
        for issue in state.validation_issues
    )


def test_publication_declaration_cannot_be_assembled_across_lines():
    source = _source()
    binding = source.publication_binding
    assert binding is not None
    binding.declaration_lines = "1-2"
    binding.declaration_excerpt = (
        "output_s unrelated{}\nmessage = unrelated"
    )
    source.execution_receipts[1].stdout = binding.declaration_excerpt
    review = _review()
    reviewed_binding = review.reviewed_publication_binding
    assert reviewed_binding is not None
    reviewed_binding.declaration_lines = binding.declaration_lines
    reviewed_binding.declaration_excerpt = binding.declaration_excerpt
    review.execution_receipts[2].stdout = binding.declaration_excerpt

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "message type and schema" in issue
        for issue in state.validation_issues
    )


def test_publication_binding_uses_the_assignment_object_not_any_token():
    source = _source()
    binding = source.publication_binding
    assert binding is not None
    binding.message_identifier = "value"
    binding.declaration_excerpt = "output_s value{}"
    source.execution_receipts[1].stdout = binding.declaration_excerpt
    review = _review()
    reviewed_binding = review.reviewed_publication_binding
    assert reviewed_binding is not None
    reviewed_binding.message_identifier = "value"
    reviewed_binding.declaration_excerpt = "output_s value{}"
    review.execution_receipts[2].stdout = binding.declaration_excerpt

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "message type and schema" in issue
        for issue in state.validation_issues
    )


def test_candidate_endpoint_must_equal_the_recorded_publication_step():
    source = _source()
    assert source.publishing_assignment is not None
    source.publishing_assignment = (
        source.publishing_assignment.model_copy(
            update={"lines": "1"}
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "does not terminate at the recorded publishing assignment" in issue
        for issue in state.validation_issues
    )


def test_candidate_endpoint_identity_ignores_path_specific_metadata():
    source = _source()
    assert source.publishing_assignment is not None
    source.publishing_assignment = (
        source.publishing_assignment.model_copy(
            update={
                "sequence": 99,
                "symbol": None,
                "runtime_conditions": ["publisher is active"],
            }
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "sufficient"
    assert state.accepted_candidate_ids == ["c1_1"]


@pytest.mark.parametrize(
    "declaration",
    [
        "::output_s message{};",
        "auto message = ::output_s();",
        "const px4::output_s *message = nullptr;",
    ],
)
def test_publication_declaration_accepts_common_cpp_forms(declaration):
    assert workflow._declaration_binds_message_type(
        declaration,
        message_identifier="message",
        message_type_identifier="output_s",
    )


def test_source_excerpt_and_file_must_share_the_same_git_receipt():
    source = _source()
    other_command = (
        "git -C PX4-Autopilot show SNAPSHOT:src/other.cpp"
    )
    for step in source.candidates[0].upstream_assignment_path:
        step.execution_commands.append(other_command)
    source.execution_receipts[0].stdout = "no relevant source here"
    source.execution_receipts.append(
        workflow.ShellExecutionReceipt(
            command=other_command,
            exit_code=0,
            stdout=SOURCE_STDOUT,
        )
    )

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert any(
        "excerpt was not returned" in issue
        for issue in state.validation_issues
    )


def test_modified_executor_receipt_fails_integrity_validation():
    initial = _initial()
    extracted = workflow._extract_execution_receipts(
        _fake_result(initial).new_items
    )
    assert extracted
    assert workflow._receipt_integrity_valid(extracted[0])
    extracted[0].stdout += " tampered"
    initial.execution_receipts = extracted

    state = workflow.derive_evidence_state(
        initial=initial,
        source=_source(),
        targeted=_targeted(),
        review=_review(),
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "unresolved"
    assert "initial inspection contains a modified execution receipt" in (
        state.validation_issues
    )


def test_local_gap_cannot_trigger_web_research():
    review = _review(web_queries=["PX4 output.value meaning"])
    review.web_gap_kind = "local_log"

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=_source(),
        targeted=_targeted(role="downstream_consistency"),
        review=review,
        cycle=1,
        allow_web_research=True,
    )

    assert state.status == "unresolved"
    assert any(
        "not external context" in issue
        for issue in state.validation_issues
    )


def test_external_gap_label_cannot_bypass_broken_local_source_lineage():
    source = _source()
    source.candidates[0].upstream_assignment_path = []

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=_targeted(role="downstream_consistency"),
        review=_review(web_queries=["PX4 output.value source history"]),
        cycle=1,
        allow_web_research=True,
    )

    assert state.status == "unresolved"
    assert state.accepted_candidate_ids == []
    assert "candidate c1_1 has an empty source lineage" in (
        state.validation_issues
    )


def test_causal_lineage_gate_does_not_block_an_observational_answer():
    source = _source()
    source.candidates[0].upstream_assignment_path = []
    initial = _initial()
    initial.question_is_causal = False
    review = workflow.EvidenceReview(
        question_is_causal=False,
        explained_logged_value="output.value",
        local_observations_sufficient=True,
        reviewed_observations=[
            workflow.ReviewedObservation(
                observation_index=0,
                assessment="supports",
                independent_result=(
                    "Independent observation: output.value=3 at 5 s"
                ),
                execution_commands=[REVIEW_COMMAND],
            )
        ],
        candidate_reviews=[],
        requested_state="sufficient",
        reason="The requested observation was read directly from the log.",
        execution_receipts=[
            workflow.ShellExecutionReceipt(
                command=REVIEW_COMMAND,
                exit_code=0,
                stdout=(
                    f"{REVIEW_STDOUT} "
                    "Independent observation: output.value=3 at 5 s"
                ),
            )
        ],
    )
    review.execution_receipts[0].stdout = workflow._receipt_json_marker(
        "reviewed_observation",
        workflow._reviewed_observation_receipt_payload(
            review.reviewed_observations[0],
            initial.observations[0],
            question_focus=initial.question_focus,
        ),
    )

    state = workflow.derive_evidence_state(
        initial=initial,
        source=source,
        targeted=workflow.TargetedLogParse(
            explained_logged_value="output.value",
            execution_receipts=[
                workflow.ShellExecutionReceipt(
                    command=TARGET_COMMAND,
                    exit_code=0,
                    stdout=TARGET_STDOUT,
                )
            ],
        ),
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    assert state.status == "sufficient"
    assert state.question_is_causal is False
    assert state.accepted_candidate_ids == []
    assert state.accepted_observation_indices == [0]

    draft = _report()
    draft.airframe_summary = "INVENTED AIRFRAME"
    draft.ranked_hypotheses[0].mechanism = "INVENTED OBSERVATION"
    draft.final_summary = "INVENTED OBSERVATIONAL SUMMARY"
    gated = workflow.apply_evidence_state_to_report(
        draft,
        evidence_state=state,
        review=review,
        source=source,
        targeted=None,
        initial=initial,
    )

    assert len(gated.ranked_hypotheses) == 1
    assert gated.ranked_hypotheses[0].known_px4_mechanism == (
        "Direct ULog observation"
    )
    assert gated.ranked_hypotheses[0].confidence == "low"
    assert gated.confirmed == []
    assert "INVENTED" not in gated.final_summary
    assert "output.value=3 at 5 s" in gated.final_summary
    assert "INVENTED" not in gated.model_dump_json()
    assert validate_report(gated).passed


def _derive_observational_state(
    initial: workflow.InitialInspection,
    review: workflow.EvidenceReview,
) -> workflow.EvidenceState:
    return workflow.derive_evidence_state(
        initial=initial,
        source=workflow.SourceInvestigation(
            source_available=False,
            explained_logged_value=initial.question_focus,
            unresolved_reason="Source is unnecessary for direct observation.",
        ),
        targeted=workflow.TargetedLogParse(
            explained_logged_value=initial.question_focus,
            execution_receipts=[
                workflow.ShellExecutionReceipt(
                    command=TARGET_COMMAND,
                    exit_code=0,
                    stdout="orientation complete",
                )
            ],
        ),
        review=review,
        cycle=1,
        allow_web_research=False,
    )


def test_observation_evidence_cannot_match_a_receipt_value_prefix():
    initial = _initial()
    initial.question_is_causal = False
    initial.observations[0].evidence = "output.value=3"
    review = _observational_review(initial)

    state = _derive_observational_state(initial, review)

    assert state.status == "unresolved"
    assert state.accepted_observation_indices == []


def test_observation_review_cannot_match_a_receipt_value_prefix():
    initial = _initial()
    initial.question_is_causal = False
    review = _observational_review(initial)
    review.reviewed_observations[0].independent_result = (
        "Independent observation: output.value=3"
    )

    state = _derive_observational_state(initial, review)

    assert state.status == "unresolved"
    assert state.accepted_observation_indices == []


def test_two_receipt_records_cannot_be_recombined_into_one_observation():
    initial = _initial()
    initial.question_is_causal = False
    initial.observations[0].evidence = "output.value=wrong"
    expected_payload = workflow._observation_receipt_payload(
        0,
        initial.observations[0],
        question_focus=initial.question_focus,
    )
    other_payload = dict(expected_payload)
    other_payload["observation_index"] = 1
    original_payload = dict(expected_payload)
    original_payload["evidence"] = INITIAL_EVIDENCE
    initial.execution_receipts[0].stdout = "\n".join(
        [
            workflow._receipt_json_marker(
                "observation",
                other_payload,
            ),
            workflow._receipt_json_marker(
                "observation",
                original_payload,
            ),
        ]
    )
    review = _observational_review(initial)

    state = _derive_observational_state(initial, review)

    assert state.status == "unresolved"
    assert state.accepted_observation_indices == []


def test_event_focus_can_bind_through_exact_observation_text():
    observation = workflow.LogObservation(
        finding="failsafe event occurred",
        evidence="failsafe event occurred at 5 s",
        topic_fields=["vehicle_status.nav_state"],
        time_or_window="5 s",
    )
    initial = workflow.InitialInspection(
        question_is_causal=False,
        question_focus="failsafe event",
        firmware_and_version="test",
        log_duration_and_timebase="10 s",
        observations=[observation],
        relevant_topic_fields=["vehicle_status.nav_state"],
        events_and_messages=["failsafe event"],
        execution_receipts=[
            workflow.ShellExecutionReceipt(
                command=INITIAL_COMMAND,
                exit_code=0,
                stdout=workflow._receipt_json_marker(
                    "observation",
                    workflow._observation_receipt_payload(
                        0,
                        observation,
                        question_focus="failsafe event",
                    ),
                ),
            )
        ],
    )
    review = _observational_review(initial)

    state = _derive_observational_state(initial, review)

    assert state.status == "sufficient"
    assert state.accepted_observation_indices == [0]


def test_observation_for_an_unrelated_focus_fails_closed():
    initial = _initial()
    initial.question_is_causal = False
    initial.question_focus = "unrelated event"
    initial.relevant_topic_fields = ["output.value"]
    review = _observational_review(initial)

    state = _derive_observational_state(initial, review)

    assert state.status == "unresolved"
    assert state.accepted_observation_indices == []


def test_unresolved_gate_downgrades_report_and_canonicalizes_lists():
    report = _report()
    report.confirmed = ["A paraphrased causal claim"]
    state = workflow.EvidenceState(
        cycle=1,
        status="unresolved",
        question_is_causal=True,
        reason="INVENTED REASON says a solar flare caused it.",
    )
    review = _review()

    gated = workflow.apply_evidence_state_to_report(
        report,
        evidence_state=state,
        review=review,
    )
    assert gated.ranked_hypotheses == []
    assert gated.confirmed == []
    assert gated.unconfirmed == []
    assert gated.final_summary == (
        "No causal explanation was confirmed. The available source and "
        "flight-log evidence did not pass the deterministic evidence gate."
    )
    assert "INVENTED" not in gated.model_dump_json()
    assert "caused the value" not in gated.final_summary


def test_unresolved_observational_state_downgrades_public_report():
    report = _report()
    state = workflow.EvidenceState(
        cycle=1,
        status="unresolved",
        question_is_causal=False,
        reason="The requested value could not be measured reliably.",
    )

    gated = workflow.apply_evidence_state_to_report(
        report,
        evidence_state=state,
        review=_review(),
    )

    assert gated.ranked_hypotheses == []
    assert gated.confirmed == []
    assert gated.unconfirmed == []
    assert gated.final_summary == (
        "The requested conclusion remains unresolved. The available source "
        "and flight-log evidence did not pass the deterministic evidence "
        "gate."
    )


def test_sufficient_report_replaces_untraceable_source_references():
    source = _source()
    targeted = _targeted()
    review = _review()
    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )
    report = _report()
    report.ranked_hypotheses[0].source_refs = [
        CodeRef(
            file="invented/not-read.cpp",
            function="fiction",
            start_line=999999,
            end_line=999999,
            snippet="invented",
        )
    ]

    gated = workflow.apply_evidence_state_to_report(
        report,
        evidence_state=state,
        review=review,
        source=source,
        targeted=targeted,
    )

    assert gated.ranked_hypotheses[0].confidence == "high"
    assert gated.confirmed == state.accepted_candidate_titles
    assert [
        (ref.file, ref.start_line, ref.end_line, ref.snippet)
        for ref in gated.ranked_hypotheses[0].source_refs
    ] == [
        (
            "src/module.cpp",
            1,
            1,
            "transformed = input * scale",
        ),
        (
            "src/module.cpp",
            2,
            2,
            "message.value = transformed",
        ),
    ]


def test_submodule_source_path_survives_gate_and_report_binding():
    submodule_command = (
        "git -C PX4-Autopilot/modules/vendor "
        "show SNAPSHOT:src/module.cpp"
    )
    canonical_file = "modules/vendor/src/module.cpp"
    source = _source()
    for step in source.candidates[0].upstream_assignment_path:
        step.file = "src/module.cpp"
        step.execution_commands = [submodule_command]
    source.execution_receipts[0].command = submodule_command
    review = _review()
    review.candidate_reviews[0].source_verification_commands = [
        submodule_command
    ]
    for reviewed_step in (
        review.candidate_reviews[0].reviewed_source_lineage
    ):
        reviewed_step.file = canonical_file
    review.execution_receipts[0].command = submodule_command
    targeted = _targeted()

    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )
    gated = workflow.apply_evidence_state_to_report(
        _report(),
        evidence_state=state,
        review=review,
        source=source,
        targeted=targeted,
    )

    assert state.status == "sufficient"
    assert {
        ref.file for ref in gated.ranked_hypotheses[0].source_refs
    } == {canonical_file}


def test_sufficient_report_replaces_untraceable_numeric_claims():
    source = _source()
    targeted = _targeted()
    review = _review()
    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )
    report = _report()
    report.ranked_hypotheses[0].numeric_checks = [
        RelationshipCheckSpec(
            type="custom",
            description="Invented result not present in any execution.",
        )
    ]
    report.ranked_hypotheses[0].evidence = ["Invented evidence."]

    gated = workflow.apply_evidence_state_to_report(
        report,
        evidence_state=state,
        review=review,
        source=source,
        targeted=targeted,
    )

    checks = gated.ranked_hypotheses[0].numeric_checks
    assert len(checks) == 1
    assert checks[0].type == "custom"
    assert checks[0].window == "4-6 s"
    assert checks[0].actual == "input.value, output.value"
    assert checks[0].source_predicate == (
        "transformed = input * scale -> message.value = transformed"
    )
    assert checks[0].supports is None
    assert state.accepted_candidate_titles[0] in (
        checks[0].description or ""
    )
    assert "maximum error 0.01" in (checks[0].description or "")
    assert gated.ranked_hypotheses[0].evidence == [
        "check_1: maximum error 0.01 Independent review: "
        "Independent same-window replay matched."
    ]


def test_sufficient_report_replaces_untraceable_mechanism_prose():
    source = _source()
    source.candidates[0].title = "INVENTED SOLAR-FLARE CAUSE"
    source.candidates[0].mechanism = "INVENTED UNRELATED CAUSE"
    targeted = _targeted()
    targeted.checks[0].source_prediction = "INVENTED SIGNATURE"
    review = _review()
    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )
    report = _report("INVENTED SOLAR-FLARE CAUSE")
    hypothesis = report.ranked_hypotheses[0]
    hypothesis.known_px4_mechanism = "Invented mechanism"
    hypothesis.mechanism = "INVENTED DIFFERENT MECHANISM"
    hypothesis.expected_logged_signature[0].description = "Invented signature"
    hypothesis.applicability.applicable = False
    report.excluded_mechanisms = ["INVENTED EXCLUSION"]
    report.final_summary = "Invented final conclusion."

    gated = workflow.apply_evidence_state_to_report(
        report,
        evidence_state=state,
        review=review,
        source=source,
        targeted=targeted,
    )

    assert gated.ranked_hypotheses[0].known_px4_mechanism == (
        "Commit-pinned PX4 source lineage"
    )
    assert gated.ranked_hypotheses[0].mechanism == (
        workflow._candidate_grounded_mechanism(source.candidates[0])
    )
    assert gated.ranked_hypotheses[0].expected_logged_signature[0].name == (
        "check_1"
    )
    assert gated.ranked_hypotheses[0].applicability.applicable is True
    assert "INVENTED" not in gated.final_summary
    assert gated.excluded_mechanisms == []
    assert "INVENTED" not in gated.ranked_hypotheses[0].title
    assert "INVENTED" not in (
        gated.ranked_hypotheses[0]
        .expected_logged_signature[0]
        .expected_behavior
        or ""
    )
    assert "INVENTED" not in (
        gated.ranked_hypotheses[0].numeric_checks[0].source_predicate
        or ""
    )
    assert workflow._candidate_grounded_mechanism(
        source.candidates[0]
    ) in gated.final_summary


def test_sufficient_report_filters_draft_hypotheses_and_plot_metadata(
    tmp_path,
):
    source = _source()
    targeted = _targeted()
    review = _review()
    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )
    report = _report()
    accepted = report.ranked_hypotheses[0]
    accepted.confidence = "low"
    accepted.plots = [
        PlotRef(
            title="INVENTED PLOT TITLE",
            path=str(tmp_path / "evidence.png"),
            purpose="INVENTED PLOT PURPOSE",
            signals=["INVENTED SIGNAL"],
        )
    ]
    unmatched = accepted.model_copy(deep=True)
    unmatched.title = "INVENTED EXTRA HYPOTHESIS"
    unmatched.mechanism = "INVENTED EXTRA MECHANISM"
    unmatched.evidence = ["INVENTED EXTRA EVIDENCE"]
    report.ranked_hypotheses.append(unmatched)

    gated = workflow.apply_evidence_state_to_report(
        report,
        evidence_state=state,
        review=review,
        source=source,
        targeted=targeted,
    )

    assert len(gated.ranked_hypotheses) == 1
    assert gated.ranked_hypotheses[0].confidence == "high"
    assert gated.confirmed == state.accepted_candidate_titles
    assert gated.ranked_hypotheses[0].plots == [
        PlotRef(
            title="Analysis plot: evidence.png",
            path=str(tmp_path / "evidence.png"),
            purpose=(
                "Plot artifact produced during the evidence workflow."
            ),
        )
    ]
    assert "INVENTED" not in gated.model_dump_json()


def test_source_role_labels_do_not_reach_public_code_references():
    source = _source()
    source.candidates[0].upstream_assignment_path[0].role = "branch"
    targeted = _targeted()
    review = _review()
    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )

    gated = workflow.apply_evidence_state_to_report(
        _report(),
        evidence_state=state,
        review=review,
        source=source,
        targeted=targeted,
    )

    assert state.status == "sufficient"
    assert {
        ref.explanation
        for ref in gated.ranked_hypotheses[0].source_refs
    } == {"Verified commit-pinned source lineage excerpt."}


def test_sufficient_report_deduplicates_an_accepted_hypothesis():
    source = _source()
    targeted = _targeted()
    review = _review()
    state = workflow.derive_evidence_state(
        initial=_initial(),
        source=source,
        targeted=targeted,
        review=review,
        cycle=1,
        allow_web_research=False,
    )
    report = _report()
    high = report.ranked_hypotheses[0].model_copy(deep=True)
    low = report.ranked_hypotheses[0].model_copy(deep=True)
    low.confidence = "low"
    report.ranked_hypotheses = [low, high]

    gated = workflow.apply_evidence_state_to_report(
        report,
        evidence_state=state,
        review=review,
        source=source,
        targeted=targeted,
    )

    assert len(gated.ranked_hypotheses) == 1
    assert gated.confirmed == state.accepted_candidate_titles


def test_plot_aliases_are_normalized_only_for_existing_run_artifacts(
    tmp_path,
):
    plots_dir = tmp_path / "outputs" / "web_run" / "plots"
    plots_dir.mkdir(parents=True)
    generated = plots_dir / "nested" / "evidence.png"
    generated.parent.mkdir()
    generated.write_bytes(b"png")
    report = _report()
    report.ranked_hypotheses[0].plots = [
        PlotRef(
            title="Evidence",
            path="/plots/nested/evidence.png",
            purpose="Show the aligned evidence.",
        ),
        PlotRef(
            title="Invented",
            path="/plots/../outside.png",
            purpose="Should not survive normalization.",
        ),
    ]

    aliases = workflow._available_plot_artifacts(plots_dir)
    normalized = workflow._normalize_report_plot_paths(
        report,
        plots_dir=plots_dir,
        available_plot_artifacts=aliases,
    )

    assert aliases == ["/plots/nested/evidence.png"]
    assert normalized.ranked_hypotheses[0].plots[0].path == str(
        generated.resolve()
    )
    assert normalized.ranked_hypotheses[0].plots[1].path == ""
    assert "not produced" in (
        normalized.ranked_hypotheses[0].plots[1].warnings[0]
    )


def test_shared_validation_keeps_unrelated_legacy_confirmation_text():
    report = _report()
    report.ranked_hypotheses[0].numeric_checks = []
    report.confirmed = ["A separate observed fact", "Upstream transform"]

    downgraded = enforce_validation_downgrades(
        report,
        validate_report(report),
    )

    assert downgraded.confirmed == ["A separate observed fact"]
    assert downgraded.unconfirmed == ["Upstream transform"]


def test_staged_agent_output_schemas_are_strict_json_compatible():
    from agents import AgentOutputSchema

    for output_type in (
        workflow.InitialInspection,
        workflow.SourceInvestigation,
        workflow.TargetedLogParse,
        workflow.EvidenceReview,
        workflow.WebResearch,
        FlightLogReport,
    ):
        AgentOutputSchema(output_type)


async def _unused_shell(_request):
    return ShellResult(output=[])


def _shell_factory(access_log: list[bool]):
    def factory(source_access: bool) -> ShellTool:
        access_log.append(source_access)
        return ShellTool(executor=_unused_shell, needs_approval=False)

    return factory


def _fake_result(output, *, raw_responses=None):
    commands: list[str] = []
    if isinstance(output, workflow.InitialInspection):
        commands = [INITIAL_COMMAND]
    elif isinstance(output, workflow.SourceInvestigation):
        commands = [
            SOURCE_COMMAND,
            DECLARATION_COMMAND,
            SCHEMA_COMMAND,
        ]
    elif isinstance(output, workflow.TargetedLogParse):
        commands = [TARGET_COMMAND]
    elif isinstance(output, workflow.EvidenceReview):
        commands = [
            SOURCE_COMMAND,
            REVIEW_COMMAND,
            DECLARATION_COMMAND,
            SCHEMA_COMMAND,
        ]

    new_items = []
    if commands:
        initial_stdout = INITIAL_STDOUT
        if isinstance(output, workflow.InitialInspection):
            initial_stdout = "\n".join(
                workflow._receipt_json_marker(
                    "observation",
                    workflow._observation_receipt_payload(
                        observation_index,
                        observation,
                        question_focus=output.question_focus,
                    ),
                )
                for observation_index, observation in enumerate(
                    output.observations
                )
            )
        targeted_stdout = TARGET_STDOUT
        review_stdout = REVIEW_STDOUT
        if isinstance(output, workflow.TargetedLogParse):
            targeted_stdout = "\n".join(
                workflow._receipt_json_marker(
                    "check",
                    workflow._targeted_check_receipt_payload(
                        check,
                        _candidate(check.candidate_id),
                        output,
                    ),
                )
                for check in output.checks
            )
        if isinstance(output, workflow.EvidenceReview):
            review_stdout = "\n".join(
                workflow._receipt_json_marker(
                    "reviewed_check",
                    workflow._reviewed_check_receipt_payload(
                        reviewed_check,
                        _check(candidate_id=candidate_review.candidate_id),
                    ),
                )
                for candidate_review in output.candidate_reviews
                for reviewed_check in candidate_review.reviewed_checks
            )
            observation_markers = [
                workflow._receipt_json_marker(
                    "reviewed_observation",
                    {
                        "observation_index": (
                            reviewed_observation.observation_index
                        ),
                        "question_focus": output.explained_logged_value,
                        "independent_result": (
                            reviewed_observation.independent_result
                        ),
                        "assessment": reviewed_observation.assessment,
                        "topic_fields": INITIAL_TOPIC_FIELDS,
                    },
                )
                for reviewed_observation in output.reviewed_observations
            ]
            if observation_markers:
                review_stdout += "\n" + "\n".join(
                    observation_markers
                )
        new_items.append(
            SimpleNamespace(
                type="tool_call_output_item",
                raw_item={
                    "type": "shell_call_output",
                    "shell_output": [
                        {
                            "command": command,
                            "stdout": {
                                INITIAL_COMMAND: initial_stdout,
                                SOURCE_COMMAND: SOURCE_STDOUT,
                                DECLARATION_COMMAND: DECLARATION_STDOUT,
                                SCHEMA_COMMAND: SCHEMA_STDOUT,
                                TARGET_COMMAND: targeted_stdout,
                                REVIEW_COMMAND: review_stdout,
                            }.get(command, ""),
                            "stderr": "",
                            "status": "completed",
                            "outcome": {
                                "type": "exit",
                                "exit_code": 0,
                            },
                            "exit_code": 0,
                        }
                        for command in commands
                    ],
                },
            )
        )
    return SimpleNamespace(
        final_output=output,
        new_items=new_items,
        raw_responses=raw_responses or [],
        context_wrapper=SimpleNamespace(
            usage=Usage(
                requests=1,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            )
        ),
    )


def test_staged_workflow_skips_web_when_local_evidence_is_sufficient(
    tmp_path,
    monkeypatch,
):
    outputs = iter([
        _initial(),
        _source(),
        _targeted(),
        _review(),
        _report(),
    ])
    agent_names: list[str] = []

    async def fake_run(agent, prompt, max_turns, hooks):
        agent_names.append(agent.name)
        assert prompt
        assert max_turns == 7
        assert hooks is not None
        return _fake_result(next(outputs))

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    access_log: list[bool] = []
    audit = DeveloperAuditLogger(tmp_path / "audit", run_id="local")

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=SimpleNamespace(
                identity="repo@abc",
                commit_sha="abc",
            ),
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=tmp_path / "output" / "plots",
            model="test-model",
            max_turns=7,
            max_total_requests=100,
            project_instructions="",
            base_instructions="Use flight.ulg and SNAPSHOT.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory(access_log),
        )
    )

    assert result.evidence_state.status == "sufficient"
    assert result.report.confirmed == (
        result.evidence_state.accepted_candidate_titles
    )
    assert result.usage.requests == 5
    assert not any("Web Researcher" in name for name in agent_names)
    assert access_log == [False, True, False, True]
    assert (
        tmp_path / "output" / "work" / "stages" / "09_gated_report.json"
    ).is_file()
    assert json.loads(audit.usage_path.read_text(encoding="utf-8"))[
        "requests"
    ] == 5
    events = [
        json.loads(line)
        for line in audit.events_path.read_text(encoding="utf-8").splitlines()
    ]
    normalized_source_event = next(
        event
        for event in events
        if event["event"] == "agent.shell_analysis_source_1.finished"
    )
    assert normalized_source_event["output"]["snapshot_identity"] == "abc"
    assert any(event["event"] == "evidence_state.final" for event in events)


def test_staged_workflow_hands_created_plots_to_the_final_writer(
    tmp_path,
    monkeypatch,
):
    plots_dir = tmp_path / "output" / "plots"
    outputs = iter([
        _initial(),
        _source(),
        _targeted(),
        _review(),
        _report(),
    ])
    final_payloads: list[dict] = []

    async def fake_run(agent, prompt, max_turns, hooks):
        if agent.name == "PX4 Initial ULog Inspector":
            plots_dir.mkdir(parents=True)
            (plots_dir / "evidence.png").write_bytes(b"png")
        output = next(outputs)
        if agent.name == "PX4 Evidence-Gated Report Writer":
            final_payloads.append(json.loads(prompt))
            output.ranked_hypotheses[0].plots = [
                PlotRef(
                    title="Evidence",
                    path="/plots/evidence.png",
                    purpose="Show the performed comparison.",
                )
            ]
        return _fake_result(output)

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    audit = DeveloperAuditLogger(tmp_path / "audit", run_id="plots")

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=SimpleNamespace(
                identity="repo@abc",
                commit_sha="abc",
            ),
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=plots_dir,
            model="test-model",
            max_turns=7,
            max_total_requests=100,
            project_instructions="",
            base_instructions="Use flight.ulg and SNAPSHOT.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory([]),
        )
    )

    assert final_payloads[0]["available_plot_artifacts"] == [
        "/plots/evidence.png"
    ]
    assert result.report.ranked_hypotheses[0].plots[0].path == str(
        (plots_dir / "evidence.png").resolve()
    )


def test_staged_workflow_runs_one_web_assisted_local_retry(
    tmp_path,
    monkeypatch,
):
    first_source = _source()
    first_source.candidates[0].upstream_assignment_path = []
    first_review = _review(
        web_queries=["PX4 output.value source history"],
    )
    first_review.web_gap_kind = "local_source"
    first_review.next_source_queries = ["Find the upstream assignment."]
    second_source = _source("c2_1", "Revised upstream transform")
    second_targeted = _targeted(candidate_id="c2_1")
    second_review = _review(candidate_id="c2_1")
    web = workflow.WebResearch(
        queries_used=["PX4 output.value source history"],
        findings=[
            workflow.WebFinding(
                title="PX4 change",
                url="https://github.com/PX4/PX4-Autopilot/pull/1",
                source_kind="px4_github",
                version_scope="test revision history",
                relevant_claim="Points to another source assignment.",
            )
        ],
        source_followup_queries=["transformed"],
    )
    outputs = iter([
        _initial(),
        first_source,
        _targeted(role="downstream_consistency"),
        first_review,
        web,
        second_source,
        second_targeted,
        second_review,
        _report("Revised upstream transform"),
    ])
    agent_names: list[str] = []
    web_payloads: list[dict] = []

    async def fake_run(agent, prompt, max_turns, hooks):
        agent_names.append(agent.name)
        if "Web Researcher" in agent.name:
            assert max_turns == workflow.WEB_FALLBACK_MAX_TURNS
            web_payloads.append(json.loads(prompt))
        output = next(outputs)
        raw = (
            [
                {
                    "action": {
                        "query": "PX4 output.value source history",
                    },
                    "citation": {
                        "url": "https://github.com/PX4/PX4-Autopilot/pull/1",
                    },
                }
            ]
            if isinstance(output, workflow.WebResearch)
            else []
        )
        return _fake_result(output, raw_responses=raw)

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    access_log: list[bool] = []
    audit = DeveloperAuditLogger(tmp_path / "audit", run_id="web")

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=SimpleNamespace(
                identity="repo@abc",
                commit_sha="abc",
            ),
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=tmp_path / "output" / "plots",
            model="test-model",
            max_turns=7,
            max_total_requests=100,
            project_instructions="",
            base_instructions="Use flight.ulg and SNAPSHOT.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory(access_log),
        )
    )

    assert result.evidence_state.status == "sufficient"
    assert result.evidence_state.cycle == 2
    assert result.report.confirmed == (
        result.evidence_state.accepted_candidate_titles
    )
    assert result.usage.requests == 9
    assert sum("Web Researcher" in name for name in agent_names) == 1
    assert web_payloads == [
        {
            "approved_queries": ["PX4 output.value source history"],
            "privacy_boundary": (
                "No flight-derived context is supplied to this stage. Use "
                "only the approved technical queries."
            ),
        }
    ]
    assert access_log == [False, True, False, True, True, False, True]
    saved_web = json.loads(
        (
            tmp_path
            / "output"
            / "work"
            / "stages"
            / "05_web_research.json"
        ).read_text(encoding="utf-8")
    )
    assert saved_web["observed_citation_urls"] == [
        "https://github.com/PX4/PX4-Autopilot/pull/1"
    ]
    assert saved_web["observed_queries"] == [
        "PX4 output.value source history"
    ]


def test_causal_source_unavailable_flow_skips_futile_web_retry(
    tmp_path,
    monkeypatch,
):
    unresolved_review = workflow.EvidenceReview(
        question_is_causal=True,
        explained_logged_value="output.value",
        local_observations_sufficient=False,
        candidate_reviews=[],
        requested_state="needs_web_research",
        reason="The exact source path is unavailable.",
        web_gap_kind="external_context",
        web_queries=["PX4 external context"],
    )
    outputs = iter([
        _initial(),
        workflow.TargetedLogParse(
            explained_logged_value="output.value"
        ),
        unresolved_review,
        _report(),
    ])
    agent_names: list[str] = []

    async def fake_run(agent, prompt, max_turns, hooks):
        agent_names.append(agent.name)
        return _fake_result(next(outputs))

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    access_log: list[bool] = []
    audit = DeveloperAuditLogger(tmp_path / "audit", run_id="no_source")

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=None,
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=tmp_path / "output" / "plots",
            model="test-model",
            max_turns=7,
            max_total_requests=100,
            project_instructions="",
            base_instructions="Use flight.ulg.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory(access_log),
        )
    )

    assert result.evidence_state.status == "unresolved"
    assert result.evidence_state.question_is_causal is True
    assert not any("Web Researcher" in name for name in agent_names)
    assert result.usage.requests == 4
    assert access_log == [False, False, False]


def test_optional_web_retry_failure_finishes_as_unresolved(
    tmp_path,
    monkeypatch,
):
    first_review = _review(
        web_queries=["PX4 output.value source history"],
    )
    web = workflow.WebResearch(
        queries_used=["PX4 output.value source history"],
        findings=[
            workflow.WebFinding(
                title="PX4 change",
                url="https://github.com/PX4/PX4-Autopilot/pull/1",
                source_kind="px4_github",
                version_scope="test history",
                relevant_claim="Suggests another source path.",
            )
        ],
    )
    outputs = iter([
        _initial(),
        _source(),
        _targeted(role="downstream_consistency"),
        first_review,
        web,
        _report(),
    ])

    async def fake_run(agent, prompt, max_turns, hooks):
        if agent.name == "PX4 Pinned Source Investigator Cycle 2":
            raise RuntimeError("retry unavailable")
        output = next(outputs)
        raw = (
            [
                {
                    "action": {
                        "query": "PX4 output.value source history",
                    },
                    "citation": {
                        "url": "https://github.com/PX4/PX4-Autopilot/pull/1",
                    },
                }
            ]
            if isinstance(output, workflow.WebResearch)
            else []
        )
        return _fake_result(output, raw_responses=raw)

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    access_log: list[bool] = []
    audit = DeveloperAuditLogger(tmp_path / "audit", run_id="retry_failure")

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=SimpleNamespace(
                identity="repo@abc",
                commit_sha="abc",
            ),
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=tmp_path / "output" / "plots",
            model="test-model",
            max_turns=7,
            max_total_requests=100,
            project_instructions="",
            base_instructions="Use flight.ulg and SNAPSHOT.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory(access_log),
        )
    )

    assert result.evidence_state.status == "unresolved"
    assert "optional local retry failed" in result.evidence_state.reason
    assert result.report.confirmed == []
    assert result.report.final_summary.startswith(
        "No causal explanation was confirmed."
    )
    assert result.usage.requests == 6
    assert access_log == [False, True, False, True, True]


def test_web_fallback_without_findings_skips_second_local_cycle(
    tmp_path,
    monkeypatch,
):
    first_review = _review(
        web_queries=["PX4 output.value source history"],
    )
    web = workflow.WebResearch(
        queries_used=["PX4 output.value source history"],
    )
    outputs = iter([
        _initial(),
        _source(),
        _targeted(role="downstream_consistency"),
        first_review,
        web,
        _report(),
    ])
    agent_names: list[str] = []

    async def fake_run(agent, prompt, max_turns, hooks):
        agent_names.append(agent.name)
        output = next(outputs)
        raw = (
            [{"action": {"query": "PX4 output.value source history"}}]
            if isinstance(output, workflow.WebResearch)
            else []
        )
        return _fake_result(output, raw_responses=raw)

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    audit = DeveloperAuditLogger(tmp_path / "audit", run_id="empty_web")

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=SimpleNamespace(
                identity="repo@abc",
                commit_sha="abc",
            ),
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=tmp_path / "output" / "plots",
            model="test-model",
            max_turns=7,
            max_total_requests=100,
            project_instructions="",
            base_instructions="Use flight.ulg and SNAPSHOT.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory([]),
        )
    )

    assert result.evidence_state.status == "unresolved"
    assert "no usable findings" in result.evidence_state.reason
    assert not any("Cycle 2" in name for name in agent_names)
    assert result.usage.requests == 6


def test_web_provenance_violation_discards_findings_and_skips_retry(
    tmp_path,
    monkeypatch,
):
    first_review = _review(
        web_queries=["PX4 output.value source history"],
    )
    web = workflow.WebResearch(
        queries_used=["reported unapproved query"],
        findings=[
            workflow.WebFinding(
                title="Unobserved result",
                url="https://example.com/unobserved",
                source_kind="primary",
                version_scope="unknown",
                relevant_claim="This finding lacks tool provenance.",
            )
        ],
        source_followup_queries=["untrusted follow-up"],
    )
    outputs = iter([
        _initial(),
        _source(),
        _targeted(role="downstream_consistency"),
        first_review,
        web,
        _report(),
    ])
    agent_names: list[str] = []

    async def fake_run(agent, prompt, max_turns, hooks):
        agent_names.append(agent.name)
        output = next(outputs)
        raw = (
            [{"action": {"query": "observed unapproved query"}}]
            if isinstance(output, workflow.WebResearch)
            else []
        )
        return _fake_result(output, raw_responses=raw)

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    audit = DeveloperAuditLogger(
        tmp_path / "audit",
        run_id="bad_web_provenance",
    )

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=SimpleNamespace(
                identity="repo@abc",
                commit_sha="abc",
            ),
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=tmp_path / "output" / "plots",
            model="test-model",
            max_turns=7,
            max_total_requests=100,
            project_instructions="",
            base_instructions="Use flight.ulg and SNAPSHOT.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory([]),
        )
    )

    saved_web = json.loads(
        (
            tmp_path
            / "output"
            / "work"
            / "stages"
            / "05_web_research.json"
        ).read_text(encoding="utf-8")
    )
    assert "reported unapproved queries" in saved_web["error"]
    assert "observed unapproved queries" in saved_web["error"]
    assert "finding URLs were not observed" in saved_web["error"]
    assert saved_web["findings"] == []
    assert saved_web["source_followup_queries"] == []
    assert result.evidence_state.status == "unresolved"
    assert not any("Cycle 2" in name for name in agent_names)


def test_web_retry_is_skipped_when_budget_cannot_finish_local_verification(
    tmp_path,
    monkeypatch,
):
    outputs = iter([
        _initial(),
        _source(),
        _targeted(role="downstream_consistency"),
        _review(web_queries=["PX4 output.value source history"]),
        _report(),
    ])
    agent_names: list[str] = []

    async def fake_run(agent, prompt, max_turns, hooks):
        agent_names.append(agent.name)
        return _fake_result(next(outputs))

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    audit = DeveloperAuditLogger(
        tmp_path / "audit",
        run_id="web_budget_skip",
    )

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=SimpleNamespace(
                identity="repo@abc",
                commit_sha="abc",
            ),
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=tmp_path / "output" / "plots",
            model="test-model",
            max_turns=7,
            max_total_requests=12,
            project_instructions="",
            base_instructions="Use flight.ulg and SNAPSHOT.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory([]),
        )
    )

    assert result.evidence_state.status == "unresolved"
    assert not any("Web Researcher" in name for name in agent_names)
    events = [
        json.loads(line)
        for line in audit.events_path.read_text(encoding="utf-8").splitlines()
    ]
    skip_event = next(
        event
        for event in events
        if event["event"]
        == "agent.shell_analysis_web_fallback.skipped"
    )
    assert skip_event["output"] == {
        "remaining_requests": 8,
        "required_requests": 9,
    }


def test_total_model_request_budget_caps_all_stages(
    tmp_path,
    monkeypatch,
):
    outputs = iter([
        _initial(),
        _source(),
        _targeted(),
        _review(),
        _report(),
    ])
    allowed_turns: list[int] = []

    async def fake_run(agent, prompt, max_turns, hooks):
        allowed_turns.append(max_turns)
        result = _fake_result(next(outputs))
        result.context_wrapper.usage = Usage(
            requests=max_turns,
            input_tokens=max_turns * 10,
            output_tokens=max_turns * 5,
            total_tokens=max_turns * 15,
        )
        return result

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    audit = DeveloperAuditLogger(tmp_path / "audit", run_id="budget")

    result = __import__("asyncio").run(
        workflow.run_staged_analysis(
            user_question="Why did output.value change?",
            inventory={"git_hash": "abc", "available_topics": ["output"]},
            source_snapshot=SimpleNamespace(
                identity="repo@abc",
                commit_sha="abc",
            ),
            mission_path=None,
            work_dir=tmp_path / "output" / "work",
            plots_dir=tmp_path / "output" / "plots",
            model="test-model",
            max_turns=7,
            max_total_requests=9,
            project_instructions="",
            base_instructions="Use flight.ulg and SNAPSHOT.",
            audit_logger=audit,
            shell_tool_factory=_shell_factory([]),
        )
    )

    assert result.usage.requests == 9
    assert allowed_turns == [2, 2, 2, 2, 1]


def test_failed_output_validation_still_records_usage(
    tmp_path,
    monkeypatch,
):
    async def fake_run(agent, prompt, max_turns, hooks):
        return _fake_result({"not": "an initial inspection"})

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    audit = DeveloperAuditLogger(
        tmp_path / "audit",
        run_id="invalid_output",
    )

    with pytest.raises(Exception):
        __import__("asyncio").run(
            workflow.run_staged_analysis(
                user_question="What happened?",
                inventory={"available_topics": []},
                source_snapshot=None,
                mission_path=None,
                work_dir=tmp_path / "output" / "work",
                plots_dir=tmp_path / "output" / "plots",
                model="test-model",
                max_turns=7,
                max_total_requests=7,
                project_instructions="",
                base_instructions="Use flight.ulg.",
                audit_logger=audit,
                shell_tool_factory=_shell_factory([]),
            )
        )

    saved_usage = json.loads(audit.usage_path.read_text(encoding="utf-8"))
    assert saved_usage["requests"] == 1


def test_runner_exception_after_llm_calls_still_records_hook_usage(
    tmp_path,
    monkeypatch,
):
    async def fake_run(agent, prompt, max_turns, hooks):
        for _ in range(2):
            await hooks.on_llm_end(
                SimpleNamespace(),
                agent,
                SimpleNamespace(
                    response_id=None,
                    request_id=None,
                    usage=Usage(
                        requests=1,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                    ),
                ),
            )
        raise RuntimeError("runner failed after model responses")

    monkeypatch.setattr(workflow.Runner, "run", fake_run)
    audit = DeveloperAuditLogger(
        tmp_path / "audit",
        run_id="runner_exception",
    )

    with pytest.raises(RuntimeError, match="runner failed"):
        __import__("asyncio").run(
            workflow.run_staged_analysis(
                user_question="What happened?",
                inventory={"available_topics": []},
                source_snapshot=None,
                mission_path=None,
                work_dir=tmp_path / "output" / "work",
                plots_dir=tmp_path / "output" / "plots",
                model="test-model",
                max_turns=7,
                max_total_requests=7,
                project_instructions="",
                base_instructions="Use flight.ulg.",
                audit_logger=audit,
                shell_tool_factory=_shell_factory([]),
            )
        )

    saved_usage = json.loads(audit.usage_path.read_text(encoding="utf-8"))
    assert saved_usage["requests"] == 2
    events = [
        json.loads(line)
        for line in audit.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["event"] == "agent.shell_analysis_initial.failed"
        for event in events
    )
