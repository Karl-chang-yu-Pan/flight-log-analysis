from __future__ import annotations

import pytest

from flight_log_agent.analysis.dag_value import DAGValueContext, DAGValueProgram, DAGValueResult
from flight_log_agent.analysis.mechanism_dag import DAGEdge, DAGVertex, MechanismDAG


def operation(name, text, line, *, scope="Controller::step", **metadata):
    return DAGVertex(
        id=name, kind="operation", file="control.cpp", line=line, expression=text,
        metadata={"expression_inputs_exact": True, "reachability": {"exact": True, "all_of": []},
                  "target_scope": {"file": "control.cpp", "callable": scope, "line": line}, **metadata},
    )


def edge(source, target, role="", kind="data"):
    return DAGEdge(id=f"{source}:{target}:{role}:{kind}", source_id=source, target_id=target, role=role, kind=kind)


def graph(*, guarded=False):
    vertices = [
        DAGVertex(id="signal", kind="evidence", sub_kind="logged_signal", signal_name="packet.input"),
        DAGVertex(id="mode", kind="evidence", sub_kind="logged_signal", signal_name="packet.mode"),
        operation("first", "input + 1", 10), operation("second", "input + 2", 20),
        operation("root", "selected * 2", 30),
    ]
    edges = [edge("signal", "first", "input"), edge("signal", "second", "input"),
             edge("first", "root", "selected"), edge("second", "root", "selected")]
    if guarded:
        for writer, predicate in [("first", "mode == 0"), ("second", "mode != 0")]:
            branch = f"gate-{writer}"
            vertices.append(DAGVertex(id=branch, kind="branch", predicate_raw=predicate,
                                      metadata={"expression_inputs_exact": True}))
            next(v for v in vertices if v.id == writer).metadata["reachability"]["all_of"] = [predicate]
            edges += [edge("mode", branch, "mode"), edge(branch, writer, kind="control")]
    return MechanismDAG(dag_id="context-contract", terminal="root", vertices=vertices, edges=edges)


@pytest.fixture(params=["ordinary", "local"])
def session_for(request):
    """Identical graph, inputs and assertions at the shared value boundary."""
    def bind(dag, *, samples=None, **kwargs):
        context = DAGValueContext(conditional_equations=True) if request.param == "local" else None
        return DAGValueProgram(dag).bind(
            sample_resolver=lambda signal, timestamp: (samples or {}).get((signal, timestamp)),
            context=context, **kwargs,
        )
    return bind


def test_shared_guarded_producers_follow_samples(session_for):
    dag = graph(guarded=True)
    samples = {(f"packet.{field}", t): value for t, mode in [(0., 0), (1., 1)]
               for field, value in [("input", 3.), ("mode", mode)]}
    session = session_for(dag, samples=samples)
    assert [session.evaluate("root", t).value for t in [0., 1.]] == [8., 10.]


def test_shared_ordered_writer_supersedes_earlier_missing_input(session_for):
    dag = graph()
    dag.edges = [e for e in dag.edges if e.target_id != "first"]
    session = session_for(dag, samples={("packet.input", 0.): 3.})
    result = session.evaluate("root", 0.)
    assert result.status == "value" and result.value == 10.
    assert not result.issues


@pytest.mark.parametrize("ambiguity", ["unknown_guard", "tied_missing", "tied_different", "other_scope", "missing_vertex"])
def test_shared_ambiguous_writers_remain_unresolved(session_for, ambiguity):
    dag = graph(guarded=ambiguity == "unknown_guard")
    first, second = (next(v for v in dag.vertices if v.id == name) for name in ["first", "second"])
    if ambiguity.startswith("tied"):
        second.line = first.line
        second.metadata["target_scope"]["line"] = first.line
    if ambiguity == "tied_missing":
        dag.edges = [e for e in dag.edges if e.target_id != "second"]
    if ambiguity == "other_scope":
        second.metadata["target_scope"]["callable"] = "Controller::other"
    if ambiguity == "missing_vertex":
        dag.vertices = [v for v in dag.vertices if v.id != "second"]
    result = session_for(dag, samples={("packet.input", 0.): 3.}).evaluate("root", 0.)
    assert result.status == "unresolved"
    assert any(i.kind == "writer_coverage" and i.vertex_id == "root" for i in result.issues)


def test_conditional_results_cannot_select_an_unknown_alternative():
    dag = graph(guarded=True)
    session = DAGValueProgram(dag).bind(
        sample_resolver=lambda signal, _timestamp: 3. if signal == "packet.input" else None,
        context=DAGValueContext(conditional_equations=True),
    )
    conditional = session.evaluate("first", 0.)
    assert conditional.status == "value" and conditional.value == 4.
    assert conditional.conditional_writer_ids == {"first"}
    assert session.evaluate("root", 0.).status == "unresolved"
    session.release_timestamp_values()
    assert not session._conditional_cache and not session._value_cache


def test_context_observation_stops_pending_history_without_affecting_ordinary_session():
    dag = graph()
    program = DAGValueProgram(dag)
    context = DAGValueContext(
        observation_resolver=lambda name, _timestamp: DAGValueResult(
            "value", value=3., observed_vertex_ids=frozenset({name}),
        ) if name == "signal" else None,
        pending_vertices=frozenset({"signal"}), conditional_equations=True,
    )
    result = program.bind(context=context).evaluate("root", 0.)
    assert result.status == "value" and result.value == 10.
    assert result.observed_vertex_ids == {"signal"}
    assert program.bind().evaluate("root", 0.).status == "unresolved"


@pytest.mark.parametrize("condition", ["false", "true"])
def test_conditional_subscription_needs_state_alignment_even_with_known_gate(condition):
    dag = graph()
    first = next(v for v in dag.vertices if v.id == "first")
    first.metadata.update(synthetic_boundary_transfer=True, boundary_direction="subscribe",
                          reachability={"exact": True, "all_of": [condition]})
    dag.vertices.append(DAGVertex(id="gate", kind="branch", predicate_raw=condition,
                                  metadata={"expression_inputs_exact": True}))
    dag.edges.append(edge("gate", "first", kind="control"))
    result = DAGValueProgram(dag).bind(
        context=DAGValueContext(conditional_equations=True),
        sample_resolver=lambda *_: 3.,
    ).evaluate("first", 0.)
    assert result.status == "unresolved"
    assert result.issues[0].kind == "state_alignment"


def test_context_does_not_accept_assumed_gate_as_writer_selection():
    dag = graph(guarded=True)
    for branch in (v for v in dag.vertices if v.kind == "branch"):
        branch.feasibility_verdict = "always_true" if branch.id == "gate-first" else "always_false"
        branch.metadata["static_evaluation"] = {"assumed": True}
    result = DAGValueProgram(dag).bind(
        context=DAGValueContext(conditional_equations=True), sample_resolver=lambda *_: 3.,
    ).evaluate("root", 0.)
    assert result.status == "unresolved"


def test_context_cannot_use_output_signal_as_an_input():
    result = DAGValueProgram(graph()).bind(
        context=DAGValueContext(forbidden_signals=frozenset({"packet.input"})),
        sample_resolver=lambda *_: 3.,
    ).evaluate("root", 0.)
    assert result.status == "unresolved"


def test_selected_writer_retains_its_observed_guard_input():
    dag = graph(guarded=True)
    result = DAGValueProgram(dag).bind(
        sample_resolver=lambda *_: 3.,
        context=DAGValueContext(
            observation_resolver=lambda name, _t: DAGValueResult(
                "value", value=0, observed_vertex_ids=frozenset({name}),
            ) if name == "mode" else None,
            conditional_equations=True,
        ),
    ).evaluate("root", 0.)
    assert result.status == "value" and result.value == 8.
    assert result.observed_vertex_ids == {"mode"}


def test_context_rechecks_branch_outside_annotated_domain():
    dag = graph(guarded=True)
    for branch in (v for v in dag.vertices if v.kind == "branch"):
        branch.feasibility_verdict = "always_true" if branch.id == "gate-first" else "always_false"
        branch.metadata["evaluation_domain"] = [0., 1.]
    result = DAGValueProgram(dag).bind(
        context=DAGValueContext(conditional_equations=True),
        sample_resolver=lambda signal, _timestamp: 1 if signal == "packet.mode" else 3.,
    ).evaluate("root", 2.)
    assert result.status == "value" and result.value == 10.


def test_shared_nested_compile_failure_keeps_its_site_and_reason(session_for):
    dag = graph(guarded=True)
    branch = next(v for v in dag.vertices if v.id == "gate-first")
    branch.predicate_raw = "receiver.update(&value)"
    result = session_for(dag, samples={("packet.input", 0.): 3.}).evaluate("root", 0.)
    assert result.status == "unresolved"
    issue = next(i for i in result.issues if i.vertex_id == branch.id and i.kind == "expression")
    assert issue.reason == "invalid expression syntax"


@pytest.mark.parametrize("pruned", [False, True])
def test_skipped_transfer_does_not_prove_persisted_receiver_equals_initializer(session_for, pruned):
    dag = graph(guarded=True)
    first = next(v for v in dag.vertices if v.id == "first")
    second = next(v for v in dag.vertices if v.id == "second")
    first.metadata["reachability"] = {"exact": True, "all_of": []}
    dag.edges = [e for e in dag.edges if not (e.target_id == "first" and e.kind == "control")]
    second.metadata.update(synthetic_boundary_transfer=True, boundary_direction="subscribe",
                           boundary_transfer_event_id="gate-second")
    if pruned:
        from flight_log_agent.analysis.mechanism_dag import prune_infeasible_operations
        next(v for v in dag.vertices if v.id == "gate-second").feasibility_verdict = "always_false"
        dag = prune_infeasible_operations(dag)
    # mode=0 disproves a transfer at this timestamp, not at all prior calls.
    result = session_for(dag, samples={("packet.input", 0.): 3., ("packet.mode", 0.): 0}).evaluate("root", 0.)
    assert result.status == "unresolved"
    assert any(issue.kind == "state_alignment" for issue in result.issues)


def single_input_graph(*, leaf_metadata):
    leaf = DAGVertex(id="reading", kind="evidence", sub_kind="logged_signal",
                     signal_name="data.value", metadata=dict(leaf_metadata))
    return MechanismDAG(
        dag_id="observation-validity", terminal="total",
        vertices=[leaf, operation("total", "reading + 1", 10)],
        edges=[edge("reading", "total", "reading")],
    )


def test_shared_type_only_observation_is_not_usable_evidence(session_for):
    """grounded_via=declared_type stays a candidate in every evaluation context.

    The context-free ordinary case is the pre-fix RED regression (it promoted
    the leaf to a value); the local-context case already rejected before the
    fix and is parity/contract coverage.
    """
    dag = single_input_graph(leaf_metadata={"observation": "observed", "grounded_via": "declared_type"})
    result = session_for(dag, samples={("data.value", 0.): 10.}).evaluate("total", 0.)
    assert result.status == "unresolved"
    assert any(issue.kind == "observation_binding" for issue in result.issues)


def test_shared_proven_observation_remains_usable(session_for):
    dag = single_input_graph(leaf_metadata={"observation": "observed"})
    result = session_for(dag, samples={("data.value", 0.): 10.}).evaluate("total", 0.)
    assert result.status == "value"
    assert result.value == 11.


def test_shared_parameter_leaf_still_resolves(session_for):
    leaf = DAGVertex(id="limit", kind="evidence", sub_kind="parameter", signal_name="SYNTH_LIMIT")
    dag = MechanismDAG(
        dag_id="observation-validity-param", terminal="total",
        vertices=[leaf, operation("total", "limit + 1", 10)],
        edges=[edge("limit", "total", "limit")],
    )
    result = session_for(dag, parameter_values={"SYNTH_LIMIT": 4.}).evaluate("total", None)
    assert result.status == "value"
    assert result.value == 5.


def _cyclic_pair_dag():
    return MechanismDAG(
        dag_id="value-cycle", terminal="a",
        vertices=[operation("a", "bval + 1", 10), operation("b", "aval + 1", 20)],
        edges=[edge("b", "a", "bval"), edge("a", "b", "aval")],
    )


def test_direct_value_cycle_stays_unresolved_without_value():
    """A → B → A must terminate without inventing a value. The cyclic
    detection itself carries no issue payload, so the wrapped parents
    surface producer-unresolved reasons and no cyclic-labeled issue."""
    dag = _cyclic_pair_dag()
    result = DAGValueProgram(dag).bind().evaluate("a", 0.)
    assert result.status == "unresolved"
    assert result.value is None
    assert result.reason == "bval: all reaching producers are unresolved"
    assert result.issues == ()


def test_inactive_operation_skips_unresolvable_operand():
    """A falsy control gate renders the operation inactive before expression
    evaluation, so an operand with no producer must not surface."""
    dag = graph(guarded=True)
    gated = next(v for v in dag.vertices if v.id == "first")
    gated.metadata["expression_inputs_exact"] = True
    gated.expression = "ghost + 1"
    result = DAGValueProgram(dag).bind(
        sample_resolver=lambda signal, timestamp: {(f"packet.{field}", timestamp): value
                                                   for field, value in [("input", 3.), ("mode", 1.)]
                                                   }.get((signal, timestamp)),
    ).evaluate("first", 0.)
    assert result.status == "inactive"
    assert not [issue for issue in result.issues if issue.kind == "source_linkage"]


def test_first_missing_producer_determines_failure():
    """Operands resolve in expression traversal order: the first operand
    whose producers are all missing fails the vertex with a role-prefixed
    reason, and later operands stay untouched (no issue for bbb)."""
    dag = MechanismDAG(
        dag_id="first-failure", terminal="total",
        vertices=[operation("total", "aaa + bbb", 10)],
        edges=[edge("ghost_a", "total", "aaa"), edge("ghost_b", "total", "bbb")],
    )
    result = DAGValueProgram(dag).bind().evaluate("total", 0.)
    assert result.status == "unresolved"
    assert result.reason == "aaa: all reaching producers are unresolved"
    assert [(i.kind, i.vertex_id, i.operand) for i in result.issues] == [
        ("source_linkage", "ghost_a", "")]


def test_repeated_operand_resolves_once_with_correct_value():
    """A repeated operand keeps one resolver fetch per signal while each
    occurrence still contributes to the value."""
    dag = MechanismDAG(
        dag_id="repeated-operand", terminal="total",
        vertices=[
            DAGVertex(id="signal", kind="evidence", sub_kind="logged_signal",
                      signal_name="packet.input"),
            operation("total", "reading + reading", 10),
        ],
        edges=[edge("signal", "total", "reading")],
    )
    fetches = []
    session = DAGValueProgram(dag).bind(
        sample_resolver=lambda signal, timestamp: (
            fetches.append((signal, timestamp)) or 3.),
    )
    result = session.evaluate("total", 0.)
    assert result.status == "value"
    assert result.value == 6.
    assert fetches == [("packet.input", 0.)]


def test_shared_unresolved_child_produces_single_issue():
    """One shared unresolved dependency reached through two parents must not
    duplicate its semantic issue."""
    dag = MechanismDAG(
        dag_id="shared-unresolved", terminal="top",
        vertices=[
            operation("shared", "aaa + 1", 5),
            operation("left", "shared + 1", 10),
            operation("right", "shared + 2", 20),
            operation("top", "left + right", 30),
        ],
        edges=[
            edge("ghost_v", "shared", "aaa"),
            edge("shared", "left", "shared"), edge("shared", "right", "shared"),
            edge("left", "top", "left"), edge("right", "top", "right"),
        ],
    )
    result = DAGValueProgram(dag).bind().evaluate("top", 0.)
    assert result.status == "unresolved"
    linkage = [issue for issue in result.issues if issue.kind == "source_linkage"]
    assert [(issue.vertex_id, issue.operand) for issue in linkage] == [("ghost_v", "")]


def _deep_chain_dag(depth):
    vertices = [DAGVertex(id="c0", kind="evidence", sub_kind="constant",
                          signal_name="base", metadata={"value": 0.0})]
    edges = []
    for level in range(1, depth + 1):
        vertices.append(operation(f"c{level}", "prev + 1", level))
        edges.append(edge(f"c{level - 1}", f"c{level}", "prev"))
    return MechanismDAG(dag_id="deep-chain", terminal=f"c{depth}",
                        vertices=vertices, edges=edges)


def test_deep_acyclic_chain_evaluates_without_recursion_limit():
    """A chain far deeper than the old recursive limit must evaluate to the
    exact semantic value: stack depth is not a correctness limit."""
    depth = 500
    result = DAGValueProgram(_deep_chain_dag(depth)).bind().evaluate(
        f"c{depth}", None)
    assert result.status == "value"
    assert result.value == float(depth)


def _deep_diamond_dag(depth, width=2):
    vertices = [DAGVertex(id="leaf", kind="evidence", sub_kind="constant",
                          signal_name="base", metadata={"value": 1.0})]
    edges = []
    level = ["leaf"]
    for layer in range(depth):
        current = []
        for slot in range(width):
            name = f"n{layer}_{slot}"
            vertices.append(operation(
                name, " + ".join(f"p{index}" for index in range(len(level))),
                layer))
            current.append(name)
        for node in current:
            for index, parent in enumerate(level):
                edges.append(edge(parent, node, f"p{index}"))
        level = current
    vertices.append(operation(
        "top", " + ".join(f"q{index}" for index in range(len(level))), depth))
    for index, parent in enumerate(level):
        edges.append(edge(parent, "top", f"q{index}"))
    return MechanismDAG(dag_id="deep-diamond", terminal="top",
                        vertices=vertices, edges=edges)


def test_deep_shared_diamond_evaluates_exact_result():
    """Depth plus true sharing (each node feeds two children) must produce
    the exact deterministic value with no duplicate semantic work."""
    depth = 150
    result = DAGValueProgram(_deep_diamond_dag(depth)).bind().evaluate(
        "top", None)
    assert result.status == "value"
    assert result.value == 2.0 ** depth


def test_computation_error_keeps_expression_issue():
    """An expression that fails after valued operands resolve keeps an
    expression issue rather than dropping the failure."""
    dag = MechanismDAG(
        dag_id="computation-error", terminal="total",
        vertices=[
            DAGVertex(id="one", kind="evidence", sub_kind="constant",
                      signal_name="one", metadata={"value": 1.0}),
            DAGVertex(id="zero", kind="evidence", sub_kind="constant",
                      signal_name="zero", metadata={"value": 0.0}),
            operation("total", "num / den", 10),
        ],
        edges=[edge("one", "total", "num"), edge("zero", "total", "den")],
    )
    result = DAGValueProgram(dag).bind().evaluate("total", None)
    assert result.status == "unresolved"
    assert result.reason == "division by zero"
    assert [(i.kind, i.vertex_id) for i in result.issues] == [
        ("expression", "total")]
