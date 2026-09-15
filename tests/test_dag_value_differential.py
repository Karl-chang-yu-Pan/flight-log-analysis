"""TEMPORARY T4 differential harness: recursive vs iterative evaluator.

Do NOT treat this file as permanent architecture. T5 owns cutover and must
delete this harness together with the obsolete recursive evaluator. It exists
only to prove zero unexplained semantic differences between the two engines
over the representative corpus, each run from equivalent fresh session state.
"""

from __future__ import annotations

import pytest

from flight_log_agent.analysis.dag_value import (
    DAGValueContext,
    DAGValueProgram,
    DAGValueResult,
)
from flight_log_agent.analysis.mechanism_dag import DAGEdge, DAGVertex, MechanismDAG


def edge(source, target, role):
    return DAGEdge(
        id=f"{source}:{target}:{role}:data",
        source_id=source, target_id=target, role=role, kind="data",
    )


def operation(name, text, line, **metadata):
    base = {"expression_inputs_exact": True,
            "reachability": {"exact": True, "all_of": []},
            "target_scope": {"file": "t.cpp", "callable": "f", "line": line}}
    base.update(metadata)
    return DAGVertex(id=name, kind="operation", file="t.cpp", line=line,
                     expression=text, metadata=base)


def const_leaf(name, signal, value):
    return DAGVertex(id=name, kind="evidence", sub_kind="constant",
                     signal_name=signal, metadata={"value": value})


def branch(name, predicate):
    return DAGVertex(id=name, kind="branch", predicate_raw=predicate,
                     metadata={"expression_inputs_exact": True})


def logged_leaf(name, signal):
    return DAGVertex(id=name, kind="evidence", sub_kind="logged_signal",
                     signal_name=signal)


def freeze(result):
    return (result.status, result.value, result.reason,
            tuple((i.kind, i.vertex_id, i.operand, tuple(i.producer_ids),
                   i.reason)
                  for i in result.issues),
            tuple(sorted(result.observed_vertex_ids)),
            tuple(sorted(result.conditional_writer_ids)))


def check_both(dag, terminal, timestamp=None, samples=None,
               parameters=None, enums=None, context=None):
    """Run both engines from equivalent fresh sessions; compare everything."""
    samples = samples or {}
    program = DAGValueProgram(dag)
    fetches = []

    def fresh():
        log = []
        fetches.append(log)
        return program.bind(
            parameter_values=parameters, enum_values=enums,
            sample_resolver=lambda signal, ts: (
                log.append((signal, ts)) or samples.get((signal, ts))),
            context=context,
        )

    recursive = fresh().evaluate(terminal, timestamp)
    iterative = fresh()._evaluate_iterative(terminal, timestamp)
    assert freeze(iterative) == freeze(recursive), (
        terminal, freeze(recursive), freeze(iterative))
    assert fetches[0] == fetches[1], (terminal, fetches)
    return recursive


def test_leaves():
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("c", "k", 2.0), operation("t", "k + 3", 1)],
        edges=[edge("c", "t", "k")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[DAGVertex(id="p", kind="evidence", sub_kind="parameter",
                            signal_name="LIM"),
                  operation("t", "limit + 1", 1)],
        edges=[edge("p", "t", "limit")]), "t", parameters={"LIM": 4.0})
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[DAGVertex(id="e", kind="evidence", sub_kind="opaque_symbol",
                            signal_name="MODE_X"),
                  operation("t", "mode + 1", 1)],
        edges=[edge("e", "t", "mode")]), "t", enums={"MODE_X": 3})
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[logged_leaf("s", "packet.in"), operation("t", "x * 2", 1)],
        edges=[edge("s", "t", "x")]), "t", 0.0, {("packet.in", 0.0): 5.0})
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[logged_leaf("s", "packet.in"), operation("t", "x * 2", 1)],
        edges=[edge("s", "t", "x")]), "t", 0.0, {("packet.in", 0.0): 5.0},
        context=DAGValueContext(forbidden_signals=frozenset({"packet.in"})))


def test_arithmetic_and_errors():
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("a", "a", 2.0), const_leaf("b", "b", 3.0),
                  operation("m", "a + b", 1), operation("t", "m * 4", 2)],
        edges=[edge("a", "m", "a"), edge("b", "m", "b"),
               edge("m", "t", "m")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("a", "a", 1.0), const_leaf("b", "b", 0.0),
                  operation("t", "a / b", 1)],
        edges=[edge("a", "t", "a"), edge("b", "t", "b")]), "t")
    ctx = DAGValueContext(
        observation_resolver=lambda vid, ts: None,
        conditional_equations=True)
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[logged_leaf("s", "packet.in"),
                  const_leaf("z", "z", 0.0), operation("t", "x / z", 1)],
        edges=[edge("s", "t", "x"), edge("z", "t", "z")]),
        "t", 0.0, {("packet.in", 0.0): 4.0}, context=ctx)


def test_missing_and_multi_producer_failure():
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[operation("t", "aaa + bbb", 1)],
        edges=[edge("ghost_a", "t", "aaa"), edge("ghost_b", "t", "bbb")]),
        "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[operation("t", "x + 1", 1)],
        edges=[edge("p1", "t", "x"), edge("p2", "t", "x")]), "t")


def test_producer_selection():
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("c", "c", 7.0), operation("t", "v * 2", 1)],
        edges=[edge("c", "t", "v")]), "t")
    early = operation("p1", "9", 10)
    late = operation("p2", "1", 20)
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[early, late, operation("t", "v", 30)],
        edges=[edge("p1", "t", "v"), edge("p2", "t", "v")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("a", "a", 1.0), const_leaf("b", "b", 2.0),
                  operation("t", "v", 1)],
        edges=[edge("a", "t", "v"), edge("b", "t", "v")]), "t")
    ctx = DAGValueContext(conditional_equations=True)
    exact_all = {"exact": True, "all_of": ["g"]}
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("k", "k", 1.0),
                  operation("p", "k + 1", 5, reachability=exact_all),
                  operation("t", "v * 3", 9)],
        edges=[edge("k", "p", "k"), edge("p", "t", "v")]),
        "t", context=ctx)
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[operation("t", "v", 1)], edges=[]), "t")


def test_activity_controls():
    gate_true = branch("g", "1 == 1")
    gate_false = branch("h", "1 == 2")
    gate_fail = branch("f", "ghost > 0")
    body = operation("t", "k + 1", 50)
    const = const_leaf("k", "k", 1.0)
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[gate_true, body, const],
        edges=[edge("k", "t", "k"),
               DAGEdge(id="c", source_id="g", target_id="t",
                       role="g", kind="control")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[gate_false, body, const],
        edges=[edge("k", "t", "k"),
               DAGEdge(id="c", source_id="h", target_id="t",
                       role="h", kind="control")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[gate_fail, gate_false, body, const],
        edges=[edge("k", "t", "k"),
               DAGEdge(id="c1", source_id="f", target_id="t",
                       role="f", kind="control"),
               DAGEdge(id="c2", source_id="h", target_id="t",
                       role="h", kind="control")]), "t")
    inexact = operation("u", "k + 1", 60,
                        reachability={"exact": False, "all_of": []})
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[inexact, const],
        edges=[edge("k", "u", "k")]), "u")


def test_expression_short_circuit():
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("a", "a", False), operation("t", "x and y", 1)],
        edges=[edge("a", "t", "x"),
               edge("ghost", "t", "y")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("a", "a", True), operation("t", "x or y", 1)],
        edges=[edge("a", "t", "x"),
               edge("ghost", "t", "y")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("c", "c", True), operation("t", "x if c else y", 1)],
        edges=[edge("c", "t", "c"),
               edge("ghost", "t", "y")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[logged_leaf("s", "packet.in"), operation("t", "x + x", 1)],
        edges=[edge("s", "t", "x")]), "t", 0.0, {("packet.in", 0.0): 3.0})


def test_repeated_call_occurrences():
    dag = MechanismDAG(
        dag_id="calls", terminal="output",
        vertices=[
            DAGVertex(id="first", kind="evidence", sub_kind="constant",
                      signal_name="first result", metadata={"value": 2.0}),
            DAGVertex(id="second", kind="evidence", sub_kind="constant",
                      signal_name="second result", metadata={"value": 5.0}),
            DAGVertex(
                id="output", kind="operation", variable="output",
                expression="reader() + reader()",
                metadata={
                    "source_expression": "reader() + reader()",
                    "source_expression_ref": {
                        "text": "reader() + reader()",
                        "lowered_text": "reader() + reader()",
                        "input_symbols": [], "input_identities": {},
                        "call_results": [
                            {"call_source_site_id": "first-site",
                             "text": "reader()"},
                            {"call_source_site_id": "second-site",
                             "text": "reader()"},
                        ],
                        "exact": True,
                    },
                    "expression_inputs_exact": True,
                    "reachability": {"exact": True},
                    "source_call_roles": {
                        "first-site": {"call": "reader()",
                                       "result": "reader()"},
                        "second-site": {"call": "reader()",
                                        "result": "reader()"},
                    },
                },
            ),
        ],
        edges=[
            DAGEdge(id="first-edge", source_id="first", target_id="output",
                    kind="data", role="call:reader", via="first-site"),
            DAGEdge(id="second-edge", source_id="second", target_id="output",
                    kind="data", role="call:reader", via="second-site"),
        ],
    )
    check_both(dag, "output")


def test_branch_fallback():
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[branch("g", "1 == 1")]), "g")
    check_both(MechanismDAG(
        dag_id="t", terminal="t", vertices=[branch("g", "1 == 2")]), "g")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[logged_leaf("m", "packet.mode"), branch("g", "mode == 0")],
        edges=[edge("m", "g", "mode")]), "g", 0.0, {("packet.mode", 0.0): 0})


def test_cycles():
    check_both(MechanismDAG(
        dag_id="t", terminal="a",
        vertices=[operation("a", "bval + 1", 10),
                  operation("b", "aval + 1", 20)],
        edges=[edge("b", "a", "bval"), edge("a", "b", "aval")]), "a", 0.)
    check_both(MechanismDAG(
        dag_id="t", terminal="a",
        vertices=[operation("a", "bval + 1", 10),
                  operation("b", "cval + 1", 20),
                  operation("c", "bval + 1", 30)],
        edges=[edge("b", "a", "bval"), edge("c", "b", "cval"),
               edge("b", "c", "bval")]), "a", 0.)


def test_shared_diamond():
    check_both(MechanismDAG(
        dag_id="t", terminal="top",
        vertices=[const_leaf("leaf", "base", 1.0),
                  operation("left", "s + 1", 10),
                  operation("right", "s + 2", 20),
                  operation("top", "left + right", 30)],
        edges=[edge("leaf", "left", "s"), edge("leaf", "right", "s"),
               edge("left", "top", "left"), edge("right", "top", "right")]),
        "top")
    check_both(MechanismDAG(
        dag_id="t", terminal="top",
        vertices=[operation("shared", "aaa + 1", 5),
                  operation("left", "shared + 1", 10),
                  operation("right", "shared + 2", 20),
                  operation("top", "left + right", 30)],
        edges=[edge("ghost", "shared", "aaa"),
               edge("shared", "left", "shared"),
               edge("shared", "right", "shared"),
               edge("left", "top", "left"),
               edge("right", "top", "right")]), "top")


def test_helper_persisted_pending():
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("c", "c", 5.0),
                  DAGVertex(id="h", kind="evidence",
                            sub_kind="helper_parameter", signal_name=" formal"),
                  operation("t", "held + 1", 9)],
        edges=[edge("c", "h", "actual"), edge("h", "t", "held")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[operation("t", "k + 1", 9),
                  const_leaf("k", "k", 1.0)],
        edges=[edge("k", "t", "k")]), "t")
    persisted = operation("t", "k + 1", 9,
                          boundary_transfer_event_id="evt-1")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[persisted, const_leaf("k", "k", 1.0)],
        edges=[edge("k", "t", "k")]), "t")
    check_both(MechanismDAG(
        dag_id="t", terminal="t",
        vertices=[const_leaf("k", "k", 1.0), operation("t", "k + 1", 9)],
        edges=[edge("k", "t", "k")]), "t",
        context=DAGValueContext(pending_vertices=frozenset({"t"})))


def test_ordinary_conditional_parity():
    inexact = operation("u", "k + 1", 60,
                        reachability={"exact": True, "all_of": ["g"]})
    dag = lambda: MechanismDAG(  # noqa: E731
        dag_id="t", terminal="t",
        vertices=[inexact, const_leaf("k", "k", 1.0),
                  operation("t", "v * 3", 9)],
        edges=[edge("k", "u", "k"), edge("u", "t", "v")])
    check_both(dag(), "t")
    check_both(dag(), "t", context=DAGValueContext(conditional_equations=True))


# TEMPORARY T4 differential harness — delete in T5 (see module header).
def test_all_producers_inactive_selection():
    def gated(name, line):
        gate = branch(f"g{name}", "1 == 2")
        gate.feasibility_verdict = "always_false"
        return gate, operation(name, "k + 0", line)
    g1, op1 = gated("p1", 10)
    g2, op2 = gated("p2", 20)
    dag = lambda: MechanismDAG(  # noqa: E731
        dag_id="t", terminal="t",
        vertices=[g1, op1, const_leaf("k1", "k", 1.0),
                  g2, op2, const_leaf("k2", "k", 1.0),
                  operation("t", "v", 30)],
        edges=[edge("k1", "p1", "k"), edge("k2", "p2", "k"),
               edge("p1", "t", "v"), edge("p2", "t", "v"),
               DAGEdge(id="c1", source_id="gp1", target_id="p1",
                       role="g", kind="control"),
               DAGEdge(id="c2", source_id="gp2", target_id="p2",
                       role="g", kind="control")])
    check_both(dag(), "t")


def test_observation_resolver_hit():
    ctx = DAGValueContext(
        observation_resolver=lambda vid, ts: (
            DAGValueResult("value", value=9.0,
                           observed_vertex_ids=frozenset({vid}))
            if vid == "s" else None),
        conditional_equations=True)
    dag = lambda: MechanismDAG(  # noqa: E731
        dag_id="t", terminal="t",
        vertices=[logged_leaf("s", "packet.in"), operation("t", "x * 2", 1)],
        edges=[edge("s", "t", "x")])
    check_both(dag(), "t", 0.0, context=ctx)


def test_branch_fast_path_verdicts():
    always_true = branch("g", "x")
    always_true.feasibility_verdict = "always_true"
    check_both(MechanismDAG(
        dag_id="t", terminal="g", vertices=[always_true]), "g")
    always_false = branch("g", "x")
    always_false.feasibility_verdict = "always_false"
    check_both(MechanismDAG(
        dag_id="t", terminal="g", vertices=[always_false]), "g")
    windowed = branch("g", "x")
    windowed.active_windows = [(0.0, 10.0)]
    check_both(MechanismDAG(
        dag_id="t", terminal="g", vertices=[windowed]), "g", 5.0)
    windowed_out = branch("g", "x")
    windowed_out.metadata = dict(
        windowed_out.metadata, evaluation_domain=[20.0, 30.0])
    check_both(MechanismDAG(
        dag_id="t", terminal="g", vertices=[windowed_out]), "g", 5.0)
