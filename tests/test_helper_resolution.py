from __future__ import annotations

import pytest

from flight_log_agent.analysis.helper_resolution import (
    HelperRegistry,
    lower_helper_call,
    substitute_helpers,
)


def _ref(
    name: str,
    *,
    parameters: list[str] | None = None,
    return_expression: str | None = None,
    lowered_return_expression: str | None = None,
    branches: list[dict] | None = None,
    unresolved_reason: str | None = None,
) -> dict:
    """Build a HelperExpressionRef-shaped dict for tests."""
    return {
        "name": name,
        "file": "fake.cpp",
        "line": 1,
        "evidence": "",
        "parameters": parameters or [],
        "statements": [],
        "assignments": {},
        "return_expression": return_expression,
        "lowered_return_expression": lowered_return_expression,
        "branches": branches or [],
        "symbol_bindings": {},
        "call_resolutions": [],
        "helper_calls": [],
        "unresolved_reason": unresolved_reason,
    }


class TestHelperRegistry:
    def test_get_by_full_name(self):
        registry = HelperRegistry([_ref("Navigator::get_acceptance_radius")])
        assert registry.get("Navigator::get_acceptance_radius") is not None

    def test_get_by_short_name_when_unique(self):
        registry = HelperRegistry([_ref("Navigator::get_acceptance_radius")])
        assert registry.get("get_acceptance_radius") is not None

    def test_short_name_collision_returns_none(self):
        registry = HelperRegistry([
            _ref("ClassA::compute"),
            _ref("ClassB::compute"),
        ])
        assert registry.get("compute") is None
        assert registry.get("ClassA::compute") is not None
        assert registry.get("ClassB::compute") is not None

    def test_empty_inputs_handled(self):
        registry = HelperRegistry([_ref(""), None, {}])  # type: ignore[list-item]
        assert registry.get("") is None
        assert registry.get("anything") is None

    def test_contains_membership(self):
        registry = HelperRegistry([_ref("foo")])
        assert "foo" in registry
        assert "bar" not in registry

    def test_from_index_substitutes_resolved_body(self):
        # When the BindingIndex has materialized a helper's resolved
        # return expression, HelperRegistry.from_index swaps the
        # helper's lowered_return_expression with that resolved form.

        class _StubResult:
            expression = "vehicle_global_position.alt"

        class _StubBindingIndex:
            helper_resolutions = {"get_alt": _StubResult()}

        helper_expressions = [
            {
                "name": "get_alt",
                "lowered_return_expression": "nested(call(chain))",
                "parameters": [],
            }
        ]
        registry = HelperRegistry.from_index(_StubBindingIndex(), helper_expressions)
        record = registry.get("get_alt")
        assert record is not None
        assert record["lowered_return_expression"] == "vehicle_global_position.alt"

    def test_from_index_with_no_resolutions_keeps_original_body(self):
        helper_expressions = [
            {"name": "untouched", "lowered_return_expression": "original_body"},
        ]

        class _Empty:
            helper_resolutions = {}

        registry = HelperRegistry.from_index(_Empty(), helper_expressions)
        record = registry.get("untouched")
        assert record is not None
        assert record["lowered_return_expression"] == "original_body"


class TestLowerHelperCall:
    def test_single_return_substitutes_parameters(self):
        # A helper that returns ``a + b``.
        registry = HelperRegistry([_ref(
            "add",
            parameters=["a", "b"],
            lowered_return_expression="a + b",
        )])
        result = lower_helper_call(
            "add",
            ["x", "y"],
            registry=registry,
            env={},
        )
        assert result == "(x) + (y)"

    def test_falls_back_to_return_expression_when_lowered_missing(self):
        registry = HelperRegistry([_ref(
            "mult",
            parameters=["a", "b"],
            return_expression="a * b",
        )])
        result = lower_helper_call("mult", ["1", "2"], registry=registry, env={})
        assert result == "(1) * (2)"

    def test_arg_count_mismatch_returns_none(self):
        registry = HelperRegistry([_ref(
            "add", parameters=["a", "b"], lowered_return_expression="a + b"
        )])
        assert lower_helper_call("add", ["x"], registry=registry, env={}) is None
        assert lower_helper_call("add", ["x", "y", "z"], registry=registry, env={}) is None

    def test_helper_with_unresolved_reason_returns_none(self):
        registry = HelperRegistry([_ref(
            "stateful",
            parameters=["x"],
            lowered_return_expression="x",
            unresolved_reason="helper body mutates pointer output",
        )])
        assert lower_helper_call("stateful", ["1"], registry=registry, env={}) is None

    def test_branched_helper_picks_first_matching_condition(self):
        registry = HelperRegistry([_ref(
            "acceptance_radius",
            parameters=[],
            branches=[
                {"condition": "vehicle_type == 1", "expression": "NAV_ACC_RAD"},
                {"condition": "vehicle_type == 2", "expression": "FW_NAV_ACC_RAD"},
                {"condition": "default", "expression": "10.0"},
            ],
        )])
        # Rotary-wing branch is active.
        result = lower_helper_call(
            "acceptance_radius",
            [],
            registry=registry,
            env={"vehicle_type": 1, "NAV_ACC_RAD": 10.0},
        )
        assert result == "NAV_ACC_RAD"

    def test_branched_helper_falls_through_to_default(self):
        registry = HelperRegistry([_ref(
            "f",
            parameters=[],
            branches=[
                {"condition": "x > 100", "expression": "1.0"},
                {"condition": "default", "expression": "0.0"},
            ],
        )])
        result = lower_helper_call("f", [], registry=registry, env={"x": 5})
        assert result == "0.0"

    def test_branched_helper_with_no_match_and_no_default_returns_none(self):
        registry = HelperRegistry([_ref(
            "f",
            parameters=[],
            branches=[
                {"condition": "x > 100", "expression": "1.0"},
                {"condition": "x < -100", "expression": "-1.0"},
            ],
        )])
        result = lower_helper_call("f", [], registry=registry, env={"x": 5})
        assert result is None

    def test_unknown_helper_returns_none(self):
        registry = HelperRegistry([])
        assert lower_helper_call("missing", [], registry=registry, env={}) is None

    def test_parameters_substituted_longest_first(self):
        # ``lat`` should NOT be substituted inside ``latitude``.
        registry = HelperRegistry([_ref(
            "f",
            parameters=["lat", "latitude"],
            lowered_return_expression="lat + latitude",
        )])
        result = lower_helper_call("f", ["A", "B"], registry=registry, env={})
        # Both formal params are replaced cleanly, no partial overwrite.
        assert result == "(A) + (B)"


class TestSubstituteHelpers:
    def test_simple_call_in_arithmetic(self):
        registry = HelperRegistry([_ref(
            "double",
            parameters=["x"],
            lowered_return_expression="x * 2",
        )])
        result = substitute_helpers(
            "alt + double(5)",
            registry=registry,
            env={},
        )
        assert result == "alt + ((5) * 2)"

    def test_nested_helper_resolved_iteratively(self):
        registry = HelperRegistry([
            _ref(
                "outer",
                parameters=["x"],
                lowered_return_expression="inner(x) + 1",
            ),
            _ref(
                "inner",
                parameters=["y"],
                lowered_return_expression="y * 2",
            ),
        ])
        result = substitute_helpers(
            "outer(3)",
            registry=registry,
            env={},
        )
        # outer(3) -> inner(3) + 1 -> (3 * 2) + 1
        assert "1" in result and "*" in result

    def test_unknown_helper_pass_through(self):
        registry = HelperRegistry([])
        result = substitute_helpers(
            "unknown(a, b) + 1",
            registry=registry,
            env={},
        )
        assert result == "unknown(a, b) + 1"

    def test_helper_calls_through_namespace_separator(self):
        registry = HelperRegistry([_ref(
            "Navigator::get_acceptance_radius",
            parameters=[],
            lowered_return_expression="NAV_ACC_RAD",
        )])
        result = substitute_helpers(
            "Navigator::get_acceptance_radius()",
            registry=registry,
            env={},
        )
        assert result == "(NAV_ACC_RAD)"

    def test_helper_called_via_short_name(self):
        registry = HelperRegistry([_ref(
            "Navigator::get_acceptance_radius",
            parameters=[],
            lowered_return_expression="NAV_ACC_RAD",
        )])
        result = substitute_helpers(
            "get_acceptance_radius()",
            registry=registry,
            env={},
        )
        assert result == "(NAV_ACC_RAD)"

    def test_empty_inputs(self):
        registry = HelperRegistry([])
        assert substitute_helpers("", registry=registry, env={}) == ""
        assert substitute_helpers(None, registry=registry, env={}) is None  # type: ignore[arg-type]

    def test_unresolved_helper_passes_through(self):
        registry = HelperRegistry([_ref(
            "stateful",
            parameters=["x"],
            lowered_return_expression="x",
            unresolved_reason="state mutation",
        )])
        result = substitute_helpers(
            "stateful(value)",
            registry=registry,
            env={},
        )
        # Helper is in the registry but unresolved → leave the call alone.
        assert result == "stateful(value)"

    def test_complex_argument_with_nested_call(self):
        registry = HelperRegistry([_ref(
            "wrap",
            parameters=["x"],
            lowered_return_expression="x + 1",
        )])
        result = substitute_helpers(
            "wrap(max(a, b))",
            registry=registry,
            env={},
        )
        assert result == "((max(a, b)) + 1)"

    def test_condition_with_signal_value(self):
        # A branched helper whose condition references a logged signal
        # bound in the env.
        registry = HelperRegistry([_ref(
            "select_cone",
            parameters=[],
            branches=[
                {"condition": "RTL_CONE_ANG > 0", "expression": "cone_branch"},
                {"condition": "default", "expression": "plain_branch"},
            ],
        )])
        result = substitute_helpers(
            "select_cone()",
            registry=registry,
            env={"RTL_CONE_ANG": 45},
        )
        assert result == "(cone_branch)"
        result_no = substitute_helpers(
            "select_cone()",
            registry=registry,
            env={"RTL_CONE_ANG": 0},
        )
        assert result_no == "(plain_branch)"
