from __future__ import annotations

import math

import pytest
from pydantic import BaseModel

from flight_log_agent.utils import (
    compare,
    copy_model,
    dedupe_keep_order,
    is_number,
    json_safe_value,
    model_dump,
    round_float,
    safe_float,
    safe_int,
    stable_id,
    timestamp_to_seconds,
)


class TestDedupeKeepOrder:
    def test_preserves_first_occurrence_order(self):
        assert dedupe_keep_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]

    def test_drops_empty_strings_by_default(self):
        assert dedupe_keep_order(["a", "", "b", "", "a"]) == ["a", "b"]

    def test_drops_none_by_default(self):
        assert dedupe_keep_order(["a", None, "b", None]) == ["a", "b"]

    def test_keeps_empty_when_drop_empty_false(self):
        assert dedupe_keep_order(["a", "", "b", ""], drop_empty=False) == ["a", "", "b"]

    def test_empty_input_returns_empty_list(self):
        assert dedupe_keep_order([]) == []

    def test_works_with_numeric_input(self):
        assert dedupe_keep_order([1, 2, 1, 3]) == [1, 2, 3]


class TestStableId:
    def test_deterministic_for_same_value(self):
        assert stable_id("check", {"a": 1, "b": 2}) == stable_id("check", {"a": 1, "b": 2})

    def test_key_order_does_not_change_hash(self):
        # JSON dump is sort_keys=True
        assert stable_id("check", {"a": 1, "b": 2}) == stable_id("check", {"b": 2, "a": 1})

    def test_different_value_gives_different_id(self):
        assert stable_id("check", {"a": 1}) != stable_id("check", {"a": 2})

    def test_prefix_is_included(self):
        identifier = stable_id("check", {})
        assert identifier.startswith("check_")
        assert len(identifier) == len("check_") + 12

    def test_handles_non_json_native_values(self):
        # default=str fallback
        class Foo:
            def __str__(self) -> str:
                return "foo"

        assert stable_id("p", Foo()) == stable_id("p", Foo())


class _Sample(BaseModel):
    name: str
    value: int | None = None
    notes: str | None = None


class TestModelDump:
    def test_excludes_none_fields(self):
        dumped = model_dump(_Sample(name="x", value=None, notes="hello"))
        assert dumped == {"name": "x", "notes": "hello"}

    def test_dict_passthrough_drops_none_values(self):
        assert model_dump({"a": 1, "b": None, "c": "x"}) == {"a": 1, "c": "x"}

    def test_list_walks_nested(self):
        items = [_Sample(name="a"), _Sample(name="b", value=2)]
        assert model_dump(items) == [{"name": "a"}, {"name": "b", "value": 2}]

    def test_scalar_passthrough(self):
        assert model_dump(42) == 42
        assert model_dump("hello") == "hello"
        assert model_dump(3.14) == 3.14


class TestCopyModel:
    def test_returns_a_copy(self):
        original = _Sample(name="x", value=1)
        copy = copy_model(original)
        assert copy == original
        assert copy is not original

    def test_applies_updates(self):
        original = _Sample(name="x", value=1)
        updated = copy_model(original, update={"value": 99})
        assert updated.value == 99
        assert original.value == 1


class TestJsonSafeValue:
    def test_bytes_decoded_and_stripped(self):
        assert json_safe_value(b"PX4\x00") == "PX4"

    def test_invalid_utf8_does_not_raise(self):
        # errors="replace" mode keeps the call safe.
        assert isinstance(json_safe_value(b"\xff"), str)

    def test_numpy_scalar_unwrapped(self):
        class Fake:
            def item(self):
                return 42

        assert json_safe_value(Fake()) == 42

    def test_scalar_passes_through(self):
        assert json_safe_value(3.14) == 3.14
        assert json_safe_value("hello") == "hello"
        assert json_safe_value(None) is None


class TestTimestampToSeconds:
    def test_microseconds_to_seconds_at_6dp(self):
        assert timestamp_to_seconds(1_234_567) == 1.234567

    def test_unwraps_numpy_scalar(self):
        class Fake:
            def item(self):
                return 2_000_000

        assert timestamp_to_seconds(Fake()) == 2.0


class TestSafeFloat:
    def test_finite_float(self):
        assert safe_float(3.14) == 3.14

    def test_int_promotes_to_float(self):
        assert safe_float(2) == 2.0

    def test_bool_rejected(self):
        assert safe_float(True) is None
        assert safe_float(False) is None

    def test_nan_rejected(self):
        assert safe_float(math.nan) is None

    def test_infinity_rejected(self):
        assert safe_float(math.inf) is None
        assert safe_float(-math.inf) is None

    def test_string_coerces(self):
        assert safe_float("2.5") == 2.5

    def test_none_returns_none(self):
        assert safe_float(None) is None

    def test_garbage_returns_none(self):
        assert safe_float("not a number") is None
        assert safe_float([]) is None


class TestSafeInt:
    def test_int_passthrough(self):
        assert safe_int(7) == 7

    def test_float_truncates(self):
        assert safe_int(7.9) == 7

    def test_none_returns_none(self):
        assert safe_int(None) is None

    def test_garbage_returns_none(self):
        assert safe_int("not a number") is None

    def test_numpy_scalar_unwrapped(self):
        class Fake:
            def item(self):
                return 13

        assert safe_int(Fake()) == 13


class TestIsNumber:
    def test_finite_number(self):
        assert is_number(2.5) is True

    def test_bool_rejected(self):
        assert is_number(True) is False

    def test_nan_rejected(self):
        assert is_number(math.nan) is False

    def test_string_coerces(self):
        assert is_number("3.14") is True


class TestRoundFloat:
    def test_rounds_to_six_dp(self):
        assert round_float(1.2345678901) == 1.234568


class TestCompare:
    @pytest.mark.parametrize(
        "left, op, right, expected",
        [
            (1, "==", 1, True),
            (1, "!=", 2, True),
            (1, "<", 2, True),
            (2, ">", 1, True),
            (1, "<=", 1, True),
            (1, ">=", 1, True),
            ("a", "==", "a", True),
            (1, "==", 2, False),
        ],
    )
    def test_supported_ops(self, left, op, right, expected):
        assert compare(left, op, right) is expected

    def test_unsupported_op_raises(self):
        with pytest.raises(ValueError, match="unsupported operator"):
            compare(1, "<>", 2)
