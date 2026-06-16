from __future__ import annotations

import pytest
from pydantic import BaseModel

from flight_log_agent.utils import copy_model, dedupe_keep_order, model_dump, stable_id


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
