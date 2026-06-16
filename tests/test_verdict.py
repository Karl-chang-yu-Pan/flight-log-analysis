from __future__ import annotations

import pytest

from flight_log_agent.analysis.verdict import (
    aggregate,
    branch_result,
    ceiling_for,
    combine_verdicts,
    verdict_from_counts,
)


class TestVerdictFromCounts:
    def test_supported_only(self):
        assert verdict_from_counts(supported=3, contradicted=0) == "supported"

    def test_contradicted_only(self):
        assert verdict_from_counts(supported=0, contradicted=2) == "contradicted"

    def test_both_is_mixed(self):
        assert verdict_from_counts(supported=1, contradicted=1) == "mixed"

    def test_empty_is_unresolved(self):
        assert verdict_from_counts(supported=0, contradicted=0) == "unresolved"


class TestCeilingFor:
    def test_supported_clean(self):
        assert ceiling_for("supported") == "high"

    def test_supported_with_unresolved_defining_caps_at_medium(self):
        assert ceiling_for("supported", has_unresolved_defining=True) == "medium"

    def test_missing_required_signals_overrides(self):
        assert ceiling_for("supported", missing_required_signals=True) == "low"

    def test_mixed_caps_medium(self):
        assert ceiling_for("mixed") == "medium"

    def test_contradicted_caps_low(self):
        assert ceiling_for("contradicted") == "low"

    def test_excluded_caps_low(self):
        assert ceiling_for("excluded") == "low"

    def test_unresolved_stays_unresolved(self):
        assert ceiling_for("unresolved") == "unresolved"

    def test_max_ceiling_caps_supported_at_medium(self):
        assert ceiling_for("supported", max_ceiling="medium") == "medium"

    def test_max_ceiling_does_not_promote(self):
        # max_ceiling is a CAP, not a floor: contradicted stays low.
        assert ceiling_for("contradicted", max_ceiling="high") == "low"


class TestCombineVerdicts:
    def test_all_supported(self):
        assert combine_verdicts("supported", "supported") == "supported"

    def test_supported_then_unresolved_takes_supported(self):
        assert combine_verdicts("supported", "unresolved") == "supported"

    def test_unresolved_then_supported_takes_supported(self):
        assert combine_verdicts("unresolved", "supported") == "supported"

    def test_supported_plus_contradicted_is_mixed(self):
        assert combine_verdicts("supported", "contradicted") == "mixed"

    def test_mixed_dominates(self):
        assert combine_verdicts("supported", "mixed") == "mixed"

    def test_all_unresolved_stays_unresolved(self):
        assert combine_verdicts("unresolved", "unresolved") == "unresolved"

    def test_empty_input_is_unresolved(self):
        assert combine_verdicts() == "unresolved"


class TestAggregateCheckList:
    def _result(self, *, role, status):
        return {"role": role, "status": status}

    def test_failed_applicability_excludes(self):
        results = [self._result(role="branch_applicability", status="failed")]
        assert aggregate(results, level="check_list") == "excluded"

    def test_unresolved_applicability_is_unresolved(self):
        results = [self._result(role="branch_applicability", status="unresolved")]
        assert aggregate(results, level="check_list") == "unresolved"

    def test_unresolved_defining_is_unresolved(self):
        results = [
            self._result(role="branch_applicability", status="passed"),
            self._result(role="mechanism_defining", status="unresolved"),
        ]
        assert aggregate(results, level="check_list") == "unresolved"

    def test_all_passed_defining_is_supported(self):
        results = [
            self._result(role="branch_applicability", status="passed"),
            self._result(role="mechanism_defining", status="passed"),
            self._result(role="mechanism_defining", status="passed"),
        ]
        assert aggregate(results, level="check_list") == "supported"

    def test_mixed_defining(self):
        results = [
            self._result(role="mechanism_defining", status="passed"),
            self._result(role="mechanism_defining", status="failed"),
        ]
        assert aggregate(results, level="check_list") == "mixed"

    def test_no_defining_is_unresolved(self):
        results = [self._result(role="branch_applicability", status="passed")]
        assert aggregate(results, level="check_list") == "unresolved"


class TestAggregateMechanism:
    def test_excluded_branches_filtered_out(self):
        items = [
            {"verdict": "excluded", "has_mechanism_defining_checks": True},
            {"verdict": "supported", "has_mechanism_defining_checks": True},
        ]
        assert aggregate(items, level="mechanism") == "supported"

    def test_branches_without_defining_filtered(self):
        items = [
            {"verdict": "supported", "has_mechanism_defining_checks": False},
            {"verdict": "contradicted", "has_mechanism_defining_checks": True},
        ]
        assert aggregate(items, level="mechanism") == "contradicted"

    def test_supported_plus_contradicted_is_mixed(self):
        items = [
            {"verdict": "supported", "has_mechanism_defining_checks": True},
            {"verdict": "contradicted", "has_mechanism_defining_checks": True},
        ]
        assert aggregate(items, level="mechanism") == "mixed"

    def test_all_contradicted_is_contradicted(self):
        items = [
            {"verdict": "contradicted", "has_mechanism_defining_checks": True},
            {"verdict": "contradicted", "has_mechanism_defining_checks": True},
        ]
        assert aggregate(items, level="mechanism") == "contradicted"

    def test_no_defining_branches_is_unresolved(self):
        items = [{"verdict": "supported", "has_mechanism_defining_checks": False}]
        assert aggregate(items, level="mechanism") == "unresolved"


class TestAggregateGraph:
    def test_all_supported(self):
        items = [{"verdict": "supported"}, {"verdict": "supported"}]
        assert aggregate(items, level="graph") == "supported"

    def test_all_contradicted(self):
        items = [{"verdict": "contradicted"}, {"verdict": "contradicted"}]
        assert aggregate(items, level="graph") == "contradicted"

    def test_mixed_inputs(self):
        items = [{"verdict": "supported"}, {"verdict": "contradicted"}]
        assert aggregate(items, level="graph") == "mixed"

    def test_object_with_verdict_attribute(self):
        class Obj:
            verdict = "supported"
        assert aggregate([Obj(), Obj()], level="graph") == "supported"


class TestBranchResult:
    class _FakeBranch:
        def __init__(self, branch_id="b", name="branch", unresolved=None, defining=True):
            self.branch_id = branch_id
            self.name = name
            self.unresolved_dependencies = unresolved or []
            self.checks = [type("Check", (), {"role": "mechanism_defining" if defining else "advisory"})()]

    def test_unresolved_dependency_forces_unresolved(self):
        branch = self._FakeBranch(unresolved=["thing"])
        result = branch_result(branch, [{"verdict": "supported"}], [])
        assert result["verdict"] == "unresolved"

    def test_supported_window_propagates(self):
        result = branch_result(self._FakeBranch(), [{"verdict": "supported"}], [])
        assert result["verdict"] == "supported"
        assert result["has_mechanism_defining_checks"] is True

    def test_all_excluded_windows(self):
        result = branch_result(
            self._FakeBranch(),
            [{"verdict": "excluded"}, {"verdict": "excluded"}],
            [],
        )
        assert result["verdict"] == "excluded"

    def test_contradicted_only(self):
        result = branch_result(self._FakeBranch(), [{"verdict": "contradicted"}], [])
        assert result["verdict"] == "contradicted"

    def test_unresolved_mechanism_defining_check_blocks(self):
        unresolved = [{"role": "mechanism_defining", "status": "unresolved"}]
        result = branch_result(self._FakeBranch(), [{"verdict": "supported"}], unresolved)
        assert result["verdict"] == "unresolved"
