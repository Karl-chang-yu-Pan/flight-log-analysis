"""P0 readiness for the staged DAG-primary migration (ADR-0005).

Test-side readiness machinery only: a pure DAG-decisive routing
predicate, benchmark-oracle loading, semantic normalization,
measurement seams, and budget-ratification contracts. No default
flip, no fallback routing, no authority changes, no production
changes. P0 GREEN means the machinery to judge readiness exists;
it never means DAG is default-ready.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "ref" / "PX4-Autopilot"
PINNED_COMMIT = "1dacb4cdef2d7145754fc788fa8dc482eed74b40"

ADR_0005 = REPO_ROOT / "docs/adr/0005-default-dag-diagnostic-migration.md"
P0_SPEC = REPO_ROOT / "docs/default_dag_p0_readiness_spec.md"

BENCHMARK_FIXTURES = {
    "rtl": REPO_ROOT / (
        "uploads/053127df84724a75998ff355b5a2519a/RTL-wierd-height-1.ulg"),
    "tecs": REPO_ROOT / (
        "uploads/tecs-switch-moded-hrate/switch-moded-hrate.ulg"),
    "takeoff": REPO_ROOT / (
        "uploads/takeoff-inconsistent-hgt/inconsistent-takeoff-hgt.ulg"),
    "airspeed": REPO_ROOT / (
        "uploads/8a8aa57fedf94caf833ab8b4ade11d25/airspeed-load-factor.ulg"),
}

BENCHMARK_SIDECARS = {
    "rtl": REPO_ROOT / "tests/acceptance/rtl_weird_height.json",
    "tecs": REPO_ROOT / "tests/acceptance/tecs_restart_transient.json",
    "takeoff": REPO_ROOT / "tests/acceptance/takeoff_minimum_altitude.json",
    "airspeed": REPO_ROOT / "tests/acceptance/airspeed_load_factor.json",
}

BENCHMARK_TESTS = {
    "rtl": REPO_ROOT / "tests/test_acceptance_rtl.py",
    "tecs": REPO_ROOT / "tests/test_acceptance_tecs.py",
    "takeoff": REPO_ROOT / "tests/test_acceptance_takeoff.py",
    "airspeed": REPO_ROOT / "tests/test_acceptance_airspeed.py",
}

BENCHMARK_MD5 = {
    "rtl": "865fd06d54611c67f37e63b9a1dbdc4d",
    "tecs": "b3dc58db95a08b8cd45ac107f28363fd",
    "takeoff": "864264eb3ceadc25c5e0b0703788be61",
    "airspeed": "2acb5b821e1dffed3edabb805f40dbaa",
}


def test_p0_t1_preflight():
    """P0-T1: governing ADR/spec, four oracle sidecars/tests,
    four real fixtures, snapshot pin, and callable live
    boundaries — all present before readiness work begins."""
    assert ADR_0005.exists(), "ADR-0005 must govern P0"
    assert P0_SPEC.exists(), "P0 readiness spec must exist"
    for name, sidecar in BENCHMARK_SIDECARS.items():
        assert sidecar.exists(), f"{name} sidecar missing"
    for name, module in BENCHMARK_TESTS.items():
        assert module.exists(), f"{name} acceptance module missing"
    for name, log in BENCHMARK_FIXTURES.items():
        assert log.exists(), \
            f"{name} fixture missing at {log} (gitignored real log)"
    import hashlib

    for name, log in BENCHMARK_FIXTURES.items():
        digest = hashlib.md5(log.read_bytes()).hexdigest()
        assert digest == BENCHMARK_MD5[name], \
            f"{name} fixture identity mismatch: {digest}"
    head = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_COMMIT, f"source snapshot {head} != pin"

    from flight_log_agent.analysis.dag_pipeline import (
        run_dag_discovery_stage,
    )
    from flight_log_agent.runner_core import analyze_flight_log

    assert callable(run_dag_discovery_stage)
    assert callable(analyze_flight_log)
    print("\nP0-T1 preflight: ADR/spec/oracles/fixtures/snapshot/boundaries")


DECISIVE = "DECISIVE"
CONTRADICTED = "CONTRADICTED"
UNDECIDED = "UNDECIDED"
UNAVAILABLE = "UNAVAILABLE"

REPLAY_NON_STATES = (None, "not_attempted", "unevaluable")


def dag_decisive_state(*, sufficient, terminal, replay_status,
                       replay_complete, validation_passed, has_dag,
                       contradicted):
    """Pure routing-readiness predicate over already-produced DAG
    stage data. Plain data in, readiness state out. No I/O, no LLM,
    no mutation, no report generation, no fallback execution.
    Routing only: never stop authority, confidence, or strength."""
    if contradicted:
        return CONTRADICTED
    if (not has_dag
            and (replay_status in REPLAY_NON_STATES or not sufficient)
            and not validation_passed):
        return UNAVAILABLE
    if (sufficient and terminal and replay_complete
            and validation_passed and has_dag):
        return DECISIVE
    return UNDECIDED


def extract_decisive_inputs(stage, *, validation_passed,
                            contradicted=False):
    """Pull predicate inputs from a real stage-result-shaped object.

    Field mapping (re-pinned from code): judged.verdict.sufficient,
    judged.verdict.selected_terminal, replay{status,complete},
    annotated_dag presence. Validation outcome is supplied by the
    caller (via validate_report), never recomputed here."""
    if stage is None:
        return {"sufficient": False, "terminal": "",
                "replay_status": None, "replay_complete": False,
                "validation_passed": False, "has_dag": False,
                "contradicted": bool(contradicted)}
    verdict = getattr(getattr(stage, "judged", None), "verdict", None)
    replay = getattr(stage, "replay", None) or {}
    dag = getattr(stage, "annotated_dag", None)
    return {
        "sufficient": bool(getattr(verdict, "sufficient", False)),
        "terminal": str(getattr(verdict, "selected_terminal", "") or ""),
        "replay_status": replay.get("status"),
        "replay_complete": bool(replay.get("complete", False)),
        "validation_passed": bool(validation_passed),
        "has_dag": bool(dag is not None) and bool(
            getattr(dag, "vertices", None)),
        "contradicted": bool(contradicted),
    }


def _made_stage(*, sufficient=True, terminal="_ead_alt",
                replay_status="matched", replay_complete=True,
                has_dag=True):
    """Fake stage-result-shaped object (test data, never production)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        judged=SimpleNamespace(
            verdict=SimpleNamespace(
                sufficient=sufficient, selected_terminal=terminal)),
        replay={"status": replay_status, "complete": replay_complete},
        annotated_dag=SimpleNamespace(vertices=[1]) if has_dag else None,
    )


def test_p0_t2_decisive_mapping():
    """P0-T2: actual DAG-output shapes map to routing-readiness
    states preserving contradiction != undecided != unavailable
    != decisive. If mapping needs authority changes: STOP A."""
    assert dag_decisive_state(
        sufficient=True, terminal="_ead_alt", replay_status="matched",
        replay_complete=True, validation_passed=True, has_dag=True,
        contradicted=False) == DECISIVE
    assert dag_decisive_state(
        sufficient=False, terminal="_ead_alt", replay_status="matched",
        replay_complete=True, validation_passed=True, has_dag=True,
        contradicted=False) == UNDECIDED
    assert dag_decisive_state(
        sufficient=True, terminal="", replay_status="matched",
        replay_complete=True, validation_passed=True, has_dag=True,
        contradicted=False) == UNDECIDED
    assert dag_decisive_state(
        sufficient=True, terminal="_ead_alt", replay_status="partial",
        replay_complete=False, validation_passed=True, has_dag=True,
        contradicted=False) == UNDECIDED
    assert dag_decisive_state(
        sufficient=True, terminal="_ead_alt", replay_status="unevaluable",
        replay_complete=False, validation_passed=False, has_dag=True,
        contradicted=False) == UNDECIDED
    assert dag_decisive_state(
        sufficient=True, terminal="_ead_alt", replay_status="matched",
        replay_complete=True, validation_passed=False, has_dag=True,
        contradicted=False) == UNDECIDED
    assert dag_decisive_state(
        sufficient=False, terminal="", replay_status=None,
        replay_complete=False, validation_passed=False, has_dag=False,
        contradicted=False) == UNAVAILABLE
    assert dag_decisive_state(
        sufficient=True, terminal="_ead_alt", replay_status="mismatched",
        replay_complete=True, validation_passed=True, has_dag=True,
        contradicted=True) == CONTRADICTED
    assert dag_decisive_state(
        sufficient=False, terminal="", replay_status="not_attempted",
        replay_complete=False, validation_passed=False, has_dag=False,
        contradicted=True) == CONTRADICTED

    extracted = extract_decisive_inputs(
        _made_stage(), validation_passed=True)
    assert extracted == {"sufficient": True, "terminal": "_ead_alt",
                         "replay_status": "matched",
                         "replay_complete": True,
                         "validation_passed": True, "has_dag": True,
                         "contradicted": False}
    assert dag_decisive_state(**extracted) == DECISIVE
    assert dag_decisive_state(**extract_decisive_inputs(
        None, validation_passed=False)) == UNAVAILABLE

    import copy

    frozen = {"sufficient": True, "terminal": "_t",
              "replay_status": "matched", "replay_complete": True,
              "validation_passed": True, "has_dag": True,
              "contradicted": False}
    before = copy.deepcopy(frozen)
    assert dag_decisive_state(**frozen) == DECISIVE
    assert dag_decisive_state(**frozen) == DECISIVE
    assert frozen == before, "predicate must not mutate inputs"
    print("\nP0-T2 decisive mapping preserves four states; pure")


def _readiness_module_tree():
    """AST of this readiness module (harness introspection)."""
    import ast

    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def test_p0_t3_authority_isolation():
    """P0-T3: readiness mapping never touches stop/proof authority.

    Static import/name scan plus behavioral facts: the predicate
    is defined test-side (not production), consumes plain data,
    and the module never names authority machinery. If this
    requires authority changes: STOP H."""
    import ast

    allowed = {
        "flight_log_agent.analysis.dag_pipeline": {
            "run_dag_discovery_stage"},
        "flight_log_agent.analysis.report_validation": {
            "validate_report"},
        "flight_log_agent.runner_core": {"analyze_flight_log"},
        "flight_log_agent.ulog.inventory": {
            "parse_ulog_inventory", "observed_signals_from_inventory"},
        "flight_log_agent.px4.msg_schema": {"load_px4_signal_policies",
                                            "load_px4_msg_schema"},
        "flight_log_agent.analysis.mechanism_judge": {
            "QuestionedCondition"},
        "flight_log_agent.analysis.dag_replay": {"EvaluationScope"},
        "flight_log_agent.audit": {"serialize_usage"},
        "flight_log_agent.px4.mechanism_source_profiler": {
            "MechanismSourceProfiler"},
        "flight_log_agent.models": {
            "FlightLogReport", "HypothesisReportItem",
            "ApplicabilityReport", "CodeRef", "RelationshipCheckSpec"},
    }
    tree = _readiness_module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and str(
                node.module or "").startswith("flight_log_agent"):
            module = str(node.module)
            assert module in allowed, \
                f"unexpected production seam: {module}"
            names = {alias.name for alias in node.names}
            assert names <= allowed[module], \
                f"unexpected names from {module}: {sorted(names)}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not str(alias.name or "").startswith(
                    "flight_log_agent"), \
                    f"unlisted production import: {alias.name}"
    # Authority machinery must be unreachable, and authority or
    # status state must never be WRITTEN. READS of report/model
    # fields (Load context) are the extraction job itself:
    # only Store context is banned. Reserved-word discipline:
    # extraction locals avoid these identifiers entirely, so any
    # future real coupling fails loudly here instead of hiding
    # behind a same-named temporary.
    forbidden_writes = {
        "legacy_verified", "branches_verified", "CheckpointDiscovery",
        "evaluate_proof_authority", "CoverageProofStore",
        "proof_store", "confirmed", "confidence", "coverage",
        "applicability", "CodeRef", "build_report_from_dag",
        "replay_terminal_expressions", "replay_dag_roots",
        "discriminate_candidates", "PROVEN",
    }
    written = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(
                node.ctx, ast.Store):
            written.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(
                node.ctx, ast.Store):
            written.add(node.attr)
    assert not (written & forbidden_writes), \
        f"authority/status writes forbidden: " \
        f"{sorted(written & forbidden_writes)}"

    assert dag_decisive_state.__module__ == __name__, \
        "predicate must live test-side, never in production"
    assert extract_decisive_inputs.__module__ == __name__
    print("\nP0-T3 authority isolation: static + behavioral (STOP H clear)")


def _made_hypothesis(*, title="primary", family="family-x",
                     contradicting=(), unresolved=(), confidence="medium",
                     source_files=("fw_pos_control/"
                                   "FixedwingPositionControl.cpp",),
                     numeric=True, missing_signals=()):
    """Real production HypothesisReportItem (test data, never live)."""
    from flight_log_agent.models import (
        ApplicabilityReport,
        CodeRef,
        HypothesisReportItem,
        RelationshipCheckSpec,
    )

    return HypothesisReportItem(
        title=title, known_px4_mechanism=family, mechanism=family,
        source_refs=[CodeRef(file=path, function="fn")
                     for path in source_files],
        expected_logged_signature=[],
        applicability=ApplicabilityReport(
            applicable=not missing_signals,
            missing_required_signals=list(missing_signals)),
        evidence=["e1"] if numeric or source_files else [],
        contradicting_evidence=list(contradicting),
        unresolved_evidence=list(unresolved),
        exclusion_checks=[],
        numeric_checks=[RelationshipCheckSpec(type="threshold",
                                              signal="s")]
        if numeric else [],
        confidence=confidence)


def _made_report(*, hypotheses=None, confirmed=(), unconfirmed=(),
                 excluded=()):
    """Real production FlightLogReport (test data, never live)."""
    from flight_log_agent.models import FlightLogReport

    items = list(hypotheses) if hypotheses is not None else [
        _made_hypothesis()]
    return FlightLogReport(
        airframe_summary="", question_intent_summary="",
        ranked_hypotheses=items, excluded_mechanisms=list(excluded),
        confirmed=list(confirmed), unconfirmed=list(unconfirmed),
        final_summary="")


def derive_stage_contradiction(stage_report, replay):
    """Test-side contradiction derivation from existing evidence.

    True iff replay is complete and mismatched, or the single
    top-ranked hypothesis carries contradicting evidence.
    Multi-hypothesis mixed evidence is NOT global contradiction
    (healthy rival exclusion must not trip it). Partial,
    unevaluable, not_attempted, and missing states never count.
    Pure getattr reads; works on real models or plain data."""
    replay = replay or {}
    if bool(replay.get("complete", False)) and (
            replay.get("status") == "mismatched"):
        return True
    items = list(getattr(stage_report, "ranked_hypotheses", None) or [])
    if len(items) == 1 and list(
            getattr(items[0], "contradicting_evidence", None) or []):
        return True
    return False


def test_p0_tdd4_contradiction_derivation():
    """TDD-4: contradiction derives only from replay mismatch or
    single-claim self-contradiction. Partial/unevaluable/missing
    and healthy rival exclusion never count. STOP E on
    derivation impossibility (not fired: derivation exists)."""
    clean = _made_report()
    assert derive_stage_contradiction(clean, {"status": "matched",
                                              "complete": True}) is False
    assert derive_stage_contradiction(
        clean, {"status": "mismatched",
                "complete": True}) is True
    assert derive_stage_contradiction(
        clean, {"status": "mismatched",
                "complete": False}) is False
    assert derive_stage_contradiction(clean, None) is False
    self_contra = _made_report(hypotheses=[
        _made_hypothesis(contradicting=["replay vs log"])])
    assert derive_stage_contradiction(self_contra, None) is True
    healthy = _made_report(hypotheses=[
        _made_hypothesis(title="a"),
        _made_hypothesis(title="b",
                         contradicting=["rival b refuted"])])
    assert derive_stage_contradiction(healthy, None) is False
    print("\nTDD-4 contradiction derives conservatively (STOP E clear)")


def test_p0_tdd6_decisive_contradiction_precedence():
    """TDD-6: derived contradiction forces CONTRADICTED in the
    routing predicate regardless of sufficient-looking fields.
    Readiness classification only; no fallback, no authority."""
    assert dag_decisive_state(**{
        "sufficient": True, "terminal": "_t",
        "replay_status": "mismatched", "replay_complete": True,
        "validation_passed": True, "has_dag": True,
        "contradicted": derive_stage_contradiction(
            _made_report(), {"status": "mismatched",
                             "complete": True})}) == CONTRADICTED
    print("\nTDD-6 derived contradiction takes precedence")


def extract_report_semantics(report, *, snapshot_provided):
    """Extract normalized semantics from an actual-shaped live
    FlightLogReport (legacy or DAG path — same report contract).

    Pure getattr reads over real model fields; no oracle access;
    prose never interpreted (titles/names carried verbatim for
    alias matching elsewhere). Check specs contribute grounding
    presence only — specs carry no outcome. Fabrication is
    undetectable post-hoc, so fabricated_refs is always False
    here (the veto applies to hand-shaped negative tests)."""
    items = list(getattr(report, "ranked_hypotheses", None) or [])
    if not items:
        return {"family_native": "", "confirmed": [],
                "unconfirmed": [], "excluded_mechanisms": [],
                "contradicting": [], "unresolved": [],
                "numeric_present": False, "source_files": [],
                "missing_signals": False, "confidence": None,
                "status": "unavailable"}
    primary = items[0]
    # Reserved-word discipline (see P0-T3): authority-adjacent
    # identifiers never appear as locals here; the applicabilities
    # and confirmations below are plain extracted data held under
    # distinct names so any future real coupling fails loudly.
    applic_report = getattr(primary, "applicability", None)
    missing = list(getattr(applic_report, "missing_required_signals",
                           None) or [])
    checks = list(getattr(primary, "numeric_checks", None) or [])
    refs = list(getattr(primary, "source_refs", None) or [])
    evidence = list(getattr(primary, "evidence", None) or [])
    contradicting = list(
        getattr(primary, "contradicting_evidence", None) or [])
    confirmed_titles = list(getattr(report, "confirmed", None) or [])
    unconfirmed = list(getattr(report, "unconfirmed", None) or [])
    title = getattr(primary, "title", "")
    if contradicting:
        status = "contradicted"
    elif title and title in confirmed_titles:
        status = "supported"
    else:
        status = "unresolved"
    grounded = not missing
    return {
        "family_native": str(
            getattr(primary, "known_px4_mechanism", "") or ""),
        "confirmed": [str(t) for t in confirmed_titles],
        "unconfirmed": [str(t) for t in unconfirmed],
        "excluded_mechanisms": [
            str(m) for m in (
                getattr(report, "excluded_mechanisms", None) or [])],
        "contradicting": [str(e) for e in contradicting],
        "unresolved": [str(e) for e in (
            getattr(primary, "unresolved_evidence", None) or [])],
        "has_evidence": bool(evidence),
        "numeric_present": bool(checks) and grounded,
        "source_files": sorted({
            str(getattr(ref, "file", "") or "") for ref in refs
            if getattr(ref, "file", "")}),
        "missing_signals": bool(missing),
        "confidence": getattr(primary, "confidence", None),
        "status": status,
    }


def test_p0_tdd1_legacy_extraction():
    """TDD-1: legacy-shaped live reports extract to stable
    semantics from real model fields. If mechanism/rival meaning
    lives only in unconstrained prose with no structured
    carrier: STOP C (not fired: carriers exist)."""
    extracted = extract_report_semantics(
        _made_report(confirmed=["primary"]), snapshot_provided=True)
    assert extracted["family_native"] == "family-x"
    assert extracted["status"] == "supported"
    assert extracted["has_evidence"] is True
    assert extracted["numeric_present"] is True
    assert extracted["source_files"] == [
        "fw_pos_control/FixedwingPositionControl.cpp"]
    assert extracted["missing_signals"] is False
    assert extracted["confidence"] == "medium"

    unresolved = extract_report_semantics(
        _made_report(unconfirmed=["primary"]), snapshot_provided=True)
    assert unresolved["status"] == "unresolved"

    assert extract_report_semantics(
        _made_report(hypotheses=[]),
        snapshot_provided=True)["status"] == "unavailable"

    gap = extract_report_semantics(
        _made_report(hypotheses=[_made_hypothesis(
            numeric=False, source_files=[],
            missing_signals=["s"])], confirmed=["primary"]),
        snapshot_provided=True)
    assert gap["numeric_present"] is False
    assert gap["source_files"] == []
    assert gap["missing_signals"] is True

    contra = extract_report_semantics(
        _made_report(hypotheses=[_made_hypothesis(
            contradicting=["replay vs log"])], confirmed=["primary"]),
        snapshot_provided=True)
    assert contra["status"] == "contradicted"
    assert contra["contradicting"] == ["replay vs log"]
    print("\nTDD-1 legacy extraction over real models (STOP C clear)")


def load_benchmark_oracle(name):
    """Load one benchmark's stable oracle semantics from its
    committed sidecar. Pure JSON transliteration: scenario,
    mechanism family, strength, rival dispositions, grounding
    bar, fixture provenance. Never live evidence, never
    expected-answer logic. If loading needs benchmark logic
    reimplemented: STOP I."""
    import json

    sidecar = json.loads(BENCHMARK_SIDECARS[name].read_text(
        encoding="utf-8"))
    expected = sidecar.get("expected") or {}
    raw_rivals = expected.get("rival_excluded", [])
    if isinstance(raw_rivals, str):
        raw_rivals = [raw_rivals]
    rivals_required = {str(rival): "excluded" for rival in raw_rivals}
    return {
        "scenario_id": sidecar.get("scenario_id", name),
        "mechanism_family": expected.get("mechanism_family", ""),
        "strength": sidecar.get("strength", ""),
        "rivals_required": rivals_required,
        "grounding_required": ["log_evidence", "snapshot",
                               "mechanism", "numeric_checks"],
        "fixture": (sidecar.get("inputs") or {}).get("log", ""),
        "snapshot": (sidecar.get("inputs") or {}).get(
            "source_snapshot", ""),
    }


def test_p0_t4_oracle_loader():
    """P0-T4: committed oracles load as independent regression
    sources. Loader reads sidecar JSON only — no ULog parsing,
    no source reads, no numeric pins. If separation from live
    inputs cannot hold: STOP B."""
    oracles = {name: load_benchmark_oracle(name)
               for name in BENCHMARK_FIXTURES}
    assert set(oracles) == {"rtl", "tecs", "takeoff", "airspeed"}
    for name, oracle in oracles.items():
        assert oracle["mechanism_family"], \
            f"{name} oracle lacks mechanism family"
        assert oracle["strength"] in ("PROVEN", "BEST_SUPPORTED"), \
            f"{name} oracle lacks benchmark strength"
        assert oracle["fixture"].endswith(".ulg"), \
            f"{name} oracle lacks fixture provenance"
        assert oracle["snapshot"] == PINNED_COMMIT, \
            f"{name} oracle snapshot mismatch"
    assert oracles["airspeed"]["strength"] == "BEST_SUPPORTED"
    assert oracles["tecs"]["rivals_required"] == {
        "persistent_1ms_limit": "excluded"}
    assert "waypoint_requested_plus20" in oracles["takeoff"][
        "rivals_required"]
    print("\nP0-T4 oracles load independently (STOP B clear)")


RIVAL_OUTCOMES = ("excluded", "inactive", "contradicted", "weaker",
                  "unresolved", "not_applicable")
RULED_OUT = ("excluded", "contradicted", "inactive")


def normalize_path_result(path_output):
    """Normalize one path's semantic output to the stable
    comparison contract. Pure data mapping; prose never
    equality-compared; confidence kept separate, never merged
    into strength."""
    rivals = dict((path_output or {}).get("rivals") or {})
    for rival, outcome in rivals.items():
        assert outcome in RIVAL_OUTCOMES, \
            f"unknown rival outcome {outcome!r} for {rival!r}"
    grounding = dict((path_output or {}).get("grounding") or {})
    return {
        "mechanism_family": str(
            (path_output or {}).get("mechanism_family") or ""),
        "rivals": rivals,
        "grounding": grounding,
        "confidence": (path_output or {}).get("confidence"),
        "status": str((path_output or {}).get("status") or "unresolved"),
    }


def compare_to_oracle(normalized, oracle):
    """Compare normalized path semantics against an oracle.

    Returns MATCH / MISMATCH / UNDECIDED / UNAVAILABLE.
    Contradiction takes precedence over every other signal
    (a contradicted claim can never read as undecided).
    Unresolved-or-weaker rivals yield UNDECIDED (never forced
    MISMATCH); strength is compatibility-checked, never
    equality-compared to confidence."""
    if normalized.get("status") == "contradicted":
        return "MISMATCH"
    if normalized.get("status") in ("unavailable", "missing"):
        return "UNAVAILABLE"
    if normalized.get("status") not in ("verified", "supported",
                                        "matched"):
        return "UNDECIDED"
    if normalized.get("mechanism_family") != oracle["mechanism_family"]:
        return "MISMATCH"
    for rival, required in oracle["rivals_required"].items():
        outcome = normalized["rivals"].get(rival, "unresolved")
        if required in ("excluded", "inactive"):
            if outcome in RULED_OUT:
                continue
            if outcome in ("weaker", "unresolved"):
                return "UNDECIDED"
            return "MISMATCH"
        if outcome != required:
            return "MISMATCH" if outcome not in (
                "weaker", "unresolved") else "UNDECIDED"
    for key in oracle["grounding_required"]:
        if not normalized["grounding"].get(key, False):
            return "UNDECIDED"
    if normalized["grounding"].get("fabricated_refs", False):
        return "MISMATCH"
    live = normalized.get("confidence")
    strength = oracle["strength"]
    if strength == "BEST_SUPPORTED" and live in ("high", "confirmed",
                                                 "PROVEN"):
        return "MISMATCH"
    if live == "contradicted":
        return "MISMATCH"
    return "MATCH"


def test_p0_t5_normalization():
    """P0-T5: normalization keeps the six rival outcomes distinct,
    never prose-matches, and compatibility-checks (never
    equality-checks) strength. If stable normalization is
    impossible: STOP C."""
    oracle = load_benchmark_oracle("tecs")
    good = normalize_path_result({
        "mechanism_family": oracle["mechanism_family"],
        "rivals": {"persistent_1ms_limit": "excluded"},
        "grounding": {"log_evidence": True, "snapshot": True,
                      "mechanism": True, "numeric_checks": True,
                      "fabricated_refs": False},
        "confidence": "medium",
        "status": "supported"})
    assert compare_to_oracle(good, oracle) == "MATCH"
    assert compare_to_oracle(
        normalize_path_result({**good, "confidence": "low"}),
        oracle) == "MATCH"

    inactive_ok = normalize_path_result({
        **good, "rivals": {"persistent_1ms_limit": "inactive"}})
    assert compare_to_oracle(inactive_ok, oracle) == "MATCH"
    weaker = normalize_path_result({
        **good, "rivals": {"persistent_1ms_limit": "weaker"}})
    assert compare_to_oracle(weaker, oracle) == "UNDECIDED"
    assert compare_to_oracle(
        normalize_path_result({**good, "status": "unresolved"}),
        oracle) == "UNDECIDED"
    assert compare_to_oracle(
        normalize_path_result({**good, "status": "missing"}),
        oracle) == "UNAVAILABLE"
    assert compare_to_oracle(
        normalize_path_result(
            {**good, "mechanism_family": "other_family"}),
        oracle) == "MISMATCH"
    assert compare_to_oracle(
        normalize_path_result(
            {**good, "grounding": {**good["grounding"],
                                   "fabricated_refs": True}}),
        oracle) == "MISMATCH"
    airspeed_oracle = load_benchmark_oracle("airspeed")
    air_good = normalize_path_result({
        "mechanism_family": airspeed_oracle["mechanism_family"],
        "rivals": {rival: "excluded" for rival in
                   airspeed_oracle["rivals_required"]},
        "grounding": {"log_evidence": True, "snapshot": True,
                      "mechanism": True, "numeric_checks": True,
                      "fabricated_refs": False},
        "confidence": "medium",
        "status": "supported"})
    assert compare_to_oracle(air_good, airspeed_oracle) == "MATCH"
    assert compare_to_oracle(
        normalize_path_result({**air_good, "confidence": "high"}),
        airspeed_oracle) == "MISMATCH"
    assert compare_to_oracle(
        normalize_path_result({**air_good, "confidence": "low"}),
        airspeed_oracle) == "MATCH"
    print("\nP0-T5 normalization preserves outcomes; strength compatible")


BENCHMARK_DAG_TERMINALS = {
    # Committed-evidence terminal symbols per benchmark (from the
    # committed stub runners that select them). Takeoff/Airspeed
    # have no built DAG stage, so no terminal is declared: their
    # native identities stay honestly undetermined (UNDECIDED),
    # never force-matched and never force-mismatched.
    "rtl": frozenset({"_rtl_alt"}),
    "tecs": frozenset({"_debug_output.altitude_rate_control"}),
    "takeoff": frozenset(),
    "airspeed": frozenset(),
}


def extract_dag_semantics(stage, *, snapshot_provided,
                          validation_passed):
    """Extract normalized semantics from an actual-shaped DAG
    stage outcome: shared report extraction plus terminal,
    replay, and decisive inputs. Pure getattr reads; no oracle
    access (oracle enters only at comparison)."""
    report = getattr(stage, "report", None)
    semantics = extract_report_semantics(
        report, snapshot_provided=snapshot_provided)
    replay = getattr(stage, "replay", None) or {}
    verdict = getattr(getattr(stage, "judged", None), "verdict", None)
    contradicted = derive_stage_contradiction(report, replay)
    semantics["terminal"] = str(
        getattr(verdict, "selected_terminal", "") or "")
    semantics["replay_status"] = replay.get("status")
    semantics["replay_complete"] = bool(replay.get("complete", False))
    semantics["contradicted"] = contradicted
    semantics["decisive_inputs"] = extract_decisive_inputs(
        stage, validation_passed=validation_passed,
        contradicted=contradicted)
    return semantics


def compare_live_output(semantics, *, terminal, oracle,
                        dag_aliases=frozenset(),
                        known_different=frozenset(),
                        snapshot_provided=True):
    """Compare extracted live semantics to an oracle.

    Family rule: native or terminal equal to the oracle family,
    or present in declared aliases, proceeds; declared-different
    identities mismatch; anything else is honestly UNDECIDED
    (unknown spelling is not evidence of a different
    mechanism). Grounding is rebuilt from extraction fields
    plus the harness-known snapshot flag. Delegates to the
    shared comparison otherwise."""
    candidates = {str(semantics.get("family_native") or "")}
    if terminal:
        candidates.add(str(terminal))
    candidates.discard("")
    if semantics.get("status") in ("unavailable", "missing"):
        return "UNAVAILABLE"
    if oracle["mechanism_family"] not in candidates and not (
            candidates & set(dag_aliases)):
        if candidates & set(known_different):
            return "MISMATCH"
        return "UNDECIDED"
    normalized = normalize_path_result({
        "mechanism_family": oracle["mechanism_family"],
        "rivals": {str(name): "excluded" for name in
                   semantics.get("excluded_mechanisms", [])},
        "grounding": {
            "log_evidence": bool(semantics.get("has_evidence", False)),
            "snapshot": bool(snapshot_provided),
            "mechanism": bool(semantics.get("source_files")),
            "numeric_checks": bool(
                semantics.get("numeric_present", False)),
            "fabricated_refs": False,
        },
        "confidence": semantics.get("confidence"),
        "status": semantics.get("status", "unresolved"),
    })
    return compare_to_oracle(normalized, oracle)


def compose_scenario_readiness(*, scenario, legacy_report, dag_stage,
                               oracle, dag_aliases=frozenset(),
                               known_different=frozenset(),
                               snapshot_status="provided",
                               dag_validation_passed=True,
                               wall_time=None, llm_usage=None):
    """Minimum composition: live outputs → extracted semantics →
    normalized comparison → decisive classification → fallback
    derivation → full 11-key readiness record. Reuses the shared
    comparison machinery; no second implementation. Pure apart
    from its explicit inputs (callers supply measured wall time
    and usage); oracle enters only here, after both extractions.

    Snapshot grounding requires a pinned snapshot status
    ("pinned:<sha>"); merely provided-but-mismatched never
    counts, and unavailable never counts."""
    snapshot_ok = str(snapshot_status).startswith("pinned:")
    leg_sem = extract_report_semantics(
        legacy_report, snapshot_provided=snapshot_ok)
    dag_sem = extract_dag_semantics(
        dag_stage, snapshot_provided=snapshot_ok,
        validation_passed=dag_validation_passed)
    leg_result = compare_live_output(
        leg_sem, terminal=None, oracle=oracle,
        dag_aliases=frozenset(), known_different=known_different,
        snapshot_provided=snapshot_ok)
    dag_result = compare_live_output(
        dag_sem, terminal=dag_sem.get("terminal", ""),
        oracle=oracle, dag_aliases=dag_aliases,
        known_different=known_different,
        snapshot_provided=snapshot_ok)
    decisive = dag_decisive_state(**dag_sem["decisive_inputs"])
    fallback = decisive in (UNDECIDED, UNAVAILABLE)
    # Compatibility tracks the DAG path: readiness judges the
    # migration subject, while the legacy result stays recorded
    # separately as fallback context (never blended in).
    compatibility = dag_result
    validation_status = ("passed" if dag_validation_passed
                         else "failed")
    return _readiness_record(
        scenario=scenario, legacy_result=leg_result,
        dag_result=dag_result, compatibility=compatibility,
        decisive_state=decisive, fallback_required=fallback,
        wall_time=dict(wall_time or {}),
        llm_usage=dict(llm_usage or {}),
        snapshot_status=str(snapshot_status),
        validation_status=validation_status,
        notes="composed from live-shaped extraction; "
              "oracle entered at comparison only")


def _match_shaped_path(oracle, live_confidence="medium"):
    """Path-shaped output carrying the oracle's own required
    semantics (test data shaped by committed evidence, never live
    LLM output). Fast-CI scope: exercises comparison logic per
    benchmark vocabulary; real-path execution belongs to the
    gated bake-off suite, never to these tests.

    The confidence axis is deliberately named live_confidence:
    a plain-data test-side value, never a read of production
    pipeline status (see P0-T3, P0-T11)."""
    return {
        "mechanism_family": oracle["mechanism_family"],
        "rivals": {rival: "excluded" for rival in
                   oracle["rivals_required"]},
        "grounding": {"log_evidence": True, "snapshot": True,
                      "mechanism": True, "numeric_checks": True,
                      "fabricated_refs": False},
        "confidence": live_confidence,
        "status": "supported",
    }


def _assert_benchmark_comparison(name):
    oracle = load_benchmark_oracle(name)
    good = normalize_path_result(_match_shaped_path(oracle))
    assert compare_to_oracle(good, oracle) == "MATCH"
    assert compare_to_oracle(
        normalize_path_result(
            {**_match_shaped_path(oracle),
             "mechanism_family": "unrelated_family"}),
        oracle) == "MISMATCH"
    first_rival = next(iter(oracle["rivals_required"]), None)
    if first_rival is not None:
        shaped = _match_shaped_path(oracle)
        shaped["rivals"][first_rival] = "unresolved"
        assert compare_to_oracle(
            normalize_path_result(shaped), oracle) == "UNDECIDED"
    return oracle


def test_p0_t6_rtl_comparison():
    """P0-T6: RTL oracle comparison (cone/floor family, PROVEN
    compatibility). Fast-CI comparison scope; see helper."""
    oracle = _assert_benchmark_comparison("rtl")
    assert oracle["strength"] == "PROVEN"
    print("\nP0-T6 RTL comparison MATCH/MISMATCH/UNDECIDED")


def test_p0_t7_tecs_comparison():
    """P0-T7: TECS oracle comparison (restart-transient family,
    PROVEN compatibility; live confirmed may stay empty)."""
    oracle = _assert_benchmark_comparison("tecs")
    assert oracle["strength"] == "PROVEN"
    print("\nP0-T7 TECS comparison MATCH/MISMATCH/UNDECIDED")


def test_p0_t8_takeoff_comparison():
    """P0-T8: Takeoff oracle comparison (minimum-altitude plus
    resume family, PROVEN compatibility)."""
    oracle = _assert_benchmark_comparison("takeoff")
    assert oracle["strength"] == "PROVEN"
    print("\nP0-T8 Takeoff comparison MATCH/MISMATCH/UNDECIDED")


def test_p0_t9_airspeed_comparison():
    """P0-T9: Airspeed oracle comparison (load-factor binding
    family, BEST_SUPPORTED compatibility with gaps intact)."""
    oracle = _assert_benchmark_comparison("airspeed")
    assert oracle["strength"] == "BEST_SUPPORTED"
    print("\nP0-T9 Airspeed comparison MATCH/MISMATCH/UNDECIDED")


def _made_dag_stage(report, *, sufficient=True, terminal="_t",
                    replay_status="matched", replay_complete=True,
                    has_dag=True):
    """Stage-shaped object carrying a real report (test data)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        judged=SimpleNamespace(
            verdict=SimpleNamespace(
                sufficient=sufficient, selected_terminal=terminal)),
        report=report,
        replay={"status": replay_status, "complete": replay_complete},
        annotated_dag=SimpleNamespace(vertices=[1]) if has_dag else None,
    )


def _tecs_shaped_reports(*, family="restart_bumpless_transient",
                         confidence="medium", contradicting=(),
                         confirmed=True, excluded=("persistent_1ms_limit",),
                         numeric=True,
                         source_files=("src/lib/tecs/TECS.cpp",),
                         missing_signals=()):
    """TECS-shaped legacy report plus DAG stage over the same
    claim (real models throughout)."""
    hypothesis = _made_hypothesis(
        title="tecs-claim", family=family, contradicting=contradicting,
        confidence=confidence, numeric=numeric,
        source_files=source_files, missing_signals=missing_signals)
    titles = ["tecs-claim"] if confirmed else []
    report = _made_report(
        hypotheses=[hypothesis],
        confirmed=titles if confirmed else [],
        unconfirmed=[] if confirmed else titles,
        excluded=list(excluded))
    return report


def test_p0_tdd7_live_composition():
    """TDD-7: live outputs compose through extraction,
    normalization, comparison, decisive classification, and
    fallback derivation into a complete readiness record —
    reusing the shared machinery, no second implementation."""
    oracle = load_benchmark_oracle("tecs")
    report = _tecs_shaped_reports()
    stage = _made_dag_stage(
        report, terminal="_debug_output.altitude_rate_control")
    record = compose_scenario_readiness(
        scenario="tecs", legacy_report=report, dag_stage=stage,
        oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["tecs"],
        snapshot_status="pinned:abc123",
        dag_validation_passed=True,
        wall_time={"legacy": 2.0, "dag": 9.0},
        llm_usage={"legacy": {"total_tokens": 10},
                   "dag": {"total_tokens": 5}})
    assert record["compatibility"] == "MATCH"
    assert record["decisive_state"] == DECISIVE
    assert record["fallback_required"] is False
    assert record["legacy_result"] == "MATCH"
    assert record["dag_result"] == "MATCH"
    assert record["wall_time"] == {"legacy": 2.0, "dag": 9.0}
    assert record["snapshot_status"] == "pinned:abc123"
    assert record["validation_status"] == "passed"
    assert record["scenario"] == "tecs"
    print("\nTDD-7 live composition reuses shared machinery")


def test_p0_tdd8_oracle_independence():
    """TDD-8: extraction/normalization/predicate signatures admit
    no oracle or expected-answer parameters. Oracle enters only
    at comparison (compare_live_output / compose), after both
    live extractions — never in steps 1-4."""
    import inspect

    extraction_fns = (extract_report_semantics, extract_dag_semantics,
                      normalize_path_result, dag_decisive_state,
                      extract_decisive_inputs,
                      derive_stage_contradiction)
    banned = {"oracle", "expected_mechanism", "expected_rivals",
              "expected_strength", "expected_numeric",
              "mechanism_family"}
    for fn in extraction_fns:
        params = set(inspect.signature(fn).parameters)
        assert not (params & banned), \
            f"{fn.__name__} admits oracle input: {params & banned}"
    print("\nTDD-8 oracle enters at comparison only")


def test_p0_tdd10_full_record_completeness():
    """TDD-10: composed pipeline over injected live-shaped
    boundary results yields a complete valid readiness record
    (extract, normalize, compare, decisive, fallback, record)
    with no network/model calls."""
    oracle = load_benchmark_oracle("tecs")
    report = _tecs_shaped_reports()
    record = compose_scenario_readiness(
        scenario="tecs", legacy_report=report,
        dag_stage=_made_dag_stage(
            report,
            terminal="_debug_output.altitude_rate_control"),
        oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["tecs"],
        snapshot_status="pinned:abc123",
        dag_validation_passed=True,
        wall_time={"legacy": 1.0, "dag": 2.0},
        llm_usage={"legacy": {"requests": 1},
                   "dag": {"requests": 2}})
    assert set(record) == {"scenario", "legacy_result", "dag_result",
                           "compatibility", "decisive_state",
                           "fallback_required", "wall_time", "llm_usage",
                           "snapshot_status", "validation_status",
                           "notes"}
    assert record["compatibility"] == "MATCH"
    assert record["decisive_state"] == DECISIVE
    assert record["fallback_required"] is False
    assert record["wall_time"]["dag"] == 2.0
    assert record["llm_usage"]["dag"] == {"requests": 2}
    print("\nTDD-10 full record from injected boundary results")


def test_p0_tdd11_mismatch_path():
    """TDD-11: a declared-different live mechanism records
    MISMATCH without becoming fallback-required (decided
    disagreement follows the existing contract)."""
    oracle = load_benchmark_oracle("tecs")
    report = _tecs_shaped_reports(family="other_family")
    record = compose_scenario_readiness(
        scenario="tecs", legacy_report=report,
        dag_stage=_made_dag_stage(
            report, terminal="_other_output"),
        oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["tecs"],
        known_different={"_other_output", "other_family"},
        snapshot_status="pinned:abc123",
        dag_validation_passed=True)
    assert record["compatibility"] == "MISMATCH"
    assert record["fallback_required"] is False
    print("\nTDD-11 mismatch recorded, not converted to fallback")


def test_p0_tdd12_contradiction_path():
    """TDD-12: actual contradiction evidence yields
    CONTRADICTED decisive state, MISMATCH compatibility, and no
    fallback — per ADR-0005 contradiction precedence, with no
    legacy override implemented here."""
    oracle = load_benchmark_oracle("tecs")
    report = _tecs_shaped_reports(
        contradicting=["replay vs observed terminal"])
    record = compose_scenario_readiness(
        scenario="tecs", legacy_report=report,
        dag_stage=_made_dag_stage(
            report, replay_status="mismatched", replay_complete=True,
            terminal="_debug_output.altitude_rate_control"),
        oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["tecs"],
        snapshot_status="pinned:abc123",
        dag_validation_passed=True)
    assert record["decisive_state"] == CONTRADICTED
    assert record["compatibility"] == "MISMATCH"
    assert record["fallback_required"] is False
    print("\nTDD-12 contradiction path: no fallback, no override")


def test_p0_tdd13_undecided_path():
    """TDD-13: incomplete replay/insufficient evidence without
    contradiction yields UNDECIDED readiness with fallback
    required — while semantic comparison still records the
    underlying agreement separation (comparison MATCH on
    semantics, decisiveness UNDECIDED on proof sufficiency)."""
    oracle = load_benchmark_oracle("tecs")
    report = _tecs_shaped_reports()
    record = compose_scenario_readiness(
        scenario="tecs", legacy_report=report,
        dag_stage=_made_dag_stage(
            report, sufficient=False, replay_status="partial",
            replay_complete=False,
            terminal="_debug_output.altitude_rate_control"),
        oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["tecs"],
        snapshot_status="pinned:abc123",
        dag_validation_passed=True)
    assert record["decisive_state"] == UNDECIDED
    assert record["compatibility"] == "MATCH"
    assert record["dag_result"] == "MATCH"
    assert record["fallback_required"] is True
    print("\nTDD-13 undecided readiness with matched semantics")


def test_p0_tdd14_unavailable_path():
    """TDD-14: missing DAG result/snapshot semantics yield
    UNAVAILABLE with fallback required."""
    oracle = load_benchmark_oracle("tecs")
    record = compose_scenario_readiness(
        scenario="tecs",
        legacy_report=_made_report(hypotheses=[]),
        dag_stage=None, oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["tecs"],
        snapshot_status="unavailable",
        dag_validation_passed=False)
    assert record["decisive_state"] == UNAVAILABLE
    assert record["compatibility"] == "UNAVAILABLE"
    assert record["fallback_required"] is True
    print("\nTDD-14 unavailable path requires fallback")


def test_p0_tdd15_airspeed_protection_composed():
    """TDD-15: BEST_SUPPORTED Airspeed still rejects an
    over-strong live claim through the composed pipeline."""
    oracle = load_benchmark_oracle("airspeed")
    hypothesis = _made_hypothesis(
        title="as-claim",
        family="bank_load_factor_adapted_minimum",
        confidence="high")
    report = _made_report(
        hypotheses=[hypothesis], confirmed=["as-claim"],
        excluded=["fixed_trim", "mission_requested_23_7",
                  "wind_scaling", "weight_scaling",
                  "measured_bank_input",
                  "persistent_unrelated_command"])
    record = compose_scenario_readiness(
        scenario="airspeed", legacy_report=report,
        dag_stage=_made_dag_stage(report, terminal=""),
        oracle=oracle, dag_aliases=frozenset(),
        snapshot_status="pinned:abc123",
        dag_validation_passed=True)
    assert record["compatibility"] == "MISMATCH"
    print("\nTDD-15 Airspeed over-strong claim still MISMATCH")


def test_p0_tdd16_proven_confidence_composed():
    """TDD-16: lower live confidence alone never mismatches a
    PROVEN benchmark through live-output extraction."""
    oracle = load_benchmark_oracle("rtl")
    hypothesis = _made_hypothesis(
        title="rtl-claim",
        family="rtl_cone_branch_acceptance_floor_wins",
        confidence="low")
    report = _made_report(
        hypotheses=[hypothesis], confirmed=["rtl-claim"])
    record = compose_scenario_readiness(
        scenario="rtl", legacy_report=report,
        dag_stage=_made_dag_stage(report, terminal="_rtl_alt"),
        oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["rtl"],
        snapshot_status="pinned:abc123",
        dag_validation_passed=True)
    assert record["compatibility"] == "MATCH"
    print("\nTDD-16 PROVEN confidence-independence survives extraction")


def test_p0_tdd17_grounding_extraction():
    """TDD-17: log/source/numeric grounding extracts from real
    report shapes; missing grounding caps at UNDECIDED, never
    MISMATCH; ref ordering/formatting never matters."""
    oracle = load_benchmark_oracle("tecs")
    full = _tecs_shaped_reports()
    record = compose_scenario_readiness(
        scenario="tecs", legacy_report=full,
        dag_stage=_made_dag_stage(
            full, terminal="_debug_output.altitude_rate_control"),
        oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["tecs"],
        snapshot_status="pinned:abc123",
        dag_validation_passed=True)
    assert record["compatibility"] == "MATCH"

    thin = _tecs_shaped_reports(
        numeric=False, source_files=[],
        missing_signals=["tecs_status"])
    record = compose_scenario_readiness(
        scenario="tecs", legacy_report=thin,
        dag_stage=_made_dag_stage(
            thin, terminal="_debug_output.altitude_rate_control"),
        oracle=oracle,
        dag_aliases=BENCHMARK_DAG_TERMINALS["tecs"],
        snapshot_status="pinned:abc123",
        dag_validation_passed=True)
    assert record["compatibility"] == "UNDECIDED"
    print("\nTDD-17 grounding extracts; gaps cap at UNDECIDED")


def test_p0_t10_prose_independence():
    """P0-T10: materially different wording normalizes identically.
    Prose/title fields never enter the semantic contract."""
    oracle = load_benchmark_oracle("tecs")
    first = normalize_path_result({
        **_match_shaped_path(oracle),
        "title": "TECS bumpless restart transient after mode change",
        "summary": "The first post-transition setpoint reflects "
                   "reinitialization, not a persistent limit."})
    second = normalize_path_result({
        **_match_shaped_path(oracle),
        "title": "Height-rate spike at transition",
        "summary": "Totally different words; identical structured "
                   "mechanism, rivals, grounding, and status."})
    assert first == second
    assert compare_to_oracle(first, oracle) == "MATCH"
    assert compare_to_oracle(second, oracle) == "MATCH"
    print("\nP0-T10 prose independence: wording never compared")


def test_p0_t11_confidence_strength_independence():
    """P0-T11: benchmark strength is never equality-compared to
    live confidence. PROVEN acceptance with lower live
    confidence still MATCHes semantically; the confidence value
    itself passes through untouched for separate assertion."""
    oracle = load_benchmark_oracle("rtl")
    low_confidence = normalize_path_result({
        **_match_shaped_path(oracle, live_confidence="low")})
    assert compare_to_oracle(low_confidence, oracle) == "MATCH"
    assert low_confidence["confidence"] == "low"
    unresolved_confidence = normalize_path_result({
        **_match_shaped_path(oracle, live_confidence="unresolved")})
    assert compare_to_oracle(unresolved_confidence, oracle) == "MATCH"
    assert unresolved_confidence["confidence"] == "unresolved"
    print("\nP0-T11 strength/compatibility separate from confidence")


def _isolated_env(**overrides):
    """Temporarily override env vars, restoring afterward."""
    import contextlib
    import os

    @contextlib.contextmanager
    def _guard():
        saved = {key: os.environ.get(key) for key in overrides}
        try:
            for key, value in overrides.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            yield dict(os.environ)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return _guard()


def test_p0_t12_parser_selection_isolation():
    """P0-T12: DAG runs select tree_sitter, legacy runs keep the
    current configuration; environment is restored afterward
    with no global leakage. Repository defaults never change."""
    import os

    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )

    assert os.environ.get("FLIGHT_LOG_DAG_DISCOVERY") is None
    assert os.environ.get("FLIGHT_LOG_SOURCE_PARSER") is None
    with _isolated_env(FLIGHT_LOG_DAG_DISCOVERY="1",
                       FLIGHT_LOG_SOURCE_PARSER="tree_sitter"):
        assert os.environ["FLIGHT_LOG_DAG_DISCOVERY"] == "1"
        dag_profiler = MechanismSourceProfiler(
            str(SOURCE_ROOT), source_parser_backend=os.environ.get(
                "FLIGHT_LOG_SOURCE_PARSER", "legacy"))
        assert dag_profiler.source_parser_backend == "tree_sitter"
    assert os.environ.get("FLIGHT_LOG_DAG_DISCOVERY") is None
    assert os.environ.get("FLIGHT_LOG_SOURCE_PARSER") is None

    legacy_profiler = MechanismSourceProfiler(
        str(SOURCE_ROOT), source_parser_backend="legacy")
    assert legacy_profiler.source_parser_backend == "legacy"
    print("\nP0-T12 parser isolation: per-run selection, env restored")


def _snapshot_status(snapshot_path_or_none):
    """Readiness view of source-snapshot availability (test-side)."""
    import subprocess

    if snapshot_path_or_none is None:
        return "unavailable"
    head = subprocess.run(
        ["git", "-C", str(snapshot_path_or_none), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False)
    if head.returncode != 0:
        return "unavailable"
    sha = head.stdout.strip()
    return f"pinned:{sha}" if sha == PINNED_COMMIT else "mismatched"


def test_p0_t13_snapshot_unavailable_measurement():
    """P0-T13: withheld snapshot is detected and classified as a
    P1 explicit-fallback requirement. Current silent fallthrough
    is measured, never fixed here; routing stays untouched."""
    assert _snapshot_status(None) == "unavailable"
    assert _snapshot_status(SOURCE_ROOT) == f"pinned:{PINNED_COMMIT}"

    runner = (REPO_ROOT / "flight_log_agent/runner_core.py").read_text()
    assert ("if dag_discovery_enabled and source_snapshot is not None"
            in runner), "gated DAG branch must exist for P1 to mark"
    classification = ("P1 explicit-fallback requirement"
                      if _snapshot_status(None) == "unavailable"
                      else "unexpected")
    assert classification == "P1 explicit-fallback requirement"
    print("\nP0-T13 snapshot-unavailable measured (P1 requirement logged)")


def _timed_call(callable_obj):
    """Wall-time seam: (result, seconds) without flakiness (only
    non-negativity and passthrough are asserted in CI; real
    values belong to real readiness runs)."""
    import time

    started = time.perf_counter()
    result = callable_obj()
    return result, time.perf_counter() - started


def test_p0_t14_wall_time_measurement():
    """P0-T14: reusable per-fixture/per-path duration seam.
    Deterministic contract only; real durations are recorded,
    never thresholded here."""
    result, seconds = _timed_call(lambda: "semantic-ok")
    assert result == "semantic-ok"
    assert isinstance(seconds, float) and seconds >= 0.0
    print("\nP0-T14 wall-time seam records, never gates")


def test_p0_t15_llm_usage_measurement():
    """P0-T15: existing audit usage facility provides stable
    token/call metrics. Model IDs and currency are out of
    scope; tokens/calls suffice per spec (STOP E clear)."""
    from types import SimpleNamespace

    from flight_log_agent.audit import serialize_usage

    assert serialize_usage(None) == {}
    usage = serialize_usage(SimpleNamespace(
        requests=3, input_tokens=1000, output_tokens=200,
        total_tokens=1200,
        input_tokens_details=SimpleNamespace(cached_tokens=100),
        output_tokens_details=SimpleNamespace(reasoning_tokens=50),
        request_usage_entries=[]))
    assert usage == {"requests": 3, "input_tokens": 1000,
                     "output_tokens": 200, "total_tokens": 1200,
                     "cached_input_tokens": 100, "reasoning_tokens": 50,
                     "request_usage_entries": []}
    print("\nP0-T15 usage metrics available; currency out of scope")


def _readiness_counts(states):
    """Aggregate DAG-decisive states into fallback readiness."""
    counts = {DECISIVE: 0, CONTRADICTED: 0, UNDECIDED: 0,
              UNAVAILABLE: 0}
    for state in states:
        assert state in counts, f"unknown readiness state {state!r}"
        counts[state] += 1
    counts["would_require_fallback"] = (
        counts[UNDECIDED] + counts[UNAVAILABLE])
    return counts


def test_p0_t16_fallback_readiness_counts():
    """P0-T16: decisive/contradicted/undecided/unavailable
    aggregate into a fallback-required rate. No legacy fallback
    report is simulated."""
    counts = _readiness_counts(
        [DECISIVE, DECISIVE, UNDECIDED, UNAVAILABLE, CONTRADICTED])
    assert counts == {DECISIVE: 2, CONTRADICTED: 1, UNDECIDED: 1,
                      UNAVAILABLE: 1, "would_require_fallback": 2}
    print("\nP0-T16 fallback-required counts aggregated")


def _readiness_record(*, scenario, legacy_result, dag_result,
                      compatibility, decisive_state,
                      fallback_required, wall_time, llm_usage,
                      snapshot_status, validation_status, notes=""):
    """Machine-readable per-scenario readiness record. Paths are
    stored repo-relative; run-local noise never enters."""
    record = {
        "scenario": str(scenario),
        "legacy_result": legacy_result,
        "dag_result": dag_result,
        "compatibility": str(compatibility),
        "decisive_state": str(decisive_state),
        "fallback_required": bool(fallback_required),
        "wall_time": dict(wall_time),
        "llm_usage": dict(llm_usage),
        "snapshot_status": str(snapshot_status),
        "validation_status": str(validation_status),
        "notes": str(notes),
    }
    assert decisive_state in (DECISIVE, CONTRADICTED, UNDECIDED,
                              UNAVAILABLE)
    assert compatibility in ("MATCH", "MISMATCH", "UNDECIDED",
                             "UNAVAILABLE")
    return record


def test_p0_t17_readiness_aggregation():
    """P0-T17: semantics plus measurements combine into a
    normalized readiness record (schema-checked, noise-free)."""
    import json

    record = _readiness_record(
        scenario="tecs", legacy_result="supported",
        dag_result="supported", compatibility="MATCH",
        decisive_state=DECISIVE, fallback_required=False,
        wall_time={"legacy": 1.5, "dag": 12.0},
        llm_usage={"legacy": {"total_tokens": 100},
                   "dag": {"total_tokens": 50}},
        snapshot_status="pinned:abc", validation_status="passed")
    dumped = json.dumps(record)
    assert str(REPO_ROOT) not in dumped
    assert "/tmp/" not in dumped
    assert set(record) == {"scenario", "legacy_result", "dag_result",
                           "compatibility", "decisive_state",
                           "fallback_required", "wall_time", "llm_usage",
                           "snapshot_status", "validation_status",
                           "notes"}
    print("\nP0-T17 readiness record schema enforced")


READINESS_BUDGETS = REPO_ROOT / "tests/acceptance/dag_default_readiness.json"


def _load_readiness_budgets(path=READINESS_BUDGETS):
    """Load the versioned budget artifact (unratified until review)."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data.get("version") == 1
    assert data.get("status") in ("unratified", "ratified")
    if data.get("status") == "ratified":
        assert isinstance(data.get("thresholds"), dict), \
            "ratified budgets must carry thresholds"
    return data


def _check_budgets(measurements, budgets):
    """Enforce ratified thresholds; report-only before adoption.

    Returns {"enforced": bool, "violations": [...]}. Null
    thresholds never fail, however large the measurements."""
    thresholds = (budgets or {}).get("thresholds")
    if not isinstance(thresholds, dict):
        return {"enforced": False, "violations": []}
    violations = []
    for scenario, observed in (measurements or {}).items():
        limits = thresholds.get(scenario, thresholds.get("*", {}))
        for metric, limit in limits.items():
            value = (observed or {}).get(metric)
            if value is not None and value > limit:
                violations.append(
                    {"scenario": scenario, "metric": metric,
                     "observed": value, "limit": limit})
    return {"enforced": True, "violations": violations}


def test_p0_t18_pre_ratification_behavior():
    """P0-T18: with null thresholds, even absurd measurements
    report without failing. Candidate numbers are never gates
    before formal adoption."""
    budgets = _load_readiness_budgets()
    assert budgets["status"] == "unratified"
    assert budgets["thresholds"] is None
    outcome = _check_budgets(
        {"tecs": {"wall_seconds": 99999.0, "total_tokens": 10 ** 9}},
        budgets)
    assert outcome == {"enforced": False, "violations": []}
    print("\nP0-T18 pre-ratification: measure, never fail")


def test_p0_t19_ratified_budget_contract():
    """P0-T19: once ratified (synthetic here — real adoption is
    a review act), threshold violations fail loudly."""
    synthetic = {"version": 1, "status": "ratified",
                 "thresholds": {"*": {"wall_seconds": 100.0,
                                      "total_tokens": 10000}},
                 "basis": "synthetic test adoption"}
    outcome = _check_budgets(
        {"tecs": {"wall_seconds": 50.0, "total_tokens": 5000}},
        synthetic)
    assert outcome == {"enforced": True, "violations": []}
    outcome = _check_budgets(
        {"tecs": {"wall_seconds": 500.0, "total_tokens": 5000}},
        synthetic)
    assert outcome["enforced"] is True
    assert outcome["violations"] == [
        {"scenario": "tecs", "metric": "wall_seconds",
         "observed": 500.0, "limit": 100.0}]
    print("\nP0-T19 ratified contract enforced (synthetic adoption)")


def _bakeoff_plan():
    """Planned real-readiness scenarios (no model calls).

    Each scenario names both live boundaries explicitly:
    legacy = analyze_flight_log with dag_discovery=False;
    DAG = run_dag_discovery_stage direct (default real agent,
    tree_sitter per P0 parser rule). Questions come from
    sidecar interrogatives only (never expected mechanisms,
    rivals, numerics, or strength). Plan entries carry no
    oracle data: oracle loading happens after both runs."""
    import json

    plan = []
    for name in ("rtl", "tecs", "takeoff", "airspeed"):
        sidecar = json.loads(BENCHMARK_SIDECARS[name].read_text(
            encoding="utf-8"))
        plan.append({
            "scenario": name,
            "log": str(BENCHMARK_FIXTURES[name]),
            "question": sidecar["question"],
            "legacy": {"entry": "analyze_flight_log",
                       "dag_discovery": False},
            "dag": {"entry": "run_dag_discovery_stage",
                    "run_agent": None,
                    "parser": "tree_sitter"},
        })
    return plan


def _usages_in(run_dir):
    """usage.json files under one run directory (test-side)."""
    root = Path(run_dir)
    if not root.exists():
        return []
    return sorted(str(path) for path in root.rglob("usage.json"))


def _sum_usage(run_dir):
    """Aggregate token usage across one path's run directory.

    Missing/unreadable files contribute nothing; absence is
    reported via the file count, never fabricated."""
    import json

    total, counted = 0, 0
    for path in _usages_in(run_dir):
        try:
            total += int(json.loads(
                Path(path).read_text(
                    encoding="utf-8")).get("total_tokens", 0) or 0)
            counted += 1
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return {"usage_files": counted, "total_tokens": total}


def test_p0_bakeoff_imports_resolve():
    """Gated-branch import preflight: every production import
    inside the real bake-off branch must resolve WITHOUT
    executing any model/API call. This catches unreachable
    gated-branch imports that skip/dry runs never touch."""
    import ast
    import importlib

    tree = _readiness_module_tree()
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == (
                "test_p0_bakeoff_readiness_run"):
            target = node
            break
    assert target is not None, "bake-off test missing"
    checked = 0
    for node in ast.walk(target):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            assert hasattr(module, alias.name), \
                f"unresolvable gated import: {node.module}.{alias.name}"
            checked += 1
    assert checked >= 5, \
        f"expected real-branch imports to check, found {checked}"
    print(f"\nBake-off gated imports resolve x{checked}, 0 model calls")


def test_p0_bakeoff_readiness_run(tmp_path):
    """Gated real-readiness bake-off (manual, never default CI).

    Without FLIGHT_LOG_BAKEOFF: skipped intentionally (no
    network/model calls). With =dry: plan assembly only.
    With =1: real legacy + DAG runs with usage recording,
    semantic comparison, and measurement artifact under the
    pytest tmp dir (requires credentials/model access; never
    run automatically)."""
    import asyncio
    import os

    gate = os.environ.get("FLIGHT_LOG_BAKEOFF", "")
    if gate not in ("1", "dry"):
        import pytest

        pytest.skip("bake-off requires FLIGHT_LOG_BAKEOFF=1 (or dry)")
    plan = _bakeoff_plan()
    assert [entry["scenario"] for entry in plan] == [
        "rtl", "tecs", "takeoff", "airspeed"]
    for entry in plan:
        assert entry["legacy"]["dag_discovery"] is False
        assert entry["dag"]["entry"] == "run_dag_discovery_stage"
        assert entry["dag"]["parser"] == "tree_sitter"
        # Plan entries carry run coordinates only: no oracle,
        # expected, or answer data may reach steps 1-4.
        assert set(entry) == {"scenario", "log", "question",
                              "legacy", "dag"}
    if gate == "dry":
        print("\nBake-off dry-run: 4 scenarios planned, 0 model calls")
        return

    from flight_log_agent.analysis.dag_pipeline import (
        run_dag_discovery_stage,
    )
    from flight_log_agent.analysis.report_validation import (
        validate_report,
    )
    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )
    from flight_log_agent.px4.msg_schema import (
        load_px4_msg_schema,
        load_px4_signal_policies,
    )
    from flight_log_agent.runner_core import analyze_flight_log
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
        parse_ulog_inventory,
    )

    snapshot_status = _snapshot_status(str(SOURCE_ROOT))
    artifact = {"scenarios": []}
    for entry in plan:
        scen_dir = tmp_path / entry["scenario"]
        # Phase 1: run both paths (no oracle contact).
        def _run_legacy(entry=entry, scen_dir=scen_dir):
            return asyncio.run(analyze_flight_log(
                entry["log"], entry["question"],
                source_path=str(SOURCE_ROOT),
                output_dir=str(scen_dir / "legacy"),
                dev_log_root=str(scen_dir / "devlogs" / "legacy"),
                dag_discovery=False))
        legacy_report, legacy_secs = _timed_call(_run_legacy)

        inventory = parse_ulog_inventory(Path(entry["log"]),
                                         SOURCE_ROOT)
        schema = load_px4_msg_schema(SOURCE_ROOT)

        def _run_dag(entry=entry, scen_dir=scen_dir,
                     inventory=inventory, schema=schema):
            return asyncio.run(run_dag_discovery_stage(
                MechanismSourceProfiler(
                    str(SOURCE_ROOT),
                    source_parser_backend="tree_sitter"),
                scen_dir / "dag-cache",
                entry["question"],
                PINNED_COMMIT,
                Path(entry["log"]),
                inventory=inventory,
                run_agent=None,
                logged_signals=set(
                    observed_signals_from_inventory(inventory)),
                schema_signals=sorted(
                    f"{topic}.{field}"
                    for topic, fields in schema.items()
                    for field in fields),
                signal_policies=load_px4_signal_policies(
                    SOURCE_ROOT)))
        dag_stage, dag_secs = _timed_call(_run_dag)
        # Phase 2: oracle load, then compare, classify, record.
        # Oracle enters here only, after both live extractions.
        oracle = load_benchmark_oracle(entry["scenario"])
        validation = bool(validate_report(
            dag_stage.report).passed)
        record = compose_scenario_readiness(
            scenario=entry["scenario"],
            legacy_report=legacy_report,
            dag_stage=dag_stage,
            oracle=oracle,
            dag_aliases=BENCHMARK_DAG_TERMINALS[entry["scenario"]],
            snapshot_status=snapshot_status,
            dag_validation_passed=validation,
            wall_time={"legacy": legacy_secs, "dag": dag_secs},
            llm_usage={
                "legacy": _sum_usage(scen_dir / "devlogs"
                                     / "legacy"),
                "dag": {"unavailable":
                        "direct stage call bypasses runner audit; "
                        "no usage.json emitted"}})
        artifact["scenarios"].append(record)
    out = tmp_path / "bakeoff_measurements.json"
    import json

    out.write_text(json.dumps(artifact, indent=2,
                              default=str), encoding="utf-8")
    print(f"\nBake-off real run recorded: {out}")


def test_p0_tdd19_usage_isolation(tmp_path):
    """TDD-19: legacy and DAG usage are read from their own
    per-path run directories — no shared stale usage.json,
    no cross-path overwrite."""
    import json

    legacy_dir = tmp_path / "devlogs" / "tecs" / "legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "usage.json").write_text(
        json.dumps({"total_tokens": 111}), encoding="utf-8")
    dag_dir = tmp_path / "devlogs" / "tecs" / "dag"
    dag_dir.mkdir(parents=True)
    (dag_dir / "usage.json").write_text(
        json.dumps({"total_tokens": 222}), encoding="utf-8")
    assert _usages_in(legacy_dir) == [str(legacy_dir / "usage.json")]
    assert _usages_in(dag_dir) == [str(dag_dir / "usage.json")]
    assert _usages_in(tmp_path / "missing") == []
    assert _sum_usage(legacy_dir) == {"usage_files": 1,
                                      "total_tokens": 111}
    assert _sum_usage(dag_dir) == {"usage_files": 1,
                                   "total_tokens": 222}
    assert _sum_usage(tmp_path / "missing") == {"usage_files": 0,
                                                "total_tokens": 0}
    print("\nTDD-19 per-path usage isolation, no stale sharing")


def _run_pytest(*args):
    """Subprocess pytest at the repo root (regression driver)."""
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        capture_output=True, text=True, check=False, cwd=str(REPO_ROOT),
    )


def _readiness_module_tree():
    """AST of this readiness module (harness introspection)."""
    import ast

    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def test_p0_t20_focused_regression():
    """P0-T20: temporal, proof/coverage/checkpoint, guard, and
    acceptance-collection suites hold. Fast suites execute;
    slow stage suites prove collection while full execution is
    covered by the full-suite run. Closed files never change."""
    fast = _run_pytest(
        "tests/test_temporal_qualification.py",
        "tests/test_no_case_specific_terms.py",
        "tests/test_dag_checkpoint.py",
        "tests/test_verdict.py",
        "tests/test_verification_plan.py",
        "tests/test_verification_graph.py",
        "-q", "-p", "no:cacheprovider",
    )
    assert fast.returncode == 0, \
        f"focused regression failed:\n{fast.stdout[-1500:]}"
    slow = _run_pytest(
        "tests/test_acceptance_rtl.py",
        "tests/test_acceptance_tecs.py",
        "tests/test_acceptance_takeoff.py",
        "tests/test_acceptance_airspeed.py",
        "--collect-only", "-q", "-p", "no:cacheprovider",
    )
    assert slow.returncode == 0, \
        f"acceptance collection broken:\n{slow.stderr[-1500:]}"
    print("\nP0-T20 fast suites GREEN; slow suites collect")


def test_p0_t21_full_suite_still_collects():
    """P0-T21: P0 readiness coexists with the whole suite.

    Full collection must succeed with no test disappearance: at
    least the pre-P0 baseline (1537 passed + 31 xfailed) plus
    every test in this module. The gated bake-off test
    collects (skipped by default) and is counted, not hidden."""
    import ast

    module_tests = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(_readiness_module_tree())
    )
    assert module_tests >= 22, \
        f"P0 module must carry T1-T21 plus bake-off, found {module_tests}"

    collected = _run_pytest("--collect-only", "-q", "-p",
                            "no:cacheprovider")
    assert collected.returncode == 0, \
        f"suite collection broken:\n{collected.stderr[-2000:]}"
    total = 0
    for line in collected.stdout.splitlines():
        if "tests collected" in line:
            total = int(line.split()[0])
    assert "test_default_dag_p0_readiness" in collected.stdout
    assert total >= 1537 + 31 + module_tests, \
        f"collected {total}, expected baseline plus {module_tests} P0 tests"
    print(f"\nP0-T21 suite collects: {total} tests "
          f"(baseline 1568 + {module_tests} P0)")
