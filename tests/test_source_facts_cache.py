from __future__ import annotations

from pathlib import Path

from flight_log_agent.px4.mechanism_source_profiler import (
    MechanismSourceProfiler,
    SourceAssignmentRef,
)
from flight_log_agent.px4.source_facts_cache import (
    SourceFileFacts,
    extract_facts_for_file,
    extractor_fingerprint,
    get_or_extract_facts,
    get_source_facts_for_file,
    layer1_cache_path,
    read_source_facts,
    write_source_facts,
)


def test_layer1_cache_path_composes_fingerprint_hash_and_file_slug(tmp_path):
    path = layer1_cache_path(tmp_path, "abcdef", "src/modules/navigator/rtl.cpp")
    assert path.parent.name == "abcdef"
    assert path.parent.parent.name == extractor_fingerprint()
    assert path.parent.parent.parent.name == "source"
    assert path.name == "src__modules__navigator__rtl.cpp.json"


def test_extractor_fingerprint_is_stable_within_process():
    assert extractor_fingerprint() == extractor_fingerprint()
    assert len(extractor_fingerprint()) == 16


def test_stale_fingerprint_trees_are_pruned_on_extraction(tmp_path):
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helper.cpp").write_text("void foo() { _x = 1; }\n", encoding="utf-8")
    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")

    cache_root = tmp_path / "cache"
    stale = cache_root / "source" / "deadbeefdeadbeef" / "oldhash"
    stale.mkdir(parents=True)
    (stale / "junk.json").write_text("{}", encoding="utf-8")

    get_or_extract_facts(profiler, cache_root, "src/modules/example/helper.cpp", "hash")

    assert not (cache_root / "source" / "deadbeefdeadbeef").exists()
    assert layer1_cache_path(cache_root, "hash", "src/modules/example/helper.cpp").exists()


def test_layer1_cache_path_handles_leading_slash_and_special_chars(tmp_path):
    path = layer1_cache_path(tmp_path, "hash", "/weird path/file+with?chars.cpp")
    # No leading slash left, path separator collapsed, non-alnum sanitized.
    assert path.name == "weird_path__file_with_chars.cpp.json"


def test_read_write_roundtrip_preserves_facts(tmp_path):
    facts = SourceFileFacts(
        file="rtl.cpp",
        source_hash="hash",
        source_assignments=[
            SourceAssignmentRef(
                target="_rtl_alt",
                expression="42",
                target_topic=None,
                target_field=None,
                function="foo",
                function_parameters=[],
                assignment_operator="=",
                file="rtl.cpp",
                line=245,
                evidence="_rtl_alt = 42;",
                control_predicates=[],
                symbol_bindings={},
            ),
        ],
    )
    path = layer1_cache_path(tmp_path, "hash", "rtl.cpp")

    assert read_source_facts(path) is None
    write_source_facts(facts, path)
    assert path.exists()

    restored = read_source_facts(path)
    assert restored is not None
    assert restored.file == "rtl.cpp"
    assert restored.source_hash == "hash"
    assert len(restored.source_assignments) == 1
    assert restored.source_assignments[0].target == "_rtl_alt"


def test_read_source_facts_returns_none_on_missing_or_corrupt(tmp_path):
    assert read_source_facts(tmp_path / "missing.json") is None

    corrupt = tmp_path / "bad.json"
    corrupt.write_text("not json {", encoding="utf-8")
    assert read_source_facts(corrupt) is None


def test_write_replaces_existing_entry_atomically(tmp_path):
    path = layer1_cache_path(tmp_path, "hash", "rtl.cpp")

    write_source_facts(
        SourceFileFacts(file="rtl.cpp", source_hash="hash"),
        path,
    )
    write_source_facts(
        SourceFileFacts(
            file="rtl.cpp",
            source_hash="hash",
            source_assignments=[
                SourceAssignmentRef(
                    target="_rtl_alt",
                    expression="99",
                    target_topic=None,
                    target_field=None,
                    function="foo",
                    function_parameters=[],
                    assignment_operator="=",
                    file="rtl.cpp",
                    line=1,
                    evidence="_rtl_alt = 99;",
                    control_predicates=[],
                    symbol_bindings={},
                ),
            ],
        ),
        path,
    )

    restored = read_source_facts(path)
    assert restored is not None
    assert len(restored.source_assignments) == 1
    assert restored.source_assignments[0].expression == "99"


def _write_git_stub(bin_dir: Path, *, exit_code: int) -> Path:
    """Write a fake ``git`` binary that returns a fixed exit code.

    Used to simulate ``git diff --quiet`` outcomes without needing an
    actual git working tree in the test tmp_path.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    git_path = bin_dir / "git"
    git_path.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
    git_path.chmod(0o755)
    return git_path


def test_get_source_facts_hits_directly_under_current_hash(tmp_path):
    path = layer1_cache_path(tmp_path, "hash_a", "rtl.cpp")
    write_source_facts(
        SourceFileFacts(file="rtl.cpp", source_hash="hash_a"),
        path,
    )

    facts = get_source_facts_for_file(tmp_path, "rtl.cpp", "hash_a")
    assert facts is not None
    assert facts.source_hash == "hash_a"


def test_get_source_facts_returns_none_without_source_root(tmp_path):
    """Cross-hash reuse needs a source_root for the git diff. Without it,
    only direct-hit is attempted."""
    write_source_facts(
        SourceFileFacts(file="rtl.cpp", source_hash="old"),
        layer1_cache_path(tmp_path, "old", "rtl.cpp"),
    )
    # Look for a different hash; no source_root → cannot reuse.
    assert get_source_facts_for_file(tmp_path, "rtl.cpp", "new") is None


def test_get_source_facts_reuses_older_hash_when_file_unchanged(tmp_path):
    write_source_facts(
        SourceFileFacts(file="rtl.cpp", source_hash="old"),
        layer1_cache_path(tmp_path, "old", "rtl.cpp"),
    )
    stub_git = _write_git_stub(tmp_path / "stubbin", exit_code=0)  # unchanged
    facts = get_source_facts_for_file(
        tmp_path,
        "rtl.cpp",
        "new",
        source_root=tmp_path,
        git_path=str(stub_git),
    )
    assert facts is not None
    assert facts.source_hash == "old"


def test_get_source_facts_skips_older_hash_when_file_changed(tmp_path):
    write_source_facts(
        SourceFileFacts(file="rtl.cpp", source_hash="old"),
        layer1_cache_path(tmp_path, "old", "rtl.cpp"),
    )
    stub_git = _write_git_stub(tmp_path / "stubbin", exit_code=1)  # differs
    facts = get_source_facts_for_file(
        tmp_path,
        "rtl.cpp",
        "new",
        source_root=tmp_path,
        git_path=str(stub_git),
    )
    assert facts is None


def test_new_write_does_not_touch_old_hash_entry(tmp_path):
    """Rebuilding under a new hash must not overwrite the old hash entry."""
    old_path = layer1_cache_path(tmp_path, "old", "rtl.cpp")
    write_source_facts(SourceFileFacts(file="rtl.cpp", source_hash="old"), old_path)

    new_path = layer1_cache_path(tmp_path, "new", "rtl.cpp")
    write_source_facts(SourceFileFacts(file="rtl.cpp", source_hash="new"), new_path)

    old = read_source_facts(old_path)
    new = read_source_facts(new_path)
    assert old is not None and old.source_hash == "old"
    assert new is not None and new.source_hash == "new"


def test_get_or_extract_facts_populates_cache_on_miss(tmp_path):
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helper.cpp").write_text(
        """
void Cone::pick()
{
    _rtl_alt = fallback();
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")

    cache_root = tmp_path / "cache"
    path = layer1_cache_path(cache_root, "hash", "src/modules/example/helper.cpp")
    assert not path.exists()

    facts = get_or_extract_facts(
        profiler,
        cache_root,
        "src/modules/example/helper.cpp",
        source_hash="hash",
    )
    assert path.exists()
    assert facts.file == "src/modules/example/helper.cpp"
    assert facts.source_hash == "hash"
    assert any(a.target == "_rtl_alt" for a in facts.source_assignments)


def test_get_or_extract_facts_uses_cache_on_hit(tmp_path):
    """Second call under the same hash should read from disk, not re-parse."""
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    file_rel = "src/modules/example/helper.cpp"
    file_abs = tmp_path / "PX4-Autopilot" / file_rel
    file_abs.write_text("void foo() { _x = 1; }\n", encoding="utf-8")

    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")
    cache_root = tmp_path / "cache"
    first = get_or_extract_facts(profiler, cache_root, file_rel, "hash")
    assert first.file == file_rel

    # Rewrite the file. If Layer 1 caching works, the second call should
    # NOT reflect the rewrite because we're still on the same hash.
    file_abs.write_text("void bar() { _y = 2; }\n", encoding="utf-8")
    second = get_or_extract_facts(profiler, cache_root, file_rel, "hash")
    # Same as the cached version — targets do not include _y.
    targets = {a.target for a in second.source_assignments}
    assert "_y" not in targets


def test_extract_facts_for_file_populates_from_profiler(tmp_path):
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "helper.cpp").write_text(
        """
void Cone::pick_altitude()
{
    if (_param_rtl_cone_half_angle_deg.get() > 0) {
        _rtl_alt = compute_cone();
    } else {
        _rtl_alt = fallback();
    }
}
""",
        encoding="utf-8",
    )

    profiler = MechanismSourceProfiler(tmp_path / "PX4-Autopilot", rg_path="missing-rg")
    facts = extract_facts_for_file(
        profiler,
        "src/modules/example/helper.cpp",
        source_hash="abcdef",
    )

    assert facts.file == "src/modules/example/helper.cpp"
    assert facts.source_hash == "abcdef"

    targets = {a.target for a in facts.source_assignments}
    assert "_rtl_alt" in targets

    # Multi-line if capture + else-tracking (from earlier profiler fixes)
    # means BOTH branches carry a control predicate.
    by_line = {a.line: a for a in facts.source_assignments if a.target == "_rtl_alt"}
    assert any(cp for a in by_line.values() for cp in a.control_predicates)


def test_legacy_layer1_keeps_pointer_effects_at_callee_storage(tmp_path):
    """DAG facts must not contain the profiler's flattened call-site write."""
    module_dir = tmp_path / "PX4-Autopilot" / "src" / "modules" / "example"
    module_dir.mkdir(parents=True)
    (module_dir / "output.cpp").write_text(
        """
struct output_s { float value; };

void fill(float input, output_s *out)
{
    out->value = input * 2.0f;
}

void run(float input)
{
    output_s result{};
    fill(input, &result);
    consume(result.value);
}
""",
        encoding="utf-8",
    )
    profiler = MechanismSourceProfiler(
        tmp_path / "PX4-Autopilot",
        rg_path="missing-rg",
        source_parser_backend="legacy",
    )

    raw = profiler.extract_source_assignments_from_source(
        ["src/modules/example/output.cpp"]
    )
    assert any(
        "legacy_pointer_output" in str(item.source_site_id or "")
        for item in raw
    )

    facts = extract_facts_for_file(
        profiler,
        "src/modules/example/output.cpp",
        source_hash="abcdef",
    )
    assert not any(
        "legacy_pointer_output" in str(item.source_site_id or "")
        for item in facts.source_assignments
    )
    assert any(item.target == "out.value" for item in facts.source_assignments)
