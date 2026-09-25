"""e2e_suite entrypoint regression tests (offline only).

Pins that the CLI main path actually reaches suite orchestration:
a past refactor stranded the run_suite(...) invocation as dead
code, turning every CLI run into a successful no-op. No provider
calls, no subprocesses, no fixtures beyond temp files.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import json as _json
from pathlib import Path as _Path


def _load_e2e_suite():
    path = (_Path(__file__).resolve().parent.parent
            / "scripts" / "e2e_suite.py")
    spec = _importlib_util.spec_from_file_location(
        "e2e_suite_under_test", path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_with_one_entry(tmp_path):
    log = tmp_path / "flight.ulg"
    log.write_bytes(b"ULog")  # existence only; never parsed here
    manifest = tmp_path / "manifest.json"
    manifest.write_text(_json.dumps(
        [{"log": str(log), "question": "Why did X happen?"}]))
    return manifest


def test_main_reaches_run_suite(monkeypatch, tmp_path):
    """A normal CLI invocation must reach run_suite exactly once
    (guards the silent-no-op regression where main returned
    before orchestration)."""
    e2e = _load_e2e_suite()
    manifest = _manifest_with_one_entry(tmp_path)
    calls = []

    def fake_run_suite(entries, out_root, timeout, limit,
                       provider_budget_json=""):
        calls.append({
            "entries": entries,
            "out_root": out_root,
            "timeout": timeout,
            "limit": limit,
            "provider_budget_json": provider_budget_json,
        })
        return 0

    monkeypatch.setattr(e2e, "run_suite", fake_run_suite)
    monkeypatch.setattr(
        "sys.argv",
        ["e2e_suite.py", "--manifest", str(manifest),
         "--out", str(tmp_path / "out"), "--limit", "1"],
    )
    assert e2e.main() == 0
    assert len(calls) == 1
    assert len(calls[0]["entries"]) == 1
    assert calls[0]["limit"] == 1
    assert calls[0]["provider_budget_json"] == ""


def test_main_forwards_budget_flags_to_run_suite(monkeypatch, tmp_path):
    """Provider-budget CLI flags serialize into the child transport
    unchanged (offline contract for future calibration runs)."""
    e2e = _load_e2e_suite()
    manifest = _manifest_with_one_entry(tmp_path)
    calls = []

    def fake_run_suite(entries, out_root, timeout, limit,
                       provider_budget_json=""):
        calls.append({"provider_budget_json": provider_budget_json})
        return 0

    monkeypatch.setattr(e2e, "run_suite", fake_run_suite)
    monkeypatch.setattr(
        "sys.argv",
        ["e2e_suite.py", "--manifest", str(manifest),
         "--out", str(tmp_path / "out"),
         "--max-provider-calls", "6",
         "--max-wall-seconds", "900"],
    )
    assert e2e.main() == 0
    assert len(calls) == 1
    payload = _json.loads(calls[0]["provider_budget_json"])
    assert payload["max_provider_calls"] == 6
    assert payload["max_wall_seconds"] == 900.0


def test_dry_run_reaches_validation_without_providers(
        monkeypatch, tmp_path, capsys):
    """Dry-run parses the manifest, validates, and exits
    successfully without touching provider-backed orchestration."""
    e2e = _load_e2e_suite()
    manifest = _manifest_with_one_entry(tmp_path)

    def forbidden_run_suite(*args, **kwargs):
        raise AssertionError("dry run must not reach run_suite")

    monkeypatch.setattr(e2e, "run_suite", forbidden_run_suite)
    monkeypatch.setattr(
        "sys.argv",
        ["e2e_suite.py", "--manifest", str(manifest),
         "--out", str(tmp_path / "out"), "--dry-run"],
    )
    assert e2e.main() == 0
    assert "no LLM call made" in capsys.readouterr().out
