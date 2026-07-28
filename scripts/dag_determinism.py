"""Deterministic DAG-construction check for RTL and airspeed.

Builds the mechanism DAG for fixed (seeds, terminal) fixtures against the
pinned PX4 source — no LLM anywhere — several times under DIFFERENT
``PYTHONHASHSEED`` values, then checks that every build produced the
byte-identical graph. Set-iteration and dict-ordering nondeterminism
only surfaces across processes with different hash seeds, so each build
runs in its own subprocess; the parent never builds.

Two fingerprints per build:

* ``fp_full``   — the whole canonical DAG including vertex/edge IDs. This
  is the property caching needs: identical inputs must yield a
  byte-identical artifact.
* ``fp_struct`` — the graph with IDs replaced by the CONTENT of the
  vertex each edge points at. If ``fp_full`` diverges while ``fp_struct``
  holds, the graph is structurally identical but its IDs are unstable;
  if ``fp_struct`` diverges too, construction itself is nondeterministic.
* ``fp_render`` — the compact rendering that is the judge's ACTUAL input.
  It groups vertices by source site and walks adjacency through
  dicts/sets, so it can be unstable while both DAG fingerprints hold.

The ULog-derived inputs (observed catalogue, parameter values) and the
schema signals are computed ONCE in the parent and passed to every
build, so this isolates DAG-construction determinism from ULog-parser or
schema-loader variance.

Usage:
    .venv/bin/python scripts/dag_determinism.py [--backend legacy|tree_sitter|both]
                                                [--seeds N] [--work DIR]

Case-specific fixture names are deliberate — dev tooling pinned to known
mechanisms, not production logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SOURCE_ROOT = REPO_ROOT / "ref" / "PX4-Autopilot"
SOURCE_HASH = "determinism"

FIXTURES: dict[str, dict] = {
    "rtl": {
        "seeds": ["RTL_RETURN_ALT", "rtl_alt", "RTL::find_RTL_destination"],
        "terminal": "_rtl_alt",
        "terminal_file": "src/modules/navigator/rtl.cpp",
        "log": "uploads/053127df84724a75998ff355b5a2519a/RTL-wierd-height-1.ulg",
    },
    "airspeed": {
        # The published setpoint field, not the bare local ``airspeed_sp``:
        # that local is declared in two functions of one file and is
        # correctly rejected as ambiguous, which would make the check
        # vacuous. The topic member is an unambiguous decision terminal.
        "seeds": ["FW_AIRSPD_TRIM", "adapt_airspeed_setpoint", "airspeed_sp", "TECS"],
        "terminal": "tecs_status.true_airspeed_sp",
        "terminal_file": "src/modules/fw_pos_control/FixedwingPositionControl.cpp",
        "log": "uploads/8a8aa57fedf94caf833ab8b4ade11d25/airspeed-load-factor.ulg",
    },
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def canonical_dag(dag) -> dict:
    """Order-independent serialization of a MechanismDAG."""
    vertices = sorted(
        (v.model_dump(mode="json") for v in dag.vertices),
        key=lambda d: d["id"],
    )
    edges = sorted(
        (e.model_dump(mode="json") for e in dag.edges),
        key=lambda d: (
            d["source_id"],
            d["target_id"],
            d["kind"],
            d.get("role") or "",
            d.get("via") or "",
        ),
    )
    return {
        "terminal": dag.terminal,
        "dag_id": dag.dag_id,
        "vertices": vertices,
        "edges": edges,
        "unresolved_symbols": sorted(dag.unresolved_symbols),
    }


def fingerprints(canonical: dict) -> tuple[str, str]:
    """Return (fp_full, fp_struct).

    ``fp_full`` hashes the canonical DAG verbatim (IDs included).
    ``fp_struct`` rebuilds every edge as (content-of-source,
    content-of-target) so it is invariant to ID spelling — it isolates
    genuine structural change from unstable IDs.
    """
    fp_full = _sha(json.dumps(canonical, sort_keys=True))

    id_to_content: dict[str, str] = {}
    for vertex in canonical["vertices"]:
        content = {k: v for k, v in vertex.items() if k != "id"}
        id_to_content[vertex["id"]] = json.dumps(content, sort_keys=True)
    struct_edges = sorted(
        json.dumps(
            {
                "src": id_to_content.get(edge["source_id"], "?missing"),
                "dst": id_to_content.get(edge["target_id"], "?missing"),
                "kind": edge["kind"],
                "role": edge.get("role"),
                "via": edge.get("via"),
            },
            sort_keys=True,
        )
        for edge in canonical["edges"]
    )
    struct_blob = json.dumps(
        {
            "vertices": sorted(id_to_content.values()),
            "edges": struct_edges,
            "terminal": canonical["terminal"],
            "unresolved_symbols": canonical["unresolved_symbols"],
        },
        sort_keys=True,
    )
    return fp_full, _sha(struct_blob)


# ---------------------------------------------------------------------------
# Fixed-input precompute (parent, once per fixture)
# ---------------------------------------------------------------------------


def build_inputs(fixture: dict) -> dict:
    """ULog catalogue + parameters + schema signals — the fixed external
    inputs to construction, computed once so per-build variance in the
    ULog parser or schema loader cannot masquerade as DAG nondeterminism.
    """
    from flight_log_agent.px4.msg_schema import load_px4_msg_schema

    schema = load_px4_msg_schema(SOURCE_ROOT)
    schema_signals = sorted(f"{t}.{f}" for t, fields in schema.items() for f in fields)

    log_path = REPO_ROOT / fixture["log"]
    observed: list[str] = []
    parameters: dict = {}
    if log_path.exists():
        from flight_log_agent.ulog.inventory import (
            observed_signals_from_inventory,
            parse_ulog_inventory,
        )

        inventory = parse_ulog_inventory(log_path, SOURCE_ROOT)
        observed = sorted(observed_signals_from_inventory(inventory))
        parameters = dict((inventory or {}).get("parameters") or {})
    else:
        print(f"  WARNING: {fixture['log']} missing — grounding on schema only")

    return {
        "logged_signals": observed,
        "schema_signals": schema_signals,
        "parameter_values": parameters,
    }


# ---------------------------------------------------------------------------
# Single build (subprocess: --emit)
# ---------------------------------------------------------------------------


def emit(fixture_name: str, backend: str, inputs_path: str, out_path: str) -> int:
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag
    from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler

    fixture = FIXTURES[fixture_name]
    inputs = json.loads(Path(inputs_path).read_text(encoding="utf-8"))
    profiler = MechanismSourceProfiler(SOURCE_ROOT, source_parser_backend=backend)

    result = discover_mechanism_dag(
        profiler,
        REPO_ROOT / ".flightlog_cache",
        fixture["seeds"],
        fixture["terminal"],
        SOURCE_HASH,
        terminal_file=fixture["terminal_file"],
        logged_signals=set(inputs["logged_signals"]),
        schema_signals=set(inputs["schema_signals"]),
        parameter_values=inputs["parameter_values"],
    )

    validation = (
        result.terminal_validation.status if result.terminal_validation else None
    )
    if result.dag is None:
        status = {
            "fp_full": None,
            "fp_struct": None,
            "vertices": 0,
            "edges": 0,
            "validation": validation,
            "terminal": fixture["terminal"],
        }
        Path(out_path).write_text("null\n", encoding="utf-8")
    else:
        canonical = canonical_dag(result.dag)
        fp_full, fp_struct = fingerprints(canonical)
        Path(out_path).write_text(
            json.dumps(canonical, sort_keys=True, indent=1) + "\n", encoding="utf-8"
        )
        # The judge's actual input is the rendering, not the DAG: it groups by
        # source site and walks adjacency through dicts/sets, so it can be
        # nondeterministic while the DAG fingerprints are stable.
        from flight_log_agent.analysis.mechanism_judge import (
            render_discovery_compact,
        )

        render_blob = json.dumps(
            render_discovery_compact(result),
            sort_keys=False,
            separators=(",", ":"),
            default=str,
        )
        status = {
            "fp_full": fp_full,
            "fp_struct": fp_struct,
            "fp_render": _sha(render_blob),
            "render_chars": len(render_blob),
            "vertices": len(canonical["vertices"]),
            "edges": len(canonical["edges"]),
            "validation": validation,
            "terminal": fixture["terminal"],
        }
    print(json.dumps(status))
    return 0


# ---------------------------------------------------------------------------
# Orchestration (parent)
# ---------------------------------------------------------------------------


def run_fixture(fixture_name: str, backend: str, seeds: int, work: Path) -> bool:
    fixture = FIXTURES[fixture_name]
    print(f"\n=== {fixture_name} / {backend} — {seeds} builds, distinct PYTHONHASHSEED")

    inputs_path = work / f"{fixture_name}.inputs.json"
    if not inputs_path.exists():
        inputs_path.write_text(json.dumps(build_inputs(fixture)), encoding="utf-8")

    builds: list[dict] = []
    for seed in range(seeds):
        out_path = work / f"{fixture_name}.{backend}.seed{seed}.json"
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--emit",
                fixture_name,
                backend,
                str(inputs_path),
                str(out_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            print(f"  seed {seed}: BUILD ERROR (rc={proc.returncode})")
            print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-8:]))
            builds.append({"_seed": seed, "_error": True})
            continue
        line = proc.stdout.strip().splitlines()[-1]
        status = json.loads(line)
        status["_seed"] = seed
        status["_out"] = str(out_path)
        builds.append(status)
        print(
            f"  seed {seed}: v={status['vertices']} e={status['edges']} "
            f"valid={status['validation']} "
            f"fp_full={status['fp_full']} fp_struct={status['fp_struct']} "
            f"fp_render={status.get('fp_render')} "
            f"render_chars={status.get('render_chars')}"
        )

    ok_builds = [b for b in builds if not b.get("_error")]
    if len(ok_builds) < seeds:
        print("  RESULT: FAIL — one or more builds errored")
        return False

    full = {b["fp_full"] for b in ok_builds}
    struct = {b["fp_struct"] for b in ok_builds}
    render = {b.get("fp_render") for b in ok_builds}
    if len(full) == 1 and len(render) == 1:
        print(f"  RESULT: PASS — byte-identical across {seeds} hash seeds")
        return True
    if len(full) == 1:
        # The DAG is stable but the judge's actual input is not.
        print("  RESULT: FAIL — DAG stable but the RENDERING is NONDETERMINISTIC")
        return False

    if len(struct) == 1:
        print("  RESULT: FAIL — structure stable but vertex/edge IDs are NONDETERMINISTIC")
    else:
        print("  RESULT: FAIL — DAG structure itself is NONDETERMINISTIC")
    # Diff the first two divergent canonical dumps.
    a, b = ok_builds[0], next(x for x in ok_builds[1:] if x["fp_full"] != a["fp_full"])
    _diff(Path(a["_out"]), Path(b["_out"]), a["_seed"], b["_seed"])
    return False


def _diff(path_a: Path, path_b: Path, seed_a: int, seed_b: int) -> None:
    import difflib

    lines_a = path_a.read_text(encoding="utf-8").splitlines()
    lines_b = path_b.read_text(encoding="utf-8").splitlines()
    diff = list(
        difflib.unified_diff(
            lines_a, lines_b, f"seed{seed_a}", f"seed{seed_b}", lineterm="", n=1
        )
    )
    print(f"  first divergence (seed{seed_a} vs seed{seed_b}), up to 30 diff lines:")
    for line in diff[:30]:
        print("    " + line)
    if len(diff) > 30:
        print(f"    … +{len(diff) - 30} more diff lines")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", nargs=4, metavar=("FIXTURE", "BACKEND", "INPUTS", "OUT"))
    parser.add_argument("--backend", default="tree_sitter",
                        choices=["legacy", "tree_sitter", "both"])
    parser.add_argument("--seeds", type=int, default=3,
                        help="number of PYTHONHASHSEED values to build under")
    parser.add_argument("--fixtures", nargs="*", choices=list(FIXTURES),
                        default=list(FIXTURES))
    parser.add_argument("--work", default=None,
                        help="artifact directory (default: a temp dir)")
    args = parser.parse_args()

    if args.emit:
        return emit(*args.emit)

    backends = ["legacy", "tree_sitter"] if args.backend == "both" else [args.backend]
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="dag_determinism_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"artifacts: {work}")

    all_pass = True
    summary: list[tuple[str, str, bool]] = []
    for fixture_name in args.fixtures:
        for backend in backends:
            ok = run_fixture(fixture_name, backend, args.seeds, work)
            summary.append((fixture_name, backend, ok))
            all_pass = all_pass and ok

    print("\n=== SUMMARY")
    for fixture_name, backend, ok in summary:
        print(f"  {'PASS' if ok else 'FAIL'}  {fixture_name} / {backend}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
