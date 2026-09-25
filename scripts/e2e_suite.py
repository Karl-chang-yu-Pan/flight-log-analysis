"""Sequential end-to-end suite runner with per-question checkpointing.

Runs :func:`analyze_flight_log` — the REAL entry point, including the question
intent stage, report plots and validation — once per manifest entry.

Why a subprocess per question, never a loop in one process:

* the DAG build peaks around 1 GB on a real slice, so questions must not
  overlap; a fresh process also returns every byte between questions;
* one question that hangs, OOMs or raises cannot take the suite with it —
  the parent records the failure and moves on;
* an interrupted suite resumes: a question whose result file already exists
  is skipped, so a crash at question 40 does not cost the first 39.

Manifest (JSON list); ``question`` is required because the repository stores
logs without the question that was asked of them::

    [{"log": "uploads/<id>/<name>.ulg", "question": "Why did ...?"}]

Usage::

    .venv/bin/python scripts/e2e_suite.py --manifest suite.json --out outputs/suite
    .venv/bin/python scripts/e2e_suite.py --manifest suite.json --dry-run
    .venv/bin/python scripts/e2e_suite.py --manifest suite.json --limit 1

``--dry-run`` validates the manifest and logs without spending a single LLM
call, which is the cheap way to confirm a suite before committing to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def question_id(log: str, question: str) -> str:
    """Stable id so a resumed run recognises an already-answered question."""
    digest = hashlib.sha256(f"{log}\x00{question}".encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Child: one question
# ---------------------------------------------------------------------------


def run_one(log: str, question: str, output_dir: str) -> int:
    import asyncio
    import json as _json
    import os as _os

    from flight_log_agent.provider_budget import (
        BudgetExceeded,
        ModelPrices,
        ProviderBudget,
    )
    from flight_log_agent.runner_core import analyze_flight_log

    budget = None
    raw_budget = _os.environ.get("FLIGHT_LOG_PROVIDER_BUDGET", "")
    if raw_budget.strip():
        payload = _json.loads(raw_budget)
        prices = None
        if isinstance(payload.get("model_prices"), dict):
            prices = {
                str(model): ModelPrices(
                    input_usd_per_token=float(spec.get("input_usd_per_token", 0.0)),
                    output_usd_per_token=float(spec.get("output_usd_per_token", 0.0)),
                    cached_input_usd_per_token=float(
                        spec.get("cached_input_usd_per_token", 0.0)),
                )
                for model, spec in payload["model_prices"].items()
                if isinstance(spec, dict)
            }
        budget = ProviderBudget(
            max_provider_calls=payload.get("max_provider_calls"),
            max_total_input_tokens=payload.get("max_total_input_tokens"),
            max_total_output_tokens=payload.get("max_total_output_tokens"),
            max_total_cost_usd=payload.get("max_total_cost_usd"),
            max_wall_seconds=payload.get("max_wall_seconds"),
            model_prices=prices,
        )

    started = time.monotonic()
    status: dict[str, object] = {"log": log, "question": question}
    try:
        report = asyncio.run(
            analyze_flight_log(
                log_path=log,
                user_question=question,
                output_dir=output_dir,
                dag_discovery=True,
                provider_budget=budget,
            )
        )
        hypotheses = list(getattr(report, "ranked_hypotheses", []) or [])
        top = hypotheses[0] if hypotheses else None
        status.update(
            {
                "ok": True,
                "hypotheses": len(hypotheses),
                "top_mechanism": (
                    str(getattr(top, "known_px4_mechanism", "") or "")
                    if top is not None
                    else ""
                ),
                "confidence": (
                    str(getattr(top, "confidence", "") or "")
                    if top is not None
                    else ""
                ),
                # The report's own verdict on whether the mechanism was shown
                # to fire in THIS flight — the nearest thing to a per-question
                # pass signal, since the repository stores no expected answers.
                "confirmed": len(list(getattr(report, "confirmed", []) or [])),
                "unconfirmed": len(list(getattr(report, "unconfirmed", []) or [])),
                "final_summary": str(getattr(report, "final_summary", "") or "")[:400],
            }
        )
    except BaseException as exc:  # noqa: BLE001 - the suite must survive anything
        status.update(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
        )
        # Explicit abort marking (machine-detectable): a budget
        # abort is neither success nor an ordinary failure.
        status.update(_abort_status_fragment(exc))
    status["seconds"] = round(time.monotonic() - started, 1)
    status["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print("SUITE_RESULT " + json.dumps(status, default=str), flush=True)
    return 0 if status.get("ok") else 1


# ---------------------------------------------------------------------------
# Parent: the suite
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> list[dict[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit("manifest must be a JSON list of {log, question}")
    cleaned: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        log = str((entry or {}).get("log") or "").strip()
        question = str((entry or {}).get("question") or "").strip()
        if not log or not question:
            raise SystemExit(f"manifest entry {index} needs both 'log' and 'question'")
        cleaned.append({"log": log, "question": question})
    return cleaned


def validate(entries: list[dict[str, str]]) -> bool:
    ok = True
    for entry in entries:
        path = REPO_ROOT / entry["log"]
        if not path.exists():
            print(f"  MISSING LOG: {entry['log']}")
            ok = False
    return ok


def run_suite(
    entries: list[dict[str, str]],
    out_root: Path,
    timeout: int,
    limit: int | None,
    provider_budget_json: str = "",
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    results_dir = out_root / "results"
    results_dir.mkdir(exist_ok=True)

    pending = []
    for entry in entries:
        qid = question_id(entry["log"], entry["question"])
        if (results_dir / f"{qid}.json").exists():
            continue
        pending.append((qid, entry))
    done_already = len(entries) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    print(
        f"suite: {len(entries)} questions, {done_already} already answered, "
        f"running {len(pending)} now (sequential, {timeout}s cap each)"
    )

    failures = 0
    for position, (qid, entry) in enumerate(pending, start=1):
        label = entry["question"][:60].replace("\n", " ")
        print(f"\n[{position}/{len(pending)}] {qid} {entry['log']}\n    “{label}”", flush=True)
        started = time.monotonic()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-one",
            entry["log"],
            entry["question"],
            str(out_root / "runs" / qid),
        ]
        environment = dict(os.environ)
        environment.setdefault("PYTHONPATH", str(REPO_ROOT))
        if provider_budget_json.strip():
            # Transport-only carrier for the child calibration guard;
            # parsed and validated by run_one, never trusted blindly.
            environment["FLIGHT_LOG_PROVIDER_BUDGET"] = provider_budget_json
        try:
            proc = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            line = next(
                (
                    raw
                    for raw in reversed(proc.stdout.splitlines())
                    if raw.startswith("SUITE_RESULT ")
                ),
                "",
            )
            if line:
                status = json.loads(line[len("SUITE_RESULT ") :])
            else:
                status = {
                    "log": entry["log"],
                    "question": entry["question"],
                    "ok": False,
                    "error": (proc.stderr or "no result line").strip()[-500:],
                    "seconds": round(time.monotonic() - started, 1),
                }
        except subprocess.TimeoutExpired:
            status = {
                "log": entry["log"],
                "question": entry["question"],
                "ok": False,
                "error": f"timeout after {timeout}s",
                "seconds": timeout,
            }
        status["question_id"] = qid
        (results_dir / f"{qid}.json").write_text(
            json.dumps(status, indent=1, default=str) + "\n", encoding="utf-8"
        )
        if status.get("ok"):
            print(
                f"    ok  {status.get('seconds')}s  rss={status.get('peak_rss_mb')}MB  "
                f"mechanism={status.get('top_mechanism') or '-'}  "
                f"confidence={status.get('confidence') or '-'}  "
                f"confirmed={status.get('confirmed')}",
                flush=True,
            )
        else:
            failures += 1
            print(f"    FAIL  {status.get('seconds')}s  {status.get('error')}", flush=True)

    summarize(results_dir)
    return 1 if failures else 0


def summarize(results_dir: Path) -> None:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_dir.glob("*.json"))
    ]
    if not rows:
        return
    ok = [r for r in rows if r.get("ok")]
    seconds = [float(r.get("seconds") or 0) for r in rows]
    peaks = [int(r.get("peak_rss_mb") or 0) for r in ok]
    print("\n=== SUITE SUMMARY ===")
    confirmed = [r for r in ok if int(r.get("confirmed") or 0) > 0]
    print(f"  answered: {len(ok)}/{len(rows)}   failed: {len(rows) - len(ok)}")
    print(f"  with a confirmed mechanism: {len(confirmed)}/{len(ok)}")
    if seconds:
        print(
            f"  time: total {sum(seconds)/60:.0f} min, "
            f"median {sorted(seconds)[len(seconds)//2]:.0f}s, max {max(seconds):.0f}s"
        )
    if peaks:
        print(f"  peak rss: max {max(peaks)} MB, median {sorted(peaks)[len(peaks)//2]} MB")
    for row in rows:
        if not row.get("ok"):
            print(f"  FAILED {row.get('question_id')}: {str(row.get('error'))[:120]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="JSON list of {log, question}")
    parser.add_argument("--out", default="outputs/suite", help="suite output root")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="per-question wall-clock cap in seconds",
    )
    parser.add_argument("--limit", type=int, default=None, help="run at most N pending")
    parser.add_argument("--max-provider-calls", type=int, default=None,
                        help="calibration guard: stop before exceeding N provider calls")
    parser.add_argument("--max-input-tokens", type=int, default=None,
                        help="calibration guard: stop before exceeding N completed input tokens")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="calibration guard: stop before exceeding N completed output tokens")
    parser.add_argument("--max-cost-usd", type=float, default=None,
                        help="calibration guard: stop before exceeding USD (needs --model-prices)")
    parser.add_argument("--max-wall-seconds", type=float, default=None,
                        help="calibration guard: stop before exceeding wall seconds")
    parser.add_argument("--model-prices", default="",
                        help='calibration guard: JSON {model: {"input_usd_per_token": f, "output_usd_per_token": f}}')
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate manifest and logs without any LLM call",
    )
    parser.add_argument("--run-one", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        return run_one(*args.run_one)
    if not args.manifest:
        parser.error("--manifest is required")

    entries = load_manifest(Path(args.manifest))
    print(f"manifest: {len(entries)} questions")
    if not validate(entries):
        return 2
    if args.dry_run:
        print("dry run: manifest and logs are valid; no LLM call made")
        return 0
def build_provider_budget_json(args) -> str:
    """Serialize calibration guard flags for the --run-one child
    transport. Empty string means no guard configured. Pure helper
    so offline tests can pin the CLI-to-guard contract."""
    budget_payload = {
        "max_provider_calls": args.max_provider_calls,
        "max_total_input_tokens": args.max_input_tokens,
        "max_total_output_tokens": args.max_output_tokens,
        "max_total_cost_usd": args.max_cost_usd,
        "max_wall_seconds": args.max_wall_seconds,
    }
    if args.model_prices.strip():
        budget_payload["model_prices"] = json.loads(args.model_prices)
    if any(value is not None
           for key, value in budget_payload.items() if key != "model_prices") \
            or budget_payload.get("model_prices"):
        return json.dumps(budget_payload)
    return ""


def _abort_status_fragment(exc: BaseException) -> dict:
    """Machine-detectable abort marking for suite results. Returns
    {} for ordinary failures so only budget aborts gain the flag."""
    try:
        from flight_log_agent.provider_budget import BudgetExceeded
    except ImportError:
        return {}
    if isinstance(exc, BudgetExceeded):
        return {"aborted_by_budget_guard": True,
                "abort_dimension": exc.dimension}
    return {}


    provider_budget_json = build_provider_budget_json(args)
    return run_suite(entries, Path(args.out), args.timeout, args.limit,
                     provider_budget_json)


if __name__ == "__main__":
    raise SystemExit(main())
