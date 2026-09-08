# DAG Checkpoint Trial

## Purpose

Borrow the successful neighboring investigations' sequence of observing a
symptom, deriving a calculation from source, and comparing it with flight data.
Keep the existing DAG as the mechanism representation. This first phase is
diagnostic: checkpoint results do not change source expansion, stopping rules,
judge prompts/payloads, or the public report schema.

## Implemented

- Resolve the existing typed questioned condition after seeding, before the
  first candidate slice. Missing units/references remain unresolved; no windows
  are not interpreted as a successful comparison. An unresolved source alias
  may be retried against each newly constructed graph.
- Group checkpoint roots by source-proven publication metadata. No new
  source-to-log mapping is inferred from names or declared types.
- Reuse one numerical replay implementation for terminal and checkpoint roots.
  Local operations execute through DAG edges with existing writer selection,
  signal policy, and resampling behavior.
- Compare within the requested windows without requiring unrelated portions of
  the log to match. Input series are not cropped, and no state initialization
  or finite history horizon is invented.
- Keep full annotations for diagnostics and apply the same existing pruning
  separately for discovery. Reuse the round's compiled program, session, and
  prepared samples. Only compact summaries survive a round.
- Reject circular comparisons where the compared output is an observed input,
  and reject declared-type-only observation bindings. Such cases need stronger
  temporal/provenance semantics rather than a numerical confidence upgrade.
- Record checkpoint errors, writer coverage, compiler dependency issues, and
  unresolved source references with their graph identities.

The numerical check covers the supplied graph, not the completeness of source
discovery. A local match does not establish that all applicable source writers
have been discovered, that upstream behavior is explained, or that the judge
should confirm a mechanism.

## Enabling The Trial

The existing runner supports these environment switches:

```bash
export FLIGHT_LOG_DAG_DISCOVERY=1
export FLIGHT_LOG_SOURCE_PARSER=tree_sitter
export FLIGHT_LOG_DAG_CHECKPOINTS=1
```

The last switch is off by default. Direct callers can instead set
`checkpoint_diagnostics=True` on `run_dag_discovery_stage` and supply an optional
`checkpoint_observer` callback. Results are returned in `checkpoint_rounds`.

The runner emits `dag_checkpoint.round.finished` audit events while discovery
is in progress. Events include separate feasibility and checkpoint wall/CPU
times and `process_peak_rss_kib`, the Linux process lifetime high-water mark,
not incremental memory allocated by a particular step.

Enabling the switch is not a deterministic-only runner: the normal agent path
still makes its existing API calls, which require user approval. Persistent
analysis caches remain disabled. The neighboring worktree is unchanged.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_dag_pipeline.py tests/test_mechanism_judge.py tests/test_mechanism_discovery.py tests/test_mechanism_dag.py
```

Shared numerical contracts exercise terminal and checkpoint entry points with
the same inputs and assertions. Source-to-pipeline trial tests run against both
legacy and tree-sitter extraction, verifying unchanged graphs, loaded files,
judge payloads, and reports with diagnostics on/off.

The stronger numerical source-fixture expectation has a strict expected failure
for legacy extraction: its assignments mark expression dependencies inexact,
so replay remains unevaluable. Tree-sitter must pass the same assertion. This
is unresolved numerical retirement parity, not a reason to relax exactness.

Small numeric fixtures reproduce the archived RTL floor of 20 m and the
airspeed load-factor result of approximately 23.698446 m/s. They do not verify
full RTL cone applicability, flight-wide grounding, or stateful airspeed slew.
No archived generated script is treated as an unquestionable oracle.

## Deferred

Before changing stopping rules or reporting scoped verification as confirmation:

1. Establish explicit, runtime-proven observation boundaries with storage,
   instance, transfer, and time identity. Remove remaining assumption-based
   construction/feasibility shortcuts rather than relying on local matches.
2. Derive claim-relevant dependency obligations, including unknown alternative
   writers. Compare diagnostic decisions with unchanged exhaustive discovery.
3. Represent persistent state updates and initialization from source facts;
   distinguish them from already-supported intra-call loop lowering.
4. Validate the full RTL and airspeed cases with resource-audited deterministic
   runs before approved live model runs. Do not introduce source-expansion
   budgets or revive a separate downstream verification plan.
