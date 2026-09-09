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
- Assess the checkpoint's value, control, and selection dependencies before
  replay. Require positive source-located publication and subscription
  transfers, actual observation availability, and known sampling policies.
- Return stable, run-local checkpoint identities, dependency vertex IDs,
  `analysis_requirements`, and the constructor's exact typed `source_requests`.
  Missing local producers are linkage requirements, not bare-name searches.
- Include every known alternative publication writer. Only exact evaluated
  false gates can discharge inactive writer inputs; assumption-based gates
  and incomplete evaluation domains cannot establish verification.
- When a terminal has no publication binding, emit `terminal_checkpoint`
  requirements rather than an unexplained empty checkpoint collection.
- Preserve internal source frontier references through feasibility annotation;
  their exclusion from the public serialized DAG schema is unchanged.

The numerical check covers the supplied graph, not the completeness of source
discovery. A local match does not establish that all applicable source writers
have been discovered, that upstream behavior is explained, or that the judge
should confirm a mechanism.

## Remaining-Analysis Requirements

`assess_checkpoint` in `analysis/dag_checkpoint.py` is a deterministic, no-I/O
entry point over an existing graph. It reuses `DAGValueProgram` and the shared
replay engine; it neither constructs a second mechanism nor searches source.
It can run on a small source-backed graph without exhaustive discovery.

Requirements distinguish scoped source discovery, broken source linkage,
missing observation bindings/data, sampling policy, control-flow exactness
and coverage, receiver-state/transfer-time alignment, expression support,
comparison scope, and incomplete replay. A false gate on an input copy does
not discharge receiver-state obligations: skipping a copy retains old storage
rather than proving it equals the latest topic sample.
Each requirement names its graph vertex and available source context. Source
requests are copied from graph-originated frontier records, never invented by
matching a local variable to a topic. A frontier without a graph origin is
reported as unclassified linkage rather than silently discarded.

An outstanding requirement prevents a numerical upgrade. A successful result
explicitly has `verification_scope="known_graph"` and
`authorizes_discovery_stop=false`. Discovery of additional writers can change
that assessment; numerical agreement does not close the source frontier.
Requirement collection is conservative, including dependencies of gates used
to prove inactivity. It does not yet trace short-circuit operand demands or
choose the next expansion request automatically.

The opt-in runner currently calls this assessment after the round's existing
feasibility pass. It therefore does not yet bypass the full-round feasibility
memory issue or candidate-admission macro-lookup cost found in the real-log
audit. Those costs, receiver-state/transfer-time semantics, and the TECS local
reference linkage remain separate work; this step makes their proof gaps
explicit instead of treating an incomplete checkpoint as verified.

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
.venv/bin/python -m pytest -q tests/test_dag_checkpoint.py tests/test_dag_pipeline.py tests/test_mechanism_judge.py tests/test_mechanism_discovery.py tests/test_mechanism_dag.py
```

Shared numerical contracts exercise terminal and checkpoint entry points with
the same inputs and assertions. Source-to-pipeline trial tests run against both
legacy and tree-sitter extraction, verifying unchanged graphs, loaded files,
judge payloads, and reports with diagnostics on/off.
The same source-to-pipeline fixture also checks missing observed inputs against
both backends. Assessment tests cover unrelated branches, scoped source
requests, declared-only observations, unlinked locals and guard operands,
assumed gates, inactive alternatives, and stale compiled programs.

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
