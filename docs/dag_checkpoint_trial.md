# DAG Checkpoint Trial

## Purpose

Borrow the successful neighboring investigations' sequence of observing a
symptom, deriving a calculation from source, and comparing it with flight data.
Keep the existing DAG as the mechanism representation. The opt-in controller
now uses checkpoint requirements to select further source discovery and to stop
with verified or explicitly unresolved evidence. Diagnostic-only mode remains
available for comparison. The public report schema is unchanged.

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
- In diagnostic-only mode, keep full annotations and apply the existing
  pruning separately for discovery. Reuse the round's compiled program,
  session, and prepared samples. Only compact summaries survive a round.
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

An outstanding requirement prevents a numerical upgrade. Assessment alone
returns `verification_scope="known_graph"` and
`authorizes_discovery_stop=false`. Numerical agreement does not close an
outstanding source-writer request.

## Discovery Control

`checkpoint_discovery.evaluate_checkpoint_round` runs before full-round
feasibility. It selects the exact questioned observation, or the explicit
terminal when no questioned condition was supplied. A different observed
intermediate cannot silently stand in for that target.

1. Take an ancestor-closed view of the selected roots in the existing DAG.
2. Load its observed inputs and independent output, prepare samples, and compile
   one run-local value program.
3. Assess structural requirements without replay. Evaluate static gates, and
   schedule dynamic work only for gates whose inputs have no outstanding
   structural requirements. Disable assumption-based freshness verdicts.
4. Reassess exact feasibility, then replay only when requirements are discharged.
5. Stop `verified` only for a complete matching selected checkpoint without
   remaining requirements or source requests. Otherwise continue through the
   constructor's relevant typed requests, or stop `unresolved` when no further
   exact source request can produce new facts.

The graph retains all source vertices; selection is a view, not a second
mechanism. Dynamic feasibility releases per-timestamp memoization after each
shared batch, preserving static values and sample preparation. Frontier records
carry every consumer origin, including helper and terminal mutator requests.

A verified stop closes the stated terminal/questioned-signal investigation,
not the entire question. The judge receives its scope and outstanding
requirements. An unresolved checkpoint cannot become report confirmation even
if a judge is sufficient or a partial numerical result exists.

No file/round budget was introduced. Exact source admission is unchanged. Missing source-writer
coverage, receiver-state/transfer-time semantics, parser/linkage gaps, and
candidate-admission costs are reported or measured, not waived by this control.

### Staged Construction

Checkpoint-controlled runs now suspend value expansion at dependency boundaries
inside the same builder. Operations retain their source and invocation IDs.
Control dependencies are walked first, and each suspension presents a coherently
wired graph to the existing checkpoint assessment. Pending value operations are
an internal field excluded from the public DAG serialization, not opaque source
gaps or a second mechanism representation.

- Missing expression helpers cannot invoke the inline provider in staged mode.
  Their existing typed frontier preserves callable, receiver, source site, and
  consumer identities for checkpoint-selected source discovery.
- Known value work needed by controls resumes before guarded value work. Exact
  inactivity can discharge a writer's inputs; unknown guards cannot. Conditional
  subscriptions retain their receiver-state obligations.
- Discharged work is reassessed at later boundaries. New definitions that remove
  the inactivity proof cause that work to resume.
- Derived edges, evidence leaves, and wiring requests are rebuilt at each
  suspension against the current producers. Provisional opaque inputs cannot
  remain attached once their source producer has been materialized.
- Pending relevant construction prevents verification unless exact inactivity
  discharges it. Outstanding source-writer requests still prevent verification;
  exhausted search does not certify writer completeness.

Construction callbacks and completed-round callbacks are separate. Intermediate
audit events have `phase=construction` and compact resource/progress fields;
completed rounds retain the proof payload with `phase=round_complete`.

Eager construction remains the default without this option. Shared source-flow
tests exercise eager/staged construction and both parser backends where supported.
The real-log measurements below predate staged construction: they are a baseline,
not evidence of its performance. Terminal caller-context registration and each
control-dependency wave still run synchronously; the implementation does not
provide a resource preemption guarantee inside those operations.
New source files still enter through the existing outer discovery rounds; local
resumption does not yet retain the builder across a source-admission round.

## Enabling The Trial

The existing runner supports these environment switches:

```bash
export FLIGHT_LOG_DAG_DISCOVERY=1
export FLIGHT_LOG_SOURCE_PARSER=tree_sitter
export FLIGHT_LOG_DAG_CHECKPOINTS=1
```

The last switch is off by default and enables checkpoint-controlled discovery.
Direct callers can set `checkpoint_discovery=True` on
`run_dag_discovery_stage`. Set only `checkpoint_diagnostics=True` to preserve
the original diagnostic-only comparison. An optional `checkpoint_observer`
receives summaries, also returned in `checkpoint_rounds`.

The runner emits `dag_checkpoint.round.finished` audit events while discovery
is in progress. Controller events include combined checkpoint wall/CPU times;
diagnostic-only events separate feasibility and assessment. Both include
`process_peak_rss_kib`, the Linux process lifetime high-water mark,
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
Controller tests additionally cover verified/unresolved stopping before full
annotation, selective exact-source expansion, exhausted requests, grouped
consumer origins, streamed feasibility equivalence, and report downgrade even
when a judge claims sufficiency. Source constants containing NaN must not be
mistaken for stale program metadata.

The stronger numerical source-fixture expectation has a strict expected failure
for legacy extraction: its assignments mark expression dependencies inexact,
so replay remains unevaluable. Tree-sitter must pass the same assertion. This
is unresolved numerical retirement parity, not a reason to relax exactness.

Small numeric fixtures reproduce the archived RTL floor of 20 m and the
airspeed load-factor result of approximately 23.698446 m/s. They do not verify
full RTL cone applicability, flight-wide grounding, or stateful airspeed slew.
No archived generated script is treated as an unquestionable oracle.

## Deterministic Audit (2026-09-09)

Prior requirements work is committed as `a83745b`. The controller changes were
then tested through the same fresh source/log path, without paid APIs or
persistent analysis caches. These are fixed-terminal tests, not question-to-report
model runs. All four first-round raw vertex/edge mappings match the previous
audit; frontier origins intentionally differ.

| Case | Wall Time | CPU Time (Process Tree) | Peak PSS | Outcome |
| --- | ---: | ---: | ---: | --- |
| RTL | 343.0 s | 211.5 s | 392.8 MiB | Unresolved after 5 rounds, 31 files |
| Airspeed | 159.2 s | 148.5 s | 380.1 MiB | Unresolved after 2 rounds, 20 files |
| TECS height rate | 14.6 s | 8.8 s | 141.4 MiB | Unresolved after 1 round, 2 files |
| Takeoff | 341.3 s | 332.3 s | 450.4 MiB sustained | Guarded during the third build |

Tests ran sequentially with one CPU, reduced priority, a ten-minute per-case
timeout, and a 450 MiB sustained tree-PSS guard. Initial full runs had isolated
PSS doubling during subprocess activity; those failures are preserved. The
rerun confirms high readings after 0.1 seconds. Takeoff's final guard is a real
sustained crossing, not the rejected transient spike (492 MiB raw peak).

RTL checkpoint evaluation remained below 3.6 seconds per round; airspeed below
1.7 seconds in the final run. Source admission, including macro-definition
lookup, dominated expansion time. Takeoff's final captured stack is inside
the DAG builder's predicate-scope handling, before that round's checkpoint.

No real case was verified. RTL's internal terminal still lacks a publication
binding; both RTL and airspeed retain source, observation and receiver-state
requirements after exhausting available exact searches. TECS stops on unlinked
`debug_output.control.altitude_rate_control` and `_tecs_is_running` dependencies.
An auxiliary real-source motor-flags case also stayed unresolved because of
caller gates and an outstanding parameter-storage writer request.

Focused checks: **309 passed, 2 expected legacy-parser failures**. A broader run
had 31 runner-test import failures because its Pydantic stub lacks
`model_rebuild`; the same failure reproduces with the committed runner.
The full unrelated repository suite and live API decisions were not verified.

Artifacts: `outputs/checkpoint_control_audit_20260909T135250Z/audit.json`,
with per-case commands, input hashes, raw graphs, frontiers, checkpoint
requirements, per-stage timings, resource samples and captured stacks in
that directory and the earlier batch directories referenced by the audit.

## Remaining Work

Before treating the general constructor as complete:

1. Establish explicit, runtime-proven observation boundaries with storage,
   instance, transfer, and time identity. Remove remaining assumption-based
   construction/feasibility shortcuts rather than relying on local matches.
2. Derive positive source-coverage certificates for outstanding alternative
   writer requests; an empty search alone is not evidence of completeness.
   Extend dependency demand analysis without erasing unknown alternatives.
3. Represent persistent state updates and initialization from source facts;
   distinguish them from already-supported intra-call loop lowering.
4. Validate the full RTL and airspeed cases with resource-audited deterministic
   runs before approved live model runs. Do not introduce source-expansion
   budgets or revive a separate downstream verification plan.
