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
New source files enter through the existing exact-source discovery rounds.
Checkpoint-controlled discovery retains one `DAGConstructionSession` and builder
across these rounds. Source admission refreshes declaration/call indexes and
suspends previously materialized values for revalidation. The demanded read
relationships and operation identities remain in the session; obsolete caller
instances cannot remain live producers after their reaching definitions change.

### Demand Contract Follow-up

The controller returns explicit `ConstructionDemand` operation IDs from checkpoint
construction requirements. Scheduling distinguishes guard construction, guard
source discovery, local calculation construction, other exact source discovery,
and blocked analysis. A missing guard helper suspends guarded value work while
its exact source request is searched. An exhausted search releases scheduling
priority, not the unresolved proof obligation: other work may resume in the same
builder while the guard stays unknown. No source/file/round budget was added.

The builder leaves unrequested work pending; an empty demand suspends instead of
expanding every deferred value. Legacy inactive-only callbacks remain available
for shared comparison tests. Source discovery still uses the existing typed
frontier, and neither pending work nor empty searches authorize verification.

Source-proven intermediate publications within the final target's dependencies
are assessed separately, including all their known alternative writers. Their
observations are loaded alongside the selected target's inputs. A local match
retains `known_graph` scope and cannot authorize stopping for the final question;
a local mismatch cannot be concealed by downstream numerical agreement. These
are views and evaluations of the same DAG, not a second mechanism or generated
formula. Intermediate requirements guide local construction without replacing
the question target.

Branch wiring considers exact source expression records before spelling-only
records at the same site, avoiding the observed consumer-arrival regression.
Shared eager/staged tests reverse consumer order and retain exact branch inputs.

Selection remains rooted in the final target's dependencies; it does not invent
bindings or seek unrelated observed outputs to obtain a passing comparison.
Derived snapshot wiring and source indexes are still rebuilt when required;
fully incremental indexing/wiring and run-local sample reuse are not claimed.
The prior experimental full runs stayed unresolved and increased RTL/airspeed
CPU cost; takeoff hit its memory guard. No full-run improvement is claimed for
this follow-up.

### Follow-up Verification

Focused checks passed: 389 tests, with two existing expected legacy-parser
failures. Coverage includes guard-source-before-value-source ordering, false
guards skipping value discovery, true guards resuming it, unknown guards
remaining obligations after exhausted searches, retained builder identity,
late-caller equivalence with a fresh build, and intermediate comparison scope.

Resource-audited RTL/airspeed first-round probes completed without guards and
stopped for guard-related source requirements with values still pending. These
are deterministic fixed-terminal checks with no APIs or persistent analysis
caches, not complete question-to-report runs. Commands, source hashes, exact
graphs/frontiers, demands and resource samples are retained under
`outputs/checkpoint_demand_audit_20260911/`.

Two-round probes also completed without resource guards. Each recorded exactly
one builder initialization and one source admission on that same builder, then
continued requesting guard-source dependencies with values still pending.

| Case | Two-Round Wall | Python CPU | Peak Tree PSS | Pending Values |
| --- | ---: | ---: | ---: | ---: |
| RTL | 24.65 s | 8.09 s | 147.72 MiB | 5 |
| Airspeed | 42.82 s | 10.62 s | 185.09 MiB | 16 |

Python CPU is from the final child event; PSS includes the sampled process tree.
Both retain unresolved requirements and neither is verified. Two rounds validate
source-admission continuation, not complete discovery, flight-wide correctness,
or a performance comparison with an exhausted full run. The audit adapter and
supervisor source are embedded in each scope's metadata JSON for reproducibility.

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
legacy and tree-sitter extraction, verifying unchanged causal renders, loaded
files, and reports with diagnostics on/off. The internal-value candidate may
add a downstream publication witness; the shared pipeline test asserts that
exact vertex/edge and diagnostic-write delta before comparing the remaining
payload. This is not unrestricted permission for judge-input differences.
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

## Conditional Observation Checks (2026-09-11)

The staged builder now retains source-backed publication copy paths to values
already present in the graph. These are downstream observation witnesses, not
invented subscription edges or inversions of arithmetic. The correspondence
requires parser-proven direct storage, scoped producers, a publication source
site, and an unambiguous observed topic instance. Casts and other transforms
are not treated as copies. The internal witness metadata is excluded from
serialized DAG/report schemas.

`dag_observation.py` checks local equations using the existing compiled graph
expressions and source-linked intermediate observations. Inputs must have the
same publication site, callable, topic instance and exact sample timestamp.
Ambiguous producers, cross-publication alignment, circular observations and
duplicate timestamps remain unevaluable. Explicit comparison windows are
honored; unresolved or assumption-dependent windows do not authorize a check.

A local match is conditional evidence only. It does not establish writer
applicability, representation/timing equivalence, alternative-writer coverage,
or intermediate-state initialization/history. The result explicitly retains
these obligations and cannot authorize discovery stopping. It does not feed
substitute values into the ordinary feasibility/replay session. No judge code,
prompt, report schema, persistent cache, or paid API was changed or used.

Source fixtures exercise the RTL floor, airspeed bank correction, TECS
reference equation and takeoff stage selection with synthetic identifiers.
These validate local calculation contracts, not the four flight explanations.
The legacy backend has an additional strict expected failure because it does
not provide the exact direct-copy metadata required by the shared contract.

### Real-Source Staged Audit

Artifacts and embedded harness/input hashes are under
`outputs/observation_checkpoint_audit_20260911/`. Commands:

```bash
.venv/bin/python /tmp/checkpoint_demand_audit.py --output outputs/observation_checkpoint_audit_20260911 --scope first-round --cases rtl airspeed tecs_height_rate takeoff
.venv/bin/python /tmp/checkpoint_demand_audit.py --output outputs/observation_checkpoint_audit_20260911 --scope two-round --cases rtl airspeed tecs_height_rate takeoff
```

These are fresh, fixed-terminal deterministic probes, with no question-intent
or judge APIs and no explicit question-event windows. They are not complete
question-to-report runs. Each case runs sequentially at reduced priority on
one CPU, with a 600-second timeout, 450 MiB sustained tree-PSS guard, 1000 MiB
emergency RSS guard, and minimum available host-memory guard. These are test
supervisor limits, not production discovery limits or completeness evidence.

| First-Round Case | Wall Time | Main Python CPU | Peak Tree PSS |
| --- | ---: | ---: | ---: |
| RTL | 15.58 s | 2.73 s | 109.84 MiB |
| Airspeed | 28.70 s | 12.28 s | 149.01 MiB |
| TECS height rate | 19.63 s | 6.39 s | 145.96 MiB |
| Takeoff | 44.34 s | 22.44 s | 186.60 MiB |

All first-round probes exited normally without resource guards. CPU figures
come from the main Python process's final event, not cumulative subprocess
CPU. Wall time includes source/log setup. Per-stage events and checkpoint
timings are preserved alongside the resource samples.

RTL still has no usable local equation observation in round one. Airspeed
has 13 publication-backed local candidates, all unevaluable because operands
have missing/alternative writers. Takeoff has 12 unevaluable candidates at
the mission-item altitude copy: its unbound member expression is not a
compiled arithmetic equation. TECS has no local arithmetic candidate and
stops on source linkage, including the debug-output member read. None of
these results verifies the flight explanation. The staged implementation is
not yet equivalent to the neighboring successful analyses.

Focused regression command:

```bash
.venv/bin/python -m pytest -q tests/test_dag_checkpoint.py tests/test_dag_pipeline.py tests/test_mechanism_judge.py tests/test_mechanism_discovery.py tests/test_mechanism_dag.py tests/test_tree_sitter_source.py tests/test_source_expression.py tests/test_mechanism_source_profiler.py
```

Result: **528 passed, 3 expected legacy-parser failures**, 16.95 seconds.
`git diff --check` also passed. The full unrelated repository suite, full
discovery-to-exhaustion probes, event-window flight explanations, and live
question-to-report/API behavior were not verified by these staged checks.

The observations expose a scheduling limitation as well as linkage gaps:
local equations whose inputs are pending still wait behind guard-source
requests. Conditional local calculation demand must eventually be distinct
from permission to consider a guarded writer active; supporting the former
must not silently grant the latter. Missing declaration/receiver links and
unknown alternative-writer coverage also remain separate obligations.

The first-round airspeed build also shows an added-cost regression relative
to `outputs/checkpoint_demand_audit_20260911/`: main-process construction CPU
increased from 1.37 to 8.77 seconds while the graph grew from 64/114 to 90/166
vertices/edges. This identifies the observation-enabled build for profiling;
it does not isolate an individual helper as the cause. Takeoff's first
post-checkpoint expansion requests 793 references (594 callable, 185 symbol,
14 storage-writer). A guard-first policy still authorizes substantial source
work before a useful local equation is available. Neither issue is addressed
by adding a production cap or treating an unresolved check as verified.

| Requested Two-Round Case | Wall Time | Main Python CPU | Peak Tree PSS | Outcome |
| --- | ---: | ---: | ---: | --- |
| RTL | 23.65 s | 8.28 s | 148.06 MiB | Guard source required, 5 pending |
| Airspeed | 72.59 s | 26.07 s | 188.20 MiB | Guard source required, 16 pending |
| TECS height rate | 29.21 s | 6.53 s | 139.53 MiB | Blocked after one round on source linkage |
| Takeoff | 548.40 s | 250.48 s | 359.72 MiB | Guard source required, 145 pending |

All exited normally without resource guards. Each used one builder;
RTL/airspeed/takeoff admitted source into that same instance once. TECS had
no source admission and did not fabricate a second round. No local equation
matched: airspeed retained 13 unevaluable candidates, takeoff 12, and the
other cases had none. Takeoff's second build consumed 324.19 seconds wall
time and 206.91 seconds main-process CPU after expansion/source admission;
its intermediate feasibility passes were short. Resumability is demonstrated,
but efficient construction and useful real-flight conditional evaluation are
not finished. Broader full-discovery runs were not launched after this
staged result exposed those remaining gaps.

## Local Input Demand Repair (2026-09-12)

The preceding implementation and audit are committed as `dd8ca3c`. The
follow-up connects local equation failures to structured `input_requirements`:
consumer vertex, operand, producer identities and existing typed source
requests. The controller may request these value dependencies before unrelated
guard-source discovery. This does not declare any guard satisfied or change
the final verification contract. Exact inactivity still prevents construction.

Source requests are selected using parser-proven value-call sites and operand
provenance, not all frontier records attached to the operation. The distinction
matters because guard and value calls can share a consumer origin. Observed
intermediate copies remain local input boundaries; a conditional match need
not reconstruct their histories. Missing/misaligned observations do not become
source-discovery requests. Alternative writers remain explicit requirements,
not an invitation to choose whichever expression matches.

New source-to-controller regressions demonstrate:

- A local equation resumes and matches while its guard remains unresolved.
- Its observed intermediate's history can remain pending at that match.
- A source-proven false guard still prevents value materialization.
- An unavailable value helper is discovered before the unknown guard helper,
  through its exact source call site; the local match cannot verify the question.
- Missing sample alignment does not authorize unrelated source work.

Fresh first-round probes used the existing one-CPU, reduced-priority supervisor
with 600-second/450-MiB sustained-PSS guards and no APIs or persistent caches:

```bash
.venv/bin/python /tmp/checkpoint_demand_audit.py --output outputs/local_input_demand_audit_20260911 --scope first-round --cases rtl airspeed
```

RTL completed in 17.60 s (104.17 MiB peak tree PSS), retaining its existing
guard-source requirement because no usable local observation is available.
Airspeed completed in 27.70 s (148.29 MiB peak tree PSS). Its next action changed
from `guard_source` to `local_calculation_source`: the local inputs were
materialized and the controller requested `_eas2tas` writer coverage. Its 13
checks remain unevaluable due to alternative writers, rather than the original
unmaterialized root. There are 31 pending operations across the larger partial
graph; that count is not itself a completion metric. Neither probe verifies
the flight explanation. These artifacts precede the final observation-error
routing refinement, which is covered by the subsequent regression run.
The same eight-file focused regression command recorded above passed:
**531 passed, 3 expected legacy-parser failures**, 7.61 seconds.
`git diff --check` passed. This is not a full-repository test result.

The initial helper regression failed on guard-first routing, then on guard/value
frontier mixing, then on rejecting a conditional helper expression. All three
were repaired without weakening final verification. Full discovery and live
question-to-report runs have not been repeated for this follow-up.

## Evaluation Consolidation (2026-09-12)

The local-input demand repair and deferred graph-compaction recommendation
are committed as `615467e`. This follow-up removes the separate recursive
local-equation evaluator. Local observations now enter `DAGValueSession`
through an explicit context; expression compilation, operand edges, producer
selection, parameter evaluation and prepared signal sampling use the shared
implementation. Publication-copy correspondence discovery is unchanged.

The context preserves the distinction between a conditional equation check
and a verified mechanism. A unique equation may be checked with an unknown
guard, with that condition recorded. Competing writers require source-backed
control and ordering evidence, never selection by agreement with the output.
Conditional results have separate session-local memoization so they cannot
leak into strict writer selection. Conditional subscriptions still require
receiver-state and transfer-time evidence; a topic sample alone is not that
evidence. Ordinary subscription semantics are not repaired by this change.

Shared producer selection also now retains missing candidate vertices and
rejects ambiguous source-order ties. A known later writer can supersede an
earlier unresolved writer only within the same proven callable/file order.
Structured dependency issues preserve the failing vertex, operand, producers
and nested failure reason. Selected values retain their observed guard inputs;
discarded writers do not contribute spurious construction requests.

The same graph fixtures and assertions exercise ordinary and local contexts.
Existing compile-once, parameter-normalization and evaluation-sharing tests
also run against both. Source-to-observation regressions cover piecewise
writers and unresolved alternatives. Focused command:

```bash
.venv/bin/python -m pytest -q tests/test_dag_value_context.py tests/test_mechanism_discovery.py tests/test_mechanism_dag.py tests/test_dag_checkpoint.py tests/test_dag_pipeline.py tests/test_source_expression.py tests/test_tree_sitter_source.py tests/test_mechanism_source_profiler.py tests/test_mechanism_judge.py
```

Result: **560 passed, 3 expected legacy-parser failures**, 6.06 seconds.
The tests exposed and then covered ambiguous writer ties and a missing-vertex
dependency-index failure during this implementation. This is not a full-suite
result or evidence that all real-flight observations are evaluable.

### Fresh Airspeed Probe

Artifacts are under
`outputs/evaluation_consolidation_final_audit_20260912/airspeed-first-round-staged/`.
The original temporary harness files were absent, so the probe reused the
supervisor and child source embedded in
`outputs/local_input_demand_audit_20260911/metadata-first-round.json`.
The new audit metadata records that harness, inputs and source hashes. It
retains the one-CPU, reduced-priority, 600-second and 450-MiB sustained-PSS
guards, emergency RSS/host-memory checks and network prohibition. These are
test-only resource guards, not production discovery limits.

The fresh first-round probe exited normally: **36.77 s wall, 17.60 s main
Python CPU, 149.21 MiB peak tree PSS**, with no resource guard triggered.
Its one builder produced 90 vertices and 168 edges, with 26 pending operations.
Construction, including embedded checkpoint passes, took 21.71 s wall and
13.94 s main-process CPU. Individual feasibility passes were below 0.09 s;
this probe did not stall in feasibility. The earlier local-input-demand probe
was 27.70 s and 148.29 MiB PSS, so this is not a demonstrated speed improvement.

All 13 local checks remain unevaluable. They are separate call instances of
the same published conversion, `_eas2tas * equivalent_airspeed_sp`, not 13
independent explanations or checks of the bank-angle minimum. The shared
evaluator now reaches the actual writer/guard obligations rather than
rejecting every operand with multiple producers unconditionally.

The immediate failure is the gate in `airspeed_poll`:

```cpp
(_param_fw_arsp_mode.get() == 0) && _airspeed_validated_sub.update(&airspeed_validated)
```

`FW_ARSP_MODE` is present as a parameter leaf. The update call has a source
call-site identity, but no graph-bound return-value operand; compiling this
gate reports `invalid expression syntax`. The two guarded `_eas2tas` writes
therefore remain unresolved alongside its header initializer. This is a
call-result linkage gap, not evidence that parameter access failed again.
The subsequent diagnostic-only addition of nested issue reasons is covered
by the final pytest run; the resource artifact predates that addition.

No RTL/TECS observation-path repair, graph compaction, judge change, persistent
analysis cache, paid API call or full-discovery run is included. Consolidating
the evaluator is implemented, but the real-flight acceptance case is not yet
verified. The next repair needs to trace the existing call-result/transfer
machinery before changing extraction or adding any new evaluator behavior.

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
