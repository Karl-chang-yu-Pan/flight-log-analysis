# DAG Source Contract Repairs

Status: source-contract repairs implemented and tested; flight verification remains incomplete.
Date: 2026-09-13.
Baseline: `a60fca9` (evaluation consolidation).

The findings below describe the pre-repair baseline. Implementation and current
validation results are recorded separately at the end of this document.

## Approved Direction

- Preserve graph relationships for boundary call execution, returned result,
  and resulting or persisted payload state. None substitutes for the others.
- When runtime evidence is missing, retain a precise unresolved obligation.
  Topic presence must not be treated as proof of subscriber consumption or
  call success, even if this prevents final verification.
- Extend shared source-to-DAG-to-evaluation contracts across both extraction
  backends. Unsupported legacy behavior must have explicit expected failures,
  not a weaker replacement-backend acceptance criterion.
- Follow small contract tests with resource-bounded RTL, airspeed, and TECS
  probes asserting the repaired relationships. Successful execution alone is
  not acceptance.
- Keep linkage repair acceptance separate from flight mechanism verification.
- Permit conditional equation checks when an independent, justified observation
  correspondence supplies the required values, even if boundary execution or
  state alignment remains unresolved. Preserve those obligations separately:
  numerical agreement cannot establish execution or complete verification.
- Reuse existing declaration, expression, call-result, producer-resolution,
  and evaluation machinery. Do not introduce mechanism-specific rules.
- Judge/report changes and graph compaction are outside this repair discussion.

See [the glossary](../CONTEXT.md) for terminology and
[the checkpoint trial](dag_checkpoint_trial.md) for earlier evaluation evidence.

## Findings and Repair Contracts

### 1. Early-Return Predicate Dependencies

In `flight_log_agent/px4/tree_sitter_source.py`, `_walk_sequence()` synthesizes
the negated fallthrough condition with exact inputs but without its symbols,
storage identities, or call-result references. The builder trusts the empty
dependency list. The audited saved graphs contain two affected RTL branches
and 13 each in airspeed and TECS.

Required contract: composing or negating a predicate preserves the dependency
identities needed to evaluate it. An exact empty operand set is valid only
when the expression genuinely requires no external operands.

Acceptance: extend the existing guard-clause contract through graph wiring
and evaluation, including parameter, local-storage, and call-result operands.
Assert the actual edges and truth values, not just predicate text or exactness.

### 2. Expression-Level Effect Reachability

`_collect_operations()` currently supplies enclosing statement controls to
every expression operation. A call on the right of `A && call()` does not
receive the additional condition `A`. An audited airspeed subscription
transfer consequently has exact, unconditional reachability.

Required contract: effects inherit the execution conditions imposed by
short-circuit and conditional-expression structure, as well as enclosing
statements. Expression evaluation and extraction of side effects must agree.

Acceptance: cover `A && call()`, `A || call()`, ternary arms, and nested forms.
Verify that skipped effects are not represented as executed. Keep the call's
execution condition distinct from whether its result indicates a transfer.

### 3. Consumed Boundary Return Values

`analysis/mechanism_discovery.py` supplies a payload transfer while
`analysis/mechanism_dag.py` excludes the boundary call from helper expansion.
No corresponding return-value producer is wired into the consuming predicate.
The raw `update(&payload)` expression therefore cannot compile as a graph-bound
operand. This is separate from expression-level reachability.

Required contract: retain a call-site-bound result and its relationship to
execution and payload state. Missing runtime knowledge remains unresolved;
do not replace the result with true or equate publication with consumption.

Acceptance: extend the existing boundary-in-predicate test beyond absence of
a helper frontier. Assert a consumed result relationship, evaluability with
sufficient evidence, and explicit unresolved behavior without that evidence.
Include non-execution, failed transfer, and persisted payload scenarios.

### 4. Braced Reference Initialization

These reference initializations should preserve equivalent result identity:

```cpp
const Status &a = object.get();
const Status &b{object.get()};
```

The braced form retains the initializer-list AST wrapper, so direct call-result
metadata is absent. Existing member projection cannot follow the alias. This
affects the audited TECS `debug_output` member path.

Required contract: derive reference binding from declaration and initializer
structure and feed the existing projection machinery. Do not assume arbitrary
single-element object construction is an identity-preserving reference bind.

Acceptance: use equivalent reference forms in the same shared contract and
assert projected member flow through the call result. Add a contrasting value
construction case to guard against overgeneralization.

### 5. Resolved Inputs and Outstanding Discovery

`_resolve_symbol_producers()` records a `storage_writers` request before
checking parameter resolution. The audited airspeed graph contains both a
resolved parameter leaf and an outstanding request for its wrapper storage.

Required contract: distinguish authoritative resolved input dependencies from
genuine mutable-state writer-coverage obligations. Resolving one operand must
not silently certify coverage for unrelated or still-mutable storage.

Acceptance: a resolved parameter must not remain blocked solely by its redundant
wrapper request. A genuine unresolved member writer must remain required.
Exhausting a search must not, by itself, establish completeness.

## Verification Sequence

1. Trace and extend existing contracts before implementation; use
   `scan-before-feature` and interchangeable retirement tests.
2. Run small synthetic source-to-evaluation cases for both backends, preserving
   identical semantic expectations where the contract is supported.
3. Run focused pytest through the repository virtual environment after edits.
4. Use `resource-aware-test-audit` for bounded real-source/flight probes, starting
   with smaller scope before broader deterministic construction.
5. Record wall time, CPU, memory, guard exits, graph relationships, unresolved
   obligations, and numerical comparisons separately. Resource guards belong
   to test supervision, not production completeness logic.
6. Claim flight verification only if the relevant execution, observations,
   calculations, and coverage justify it. No paid API calls are authorized by
   this documentation approval.

## Runtime Evidence Boundary

- Boundary results and payload-transfer events have distinct graph identities
  tied to their invocation. Their shared provenance is not a success predicate:
  no Boolean or return-code convention is guessed.
- Without invocation-aligned evidence, result and receiver-state obligations
  remain unresolved. The repair does not reconstruct subscriber generation,
  consumption times, or receiver history from topic snapshots.
- Independent receiver observations may support conditional calculations but
  do not discharge those execution obligations or certify writer coverage.

## Pre-Implementation Brief

### Existing Path and Precedents

The active path is source extraction, `dag_inputs_from_facts()`, DAG building,
`DAGValueProgram` evaluation, and checkpoint assessment. The adapter is shared
by both extraction backends; it is not the retired heuristic parser.

- `_termination_expression()` and switch-exit helpers currently return text
  and exactness. `_walk_sequence()` cannot preserve operand identities from
  that interface. Commit `f34379b` added exactness without those dependencies.
- `_collect_operations()` walks expression descendants with one control set.
  Existing assignment/call emitters can accept richer controls; they need an
  expression-structure-aware traversal rather than a second operation emitter.
- `boundary_operation()` already preserves transfer site, storage identity,
  direction, and controls. It does not model the consumed returned result.
- `_wire_expression_helper_calls()` and `_record_call_grounding_roles()` already
  connect call-site-qualified result operands. Boundary classification currently
  exits before this wiring, avoiding bus-internal expansion but losing the result.
- `_initializer_ref()` and alias registration should share declaration-proven
  initializer unwrapping and retain the existing direct-result projection path.
- `_resolve_symbol_producers()` emits writer-discovery requests before classifying
  resolved inputs. Genuine mutable producer coverage must survive this repair.
- `DAGValueSession` and checkpoint assessment already distinguish unresolved
  state alignment and block verification on outstanding obligations. Local
  observation checks already have a separate conditional-evaluation context.

### Proposed Changes

1. Carry structured predicate dependencies through termination and switch-exit
   composition and negation, including original call sites and storage identities.
   Preserve existing conservative handling of unsupported control flow.
2. Extend expression traversal to derive controls for short-circuit RHS effects
   and conditional arms. Preserve existing source ordering, deduplication, and
   exclusion of nested callable bodies. Do not assume overloaded operators have
   built-in short-circuit semantics when available type evidence contradicts it.
3. Wire consumed boundary results through the existing call-site operand
   mechanism while retaining exclusion from unrestricted helper expansion.
   Represent execution, transfer, and persisted state as related but distinct
   graph facts. A result lacking evidence remains an explicit runtime obligation,
   not malformed expression syntax or a generic source-search request. Derive
   transfer-success conventions from source; do not equate every result with
   Boolean success or invent samples from topic presence.
4. Unwrap a braced reference initializer only when declaration structure proves
   reference binding. Reuse that decision for expression and alias metadata so
   existing member projection works consistently.
5. Classify authoritative parameter/constant inputs before scheduling redundant
   wrapper discovery. Preserve requests for genuine mutable reaching definitions;
   do not remove requests solely because one producer or an empty search exists.

### Impact and Compatibility

Expected implementation scope: `px4/tree_sitter_source.py`, the shared source
fact adapter and DAG builder, and value/checkpoint handling for explicit boundary
obligations. Extend existing source contracts and downstream tests. If shared
fact metadata needs extension, keep existing serialized fields compatible and
update both adapters; do not change report schemas or judge behavior.

Confirmed policy: generic source-derived behavior, graph-native dependencies,
strict verification obligations, and conditional local checks are required.
The boundary success/state semantics available from particular source definitions
and runtime observations remain evidence to establish, not assumptions to add.

The existing boundary-in-predicate test currently asserts only transfer presence
and absence of helper discovery. Extend that same contract through result wiring
and evaluation. Similarly extend guard-clause and reference-projection contracts,
then test checkpoint closure for resolved inputs versus mutable coverage.

Use the verification sequence above. No new resolver, persistent cache, production
budget, paid API call, or parser retirement is proposed. The user approved this
brief before implementation.

## Implementation and Test Audit

### Implemented

- Termination and switch-exit composition carry structured operand identities
  and call results through negation rather than emitting exact empty inputs.
- Expression effects inherit short-circuit and ternary-arm controls. Declared
  class operands are conservatively retained without invented short-circuit
  guards, pending operator resolution.
- Boundary returned values use existing call-site operand edges. Transfer events
  gate payload-copy definitions and remain distinct from return values. Projected
  fields reuse their preserved original source site to find the same event.
- The compiler tokenizes parser-proven C++ call operands containing address
  arguments before Python AST parsing. It does not execute or numerically
  reinterpret those calls; repeated occurrences retain separate operand IDs.
- Braced reference initializers feed existing alias and call-result projection.
  Ordinary value construction is not promoted to a direct reference alias.
- Parameter and source-constant classification precedes redundant storage-writer
  requests. Genuine mutable producer coverage remains required.
- A failed/skipped transfer at the current timestamp cannot make an initializer
  stand in for persisted receiver state. Independent receiver observations use
  the existing evaluation context; otherwise the temporal obligation remains.

### Focused Verification

```bash
.venv/bin/python -m pytest -q tests/test_mechanism_discovery.py tests/test_mechanism_source_profiler.py tests/test_tree_sitter_source.py tests/test_source_expression.py tests/test_dag_value_context.py tests/test_mechanism_dag.py tests/test_dag_checkpoint.py tests/test_dag_pipeline.py tests/test_mechanism_judge.py --tb=short
```

Latest result: **593 passed, 29 xfailed in 11.56 seconds**. Expected failures
are explicitly labelled legacy contract gaps, including parameterized cases;
they are not evidence of completed backend parity. This is the focused analysis
suite, not the entire repository suite or a live question-to-report run.

The shared tests cover early-return value evaluation, effect reachability,
object and C-API boundary results in eager/staged construction, reference-form
equivalence and nested projection, and resolved versus mutable writer requests.
Additional tests cover address-call occurrence identity, class logical operands,
ordinary value construction, and persisted receiver state after a skipped copy.

During implementation, stronger tests exposed the address-call compilation
ordering defect. The first flight probes then exposed projected transfers missing
their event edges; the repair now uses `projection_source_site_id`, and the shared
boundary test asserts the edge for every copy, not just the aggregate.

Connecting those projected copies exposed eight pre-existing test expectations
that topic samples prove receiver state. These conflicted with the approved
contract. Numerical checks now supply explicit synthetic receiver observations;
negative checks verify that topic-only evidence cannot authorize checkpoint
completion. Those failing intermediate runs were investigated, not hidden by
retries or by weakening the graph assertions.

The final review additionally tested pruning of a skipped transfer. Removing its
persisted-state definition incorrectly exposed the initializer as the current
value in both ordinary and local contexts. Pruning now retains that definition
and its false control gate. Nested execution guards also retain source order,
so an outer false guard does not request an unevaluable inner operand.

### Bounded Flight Probes

The existing supervisor and child harness are loaded from
`outputs/local_input_demand_audit_20260911/metadata-first-round.json`, without
changing its deterministic driver or invoking seed/judge APIs. The final audit
stores the executable harness, input fixtures, source hashes, and guards in
`outputs/dag_source_contract_final_audit_20260913/metadata.json`.

Cases are RTL, airspeed, and TECS height rate, each first-round then two-round.
Runs are sequential, one CPU, reduced priority, no network, with 600 seconds per
probe, 450 MiB sustained tree PSS, 1000 MiB emergency tree RSS, and a 180 MiB
host-available-memory guard. These are test limits, not production search caps.
Historical first-round measurements suggested roughly 10-40 seconds and 110-160
MiB PSS per case; larger follow-up rounds remain supervised rather than assumed
to have the same cost.

Earlier diagnostic artifacts remain in
`outputs/dag_source_contract_audit_20260913/`; they precede the projected-event
and persisted-state fixes and must not be presented as final-code acceptance.

### Probe Results

All six probes exited normally, with no resource guard triggered. Times are
supervisor wall time; memory is peak process-tree PSS.

| Case | Scope | Wall seconds | Peak PSS MiB |
| --- | --- | ---: | ---: |
| RTL | First round | 9.05 | 111.36 |
| Airspeed | First round | 37.77 | 156.94 |
| TECS height rate | First round | 27.68 | 153.42 |
| RTL | Two rounds | 22.63 | 151.03 |
| Airspeed | Two rounds | 88.16 | 361.18 |
| TECS height rate | Two rounds | 64.47 | 179.37 |

Two-round main-process CPU was 8.49 s, 38.25 s, and 31.85 s respectively.
Construction (including embedded checkpoints) occupied 3.14 s, 43.36 s, and
43.77 s wall respectively. Individual feasibility passes were below 0.02 s.
Airspeed spent about 25.6 s between starting expansion and admitting source.
These runs did not stall in feasibility; expansion and construction dominate.

Direct artifact checks established:

- RTL's two affected fallthrough branches now have parameter and enum edges
  and both evaluate `always_true` against the recorded parameter values.
- Airspeed and TECS each have all 13 `_tecs_is_running` fallthrough inputs
  connected to source writers, rather than exact empty operand lists.
- Airspeed's update predicate compiles successfully through its call-result
  edge. Both projected airspeed fields carry the mode guard and the same
  transfer-event identity. The redundant parameter-wrapper request is absent.
- TECS no longer has opaque `debug_output.control.altitude_rate_control`
  leaves. First-round projection requests `_tecs.getStatus`; second-round
  discovery resolves 13 call-instance return projections from `TECS.hpp`.

These checks do not establish a flight answer. RTL retains five pending
operations and no local numerical checks after two rounds. Airspeed retains
114 pending operations and 13 unevaluable checks, with zero comparisons;
receiver-state and alternative-writer obligations remain. TECS has 13
unevaluable checks in round zero, then zero local candidates in round one:
its getter projections become pure-copy operands and are skipped by the current
local-check selector while their return dependencies remain pending. The
controller returns to guard discovery with 54 pending operations. That is a
remaining observation/checkpoint scheduling gap, not repaired by braced-reference
extraction alone and not counted as a verified equation.

The final pruning-retention and nested-guard-order safeguards were added after
these resource measurements. They are covered by the final focused suite.
The saved annotated graphs were additionally passed through the updated pruner,
checking retention of every persisted receiver definition. The recorded probes
contain no prune-phase calls; they were not repeated after those final safeguards.
Their recorded source hashes must therefore not be represented as the final
working-tree revision.

No full-discovery completion, live question-to-report verification, or automatic
runtime transfer reconstruction is claimed by these bounded probes. Judge and
report integration remain outside this change. No paid API calls or persistent
analysis caches are used.

No ADR is created yet: these are repairs to existing contracts, and no new
architectural trade-off has been settled. Production edits require a completed
pre-implementation brief and user approval.

## Follow-Up: Forwarded Calculation Demand

The user approved retaining local-calculation demand after getter resolution,
staged acceptance tests, and recording this repair here. Existing evidence,
alignment, writer-coverage, and verification restrictions remain unchanged.

### Pre-Implementation Brief

The existing `dag_observation.evaluate_local_observed_equations()` skips a
non-pending root whose compiled expression is a single operand. That avoids
checking trivial copies, but also skips a resolved call-result operand while
its return dependencies remain pending. `checkpoint_discovery` prioritizes
local construction only when evaluation supplies those requirements.

Reuse `DAGValueSession` traversal, pending-construction diagnostics, and the
existing checkpoint priority order. Retain candidates whose single compiled
operand is a graph-bound call result supported by extracted call metadata.
Do not propagate the output observation into that producer, alter ordinary
copy filtering, or create a second resolver. The original publication and
comparison domain remain attached to the check.

### Implementation and Acceptance

- Extended the existing staged helper-discovery test with a forwarding call.
  Before the repair, source discovery succeeded but no numerical check matched.
  Afterward, the controller explicitly requests the pending helper return as
  local-calculation construction and reaches a conditional match despite an
  unresolved unrelated guard.
- The same staged contract runs against both extraction backends. Legacy
  remains a strict expected failure for missing exact local-equation operands;
  this is not completed retirement parity.
- Extended observation-safety tests to forwarding calls: ambiguous producers,
  cyclic dependencies, circular output evidence, sample alignment, publication
  identity, comparison scope, and numerical mismatch retain their safeguards.
  Initial negative fixtures changed an unused raw-input edge; they were corrected
  to change the consumed call-result edge before accepting these assertions.
- Focused suite: **604 passed, 31 xfailed in 12.12 seconds**. A subsequently
  strengthened pending-return assertion passed its targeted rerun:
  **2 passed, 2 xfailed**.

No real-flight construction was rerun for this follow-up. The earlier flight
artifacts still describe the pre-fix selector; flight acceptance remains pending.
Receiver-state evidence and runtime boundary outcomes are outside this repair.
No glossary term or architectural decision was added: this restores the existing
conditional-equation and construction contracts rather than introducing a new
domain concept.

## Repository Review: Remaining Failures

Review date: 2026-09-15. Reviewed revision: `f388ab9`.
Status: findings and proposed repairs, not implemented fixes.

The review traced current code, relevant commits, shared and implementation-specific
tests, saved flight artifacts, and small isolated reproductions. The main gaps are
composition between existing source, graph, discovery, and evaluation mechanisms;
they are not all missing tree-sitter extraction features.

### Latest Flight Evidence

After the forwarding repair, all four fixed-terminal cases were run at first-round
and two-round scope. This supersedes the earlier follow-up's statement that flight
probes had not yet been repeated. The run preceded the commit, so its metadata
records `da31d87` plus working-tree source hashes; the forwarding changes were then
committed as `f388ab9`.

Artifacts: `outputs/forwarding_demand_audit_20260913/`, including `metadata.json`,
`audit.json`, per-run graphs, frontier requests, checkpoint summaries, resource
samples, and stderr. These are generated local artifacts, not committed fixtures.

| Case | First-round wall seconds | Two-round wall seconds | Two-round peak tree PSS MiB | Outcome |
| --- | ---: | ---: | ---: | --- |
| RTL | 10.57 | 19.63 | 149.31 | Five pending operations; no local comparisons |
| Airspeed | 33.75 | 77.09 | 196.53 | Thirteen unevaluable local checks; zero comparisons |
| TECS height rate | 24.69 | 58.46 | 177.65 | Thirteen local checks retained; 41 pending operations; zero comparisons |
| Takeoff altitude | 41.26 | 332.74 | 339.49 | First round completed; second round aborted with stack overflow |

No resource guard triggered. These were bounded construction probes with no
LLM/API calls, not full discovery or question-to-report verification. Successful
process exit did not establish a flight mechanism. The review itself did not
rerun these expensive probes.

### 1. Inconsistent Observation Provenance

**Confirmed defect.** `_member_observation_leaf()` in
`analysis/mechanism_dag.py` converts a declared message-struct type into a logged
leaf without proving a transfer or publication relationship. Producer resolution
can prefer that leaf over ambiguous source writers. `dag_checkpoint.assess_checkpoint()`
rejects `grounded_via="declared_type"`, while ordinary value evaluation accepts it.

A small source fixture with a message-typed member and no boundary operation
evaluated `data.value + 1` as 11 from a topic sample of 10. This demonstrates the
builder/evaluator mismatch independently of the flight cases. Saved RTL, airspeed,
and TECS graphs contain such type-only leaves.

Proposed repair: retain a type match as an observation candidate, not usable
evidence. Keep source producers until an actual source/observation relationship
justifies using the recorded value. Apply the same provenance contract to ordinary
evaluation, feasibility, conditional checks, and checkpoint validation. Do not
make checkpoints accept the unsupported inference merely to achieve parity.

### 2. Evaluator Depth and the Takeoff Crash

**Confirmed depth defect; exact takeoff crash cause remains unproven.**
`DAGValueSession._evaluate_vertex()` and `_select_producer()` recursively evaluate
dependencies. Cycle detection does not protect deep acyclic graphs.

Separate resource-limited processes evaluated simple acyclic `x + 1` chains:

| Vertices, including constant leaf | Result |
| ---: | --- |
| 17 | Value 17 |
| 65 | Value 65 |
| 129 | `RecursionError` |
| 257 | `RecursionError` |

Each probe used under 50 MiB process RSS, a five-second CPU limit, a ten-second
parent timeout, and disabled core dumps. These diagnostics used the repository
virtual environment and did not alter the interpreter recursion limit. The
thresholds are observations of this runtime and expression shape, not proposed
production limits.

Takeoff separately aborted with `_Py_CheckRecursiveCall: Cannot recover from stack
overflow`. Its last completed event was checkpoint assessment during resumed
construction, before the next feasibility-start event. The traceback contains
asyncio frames but does not identify the recursive application call. Approximately
118 seconds elapsed between expansion start and resumed construction, followed by
175 seconds of resumed construction/checkpoint work. Pending operations reached
833 after source admission. Neither graph size alone nor the last completed event
proves the fatal call site.

Proposed repair: replace dependency recursion with explicit evaluation frames or
a worklist, preserving short-circuiting, writer selection, cycle diagnostics,
observation boundaries, and shared memoization. Independently capture the last
construction snapshot and application stack in a supervised takeoff reproduction.
Do not raise recursion limits, cap graph depth, or declare the crash fixed solely
because the small chain test passes.

### 3. Receiver Identity Lost Through Aggregate Projection

**Confirmed TECS defect, reproduced without a ULog.** A synthetic receiver updated
before a scalar getter returns the expected value 7. Returning the same state as
an aggregate and then projecting its field becomes unresolved.

The relevant composition is:

```text
receiver storage identity
  -> call-instance member identity
  -> aggregate helper return
  -> demanded field projection
  -> producer lookup / source-discovery request
```

`_identity_in_call_scope()` constructs a receiver-qualified subobject identity.
`_project_identity()` then preserves the composite declaration but replaces its
symbol with the callee-local spelling. `_resolve_symbol_producers()` requests
writers using the receiver root, and `_record_unresolved()` re-resolves that name
inside the callee scope rather than retaining the already-proven identity.

In TECS, `_debug_status.control.altitude_rate_control` consequently becomes opaque
while a writer request asks for `_tecs` in `TECS::getStatus`, where `_tecs` is
unknown. This corrects the earlier shorthand that no request exists: a request
exists in the raw frontier, but its identity/scope is degraded.

There is a second association defect. Local evaluation reports the opaque leaf;
the request's origin is its consuming helper-return operation. The local-check
`require()` function matches origins and operands only at the reported vertex,
so the request is absent from the local input requirement. The controller returns
to guard discovery even though the unresolved value has a source-discovery need.

Proposed repair: compose field projections with the existing storage identity,
keeping receiver instance, declaration ownership, and member path coherent and
distinct from source spelling. Pass that identity to discovery without rebuilding
it from a bare name. Preserve consuming vertex/operand provenance when propagating
unresolved-value diagnostics so the exact request reaches the controller. Do not
introduce a second resolver or broaden the search to compensate for lost scope.

The forwarding-demand repair itself works: all 13 TECS local candidates now
survive getter resolution and the pending return definitions are materialized.
That exposes this next defect; it does not complete flight verification.

### 4. Missing Parameter Scope in Call-Effect Discovery

**Confirmed RTL-related defect.** `_writers_from_relevant_calls()` calls
`_match_parameter()` without file, callable, or source-site context. The resolver
requires ownership when source declarations are authoritative. A small probe
returned `None` without scope and `CUSTOM_MODE` for the same accessor with scope.
The saved RTL frontier contains six parameter `get` requests.

Proposed repair: use the existing scoped parameter classification consistently
at call-effect discovery, including the actual caller's context. Assert both
positive parameter resolution and absence of redundant callable requests. Do not
weaken ownership checks or infer parameter identity from spelling.

This is not all of RTL's unfinished work. Navigator storage and clock dependencies
remain, the cone/max-altitude calculations are pending, and the internal `_rtl_alt`
terminal lacks a source-proven publication binding in the bounded graph. Further
discovery and output-observation investigation are still needed.

### 5. Runtime Evidence and Mutable-Writer Coverage

**Confirmed production capability gap; recording insufficiency is not established.**
Airspeed has the initializer, default assignment, constrained-ratio assignment,
and boundary relationships for `_eas2tas`. Selecting the retained value depends
on invocation outcomes and receiver history that the current production path
does not supply. `DAGValueSession` intentionally returns unresolved for those
obligations unless an independent observation supplies the required value.

The publication directly records equivalent airspeed, but not `_eas2tas` itself.
Deriving that factor from the true-airspeed output being compared would make the
check circular. More source may reveal valid observations, exclude dependencies,
or establish needed relationships; it cannot be assumed either sufficient or
useless before those paths are investigated.

Mutable-writer coverage has a separate completion gap. `storage_writers` requests
are retained as obligations; exhausted searches release scheduling priority but
do not certify coverage. No positive production path was demonstrated that closes
these retained obligations from a source-backed coverage result. An unsuccessful
search must remain distinct from proof that all relevant writers were considered.

Proposed repair: establish independent source-linked observations or temporal
relationships where evidence permits, and represent justified writer coverage
separately from search exhaustion. Reuse the existing graph and checkpoint
contracts. Retain unresolved status when evidence genuinely cannot establish the
value. Neither fabricated call success nor unrestricted source expansion is a fix.

### History and Coverage Gaps

| Commit(s) | What the work covered | Missing composition or acceptance boundary |
| --- | --- | --- |
| `558c849` | Receiver-object state and call order | Scalar getter tests assert graph reachability, not aggregate projection through numerical evaluation and discovery |
| `208641e` | Graph-native source flow, projection, and scoped identities | Receiver-qualified identity surviving field projection into producer lookup and frontier requests |
| `0536788` followed by `208641e` | Parameter-call filtering, then stricter ownership | The older call-effect caller still omits the context needed by the stricter resolver |
| `5f0ec4d`, `c79e13f`, `a60fca9` | Alias-cycle protection, shared evaluation, memoization, and local/ordinary parity | Deep acyclic graph evaluation; cycle termination and compile/sample counts do not cover stack depth |
| `684b57d` | Type-based observation fallback | No accompanying tests; later checkpoint rejection does not constrain builder or ordinary evaluator behavior |
| `da31d87` | Boundary obligations and rejection of unsupported receiver evidence | Positive numerical fixtures inject receiver observations, not a production evidence-acquisition path |
| `615467e`, `f388ab9` | Local-demand priority and call-result forwarding | The combination with aggregate receiver state and source-request propagation remains uncovered |

Relevant examples include `test_receiver_getter_reads_state_written_by_prior_call_site`,
`test_helper_call_instance_reaches_member_writer_in_sibling_method`,
`test_source_projection_respects_scope_and_terminates_cycles`,
`test_reference_initialization_projects_nested_call_result`, and
`test_local_equation_discovers_its_helper_before_unknown_guard`.

The existing tests are useful but often exercise one axis: scalar receiver state,
same-owner reference projection, static fake bindings, or a manually constructed
checkpoint graph. Shared backend tests can still miss the same downstream
composition on both paths. A numerical value can also coexist with redundant or
mis-scoped requests that prevent checkpoint completion.

The boundary repair intentionally changed earlier topic-only success assertions
into rejection assertions. That is correct under the approved evidence contract,
but successful rejection is not positive end-to-end capability. Likewise,
`requested_rounds_complete` in a bounded harness is not mechanism verification.
Previously reported pytest counts must not be used as proof of those missing
production paths.

### Proposed Repair and Acceptance Order

1. Extend existing shared contracts through identity, projection, request creation,
   source admission, numerical evaluation, and checkpoint scheduling. Cover scalar
   versus aggregate returns, nested fields, local/member receivers, pointer/reference
   forms, same-named storage in unrelated classes, and eager/staged construction.
   Keep genuine missing evidence as an explicit negative case, not a universal
   expected failure that hides loss of positive capability.
2. Repair TECS identity/request propagation and RTL's omitted parameter scope.
   A fully source-resolvable synthetic case must reach the expected value without
   an opaque substitute or redundant blocker. A missing writer must yield the
   exact scoped request that allows construction to resume.
3. Repair evaluator depth handling and isolate the takeoff abort. Cover deep
   acyclic graphs, cycles, short-circuit branches, multiple writers, and ordinary
   versus conditional evaluation without introducing production depth budgets.
4. Unify observation provenance and add positive production-path evidence and
   writer-coverage tests, alongside the existing fail-closed checks. Type alone
   cannot prove receiver state; an empty search cannot prove coverage.
5. Rerun all four cases with resource supervision and explicit assertions for
   numerical progress, pending work, source-request identity, and justified
   unresolved outcomes. Do not label bounded-run completion as a verified answer.

This review made no production changes and did not rerun the full pytest suite or
the flight probes. The new small diagnostics used the repository virtual environment,
no paid APIs, and no persistent analysis caches. Repair implementation requires a
completed pre-implementation brief and user approval; documenting these findings
does not authorize changing evidence policy or judge/report behavior.
