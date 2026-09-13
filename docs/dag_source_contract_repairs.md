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
