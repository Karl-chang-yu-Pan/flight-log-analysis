# Implementation Spec (rev. 2): DAG evaluator recursion → explicit worklist

Status: specification only. No code implemented. No tests written.

Contract: preserve all existing evaluator semantics exactly (frozen behavior,
including known approximations). No domain, boundary, or ownership change.
Identity/provenance workstream (Steps A–D, F1) is closed and must stay green.

## 1. Problem statement

`DAGValueSession` evaluates through mutual Python recursion
(`_evaluate_vertex` → activity/expression/branch → `_select_producer` →
`_evaluate_vertex`, plus expression evaluation driving operand resolution
through callbacks). A synthetic acyclic chain fails with `RecursionError` at
~129 vertices, so Python call-stack depth is a correctness/scale limit. A
takeoff run separately aborted with stack overflow, but that abort's exact
call site is unproven — this workstream fixes the proven recursion limit, not
the takeoff run. Forbidden fixes: raising `sys.setrecursionlimit`, depth
caps, truncation, swallowing `RecursionError`, weakening semantics.

## 2. Current recursive call graph

All recursion lives in `DAGValueSession`
(`flight_log_agent/analysis/dag_value.py`); pure helpers
(`_latest_ordered_producer`, `_producer_order`, `_unresolved`,
`with_dependencies`) and `_derive_leaf_dependencies` (already `deque`
iterative) never recurse.

- `evaluate` → `_evaluate_vertex` (entry; `active=frozenset()`, conditional
  from context).
- `_evaluate_vertex` → `_evaluate_evidence` (leaf, no recursion) /
  `_operation_activity` / `_evaluate_expression_vertex` / `_evaluate_branch`.
- `_operation_activity` → `_evaluate_vertex` per control branch, in
  `controls_by_target` order, with early `inactive` return on the first falsy
  value; non-value marks `unknown` and continues.
- `_evaluate_expression_vertex` → `resolve_operand` → `_select_producer`,
  driven inside `expression.evaluate(callback)`; first non-value operand
  raises and aborts the rest; missing operand records `source_linkage` and
  raises. Operand order is the expression engine's traversal order, not dict
  order.
- `_select_producer` → `_evaluate_vertex` for ALL producers in unique order
  (no early exit); conditional narrows to `conditional and
  len(unique_ids)==1` at selection start.
- `_evaluate_branch` → `_evaluate_expression_vertex` only on predicate
  fallback paths (tail position, unwrapped); otherwise inline verdicts.
- `helper_parameter` evidence → `_select_producer` like any operand.

## 3. Failure mechanism

Each dependency level consumes several Python frames, so depth scales with
graph depth (~7.75 frames/vertex observed). Default limits exhaust at ~129
chain vertices. Wide graphs are unaffected. No limit handling exists.

## 4. Existing cache/cycle model

- `_value_cache` / `_conditional_cache` keyed `(vertex_id, timestamp)`,
  chosen by the call's conditional flag (never cross-leak); read on every
  entry; written on every completion below.
- `_activity_results` keyed `(vertex_id, timestamp)`; read on activity
  entry (mapped back to active/inactive/unknown); written on all activity
  exits.
- `_sample_cache` keyed `(signal, timestamp)`; input sampling only.
- Cycle check is `vertex_id in active` (vertex ID ONLY — not timestamp or
  conditional flag) and runs AFTER memo lookup. The path-independent memo
  is a known approximation: preserve it, do not fix it.
- The two immediate outcomes below are NOT memoized and must stay that way.

## 5. Scope

Evaluator recursion only; explicit frames; evaluator-local memo/cycle
handling; equivalence + deep-stack-safety tests.

## 6. Non-goals

Identity/provenance/scope/linkage redesign; positive coverage; publication
consolidation; forwarding; receiver history; cache redesign beyond stack
safety; compaction; legacy retirement; `mechanism_dag.py` decomposition;
recursion-limit increases; any domain change.

## 7. Proposed explicit evaluator design

Public boundaries stay identical (`evaluate`, `evaluate_many`, `bind`,
`DAGValueProgram`, `DAGValuePlan`, `release_timestamp_values`,
`DAGValueResult`/`Issue` shapes, cache attribute names — tests read
`_value_cache`, `_conditional_cache`, `_sample_cache`, so names are
contract). The interior becomes a driver loop over an explicit heap stack,
reusing pure selection/merging helpers unchanged. Expressions keep running
inside the existing `expression.evaluate(callback)` via suspend/resume (§8).

## 8. Frame/state model

Settled states (no generic continuations; parent handling is a fixed
three-case enum: activity-parent, producers-parent, expression-parent):

- `ENTER_VERTEX(vertex_id, timestamp, conditional, path)`: memo lookup →
  hit delivers; else vertex-id path-membership check → cyclic result (no
  memo write); missing vertex → `source_linkage` result (no memo write);
  else context/resolver/pending/persisted gates inline, then dispatch by
  kind (evidence completes inline; operation pushes `ACTIVITY`; branch runs
  inline checks or tail-transitions to `EXPRESSION` for the same vertex —
  no branch continuation state exists because both fallbacks return the
  expression result unwrapped).
- `ACTIVITY(vertex_id, timestamp, branch_ids, index, unknown, dependencies)`:
  schedule control branches one at a time; falsy value completes `inactive`
  immediately attaching ONLY that result (prior accumulated dependencies
  are discarded, matching current behavior); non-value sets unknown and
  continues; end completes the proof with all evaluated gate dependencies;
  writes `_activity_results` on every exit.
- `OP_DISPATCH(vertex_id, timestamp, activity outcome)`: consumes a
  completed activity result and reproduces the operation ladder exactly —
  inactive returns immediately; unknown under ordinary evaluation returns
  the activity proof without touching producers/expression; unknown under
  the conditional+exact path continues into expression and tags
  `conditional_writer_ids` WITHOUT merging activity dependencies (differs
  from the ordinary path — preserved as is); active continues into
  expression and merges the activity proof.
- `PRODUCERS(client, producer_ids in unique order, index, candidates,
  conditional-at-start)`: schedules ALL producers sequentially (never
  short-circuits); on completion runs the unchanged selection ladder
  (single passthrough, ordered tie-break, unanimous merge, ambiguous/
  all-unresolved/inactive-none outcomes with their exact dependency rules).
  Clients are closed to: operand resolution and `helper_parameter` only.
- `EXPRESSION(vertex_id, role_order[], values{}, dependencies[])`: resume
  loop around `evaluate()` with a table-backed resolver (see §9); missing
  operand (`producer_ids == []`) records `source_linkage` and fails
  immediately — never a suspension.
- COMPLETE is a driver action, not a persistent frame: write the frame's
  memo entry (all outcomes except the two §4 exclusions), pop, deliver to
  the parent continuation.

## 9. Child-result/continuation handling and the `_Suspend` contract

`_Suspend` is a private evaluator-control-flow exception inheriting directly
from `Exception` — explicitly NOT a subtype of `SourceExpressionError`,
`ValueError`, `TypeError`, or `ArithmeticError` (all caught at the
evaluation-failure site). Only the driver catches it. Verified safe:
`evaluate()` and `evaluate_node` contain zero `try/except` (all existing
handlers are compile-time), so it propagates untouched; re-invocation is
valid because the operand environment is recreated per call, evaluation is
side-effect-free against DAG state, completed values come from the driver
table on resume, traversal/short-circuit stays inside the unchanged engine,
and repeated aliases stay separate occurrences (never deduplicated by
spelling — per-occurrence call operands suspend independently). The
resolver MUST consult the completed-operand table first and suspend only on
genuinely uncomputed operands. A pre-implementation transparency check
proving propagation is required before building around it (TDD step 5).

## 10. Cache/memo interaction

Identical keys, values, read-before-compute, and write timing as §4 —
including the two no-write outcomes. Frames plus the per-path vertex-ID set
are the sole in-progress representation: strict depth-first scheduling
means a second encounter either memo-hits or path-hits, so no pending-key
map is needed and none is added. Conditional/ordinary isolation rides the
frame flag into cache choice.

## 11. Cycle handling

Path stack replaces the `active` frozenset with identical vertex-id-only
membership semantics; memo-before-cycle ordering preserved; cyclic result
reason unchanged; missing vertex unchanged. Same vertex under different
timestamps or conditional flags behaves exactly as today (path ignores
both). No cycle "improvement" of any kind.

## 12. Failure/unresolved propagation

Every reason/issue/dedup behavior in §2's inventory is reproduced by
reusing the same construction code paths: cyclic (bare, uncached), missing
vertex/operand/producer (`source_linkage` with operand/vertex attribution),
`observation_binding`, `forbidden` (bare, no issue), `construction`,
`state_alignment` (both sites), `expression` (pure-computation errors only),
`control_flow` (assumption/unresolved-transfer/unreachable), `inactive`
writer short-circuit, `writer_coverage` (multi-producer failures only),
ambiguous/all-unresolved/no-active selection outcomes, timestamp-missing and
unevaluable signals. `with_dependencies` merging (first-seen order) is
unchanged, so issue chains and reason strings depending on evaluation
order (first-falsy gate, first-failed role `"{role}: …"`, candidate order)
are preserved by preserving traversal order — never by re-sorting.

## 13. Observable compatibility requirements

Same result sets for the whole existing corpus; same downstream
`ReplayStatus` behavior (replay/checkpoint consume `session.evaluate`
only); Steps A–D/F1 suites green with zero required semantic changes;
deterministic results for identical inputs; public signatures
(`evaluate(vertex_id, timestamp)`, `evaluate_many`, `bind` kwargs) and
cache attribute names preserved (tests monkeypatch and read them).

## 14. Affected files/functions

Production: `flight_log_agent/analysis/dag_value.py` only —
`DAGValueSession.evaluate` (delegates to driver), `_evaluate_vertex`
(dispatch or retirement subject to caller audit), `_operation_activity`,
`_evaluate_expression_vertex` (+ `resolve_operand` → table-resolver +
`_Suspend` protocol), `_select_producer` (loop → `PRODUCERS` frame,
ladder reused), `_evaluate_branch` (predicate-fallback tail path).
Tests: new deep/equivalence/short-circuit/cycle fixtures; REQUIRED updates
to `test_dag_value_session_shares_vertex_activity_and_sample_results`
(invocation counts are implementation detail — re-anchor to values +
single sample fetch with an explanatory comment) with justification;
`evaluate`/`evaluate_many` patches and cache reads keep working unchanged.

## 15. TDD plan

Seam (existing, highest): `DAGValueProgram(dag).bind(...).evaluate(vertex,
t)` — the Step A parity suite's seam; no new seams.
1. Equivalence fixtures (green pre-change): arithmetic, single/ordered/tied/
   ambiguous/missing producers, conditional writer + cache isolation,
   inactive writer, unknown activity, unresolved/missing dependency,
   evidence/parameter/constant leaves, persisted-state + pending holds,
   explicit 2-cycle (`unresolved` + exact cyclic reason — currently
   unpinned anywhere), observation resolver + forbidden signals.
2. Short-circuit pins: inactive-gated op with unresolvable operand stays
   `inactive` with no operand issue; first-failure `"{role}: …"` reason;
   per-occurrence call aliases stay separate.
3. Shared-child diamond: correct merged values, no duplicate issues.
4. RED deep chain (~500 vertices, exact terminal value — fails pre-fix via
   recursion depth; committed assertion semantic only, never
   `RecursionError`); RED layered diamond (depth ~150, true sharing,
   expression→producer→vertex recursion, semantic + issue assertions).
5. No constant folding exists (compile normalizes only), so depth is real.

## 16. Implementation sequence

1. Pin equivalence fixtures (§15.1–15.3). 2. Add RED deep fixtures;
   confirm recursion-depth failure. 3. Audit private-method callers/
   monkeypatched tests. 4. Verify `_Suspend` transparency + table-first
   resolver. 5. Build driver + finalized states (ENTER_VERTEX, ACTIVITY,
   OP_DISPATCH, PRODUCERS, EXPRESSION, completion action) reusing pure
   helpers. 6. Preserve exact memo/cycle semantics incl. no-write
   outcomes. 7. Mandatory temporary differential harness (test-only):
   full-result compare (status/value/reason/issues/observed/conditional
   ids) over corpus + new fixtures. 8. Resolve every unexplained diff.
   9. Swap public evaluation to the iterative engine. 10. Delete harness
   + obsolete recursive engine in the same slice (final diff: one
   evaluator). 11. Update implementation-detail tests only as §14
   requires, preserving intent. 12. Full nearby regression incl. A–D/F1.
   13. Optional supervised takeoff probe under the non-verification
   boundary (past the prior abort point only; never a correctness claim).

## 17. Acceptance criteria

Deep acyclic (~500) and branching graphs yield exact semantic values with
no recursion failure and no limit workaround (`setrecursionlimit`/
catching `RecursionError` absent — grep-verifiable); existing corpus green
with no semantic test edits (the §14 count-test revision excepted and
justified); cycle reason/behavior identical; memo-before-cycle and no-write
outcomes preserved; activity/operand/producer short-circuit distinctions
preserved; conditional/ordinary caches isolated; no duplicate shared-child
issues; issue order/reasons equivalent; A–D/F1 green; differential zero
unexplained diffs; final tree holds one evaluator.

## 18. Risks/open questions

`_Suspend` transparency (§16.4 gate) is the last load-bearing unknown;
everything else is settled design. Resume overhead (O(operands²) resolver
calls/vertex worst case) is accepted: real arities are single-digit.
Frames are O(depth) small heap dicts replacing O(depth) Python frames;
memo contract identical, so no new retained-state asymptotics are
introduced (byte-for-byte memory equivalence is explicitly not claimed).

## 19. Takeoff/large-case verification boundary

Outside the GREEN gate. If run supervised afterward: no abort at the prior
evaluation stage, advancement past it, observable pending/request/numeric
state only. Never a verified answer, coverage, or exit-code correctness
claim. No persistent caches or paid APIs involved.
