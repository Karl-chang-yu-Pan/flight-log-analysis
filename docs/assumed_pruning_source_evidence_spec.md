# Assumed-pruning causal terminal provenance (Workstream A spec)

Implementation-ready product contract. Workstream A closes one
confirmed defect: runtime terminal writers removed solely under an
ASSUMED feasibility verdict silently disappear from report source
evidence. General upstream/helper causal provenance is explicitly
out of scope here (see `docs/upstream_causal_source_evidence_spec.md`).

Status: specification only. No implementation, no fixtures, no tests.

## 1. Narrowed problem statement

For the RTL `_rtl_alt` terminal the discovery DAG holds five
terminal writer operations (`rtl.cpp:245` ×2, `rtl.cpp:248` ×2,
`rtl.h:163` ×1). During `annotate_dag → evaluate_feasibility →
prune_infeasible_operations`, the temporary freshness stopgap
marks the shared hrt control predicate `always_false`; the four
computation operations are removed together with the branch. The
final report can then point only at the `rtl.h:163` member
initializer, although deterministic reconstruction proves the
computation (`destination + 2 × acceptance radius = 103.93812`)
caused the logged command. Nothing records that the removal rested
on an assumption.

Product invariant (Workstream A scope only):

> A runtime terminal writer removed solely because of an ASSUMED
> feasibility verdict must remain available as clearly-marked
> causal provenance for reporting, without being reintroduced
> into execution, replay, checkpoint, coverage, applicability,
> retirement, or stop authority.

## 2. Current behavior (verified)

- Branch verdicts: `feasibility_verdict` in `unknown` /
  `always_true` / `always_false`; sample-derived verdicts carry
  `active_windows`; the stopgap records
  `metadata["static_evaluation"] = {"status": "value",
  "assumed": True, "reason": "temporary: uORB freshness gate
  assumed fresh"}` versus `{"status", "reason"}` without the flag
  (`mechanism_dag.py:5990-5994, 6118-6165`).
- `prune_infeasible_operations` (`mechanism_dag.py:6168-6197`)
  removes `always_false` branches and their control-targets
  (single false conjunct kills; `persisted_receivers` boundary
  transfers survive via the existing retention precedent at
  `:6170-6191`). The checkpoint assessment path already runs
  with `prune_dead=False` and `allow_assumptions=False`
  (`checkpoint_discovery.py:545-546`); the discovery-loop
  fallback and the stage annotate path prune with assumptions on
  (`mechanism_discovery.py:2208`, `dag_pipeline.py:978`).
- `build_report_from_dag` emits one `CodeRef` per `operation +
  file + is_terminal` vertex as `terminal write: {variable} <-
  {expression}`, no dedup, truncated `[:8]`
  (`dag_pipeline.py:638-669,779`). Upstream non-terminal
  operations are excluded by construction.
- Replay enumerates terminal operations of the annotated DAG and
  compares each writer only inside its gating branch windows
  (`dag_pipeline.py:490-560`).
- `CodeRef` (`models.py:8-14`) is structural location only
  (`file`, `function?`, `start_line?`, `end_line?`, `snippet?`,
  `explanation`); no status field exists.

## 3. Feasibility triple-read rule

Status is read off three existing fields together, never verdict
alone, never reason-string matching:

```text
ASSUMED  iff  metadata["static_evaluation"].assumed is True
DERIVED  iff  verdict from sample/window evaluation
         (active_windows / evaluation_domain present)
         and assumed is not true
PROVEN   iff  static value evaluation establishes the result
         and assumed is not true
```

Implementation invariant to test: DERIVED ∧ ASSUMED is
impossible under current logic (the assumption path at
`mechanism_dag.py:6125` is guarded by absence of windows).

## 4. Proof/replay/scheduling isolation

Retained provenance is diagnostic/report provenance only. It
must not affect graph scheduling (discovery-loop frontier keeps
current behavior), replay enumeration (vertices only; retained
writers would carry empty windows in any case), terminal
selection, coverage, applicability, retirement, checkpoint
evaluation, T6B authority, or proof stores. Audited consumers
(judge render, replay, checkpoint allowlist, value engine,
terminal/source selectors, proof stores, diagnostic counts)
read vertices, explicit keys, or proof state — never generic
DAG-level collections — so an additive DAG-level record list is
compatible. The checkpoint path's existing
`prune_dead=False` + `allow_assumptions=False` posture is
unchanged and is what keeps proof authority assumption-free.

## 5. Retained-record owner and capture

Owner: additive internal field on the annotated `MechanismDAG`,
following the `exclude=True` precedent (`observation_witnesses`,
`pending_construction`): internal-only, never serialized into
report schemas. Rationale: the report builder receives only the
annotated DAG (`dag_pipeline.py:571-586,1208`); diagnostics
dicts are run-local and `DiscoveryResult`/proof stores never
reach it. Capture inside `prune_infeasible_operations` before
the final pruned `model_copy` is constructed — while operation,
branch, verdict, and assumption metadata are still live. No
post-pruning reconstruction, ever.

## 6. Retained-record fields (minimal)

Per removed causal terminal writer, all plain data (no vertex
object, no graph snapshot):

```text
source file, source line/range, exact target symbol,
normalized expression / writer identity,
callable scope where available,
source-site / assignment identity,
terminal status held, controlling predicate identity,
feasibility verdict, windows state, assumed flag,
stable assumption category + exact stopgap reason
```

Vertex IDs may appear as diagnostic reference only — never as
semantic identity, ordering, or dedup input.

## 7. Stable identity and dedup

Dedup key (explicit, repository-grounded):

```text
(file, line/range, exact_target_symbol,
 normalized_expression,
 declaration_id OR assignment-path identity OR callable scope,
 call_source_site_id where semantically relevant)
```

Same source writer instantiated multiple graph times → one
ref; different source assignments → distinct. `file=None` /
`line=None` are legal: fall back to symbol + expression +
scope tuple, never drop silently, never synthesize identity.

## 8. Proven-false semantics

Genuinely PROVEN_FALSE → operation remains prunable and never
becomes causal report evidence. "Not report evidence" does not
mean "vanish everywhere": pre-prune views, the
`prune_dead=False` checkpoint path, and existing tests pinning
gone-ness keep current behavior. Execution semantics unchanged.

## 9. source_refs semantics (Workstream A)

For a selected terminal symbol, `source_refs` may contain
runtime terminal computation writers plus supporting
declaration/storage anchors. Assumed-pruned runtime writers are
candidate causal evidence, never proven executed operations.
Upstream non-terminal causal operations belong to Workstream B
and are not selected here.

## 10. Candidate marking convention (BLOCKING correctness)

Current `terminal write: {variable} <- {expression}` wording
implies surviving terminal evidence and must NOT be used for
assumed candidates. Exact convention (TDD finalizes only
cosmetic details, and must preserve these properties):

```text
candidate terminal write excluded by assumed feasibility
condition: {variable} <- {expression} [{file}:{line}]
```

Requirements: REPLACES ordinary wording (never appends to it);
assumption status unmissable; temporary internal reason text
never leaks verbatim (fixed generic vocabulary only);
survives serialization/rendering as a plain string field;
never golden-pinned as exact prose (normalization forbids it).

Validator interplay (verified against
`report_validation.py:6-70`): presence checks
(`source_refs`, `numeric_checks`, signature) may be satisfied
by candidate refs — acceptable because candidate status is
explicit, confidence derives from replay/coverage paths rather
than ref counts, and `enforce_validation_downgrades` still
demotes unsupported high/medium claims. Candidates must never
by themselves upgrade confidence, confirm a hypothesis, or
imply replay/applicability proof.

## 11. Total ordering (Workstream A: no helper tier)

1. runtime computation writers;
2. declaration/storage anchors.
Within a role: derived/proven surviving writer before
assumed-pruned candidate. Deterministic tiebreak: source
order. An assumed runtime computation may therefore precede a
proven declaration anchor — honest ONLY because candidate
marking (§10) is unmistakable.

## 12. Declaration role without declaration_kind

Vertices carry no `declaration_kind` (bindings-only metadata),
so classify via binding/source-site join: resolve the
operation's `source_site_id` / `source_order` back to its
binding and read declaration role there. Fallback where join
is unavailable: a header-file initializer-shaped write with no
runtime writer for the same symbol is a supporting anchor, not
primary causal evidence. Never classify by `header file ==
declaration` alone. Semantic rule: runtime writer exists for
the selected symbol → declaration initializer is supporting;
no runtime writer and the initializer itself establishes value
→ it may remain primary.

## 13. Assumed-writer relevance rule

An assumed-pruned writer is eligible iff ALL hold: same
selected terminal symbol; source-parsed writer; terminal-marked
before pruning; removed solely because of ASSUMED
feasibility; belongs to the selected mechanism/result via
actual graph/result relations. Nothing unrelated is emitted;
no dumping of every retained writer.

## 14. Budget invariant

`[:8]` unchanged. Invariant: where eligible causal runtime
computation writer evidence exists, at least one such ref
survives selection — a declaration anchor must never consume
all slots ahead of it (§11 ordering enforces this).

## 15. Assumption diagnostics

Primary owner: annotated-DAG retained metadata (§5). The
`checkpoint_rounds` summaries may carry diagnostic summaries
only through existing fields; checkpoint semantics must never
depend on them. Do NOT add assumed-pruning traces to
`unresolved_evidence` by default (verified: it would pollute
user-facing gaps; confidence does not route on list contents,
but gap semantics would degrade).

## 16. Retained branch verdicts

Retention revives nothing: original verdicts stay
`always_false` + `assumed=True`. The `branches_verified` check
(`dag_pipeline.py:691-708`) therefore keeps treating a
judge-named assumed-pruned branch as feasibility-dead per
current semantics. No transform of retained provenance into
surviving graph branches, ever.

## 17. Determinism and fingerprints

The retained list is built deterministically (stable §7 dedup
and §11 order). Construction-DAG fingerprints, cache keys, and
source+ulog+terminal identity are untouched (retention lives
only on the annotated DAG). The additive field follows the
`exclude=True` precedent so model-serialization fingerprints
do not see it; TDD must confirm no pinned round-trip test
regresses (existing `model_dump` equality tests compare
same-code paths, which gain the field symmetrically).

## 18. Existing-test compatibility

Records are DAG-level, never per-vertex (a vertex-level exact
dict pin at `test_mechanism_discovery.py:770-778` forbids
per-vertex keys). No weakening of branch/vertex-count,
prune-gone, or round-trip tests; additive expectations only
where the new behavior is asserted.

## 19. RTL defect closure

Post-fix report carries deduped retained `rtl.cpp:245/248`
candidate refs plus the existing `rtl.h:163` anchor, ordered
per §11. Numeric benchmark (`destination + 2 × acceptance
radius ≈ 103.93812`), diagnostic strength (PROVEN), and A1–A6
unchanged. When real log-derived freshness later flips the
path to DERIVED/PROVEN, the same computation appears as
ordinary surviving evidence with no marking and no contract
change. Helper `696–731` refs are explicitly NOT required
here.

## 20. Workstream A TDD (no helper tests)

- T1: generic runtime writer under ASSUMED_FALSE is removed
  from the execution graph but retained as candidate
  provenance (record fields per §6, DAG-level).
- T2: generic writer under PROVEN_FALSE is pruned and does
  NOT become report causal evidence.
- T3: runtime terminal writer + declaration initializer →
  runtime computation ordered ahead; candidate marking
  explicit while assumed.
- T4: duplicate graph instances of one stable writer dedup
  without vertex IDs; distinct assignments stay distinct.
- T5: with eligible causal runtime evidence, ≥1 causal
  runtime ref survives `[:8]`.
- T6: RTL acceptance strengthens declaration-only → retained
  computation candidates + anchor; numerics untouched.
- T7: replay, checkpoint, coverage, applicability, T6B,
  Gate-B, R1, P0–P4 suites unchanged.
- Negatives: assumed never presented as proven executed
  (wording); proven-dead never promoted; declaration never
  crowds out runtime evidence; budget stable; no vertex-ID
  ordering/dedup.

## 21. Workstream A slices

- D1: triple-read formalization + unit coverage (no
  proof-authority contact).
- D2: retained-record capture in pruning path.
- D3: report selection/dedup/ordering/marking (terminal
  writers + anchors only).
- D4: RTL acceptance strengthening.
- D5: focused + subsystem + full regression.

## 22. Stop gates (assessed, none triggered)

A: no schema change needed (additive `exclude=True` field +
explanation convention). B: candidate marking representable in
existing `CodeRef.explanation` (plain string, rendered and
serialized; TDD verifies end-to-end). C: identity tuple needs
no vertex IDs (§7, with None-handling). D: replay enumerates
vertices, coverage reads proof stores, checkpoint path is
`prune_dead=False` + `allow_assumptions=False` — metadata
cannot leak into proof semantics. E: 245/248 are terminal
writers; defect closes without helper provenance.
