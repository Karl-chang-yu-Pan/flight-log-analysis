# Implementation Spec: Identity / Provenance Consistency

Status: specification only. No code implemented. No tests written.

Architectural contract: ADR 0001 (`docs/adr/0001-evidence-identity-ownership.md`)
and the CONTEXT.md glossary are fixed constraints. This slice implements only the
smallest observation-validity portion of that contract, not the complete
evidence-promotion abstraction.

## 1. Problem statement

The DAG subsystem has no single enforced contract for what counts as usable
evidence or how identity survives transformation:

- Ordinary evaluation accepts a type-grounded leaf
  (`grounded_via="declared_type"`) that checkpoint and replay reject, so one
  graph yields different verdicts depending on entry point.
- Caller scope is dropped on one writer-discovery path, producing redundant
  degraded parameter requests.
- Aggregate/member projection drops receiver/consuming identity, so the
  downstream frontier request names the wrong scope and observation linkage
  cannot associate it.
- Equivalent publication-proof logic is duplicated in several places (recorded
  as a deferred follow-up, not repaired here).

## 2. Current behavior

**Type-only leaf.** `_member_observation_leaf` (`mechanism_dag.py:935-973`)
emits a `logged_signal` leaf with `grounded_via="declared_type"`, no file/line,
gated only on `member_type.endswith("_s")` plus log placement.
`_resolve_symbol_producers` (`:4579-4586`) returns it in place of zero or
multiple producers. Ordinary `_evaluate_evidence` (`dag_value.py:444-450`)
rejects it only when `context is not None`; context-free binds fall through to
the sample path (`:451-466`) and report `value`. Checkpoint
(`dag_checkpoint.py:277-286`) and replay (`dag_replay.py:242-251`) always
reject it.

**Scope loss.** `_writers_from_relevant_calls` (`mechanism_dag.py:1975-2054`)
holds full scope (`scope_file/scope_callable/scope_line/scope_order`,
`:1988-1990`) but calls `_match_parameter(f"{receiver}.{name}")` with no scope
(`:2029-2031`). `_match_parameter` (`:5629-5664`) resolves ownership via
`_reference_identity(root, file, scope_function, line)` plus lineage
(`:5647-5658`) and returns `None` under authoritative declarations without an
owner (`:5659-5660`). The scoped call in `_resolve_symbol_producers`
(`:4566-4568`) shows the correct pattern.

**Projection loss.** `_project_identity` (`:3020-3029`) preserves all fields
and rewrites only `symbol` — correct. The loss is downstream: the composed
receiver identity (built by `_identity_in_call_scope`, `:2175-2242`, as
`receiver.member` with `recvDecl::subobject::decl`) passes through projection,
but `_resolve_symbol_producers` emits the frontier keyed on `identity.root`
and `_record_unresolved` (`:3457-3520`) re-derives identity via
`_reference_identity(symbol, file, scope_function, line)` (`:3476-3479`) in the
callee scope instead of carrying the proven composite.

**Linkage.** `require()` (`dag_observation.py:119-134`) matches frontier refs
by `vertex_id in origin_vertex_ids` plus operand/symbol intersection (or
call-site match for `callable`). With the degraded request above, the match
misses: the ref exists in the raw frontier but is absent from local
`input_requirements`. On current evidence this is a consequence of identity
loss, not an independent search gap.

## 3. Architectural constraints from ADR 0001

Builder emits candidates and scoped identities, never final evidence.
Evaluation owns promotion; checkpoint owns gating/completion, not evidence
meaning. Type-only grounding is candidate-only in every consumer. Proven
identity is preserved semantically through explicitly allowed parser-proven
re-spellings (suffix/member projection, receiver composite, call-instance
scope patch, storage-key use-site erasure). Fallback is a constrained,
revalidated lifecycle stage; `declaration_proven == False` alone permits
nothing. Exhaustion is scheduling only. The single future producer of positive
coverage/stop facts is checkpoint discovery. Forwarding stays narrow.
Legacy paths stay frozen.

## 4. Scope

1. Shared observation-validity clause eliminating the `declared_type`
   divergence (Step A).
2. Caller-scope threading through `_writers_from_relevant_calls` into
   `_match_parameter` (Step B).
3. Proven-composite population of the existing frontier identity field through
   projection (Step C).
4. `require()` linkage verified as a consequence; code change only on the
   four-part gate in section 11 (Step D).
5. Named follow-up recording publication-proof consolidation (section 15).

## 5. Non-goals

Evaluator recursion/worklist conversion; positive coverage implementation;
receiver-history reconstruction; persistent-state redesign; cache redesign;
judge/report schema changes; analysis-history UI/API work; generalized
multi-hop forwarding; legacy path retirement; large-scale `mechanism_dag.py`
decomposition; unrelated correctness-plan items; flight-result verification
claims; publication-proof consolidation; any evidence-kind dispatch framework.
Legacy semantic paths remain frozen unless a minimal compatibility change is
required; any such change must be called out explicitly in the implementation.

## 6. Proposed design

**Step A — observation-validity clause.** Introduce one shared pure rule owned
by evaluation:

observation candidate + observation metadata + forbidden-signal context
→ usable observation OR `observation_binding` issue.

Wire it into ordinary `_evaluate_evidence` unconditionally (removing the
`context is not None` gate around this clause only). Checkpoint, replay, and
conditional evaluation may reuse this clause while keeping every existing
completion/policy rule; the required behavior is agreement (their existing
inline checks may remain), and publication/root-proof consolidation stays
deferred. See section 7 for the narrowed interface.

**Step B — scope threading.** Pass `scope_file`, raw `scope_callable`, and
`scope_line` as supplied by the caller at the `:2029-2031` call. See
section 8.

**Step C — identity carry-through.** Populate the existing
`UnresolvedSourceReference.identity` field with the proven composite identity
when recording `storage_writers`/`member_writers` for a proven identity.
Retain collapsed storage dedup; accumulate consumer origins. See section 8.

**Step D — require() verification gate.** Test corrected identity through the
existing matcher. No code change if it associates; a minimal scoped-identity
repair only if all four gate conditions in section 11 are demonstrated.

## 7. Observation-validity clause (narrowed Step A interface)

- **Responsibility**: answer one question only — "is this observation candidate
  usable as an observation."
- **Inputs** (exact): evidence sub-kind; signal name; observation metadata
  (`observation` status, `grounded_via`); forbidden-signal set from context
  when a context exists.
- **Explicit non-inputs**: samples, parameter values, enum values, replay
  domains, policy objects, completion state, checkpoint requirements,
  discovery state. The clause must not accept or access them.
- **Output**: usable observation, or issue `("observation_binding", vertex_id,
  reason)`.
- **Kind coverage**: `logged_signal` only. Parameter, constant,
  opaque-symbol, helper-parameter, and all other evidence-kind behavior stays
  inline where it currently is and remains unchanged.
- **No dispatch framework**: no kind router, registry, or base class. ADR 0001
  defines the long-term ownership contract; this slice implements only the
  smallest shared observation-validity portion required by the defect.
- **Callers**: ordinary `_evaluate_evidence` (clause applied unconditionally,
  then existing sample/parameter paths), checkpoint leaf check, replay
  pre-gate, conditional context path — each retaining all other rules.

## 8. Identity propagation rules

- Proven `declaration_id + kind` is never replaced by bare-spelling lookup.
- Allowed re-spellings only: `_project_identity` symbol extension, receiver
  composite transformation, call-instance scope patching, `storage_key()`
  use-site erasure. Each preserves semantic identity while spelling changes.
- **Step B**: pass `scope_file`, raw `scope_callable`, `scope_line` as
  supplied. Do not pre-normalize `scope_callable`: `_reference_identity`
  (`:3379-3384`) already attempts raw and `_base_callable_scope`-stripped
  forms, matching the existing scoped call at `:4566-4568`. Line is use-site
  resolution input, never storage identity. Passing caller scope narrows
  resolution rather than broadening it: the current unscoped path can fall
  back to broad symbol lookup and fails closed under authoritative
  declarations, while the scoped path adds ownership/lineage context.
  Ownership and lineage filtering (`:5650-5658`), the unique-name rule
  (`:5662-5664`), and later-call/order filtering (`:2010-2024`) remain
  unchanged. No unrelated same-name parameter may become visible.
- **Step C carrier**: `UnresolvedSourceReference.identity` already exists.
  This step populates that field with the proven composite (receiver,
  receiver type, declaration, projected symbol, storage relationship). It is
  not a new dataclass field, schema change, or serialization redesign.
- **Dedup stays collapsed**: proven `storage_writers` keep the
  `(kind, kind, declaration_id)` key (`:3506-3515`). Rationale: multiple
  consumers of the same storage obligation share one frontier item where the
  model considers the obligation identical. Per-consumer provenance is
  preserved by accumulated `origin_vertex_ids`, `origin_operands`, and
  existing origin metadata (`:3519-3539`). The implementation must not emit
  one discovery request per consumer merely to preserve receiver identity.
  Shared storage obligation → one deduplicated frontier item; multiple
  consuming contexts → accumulated origins on that item. `visit_key()`
  continues carrying full reference identity where applicable. Nested
  projection re-applies `_project_identity` to the already-composite
  identity. Fallback and proven identity coexist only under ADR revalidation
  rules.

## 9. Affected modules/functions

- `analysis/dag_value.py::_evaluate_evidence` (`:430-487`) — apply the Step A
  clause unconditionally; `:373-402` resolver/pending/persisted behavior
  unchanged.
- `analysis/dag_checkpoint.py::assess_checkpoint` leaf clause (`:271-286`) —
  delegate observation validity only; all policy requirements unchanged.
- `analysis/dag_replay.py::replay_dag_roots` pre-gate (`:242-251`) —
  delegate observation validity only; domain/completeness policy unchanged.
- `analysis/mechanism_dag.py::_writers_from_relevant_calls` (`:1975-2054`)
  and `_match_parameter` (`:5629-5664`) — Step B scope threading.
- `analysis/mechanism_dag.py::_resolve_symbol_producers` (`:4522-4590`),
  `_record_unresolved` (`:3457-3521`), `_project_identity` /
  `_project_binding_to_reference` (`:3020-3153`), `_identity_in_call_scope`
  (`:2175-2242`) — Step C carry-through.
- `analysis/dag_observation.py::require` (`:119-134`) — read-only unless the
  section 11 gate is met. Forwarding gate (`:100-112`), correspondence
  (`:20-57`) frozen.
- `analysis/checkpoint_discovery.py` (`:124-186`) — untouched; consumes
  improved requests automatically.

## 10. Detailed behavioral requirements

1. Type-only match never becomes usable evidence by itself, in any entry.
2. Observation-validity verdicts identical across ordinary/checkpoint/replay/
   local wrappers; completion-policy differences remain legitimate.
3. Proven declaration/storage identity never replaced by unvalidated
   bare-name lookup.
4. Parser-proven re-spelling preserves semantic identity.
5. Caller scope survives relevant-call parameter writer discovery.
6. Member projection preserves storage and receiver/consumer identity,
   including nesting.
7. Same symbol names in unrelated scopes/classes never interchangeable.
8. Empty discovery/exhaustion never becomes evidence or coverage.
9. Local conditional agreement carries `authorizes_discovery_stop=False`
   with scoped `matched`; never terminal stop authority.
10. Existing forwarding restrictions unchanged byte-for-byte.

## 11. Failure semantics and the Step D gate

Missing writer → `opaque_symbol` plus exact scoped `storage_writers`
request; checkpoint `source_lookup`; replay `partial`/`unevaluable` — never a
substituted leaf. Exhausted search → scheduling release with `unresolved`
stop. Ambiguous fallback → non-authoritative, no producer edge. Unknown guard
with conditional match → `conditional_writer_ids` tagged, scope-limited, no
stop. Complete mismatch → `mismatched` with existing downgrade behavior.

`dag_observation.require()` remains unchanged by default. A change is allowed
only if a minimal fixture proves all four: (1) the corrected exact scoped
frontier reference exists; (2) that same reference is absent from the
expected `input_requirements[].source_requests`; (3) the failure is inside
the existing matcher (vertex/origin, operand, or legitimate call-site
association); (4) the repair uses scoped/declaration/storage identity with no
spelling broadening. Flight probes, counts, incomplete results, or general
TECS failure never satisfy this gate. If Steps B/C repair linkage under the
existing matcher, record Step D explicitly as "no code change required."

## 12. Compatibility constraints

Legacy backend strict expected-failure parity preserved; no weaker
replacement acceptance. Report/judge schemas and local-check payload
constants untouched. Judge-I/O diff limited to corrected vertices/edges. Any
required legacy compatibility shim must be additive and called out in the
implementation.

## 13. Test plan

- **Validity agreement**: same graph with equivalent bindings across
  ordinary/checkpoint/replay/local; assert leaf-level usable vs
  `observation_binding`; explicitly cover the context-free ordinary path. No
  reliance on final replay/checkpoint status alone.
- **RTL scope**: (i) expected parameter leaf with correct identity/name;
  (ii) no degraded `(callable, get)` for that accessor; (iii) unknown-parameter
  control still yields an exact scoped request. Counts alone forbidden.
- **TECS/projection**: assert receiver, declaration identity, consuming/origin
  vertex, and projected origin operand on the frontier item; assert the same
  scoped ref in `input_requirements[].source_requests`; degraded duplicates
  rejected.
- **Dedup**: multi-consumer same-storage case → one obligation where
  appropriate with all consumer origins retained (detects over-collapsing and
  over-specific duplication).
- **Init/spelling**: braced vs parenthesized equivalence; parser-proven suffix
  re-spelling; call-instance-local separation (extend existing tests).
- **Failure**: missing writer unresolved with exact scoped request; exhausted
  stays unresolved; local match asserts no stop authority with retained
  conditions; ambiguous fallback non-authoritative.
- **Regression**: rename invariance on new fixtures only; both backends where
  supported; judge-I/O diff against a recorded baseline, marked conditional
  if no baseline exists (never replaced by exit-code assertions).

Extend (do not replace): `test_observation_requires_positive_source_provenance`,
`test_type_only_observation_is_not_checkpoint_proof`,
`test_resolved_parameter_does_not_discharge_mutable_writer_coverage`,
`test_reference_initialization_projects_nested_call_result`,
`test_boundary_call_in_predicate_is_not_a_helper_gap`,
`test_local_equation_discovers_its_helper_before_unknown_guard`.

## 14. Implementation sequence

### A — Observation-validity clause

Pure shared observation-validity rule; remove context-free declared-type
acceptance; agreement tests.

### B — Caller scope threading

`_writers_from_relevant_calls` into `_match_parameter`; scoped RTL fixtures.

### C — Proven identity carry-through

Populate existing `UnresolvedSourceReference.identity`; preserve
storage/receiver/projection semantics; retain collapsed storage dedup;
accumulate consumer origins; projection/dedup fixtures.

### D — require() verification gate

Test corrected identity through existing matching; no code change if it
associates; minimal scoped-identity repair only if all four gate conditions
are demonstrated.

Publication-proof dedup is not part of this sequence. Evaluator
recursion/worklist remains separate. Positive coverage remains separate.

## 15. Deferred follow-up

**Follow-up: consolidate duplicated publication-proof validity logic.**
Deferred because root-publication checks require graph/edge/file/line context
(combining them with the pure observation-validity clause would broaden that
interface incorrectly), root gating affects stop/completion behavior deserving
its own regression surface, and this slice stays focused on demonstrated
correctness defects. Purpose recorded only; not designed here.

## 16. Risks / open questions

A misused clause for non-observation kinds is guarded by the explicit
non-input list and the no-dispatch rule. Nested composites beyond two levels
are thinly fixtured; a small second carry-through point is acceptable
in-slice, a larger one becomes a named follow-up.
