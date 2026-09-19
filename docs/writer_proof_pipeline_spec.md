# Writer proof-production and proof-lifetime specification

Implementation-ready specification for the post-T6B proof-production /
proof-lifetime architecture (slices 0–3). T1–T6B ownership is frozen;
this spec completes production wiring without moving any boundary.

Authority: `docs/adr/0001-evidence-identity-ownership.md`,
`docs/adr/0002-coverage-stop-authority.md`,
`docs/writer_coverage_stop_authority_spec.md` (rev. 2),
`CONTEXT.md` (Coverage Proof glossary).

## 1. Purpose

Give production discovery a complete, fail-closed path from source
resolution to T6B authority: retain T1 evidence, preserve exact use
pairs, derive/store versioned T3/T5 proof incrementally, and thread an
immutable snapshot into checkpoint evaluation — without moving any
T1–T6B ownership boundary.

## 2. Non-goals

No per-output stop scope; no T3/T5 derivation inside checkpoint; no
retirement-as-authority; no queue-derived coverage; no
heuristic-search completeness; no receiver/history reasoning beyond
§8 bases; no report/schema changes; no PX4-specific rules.

## 3. Existing architecture/ownership

T1/T2 evidence+version → T3 certificates (pure, zero production
callers) → T4/F1 relevance (record-only) → T5 per-use proofs (pure,
zero production callers) → T6A retirement (never driven in production)
→ T6B sole authority (production callers omit proof → fail-closed
veto). Discovery loop order today: settle universe → advance version
→ build DAG → checkpoint → resolve (legacy path) → repeat.

## 4. Root causes

(a) Pairing lost at unresolved-reference merge
(`mechanism_dag.py` independent list union; operand-less terminal
append). Cartesian use invention is the symptom. (b) Proof never born
in production (legacy `resolve` on the scheduling path; helpers-only
evidence sink). Missing checkpoint args are the symptom.
(c) Unreachability is lifetime-first: with proofs derived/stored/
threaded, the F1-local path fits current global scope.

## 5. Exact-use provenance model

New field `origin_uses: list[tuple[str, str]]` on
`UnresolvedSourceReference` (Pydantic-coercible, JSON-safe as pair
arrays). Sorted, deduplicated, immutable-by-convention (`model_copy`
on merge). Conceptual model: obligation `O` owns uses
`{(A,x), (B,y)}` — never split into `O-A`/`O-B` obligations. Origins /
operands lists retained for compatibility. `visit_key`,
`source_reference_resolution_key`, and merge-dedup keys MUST NOT
include `origin_uses` (verified: neither key function reads origins
today). Operand-less origins (e.g. terminal-id append) contribute
**no** concrete use — they are not governable uses (T5 needs a role
edge for a specific operand); legacy linkage still sees them via
retained origins lists. Receiver/call-scope preserved: pairs carry no
scheduling context themselves; binding to visits happens at T5/T6B as
today.

## 6. Pairing merge semantics

Creation populates the single known pair (writer-obligation path
already passes one origin + one operand). Merge unions **pairs as
pairs**: `{(A,x)} ∪ {(B,y)} = {(A,x),(B,y)}` — never re-derived from
merged lists. Deterministic order (sorted), duplicates dropped.

## 7. Pairing migration/backfill

Consumers updated: **exactly two** — T5 membership checks and T6B
`collect_applicability_uses` (exact pairs primary). Legacy
set-consumers explicitly unchanged (`dag_checkpoint` origin/relevant
and opaque-linkage sites, `dag_observation` linkage filter,
discovery owner filter, builder reachability filter). Fallback
(bounded): refs with empty pairs but non-empty origins/operands →
cartesian, fail-closed. Backfill at DAG load/validation **only** when
exactly one pair is mechanically entailed (single origin + single
operand); ambiguous old shapes stay unpaired. Sunset: fallback
removed on next DAG-snapshot format version; until then it covers
only `model_validate`d old payloads.

## 8. Production evidence-source design

**Option A (chosen)**: production frontier scheduling calls
`resolve_with_evidence` with a session-owned sink. Rejected B (dual
resolve): doubles query cost and risks candidate divergence. A is
safe because T2 record-suppression semantics were built for exactly
this — suppressed stages still execute; candidates/admission
identical (T1-committed invariant).

## 9. Evidence ownership/lifetime

Owner: `discover_mechanism_dag` session (single writer). Sink:
session-local ordered list, entries `(round_index, evidence)`; key
`(search-universe version, scheduling key)`; latest-wins per key per
version. Visible to derivation after the round's resolution phase.
Same-version repeats: suppressed by `visited` (skip) and
`skip_stages` (record-only suppression) — no duplication/corruption.
Candidates, order, admission, and unavailable/partial/ambiguous
honesty unchanged (same stages execute). Version-stamped by resolver
(`universe_version`) as today.

## 10. Proof-store model

Session-scoped **derived-state cache** (certificates, applicability
proofs, optional evidence index). Not an authority: nothing
authorizes from it; checkpoint alone does.

## 11. Proof-store ownership decision

**Option B (chosen): separate session proof state** (e.g.
`CoverageProofStore` beside T3/T5 models in `coverage.py`).
`CoverageSearchState` rejected: it is scheduling suppression consumed
by the resolve loop; mixing semantic cache risks authority coupling.
Sync cost is trivial — both sides key off `search_state.version`,
cross-reads only.

## 12. Keys and identities

Evidence: `(version, scheduling key)`. Certificates:
`(version, scheduling key)` retrieval + `(version, obligation key)`
index; never spelling. Proofs: `(version, use_key, scheduling key)`.
All keys from existing vocabulary; no new namespace.

## 13. Incremental T3 derivation

Trigger: relevant obligation with new/changed evidence
(attempt-count/`universe_ref` fingerprint differs) and no current
cert. Same-version unchanged → reuse. Added same-version evidence →
re-derive. Refusal → diagnostic only, no entry. Replacement only via
newer evidence, same version.

## 14. Incremental T5 derivation

Trigger: current cert present + use-graph facts available for an
unproven exact use (pairs from §5). Same-version unchanged → reuse.
Refusal (conditional/control/competitor) → diagnostic, use stays
missing. Never re-runs T5 derivation logic inside checkpoint.

## 15. T6A retirement relationship

**Option A (chosen)**: retirement records stay in
`CoverageSearchState`; the loop calls
`apply_writer_coverage_retirement` after deriving a current cert
(this also wires T6A into production for the first time). Invariant:
certificate → may cause retirement; retirement ⇏ certificate truth;
T6B never reads retirement.

## 16. Version/invalidation semantics

`origin_uses`: structural provenance, **survives** versions.
Evidence/certificates/proofs/retirement: version-bound. Lookup
filters to current version (stale never authoritative). On advance:
prune non-current entries by default (diagnostic retention
explicit/opt-in). Stale applicability can never pair with fresh
coverage — authority already binds both to one `proof_version`.

## 17. Proof snapshot model

Immutable value: `(version, certificates: tuple,
applicability_proofs: tuple)`. Frozen inputs (T3/T5 models already
frozen dataclasses) — copy cost negligible. Checkpoint receives
snapshot, never the store; no mutation path.

## 18. Round ordering/snapshot timing (authoritative eight-step sequence)

1. load new files;
2. advance version iff universe extended;
3. rebuild relevant inputs/DAG;
4. incrementally derive proof from evidence already available and
   apply T6A retirement where current certificate permits;
5. create immutable current-version proof snapshot;
6. evaluate checkpoint using snapshot;
7. resolve remaining frontier through evidence-producing resolver;
8. next discovery iteration consumes the newly recorded evidence.

Critical dependency (preserved over superficial line order):

```text
version settles
→ proof derivation
→ snapshot
→ checkpoint
→ new resolution/evidence
→ next iteration
```

A snapshot must never be created just before an immediate version
advance that invalidates it.

## 19. Cross-round lifetime

Version unchanged: current certs, still-bound proofs, and retirement
survive; derivation skipped for unchanged inputs. Version advanced:
currentness flips by key filter; obligations re-derive under the new
version as evidence arrives.

## 20. T6B threading contract

Existing optional args are the contract
(`coverage_certificates`, `applicability_proofs`, `proof_version`);
omitted-proof fail-closed semantics unchanged. Slice 3 wires
production call sites (discovery-internal evaluators and pipeline
`control_round` session access — implementation chooses closure vs
param within this constraint).

## 21. Fail-closed intermediate migration states

After Slice 0 only: evidence exists, no store/threading → vetoes
stand. Slice 1 only: exact uses, no lifecycle → vetoes stand.
Slice 2 only: store without threading → production checkpoint still
omits proof → vetoes stand. Slice 3: stop true only under the
existing T6B conjunction. No slice weakens T6B.

## 22. Performance/resource constraints

No whole-repo derivation per round (trigger-gated §13–14);
same-version reuse; store bounded by session
obligations×uses, pruned on advance; no per-round full
re-resolution (`visited`/`skip_stages` stand).

## 23. Production-positive acceptance scenario

Internal-linkage obligation (only certifiable class today): round N
resolves with retained evidence → T3 cert + exact pairs → T5 proof →
stored/retired; round N+1 relevance persists, legacy blockers clear,
snapshot threaded → `writer_coverage_verified=True`,
`applicability_verified=True`, `authorizes_stop=True`. Scoped
liveness claim only: general heuristic obligations remain
fail-closed by design.

## 24. Test matrix

Evidence: retained from real `resolve_with_evidence`; candidates
byte-identical to legacy path; ambiguity/partial/unavailable verdicts
preserved; same-version repeat no-duplication. Pairing: merge
`{(A,x)}+{(B,y)}` exact; no invention; dedup; obligation identity
unchanged; T5/T6B consume pairs; unpaired legacy input fail-closed;
operand-less origins yield no use. Store: current-only authority;
cross-obligation/use isolation; advance invalidates; unchanged inputs
reuse (spy on derivation). Threading: immutability; post-settle
timing; stale snapshot vetoes; no store mutation from checkpoint;
omitted+non-empty still vetoes. End-to-end: honest positive;
minus-coverage veto; minus-applicability veto; extension invalidates;
pairing prevents fabricated-use veto.

## 25. Migration slices 0–3

As §18: Slice 0 evidence source → Slice 1 pairing → Slice 2
store+incremental derivation (+T6A wiring) → Slice 3
snapshot/threading. Order fixed by dependency (derivation needs
evidence; threading needs store).

## 26. Stop/reopen conditions

STOP on: (A) evidence retention changes resolver output; (B) pairing
preservation splits obligation identity; (C) derivation needs
checkpoint logic (reject — ownership); (D) invalidation not atomic
per version; (E) snapshot threaded yet no honest legacy-clean
positive round possible (reopen scope, never weaken T6B).

## 26b. P4 Gate-B resolution

The Gate-B deadlock (proven unresolved obligations keeping legacy
`source_requests` dirty despite complete T3/T5 proof) is repaired by
proof-aware outstanding-work discharge specified in
`docs/outstanding_source_work_authority_spec.md`: covered visits stop
counting as outstanding source work while raw diagnostics, relevance,
and the raw legacy verdict stay unchanged. The second structural gap
found during that repair — numeric replay gated on zero raw
requirements, so `matched`/`complete` could never hold while any
reference persisted — is repaired by the replay-gating rule specified
in `docs/replay_outstanding_requirements_spec.md`: only well-formed
`source_lookup` requirements are replay-non-blocking, with the
diagnostic observer path frozen replay-free. The remaining P4 path
is an honest publication-terminal fixture (source-proven publication
binding plus aligned samples) for the production loop; the P4
strict-xfail tripwire stays until that fixture yields a genuine
green.

## 27. Deferred/non-goals

Per-output authority (future seam iff §23-class evidence proves
global scope insufficient); §14 matching; heuristic completeness;
history/transfer engines.

## 28. Required documentation amendments

`CONTEXT.md`: add `origin_uses`/`concrete use` + proof-store/snapshot
glossary (terms only). ADR 0001/0002: no semantic change.
Writer-coverage spec: minor §7 (store as retained state), §8
(exact-pair provenance), §11 (store ownership), §12 (slices)
implementation-completion notes.

## 29. Handoff to ticketing

Slices 0–3 map 1:1 to ticket groups; §24 rows are per-ticket
acceptance; §26 are ticket exit gates.
