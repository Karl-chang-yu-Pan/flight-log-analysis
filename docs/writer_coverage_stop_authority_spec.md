# Positive writer coverage and stop authority

Finalized specification (rev. 2) for the writer-coverage workstream. This
document materializes already-agreed decisions; it invents no new semantics.
It constrains T3–T6B and records T1–T2 as completed gates.

Authority: `docs/adr/0002-coverage-stop-authority.md` (decision),
`docs/adr/0001-evidence-identity-ownership.md` (identity/promotion
boundaries), `CONTEXT.md` (domain vocabulary, Coverage Proof glossary).
Consistency evidence only: committed T1 structured evidence
(`SearchAttempt`/`CoverageEvidence`) and committed T2 search-universe
versioning (`CoverageSearchState`, `planned_stages`, record-suppression
cursor, version-stamped `universe_ref`).

## 1. Core invariants

These hold across the whole workstream and are not renegotiable per ticket:

1. Broad member/global searches are heuristic only. Owner-lineage,
   declaration/assignment-shape, bare-name, callable, and query-fallback
   search generate candidate locations; they never establish completeness.
2. Exact admission proves identity precision, NOT search completeness.
   Admitting one exact writer says nothing about writers not examined.
3. Coverage certificates may come only from provably closed search
   classes (see §2). A useful exact writer alone is never enough, and a
   zero-result heuristic search is never enough.
4. Applicability is positive and per use (see §8). Coverage lists
   writers; applicability qualifies them for one specific use.
5. Retirement must retain covered writer/provenance state (see §7). A
   retired search stays reconstructible; retirement is not deletion.
6. Search-universe version is discovery-session-owned (see §4). It lives
   in the discovery session, never on graphs, snapshots, caches, or globals.
7. Evidence recording on the coverage path is mandatory. A
   coverage-enabled search cannot run without producing its attempt
   records (`resolve_with_evidence` requires a sink).
8. Mutable proof/search state must not live on snapshot DAGs. DAGs store
   graph/domain facts only; `split_by_terminal` and copies never own,
   fork, or duplicate version/progress state.
9. Spelling fallback never establishes identity. Bare-name and
   query-shape hits are heuristic recall aids; only declaration/storage
   identity rules admit (per ADR 0001, including its explicitly allowed
   parser-proven re-spellings, which preserve semantic identity).
10. Exhaustion remains scheduling-only and proves nothing about coverage
    (per ADR 0001 §Decision and ADR 0002). An empty search releases
    scheduling priority only. Visited state, empty queues, and
    filtered-empty states likewise prove nothing.
11. A writer obligation is per declaration/storage entity; multiple
    consuming origins share one obligation through accumulated origins
    (per ADR 0002). Splitting or merging obligations by spelling,
    call site, or file is prohibited.
12. Fail-closed defaults stand (per ADR 0002): unknown writers, unknown
    applicability, and missing history/transfer evidence keep their
    obligations open. Nothing in this spec upgrades an unknown to a
    proof.

## 2. Search taxonomy

Search classes for certificate purposes. Only **EXACT-COMPLETE**
evidence may contribute to certificate derivation (§6).

### Certifiable only under explicit closure assumptions

#### Same-callable local enumeration

EXACT-COMPLETE only when:

- one loaded/parsed callable body is the complete lexical boundary;
- all relevant writer syntax inside that boundary is represented;
- assumptions/gaps such as macros, initializer forms, increment forms,
  aliases, and similar syntax coverage limits are explicit.

The current local short-circuit (`STRATEGY_LOCAL_SHORTCIRCUIT`) records
the boundary decision (`local-scope-rule`); the explicit assumption list
is a T3 certificate input, not current output.

#### Internal-linkage global

EXACT-COMPLETE only when:

- linkage proves one file is the complete boundary
  (current predicate: declaration linkage `internal`, or a
  `global:internal:` declaration identity);
- syntax coverage assumptions are explicit.

Current evidence: `STORAGE_INTERNAL_ONLY` with
`broad_search_skipped: internal-linkage`. The broad search is skipped
*because* linkage closes the boundary; the closure assumption must be
carried into any certificate, not assumed by it.

#### Explicitly enumerated closed file/declaration set

EXACT-COMPLETE only when every member of the declared closed set was
examined. The declared set, its closure justification, and the
per-member examination record are all certificate inputs. No current
strategy claims this class; a future strategy may, subject to the same
explicitness rule.

### Heuristic

- Owner-lineage + owner-qualified search.
- Declaration + assignment-shape search.
- Bare-name / callable / query fallback (including definition search and
  per-group query stages).

Heuristic evidence may generate candidate locations and may support
scheduling, but never contributes to certificate derivation.

### Unavailable

Search cannot legally or meaningfully run for the current
obligation/scope (unproven identity, unknown receiver type, unknown
owner, no issued query). Unavailable/failure outcomes must never become
absence: they record that nothing was examined, which is distinct from
examining a closed domain and finding no writer.

### Note on code-level completeness classes

Committed evidence code labels some strategies `complete`, meaning
*closed over the enumerated set as constructed* — explicitly not
tree-wide completeness. That label is an examined-domain description,
not a certificate class. Mapping a code-`complete` strategy to
EXACT-COMPLETE additionally requires the closure assumptions of this
section. In particular, owner-lineage enumeration is code-`complete`
over its constructed file set but remains **heuristic** for
certification, because the owner set itself is not a proven-closed
search boundary.

## 3. Evidence model

### `SearchAttempt`

One use of one search strategy against one obligation. Fields:

- **obligation key**: which writer obligation was searched
  (`source_reference_resolution_key`: semantic lookup identity,
  independent of runtime call sites);
- **scheduling key**: which scheduled request this attempt served
  (`visit_key`: call-site-scoped scheduling identity);
- **strategy**: the dispatch branch executed (e.g.
  `storage-owner-files`, `callable-query-group`, `storage-internal-only`,
  `unique-entity-filter`, `local-shortcircuit`);
- **dispatch**: callable dispatch kind where applicable
  (receiver / qualified / unqualified-member / free);
- **flags/config**: search configuration in force;
- **issued queries**: profiler queries as issued pre-normalization;
- **intended boundary**: the boundary the strategy aimed at
  (owner lineage, declaration, definition search, query);
- **examined domain**: files actually opened and admission-checked,
  plus backend; only the examined domain counts, never the intended one;
- **search-universe version**: the version the attempt ran under
  (stamped when the caller supplies it; absent otherwise);
- **outcome**: one admission verdict (below).

### `AdmissionVerdict`

The recorded outcome for one examined candidate or examined domain.
Current taxonomy: `admitted`, `rejected-declaration`, `rejected-owner`,
`non-writer`, `ambiguous`, `inapplicable`, `no-query-issued`,
`unavailable`, `no-candidate-complete-domain`,
`no-candidate-partial-domain`. Per-file verdicts (`exact:<site>`,
`index-miss`, `multi-match:<ids>`, incompatibility clauses) are part of
the verdict, not commentary. Rules:

- Ambiguity must preserve the colliding identities
  (`colliding_identities`); an ambiguous outcome never collapses to
  empty.
- Unavailable/failure must not become absence.
- An empty examined set must fall through to the domain-class rule
  (complete vs partial vs unavailable); it must never become a
  rejection class (`non-writer`, `rejected-*`) or `ambiguous`.
- Heuristic discovery may generate candidate locations but does not
  establish identity; exact admission remains controlled by
  declaration/storage identity rules (`same_declaration_entity`,
  `compatible`, owner lineage), unchanged by this workstream.

### `CoverageEvidence`

The attempt records for one resolution: obligation key, scheduling key,
ordered attempts, and `universe_ref`. Evidence is input to a coverage
decision, never the decision. It carries no authority.

## 4. Search-universe state

`CoverageSearchState` is discovery-session-owned scheduling/progress
state (current production owner: one instance per `discover_mechanism_dag`
session driving `was_visited` / `mark_visited` / `advance_version`).

Committed T2 semantics, preserved here as finalized:

- The version increments when newly loaded source files extend the
  current production search universe.
- This trigger is exhaustive for the current production flow, because
  structure/owner/boundary facts derive from loaded files while
  source hash, parser backend, profiler tree, seeds, and preranked
  inputs are fixed for one discovery session. No other current event
  changes resolver results independently.
- Stale version-N visited suppression cannot block version-N+1 search:
  advancing the version prunes visited/completed/exhausted scheduling
  state without discarding recorded evidence (evidence lives in sinks).
- Retry must settle when no further file is admitted: versions advance
  only on newly loaded files, the loaded set grows monotonically, and
  retried resolution over cached per-file facts is deterministic, so a
  fruitless retry produces an empty pending set and the loop exits.
- Same-version retry is currently unnecessary: one resolver invocation
  runs all applicable stages (ambiguity/empty never blocks a later
  stage; only successful admission short-circuits by pre-existing
  first-group-wins behavior), and no within-version event contributes
  new search information.

Do NOT generalize this into speculative invalidation machinery. Future
capability changes that introduce same-version new-information events
belong to a later ticket and spec change, not to silent extension here.

## 5. Stage semantics

Planned stages (`planned_stages`) describe real resolver strategy
stages for progress representation. The plan reuses the resolver's own
dispatch helpers (query-group derivation, query-shape derivation) and
must never drift from recordable stages; the internal-linkage predicate
is shared verbatim between resolver and planner for exactly this reason.
A plan may over-approximate (a stage that never records because an
earlier stage admitted first is still plannable); it must never omit a
stage the resolver can record, nor list a stage the resolver cannot
record on that path.

T2 clarification, binding on T3:

- `skip_stages` (and any equivalent current mechanism) suppresses
  duplicate **evidence recording only**. The underlying search still
  executes; candidates are identical with and without suppression.
- A suppressed record means "already recorded for this version". It does
  NOT mean the resolver stage was skipped, the expensive search did not
  rerun, or the work is complete.
- Absence of a repeated `SearchAttempt` record does NOT prove the
  underlying stage did not execute.

T3 MUST derive completeness from actual attempt semantics (recorded
strategies, examined domains, verdicts), never from
record-suppression/cursor assumptions.

## 6. Certificate model

A future `WriterCoverageCertificate` binds at least:

- exact obligation/resolution identity;
- storage identity;
- symbol/declaration identity;
- receiver/composite context where the boundary depends on it;
- exact search boundary (the declared closed set or closed lexical unit);
- examined domain (what was actually opened and checked);
- search-universe version (certificates are valid only for the index
  snapshot they derive from);
- certifying strategy assumptions (the explicit closure assumptions
  from §2).

Certificate derivation must be pure: a deterministic function of
obligation + evidence + declared boundary, producing no scheduling
effects and owning no scheduler state.

A certificate may be produced only if ALL hold:

- the required strategy is EXACT-COMPLETE under §2 with explicit
  assumptions;
- the evidence is from the current universe version;
- every member of the declared boundary was examined;
- no unavailable / unexamined / truncated / ambiguous / failure state
  remains on the certified path;
- relevant candidates carry explicit admission verdicts.

A useful exact writer alone is NOT enough (identity precision ≠
completeness, §1.2). A zero-result heuristic search is NOT enough
(§1.1, §1.3). Ambiguity, unavailability, and empty-examined outcomes
can never support derivation.

## 7. Covered/satisfied search state

Future scheduling transition, owned by the discovery session (not the
graph):

```text
OPEN → COVERED   (only after a valid §6 certificate exists)
```

Covered/satisfied state must retain everything needed to use and
revalidate the result without re-deriving identity from spelling:

- covered writer IDs;
- consuming origins;
- operands/context required for reconstruction;
- semantic and scheduling key namespaces;
- the certificate, its boundary, and its search-universe version.

Covered is distinct from exhausted: exhausted releases scheduling
priority with no proof; covered records proof with retained state.

Universe-version invalidation (new files admitted) must revoke, for
dependent obligations, certificate authority, covered scheduling
authority, derived applicability (§8), and stop authority — and must
make the obligation schedulable again **without reconstructing identity
from spelling** (identities and origins are retained; only authority is
revoked). Recorded evidence is never discarded by invalidation.

## 8. Applicability

Applicability is per `(use, covered-writer-set)`: whether the covered
writers could actually govern one specific use (one receiver, one call
scope, one ordering context). Coverage alone never proves it.

Require at least one positive basis, for example:

- one exactly ordered producer with no relevant competitor;
- exact gate/control coverage bound to the specific receiver/use;
- an already-proven transfer/execution fact.

And require absence of unresolved blockers, for example:

- control/state/construction uncertainty;
- `conditional_writer_ids` (current evaluator marker for
  invocation-dependent producers);
- history/transfer obligations.

Presence of `conditional_writer_ids` blocks applicability. Absence of
that marker does NOT prove applicability — it is a necessary
condition, never a sufficient one.

Cross-method same-declaration situations with unknown
invocation/order may therefore hold simultaneously:

- coverage = true,
- applicability = false,
- stop = false.

Static writer enumeration cannot answer control, invocation, receiver,
or order questions; merging those questions into coverage is prohibited
(per ADR 0002 considered-options).

## 9. Checkpoint relevance and non-vacuity

Relevant obligations must be determined **before** scheduling/filtering
effects (reachability pruning, priority exhaustion, queue filtering)
remove them. Filtered or exhausted obligations remain
relevant-but-uncovered unless actual proof exists; scheduling state
must never shrink the relevance set.

Do not allow `all([])`, an empty pending queue, or an empty filtered
queue to become proof of anything. Vacuous success is prohibited.

A genuinely empty relevant set may be accepted only when the checkpoint
is independently non-degenerate (terminal validated, graph materialized
and feasible, replay complete) and all other existing
matched/replay/requirement conditions hold.

## 10. Stop authority

`evaluate_checkpoint_round` remains the sole stop-authority producer.
No other component — evaluator, replay, resolver, discovery scheduler,
or certificate deriver — may authorize stop.

Final authorization requires the existing verified conjunction PLUS,
where required:

- coverage for every relevant obligation (§6–§7, current version);
- positive applicability wherever applicability is required (§8);
- current-version proof throughout (no stale-version authority);
- non-vacuous relevant scope (§9).

Explicitly prohibited as proof, jointly or severally: exhaustion,
visited state, empty queue, filtered-empty state, absence of a marker,
one discovered writer, exact admission alone, heuristic empty result.
(Per ADR 0002: zero pending work alone is never authority; per ADR
0001: exhaustion never means coverage, local matches never authorize
stopping.)

Proof-aware outstanding-work rule (Gate-B repair): a raw
`source_requests` entry, and its paired `source_lookup` analysis
requirement, remain reported as structural diagnostics but cease to
count as outstanding work in the final authority conjunction when
their exact visit key belongs to the current accepted
`covered_obligation_keys`. Raw `source_requests`, raw requirements,
and the raw `legacy_verified` verdict keep their existing meanings;
only current accepted coverage proof discharges, never retirement,
exhaustion, visited state, or spelling. See
`docs/outstanding_source_work_authority_spec.md`.

Replay-gating rule (R1 repair): numeric replay is attempted unless a
replay-blocking requirement exists, and only a well-formed
`source_lookup` — kind exactly `source_lookup` with a
`source_reference` mapping payload validating as a source reference —
is non-blocking. Every other kind (including malformed or
payload-less `source_lookup`) blocks replay exactly as before, and
the diagnostic observer path keeps replay disabled. A `matched`
replay therefore reports numeric agreement of the known evaluable
cone only: replay agreement is never discovery completeness
(completeness belongs to T3 alone), and final safety still comes
exclusively from the §10 conjunction. See
`docs/replay_outstanding_requirements_spec.md`.

## 11. Ownership

- **Source expansion** records attempts and evidence
  (`SearchAttempt`, admission verdicts, `CoverageEvidence`). It decides
  nothing about coverage, applicability, retirement, or stop.
- **Discovery session** owns search-universe state: version, visited
  suppression, progress (`CoverageSearchState`). Scheduling effects
  only.
- **DAG** stores graph/domain facts only. No version, progress,
  certificate, or authority state on graphs or snapshots.
- **Certificate derivation** (T3) consumes evidence and declared
  boundaries and produces certificates. Pure function; owns no
  scheduling state and issues no scheduling effects.
- **Checkpoint round** alone owns stop authorization
  (`evaluate_checkpoint_round`), seeing obligations, coverage,
  applicability, replay, and requirements together.

No persisted/disk proof state is required for this workstream. Caches
(fact caches, admission indexes, profiler search caches) are keyed by
file/query content and are immutable within a discovery session; they
carry no version or authority.

## 12. Current implementation sequence

Finalized gates (linear; each gate unlocks the next, none implies the
next):

```text
T1 evidence
→ T2 version/retry
→ T3 certificate derivation
→ T4 record-only checkpoint threading
→ T5 applicability
→ T6A retire-as-satisfied
→ T6B stop activation
```

- **T1 and T2 are completed before this document is filed.**
  T1: structured attempt/verdict/evidence recording with unchanged
  legacy candidates. T2: session-owned version/visited state,
  invalidation on universe extension, version-stamped evidence,
  faithful stage plans, record-suppression-only cursor.
- **T3** derives certificates per §6. It must not change scheduling or
  stop authority. It must satisfy the §13 precondition first.
- **T4** threads proof state (certificates, covered sets, versions)
  through checkpoint records with **zero behavior change**: evidence
  flows further, decisions do not move.
- **T5** adds positive applicability per §8. Stop remains impossible:
  applicability without retirement/activation authorizes nothing.
- **T6A** may retire certified searches to covered/satisfied state per
  §7, retaining all reconstruction state. Stop remains disabled.
- **T6B is the sole ticket allowed to change discovery stop
  authority**, activating the §10 conjunction. No earlier ticket may
  grant, imply, or approximate stop permission.

## 13. Mandatory T3 precondition from T2 review

Recorded explicitly as a T3 consumption gate:

`CoverageSearchState.eligible()` currently contains a global-exhaustion
coupling — any non-empty `_exhausted` set suppresses retry of every
seen obligation with remaining stages, regardless of which obligation
the exhaustion belongs to. This is safe only because `_completed`,
`_exhausted`, and `eligible()` are **not consumed by production T2**
(tests only); production scheduling uses the version-scoped `visited`
set, which has correct per-key semantics.

Before T3 consumes this state, it MUST be corrected so
exhaustion/progress is keyed per relevant obligation rather than one
global exhausted condition suppressing unrelated obligations. T3 must
either fix it before use or avoid consuming that API until corrected.

This is NOT a T2 correctness blocker. It is a T3 consumption
precondition. (Minor companion notes, also non-blocking: the
`skip_stages` parameter name overclaims execution skipping — its
record-suppression-only semantics are explicit in docs, comments, and
tests; the `stage_id` docstring's "skip" verb should be aligned on next
touch.)

## 14. Deferred matching issue

Non-blocking backlog, recorded separately:

Field-level / member-path matching — e.g. a `Cfg::gain` field writer
satisfying a `cfg` struct obligation — is NOT part of T2 and was
deliberately deferred. T2 scheduling-correctness does not depend on it:
the cross-version retry guarantee is proven independently of new
matching semantics (the pinning test counts re-resolution attempts,
not new admissions).

T3 certificate logic must not silently assume this matching capability
exists. Coverage proof is defined only over source/admission classes
the resolver actually supports. If a future ticket introduces
member-path matching, it must extend §2's certifiable classes and this
section explicitly — never by reinterpreting existing evidence.

## 15. TDD / acceptance requirements

Adversarial acceptance cases retained from the finalized design. Each
must hold at the ticket that introduces the relevant authority, and
none may be weakened to make a ticket pass:

1. Incomplete index cannot certify (missing members/unexamined files
   block derivation even with an exact writer in hand).
2. Exact writer + unseen candidate cannot certify (one admission never
   closes the boundary).
3. Invalidation after retirement reopens work: admitting new source
   revokes certificate, covered, applicability, and stop authority for
   dependent obligations while retaining identities, origins, and
   recorded evidence.
4. The same declaration may share coverage across uses while
   applicability differs by use (§8 cross-method case).
5. A vacuous checkpoint cannot authorize stop (§9).
6. Version retry is proven independently of new matching semantics
   (committed T2 pinning test: resolve-count, not admission).
7. A heuristic empty result cannot certify.
8. Ambiguity cannot certify (colliding identities preserved, never
   collapsed).
9. Unavailable search cannot certify (unexamined ≠ absent).
10. Closed-empty may certify only with an explicitly closed boundary:
    an examined closed domain with no writers is a positive finding,
    distinct from every empty/failed outcome above.
11. The full positive case (coverage + applicability + current version
    + non-vacuity + replay + requirements) may authorize stop only at
    T6B, via `evaluate_checkpoint_round` alone.

## 16. Non-goals

Explicitly out of this workstream:

- Redesigning DAG identity/provenance (owned by ADR 0001 and its
  implementation track).
- Redesigning the evaluator worklist.
- Solving arbitrary receiver/invocation/history reasoning beyond the
  positive bases of §8.
- Implementing the §14 field-level matching backlog as part of
  coverage proof.
- Making heuristic repository search globally complete.
- Altering legacy/non-checkpoint execution paths beyond what
  scheduling requires (legacy `resolve()` candidates are frozen).
- Introducing module-, PX4-, or symbol-specific rules (per AGENTS.md:
  derive from source, schema, or log — never tabulate).

---

## Status and handoff

- T1 (evidence) and T2 (version/retry) are complete and committed;
  focused review verdict on the T2 follow-ups: APPROVE WITH MINOR
  COMMENTS, with D1 (§13) gated on T3 consumption.
- Next: T3 certificate derivation under §§6, 12, 13, 15 — pure
  derivation, no scheduling or stop-authority change.
- This document is the contract T3 builds against. Contradictions
  between this spec and repository findings must be reported per
  AGENTS.md, never silently resolved in either direction.
