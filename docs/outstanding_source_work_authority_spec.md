# Outstanding source work and proof-aware stop authority

Implementation-ready specification for the P4 Gate-B liveness repair:
a covered writer obligation must stop counting as outstanding legacy
source work, while remaining structurally unresolved and T6B-relevant.
Changes the final authority conjunction only. Freezes every diagnostic,
every key namespace, and the raw legacy verdict byte-for-byte.

Authority: `docs/adr/0002-coverage-stop-authority.md` (decision, unchanged),
`docs/adr/0001-evidence-identity-ownership.md`,
`docs/writer_coverage_stop_authority_spec.md` rev. 2 (T6B conjunction),
`docs/writer_proof_pipeline_spec.md` (P0–P3 pipeline, Gate-B finding),
`CONTEXT.md` (Coverage Proof glossary).

## 1. Problem

An honest production round can reach `coverage_ok`, `applicability_ok`
and `non_vacuous_ok` all true with empty uncovered/missing/conflict
sets, yet `authorizes_discovery_stop` stays false because
`legacy_verified` is false: the proven obligation itself remains in
the selected checkpoint's `source_requests`. Relevance and
source-requests derive from the same origin sets, and producer arrival
(which alone clears source requests) simultaneously destroys the
relevance a non-vacuous positive stop requires. Retired, exhausted and
visited obligations never leave `source_requests` either. No
non-vacuous positive stop is therefore reachable, although every proof
layer works honestly end to end.

## 2. Semantic rule

A raw `source_requests` entry, and its paired `source_lookup`
analysis requirement, remain present as structural diagnostics, but
they cease to count as **outstanding work** in the final authority
conjunction when their exact visit key belongs to the current proof
observation's `covered_obligation_keys`.

```text
raw source_requests            → unchanged (structural diagnostic)
raw analysis_requirements      → unchanged (structural diagnostic)
raw legacy_verified            → unchanged (pure legacy verdict)
outstanding source requests    → raw minus proof-covered visits
outstanding requirements       → raw minus proof-covered source_lookup
verified                       → proof-adjusted legacy ok AND proof conjunction
```

Discharge can only remove a veto the proof conjunction independently
demands be satisfied: a discounted visit is always a member of the
current covered set, which the unchanged coverage conjunct requires
for every relevant obligation. Discharge therefore never establishes
coverage; current coverage proof discharges legacy source-search work
for the same visit.

## 3. Ownership

All discharge computation lives in `evaluate_checkpoint_round`, the
sole stop-authority owner, after current proof matching is already
available. `assess_checkpoint` stays proof-blind: no snapshot,
certificate set, or version enters it. No other layer changes.

## 4. Key namespace

Discharge keys are **visit/scheduling keys**: tuples exactly as
produced by `UnresolvedSourceReference.visit_key()`, compared by tuple
equality. `covered_obligation_keys` is that same namespace
(established T4 behavior: certificates associate by `scheduling_key`
equality under the current proof version). No symbol spelling, no
declaration name, no file heuristic participates.

- Raw `source_requests` entry → visit via `model_validate(raw)`
  `.visit_key()`, the same parsing the round already performs for
  relevance. Unparseable entry → retained as outstanding.
- `source_lookup` requirement → visit via its embedded
  `source_reference` payload through the same parsing. Missing payload
  or parse failure → retained. Only `kind == "source_lookup"`
  requirements participate; `source_linkage` (same payload shape) and
  every other kind are never discounted.

## 5. Proof basis

The only discharge basis is membership in the current round's
`covered_obligation_keys`. Never retirement, exhaustion, visited
state, queue absence, resolution success, spelling, or symbol name.
`proof_version is None` yields an empty covered set through existing
observation behavior, hence zero discharge. Stale or foreign
certificates never enter the covered set, hence never discharge.

## 6. Final conjunction (field-exact)

With `covered = set(observation.covered_obligation_keys)`:

```text
outstanding_source_requests =
    [raw for raw in selected["source_requests"]
     if visit_key_or_unparseable(raw) not in covered]
outstanding_requirements =
    [r for r in selected["analysis_requirements"]
     if not (r.kind == "source_lookup"
             and visit_key_of(r) in covered)]
proof_adjusted_legacy_ok = bool(
    selected and selected["observed"]
    and selected["status"] == "matched"
    and selected["complete"]
    and not outstanding_requirements
    and not outstanding_source_requests
    and not any mismatched intermediates)
verified = proof_adjusted_legacy_ok
         and proof_authority.authorizes_stop
```

`legacy_verified` keeps its existing byte-identical definition and
continues to feed `ProofAuthority` unchanged. `authorizes_stop` keeps
its existing meaning (raw legacy AND proof conditions) as a
diagnostic: because it embeds the raw legacy verdict, reusing it in
the final gate would nullify the discharge, so the final gate
conjoins the proof-adjusted legacy gate with the proof flags
directly:

```text
verified = proof_adjusted_legacy_ok
         and proof_authority.coverage_ok
         and proof_authority.applicability_ok
         and proof_authority.non_vacuous_ok
```

Final stop is governed by `verified` only. Unparseable entries are retained by
construction (mirror the existing try/except-continue precedent).

## 7. Relevance preservation

Discharge never removes anything from T6B proof relevance. The
structural unresolved reference remains; relevance, coverage matching,
and applicability matching run before and independently of the
discount. Relevance must not be re-derived from outstanding work
after discounting, or the proof requirement would vanish with it.

## 8. Multi-checkpoint and selection

Discharge operates on the selected checkpoint exactly as the existing
final conjunction does. No new selection policy. When `selected` is
`None`, legacy is false as today and discharge is moot. Exact visit
identity only: a covered visit never discharges another checkpoint's
request. Same-spelling declarations never cross-discharge (visit keys
embed declaration identity).

## 9. Diagnostics

- `source_requests` field: unchanged contents.
- `analysis_requirements` field: unchanged contents.
- `legacy_verified` (record + `ProofAuthority` field): unchanged meaning.
- New: `ProofAuthority.discharged_source_request_keys: tuple = ()`,
  the discounted visit keys in deterministic order, for triage
  legibility. Defaulted, immutable tuple style like its sibling
  fields, kept off stable external summaries exactly as the existing
  authority fields are. No other schema change.

Glossary (for `CONTEXT.md`): a **source request** is a structural
checkpoint diagnostic indicating an unresolved source relation;
**outstanding source work** is a source request not yet satisfied by
current accepted coverage evidence.

## 10. Critical invariants (preserved, with tests)

1. Relevant request, no current coverage → outstanding → stop false.
2. Coverage accepted, source work discharged, applicability missing →
   applicability veto → stop false (discharge and applicability are
   independent dimensions).
3. Stale certificate/proof → visit absent from current covered set →
   outstanding → stop false.
4. `is_proof_retired` true with no matching current certificate →
   no discharge → stop false. Retirement stays scheduling-only.
5. Exhausted without certificate → no discharge → stop false.
6. Empty relevance → existing non-vacuity rule unchanged; discharge
   of nothing authorizes nothing.
7. Unparseable/missing-key request or requirement → outstanding.
8. `proof_version is None` → covered set empty → zero discharge.
9. Multi-checkpoint visits never cross-discharge.
10. Same-spelling declarations never cross-discharge.

## 11. Secondary blocker: unobserved terminal

The current P4 tripwire fixture additionally reports an unobserved
terminal. That is fixture dirtiness, not architecture: observed and
matched publication terminals are production-routine (publication
validation succeeds with logged signals; replay matching is routine
with aligned samples). No semantic change is specified for it. The
implementation ticket resolves it with an honest generic fixture (a
logged publication terminal plus aligned synthetic samples), never
with flag forcing.

## 12. P4 tripwire lifecycle

`test_p4_honest_production_positive_stop` remains `xfail(strict=True)`
until the implementation lands. The implementation ticket removes the
marker only when the fixture genuinely passes with non-empty
relevance, non-empty applicability-use set, real P0 evidence, real T3,
real T5, real retirement, real snapshot, and both verification flags
true. Never weaken its assertions.

## 13. Implementation slices

- **G1 — authority-model diagnostic**: add defaulted
  `discharged_source_request_keys` to the proof-authority result. No
  semantics change.
- **G2 — discharge computation**: derive covered-visit set from the
  existing proof observation; map selected raw requests and paired
  `source_lookup` requirements; fail closed on unparseable entries.
- **G3 — conjunction integration**: use outstanding work in the final
  stop conjunction; keep raw `legacy_verified` byte-identical.
- **G4 — tripwire promotion**: apply Gate-B repair, resolve
  fixture-level terminal dirt honestly, promote the tripwire.
- **G5 — docs**: `CONTEXT.md` glossary, ADR 0002 clarification note
  (no decision change), writer-coverage spec §10 note, proof-pipeline
  spec Gate-B note.

## 14. TDD matrix

- A: covered visit discharged from both veto sets; raw fields intact.
- B: paired `source_lookup` discharged; other kinds untouched.
- C: uncovered request stays outstanding; stop vetoed.
- D: covered + missing applicability → stop false.
- E: stale proof → no discharge → stop false.
- F: retired without certificate → no discharge.
- G: exhausted without certificate → no discharge.
- H: malformed/unparseable → outstanding.
- I: `proof_version=None` → zero discharge.
- J: multi-checkpoint visit isolation.
- K: same-spelling isolation.
- L: raw diagnostics unchanged while authority proceeds.
- M: existing vacuity coverage cited/reused, not duplicated.
- N: honest P4 promotion, non-vacuous and production-derived.

## 15. Stop gates for the implementation ticket

- **STOP A**: covered-visit set untrustworthy → do not recompute
  coverage independently; report missing contract.
- **STOP B**: requirement→visit pairing needs spelling heuristics →
  stop; fail-closed default already holds.
- **STOP C**: cannot separate raw legacy from proof-aware
  cleanliness without redefining `legacy_verified` → revisit model.
- **STOP D**: P4 still blocked after discharge and the terminal is
  structurally unobservable → report second liveness gap.
- **STOP E**: any uncovered/retired/exhausted obligation stops
  vetoing → reject the change.

## 16. Non-goals

Per-output proof authority or per-output discharge; persisted
outstanding-work models; pipeline changes; resolver/builder/T3/T5/T6A
changes; public report format changes; heuristic-search completeness;
history/transfer reasoning.

## 17. Documentation impact

`CONTEXT.md`: glossary terms only. ADR 0001: untouched. ADR 0002:
clarification note only — "unresolved source work" in authority
discussion means unproven/outstanding work, not the absence of a
structural unresolved reference; retirement/exhaustion stay
non-authoritative (no decision change). Writer-coverage spec: §10
note. Proof-pipeline spec: Gate-B resolution note.
