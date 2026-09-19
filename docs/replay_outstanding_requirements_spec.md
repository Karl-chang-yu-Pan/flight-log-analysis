# Replay gating on outstanding requirements

Implementation-ready specification for the replay-gating liveness
repair: numeric replay is currently attempted only when the raw
`analysis_requirements` set is empty, so `matched`/`complete` can
never hold while any unresolved reference persists — even a
proof-discharged one. This blocks every non-vacuous positive stop
independently of the Gate-B veto repair.

Authority: `docs/adr/0002-coverage-stop-authority.md` (decision,
unchanged), `docs/adr/0001-evidence-identity-ownership.md`,
`docs/writer_coverage_stop_authority_spec.md` (T6B conjunction),
`docs/writer_proof_pipeline_spec.md` (pipeline + Gate-B note),
`docs/outstanding_source_work_authority_spec.md` (Gate-B discharge).

## 1. Semantic rule

Replay eligibility is a question about **numeric evaluability**, not
about **search completeness**. A well-formed `source_lookup`
requirement records that a writer search is outstanding; it says
nothing about whether the current graph values are computable. All
other requirement kinds — and any malformed or payload-less
`source_lookup` — record missing numeric/equation inputs or unknown
states and remain replay-blocking.

```text
replay is attempted
iff
every analysis requirement is a well-formed source_lookup,
where well-formed means:
    requirement.kind == "source_lookup"
    AND requirement carries a source_reference mapping payload
    that validates as a source reference (same model_validate
    pattern used for relevance mapping; validation failure blocks)
```

A `source_lookup` without a usable payload, an unknown requirement
kind, or any non-mapping requirement entry blocks replay exactly as
today (fail closed; conservative by construction, so future kinds
are blocking by default).

## 2. Why this is sound

Replay's internal machinery decides only **numeric agreement over
the currently modeled/evaluable cone**: unevaluable expressions,
missing variables, unproven or circular input observations,
scope/window/sample defects, and inexact reachability yield
`not_attempted`, `partial`, or `unevaluable` — never `matched`.

Crucially, replay does NOT know source-search completeness:
it evaluates only the known graph and cannot see undiscovered
writers. A `matched` verdict therefore means numeric agreement on
the known cone — never "all possible writers have been discovered".
Search completeness belongs solely to current T3 coverage proof,
and final safety comes exclusively from composition:

```text
replay numeric agreement
AND current coverage
AND current applicability
AND other legacy conditions
```

A `matched` replay beside an uncovered obligation is honest
*as a replay diagnostic* and still vetoes final stop through the
independent T6B coverage conjunct, which this slice does not touch.
Mandatory adversarial test: source_lookup present, replay evaluable
and matched, NO current accepted coverage → final round authority
remains false.

- The evaluator never substitutes observations for writers
  (`replay_dag_roots` contract): sampled values satisfy boundary
  inputs only, never the writers under test.
- Direction of risk is fail-closed: the only new outcomes in dirty
  rounds are attempted evaluations (`partial`/`unevaluable`, or a
  `mismatched` veto on genuinely contradictory evidence — itself a
  correct fail-closed outcome).

## 3. Ownership and layering

- `assess_checkpoint` keeps full ownership of replay eligibility and
  evaluation. No certificate, proof, version, or store state enters
  it — the gate predicate reads requirement kinds plus payload
  presence only, so the proof-blind assess invariant is preserved
  and no new plumbing exists to go stale.
- `evaluate_checkpoint_round` is unchanged except for receiving the
  replay verdicts `assess_checkpoint` now produces in more rounds.
  No ordering change: the call sequence
  assess → relevance → observation → authority is untouched.
- No dependency cycle is possible: the gate needs kinds (available
  before replay), coverage matching needs replay-independent
  observation inputs, and authority consumes both downstream.

## 4. Requirement classification

| kind | blocks replay | reason |
|---|---|---|
| well-formed `source_lookup` (with `source_reference` mapping payload) | no | search completeness, not numeric evaluability; values compute from the current graph + observations |
| malformed/payload-less `source_lookup` | yes | unknown structure; fail closed |
| unknown requirement kind | yes | conservative by construction; future kinds block by default |
| non-mapping requirement entry | yes | cannot classify; fail closed |
| `source_linkage` | yes | unbound operand/vertex or missing graph origin — unevaluable by construction |
| `construction` | yes | unmaterialized vertices |
| `expression` | yes | uncompilable equation |
| `control_flow` | yes | unknown reachability/domain gating |
| `observation_binding` | yes | no source-proven publication to compare against |
| `observation_data` | yes | absent/insufficient samples |
| `sampling_policy` | yes | underivable comparison policy |
| `parameter_data` | yes | unavailable required value |
| `evaluation_scope` | yes | unresolved comparison domain |
| `state_alignment` | yes | missing history/transfer evidence |
| `numerical_replay` | n/a (output) | recorded when a prior attempt reports incomplete |

Only well-formed `source_lookup` changes status. Every other kind
blocks exactly as today, including all genuinely missing numeric
inputs.

## 5. Exact change

In `assess_checkpoint`, the replay gate changes from emptiness of the
raw requirement set to absence of replay-blocking requirements per
§1/§4 (only well-formed `source_lookup` entries are non-blocking).
Preflight (`attempt_replay=False`) is untouched. Result
construction, `numerical_replay` recording on incompleteness,
diagnostic fields, and the `not_attempted` fallback shape are
unchanged. No signature change is required; no new model, field, or
plumbing.

The two diagnostic `assess_checkpoint` call sites in the pipeline
diagnostics layer (currently inheriting the default
`attempt_replay=True`) must be frozen with explicit
`attempt_replay=False` in the same slice, so observer summaries keep
emitting `not_attempted` exactly as today. Only the production
checkpoint-discovery path adopts the relaxed gate.

## 6. Interaction with Gate-B discharge

The two repairs compose without contact: Gate-B discounts covered
visits from authority vetoes; replay gating keys off requirement
kind plus payload presence (never coverage state). A covered
obligation with an attempted replay can therefore yield `matched`
while its raw `source_lookup` entry persists diagnostically. An
uncovered obligation may now also yield `matched`, but final
authority still vetoes through the independent coverage conjunct —
verified by dedicated tests. Neither repair reads the other's output.

## 7. Diagnostics honesty

Checkpoint summaries in the production discovery path may now report
`matched`/`partial`/`unevaluable` statuses (instead of
`not_attempted`) in rounds whose only requirements are well-formed
`source_lookup`. Raw `source_requests` and `analysis_requirements`
remain intact, so a `matched` replay beside an outstanding source
lookup reads exactly as what it is:

```text
Replay:
    numeric consistency of the known evaluable mechanism

T3:
    source/writer search completeness

T5:
    applicability of the covered writer set to the exact use

T6B:
    conjunction of those independent claims
```

A `matched` replay never implies discovery completeness; global
source-search completeness belongs to T3 alone. Diagnostic observer
summaries keep emitting `not_attempted` via the frozen flag (§5), so
no downstream reader sees an unexplained verdict change. No report
schema changes.

## 8. Performance

Replay now runs in rounds previously skipped for `source_lookup`-only
requirements. Cost is bounded by the existing evaluation machinery
over the terminal dependency view; unchanged inputs re-derive nothing
new elsewhere. No caching layer is added in this slice; add one only
with measured justification.

## 9. Terminal-gap status

Terminal *observation* (publication binding) is fixture-achievable and
orthogonal. Terminal *matched* additionally needs an evaluable cone:
plain-name equations plus observed boundary inputs; member-write
equations currently break `safe_eval` (fixture constraint, not
architecture). If promotion-time probing shows transfer-terminal
evaluability blocked beyond fixtures, report it as the next gap
rather than extending evaluator semantics here.

## 10. Implementation slices

- **R1 — gate change + tests**: repoint the replay gate to
  well-formed-`source_lookup`-only requirements; freeze both
  diagnostic call sites with `attempt_replay=False`; unit/round
  tests per §12 (including the report-path pin: replay `matched`
  with unverified final authority keeps downstream confidence
  unresolved/non-authoritative).
- **R2 — P4 promotion attempt**: honest publication-terminal fixture
  with aligned samples; promote tripwires only on genuine green.
- **R3 — docs**: Gate-B spec note, pipeline spec note, matched-meaning
  paragraph (§7), this file's retirement (fold into R1 commit if preferred).

## 11. TDD matrix

1. well-formed `source_lookup`-only requirements → replay attempted
   (status ≠ `not_attempted`).
2. Uncovered source lookup + matched-capable graph → replay runs;
   final authority still vetoes via coverage (no false stop).
   Mandatory companion: matched replay + no current coverage →
   downstream confidence stays unresolved/non-authoritative.
3. Non-source requirement (e.g. construction, expression) →
   replay still skipped.
4. Malformed/payload-less `source_lookup` → replay blocked.
5. Unknown requirement kind → replay blocked.
6. Incomplete data → `partial`/`unevaluable`, never `matched`;
   `numerical_replay` requirement recorded.
7. Genuinely contradictory evidence → `mismatched` veto stands.
8. Raw requirements/diagnostics unchanged by the gate change
   (including both frozen diagnostic call sites still emitting
   `not_attempted`).
9. Gate-B discharge tests still green unchanged.
10. Existing replay skipped/attempted unit tests updated only where
    they pinned the old gate, with justification per test.
11. Vacuity suite green unchanged.
12. P4 tripwire progression (blocked only on fixture/terminal work
    after this slice, if replay behaves).
13. Full nearby suites green.
14. No retirement/exhaustion authority (regression).
15. No vacuous positive (regression).

## 12. Stop gates

- **STOP A**: replay correctness is shown to depend on current proof
  coverage (e.g. matched requires knowing writers are all found).
  Report the ownership change instead of proceeding.
- **STOP B**: ignoring a `source_lookup` lets replay falsely report
  `matched`+`complete` for an uncovered obligation in a way T6B
  cannot contain (i.e. final authority could succeed without
  coverage). Reject the change.
- **STOP C**: the fix requires redefining raw requirements or their
  diagnostic meaning. Reject.
- **STOP D**: a dependency cycle appears between replay verdicts and
  proof-adjusted discharge. Reject.
- **STOP E**: after honest replay, terminal observation/matching is
  structurally impossible for a non-vacuous proof-positive session.
  Report the next liveness gap; do not touch evaluator semantics
  here.
- **STOP F**: well-formed source_lookup cannot be distinguished from
  malformed/unknown requirements without spelling heuristics or
  proof data. The payload-presence rule above is the minimum
  structural check; if even that proves unimplementable, stop.

## 13. Documentation

- `CONTEXT.md`: one line on replay eligibility if glossary
  warrants it; otherwise none.
- ADR 0002: no change (still requires complete replay; gate
  refinement is implementation).
- Writer-coverage spec §10: note replay now runs with
  well-formed-`source_lookup`-only requirements, plus the
  matched-meaning decomposition from §7 (replay agreement ≠
  discovery completeness).
- Proof-pipeline spec: note replay-gating resolution + P4 path.
- This file is the durable record; fold a retirement pointer into
  R1's commit message or docs index if the repo requires it.

## 14. Non-goals

Evaluator semantics changes; member-write expression support;
transfer-execution proof; per-output scoping; proof-aware assess
plumbing; report schema changes; caching layers; T3/T5/T6A behavior.
