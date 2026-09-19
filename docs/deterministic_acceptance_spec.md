# Deterministic acceptance framework

Shared contract for deterministic full-system acceptance cases of
`flight-log-analysis-agent`.

Status: specification only. No implementation, no fixtures, no tests.

## 1. Problem statement

Subsystem tests (including the P4 stop-authority E2E) stay green
while the supported deterministic pipeline as a whole can still
fail to turn a real log plus a real source snapshot into a grounded,
schema-valid report. Each future acceptance case needs the same
rules for what counts as evidence, what counts as proven, and what
must never be frozen — without duplicating them per case.

## 2. Solution

One shared framework spec (this document) owning evidence
semantics, strength semantics, sidecar schema, normalization, and
tolerance policy; thin per-case specs owning question, fixtures,
mechanism, numbers, and case evidence. RTL is the first case
(`docs/rtl_deterministic_acceptance_spec.md`).

## 3. Benchmark ground truth policy

Benchmark ground truth is the semantic conclusion of a
human-accepted successful reference run, subject to deterministic
refinement. Default rule: human-accepted conclusion becomes the
benchmark semantic target. Never reopen it merely because current
deterministic tooling cannot yet reproduce every piece of evidence.

Revise a benchmark conclusion only on: direct source
contradiction, direct log contradiction, numerical contradiction,
or a materially stronger source/log-supported explanation. A
meaning-preserving refinement is allowed; RTL is the model
(cone-related altitude behavior refined to cone-enabled helper
selected, outside-acceptance-radius floor wins numerically).

## 4. Evidence-strength model

```text
PROVEN: source/log evidence deterministically reconstructs or
uniquely identifies the mechanism; material alternatives excluded.
Benchmarks: RTL, TECS, Takeoff.

BEST_SUPPORTED: evidence strongly favors the accepted mechanism
and important alternatives are weaker/contradicted, but one or
more discriminating facts are unrecoverable today. Benchmark:
Airspeed (peak mechanism numerically proven; full-window slew
residual/state not currently modeled).

PLAUSIBLE: consistent with evidence, meaningful alternatives
remain similarly viable. No current benchmark requires it.

UNKNOWN: evidence cannot discriminate meaningfully. No current
benchmark requires it.

CONTRADICTED: evidence conflicts with the benchmark mechanism.
No current benchmark is contradicted.
```

Do not add categories without an actual case requiring them.

## 5. Strength versus pipeline status

Diagnostic evidence strength (above) and pipeline-derived
verification status are separate concepts. Pipeline status uses
existing fields: `confirmed` / `unconfirmed`, per-hypothesis
`confidence`, `complete` / `partial` / `guess`, replay/checkpoint
state, coverage/applicability proof, T6B stop authority. Never map
these one-to-one onto PROVEN / BEST_SUPPORTED automatically.

Example: Airspeed evidence is BEST_SUPPORTED while the current
deterministic pipeline may remain `unconfirmed` / low — a tooling
gap, not contradictory evidence. Likewise a PROVEN diagnostic
mechanism must NEVER manufacture T3 coverage, T5 applicability, or
T6B stop authority; proof authority remains strict and separate.

## 6. Verification primitives

Reusable forms; no case must exercise all of them:

- source equation → predicted quantity → logged
  command/observation comparison (with stated tolerance);
- command-vs-observed tracking, with lag accounting where the
  controller slews;
- parameter-controlled branch/gate evaluation from logged
  parameters;
- temporal ordering (command → response → transition);
- rival-hypothesis exclusion using logged gates;
- source file + symbol anchoring (file+symbol+containing range,
  never exact line equality);
- setpoint/text corroboration (logged setpoints, logged
  Navigator text);
- acceptable derivation of unlogged locals (substitution, §8).

## 7. Evidence categories

`LOG_DIRECT`, `LOG_DERIVED`, `PARAMETER`, `SOURCE_DIRECT`,
`SOURCE_DERIVED`, `SOURCE_PLUS_LOG_DERIVED`, `UNAVAILABLE`.
`AGENT_INFERENCE` may be recorded for provenance analysis but must
never satisfy a deterministic golden evidence requirement.

## 8. Evidence substitution

A source-level local need not be directly logged if established
via: another logged signal, logged parameter, source constant,
deterministic computation, coordinate/distance derivation, a
bounded interval sufficient to decide a branch, or a downstream
output uniquely constraining the source calculation. Test the fact
being established, not one observation path.

## 9. Missing-evidence impact

`NON_DECISIVE` (does not distinguish surviving explanations;
e.g. RTL mission-vs-rally destination label after altitude and
rule are reconstructed). `DISCRIMINATING` (would strengthen or
distinguish, but remaining evidence still permits a
best-supported conclusion; e.g. airspeed per-sample slew state).
`CRITICAL` (prevents useful discrimination). No current benchmark
has a CRITICAL gap.

## 10. Missing evidence is not negative evidence

Missing ≠ contradictory; not-deterministically-proven ≠ false;
plausible ≠ proven. Diagnostic reasoning only — never weakens
strict proof authority.

## 11. Sidecar schema

Minimal shared semantic schema (JSON; field names illustrative,
TDD adopts):

```text
scenario_id, question, log, snapshot,
mechanism_family,
verification_claims[] {id, statement, expected_status,
  evidence_requirements, acceptable_substitutions},
required_evidence {source[], log[], param[]},
strength, unavailable[], forbidden[],
normalization {unstable_fields[]}
```

Case-specific extensions ride alongside, never in every case:
numeric equation + inputs + output + tolerance, temporal
sequence, branch conditions, command-vs-observed relation,
sensor-reliability note, missing mission/dataman state,
text-message corroboration.

## 12. Deterministic-control separation

Scripted `run_agent` responses live in the test harness, never in
the semantic sidecar. Golden data states what must be true; the
harness states how nondeterministic model decisions are
controlled. A shared sidecar must stay reusable for a future
live-model run of the same scenario.

## 13. Normalization

Never golden: exact prose, report titles, generated IDs,
vertex/edge counts, fingerprints, session IDs, UUIDs, absolute
paths, timing/runtime, RSS, incidental ordering, exact LLM call
ordering unless contractually required, `explaining_branches`
IDs, exact `unresolved_evidence` strings, replay status strings,
`match_fraction` values.

## 14. Future-improvement tolerance

Golden tests keep passing when implementation legitimately
discovers more evidence, raises confidence, moves unconfirmed →
confirmed, resolves unavailable evidence, or finds supporting
source paths — provided the diagnosis stays semantically
compatible and no forbidden outcome fires. Never freeze today's
pipeline limitations as golden truth.

## 15. Contradiction policy

Genuinely conflicting future evidence (snapshot mismatch,
equation-vs-command contradiction, false branch condition, a
competitor that explains the result while the benchmark cannot)
fails loudly and triggers benchmark/spec review. Never silently
loosen the golden.

## 16. Benchmark inventory (short)

RTL (PROVEN, first target), Airspeed (BEST_SUPPORTED: load-factor
reconstruction, gated rival exclusion, wind-reliability context),
TECS (PROVEN: transition timing, first-sample equation,
multi-sample RMSE, counterfactual exclusion), Takeoff (PROVEN:
parameter state machine, TAKEOFF→WAYPOINT sequence,
command-vs-home differences, Navigator text). Inventory only —
thin specs for the latter three do not exist yet and RTL TDD must
not depend on them.

## 17. File structure

- `docs/deterministic_acceptance_spec.md` (this framework).
- `docs/<case>_deterministic_acceptance_spec.md` per thin case
  (RTL: `docs/rtl_deterministic_acceptance_spec.md`).
- `tests/acceptance/<case>.json` per semantic sidecar (RTL:
  `tests/acceptance/rtl_weird_height.json`). One directory is
  justified by the planned four-case suite, not proliferation.
- Acceptance test modules follow the existing `tests/test_*.py`
  convention (RTL: `tests/test_acceptance_rtl.py`, created in
  TDD, not here).

## 18. Documentation disposition

Stale P4-xfail statements (outstanding-work spec, pipeline spec
§26b-class note, replay spec R2 notes) are fixed in a separate
tiny docs commit — never inside an acceptance-test commit. No
broad cleanup.

## 19. Stop gates

STOP framework work if: (A) the four cases cannot share one
evidence model; (B) RTL needs semantics incompatible with the
others; (C) strength cannot be separated from pipeline status;
(D) the sidecar would have to embed LLM control; (E) RTL cannot
remain a bounded first TDD target. None triggered: all four cases
reduce to equation/temporal/gate evidence over the same
categories, strength separates cleanly from status (§5), control
separates cleanly from semantics (§12), and RTL stays bounded.

## 20. Out of scope

Live-LLM test and scoring; golden dataset beyond the inventory;
recommendation generation; legacy-backend parity;
history/transfer implementation; harness frameworks; performance
work; fixture provisioning infrastructure.
