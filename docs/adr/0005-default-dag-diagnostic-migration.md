# Default-DAG diagnostic migration with legacy fallback

We migrate the user-facing diagnostic path to DAG-primary in stages, keeping
legacy as an explicitly marked temporary fallback. The existing T6B
stop-authority conjunction (`legacy_verified` AND coverage AND applicability
AND non-vacuous proof AND Gate-B discharge, per
`flight_log_agent/analysis/checkpoint_discovery.py`) stays unchanged: this
migration changes routing and fallback behavior, never proof or stop
authority. Legacy retirement is not decided here.

## Status

Accepted. Staged migration; stop-authority decoupling explicitly out of scope.

## Decisions

- **D1 target.** Staged DAG-primary migration. DAG becomes the default
  user-facing diagnostic path once readiness gates pass; legacy remains an
  explicitly marked temporary fallback. Not permanent dual authority, not
  immediate removal.
- **D2 legacy role.** Temporary migration fallback, operational throughout.
  Retirement needs a separate later decision against an explicit S5-style
  gate; this ADR does not state that legacy will be removed.
- **D3 readiness gates.** Zero semantic mismatches on the four deterministic
  fixtures (mechanism family, rival exclusions, source/log grounding,
  strength compatibility — never prose or confidence); fallback markers
  tested; silent missing-snapshot fallthrough eliminated; wall time, LLM
  cost, and fallback rate measured with budgets ratified from measurements.
- **D4 stop authority.** The T6B conjunction is unchanged through this
  migration, including the `legacy_verified` conjunct, the
  required-proof-absent rule for omitted inputs, and the
  empty-non-degenerate defers-to-legacy path. Routing changes; authority
  does not.
- **D5 replacement authority.** None. Nothing is removed, so no new
  authority boolean, arbitration layer, or stop token is introduced.
- **DAG-decisive routing predicate.** DAG-decisive means the existing
  DAG-side conditions already hold together (replay-complete,
  judge-verified, feasibility-passing, validation-passing). It decides
  routing/precedence only. DAG-decisive is not a stop-authority layer.
- **D6 proof plumbing.** T3 certificates already feed authority
  observation; not a blocker. `record_stages` stays inert with one
  revisit trigger: only if coverage-driven scheduling work is proposed.
  Coverage retirement stays scheduling-only; retirement/queue/visited/
  exhaustion never authorize stop.
- **D7 parser parity.** The 31 strict legacy-parser xfails block legacy
  retirement only, never DAG default: DAG runs select tree_sitter
  per run while legacy remains available as fallback.
- **D8 parser target.** tree_sitter per-run for DAG work;
  `FLIGHT_LOG_SOURCE_PARSER` retained as explicit selector. Parser
  retirement is a future migration decision.
- **D9/D10 reports and LLM role.** The DAG deterministic report becomes
  default at flip; diagnostic meaning stays with deterministic findings.
  LLMs are not eliminated: seeder/judge remain under replay, proof, and
  validation ceilings, and any report LLM is presentation-only. The
  deterministic part is the constrained evidence/report boundary, not
  every upstream inference.
- **D11 benchmark gate.** Acceptance fixtures act as regression oracle
  and migration gate (family, exclusions, grounding, strength
  compatibility); never runtime lookup, prose, or confidence equality.
  Airspeed stays BEST_SUPPORTED absent independent benchmark evidence.
- **D12/D13 fallback and precedence.** Missing snapshot, unsupported or
  unevaluable writers, missing signals, failed applicability/Gate-B, or
  unresolved evidence permit legacy fallback that is explicitly marked,
  auditable, and certainty-preserving (existing report/evidence slots;
  no schema change without a later spec). One small named pure
  precedence function centralizes the fixed rules: DAG-demonstrated
  contradiction reports contradiction without consulting legacy
  (fail-closed); else DAG-decisive routes to DAG; else explicit legacy
  fallback. Both-verified-different fails the migration gate as a
  semantic mismatch. Unknown never becomes verified via fallback.
- **D14 shadow.** Harness-only sequential comparison over the four
  fixtures plus representative samples, using existing audit
  artifacts/tests. No production dual-run, no new telemetry.
- **D15 flags.** `FLIGHT_LOG_DAG_DISCOVERY` retained through
  measurement and bake-off as the rollback mechanism; removal only
  after explicit exit criteria and soak. `FLIGHT_LOG_SOURCE_PARSER`
  retained as explicit selector.
- **D16 snapshots.** Missing snapshot becomes an explicit
  unavailable-source condition with optional marked legacy fallback.
  Never fabricate source; never silently claim DAG grounding.
- **D17 performance.** Wall time, LLM cost, and fallback rate measured
  before flip; budgets ratified from measurements, none fabricated here.
- **D18 stateful replay.** Not required; stateless replay stands.
- **D19 coverage retirement.** Scheduling-only; never stop, readiness,
  or fallback authority.
- **D20 retirement gate (future).** Legacy retirement remains UNDECIDED.
  It will require its own decision with evidence such as completed
  bake-off, sustained compatibility, acceptable fallback rate,
  sufficient contract parity, rollback evidence, and stable report
  behavior — adapted from the still-valid retirement-gate concept in
  the untracked prior plan, whose old phase numbering is not adopted.

## Considered Options

- **DAG primary, legacy fallback (chosen).** Smallest change satisfying
  correctness (proven capability reaches default runs), safety
  (authority untouched, explicit fallback, flag rollback), and
  maintainability (one precedence function, no arbitration layer).
- **Permanent dual authority (rejected).** An arbitration component
  deciding between two first-class diagnosers adds ownership and
  coupling without a force requiring it; fixed precedence rules with
  tests dominate it.
- **Immediate flip or immediate retirement (rejected).** Removes the
  proven empty-relevance behavior and the only cross-check while
  seeder/judge LLMs and backend divergence are unmeasured; rollback
  would have no harness behind it.

## Consequences

- P0 (readiness measurement + routing-predicate definition) is the next
  implementation phase; phases P1–P4 follow the staged sequence with
  per-phase rollback intact.
- Parser xfails, doc cleanup, and untracked-artifact triage proceed in
  parallel; stateful replay and stop-authority decoupling are
  explicitly not part of this migration.
- Non-negotiable invariants preserved: generic production behavior, no
  benchmark-specific paths, source evidence ≠ proof authority, temporal
  qualification ≠ execution identity, benchmark strength ≠ live
  confidence, missing evidence ≠ contradiction, fail-closed
  unsupported behavior, inspectable raw evidence, no silent certainty
  upgrades, no queue/exhaustion/retirement stop authority.
