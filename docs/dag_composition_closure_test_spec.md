# DAG composition-closure test spec

Deterministic regression coverage proving that known narrow
DAG composition behaviors stay conservative, so future
benchmarks do not silently turn a test gap into another
architecture-discovery cycle. This document specifies TESTS
only. It proposes no DAG architecture, no new primitives,
no authority changes, and no benchmark-specific production
behavior.

Status: specification only. No implementation, no tests, no commits.

## 1. Purpose

Pin known conservative composition behavior — especially the
two narrow gaps found by the capability-closure audit (C4
per-window discrimination; history/persistence composition)
— so that a future benchmark using only existing DAG
primitives requires evidence and tests, never a new
primitive merely because a composition was untested.

## 2. Closure criterion

A new diagnostic case using only already-modeled concepts
should not require a new DAG primitive or authority path.

It may require:
- new fixtures
- new oracle evidence
- acceptance tests
- composition tests
- bounded bug fixes
- heuristic tuning

A genuinely new diagnostic concept may legitimately
require architecture extension.

## 3. Known narrow composition gaps

Two, neither foundational:

- **C4 — per-window writer discrimination.** Production
  `discriminate_candidates`
  (`flight_log_agent/analysis/temporal_selection.py`)
  evaluates overlap against the diagnostic window list as
  given; it has no per-window sole-overlap loop. A writer
  overlapping only window W1 and another overlapping only
  W2 are therefore jointly retained as eligible +
  unresolved under a merged scope, even where per-window
  reasoning could distinguish them.
- **History/persistence composition.** Temporal ordering
  plus value continuity does not automatically constitute
  a retention proof. No rule composes an ordering fact
  with a continuity fact into persistence; such a
  conclusion remains unavailable without explicit
  evidence (a logged/derived continuity relation or an
  equivalent explicit record).

## 4. C4 conservative contract

Production behavior to pin: given writers A (domain
overlaps only W1) and B (domain overlaps only W2) under
a merged scope, the selector must NOT claim one writer
uniquely applies across the combined scope. Expected
conservative outcome: both retained, unresolved note
present as appropriate, no uniqueness claim, no
authority upgrade. Failing closed here is correct
behavior, not a bug.

## 5. C4 per-window acceptance pattern

Acceptance-side (tests/oracles only, never production):
apply the existing generic per-window qualification
independently per window to derive W1 → writer A and
W2 → writer B, using only generic window/writer/overlap
structures. This test proves the missing production
precision is a composition rule, not a missing
primitive: everything it consumes already exists.

## 6. C4 production-rule deferral rule

C4 production per-window discrimination is a DEFERRED
PRECISION IMPROVEMENT, not current work. It becomes
required only if a governing spec mandates that live
production reports distinguish writers per window. No
such requirement exists today; Takeoff‐style ordered
evidence is carried acceptance-side. Do not implement
the rule under this spec.

## 7. History/persistence conservative contract

Given value V observed at T1, same/similar value at T2,
and known ordering — but no explicit persistence or
continuity evidence — the DAG must NOT manufacture
"V persisted continuously from T1 to T2". Expected:
retention unresolved, `state_alignment`/history
requirement remains open, no authority upgrade. If
current abstractions CAN represent a logged/derived
continuity relation, a composition test must use it;
otherwise the spec records that history-persistence
conclusions remain unavailable without explicit
evidence, rather than inventing a relation.

## 8. Unknown-state contracts

Pin the conservative handling the audit verified:
`_from_graph` unknown → `mixed` (a mixed verdict
authorizes no stronger conclusion than unknown); a
`dag_observation` vertex with no correspondence is
unresolved, never silently satisfiable; partial /
unevaluable replay stays unresolved evidence, never
support. These tests pin behavior, they do not change it.

## 9. Polarity split

Record as specified behavior (not a bug): temporal
unknown eligibility may retain candidates (existence is
not a claim), while replay/checkpoint unknown fails
closed for authority; `None`-scope may mean full-domain
evaluation while `None`-windows means unresolved.
Candidate-retention polarity is not authority polarity.
If no ADR records this, this section serves as the
record; a future ADR may supersede it without changing
behavior.

## 10. Synthetic composition matrix

Deterministic tests over synthetic DAG structures (no
benchmark fixtures required):

1. Multiple writers + per-window applicability +
   replay unavailable → no false exact-writer claim
   (both retained, unresolved as appropriate).
2. Retained source evidence + later temporal window +
   missing persistence proof → evidence available,
   persistence unresolved (no manufactured continuity).
3. Per-window writer discrimination + contradiction in
   one window → contradiction does not contaminate the
   unrelated window, where temporal evidence
   representation supports the distinction; otherwise
   the test records the exact limitation instead of
   asserting separation.
4. Multiple equivalent writers + different windows +
   claim independent of exact execution writer →
   mechanism-level conclusion may stand while exact
   writer stays unresolved (existing W3 rules).

## 11. Test classifications

Every planned test is classified up front:

- PASS CURRENT PRODUCTION (conservative behavior pins).
- PASS ACCEPTANCE-SIDE HELPER (per-window pattern,
  alias/oracle-free generic structures).
- EXPECTED CONSERVATIVE LIMITATION (C4 merged-scope
  retention; unavailable persistence conclusions).
- WOULD REQUIRE PRECISION IMPLEMENTATION (explicitly
  deferred; never asserted as passing today).

No test may be written in a way that assumes the
deferred C4 production rule exists.

## 12. Benchmark relationship

Benchmark acceptance suites (RTL/TECS/Takeoff/Airspeed)
remain the semantic oracles; these composition tests
use generic synthetic fixtures and assert primitive
behavior, never benchmark output strings. A composition
test that needs a benchmark name to state its
expectation is misspecified — rewrite it in primitive
terms.

## 13. Genericity requirements

All production-facing composition tests use generic
synthetic data (abstract writers, windows, verdicts,
evidence kinds). No RTL/TECS/Takeoff/Airspeed literals,
no module names, no commit-specific values in
production-facing assertions. Benchmark-specific
acceptance tests may verify known cases separately.
Enforced by the existing case-term guard patterns.

## 14. Architecture-vs-test distinction

A test failure here means either a conservative
behavior regressed (fix production) or a deferred
precision was accidentally implemented (update this
spec's classification, do not silently accept new
semantics). Test gaps are never architecture gaps;
a missing test is a missing test.

## 15. Conditions that would reopen DAG architecture

Only these: a genuinely missing evidence category; a
missing authority state; a composition unrepresentable
with existing primitives; a benchmark-specific
production requirement; a contradiction with the
closure audit. Each classifies as FOUNDATIONAL DAG GAP
and stops test-plan work until re-scoped. None is
currently known.

## 16. Recommended implementation order

1. Deterministic DAG composition-regression tests
   (this spec).
2. Bounded-frontier implementation (separate spec).
3. Bounded-frontier offline acceptance (separate spec).
4. Paid validation only with explicit user approval.

Historical convergence note (evidence, not proof):
the last production capability addition was W3A
temporal qualification; subsequent W3B/W3C/Airspeed
work was acceptance, observability, and docs. Do not
claim mathematical completeness from this history.
