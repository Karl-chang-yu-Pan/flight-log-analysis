# Temporal causal qualification + TECS acceptance spec (W3A+W3B)

Combined implementation-ready contract for the shared temporal
diagnostic-window / qualification capability (W3A) driven jointly
by the TECS deterministic acceptance case (W3B). Builds on the
ownership decisions of
`docs/adr/0004-temporal-causal-qualification-ownership.md` without
reopening them. Out of scope: W3C Takeoff acceptance (must not be
foreclosed), Airspeed representation, numeric-check ↔
source-operation linkage ownership, new graph or view types, new
persistent temporal models.

Status: specification only. No implementation, no fixtures, no tests.

## 1. Problem statement

Static source topology shows an operation is reachable without
showing it is the causally relevant writer for the time window a
diagnostic question asks about. The accepted TECS conclusion — the
~1.012 m/s height-rate setpoint immediately after the relevant
mode transition is a bumpless/restart transient, not a persistent
~1 m/s limit — is decided temporally (first sample/window after
transition/reinitialization), not by static graph reachability:
initialize/update paths share or partially share the static
backward cone. Report selection has no rule that consumes a
diagnostic window, so the transient-vs-limit distinction cannot be
reproduced deterministically today.

## 2. Transition-event specification input

`QuestionedCondition` (mechanism_judge.py seed structures) gains
one optional additive field carrying a generic
transition-event specification. Minimal schema (each field
justified; no temporal query language):

```text
transition_signal: str  (REQUIRED to activate)
    Logged topic.field whose value changes mark candidate events
    (e.g. a mode/state signal). Justification: identifies the
    observation stream without naming any mechanism.

from_value / to_value: optional
    Exact logged values bounding the change. None means any value
    on that side (any-change). Justification: expresses A→B
    transitions when the case needs them, while plain changes
    stay expressible.

event_selection: optional, "first" | "last"
    Which matching event to use. Justification: explicit
    disambiguation chosen by intent, never an implicit default.
    Absent plus multiple matches → unresolved (see §5).

window: relation "after" | "before" | "around"
    plus ONE extent form:
      duration_s: float,  OR
      first_sample_of: str (a logged topic.field)
    Justification: "after/before/around" covers phase-relative
    questions for both W3B and W3C; duration covers phase spans;
    first-sample covers transient-at-onset questions (TECS) using
    exact sample timestamps, never wall-clock epsilon tricks.
```

No TECS/nav-state/mode-integer/parameter names appear in this
schema or in any production code consuming it. Case mappings
live in acceptance seeds/sidecars.

## 3. Backward compatibility

Absent transition specification → byte-/semantic-equivalent
temporal behavior: questioned-condition windows as today,
`EvaluationScope` construction unchanged, no scope threading
effects, no candidate filtering. Pin with T1. No question is
required to carry transition intent.

## 4. Deterministic event derivation

A small pure derivation helper in the questioned-condition /
diagnostic-window derivation seam (the layer around
`evaluate_questioned_condition_windows`; `$tdd` may extend that
function or adjoin a helper without changing architecture)
evaluates the specification against prepared logged-signal
samples (exact timestamps; timeline value-change events remain
corroboration):

```text
transition spec + prepared samples(signal) → matched event(s) | unresolved
```

Rules: scan ascending timestamps; an event is a consecutive
sample pair whose previous value satisfies `from_value` (or any
when absent) and whose next value satisfies `to_value` (or any
change when absent); value comparison uses logged
`json_safe_value` semantics. Deterministic, source-order
independent, no LLM interpretation, no source-code
interpretation. Operates over logged observations only.

## 5. Event ambiguity

Zero matching events → no diagnostic window; fail closed per
§18. Multiple matching events → `event_selection` ("first" |
"last") chooses explicitly. No selection field plus multiple
matches → unresolved; never default to first/last silently.

## 6. Diagnostic window derivation

A matched event becomes concrete `EvaluationScope.windows`:

- `after` + `duration_s: d` → `[t_event, t_event + d]`.
- `before` + `duration_s: d` → `[t_event - d, t_event]`.
- `around` + `duration_s: d` → `[t_event - d, t_event + d]`.
- Any relation + `first_sample_of: S` → `[t_event, t_first]`
  where `t_first` is the first prepared sample timestamp of S
  with `t >= t_event` (closed comparison on exact timestamps).
  A degenerate point window is valid for selection/eligibility
  overlap; replay over a zero-duration domain keeps its
  existing `not_attempted` invariant unchanged.

Duration and first-sample forms are alternatives per window
spec, not combined. Only closed-interval, sample-timestamp
semantics; no epsilon widening.

## 7. TECS first-sample intent (fixture/seed, NOT production)

The TECS acceptance seed supplies the case mapping: the
relevant logged mode/state transition signal with its
from/to values, `event_selection` as pinned during TDD, and a
window of relation `after` with `first_sample_of` naming the
logged TECS height-rate setpoint signal. Production sees only
the generic §2 schema.

## 8. Multiple windows

`EvaluationScope.windows` stays plural throughout derivation,
construction, threading, and selection. W3B may exercise one
effective window; nothing may assume, require, or collapse to
exactly one interval, preserving W3C Takeoff ordered windows.
Pin multi-window preservation in unit tests.

## 9. Scope construction

Derived transition windows feed the existing
`EvaluationScope.from_result` shape unchanged (windows +
signal/reference/units/frame/assumptions/error). Signal,
reference, units, and frame continue to come from the existing
questioned-condition evaluation. When both a questioned
predicate and a transition spec are present, scope windows are
their intersection; transition-only seeds use derived windows;
questioned-only seeds use current behavior. No parallel
temporal-scope type.

## 10. Scope threading (minimum plumbing)

`EvaluationScope` (or its already-derived windows) must reach
the temporal-selection boundary. Minimum, all optional and
pass-through only (no intent parsing outside derivation, no
scope mutation):

- `run_dag_discovery_stage` already builds scope; thread it
  (or its windows) into `replay_terminal_expressions` as a new
  optional parameter (default None = current full-domain
  behavior), which forwards to the existing `scope` parameter
  of `replay_dag_roots`.
- Thread it into `build_report_from_dag` as a new optional
  parameter consumed ONLY by the report-adjacent temporal
  selector before source-ref normalization. `build_report_from_dag`
  itself parses no intent.

Outside temporal-seeded cases replay and report semantics are
unchanged (T1 pins this).

## 11. Derivation placement

Prefer extending the questioned-condition / diagnostic-window
derivation seam (around `evaluate_questioned_condition_windows`)
or adjoining a small pure helper in the same ownership area.
Do not place derivation in the report builder, replay, or the
source-ref selector. Exact function placement is `$tdd`
territory within this fixed seam.

## 12. Temporal eligibility (exact)

A represented candidate is temporally eligible iff its
candidate domain intersects the relevant diagnostic window
under closed-interval semantics. Overlap means the candidate
could be relevant during the questioned interval; it never
means the candidate uniquely caused the observed behavior.

## 13. Candidate domain (existing structures only)

Deterministic priority, first available wins, per candidate:

1. Replay-result `active_windows` for that writer op id when
   a replay result exists (already comparison-intersected by
   replay construction).
2. Otherwise the writer's gating control-branch windows
   intersected with scope windows: `always_true` contributes
   the scope span; `always_false` contributes empty;
   `unknown` branches without windows constrain nothing
   (the candidate stays eligible but unverified — missing
   feasibility information never excludes).
3. Intersect the result with the diagnostic window(s); empty
   intersection → temporally ineligible.

No second writer-window model is introduced.

## 14. Writer domain (terminology)

"Writer domain" is ADR-0004 shorthand for item 13's outcome.
Nothing named `WriterDomain` (or `TemporalEvidence`,
`TemporalSelectionResult`, `TemporalGraph`, `PhaseGraph`) is
persisted. If implementation ever claims to need one: STOP I.

## 15. Unique discrimination (exact, allow-listed)

After eligibility, at most one of the following existing
signals may establish unique causal selection, checked in
this order:

1. Sole overlapping writer domain (exactly one eligible
   candidate).
2. Domain separation across ordered windows (each window
   admits a different single candidate).
3. Available in-window replay support for one candidate over
   rivals (an evaluable replay result for that writer op
   meeting the existing replay matched threshold over a
   window-overlapping domain).
4. Observable temporal ordering excluding rivals (timeline
   order contradicts a rival's required ordering).

ForbiddenChoosers: source order, vertex ID, insertion order,
report rank, or any incidental ordering. Do NOT solve generic
numeric-check ↔ source-operation linkage: discriminator 3
uses only already-linked per-writer replay results.

## 16. Replay is optional discrimination

Replay match is never required for temporal eligibility.
Replay discriminates only where existing per-writer results
already provide in-window evidence through unchanged replay
semantics (plus the §10 scope threading, which preserves
replay behavior outside temporal-seeded cases).

## 17. Multiple eligible writers (fail-closed, pinned)

Candidate A eligible, candidate B eligible, no item-15
evidence distinguishes them → retain both eligible
candidates, make no unique causal selection, surface
unresolved evidence through the existing
`unresolved_evidence` report path with deterministic
wording. No arbitrary winner. Pin with T8 (including an
input-permutation variant per T11).

## 18. Missing diagnostic window

Event/window derivation failure (zero events, ambiguous
events without selection, unresolvable scope) → no
temporally-selected causal claim; preserve broader evidence
that remains valid under existing rules; surface unresolved
evidence through existing mechanisms. Missing window is not
contradiction.

## 19. Temporal contradiction (existing owners only)

Real contradiction, each observable by its current owner:

- candidate domain has no overlap with the required window
  (selector);
- claimed event ordering contradicts observed timeline
  ordering (timeline);
- available in-window replay directly contradicts the
  candidate (replay → existing mismatch reporting).

Missing hidden state, multiple eligible writers, and missing
unique discrimination are NOT contradictions.

## 20. Data-flow relationship

Existing DAG candidate relationships are preserved. Temporal
qualification filters WS1/WS2-qualified candidates (terminal
writers and helper entries) by window; it replaces no
terminal-writer identity, helper data qualification, or
source identity.

## 21. Control/branch relationship

Temporal logic may READ feasibility and `active_windows` for
context/eligibility. It must NOT verify unknown branches,
revive `always_false` branches, or create source refs from
control ancestry alone.

## 22. WS1/WS2 source-ref integration

Order of operations inside report construction:

```text
represented source candidates
→ temporal eligibility/discrimination (only when a diagnostic
  window was successfully derived; otherwise unchanged)
→ existing source-ref normalization
→ dedup/order/budget/render
```

Do NOT rebuild source identity, helper identity,
cross-role overlap, tiering, or the `[:8]` budget.

## 23. Helper interaction

WS2 helper identity (callable + stable caller SOURCE site +
result path) is unchanged, as is helper data qualification.
An already-WS2-qualified helper lying wholly outside the
relevant diagnostic window may be filtered by the temporal
layer using its gating-branch/consumer domain per §13.
Filtering never redefines helper identity.

## 24. TECS source candidates (as known today)

The spec documents, without re-diagnosing, the represented
landscape `$tdd` must confront (per trial/contract audits):

- initialize/update paths share or partially share static
  backward topology;
- `_tecs_is_running` fallthrough/gating inputs (order-13
  local candidates observed);
- `debug_output.control.altitude_rate_control` opaque member
  path with known receiver-identity degradation;
  receiver-state and alternative-writer obligations may
  remain open;
- replay enumerates terminal ops only; upstream helper
  writers appear in ancestors/dependency issues, with
  per-writer replay domains only where results exist.

The spec therefore requires `$tdd` to record per candidate:
represented writer identity, available domain per §13,
whether replay evaluates it, and what evidence from §15 can
distinguish it. Assumptions beyond this record are
forbidden.

## 25. STOP if TECS remains nondiscriminable

If, after the transition-relative window, multiple TECS
writer candidates remain eligible and writer domains plus
existing replay plus observable ordering cannot distinguish
them: STOP. Invent no winner. Report the exact missing
discriminator and recommend a narrow `$wayfinder` for that
relation. The accepted TECS benchmark stays PROVEN
externally; current production selection simply cannot
reproduce it yet.

## 26. TECS acceptance sidecar

Extend `tests/acceptance/` conventions (follow
`rtl_weird_height.json`: scenario_id/question/inputs/
expected/strength/unavailable/forbidden/normalization) with
temporal-sequence extensions already anticipated by the
framework:

```text
scenario identity, question, log path (provisioned real log
  with mode transition + tecs_status height-rate telemetry,
  gitignored uploads path per RTL A1 pattern),
source snapshot pin, accepted mechanism family
  (restart/bumpless transient, not persistent limit),
transition-event intent (§2 schema, case mapping),
diagnostic-window intent (after + first-sample),
expected early transient behavior,
expected later steady-state behavior,
required source/observation/temporal/numeric evidence,
strength = PROVEN,
unavailable/non-critical evidence,
forbidden claims (persistent-limit diagnosis,
  fabricated helper/interior refs, numeric claims beyond
  tolerance, strength downgrade on tooling limits).
```

## 27. TECS benchmark source facts

In-repo verified values (do NOT re-round): reference
`3.3737552`, first expected setpoint `1.0121266`, relation
`(a - b) / 5.f + 0.3f * c`, per
`tests/test_mechanism_discovery.py:4857`. Transition time
(~24.9127 s), first-sample lag (~17.9 ms), and later ~5 m/s
behavior are approximate reminders ONLY; `$tdd` pins exact
values measured from the provisioned log in the sidecar.
Production logic must contain none of these numbers.

## 28. TECS numeric relation

Acceptance preserves first expected setpoint ≈ `0.3 ×
reference` with tolerance following deterministic-acceptance
conventions (RTL precedent: absolute tolerance pinned in the
sidecar; exact value set during TDD from log quantization).
This equation is benchmark verification evidence in the
sidecar/acceptance test, never generic temporal-qualification
logic.

## 29. TECS early-vs-later behavior

Acceptance proves both the early post-transition transient
(≈1.012) and the later return to normal ~5 m/s setpoint
behavior. The pair is what excludes the rival persistent
~1 m/s limit diagnosis.

## 30. TECS hidden reinitialize state

Direct logging of `_reinitialize_tecs` (or equivalent
internal state) is NOT required where deterministic
observable/source evidence already establishes the accepted
conclusion. Missing internal state is not contradiction.

## 31. TECS benchmark strength

Sidecar strength stays PROVEN. Never altered on report
confidence alone.

## 32. TECS live pipeline status

No promotion of `confirmed`/confidence from temporal
qualification. Existing confidence rules authoritative;
benchmark truth and live pipeline status stay separate (a
PROVEN benchmark may remain unconfirmed/low while tooling
catches up).

## 33. Observation evidence

Timeline/signal values used for event/window derivation
remain observation evidence. They never become `CodeRef`
grounding or proof authority.

## 34. Logged text

TECS needs no textual-event dependence; do not add it to
exercise the abstraction. (Takeoff may use message
corroboration under W3C.)

## 35. Unresolved evidence

Reuse the existing `unresolved_evidence` report path with
deterministic wording. No new generic unresolved
classification model. Sidecars may use NON_DECISIVE /
DISCRIMINATING / CRITICAL where already appropriate.

## 36. Confidence isolation

Temporal eligibility/selection never directly sets
`confirmed`, `high` confidence, PROVEN, or BEST_SUPPORTED.

## 37. branches_verified isolation

No write, no change, no helper-awareness change.

## 38. Replay isolation

No changes to writer enumeration, expression domains,
`match_fraction` semantics, or completeness semantics —
except the §10 optional scope threading, which preserves
existing semantics outside temporal-seeded cases (T1 pins).

## 39. Proof isolation

No changes to T3, T5, T6B, Gate-B, R1, coverage,
applicability, checkpoint authority, proof stores, or
retirement.

## 40. Backward compatibility

T1 pins: existing analyses without a temporal seed produce
unchanged window/scope/selection behavior. Transition
intent is strictly opt-in; no question is required to
carry it.

## 41. Generic code rule

Production additions must contain no case-specific literals
(TECS names, this case's nav-state values, module paths,
parameter names, timestamps, or benchmark numbers) except
where an existing generic parser naturally knows a
source/log field schema. Case values live in
tests/fixtures/sidecars. Pin with a case-term scan test per
repo convention (`test_no_case_specific_terms.py` pattern).

## 42. Workstream scope

This spec covers W3A+W3B together. W3C Takeoff is excluded
from implementation, but §8 plus multi-window derivation,
selection, and discrimination rules must not foreclose
ordered multi-window diagnostics.

## 43. Implementation ownership (expected, minimal)

- Question-intent seed dataclasses (mechanism_judge.py):
  additive optional transition-spec field(s) plus minimal
  seeder guidance consumable deterministically.
- Diagnostic-window derivation (dag_pipeline scope-derivation
  area per §11): event matching + window construction.
- `EvaluationScope` construction/threading: feed derived
  windows through the existing constructor; thread scope to
  replay entry and report-adjacent selection (§10).
- Report-adjacent temporal candidate selector (dag_pipeline
  selection neighborhood): eligibility (§§12–13), unique
  discrimination (§15), fail-closed retention + unresolved
  surfacing (§§17–18).
- TECS acceptance fixture/sidecar/tests (fixtures, sidecar,
  A-series tests following the RTL pattern).

No other owners are expected to change; `$tdd` must justify
any expansion beyond this list before making it.

## 44. No persistent temporal model

Pure functions plus existing structs only. No
`TemporalEvidence`, `TemporalSelectionResult`,
`TemporalGraph`, or `PhaseGraph`. If implementation proves
one unavoidable: STOP I.

## 45. TDD plan

Seams first: derivation helpers (pure, log-samples in →
event/windows out), `EvaluationScope` construction,
report-adjacent selector (dag + windows + replay results in
→ narrowed candidates out), then TECS acceptance. One
vertical slice per cycle (single test → minimal code).

- T1 existing non-temporal behavior unchanged: analyses
  without temporal seed keep current windows, scope,
  replay domains, and source-ref output (pin with golden
  equivalence on existing cases).
- T2 transition-event derivation: generic logged state
  transition + intent → deterministic matched event
  (timestamp, from/to values).
- T3 zero matching events → no diagnostic window;
  unresolved surfaced; no selection claim.
- T4 ambiguous multiple events without selection →
  fail closed (no window); with explicit first/last →
  deterministic choice.
- T5 transition-relative diagnostic window: matched event +
  relation/duration → expected `EvaluationScope.windows`.
- T6 first-sample-after-event: generic target signal →
  first sample at/after event timestamp; degenerate
  point-window overlap valid for selection.
- T7 temporal eligibility: overlapping domain → eligible;
  disjoint domain → excluded from temporal eligibility.
- T8 overlap not unique: two overlapping writers → both
  retained + unresolved surfaced.
- T9 unique domain discrimination: exactly one overlapping
  writer → uniquely selected.
- T10 optional replay discrimination: where existing
  per-writer replay results distinguish eligible writers,
  the supported writer may be selected; replay semantics
  unchanged. If not cleanly available, omit this test
  rather than manufacturing it.
- T11 no source-order tiebreak: input permutation → same
  eligible/selected result.
- T12 helper/source identity preservation: temporal
  filtering leaves all WS1/WS2 identities unchanged.
- T13 proof/status isolation: branches/proof/confidence
  suites unchanged by temporal eligibility alone.
- T14 TECS event/window derivation on the real log →
  correct transition-relative diagnostic window.
- T15 TECS early transient: first post-transition relevant
  sample ≈ accepted transient value (per sidecar).
- T16 TECS later behavior: later setpoint approaches
  accepted ~5 m/s behavior (per sidecar).
- T17 TECS rival exclusion: early-vs-later temporal
  evidence rejects the persistent-limit diagnosis.
- T18 TECS source evidence: refs grounded in represented
  computations available to the pipeline; no fabricated
  refs (missing-fixture and snapshot pins per RTL A1
  pattern).
- T19 TECS benchmark strength: sidecar stays PROVEN.
- T20 full isolation regression: replay/checkpoint/
  coverage/applicability/proof suites unchanged.

## 46. Stop gates

- STOP A: no minimal generic transition-event intent can be
  added without TECS-specific production policy.
- STOP B: logged timeline data cannot deterministically
  identify the required TECS transition.
- STOP C: first target-signal sample after the transition
  cannot be derived from available logged timestamps.
- STOP D: `EvaluationScope` cannot carry the derived window
  without schema redesign.
- STOP E: scope cannot reach temporal selection without
  report-layer intent parsing.
- STOP F: represented TECS writer candidates carry no
  temporal/domain information capable of narrowing them.
- STOP G: after eligibility, multiple TECS candidates remain
  and no existing writer-domain/replay/ordering evidence
  distinguishes them (§25 governs the report).
- STOP H: unique selection would require source-order/
  insertion-order tiebreaking.
- STOP I: a new persistent temporal evidence/phase graph is
  unavoidable (§44 governs the report).
- STOP J: proof-authority or confidence-semantics changes
  are required.

On any stop gate: do not widen scope; recommend a narrowly
targeted `$wayfinder` for the exact missing relation.

## 47. Spec path

This file (`docs/temporal_causal_qualification_spec.md`) is
the single combined W3A+W3B contract, following the repo
precedent of pairing a production-evidence contract with its
driving acceptance case (cf. helper-source-evidence §16 RTL
target). No second spec is created.

## 48. Explicit exclusions

W3C Takeoff implementation; Takeoff fixtures, sidecars, and
acceptance tests. Airspeed representation or acceptance.
Logged-message event sources (no W3B need; revisit only if a
driving case requires them). New benchmark frameworks (extend
`tests/acceptance/` conventions). Numeric-linkage ownership.
Live-LLM behavior changes beyond minimal deterministic
seeder guidance. Performance work. Fixture provisioning
infrastructure beyond the TECS log/sidecar.

## 49. Deterministic acceptance conformance

Sidecar and tests follow `docs/deterministic_acceptance_spec.md`:
human-accepted benchmark semantics frozen; strength separate
from pipeline status (§5); future-improvement tolerance — more
evidence or higher confidence later must not break the golden
while the diagnosis stays compatible and no forbidden claim
fires; contradiction fails loudly with spec review. Logged
text unused for TECS; normalization follows the shared
unstable-fields list.
