# Temporal causal qualification ownership

Static source topology can show that an operation is reachable without
showing it is the causally relevant writer for the time window a
diagnostic question asks about (TECS restart transient versus steady
limit; takeoff climb phase versus resumed waypoint). We decide that
question intent specifies WHAT temporal event matters, deterministic
derivation maps that specification plus logged timeline observations
to concrete bounds, `EvaluationScope` carries the resulting bounds
downstream, replay keeps evaluating represented writers inside
whatever domain it is given, and a report-adjacent selector — not
the report builder, not replay, not proof — consumes scope plus
replay plus existing DAG selection to qualify which feasible writer
is relevant for that window. Temporal overlap establishes
eligibility; unique causal selection additionally requires evidence
that actually distinguishes one candidate.

## Definitions

- **Diagnostic window**: the time bounds within which the questioned
  behavior is evaluated. Stated directly or derived relative to a
  logged transition event; never invented from source order. One
  diagnostic scope may carry several ordered windows.
- **Transition-event specification**: the question-intent input saying
  WHAT temporal event a diagnostic window should be derived from
  (observe a transition in some logged signal/state, select the
  matching event, derive bounds relative to it). Semantic intent
  only; no schema is fixed here.
- **Transition event**: a logged value change (timeline signal event
  or logged message) marking a mode/phase boundary, e.g. a nav-state
  change or a takeoff/waypoint sequence advance.
- **Temporally eligible candidate**: a represented candidate whose
  feasible/active domain overlaps the diagnostic window. Eligibility
  means the candidate could be relevant during the questioned
  interval; it never by itself means the candidate uniquely caused
  the observed behavior.
- **Temporal qualification**: the decision that available evidence
  further establishes a temporally eligible candidate as the
  relevant causal source for the diagnostic claim. Qualification
  may remain non-unique: several candidates may stay eligible with
  no unique selection. It filters already-identified candidates; it
  never changes source identity, feasibility verdicts, or proof
  claims.
- **Writer domain**: the writer's existing replay-result
  `active_windows` restricted to the supplied diagnostic/comparison
  domain; descriptive terminology, not a new stored domain type.
- **Corroborating observation**: timeline events, logged text, and
  setpoint sequences that support a diagnosis without being source
  provenance and without carrying proof authority.

## Decision

- Question intent (the diagnostic seed) specifies WHAT temporal
  event a diagnostic window should be derived from. It never lives
  in `EvaluationScope`, the report builder, replay, or the
  source-ref selector; exact schema belongs to `$to-spec`.
- Deterministic window derivation maps that specification plus
  logged timeline/signal observations to concrete diagnostic
  window(s). This responsibility belongs to the existing
  questioned-condition / diagnostic-window derivation seam (the
  layer around `evaluate_questioned_condition_windows`), which
  `$to-spec` may extend or adjoin with a small pure derivation
  helper without changing architecture.
- `EvaluationScope` owns ONLY the derived result: concrete
  diagnostic/comparison window(s) plus the signal/reference/units
  /frame information already in scope. It is not an event
  detector, a transition interpreter, or a question-intent
  parser. Ownership chain: question intent says WHAT temporal
  event matters; timeline/logs say WHAT happened and WHEN;
  window derivation deterministically maps those inputs to
  bounds; `EvaluationScope` carries the resulting bounds
  downstream. The derived scope must be available to the
  temporal-selection boundary; exact threading belongs to
  `$to-spec`.
- Timeline/inventory own raw temporal events (signal value-change
  events, logged messages). Transition-relative windows are derived
  deterministically from those logged events, never from source
  text, and never by the report builder. Case-specific transition
  semantics (which signal, which values, which message) travel
  with question intent and acceptance sidecars, never as
  production branching: production operates on generic
  transition/event specifications over logged observations.
- Replay keeps evaluating represented writer values over the
  supplied temporal domain and reporting per-writer domains with
  match evidence. It never decides diagnostic meaning, never
  selects prose source refs, and never declares causal sufficiency.
- Temporal qualification lives in a report-adjacent causal
  selector that consumes scope windows plus replay writer domains
  plus the existing WS1/WS2 DAG selection, and narrows to the
  writers relevant for the diagnostic window. The report builder
  formats already-qualified evidence only.
- Composition rule: data edges identify value-producing
  candidates; control/branch state (feasibility plus
  `active_windows`) identifies feasibility context; temporal
  windows identify which feasible writer is relevant. Control
  ancestors never become source evidence by themselves.
  `active_windows` and branch feasibility may support temporal
  eligibility and selection; they never by themselves establish
  unique causality, proof authority, or diagnostic strength.
  `evaluation_domain` records only the span examined by
  evaluation/value gating and is not authoritative for
  diagnostic relevance.
- Source selection is two-level. Level 1, temporal eligibility: a
  represented source candidate whose feasible/active domain
  overlaps the diagnostic scope becomes temporally eligible.
  Level 2, causal selection: unique selection occurs only when
  available existing evidence actually distinguishes one
  candidate — a sole overlapping writer domain, domain separation
  across ordered windows, in-window replay support for one
  candidate over rivals, or observable temporal ordering
  excluding rivals. Replay support corroborates but is never
  mandatory for eligibility, so WS3 never depends on
  numeric-check ↔ source-operation linkage ownership.
- Multi-writer fail-closed rule: if several writers overlap the
  diagnostic window and current evidence does not distinguish
  them, retain all as temporally eligible, pick none by source
  order, claim no unique causal selection, and surface the
  unresolved/discriminating evidence through existing report
  paths. Missing unique discrimination is not contradiction.
  Transition-relative overlap alone does not solve
  shared-cone cases (e.g. TECS initialize versus update writers
  eligible in the same window): the window narrows the evidence
  domain, writer-domain separation, replay, or ordering may
  further discriminate, and a remaining tie stays retained and
  unresolved rather than overclaimed. The same rule governs
  Takeoff: writers relevant in earlier versus later windows may
  separate uniquely, and indistinguishable candidates inside one
  window are retained, never ordered by source position.
- Temporal non-uniqueness is evaluated at the level of the claim
  being made. If multiple temporally eligible writers imply
  materially different diagnostic mechanisms, lack of a
  discriminator remains fail-closed: retain the candidates, make
  no unique mechanism claim, and surface unresolved evidence.
  If multiple eligible writers are execution-distinct but
  mechanism-equivalent for the questioned diagnostic claim, the
  system may retain all such writers without asserting which one
  executed and may proceed with the shared mechanism-level
  conclusion when that conclusion is independently supported.
  Writer non-uniqueness must be recorded honestly and must never
  be converted into an execution claim. Mechanism equivalence
  exists only when, for the actual questioned claim, every
  retained candidate supports the same material causal mechanism
  and no accepted conclusion depends on distinguishing which
  candidate executed; identical source text, destination symbol,
  proximity, or lifecycle variant alone never establish it.
- Unresolved writer non-uniqueness surfaces through existing
  report paths when a report hypothesis exists. When no report
  hypothesis is produced, acceptance/verification may record the
  unavailable exact-writer attribution as NON_DECISIVE evidence
  without fabricating a report claim. This is acceptance-side
  metadata only; no new production severity enum is introduced.
- Temporal qualification may FILTER an already-qualified
  helper/source candidate by window. It is orthogonal to
  identity: WS1/WS2/ADR-0003 identities are consumed unchanged.
- One source-ref normalization pipeline is kept: the qualifier
  narrows the candidate set, and the existing WS1/WS2 pipeline
  renders it with unchanged ordering, dedup, and budget rules.
- Missing-window behavior fails closed: with no transition event,
  no window, or no writer domain where discrimination is needed,
  claim no temporally selected causality, retain the broader
  plausible evidence, and mark the gap unresolved (acceptance-side
  `DISCRIMINATING`/`CRITICAL` classification). Never pick a
  writer by source order.
- A real temporal contradiction is ordering/domain conflict
  (transient predicted before its transition, writer active only
  outside the diagnostic window, observed setpoint ordering
  reversing the claimed mechanism, replay mismatch inside the
  qualified domain). Missing internal state alone is not
  contradiction.
- Logged text (e.g. takeoff/Navigator messages) is corroborating
  observation evidence only: never a source `CodeRef`, never
  proof authority by itself.
- Strength, confidence, and proof are untouched: temporal
  qualification never sets confidence/confirmed/PROVEN by itself
  (effects flow only through existing evidence/replay/branch
  rules); TECS and Takeoff stay PROVEN benchmarks; no benchmark
  is relabeled to cover BEST_SUPPORTED; T3/T5/T6B/Gate-B/R1,
  coverage, applicability, checkpoint, proof stores, and
  `branches_verified` (readable, never writable) are unaffected.
- No new persistent graph, view, or evidence types: temporal
  selection composes `EvaluationScope`, timeline events,
  `active_windows`, replay results, and the existing DAG slice.
  A pure qualification function is allowed; a stored model is
  rejected.
- Acceptance sequencing: TECS is the first WS3 acceptance case,
  Takeoff the second, both on this one temporal architecture.
  Case-specific semantics (transition signals, phase expectations,
  numbers, text corroboration) live in acceptance sidecars, never
  as production branching. `EvaluationScope.windows` already
  carries several ordered windows, so Takeoff-style ordered
  observations need no persistent phase model.

## Considered Options

- Report builder decides temporal relevance: rejected. Formatting
  must not infer first-sample-after-transition, phase changes, or
  timing winners; those decisions must originate upstream where
  windows and replay evidence live.
- Replay decides causal meaning: rejected. Replay's authority is
  value comparison inside a given domain; promoting it to
  sufficiency/meaning would merge evaluation with policy against
  ADR-0001/ADR-0002.
- New TemporalEvidence graph/model: rejected. `EvaluationScope`,
  timeline events, `active_windows`, replay results, and the DAG
  slice already partition the concern; a new stored type adds
  ownership without a consumer beyond selection.
- Case-specific TECS/Takeoff production rules: rejected. Per-case
  branching violates the generic-code rule and would need manual
  upkeep per mechanism; benchmarks carry case semantics in
  sidecars while production operates on events, windows,
  writers, and transitions. Overlap alone is eligibility rather
  than unique selection for the same reason: hardcoding which
  rival wins would be case logic in disguise.

## Consequences

- WS3 decomposes into one shared production capability (window
  derivation plus temporal qualification plus selection
  interaction, with isolation pins) and two acceptance slices
  (TECS, then Takeoff); no production mechanism per benchmark.
  The shared capability is implemented jointly with, and driven
  by, TECS acceptance — never as unused generic machinery in
  isolation — and Takeoff then reuses it.
- `branches_verified`, confidence, confirmation, replay
  completeness, `match_fraction`, and verification status keep
  their existing producers and meanings; temporal evidence is
  diagnostic selection, not proof authority.
- BEST_SUPPORTED acceptance coverage stays absent until a
  natural case needs it; Airspeed remains deferred on its
  representation gap.

## Status

Proposed as the ownership contract for WS3 temporal work.
Implementation follows separately (`$to-spec`); Airspeed state
representation stays out of scope.
