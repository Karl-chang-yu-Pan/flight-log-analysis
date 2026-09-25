# Bounded discovery frontier spec

Implementation-ready specification for narrowing the LLM discovery
role: deterministic code presents only the information required for
the next semantic discovery decision, while preserving everything
that can materially change that decision. First implementation
target is the legacy PX4 Source Mechanism Discovery Decision input;
the design is reusable by the DAG judge later without requiring it.

Status: specification only. No implementation, no tests, no commits.

## 1. Problem statement

The paid TECS calibration measured two discovery-decision requests
of 186,947 and 190,980 input tokens (~$2.00 of the $2.13 total),
with ~97% of the accumulated structured corpus repeated verbatim
between them. Dominant categories were `source_assignments`
(~254K chars), `helper_expressions` (~174K),
`parameter_requirements` (292 items, ~117K), `function_calls`
(~71K), and verification candidates (~42K); raw source snippets
were ~2%. These structures are already deterministically
collected, yet every Runner call is stateless, so each decision
re-reads the entire corpus to decide anything at all.

## 2. Measured evidence

- Paid run (TECS fixture, `uploads/tecs-switch-moded-hrate/`):
  intent 685 in/949 out ($0.032); four discovery decisions at
  4,904 / 186,947 / 5,323 / 190,980 in (~$0.04 / $0.97 / $0.06 /
  $1.03); final report never reached; DAG never started;
  cumulative $2.130105 triggered the hard guard. No cached-token
  discount observed on any call; max input 190,980 tokens (no
  272K breach).
- Overlap: per-category item-level intersection of the two
  expensive packets is ~100% (assignments 60/60, branches
  80/80, EVC 40/40, calls 80/80, helpers 40/40, requirements
  292/292); only related-files, new-file snippets, prior notes,
  and visited lists differ (~2–3% new content).
- Historical bounds: judge inputs up to 114K chars observed;
  render safety valves at 4000 ops/branches/evidence/unresolved
  entries each; discovery capped at 18 total files, 3 profiled
  per iteration, 4 snippets × 12KB per file.
- All five agents declare `tools=[]`; every audited invocation
  shows exactly 1 provider request; no reasoning effort,
  temperature, or model settings are configured anywhere
  (provider/model defaults apply); no prompt caching is
  configured (one 1,792-token cached case observed, otherwise
  zero — assume no discount).

## 3. Current architecture

Legacy discovery (`SourceMechanismResolver.discover`,
`flight_log_agent/px4/source_mechanism_resolver.py`) loops to
`max_depth=2`, issuing up to two `decide` calls per depth
(search-hits packet, then expansion packet with the FULL
accumulated corpus). Packets are built by
`_build_search_iteration_packet` /
`_build_iteration_packet`; snippets by
`_source_snippets_for_files` (4 per file, 12KB each, 180
lines, 12 context lines). Parameters flow through
`parameter_gate.evaluate(dedupe_parameter_predicates(...))`
with NO dedup or bound on the resulting requirements list
(an existing `dedupe_parameter_requirements` helper exists
but is not called on this path). Loop exit is LLM-advised
(`decision.stop`) with deterministic exhaustion fallbacks
(no queries, file cap reached). T6B stop authority
(`legacy_verified` AND coverage AND applicability AND
non-vacuous proof AND Gate-B discharge, in
`flight_log_agent/analysis/checkpoint_discovery.py`) is
downstream and untouched by this work. DAG seeder/judge and
the report writer are separate roles, out of first scope.

## 4. Desired LLM responsibility

LLMs resolve semantic ambiguity over a bounded unresolved
frontier. LLMs do not act as storage, transport, or repeated
readers of the entire accumulated deterministic evidence
state. Concretely after this work: the discovery LLM sees
new/changed/ambiguous material plus a compact deterministic
carry-forward summary, never the full retransmitted corpus.
Authority stays where it is: deterministic evidence and
proof machinery decide acceptance; LLM output remains
advisory to traversal.

## 5. Scope

First and only implementation target: legacy discovery
`decide` input construction (packet builder, requirement
aggregation, novelty/dedup, mass instrumentation, P0 gate
integration). Everything else in this spec is either a
reusable abstraction definition or an explicitly deferred
direction.

## 6. Non-goals

P1 routing/default flip; DAG-primary flip; DAG judge
migration (reusable abstractions only); report-writer
refactor (direction recorded in §13); persistent LLM
conversation state (`conversation_id`, `previous_response_id`,
hidden model memory); new proof authority; T6B changes;
replay changes; Gate-B changes; benchmark conclusion
changes; model changes; prompt-model tuning; paid benchmark
execution; telemetry/persistence architecture; cache-layer
design.

## 7. Terminology

- **Discovery frontier**: the bounded set of evidence items
  requiring a semantic decision in the current round:
  new/changed candidates, edges, requirements, gates, and
  explicitly unresolved questions. _Avoid_: full packet,
  accumulated corpus.
- **Carry-forward summary**: deterministic, compact
  restatement of previously observed evidence sufficient to
  interpret the frontier without retransmission. Produced by
  deterministic code, consumed by the LLM, never owned as
  persisted memory by the model. _Avoid_: conversation
  history, cached context.
- **Novelty**: the property of an evidence item of not
  having appeared, in identity terms, in any prior round
  of the same discovery run.
- **Evidence lifecycle state**: ACTIVE (in contention now),
  RESOLVED (decided, still constraining), SUPERSEDED (replaced
  by newer evidence, retained by reference), CONTRADICTED
  (refuted, visible as refutation), RETAINED-SUMMARY
  (folded into the carry-forward summary with its identity
  preserved). IRRELEVANT items are dropped with a count,
  never silently. _Avoid_: visited/resolved/old as
  deletion synonyms.
- **Packet mass**: serialized size of a decide input in
  bytes/chars, total and per section, measured before any
  model call. No tokenizer dependency.

## 8. DiscoveryFrontier model

A per-round structure containing ONLY:

- `new_candidates`: source candidates first seen this
  round. Identity: declaration-derived source identity
  (file, callable, scope, symbol) where available, else
  the existing candidate key. Dedup: exact identity match
  against all prior rounds. Order: deterministic
  (discovery order, then identity sort). Delta: always
  current-delta. Needed because: the relevance judgment
  is the core semantic decision.
- `changed_assignments`, `changed_calls`,
  `changed_branches`: items whose content changed since
  last sent (by stable identity + content hash), else
  omitted. Identity: existing `visit_key` /
  declaration-derived identities. Dedup: identity match;
  ordering: stable identity sort. Delta. Needed because:
  changed edges alter mechanism hypotheses.
- `new_helper_relationships`: helper edges/refs first seen
  this round (same identity rules as calls). Delta.
- `new_requirements`: parameter requirements first seen
  this round, grouped per §12 (never blindly truncated).
  Identity: (name, source_predicate, source_file,
  source_line) plus retained gate/role variants (§12).
  Delta.
- `new_verification_candidates`: first-seen verification
  candidates. Delta.
- `unresolved_questions`: open items whose full text lives
  in the canonical deterministic store under a stable
  question identity; the carry-forward representation holds
  the identity plus state/counter, change flag/revision,
  and bounded decision-critical metadata — never bare
  counters alone. Full text is prefetched whenever an
  open question is decision-relevant this round (per the
  §27 relevance closure); it is never re-derived by the
  LLM from absence.
- `contradictions`: refuted claims with their refutation,
  always visible while in contention. Delta on change.
- `work_state`: deterministic termination facts — queries
  remaining, files remaining under cap, depth remaining.
  Current-delta. Needed because: the LLM's stop advice
  must see the same budget the loop enforces.

Excluded by construction: unchanged assignments, calls,
helpers, branches, predicates, fields, topics, requirements
already sent (they live in the summary by identity);
visited-file full listings beyond counts + new names;
repeated log-context blobs (vehicle/mode/params deltas
only).

## 9. CarryForwardSummary model

Deterministically produced per round from full resolver
state; consumed, never modified, by the LLM:

- `coverage_map`: per searched domain/file: files seen,
  ref counts by category, requirement counts by gate
  outcome. Source: accumulated lists + gate results.
  Lossless as counts; lossy as content (content lives
  behind identities). Stable: monotonic counters.
- `resolved_claims`: decided items with outcome
  (admitted/rejected + reason class). Source: admission
  verdicts + gate results. Lossless at verdict
  granularity. Stable: append-only.
- `open_claims`: unresolved items by identity with age
  (rounds waiting). Source: frontier leftovers +
  requirements with `verification_required` outcomes.
- `contradiction_ledger`: refuted claims and their
  refutations. Never dropped while the refuted claim is
  referenced. Each entry preserves `window_identity`
  wherever temporal qualification is part of the evidence
  identity or state, so the same source/writer under a
  different window never collides with an earlier
  refutation.
- `candidate_standings`: mechanisms in contention with
  supporting/pending evidence counts (not prose).
- `gate_tally`: parameter-gate outcome histogram
  (satisfied / contradicted / verification_required /
  unknown). This is what makes the 292-item list
  compressible without loss of control information.
- Conflicting evidence is represented as explicit
  contradiction entries, never averaged or merged away.
  The LLM may not modify the summary; there is no
  conversation back-channel (no `conversation_id`,
  no `previous_response_id`, no hidden memory).

## 10. Evidence lifecycle

Every evidence item carries a lifecycle state
(ACTIVE, RESOLVED, SUPERSEDED, CONTRADICTED,
RETAINED-SUMMARY; counts for IRRELEVANT). Transition rules
are deterministic and owned by the resolver:

- Newly extracted → ACTIVE (enters frontier once).
- Admitted/rejected with reason → RESOLVED (leaves
  frontier; constrains via summary + verdict records).
- Replaced by newer extraction → SUPERSEDED (identity
  retained, content retrievable only through the §27
  retrieval protocol, never retransmitted by default).
- Refuted → CONTRADICTED (stays visible while referenced).
- Unchanged across N rounds with no open question
  attached → RETAINED-SUMMARY (identity + verdict in
  summary).
- A RESOLVED fact still constrains later hypotheses
  through verdict records and gate tallies; resolution
  never equals deletion. Round-trip invariant: the union
  of frontier + summary + verdict records covers exactly
  the evidence set of the full packet (asserted by the
  offline harness, §19).

### 10A. Reactivation (deterministic, no LLM memory needed)

The W→H1→H2 counterexample must be impossible: a
RESOLVED, SUPERSEDED, or IRRELEVANT item re-enters the
frontier as ACTIVE whenever a deterministic trigger fires:

- a new hypothesis references the same evidence identity;
- a new candidate shares writer, source, call, parameter,
  or gate relationship with retained evidence (via
  `visit_key` / declaration-derived identities);
- a new contradiction touches retained evidence;
- a new branch/gate makes prior evidence discriminating
  (parameter-gate outcome change on an overlapping site);
- a new parameter requirement intersects retained
  evidence by identity;
- a new temporal window changes applicability of
  retained evidence;
- a new source relationship connects a retained node to
  the active frontier (call/assignment edge either
  direction).

Triggers are evaluated deterministically per round
BEFORE frontier construction, so reactivation needs no
model recall. A reactivated item carries its full prior
verdict history with it (admitted/rejected + reason),
never as a blank new item.

### 10B. Irrelevance ownership

The LLM may PROPOSE semantic irrelevance in its output,
but that proposal is advisory only. Deterministic code
classifies evidence as inactive only under ALL of these
explicit conditions: SUPERSEDED by a newer extraction of
the same identity, unreferenced by every open claim,
candidate, contradiction, and requirement (per §10C),
and unchanged for at least one full round. There is NO
safe irreversible deletion in this model: IRRELEVANT
means "dropped from this packet with a count, retained
in the canonical store under identity, retrievable via
§27." If these conditions cannot be met, the item stays
RETAINED-SUMMARY, never IRRELEVANT.

### 10C. Referenced semantics

`referenced` is machine-checkable identity intersection,
never vague relevance prose. Evidence identity X counts
as referenced iff ANY holds: an active hypothesis lists
X in its support set; an open claim depends on X; a
rival exclusion cites X; a contradiction is supported by
X; a parameter requirement points to X's source or gate
identity. Contradicted evidence satisfying ANY clause
stays accessible for rival exclusion, mechanism
disqualification, future comparison, and explanation —
it is never discarded merely because its candidate
stopped leading.

### 10D. Temporal and change provenance

Every carried item records `first_seen_round`,
`last_changed_round`, and a value-revision counter;
requirements additionally record gate-outcome history;
window-sensitive facts record `window_identity`.
Round/version identity suffices — no wall-clock
invention. These fields distinguish same-identity
changed values, same requirement new rounds, and same
writer under new questioned windows. Open claims and
parameter requirements preserve variant-level identity
(name + predicate + site + gate + value revision),
never parameter-name-only identity: a changed actual
value or gate outcome is a distinct decision-relevant
variant by construction (§12). Open claims use the same
variant contract as requirements: an open claim whose
underlying variant changed is a new open-claim version,
not a counter increment on the old one.

`value_revision` bump rule (deterministic, no LLM
involvement): increment by exactly one when, and only
when, a new extraction round observes a difference in
`actual_value`, gate outcome, semantic effect, or
window-specific applicability for the same identity.
Serialization order, whitespace, counter values, and
round numbers never bump the revision. Two extractions
with equal identity and equal revision are
interchangeable by construction.

## 11. Identity/dedup rules

Reuse the strongest existing identities; do not invent
weaker text-only ones:

- Source references: `visit_key` (kind, exact symbol,
  file, callable, owner, receiver, resolved callable,
  argument count, source site).
- Assignments/calls/helpers/branches/fields/topics:
  the existing `dedupe_*` family in
  `source_mechanism_resolver.py` (assignment, call,
  helper-expression, branch-condition, field, topic,
  predicate variants).
- Requirements: (name, source_predicate, source_file,
  source_line) per existing
  `dedupe_parameter_requirements` — extended ONLY by
  retaining distinct gate/role variants explicitly (see
  §12), never by weakening the key to bare names.
- Files/sites: path + stable ranges (`dedupe_line_ranges`
  exists).
- Branch conditions: source file + line +
  `source_site_id` + `source_order` + normalized predicate
  (`condition_ref` where available), per the
  `BranchConditionRef` model. Callable/scope ownership is
  derived deterministically from the source-site context
  (enclosing scope lookup by file + line), not a stored
  field — the spec records the derivation rule, not a
  claim that the model stores it. Text-only, file-only,
  or symbol-name-only branch identity is rejected: two
  branch arms, two call sites, or two gate predicates at
  nearby lines must never collapse.
- Ordering everywhere: deterministic (stable identity
  sort after discovery order) so repeated runs diff
  cleanly.

## 12. Parameter-requirement representation

Measured state: 292 items / 82 names; roles threshold-160,
unknown-62, tuning/shaping-61, branch_selector-9; 283
`verification_required`, 7 unknown, 1 satisfied,
1 contradicted; ~438 chars each.

1. Stable identity: (name, source_predicate, source_file,
   source_line) as today.
2. Distinctness: items differing in role, gate_result,
   actual_value, or effect are semantically distinct
   evaluations, not duplicates; the 292 are all distinct
   as JSON.
3. Grouping: group under (name) for display/counting ONLY,
   retaining per-item gate/role/site variants explicitly
   (grouped list, never collapsed scalar). The existing
   `dedupe_parameter_requirements` key is adopted for
   identity, with variant retention added.
4. Summarization: per-parameter rollups (counts by gate
   outcome + site list) may stand in for full re-listing
   across rounds; full variants re-emitted on change.
5. Outcomes: satisfied / contradicted / verification_
   required / unknown travel explicitly; contradicted
   items route to the contradiction ledger (§9).
6. Paging: if a grouped representation still exceeds the
   mass budget (§17), page by parameter-name shards with
   explicit continuation markers, never silent truncation.
   Paging decision rule (fail-safe): a decision depending
   on a paged section is complete ONLY after every
   relevant page has been represented or deterministically
   classified as not decision-relevant. From partial pages
   only existential claims are permitted ("at least one
   X exists"); universal claims ("no X exists") require
   all pages. A paged section whose pages were all seen
   behaves exactly like an unpaged section for §13.
7. Over-limit behavior: FRONTIER_TOO_LARGE unavailable
   result per §16 (never blind truncation).

## 13. Information-preservation contract

Full packet ≡ summary + frontier for discovery decisions
iff ALL hold (acceptance criteria, offline-testable):

- Same unresolved claims represented (unresolved set
  equality by identity, at requirement-variant
  granularity per §10D/§12).
- Same surviving candidate mechanisms representable
  (candidate identity set equality).
- Same contradictions visible (refuted-claim set
  equality, with §10C reference scope).
- Same outstanding evidence requirements visible
  (open-requirement set equality at variant granularity).
- Same source identities/references available when
  relevant (identity-keyed lookup succeeds for every
  referenced item, via the §27 protocol — never by
  assumption).
- Same branch/gate facts available (gate tally +
  variant lists equal).
- Reactivation completeness: every item meeting a §10A
  trigger in the round is ACTIVE in the frontier (the
  round-trip invariant covers presence; this covers
  return).
- No omitted evidence could change the decision:
  operationalized as — every item excluded from the
  frontier is either RESOLVED with a recorded verdict,
  SUPERSEDED with identity retained, or IRRELEVANT with
  a counted drop AND no §10A trigger firing; the harness
  asserts these partitions cover the full evidence set
  exactly.

Explicitly NOT required: same prose, same output text,
same candidate order (order is not contractual).

Decision-relative representation: the bounded input is
constructed with respect to the active hypotheses, open
claims, traversal state, new evidence, and current
question/window — never as one universal lossy summary.
Evidence safely summarized for H1 MUST re-enter the
frontier for H2 when §10A triggers fire; the contract
above is evaluated per decision round, not once globally.

Canonical store statement: the CarryForwardSummary is NOT
the evidence store. The deterministic full evidence
store (accumulated resolver state) remains canonical.
The summary is a decision-oriented projection of it.
Lifecycle and relevance changes may cause canonical
evidence to re-enter the frontier at any round.

Central acceptance property: for every semantic decision
D, if two full evidence states E1 and E2 would
legitimately permit different outcomes for D, then their
bounded representations B(E1,D) and B(E2,D) must differ
in some decision-relevant field, OR deterministic
prefetch/lookup must retrieve the distinguishing
evidence before D is finalized.

## 14. Discovery-decision input contract

Per round the LLM receives: question + intent (unchanged,
small) + work_state + frontier (§8) + carry-forward
summary (§9) + bounded log-context deltas. It MUST NOT
receive: retransmitted unchanged corpus, full visited-file
contents, repeated parameter blobs, or anything derivable
from the summary without loss. Estimated steady-state
shape from measured data (NON-NORMATIVE ESTIMATE, not an
acceptance threshold): summary ~15–25KB + frontier
(new files ≤3 × capped snippets + new refs), versus
~680KB full packets today. Implementation GREEN depends
only on direction (frontier + summary < full packet with
§13 invariants green), never on these illustrative sizes.

## 15. Discovery-decision output contract

The LLM returns advisory-only fields: semantic priorities
among frontier candidates, relevance judgments, unresolved
ambiguity restated, next semantic question, and a
need-more-evidence recommendation with reasons. Deterministic
code validates: priorities reference known identities;
relevance never creates writers, coverage, applicability,
or stop authority; stop advice is consumed as a hint
alongside deterministic exhaustion (queries/files/depth
remaining), never as authority. Ignored-for-authority:
everything the LLM emits about sufficiency, proof, or
stopping.

Output authority classification (every field):

- candidate relevance, semantic priority, suggested
  query/file, need-more-evidence, discovery-stop
  suggestion, lookup request (if retained per §27),
  hypothesis update → ADVISORY.
- Relevant-files/expansion-queries as emitted by the
  LLM → ADVISORY and validated: only known identities
  survive; see traversal ownership below.
- Nothing in this contract is AUTHORITY. There is no
  field whose content can create writers, coverage,
  applicability, stop, proof, replay, or confidence
  outcomes.

Traversal ownership (deterministic, existing machinery):
eligible expansion targets, visited handling, candidate
eligibility, and search exhaustion are owned by
deterministic code — specifically the existing
`_build_expansion_queries` owner and the resolver loop —
not by LLM output. Control flow per round:

```text
deterministic traversal state
        ↓
_build_expansion_queries (eligible expansion candidates)
        ↓
LLM semantic prioritization over the eligible set if needed
        ↓
deterministic validation (known identities only)
        ↓
actual expansion set
```

LLM-suggested files/queries beyond the eligible set are
advisory input to a future round's deterministic
eligibility check, never direct traversal commands.
`decision.stop` remains a hint consumed conjunctively
with deterministic exhaustion: the loop breaks on
exhaustion regardless of the hint, and the hint alone
never breaks a loop with remaining eligible work —
preserving exhaustion ≠ coverage ≠ applicability ≠ stop
(which T6B owns downstream).

## 16. Failure/fallback behavior

- Frontier exceeding its mass budget → deterministic
  paging first; if still over budget →
  FRONTIER_TOO_LARGE unavailable result (fail-closed,
  counted, never silent truncation).
- Summary construction failure → fail-closed: fall back
  to the current full packet for that round ONLY,
  observably counted as unoptimized, never interpreted
  as optimized success.
- Any invariant violation detected at runtime →
  fail-closed to current behavior for that round, counted.
- Blind truncation is forbidden in all paths.
- Fallback accounting fields (recorded per round,
  surfaced to P0 mass measurement):
  `full_packet_fallback_used` (bool),
  `full_packet_fallback_reason` (summary-failure |
  invariant-violation | frontier-over-budget),
  `full_packet_fallback_count` (monotonic per run),
  `full_packet_fallback_bytes` (sent size that round).
  A fallback run MUST NOT count as bounded-frontier
  success in any mass or readiness aggregation.

## 17. Packet-mass instrumentation

Deterministic, pre-call measurements (no tokenizer
needed): total serialized bytes/chars; per-section
bytes/chars and item counts; new-item counts;
summary size; frontier size; retransmitted-equivalent
size avoided (full-packet reconstruction size minus sent
size, computed from retained state). Prefer measuring
from the actual serialized payload object. These fields
feed P0 readiness measurement (mass gate: measurement
required, threshold unratified — record into the
readiness artifact shape, never gate GREEN on values).

## 18. P0 readiness integration

Packet-mass fields plug into existing P0 measurement
helpers (per-fixture/per-path durations + the readiness
JSON `version/status/thresholds/basis` contract, thresholds
remaining null until ratified). No P0 test changes are
required by this spec except additive mass assertions;
existing T1–T21 and bake-off gating are untouched.

## 19. Offline acceptance harness

A deterministic harness (tests/, no models) that takes a
recorded full discovery state — primarily the stored TECS
paid-run packet artifacts — and constructs old packet,
summary, and frontier, asserting the §13 contract
dimensions: unresolved/candidate/contradiction/
requirement/identity/gate equality plus exact partition
coverage. Required suites: preservation invariants on
TECS recorded states; construction determinism
(byte-identical repeated builds); stable ordering;
mass-metric sanity (frontier + summary < full packet
with all contract dimensions green); genericity (no
benchmark/module/commit-specific branches — enforced by
existing case-term guard patterns plus a spec-named test
that fails on per-case production literals); reactivation
(resolved evidence becomes ACTIVE for a later rival);
changed-value variants (same identity, new value/gate
stays distinguishable); deterministic prefetch (retained
evidence required by an active claim re-enters before
judgment); lookup budget enforcement (lookup rounds,
identities, and bytes stay within declared caps);
traversal ownership (LLM-suggested targets pass through
deterministic eligibility; exhaustion behavior
unchanged); stateless equivalence (each round's bounded
input decides without prior conversation state).

## 20. Benchmark acceptance

Offline structural acceptance per benchmark (RTL, TECS,
Takeoff, Airspeed), no model calls: evidence-set,
unresolved-claim, contradiction, and candidate
preservation from recorded or deterministically
reconstructed states; mass measurements; no
benchmark-specific production branches. Paid semantic
acceptance (LLM decision compatibility, benchmark
semantic compatibility, actual token/cost reduction)
requires separate explicit user approval and is NOT part
of implementation GREEN.

### 20A. Must-survive evidence (test obligations, never
production special cases)

These benchmark-specific must-survive items are derived
acceptance obligations from each benchmark's governing
specification. They do not redefine benchmark truth. If
this document and the governing benchmark specification
ever disagree about the benchmark's accepted semantics,
the governing benchmark specification is authoritative.
This section only identifies which already-governed
evidence must remain representable after
bounded-frontier transformation. No benchmark semantics
are duplicated here and no accepted benchmark conclusion
is changed.

Governing specifications (all committed):

- RTL: `docs/deterministic_acceptance_spec.md` (shared
  framework; RTL is the first PROVEN case; its §3
  benchmark ground-truth policy owns the accepted
  semantics).
- TECS: `docs/deterministic_acceptance_spec.md` (§16
  inventory: PROVEN — transition timing, first-sample
  equation, multi-sample RMSE, counterfactual
  exclusion).
- Takeoff:
  `docs/takeoff_temporal_acceptance_spec.md` (accepted
  Takeoff benchmark), under the conventions of
  `docs/deterministic_acceptance_spec.md`.
- Airspeed:
  `docs/airspeed_load_factor_acceptance_spec.md`
  (accepted Airspeed benchmark), under the conventions
  of `docs/deterministic_acceptance_spec.md`.

Must-survive items (unchanged):

- RTL: cone/floor relationship (destination + 2×
  acceptance radius floor, winning altitude relation)
  and the helper-selection evidence supporting it.
- TECS: restart/transient relation, the 0.3 × reference
  rate relationship, and relevant writer/mechanism
  identities.
- Takeoff: MIS_TAKEOFF_ALT minimum target, initial
  home +20 relationship, later retained/resumed ~+7m
  mission target, and ordering/window distinction.
- Airspeed: load_factor = 1/cos(setpoint roll),
  minimum airspeed × sqrt(load factor), adapted-minimum
  binding versus trim/max, and the setpoint-roll vs
  measured-roll distinction.

## 21. Later paid acceptance

Only after offline GREEN plus explicit approval: rerun
bounded scenarios asserting decision compatibility with
recorded full-packet decisions where those exist, plus
measured token/cost deltas. Thresholds stay unratified
until measured. Paid acceptance never gates implementation
GREEN.

## 22. Migration plan

Phase 1 (this spec): legacy decide-input narrowing +
instrumentation + offline harness + P0 mass fields.
Phase 2 (separate): paid decision-compatibility sampling.
Phase 3 (separate): DAG judge input narrowing ONLY IF
code inspection proves shared abstraction (do not assume);
otherwise a second bounded spec. Report-writer work is
not in any phase (direction only, §13 of tasking). P1
routing/flip work is untouched.

## 23. Risks

- Over-narrowing hides needed context → mitigated by
  §13 contract tests + fail-closed fallback.
- Alias/dedup key collisions merge distinct evidence →
  mitigated by identity-strength rule (§11) + variant
  retention (§12).
- Summary staleness across rounds → mitigated by
  monotonic counters + change-triggered re-emission.
- LLM behavior shift on smaller inputs → mitigated by
  paid decision-compatibility sampling (§21), not assumed.
- Scope creep into authority/replay/proof → mitigated by
  frozen-architecture list + T3-style guard tests.

## 24. Open questions

- Exact mass-budget numbers (measure first; §17 before
  any threshold).
- Whether `decision.stop` advisory can eventually retire
  in favor of purely deterministic exhaustion + semantic
  need signaling (deferred; current advisory use stays).
- Paging shard-size tuning (deferred to implementation
  measurement).
- Lookup-budget tuning: max rounds, identities per
  lookup, and byte caps for Mode B (§27) — set from
  implementation measurement, unratified until then.
- DAG judge shared-abstraction proof (deferred to Phase 3
  inspection).

## 25. Test matrix

| Area | Offline deterministic | Paid (separately authorized) |
|---|---|---|
| Preservation contract (§13) | required GREEN | — |
| Construction determinism/ordering | required GREEN | — |
| Mass metrics + P0 fields | required GREEN | — |
| Genericity (no per-case literals) | required GREEN | — |
| Decision compatibility | — | required when authorized |
| Token/cost deltas | — | required when authorized |
| Benchmark conclusions | unchanged, structural | unchanged, semantic |

## 26. STOP conditions

STOP and document if: (1) preservation needs LLM hidden
state; (2) the summary itself needs an LLM to produce;
(3) dedup identities don't exist (they do — §11);
(4) narrowing removes evidence needed by accepted
benchmarks; (5) benchmark-specific production logic is
required; (6) the work cannot stay within discovery-input
responsibility without touching proof authority. Route
to `$wayfinder` for missing structure,
`$architecture-critic` for genuine architecture conflict.

## 27. Evidence retrieval protocol

Mode A — deterministic prefetch (the normal path).
Before frontier construction each round, deterministic
code computes a relevance closure over: active
candidates, open claims, new contradictions, new
branches, reactivated evidence (§10A), and unresolved
parameter requirements. Every retained item intersecting
that closure by identity is prefetched into the
frontier (or summary, per lifecycle state) BEFORE the
LLM is invoked. The LLM is never required to discover
that omitted evidence exists.

Mode B — bounded explicit lookup (exceptional, only for
semantic relationships deterministic code cannot know
in advance, e.g. an LLM-proposed hypothesis naming an
identity outside the closure). The LLM may request
 retained identities subject to: declared maximum
 lookup rounds per discovery round (default 1),
 declared maximum identities per lookup, declared
 maximum returned bytes/items, and a hard budget.
 Rationale for the default of 1: one follow-up lookup
 round exists for the narrow case where the initial
 semantic judgment exposes a need for already-known,
 identity-addressable canonical evidence that
 deterministic prefetch could not identify in advance.
 The default is 1 rather than 0 because that
 named-evidence case must remain representable. The
 default is 1 rather than >1 because repeated lookup
 rounds would recreate iterative model-driven search,
 weaken cost boundedness, and blur the boundary
 between retrieval and traversal. Changing this
 default requires explicit evidence and spec review;
 implementations must not silently raise it. Lookup
 is not traversal: lookup identities must already
 exist in the canonical evidence store, lookup does
 not expand visited/source traversal state, there is
 no free-form search and no hidden model memory, and
 lookup budget exhaustion fails closed.
Exceeding any bound yields LOOKUP_BUDGET_EXHAUSTED
(fail-closed, counted); unresolvable identities yield
LOOKUP_UNAVAILABLE; completed retrieval yields
LOOKUP_COMPLETE. Each lookup is a separate model
invocation and counts as one in packet-mass and cost
accounting. Lookup requests are ADVISORY: deterministic
code validates every returned identity against the
canonical store and drops unknown identities with a
count. Arbitrary free-form source search, unbounded
follow-up, and hidden conversation memory are
forbidden; the stateless-call invariant (§15 context)
holds for lookup calls exactly as for decision calls.

Ordering per round: (1) deterministic state update;
(2) lifecycle/reactivation evaluation (§10A); (3)
deterministic relevance closure; (4) prefetch; (5)
summary + frontier construction; (6) LLM invocation;
(7) output validation; (8) optional bounded lookup
request within budget; (9) frontier reconstruction
with retrieved material; (10) at most one more
invocation while budget permits, then decide-or-defer.
Core contract: the LLM must never be required to make
the final semantic decision before evidence necessary
for that decision is available; any other ordering that
preserves this property is acceptable.

Lookup mass fields (measured pre-call, thresholds
unratified): `prefetched_item_count`,
`prefetched_bytes`, `lookup_request_count`,
`lookup_result_bytes`, `lookup_round_count`,
`lookup_budget_exhausted` (bool). These join the §17
fields in P0 readiness measurement.
