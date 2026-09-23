# Default-DAG P0 readiness spec

Implementation-ready specification for P0 of ADR-0005 (staged
DAG-primary migration): readiness measurement, DAG-decisive
routing-predicate definition, semantic comparison harness,
performance/fallback measurement, and budget-ratification
procedure. P0 observes and measures. It flips no defaults,
changes no routing or fallback behavior, rewrites no authority,
and builds no production architecture.

Status: specification only. No implementation, no fixtures beyond
the four committed acceptance logs, no tests.

## 1. Scope and non-goals

P0 answers: is the current DAG path ready to become the default
diagnostic path, with legacy retained as marked fallback, without
changing proof authority? P0 produces measurements and executable
readiness checks. It does not itself flip the default.

Explicit non-goals: P1 explicit fallback and precedence routing;
P2 shadow bake-off execution; stop-authority changes of any kind;
legacy retirement; parser parity work; stateful replay; DAG cache
reactivation; new telemetry or persistence architecture; report
schema changes; benchmark-strength changes.

## 2. Governing constraints (ADR-0005, frozen)

P0 preserves: staged migration; legacy as temporary fallback;
retirement undecided; T6B stop-authority conjunction unchanged
(`legacy_verified` AND coverage AND applicability AND
non-vacuous proof AND Gate-B discharge, with required-proof-absent
for omitted inputs and empty-non-degenerate defers-to-legacy);
DAG-decisive as routing only; no new authority layer;
harness-only shadow; tree_sitter selectable per DAG run; 31
legacy-parser xfails block retirement only; stateful replay not
required. LLMs remain seeder/judge components under
deterministic constraints; the deterministic part is the
constrained evidence/report boundary, not every upstream
inference.

## 3. Seams (all existing, highest first)

forthcoming P0 tests live at exactly these seams; no new seams,
no production seams touched:

1. Real-log parsing: production inventory
   (`parse_ulog_inventory`, `observed_signals_from_inventory`)
   plus raw pyulog ground truth (Takeoff/Airspeed pattern).
2. Pinned source/snapshot assertions (file + symbol + content;
   rev-parse snapshot pin).
3. DAG stage result structures (`DagStageResult`, verdict
   model, replay dict, `validate_report`, `EvaluationScope`)
   consumed read-only through existing public entry points.
4. Agent-injection seam (`run_agent` callable on
   `run_dag_discovery_stage`; proven by the TECS stub-runner
   pattern) for deterministic fast tests.
5. Sidecar/oracle loading from committed acceptance artifacts
   (`tests/acceptance/*.json` + acceptance test modules) as
   regression oracles only — never as live evidence.
6. Audit artifacts (`usage.json` per run: requests, input /
   output / total tokens, cached and reasoning tokens) for
   LLM-usage measurement; wall-clock timers around harness
   invocations.

## 4. DAG-decisive routing predicate

### 4.1 Contract

A pure deterministic function over an already-produced DAG
stage outcome. No I/O. No LLM call. No mutation. No report
generation. No fallback execution. It answers routing
readiness only. It never authorizes stop, replaces
`legacy_verified`, changes proof authority, upgrades
confidence, or changes evidence strength.

### 4.2 Exact inputs (existing state only)

- `judged.verdict.sufficient` (bool) and
  `judged.verdict.selected_terminal` (non-empty) from the
  stage result.
- `replay` dict `status` / `complete` from the stage result
  (existing replay vocabulary: matched, mismatched, partial,
  unevaluable, not_attempted; completeness flag).
- `validate_report(report).passed` (bool) from the existing
  report validator.
- `annotated_dag` presence (non-None, non-empty vertices).
- Report hypothesis verdict ceilings
  (supported / contradicted / mixed / unresolved / excluded
  vocabulary) for the contradiction signal.

TDD must re-pin exact field names from the models if any
name above has drifted; the mapping rule (below) is
normative, spellings are bound to code.

Feasibility enters through verdict + validation (the judge
sees feasibility; the validator checks applicability
completeness), not as a separate invented boolean. No new
booleans are introduced when existing state expresses the
semantics.

### 4.3 Outcomes (test-side normalization, not production)

Four states using existing vocabularies only:

```text
DECISIVE: sufficient AND replay complete AND validation
    passed AND annotated DAG present AND no contradicted
    ceiling on the selected claim.

CONTRADICTED: complete replay mismatch on the terminal
    domain OR a contradicted ceiling on the selected claim.
    (Takes precedence over DECISIVE inputs.)

UNDECIDED: not contradicted, but insufficient verdict,
    incomplete replay, failed validation, or absent DAG.

UNAVAILABLE: the stage could not run or be evaluated —
    missing snapshot, missing required signals, degenerate
    scope, unevaluable domain.
```

This preserves contradiction ≠ unknown ≠ unsupported ≠
verified. No new production enum is created; the four
states live in test/harness code. A future P1 home is
suggested (the `dag_pipeline` selection neighborhood where
the temporal selector lives) without forcing a module.

### 4.4 Owner

P0 implements the predicate as a pure helper inside the
readiness tests/harness. It is acceptance-side analysis
code, never imported by production.

## 5. Semantic comparison harness

### 5.1 Shape

A test-only harness comparing the legacy diagnostic path
against the DAG diagnostic path, run sequentially by the
driver. No production dual-run. Reusable for the four
deterministic fixtures plus future representative samples.

### 5.2 Live-path boundaries (justified)

- DAG path: `run_dag_discovery_stage` — the exact production
  stage entry, proving real stage behavior rather than a
  subcomponent. Agent mode varies by suite (see §9).
- Legacy path: `analyze_flight_log` with the DAG flag off —
  the exact default production entry, proving real default
  behavior. No narrower legacy seam exists that preserves
  end-to-end diagnostic meaning; sub-pipeline slicing would
  test parts, not the behavior being migrated.

Both boundaries are the narrowest entries that still
represent real live behavior on their side.

### 5.3 Harness independence

The harness must NOT feed benchmark expected answers into
live diagnosis, alter prompts to force mechanisms, reuse
sidecar numeric outputs as live evidence, change feature
flags globally across unrelated tests (per-run env isolation
with restoration), or mutate persistent state. Each path
diagnoses independently. Sidecars and acceptance modules
are oracle loaders only.

### 5.4 Benchmark oracles (independent sources)

Oracle semantics load from committed acceptance artifacts
without executing their assertions as live evidence:

- RTL: cone/floor mechanism family + PROVEN compatibility
  (`tests/acceptance/rtl_weird_height.json`,
  `tests/test_acceptance_rtl.py`).
- TECS: restart-transient mechanism + PROVEN compatibility
  (`tests/acceptance/tecs_restart_transient.json`,
  `tests/test_acceptance_tecs.py`).
- Takeoff: minimum-altitude + resume mechanism + PROVEN
  compatibility (`tests/acceptance/takeoff_minimum_altitude.json`,
  `tests/test_acceptance_takeoff.py`).
- Airspeed: load-factor binding + BEST_SUPPORTED
  compatibility (`tests/acceptance/airspeed_load_factor.json`,
  `tests/test_acceptance_airspeed.py`).

### 5.5 Normalization schema (harness-side only)

Per (fixture, path) normalize exactly:

```text
mechanism_family: canonical family string derived from the
    path's structured findings (never prose similarity).
rival_disposition: per-benchmark rival outcomes over the
    existing vocabulary (excluded / inactive /
    contradicted / weaker / unresolved / not-applicable);
    no new normalization that loses meaning.
grounding: required log evidence present + required source
    snapshot present + required source mechanism grounded +
    numeric checks present where the benchmark requires +
    no fabricated source refs. Source-ref formatting
    differences with identical semantics are compatible.
strength_compat: benchmark strength not contradicted by the
    live result (PROVEN must not degrade to unsupported or
    contradicted; BEST_SUPPORTED must not be silently
    upgraded to PROVEN).
status: one of MATCH / MISMATCH / UNDECIDED / UNAVAILABLE
    for the comparison (harness vocabulary).
```

Never compared: exact prose, titles, tool order, generated
IDs, source-ref ordering, LLM call counts, live confidence
equality. Confidence is computed independently per path and
asserted separately from strength compatibility.

### 5.6 Rival normalization

Map each path's rival evidence onto excluded / inactive /
contradicted / weaker / unresolved / not-applicable using
the benchmark's own rival vocabulary where it exists
(Takeoff A/B/C, Airspeed A–F patterns); otherwise the
generic six values apply without redefining benchmark
meaning.

### 5.7 Parser selection

DAG comparison runs select tree_sitter per run; legacy
comparison runs use the expected current configuration
(legacy default). The asymmetry is intentional during
migration and must be recorded in the artifact, not hidden.
Environment is restored after each run; defaults never
change globally.

### 5.8 Snapshot-unavailable measurement

Run the harness once per fixture with the snapshot
withheld (or record the current silent-fallthrough branch
by direct inspection plus a no-snapshot run): classify the
outcome as `would require explicit fallback in P1`. Do NOT
fix the fallthrough in P0.

### 5.9 Prose decoupling

Include at least one normalization test showing semantically
equivalent differently worded findings normalize
identically. Never use an LLM as judge to compare LLM
outputs. Where legacy exposes semantics only through final
prose, specify the narrow deterministic extraction used —
or record the field as uncomparable (UNDECIDED-leaning)
rather than guessing.

### 5.10 Comparison prior art

`scripts/judge_io_diff.py` is untracked, non-authoritative
prior art, partially reusable conceptually for its
pinned-fixture normalization and diff discipline. P0 does
not adopt or modify it and does not depend on it: the P0
harness compares independently produced live-path semantic
outputs, not judge inputs or expected-answer fixtures.

## 6. Measurements

### 6.1 Required measurements (never guessed)

Per fixture, per path, plus aggregate summary (never hide
outliers in one aggregate): wall-clock duration; LLM usage
at actually available granularity (requests, input/output/
total tokens, cached and reasoning tokens from existing
audit `usage.json`; currency conversion explicitly out of
scope); DAG-decisive state counts (decisive vs
undecided/unavailable → fallback-required rate); semantic
mismatch count (0 required pre-P2).

### 6.2 Silent-fallthrough exposure

The missing-snapshot case (§5.8) is a measured readiness
case, classified as P1-fallback-required, not fixed.

### 6.3 Four-fixture gate semantics

Pre-P2 progression requires 0 semantic mismatches on all
four benchmarks. DAG-undecided is NOT a mismatch, but it
may still mean not-ready depending on benchmark and future
fallback policy; the readiness report must state per
fixture whether undecided blocks default (e.g. undecided
on a PROVEN benchmark with no fallback coverage blocks;
undecided with explicit fallback coverage is reported, not
hidden).

### 6.4 Representative samples

The four benchmarks are mandatory for P0 correctness.
Additional samples are optional exploratory input in P0
and required only for the later P2 bake-off. No arbitrary
sample count is set here.

## 7. Budget ratification

### 7.1 No invented thresholds

P0 measurement reports; nothing fails on invented
thresholds. The architecture-critic candidate examples
(wall < 2× legacy, LLM usage ≤ legacy) are labeled
candidate thresholds pending P0 measurement unless
repository evidence justifies them.

### 7.2 Ratification procedure

Collect measurements → summarize fixture-by-fixture →
propose candidate thresholds → review/ratify → store
ratified readiness values in a versioned tracked artifact.
Ratification is a review act recorded in the artifact, not
a code constant and never environment-only undocumented
numbers. After formal adoption, acceptance fails loudly
when ratified thresholds are exceeded.

### 7.3 Artifact placement

- Tracked: `tests/acceptance/dag_default_readiness.json`
  with schema `{version, status:
  pre-ratification | ratified, thresholds: null | {...},
  basis: measurement references}`. Thresholds stay null
  until ratified; budget tests report-only while null.
- Untracked: raw per-run measurement outputs (gitignored
  paths / CI artifacts), never asserted, never committed.

### 7.4 Enforcement contract

Specify and test the contract: null thresholds →
report-only (pass with note); ratified thresholds →
loud failure on exceedance. The contract test may run
against synthetic thresholds so it does not depend on
ratification having happened.

## 8. Measurement artifact schema (per run)

Machine-readable, reviewable, free of machine-specific
noise (no absolute paths, random IDs, run timestamps, raw
temp filenames unless explicitly diagnostic and excluded
from assertions):

```text
scenario, legacy semantic result, DAG semantic result,
compatibility, DAG-decisive state, fallback-required,
wall time per path, LLM usage per path, source snapshot
status, validation status, parser selection per path,
notes
```

## 9. CI vs bake-off suites

- Fast deterministic CI (default pytest): predicate unit
  tests over constructed stage outcomes (no I/O, no
  models); oracle loader tests; normalization tests
  including prose-variance; env-isolation tests; snapshot-
  unavailable detection; budget report-only behavior;
  ratification contract tests (synthetic thresholds).
  No test in this suite may require model/network
  availability.
- Manual bake-off suite (explicit marker + env gate,
  deselected by default, e.g. marker `bakeoff` requiring
  `FLIGHT_LOG_BAKEOFF=1`): real `run_agent` on the DAG
  side and real model behavior on the legacy side across
  the four fixtures; writes untracked measurement
  artifacts; asserts nothing about thresholds
  pre-ratification (records only).
- DAG-side stub precedent (`run_agent` injection, TECS
  stub-runner pattern) is used ONLY for CI mechanics
  tests, never as readiness evidence. Bake-off readiness
  comes exclusively from real-behavior runs.

## 10. P0 test plan

- P0-T1 architecture/preflight: ADR-0005 expectations,
  required fixtures, source snapshot presence.
- P0-T2 predicate mapping: constructed stage outcomes
  covering decisive, undecided, validation-fail, replay-
  incomplete, judge-fail, feasibility-fail map to the §4
  contract correctly.
- P0-T3 authority isolation: predicate invocation never
  touches stop authority, writes `legacy_verified`,
  mutates checkpoint/proof state, or changes confidence.
- P0-T4 oracle loader: committed acceptance semantics
  load without feeding diagnostic execution.
- P0-T5 normalization: family, rivals, grounding,
  strength-compat, status normalize per §5.
- P0-T6/T7/T8/T9 RTL/TECS/Takeoff/Airspeed comparisons:
  each records MATCH / MISMATCH / UNDECIDED / UNAVAILABLE
  (harness states for the comparison itself).
- P0-T10 no prose coupling: paraphrased equivalents
  normalize identically.
- P0-T11 confidence/strength independence: live
  confidence never equality-compared to benchmark
  strength.
- P0-T12 parser isolation: per-run selection recorded,
  env restored, defaults unchanged.
- P0-T13 snapshot-unavailable: detected, reported as
  P1-fallback-required; not fixed.
- P0-T14 wall time: captured per path, report-only.
- P0-T15 LLM usage: captured at audit granularity,
  report-only.
- P0-T16 fallback readiness: decisive vs
  undecided/unavailable counts.
- P0-T17 aggregation: semantics + measurements combine
  into the §8-shaped readiness report.
- P0-T18 pre-ratification: null thresholds never fail.
- P0-T19 ratification contract: synthetic-threshold
  enforcement specified and tested.
- P0-T20 regression isolation: RTL/TECS/Takeoff/
  Airspeed/temporal/proof suites unchanged.
- P0-T21 full regression: no test disappearance (count
  against the post-Airspeed baseline plus new P0 tests).

## 11. Stop gates

- STOP A: existing DAG outputs cannot map to a pure
  routing-readiness predicate without changing proof
  authority.
- STOP B: semantic comparison requires feeding expected
  answers into live diagnosis.
- STOP C: legacy and DAG outputs lack stable structure
  for mechanism/rival/grounding normalization.
- STOP D: measurement needs new production persistence
  or telemetry architecture.
- STOP E: real LLM usage unmeasurable at any stable
  hook. Non-blocking if audit calls/tokens suffice;
  otherwise STOP.
- STOP F: per-run parser selection cannot be isolated
  safely.
- STOP G: P0 requires changing default routing or
  fallback behavior.
- STOP H: P0 requires changing T6B stop authority.
- STOP I: a benchmark oracle cannot load without
  duplicating its expected-answer logic.
- STOP J: current live evidence contradicts a closed
  benchmark (report contradiction; do not widen scope).

On any fired stop: do not widen P0 scope. Route to
`$wayfinder` for missing evidence, `$architecture-critic`
only for a real architecture contradiction.

## 12. P0 success condition

GREEN requires all of: DAG-decisive semantics mapped
from current state; predicate routing-only (T3-proven);
semantic comparison representation exists; four oracles
load independently; no prose/confidence-equality
dependence; parser isolation explicit; snapshot-
unavailable measured; wall time measurable; LLM usage
measurable at audit granularity; fallback-required rate
measurable; readiness report generatable; no invented
threshold gates GREEN; ratification procedure defined;
no production behavior changed; T6B untouched; existing
regressions preserved.

P0 GREEN does NOT mean DAG is ready for default. It means
the machinery and measurements to decide readiness exist.

## 13. P0/P1/P2 handoff

P0 produces: predicate contract, comparison harness,
measurement results, candidate budgets, readiness
findings. P1 later implements explicit fallback,
precedence routing, and markers. P2 later executes the
shadow bake-off against ratified gates. No P1 behavior
enters P0.

## 14. Explicit exclusions

Default flip; fallback implementation; precedence
function implementation; stop-authority changes; legacy
retirement; parser parity work; stateful replay; DAG
cache reactivation; report schema changes; proof-
authority changes; confidence changes; benchmark-
strength changes; new telemetry/persistence; live-LLM
behavior changes; performance optimization; fixture
provisioning beyond the four accepted logs.
