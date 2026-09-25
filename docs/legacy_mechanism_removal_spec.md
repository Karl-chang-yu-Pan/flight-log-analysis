# Stage-2 legacy mechanism-discovery physical removal spec

Implementation-ready specification for physically removing the
retired legacy source-mechanism discovery path from production,
moving from "legacy provider unreachable by default" to "legacy
mechanism-discovery provider does not exist in production".

Status: specification only. No implementation, no tests, no commits.

## 1. Purpose

Runtime retirement (`4da67e5`) made DAG the default and left legacy
reachable only via explicit opt-in or model-free missing-source
handling. Dead-but-present provider code is a budget and
maintenance risk: any future caller, flag, or refactor can
re-enable spending silently. This spec removes the legacy
mechanism-discovery implementation while preserving every shared,
DAG-reachable, and migration-needed behavior.

## 2. Current state

- HEAD `4da67e5`: `resolve_mechanism_path` routes DAG by default;
  legacy runs only on `dag_discovery=False`, opt-out env values,
  or missing source snapshot (model-free there).
- `source_discovery_agent` has exactly one live call path
  (`decide_source_discovery` → `discover_source_mechanisms` →
  `SourceMechanismResolver.discover`).
- `SourceMechanismResolver` has no DAG production consumers;
  `discovery_frontier.py` has no non-legacy production consumers.
- `SourceOutputBindingRecord` is the one shared model with live
  DAG-side consumers.
- P0 bakeoff plan references `dag_discovery=False` as an
  explicitly-invoked offline comparison boundary.
- Readiness artifact unratified; P1 incomplete by design.

## 3. Final target state

```text
source_discovery_agent does not exist in production
no production runner can invoke legacy mechanism discovery
no legacy runtime configuration can restore it
DAG is the sole mechanism-analysis implementation
missing-source handling remains model-free
no paid legacy comparison is required
```

## 4. Scope

Delete legacy provider path, resolver, frontier module,
legacy-only schemas, legacy-only tests, and opt-in routing;
relocate the one shared record; rewrite mode tests;
update P0/bakeoff plan and migration docs. Nothing else.

## 5. Non-goals

DAG primitives/traversal/proof/coverage/applicability/replay/
stop/report/intent redesigns; new DAG bounded frontier; intent
fast path (parked); seeder/judge/report cross-checks; budget
thresholds; P1 completion; paid validation of any kind.

## 6. Runtime routing contract

End state of `analyze_flight_log` mechanism selection:

- No `dag_discovery` parameter effect on mechanism path; no
  `FLIGHT_LOG_DAG_DISCOVERY` opt-out values. (Parameter and env
  handling: §8.)
- DAG stage runs whenever its existing prerequisites hold
  (snapshot, inventory as today).
- Missing/invalid snapshot takes the §7 no-source path.
- `resolve_mechanism_path`, `_DAG_EXPLICIT_OFF`, and the
  opt-in branch are deleted; the gate collapses to the DAG
  stage plus the no-source path. Do not leave a
  constant-`"dag"` resolver behind — remove the indirection
  with the branch.

## 7. Missing-source contract

Replace the legacy-shaped missing-source fallthrough with an
explicit deterministic no-source result reusing existing
contracts:

- Result object: the normal report pipeline output with zero
  mechanism candidates and `unresolved_questions` carrying
  the existing wordings ("Exact PX4 source snapshot is
  unavailable …" and the cache-hit wording where cache hits
  exist). No new diagnostic-strength model.
- Cache-hit behavior from the current
  `cached_candidates and source_path_obj is None` branch is
  preserved verbatim (same message, same candidate flow).
- Report writer still runs (unchanged downstream pipeline).
- Source absence is exposed to the user via
  `unresolved_questions` and inventory warnings, as today.
- Log-evidence stages (inventory, timeline, plots) remain
  useful without source, as today.
- Absent vs invalid snapshots behave identically (both mean
  "no pinned source"); no new distinction is introduced.
- Invariants: missing source != successful mechanism proof;
  missing source != legacy fallback; missing source !=
  provider invocation (no `decide` construction, no agent
  call — assert structurally in tests).

## 8. Deprecated legacy-option contract

Contract **B (ignore and use DAG with deprecation logging)**,
chosen because: no caller in-repo passes `False` except the
P0 bakeoff plan (neutralized in S1, same ticket);
fail-closed DAG internals bound the blast radius; explicit
rejection (A) would break the bakeoff plan's call shape
before its S1 neutralization lands; silent preservation is
forbidden.

- `dag_discovery=False` (or any legacy/off env value):
  routed to the DAG path; one deprecation warning through
  the existing audit-log event for mechanism selection
  (no new logging surface).
- P0 `_run_legacy` is a known caller of
  `dag_discovery=False` and MUST be neutralized in S1
  (same ticket): an explicitly requested legacy-vs-DAG
  bakeoff becomes an explicit unavailable/deprecated
  failure — never a silent DAG-twice comparison.
  Historical P0 data/oracles remain untouched.
- Backward-compatibility impact: only explicit legacy
  opt-in callers change behavior (P0 plan, manual
  rollback); default callers already run DAG.
- Tests: mode tests rewritten to DAG-only + deprecation
  path (S1); P0 bakeoff neutralization covered by S1
  bakeoff tests (unavailable/deprecated failure asserts).
- Removal timing: this contract dies with the opt-in at
  S1; no lingering deprecated flag afterwards.
- End invariant: no configuration value can invoke
  `source_discovery_agent` (it no longer exists — §15
  acceptance G).

## 9. Legacy provider deletion

Delete from `runner_core.py`: `source_discovery_agent`
definition, `decide_source_discovery` closure,
`discover_source_mechanisms`,
`source_mechanisms_to_candidates`,
`source_mechanism_to_candidate`, runner-local
`SignalCanonicalizer` and its uses,
`canonicalize_mechanism_candidate_signals` (verify no
remaining caller at deletion time), legacy imports,
legacy routing branch. Shared roles (intent, seeder,
judge, report writer, prepass, cache retrieval of
`models.py` candidates, binding index) stay untouched.

## 10. SourceMechanismResolver deletion

Delete the module `px4/source_mechanism_resolver.py` in full:
`SourceMechanismResolver` (all ~35 methods),
`ParameterFeasibilityGate`, `build_source_discovery_log_context`,
all `dedupe_*`/`extract_*`/binding/candidate/snippet/seed
helpers. No production consumer exists outside legacy
discovery (verified: only `runner_core` legacy branch;
DAG imports profiler/schema/symbols/facts-cache only).
No generic-helper extraction: rule is no concrete current
consumer → delete with the module. Profiler tests using
the gate/context builder are retargeted in S5 (drop those
cases with the helpers; the profiler module itself stays).

## 11. discovery_frontier deletion

Delete `px4/discovery_frontier.py` in full (`CanonicalStore`,
lifecycle/reactivation, identity helpers,
`relevance_closure`, `closed_prefetch`, summary projection,
bounded packet machinery, Mode-B lookup, budgets, fallback
and packet-mass accounting, P0 fragment). Only consumer is
the retired resolver loop. No DAG consumer exists or is
planned; a future DAG bounded-frontier design gets its own
spec and module. Do not preserve on reuse speculation.

## 12. Model/schema cleanup

After relocating `SourceOutputBindingRecord` (§13), delete
from `px4/source_mechanism_models.py`: `SourceDiscoveryLogContext`,
`TopicFieldRef`, `SourceFieldRef`, `ParameterRequirement`,
`SourceSnippet`, `SourceBackedParameterPredicate`,
`SourceBackedVerificationCheck`,
`SourceMechanismBranchGroup`, `SourceMechanismCandidate`,
`SourceMechanismCandidateSet`,
`SourceDiscoveryCandidateDraft`,
`SourceDiscoveryIterationPacket`,
`SourceDiscoveryDecision`. Each was re-verified legacy-only
(no production use outside resolver/runner-legacy; DAG and
report use `models.py` candidates). If the file empties to
the relocated record alone, delete the file and move the
record (preferred) rather than keeping a one-class legacy-
named module.

## 13. SourceOutputBindingRecord relocation

Target: `flight_log_agent/models.py` (shared analysis
contract module). Rationale: already the shared model home
imported by binding/verification flows; the record's fields
are primitives-only (no px4 imports needed), so the move
creates no cycle in either direction (`models.py` imports
typing + pydantic only, verified); semantically adjacent to
`VerificationSignalResolution` consumers; not a dumping
ground (one record with four live consumer files).
Single-definition rule: there MUST be exactly one runtime
class definition, canonical at
`flight_log_agent.models.SourceOutputBindingRecord`. If
compatibility through
`flight_log_agent.px4.source_mechanism_models` is required
between S3 and S4, it MUST be a same-object
import/re-export of the canonical class; creating a second
independent Pydantic class definition is forbidden. The
contract is one class identity, not necessarily two import
paths — prefer updating all imports directly so no
re-export is needed.
Update imports in `test_verification_plan.py`,
`test_verification_graph.py`, `test_binding_index.py`,
`test_graph_execution.py`, `analysis/binding_index.py`
(any comment references), and `runner_core.py` legacy uses
before deleting the source. Verify zero remaining imports
of the old path.

## 14. runner_core cleanup

Beyond §9 deletions: remove `resolve_mechanism_path`,
`_DAG_EXPLICIT_OFF`, and the mode-selection test's target
(rewrite tests per S1); keep `FLIGHT_LOG_DAG_DISCOVERY`
reads only if still meaningful (else remove the env read
with the opt-in); keep the web/`__main__`/e2e call sites
unchanged (they pass no flag or explicit `True`, both
already DAG). Preserve intent, seeder, judge, report,
prepass, cache-of-`models.py`-candidates, and binding
index paths byte-for-byte in behavior.

## 15. Test deletion/rewrites

- DELETE with implementation:
  `tests/test_source_mechanism_resolver.py`,
  `tests/test_discovery_frontier.py` (whole files; both
  exclusively cover deleted code).
- REWRITE `tests/test_mechanism_mode_selection.py`:
  DAG-always routing + deprecation-path behavior
  (§8) + structural no-legacy-provider assertions.
- RETARGET subsets: `test_mechanism_source_profiler.py`
  (gate/context-builder cases), `test_runner_tools.py`
  (`source_mechanism_to_candidate`, numeric-constants
  cases — drop with the helpers).
- KEEP untouched: composition, pipeline, mechanism DAG,
  verdict, temporal, checkpoint, acceptance/oracle,
  binding/plan/graph/execution (after import updates),
  P0 DAG-measurement tests.
- P0 bakeoff `_run_legacy` plan + `dag_discovery=False`
  boundary: neutralized to explicit-unavailable in S1;
  rewritten to DAG-only measurement wording in S6b;
  historical results stay.

## 16. P0/bakeoff cleanup

KEEP: readiness artifact shape, sidecar/oracle loading, DAG
measurement helpers and accounting, historical results.
REMOVE/REWRITE: `_run_legacy` plan entries,
`dag_discovery=False` comparison boundary, any requirement
demanding paid legacy parity (replace with DAG-acceptance
wording; never execute a legacy run to satisfy it).
Historical evidence is not deleted with the comparator.

## 17. Migration-doc cleanup (S6b)

- `docs/adr/0005-default-dag-diagnostic-migration.md`:
  record the runtime-default commit (`4da67e5`) and this
  removal commit; mark D1 default-flip done; mark D2
  temporary-fallback retired (keep history; note the
  superseding decision); include the §17b D20 retirement
  decision record with satisfied/waived evidence items;
  confirm D7 parser parity untouched.
- `docs/default_dag_p0_readiness_spec.md`: legacy-default
  asymmetry passages, fallback-required classifications
  tied to comparison, bakeoff legacy boundary.
- `flight_log_agent/analysis/dag_pipeline.py:1-6`
  module docstring ("Flag-gated … when enabled"):
  update the stale flag-gated wording to DAG-default.
- Check `docs/dag_checkpoint_trial.md` and
  `docs/next_implementation_plan.md` for stale legacy-
  default references at cleanup time; update only lines
  that contradict removal.
- Do not rewrite history: preserve decisions, mark
  superseded steps explicitly.

## 18. P1/superseded criteria

- Ratified budget/threshold policy: STILL REQUIRED
  (separate cost/operations work; untouched by deletion).
- Explicit fallback/rollback routing: SUPERSEDED BY
  RETIREMENT DECISION (no fallback to route to; rollback
  dies with the opt-in).
- Legacy-vs-DAG parity measurement: SUPERSEDED (oracle
  policy is acceptance specs + sidecars, never legacy
  agreement).
- P0 bakeoff completion: REPLACED BY DAG-ONLY ACCEPTANCE
  (historical results kept).
- Do not mark P1 complete: budget policy remains open.

## 19. Dependency DAG

```text
S1 routing simplification + no-source contract
        + bakeoff neutralization + deprecation path
        ↓
S2 runner legacy path/provider deletion
        ↓
S3 relocate SourceOutputBindingRecord ONLY
        + update shared imports (additive; no deletions)
        ↓
S4 atomic legacy subsystem erase:
        resolver + frontier modules,
        legacy model schemas / slim-delete models file,
        direct owned tests (resolver, frontier),
        profiler/runner-tools legacy subsets
        ↓
S5 final orphan sweep + import/reference scan
        + broad offline regression
        ↓
S6b final migration documentation
        + ADR-0005 D20 retirement decision record
        + P0 spec cleanup
        + stale DAG-default wording
```

Global invariant (applies to every ticket; repeated in §20,
§22, and §26): EVERY ticket commit must import the package
successfully, allow pytest collection, contain no tests
importing symbols deleted in that same commit, and leave all
relevant offline suites green. There are NO intentionally
broken intermediate commits. In particular: S3 performs no
deletions (so the resolver's top-level model imports keep
resolving); test files die or are retargeted in the same
slice as their production owner (S1 mode tests, S4 owned
tests and subsets).

S6b depends on S5 (final production shape known; S5 tail
overlap allowed only for non-staling text).

## 20. Ticket definitions S1–S6b

- **S1 — routing simplification + no-source contract +
  bakeoff neutralization.**
  Deps: none. Owns: `runner_core` gate + opt-in removal,
  `resolve_mechanism_path` + `_DAG_EXPLICIT_OFF` removal,
  deprecation path (§8, audit event
  `mechanism_selection.deprecated_legacy_requested` with
  payload `{requested_source: parameter|environment,
  requested_value, effective_route: dag,
  reason: legacy_removed}`), no-source result (§7),
  P0 `_run_legacy` plan neutralization (explicit
  unavailable/deprecated failure — never silent DAG-twice;
  historical data untouched), mode-test rewrite to
  DAG-only + deprecation. Non-goals: provider deletion
  (stubs remain callable but unreachable). Tests:
  rewritten mode tests (§22 A–F) plus bakeoff-neutral
  assertions. Done when: DAG-always routing + deprecation
  + no-source unresolved green; bakeoff cannot silently
  compare DAG against DAG; suite green.
- **S2 — runner legacy path/provider deletion.**
  Deps: S1. Owns: §9 symbol list. Non-goals: resolver
  module deletion (imports die in S4; keep module file).
  Tests: mode tests stay green; no runner-branch tests
  reference removed symbols (S1 already rewrote them).
  Done when: no `source_discovery_agent`/closure/wrapper/
  converters in production; grep-clean.
- **S3 — record relocation ONLY + shared imports.**
  Deps: none strictly (additive-only; may precede S2 if
  scheduling prefers, but numeric order keeps review
  order aligned with risk order — do not reorder merely
  for concurrency). Owns: move
  `SourceOutputBindingRecord` to `models.py`, update
  imports in binding/plan/graph/execution tests and
  flows. Non-goals: any deletion (models file,
  resolver, tests all stay importable). Tests:
  binding/plan/graph/execution green from both import
  paths during transition (keep a re-export only if
  needed to stay green; prefer direct updates).
  Done when: shared consumers import from the new home;
  old path still intact.
- **S4 — atomic legacy subsystem erase.**
  Deps: S2–S3. Owns: delete
  `px4/source_mechanism_resolver.py` and
  `px4/discovery_frontier.py`; slim/delete legacy
  models file (§12 list, after S3 relocation);
  delete `tests/test_source_mechanism_resolver.py`
  and `tests/test_discovery_frontier.py` in the SAME
  slice; drop/retarget resolver-owned portions of
  `test_mechanism_source_profiler.py` and
  `test_runner_tools.py`; update any imports of
  deleted legacy schemas/helpers. Tests: deleted with
  owners; remaining suites green; collection clean.
  Done when: files gone, zero importers of deleted
  paths, suite green.
- **S5 — final orphan sweep + broad regression.**
  Deps: S4. Owns: orphan-reference scan, leftover
  docstring/comment references, broad offline
  regression. S5 restores nothing and fixes no
  behavior; it must NOT be required to restore
  importability (guaranteed by S1–S4 construction).
  Done when: orphan scan clean; suite at baseline
  health.
- **S6b — final migration documentation.**
  Deps: S5 (S6b records the final regression status and
  resulting implementation commit(s), which require the
  S5 sweep result; S5-tail overlap allowed only for text
  that cannot go stale — default to strict sequence). Owns: ADR-0005
  retirement record including the D20 decision
  record (§17b below); P0 spec cleanup; stale
  DAG-default wording including the `dag_pipeline`
  module docstring ("Flag-gated … when enabled",
  `analysis/dag_pipeline.py:1-6`); record the
  physical-removal implementation commit after it
  exists. Done when: no stale
  legacy-default/fallback/parity requirements remain.

## 17b. ADR-0005 D20 retirement decision record

D20 (`docs/adr/0005-default-dag-diagnostic-migration.md:86`)
states legacy retirement remains UNDECIDED and requires
its own decision with evidence (completed bake-off,
sustained compatibility, acceptable fallback rate,
sufficient contract parity, rollback evidence, stable
report behavior). S6b must record that decision
explicitly — no new ADR required. Classification:

- Completed bake-off: WAIVED BY EXPLICIT RETIREMENT
  DECISION — paid legacy bakeoff/parity is intentionally
  not executed because legacy output is no longer
  treated as the correctness oracle. DAG validation
  relies instead on deterministic acceptance specs,
  accepted semantic sidecars/oracles, and deterministic
  DAG authority checks.
- Sustained compatibility: SATISFIED FOR THE AVAILABLE
  OFFLINE/DETERMINISTIC EVIDENCE BOUNDARY — DAG
  deterministic regression suites, composition boundary,
  accepted semantic oracle/sidecar tests, DAG-default
  routing regression, and relevant offline integration
  coverage. This does not claim longitudinal
  production-time observation. No paid validation is
  added.
- Acceptable fallback rate: SUPERSEDED — no automatic
  fallback exists by construction (early return /
  propagate); explicit fallback routing died with the
  opt-in.
- Sufficient contract parity: SUPERSEDED — parity
  against legacy output is not a correctness criterion;
  acceptance specs + sidecars are.
- Rollback evidence: SUPERSEDED — post-removal rollback
  is `git revert` / redeploy, stated in S6b docs. This is
  a documented operational procedure, not a tested
  runtime capability, and must not be described as
  validated rollback evidence. No rollback tests are
  manufactured.
- Stable report behavior: SATISFIED — deterministic
  report pipeline unchanged; validation + downgrades
  intact.

## 21. Allowed parallelism

CORE STRICTLY SEQUENTIAL: S1 → S2 → S3 → S4 → S5,
then S6b (which depends on S5). S6b is docs-only and may
overlap the S5 tail only where the text cannot go stale
(it references final symbol names, statuses, regression
results, and commit hashes, so overlapping is
discouraged — default to strict sequence). Do not
design this workstream around parallelism;
`implement-spec` may still orchestrate the dependency
graph sequentially.

## 22. Offline acceptance matrix

- A. Default call → DAG (mode tests, no provider).
- B. `dag_discovery=False` → DAG + deprecation audit
  event (no legacy provider).
- C. Legacy/off env values → same as B.
- D. Missing source → deterministic unresolved report
  (unresolved questions preserved); zero provider calls
  (structural: no agent/resolver construction on path).
- E. Invalid source → same safety property as D.
- F. DAG failure → propagates/fails closed; no legacy
  invocation (existing structural pins retained).
- G. Production grep-clean for `SourceMechanismResolver`,
  `source_discovery_agent`, `discovery_frontier`,
  legacy `SourceDiscovery*` schemas (shared record
  excepted at its new home).
- H. Binding/plan/graph/execution suites green after
  relocation.
- I. Acceptance specs/sidecars load and run offline.
- J. No benchmark-specific production logic introduced.
- K. Legacy bakeoff neutralization: an explicitly
  requested legacy-vs-DAG bakeoff (`FLIGHT_LOG_BAKEOFF`,
  `_run_legacy`, `dag_discovery=False` boundary) yields
  an explicit unavailable/deprecated outcome — never a
  silent DAG-twice comparison labeled as parity, never a
  legacy provider invocation.

## 23. Import/dependency invariants

After Stage 2: zero production imports of deleted paths
(assert in tests); `models.py` gains no px4 imports;
`binding_index.py` and verification flows import the
record from its new home; no circular imports (record
dependency-free — verified).

## 24. API-budget invariant

Acceptance: `source_discovery_agent` absent from
production; no runner path constructs a legacy decide
closure; no legacy configuration restores it (opt-in
deleted); DAG sole mechanism implementation;
missing-source model-free; no paid legacy comparison
required. Stronger than today's reachability invariant.

## 25. Rollback implications

Stage 2 deletes the rollback itself. After S1, rollback
means reverting commits (standard git), not a runtime
flag. State this in the S6 doc updates so no operator
expects a flag that no longer exists.

## 26. STOP conditions

STOP and report if: DAG consumes legacy mechanism output
(verified it does not — profiler/schema/symbols only);
a legacy-only helper has a real DAG consumer (none
found — exhaustive import grep); the record cannot move
without cycles (dependency-free — verified); no-source
handling needs new architecture (§7 reuses existing
contracts); governance forbids removal now (ADR-0005
anticipates it; D2's separate decision is this chain);
P0 correctness requires running legacy provider (only
the opt-in bakeoff plan does — rewritten, never
executed). None triggered. Implementation-ready.

## Deferred ledger

Intent fast path (parked spec); seeder/judge/report
redesigns and cross-check; DAG bounded frontier; budget
thresholds; P1 completion; paid validation of any kind.
