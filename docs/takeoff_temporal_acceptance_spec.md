# Takeoff temporal acceptance spec (W3C)

Deterministic acceptance for the accepted Takeoff benchmark, driven
by the already-landed W3A temporal causal-qualification machinery
(ADR-0004, `docs/temporal_causal_qualification_spec.md`). W3A and
W3B are CLOSED; this spec adds acceptance only and reopens
nothing. Out of scope: W3A/W3B redesign, Airspeed representation,
numeric-check ↔ source-operation linkage ownership, new graph or
view types, new persistent temporal models.

Status: implemented and acceptance-verified. W3C acceptance is
complete on the existing temporal architecture with no
production-code changes. Acceptance coverage lives in
`tests/test_acceptance_takeoff.py` and
`tests/acceptance/takeoff_minimum_altitude.json`; the real ULog
fixture is provisioned through the existing gitignored uploads
test-resource mechanism and is not committed.

## 1. Accepted benchmark truth (frozen)

Do NOT rediagnose or downgrade this case unless deterministic
repository evidence directly contradicts it.

Accepted conclusion:

```text
Navigator intentionally commands the initial takeoff target using
MIS_TAKEOFF_ALT = 20 m above home.

The mission waypoint at approximately 7 m above home is retained
and becomes active later.

Therefore the initial ~20 m climb is not an unexplained overshoot
or replacement of the 7 m mission waypoint.
```

Benchmark strength: PROVEN.

## 2. Real fixture (provision in TDD, never fabricate)

The accepted real log is `inconsistent-takeoff-hgt.ulg`
(md5 `864264eb3ceadc25c5e0b0703788be61`, ~18 MB). It is NOT
currently stored in this repository's `uploads/` tree; accepted
copies exist alongside prior analysis. TDD must provision it
into the repository's gitignored upload/test-resource path
following the RTL/TECS pattern, and acceptance must fail
loudly if the fixture is missing (missing-fixture error, same
convention as RTL A1 / TECS T1-analog). Do NOT synthesize or
substitute data. Do NOT commit the ULog.

## 3. Source snapshot

Use the pinned source snapshot `1dacb4cdef2d7145754fc788fa8dc482eed74b40`
(the same pin as RTL/TECS acceptance). Acceptance must fail
loudly if the source checkout is missing or mismatched
(rev-parse check, same convention).

## 4. Re-verified source anchors

The following regions were re-verified against the pinned
snapshot for this spec; TDD re-pins exact lines/sites before
putting them in tests (line reminders below are starting
points, not golden values):

```text
mission_params.c (~47–58)
    MIS_TAKEOFF_ALT parameter definition (minimum takeoff
    altitude above home/ground).

mission.cpp:793–821
    vertical-takeoff block: `mission_item_next_position =
    _mission_item` preserves the original mission item while
    `_mission_item` is rewritten as NAV_CMD_TAKEOFF with the
    computed takeoff altitude; emits the "Takeoff to %.1f
    meters above home" log line.

mission.cpp:~847–853
    "if we just did a normal takeoff navigate to the actual
    waypoint now" — resume logic converting the takeoff item
    back toward waypoint handling.

mission.cpp:1239–1273
    `do_need_vertical_takeoff()` gate (rotary-wing check,
    landed/altitude conditions, takeoff-item conditions).

mission.cpp:1338–1351
    `calculate_takeoff_altitude()`: `fmaxf(takeoff_alt,
    home_alt + takeoff_min_alt)` (landed variant uses
    global-position altitude instead of home).

navigator.h:292
    `get_takeoff_min_alt()` returns `_param_mis_takeoff_alt`,
    binding the parameter to the fmaxf rule.

mission.cpp:~1093 / ~1153
    `mission_item_to_position_setpoint(mission_item_next_position,
    ...)` — the retained original item flowing back into the
    position triplet (resume path).
```

## 5. Verified log facts (re-derive exact values in TDD)

Measured from the real fixture; TDD re-parses and pins exact
values rather than copying these reminders:

```text
home altitude (time-varying, hold-last semantics):
    12.385593 @ 547.583 s (initial)
    12.183000 @ 682.506 s (at arming/first takeoff)
    11.622925 @ 836.838 s (landing)
    → home-at-command (latest at/before 791.477): 12.183000

MIS_TAKEOFF_ALT (logged parameter): 20.0

nav_state 2 → 3 (posctl → mission): 791.253 → 791.408

logged text:
    [navigator] Executing Mission @ 791.475
    [navigator] Takeoff to 20.0 meters above home @ 791.476

position_setpoint_triplet.current (only 4 samples total):
    670.635: alt 0.000, invalid
    791.477: alt 32.183, valid   (= 12.183000 + 20)
    805.314: alt 19.183, valid   (= 12.183000 + 7, see §8)
    808.467: alt 0.000, invalid  (pilot takeover logged @ 808.457)

navigator_mission_item (2 samples):
    791.477: nav_cmd 22 (TAKEOFF), alt 32.183
    805.314: nav_cmd 16 (WAYPOINT), alt 7.000

mission_result.seq_current: always 0 — NOT retention evidence.

vehicle_global_position.alt max in [785, 815]: 31.472 @ 805.815
    → 31.472 − 12.183000 = 19.289 above home-at-command.

vehicle_land_detected.landed: 0 with 1 present (phase context).

takeoff_status.takeoff_state: 3@680.5, 4@681, 5@684
    (early RC-takeoff phase), 3@835, 1@836 — belongs to the
    earlier ~682 s manual takeoff, NOT the 791 s mission phase.
    Do NOT use takeoff_state for Window A/B derivation.
```

## 6. Temporal structure (two ordered windows)

Minimum generic seeds, both derivable with the existing
`TransitionEventSpec` value-change matching (no new
derivation machinery):

```text
Window A — initial takeoff command phase:
    derived from nav_state 2 → 3 (≈791.408) and/or the
    first valid triplet sample (791.477, alt 32.183).
    Covers the initial +20 m takeoff command.

Window B — retained waypoint resume phase:
    derived from the triplet/mission-item change at 805.314
    (alt 32.183 → 19.183; nav_cmd 22 → 16).
    Covers the resumed +7 m mission command.
```

Ordered-window rule: every assertion that compares phases
must verify `Window A precedes Window B` from exact fixture
timestamps. Representation: the existing plural
`EvaluationScope.windows` (already multi-window capable per
W3A tests); evaluate eligibility/discrimination PER window.
No production phase model, no labeled phase objects.

## 7. Initial target arithmetic (benchmark-side numeric evidence)

Acceptance verifies deterministically from logged values:

```text
home-at-command (hold-last home sample at/before command time)
+ MIS_TAKEOFF_ALT (logged parameter)
≈ initial commanded triplet altitude
```

Accepted instance: `12.183000 + 20.0 ≈ 32.183`. TDD pins the
hold-last home selection rule (latest home sample at or
before the command timestamp) and the tolerance from log
quantization conventions. This equation lives in acceptance
only — never in generic temporal selection.

## 8. Later waypoint arithmetic

Acceptance verifies:

```text
later mission target ≈ home-at-command + mission waypoint
relative altitude
```

Accepted instance: logged mission-item alt `7.000` with
`12.183000 + 7.0 = 19.183` matching the later triplet
exactly to log precision. TDD must confirm the
relative-to-home conversion rule from
`mission_item_to_position_setpoint` source (not assumed);
if the conversion cannot be confirmed deterministically,
the later target is still pinned as the logged triplet
value with the item alt recorded as corroboration. Do not
infer unavailable mission-plan contents.

## 9. Retention claim (honest scoping)

The accepted mechanism includes waypoint retention. Support
it with the strongest deterministic combination available:

```text
source logic preserving the original item
    (mission.cpp ~802–803: mission_item_next_position)

later command consistent with the original item
    (cmd 16 at 805.314 following cmd 22 at 791.477;
    waypoint coordinates/altitude continuity)

later setpoint returning to ~7 m-relative behavior
```

Explicitly distinguish in assertions:

```text
observed later ~7 m command        (required)

same retained original waypoint    (required ONLY where
    source/log evidence supports it deterministically)
```

`mission_result.seq_current` (always 0 here) MUST NOT be
cited as retention evidence. Only require the stronger
retained-identity claim where the evidence supports it;
otherwise the observed-command claim plus source retention
logic carries the mechanism.

## 10. Source mechanism grounding

Represented source candidates (re-verify in TDD against the
pinned snapshot + real DAG):

```text
minimum takeoff-altitude computation
    (calculate_takeoff_altitude + fmaxf rule, §4)

original-item retention
    (mission_item_next_position assignment, §4)

takeoff-item rewrite + resume logic
    (§4 blocks above)
```

The earlier investigation found relevant nonterminal source
operations represented; re-check current behavior after
W3A/W3B changes. TDD records per-candidate represented
identity, available domain, and discriminating evidence
(the TECS §24 pattern). Assumptions beyond that record are
forbidden.

## 11. Data-flow role

Initial target: value-producing path (home + MIS_TAKEOFF_ALT
through the fmaxf rule into the triplet). Later target:
value-producing/resume path (retained item through
`mission_item_to_position_setpoint`). Temporal windows
distinguish WHEN each mechanism is relevant. Do not turn
arbitrary control ancestors into source refs.

## 12. Control/state role

Takeoff uses more explicit state/context than TECS
(nav_state, landed flags, takeoff-item vs waypoint-item
identity). Use control/branch facts as feasibility/context
only. Do NOT alter `branches_verified`. Control ancestors
never become source refs by themselves.

## 13. Logged text (corroboration only)

`Takeoff to 20.0 meters above home` (@791.476) and
`Executing Mission` (@791.475) are corroborating
observation evidence: assert presence as corroboration.
They are NOT CodeRefs, proof authority, or source
provenance. The mechanism must remain supported even if
text is treated as corroborative rather than primary.

## 14. Mechanism-level claim (exact wording bound)

The accepted W3C conclusion is approximately:

```text
The initial +20 m target is the intentional minimum
takeoff-altitude behavior.

The later +7 m target belongs to the mission waypoint
sequence.

The observed ordering is expected state-machine behavior
rather than an inconsistent altitude command.
```

Do not overclaim exact internal state that is not
observable (e.g. which internal branch fired on which
cycle) unless deterministic evidence supports it.

## 15. Rival hypotheses (evaluated only where evidence permits)

```text
Rival A: the mission waypoint itself requested +20 m.
    Excluded by: waypoint item alt 7.000 logged at 805.314
    plus fmaxf/home arithmetic producing 32.183 only via
    the takeoff rule.

Rival B: the +7 m waypoint was overwritten/lost.
    Excluded by: later 19.183 command consistent with the
    retained item (§9) plus source retention logic.

Rival C: the +20 m target was uncontrolled overshoot, not
    an intentional commanded takeoff target.
    Excluded by: exact home+MIS_TAKEOFF_ALT arithmetic,
    takeoff log text, and ordered phase structure
    (command precedes achievement; max 19.289 above home
    is consistent tracking, not the command itself).
```

Only include rivals deterministically evaluable from the
available evidence; drop or downgrade the rest rather than
stretching.

## 16. Command vs observed altitude

The mechanism concerns the intentional TARGET
(triplet 32.183, then 19.183). Observed tracking
(`vehicle_global_position` max 31.472 → 19.289 above home;
local-position climb) corroborates the command but is not
the same fact. Never substitute achieved altitude for
commanded altitude in the numeric relations.

## 17. Evidence roles (ADR-0004, unchanged)

```text
source evidence:
    code computation / mission behavior (§4, §10)

observation evidence:
    logged setpoints / states / messages (§5)

temporal evidence:
    takeoff target precedes later mission target (§6)

numeric evidence:
    +20 / +7 relationships (§§7–8)
```

No new universal evidence model.

## 18. Temporal eligibility (W3A unchanged)

A source candidate overlapping a relevant window is only
temporally eligible, never automatically uniquely causal.
Apply per diagnostic window.

## 19. Ordered-window discrimination (specified, not built)

Assert per window using current DAG/window metadata:

```text
candidates overlapping Window A only
    → initial-takeoff mechanism evidence

candidates overlapping Window B only
    → waypoint-resume mechanism evidence
```

Each single-window evaluation uses the existing sole-overlap
rule; no cross-window discriminator is built (W3A's
`discriminate_candidates` has none — verified). Do not use
source order. If per-window evaluation cannot separate the
relevant paths, STOP per §21 rather than inventing
separation.

## 20. Refined STOP G for W3C

Case 1 (mechanism-equivalent tie with independently
supported shared mechanism): proceed per the amended
temporal spec — but do NOT auto-reuse TECS's equivalence.
Takeoff equivalence must be re-established for Takeoff's
claim if it is invoked at all.

Case 2 (mechanism-different tie, no discriminator): STOP G.

Case 3 (equivalence unestablished): STOP G, mandatory.

Expected path (to be confirmed by TDD, not assumed):
ordered-window separation makes Case 2/3 unlikely, because
takeoff-phase and resume-phase writers occupy different
windows. If the real DAG instead yields an unresolved
mechanism-different tie inside one window: STOP, do not
force the TECS pattern onto Takeoff.

## 21. Candidate-domain information

TDD records, per mission/takeoff writer candidate (same
per-candidate record pattern as TECS §24):

```text
replay-result active_windows (if any)

gating/branch windows and feasibility

EvaluationScope windows overlapping it

candidate operation identity
```

Specify exactly what distinguishes the takeoff-phase from
the resume-phase writers. Mission branches referencing
logged state (vehicle type, landed flags, altitude
comparisons) may be evaluable — verify, do not assume.

## 22. Replay role

Use replay only if existing operation-linked replay
naturally contributes. Do NOT build new replay linkage
for W3C. If ordered windows + source/log evidence
suffice: replay is optional.

## 23. Source refs

Do NOT require live report source refs if deterministic
acceptance can ground source computations honestly through
the DAG/source snapshot (TECS 2a precedent). Do NOT
fabricate refs. Empty-signature behavior may remain.

## 24. Benchmark strength

Preserve PROVEN unless actual deterministic evidence
directly contradicts the accepted benchmark. Do not
downgrade merely because some internal state is unlogged.

## 25. Missing evidence

The core conclusion does not require `mission.plan`,
full mission-file provenance, or innovation/internal
estimator signals (per accepted analysis). Classify
unavailable evidence honestly (NON_DECISIVE where it does
not distinguish surviving explanations); do not turn it
into a blocker.

## 26. Acceptance sidecar

Extend `tests/acceptance/` conventions (same schema as
RTL/TECS sidecars: scenario_id/question/inputs/expected/
strength/unavailable/forbidden/normalization). Minimum
fields: scenario id, question, real log path/resource
identity, source snapshot, mechanism family, transition
intent(s), diagnostic window expectations (A then B),
required source anchors, required command/setpoint
observations, numeric +20 relationship, numeric +7
relationship, ordered-window requirement, logged-text
corroboration if available, rival exclusions, strength =
PROVEN, unavailable evidence with classifications,
forbidden claims. No Takeoff-specific framework.

## 27. K1–K15 acceptance test plan

Seams first (derivation helpers → scope → selector →
acceptance); one vertical slice per cycle:

- K1 real fixture preflight: Takeoff ULog exists at the
  provisioned path; source snapshot matches; required
  signals surfaced (triplet, mission item, nav_state,
  home, takeoff text).
- K2 initial window derivation: deterministic early
  takeoff-target window from the real log.
- K3 later window derivation: deterministic later
  mission-target/resume window from the real log.
- K4 ordered windows: early window precedes later window,
  exact fixture timestamps.
- K5 initial target numeric relation: initial target ≈
  home-at-command + MIS_TAKEOFF_ALT (exact values).
- K6 later target numeric relation: later target ≈
  home-at-command + mission relative altitude, or the
  strongest deterministic equivalent per §8.
- K7 source takeoff-altitude mechanism: represented
  source/DAG anchors for the minimum takeoff-altitude
  computation (§4, §10).
- K8 waypoint retention/resume grounding: strongest
  deterministic source/log evidence per §9.
- K9 temporal mechanism discrimination: initial and later
  commands belong to ordered, materially different
  phases/mechanisms; per-window evaluation (§19); no
  source-order tiebreak.
- K10 logged-text corroboration: `Takeoff to 20.0 meters
  above home` present as observation corroboration only
  (required only if the fixture text is stable; §13).
- K11 rival exclusion: reject supported rivals A/B/C
  only where evidence permits (§15).
- K12 benchmark contract: strength = PROVEN, unavailable
  evidence classified, forbidden overclaims pinned.
- K13 live pipeline status isolation: no forced
  `confirmed`/high confidence.
- K14 proof/source isolation: WS1/WS2/replay/checkpoint/
  proof semantics unchanged.
- K15 full regression: temporal + TECS + RTL acceptance
  remain GREEN.

## 28. Timestamps and source lines

Pin exact values/lines only after parsing/re-verifying
the real fixture and pinned snapshot during TDD. The
reminders in §§4–5 are starting points, never golden
values. Production gets no case timestamps.

## 29. Case-specific code guard

Production must contain no new MIS_TAKEOFF_ALT, takeoff,
waypoint, mission-command, +20/+7, or case-timestamp
diagnostic policy. Case values belong in
fixture/sidecar/tests. Enforce via the existing
case-specific-term guard pattern.

## 30. Regression

W3C TDD must preserve `tests/test_temporal_qualification.py`,
`tests/test_acceptance_tecs.py`, and
`tests/test_acceptance_rtl.py` with no semantic changes to
completed benchmarks.

## 31. Proof isolation

No changes to `branches_verified`, coverage,
applicability, checkpoint, T3, T5, T6B, Gate-B, R1, proof
stores, or retirement.

## 32. Stop gates

- STOP A: real accepted Takeoff ULog unavailable through
  the repository test-resource mechanism.
- STOP B: logged data cannot deterministically identify
  early and later target windows.
- STOP C: current `EvaluationScope.windows` cannot express
  the required ordered windows.
- STOP D: initial +20 target cannot be tied
  deterministically to the source takeoff-altitude
  computation.
- STOP E: later ~7 m target cannot be grounded as mission
  waypoint/resume behavior strongly enough for the
  accepted claim.
- STOP F: Takeoff source candidates remain
  mechanism-different and temporally indistinguishable.
- STOP G: mechanism equivalence is required but cannot be
  established.
- STOP H: only source/insertion order could distinguish
  candidates.
- STOP I: implementation requires new production
  state-history / phase / DAG / replay architecture.
- STOP J: acceptance would require changing confidence or
  proof authority.

On any gate: do not widen scope. Report the missing
evidence and recommend the narrowest appropriate skill:
`$wayfinder` for representation/evidence-location
questions, `$architecture-critic` for genuine
ownership/model decisions.

## 33. Explicit exclusions

W3A/W3B redesign; Takeoff fixtures beyond the single
accepted log/sidecar; Airspeed representation or
acceptance; logged-message event-source machinery beyond
text corroboration; new benchmark frameworks;
numeric-linkage ownership; live-LLM behavior changes;
performance work; fixture provisioning infrastructure
beyond the Takeoff log/sidecar.
