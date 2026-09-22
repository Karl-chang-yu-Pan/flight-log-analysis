# Airspeed load-factor acceptance spec

Deterministic acceptance for the accepted Airspeed benchmark, driven
by the landed W3A temporal causal-qualification machinery
(ADR-0004, `docs/temporal_causal_qualification_spec.md`) and the
RTL/TECS/Takeoff acceptance conventions
(`docs/deterministic_acceptance_spec.md`). W3A, W3B, and W3C are
CLOSED; this spec adds acceptance only and reopens nothing. Out of
scope: W3A/W3B/W3C redesign, Takeoff/TECS/RTL re-evidence,
numeric-check ↔ source-operation linkage ownership, new graph or
view types, new persistent temporal models, new replay capability,
new state-history architecture, benchmark-strength upgrades.

Status: specification only. No implementation, no fixtures, no tests.

## 1. Accepted benchmark truth (frozen)

Do NOT rediagnose, upgrade, or downgrade this case unless
deterministic repository evidence directly contradicts it.

Accepted question:

```text
Why did equivalent_airspeed_sp reach approximately 23.7 m/s,
above the 21 m/s trim value, during the approximately 50-degree
banked turn, instead of remaining at trim or being explained by
mission, wind, or another persistent command?
```

Accepted conclusion:

```text
Above-trim equivalent airspeed setpoint excursions are best
supported by bank/load-factor adaptation in
FixedwingPositionControl.

The adapted minimum airspeed rises with bank/load factor and
binds the commanded airspeed setpoint during the excursion.

Mission-command, wind-scaling, and weight-scaling alternatives
are deterministically inactive or weaker for this event.
```

Benchmark strength: BEST_SUPPORTED. This is intentional, not a
tooling gap to be closed by fiat (see §18). Do NOT specify
PROVEN.

## 2. Why BEST_SUPPORTED is intentional

Deterministically supported (acceptance must establish all of
these):

```text
peak numeric reconstruction
load-factor/minimum-airspeed binding
source mechanism
exclusive publication writer
setpoint-roll coincidence
wind rival exclusion
mission-command rival exclusion
weight-scaling inactivity
trajectory fingerprints
```

Not fully reconstructed (acceptance must disclose, never hide):

```text
slew-controller entry state at arbitrary windows
per-sample NPFG benign-branch proof
per-cycle dt / complete trajectory history
```

Therefore:

```text
strongest surviving explanation = load-factor adaptation

but

full-loop execution proof = unavailable
```

Missing full-loop proof is not contradiction. Do NOT treat it
as negative evidence, and do NOT convert it into false
confidence.

## 3. Real fixture (provision in TDD, never fabricate)

The accepted real log is `airspeed-load-factor.ulg`. A
legitimate copy is expected at the repository's gitignored
upload/test-resource path following the RTL/TECS/Takeoff
pattern (historical location:
`uploads/8a8aa57fedf94caf833ab8b4ade11d25/airspeed-load-factor.ulg`).
TDD must provision it into that mechanism, pin its hash in
acceptance, and fail loudly if the fixture is missing
(missing-fixture error, same convention as RTL A1 / TECS
T1-analog / Takeoff K1). Do NOT synthesize or substitute data.
Do NOT commit the ULog.

## 4. Source snapshot

Use the pinned source snapshot `1dacb4cdef2d7145754fc788fa8dc482eed74b40`
(the same pin as RTL/TECS/Takeoff acceptance). Acceptance must
fail loudly if the source checkout is missing or mismatched
(rev-parse check, same convention).

## 5. Corrected source mechanism

The historical shorthand ("trim times load factor") must NOT be
carried forward. The pinned implementation, re-verified against
the snapshot, is:

```text
load_factor_from_bank_angle
    = 1 / cos(attitude_setpoint.roll_body)

weight_ratio
    = 1.0 for this log (§7)

calibrated_min_airspeed
    *= sqrt(load_factor_from_bank_angle * weight_ratio)

airspeed setpoint
    = constrain(requested_setpoint,
                adapted_minimum,
                FW_AIRSPD_MAX)
```

The load-factor formula scales the MINIMUM airspeed, not trim
directly. The adapted value then passes through:

```text
slew controller (SlewRate member, ASPD_SP_SLEW_RATE)
→ NPFG airspeed nominal/reference path
→ tecs_update_pitch_throttle(... airspeed_sp ...)
→ tecs_status_publish(...)
→ tecs_status.equivalent_airspeed_sp
```

Re-pin exact symbols/sites in TDD; the reminders below are
starting points, not golden values:

```text
FixedwingPositionControl.cpp: adapt_airspeed_setpoint
    (load-factor term, weight term, wind gate, ground-speed
    undershoot gate, constrain, slew forced/rate logic)

FixedwingPositionControl.cpp: control_auto_position
    (mission path: adapt with FW_AIRSPD_MIN, then NPFG
    overwrite of the local target)

FixedwingPositionControl.cpp: tecs_update_pitch_throttle
    (passes airspeed_sp through to publication)

FixedwingPositionControl.cpp: tecs_status_publish
    (sole writer of tecs_status.equivalent_airspeed_sp)

SlewRate.hpp: update / setForcedValue / getState

npfg.cpp: refAirVelocity → airspeed_ref_ norm
```

## 6. Publication ownership

`tecs_status.equivalent_airspeed_sp` has a sole repository
writer at the pinned `tecs_status_publish` site in
`FixedwingPositionControl.cpp`. TDD must re-pin exact
file/symbol/source location (file + symbol + containing
content, never exact line equality).

The `airspeed_sp` value is passed into TECS by value. TECS
internal load-factor handling (`set_load_factor` from measured
euler roll, `update`) is a separate consumer and must NOT be
confused with the source of the published field.

Acceptance must explicitly distinguish:

```text
published EAS setpoint path (adapt → slew → NPFG → publish)
```

from:

```text
TECS internal load-factor logic (measured-roll consumer)
```

## 7. Accepted parameter evidence

The real fixture deterministically provides (TDD re-pins exact
values from the log; reminders only):

```text
FW_AIRSPD_TRIM = 21.0
FW_AIRSPD_MIN  = 19.0
FW_AIRSPD_MAX  = 35.0
FW_WIND_ARSP_SC = 0.0
FW_GND_SPD_MIN = 5.0
```

Do NOT specify historical `FW_WGT_SCA`: it does not exist in
this snapshot or log. The real weight semantics are
`WEIGHT_BASE` / `WEIGHT_GROSS`, absent from the log and taking
defaults of `-1.0`, which fail the positive/epsilon scaling
gate in source, yielding `weight_ratio = 1.0`. TDD must verify
this from pinned source/defaults rather than hard-code the
conclusion. A historical-correction note in rationale is
allowed; `FW_WGT_SCA` must not appear in acceptance logic.

## 8. Verified log facts (re-derive exact values in TDD)

Measured from the real fixture; TDD re-parses and pins exact
values rather than copying these reminders:

```text
equivalent_airspeed_sp peak ≈ 23.698446 @ ≈297.961 s
    (296 tecs_status samples; observed range ≈19.985–23.698)

attitude-setpoint roll_body ≈ 0.8727 rad ≈ 50.00°
    at effectively the same timestamp as the peak (dt ≈ 0)

measured vehicle roll ≈ 44.28° at the peak sample
    (corroboration only; NOT the formula input)

nav_state 3 (mission) at peak; landed = 0

position_setpoint_triplet cruising_speed = -1.0
    (current and previous)

wind estimate present and healthy (≈0.67 N, ≈0.59 E m/s)
    but unscaled (see §11)

local horizontal velocity ≈ 29 m/s at peak
    (vs FW_GND_SPD_MIN = 5)

excursion shape: fast rise tracking the adapted minimum,
    flat peak while roll holds 50°, fall at ≈ -1.000 m/s/s
    (== ASPD_SP_SLEW_RATE), return toward trim
```

## 9. Roll input (critical)

The load-factor formula consumes `_att_sp.roll_body`, i.e.
attitude SETPOINT roll. It does NOT use measured vehicle
roll. Acceptance must align the setpoint-roll sample to the
EAS peak and must pin that measured roll is NOT substituted
(assert the distinction; measured attitude stays
corroborative).

## 10. Core numeric acceptance

Acceptance verifies deterministically from logged values under
the verified `weight_ratio = 1`:

```text
expected = FW_AIRSPD_MIN * sqrt(1 / cos(attitude_setpoint_roll))
```

Accepted instance: `19 * sqrt(1 / cos(50°)) ≈ 23.698445`
against observed `23.698446…`, i.e. agreement at float32
serialization scale. TDD must derive exact roll and setpoint
values from the fixture (never blindly pin rounded 50°) and
justify tolerance from float32 conventions (RTL 0.02 / TECS
0.001 precedent; tight enough to discriminate the ~13 m
separations in play, loose enough for serialization noise).

## 11. Binding proof (load-bearing)

Numeric match alone is not enough. Acceptance must verify:

```text
requested/fallback airspeed < adapted minimum < FW_AIRSPD_MAX
```

therefore:

```text
constrain(requested, adapted_min, max) = adapted_min
```

For this event: requested fallback ≈ trim = 21 < adapted
minimum ≈ 23.698 < max = 35, so the adapted minimum binds.
Mission `cruising_speed = -1.0` falls back to trim per pinned
source; trim cannot independently explain 23.698.

## 12. Rival exclusions

Rival A — trim should remain fixed: rejected because the
adapted minimum exceeds trim and `constrain` therefore raises
the command (§11).

Rival B — mission requested ~23.7: rejected using
`cruising_speed = -1` plus pinned fallback-to-trim source
semantics plus the binding proof (effective request ≈ 21).

Rival C — wind scaling caused it: rejected using
`FW_WIND_ARSP_SC = 0` plus the exact disabled source gate
(`> FLT_EPSILON` fails). Classify wind scaling INACTIVE, not
merely unlikely.

Rival D — weight scaling caused it: rejected using
absent/default weight parameters and the inactive gate
(`weight_ratio = 1.0` via §7). Classify INACTIVE.

Rival E — measured bank alone explains it: clarified, not
merely rejected — source consumes attitude-setpoint roll;
measured roll (≈44.28°, relation gives ≈22.45) is
corroborative but is NOT the formula input. Acceptance must
assert the correct input.

Rival F — persistent unrelated 23.7 command: rejected only to
the degree deterministically supported — excursion correlates
with the load-factor/bank shape, the falling edge follows the
slew fingerprint, and no logged persistent command holds that
value. Do not overclaim beyond the evidence.

## 13. Ground-speed undershoot

Pinned source gates the undershoot path on `!_wind_valid`,
`!in_takeoff_situation`, and body-forward velocity below
`FW_GND_SPD_MIN`. At peak the logged horizontal velocity is
≈29 m/s (far above 5) and the requested setpoint sits below
the adapted minimum regardless. TDD must determine from
pinned source plus the exact sample whether the path is
inactive and assert inactivity only if deterministic;
otherwise classify it honestly as residual uncertainty. Do
not overclaim.

## 14. Takeoff/landing gate

At peak: `nav_state = 3`, `landed = 0` — normal mission
operation. The active auto-position adaptation call uses the
`in_takeoff = false` default; takeoff/landing
adjusted-minimum paths belong to other modes and are not the
active explanation for this sample. Specify deterministic
tests showing this from mode/landed evidence plus the
call-site default. Do NOT introduce a phase model.

## 15. Temporal window

Use existing W3A semantics only. The acceptance window is the
`equivalent_airspeed_sp > trim` excursion span around the
peak, expressed through the existing questioned-condition
seam (signal Hint + comparison + reference over logged
signals — the backward-compatible, transition-free form per
the temporal spec). An optional mission/nav transition may
provide surrounding context. Do NOT invent a new production
event type.

The window bounds the above-trim excursion for numeric
checks, trajectory fingerprints, and rival evidence. It is
not required to prove exact internal writer identity.

## 16. Slew-controller behavior (BEST_SUPPORTED boundary)

Three trajectory fingerprints are available from
fixture/source (verify in TDD only those deterministically
reconstructible; do NOT require arbitrary-window slew entry
state):

```text
1. peak/flat region: slew state equals the adapted minimum
   (no-lag equality at peak)

2. falling edge: ≈ -1.000 m/s/s, matching ASPD_SP_SLEW_RATE

3. rising behavior: setForcedValue(minimum) when prior slew
   state is below the newly adapted minimum
```

## 17. Converged anchors

Samples where published EAS equals the instantaneous
computed adapted/clamped input may act as deterministic
convergence anchors supporting local trajectory checks. Do
NOT treat anchors as generic replay/state-history
infrastructure; they belong only to acceptance analysis.

## 18. NPFG treatment

The value path includes `_npfg.setAirspeedNom(...)` and
`_npfg.getAirspeedRef()`. Peak value is strongly consistent
with identity/benign NPFG behavior (final 1e-6 agreement
bounds any peak-sample deviation to serialization scale).
Complete per-sample NPFG branch proof was not reconstructed.
Distinguish the supported peak-identity claim from the
unavailable whole-excursion branch proof; classify the
latter DISCRIMINATING.

## 19. `_eas2tas` treatment

The conversion factor cancels through the NPFG set/get
round-trip for the relevant EAS relationship. TDD must
re-verify exact source structure; if confirmed, classify the
exact `_eas2tas` value NON_DECISIVE.

## 20. Evidence roles (ADR-0004, unchanged)

```text
source evidence:
    adapt computation / slew rule / NPFG path / gates (§5)

observation evidence:
    EAS series / setpoint roll / params / mode / wind (§8)

temporal evidence:
    above-trim excursion span (§15)

numeric evidence:
    load-factor reconstruction + binding (§§10–11)
```

No new universal evidence model.

## 21. Replay role

Replay is OPTIONAL. Do NOT require or build new replay
implementation. Working classification: minimum computation
replay-evaluable in principle; full stateful trajectory
incomplete without state anchors. If existing replay
naturally covers a bounded calculation, acceptance may use
it; otherwise `replay N/A / partial` is acceptable (Takeoff
precedent).

## 22. DAG role

No Airspeed DAG stage currently exists. Acceptance may use
pinned source assertions, real log values, deterministic
arithmetic, and existing inventory/source structures without
a real DAG where deterministic acceptance conventions permit
(RTL/TECS/Takeoff precedent for source-grounded claims with
empty live refs). Do NOT fabricate DAG results. If TDD
naturally constructs the relevant existing generic DAG with
no production changes, its source assertions may be used —
but a DAG is not a requirement.

## 23. Source refs

Live report/source-ref availability is unknown. Do NOT
require report CodeRefs. Acceptance may ground the mechanism
through the pinned snapshot. No fabricated refs.

## 24. BEST_SUPPORTED contract

The sidecar must record `strength = BEST_SUPPORTED` and
clearly separate the supported mechanism (§2, first list)
from missing discriminating evidence (§2, second list). Do
not convert missing evidence into false confidence. Missing
internal state alone is not contradiction.

## 25. Missing-evidence classifications

DISCRIMINATING (at minimum):

```text
slew-controller _value at arbitrary window starts
per-sample NPFG branch proof
per-cycle dt / complete trajectory reconstruction
```

NON_DECISIVE (at minimum):

```text
TECS internal state
takeoff/landing state-machine internals beyond the observed
    inactive gate
_eas2tas exact value if cancellation is source-confirmed
mission-file-style provenance
```

CRITICAL: expected none. If TDD discovers a genuinely
CRITICAL gap, acceptance must stop rather than preserve
BEST_SUPPORTED by fiat.

## 26. Mechanism equivalence

Do NOT use the TECS mechanism-equivalent-writer rule unless
TDD discovers an actual writer tie requiring it. The
publication writer is unique, and exact writer equivalence
is unnecessary for this claim. The spec explicitly
discourages copying the TECS pattern here: if unresolved
alternatives would materially affect the numeric mechanism,
equivalence is invalid.

## 27. Acceptance sidecar

Use existing RTL/TECS/Takeoff schema conventions
(scenario_id/question/inputs/expected/strength/unavailable/
forbidden/normalization). Minimum fields:

```text
scenario id, question, fixture (+ hash requirement),
source snapshot, mechanism family, strength BEST_SUPPORTED,
excursion/window intent, peak timestamp, observed peak,
attitude-setpoint roll, FW_AIRSPD_MIN/TRIM/MAX, wind
parameter, mission/cruising input, weight defaults/gates,
numeric reconstruction, binding rule, publication owner,
trajectory fingerprints, rival exclusions, DISCRIMINATING
missing evidence, NON_DECISIVE missing evidence, forbidden
claims
```

Do not create another sidecar framework.

## 28. Forbidden claims

Acceptance must prohibit claiming:

```text
benchmark is PROVEN
measured roll is the formula input
trim itself is directly multiplied by load factor
FW_WGT_SCA controls this snapshot
TECS internal load-factor handling writes the published EAS
    setpoint
full trajectory state is reconstructed
every NPFG branch is proven
slew entry state is known at arbitrary times
live pipeline confidence equals benchmark strength
missing state is contradictory evidence
source existence alone proves runtime execution
```

## 29. Case-specific code guard

Production must contain no new airspeed/load-factor policy,
parameter literals, or case timestamps. Case values belong
in fixture/sidecar/tests. Enforce via the existing
case-specific-term guard pattern.

## 30. Regression

Airspeed TDD must preserve `tests/test_temporal_qualification.py`,
`tests/test_acceptance_tecs.py`,
`tests/test_acceptance_rtl.py`, and
`tests/test_acceptance_takeoff.py` with no semantic changes
to completed benchmarks.

## 31. Proof isolation

No changes to `branches_verified`, coverage, applicability,
checkpoint, T3, T5, T6B, Gate-B, R1, proof stores,
retirement, confidence, or confirmation. Strength,
confidence, and proof stay untouched: BEST_SUPPORTED never
sets confidence/confirmed by itself.

## 32. TDD plan (seams first)

Seams, highest first, all existing: real-log
parsing (production inventory + raw pyulog ground truth) →
pinned-source assertions (file + symbol + content) →
existing W3A questioned-condition windows → sidecar
contract. No new seams; no production seams touched.

- A1 fixture/source preflight: real Airspeed ULog exists at
  the provisioned path with pinned hash; source snapshot
  matches; required signals/topics surfaced.
- A2 published-field ownership: sole source writer for
  `tecs_status.equivalent_airspeed_sp` re-pinned; TECS
  internal load-factor logic distinguished (not the source).
- A3 peak extraction: exact peak EAS + timestamp parsed
  from the real fixture (re-derive, never trust reminders).
- A4 correct roll input: nearest/exact attitude-setpoint
  `roll_body` aligned to the peak; measured roll pinned as
  non-substitute.
- A5 parameter state: MIN/TRIM/MAX/WIND verified from the
  log; weight defaults/gates verified from pinned
  source/defaults.
- A6 load-factor numeric reconstruction:
  `MIN * sqrt(1 / cos(setpoint_roll))` vs peak, with
  justified float tolerance.
- A7 constrain binding: requested/fallback < adapted
  minimum < maximum, therefore the minimum binds.
- A8 mission-command rival: cruising input/default
  behavior cannot produce the peak independently.
- A9 wind rival: wind-scaling gate inactive (INACTIVE).
- A10 weight rival: weight ratio resolves to 1.0 through
  the source/default gate (INACTIVE).
- A11 normal-mode gating: mission/airborne state; no
  takeoff/landing adjusted-min path active at peak.
- A12 excursion window: above-trim span derived via
  existing temporal/questioned-condition semantics; no new
  production event type.
- A13 trajectory fingerprints: peak convergence, fall
  rate ≈ −ASPD_SP_SLEW_RATE, available converged anchors;
  no unknown initial slew state required.
- A14 rival bundle: all deterministic exclusions together,
  without collapsing missing DISCRIMINATING evidence into
  proof.
- A15 BEST_SUPPORTED contract: sidecar strength,
  missing-evidence classifications, forbidden claims.
- A16 live status/proof isolation: no forced
  confidence/confirmed/proof changes.
- A17 existing acceptance regression: RTL + TECS +
  Takeoff + temporal suites remain GREEN.
- A18 full regression: no test disappearance (count
  against the post-W3C baseline plus new Airspeed tests).

## 33. Stop gates

- STOP A: real accepted Airspeed ULog unavailable or
  identity cannot be pinned.
- STOP B: correct attitude-setpoint roll input cannot be
  deterministically aligned to the EAS peak.
- STOP C: adapted-min formula cannot reconstruct the
  observed peak within justified tolerance.
- STOP D: constrain binding cannot be established.
- STOP E: mission/wind/weight rivals cannot be evaluated
  strongly enough for BEST_SUPPORTED.
- STOP F: sole publication writer/source path cannot be
  grounded.
- STOP G: missing state becomes CRITICAL rather than
  merely DISCRIMINATING.
- STOP H: acceptance requires reconstructing arbitrary
  slew-controller history.
- STOP I: acceptance requires new DAG/replay/state-history
  architecture.
- STOP J: acceptance requires changing benchmark strength,
  live confidence, or proof authority.

On any gate: do not widen scope. Report the missing
evidence and recommend the narrowest appropriate skill:
`$wayfinder` for representation/evidence-location
questions, `$architecture-critic` for genuine
ownership/model decisions.

## 34. Explicit BEST_SUPPORTED success condition

GREEN is possible when all of the following hold:

```text
real fixture and source pinned

correct setpoint-roll input aligned to peak

adapted minimum numerically reconstructs peak

constrain binding established

publication writer uniquely grounded

mission-command rival excluded

wind scaling inactive

weight scaling inactive

normal mission/airborne mode established

trajectory fingerprints support adaptation

remaining hidden-state gaps disclosed as DISCRIMINATING
    rather than contradictory

no CRITICAL evidence gap remains

benchmark remains BEST_SUPPORTED

no production/proof/status architecture changed
```

## 35. Explicit exclusions

W3A/W3B/W3C redesign or re-evidence; Takeoff/TECS/RTL
fixtures beyond the single accepted Airspeed log/sidecar;
logged-message event-source machinery; new benchmark
frameworks; numeric-linkage ownership; live-LLM behavior
changes; performance work; fixture provisioning
infrastructure beyond the Airspeed log/sidecar; PROVEN
upgrade; full-trajectory state reconstruction.
