# DAG Construction Completeness Specification

Status: draft for review. No implementation. No authority changes.
Next: `$grill-with-docs` (see §20).

## 1. Problem Statement

Paid flight-log acceptance runs are being used as graph debuggers: a
calibration run returns `sufficient: false`, and only afterwards does
offline analysis discover that deterministic source-dependency edges
were never constructed (missing caller bindings, unrepresented state
transfer, unreachable writers). Each failure then produces a narrow
parser patch for one scenario, followed by the next scenario's patch.

The cross-oracle audit (RTL, TECS, Takeoff, Airspeed) established that
this is not a series of isolated bugs but missing capability families
in production DAG construction: plain-call binding and cross-call
state transfer are unrepresentable today, multi-terminal composition
is unscoped (synthesis-first per §4C), while temporal/window state
and history persistence are explicitly deferred. Meanwhile slicing,
rendering, and stop authority are complete and fail-closed — the
system correctly refuses, it just cannot yet build several accepted
evidence paths.

This spec defines what "construction complete" means, which generic
capabilities it requires, how each is tested before any corpus claim,
and the gate that must pass before any future paid acceptance run.

## 2. Corrected Completeness Terminology

The unqualified phrase "DAG complete" is retired and must not appear in
decisions, reports, or commit messages. The following levels are
distinct; a claim at one level never implies any other:

- **Primitive-vocabulary completeness**: the dependency kinds the graph
  can represent exist as primitives. (Current verdict: incomplete —
  plain-call binding, member mapping, and cross-call state transfer
  have no edge kind; cross-terminal composition is unscoped, see
  Family C.)
- **Extraction completeness**: the profiler emits the facts each
  capability needs. (Current verdict: partial — multiline and
  braced-initializer extraction now generic; member-to-aggregate
  mapping missing.)
- **Edge-construction completeness**: required dependency edges are
  created during the deterministic build. (Current verdict: incomplete —
  synthetic edge creation is gated on helper-call classification.)
- **Traversal/reachability completeness**: edges are traversable in the
  direction each explanation requires. (Current verdict: incomplete —
  backward-only, single-root.)
- **Slicing completeness**: the production slice retains all modeled
  evidence without starving the judge. (Current verdict: complete and
  conservative for modeled edges.)
- **Rendering completeness**: the judge packet exposes retained
  evidence with explicit truncation tails. (Current verdict: complete.)
- **Authority completeness**: stop/continue verdicts are fail-closed
  and well-specified. (Current verdict: complete and mature.)
- **Acceptance-corpus constructibility**: every accepted oracle chain
  reconstructs offline through the production path. (Current verdict:
  incomplete — TECS blocked; Takeoff/Airspeed state links unbuilt;
  RTL proven opt-in tree-sitter only.)

## 3. E0–E8 Support Model (Preserved)

A capability is NOT supported merely because a parser extracts it:

- E0 source fact exists; E1 extracted; E2 canonicalized;
- E3 graph node created; E4 dependency edge created;
- E5 traversable in required direction; E6 retained by production slice;
- E7 present in judge rendering; E8 usable by authority/replay as required.

A capability counts as supported only if it reaches the highest stage
its scenario requires. "Writer parsed" (E1) is never equivalent to
"writer usable" (E7/E8).

## 4. Capability Families

### Family A — Plain-Call Binding (audit classes B/F)

Required: plain-call actual→formal, plain-call return→caller, member
actual→formal, aggregate/braced-initializer member mapping.

Constraint: the architecture already creates analogous synthetic
bindings for recognized helper calls (formal←actual, return bindings,
receiver-member inputs, pointer-output caller targets). The spec
generalizes that machinery; a second incompatible call model is
forbidden. Governing requirement: **a call's semantic dependency edges
must not depend on whether the callee happened to be classified as a
special helper**, subject to bounded, validated resolution rules
(exact callable identity, arity/default compatibility, scope
discipline). The existing helper tests become the regression floor for
the generalized semantics.

Admission rules (normative, closing review gaps):

- A plain call is admitted only by resolving to exactly one
  declaration-proven callable through the existing dispatch
  vocabulary (receiver type lineage, explicit qualification, lexical
  owner, or ownerless free call). Dynamic dispatch, function
  pointers, and unresolved receivers fail closed to unresolved —
  never to a guessed callee.
- Overloads resolve by exact identity first, then arity/default
  compatibility. Same-arity ambiguity across overloads fails closed;
  the design must state the ambiguity rule rather than pick textually.
- Actual→formal and return→caller are separate edge kinds with
  separate tests; return propagation is not implied by argument
  binding.
- Member mapping requires an explicit per-construct support table
  stating, for each of positional init, designated init, nested
  aggregates, partial initializers, default-initialized fields,
  anonymous structs/unions, aggregate references/aliases, copy
  construction, and temporaries: supported with its identity rule,
  or fail-closed with the exact unresolved marker. Unlisted
  constructs fail closed by default.

### Family B — Cross-Call State Transfer (audit class F)

Required, with no scenario-specific state names: object/member state
mutation across calls; callee member write → caller-visible object
state; later read of mutated state; stateful helper/object evolution.
Current condition (parsed call, possibly parsed writer, no E4 edge) is
the acceptance criterion's negation. The spec must define identity and
alias rules strong enough to keep transfer sound: exact storage
identity, receiver/consuming identity survival through projection,
reference/pointer containment directionality (an aggregate actual may
contain a member effect; a field actual never proves its aggregate).
Required by TECS, Takeoff, and Airspeed.

Versioning rules (normative, closing review gaps — without these,
Family B smuggles persistence, i.e. Family E, through the back door):

- Object state is versioned per proven mutation: a state edge carries
  the mutating call site, and a later read consumes the latest
  preceding version only.
- A write supersedes prior versions of the same member; an unknown or
  unresolvable intervening call invalidates (never preserves) the
  affected member versions.
- Receiver identity must be declaration-proven, not merely
  same-spelled; two aliases bind the same object only through a
  source-proven alias/containment chain, never by name similarity.
- Partial mutation is member-granular: mutating one member leaves
  sibling versions intact only where the write set is proven;
  otherwise the whole-object version invalidates.

### Family C — Multi-Terminal / Cross-Root Composition (audit class F,
rescoped by grill review)

TECS showed an intermediate root whose accepted writers are downstream
— but the oracle root alone already carries both writers and the
shared helper mechanism (proven on both backends), and the judge
already receives one render per candidate terminal. The spec therefore
distinguishes, normatively:

- **ROOT-SELECTION PROBLEM** (proven for TECS): the paid-run root was
  an intermediate; the oracle root suffices structurally. Fixed by
  terminal-selection contract work and the §16 gate's known-roots
  requirement, not by new graph machinery.
- **TRUE MULTI-ROOT COMPOSITION**: verdicts combining sufficient
  evidence across renders (e.g. mechanism in one slice, rival
  exclusion corroborated by another).

Default scope is multi-render verdict/report synthesis over per-root
DAGs that each satisfy today's sufficiency rules independently
(conjunction = aggregation of existing verdicts, no new authority
kind). Graph-level cross-root edges require a prior proof obligation:
name the specific accepted fact that per-root DAGs plus multi-render
synthesis cannot express, and only then design the link kind.
Increasing backward depth is explicitly not the answer.

### Family D — Temporal/Window State (audit class H, required)

Takeoff requires phase/window discrimination. The existing C4
per-window deferral is preserved as scheduling, not silently dropped:
the spec defines what state is window-specific, how two ordered
windows are distinguished, how writer/state identity interacts with
windows, and what can and cannot be inferred across windows.
Preserved invariants: ordering plus endpoint equality never implies
continuity; existing fail-closed rules are not weakened. The
D-restriction (window qualification of already-modeled edges) versus
D-state (window-specific state, requiring Family B facts) split is
normative per §18; attach point (node, edge, observation) is open
design question Q4.

### Family E — History/Runtime-State Persistence (audit class H, required)

Takeoff and Airspeed require behavior current static dependencies
cannot represent. This family may land in a later milestone, but an
unimplemented Family E forces the final status to remain
"acceptance corpus NOT fully constructible" — it must never be
excluded to manufacture a green label. Preserved distinction:
retained evidence is not runtime-state persistence.

Persistence authorization rule (normative shape; the design fills the
producer): a persistence link is admitted only on positive,
deterministically produced continuity evidence bound to an explicit
window pair — a logged value-continuity relation, a derived
continuity check with stated tolerance and scope, or a source-proven
retention statement naming the storage identity. Ordering plus value
similarity, absence of observed writers, and unchanged-looking state
never authorize persistence. Reset/reinitialization source facts
terminate persistence links. Replay may corroborate a persistence
link but never creates one.

## 5. Conditional Capabilities (Explicitly Preserved, Not Redesigned)

Parameter-value grounding, branch feasibility, signal binding, and
numeric replay are conditional on runtime/log inventory, not absent
construction primitives. The spec describes their required inputs and
failure behavior only; no replacement subsystems without evidence.
The consolidated product-domain boundary lives in §24.

## 6. Production Ownership and Functions

Construction flows through the existing seam chain, unchanged in shape:
source snapshot → profiler facts (`extract_facts_for_file`, both parser
backends) → discovery fixed point (`discover_mechanism_dag`:
`validate_terminal`, `SourceExpansionResolver`, `binding_from_assignment`)
→ graph build (`build_mechanism_dag` / `_DAGBuilder`) → slice
(`evaluate_feasibility`, terminal-scoped projection) → rendering
(`render_discovery_compact`) → judge packet (`discover_with_judge`) →
STOP BEFORE MODEL. Families A–C land as builder/resolver edge semantics
plus fact-mapping rules; Families D–E land as first-class dependency
and requirement kinds flowing through the same seams. No new pipeline
stages.

## 7. Identity, Alias, and State Safety Rules

All new edges obey the existing identity regime (declaration-derived
source identity, storage identity, projection identity preserving the
aggregate chain, receiver/consuming identity, call-site scoping):
exact-symbol matching; scope-disciplined instantiation per invocation
path; no compatible-type inference; no topic-presence inference;
aggregate-containment directionality for reference/pointer effects.
Any binding that cannot prove these rules fails closed to unresolved,
exactly as today.

## 8. Multi-Root Composition Semantics

Default envelope: each root builds exactly as today (single backward
DAG, unchanged authority); verdict/report synthesis reasons over the
existing per-terminal renders. This needs no new graph machinery and
no new authority kind — conjunction is aggregation of existing
per-root verdicts, each of which must satisfy today's sufficiency
rules independently.

Graph-level cross-root links are designed only after the Family C
proof obligation (§4C) names an accepted fact that synthesis cannot
express. If that happens, the link design inherits these constraints:
shared vertex identity (same source-site identities merge; distinct
call instances stay distinct per the existing helper-instance rule);
links only where a source-proven relationship exists (call binding,
state transfer, or observation correspondence — never co-occurrence);
rendering exposes one explanation with per-root provenance.

## 9. Temporal and History Semantics

Windows remain evaluation-scope annotations owned outside source order.
Family D adds window-qualified dependency links (a writer/state fact
usable only within its active windows) without changing source
identity. Family E adds persistence links only on explicit evidence
(logged/derived continuity relations or equivalent records, per the
existing conservative contract) — never from ordering plus value
similarity. `state_alignment`/history requirements stay open until
such evidence exists.

## 10. Parser Backend Boundary

Production default stays `source_parser_backend="legacy"`; tree-sitter
stays opt-in. This project changes no default. Construction semantics
must be backend-independent: identical fact shapes in, identical edge
semantics out; backend differences are confined to extraction fidelity
and covered by parity tests. Every corpus claim names the backend
tested. The ADR-0005 target-vs-default mismatch is recorded for a
separate migration decision, out of scope here.

## 11. Seed-Selection Boundary

Two contracts, never conflated:

- **ORACLE-SEEDED CONSTRUCTION COMPLETENESS**: given a valid accepted
  root/terminal set, deterministic construction carries every
  load-bearing fact to the judge-input boundary. This is what the
  project proves.
- **NORMAL TERMINAL-SELECTION QUALITY**: whether the LLM seeder
  chooses those roots in production. Out of scope; this spec claims
  nothing about it.

## 12. Neighbor/Replay Clarification (Normative)

`pending_construction`, `ConstructionDemand`, `replay_dag_roots`,
coverage/search versions, and checkpoint reconstruction do NOT create
source-dependency edges and must never be cited as construction
completeness. Their roles: scheduling, materialization requests,
retention, authority, numeric replay. Only the Family A–C edge
semantics in §4 count toward construction.

## 13. Cross-Oracle Capability Matrix (Normative Witness Set)

| Capability | RTL | TECS | Takeoff | Airspeed |
|---|:---:|:---:|:---:|:---:|
| Local assignment / writer→reader flow | R | R | R | R |
| Multiline / braced-init extraction | · | R | · | · |
| Helper actual→formal / return, expansion | R | R | · | (r) |
| Plain-call actual→formal / return (Fam. A) | · | R | (r) | · |
| Member mapping (Fam. A) | · | R | · | · |
| Cross-call state transfer (Fam. B) | · | R | R | R |
| Multi-terminal composition (Fam. C) | · | R | (r) | · |
| Temporal/window state (Fam. D) | · | · | R | (r) |
| History/persistence (Fam. E) | · | · | R | R |
| Param / branch / signal / replay (conditional) | R | R | R | R |
| Multi-writer, slicing, rendering, authority | as today (complete) |

(R = required, (r) = partially required, · = not required.)

- **RTL**: opt-in tree-sitter path structurally sufficient; requires the
  §16 gate on the production default. No logic redesign.
- **TECS**: requires Families A–B plus conditional support, with
  Family C in synthesis scope (root-selection contract + multi-render
  synthesis; graph links only on proof); production
  conditions must never name TECS symbols or lines (fixture
  expectations stay in acceptance tests/sidecars).
- **Takeoff**: requires Families A (cross-function call binding for
  the takeoff-altitude mechanism), B, D, E plus grounding; existing
  K-tests/source-grep do not count as construction proof.
- **Airspeed**: requires math flow, binding, Families B and E; remains
  BEST_SUPPORTED until deterministic requirements justify more —
  this spec upgrades no strength.

## 14. Generic Test Strategy

Per family: small synthetic generic RED test → generic implementation
→ generic GREEN → corpus acceptance witness. Production code must not
contain RTL/TECS/Takeoff/Airspeed literals, fixture hashes, accepted
line numbers, or diagnostic questions (enforced by existing case-term
guard patterns). Scenario expectations live only in acceptance
tests/sidecars. Precedent: the multiline/braced-init unit tests
parameterized over both parser backends. The required generic
synthetic witness shapes live in §25.

## 15. Production Deterministic Acceptance Harness

One reusable offline harness (not four bespoke ones) running: source
snapshot → selected-backend profiler (production default required;
other backends optional per claim) → fact extraction →
production DAG construction → deterministic expansion/traversal →
production slice → production rendering → STOP BEFORE MODEL, with
provider/seeder/judge stubbed as appropriate and provider-call count
asserted zero. Runtime/log annotations (inventory, parameter values,
windows) are injectable inputs, explicit per run. It emits staged E0–E8
diagnostics per required fact (scenario, root, fact, last-green
E-stage, first-red E-stage, nearest reachable identity, missing edge
kind, source location, slice/render disposition). This harness is the
authoritative construction-completeness gate. The untracked TECS
evidence probe may inform its design but is not normative.

The harness runs in two modes: (a) oracle-seeded construction with
accepted roots, and (b) captured production-seed replay using the
exact recorded seeds of a prior paid run (e.g. from run event logs),
which pins the production input without solving seeder quality.
Mode (b) is required wherever a paid run previously occurred, so
oracle seeding cannot mask a production contract bug.

## 16. Paid-Run Gate (Project Rule)

NO PAID ACCEPTANCE RUN until the scenario passes its deterministic
production-path evidence-chain gate through E7 (and E8 wherever
authority/replay is required). The gate additionally requires, each
recorded with the run: the declared backend including an explicit
production-default-backend leg (claims may be backend-qualified, but
M7 needs the default leg); a clean commit; the known accepted
root/terminal set; every load-bearing unresolved symbol classified
(grounded, NON_DECISIVE per oracle, or explicitly open — never
silently dropped); and all conditional runtime inputs present (log
inventory, parameter values, windows), so E8 cannot pass vacuously.
A paid run tests LLM interpretation/selection of sufficient evidence;
it must never be used to discover parser defects, missing edges,
traversal gaps, slice omissions, or render omissions.

## 17. Milestones

- M0 Authority-safe — existing; fail-closed. No work.
- M1 Local/static construction — assignments, writers/readers,
  helpers, branches. Substantially existing; multiline/braced
  extraction closes it.
- M2 Call-binding complete (Family A).
- M3 Cross-call state complete (Family B).
- M4 Multi-root synthesis complete (Family C, synthesis scope unless
  the proof obligation forces link design).
- M5 Temporal/window state complete (Family D).
- M6 History/persistence complete (Family E).
- M7 Acceptance-corpus constructible — all required RTL/TECS/
  Takeoff/Airspeed chains green on declared backend(s).

No later milestone completes while an earlier required capability is
red. Families D–E may schedule after A–C, but M7 stays red until
they land.

Exit criteria (normative minima; each milestone exits only when all
hold on the declared backend(s)):

- M2: generic RED→GREEN suites for actual→formal, return→caller, and
  member mapping (including the §4A per-construct table) green on
  both backends; existing helper suites unchanged; TECS caller
  grounding RED resolved.
- M3: versioned state-transfer edges with invalidation semantics
  (§4B) proven generically; TECS writer-chain reachability resolved.
- M4: the Family C proof obligation discharged one way or the other
  — either multi-render synthesis rules specified and witnessed, or
  a named cross-root fact forcing link-kind design.
- M5: window-qualified links with the §4D definitions witnessed on
  Takeoff windows.
- M6: persistence authorization rule specified with its producer,
  witnessed on Takeoff/Airspeed state.
- M7: every required oracle chain green through its required
  E-stage, including the production-default-backend leg.

## 18. Implementation Dependency Graph

Validated shape (to be confirmed against code during design, not
assumed):

```text
A Call Binding
    ↓
B Cross-call State (needs call resolution + arg mapping)
    ↓ (per-root completeness for stateful facts)
C Multi-render synthesis (needs A; parallelizable with B
    given stable shared identity — no B dependency)

A (call structure) ········→ D-restriction (window qualification
                                of existing edges; independent)
B (state facts) ···········→ D-state (window-specific state)
D ························→ E History/Persistence
A–E ······················→ F Cross-oracle acceptance closure
```

Rationale: state transfer consumes call-binding identities;
synthesis needs complete per-root DAGs but no state machinery of
its own; window restriction applies to already-modeled edges while
window-specific state needs B's facts; persistence generalizes
windowed state. No parallelism is claimed beyond C∥B and
D-restriction independence; the design phase must confirm or revise
with `$wayfinder` evidence.

## 19. Migration and Compatibility

Families A–C extend edge semantics within the existing builder and
fact shapes; no schema change is anticipated, and any schema change
requires its own decision. Helper-call behavior is the regression
floor: generalization must preserve every existing helper test
outcome. Backend parity tests must pass on both backends for every
new capability. No flag or default changes ship inside this project.

## 20. Non-Goals (Scope Exclusions)

Prompt tuning; model changes; paid calibration; legacy
mechanism-discovery restoration; question-intent fast path; budget
threshold policy; report prose redesign; benchmark-specific
production hacks; parser-default migration; seeder quality work.

## 21. Open Design Questions (for `$grill-with-docs`)

1. Is generalizing helper-call instantiation to plain calls sound
   under the §7 identity rules, or does plain-call dispatch
   (virtuals, overloads, function pointers) need a narrower
   admission rule than §4A states?
2. What is the exact edge kind for cross-call member-state transfer,
   and how do repeated mutations compose without exploding vertices?
3. For the Family C proof obligation: name the accepted fact (if
   any) that per-root DAGs plus multi-render synthesis cannot
   express, or confirm synthesis scope and close the link-kind
   question.
4. Can window qualification reuse branch-feasibility machinery, or
   does Family D need a new link kind? On which element (node,
   edge, observation) does qualification attach?
5. Who produces the §4E persistence evidence deterministically, and
   does replay corroboration need new plumbing?
6. Does the §18 dependency graph match the actual code seams?
7. Would M7 as defined make the corpus genuinely constructible, or
   does any oracle need a capability outside Families A–E?
8. What is the §4A per-construct member-mapping table content for
   each listed C++ construct (supported rule vs. fail-closed
   marker)?

## 22. Measurable Exit Criteria

The spec is review-ready when §§1–21 are grilled and stable. The
project exits when: generic RED→GREEN suites per family; parity on
both backends; the §15 harness green per scenario through its
required E-stage; M7 declared with backend annotations; zero
scenario-specific production literals; existing suites green.

## 23. Appendix: Audit Provenance

Cross-oracle audit findings incorporated: E0–E8 model; capability
catalog and matrix; C4/history deferrals preserved; conditional
capabilities preserved; neighbor/replay non-construction roles;
oracle-seeded vs normal-selection distinction; traversal-direction
analysis (TECS intermediate-root contract mismatch; Takeoff
forward-temporal need); A–H classifications; "DAG complete" phrase
retirement. Prior investigation RED boundaries (caller grounding,
state transfer, parameter values, all pinned offline with stubbed
providers) are design evidence, not normative structure.

## 24. Supported Product Domain Boundary

The project capability is:

```text
Given a diagnostic question, a pinned reference-source snapshot, and
runtime/log evidence, reconstruct minimal sufficient source
dependency chains that explain the observed behavior, ground every
load-bearing link, eliminate rivals where evidence permits, and fail
closed with explicit gaps otherwise.
```

It is NOT whole-program C++ analysis. This section is the boundary
against which future construction-completeness claims are made, per
the hierarchy in §2 and the claim requirements in §22.

### SUPPORTED

Lexical/source identity; local value/data flow; writer→reader
dependencies; control dependencies; branch predicates; statically
resolved calls; cross-file/module dependencies; runtime observation
binding; numeric replay where evaluable.

### SUPPORTED ONLY WITH VERIFIED STATIC IDENTITY

Actual→formal binding; return→caller binding; receiver identity;
member/field mapping; state mutation; construction/initialization
mapping. Ambiguous identity fails closed — no synthetic edge without
a declaration-proven identity meeting the §7 rules.

### SUPPORTED ONLY WITH REQUIRED RUNTIME EVIDENCE

Branch feasibility; parameter/config values; window-specific
applicability; state persistence; runtime observation bindings.
Without the required runtime input, each fails closed to unresolved;
§5 governs inputs and failure behavior.

### CONDITIONAL

Global/static state, only when the writer set and identity can be
closed sufficiently; otherwise fail closed. Multi-root composition,
only when an accepted explanatory fact cannot be expressed through
correct root selection plus per-root evidence synthesis (§4C proof
obligation).

### FAIL-CLOSED

Indirect/function-pointer calls unless target identity is proven;
virtual dispatch unless implementation identity is statically or
otherwise conclusively established; ambiguous overloads; unproven
receiver identity; unproven aliases; unknown intervening mutations;
state crossing windows without continuity evidence; unevaluable
replay. Each fails closed to unresolved with its marker — never to a
guessed dependency, and never silently satisfiable.

### OUT OF SCOPE FOR THE SUPPORTED PRODUCT DOMAIN

Unless separately specified later: concurrency/thread
interleavings; undefined-behavior reasoning; reflection;
generated-code semantics beyond the available source snapshot;
external-library internals outside the admitted source boundary
(boundary transfer sites only, never library interiors).

Templates: angle-bracket syntax is tolerated where patterns name it,
but there is no instantiation semantics — constructs whose meaning
depends on template instantiation fail closed where semantic
identity cannot be established. Macros: simple object-like defines
are supported as facts; function-like and generated semantics are
supported only where definition-anchored projection proves identity,
and fail closed otherwise. Neither implies perfect
expansion/instantiation support.

## 25. Required Generic Synthetic Capability Witnesses

Every newly implemented semantic family first needs a small
source-shape acceptance witness proving the capability, before any
corpus claim. Required shapes, at minimum:

```text
ordinary call actual→formal binding
ordinary call return→caller flow
member mutation through a method
member mapping through aggregate/initializer construction
reference/alias mutation with proven identity
conditional writer with rival branch
state mutation across two calls
window-scoped writer usability
persistence requiring positive continuity evidence
indirect call that must fail closed
ambiguous overload that must fail closed
```

These tests contain no RTL, TECS, Takeoff, Airspeed, fixture-hash,
or benchmark-specific production assumptions. Per family:

```text
synthetic RED → generic implementation → synthetic GREEN
→ existing corpus witness
```

The synthetic witness proves semantic capability; the corpus witness
proves applicability to accepted scenarios. Neither replaces the
other.