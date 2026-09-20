# Helper causal source representative identity and ownership

Upstream helper computations that explain a reported mechanism are
represented in `source_refs` by at most one representative ref per
helper callable plus source call site (plus output discriminator
where one site yields multiple distinct returns), anchored at the
real helper-return source location. Helper refs never establish
sufficiency, and never affect coverage, applicability, stop
authority, branch verification, replay, or confidence beyond
satisfying generic source-grounding presence checks; temporal
discrimination belongs to timeline scope, not topology.

## Definitions

- **Terminal source writer**: an operation vertex with terminal
  marking whose write the report attributes to the selected
  mechanism.
- **Helper-return vertex**: the single per-call-site `__return__`
  operation through which a helper call's value re-enters the
  caller cone (private call-instance scope). Helper bodies do not
  materialize as individual DAG operations.
- **Helper call site**: the caller-side source location issuing
  the call (file, line, `call_source_site_id` where recorded).
  This is distinct from the callee-side helper-return source
  location used as the displayed anchor (see below).
- **Call instance**: one graph instantiation of a call site. Call
  instances share the call site; they never define report
  identity.
- **Helper representative**: the one `CodeRef` standing for a
  (helper callable, source call site, conditional output path)
  triple in `source_refs`.
- **Source-writer identity**: the stable writer key used by
  terminal dedup (file, range, symbol, expression, scope).
- **Helper representative identity**: `(helper_callable_id,
  call_source_site_id)`, plus the output/result-path
  discriminator (see Decision) whenever one callable+site pair
  yields multiple semantically distinct returns. The identity
  is caller-side; it is resolved to the helper-return vertex's
  callee-side file/line only for display anchoring. Never a
  vertex ID, never a call instance.
- **Qualified helper**: a helper representative satisfying the
  eligibility rule below.
- **Selected mechanism**: the judge-selected terminal slice whose
  report hypothesis the refs support.
- **Source ref**: a `CodeRef` grounding a claim to source, not a
  proof of execution.

## Decision

- Representative granularity is the helper-return source
  anchor: the callee-side file/line of the represented
  `__return__` vertex (never the caller call-expression
  location, never a synthesized whole-helper or arm range).
  Inner helper arms are not modeled; numeric arm attribution
  lives in replay and numeric checks, not in report topology.
  A single-return helper needs no discriminator; a helper
  whose one call site yields multiple distinct returns (e.g.
  complementary `early_alt` / `fallback_alt` paths) must
  include the result/output path (as in `call-result:{path}`
  edge roles and `__return__.{path}` spellings) in the
  representative identity, or the refs would wrongly collapse.
- One representative ref per `(helper_callable_id,
  call_source_site_id)` plus the conditional output
  discriminator above; duplicates across call instances
  collapse; genuinely different call sites stay distinct.
  The discriminator is part of identity: same callable, same
  site, different represented result paths are different
  representatives.
- A helper qualifies iff it lies on the selected mechanism cone
  (backward data/control reachability from the selected
  terminal, reusing replay's ancestor-walk shape) AND supplies
  the terminal value through a `call:{name}` / call-result data
  relationship. No new traversal machinery: compose the replay
  ancestor walk with the judge-selected terminal slice.
- Tier order under the fixed `[:8]` budget: causal terminal
  writers, then qualified helper representatives, then
  declaration/storage anchors. Helpers precede anchors but
  never displace causal terminal refs: if the full budget is
  consumed by required causal terminal evidence, a qualified
  helper is omitted rather than crowding it out. Concretely, if
  fewer than the full budget is consumed by higher-priority
  causal terminal evidence and at least one qualified helper
  exists, at least one helper representative survives before
  any anchor does. Tiers never imply execution order.
- Where a helper representative and a terminal writer resolve
  to the same physical source statement, emit one source ref
  with terminal role dominating; different roles never justify
  duplicate budget consumption for one statement.
- Helper refs use honest non-terminal wording distinct from both
  `terminal write:` and the assumed-candidate wording; they
  assert grounding location only.
- `branches_verified` stays helper-unaware: helper refs never
  make an unknown branch verified, revive `always_false`
  paths, or change feasibility, confirmation, or stop
  authority. Retained/report provenance never creates
  proof (ADR-0002 unchanged).
- Helper refs may satisfy generic source-grounding presence
  checks exactly like other valid source refs (e.g. a
  presence-based validation rule must not treat a grounded
  helper ref as absent evidence). Presence is all they
  contribute: helper presence alone never establishes
  evidence sufficiency, replay status, branch verification,
  confidence, confirmation, benchmark strength, or T6B stop
  authority.
- Temporal discrimination (e.g. first-sample-after-transition)
  belongs to `EvaluationScope`/timeline and replay windows; the
  DAG selector consumes windows as input and the report builder
  passes them through. Logged text stays independent
  corroboration, never source provenance.
- Numeric verification stays disconnected from source
  operations for now: no claim↔operation linkage is introduced,
  and helper eligibility does not depend on it. The
  `ExpressionHelperDependency.source_file/source_line` fields
  remain populated from verification plans only.
- Proof-blind spots stay blind: void-mutator member state with
  no routable return (Airspeed load-factor path) gets no
  synthetic value edges and no refs; absent DAG evidence is a
  representation gap for a future decision, not something
  selection may invent.
- No `MechanismCone`, `EvidenceView`, or `CausalSlice` types;
  no macro/generated-code normalization; no per-case
  production branching.

## Relationship to ADR-0001 / ADR-0002

Excluding `call_instance_scope` from report helper identity is
an application of ADR-0001's existing rule: call-instance scope
patching is an explicitly allowed parser-proven re-spelling
that preserves semantic identity while spelling changes. The
helper identity therefore keys off the stable call-site and
callable components and ignores instantiation scope, exactly as
evidence identity ignores spelling-level instance variation.
Helper refs are a new grounding class only: coverage,
applicability, retained-state/history proof, and stop authority
keep their separate producers and meanings per ADR-0002, and
nothing in this decision promotes a helper ref into any of
those claims.

## Status

Proposed as the ownership contract for future helper-evidence
work. Implementation follows separately (WS2); Airspeed state
representation and temporal/windowed selection (WS3) are
explicitly out of scope here.

## Considered Options

- Inner-arm operation refs: rejected. Helper bodies do not
  materialize as vertices; synthesizing arm refs would invent
  source operations, and one helper can contain several
  important lines (transform, clamp, winning max/min, return)
  that a single ref cannot disambiguate.
- Callable-only collapse: rejected. It fuses genuinely
  different call sites and loses the only stable per-use
  anchor the graph provides.
- Callable-plus-call-site as universally sufficient identity:
  rejected. Real multi-return helpers exist (complementary
  return paths from one call site); the identity must carry
  the conditional output/result-path discriminator, or
  distinct represented results would wrongly collapse.
- Pre-prune helper-cone retention: rejected. It duplicates
  cone storage, couples selection to the pruning lifecycle,
  and risks proof leakage; post-prune traversal of surviving
  graphs plus existing assumed-pruning retention covers the
  reachable cases.
- Separate evidence graph/view: rejected. Annotated DAG,
  retained provenance, judge projection, verification plans,
  and replay results already partition the concerns; a new
  type adds ownership without a consumer beyond `source_refs`.
- Modeling void-mutator state (Airspeed): rejected here.
  Unbounded graph growth, collapsed call-scope identity, and
  severe replay/coverage risk (per-declaration certificates,
  single-exact-producer basis, T6B veto) outweigh diagnostic
  value at this stage; deferred to its own future ADR if ever
  pursued.
- Numeric-link-required selection: rejected. The linkage owner
  does not exist yet; making helper qualification depend on it
  would block all helper evidence on unrelated architecture.

## Consequences

- WS2 implements representative selection, cross-path dedup by
  site-group key, ordering insertion with budget proof, and
  RTL helper coverage; TECS/Takeoff temporal behavior and
  Airspeed representation remain out of scope for it.
- `branches_verified`, confidence, confirmation, replay
  completeness, `match_fraction`, and verification status are
  unaffected by helper-ref presence.
- BEST_SUPPORTED stays reachable with excellent grounding and
  does not require Airspeed; grounding and sufficiency remain
  separate claims.
