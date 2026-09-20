# Helper source-evidence spec (WS2)

Implementation-ready contract for qualified helper representative
source evidence in `source_refs`. Builds on the terminal
normalization of `docs/terminal_source_ref_normalization_spec.md`
and the ownership decisions of
`docs/adr/0003-helper-source-representative-ownership.md`.
Out of scope: temporal/windowed selection (WS3), Airspeed
representation, numeric↔source linkage ownership, new graph or
view types.

Status: specification only. No implementation, no fixtures, no tests.

## 1. Problem statement

Terminal-only `source_refs` omit the upstream helper computation
that actually explains a reported mechanism (e.g. RTL's cone
helper behind the `_rtl_alt` terminal writers). The DAG does
represent the helper as one `__return__` vertex per call site,
wired to its caller by data edges — but report selection has no
rule that admits it, so the explanatory equation never reaches
the report.

## 2. Helper callable extraction

Use the existing `_helper_callable_id` canonical form
(`file:line:name:params`, preferring the definition record's
`callable_id` when present). Do not parse diagnostic
`helper_return:` provenance strings. If neither a `callable_id`
nor file/line/name components exist, the helper has no stable
callable identity and is ineligible — fall back to omitting it,
never to a synthesized identifier.

## 3. Caller call-site extraction

Priority, first present wins:

1. Structured `call_site_id` (vertex metadata or binding field).
2. `expression_ref.call_results[].call_source_site_id`.
3. Edge `via` source-site on the helper-return→caller data edge.
4. `source_call_roles` map entries keyed by invocation site.

Never substitute the callee return-site `source_site_id` for
caller call-site identity; those are distinct semantic
locations (verified: return-vertex file/line is callee-side,
call site is caller-side).

## 4. Result/path discriminator

Use the represented result identity already in the graph:
`__return__` vs `__return__.{path}` variable spellings and
`call-result:{path}` (vs plain `call:{name}`) edge roles. For a
single-return helper the discriminator is the canonical empty
value and adds nothing. Do not invent discriminators for
unrepresented interior arms.

## 5. Exact helper identity

```text
(helper_callable_identity,
 caller_call_site_identity,
 result_path_discriminator)
```

No vertex IDs, no insertion order, no `call_instance_scope`
(multiple instances of one site collapse by construction).

## 6. Source anchor

The helper `CodeRef` anchors to the callee helper-return
vertex file/line — never the caller call-expression location,
never a synthetic function range, never a helper-body
interior line. If no reportable helper-return file exists, emit
no helper ref (existing `_retained_candidate_ref`-style file
gate applies by analogy).

## 7. Qualification relation

A helper qualifies iff all hold on the surviving annotated DAG:

- it is reachable from the selected terminal by backward
  data-edge ancestry (reuse replay's ancestor-walk shape; no
  new traversal type);
- it supplies the terminal value through an incoming
  `call:{name}` / `call-result:{path}` data relationship
  toward the selected terminal (edge `role` + `via`, as
  recorded by the existing call-grounding map).

Control-only reachability never qualifies (feasibility
context, not value contribution). This excludes Takeoff-style
state-machine provenance by construction.

## 8. Selected-terminal ancestry

Reuse the existing incoming-edge ancestor walk shape plus the
judge-selected terminal slice. No `MechanismCone`,
`EvidenceView`, or `CausalSlice` types; no stored traversal
abstraction.

## 9. Surviving-DAG boundary

Selection runs only over surviving annotated-DAG
relationships. Never traverse from `assumed_pruned_provenance`
(its incoming edge structure no longer exists). A retained
terminal candidate therefore has terminal source evidence but
no helper representative — acceptable and intentional. Do not
expand Workstream A retained records.

## 10. Call-instance collapse and distinct sites

Same callable + same caller call site + same result path
across `call_instance_scope` values normalizes to one helper
representative (pin with a test). Same callable at two
genuinely different caller source call sites yields two
representatives, subject to budget. Same helper identity
reached through multiple selected paths yields one ref; path
count never consumes budget.

## 11. Cross-path dedup and helper-vs-terminal overlap

Dedup key: the §5 identity. If a helper representative and a
terminal writer resolve to the SAME physical source statement,
emit one ref with terminal semantic class winning — duplicate
roles never consume two budget slots for one statement.

## 12. Explanation semantic class

Helper refs carry a stable semantic prefix/class distinct from
`terminal write:` and from `candidate terminal write excluded
by assumed feasibility condition:`. Conceptually
`upstream helper contribution:`. Tests assert semantic
class/prefix, never incidental full prose.

## 13. Tier ordering and budget

Order: Tier 1 causal terminal writers, Tier 2 qualified helper
representatives, Tier 3 declaration/storage anchors — reusing
the WS1 normalized flow (identity → dedup → tier ordering →
budget → emission). Budget stays `[:8]`, applied after
selection: helpers NEVER displace required causal terminal
refs; if fewer than 8 slots are occupied by causal terminal
refs, qualified helpers receive remaining slots before
anchors. For more helper identities than remaining slots, use
the existing deterministic source-derived ordering (no
semantic "best helper" ranking).

## 14. Validation, confidence, branches

Qualified helper refs count as valid source grounding present
(they may prevent an empty-source-ref downgrade) and MUST NOT
themselves establish verified replay, branch verification,
sufficient evidence, coverage, applicability, confirmation,
confidence, benchmark strength, or stop authority.
`branches_verified` remains helper-unaware. Presence alone
never upgrades confidence/confirmation — grounding and
sufficiency stay separate claims.

## 15. Isolation

No helper source-ref behavior may affect T3, T5, T6B, Gate-B,
R1, proof stores, checkpoint authority, replay enumeration,
comparison domains, match fractions, completeness, or
verification results. Replay/DAG context may be read, never
mutated, by selection.

## 16. RTL target

Post-WS2, the RTL report additionally contains one qualified
helper representative for
`calculate_return_alt_from_cone_half_angle` at its actual
represented helper-return anchor:
- callable identity: the RTL cone helper canonical form;
- caller call-site: the `rtl.cpp:245` call site;
- result path: single-return, no extra discriminator;
- rendered anchor: callee return-site source location.
Required: `rtl.cpp:245`, `rtl.cpp:248`, `rtl.h:163` behavior
unchanged; numeric benchmark (`destination + 2 × acceptance
radius ≈ 103.93812`), PROVEN strength, and
unconfirmed/low pipeline status unchanged. Forbidden: 696–731
interior refs; any claim about which internal arm won (that
conclusion stays with deterministic numeric verification).

## 17. Ownership and slices

Primary change: `dag_pipeline.py` report source-ref
selection, adding one helper evidence class into the existing
normalized flow. Reuse ancestry/call-result helpers where
already owned; do not move ordering logic into
`mechanism_dag`, change parser/profiler/DAG representation, or
touch proof/replay/checkpoint modules. Slices: (1) identity
extraction + collapse + multi-return tests; (2) qualification
+ anchor + explanation class; (3) tier/budget/dedup insertion
+ isolation pins; (4) RTL acceptance strengthening; (5)
focused + subsystem + full regression.

## 18. Explicit exclusions

Helper representative selection beyond this contract;
helper-return source refs for non-qualifying helpers;
upstream traversal beyond selected-terminal ancestry;
numeric claim ↔ operation linkage; temporal/windowed causal
selection; TECS acceptance; Takeoff acceptance; Airspeed
acceptance or void-mutator representation; new ADRs; new
graph/view types; macro/generated-code normalization;
mechanism-cone type; evidence-view type.

## 19. Stop gates (assessed clear)

A (callable identity without diagnostic parsing): CLEAR —
`_helper_callable_id` exists with structured fallback chain.
B (caller call-site identity): CLEAR — `call_site_id`,
`call_results[].call_source_site_id`, edge `via`,
`source_call_roles` all exist. C (multi-return identity
without vertex IDs): CLEAR — result-path spellings and edge
roles exist. D (helper-return anchor reportable): CLEAR —
callee file/line on the vertex. E (no synthetic helper-body
ops needed): CLEAR — contract uses only represented
vertices. F (no replay/proof changes): CLEAR — read-only
selection. G (no schema/budget change): CLEAR — `CodeRef`
unchanged, `[:8]` unchanged. H (RTL selectable from
surviving topology): CLEAR — caller op + helper-return vertex
+ data edge all survive annotation in the accepted reports.
