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

Amended after WS2 STOP H: the original unconditional live-RTL-helper
rule (old §16 + old STOP H) is replaced by a conditional
surviving-topology contract (§§7–9, §16). No ADR-0003 change:
helper selection stays on the surviving annotated DAG only.

## 1a. Selector correctness versus live availability

These are different claims and the spec treats them separately:

- **Helper selector correctness**: given a surviving represented
  helper→selected-terminal data relationship, the selector must
  correctly identify the helper, preserve stable identity, collapse
  repeated graph instances, preserve distinct call sites/results,
  deduplicate cross-path duplicates, place helpers in the correct
  tier, and respect the fixed `[:8]` budget.
- **Live helper availability**: whether a qualifying
  helper→terminal relationship still exists in the annotated DAG
  AFTER pruning.

A selector can be correct even when a particular live case has no
surviving helper relationship. Absence of a helper ref under §9 is
a surviving-topology limitation, never evidence that the helper did
not execute or did not contribute (see §9 honesty rule).

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

## 3. Caller call-site extraction (stable SOURCE site only)

The report identity component is the stable caller SOURCE call-site
identity: the source AST/call statement issuing the call, not a
runtime/graph call instance.

Priority, first stable present wins:

1. Consumer-side structured
   `metadata.source_expression_ref.call_results[].call_source_site_id`
   for the call-result entry matching the helper edge (by
   `result_path`, else by helper short name, else the single entry).
   Format `file:start_byte:end_byte:type` is source-stable and never
   carries an `invocation_` prefix.
2. Raw helper-vertex/binding `call_site_id` ONLY when it is not an
   instance-scoped value (it must not start with `invocation_`).
   Top-level calls satisfy this; nested/contextualized calls do not.
3. Edge `via` source-site on the helper-return→caller data edge ONLY
   when it is not an `invocation_` hash.
4. `source_call_roles` map entries ONLY when the key is not an
   `invocation_` hash.

Never use as report identity: `call_instance_scope`, vertex IDs,
operation IDs, `invocation_` hashes, graph insertion order.
Multiple graph instances whose raw `call_site_id` values are distinct
`invocation_<hash>` values from the SAME source call MUST collapse to
one representative via the stable site above (see §10).

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
 stable caller SOURCE call-site identity,
 result_path_discriminator)
```

The second component is the §3 stable SOURCE site, never a
`call_instance_scope`, never an `invocation_` hash, never a vertex
ID. The displayed helper `CodeRef` still anchors at the callee
helper-return file/line (§6); the return vertex's callee-side
`source_site_id` is never substituted for the caller source-site
identity.

No vertex IDs, no insertion order, no `call_instance_scope`
(multiple instances of one site collapse by construction, including
multiple `invocation_<hash>` instances of one source call).

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

Positive helper contract (conditional, topology-bound): if a
qualified helper relationship survives in the annotated DAG, the
report MUST emit the corresponding helper representative, subject
to identity (§5), dedup (§§10–11), ordering, overlap, and budget
(§13) rules. There is no unconditional live-helper requirement;
§9 states the exclusion side and §16 states the RTL consequence.

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

Topology-limited exclusion: if the helper existed before pruning
but all qualifying consumers / data relationships were removed,
and no surviving data path connects that helper to the selected
terminal, WS2 MUST NOT synthesize, reconstruct, or infer a helper
ref. The surviving declaration terminal having no incoming helper
data path is the defining instance of this rule.

Source-grounding honesty rule: absence of a helper ref under this
section is an observability / surviving-topology limitation. It is
never evidence that the helper did not execute or did not
contribute, and the spec MUST NOT be read to imply that.

Retained-candidate interaction (intentional): retained candidate
terminal + deleted helper data ancestry → candidate terminal ref
allowed, helper representative unavailable. Do not expand
`assumed_pruned_provenance` to recover helper ancestry.

## 10. Call-instance collapse and distinct sites

Same callable + same stable caller SOURCE call site + same result
path across `call_instance_scope` values (including distinct
`invocation_<hash>` raw site values from one source call)
normalizes to one helper representative (pin with a test). Same
callable at two genuinely different caller source call sites
yields two representatives, subject to budget. Same helper
identity reached through multiple selected paths yields one ref;
path count never consumes budget.

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

## 16. RTL target (topology-limited acceptance)

STOP H established that the real RTL helper is not reachable
through the surviving represented data topology: helper
`__return__` vertices exist, but their only relevant consumers
are the runtime terminal operations around `rtl.cpp:245`, those
terminal operations are pruned, pruning removes their incident
data edges, and the surviving declaration terminal has no
incoming helper data path. A selector restricted to the surviving
annotated DAG therefore cannot qualify the helper, and returning
no helper ref is correct under the frozen architecture.

The live RTL report MUST NOT be required to contain a helper
representative. RTL WS2 acceptance verifies the honest
topology-limited behavior instead:

1. No fabricated helper ref is emitted when the surviving DAG has
   no qualifying helper→terminal data relationship.
2. Existing candidate computation refs remain present:
   `rtl.cpp:245`, `rtl.cpp:248`.
3. Existing `rtl.h` storage/declaration anchor remains present.
4. Existing numeric reconstruction remains unchanged
   (`destination + 2 × acceptance radius ≈ 103.93812`).
5. Benchmark strength remains PROVEN.
6. Pipeline status remains `confirmed=[]`, unconfirmed slice, low
   confidence.
7. No synthetic helper-body interior refs are emitted
   (696–731 interior refs forbidden; which internal arm won stays
   with deterministic numeric verification, never report
   topology).

No statement in this spec implies absence of a helper ref means
the helper did not contribute (§9 honesty rule applies to RTL).

## 17. Ownership and slices

Primary change: `dag_pipeline.py` report source-ref
selection, adding one helper evidence class into the existing
normalized flow. Reuse ancestry/call-result helpers where
already owned; do not move ordering logic into
`mechanism_dag`, change parser/profiler/DAG representation, or
touch proof/replay/checkpoint modules. Slices: (1) identity
extraction + stable-source-site collapse + multi-return tests;
(2) qualification + anchor + explanation class; (3) tier/budget/
dedup insertion + isolation pins + topology-limited negative
regression; (4) RTL topology-limited acceptance (§16); (5)
focused + subsystem + full regression.

Test roles (conceptual): H1 single-return qualification; H2
repeated call-instance collapse (including `invocation_<hash>`
variants of one source call); H3 distinct source call sites
preserved; H4 multi-return result/path discriminator; H5
control-only helper excluded; H6 cross-path helper dedup; H7
terminal/helper same-statement overlap; H8 terminal → helper →
anchor tier ordering; H9 8 terminals + helper (helper does not
displace terminal evidence); H10 7 terminals + helper + anchors
(helper receives remaining slot before anchors); H11 helper
counts as grounded source presence but establishes no
confidence/proof/status; H12 topology-limited/pruned-consumer
exclusion (helper vertex exists, only qualifying consumer/path
absent from surviving DAG → no helper ref); H13 real RTL
acceptance of topology-limited behavior (§16); H14
proof/replay/checkpoint isolation. Exact numbering may differ;
H1–H11 positive behavior is retained, not weakened.

## 18. Explicit exclusions

Helper representative selection beyond this contract;
helper-return source refs for non-qualifying helpers;
upstream traversal beyond selected-terminal ancestry;
numeric claim ↔ operation linkage; temporal/windowed causal
selection; TECS acceptance; Takeoff acceptance; Airspeed
acceptance or void-mutator representation; new ADRs; new
graph/view types; macro/generated-code normalization;
mechanism-cone type; evidence-view type.

## 19. Stop gates (A–G kept; H replaced; I added)

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
unchanged, `[:8]` unchanged. H (live helper required only on
surviving relationship): REASSESSED — a live helper ref is
required only when a surviving represented
helper→selected-terminal data relationship exists; if such a
relationship exists and the selector cannot emit the helper,
STOP. For real RTL no such surviving relationship exists, so no
live helper ref is required (see §16). I (stable caller SOURCE
call-site identity reconstructible): CLEAR on current tree —
consumer-side `call_source_site_id`, non-`invocation_` raw site,
and non-`invocation_` edge `via` provide a source-stable site
without per-instance hashes; if a future case offers no stable
site without instance IDs/hashes, STOP I fires: do not resume
TDD, recommend a narrow `$wayfinder` focused only on
source-call-site identity.
