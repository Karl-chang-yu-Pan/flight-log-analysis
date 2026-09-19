# Terminal source-ref normalization spec (WS1)

Deterministic normalization, deduplication, ordering, and budget
policy for terminal source-write evidence in
`build_report_from_dag`, plus retained assumed-pruned terminal
candidates. This is Workstream 1 of the causal-provenance roadmap:
report-layer hardening over existing structures. No helper
selection, no upstream traversal, no temporal logic, no new graph
or view types.

Status: specification only. No implementation, no fixtures, no tests.

## 1. Problem statement

Multiple graph operations may represent the same source writer
(e.g. one publication site instantiated across call-instance
scopes, as seen in Airspeed/TECS terminal duplication). Each
currently becomes one `CodeRef`, so `[:8]` truncation cuts before
any semantic normalization. Causal runtime evidence can therefore
lose budget slots to duplicate anchors, and duplicate instances
consume slots that distinct writers need.

## 2. Normalization boundary

One deterministic stage inside report source-ref selection,
owned by the report builder (`dag_pipeline`), reusing the current
`_order_source_ref_entries` seam:

```text
raw eligible source evidence (surviving terminal writes +
    retained assumed-pruned terminal candidates)
→ stable identity per candidate
→ cross-path dedup (surviving wins over retained duplicates)
→ semantic tier ordering
→ budget truncation ([:8])
→ CodeRef emission
```

No new architecture layer. No changes to discovery, feasibility,
replay, checkpoint, coverage, applicability, retirement, or stop
authority. The report builder remains presentation-selection
owner; it performs no causal analysis beyond the rules below.

## 3. Identity definitions

Three distinct identities; dedup in this spec operates ONLY on
source-writer identity:

- **Graph-instance identity**: the emitted vertex (`id`, insertion
  order). Never used for dedup or ordering.
- **Source-writer identity** (dedup key, §5): the source
  statement that performs the write, independent of how many
  graph times it was instantiated.
- **Report-ref identity**: the emitted `CodeRef` (file + range +
  explanation). Two refs are duplicates iff their source-writer
  identities match.

## 4. Exact dedup key

```text
(file, start_line, end_line,
 exact_target_symbol,
 normalized_expression,
 scope_identity,
 call_site_note)
```

Field sources, in order:

- `file`: `vertex.file` / record `file`; empty string if absent.
- `start_line` / `end_line`: `vertex.line` for both when only a
  single line is known; record `line`/`end_line`; `None`
  preserved as `None` (never coerced to a sentinel that could
  collide with a real line).
- `exact_target_symbol`: `vertex.variable` / record `variable`,
  unstripped except surrounding whitespace.
- `normalized_expression`: §6.
- `scope_identity`, first present wins:
  1. `target_identity.declaration_id` from vertex metadata when
     present and non-empty;
  2. vertex metadata `source_site_id`, else record
     `source_site_id`;
  3. vertex metadata `function` / record `callable`;
  4. target-scope `(file, callable)` from vertex metadata.
- `call_site_note`: metadata `call_site_id` / record callable
  site where present, used ONLY to keep genuinely different
  call-site writers distinct (see §5); never a vertex/op ID.

Rationale for each component: file+range locates the statement;
symbol+expression distinguishes same-line distinct writes;
scope identity distinguishes same-spelling writers in different
declarations/callables; call-site note preserves genuinely
different instantiation contexts without making instances the
identity unit.

## 5. Call-site handling and WS2 compatibility

Same source statement instantiated repeatedly (same file, line,
symbol, expression, scope) collapses to one ref regardless of
call-instance count — this is the Airspeed/TECS duplication fix.
Same helper/callable source reached from distinct semantically
relevant call sites keeps per-site distinction ONLY through the
`call_site_note` component; WS1 never selects on it and never
emits helper refs. The note field exists in the key shape now so
WS2 can promote it into per-helper collapse identity later
without reshaping WS1 records. No helper selection, no
`helper_callable_id` logic, no backward traversal in WS1.

## 6. None-location behavior

`file=None` / `line=None` / `end_line=None` never crash
normalization and never synthesize a location:
- Dedup identity still exists (other tuple components decide).
- A ref is EMITTED only if `file` is non-empty (existing
  `_retained_candidate_ref` behavior; report refs without a file
  are not valid `CodeRef`s).
- `None` line sorts deterministically (see §11) and renders
  without a line suffix.

## 7. Expression normalization (minimal)

Normalize only representation differences already produced by
existing parser/DAG structures:

- surrounding whitespace;
- redundant outer parentheses;
- the already-lowered expression form when the vertex carries
  `lowered_expression` (prefer it over raw `expression`;
  record both spellings' equality is NOT required, only that
  one deterministic choice is used consistently);
- stable serialization of the chosen form (single spaces,
  no trailing semicolons).

Do NOT canonicalize semantically different expressions into one
identity (`a*3` vs `a * 3`normalize together; `a*3` vs `a*4`
never do). No source canonicalizer, no new parser behavior.

## 8. Surviving-vs-retained precedence

Cross-dedup rule (existing behavior, pinned): same stable
writer identity present as both surviving ordinary evidence and
retained candidate → emit ordinary surviving evidence once.
Candidate provenance must never downgrade stronger surviving
provenance. Comparison uses the §4 key, never explanation text.

## 9. Terminal-duplication normalization

Repeated terminal operations sharing one stable writer identity
(§4) collapse to a single semantic ref. Genuinely different
writes (different key) stay distinct even for the same symbol.
This fixes the Airspeed/TECS pattern (same publication source
location across call-instance scopes → one ref) without any
acceptance case for those logs and without touching their
pipelines.

## 10. Evidence tiers (WS1 only)

- TIER 1 — causal/runtime terminal computation writer
  (surviving ordinary evidence + retained assumed candidates).
- TIER 2 — supporting declaration/storage anchor.
- Reserved future slot, NOT implemented: qualified helper
  representative between TIER 1 and TIER 2. No helper records,
  no helper selection, no helper ordering in WS1.

## 11. Total order (explicit, deterministic)

Sort key, in order:

1. Tier: TIER 1 (0) before TIER 2 (1).
2. Status within tier: surviving ordinary (0) before assumed
   candidate (1).
3. Source order: vertex/record `source_order` ascending where
   present on all compared items; else `source_site_id`
   ordering via the existing `_source_site_order`
   compatibility rule; else `(file, line or -1)`.
4. No further fallback: Python sort stability over a
   deterministically built input list preserves input order
   for full ties, and input construction order is itself
   deterministic (vertex emission order, then retained-record
   capture order).

Graph insertion order and vertex IDs are never ordering inputs
beyond what stable-sort preservation already guarantees.

## 12. Candidate marking (preserved)

Retained assumed-pruned writers keep the established candidate
wording replacing ordinary `terminal write:` wording, generic
vocabulary only, no verbatim internal reason text. Normalization
and dedup must preserve the marking through collapse (a merged
duplicate set containing any candidate marks the surviving ref
as candidate; surviving-ordinary-wins (§8) applies only to
exact cross-dedup hits).

## 13. Budget placement and anti-crowding invariants

Truncation `[:8]` applies AFTER identity normalization, dedup,
and ordering — never before. Limit value unchanged.

- **Invariant A**: if ≥1 eligible causal runtime terminal ref
  exists, ≥1 survives `[:8]`. Enforced structurally: TIER 1
  sorts before TIER 2, so truncation can only remove causal
  refs when >8 causal refs exist (in which case survivors are
  still causal).
- **Invariant B**: anchors cannot consume the full budget
  ahead of causal runtime evidence (follows from tier order).
- **Invariant C**: duplicate instances consume one slot after
  normalization (follows from §9).
- **Invariant D**: an ASSUMED candidate cannot crowd out an
  equivalent surviving ordinary writer (follows from
  cross-dedup §8 + surviving-first ordering).

## 14. Helper insertion point (reserved, not built)

Future qualified helper refs insert between TIER 1 and TIER 2
with their own assumed/ordinary status sub-ordering. WS1 code
must not foreclose this slot but must not implement any part
of it: no helper record type, no helper selection predicate,
no helper collapse.

## 15. Proof isolation (frozen)

Normalization output affects only `source_refs` list content
and order. It must not affect DAG vertices/edges, replay
enumeration (vertices only), writer coverage, applicability,
retirement, checkpoint authority, T6B, Gate-B, R1, or P0–P4.
`assumed_pruned_provenance` stays `exclude=True` diagnostic
metadata; no proof path reads it (verified: only report
selection consumes it).

## 16. Branch verification and confidence (frozen)

`branches_verified` keeps current judge-named-branch
feasibility semantics; normalization never touches it. More or
better-ordered refs never upgrade confidence, confirmed
status, replay status, coverage, or applicability. No
`unresolved_evidence` entries are added by normalization.

## 17. Retained-record changes

NO retained-record shape change is required for WS1. All §4
key components are already present: retained records carry
`file/line/end_line/variable/expression/callable/source_site_id`;
surviving vertices carry `file/line/variable/expression` plus
metadata `source_site_id`/`source_order`/`call_site_id`/
`target_identity`/`function`/scopes where the emitter recorded
them. The key degrades gracefully via §4 preference order and
§6 None-handling. (Resolves §18 below: dict stays.)

## 18. Typed-record verdict: KEEP DICT FOR WS1

Correctness leverage does not justify typing now: every
producer (one capture site) and consumer (one selection site)
is covered by the TDD list below, all accesses go through
`.get()` with explicit defaults, and the shape is frozen by
this spec. Revisit only if a second producer/consumer pair
appears (e.g. WS2 helper records).

## 19. Mixed-control test (include)

Add the hardening test: one operation controlled by
ASSUMED_FALSE + independent PROVEN_FALSE → not retained as a
candidate (assumption is not the sole removal cause). Cheap
(two predicates on one fake binding) and pins §8-adjacent
eligibility at the selection layer's input contract.

## 20. Auto-heal lifecycle test (include)

Add: ASSUMED candidate record + later DERIVED/PROVEN
surviving equivalent → exactly one ordinary ref, candidate
wording gone, no duplicate slot. Feasible without new
infrastructure (hand-built DAG + retained list through the
selection function). Directly relevant to dedup + precedence.

## 21. RTL acceptance compatibility

RTL stays GREEN with computation-first ordering. Numeric
benchmark, PROVEN strength, and unconfirmed/low pipeline
status unchanged. If current A5 pins only declaration-anchor
presence, extend it to require the computation-first order —
without touching numerics, confidence, or authority. Future
valid extra refs must not break acceptance (assert presence +
ordering properties, not exact ref lists).

## 22. Generic duplication regression (no real logs)

Synthetic unit testindependent of large fixtures: N
hand-built terminal graph instances sharing one stable writer
identity → exactly one normalized `CodeRef`. This proves the
Airspeed/TECS duplication fix generically.

## 23. TDD scope (bounded)

- N1 duplicate surviving terminal refs → one `CodeRef`.
- N2 distinct writers (same symbol, distinct identity) →
  separate `CodeRef`s.
- N3 retained+surviving duplicate → one ordinary `CodeRef`.
- N4 runtime-before-anchor ordering (both present orders).
- N5 budget: causal survives `[:8]` among anchors.
- N6 order stability across input permutations.
- N7 mixed-control exclusion (§19).
- N8 auto-heal dedup (§20).
- N9 RTL regression (computation-first, anchor retained).
- N10 proof non-interference (existing replay/checkpoint/
  coverage/applicability/T6B suites unchanged).
Prior D3 T3/T4/T5 remain valid and are extended, not replaced.

## 24. Ownership and slices

Owner: report source-ref selection/normalization in
`dag_pipeline` (selection function + helpers); no DAG-builder
ordering logic. Slices: (N1) identity + key + normalization;
(N2) generic dedup + cross retained/surviving + budget/order;
(N3) tiers + candidate preservation + ordering; (N4) N7/N8
hardening + RTL + full regression.

## 25. Explicit exclusions

Helper representative selection; helper-return refs; upstream
traversal; numeric-claim↔operation linkage; temporal/window
selection; TECS/Takeoff/Airspeed cases; Airspeed
void-mutator representation; new ADRs; new graph/view types;
macro normalization; mechanism-cone type; evidence-view type.

## 26. Stop-gate dispositions

- STOP A (identity needs vertex IDs): CLEAR — key uses only
  vertex metadata + record fields verified present.
- STOP B (dedup needs vertex IDs): CLEAR — same key, no IDs.
- STOP C (duplicates are genuinely distinct): CLEAR — key
  distinguishes assignments; call-site note preserves
  per-site distinction where semantically relevant.
- STOP D (budget needs schema change): CLEAR — order-then-
  truncate in list code, `[:8]` unchanged.
- STOP E (normalization changes replay/proof): CLEAR —
  report-list-only change; retained field unread by those
  paths (verified by consumer grep).
- STOP F (needs helper semantics): CLEAR — terminal-only
  scope; helper slot reserved, not built.
