# Positive writer coverage and stop authority

Unresolved writer obligations can be represented, linked, and scheduled, but
nothing today positively closes them: exhausted searches release scheduling
priority only, and local matches never authorize stopping. We decide that
coverage evidence (what was searched, with what result) is produced by source
expansion and discovery, while the final coverage/stop verdict belongs solely
to checkpoint discovery; coverage, applicability, history, and value evidence
remain distinct claims with separate producers.

## Decision

- A writer obligation is per declaration/storage entity; consumers share it
  through accumulated origins.
- Search attempts record request, strategy, examined domain, and index state;
  misses carry verdicts and never collapse into bare emptiness.
- A coverage certificate binds one obligation to one declared search boundary
  plus the examined-domain evidence showing no unaccounted writer
  possibilities inside that boundary. It is valid only for the index snapshot
  it was derived from; admitting new source invalidates dependent certificates
  and reopens scheduling without discarding the evidence.
- Coverage (which writers are accounted for), applicability (which could
  govern this use), retained-state/history proof, and usable value evidence
  are separate claims with separate producers; none implies the others.
- Stop authority additionally requires complete replay with no unresolved
  requirements; zero pending work alone is never authority.
- Fail-closed defaults stand: unknown writers, unknown applicability, and
  missing history/transfer evidence keep their obligations open.

## Considered Options

- Exhaustion-as-coverage: rejected. An empty search cannot distinguish no
  writer from unsearched domain, and visited-state suppresses retry.
- Single found writer as coverage: rejected. One writer never establishes
  that all writers are known.
- Topic presence as transfer/execution proof: rejected. Publication and
  consumption are distinct; invocation outcomes are not logged.
- Joint evaluator/checkpoint coverage ownership: rejected. Two producers of
  stop authority would disagree; one owner sees obligations, numeric
  completeness, and intermediate checks together.
- Merging applicability into coverage: rejected. Static writer enumeration
  cannot answer control/invocation/receiver questions; merging would certify
  writers that cannot govern the use.
- Treating retained-state/history as statically coverable: rejected. History
  needs invocation-scoped evidence no source search supplies; forcing it
  into coverage would either fabricate success or block static certification.

## Consequences

- More `unresolved` outcomes until positive evidence exists; this is intended.
- Expansion/discovery must start recording searched domains, strategies, and
  admission verdicts where they currently return bare emptiness.
- Stop requires a positive certificate where one is required; history
  obligations cap verification scope until separately solved.
- Legacy default-path behavior is unchanged by this decision.

## Non-goals

No implementation, no receiver-history or transfer-time evidence production
design, no invocation reconstruction, no legacy-path retirement.
