# Single evidence-promotion interface with preserved identity

The DAG builder emits candidates and scoped identities; evaluation owns a shared evidence-promotion interface with kind-specific validity rules; verification/completion policy is separate from validity; checkpoint gates completion but defines no separate evidence meaning; type-only grounding stays candidate-only; proven identity is preserved semantically through explicitly allowed parser-proven re-spellings; bare spelling never replaces proven identity; exhaustion never means coverage; the single producer of a positive coverage/stop fact is checkpoint discovery; forwarding stays narrow; legacy evaluation paths stay frozen.

## Decision

- The builder emits candidate evidence and scoped identities, never final evidence.
- Evaluation owns the evidence-promotion interface: candidate plus kind-specific provenance plus run binding plus pointwise validity yields usable evidence.
- Evidence validity rules may differ by kind (logged observation, parameter, constant, projection, boundary result, boundary transfer, persisted state, witness/correspondence, conditional-equation evidence) under that one interface.
- Verification/completion policy (all writers accounted for, domain covered, history complete, alternatives exhausted, no unsupported assumptions, stop authority) is separate from validity and lives in checkpoint/replay/feasibility aggregation. Policy must not upgrade invalid evidence.
- Checkpoint owns gating/completion policy, not evidence meaning.
- Type-only grounding remains candidate-only in every consumer.
- Proven identity is preserved semantically. Explicitly allowed parser-proven re-spellings are suffix/member projection, receiver composite transformation, call-instance scope patching, and storage-key use-site erasure. Each must preserve semantic identity while spelling changes.
- Bare symbol spelling cannot replace proven identity. Fallback identity is a lifecycle stage constrained by exact-match/admission/compatibility rules with revalidation, never proof by itself.
- Exhaustion never means coverage. An empty search releases scheduling priority only.
- Positive coverage/stop has one producer: checkpoint discovery. Evaluation produces value results, replay produces replay results, checkpoint assessment produces obligations, conditional checks produce scoped results, exhaustion produces scheduling state; none of these independently becomes coverage.
- Forwarding means a single-producer, unconditional, parser-exact direct-storage copy along an existing graph edge with its guard preserved. One hop is one such edge. Projection, transfer, and evaluation remain distinct.
- Legacy evaluation paths remain frozen rather than semantically extended.

## Considered Options

- Per-path evidence semantics: rejected. It produced three verdicts for one declared-type leaf (ordinary promotion, checkpoint rejection, replay refusal) and would drift again on the next edit.
- Declared type as usable evidence: rejected. A type match carries no transfer or invocation proof; accepting it let ordinary evaluation report values the checkpoint correctly forbids.
- Arbitrary identity reconstruction from symbol spelling: rejected. It fuses unrelated storages (same names under unrelated classes/scopes) and degrades requests (receiver-qualified identities collapsed to bare names).
- Absolute identity preservation with no re-spelling: rejected. Receiver composites, call-instance scopes, member suffixes, and storage-key erasure are load-bearing; forbidding them leaves element demands permanently unresolved and fuses invocations.
- Exhaustion-as-coverage: rejected. An unsuccessful search is not proof all writers were considered; treating it as coverage would certify incomplete graphs.
- Broadening discovery to compensate for degraded identity: rejected. It masks the identity defect and expands search cost instead of fixing the request.
- Evaluator/checkpoint joint ownership of positive coverage: rejected. Two producers of stop authority disagree about completeness; a single producer sees obligations, numeric completeness, and intermediate checks together.
- Forwarding as a generic dataflow mechanism: rejected. Aliases, temporaries, call-result evaluation, pointer semantics, conditional and multi-producer copies each need their own rules; a generic forwarder would reimplement evaluation badly.

## Consequences

- More unresolved outcomes are acceptable until valid evidence exists; fail-closed stands.
- Evidence kinds may carry distinct validity rules under the shared interface.
- Policy must not upgrade invalid evidence, and a local conditional match never carries stop authority.
- Semantic identity may change representation through validated transformations only.
- Positive coverage remains unavailable until separately implemented; recursion/worklist stays an independent track.
- Legacy semantic paths stay frozen; future implementation specs must conform to this decision.
