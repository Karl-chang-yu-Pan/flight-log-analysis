# DAG Graph Compaction: Deferred Review

## Status

Recorded on 2026-09-12 for future review. This is a recommendation, not an
implemented feature or approval to change evaluation semantics. The immediate
priority is consolidating evaluation through the existing graph-native value
machinery. Revisit compaction after useful local checks work on real flight data.

Related implementation and resource audits: [DAG checkpoint trial](dag_checkpoint_trial.md).

## Recommendation

Prefer reversible compaction of the working representation of the same DAG.
Do not erase the source mechanism, create a second mechanism, or turn a local
numerical match into proof of execution, writer completeness, or state history.

Candidate transformations, subject to correctness checks:

- Fold source-proven constant calculations while retaining their dependencies.
  Parameter-derived results apply only within verified parameter-value intervals.
- Exclude exactly inactive operations from a working view within the proven
  domain. Unknown or assumption-dependent conditions cannot justify exclusion.
- Represent completed local calculations compactly, retaining their inputs,
  outputs, evaluation results, provenance, and remaining obligations.
- Share immutable expression structure without merging distinct call instances,
  receiver storage, source sites, or operand bindings.

These are different operations from simply folding nodes in a diagram. No
production graph-size, file-count, or expansion budget is proposed.

## Required Information

A compact region must retain, or provide recoverable access to:

- Original vertex, edge, source-site, callable, receiver, and storage identities.
- Boundary inputs and every externally consumed output, including control,
  selection, observation, and state relationships, not just arithmetic values.
- Applicable time windows, evaluable domains, parameter values, signal instances,
  sampling policies, and numerical tolerances used by the check.
- Evaluation status and evidence: a mismatch is not unavailable evidence, and a
  local match does not prove that the writer ran.
- Pending construction, typed source requests and their consumer origins,
  alternative-writer coverage, and unresolved initialization/alignment conditions.
- Dependencies that require reopening the region when newly admitted source,
  changed comparison scope, or changed input evidence alters the result.

Keep source-derived control and temporal semantics. A condition false during the
comparison window does not prove an earlier state update was irrelevant. Unknown
subscription timing must not be replaced by an assumption that stored receiver
state equals the latest logged topic sample.

## Risks

- Replacing a calculation with only its numerical result loses the explanation
  and its parameter/source dependencies.
- A time-varying or stateful computation cannot generally become one scalar.
  Materializing a full result series for every node can increase memory usage.
- A later source admission can reveal additional writers. Compaction must not
  prevent revalidation or preserve a previously justified result as unconditional.
- Equal expressions or values do not establish equal source or invocation identity.
- Deleting underlying derivations can prevent auditing, alternative checks, and
  resource-safe continuation. Keeping every original object resident, however,
  means presentation compaction alone will not materially reduce retained memory.

## Existing Mechanisms To Reuse

- `dag_checkpoint.dependency_view` selects an ancestor-closed view while keeping
  original identities.
- `mechanism_dag.prune_infeasible_operations` already removes operations gated by
  false branches. Any future compaction must establish exactness, scope, and
  completeness of those verdicts rather than treating the verdict label as proof.
- `DAGValueProgram` already compiles local expressions and graph adjacency.
- `DAGValueSession.release_timestamp_values` releases dynamic memoization after
  a timestamp batch; it does not discard modeled vehicle state or source history.
- `mechanism_judge.render_discovery_compact` already groups source sites for
  presentation while preserving the underlying per-instance graph. That grouped
  rendering is not an execution graph and must not become one.
- The builder's `admit_source` revalidates existing reads when new facts arrive.
  Compaction must preserve that behavior.

This proposal does not re-enable persistent analysis caches or cross-run reuse.
Recoverable provenance outside the live working set, if needed, would require a
separate reviewed storage/lifetime design; it is not specified here.

## Resource Expectations

Presentation compaction primarily reduces rendering size and model input.
Execution compaction may reduce repeated traversal and evaluation. Actual memory
savings require releasing objects no longer needed by any live owner, including
builder indexes, compiled programs, and retained snapshots.

Collapsing a graph only after construction cannot prevent that construction's
peak resource use. Recent audits identify substantial source-admission and
construction costs, so compaction alone is not a remedy for excessive discovery.
No quantified saving has been demonstrated for the proposed transformations.

## Review And Acceptance

Before implementation, identify the live objects and operations responsible for
the measured cost. Compare uncompacted and compacted execution using the same
source, log, comparison windows, and semantic requirements.

Require equivalent numerical results, active domains, mismatch/unresolved states,
source provenance, and pending/source-discovery obligations. Include late-writer
admission, changing parameter intervals, state carried from before the comparison
window, conditional subscriptions, shared dependencies, and separate call sites.

Use small deterministic contracts first, then the existing resource-audited
real-source workflow. Measure wall time, CPU, retained/peak memory, and rendering
size separately. Test resource guards remain distinct from production semantics.
Do not describe compaction as successful merely because fewer nodes are displayed.
