# Upstream causal source evidence (Workstream B design seed)

Bounded future contract seed. NOT implementation-ready; does not
block Workstream A (`docs/assumed_pruning_source_evidence_spec.md`).

Status: design seed only. No implementation, no fixtures, no tests.

## 1. Problem statement

Useful causal computation may be upstream and non-terminal even
when no assumed-pruning defect occurs: the report's
terminal-only `source_refs` rule then omits the equation that
actually explains the behavior. Known future instances:

- Airspeed: load-factor adaptation computation upstream of the
  publication terminal.
- TECS: restart/transient computation upstream of the reported
  signal.
- Takeoff: `MIS_TAKEOFF_ALT` / `fmaxf` parameter-controlled
  computation upstream of state/output terminal.
- RTL: `calculate_return_alt_from_cone_half_angle` helper body
  (`rtl.cpp:696–731`).

Workstream A may ship independently; Workstream B may follow
independently. RTL defect closure requires retained `rtl.cpp`
terminal refs, never helper refs.

## 2. Retained findings (do not re-derive)

- Helpers are ordinary operations in private call scopes with
  `__return__[.path]` producers wired to consumers via `data`
  edges (`role=call:{name}/call-result:{path}`, `via`
  source-site); call-site provenance survives only as edge
  `via` + `metadata{call_site_id, call_instance_scope,
  source_site_id, source_order}`; helper vertex file/line is
  the callee definition site.
- Cone reachability is answerable today via backward
  data/control traversal (replay builds `ancestors` this way;
  `DAGValueProgram` is edge-native).
- Replay-equation participation is NOT directly answerable:
  `replay_terminal_expressions` enumerates terminals only;
  helper ids appear solely implicit in `ancestors` /
  `dependency_issues`. Any "participates in replay" criterion
  must be restated operationally (data-path + value-use +
  numeric-claim support).
- Candidate qualification direction: on the selected
  data-flow cone AND (supplies terminal value via data edge
  OR supports the numeric verification claim).

## 3. Mandatory invariant before helper refs ship

At most one representative ref per
`helper_callable_id + call_source_site_id`, with an explicit
rule choosing the representative source operation/range.
Unbounded cone admission (~5–15 ops observed) followed by
arbitrary `[:8]` truncation is not acceptable.

## 4. Budget interaction

`[:8]` stays. Causal terminal computation (Workstream A)
orders ahead of helper refs; declaration anchors last.
Future spec must prove the winning computation cannot be
crowded out once helpers compete for slots.

## 5. Open design questions (for the future spec)

- Representative operation/range selection per helper.
- Restated replay-participation criterion in DAG-native terms.
- Dedup across same helper body on two mechanism paths.
- Macro/generated-code normalization.
- Whether `branches_verified`-style checks need helper
  awareness (default: no).
- TDD coverage: bounded helper selection, collapse,
  no-explosion, Airspeed-shaped upstream equation surfacing.

## 6. Explicitly out of scope for Workstream A

Helper-body refs, upstream non-terminal equation refs,
per-helper collapse implementation, extended ordering tiers
beyond runtime-computation → declaration-anchor, and any
Airspeed/TECS/Takeoff case work.
