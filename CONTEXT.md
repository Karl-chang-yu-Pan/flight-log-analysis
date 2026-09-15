# Flight Mechanism Analysis

The language used to distinguish source-defined behavior from evidence of what
happened during a recorded flight.

## Language

**Operation reachability**:
The source-defined condition under which an operation can execute. Exact
reachability does not, by itself, establish that the operation executed in a flight.

**Boundary call execution**:
An invocation of an operation that exchanges data with an external system.
Execution is distinct from its returned result and any resulting payload change.

**Boundary call result**:
The value returned by a boundary invocation. A successful result cannot be
inferred solely from the presence of a related topic in a log.

**Payload state**:
The value retained by receiving storage, including persistence between transfers.
A published value and the value consumed by a particular receiver are distinct.

**Observed evidence**:
A value actually recorded in the current flight log. Its presence does not alone
prove a particular source operation consumed or produced it.
_Avoid_: Schema-valid evidence

**Observation correspondence**:
A justified relationship between a recorded value and a source-defined value.
It includes the identity and timing conditions needed to use that observation.

**Writer coverage**:
The extent to which assignments and transfers that could determine a value have
been accounted for. An unsuccessful search is not proof of complete coverage.

**Conditional equation check**:
A numerical comparison of a source-defined equation with observations under
explicit conditions. A match alone does not prove writer execution or coverage.
_Avoid_: Verified mechanism

**Linkage repair acceptance**:
Evidence that source relationships survive extraction, graph construction, and
evaluation under the agreed contract. It is distinct from flight verification.

**Flight mechanism verification**:
An evidence-backed conclusion about the questioned flight behavior, accounting
for the relevant calculations, execution conditions, and writer coverage.

## Evidence

**Observation**:
A value recorded in the current flight log at an exact topic placement and time.
_Avoid_: Schema-valid evidence

**Candidate evidence**:
A possible grounding emitted for later validation. It is not yet usable.

**Usable evidence**:
A candidate that passed promotion: kind-specific provenance plus run binding
plus pointwise validity.

**Evidence validity**:
Whether one candidate may be used at one point. Validity never covers all
writers, all times, history, or stop authority.

**Completion / verification policy**:
The separate judgment that writers, domain, history, and assumptions together
justify stopping. Policy never upgrades invalid evidence.

**Provenance**:
The justification for using a value: source link, transfer or publication site,
timing, and policy, plus the identity and proof kind the evidence class requires.

**Replay**:
A deterministic comparison with one vocabulary: not attempted, unevaluable,
partial, matched, mismatched. A matched result must carry its scope; a local
same-packet match and a terminal replay match have different authority.
_Avoid_: bare matched

## Identity

**Source identity**:
The declaration-derived identity of a source symbol, distinct from its spelling.

**Storage identity**:
The written location a value belongs to. Coverage is owed per storage identity.

**Projection identity**:
A member view of an aggregate that preserves the aggregate's storage chain.

**Receiver / consuming identity**:
The object instance and call scope consuming a value. It must survive projection
into the downstream request.

**Scope**:
Where a source use is evaluated: file, callable, line, and order.

**Fallback identity**:
An unproven candidate identity and lifecycle stage, not proof. It requires
exact-match and compatibility constraints plus revalidation before authority.

## Work tracking

**Transfer**:
A source-backed movement across a boundary for one invocation. It is never
inferred from compatible types or topic presence alone.

**Forwarding**:
An identity-preserving copy along one existing single-producer unconditional
exact edge with its guard kept. One hop is one such edge.
_Avoid_: generic dataflow

**Frontier**:
The builder's typed unknown-source items with originating context. The canonical
expansion demand.

**Source request**:
The frontier filtered to one checkpoint's relevant origins. A scoped view, not a
new concept.

**Need**:
One blocked operand's evaluation requirement with its linked source requests.
Narrower than an analysis requirement.

**Requirement**:
A graph proof obligation such as observation binding, source lookup,
construction, state alignment, or replay completeness.

**Pending construction**:
Deferred known work awaiting materialization. Distinct from unknown source.

**Coverage**:
Positive proof that the determining writers and transfers are accounted for.
_Avoid_: exhausted search

**Exhaustion**:
Scheduling state recording that a search found nothing. It releases priority
only and never proves coverage.
