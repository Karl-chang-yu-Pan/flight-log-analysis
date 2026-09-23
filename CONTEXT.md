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

**Helper representative**:
A single source reference standing for one helper callable at one caller
source call site (plus output discriminator where one site yields multiple
distinct returns), anchored at the helper's return-site source location.
It never denotes an unrepresented helper-body interior operation.

**Replay**:
A deterministic comparison with one vocabulary: not attempted, unevaluable,
partial, matched, mismatched. A matched result must carry its scope; a local
same-packet match and a terminal replay match have different authority.
_Avoid_: bare matched

## Temporal

**Diagnostic window**:
The time bounds within which the questioned behavior is evaluated, owned
by the evaluation scope; stated directly or derived relative to a logged
transition event, never from source order.
_Avoid_: observation window, comparison domain, active window

**Temporal qualification**:
The decision that a represented writer or helper is the causally relevant
one for a diagnostic window, made from scope windows plus replay and
selection evidence. It filters candidates without changing source
identity, feasibility, or proof claims.

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

**Call site**:
The source location issuing a call. It is distinct from a graph call
instance; repeated call instances do not by themselves create distinct
report identities.

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

**Outstanding source work**:
A source request whose source-search obligation is not yet satisfied by current
accepted coverage proof. Structurally unresolved references remain reported as
source requests even after their search work is proven complete.

**Need**:
One blocked operand's evaluation requirement with its linked source requests.
Narrower than an analysis requirement.

**Requirement**:
A graph proof obligation such as observation binding, source lookup,
construction, state alignment, or replay completeness. Only a
well-formed source lookup (with a validating source-reference
payload) is replay-non-blocking; every other kind blocks numeric
replay.

**Pending construction**:
Deferred known work awaiting materialization. Distinct from unknown source.

**Coverage**:
Positive proof that the determining writers and transfers are accounted for.
_Avoid_: exhausted search

**Exhaustion**:
Scheduling state recording that a search found nothing. It releases priority
only and never proves coverage.

## Coverage Proof

**Writer obligation**:
An unresolved requirement to account for the writers of one storage or
declaration identity. Multiple consuming origins may share it.

**Search attempt**:
One use of one search strategy against one obligation, identified by request,
strategy, examined domain, and index state. An attempt alone proves nothing
complete.

**Searched domain**:
The files, owners, declarations, and query universe a strategy actually
examined. Only the examined domain counts, never the intended one.

**Admission verdict**:
The recorded outcome for one examined candidate: admitted writer, wrong
declaration or owner, non-writer, ambiguous, or search-unavailable. The
miss reason is part of the verdict.

**Coverage evidence**:
Facts supporting sufficiency of search: strategies, examined domains,
candidates, verdicts, limits. Input to a coverage decision, not the decision.

**Coverage certificate**:
A positive statement binding one obligation to one declared search boundary
whose examination leaves no unaccounted writer possibilities inside it.
Valid only for the index snapshot it derives from. Never established by
exhaustion alone, one found writer, topic presence, or visited state.
_Avoid_: exhausted search, visited state

**Applicability proof**:
Evidence that a covered writer could actually govern the use: control,
invocation, receiver, order, and conditions. Coverage lists writers;
applicability qualifies them.
_Avoid_: writer coverage

**Stop authority**:
The checkpoint verdict permitting discovery to stop: no unresolved
requirements, satisfied coverage and applicability where required, and
complete replay. Zero pending work alone is never authority.

## Diagnostic routing

**DAG-decisive**:
A routing predicate (replay-complete, judge-verified,
feasibility-passing, validation-passing) deciding whether the DAG
result is user-facing. It never authorizes stop; stop authority is
separate.
_Avoid_: stop-authority verdict
