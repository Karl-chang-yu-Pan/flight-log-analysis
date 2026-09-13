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
