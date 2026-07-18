---
name: interchangeable-retirement-tests
description: Design and audit interchangeable contract tests during staged implementation retirement. Use when an old and replacement parser, resolver, pipeline, service, backend, or downstream mechanism coexist; when adding shadow or compare modes; when moving consumers to a replacement; or before removing the old path.
---

# Interchangeable Retirement Tests

## Core Rule

While old and replacement implementations coexist, define shared behavior once and run the same fixtures and assertions unchanged against both implementations.

Do not copy an existing behavioral test into a replacement-specific test. A separate test can let the two paths drift while both suites remain green.

## Establish the Replacement Boundary

1. Identify the old and replacement entry points.
2. Identify their common input, output, errors, side effects, and downstream consumers.
3. Place adapters outside the contract when their purpose is only to present the common interface.
4. Test through the deepest shared consumer boundary practical, not only through each implementation's internal records.

For a parser replacement, exercise source through extraction and the downstream consumer. For a pipeline replacement, exercise the same request or fixture through both pipelines to the stable result consumed by the next stage.

## Classify Existing Tests

Classify every relevant test before adding coverage:

- **Shared contract:** Observable behavior both implementations claim to provide. Run unchanged against both.
- **Implementation detail:** Internal AST nodes, scanner state, query plans, or private helpers. Keep implementation-specific.
- **Replacement-only capability:** Syntax or behavior the old implementation never supported. Test separately, but do not count it as parity evidence.
- **Intentional behavior change:** Define the desired result independently of the old output. Do not use the old implementation as the correctness oracle.

Move existing shared tests into the contract suite instead of creating parallel replacement tests.

## Build One Harness

Select the implementation through a fixture, factory, dependency, or feature flag while preserving identical test code:

```python
@pytest.mark.parametrize("implementation", ["old", "replacement"])
def test_source_value_reaches_observed_output(tmp_path, implementation):
    system = build_system(tmp_path, implementation=implementation)
    result = run_shared_workflow(system, SHARED_INPUT)

    assert_shared_contract(result)
```

The shared workflow must include the same:

- input files and companion files;
- schemas, runtime inventory, and configuration;
- orchestration and adapters;
- downstream transformation;
- expected output and failure assertions.

Do not weaken assertions for one implementation merely to obtain parity.

## Define Correctness Independently

Derive expected behavior from requirements, stable interfaces, schemas, source semantics, and approved design decisions. Do not snapshot the old implementation and call that output correct.

When a shared contract exposes an old defect:

1. Keep the desired contract shared.
2. Record any temporary old-path failure explicitly with its retirement or fix condition.
3. Require the replacement to pass the desired contract.
4. Prevent the old failure from becoming the replacement's accepted behavior.

An approved temporary divergence is not parity. Report it as unresolved migration work.

## Audit Interchangeability

Run the entire shared contract suite against both implementations. Also compare normalized public outputs when deterministic comparison is useful.

Comparison diagnostics, shadow mode, and implementation-specific unit tests supplement the shared contracts; they do not replace them.

Check specifically for:

- fixtures executed by only one implementation;
- assertions duplicated with different expectations;
- adapters that silently drop information for one path;
- fallback behavior that masks replacement failures;
- tests that stop before the shared downstream consumer;
- old implementation types or helpers leaking into the contract.

## Gate Retirement

Retire the old implementation only when:

1. The replacement passes every shared contract required by current behavior.
2. Every intentional divergence is approved and represented by the desired contract.
3. Production consumers use the replacement path without semantic fallback to the old path.
4. Compare or shadow results contain no unexplained material differences.
5. No downstream code depends on old implementation-specific types or state.
6. The shared contract suite remains after retirement and runs against the replacement.

After removal, collapse the implementation parameter to the replacement or keep the reusable contract harness for future replacements. Never delete the behavioral contracts merely because one implementation remains.

## Review Standard

Reject a staged replacement test plan when it adds only replacement-specific tests for behavior already covered on the old path. Require the existing behavioral fixture and assertion to execute against both paths through a shared interface.

Accept implementation-specific tests only when they cover genuinely private mechanics or a capability absent from the old implementation, and label them as non-parity evidence.
