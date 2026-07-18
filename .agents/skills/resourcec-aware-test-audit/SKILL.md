---
name: resource-aware-test-audit
description: Plan and audit testing when the appropriate test suite may consume excessive time, compute, memory, storage, network bandwidth, external-service quota, or money. Use before running expensive tests or when considering replacing a full test plan with smaller staged checks.
---

# Resource-Aware Test Audit

Use an evidence-based test strategy without silently weakening verification.

When the normally appropriate test suite is likely to consume excessive resources, estimate its impact, propose a smaller staged alternative, and ask the user for approval before replacing, omitting, or deferring the full test plan.

## Core rules

1. Do not silently skip an appropriate test because it is expensive.
2. Do not claim that a reduced test plan is equivalent to the full suite.
3. Focused tests may be run first without approval when they are an additional diagnostic step and the original verification plan remains intact.
4. User approval is required before changing the final verification plan by:
   - omitting the full suite;
   - replacing it with narrower tests;
   - deferring it to another environment or person;
   - disabling expensive test categories;
   - reducing test data, repetitions, platforms, services, or configurations in a way that lowers coverage.
5. After testing, produce a test audit that distinguishes executed, passed, failed, skipped, deferred, and not applicable checks.

Repository-specific instructions and explicit user instructions override this skill.

## Step 1: Discover the expected test plan

Before deciding that testing is too expensive:

- Read applicable `AGENTS.md` files and repository testing documentation.
- Inspect package scripts, build files, CI workflows, test configuration, and relevant prior test commands.
- Identify the tests normally expected for the changed files and behavior.
- Determine whether the repository already defines quick, focused, integration, end-to-end, nightly, or full-suite tiers.
- Check whether CI is expected to run tests that should not be run locally.

Do not classify a suite as excessive merely because it is inconvenient.

## Step 2: Estimate resource impact

Use available evidence to estimate:

- Expected duration
- CPU and memory usage
- Disk usage and generated artifacts
- Network traffic
- Container, emulator, browser, database, or service requirements
- Paid API, cloud, or external-service usage
- Test-data size
- Parallelism and worker count
- Known flakiness or retry cost
- Risk to the local machine or shared environment

Prefer repository evidence such as CI timings, scripts, configuration, previous logs, and documentation.

When precise numbers are unavailable, label estimates as approximate and explain the evidence used.

## Step 3: Decide whether approval is needed

Approval is not required merely to run a narrow test first.

For example, the agent may run:

- A test file directly related to the changed code
- A package-level test before a monorepo-wide test
- Static analysis before runtime tests
- A single deterministic reproduction before a stress test

Approval is required when the agent proposes that these smaller checks will replace, omit, or defer the normally appropriate broader verification.

Treat testing as resource-intensive when one or more of the following is reasonably likely:

- The suite takes substantially longer than ordinary repository checks.
- It uses significant CPU, memory, disk, or network resources.
- It launches many containers, browsers, emulators, virtual machines, or services.
- It invokes paid or rate-limited external systems.
- It requires credentials, production-like infrastructure, or shared environments.
- It processes very large fixtures or datasets.
- It runs broad cross-platform, load, soak, fuzz, or end-to-end coverage.
- It may disrupt other work on the machine or shared runner.

## Step 4: Design the staged alternative

Build the smallest sequence that gives useful evidence while preserving traceability to the full plan.

A typical sequence is:

1. Syntax, formatting, lint, or type checks
2. Tests for directly modified units
3. Tests for immediate callers and integration boundaries
4. Package, module, or service-level tests
5. Selected regression tests for the affected policy or behavior
6. Broader integration or end-to-end tests
7. Full suite, stress, soak, fuzz, or cross-platform checks

Choose stages based on the repository. Do not include irrelevant stages merely to make the plan look complete.

For every proposed reduced stage, state:

- What it validates
- Why it is relevant
- What broader test it substitutes for or precedes
- What it cannot validate
- The residual risk if broader testing is not run

## Step 5: Request approval before weakening the plan

Before replacing, omitting, or deferring the full test plan, ask the user using this structure:

### Resource concern

State which test or command is expensive and why.

### Original verification plan

State what would normally be run.

### Proposed staged plan

List the smaller commands or test groups in execution order.

### Coverage difference

Explain what the staged plan covers and what it does not.

### Residual risk

Describe failures that could remain undetected.

### Approval request

Ask the user to choose between:

- Run the full appropriate test plan
- Use the proposed reduced plan
- Use the reduced plan now and leave the full suite explicitly deferred

Do not phrase silence or lack of response as approval.

Do not execute the weakened plan until approval is received. Safe, non-destructive inspection and already-approved focused diagnostics may continue.

## Step 6: Execute approved testing

When running tests:

- Use the approved scope.
- Record the exact commands.
- Record relevant environment, configuration, filters, seeds, and test data.
- Preserve meaningful exit codes and failure output.
- Do not hide failures with unconditional retries.
- If retries are justified, report the original failure and the retry result.
- Stop and reassess if actual resource use is materially higher than estimated.
- Ask again if a further reduction in coverage becomes necessary.

A passing small test does not justify silently cancelling an approved broader test.

## Step 7: Audit the test coverage

Create a test audit after implementation or testing.

The audit must include:

### Change under test

Summarize the behavior, modules, interfaces, and policies affected.

### Expected verification

List the tests that would normally provide appropriate confidence.

### Resource assessment

Record the expensive commands, estimated or observed costs, and evidence for the assessment.

### Approval record

State:

- Whether the user approved a reduced plan
- Which option was approved
- Any limits or conditions the user specified

Do not claim approval that is not present in the conversation.

### Executed checks

For each command, record:

- Exact command
- Scope
- Result
- Duration when available
- Relevant configuration or environment
- Failures, warnings, retries, and flaky behavior

### Coverage map

Map each affected behavior or risk to the test evidence that covers it.

Use statuses such as:

- **Covered:** directly exercised by a passing check
- **Partially covered:** only some paths or layers were exercised
- **Not covered:** no executed check validates it
- **Deferred:** intentionally left for a later environment or workflow

### Omitted or deferred checks

List every normally relevant test that was not run and why.

### Residual risk

Explain realistic defects that the executed checks may not detect.

### Confidence statement

Give a bounded conclusion, for example:

- High confidence for the modified unit behavior; integration behavior not verified.
- Moderate confidence within the tested package; full regression suite was deferred.
- Low confidence because only static checks completed.

Never report unqualified success when relevant tests were omitted.

## Step 8: Preserve an auditable final report

The final response must clearly separate:

- Implementation status
- Test results
- Test scope limitations
- User-approved deviations
- Recommended remaining verification

Use precise wording:

- Say `The focused tests passed`, not `All tests passed`, unless all applicable tests actually ran.
- Say `The full suite was not run with user approval`, when that occurred.
- Say `Not verified`, rather than inferring a result from unrelated checks.

## Example approval request

Use wording similar to:

> The full end-to-end suite is the normal verification for this change, but it launches 12 service containers and historically takes about 70 minutes. I can instead run the affected unit tests, the service integration tests, and the three ownership-flow end-to-end cases. This will not cover unrelated workflows, cross-browser behavior, or the complete regression surface. Should I use this reduced plan, run the full suite, or run the reduced plan now and mark the full suite as deferred?

Adapt the details to the actual repository and task.

## Prohibited shortcuts

Do not:

- Call a suite expensive without inspecting available evidence.
- Skip tests and mention it only after implementation.
- Use a passing lint or type check as evidence that runtime behavior works.
- Treat one test file as repository-wide regression coverage.
- Reduce data size, repetitions, platforms, or environments without disclosing the coverage loss.
- Run paid, destructive, production-facing, or shared-environment tests without required authorization.
- State or imply that deferred tests passed.
- Hide the exact commands used.
- Omit failed attempts from the audit.

