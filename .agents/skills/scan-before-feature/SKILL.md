---

name: scan-before-feature
description: Inspect the repository before implementing or expanding a feature. Use when the user asks to add, change, extend, or refactor product behavior, especially when similar behavior may already exist or the change spans multiple modules, layers, services, or policies.
---

# Repository Feature Preflight

Before implementing a feature, discover how the repository already solves the same or a closely related problem. Reuse the repository's established architecture, domain rules, user-facing behavior, and policy decisions unless the user explicitly requests a change.

## Core rule

Do not begin implementation from the requested feature name alone.

First:

1. Understand the requested behavior.
2. Search for analogous behavior and related domain concepts.
3. Trace the relevant execution paths.
4. Identify the existing policy and behavioral contract.
5. Choose whether to reuse, extend, or introduce a new pattern.
6. Only then edit code.

Do not treat the first matching file as sufficient evidence.

## Step 1: Read repository guidance

Before searching for implementation details:

* Read all applicable `AGENTS.md` files, starting at the repository root and continuing toward the target directory.
* Read relevant README files, architecture documents, contribution guides, and local design notes.
* Identify the build, test, lint, formatting, and type-check commands.
* Note generated-code boundaries, vendored code, deprecated modules, and directories that should not be edited.

Repository-specific instructions override this skill.

## Step 2: Translate the request into search concepts

Extract the following from the request:

* User-visible action or outcome
* Domain entities and state transitions
* Likely API, command, event, job, page, or component names
* Authorization or entitlement concepts
* Validation rules and failure cases
* Persistence or schema concepts
* User-facing strings and error messages
* Synonyms, abbreviations, older terminology, and likely renamed concepts

Search using both exact terms and semantic neighbors. A feature may already exist under a different name.

## Step 3: Map the repository before editing

Establish the relevant repository structure:

* Application entry points
* API routes, controllers, handlers, commands, or resolvers
* Domain and service layers
* Models, schemas, repositories, and migrations
* UI components and client-side state
* Background jobs, queues, events, and integrations
* Authentication, authorization, feature flags, and configuration
* Tests, fixtures, examples, and documentation
* Observability, audit logging, analytics, and error handling

Use available repository search tools such as `rg`, `git grep`, language-aware symbol search, or IDE references. Do not rely only on filenames.

## Step 4: Find similar features and precedents

Search for at least one of the following:

* The same user outcome implemented elsewhere
* The same domain entity used in another workflow
* A neighboring feature with similar authorization or validation
* A previous version, deprecated implementation, or migration
* Tests describing the expected behavior
* Documentation or examples describing product intent
* Git history showing why the behavior was introduced or changed

Useful history searches include:

* `git log -- <path>`
* `git log -S'<term>' -- <paths>`
* `git log -G'<pattern>' -- <paths>`
* `git blame <path>`

You can also examine existing tests to understand the previously intended or desired behavior of a feature, especially when implementation details are unclear or have changed over time.

Inspect history only where it helps explain intent; do not scan history without a concrete question.

## Step 5: Trace the complete behavior

For each relevant precedent, trace the vertical path from entry point to observable outcome.

Include, when applicable:

1. Request, event, command, or UI entry point
2. Authentication and authorization
3. Input parsing and validation
4. Domain rules and state transitions
5. Service orchestration
6. Data reads and writes
7. Transactions, locking, idempotency, and retries
8. Events, queues, webhooks, or external calls
9. Response mapping and user-visible output
10. Audit logs, metrics, analytics, and error reporting
11. Unit, integration, end-to-end, and regression tests

Follow callers and callees. Record relevant functions, methods, classes, types, constants, flags, tables, events, and tests by symbol name and file path.

Do not assume a helper's behavior from its name. Read its implementation and important callers.

## Step 6: Recover the existing policy contract

Determine what behavior users and maintainers already rely on.

Look specifically for:

* Who may perform the action
* Which resource states allow or deny it
* Defaults and fallback behavior
* Validation boundaries
* Error type, status, code, and message conventions
* Idempotency and duplicate-request behavior
* Ordering and precedence rules
* Data ownership and visibility
* Privacy, security, and audit requirements
* Compatibility and migration behavior
* Feature-flag and rollout rules
* Retry, timeout, and partial-failure behavior
* Side effects and notification behavior

Use the following evidence priority:

1. Explicit repository instructions and approved design documents
2. Tests that assert product behavior
3. Public interfaces and stable callers
4. Multiple consistent implementations
5. Current implementation details
6. Git history and comments

Treat a single implementation as evidence, not automatically as policy.

When evidence conflicts, state the conflict. Do not silently choose the most convenient behavior.

## Step 7: Classify the feature scope

Treat the feature as **large-scope** when any of these apply:

* It touches three or more architectural layers.
* It crosses multiple modules, packages, services, or applications.
* It changes authorization, billing, privacy, lifecycle state, persistence, or another product policy.
* It changes a shared interface, schema, event, protocol, or public API.
* It requires migration, backward compatibility, rollout, or coordinated updates.
* Several plausible implementations exist with different user-visible behavior.
* The relevant call graph cannot be understood from one local module.

For a small, local change, perform the same discovery with less reporting.

## Step 8: Produce a pre-implementation brief

Before editing a large-scope feature, write a concise brief containing:

### Requested behavior

Restate the intended user-visible outcome and explicit constraints.

### Existing precedents

List similar features and explain what each one does.

### Relevant execution paths

List all relevant symbols and paths, grouped by layer. For each important symbol, state its role and relationship to the requested behavior.

### Existing policy

State the behavior that appears intentional and should remain consistent. Separate:

* **Confirmed:** directly supported by instructions, tests, or stable interfaces
* **Inferred:** supported by repeated implementation patterns
* **Unknown:** not established by repository evidence

### Proposed approach

State whether to reuse, extend, extract, or create a pattern, and why.

### Impact and verification

List affected modules, compatibility concerns, tests to add or update, and commands to run.

Do not ask for approval merely because the feature is large. Ask the user only when unresolved policy conflicts would produce materially different user-visible behavior.

## Step 9: Implement consistently

During implementation:

* Prefer extending an established abstraction over duplicating logic.
* Preserve existing policy unless the request explicitly changes it.
* Keep naming, layering, dependency direction, errors, and response shapes consistent.
* Reuse shared authorization, validation, transaction, event, and audit mechanisms.
* Avoid broad refactors unless required for correctness or requested by the user.
* Update all affected callers, types, schemas, tests, fixtures, docs, flags, and telemetry.
* Add regression coverage for the policy being preserved.
* Add tests for new behavior, denied behavior, boundary cases, and failure paths.

If a new abstraction is necessary, explain why existing ones are insufficient.

## Step 10: Verify and review

Run the narrowest relevant checks first, then broader checks when practical:

1. Focused unit or component tests
2. Relevant integration tests
3. Type checking, linting, and formatting
4. Broader test suites affected by shared interfaces
5. Migration or compatibility checks
6. Diff review for unintended behavior changes

After implementation, compare the final behavior against both:

* The user's request
* The recovered existing policy contract

Report:

* What was reused
* What was changed
* Which policy decisions were preserved
* Any unresolved uncertainty
* Tests and checks run
* Important checks not run and why

## Prohibited shortcuts

Do not:

* Implement immediately after finding one keyword match.
* Copy a similar feature without tracing its callers and policy.
* Create a second authorization or validation path when a shared one exists.
* Assume tests are irrelevant because the code appears straightforward.
* Change user-visible policy as an accidental side effect of refactoring.
* Ignore deprecated code without confirming what replaced it.
* Claim repository-wide consistency without searching the relevant modules.
* List every repository symbol; include symbols that affect behavior, policy, integration, or verification.

