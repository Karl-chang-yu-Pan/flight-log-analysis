# Repository Agent Instructions

## General engineering rules

* Preserve existing behavior unless the user explicitly asks to change it.

* Keep the agent generic. Do not hardcode case-specific behavior — per-module shortcuts, per-mechanism formulas, per-question heuristics, or per-PX4-commit special cases. The same code path must work across any PX4 module, including those the agent has not been pointed at yet.

* Do not maintain lookup tables such as parameter lists, topic lists, mechanism catalogs, constant deny-lists, or helper-name registries that require manual updates as new domains, mechanisms, or PX4 versions are encountered. If a fact can be derived from the source under inspection, the `.msg` schema, or the ULog itself, derive it instead of tabulating it.

* Keep generated reports and schemas stable unless the user explicitly asks to change them.

* Prefer small, testable modules over concentrating unrelated logic in `runner.py` or another large orchestration module.

* Prefer simple, runnable code over unnecessary abstraction.

* Keep changes focused on the requested scope. Do not absorb adjacent cleanup, refactoring, or architectural work unless it is required for correctness.

## Repository inspection before implementation

Before implementing or substantially changing a feature, use the `$wayfinder` skill to inspect the current repository.

Do not begin implementation until you have:

1. Searched for similar existing behavior.
2. Traced the relevant callers, callees, tests, and data flow.
3. Identified existing validation, authorization where applicable, state transitions, error handling, and user-facing behavior.
4. Determined which existing logic should be reused, extended, or deliberately left unchanged.
5. Checked relevant documentation, ADRs, specifications, and existing tests for constraints that apply to the change.

For large-scope or architecture-sensitive changes, produce a concise pre-implementation assessment before modifying code.

If repository findings conflict with an existing document or specification, report the conflict instead of silently choosing one interpretation.

## Editing and approval

Before the first file modification for a task, explain what you intend to change and which files or areas are expected to be affected.

Ask for explicit approval before editing when the user has requested analysis, planning, review, investigation, or specification only.

If the user has already explicitly instructed you to implement, fix, edit, apply, or carry out the described change, that instruction counts as approval for changes within the stated scope. Do not repeatedly ask for approval before every file.

Do not expand beyond the approved scope without reporting the newly discovered dependency and obtaining approval when it materially changes the task.

## Testing

* Add or update tests when implementing new behavior or correcting behavior that should be pinned by regression coverage.

* Prefer focused tests that directly prove the intended semantic behavior rather than relying only on counts, exit status, absence of exceptions, or broad end-to-end success.

* After code changes, run the focused relevant pytest scope first.

* Then run the appropriate nearby regression suite when practical.

* Run pytest only through the user-created virtual environment, for example:

  `.venv/bin/python -m pytest`

  so repository-local imports and `pytest.ini` are honored.

* Run Python scripts and install Python packages only inside the user-created virtual environment.

* If no suitable virtual environment exists, ask the user to create one before running Python scripts, pytest, or package installation.

* Do not claim a change is verified solely because a process exited successfully. Verification should assert the behavior relevant to the change.

## Commit requirements

Before creating any commit:

1. Carefully read the complete diff and understand the changes being committed.
2. Verify that only files belonging to the intended logical change are staged.
3. Do not include unrelated modified or untracked files.
4. Run the tests required for that change.
5. Follow the repository commit format below.

Use the commit subject format:

`type(scope): description`

Use an imperative, concise description.

The commit body must use list-format bullets describing the material changes.

Example:

`fix(dag): preserve scoped parameter identity`

`- Thread caller scope through parameter writer resolution.`

`- Preserve exact unresolved source identity for downstream matching.`

`- Add regression coverage for unrelated same-name parameters.`

Keep logically independent changes in separate commits unless the user explicitly asks to squash or consolidate them.

### AI co-author trailer

When an AI coding agent materially contributes to a commit, include a `Co-Authored-By` trailer identifying the model that materially produced the change.

Leave one blank line between the list-format body and the trailer.

Do not infer the model identity from the coding frontend or harness. For example, running through OpenCode, Codex CLI, or another agent frontend does not by itself establish which model produced the change.

If the exact model identity is known, use its established repository attribution.

For commits materially produced by Codex GPT-6, use exactly:

`Co-Authored-By: Codex GPT-6 <noreply@openai.com>`

If the agent is not certain which model is currently being used, or is not certain which attribution identity corresponds to that model, it must ask the user before creating the commit.

Do not guess the model name.

Do not copy a `Co-Authored-By` identity from an earlier commit merely because it appears in repository history.

Do not silently substitute the frontend name, provider name, or a generic AI identity for the actual model.

If the user identifies the model or specifies the required trailer, use that identity exactly unless it conflicts with another explicit repository rule.

When amending, squashing, rebasing, or otherwise rewriting AI-assisted commits, preserve the appropriate `Co-Authored-By` trailer on each resulting commit.

If multiple AI models materially contributed to the same resulting commit and the user wants each contribution attributed, include one valid `Co-Authored-By` trailer per materially contributing model.

## Commit history safety

* Do not rewrite pushed/shared history unless the user explicitly approves it.

* Local, unpushed commits may be amended, squashed, or rebased when the user requests history cleanup.

* Before rewriting history, inspect the working tree and preserve unrelated modified/untracked files.

* After history rewriting, verify both the resulting commit contents and the remaining working-tree state.

## Agent skills and workflow

Use the available repository skills according to the task rather than treating every change as the same workflow.

* `$wayfinder`: inspect and trace the repository before implementation or when verifying current behavior.
* `$architecture-critic`: challenge architecture, boundaries, ownership, and proposed repair direction.
* `$domain-modeling`: define or refine domain terminology, invariants, CONTEXT.md, and ADR decisions.
* `$grill-with-docs`: challenge proposed decisions or models against repository documentation and evidence.
* `$grilling`: adversarially review a plan or implementation specification before execution.
* `$to-spec`: turn an accepted design into an implementation-ready specification.
* `$to-tickets`: split an accepted specification into independently actionable work items when ticketing is useful.
* `$tdd`: use red/green/refactor for behavior that should be established through tests.
* `$implement-spec`: implement an accepted specification without reopening already-decided architecture.
* `$implement`: perform bounded implementation work when a separate formal spec is unnecessary.
* `$code-review`: review completed changes against intended behavior, repository constraints, tests, and regressions.
* `$codebase-design` / `$improve-codebase-architecture`: use only when the task explicitly concerns broader structural design or architectural improvement.

Do not run every skill mechanically. Use the smallest workflow appropriate to the task.

When an ADR or accepted specification exists, treat it as a constraint during implementation rather than casually reopening the decision. Report contradictions instead.

## External services and cost

Ask for explicit user approval before:

* using user-provided API keys;
* calling paid APIs;
* running commands that can spend tokens, credits, or other paid usage.

Do not expose credentials in logs, commits, generated files, or command output.

## Instruction placement

`AGENTS.md` contains coding-agent instructions for repository analysis, editing, testing, and Git operations.

Runtime flight-log-agent behavior belongs in runtime prompts, application code, configuration, or other runtime-specific instructions rather than in `AGENTS.md`.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (uses the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

