- Preserve existing behavior unless the user explicitly asks to change it.
- Keep the agent generic. Do not hardcode case-specific behavior — per-module shortcuts, per-mechanism formulas, per-question heuristics, or per-PX4-commit special cases. The same code path must work across any PX4 module, including those the agent has not been pointed at yet.
- Do not maintain lookup tables (parameter lists, topic lists, mechanism catalogs, constant deny-lists, helper-name registries) that need updating as new domains, mechanisms, or PX4 versions are encountered. If a fact can be derived from the source under inspection, the .msg schema, or the ULog itself, derive it instead of tabulating it.
- Keep generated reports and schemas stable unless the user asks to change them.
- Prefer small, testable modules over putting all logic in runner.py.
- Add or update tests when implementing new features.
- After code changes, run pytest.
- Keep changes focused on the requested scope.
- Prefer simple, runnable code over abstract architecture.
- When creating commits, use the format `type(scope): description` and include a list-format body describing the changes.
- Before creating commits, carefully read the diff and understand the changes being committed.
- Before editing a file or creating a new file, explain to the user what you're about to change and ask for approval.
- Do not edit or create files until the user approves the proposed change.
- Ask for explicit user approval before using user-provided API keys, calling paid APIs, or running commands that can spend tokens or usage credits.
- AGENTS.md is for Codex build/edit instructions; runtime flight-log agent behavior belongs in code prompts or agent instructions.
- Run Python scripts and install Python packages only inside the user-created virtual environment. If no virtual environment exists, ask the user to create one before proceeding.
- Run pytest through the virtual environment's Python module invocation, for example `.venv/bin/python -m pytest`, so repository-local imports and pytest.ini are honored.

Before implementing or substantially changing a feature, use the
`$scan-before-feature` skill.

Do not begin editing until you have:

1. Searched for similar existing behavior.
2. Traced the relevant callers, callees, tests, and data flow.
3. Identified existing authorization, validation, state-transition,
   error-handling, and user-facing policy.
4. Determined which existing logic should be reused or extended.

For large-scope changes, produce the pre-implementation brief required by
the skill before modifying code.
