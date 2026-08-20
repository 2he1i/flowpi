# Agent instructions

- After completing a code-change task and validating it, commit the task-scoped changes and push the current branch by default.
- Before committing, inspect the worktree and stage only files required for the requested task.
- Do not include tests, generated files, temporary files, or unrelated pre-existing changes unless the user explicitly asks for them.
- Preserve unrelated uncommitted work in the worktree.
- If a task explicitly excludes tests or other files, leave those files uncommitted even when they were created during investigation.
- Do not run GPU tests unless they are necessary for validation. If GPU validation is necessary, stop the user's existing process first to avoid low GPU utilization.
