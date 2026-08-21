# Agent instructions

- After completing a code-change task and validating it, commit the task-scoped changes and push the current branch by default.
- Before committing, inspect the worktree and stage only files required for the requested task.
- Do not include tests, generated files, temporary files, or unrelated pre-existing changes unless the user explicitly asks for them.
- Preserve unrelated uncommitted work in the worktree.
- If a task explicitly excludes tests or other files, leave those files uncommitted even when they were created during investigation.
- Prefer GPU execution for tests that support and benefit from it; use CPU only when GPU execution is unavailable or not meaningful. Before GPU validation, stop the two intentional empty keepalive workers started by `/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zheli/gpu_stress.sh`; after validation, restart that script.
