# Deletion stage (delete saved sessions)

Goal: remove a persisted session entry by session name or `thread_id`, or delete the entire workspace mapping file.

Notes:

- This stage does not call the `codex` CLI. It only deletes entries from the workspace-scoped manifest file created by the creation stage.
- If you use multiple workspaces, be explicit about which workspace you are operating on via `--workspace-dir` or `--mapping-file`.

## Delete a particular saved session

Delete by session name, which is preferred:

```bash
python3 scripts/invoke_persist.py delete-session --session-name "review-src"
```

Delete by `thread_id`, which removes any alias entries pointing at that thread:

```bash
python3 scripts/invoke_persist.py delete-session --thread-id "019d063a-8371-7ba3-9710-7388fc43c8df"
```

Operate on a different workspace dir:

```bash
python3 scripts/invoke_persist.py delete-session --session-name "review-src" --workspace-dir "/abs/path/to/other/workspace"
```

## Delete all saved sessions for a workspace

Delete the entire manifest file:

```bash
python3 scripts/invoke_persist.py delete-all-sessions
```

For a different workspace:

```bash
python3 scripts/invoke_persist.py delete-all-sessions --workspace-dir "/abs/path/to/other/workspace"
```

## Output contract (what to respond with)

When the deletion stage runs, respond with:

- What was deleted: alias names and or mapping file.
- The workspace dir and mapping file path used.
- The count of remaining aliases when applicable.
