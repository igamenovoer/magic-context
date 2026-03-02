# Listing stage (list existing sessions)

Goal: list session aliases (and their stored `session_id` metadata) for a given workspace.

## Default behavior

By default, list sessions for the current working directory's workspace mapping file.

```bash
python3 scripts/invoke_persist.py list-sessions
```

## List sessions for a different workspace dir

```bash
python3 scripts/invoke_persist.py list-sessions --workspace-dir "/abs/path/to/other/workspace"
```

## Optional output shapes

Print only alias names (one per line):

```bash
python3 scripts/invoke_persist.py list-sessions --print-aliases
```

Print machine-readable JSON (default):

```bash
python3 scripts/invoke_persist.py list-sessions
```

## Notes

- The mapping file lives under system temp and is keyed by the workspace absolute path.
- Each alias entry may include `last_model` and `last_reasoning_effort` (the defaults used when resuming that session, unless explicitly overridden).
- If the mapping file does not exist yet, the command returns an empty list.
