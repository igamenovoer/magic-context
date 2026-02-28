# Claude Code session creation (headless)

Goal: create a Claude Code conversation session and capture the returned `session_id` so later invocations can resume with `--resume <session_id>`.

## Create a new session (JSON output)

```bash
claude -p "Initialize a new session. Reply only: OK" --output-format json
```

Parse:

- `.result` (usually the assistant text)
- `.session_id` (persist this for multi-turn)

## Resume a session (deterministic automation)

```bash
claude -p "Continue from where we left off" --resume "<session_id>" --output-format json
```

Prefer `--resume` over `--continue` in automation because it is explicit and deterministic.

## Persist alias→session mapping (system temp)

For multi-process workflows, persist an alias→session mapping file:

- Default path: `<system-tmp>/agent-sessions/<basename-of-workspace>-<md5-hex-string-of-abs-path-of-workspace>/claude-code-alias-mapping.json`
- Store `workspace_dir` once at top-level; store per-alias metadata under `aliases`: `session_id`, `created_at`
- Overwrite alias entries unconditionally
- Conflict handling is left to the user (avoid unintentional alias collisions across processes)

The helper script `scripts/create_session.py` implements this behavior and will still succeed if the mapping file cannot be written (it prints a warning to stderr).
