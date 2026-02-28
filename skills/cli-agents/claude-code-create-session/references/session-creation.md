# Claude Code session creation (headless)

Goal: create a Claude Code conversation session and capture the returned `session_id` so later invocations can resume with `--resume <session_id>`.

## Create a new session (JSON output)

```bash
claude -p "Initialize a new session. Reply only: OK" --output-format json
```

If you have a Claude wrapper (executable, `.sh`, or shell alias), substitute it for `claude` (example: `claude-wrapper -p ...`).

Parse:

- `.result` (usually the assistant text)
- `.session_id` (persist this for multi-turn)

Run the command from the intended workspace root (or pass `--workspace-dir /abs/path` when recording the mapping).

## Resume a session (deterministic automation)

```bash
claude -p "Continue from where we left off" --resume "<session_id>" --output-format json
```

Prefer `--resume` over `--continue` in automation because it is explicit and deterministic.

## Persist session-name→session mapping (system temp)

For multi-process workflows, persist a session-name→session mapping file:

- Default path: `<system-tmp>/agent-sessions/<basename-of-workspace>-<md5-hex-string-of-abs-path-of-workspace>/claude-code-session-mapping.json`
- Store `workspace_dir` once at top-level; store per-session metadata under `sessions`: `session_id`, `created_at`
- Overwrite session names unconditionally
- Conflict handling is left to the user (avoid unintentional session name collisions across processes)

The helper script `scripts/record_session.py` updates the mapping file by extracting `.session_id` from stdin JSON (or `--session-id`) and will still succeed if the mapping file cannot be written (it prints a warning to stderr).
