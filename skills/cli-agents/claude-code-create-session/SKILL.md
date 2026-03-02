---
name: claude-code-create-session
description: Create a new Claude Code dialog session and capture its `session_id` under a user-provided session name/session alias (identity string) for later multi-turn resumes with `claude --resume`. Invoke explicitly as `$claude-code-create-session` to start and name a persistent Claude Code conversation and persist a session-name→session mapping in a workspace-scoped JSON file under the system temp directory (overwriting the existing entry if the same session name/alias is reused). If the mapping file cannot be written, keep the mapping in chat history and warn the user, but still proceed.
---

# Claude Code Create Session

## Manual invocation

Invoke this skill explicitly by name (`$claude-code-create-session`). It is designed to be run on-demand so the session-name→session mapping is created intentionally.

## Required input

- `session name` / `session alias` (identity string): a short name the user will use later to refer to this session (example: `review-src`, `issue-1234`, `plan-feature-x`).

When determining the name from chat context, use this priority:

1. Explicit `session name` value.
2. Explicit `session alias` value.
3. Fallback legacy `alias` value.

If none are provided, ask the user for the session name/alias (don’t invent it).

## Create the session (headless)

Prefer machine-readable output:

```bash
claude -p "Initialize a new session named '<session_name_or_alias>'. Reply only: OK" --output-format json
```

Parse and persist `.session_id` from the JSON output.

Record the returned `.session_id` under the session name (without hiding how `claude` is called):

```bash
claude -p "Initialize a new session named '<session_name_or_alias>'. Reply only: OK" --output-format json | \
  python3 scripts/record_session.py --session-name "<session_name_or_alias>" --print-mapping-json
```

Run the command from the intended workspace root (or pass `--workspace-dir /abs/path` to `scripts/record_session.py`).

If you use a Claude wrapper (executable, `.sh`, or shell alias), replace `claude` with it (placeholder):

```bash
<custom-wrapper-name> -p "Initialize a new session named '<session_name_or_alias>'. Reply only: OK" --output-format json | \
  python3 scripts/record_session.py --session-name "<session_name_or_alias>" --print-mapping-json
```

Credentials: ensure the chosen Claude command can authenticate in your environment (for example by exporting env vars or sourcing an env file before running the command).

## Record the session-name→session mapping (in chat context)

Persist the session-name mapping to a workspace-scoped JSON file in system temp:

- Path: `<system-tmp>/agent-sessions/<basename-of-workspace>-<md5-hex-string-of-abs-path-of-workspace>/claude-code-session-mapping.json`
- File fields: top-level `workspace_dir` plus `sessions` (session name→{`session_id`, `created_at`})
- Overwrite behavior: if a session name already exists in the file, overwrite it unconditionally
- Conflict handling: user is responsible for avoiding unintended session name collisions across processes

Use the helper script output’s `mapping_file` field as the source of truth for the exact path.

Also keep the mapping in the current chat history (so later turns in this same chat can resume by session name/alias without reading the file).

If writing the mapping file fails for any reason (permissions, missing tmp, etc.), still keep the chat-history mapping and **warn** the user, but do not error out.

Registry format (JSON):

```json
{
  "workspace_dir": "/abs/path/to/workspace",
  "sessions": {
    "<session-name>": {
      "session_id": "REDACTED",
      "created_at": "2026-02-28T00:00:00+00:00"
    }
  }
}
```

## Output contract (what to respond with)

When this skill runs, respond with:

- The created `session_id` (and the session name/session alias it was stored under).
- The mapping file path used and whether it was written successfully.
- The updated session-name→session mapping JSON block.
- If the mapping file write failed: include a warning.

## Resources

- Reference: `references/session-creation.md`
- Helper script: `scripts/record_session.py` (records `session_id` from stdin JSON or `--session-id` and updates the mapping file)
