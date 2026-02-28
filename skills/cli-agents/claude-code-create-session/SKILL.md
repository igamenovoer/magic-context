---
name: claude-code-create-session
description: Create a new Claude Code dialog session and capture its `session_id` under a user-provided alias (identity string) for later multi-turn resumes with `claude --resume`. Invoke explicitly as `$claude-code-create-session` to start and name a persistent Claude Code conversation and persist an alias→session mapping in the system temp directory at `agent-sessions/claude-code-alias-mapping.yaml` (overwrite alias if it already exists). If the mapping file cannot be written, keep the mapping in chat history and warn the user, but still proceed.
---

# Claude Code Create Session

## Manual invocation

Invoke this skill explicitly by name (`$claude-code-create-session`). It is designed to be run on-demand so the alias→session mapping is created intentionally.

## Required input

- `alias` (identity string): a short name the user will use later to refer to this session (example: `review-src`, `issue-1234`, `plan-feature-x`).

If `alias` is missing, ask the user for it (don’t invent it).

## Create the session (headless)

Prefer machine-readable output:

```bash
claude -p "Initialize a new session named '<alias>'. Reply only: OK" --output-format json
```

Parse and persist `.session_id` from the JSON output.

Recommended helper (also updates the default alias mapping file in system temp):

```bash
python3 scripts/create_session.py --alias "<alias>" --print-mapping-json
```

Credentials: ensure the `claude` CLI can authenticate in your environment. If you want the helper to load env vars from a file, pass `--env-file /path/to/vars.env` (KEY=VALUE lines).

## Record the alias→session mapping (in chat context)

Persist the alias mapping to a YAML file in system temp:

- Path: `<system-tmp>/agent-sessions/claude-code-alias-mapping.yaml`
- Fields per alias: `session_id`, `workspace_dir`, `created_at`
- Overwrite behavior: if `alias` already exists in the file, overwrite it unconditionally
- Conflict handling: user is responsible for avoiding unintended alias collisions across processes

Also keep the mapping in the current chat history (so later turns in this same chat can resume by alias without reading the file).

If writing the mapping file fails for any reason (permissions, missing tmp, etc.), still keep the chat-history mapping and **warn** the user, but do not error out.

Registry format (YAML):

```yaml
claude_code_sessions:
  <alias>:
    session_id: "REDACTED"
    workspace_dir: "/abs/path/to/workspace"
    created_at: "2026-02-28T00:00:00+00:00"
```

## Output contract (what to respond with)

When this skill runs, respond with:

- The created `session_id` (and the `alias` it was stored under).
- The mapping file path used and whether it was written successfully.
- The updated `claude_code_sessions` registry block.
- If the mapping file write failed: include a warning.

## Resources

- Reference: `references/session-creation.md`
- Helper script: `scripts/create_session.py` (creates a session and updates the default mapping file in system temp)
