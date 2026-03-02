---
name: claude-code-session-call
description: Resume a persisted Claude Code conversation via the `claude` CLI using `--resume` with either an explicit `session_id` or a user-provided `session name`/`session alias` (legacy `alias`) stored by `$claude-code-create-session` in a workspace-scoped alias-to-session_id mapping file. Use when the user wants deterministic, session-persistent Claude Code automation.
---

# Claude Code Session Call

## Manual invocation

Invoke this skill explicitly by name (`$claude-code-session-call`) because it may execute the `claude` CLI.

## Required input (don't invent)

- `prompt`: what to send to Claude.
- Session selector (pick one):
  - `session_id` (preferred if provided)
  - `session name` or `session alias` (preferred naming)
  - `alias` (legacy fallback)

If the user asks to resume but does not specify which session, ask for the session name/alias (or the `session_id`).

If the user says "continue last chat" (or similar), use the most recently referenced `session_id` or session name/alias in this conversation. If none exists, ask for the session name/alias.

## Resolve `session_id`

Resolution order:

1. If the user provided `session_id`, use it as-is.
2. Otherwise, resolve `session name` / `session alias` / `alias` via the alias mapping file created by `$claude-code-create-session`.

Default mapping file path:

- `<system-tmp>/agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/claude-code-alias-mapping.json`

This default path depends on your current working directory. Run from the same workspace directory used when creating the session (or pass `--mapping-file`).

Helper (resolve without calling Claude):

```bash
python3 scripts/session_call.py resolve --session-alias "review-src" --print-session-id
```

If the alias is missing, tell the user to create it first with `$claude-code-create-session`.

## Resume and call Claude Code

Machine-readable JSON (recommended):

```bash
python3 scripts/session_call.py resume-json --session-alias "review-src" --prompt "Continue from where we left off"
```

Print just the assistant text:

```bash
python3 scripts/session_call.py resume-json --session-alias "review-src" --prompt "Continue from where we left off" --print-result
```

Streaming JSONL (progress/events):

```bash
python3 scripts/session_call.py resume-stream --session-alias "review-src" --prompt "Continue from where we left off" --verbose --include-partials
```

Credentials: ensure `claude` is on PATH and authenticated. To load env vars from a file, pass `--env-file /path/to/vars.env` (KEY=VALUE lines).

## Output contract (what to respond with)

When this skill runs, respond with:

- The `session_id` used (and the session name/session alias if applicable).
- The Claude answer (or a short parsed summary if streaming).

## Resources

- Reference: `references/session-call.md`
- Helper script: `scripts/session_call.py` (resolve alias->session_id and call `claude --resume`)
