---
name: claude-code-invoke-persist
description: Create and resume Claude Code CLI sessions with a persistent alias-to-session_id mapping stored in a workspace-scoped JSON file under system temp. Use when the user wants deterministic, session-persistent Claude Code automation across turns and processes.
---

# Claude Code Invoke Persist

## Manual invocation

Invoke this skill explicitly by name (`$claude-code-invoke-persist`) because it may execute the `claude` CLI.

## Claude wrappers

If the user provides a claude-compatible wrapper command (drop-in replacement for `claude`, like in `call-claude-code`), use it by passing `--claude-cmd "<wrapper command...>"` to `scripts/invoke_persist.py` (or set `CLAUDE_CMD`). The wrapper must be an executable command (not a shell alias) and must accept the `claude` flags used here (`-p`, `--output-format`, `--resume`, and streaming flags).

## Heartbeats and deadlines

- For long-running work, prefer streaming (`resume-stream`) so stdout emits a steady heartbeat (one JSON object per line).
- Do not add timeouts or terminate the Claude process while it is still working unless the user explicitly requested a deadline (for example: "up to 5 minutes"). If a deadline is requested, pass `--deadline-seconds` to the helper.

## Example triggers (user intent -> stage)

The skill itself is triggered by name, but once invoked, pick a stage based on user intent:

- Create/persist a new session (creation stage):
  - "Create a Claude session named review-src"
  - "Start a new Claude chat and save it as issue-1234"
  - "Make a new persistent Claude Code session (session alias: plan-feature-x)"
- Resume/call an existing session (invocation stage):
  - "Resume session review-src and ask: summarize the repo"
  - "Continue last chat" (use the latest referenced session_id or alias in this conversation)
  - "Use this session_id and continue: <session_id>"
- List existing sessions (listing stage):
  - "List my saved Claude sessions for this workspace"
  - "Show available session aliases"
  - "List sessions for workspace /abs/path/to/other/workspace"

## Stages (sub-skills)

- Creation (`create-session`): required `session name` / `session alias` (legacy fallback: `alias`); keywords like "create session", "start new chat", "save as <name>"; writes the workspace mapping JSON (may create `.bad.<pid>` and `.tmp.<pid>` files during safe writes); supports `--claude-cmd`/`CLAUDE_CMD` if the user provides a wrapper.
- Invocation (`resolve`, `resume-json`, `resume-stream`): required `prompt` plus `session_id` or `session name`/`session alias` (legacy fallback: `alias`); keywords like "continue latest chat" / "continue last chat"; reads the workspace mapping JSON (unless `session_id` is given) and runs `claude --resume <session_id>` (or wrapper via `--claude-cmd`/`CLAUDE_CMD`); supports `--deadline-seconds` only when the user requested a time limit.
- Listing (`list-sessions`): required input: none (defaults to current workspace); optional `workspace dir` or explicit `mapping file`; keywords like "list sessions", "show aliases"; reads the workspace mapping JSON and returns an empty list if it does not exist yet.

- Management file (alias mapping JSON): `<system-tmp>/agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/claude-code-alias-mapping.json` (override with `--mapping-file`).

## Workflow decision

- Create a new persistent session: follow `references/creation.md`.
- Resume an existing session: follow `references/invocation.md`.
- List existing sessions for a workspace: follow `references/listing.md`.

## Naming and selection rules

- Prefer the user's `session name` / `session alias` if provided (fallback legacy: `alias`).
- If the user provides a `session_id`, always use it as the resume target.
- If the user says "continue latest chat", "continue last chat", or similar, reuse the most recently referenced `session_id` or session name/alias in this conversation. If none exists, ask which session to use.

## Resources

- Stage guide (creation): `references/creation.md`
- Stage guide (invocation): `references/invocation.md`
- Stage guide (listing): `references/listing.md`
- Helper script: `scripts/invoke_persist.py` (create-session, resolve, list-sessions, resume-json, resume-stream)
