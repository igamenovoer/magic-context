---
name: claude-code-invoke-persist
description: Create and resume Claude Code CLI sessions with a persistent session-name-to-session_id mapping (plus last-used model and reasoning effort) stored in a workspace-scoped JSON file under system temp. Use when the user wants deterministic, session-persistent Claude Code automation across turns and processes.
---

# Claude Code Invoke Persist

## Manual invocation

Invoke this skill explicitly by name (`$claude-code-invoke-persist`) because it may execute the `claude` CLI.

## Claude wrappers

If the user provides a claude-compatible wrapper command (drop-in replacement for `claude`), use it by passing `--claude-cmd "<wrapper command...>"` to `scripts/invoke_persist.py` (or set `CLAUDE_CMD`). The wrapper must be an executable command (not a shell alias) and must accept the `claude` flags used here (`-p`, `--output-format`, `--resume`, `--model`, `--effort`, `--append-system-prompt`, and streaming flags).

## Heartbeats and deadlines

- For long-running work, prefer streaming (`resume-stream`) so stdout emits a steady heartbeat (one JSON object per line).
- Do not add timeouts or terminate the Claude process while it is still working unless the user explicitly requested a deadline (for example: "up to 5 minutes"). If a deadline is requested, pass `--deadline-seconds` to the helper.

## Example triggers (user intent -> stage)

The skill itself is triggered by name, but once invoked, pick a stage based on user intent:

- Create/persist a new session (creation stage):
  - "Create a Claude session named review-src"
  - "Start a new Claude chat and save it as issue-1234"
  - "Make a new persistent Claude Code session (session name: plan-feature-x)"
  - "Create a session named perf-review using role definition roles/perf_reviewer.md"
- Resume/call an existing session (invocation stage):
  - "Resume session review-src and ask: summarize the repo"
  - "Continue last chat" (use the latest referenced session_id or session name in this conversation)
  - "Use this session_id and continue: <session_id>"
- List existing sessions (listing stage):
  - "List my saved Claude sessions for this workspace"
  - "Show available session names"
  - "List sessions for workspace /abs/path/to/other/workspace"
- Delete saved sessions (deletion stage):
  - "Delete session name review-src"
  - "Remove saved Claude session review-src from this workspace"
  - "Delete all saved Claude sessions for this workspace"

## Stages (sub-skills)

- **Creation** (`create-session`)
  - **Required input:** `session name` (the user may call it a "session alias")
  - **Optional input:**
    - Role definition `.md` (system prompt): `--role-definition-md /abs/path/to/role.md` (appended via `claude --append-system-prompt`)
    - Model: `--model ...` (stored as `last_model` for later resumes)
    - Reasoning effort: `--reasoning-effort ...` (mapped to `claude --effort`; stored as `last_reasoning_effort`)
    - Wrapper command: `--claude-cmd "..."` / `CLAUDE_CMD`
    - Deadline: `--deadline-seconds ...` only when the user explicitly requested a time limit
  - **Files touched:**
    - Reads: role definition `.md` (if provided)
    - Writes: workspace mapping JSON (safe writes may create `.bad.<pid>` and `.tmp.<pid>`)
  - **Keywords:** "create session", "start new chat", "save as <name>"

- **Invocation** (`resolve`, `resume-json`, `resume-stream`)
  - **Required input:** `prompt` plus `session_id` or `session name`
  - **Behavior:**
    - Resolves `session name` via the workspace mapping JSON (unless `session_id` is provided)
    - Runs `claude --resume <session_id>` (or wrapper via `--claude-cmd`/`CLAUDE_CMD`)
    - Defaults `--model` / `--reasoning-effort` from stored `last_model` / `last_reasoning_effort` when not explicitly provided
    - Persists the effective model/effort as the new `last_*` values after a successful call
  - **Deadlines:** support `--deadline-seconds` only when the user explicitly requested a time limit
  - **Keywords:** "continue latest chat", "continue last chat", "resume session"

- **Listing** (`list-sessions`)
  - **Required input:** none (defaults to current workspace)
  - **Optional input:** `workspace dir` or explicit `mapping file`
  - **Behavior:** reads the workspace mapping JSON and returns an empty list if it does not exist yet
  - **Keywords:** "list sessions", "show session names"

- **Deletion** (`delete-session`, `delete-all-sessions`)
  - **Required input:** either `session_id` or `session name`, or explicit "all sessions"
  - **Optional input:** `workspace dir` or explicit `mapping file`
  - **Behavior:** removes entries from the workspace mapping JSON (safe rewrites may create `.bad.<pid>`/`.tmp.<pid>`) or deletes the mapping file entirely; does not call `claude`
  - **Keywords:** "delete session", "remove saved session", "clear all sessions"

- Management file (session mapping JSON): `<system-tmp>/agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/claude-code-alias-mapping.json` (override with `--mapping-file`).

## Workflow decision

- Create a new persistent session: follow `references/creation.md`.
- Resume an existing session: follow `references/invocation.md`.
- List existing sessions for a workspace: follow `references/listing.md`.
- Delete saved sessions: follow `references/deletion.md`.

## Naming and selection rules

- Prefer the user's `session name` (the user may call it a "session alias").
- If the user provides a `session_id`, always use it as the resume target.
- If the user says "continue latest chat", "continue last chat", or similar, reuse the most recently referenced `session_id` or session name in this conversation. If none exists, ask which session to use.

## Model and reasoning-effort persistence

- The manifest (mapping JSON) stores `last_model` and `last_reasoning_effort` per session name.
- On reuse: if the user does not specify `--model` / `--reasoning-effort`, use the stored `last_model` / `last_reasoning_effort`.
- On override: if the user explicitly specifies `--model` and/or `--reasoning-effort`, use those values for this call and persist them as the new `last_*` defaults for next time.

## Resources

- Stage guide (creation): `references/creation.md`
- Stage guide (invocation): `references/invocation.md`
- Stage guide (listing): `references/listing.md`
- Stage guide (deletion): `references/deletion.md`
- Helper script: `scripts/invoke_persist.py` (create-session, resolve, list-sessions, delete-session, delete-all-sessions, resume-json, resume-stream)
