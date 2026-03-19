---
name: codex-cli-invoke-persist
description: Create and resume Codex CLI sessions with a persistent session-name-to-thread_id mapping plus last-used model stored in a workspace-scoped JSON file under system temp. Invoke only when the user explicitly names `$codex-cli-invoke-persist` or otherwise clearly requests this exact skill. Use it for deterministic, session-persistent Codex automation across turns and processes.
---

# Codex CLI Invoke Persist

## Manual invocation

Invoke this skill explicitly by name (`$codex-cli-invoke-persist`) because it may execute the `codex` CLI.
Do not invoke it implicitly from a generic request for Codex help.

## Codex wrappers

Users may provide Codex wrappers with any name. They are often named `codex-<something>`, but they do not need to be.

- If the wrapper is an executable command or script such as `.sh`, `.bat`, `.cmd`, or a normal binary on `PATH`, pass it via `--codex-cmd "<wrapper command...>"` or `CODEX_CMD`.
- If the wrapper is a POSIX alias/function or a PowerShell function, use shell-launch mode: pass the wrapper name via `--codex-cmd`, add `--codex-shell posix|powershell`, and usually pass profile/setup code via `--codex-shell-init`. Optionally override the launcher with `--codex-shell-cmd`.
- The wrapper must accept the `codex exec` / `codex exec resume` flags used here, especially `--json`, `-o/--output-last-message`, `--model`, and the resume selector.

## Heartbeats and deadlines

- For long-running work, prefer streaming (`resume-stream`) so stdout emits a steady JSONL heartbeat from `codex exec resume --json`.
- Do not add timeouts or terminate the Codex process while it is still working unless one of these applies: (1) the user explicitly requested a deadline, (2) the process is clearly stalled, or (3) the process is in a hard error state. If a deadline is requested, pass `--deadline-seconds` to the helper.
- Do not interrupt simply because intermediate output appears sufficient. Let Codex run to normal completion and evaluate the final result afterward.
- `resume-json` uses Codex JSONL internally, but the helper emits one final JSON object on stdout instead of forwarding the heartbeat stream.

## Example triggers (user intent -> stage)

The skill itself is triggered by name, but once invoked, pick a stage based on user intent:

- Create/persist a new session (creation stage):
  - "Create a Codex session named review-src"
  - "Start a new Codex chat and save it as issue-1234"
  - "Make a new persistent Codex CLI session (session name: plan-feature-x)"
  - "Create a session named perf-review using role definition roles/perf_reviewer.md"
- Resume/call an existing session (invocation stage):
  - "Resume session review-src and ask: summarize the repo"
  - "Continue last chat"
  - "Use this thread_id and continue: <thread_id>"
- List existing sessions (listing stage):
  - "List my saved Codex sessions for this workspace"
  - "Show available session names"
  - "List sessions for workspace /abs/path/to/other/workspace"
- Delete saved sessions (deletion stage):
  - "Delete session name review-src"
  - "Remove saved Codex session review-src from this workspace"
  - "Delete all saved Codex sessions for this workspace"

## Stages (sub-skills)

- **Creation** (`create-session`)
  - **Required input:** `session name` (the user may call it a "session alias")
  - **Optional input:**
    - Role definition `.md`: `--role-definition-md /abs/path/to/role.md`
      Codex CLI has no `--append-system-prompt`, so the helper prepends the file contents to the initial user prompt.
    - Model: `--model ...` (stored as `last_model` for later resumes)
    - Wrapper command: `--codex-cmd "..."` / `CODEX_CMD`
    - Shell-launch mode for alias/function wrappers: `--codex-shell posix|powershell`, usually with `--codex-shell-init "..."` / `CODEX_SHELL_INIT`, and optionally with `--codex-shell-cmd "..."` / `CODEX_SHELL_CMD`
    - Deadline: `--deadline-seconds ...` only when the user explicitly requested a time limit
    - Exec pass-throughs supported by the helper: `--config`, `--enable`, `--disable`, `--image`, `--full-auto`, `--dangerously-bypass-approvals-and-sandbox`, `--skip-git-repo-check`
  - **Files touched:**
    - Reads: role definition `.md` (if provided)
    - Writes: workspace mapping JSON (safe writes may create `.bad.<pid>` and `.tmp.<pid>`)
  - **Keywords:** "create session", "start new chat", "save as <name>"

- **Invocation** (`resolve`, `resume-json`, `resume-stream`)
  - **Required input:** `prompt` plus `thread_id`/`session_id` or `session name`
  - **Behavior:**
    - Resolves `session name` via the workspace mapping JSON unless a direct `thread_id` is provided
    - Runs `codex exec resume <thread_id> --json`
    - Defaults `--model` from stored `last_model` when not explicitly provided
    - Persists the effective model as the new `last_model` after a successful call
    - Supports `--role-definition-md` by prepending the file contents to the resumed prompt for that call
    - Lets the Codex process run to natural completion unless deadline/stall/error criteria are met
  - **Deadlines:** support `--deadline-seconds` only when the user explicitly requested a time limit
  - **Keywords:** "continue latest chat", "continue last chat", "resume session"

- **Listing** (`list-sessions`)
  - **Required input:** none (defaults to current workspace)
  - **Optional input:** `workspace dir` or explicit `mapping file`
  - **Behavior:** reads the workspace mapping JSON and returns an empty list if it does not exist yet
  - **Keywords:** "list sessions", "show session names"

- **Deletion** (`delete-session`, `delete-all-sessions`)
  - **Required input:** either `thread_id`/`session_id` or `session name`, or explicit "all sessions"
  - **Optional input:** `workspace dir` or explicit `mapping file`
  - **Behavior:** removes entries from the workspace mapping JSON or deletes the mapping file entirely; does not call `codex`
  - **Keywords:** "delete session", "remove saved session", "clear all sessions"

- Management file (session mapping JSON): `<system-tmp>/agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/codex-cli-alias-mapping.json` (override with `--mapping-file`).

## Workflow decision

- Create a new persistent session: follow `references/creation.md`.
- Resume an existing session: follow `references/invocation.md`.
- List existing sessions for a workspace: follow `references/listing.md`.
- Delete saved sessions: follow `references/deletion.md`.

## Naming and selection rules

- Prefer the user's `session name`.
- If the user provides a `thread_id` or `session_id`, always use it as the resume target.
- If the user says "continue latest chat", "continue last chat", or similar, reuse the most recently referenced `thread_id` or session name in this conversation. If none exists, ask which session to use.

## Model persistence

- The manifest stores `last_model` per session name.
- On reuse: if the user does not specify `--model`, use the stored `last_model` for that session name when present.
- On override: if the user explicitly specifies `--model`, use it for this call and persist it as the new default for next time.

## Codex-specific limitations

- `codex exec resume` supports fewer flags than `codex exec`, so this helper intentionally persists only `last_model`, not broader exec-only options.
- Codex CLI has no Claude-style `--append-system-prompt`, so role definitions are prepended to the prompt text rather than attached as a separate system-prompt layer.
- `resume-json` is helper-generated JSON, not raw Codex CLI JSON, because the CLI itself exposes JSONL event streams via `--json`.
- Alias/function wrappers are supported only through shell-launch mode. If you pass an alias/function name without `--codex-shell`, the helper will treat it like a direct executable and it will likely fail.

## Resources

- Stage guide (creation): `references/creation.md`
- Stage guide (invocation): `references/invocation.md`
- Stage guide (listing): `references/listing.md`
- Stage guide (deletion): `references/deletion.md`
- Helper script: `scripts/invoke_persist.py` (create-session, resolve, list-sessions, delete-session, delete-all-sessions, resume-json, resume-stream)
