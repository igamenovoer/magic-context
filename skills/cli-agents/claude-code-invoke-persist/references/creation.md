# Creation stage (persist a new Claude Code session)

Goal: create a new Claude Code session, capture its `session_id`, and persist a user-facing session name (the user may call it a "session alias") for later deterministic resumes.

## Required input (don't invent)

- `session name` (the user may call it a "session alias")

If missing, ask the user for it.

## Optional input (only if user provides it)

- Role definition Markdown (`.md`) file path: appended to Claude's default system prompt via `--append-system-prompt`

## Create a new session (headless)

Prefer machine-readable JSON output and store the returned `session_id`:

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --print-mapping-json
```

If the user provides a role definition Markdown file (system prompt), append it at session creation:

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --role-definition-md "/abs/path/to/role.md" --print-mapping-json
```

Optional: if the user explicitly chooses a Claude model and/or reasoning effort, pass them at creation time so they become the stored defaults for later resumes:

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --model "MODEL" --reasoning-effort "EFFORT" --print-mapping-json
```

Credentials: ensure the `claude` CLI is on PATH and authenticated. To load env vars from a file, pass `--env-file /path/to/vars.env` (KEY=VALUE lines).

If the user provides a claude-compatible wrapper command (executable, drop-in replacement for `claude`), pass it via `--claude-cmd` (or set `CLAUDE_CMD`):

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --print-mapping-json --claude-cmd "claude-wrapper"
```

If you also use `--role-definition-md`, the wrapper must accept `--append-system-prompt` (drop-in compatibility).

Do not set deadlines unless the user explicitly requested a time limit. If they did, pass `--deadline-seconds`:

```bash
python3 scripts/invoke_persist.py create-session --session-name "review-src" --print-mapping-json --deadline-seconds 30
```

Do not interrupt a running create/resume Claude call just because intermediate output appears to drift from the prompt. Let the process finish unless deadline, stall, or hard error criteria are met.

## Persist the alias mapping (system temp)

By default, this skill writes a workspace-scoped JSON file under system temp:

- Path: `<system-tmp>/agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/claude-code-alias-mapping.json`
- Shape: top-level `workspace_dir` plus `aliases` mapping session_name->{`session_id`, `created_at`, optional `last_model`, optional `last_reasoning_effort`}
- Overwrite behavior: overwrite alias entries unconditionally

The default path depends on your current working directory. Run from the same workspace directory across create/resume, or pass `--mapping-file` explicitly.

If writing the mapping file fails (permissions, missing tmp, etc.), the script warns to stderr but still succeeds and prints the `session_id`. Keep the alias and `session_id` in the current chat context so you can resume within this conversation.

## Output contract (what to respond with)

When the creation stage runs, respond with:

- The created `session_id` and the session name it was stored under.
- The mapping file path used and whether it was written successfully.
- The updated alias mapping JSON block (from `--print-mapping-json`).
