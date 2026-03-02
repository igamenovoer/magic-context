# Invocation stage (resume and call Claude Code)

Goal: run a prompt in an existing Claude Code conversation deterministically using `--resume` with a known `session_id`.

## Required input (don't invent)

- `prompt`: what to send to Claude
- Session selector (pick one):
  - `session_id` (preferred if provided)
  - `session name` (the user may call it a "session alias")

If the user asks to resume but does not specify which session, ask for the session name/alias (or the `session_id`).

## Model and reasoning-effort defaults (manifest-backed)

- If the user does not specify `--model` / `--reasoning-effort`, this skill defaults to the stored `last_model` / `last_reasoning_effort` for that session name (if present in the mapping file).
- If the user explicitly specifies `--model` and/or `--reasoning-effort`, those values override for this call and are persisted as the new `last_*` defaults for next time.

## Resolve session name to `session_id`

Resolve without calling Claude:

```bash
python3 scripts/invoke_persist.py resolve --session-name "review-src" --print-session-id
```

If the alias is missing, create it first via the creation stage guide.

## Resume (JSON output)

Machine-readable JSON (simple, but no heartbeat until completion):

```bash
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off"
```

Override the stored defaults explicitly:

```bash
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off" --model "MODEL" --reasoning-effort "EFFORT"
```

Print just the assistant text:

```bash
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off" --print-result
```

## Resume (streaming JSONL output)

```bash
python3 scripts/invoke_persist.py resume-stream --session-name "review-src" --prompt "Continue from where we left off" --verbose --include-partials
```

Credentials: ensure `claude` is on PATH and authenticated. To load env vars from a file, pass `--env-file /path/to/vars.env` (KEY=VALUE lines).

If the user provides a claude-compatible wrapper command (executable, drop-in replacement for `claude`), pass it via `--claude-cmd` (or set `CLAUDE_CMD`):

```bash
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off" --claude-cmd "claude-wrapper"
```

## Heartbeat and deadlines

- Prefer `resume-stream` for long-running work. Each JSONL line on stdout can be treated as a heartbeat.
- Do not terminate the Claude process due to lack of output unless the user explicitly requested a deadline. If they did, pass `--deadline-seconds` (example: "up to 5 minutes" -> `--deadline-seconds 300`).

Examples:

```bash
python3 scripts/invoke_persist.py resume-stream --session-name "review-src" --prompt "Continue from where we left off" --deadline-seconds 300
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off" --deadline-seconds 300
```

## Output contract (what to respond with)

When the invocation stage runs, respond with:

- The `session_id` used (and the session name if applicable).
- The Claude answer (or a short parsed summary if streaming).
