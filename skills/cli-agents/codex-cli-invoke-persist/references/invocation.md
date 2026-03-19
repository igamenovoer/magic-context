# Invocation stage (resume and call Codex CLI)

Goal: run a prompt in an existing Codex CLI conversation deterministically using `codex exec resume` with a known `thread_id`.

## Required input (don't invent)

- `prompt`: what to send to Codex
- Session selector (pick one):
  - `thread_id` or `session_id` (preferred if provided)
  - `session name`

If the user asks to resume but does not specify which session, ask for the session name or `thread_id`.

## Model defaults (manifest-backed)

- If the user does not specify `--model`, this skill defaults to the stored `last_model` for that session name when present in the mapping file.
- If the user explicitly specifies `--model`, it overrides for this call and is persisted as the new `last_model` default for next time.

## Resolve session name to `thread_id`

Resolve without calling Codex:

```bash
python3 scripts/invoke_persist.py resolve --session-name "review-src" --print-thread-id
```

If the alias is missing, create it first via the creation stage guide.

## Resume (final JSON summary)

`resume-json` returns one helper-generated JSON object on stdout. Internally it still uses `codex exec resume --json` and `-o` to capture the final assistant message.

```bash
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off"
```

Override the stored default model explicitly:

```bash
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off" --model "MODEL"
```

Print just the assistant text:

```bash
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off" --print-result
```

## Resume (streaming JSONL output)

Use this when heartbeat visibility matters:

```bash
python3 scripts/invoke_persist.py resume-stream --session-name "review-src" --prompt "Continue from where we left off"
```

Reassert a role definition for this specific resumed call:

```bash
python3 scripts/invoke_persist.py resume-stream --session-name "review-src" --prompt "Continue from where we left off" --role-definition-md "/abs/path/to/role.md"
```

Credentials: ensure `codex` is on PATH and authenticated. To load env vars from a file, pass `--env-file /path/to/vars.env` (KEY=VALUE lines).

If the user provides a codex-compatible executable wrapper command or script, pass it via `--codex-cmd` (or set `CODEX_CMD`):

```bash
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off" --codex-cmd "codex-wrapper"
```

If the user provides a POSIX alias/function wrapper, use shell-launch mode:

```bash
python3 scripts/invoke_persist.py resume-json \
  --session-name "review-src" \
  --prompt "Continue from where we left off" \
  --codex-cmd 'codex-work' \
  --codex-shell posix \
  --codex-shell-init 'shopt -s expand_aliases; source ~/.bashrc' \
  --codex-shell-cmd 'bash -lc'
```

If the user provides a PowerShell function wrapper, use PowerShell shell-launch mode:

```bash
python3 scripts/invoke_persist.py resume-json \
  --session-name "review-src" \
  --prompt "Continue from where we left off" \
  --codex-cmd 'codex-work' \
  --codex-shell powershell \
  --codex-shell-init '. $PROFILE' \
  --codex-shell-cmd 'pwsh -NoLogo -Command'
```

Do not assume wrapper names start with `codex-`. Use the exact wrapper name or shell snippet the user gave you.

Additional `codex exec resume` flags supported by the helper:

```bash
python3 scripts/invoke_persist.py resume-stream \
  --session-name "review-src" \
  --prompt "Continue from where we left off" \
  --enable my_feature \
  --image /abs/path/to/image.png \
  --full-auto
```

## Heartbeat and deadlines

- Prefer `resume-stream` for long-running work. Each JSONL line on stdout can be treated as a heartbeat.
- `resume-json` still monitors Codex JSONL internally, but it only emits the final helper-generated JSON summary on stdout after the run completes.
- Do not terminate the Codex process due to lack of output unless the user explicitly requested a deadline, the process is clearly stalled, or the process is in a hard error state. If a deadline is requested, pass `--deadline-seconds`.
- Do not interrupt simply because intermediate output appears to drift from the prompt; let Codex run to normal completion and evaluate the final result afterward.

Examples:

```bash
python3 scripts/invoke_persist.py resume-stream --session-name "review-src" --prompt "Continue from where we left off" --deadline-seconds 300
python3 scripts/invoke_persist.py resume-json --session-name "review-src" --prompt "Continue from where we left off" --deadline-seconds 300
```

## Important Codex CLI limitation

`codex exec resume` supports fewer flags than `codex exec`. In particular, this helper does not attempt to persist or replay exec-only options like `--cd`, `--add-dir`, `--sandbox`, `--search`, `--profile`, or `--output-schema`.

## Output contract (what to respond with)

When the invocation stage runs, respond with:

- The `thread_id` used and the session name if applicable.
- The Codex answer or a short parsed summary if streaming.
