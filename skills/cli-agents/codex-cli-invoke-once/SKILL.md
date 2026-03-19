---
name: codex-cli-invoke-once
description: Programmatically invoke Codex via the `codex` CLI for headless, non-interactive runs using `codex exec`, machine-readable JSONL via `--json`, final-message capture via `-o/--output-last-message`, and multi-turn continuation via `codex exec resume`. Use when you need scriptable Codex automation for repo reviews, issue investigation, implementation planning, or structured one-shot outputs.
---

# Codex CLI Invoke Once

## Wrapper commands

Users may provide Codex wrappers with any name. Many teams use names like `codex-dev`, `codex-prod`, or `codex-openai`, but do not assume a `codex-*` prefix.

- If the wrapper is an executable script or command on PATH, such as `.sh`, `.bat`, or a normal executable, substitute it directly for `codex`.
- If the wrapper is a shell alias, shell function, or PowerShell function, invoke it through the corresponding shell instead of treating it like a plain executable.

## Quick start (shell)

One-shot run with the last assistant message written to a file:

```bash
codex exec -o /tmp/codex-last.txt "Summarize this repo"
cat /tmp/codex-last.txt
```

Machine-readable JSONL events on stdout:

```bash
codex exec --json -o /tmp/codex-last.txt "Explain recursion"
```

Persist a session and resume it later:

```bash
log="$(mktemp)"
codex exec --json -o /tmp/codex-last.txt "Start a review of this repo" | tee "$log"
thread_id="$(jq -r 'select(.type=="thread.started").thread_id' "$log" | head -n1)"
codex exec resume "$thread_id" --json -o /tmp/codex-last-2.txt "Continue that review with a focus on tests"
```

Structured final output with a JSON Schema:

```bash
codex exec --output-schema /path/to/schema.json -o /tmp/result.json "Return the requested fields only"
```

## Heartbeats and deadlines

- For long-running work, prefer `--json` so stdout emits a stream of JSONL events. Treat each line as a heartbeat and update a last-seen timestamp while the process is active.
- In automation, surface minimal heartbeat status instead of dumping every event unless the caller explicitly wants raw event logs.
- Do not add timeouts or terminate a healthy Codex run while heartbeats are still arriving unless one of these applies: (1) the user explicitly requested a deadline, (2) the process is clearly stalled and has stopped emitting heartbeats beyond the caller's chosen inactivity threshold, or (3) the process is in a hard error state.
- Do not interrupt simply because an intermediate message already looks good enough. Let the run reach normal completion and then use the final assistant message from `-o` or the terminal JSONL event stream.

## Workflow decision

1. Need one final answer: use `codex exec` and `-o` to capture the last assistant message.
2. Need streaming machine events: add `--json` and parse JSONL from stdout.
3. Need multi-turn state across invocations: capture the `thread_id` from `thread.started` and use `codex exec resume`.
4. Need scripted repo reviews: prefer `codex exec review` over top-level `codex review` because `exec review` supports `--json` and `-o`.
5. Need no durable session files: add `--ephemeral` and do not expect resume to work afterward.

## Session handling (must follow)

- If the user provides a Codex `thread_id` or session id, use `codex exec resume <id>`.
- If the user says “continue latest chat”, “continue last chat”, or similar, use `codex exec resume --last`.
- If the prior run used `--ephemeral`, start a new session instead of attempting resume.
- Persist the first `thread.started` event if you may need a follow-up turn later.

## Codex-specific CLI notes

- `codex exec` does not use Claude-style `-p` or `--output-format`; pass the prompt as a positional argument and use `--json` for JSONL events.
- `codex exec` does not accept `-a` or `--ask-for-approval`. Use the flags listed in `codex exec --help` instead.
- `codex exec` supports `--sandbox`, `--full-auto`, `--dangerously-bypass-approvals-and-sandbox`, `--ephemeral`, `--output-schema`, `--json`, and `-o`.
- `codex exec resume` supports `--json`, `-o`, `--last`, and `--ephemeral`, but its option surface is smaller than plain `exec`.
- Users may provide wrapper commands with arbitrary names. Prefer the exact wrapper name they gave you instead of guessing from `codex`.
- Treat stdout as the machine channel and stderr as diagnostics.
- Parse JSONL defensively and prefer matching on event `type` instead of hard-coding full schemas.

## Example prompts (to trigger this skill)

- “Use Codex CLI to generate a machine-readable repo summary.”
- “Use Codex CLI to investigate this error and save the last reply to a file.”
- “Use Codex CLI to start a review, then continue it in a second invocation.”

## Credentials

Ensure the `codex` CLI is installed and authenticated in the current environment before automating it.

## Resources

- Full guide: `references/howto-control-codex-cli-programmatically.md`
