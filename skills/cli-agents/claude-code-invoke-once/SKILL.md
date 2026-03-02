---
name: claude-code-invoke-once
description: Programmatically invoke Claude Code via the `claude` CLI (headless `-p`, `--output-format json|stream-json`, `--resume`/`--continue`) to automate tasks like producing code review reports for specific files/dirs, investigating an issue, or drafting an implementation plan. Use when you need scriptable, machine-readable output and/or session-persistent multi-turn workflows with Claude Code.
---

# Claude Code Invoke Once

## Quick start (shell)

One-shot (machine-readable JSON):

```bash
claude -p "Summarize this repo" --output-format json
```

If you use a Claude wrapper (executable, `.sh`, or shell alias), substitute it for `claude` (example: `claude-wrapper -p ...`).

Streaming progress/events (JSONL):

```bash
claude -p "Explain recursion" --output-format stream-json --verbose --include-partial-messages
```

Persist multi-turn state and resume later (deterministic automation):

```bash
session_id="$(claude -p "Start a review" --output-format json | jq -r '.session_id')"
claude -p "Continue that review" --resume "$session_id" --output-format json
```

## Workflow decision

1. Need a single final answer: use `--output-format json` and parse `.result`.
2. Need realtime progress: use `--output-format stream-json` and parse one JSON object per stdout line.
3. Need multi-turn state: persist `.session_id` and call `--resume <session_id>` for subsequent turns (prefer over `--continue` in automation).

## Session handling (must follow)

- If the user provides a `session_id`, always use `--resume <session_id>` for the next Claude Code call.
- If the user says “continue latest chat”, “continue last chat”, or similar, use the most recently captured `session_id` from this conversation.
- If the user does not provide a `session_id` and does not explicitly request a continuation, determine from context whether they want to persist the previous Claude session. If unclear, start a new session.

## Example prompts (to trigger this skill)

- “Use Claude Code to make a review report of `src/` and `tests/`.”
- “Use Claude Code to look into the issue of: <paste error/log>.”
- “Use Claude Code to make an implementation plan for: <feature/PRD>.”

## Integration rules (don’t skip)

- Treat `stdout` as the machine channel; treat `stderr` as diagnostics.
- Assume output schemas can evolve; parse defensively and avoid hard-coding event keys.
- Store the returned `session_id` if you will send follow-up prompts in a later process/run.

## Credentials (optional)

Ensure your chosen Claude command can authenticate in your environment (for example by exporting env vars or sourcing an env file before running the command).

## Resources

- Full guide: `references/howto-control-claude-code-programmatically.md`
