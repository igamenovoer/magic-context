# How to Control Claude Code Programmatically (CLI / Headless)

Drive Claude Code from another program by spawning the `claude` CLI and parsing machine-readable output.

This guide assumes **headless invocations** plus **session continuation**, not a long-lived JSON-RPC chat server.

If you use a Claude wrapper (executable, `.sh`, or shell alias), substitute it for `claude` in the examples (example: `claude-wrapper -p ...`).

## One-shot prompt (structured JSON)

```bash
claude -p "Summarize this project" --output-format json
```

Typical use: parse `.result` and persist `.session_id` (schema may evolve; treat it as API output).

## Streaming progress/events (JSONL)

```bash
claude -p "Explain recursion" --output-format stream-json --verbose --include-partial-messages
```

Typical use: parse one JSON object per line for realtime progress.

## Maintaining conversational state (multi-turn across invocations)

### Continue the latest conversation

```bash
claude -p "Review this codebase for performance issues"
claude -p "Now focus on the database queries" --continue
claude -p "Generate a summary of all issues found" --continue
```

### Resume by session ID (recommended for deterministic automation)

```bash
session_id="$(claude -p "Start a review" --output-format json | jq -r '.session_id')"
claude -p "Continue that review" --resume "$session_id"
```

## Credentials (env vars)

If you keep credentials in an env var file (KEY=VALUE lines), load it before invoking `claude`:

```bash
set -a
source /path/to/vars.env
set +a

codeword="CLAUDE-BANANA-$(date +%s)"
session_id="$(
  claude -p "Remember this codeword for later: $codeword. Reply only: OK" --output-format json |
    jq -r '.session_id'
)"

set -a
source /path/to/vars.env
set +a

claude --resume "$session_id" -p "What codeword did I ask you to remember? Reply only the codeword." --output-format json |
  jq -r '.result'
```

## Python subprocess example (stream-json)

Spawn the CLI, read stdout line-by-line, and parse JSONL events.

```python
import json
import subprocess


def run_claude_stream(prompt: str) -> list[dict]:
    proc = subprocess.Popen(
        ["claude", "-p", prompt, "--output-format", "stream-json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered for JSONL
    )

    assert proc.stdout is not None
    events: list[dict] = []
    for line in proc.stdout:
        events.append(json.loads(line))

    rc = proc.wait()
    if rc != 0:
        err = (proc.stderr.read() if proc.stderr else "").strip()
        raise RuntimeError(f"claude failed rc={rc}: {err}")

    return events
```

Notes:

1. Treat `stdout` as the machine channel and `stderr` as diagnostics.
2. For robust automation, prefer `--output-format json` or `--output-format stream-json` over plain text parsing.
3. If you need multi-turn state, persist the `session_id` and use `--resume`.
4. If you use a wrapper command, replace `"claude"` with that executable (or pass it as a parameter).
5. Do not interrupt a running Claude process only because partial output seems off-prompt; let it reach normal completion, then assess the final result.
6. Early termination is reserved for explicit deadlines requested by the user, confirmed stalls, or hard error states (for example: lost connection, timeout, invalid API key, process failure).

## Optional: running Claude Code as an MCP server (tools, not chat)

Claude Code can run as an MCP server over stdio:

```bash
claude mcp serve
```

This is useful if you have an MCP client that wants to call Claude Code’s tools (filesystem, editing, shell, etc.).
It is not the same as a stable “chat session server” API for sending prompts and receiving replies.

## Sources

- Claude Code headless mode: `https://code.claude.com/docs/en/headless`
- Claude Code best practices: `https://code.claude.com/docs/en/best-practices`
- Claude Code as an MCP server: `https://docs.anthropic.com/en/docs/claude-code/mcp`
