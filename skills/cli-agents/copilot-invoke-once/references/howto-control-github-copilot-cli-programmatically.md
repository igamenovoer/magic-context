# How to Control GitHub Copilot CLI Programmatically (CLI / Headless)

Drive GitHub Copilot from another program by spawning the `copilot` CLI in non-interactive mode.

This guide assumes **headless invocations** plus **session continuation** (not a long-lived JSON-RPC chat server).

If you use a Copilot wrapper command, substitute it for `copilot` in all examples.

## One-shot prompt (script-friendly text output)

```bash
copilot -p "Summarize this project" --allow-all-tools --silent --stream off
```

Typical use: capture stdout as the final answer string.

## Streaming progress (heartbeat)

```bash
copilot -p "Review this repository for performance issues" --allow-all-tools --stream on
```

Typical use: consume stdout continuously so the caller can observe progress while the run is active.

## Maintaining conversational state (multi-turn across invocations)

### Continue the latest conversation

```bash
copilot -p "Review this codebase for performance issues" --allow-all-tools --stream off
copilot -p "Now focus on the database queries" --continue --allow-all-tools --stream off
copilot -p "Generate a summary of all issues found" --continue --allow-all-tools --stream off
```

### Resume by session ID (deterministic automation)

```bash
copilot -p "Continue that review" --resume "<session-id>" --allow-all-tools --stream off
```

## Permissions and auth

For non-interactive automation, set explicit permissions according to your risk tolerance:

```bash
# Broad permissions (fastest for automation)
copilot -p "Explain this file" --allow-all

# Example least-privilege pattern
copilot -p "Analyze src/main.py" \
  --allow-tool 'shell(git:*)' \
  --allow-tool write \
  --add-dir "$(pwd)"
```

If authentication is not configured yet:

```bash
copilot login
```

## Optional: MCP web search providers (Tavily / Exa)

Copilot CLI supports MCP servers, and web-search MCP providers can be configured (for example `tavily-mcp` or `exa`).
When available, prefer them for prompts that explicitly require web research, and avoid forcing web search for prompts that don't.

In this environment, Copilot CLI is configured with `tavily-mcp` via `~/.copilot/mcp-config.json`.

## Python subprocess example (streaming stdout)

```python
import subprocess


def run_copilot_stream(prompt: str) -> str:
    proc = subprocess.Popen(
        [
            "copilot",
            "-p",
            prompt,
            "--allow-all-tools",
            "--stream",
            "on",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    chunks: list[str] = []
    for line in proc.stdout:
        chunks.append(line)

    rc = proc.wait()
    if rc != 0:
        err = (proc.stderr.read() if proc.stderr else "").strip()
        raise RuntimeError(f"copilot failed rc={rc}: {err}")

    return "".join(chunks)
```

Notes:

1. Treat `stdout` as the primary machine channel and `stderr` as diagnostics.
2. Prefer `--stream on` for long jobs where heartbeat/progress visibility matters.
3. Use `--continue` for "latest session" and `--resume <session-id>` when the caller has a specific target session.
4. Output format is text-oriented; parse defensively.

## Sources

- GitHub Docs: Copilot CLI overview and how-tos:
  `https://docs.github.com/en/copilot/how-tos/copilot-cli`
- GitHub Docs: Using GitHub Copilot in the command line:
  `https://docs.github.com/en/copilot/how-tos/copilot-cli/using-github-copilot-in-the-command-line`
- GitHub Docs: Copilot CLI command reference:
  `https://docs.github.com/en/copilot/how-tos/copilot-cli/reference/cli-command-reference`
