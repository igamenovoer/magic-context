# How to Control GitHub Copilot CLI Programmatically (CLI / Headless)

Drive GitHub Copilot from another program by spawning the `copilot` CLI in non-interactive mode.

This guide assumes **headless invocations** plus **session continuation** (not a long-lived JSON-RPC chat server).

If you use a Copilot wrapper command, substitute it for `copilot` in all examples.

## Config composition for non-CLI settings (required)

Some Copilot settings (notably `reasoning_effort`) are not controllable via CLI flags.
For deterministic control, compose a temporary config directory per invocation and pass it through `--config-dir`.

Use this skill's helper:

```bash
skill_dir=".codex/skills/copilot-invoke-once"
tmp_cfg="$(python3 "$skill_dir/scripts/compose_config.py" --preset "$skill_dir/presets/reasoning-high.json")"
trap 'rm -rf "$tmp_cfg"' EXIT
```

How composition works:

1. Start from `~/.copilot/config.json` (or `--base-config-dir`).
2. Overlay one preset from `presets/`.
3. Optionally overlay additional JSON via `--overlay-file` / `--overlay-json`.
4. Write the merged result to temp `config.json`.
5. Mirror non-`config.json` base entries into the temp config dir (for example `mcp-config.json`, session state) so auth/session behavior remains consistent.

Merge semantics:

- JSON objects merge recursively.
- Non-object values (including arrays) replace the base value.

## One-shot prompt (script-friendly text output)

```bash
copilot --config-dir "$tmp_cfg" -p "Summarize this project" --yolo --no-ask-user --silent --stream on
```

Typical use: capture stdout as the final answer string.

## Streaming progress (heartbeat)

```bash
copilot --config-dir "$tmp_cfg" -p "Review this repository for performance issues" --yolo --no-ask-user --silent --stream on
```

Typical use: consume stdout continuously so the caller can observe progress while the run is active.

## Maintaining conversational state (multi-turn across invocations)

### Continue the latest conversation

```bash
copilot --config-dir "$tmp_cfg" -p "Review this codebase for performance issues" --yolo --no-ask-user --silent --stream on
copilot --config-dir "$tmp_cfg" -p "Now focus on the database queries" --continue --yolo --no-ask-user --silent --stream on
copilot --config-dir "$tmp_cfg" -p "Generate a summary of all issues found" --continue --yolo --no-ask-user --silent --stream on
```

### Resume by session ID (deterministic automation)

```bash
copilot --config-dir "$tmp_cfg" -p "Continue that review" --resume "<session-id>" --yolo --no-ask-user --silent --stream on
```

## Permissions and auth

For non-interactive automation, set explicit permissions according to your risk tolerance:

```bash
# Broad permissions (fastest for automation)
copilot --config-dir "$tmp_cfg" -p "Explain this file" --yolo --no-ask-user

# Example least-privilege pattern
copilot --config-dir "$tmp_cfg" -p "Analyze src/main.py" \
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


def run_copilot_stream(prompt: str, config_dir: str) -> str:
    proc = subprocess.Popen(
        [
            "copilot",
            "--config-dir",
            config_dir,
            "-p",
            prompt,
            "--yolo",
            "--no-ask-user",
            "--silent",
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
5. Remove the temp config dir after invocation when you no longer need it.

## Sources

- GitHub Docs: Copilot CLI overview and how-tos:
  `https://docs.github.com/en/copilot/how-tos/copilot-cli`
- GitHub Docs: Using GitHub Copilot in the command line:
  `https://docs.github.com/en/copilot/how-tos/copilot-cli/using-github-copilot-in-the-command-line`
- GitHub Docs: Copilot CLI command reference:
  `https://docs.github.com/en/copilot/how-tos/copilot-cli/reference/cli-command-reference`
