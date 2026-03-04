---
name: copilot-invoke-once
description: Programmatically invoke GitHub Copilot CLI via the `copilot` command (headless `-p`, `--silent`, `--stream on|off`, `--resume`/`--continue`) to automate tasks like producing code review notes for specific files/dirs, investigating an issue, or drafting an implementation plan. Use when you need scriptable, non-interactive Copilot runs with optional streaming progress and session continuation.
---

# GitHub Copilot Invoke Once

## Quick start (shell)

One-shot (script-friendly final response with a high-reasoning preset):

```bash
skill_dir=".codex/skills/copilot-invoke-once"
tmp_cfg="$(python3 "$skill_dir/scripts/compose_config.py" --preset "$skill_dir/presets/reasoning-high.json")"
trap 'rm -rf "$tmp_cfg"' EXIT
copilot --config-dir "$tmp_cfg" -p "Summarize this repo" --yolo --no-ask-user --silent --stream on
```

## Dynamic config overlays (must follow)

- Copilot CLI does not expose every model-control setting via CLI flags (for example `reasoning_effort`), so always generate a temporary config directory per invocation and run Copilot with `--config-dir <temp-dir>`.
- Use `scripts/compose_config.py` (this skill) to compose:
  1. base config: `~/.copilot/config.json` (or `--base-config-dir`)
  2. preset overlay: `presets/*.json`
  3. optional runtime overlays: `--overlay-file` and/or `--overlay-json`
- Merge rule: recursive object merge, with overlay values replacing base values on key conflicts.
- The helper links non-`config.json` entries from the base config dir into the temp dir (for example `mcp-config.json`, session state) so behavior/auth remains consistent.
- The helper prints the generated config directory path on stdout; clean it up after the command (`rm -rf "$tmp_cfg"`).

## Presets

Preset overlays live in `presets/`:

- `presets/reasoning-high.json`
- `presets/reasoning-medium.json`
- `presets/reasoning-low.json`

Default behavior: use `presets/reasoning-high.json` unless the user explicitly requests a different reasoning level or a custom overlay.

## Copilot wrappers

If the user provides a Copilot wrapper command (drop-in replacement for `copilot` that sets preset env vars), use it by substituting it for `copilot` in all examples and invocations.

- The wrapper may be a command on `PATH` (executable or `.sh` script).
- The wrapper may be an explicit filesystem path (for example `/abs/path/to/copilot-wrapper`).
- The wrapper may be a shell alias defined in the user's shell profile (it must be directly runnable in a shell prompt).
- The wrapper must accept the `copilot` flags used here (`-p`, `--silent`, `--stream`, `--continue`, `--resume`, `--model`, and permission flags).
- If the wrapper is a shell alias (not an executable), invoke it via a shell that loads aliases (for example `bash -lc '<alias> ...'`) so it resolves correctly.

## Default permissions (maximum power)

- To minimize permission prompts and give Copilot the most power, launch with `--yolo` (equivalent to `--allow-all-tools --allow-all-paths --allow-all-urls`) and `--no-ask-user`.
- `--allow-all-tools` (and env var `COPILOT_ALLOW_ALL=true`) only covers tool approvals. It does not automatically allow all paths or URLs, so `--yolo`/`--allow-all` is the simplest true "most power" mode.
- This skill defaults to maximum power flags. If the user requests tighter safety constraints, switch to least-privilege permissions (`--allow-tool`, `--allow-url`, `--add-dir`) and remove `--yolo`.

Streaming progress (heartbeat on stdout):

```bash
copilot --config-dir "$tmp_cfg" -p "Explain recursion with examples" --yolo --no-ask-user --silent --stream on
```

Continue a multi-turn session:

```bash
copilot --config-dir "$tmp_cfg" -p "Start a review of src/" --yolo --no-ask-user --silent --stream on
copilot --config-dir "$tmp_cfg" -p "Continue the same review and focus on tests/" --continue --yolo --no-ask-user --silent --stream on
```

Resume by explicit session ID:

```bash
copilot --config-dir "$tmp_cfg" -p "Continue that review" --resume "<session-id>" --yolo --no-ask-user --silent --stream on
```

## Heartbeats and deadlines

- Always run with `--stream on` so stdout emits a steady progress heartbeat while Copilot is still working.
- In automation, treat streaming output as a heartbeat signal: only surface minimal progress (for example, log one short "still running" line periodically) instead of printing the full streamed content.
- Avoid relying on timeouts to determine liveness; prefer streaming heartbeats. Do not terminate the Copilot process while it is still working unless the user explicitly requested a deadline (for example: "up to 5 minutes").

## Model selection

- Per-invocation: pass `--model <model>` (preferred for deterministic automation).
- Environment default: set `COPILOT_MODEL=<model>` (overridden by `--model`).
- Interactive: start `copilot` and use `/model` to select a model (may persist in `~/.copilot/config.json`).
- Config overlay: set model- or reasoning-related keys in a preset/overlay JSON when no matching CLI flag exists.
- If the user specifies a model by a short name or alias (for example "opus", "sonnet", "gpt-5"), resolve it to an exact `copilot --model <model>` choice.
- If the exact model name is not obvious from chat context, check the locally installed Copilot CLI for its supported model IDs (for example by running `copilot --help` and reading the `--model` choices, or by using interactive `/model`).
- Do not guess a model ID when multiple choices could match; ask the user to pick the exact model string or choose the closest unambiguous match.

## Workflow decision

1. Need one final response for automation: use `--silent --stream on` and capture stdout, but only display minimal heartbeat info while it runs.
2. Need realtime progress: use `--stream on` and consume stdout continuously.
3. Need multi-turn state: use `--continue` (latest session) or `--resume <session-id>` (deterministic target), still with `--stream on` for heartbeats.

## Session handling (must follow)

- If the user provides a `session_id`, always use `--resume <session_id>` for the next Copilot call.
- If the user says "continue latest chat", "continue last chat", or similar, use `--continue`.
- If the user does not provide a `session_id` and does not request continuation, determine from context whether to continue or start fresh. If unclear, start a new session.

## Permissions and safety (don't skip)

- If the user asks for a safer mode, prefer least privilege (`--allow-tool`, `--allow-url`, `--add-dir`) and remove `--yolo`/`--allow-all`.
- Treat `stdout` as the machine channel and `stderr` as diagnostics.
- Copilot CLI output can change across versions; parse defensively and avoid brittle string matching.

## Web search / online research

- Only explicitly ask Copilot to do web search when the user prompt implies web research (for example: "look it up", "latest", "news", "what does X mean", "compare providers"). Otherwise, let Copilot decide whether web research is needed.
- GitHub Copilot CLI can do web-backed research in **interactive mode** via the `/research` command (it gathers information from GitHub and the web and produces a report with citations).
- Before asking Copilot to do web search, check whether MCP web-search providers are configured (for example `tavily-mcp` and/or `exa`) in `~/.copilot/mcp-config.json`.
- If they are available, prefer those MCP tools for web search when needed and ask Copilot to use them. In this environment, `tavily-mcp` is present in `~/.copilot/mcp-config.json`.
- If you must stay in non-interactive `-p` mode, do not claim citations unless Copilot actually provided them.

## Example prompts (to trigger this skill)

- Users may refer to the same tool as "GitHub Copilot", "github copilot", or "Copilot" (even when they mean the terminal CLI). Treat those as triggers for this skill.
- "Use GitHub Copilot to review `src/` and `tests/` and summarize issues."
- "Use github copilot to investigate this error log and suggest fixes."
- "Use Copilot CLI to draft an implementation plan for this feature."

## Credentials (optional)

Ensure the command can authenticate in your environment (for example: `copilot login` or a configured token).

## Resources

- Full guide: `references/howto-control-github-copilot-cli-programmatically.md`
- Helper: `scripts/compose_config.py`
- Presets: `presets/`
