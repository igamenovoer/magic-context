# Configure GitHub Copilot CLI (Model + Reasoning Effort)

This note documents how to configure the GitHub Copilot CLI (`copilot`) to:

- Select an AI model (per command or by default).
- Set a default reasoning effort level (as supported by the CLI and model).

This is written for Copilot CLI versions similar to `0.0.420` (Linux), but the concepts should transfer.

## Model Selection

### Per invocation (recommended for deterministic automation)

Pass `--model <model-id>`:

```bash
copilot --model gpt-5.3-codex -p "Summarize this repo" --yolo --no-ask-user --silent --stream on
```

To see the exact model IDs your installed CLI supports:

```bash
copilot --help
```

Look for the `--model <model>` option and its `choices:` list.

### Environment default (overridden by `--model`)

Set `COPILOT_MODEL`:

```bash
export COPILOT_MODEL="claude-opus-4.6"
copilot -p "Explain this file" --yolo --no-ask-user --silent --stream on
```

### Interactive default

Start `copilot` interactively and use:

```text
/model
```

This can persist your selection to the config file (see below).

## Default Reasoning Effort

### Important limitation: no per-invocation flag (in current CLI)

At least in Copilot CLI `0.0.420`, there is no `--reasoning-effort` flag.
Reasoning effort is configured as a default in the Copilot config JSON and is only applied for models that support it.

### Config file key

Copilot stores config under `~/.copilot/config.json` by default (unless `--config-dir` is used).

The keys commonly used are:

- `model`: default model ID (string)
- `reasoning_effort`: default reasoning effort level (string)

Typical values for `reasoning_effort` are:

- `low`
- `medium`
- `high`
- `xhigh`

Not every model will honor this setting; Copilot applies it only when supported by the selected model.

### Safe editing workflow

Warning: `~/.copilot/config.json` may contain authentication tokens and user info.
Do not paste it into chats, tickets, or logs.

To update only `model` and `reasoning_effort` while preserving everything else:

```bash
set -euo pipefail

cfg="$HOME/.copilot/config.json"
bak="$HOME/.copilot/config.json.bak.$(date +%Y%m%d-%H%M%S)"

cp -a "$cfg" "$bak"

jq '.model="claude-opus-4.6" | .reasoning_effort="high"' "$cfg" > "$cfg.tmp"
chmod --reference="$cfg" "$cfg.tmp" || true
mv "$cfg.tmp" "$cfg"

echo "Updated: $cfg (backup: $bak)"
```

Notes:

- The `cp -a` backup is important because the file may include auth data.
- `chmod --reference` helps preserve strict permissions (some setups keep this file `0600`).

### Verifying your current defaults

Avoid printing the entire config file. Instead, extract just the relevant keys:

```bash
jq -r '{model: .model, reasoning_effort: .reasoning_effort} | @json' ~/.copilot/config.json
```

## Related flags

- `--model <model-id>`: select model for this run.
- `--yolo` / `--allow-all`: maximum permissions (`--allow-all-tools --allow-all-paths --allow-all-urls`).
- `--no-ask-user`: prevents the agent from pausing to ask interactive questions.
- `--silent`: reduces extra CLI output (useful for scripting).
- `--stream on`: stream progress/heartbeat while Copilot is running.
