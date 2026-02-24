---
name: init-config-devbox
description: Performs initial configuration of a development host with a fixed sequence: setup proxy first, then configure pixi/uv/npm/bun global tooling, and finally configure Claude Code. Use only when explicitly invoked by name for first-time host bootstrap or full re-bootstrap.
compatibility: Linux host setup; requires bash, curl or wget, jq, node/npm, pixi, uv, bun, and network access.
metadata:
   author: local-workspace
---

# Initial development host configuration skill

Use this skill to run the full host configuration process in a repeatable way.

## Activation
- This skill must be triggered manually by name: `init-config-devbox`.
- Do not auto-activate this skill for unrelated tasks.

## Network policy
- Treat all network access as fallback.
- Prefer local assets first (`scripts/`, existing binaries, local config/state).
- Use network only when required inputs/tools are missing or explicitly requested.
- Proxy usage is controlled by `[proxy].candidate_list` from selected TOML:
   - If `candidate_list` is present and non-empty, use proxy-aware flow.
   - If `candidate_list` is missing or empty, run all installation/configuration commands without proxy setup/sourcing.

## Inputs and references
- Proxy script: `scripts/setup-proxy.sh`
- Unified configuration file: `config.toml`
- Config schema: `config-schema.json`
- Vendored scripts: `scripts/`
   - Refresh vendored scripts from raw GitHub URLs (no git checkout):
      - `bash scripts/sync-from-lanren-ai.sh --ref main`

Configuration file selection:
- Allow user to specify an alternate TOML path (for example via prompt input or an explicit argument).
- If user does not provide a TOML path, fall back to the skill-accompanied file:
   - `./config.toml`
- Validate selected TOML against `./config-schema.json` before applying changes.

## Required execution order

### 1) Setup proxy (`setup-proxy.sh` and `.bashrc`)
1. If `[proxy].candidate_list` is present and non-empty, use accompanied local script first:
   - `install -m 0755 ./scripts/setup-proxy.sh "$HOME/setup-proxy.sh"`
2. If `[proxy].candidate_list` is present and non-empty, ensure `.bashrc` contains proxy candidates in order:
   - `export PROXY_CANDIDATE_LIST="<candidate_list joined by comma>"`
   - Do not hardcode host-specific proxy entries in this skill; use values from TOML or existing host `.bashrc`.
3. For proxy-required install operations, source first only when `[proxy].candidate_list` is present and non-empty:
   - `source ~/setup-proxy.sh`
4. If `[proxy].candidate_list` is missing or empty, skip proxy setup/sourcing entirely.

### 2) Configure/install pixi
Rules:
- If `[proxy].candidate_list` is present and non-empty, source proxy before pixi install/update commands.
- If `[proxy].candidate_list` is missing or empty, run pixi commands directly without proxy sourcing.
- Use global tool install: `pixi global install ...`
- Read channel priority from `[pixi].channel_priority` in TOML (do not hardcode channels in this skill).
- If `[paths].pixi_cache_dir` is set and non-empty, redirect pixi/rattler cache via `.bashrc`:
   - `export PIXI_CACHE_DIR="<pixi_cache_dir>"`
   - `export RATTLER_CACHE_DIR="<pixi_cache_dir>"`
- If `[paths].pixi_cache_dir` is missing or empty, do not set `PIXI_CACHE_DIR`/`RATTLER_CACHE_DIR`; use Pixi/Rattler defaults.

Global tool set:
- Read from `[pixi].global_tools` in TOML (do not hardcode tool names in this skill).

Install command:
- If `[proxy].candidate_list` is present and non-empty:
   - `source ~/setup-proxy.sh; pixi --no-progress global install <tools from [pixi].global_tools>`
- If `[proxy].candidate_list` is missing or empty:
   - `pixi --no-progress global install <tools from [pixi].global_tools>`

### 3) Configure/install uv
Rules:
- If `[proxy].candidate_list` is present and non-empty, source proxy before uv install/update commands.
- If `[proxy].candidate_list` is missing or empty, run uv commands directly without proxy sourcing.
- Use global tool install: `uv tool install ...`
- If `[paths].uv_cache_dir` is set and non-empty, redirect uv cache via `.bashrc`:
   - `export UV_CACHE_DIR="<uv_cache_dir>"`
- If `[paths].uv_cache_dir` is missing or empty, do not set `UV_CACHE_DIR`; use uv defaults.

Global tool set:
- Read from `[uv].global_tools` in TOML (do not hardcode tool names in this skill).

Install commands:
- If `[proxy].candidate_list` is present and non-empty:
   - `source ~/setup-proxy.sh; uv tool install <each tool from [uv].global_tools>`
- If `[proxy].candidate_list` is missing or empty:
   - `uv tool install <each tool from [uv].global_tools>`

### 4) Configure/install npm (skip bootstrap if already present)
Rules:
- If `npm --version` succeeds, skip npm bootstrap installation.
- If `[proxy].candidate_list` is present and non-empty, source proxy before npm config/install operations.
- If `[proxy].candidate_list` is missing or empty, run npm config/install operations directly without proxy sourcing.
- Configure npm mirror from TOML:
   - `npm config set registry <[urls].npm_registry_mirror>`
- Install global packages with latest policy:
   - Read package list from `[npm].global_packages` and apply `[npm].install_global` / `[npm].install_latest`.

Note:
- If no npm global package list is explicitly provided, only set registry/config and skip global installs.

### 5) Configure/install bun (skip bootstrap if already present)
Rules:
- If `bun --version` succeeds, skip bun bootstrap installation.
- If bootstrap is needed and `[proxy].candidate_list` is present and non-empty, proxy may be used for bootstrap only.
- If `[proxy].candidate_list` is missing or empty, run bun bootstrap/install/update directly without proxy sourcing.
- Install global packages with latest policy:
   - Read package list from `[bun].global_packages` and apply `[bun].install_global` / `[bun].install_latest`.
- Update globals:
   - `bun update -g --latest`

### 6) Configure Claude Code
Use local vendored scripts in `scripts/claude-code-cli/`.

Required outcomes:
1. Skip login/onboarding enabled.
2. Custom alias launcher created (read alias from `[claude].alias`; do not hardcode alias in this skill).
3. Tavily MCP configured.
4. Context7 MCP configured.

Credential policy:
- Read secrets and all configurable values from the selected TOML (user-provided TOML if specified, otherwise `./config.toml`).
- Do not write secrets into markdown docs.

Missing credential handling:
- For each required credential (`anthropic_api_key`, `tavily_api_key`, `context7_api_key`), if the TOML value is empty or missing, explicitly ask the user how to proceed.
- Offer exactly these choices per missing credential:
   1. Provide the credential value directly.
   2. Provide an environment variable name that already contains the credential.
   3. Refuse to provide it.
- If user provides a value or env reference, use it only for the current run unless the user explicitly asks to persist it.
- If user refuses to provide a credential, skip only the steps that require that credential and continue with all remaining applicable steps.
- Never block the full workflow due to one missing credential if unaffected steps can still run.

Credential-to-step mapping:
- `anthropic_api_key` is required for `scripts/claude-code-cli/config-custom-api-key.sh`.
- `tavily_api_key` is required for `scripts/claude-code-cli/config-tavily-mcp.sh`.
- `context7_api_key` is required only when applying Context7 API key via `claude mcp add-json`; `scripts/claude-code-cli/config-context7-mcp.sh` can still run.

Execution sequence (using vendored scripts):
- `sh scripts/claude-code-cli/config-skip-login.sh`
- `sh scripts/claude-code-cli/config-custom-api-key.sh --alias-name <alias> --api-key <claude_api_key>`
- `sh scripts/claude-code-cli/config-tavily-mcp.sh --api-key <tavily_api_key>`
- `sh scripts/claude-code-cli/config-context7-mcp.sh`
- Then ensure Context7 key is applied in MCP env via `claude mcp add-json` if needed.

## Verification checklist
- Proxy selection works: `source ~/setup-proxy.sh`
- Pixi tools present: `pixi global list`
- uv tools present: `uv tool list`
- npm available and configured: `npm --version` and `npm config get registry`
- bun globals present: `bun pm -g ls`
- Claude onboarding skipped: `~/.claude.json` contains `hasCompletedOnboarding=true`
- Claude MCP healthy: `claude mcp list` shows `tavily` and `context7`

## Documentation requirements
- Keep all credentials only in the selected TOML (fallback: `./config.toml`).
- Store non-secret execution notes in your current task output/logs; do not duplicate secrets in docs.
