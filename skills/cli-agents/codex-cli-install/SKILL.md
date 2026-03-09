---
name: codex-cli-install
description: A three-part Codex CLI setup skill with subskills for installation, skip-login configuration, and custom API-key launcher setup. Use when the user wants Codex CLI installed or configured on the current host, with instructions adapted to the actual operating system and runtime environment.
---

# Codex CLI Install

## Manual invocation

Invoke this skill explicitly by name (`$codex-cli-install`) because it modifies the local host environment.

## Core operating rules

- Detect the host OS first and choose commands accordingly.
- On Windows, prefer the PowerShell helpers in `scripts/*.ps1`; if the user needs a double-clickable entrypoint, use the matching `scripts/*.bat` wrapper.
- On Linux or macOS, prefer the POSIX shell helpers in `scripts/*.sh`.
- Do not mix command styles across OS families.
- If the OS is unclear from context, check it before acting.
- Treat the scripts in `scripts/` as optional accelerators and reference implementations, not the only path.
- Before running a helper script, inspect the actual environment: available shell, available package managers, PATH layout, permission constraints, and existing Codex configuration.
- If a helper script works in the current environment, use it.
- If a helper script fails, do not stop at the script failure. Read the relevant subskill guide under `references/`, infer the required actions, and execute an environment-appropriate plan manually.

## Subskills

This skill has exactly three subskills.

1. `installation`
   Install the Codex CLI in the current environment.
   Primary guide: `references/installation.md`
   Optional helpers: `scripts/install-comp.sh`, `scripts/install-comp.ps1`, `scripts/install-comp.bat`

2. `skip-login-config`
   Configure Codex to use a custom provider in `config.toml` with `requires_openai_auth = false`, so it no longer relies on the built-in login flow.
   Primary guide: `references/skip-login-config.md`
   Optional helpers: `scripts/config-skip-login.sh`, `scripts/config-skip-login.ps1`, `scripts/config-skip-login.bat`

3. `add-custom-api-key`
   Create a custom alias, function, or optional launcher that injects an OpenAI-compatible API key and base URL, and wires Codex to the matching provider configuration.
   Primary guide: `references/add-custom-api-key.md`
   Optional helpers: `scripts/config-custom-api-key.sh`, `scripts/config-custom-api-key.ps1`, `scripts/config-custom-api-key.bat`

## Subskill selection

- If the user wants Codex installed, use `installation`.
- If Codex is already installed but the user wants to bypass the built-in login flow, use `skip-login-config`.
- If the user wants a wrapper such as `codex-openai-proxy`, a custom base URL, or a custom API key, use `add-custom-api-key`.
- If the user wants a complete setup, run the subskills in this order: `installation`, then `skip-login-config`, then `add-custom-api-key`.

## How to execute each subskill

For each subskill:

1. Read the corresponding file in `references/` first.
2. Detect the OS and shell family.
3. Check which runtime tools are actually available.
4. Prefer the documented manual steps from the reference guide.
5. Use a helper script from `scripts/` only if it matches the environment and clearly reduces work.
6. If the helper script fails, continue by applying the guide manually instead of retrying blindly.
7. Verify the result with the verification section from the corresponding reference guide.

## Example prompts

- "Use the Codex CLI install skill to install Codex on this machine."
- "Use the skip-login subskill so Codex no longer shows its built-in login flow on this host."
- "Use the custom API key subskill to make a `codex-openai-proxy` alias that reads the key from an environment variable."
- "Set up Codex fully, and adapt the steps if the bundled scripts do not work in this environment."

## Resources

- Installation guide: `references/installation.md`
- Skip-login guide: `references/skip-login-config.md`
- Custom API-key guide: `references/add-custom-api-key.md`
- Helper scripts: `scripts/`