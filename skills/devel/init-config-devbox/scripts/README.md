# Vendored setup scripts

These scripts are synced from raw GitHub URLs (no git clone/checkout):
- repo: https://github.com/igamenovoer/lanren-ai
- raw base: https://raw.githubusercontent.com/igamenovoer/lanren-ai/main/components
- ref: main

Vendored components in this workspace:
- scripts/nodejs
- scripts/bun
- scripts/uv
- scripts/pixi
- scripts/claude-code-cli

Sync command:
- bash scripts/sync-from-lanren-ai.sh --ref main

Purpose:
- Keep host installation/bootstrap scripts local so future runs do not require cloning the full lanren-ai repository.
