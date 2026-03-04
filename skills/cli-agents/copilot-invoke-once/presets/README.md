# Copilot config presets

These files are JSON overlays for `~/.copilot/config.json`.

They are intended to be used with `scripts/compose_config.py`, which:

1. Loads the user's base config (`~/.copilot/config.json` by default),
2. Applies one preset from this directory,
3. Applies optional runtime overlays,
4. Writes the composed config to a temp config directory.

Merge semantics:

- JSON objects merge recursively.
- Non-object values (including arrays) replace the base value at the same key.

Available presets:

- `reasoning-low.json`
- `reasoning-medium.json`
- `reasoning-high.json`
