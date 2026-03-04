#!/usr/bin/env python3
"""Compose a Copilot config directory from base config and JSON overlays."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for config composition."""
    parser = argparse.ArgumentParser(
        description=(
            "Compose a Copilot config directory by overlaying preset/runtime JSON "
            "on top of the user's base ~/.copilot/config.json."
        )
    )
    parser.add_argument(
        "--preset",
        required=True,
        help="Path to preset JSON overlay (required).",
    )
    parser.add_argument(
        "--base-config-dir",
        default="~/.copilot",
        help="Base Copilot config directory (default: ~/.copilot).",
    )
    parser.add_argument(
        "--output-config-dir",
        help=(
            "Destination config directory. If omitted, creates a new temp dir and "
            "prints that path."
        ),
    )
    parser.add_argument(
        "--overlay-file",
        action="append",
        default=[],
        help="Additional JSON overlay file path (can be used multiple times).",
    )
    parser.add_argument(
        "--overlay-json",
        action="append",
        default=[],
        help="Additional JSON overlay string (can be used multiple times).",
    )
    parser.add_argument(
        "--no-link-base-state",
        action="store_true",
        help=(
            "Do not mirror non-config.json entries from the base config directory "
            "into the generated directory."
        ),
    )
    return parser.parse_args()


def _load_json_object(path: Path) -> JsonObject:
    """Load a JSON object from ``path``."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(value).__name__}")
    return value


def _deep_merge(base: JsonObject, overlay: JsonObject) -> JsonObject:
    """Recursively merge overlay object into base object."""
    merged = dict(base)
    for key, value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
            continue
        merged[key] = value
    return merged


def _link_or_copy(source: Path, destination: Path) -> None:
    """Create a symlink at ``destination``; fallback to copy if needed."""
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
        return
    except OSError:
        pass

    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def _mirror_base_state(base_config_dir: Path, output_config_dir: Path) -> None:
    """Mirror non-config.json entries from base config directory to output dir."""
    if not base_config_dir.exists():
        return

    for entry in base_config_dir.iterdir():
        if entry.name == "config.json":
            continue
        target = output_config_dir / entry.name
        if target.exists() or target.is_symlink():
            continue
        _link_or_copy(entry, target)


def _resolve_output_dir(output_config_dir: str | None) -> Path:
    """Resolve or create output config directory."""
    if output_config_dir:
        path = Path(output_config_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    tmp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    return Path(tempfile.mkdtemp(prefix="copilot-config-", dir=tmp_root))


def main() -> int:
    """Compose the config directory and print its path."""
    args = _parse_args()

    base_config_dir = Path(args.base_config_dir).expanduser().resolve()
    preset_path = Path(args.preset).expanduser().resolve()
    output_config_dir = _resolve_output_dir(args.output_config_dir)

    if output_config_dir == base_config_dir:
        raise ValueError("Refusing to write into base config directory; use a separate output path")

    if not preset_path.is_file():
        raise FileNotFoundError(f"Preset not found: {preset_path}")

    merged: JsonObject = {}
    base_config_path = base_config_dir / "config.json"
    if base_config_path.is_file():
        merged = _load_json_object(base_config_path)

    merged = _deep_merge(merged, _load_json_object(preset_path))

    for overlay_file in args.overlay_file:
        overlay_path = Path(overlay_file).expanduser().resolve()
        merged = _deep_merge(merged, _load_json_object(overlay_path))

    for overlay_json in args.overlay_json:
        parsed = json.loads(overlay_json)
        if not isinstance(parsed, dict):
            raise ValueError("Each --overlay-json must decode to a JSON object")
        merged = _deep_merge(merged, parsed)

    if not args.no_link_base_state:
        _mirror_base_state(base_config_dir=base_config_dir, output_config_dir=output_config_dir)

    config_path = output_config_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(output_config_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
