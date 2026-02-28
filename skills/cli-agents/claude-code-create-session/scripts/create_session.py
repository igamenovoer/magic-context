#!/usr/bin/env python3
"""
Create a Claude Code session and print the resulting session_id.

This is a thin wrapper around the `claude` CLI intended for automation.
By default it also updates a YAML mapping file in the system temp directory:
`agent-sessions/claude-code-alias-mapping.yaml` (alias entries are overwritten
unconditionally). If the mapping file cannot be written, the script prints a
warning to stderr and still succeeds.

Examples
--------
Create a session and print an alias->session_id JSON mapping:
  python3 scripts/create_session.py --alias review-foo --print-mapping-json

Load credentials from an env file before calling `claude`:
  python3 scripts/create_session.py --alias review-foo --env-file /path/to/vars.env
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any


@dataclass(frozen=True)
class SessionInfo:
    alias: str
    session_id: str
    created_at: str
    workspace_dir: str
    raw: dict[str, Any]


def _read_env_vars_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            env[key] = value
    return env


def _default_mapping_file() -> Path:
    """Return the default alias→session mapping file path."""

    return Path(tempfile.gettempdir()) / "agent-sessions" / "claude-code-alias-mapping.yaml"


def _split_key_value(line: str) -> tuple[str, str] | None:
    """Split a YAML-ish `key: value` line on the first ':' outside quotes."""

    in_single = False
    in_double = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_double:
            escaped = True
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == ":" and not in_single and not in_double:
            return line[:i], line[i + 1 :]
    return None


def _parse_scalar(value: str) -> str:
    """Parse a limited YAML scalar (plain or double-quoted string)."""

    value = value.strip()
    if not value:
        return ""
    if value.startswith('"') and value.endswith('"'):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, str) else str(parsed)
        except json.JSONDecodeError:
            return value.strip('"')
    if value.startswith("'") and value.endswith("'"):
        inner = value[1:-1]
        return inner.replace("''", "'")
    return value


def _load_mapping(path: Path) -> dict[str, dict[str, str]]:
    """Load alias→metadata mapping from the YAML-ish mapping file.

    The parser is intentionally minimal and only supports the structure this
    script writes:

    <alias>:
      session_id: "..."
      workspace_dir: "..."
      created_at: "..."
    """

    if not path.exists():
        return {}

    mapping: dict[str, dict[str, str]] = {}
    current_alias: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))

        if indent == 0:
            parts = _split_key_value(line)
            if not parts:
                current_alias = None
                continue
            key_part, rest = parts
            if rest.strip():
                # Unexpected inline value; treat as new alias but ignore value.
                pass
            alias = _parse_scalar(key_part)
            current_alias = alias
            mapping.setdefault(alias, {})
            continue

        if indent == 2 and current_alias is not None:
            parts = _split_key_value(line.lstrip(" "))
            if not parts:
                continue
            key_part, rest = parts
            k = key_part.strip()
            v = _parse_scalar(rest)
            if k:
                mapping[current_alias][k] = v

    return mapping


def _quote_key(key: str) -> str:
    if key and all(ch.isalnum() or ch in "._-" for ch in key):
        return key
    return json.dumps(key, ensure_ascii=False)


def _dump_mapping(mapping: dict[str, dict[str, str]]) -> str:
    lines: list[str] = []
    for alias in sorted(mapping):
        lines.append(f"{_quote_key(alias)}:")
        entry = mapping[alias]
        ordered_keys = ["session_id", "workspace_dir", "created_at"]
        for k in ordered_keys + sorted(set(entry) - set(ordered_keys)):
            if k not in entry:
                continue
            lines.append(f"  {k}: {json.dumps(str(entry[k]), ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_mapping_file(
    mapping_file: Path,
    *,
    alias: str,
    session_id: str,
    workspace_dir: str,
    created_at: str,
) -> bool:
    """Write (or update) the mapping file; overwrite alias unconditionally."""

    try:
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        mapping = _load_mapping(mapping_file)
        mapping[alias] = {
            "session_id": session_id,
            "workspace_dir": workspace_dir,
            "created_at": created_at,
        }
        payload = _dump_mapping(mapping)
        tmp_path = mapping_file.with_name(mapping_file.name + f".tmp.{os.getpid()}")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, mapping_file)
        return True
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[WARN] Failed to write mapping file {mapping_file}: {e}\n")
        return False


def create_session(
    *,
    alias: str,
    init_prompt: str,
    extra_env: dict[str, str] | None = None,
) -> SessionInfo:
    """Create a new Claude Code session and return session metadata.

    Parameters
    ----------
    alias
        User-facing name for the session (not sent to Claude unless included in
        `init_prompt`).
    init_prompt
        First prompt used to create the session.
    extra_env
        Extra environment variables to set for the `claude` subprocess.
    """

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = ["claude", "-p", init_prompt, "--output-format", "json"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)
    except FileNotFoundError as e:
        raise RuntimeError("`claude` CLI not found on PATH. Install Claude Code and ensure `claude` is available.") from e

    if proc.returncode != 0:
        raise RuntimeError(f"claude failed rc={proc.returncode}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    session_id = data.get("session_id")
    if not session_id or not isinstance(session_id, str):
        raise RuntimeError("Missing `session_id` in `claude --output-format json` output.")

    created_at = datetime.now(timezone.utc).isoformat()
    workspace_dir = str(Path.cwd().resolve())
    return SessionInfo(
        alias=alias,
        session_id=session_id,
        created_at=created_at,
        workspace_dir=workspace_dir,
        raw=data,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="create_session.py")
    parser.add_argument("--alias", required=True, help="User-facing name for this session (stored in mapping file).")
    parser.add_argument(
        "--init-prompt",
        default=None,
        help="Optional first prompt; defaults to a minimal init prompt that replies 'OK'.",
    )
    parser.add_argument("--env-file", type=Path, help="Optional KEY=VALUE env file to load.")
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional alias→session mapping YAML file path (defaults to system tmp).",
    )
    parser.add_argument("--print-session-id", action="store_true")
    parser.add_argument("--print-mapping-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)
    init_prompt = ns.init_prompt or f"Initialize a new session named '{ns.alias}'. Reply only: OK"
    extra_env = _read_env_vars_file(ns.env_file) if ns.env_file else None

    info = create_session(alias=ns.alias, init_prompt=init_prompt, extra_env=extra_env)

    mapping_file = ns.mapping_file or _default_mapping_file()
    wrote_mapping = _write_mapping_file(
        mapping_file,
        alias=info.alias,
        session_id=info.session_id,
        workspace_dir=info.workspace_dir,
        created_at=info.created_at,
    )

    if ns.print_session_id:
        sys.stdout.write(info.session_id + "\n")

    if ns.print_mapping_json:
        sys.stdout.write(
            json.dumps(
                {
                    "alias": info.alias,
                    "session_id": info.session_id,
                    "workspace_dir": info.workspace_dir,
                    "created_at": info.created_at,
                    "mapping_file": str(mapping_file),
                    "mapping_file_written": wrote_mapping,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    if not ns.print_session_id and not ns.print_mapping_json:
        sys.stdout.write(info.session_id + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
