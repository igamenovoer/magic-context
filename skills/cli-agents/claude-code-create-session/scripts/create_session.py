#!/usr/bin/env python3
"""
Create a Claude Code session and print the resulting session_id.

This is a thin wrapper around the `claude` CLI intended for automation.
By default it also updates a JSON mapping file in the system temp directory:
`agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/claude-code-alias-mapping.json`
(alias entries are overwritten unconditionally). The mapping file stores the
workspace directory once (top-level), not repeated per alias. If the mapping file
cannot be written, the script prints a warning to stderr and still succeeds.

Examples
--------
Create a session and print an alias->session_id JSON mapping:
  python3 scripts/create_session.py --alias review-foo --print-mapping-json

Load credentials from an env file before calling `claude`:
  python3 scripts/create_session.py --alias review-foo --env-file /path/to/vars.env
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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


def _workspace_dir_key(workspace_dir: str) -> str:
    """Convert a workspace directory into a filesystem-safe key string.

    Format: <workspace-basename>-<md5(abs-workspace-dir)>
    """

    resolved = str(Path(workspace_dir).resolve())
    base_name = Path(resolved).name or "workspace"
    base_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("_") or "workspace"
    digest = hashlib.md5(resolved.encode("utf-8")).hexdigest()
    return f"{base_name}-{digest}"


def _default_mapping_file(workspace_dir: str) -> Path:
    """Return the default alias→session mapping file path for a workspace."""

    key = _workspace_dir_key(workspace_dir)
    return Path(tempfile.gettempdir()) / "agent-sessions" / key / "claude-code-alias-mapping.json"


def _load_mapping_json(path: Path) -> dict[str, Any]:
    """Load mapping data from the JSON mapping file."""

    if not path.exists():
        return {"workspace_dir": "", "aliases": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("mapping file must be a JSON object")

    if "aliases" in data and isinstance(data["aliases"], dict):
        workspace_dir = data.get("workspace_dir")
        if not isinstance(workspace_dir, str):
            workspace_dir = ""
        aliases: dict[str, dict[str, str]] = {}
        for alias, entry in data["aliases"].items():
            if not isinstance(alias, str):
                continue
            normalized: dict[str, str] = {}
            if isinstance(entry, dict):
                session_id = entry.get("session_id")
                if isinstance(session_id, str) and session_id:
                    normalized["session_id"] = session_id
                created_at = entry.get("created_at")
                if created_at is not None:
                    normalized["created_at"] = str(created_at)
            elif isinstance(entry, str) and entry:
                normalized["session_id"] = entry
            if "session_id" in normalized:
                aliases[alias] = normalized
        return {"workspace_dir": workspace_dir, "aliases": aliases}

    # Back-compat: older file shape where the top-level object was the alias map.
    aliases: dict[str, dict[str, str]] = {}
    for alias, entry in data.items():
        if not isinstance(alias, str) or alias in {"workspace_dir", "aliases"}:
            continue
        normalized = {}
        if isinstance(entry, dict):
            session_id = entry.get("session_id")
            if isinstance(session_id, str) and session_id:
                normalized["session_id"] = session_id
            created_at = entry.get("created_at")
            if created_at is not None:
                normalized["created_at"] = str(created_at)
        elif isinstance(entry, str) and entry:
            normalized["session_id"] = entry
        if "session_id" in normalized:
            aliases[alias] = normalized
    workspace_dir = data.get("workspace_dir") if isinstance(data.get("workspace_dir"), str) else ""
    return {"workspace_dir": workspace_dir, "aliases": aliases}


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
        try:
            data = _load_mapping_json(mapping_file)
        except Exception as e:  # noqa: BLE001
            backup = mapping_file.with_name(mapping_file.name + f".bad.{os.getpid()}")
            try:
                if mapping_file.exists():
                    mapping_file.replace(backup)
                sys.stderr.write(
                    f"[WARN] Mapping file was invalid JSON; moved to {backup} and recreated: {e}\n"
                )
                data = {"workspace_dir": "", "aliases": {}}
            except Exception as backup_error:  # noqa: BLE001
                sys.stderr.write(
                    f"[WARN] Failed to read/backup mapping file {mapping_file}: {backup_error}\n"
                )
                return False

        aliases = data.get("aliases")
        if not isinstance(aliases, dict):
            aliases = {}
        aliases[alias] = {"session_id": session_id, "created_at": created_at}
        data["workspace_dir"] = workspace_dir
        data["aliases"] = aliases

        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
        help="Optional alias→session mapping JSON file path (defaults to system tmp, workspace-scoped).",
    )
    parser.add_argument("--print-session-id", action="store_true")
    parser.add_argument("--print-mapping-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)
    init_prompt = ns.init_prompt or f"Initialize a new session named '{ns.alias}'. Reply only: OK"
    extra_env = _read_env_vars_file(ns.env_file) if ns.env_file else None

    info = create_session(alias=ns.alias, init_prompt=init_prompt, extra_env=extra_env)

    mapping_file = ns.mapping_file or _default_mapping_file(info.workspace_dir)
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
