#!/usr/bin/env python3
"""
Record a Claude Code session_id under a user-provided session name.

This script does NOT call `claude`. You run `claude` (or your wrapper) yourself with
`--output-format json`, then pass either:
- the JSON to this script via stdin, or
- the extracted `--session-id`.

By default it updates a JSON mapping file in the system temp directory:
`agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/claude-code-session-mapping.json`
(session name entries are overwritten unconditionally). The mapping file stores the
workspace directory once (top-level), not repeated per session name. If the mapping file
cannot be written, the script prints a warning to stderr and still exits successfully.

Examples
--------
Record from a `claude` JSON response:
  claude -p "Initialize a new session. Reply only: OK" --output-format json | \
    python3 scripts/record_session.py --session-name review-foo --print-mapping-json

Record by passing the extracted session_id:
  python3 scripts/record_session.py --session-name review-foo --session-id "<id>" --print-mapping-json

Use a Claude wrapper command instead of `claude`:
  claude-wrapper -p "Initialize a new session. Reply only: OK" --output-format json | \
    python3 scripts/record_session.py --session-name review-foo --print-mapping-json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any


@dataclass(frozen=True)
class SessionInfo:
    session_name: str
    session_id: str
    created_at: str
    workspace_dir: str


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
    """Return the default session-name→session mapping file path for a workspace."""

    key = _workspace_dir_key(workspace_dir)
    return Path(tempfile.gettempdir()) / "agent-sessions" / key / "claude-code-session-mapping.json"


def _load_mapping_json(path: Path) -> dict[str, Any]:
    """Load mapping data from the JSON mapping file."""

    if not path.exists():
        return {"workspace_dir": "", "sessions": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("mapping file must be a JSON object")

    sessions_key = "sessions" if "sessions" in data else "aliases"
    if sessions_key in data and isinstance(data[sessions_key], dict):
        workspace_dir = data.get("workspace_dir")
        if not isinstance(workspace_dir, str):
            workspace_dir = ""
        sessions: dict[str, dict[str, str]] = {}
        for session_name, entry in data[sessions_key].items():
            if not isinstance(session_name, str):
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
                sessions[session_name] = normalized
        return {"workspace_dir": workspace_dir, "sessions": sessions}

    # Back-compat: older file shape where the top-level object was the session-name map.
    sessions: dict[str, dict[str, str]] = {}
    for session_name, entry in data.items():
        if not isinstance(session_name, str) or session_name in {"workspace_dir", "sessions", "aliases"}:
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
            sessions[session_name] = normalized
    workspace_dir = data.get("workspace_dir") if isinstance(data.get("workspace_dir"), str) else ""
    return {"workspace_dir": workspace_dir, "sessions": sessions}


def _write_mapping_file(
    mapping_file: Path,
    *,
    session_name: str,
    session_id: str,
    workspace_dir: str,
    created_at: str,
) -> bool:
    """Write (or update) the mapping file; overwrite session name unconditionally."""

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
                data = {"workspace_dir": "", "sessions": {}}
            except Exception as backup_error:  # noqa: BLE001
                sys.stderr.write(
                    f"[WARN] Failed to read/backup mapping file {mapping_file}: {backup_error}\n"
                )
                return False

        sessions = data.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
        sessions[session_name] = {"session_id": session_id, "created_at": created_at}
        data["workspace_dir"] = workspace_dir
        data["sessions"] = sessions

        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        tmp_path = mapping_file.with_name(mapping_file.name + f".tmp.{os.getpid()}")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, mapping_file)
        return True
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[WARN] Failed to write mapping file {mapping_file}: {e}\n")
        return False


def _parse_json_from_maybe_dirty_stdout(stdout: str) -> dict[str, Any]:
    """Parse JSON from stdout that may contain extra non-JSON lines."""

    stdout = stdout.strip()
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data
        raise ValueError("expected JSON object")
    except Exception:
        decoder = json.JSONDecoder()
        last_obj: dict[str, Any] | None = None
        for i, ch in enumerate(stdout):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(stdout[i:])
            except Exception:
                continue
            if isinstance(obj, dict):
                last_obj = obj
        if last_obj is None:
            raise ValueError("failed to find a JSON object in stdout")
        return last_obj


def _read_session_id_from_stdin() -> str:
    if sys.stdin.isatty():
        raise ValueError("stdin is empty; provide --session-id or pipe `claude --output-format json` output")
    payload = sys.stdin.read()
    data = _parse_json_from_maybe_dirty_stdout(payload)
    session_id = data.get("session_id")
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Missing `session_id` in JSON input.")
    return session_id


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="record_session.py")
    parser.add_argument(
        "--session-name",
        dest="session_name",
        required=True,
        help="User-facing name for this session (stored in mapping file).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session id to record. If omitted, read `claude --output-format json` from stdin and extract `.session_id`.",
    )
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="Workspace directory to scope the mapping file (defaults to current working directory).",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional session-name→session mapping JSON file path (defaults to system tmp, workspace-scoped).",
    )
    parser.add_argument("--print-session-id", action="store_true")
    parser.add_argument("--print-mapping-json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)
    session_id = ns.session_id or _read_session_id_from_stdin()
    created_at = datetime.now(timezone.utc).isoformat()
    workspace_dir = str(Path(ns.workspace_dir).resolve() if ns.workspace_dir else Path.cwd().resolve())
    info = SessionInfo(
        session_name=ns.session_name,
        session_id=session_id,
        created_at=created_at,
        workspace_dir=workspace_dir,
    )

    mapping_file = ns.mapping_file or _default_mapping_file(info.workspace_dir)
    wrote_mapping = _write_mapping_file(
        mapping_file,
        session_name=info.session_name,
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
                    "session_name": info.session_name,
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
