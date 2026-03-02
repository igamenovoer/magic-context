#!/usr/bin/env python3
"""
Resume a Claude Code session by alias and run a prompt.

This is a thin wrapper around the `claude` CLI intended for deterministic
automation. It resolves a user-facing session name/session alias to a
`session_id` using the mapping file created by the `claude-code-create-session`
skill:

  agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/claude-code-alias-mapping.json

Examples
--------
Resolve an alias to a session_id (no Claude call):
  python3 scripts/session_call.py resolve --session-alias review-src --print-session-id

Resume and run a prompt (JSON output):
  python3 scripts/session_call.py resume-json --session-alias review-src --prompt "Continue"

Stream events (JSONL):
  python3 scripts/session_call.py resume-stream --session-alias review-src --prompt "Continue" --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedSession:
    """Resolved session selector."""

    session_id: str
    alias: str | None
    mapping_file: str | None
    source: str


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
    """Return the default alias->session mapping file path for a workspace."""

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


def _select_alias(*, session_name: str | None, session_alias: str | None, alias: str | None) -> str | None:
    for value in (session_name, session_alias, alias):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_session(
    *,
    session_id: str | None,
    session_name: str | None,
    session_alias: str | None,
    alias: str | None,
    mapping_file: Path | None,
) -> ResolvedSession:
    """Resolve a session selector to a concrete `session_id`."""

    if isinstance(session_id, str) and session_id.strip():
        return ResolvedSession(
            session_id=session_id.strip(),
            alias=None,
            mapping_file=None,
            source="session_id",
        )

    chosen_alias = _select_alias(session_name=session_name, session_alias=session_alias, alias=alias)
    if not chosen_alias:
        raise ValueError(
            "Missing session selector: provide --session-id or --session-name/--session-alias/--alias."
        )

    workspace_dir = str(Path.cwd().resolve())
    mapping_path = mapping_file or _default_mapping_file(workspace_dir)
    data = _load_mapping_json(mapping_path)
    aliases = data.get("aliases")
    if not isinstance(aliases, dict):
        aliases = {}

    entry = aliases.get(chosen_alias)
    if not isinstance(entry, dict):
        entry = {}
    resolved_id = entry.get("session_id")
    if not isinstance(resolved_id, str) or not resolved_id:
        raise KeyError(
            f"Session alias '{chosen_alias}' not found in mapping file: {mapping_path}. "
            "Create it first with $claude-code-create-session."
        )

    return ResolvedSession(
        session_id=resolved_id,
        alias=chosen_alias,
        mapping_file=str(mapping_path),
        source="mapping_file",
    )


def _run_claude(args: list[str], *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(args, text=True, capture_output=True, env=env, check=False)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="session_call.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    session_selector = argparse.ArgumentParser(add_help=False)
    session_selector.add_argument("--session-id", help="Resume a specific Claude session_id (preferred if provided).")
    session_selector.add_argument("--session-name", help="Session name to resolve via the mapping file.")
    session_selector.add_argument("--session-alias", help="Session alias to resolve via the mapping file.")
    session_selector.add_argument("--alias", help="Legacy alias name to resolve via the mapping file.")
    session_selector.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional alias->session mapping JSON file path (defaults to system tmp, workspace-scoped).",
    )

    resolve = sub.add_parser("resolve", parents=[session_selector], help="Resolve alias/name to session_id (no Claude call).")
    resolve.add_argument("--print-session-id", action="store_true")

    common_call = argparse.ArgumentParser(add_help=False)
    common_call.add_argument("--prompt", required=True)
    common_call.add_argument("--env-file", type=Path, help="Optional KEY=VALUE env file to load.")

    resume_json = sub.add_parser(
        "resume-json",
        parents=[session_selector, common_call],
        help="Run `claude -p` resuming a session with JSON output.",
    )
    resume_json.add_argument("--print-result", action="store_true")
    resume_json.add_argument("--print-session-id", action="store_true")

    resume_stream = sub.add_parser(
        "resume-stream",
        parents=[session_selector, common_call],
        help="Run `claude -p` resuming a session with streaming JSONL output.",
    )
    resume_stream.add_argument("--include-partials", action="store_true")
    resume_stream.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)

    if ns.cmd == "resolve":
        resolved = resolve_session(
            session_id=ns.session_id,
            session_name=ns.session_name,
            session_alias=ns.session_alias,
            alias=ns.alias,
            mapping_file=ns.mapping_file,
        )
        if ns.print_session_id:
            sys.stdout.write(resolved.session_id + "\n")
            return 0
        sys.stdout.write(
            json.dumps(
                {
                    "session_id": resolved.session_id,
                    "alias": resolved.alias,
                    "mapping_file": resolved.mapping_file,
                    "source": resolved.source,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 0

    extra_env = _read_env_vars_file(ns.env_file) if ns.env_file else None

    resolved = resolve_session(
        session_id=ns.session_id,
        session_name=ns.session_name,
        session_alias=ns.session_alias,
        alias=ns.alias,
        mapping_file=ns.mapping_file,
    )

    if ns.cmd == "resume-json":
        cmd = ["claude", "-p", ns.prompt, "--resume", resolved.session_id, "--output-format", "json"]
        proc = _run_claude(cmd, extra_env=extra_env)
        if proc.returncode != 0:
            raise RuntimeError(f"claude failed rc={proc.returncode}: {proc.stderr.strip()}")

        if not ns.print_result and not ns.print_session_id:
            sys.stdout.write(proc.stdout)
            return 0

        data = json.loads(proc.stdout)
        if ns.print_result:
            sys.stdout.write(f"{data.get('result')}\n")
        if ns.print_session_id:
            sys.stdout.write(f"{data.get('session_id') or ''}\n")
        return 0

    if ns.cmd == "resume-stream":
        cmd = ["claude", "-p", ns.prompt, "--output-format", "stream-json", "--resume", resolved.session_id]
        if ns.verbose:
            cmd.append("--verbose")
        if ns.include_partials:
            cmd.append("--include-partial-messages")

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        rc = proc.wait()
        if rc != 0:
            err = (proc.stderr.read() if proc.stderr else "").strip()
            raise RuntimeError(f"claude failed rc={rc}: {err}")
        return 0

    raise AssertionError(f"Unhandled cmd: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
