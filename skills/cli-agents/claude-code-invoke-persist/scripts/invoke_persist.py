#!/usr/bin/env python3
"""
Create and resume Claude Code sessions with a persistent alias mapping.

This is a thin wrapper around the `claude` CLI intended for deterministic
automation. It supports:

- create-session: create a new session and persist alias->session_id mapping
- resolve: resolve a session name to session_id (no Claude call)
- list-sessions: list session names for a workspace mapping file (no Claude call)
- delete-session: delete an alias entry from the mapping file (no Claude call)
- delete-all-sessions: delete the mapping file for a workspace (no Claude call)
- resume-json: resume and run a prompt with JSON output
- resume-stream: resume and run a prompt with streaming JSONL output

If you use a claude-compatible wrapper command (drop-in replacement for `claude`),
pass it via `--claude-cmd "..."` or set `CLAUDE_CMD`.

Do not terminate a running Claude call unless one of these applies: an explicit
user-requested deadline, a confirmed stall, or a hard error state.
Do not terminate only because intermediate output appears off-prompt; let it
run to completion and evaluate the final result.

The default alias mapping file path is workspace-scoped under system temp:
  agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/claude-code-alias-mapping.json

Examples
--------
Create a session and print the session_id:
  python3 scripts/invoke_persist.py create-session --session-name review-src --print-session-id

Create a session and print a JSON summary (includes mapping file path):
  python3 scripts/invoke_persist.py create-session --session-name review-src --print-mapping-json

Resolve an alias to a session_id (no Claude call):
  python3 scripts/invoke_persist.py resolve --session-name review-src --print-session-id

List known aliases for the current workspace:
  python3 scripts/invoke_persist.py list-sessions --print-aliases

Delete one alias from the mapping file:
  python3 scripts/invoke_persist.py delete-session --session-name review-src

Delete the workspace mapping file:
  python3 scripts/invoke_persist.py delete-all-sessions

Resume and run a prompt (JSON output):
  python3 scripts/invoke_persist.py resume-json --session-name review-src --prompt "Continue"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionInfo:
    alias: str
    session_id: str
    created_at: str
    workspace_dir: str


@dataclass(frozen=True)
class ResolvedSession:
    session_id: str
    alias: str | None
    mapping_file: str | None
    last_model: str | None
    last_reasoning_effort: str | None
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


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    key = _workspace_dir_key(workspace_dir)
    return Path(tempfile.gettempdir()) / "agent-sessions" / key / "claude-code-alias-mapping.json"


def _load_mapping_json(path: Path) -> dict[str, Any]:
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
                last_model = entry.get("last_model")
                if isinstance(last_model, str) and last_model:
                    normalized["last_model"] = last_model
                last_reasoning_effort = entry.get("last_reasoning_effort")
                if isinstance(last_reasoning_effort, str) and last_reasoning_effort:
                    normalized["last_reasoning_effort"] = last_reasoning_effort
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
        normalized: dict[str, str] = {}
        if isinstance(entry, dict):
            session_id = entry.get("session_id")
            if isinstance(session_id, str) and session_id:
                normalized["session_id"] = session_id
            created_at = entry.get("created_at")
            if created_at is not None:
                normalized["created_at"] = str(created_at)
            last_model = entry.get("last_model")
            if isinstance(last_model, str) and last_model:
                normalized["last_model"] = last_model
            last_reasoning_effort = entry.get("last_reasoning_effort")
            if isinstance(last_reasoning_effort, str) and last_reasoning_effort:
                normalized["last_reasoning_effort"] = last_reasoning_effort
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
    last_model: str | None = None,
    last_reasoning_effort: str | None = None,
) -> bool:
    try:
        mapping_file.parent.mkdir(parents=True, exist_ok=True)

        aliases: dict[str, dict[str, str]] = {}
        data: dict[str, Any] = {"workspace_dir": workspace_dir, "aliases": aliases}
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
            except Exception as backup_error:  # noqa: BLE001
                sys.stderr.write(
                    f"[WARN] Failed to read/backup mapping file {mapping_file}: {backup_error}\n"
                )
                return False
            data = {"workspace_dir": workspace_dir, "aliases": {}}

        existing_aliases = data.get("aliases")
        if isinstance(existing_aliases, dict):
            aliases = existing_aliases
        else:
            aliases = {}

        entry: dict[str, str] = {"session_id": session_id, "created_at": created_at}
        if isinstance(last_model, str) and last_model:
            entry["last_model"] = last_model
        if isinstance(last_reasoning_effort, str) and last_reasoning_effort:
            entry["last_reasoning_effort"] = last_reasoning_effort
        aliases[alias] = entry
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


def _update_last_call_metadata(
    mapping_file: Path,
    *,
    alias: str,
    session_id: str,
    workspace_dir: str,
    last_model: str | None,
    last_reasoning_effort: str | None,
) -> bool:
    if last_model is None and last_reasoning_effort is None:
        return True

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
            except Exception as backup_error:  # noqa: BLE001
                sys.stderr.write(
                    f"[WARN] Failed to read/backup mapping file {mapping_file}: {backup_error}\n"
                )
                return False
            data = {"workspace_dir": workspace_dir, "aliases": {}}

        stored_workspace_dir = data.get("workspace_dir")
        if not isinstance(stored_workspace_dir, str) or not stored_workspace_dir:
            data["workspace_dir"] = workspace_dir

        aliases = data.get("aliases")
        if not isinstance(aliases, dict):
            aliases = {}

        entry = aliases.get(alias)
        if not isinstance(entry, dict):
            entry = {}

        entry["session_id"] = session_id
        if "created_at" not in entry:
            entry["created_at"] = datetime.now(timezone.utc).isoformat()
        if isinstance(last_model, str) and last_model:
            entry["last_model"] = last_model
        if isinstance(last_reasoning_effort, str) and last_reasoning_effort:
            entry["last_reasoning_effort"] = last_reasoning_effort

        aliases[alias] = entry
        data["aliases"] = aliases

        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        tmp_path = mapping_file.with_name(mapping_file.name + f".tmp.{os.getpid()}")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, mapping_file)
        return True
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[WARN] Failed to update mapping file {mapping_file}: {e}\n")
        return False


def _normalize_optional_str(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _validate_deadline_seconds(deadline_seconds: float | None) -> None:
    if deadline_seconds is None:
        return
    if not (deadline_seconds > 0):
        raise ValueError("--deadline-seconds must be > 0 when provided.")


def _normalize_nonempty_str(value: str | None, *, flag_name: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"--{flag_name} cannot be empty.")
    return stripped


def _resolve_claude_cmd(raw: str | None) -> list[str]:
    """Resolve a claude-compatible command (wrapper) into argv tokens.

    Notes
    -----
    - The wrapper must be compatible with the `claude` CLI flags used here:
      `-p`, `--output-format`, `--resume`, `--model`, `--effort`, `--append-system-prompt`, and
      (for streaming) `--verbose`, `--include-partial-messages`.
    - Shell aliases are not supported (this uses subprocess without a shell).
    """

    candidate = raw if (isinstance(raw, str) and raw.strip()) else os.environ.get("CLAUDE_CMD", "")
    if isinstance(candidate, str) and candidate.strip():
        parts = shlex.split(candidate)
        if not parts:
            raise ValueError("Empty --claude-cmd/CLAUDE_CMD after parsing.")
        return parts
    return ["claude"]


def _run_claude(
    claude_cmd: list[str],
    args: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        return subprocess.run(
            claude_cmd + args,
            text=True,
            capture_output=True,
            env=env,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as e:
        cmd_display = " ".join(claude_cmd) if claude_cmd else "<empty>"
        raise RuntimeError(
            f"Claude command not found: {cmd_display}. Provide --claude-cmd or set CLAUDE_CMD, "
            "or ensure `claude` is on PATH."
        ) from e
    except subprocess.TimeoutExpired as e:
        cmd_display = " ".join(claude_cmd) if claude_cmd else "<empty>"
        raise RuntimeError(
            f"Claude command exceeded deadline ({timeout_seconds}s) and was terminated: {cmd_display}"
        ) from e


def create_session(
    *,
    alias: str,
    init_prompt: str,
    claude_cmd: list[str],
    append_system_prompt: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    extra_env: dict[str, str] | None = None,
    deadline_seconds: float | None = None,
) -> SessionInfo:
    """Create a new Claude Code session and return session metadata."""

    cmd = ["-p", init_prompt, "--output-format", "json"]
    if isinstance(append_system_prompt, str) and append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])
    if isinstance(model, str) and model:
        cmd.extend(["--model", model])
    if isinstance(reasoning_effort, str) and reasoning_effort:
        cmd.extend(["--effort", reasoning_effort])
    proc = _run_claude(claude_cmd, cmd, extra_env=extra_env, timeout_seconds=deadline_seconds)

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
    )


def resolve_session(
    *,
    session_id: str | None,
    session_name: str | None,
    mapping_file: Path | None,
) -> ResolvedSession:
    """Resolve a session selector to a concrete `session_id`."""

    if isinstance(session_id, str) and session_id.strip():
        stripped_id = session_id.strip()

        # Best-effort: if the session_id is already present in the workspace mapping,
        # reuse the stored metadata defaults (model/reasoning effort) and enable
        # updating the entry after the call.
        workspace_dir = str(Path.cwd().resolve())
        mapping_path = mapping_file or _default_mapping_file(workspace_dir)
        if mapping_path.exists():
            try:
                data = _load_mapping_json(mapping_path)
                aliases = data.get("aliases")
                if isinstance(aliases, dict):
                    for alias_name in sorted(a for a in aliases.keys() if isinstance(a, str)):
                        entry = aliases.get(alias_name)
                        if not isinstance(entry, dict):
                            continue
                        if entry.get("session_id") != stripped_id:
                            continue
                        last_model = entry.get("last_model")
                        if not isinstance(last_model, str) or not last_model:
                            last_model = None
                        last_reasoning_effort = entry.get("last_reasoning_effort")
                        if not isinstance(last_reasoning_effort, str) or not last_reasoning_effort:
                            last_reasoning_effort = None
                        return ResolvedSession(
                            session_id=stripped_id,
                            alias=alias_name,
                            mapping_file=str(mapping_path),
                            last_model=last_model,
                            last_reasoning_effort=last_reasoning_effort,
                            source="mapping_file_by_session_id",
                        )
            except Exception:  # noqa: BLE001
                pass

        return ResolvedSession(
            session_id=stripped_id,
            alias=None,
            mapping_file=None,
            last_model=None,
            last_reasoning_effort=None,
            source="session_id",
        )

    chosen_name = _normalize_optional_str(session_name)
    if not chosen_name:
        raise ValueError("Missing session selector: provide --session-id or --session-name.")

    workspace_dir = str(Path.cwd().resolve())
    mapping_path = mapping_file or _default_mapping_file(workspace_dir)
    data = _load_mapping_json(mapping_path)
    aliases = data.get("aliases")
    if not isinstance(aliases, dict):
        aliases = {}

    entry = aliases.get(chosen_name)
    if not isinstance(entry, dict):
        entry = {}
    resolved_id = entry.get("session_id")
    if not isinstance(resolved_id, str) or not resolved_id:
        raise KeyError(
            f"Session name '{chosen_name}' not found in mapping file: {mapping_path}. "
            "Create it first with the creation stage."
        )

    last_model = entry.get("last_model")
    if not isinstance(last_model, str) or not last_model:
        last_model = None
    last_reasoning_effort = entry.get("last_reasoning_effort")
    if not isinstance(last_reasoning_effort, str) or not last_reasoning_effort:
        last_reasoning_effort = None

    return ResolvedSession(
        session_id=resolved_id,
        alias=chosen_name,
        mapping_file=str(mapping_path),
        last_model=last_model,
        last_reasoning_effort=last_reasoning_effort,
        source="mapping_file",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="invoke_persist.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--session-id", help="Resume a specific Claude session_id (preferred if provided).")
    selector.add_argument("--session-name", help="Session name to resolve via the mapping file.")
    selector.add_argument("--session-alias", dest="session_name", help=argparse.SUPPRESS)
    selector.add_argument("--alias", dest="session_name", help=argparse.SUPPRESS)
    selector.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional alias mapping JSON file path (defaults to system tmp, workspace-scoped).",
    )

    common_call = argparse.ArgumentParser(add_help=False)
    common_call.add_argument("--env-file", type=Path, help="Optional KEY=VALUE env file to load.")
    common_call.add_argument(
        "--claude-cmd",
        default=None,
        help="Optional claude-compatible wrapper command (default: CLAUDE_CMD env var or `claude`).",
    )
    common_call.add_argument(
        "--model",
        default=None,
        help="Optional model to request (persisted per alias as last_model).",
    )
    common_call.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        default=None,
        help="Optional reasoning effort to request (persisted per alias as last_reasoning_effort).",
    )
    common_call.add_argument("--effort", dest="reasoning_effort", default=None, help=argparse.SUPPRESS)
    common_call.add_argument(
        "--deadline-seconds",
        type=float,
        default=None,
        help="Optional overall deadline in seconds. Do not set unless the user requested a time limit.",
    )

    workspace_ops = argparse.ArgumentParser(add_help=False)
    workspace_ops.add_argument(
        "--workspace-dir",
        default=None,
        help="Workspace directory to operate on (defaults to current working directory).",
    )
    workspace_ops.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional alias mapping JSON file path. If set, overrides --workspace-dir.",
    )

    create = sub.add_parser("create-session", parents=[selector, common_call], help="Create a new Claude session.")
    create.add_argument(
        "--role-definition-md",
        type=Path,
        default=None,
        help="Optional Markdown file whose contents are appended to the default system prompt.",
    )
    create.add_argument(
        "--init-prompt",
        default=None,
        help="Optional first prompt; defaults to a minimal init prompt that replies 'OK'.",
    )
    create.add_argument(
        "--fake-session-id",
        default=None,
        help="Testing helper: skip calling `claude` and use this session_id.",
    )
    create.add_argument("--print-session-id", action="store_true")
    create.add_argument("--print-mapping-json", action="store_true")

    resolve = sub.add_parser("resolve", parents=[selector], help="Resolve alias/name to session_id (no Claude call).")
    resolve.add_argument("--print-session-id", action="store_true")

    list_sessions = sub.add_parser("list-sessions", help="List sessions for a workspace mapping file.")
    list_sessions.add_argument(
        "--workspace-dir",
        default=None,
        help="Workspace directory to list sessions for (defaults to current working directory).",
    )
    list_sessions.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Optional alias mapping JSON file path. If set, overrides --workspace-dir.",
    )
    list_sessions.add_argument("--print-aliases", action="store_true", help="Print only alias names (one per line).")

    delete_session = sub.add_parser(
        "delete-session",
        parents=[common_call, workspace_ops],
        help="Delete a saved session entry from the workspace mapping file (no Claude call).",
    )
    delete_session.add_argument("--session-id", help="Delete entries matching this session_id (if no alias provided).")
    delete_session.add_argument("--session-name", help="Session name to delete from the mapping file.")
    delete_session.add_argument("--session-alias", dest="session_name", help=argparse.SUPPRESS)
    delete_session.add_argument("--alias", dest="session_name", help=argparse.SUPPRESS)
    delete_session.add_argument(
        "--print-mapping-json",
        action="store_true",
        help="Include the mapping JSON in the output (even if unchanged).",
    )

    delete_all = sub.add_parser(
        "delete-all-sessions",
        parents=[common_call, workspace_ops],
        help="Delete the workspace mapping file entirely (no Claude call).",
    )

    resume_json = sub.add_parser(
        "resume-json",
        parents=[selector, common_call],
        help="Run `claude -p` resuming a session with JSON output.",
    )
    resume_json.add_argument("--prompt", required=True)
    resume_json.add_argument("--print-result", action="store_true")
    resume_json.add_argument("--print-session-id", action="store_true")

    resume_stream = sub.add_parser(
        "resume-stream",
        parents=[selector, common_call],
        help="Run `claude -p` resuming a session with streaming JSONL output.",
    )
    resume_stream.add_argument("--prompt", required=True)
    resume_stream.add_argument("--include-partials", action="store_true")
    resume_stream.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)

    if ns.cmd == "create-session":
        session_name = _normalize_optional_str(getattr(ns, "session_name", None))
        if not session_name:
            raise ValueError("Missing session name: provide --session-name.")

        extra_env = _read_env_vars_file(ns.env_file) if ns.env_file else None
        claude_cmd = _resolve_claude_cmd(ns.claude_cmd)
        model = _normalize_nonempty_str(getattr(ns, "model", None), flag_name="model")
        reasoning_effort = _normalize_nonempty_str(getattr(ns, "reasoning_effort", None), flag_name="reasoning-effort")
        _validate_deadline_seconds(ns.deadline_seconds)
        append_system_prompt: str | None = None
        if ns.role_definition_md is not None:
            role_path = ns.role_definition_md
            if role_path.suffix.lower() != ".md":
                raise ValueError("--role-definition-md must point to a .md file.")
            if not role_path.exists():
                raise FileNotFoundError(f"Role definition file not found: {role_path}")
            append_system_prompt = _read_text_file(role_path)
        if ns.fake_session_id:
            created_at = datetime.now(timezone.utc).isoformat()
            workspace_dir = str(Path.cwd().resolve())
            info = SessionInfo(
                alias=session_name,
                session_id=str(ns.fake_session_id),
                created_at=created_at,
                workspace_dir=workspace_dir,
            )
        else:
            init_prompt = ns.init_prompt or f"Initialize a new session named '{session_name}'. Reply only: OK"
            info = create_session(
                alias=session_name,
                init_prompt=init_prompt,
                claude_cmd=claude_cmd,
                append_system_prompt=append_system_prompt,
                model=model,
                reasoning_effort=reasoning_effort,
                extra_env=extra_env,
                deadline_seconds=ns.deadline_seconds,
            )

        mapping_file = ns.mapping_file or _default_mapping_file(info.workspace_dir)
        wrote_mapping = _write_mapping_file(
            mapping_file,
            alias=info.alias,
            session_id=info.session_id,
            workspace_dir=info.workspace_dir,
            created_at=info.created_at,
            last_model=model,
            last_reasoning_effort=reasoning_effort,
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
                        "last_model": model,
                        "last_reasoning_effort": reasoning_effort,
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

    if ns.cmd == "resolve":
        resolved = resolve_session(
            session_id=ns.session_id,
            session_name=ns.session_name,
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
                    "last_model": resolved.last_model,
                    "last_reasoning_effort": resolved.last_reasoning_effort,
                    "source": resolved.source,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 0

    if ns.cmd == "list-sessions":
        workspace_dir = ns.workspace_dir or str(Path.cwd().resolve())
        mapping_path = ns.mapping_file or _default_mapping_file(workspace_dir)
        if not mapping_path.exists():
            if ns.print_aliases:
                return 0
            sys.stdout.write(
                json.dumps(
                    {
                        "workspace_dir": workspace_dir,
                        "mapping_file": str(mapping_path),
                        "aliases": {},
                        "alias_count": 0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            return 0

        data = _load_mapping_json(mapping_path)
        aliases = data.get("aliases")
        if not isinstance(aliases, dict):
            aliases = {}

        if ns.print_aliases:
            for alias in sorted(a for a in aliases.keys() if isinstance(a, str)):
                sys.stdout.write(alias + "\n")
            return 0

        sys.stdout.write(
            json.dumps(
                {
                    "workspace_dir": workspace_dir,
                    "mapping_file": str(mapping_path),
                    "aliases": aliases,
                    "alias_count": len(aliases),
                    "stored_workspace_dir": data.get("workspace_dir", ""),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 0

    if ns.cmd == "delete-session":
        workspace_dir = ns.workspace_dir or str(Path.cwd().resolve())
        mapping_path = ns.mapping_file or _default_mapping_file(workspace_dir)

        requested_session_name = _normalize_optional_str(ns.session_name)
        requested_session_id = ns.session_id.strip() if isinstance(ns.session_id, str) and ns.session_id.strip() else None
        if not requested_session_name and not requested_session_id:
            raise ValueError(
                "Missing delete selector: provide --session-name or --session-id."
            )

        data: dict[str, Any] = {"workspace_dir": workspace_dir, "aliases": {}}
        stored_workspace_dir = ""
        mapping_file_existed = mapping_path.exists()
        recreated_from_invalid = False
        wrote_mapping = False
        removed_aliases: list[str] = []
        aliases_before = 0
        aliases_after = 0

        if mapping_file_existed:
            try:
                data = _load_mapping_json(mapping_path)
            except Exception as e:  # noqa: BLE001
                backup = mapping_path.with_name(mapping_path.name + f".bad.{os.getpid()}")
                try:
                    if mapping_path.exists():
                        mapping_path.replace(backup)
                    sys.stderr.write(
                        f"[WARN] Mapping file was invalid JSON; moved to {backup} and recreated: {e}\n"
                    )
                except Exception as backup_error:  # noqa: BLE001
                    raise RuntimeError(
                        f"Failed to read/backup mapping file {mapping_path}: {backup_error}"
                    ) from backup_error
                data = {"workspace_dir": workspace_dir, "aliases": {}}
                recreated_from_invalid = True

            stored_workspace_dir = data.get("workspace_dir", "") if isinstance(data.get("workspace_dir"), str) else ""
            aliases = data.get("aliases")
            if not isinstance(aliases, dict):
                aliases = {}
            aliases_before = len(aliases)

            if requested_session_name:
                if requested_session_name in aliases:
                    removed_aliases.append(requested_session_name)
                    aliases.pop(requested_session_name, None)
            else:
                assert requested_session_id is not None
                for alias_name in sorted(a for a in list(aliases.keys()) if isinstance(a, str)):
                    entry = aliases.get(alias_name)
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("session_id") != requested_session_id:
                        continue
                    removed_aliases.append(alias_name)
                    aliases.pop(alias_name, None)

            if not isinstance(data.get("workspace_dir"), str) or not data.get("workspace_dir"):
                data["workspace_dir"] = workspace_dir
            data["aliases"] = aliases

            if removed_aliases or recreated_from_invalid:
                payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                tmp_path = mapping_path.with_name(mapping_path.name + f".tmp.{os.getpid()}")
                tmp_path.write_text(payload, encoding="utf-8")
                os.replace(tmp_path, mapping_path)
                wrote_mapping = True

            aliases_after = len(aliases)

        if ns.print_mapping_json and not mapping_file_existed:
            data = {"workspace_dir": workspace_dir, "aliases": {}}

        sys.stdout.write(
            json.dumps(
                {
                    "workspace_dir": workspace_dir,
                    "mapping_file": str(mapping_path),
                    "mapping_file_existed": mapping_file_existed,
                    "stored_workspace_dir": stored_workspace_dir,
                    "requested_session_name": requested_session_name,
                    "requested_session_id": requested_session_id,
                    "removed_aliases": removed_aliases,
                    "removed_count": len(removed_aliases),
                    "alias_count_before": aliases_before,
                    "alias_count_after": aliases_after,
                    "mapping_file_written": wrote_mapping,
                    "mapping_recreated_from_invalid_json": recreated_from_invalid,
                    "mapping": data if ns.print_mapping_json else None,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 0

    if ns.cmd == "delete-all-sessions":
        workspace_dir = ns.workspace_dir or str(Path.cwd().resolve())
        mapping_path = ns.mapping_file or _default_mapping_file(workspace_dir)

        existed = mapping_path.exists()
        deleted = False
        deleted_parent_dir = False
        if existed:
            try:
                mapping_path.unlink()
                deleted = True
            except FileNotFoundError:
                existed = False
                deleted = False
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Failed to delete mapping file {mapping_path}: {e}") from e

        if deleted:
            try:
                mapping_path.parent.rmdir()
                deleted_parent_dir = True
            except OSError:
                deleted_parent_dir = False

        sys.stdout.write(
            json.dumps(
                {
                    "workspace_dir": workspace_dir,
                    "mapping_file": str(mapping_path),
                    "mapping_file_existed": existed,
                    "mapping_file_deleted": deleted,
                    "mapping_parent_dir_deleted_if_empty": deleted_parent_dir,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        return 0

    extra_env = _read_env_vars_file(ns.env_file) if ns.env_file else None
    claude_cmd = _resolve_claude_cmd(ns.claude_cmd)
    explicit_model = _normalize_nonempty_str(getattr(ns, "model", None), flag_name="model")
    explicit_reasoning_effort = _normalize_nonempty_str(
        getattr(ns, "reasoning_effort", None), flag_name="reasoning-effort"
    )
    deadline_seconds = getattr(ns, "deadline_seconds", None)
    _validate_deadline_seconds(deadline_seconds)

    resolved = resolve_session(
        session_id=ns.session_id,
        session_name=ns.session_name,
        mapping_file=ns.mapping_file,
    )
    effective_model = explicit_model or resolved.last_model
    effective_reasoning_effort = explicit_reasoning_effort or resolved.last_reasoning_effort

    if ns.cmd == "resume-json":
        cmd = ["-p", ns.prompt, "--resume", resolved.session_id, "--output-format", "json"]
        if effective_model:
            cmd.extend(["--model", effective_model])
        if effective_reasoning_effort:
            cmd.extend(["--effort", effective_reasoning_effort])
        proc = _run_claude(claude_cmd, cmd, extra_env=extra_env, timeout_seconds=deadline_seconds)
        if proc.returncode != 0:
            raise RuntimeError(f"claude failed rc={proc.returncode}: {proc.stderr.strip()}")

        if resolved.alias and resolved.mapping_file:
            _update_last_call_metadata(
                Path(resolved.mapping_file),
                alias=resolved.alias,
                session_id=resolved.session_id,
                workspace_dir=str(Path.cwd().resolve()),
                last_model=effective_model,
                last_reasoning_effort=effective_reasoning_effort,
            )

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
        cmd = claude_cmd + ["-p", ns.prompt, "--output-format", "stream-json", "--resume", resolved.session_id]
        if effective_model:
            cmd.extend(["--model", effective_model])
        if effective_reasoning_effort:
            cmd.extend(["--effort", effective_reasoning_effort])
        if ns.verbose:
            cmd.append("--verbose")
        if ns.include_partials:
            cmd.append("--include-partial-messages")

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        deadline_hit = threading.Event()
        stop_deadline = threading.Event()

        def _deadline_killer() -> None:
            if deadline_seconds is None:
                return
            if stop_deadline.wait(deadline_seconds):
                return
            deadline_hit.set()
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                return
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except FileNotFoundError as e:
            cmd_display = " ".join(claude_cmd) if claude_cmd else "<empty>"
            raise RuntimeError(
                f"Claude command not found: {cmd_display}. Provide --claude-cmd or set CLAUDE_CMD, "
                "or ensure `claude` is on PATH."
            ) from e

        killer_thread = threading.Thread(target=_deadline_killer, daemon=True)
        killer_thread.start()

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        rc = proc.wait()
        stop_deadline.set()
        if deadline_hit.is_set():
            raise RuntimeError(f"claude exceeded deadline ({deadline_seconds}s) and was terminated.")
        if rc != 0:
            err = (proc.stderr.read() if proc.stderr else "").strip()
            raise RuntimeError(f"claude failed rc={rc}: {err}")
        if resolved.alias and resolved.mapping_file:
            _update_last_call_metadata(
                Path(resolved.mapping_file),
                alias=resolved.alias,
                session_id=resolved.session_id,
                workspace_dir=str(Path.cwd().resolve()),
                last_model=effective_model,
                last_reasoning_effort=effective_reasoning_effort,
            )
        return 0

    raise AssertionError(f"Unhandled cmd: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
