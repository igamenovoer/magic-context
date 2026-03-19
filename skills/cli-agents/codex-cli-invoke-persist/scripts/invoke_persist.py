#!/usr/bin/env python3
"""
Create and resume Codex CLI sessions with a persistent alias mapping.

This is a thin wrapper around the `codex` CLI intended for deterministic
automation. It supports:

- create-session: create a new session and persist alias -> thread_id mapping
- resolve: resolve a session name to thread_id without calling Codex
- list-sessions: list session names for a workspace mapping file
- delete-session: delete an alias entry from the mapping file
- delete-all-sessions: delete the mapping file for a workspace
- resume-json: resume and run a prompt, then print one JSON summary object
- resume-stream: resume and run a prompt with streaming JSONL output

If you use a codex-compatible wrapper command, pass it via `--codex-cmd "..."`
or set `CODEX_CMD`. For shell aliases/functions, also pass `--codex-shell`
and usually `--codex-shell-init`.

Do not terminate a running Codex call unless one of these applies: an explicit
user-requested deadline, a confirmed stall, or a hard error state.
Do not terminate only because intermediate output appears off-prompt; let it
run to completion and evaluate the final result.

The default alias mapping file path is workspace-scoped under system temp:
  agent-sessions/<workspace-basename>-<md5(abs-workspace-dir)>/codex-cli-alias-mapping.json
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionInfo:
    alias: str
    thread_id: str
    created_at: str
    workspace_dir: str


@dataclass(frozen=True)
class ResolvedSession:
    thread_id: str
    alias: str | None
    mapping_file: str | None
    last_model: str | None
    source: str


@dataclass(frozen=True)
class CodexRunResult:
    thread_id: str | None
    events: list[Any]
    stderr: str
    final_message: str | None


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
    resolved = str(Path(workspace_dir).resolve())
    base_name = Path(resolved).name or "workspace"
    base_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("_") or "workspace"
    digest = hashlib.md5(resolved.encode("utf-8")).hexdigest()
    return f"{base_name}-{digest}"


def _default_mapping_file(workspace_dir: str) -> Path:
    key = _workspace_dir_key(workspace_dir)
    return Path(tempfile.gettempdir()) / "agent-sessions" / key / "codex-cli-alias-mapping.json"


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
                thread_id = entry.get("thread_id")
                if isinstance(thread_id, str) and thread_id:
                    normalized["thread_id"] = thread_id
                created_at = entry.get("created_at")
                if created_at is not None:
                    normalized["created_at"] = str(created_at)
                last_model = entry.get("last_model")
                if isinstance(last_model, str) and last_model:
                    normalized["last_model"] = last_model
            elif isinstance(entry, str) and entry:
                normalized["thread_id"] = entry
            if "thread_id" in normalized:
                aliases[alias] = normalized
        return {"workspace_dir": workspace_dir, "aliases": aliases}

    aliases = {}
    for alias, entry in data.items():
        if not isinstance(alias, str) or alias in {"workspace_dir", "aliases"}:
            continue
        normalized: dict[str, str] = {}
        if isinstance(entry, dict):
            thread_id = entry.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                normalized["thread_id"] = thread_id
            created_at = entry.get("created_at")
            if created_at is not None:
                normalized["created_at"] = str(created_at)
            last_model = entry.get("last_model")
            if isinstance(last_model, str) and last_model:
                normalized["last_model"] = last_model
        elif isinstance(entry, str) and entry:
            normalized["thread_id"] = entry
        if "thread_id" in normalized:
            aliases[alias] = normalized
    workspace_dir = data.get("workspace_dir") if isinstance(data.get("workspace_dir"), str) else ""
    return {"workspace_dir": workspace_dir, "aliases": aliases}


def _write_mapping_file(
    mapping_file: Path,
    *,
    alias: str,
    thread_id: str,
    workspace_dir: str,
    created_at: str,
    last_model: str | None = None,
) -> bool:
    try:
        mapping_file.parent.mkdir(parents=True, exist_ok=True)

        aliases: dict[str, dict[str, str]] = {}
        data: dict[str, Any] = {"workspace_dir": workspace_dir, "aliases": aliases}
        try:
            data = _load_mapping_json(mapping_file)
        except Exception as exc:  # noqa: BLE001
            backup = mapping_file.with_name(mapping_file.name + f".bad.{os.getpid()}")
            try:
                if mapping_file.exists():
                    mapping_file.replace(backup)
                sys.stderr.write(
                    f"[WARN] Mapping file was invalid JSON; moved to {backup} and recreated: {exc}\n"
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

        entry: dict[str, str] = {"thread_id": thread_id, "created_at": created_at}
        if isinstance(last_model, str) and last_model:
            entry["last_model"] = last_model
        aliases[alias] = entry
        data["workspace_dir"] = workspace_dir
        data["aliases"] = aliases

        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        tmp_path = mapping_file.with_name(mapping_file.name + f".tmp.{os.getpid()}")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, mapping_file)
        return True
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[WARN] Failed to write mapping file {mapping_file}: {exc}\n")
        return False


def _update_last_model(
    mapping_file: Path,
    *,
    alias: str,
    thread_id: str,
    workspace_dir: str,
    last_model: str | None,
) -> bool:
    if last_model is None:
        return True

    try:
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = _load_mapping_json(mapping_file)
        except Exception as exc:  # noqa: BLE001
            backup = mapping_file.with_name(mapping_file.name + f".bad.{os.getpid()}")
            try:
                if mapping_file.exists():
                    mapping_file.replace(backup)
                sys.stderr.write(
                    f"[WARN] Mapping file was invalid JSON; moved to {backup} and recreated: {exc}\n"
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

        entry["thread_id"] = thread_id
        if "created_at" not in entry:
            entry["created_at"] = datetime.now(timezone.utc).isoformat()
        entry["last_model"] = last_model

        aliases[alias] = entry
        data["aliases"] = aliases

        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        tmp_path = mapping_file.with_name(mapping_file.name + f".tmp.{os.getpid()}")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, mapping_file)
        return True
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[WARN] Failed to update mapping file {mapping_file}: {exc}\n")
        return False


def _normalize_optional_str(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _normalize_nonempty_str(value: str | None, *, flag_name: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"--{flag_name} cannot be empty.")
    return stripped


def _validate_deadline_seconds(deadline_seconds: float | None) -> None:
    if deadline_seconds is None:
        return
    if not (deadline_seconds > 0):
        raise ValueError("--deadline-seconds must be > 0 when provided.")


def _resolve_codex_cmd(raw: str | None) -> list[str]:
    candidate = raw if (isinstance(raw, str) and raw.strip()) else os.environ.get("CODEX_CMD", "")
    if isinstance(candidate, str) and candidate.strip():
        parts = shlex.split(candidate)
        if not parts:
            raise ValueError("Empty --codex-cmd/CODEX_CMD after parsing.")
        return parts
    return ["codex"]


def _resolve_codex_wrapper(raw: str | None) -> str:
    candidate = raw if (isinstance(raw, str) and raw.strip()) else os.environ.get("CODEX_CMD", "")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return "codex"


def _resolve_codex_shell(raw: str | None) -> str | None:
    candidate = raw if (isinstance(raw, str) and raw.strip()) else os.environ.get("CODEX_SHELL", "")
    if not isinstance(candidate, str):
        return None
    stripped = candidate.strip().lower()
    if not stripped:
        return None
    if stripped not in {"posix", "powershell"}:
        raise ValueError("--codex-shell/CODEX_SHELL must be one of: posix, powershell.")
    return stripped


def _resolve_codex_shell_cmd(raw: str | None, *, shell_kind: str | None) -> list[str] | None:
    candidate = raw if (isinstance(raw, str) and raw.strip()) else os.environ.get("CODEX_SHELL_CMD", "")
    if isinstance(candidate, str) and candidate.strip():
        parts = shlex.split(candidate)
        if not parts:
            raise ValueError("Empty --codex-shell-cmd/CODEX_SHELL_CMD after parsing.")
        return parts
    if shell_kind == "posix":
        return ["bash", "-lc"]
    if shell_kind == "powershell":
        return ["pwsh", "-NoLogo", "-Command"]
    return None


def _resolve_codex_shell_init(raw: str | None) -> str | None:
    candidate = raw if (isinstance(raw, str) and raw.strip()) else os.environ.get("CODEX_SHELL_INIT", "")
    if not isinstance(candidate, str):
        return None
    stripped = candidate.strip()
    return stripped if stripped else None


def _build_codex_argv(
    wrapper: str,
    *,
    shell_kind: str | None,
    shell_cmd: list[str] | None,
    shell_init: str | None,
    args: list[str],
) -> list[str]:
    if shell_kind is None:
        parts = shlex.split(wrapper)
        if not parts:
            raise ValueError("Empty codex wrapper command after parsing.")
        return parts + args

    if shell_cmd is None:
        raise ValueError("Shell launch mode requires a resolved shell command.")

    if shell_kind == "posix":
        script = f'{wrapper} "$@"'
        if shell_init:
            script = f"{shell_init}\n{script}"
        return shell_cmd + [script, "__codex_wrapper__", *args]

    if shell_kind == "powershell":
        script = f"& {{ {wrapper} @args }}"
        if shell_init:
            script = f"& {{ {shell_init}; {wrapper} @args }}"
        return shell_cmd + [script, *args]

    raise ValueError(f"Unsupported shell kind: {shell_kind}")


def _compose_prompt(prompt: str, role_definition_text: str | None = None) -> str:
    if not isinstance(role_definition_text, str) or not role_definition_text.strip():
        return prompt
    role_text = role_definition_text.strip()
    return (
        "Follow this role definition for this request and the surrounding session context unless the user overrides it.\n\n"
        f"{role_text}\n\n"
        "User request:\n"
        f"{prompt}"
    )


def _extract_thread_id(events: list[Any]) -> str | None:
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id
    return None


def _extract_final_message(events: list[Any]) -> str | None:
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str):
            return text
    return None


def _build_common_exec_args(ns: argparse.Namespace) -> list[str]:
    args: list[str] = []
    for item in getattr(ns, "config_overrides", []) or []:
        args.extend(["--config", item])
    for item in getattr(ns, "enable_features", []) or []:
        args.extend(["--enable", item])
    for item in getattr(ns, "disable_features", []) or []:
        args.extend(["--disable", item])
    for item in getattr(ns, "images", []) or []:
        args.extend(["--image", item])

    model = _normalize_nonempty_str(getattr(ns, "model", None), flag_name="model")
    if model:
        args.extend(["--model", model])
    if getattr(ns, "full_auto", False):
        args.append("--full-auto")
    if getattr(ns, "dangerously_bypass_approvals_and_sandbox", False):
        args.append("--dangerously-bypass-approvals-and-sandbox")
    if getattr(ns, "skip_git_repo_check", False):
        args.append("--skip-git-repo-check")
    return args


def _run_codex_jsonl(
    codex_cmd: list[str],
    args: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    deadline_seconds: float | None = None,
    stream_stdout: bool = False,
    output_last_message_file: Path | None = None,
) -> CodexRunResult:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    try:
        proc = subprocess.Popen(
            codex_cmd + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
    except FileNotFoundError as exc:
        cmd_display = " ".join(codex_cmd) if codex_cmd else "<empty>"
        raise RuntimeError(
            f"Codex command not found: {cmd_display}. Provide --codex-cmd/--codex-shell as needed, "
            "or ensure the wrapper command is available in the chosen shell."
        ) from exc

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

    killer_thread = threading.Thread(target=_deadline_killer, daemon=True)
    killer_thread.start()

    events: list[Any] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            event = {"type": "raw_line", "raw": stripped}
        events.append(event)
        if stream_stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

    rc = proc.wait()
    stop_deadline.set()
    stderr_text = (proc.stderr.read() if proc.stderr else "").strip()

    if deadline_hit.is_set():
        raise RuntimeError(f"codex exceeded deadline ({deadline_seconds}s) and was terminated.")
    if rc != 0:
        raise RuntimeError(f"codex failed rc={rc}: {stderr_text}")

    final_message: str | None = None
    if output_last_message_file is not None and output_last_message_file.exists():
        final_message = output_last_message_file.read_text(encoding="utf-8")
    if final_message is None:
        final_message = _extract_final_message(events)

    return CodexRunResult(
        thread_id=_extract_thread_id(events),
        events=events,
        stderr=stderr_text,
        final_message=final_message,
    )


def _mktemp_path(*, prefix: str, suffix: str) -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(path)


def create_session(
    *,
    alias: str,
    init_prompt: str,
    codex_cmd: list[str],
    role_definition_text: str | None = None,
    common_exec_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    deadline_seconds: float | None = None,
) -> SessionInfo:
    prompt = _compose_prompt(init_prompt, role_definition_text)

    output_file = _mktemp_path(prefix="codex-persist-create-", suffix=".txt")
    try:
        args = ["exec", "--json", "-o", str(output_file), prompt]
        if common_exec_args:
            args[1:1] = common_exec_args
        result = _run_codex_jsonl(
            codex_cmd,
            args,
            extra_env=extra_env,
            deadline_seconds=deadline_seconds,
            output_last_message_file=output_file,
        )
    finally:
        try:
            output_file.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    thread_id = result.thread_id
    if not isinstance(thread_id, str) or not thread_id:
        raise RuntimeError("Missing `thread_id` in `codex exec --json` output.")

    created_at = datetime.now(timezone.utc).isoformat()
    workspace_dir = str(Path.cwd().resolve())
    return SessionInfo(
        alias=alias,
        thread_id=thread_id,
        created_at=created_at,
        workspace_dir=workspace_dir,
    )


def resolve_session(
    *,
    thread_id: str | None,
    session_name: str | None,
    mapping_file: Path | None,
) -> ResolvedSession:
    if isinstance(thread_id, str) and thread_id.strip():
        stripped_id = thread_id.strip()
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
                        if entry.get("thread_id") != stripped_id:
                            continue
                        last_model = entry.get("last_model")
                        if not isinstance(last_model, str) or not last_model:
                            last_model = None
                        return ResolvedSession(
                            thread_id=stripped_id,
                            alias=alias_name,
                            mapping_file=str(mapping_path),
                            last_model=last_model,
                            source="mapping_file_by_thread_id",
                        )
            except Exception:  # noqa: BLE001
                pass

        return ResolvedSession(
            thread_id=stripped_id,
            alias=None,
            mapping_file=None,
            last_model=None,
            source="thread_id",
        )

    chosen_name = _normalize_optional_str(session_name)
    if not chosen_name:
        raise ValueError("Missing session selector: provide --thread-id/--session-id or --session-name.")

    workspace_dir = str(Path.cwd().resolve())
    mapping_path = mapping_file or _default_mapping_file(workspace_dir)
    data = _load_mapping_json(mapping_path)
    aliases = data.get("aliases")
    if not isinstance(aliases, dict):
        aliases = {}

    entry = aliases.get(chosen_name)
    if not isinstance(entry, dict):
        entry = {}
    resolved_id = entry.get("thread_id")
    if not isinstance(resolved_id, str) or not resolved_id:
        raise KeyError(
            f"Session name '{chosen_name}' not found in mapping file: {mapping_path}. "
            "Create it first with the creation stage."
        )

    last_model = entry.get("last_model")
    if not isinstance(last_model, str) or not last_model:
        last_model = None

    return ResolvedSession(
        thread_id=resolved_id,
        alias=chosen_name,
        mapping_file=str(mapping_path),
        last_model=last_model,
        source="mapping_file",
    )


def _read_role_definition(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.suffix.lower() != ".md":
        raise ValueError("--role-definition-md must point to a .md file.")
    if not path.exists():
        raise FileNotFoundError(f"Role definition file not found: {path}")
    return _read_text_file(path)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="invoke_persist.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--thread-id", dest="thread_id", help="Resume a specific Codex thread_id.")
    selector.add_argument("--session-id", dest="thread_id", help=argparse.SUPPRESS)
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
        "--codex-cmd",
        default=None,
        help=(
            "Optional codex-compatible wrapper command. In shell-launch mode, "
            "this is usually the wrapper name to invoke after shell initialization "
            "(default: CODEX_CMD env var or `codex`)."
        ),
    )
    common_call.add_argument(
        "--codex-shell",
        choices=["posix", "powershell"],
        default=None,
        help=(
            "Use shell-launch mode for alias/function wrappers. "
            "When set, --codex-cmd is invoked through the selected shell instead of "
            "being treated as a direct executable."
        ),
    )
    common_call.add_argument(
        "--codex-shell-cmd",
        default=None,
        help=(
            "Optional shell launcher command for --codex-shell, for example "
            "'bash -lc' or 'pwsh -NoLogo -Command'."
        ),
    )
    common_call.add_argument(
        "--codex-shell-init",
        default=None,
        help=(
            "Optional shell initialization snippet to run before invoking the wrapper, "
            "for example 'shopt -s expand_aliases; source ~/.bashrc' or '. $PROFILE'."
        ),
    )
    common_call.add_argument("--model", default=None, help="Optional model to request and persist per alias.")
    common_call.add_argument(
        "--config",
        dest="config_overrides",
        action="append",
        default=[],
        help="Repeatable `codex --config key=value` override.",
    )
    common_call.add_argument(
        "--enable",
        dest="enable_features",
        action="append",
        default=[],
        help="Repeatable `codex --enable FEATURE` flag.",
    )
    common_call.add_argument(
        "--disable",
        dest="disable_features",
        action="append",
        default=[],
        help="Repeatable `codex --disable FEATURE` flag.",
    )
    common_call.add_argument(
        "--image",
        dest="images",
        action="append",
        default=[],
        help="Repeatable image path to attach to the prompt.",
    )
    common_call.add_argument("--full-auto", action="store_true")
    common_call.add_argument("--dangerously-bypass-approvals-and-sandbox", action="store_true")
    common_call.add_argument("--skip-git-repo-check", action="store_true")
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

    create = sub.add_parser("create-session", parents=[selector, common_call], help="Create a new Codex session.")
    create.add_argument(
        "--role-definition-md",
        type=Path,
        default=None,
        help="Optional Markdown file whose contents are prepended to the initial prompt.",
    )
    create.add_argument(
        "--init-prompt",
        default=None,
        help="Optional first prompt; defaults to a minimal init prompt that replies 'OK'.",
    )
    create.add_argument(
        "--fake-thread-id",
        default=None,
        help="Testing helper: skip calling `codex` and use this thread_id.",
    )
    create.add_argument("--print-thread-id", dest="print_thread_id", action="store_true")
    create.add_argument("--print-session-id", dest="print_thread_id", action="store_true", help=argparse.SUPPRESS)
    create.add_argument("--print-mapping-json", action="store_true")

    resolve = sub.add_parser("resolve", parents=[selector], help="Resolve alias/name to thread_id.")
    resolve.add_argument("--print-thread-id", dest="print_thread_id", action="store_true")
    resolve.add_argument("--print-session-id", dest="print_thread_id", action="store_true", help=argparse.SUPPRESS)

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
    list_sessions.add_argument("--print-aliases", action="store_true", help="Print only alias names, one per line.")

    delete_session = sub.add_parser(
        "delete-session",
        parents=[workspace_ops],
        help="Delete a saved session entry from the workspace mapping file without calling Codex.",
    )
    delete_session.add_argument("--thread-id", dest="thread_id", help="Delete entries matching this thread_id.")
    delete_session.add_argument("--session-id", dest="thread_id", help=argparse.SUPPRESS)
    delete_session.add_argument("--session-name", help="Session name to delete from the mapping file.")
    delete_session.add_argument("--session-alias", dest="session_name", help=argparse.SUPPRESS)
    delete_session.add_argument("--alias", dest="session_name", help=argparse.SUPPRESS)
    delete_session.add_argument(
        "--print-mapping-json",
        action="store_true",
        help="Include the mapping JSON in the output, even if unchanged.",
    )

    sub.add_parser(
        "delete-all-sessions",
        parents=[workspace_ops],
        help="Delete the workspace mapping file entirely without calling Codex.",
    )

    resume_json = sub.add_parser(
        "resume-json",
        parents=[selector, common_call],
        help="Resume a Codex session and print a final JSON summary object.",
    )
    resume_json.add_argument("--prompt", required=True)
    resume_json.add_argument(
        "--role-definition-md",
        type=Path,
        default=None,
        help="Optional Markdown file whose contents are prepended to the resumed prompt.",
    )
    resume_json.add_argument("--print-result", action="store_true")
    resume_json.add_argument("--print-thread-id", dest="print_thread_id", action="store_true")
    resume_json.add_argument("--print-session-id", dest="print_thread_id", action="store_true", help=argparse.SUPPRESS)

    resume_stream = sub.add_parser(
        "resume-stream",
        parents=[selector, common_call],
        help="Resume a Codex session with streaming JSONL output.",
    )
    resume_stream.add_argument("--prompt", required=True)
    resume_stream.add_argument(
        "--role-definition-md",
        type=Path,
        default=None,
        help="Optional Markdown file whose contents are prepended to the resumed prompt.",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)

    if ns.cmd == "create-session":
        session_name = _normalize_optional_str(getattr(ns, "session_name", None))
        if not session_name:
            raise ValueError("Missing session name: provide --session-name.")

        extra_env = _read_env_vars_file(ns.env_file) if ns.env_file else None
        shell_kind = _resolve_codex_shell(ns.codex_shell)
        shell_cmd = _resolve_codex_shell_cmd(ns.codex_shell_cmd, shell_kind=shell_kind)
        shell_init = _resolve_codex_shell_init(ns.codex_shell_init)
        codex_cmd = _build_codex_argv(
            _resolve_codex_wrapper(ns.codex_cmd),
            shell_kind=shell_kind,
            shell_cmd=shell_cmd,
            shell_init=shell_init,
            args=[],
        )[:]
        _validate_deadline_seconds(ns.deadline_seconds)
        common_exec_args = _build_common_exec_args(ns)
        role_definition_text = _read_role_definition(ns.role_definition_md)
        model = _normalize_nonempty_str(getattr(ns, "model", None), flag_name="model")

        if ns.fake_thread_id:
            created_at = datetime.now(timezone.utc).isoformat()
            workspace_dir = str(Path.cwd().resolve())
            info = SessionInfo(
                alias=session_name,
                thread_id=str(ns.fake_thread_id),
                created_at=created_at,
                workspace_dir=workspace_dir,
            )
        else:
            init_prompt = ns.init_prompt or (
                f"Initialize a new persistent Codex session named '{session_name}'. Reply exactly: OK"
            )
            info = create_session(
                alias=session_name,
                init_prompt=init_prompt,
                codex_cmd=codex_cmd,
                role_definition_text=role_definition_text,
                common_exec_args=common_exec_args,
                extra_env=extra_env,
                deadline_seconds=ns.deadline_seconds,
            )

        mapping_file = ns.mapping_file or _default_mapping_file(info.workspace_dir)
        wrote_mapping = _write_mapping_file(
            mapping_file,
            alias=info.alias,
            thread_id=info.thread_id,
            workspace_dir=info.workspace_dir,
            created_at=info.created_at,
            last_model=model,
        )

        if ns.print_thread_id:
            sys.stdout.write(info.thread_id + "\n")

        if ns.print_mapping_json:
            sys.stdout.write(
                json.dumps(
                    {
                        "alias": info.alias,
                        "thread_id": info.thread_id,
                        "workspace_dir": info.workspace_dir,
                        "created_at": info.created_at,
                        "last_model": model,
                        "mapping_file": str(mapping_file),
                        "mapping_file_written": wrote_mapping,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if not ns.print_thread_id and not ns.print_mapping_json:
            sys.stdout.write(info.thread_id + "\n")
        return 0

    if ns.cmd == "resolve":
        resolved = resolve_session(
            thread_id=ns.thread_id,
            session_name=ns.session_name,
            mapping_file=ns.mapping_file,
        )
        if ns.print_thread_id:
            sys.stdout.write(resolved.thread_id + "\n")
            return 0
        sys.stdout.write(
            json.dumps(
                {
                    "thread_id": resolved.thread_id,
                    "alias": resolved.alias,
                    "mapping_file": resolved.mapping_file,
                    "last_model": resolved.last_model,
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
        requested_thread_id = ns.thread_id.strip() if isinstance(ns.thread_id, str) and ns.thread_id.strip() else None
        if not requested_session_name and not requested_thread_id:
            raise ValueError("Missing delete selector: provide --session-name or --thread-id/--session-id.")

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
            except Exception as exc:  # noqa: BLE001
                backup = mapping_path.with_name(mapping_path.name + f".bad.{os.getpid()}")
                try:
                    if mapping_path.exists():
                        mapping_path.replace(backup)
                    sys.stderr.write(
                        f"[WARN] Mapping file was invalid JSON; moved to {backup} and recreated: {exc}\n"
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
                assert requested_thread_id is not None
                for alias_name in sorted(a for a in list(aliases.keys()) if isinstance(a, str)):
                    entry = aliases.get(alias_name)
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("thread_id") != requested_thread_id:
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
                    "requested_thread_id": requested_thread_id,
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
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to delete mapping file {mapping_path}: {exc}") from exc

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
    shell_kind = _resolve_codex_shell(ns.codex_shell)
    shell_cmd = _resolve_codex_shell_cmd(ns.codex_shell_cmd, shell_kind=shell_kind)
    shell_init = _resolve_codex_shell_init(ns.codex_shell_init)
    codex_cmd = _build_codex_argv(
        _resolve_codex_wrapper(ns.codex_cmd),
        shell_kind=shell_kind,
        shell_cmd=shell_cmd,
        shell_init=shell_init,
        args=[],
    )[:]
    _validate_deadline_seconds(ns.deadline_seconds)
    common_exec_args = _build_common_exec_args(ns)
    explicit_model = _normalize_nonempty_str(getattr(ns, "model", None), flag_name="model")

    resolved = resolve_session(
        thread_id=ns.thread_id,
        session_name=ns.session_name,
        mapping_file=ns.mapping_file,
    )
    effective_model = explicit_model or resolved.last_model

    if ns.cmd == "resume-json":
        role_definition_text = _read_role_definition(ns.role_definition_md)
        prompt = _compose_prompt(ns.prompt, role_definition_text)
        output_file = _mktemp_path(prefix="codex-persist-resume-", suffix=".txt")
        try:
            args = ["exec", "resume", resolved.thread_id, "--json", "-o", str(output_file), prompt]
            if common_exec_args:
                args[2:2] = common_exec_args
            if effective_model and explicit_model is None:
                args[2:2] = ["--model", effective_model]
            result = _run_codex_jsonl(
                codex_cmd,
                args,
                extra_env=extra_env,
                deadline_seconds=ns.deadline_seconds,
                output_last_message_file=output_file,
            )
        finally:
            try:
                output_file.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

        if resolved.alias and resolved.mapping_file:
            _update_last_model(
                Path(resolved.mapping_file),
                alias=resolved.alias,
                thread_id=resolved.thread_id,
                workspace_dir=str(Path.cwd().resolve()),
                last_model=effective_model,
            )

        final_thread_id = result.thread_id or resolved.thread_id
        final_message = result.final_message or ""

        if ns.print_result:
            sys.stdout.write(final_message + ("\n" if not final_message.endswith("\n") else ""))
        if ns.print_thread_id:
            sys.stdout.write(final_thread_id + "\n")
        if not ns.print_result and not ns.print_thread_id:
            sys.stdout.write(
                json.dumps(
                    {
                        "thread_id": final_thread_id,
                        "alias": resolved.alias,
                        "mapping_file": resolved.mapping_file,
                        "last_model": effective_model,
                        "source": resolved.source,
                        "event_count": len(result.events),
                        "final_message": final_message,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return 0

    if ns.cmd == "resume-stream":
        role_definition_text = _read_role_definition(ns.role_definition_md)
        prompt = _compose_prompt(ns.prompt, role_definition_text)
        args = ["exec", "resume", resolved.thread_id, "--json", prompt]
        if common_exec_args:
            args[2:2] = common_exec_args
        if effective_model and explicit_model is None:
            args[2:2] = ["--model", effective_model]
        _run_codex_jsonl(
            codex_cmd,
            args,
            extra_env=extra_env,
            deadline_seconds=ns.deadline_seconds,
            stream_stdout=True,
        )
        if resolved.alias and resolved.mapping_file:
            _update_last_model(
                Path(resolved.mapping_file),
                alias=resolved.alias,
                thread_id=resolved.thread_id,
                workspace_dir=str(Path.cwd().resolve()),
                last_model=effective_model,
            )
        return 0

    raise AssertionError(f"Unhandled cmd: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
