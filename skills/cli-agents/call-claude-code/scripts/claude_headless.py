#!/usr/bin/env python3
"""
Minimal wrappers for calling Claude Code programmatically via the `claude` CLI.

This script is intentionally dependency-free (stdlib only) and focuses on:
- one-shot JSON output
- streaming JSONL output
- resuming a session via `--resume <session_id>`

Examples:
  python3 scripts/claude_headless.py one-shot --prompt "Hello" --print-result
  python3 scripts/claude_headless.py stream --prompt "Explain recursion" --include-partials
  python3 scripts/claude_headless.py one-shot --prompt "Start" --print-session-id
  python3 scripts/claude_headless.py resume --session-id <id> --prompt "Continue" --print-result
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ClaudeJsonResult:
    """Parsed result from `claude --output-format json`."""

    result: Any
    session_id: str | None
    raw: dict[str, Any]


def _read_env_vars_file(path: Path) -> dict[str, str]:
    """Read a simple KEY=VALUE env file (comments and blanks allowed).

    Parameters
    ----------
    path
        Path to an env vars file (e.g. `.../env/vars.env`).

    Returns
    -------
    dict[str, str]
        Parsed environment variables.
    """

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


def _run_claude(args: list[str], *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(args, text=True, capture_output=True, env=env, check=False)


def claude_one_shot_json(
    prompt: str,
    *,
    continue_latest: bool = False,
    resume_session_id: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> ClaudeJsonResult:
    """Run a single `claude -p` prompt and parse JSON output.

    Parameters
    ----------
    prompt
        Prompt string passed to `claude -p`.
    continue_latest
        If true, pass `--continue` to use the latest conversation.
    resume_session_id
        If set, pass `--resume <session_id>` to resume a specific session.
    extra_env
        Extra environment variables to use for the subprocess.

    Returns
    -------
    ClaudeJsonResult
        Parsed `result` and optional `session_id` plus raw JSON dict.
    """

    cmd = ["claude", "-p", prompt, "--output-format", "json"]
    if continue_latest:
        cmd.append("--continue")
    if resume_session_id:
        cmd.extend(["--resume", resume_session_id])

    proc = _run_claude(cmd, extra_env=extra_env)
    if proc.returncode != 0:
        raise RuntimeError(f"claude failed rc={proc.returncode}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    return ClaudeJsonResult(
        result=data.get("result"),
        session_id=data.get("session_id"),
        raw=data,
    )


def claude_stream_jsonl(
    prompt: str,
    *,
    include_partial_messages: bool = False,
    verbose: bool = False,
    continue_latest: bool = False,
    resume_session_id: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> Iterable[dict[str, Any]]:
    """Run `claude --output-format stream-json` and yield JSON objects per line.

    Notes
    -----
    - Treat `stdout` as the machine channel and `stderr` as diagnostics.
    - The event schema may evolve; do not hard-code exact keys unless needed.
    """

    cmd = ["claude", "-p", prompt, "--output-format", "stream-json"]
    if verbose:
        cmd.append("--verbose")
    if include_partial_messages:
        cmd.append("--include-partial-messages")
    if continue_latest:
        cmd.append("--continue")
    if resume_session_id:
        cmd.extend(["--resume", resume_session_id])

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
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)

    rc = proc.wait()
    if rc != 0:
        err = (proc.stderr.read() if proc.stderr else "").strip()
        raise RuntimeError(f"claude failed rc={rc}: {err}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="claude_headless.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--prompt", required=True)
    common.add_argument("--continue-latest", action="store_true")
    common.add_argument("--resume", dest="resume_session_id")
    common.add_argument("--env-file", type=Path, help="Optional KEY=VALUE env file to load.")

    one = sub.add_parser("one-shot", parents=[common], help="Run `claude -p` with JSON output.")
    one.add_argument("--print-result", action="store_true")
    one.add_argument("--print-session-id", action="store_true")

    res = sub.add_parser("resume", parents=[common], help="Run `claude -p` resuming a session.")
    res.add_argument("--session-id", required=True)
    res.add_argument("--print-result", action="store_true")

    stream = sub.add_parser("stream", parents=[common], help="Run `claude -p` with streaming JSONL output.")
    stream.add_argument("--include-partials", action="store_true")
    stream.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)
    extra_env = _read_env_vars_file(ns.env_file) if ns.env_file else None

    if ns.cmd == "one-shot":
        out = claude_one_shot_json(
            ns.prompt,
            continue_latest=ns.continue_latest,
            resume_session_id=ns.resume_session_id,
            extra_env=extra_env,
        )
        if ns.print_result:
            sys.stdout.write(f"{out.result}\n")
        if ns.print_session_id:
            sys.stdout.write(f"{out.session_id or ''}\n")
        return 0

    if ns.cmd == "resume":
        out = claude_one_shot_json(
            ns.prompt,
            continue_latest=ns.continue_latest,
            resume_session_id=ns.session_id,
            extra_env=extra_env,
        )
        if ns.print_result:
            sys.stdout.write(f"{out.result}\n")
        return 0

    if ns.cmd == "stream":
        for event in claude_stream_jsonl(
            ns.prompt,
            include_partial_messages=ns.include_partials,
            verbose=ns.verbose,
            continue_latest=ns.continue_latest,
            resume_session_id=ns.resume_session_id,
            extra_env=extra_env,
        ):
            sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        return 0

    raise AssertionError(f"Unhandled cmd: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

