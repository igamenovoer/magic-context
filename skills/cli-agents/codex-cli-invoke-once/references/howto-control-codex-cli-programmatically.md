# How to Control Codex CLI Programmatically (CLI / Headless)

Drive Codex from another program by spawning `codex exec` and parsing JSONL or by capturing the final assistant message with `-o`.

This guide assumes headless invocations plus session continuation, not the interactive TUI.

Examples below were locally inspected against `codex-cli 0.115.0` on 2026-03-19. If the CLI changes, re-check `codex exec --help` and `codex exec resume --help`.

## Wrapper commands

Users may provide Codex wrappers with any name. They are often named `codex-<something>`, but they do not need to be.

Executable wrappers can be substituted directly:

```bash
codex-dev exec --json -o /tmp/codex-last.txt "Summarize this project"
```

For POSIX shell aliases or functions, invoke them through a shell that defines them:

```bash
bash -lc 'shopt -s expand_aliases; source ~/.bashrc; codex-work exec --json -o /tmp/codex-last.txt "Summarize this project"'
```

For PowerShell functions, invoke them through PowerShell:

```powershell
pwsh -NoLogo -Command '. $PROFILE; codex-work exec --json -o "$env:TEMP\codex-last.txt" "Summarize this project"'
```

If the wrapper is an alias or function, do not pass it to `subprocess.Popen([...])` as if it were a standalone executable. Wrap it in the shell that defines it.

## One-shot prompt with final output capture

```bash
codex exec -o /tmp/codex-last.txt "Summarize this project"
cat /tmp/codex-last.txt
```

Typical use: treat `/tmp/codex-last.txt` as the final assistant message for downstream automation.

If you want to read the prompt from stdin, pass `-` as the prompt argument:

```bash
printf '%s\n' "Summarize this project" | codex exec --json -o /tmp/codex-last.txt -
```

## Streaming machine output (JSONL)

```bash
codex exec --json -o /tmp/codex-last.txt "Explain recursion"
```

A minimal observed event stream looks like this:

```json
{"type":"thread.started","thread_id":"019d063a-8371-7ba3-9710-7388fc43c8df"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"OK"}}
{"type":"turn.completed","usage":{"input_tokens":16214,"cached_input_tokens":12288,"output_tokens":39}}
```

Typical use:

1. Capture `thread_id` from `thread.started`.
2. Watch `item.completed` events for final agent messages.
3. Prefer `-o` for the canonical final answer when you need a stable file output.

Treat stdout as the machine channel and stderr as diagnostics.

## Heartbeat monitoring and deadlines

When you run with `--json`, each stdout line can be treated as a heartbeat.

Recommended policy:

1. Record the timestamp of every JSONL line received from stdout.
2. Optionally emit a compact caller-facing heartbeat such as `still running` every N seconds without replaying the full event stream.
3. Do not kill the process while heartbeats are still arriving.
4. Only enforce an overall deadline when the caller explicitly asked for one.
5. Only treat silence as a stall if your automation deliberately defines an inactivity threshold.

This keeps long-running Codex calls observable without cutting them off prematurely.

## Maintaining conversational state across invocations

Resume by thread id:

```bash
log="$(mktemp)"
codex exec --json -o /tmp/first.txt "Start a review of this repository" | tee "$log"
thread_id="$(jq -r 'select(.type=="thread.started").thread_id' "$log" | head -n1)"
codex exec resume "$thread_id" --json -o /tmp/second.txt "Continue that review with a focus on tests"
```

Resume the latest recorded session:

```bash
codex exec resume --last --json -o /tmp/latest.txt "Continue the latest session"
```

Notes:

1. `codex exec` persists session state by default.
2. `--ephemeral` disables durable session files, so do not use it if you plan to resume later.
3. For automation, prefer `codex exec resume` over interactive `codex resume`.

## Structured final output with JSON Schema

Use `--output-schema` when a downstream program expects a specific JSON shape.

```bash
codex exec --output-schema /path/to/schema.json -o /tmp/result.json "Return the requested fields only"
```

The final assistant message written by `-o` should conform to the provided schema.

## Review mode for automation

For scripted repo reviews, prefer `codex exec review` instead of top-level `codex review` because `exec review` exposes `--json`, `-o`, and `--ephemeral`.

```bash
codex exec review --uncommitted --json -o /tmp/review.txt
codex exec review --base main --json -o /tmp/review.txt "Focus on correctness and regressions"
```

## Option differences and common pitfalls

1. `codex exec` does not accept `-a` or `--ask-for-approval`. Passing `-a` currently fails with `unexpected argument '-a' found`.
2. `codex exec` takes the prompt as a positional argument. There is no Claude-style `-p`.
3. `codex exec` uses `--json` for JSONL events. There is no `--output-format json|stream-json`.
4. `codex exec resume` supports fewer flags than plain `exec`. If you need custom behavior across both commands, prefer a shared config/profile or keep the resumed call simple.
5. If you need a different working root, use `-C` or `--cd`.
6. If you need attached images on the initial prompt, use `-i` or `--image`.
7. If the user provides a wrapper, use the exact wrapper name or shell snippet they gave you. Do not assume the wrapper starts with `codex-`.

## Python subprocess example (JSONL + heartbeat monitoring)

```python
import json
import pathlib
import subprocess
import tempfile
import time
from typing import Callable


def run_codex_stream(
    prompt: str,
    *,
    heartbeat_interval_seconds: float = 15.0,
    inactivity_timeout_seconds: float | None = None,
    overall_deadline_seconds: float | None = None,
    on_heartbeat: Callable[[str], None] | None = print,
) -> tuple[str | None, list[dict], str]:
    out_file = tempfile.NamedTemporaryFile(delete=False)
    out_path = pathlib.Path(out_file.name)
    out_file.close()

    proc = subprocess.Popen(
        ["codex", "exec", "--json", "-o", str(out_path), prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    events: list[dict] = []
    thread_id: str | None = None
    started_at = time.monotonic()
    last_event_at = started_at
    last_heartbeat_log_at = started_at

    while True:
        line = proc.stdout.readline()
        now = time.monotonic()

        if line:
            event = json.loads(line)
            events.append(event)
            last_event_at = now

            if event.get("type") == "thread.started":
                thread_id = event.get("thread_id")

            continue

        if proc.poll() is not None:
            break

        if (
            heartbeat_interval_seconds > 0
            and on_heartbeat is not None
            and now - last_heartbeat_log_at >= heartbeat_interval_seconds
        ):
            on_heartbeat(
                f"codex still running; last event {now - last_event_at:.1f}s ago"
            )
            last_heartbeat_log_at = now

        if (
            inactivity_timeout_seconds is not None
            and now - last_event_at > inactivity_timeout_seconds
        ):
            proc.kill()
            raise TimeoutError(
                f"codex exec produced no heartbeat for {inactivity_timeout_seconds}s"
            )

        if (
            overall_deadline_seconds is not None
            and now - started_at > overall_deadline_seconds
        ):
            proc.kill()
            raise TimeoutError(
                f"codex exec exceeded overall deadline of {overall_deadline_seconds}s"
            )

        time.sleep(0.2)

    rc = proc.wait()
    if rc != 0:
        err = (proc.stderr.read() if proc.stderr else "").strip()
        raise RuntimeError(f"codex exec failed rc={rc}: {err}")

    final_message = out_path.read_text()
    return thread_id, events, final_message
```

Notes:

1. The example separates heartbeat monitoring from hard deadlines.
2. Leave `inactivity_timeout_seconds=None` unless your caller truly wants stall detection.
3. Leave `overall_deadline_seconds=None` unless the user asked for a hard deadline.
4. Prefer using `-o` for the final answer even when you also parse JSONL events.

## Sources

- Local CLI help: `codex --help`
- Local CLI help: `codex exec --help`
- Local CLI help: `codex exec resume --help`
- Local CLI help: `codex exec review --help`
- Local CLI version: `codex --version`
