---
name: test-and-log
description: Test a target (script, demo, pipeline, CLI command, integration) without modifying any source code, then write a structured log of the process, outcomes, anomalies, and issues. Use when the user says "test X and log", "run X and document findings", or "try X without changing code". Default log location is context/logs/TIMESTAMP-task-name/TIMESTAMP.md.
---

# Test and Log

Run a test target, capture every step's outcome, and write a structured log. **Never modify source code.**

**This skill must be invoked explicitly by name** (e.g., `$test-and-log`). It will not trigger automatically from context.

Example prompts:

- `$test-and-log scripts/demo/cao-interactive-full-pipeline-demo`
- `$test-and-log — run the cao-claude-session demo and log any issues to context/logs`
- `$test-and-log pixi run test-runtime, log results to tmp/test-logs`
- `$test-and-log the gemini-headless-session demo without touching any code`

## References

- **Log template**: See [references/log-template.md](references/log-template.md) for the exact log structure to fill in.

## Workflow

### 1. Clarify parameters

Collect before starting:

| Parameter | Default | Notes |
|---|---|---|
| **Subject** | (required) | What to test — command, script, demo path, etc. |
| **Log dir** | `context/logs/` | Root directory for log output |
| **Task name** | Derived from subject | Kebab-case slug for directory name |

Derive `<ts>` from the moment the run starts: `YYYYMMDD-HHMMSS` (UTC).
Full log path: `<log-dir>/<ts>-<task-name>/<ts>.md`

### 2. Check prerequisites

Before running anything, verify all stated prerequisites of the subject. Record each check result in the log's **Environment** section. If a prerequisite is missing, record it and decide (with the user if interactive) whether to abort or proceed with a caveat.

### 3. Run steps

Execute each step of the subject in sequence. For each step:

- Capture the full command run
- Capture exit code and relevant output excerpts
- Note any warnings, anomalies, or unexpected outputs — **even when exit code is 0**
- Record timing where meaningful

### 4. Identify issues and anomalies

After all steps complete, review the captured outputs for:

- Hard failures (non-zero exit codes, exceptions, error messages)
- Soft anomalies (wrong output, stale data, unexpected warnings, parser drift, etc.)
- Partial successes (pipeline succeeded structurally but produced incorrect results)

For each issue, record: what happened, where in the run it occurred, relevant output excerpt, and interpretation if known.

### 5. Write the log

Create the log file at the derived path. Fill in the template from [references/log-template.md](references/log-template.md). Key rules:

- Record actual command output verbatim (truncate only if very long — mark truncations with `[…]`)
- Do not editorialize in the per-step sections; save interpretation for the **Anomalies** and **Summary** sections
- Do not include speculation about fixes unless explicitly asked

### 6. Do not modify code

This skill is strictly read-and-run. If a failure or anomaly is discovered, document it in the log and stop. Do not attempt to fix, patch, or work around issues in source files.
