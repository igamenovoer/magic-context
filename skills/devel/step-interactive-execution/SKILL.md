---
name: step-interactive-execution
description: "Prepare and run a user-confirmed interactive execution or test session for a source file, executable, script, demo, or directory. This skill must be invoked manually: use it only when the user explicitly invokes step-interactive-execution by name or clearly asks for this exact manual workflow. Read the target first without pausing, then switch to a strict step-by-step loop where the agent prints the exact CLI command to run next, waits for confirmation such as \"continue\", executes only that approved command, and reports the result before proposing the next step."
---

# Step Interactive Execution

This skill must be invoked explicitly by name. Do not apply it implicitly from context alone.

Run the session in two phases: unattended preparation first, then confirmed execution.

## Preparation Phase

Inspect the target before asking the user to approve any command.

- Identify whether the target is a source file, executable, script, or directory.
- Read the smallest useful set of nearby materials: local README files, runner scripts, tests, inputs, wrappers, config, and obvious entrypoints.
- Use read-only inspection commands freely during this phase.
- Do not pause for approval during preparation.
- Do not start the real test flow or other side-effecting commands during preparation unless the user explicitly asks.

At the end of preparation, summarize only the operational facts needed for the next step:

- what the target appears to do
- how it is likely meant to be exercised
- visible prerequisites or environment constraints
- the most likely next command to run

## Target-Specific Reading Heuristics

### Source File

- Read the file itself first.
- Read nearby callers, entrypoints, tests, or wrappers that define how the file is exercised.
- If it is a runnable script, inspect its usage surface before executing it.

### Executable Or CLI

- Prefer local docs, wrapper scripts, and checked-in examples first.
- Use `--help` or equivalent only when it is cheap and safe.
- Avoid commands that may mutate state just to discover usage.

### Directory

- Read the local operator guidance first.
- For demo or tutorial directories, inspect the runner, input fixtures, expected outputs, cleanup behavior, and any verification helpers.
- Distinguish convenience wrappers from the underlying manual command flow.

## Confirmed Execution Loop

After preparation, handle real execution one step at a time.

1. Derive the single next command that should run.
2. Print that exact command before executing it.
3. Wait for the user's approval signal.
4. Execute only the approved command.
5. Report the result briefly.
6. Propose the next command and wait again.

Treat this loop as the default contract for the session.

## Command Presentation Rules

- Use the exact command that is about to run, not a paraphrase.
- Present the command in a fenced `bash` block.
- Prefer one command per approval step.
- If the user explicitly asks to batch several commands behind one confirmation, print the full batch together and wait once.
- Never execute additional side-effecting commands that were not shown to the user first.
- Preparatory inspection commands do not need user confirmation and do not need to be shown as pending execution.

Use this response shape when requesting approval:

```text
CLI command to execute next:
```

```bash
<exact command>
```

```text
Send `continue` when you want me to run it.
```

After approval, use this response shape:

```text
Executed:
```

```bash
<exact command>
```

```text
<brief result summary>
```

## Failure Handling

- If a command fails, report the failure before proposing anything else.
- Do not hide retries, cleanup, or fallback actions behind the same approval.
- If recovery needs another command, show that command and wait for approval.
- Do not jump to code changes or fixes unless the user explicitly changes the task.

## Boundaries

- Treat this skill as read-and-run by default.
- Do not modify code during the preparation phase.
- Do not widen scope beyond the user's target unless surrounding files are needed to understand or execute the target safely.
- If a safer or more accurate command becomes necessary, show the revised command and wait again.

## Examples

- "Use step-interactive-execution on `scripts/demo/foo`."
- "Use step-interactive-execution for this binary and walk me through the test."
- "Use step-interactive-execution on `src/app/main.py`; read it first, then wait before each command."
