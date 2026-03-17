---
name: do-interactive-test
description: Prepare for and run user-driven interactive testing of a directory the user points to. Use when the user wants the agent to read what is already there first, be prepared, follow step-by-step test instructions, or honor a constrained edit boundary during testing. Handle generic directories, demo/tutorial directories, and OpenSpec change directories differently; for OpenSpec change directories, use openspec CLI commands to gather context instead of assuming a file layout inside the directory. During interactive testing, do not automatically modify the system under test; report issues first, let the developer decide whether to log them or proceed to a fix, and only modify demo-specific code when a fix is explicitly requested. Do not create extra logs unless the developer asks for issue logging or step logging.
---

# Do Interactive Test

Run an interactive test as a guided session. Inspect first, execute only the step the user asks for, log the session when requested, and keep any code changes inside the user-approved boundary.

## Preparation Workflow

1. Identify the target under test and the edit boundary.
2. Classify the target directory before reading deeply.
3. Gather context with the workflow that matches that directory type.
4. Separate intended behavior from implemented behavior.
5. Capture the test constraints in working memory before proceeding.
6. Wait for the user's next instruction and execute one step at a time.

## Directory Classification

Choose the narrowest fitting category:

- **Generic directory**: a normal code or docs directory with no obvious runnable demo wrapper or OpenSpec identity
- **Demo/tutorial directory**: a directory centered on runnable instructions, demo scripts, inputs, expected outputs, or verification helpers
- **OpenSpec change directory**: a directory that appears to correspond to an OpenSpec change; do not assume its internal file layout, and use `openspec` commands first

During preparation, do not edit code. Summarize:

- what the target is intended to do
- what files actually implement it
- what tests exist
- what prerequisites or environment dependencies are visible
- what looks ready to test next

## Context Gathering By Directory Type

### Generic Directory

Inspect the directory itself first.

- List files and subdirectories.
- Read the most relevant entrypoints such as `README`, runnable scripts, test files, or config files.
- Identify how the directory is meant to be exercised.
- Avoid loading unrelated parts of the repository.

### Demo Or Tutorial Directory

Prioritize operator guidance and runnable surfaces.

- Read the local README or run instructions first.
- Read the runner script, verification helper, and expected output contract if present.
- Read the tests that exercise the demo flow if they exist.
- Identify prerequisites, environment variables, output directories, and cleanup behavior.
- Distinguish convenience wrappers from the underlying manual command flow.

### OpenSpec Change Directory

Do not assume files such as `proposal.md` or `design.md` exist just because the target lives under an OpenSpec-looking path.

Use OpenSpec tooling first:

1. Derive the candidate change name from the target directory name.
2. Confirm it through `openspec list --json`.
3. Gather structured change context through:
   - `openspec show --type change --json --no-interactive <change-name>`
   - `openspec status --change <change-name> --json`
4. Use `openspec validate --type change --strict --json --no-interactive <change-name>` when validation state matters to the test.
5. Only after that, open specific files that are directly relevant to the user's requested test step.

When reading files for an OpenSpec change, use the OpenSpec tool output to decide what to inspect next. Do not hard-code or assume the artifact layout inside the directory.

## Logging Rules

Do not assume any repository-local logging directory exists.

By default, create no extra logs at all. The only logging that should happen by default is whatever the system or demo already implements in its own code or runner.

Derive `<demo-slug>` from the target directory name by normalizing it to lowercase hyphen-case.

If the developer says `demo-root=<some-dir>`, use that exact directory as `<demo-root>` for this session.

Otherwise, when this skill uses its default logging layout, resolve `<repo-root>` as the main Git repository root that owns the target under test and define:

- `<demo-root>`: `<repo-root>/.agent-automation/demos/<demo-slug>`

If this skill creates `.agent-automation/demos/`, add `.agent-automation/demos/` to the main repository `.gitignore` so helper-managed demo artifacts stay out of commits. If `.gitignore` already has commented `.agent-automation/demos` entries, treat that as a developer signal not to auto-add the ignore rule.

If the developer specifies `demo-root=<some-dir>`, treat that as a user-chosen location and do not modify `.gitignore` unless the developer explicitly asks.

### Issue Logging

Enable issue logging only when the developer says things like:

- "log the issues"
- "log errors"
- "log problems"
- or equivalent wording

Default issue-log directory:

- `<demo-root>/issues`

If the developer specifies a location directory, use that directory instead of the default issue-log directory.

If the default issue-log directory is used, create `<demo-root>/issues` as needed. When `<demo-root>` is the default helper-managed root and creating that path also creates `.agent-automation/demos/`, add `.agent-automation/demos/` to the main repository `.gitignore` unless commented `.agent-automation/demos` entries are already present.

If the developer specifies a logging directory, do not touch `.gitignore`.

After issue logging is enabled, whenever the interactive run exposes an issue, create one Markdown file in the issue-log directory:

- filename: `<ts>-<what>.md`
- timestamp format: `YYYYMMDD-HHMMSS`
- `<what>`: short hyphen-case summary of the issue

Each issue log should include:

- what problem was hit
- where and when it happened
- the command or step that triggered it, if applicable
- the observed output or error
- whether it appears demo-specific or system-level
- suggested solutions or next actions

### Step Logging

Enable step logging only when the developer says things like:

- "log the steps"
- "log the process"
- or equivalent wording

Default step-log directory:

- `<demo-root>/steps`

If the developer specifies a location directory, use that directory instead of the default step-log directory.

If the default step-log directory is used, create `<demo-root>/steps` as needed. When `<demo-root>` is the default helper-managed root and creating that path also creates `.agent-automation/demos/`, add `.agent-automation/demos/` to the main repository `.gitignore` unless commented `.agent-automation/demos` entries are already present.

If the developer specifies a logging directory, do not touch `.gitignore`.

After step logging is enabled, create one Markdown file per qualifying step:

- filename: `<ts>-<step-slug>.md`
- timestamp format: `YYYYMMDD-HHMMSS`
- `<step-slug>`: short hyphen-case summary of the step

Only log steps that result in code execution or code changes.

- log: shell commands, test runs, demo runs, validation runs, file edits
- do not log: normal chatting, planning-only messages, clarifications with no code run or code change

Each step log should include:

- what the developer instructed the agent to do
- what command or code change was actually run
- what result was observed

Keep all logs factual. Separate observed behavior from proposed fixes.

## Edit Boundary

Treat interactive testing as read-run-log by default.

- Do not modify anything during the preparation phase.
- Respect the edit boundary the user sets for the session.
- Do not automatically modify the system under test when a failure or anomaly appears.
- Report the issue first and let the developer decide whether to log it, continue testing, or proceed to a fix.
- If the user says demo-owned code only, keep changes inside the target demo or tutorial area and treat shared system source as off-limits.
- Only modify demo-specific code, and only after the developer explicitly asks for a fix.
- If the user does not authorize fixes, test and log only.
- If a real fix would require stepping outside the approved boundary, stop, log the finding, and ask the user how they want to proceed.

When the boundary is ambiguous, use the narrowest reasonable interpretation and prefer asking before expanding scope.

Treat shared runtime modules, common CLI internals, and the broader system being tested as non-editable unless the developer explicitly changes the rule for the session.

## Issue Handling

When the interactive test exposes a problem:

1. Record or report what happened.
2. Identify whether the problem appears to be in demo-specific code or in the broader system under test.
3. Do not patch anything yet.
4. Let the developer choose the next action:
   - log the issue
   - keep testing without fixing
   - request a fix inside demo-specific code only

If the likely fix is outside demo-specific code, say so clearly and stop before editing.

## Interaction Pattern

- Start with a short status update describing the first inspection step.
- Keep the user updated as context is gathered.
- After preparation, summarize the current state and wait for the next instruction instead of freelancing the test plan.
- Execute the user's requested step, then report the result briefly and update the log if logging is active.
- If something fails, document or report the failure first. Do not jump to a fix unless the developer explicitly asks for one, and even then stay inside demo-specific code only.

## Examples

- "Use `$do-interactive-test` on this directory and tell me what is implemented before I give more steps."
- "Be prepared to test `openspec/changes/add-foo`; read what is there first and wait for my instructions."
- "Use `$do-interactive-test` for `scripts/demo/bar-pack`, and log the issues."
- "Use `$do-interactive-test` for this demo, `demo-root=tmp/demo-logs`, and log the issues."
- "Use `$do-interactive-test` for this demo, log the process under `tmp/test-logs`."
- "Interactively test this tutorial pack, but do not modify system source if it breaks."
