---
name: hack-through-testing
description: Manual invocation only. Drive a crashy, hanging, or half-broken system forward by applying temporary unblockers in a disposable snapshot worktree so later failures can be discovered quickly.
---

# Hack Through Testing

Manual invocation only: use this skill only when the developer explicitly wants this workflow.

Drive a fragile system to the end by patching forward in a disposable snapshot instead of solving each issue cleanly on first contact. Preserve the original checkout, keep each workaround reviewable, and finish with a synthesis that guides the real implementation.

The target can be anything from a single script to a multi-step test sequence that the agent drives by invoking multiple commands, inspecting intermediate state, and exercising different surfaces of the system under test. A "run" is not limited to launching one program — it is whatever sequence of actions is needed to reach the next blocker or confirm a workaround.

If the developer wants testing without code changes, use `test-and-log` instead. If the developer wants a slow, stepwise session with approval before each action, use `do-interactive-test` instead.

## Defaults

Unless the developer says otherwise, use these defaults:

- **Topic slug**: derive from the target; normalize to hyphen-case; keep stable for the whole session
- **HTT home** (`htt-home`): resolved by the helper to `<repo-root>/.agent-automation/hacktest/<topic-slug>` unless the developer explicitly sets `htt-home=<dir>`
- **Throwaway branch** (`htt-branch`): `hacktest/<topic-slug>`
- **Throwaway worktree**: `<htt-home>/repo` unless `--path` is provided
- **Log root**: `<htt-home>/logs`
- **Runs root**: `<htt-home>/runs`
- **Session log**: `<log-root>/<ts>.md`
- **Issue notes**: `<log-root>/issues/<ts>-<what>.md` — one file per underlying issue; append later fixes to the same note rather than creating a new one
- **Run artifacts**: `<runs-root>/<run-ts>/` — always copy generated artifacts here so they survive worktree deletion
- **Issue IDs**: `HT-01`, `HT-02`, ...
- **Commit message format**: `hack-through: <issue-id> <short workaround>`
- **Stopping rule**: first successful end-to-end run, 10 distinct issues, or 90 minutes — whichever comes first
- **Data realism**: prefer realistic inputs, real data, and real read-only API calls. Only treat the session as a smoke-style run when the developer says so explicitly.
- **Path references in logs**: always use repo-relative paths on `htt-branch` plus commit SHAs — never absolute worktree paths. Logs must remain useful after the worktree is deleted.

If this skill creates `.agent-automation/hacktest/`, add it to `.gitignore`. If `.gitignore` already has commented `.agent-automation/hacktest` entries, do not auto-add the rule.

## Context Gathering By Directory Type

The developer may provide an explicit command, script, or entrypoint, or they may point at a directory and expect this skill to figure out what to test next. The target may also be a multi-step workflow — a sequence of commands, checks, and interactions that the agent drives rather than a single invocation.

If the target is a directory rather than a concrete command or sequence, classify it before reading deeply and use that classification to identify the testable surface.

Choose the narrowest fitting category:

- **Generic directory**: a normal code or docs directory with no obvious runnable demo wrapper or OpenSpec identity
- **Demo/tutorial directory**: a directory centered on runnable instructions, demo scripts, inputs, expected outputs, or verification helpers
- **OpenSpec change directory**: a directory that appears to correspond to an OpenSpec change; do not assume its internal file layout, and use `openspec` commands first

### Generic Directory

- List files and subdirectories.
- Read the most relevant entrypoints such as `README`, runnable scripts, test files, or config files.
- Identify the best hack-through target — a single command, a test suite, or a multi-step interaction sequence.
- Avoid loading unrelated parts of the repository.

### Demo Or Tutorial Directory

- Read the local README or run instructions first.
- Read the runner script, verification helper, and expected output contract if present.
- Read the tests that exercise the demo flow if they exist.
- Identify prerequisites, environment variables, output directories, and cleanup behavior.
- Distinguish convenience wrappers from the underlying manual command flow, then choose the best hack-through target.

### OpenSpec Change Directory

Do not assume files such as `proposal.md` or `design.md` exist just because the target lives under an OpenSpec-looking path.

Use OpenSpec tooling first:

1. Derive the candidate change name from the target directory name.
2. Confirm it through `openspec list --json`.
3. Gather structured change context through:
   - `openspec show --type change --json --no-interactive <change-name>`
   - `openspec status --change <change-name> --json`
4. Use `openspec validate --type change --strict --json --no-interactive <change-name>` when validation state matters to the target under test.
5. Only after that, open specific files that are directly relevant to the testable surface you plan to hack through.

When reading files for an OpenSpec change, use the OpenSpec tool output to decide what to inspect next. Do not hard-code or assume the artifact layout inside the directory.

## Workflow

### 1. Resolve the target and stopping rule

Identify before touching Git:

- the test target: a command, script, test suite, or multi-step interaction sequence (use directory-type guidance above if needed)
- the topic slug for `htt-branch`
- whether the developer set `htt-home=...`
- what counts as "far enough" or "done" (fall back to the default stopping rule)
- whether there is a time budget, issue budget, or both
- whether external services, network calls, or persistent state are in scope

### 2. Snapshot and prepare the worktree

Never switch branches in the current worktree. Preserve the developer's exact dirty state, including untracked files.

Use the bundled helper:

```bash
bash ./scripts/create_snapshot_worktree.sh --topic TOPIC_SLUG
```

Optional arguments:

```bash
bash ./scripts/create_snapshot_worktree.sh --repo PATH --topic TOPIC_SLUG --branch hacktest/TOPIC_SLUG --htt-home HTT_HOME --path WORKTREE_PATH
```

Read [references/git-snapshot-plumbing.md](./references/git-snapshot-plumbing.md) only when you need the underlying Git plumbing or must adjust the helper script.

Run the workflow at the Git repo root that actually owns the files you expect to patch. If the interesting changes live inside nested Git repositories, snapshot those repositories separately; the helper can only preserve what the current repository is able to stage.

**Local resource bridging.** Before the first test run, inspect the target and do a best-effort setup pass. If the target depends on untracked, ignored, external, or otherwise non-snapshotted resources, create the narrowest useful symlink into the worktree so the system under test can function. This bridging may also happen later when a missing resource is discovered during the run loop. Treat these symlinks as agent-managed test setup, not as workaround commits, and do not record setup gaps as product issues.

### 3. Start logs

Use [references/log-template.md](./references/log-template.md) for the session record and [references/issue-template.md](./references/issue-template.md) for each issue note. Record:

- repository root, original `HEAD`, `htt-home`, `htt-branch`, and throwaway worktree path
- the helper-reported log root, runs root, session log path, and issue note directory
- test target and stopping rule
- an issue ledger mapping issue IDs to issue note files, latest verification state, workaround commits, and status

Keep live notes concise but factual. Save interpretation for the synthesis.

### 4. Loop: test, log, patch forward, verify, commit, repeat

`cd` into the throwaway worktree and do the rest of the session there.

Each iteration exercises the target — which may mean running a single command, executing a test suite, or driving a multi-step interaction sequence (invoking commands, inspecting outputs, checking state, calling APIs). The agent decides what to run next based on where the previous iteration stopped.

For each issue encountered:

1. Choose a fresh `run-ts` and create `<runs-root>/<run-ts>/`.
2. Execute the next test step or sequence. Use timeouts when hangs are plausible.
3. If the failure is just a missing local resource, bridge it and rerun — this is setup, not a product issue.
4. Copy generated artifacts worth keeping into the run directory.
5. Record the commands executed, failure mode, furthest point reached, and run artifact directory.
6. Decide: new underlying issue, or another manifestation of an existing one?
7. New issue → assign the next `HT-nn` ID and create an issue note. Same issue → append to the existing note.
8. Apply the smallest reversible workaround that unlocks the next stretch of execution.
9. Re-run to verify the workaround actually fixes this manifestation. Use another fresh `run-ts` if the rerun produces outputs worth keeping.
10. Commit only after verification succeeds. Use the issue ID in the commit message and record the commit SHA in the issue note.
11. Continue from the newly reached point.

If a rerun shows the issue is not actually fixed, or later testing reveals a regression, keep iterating in the same issue note.

**Preferred unblockers:**

- guards around bad inputs or absent state
- temporary shims around non-essential or already-stubbed integrations
- force-fail-fast checks instead of hanging behavior
- narrow skips with loud comments and log entries
- temporary defaults that keep the path moving while making the compromise obvious

Avoid "real" solution work during the loop. The goal is discovery density, not code quality.

### 5. Keep temporary fixes obviously temporary

Tag workaround code clearly when a marker helps:

```text
HACK-THROUGH(HT-03): temporary unblocker to continue session
```

One verified workaround step, one commit. Do not refactor unrelated code or fold multiple distinct issues into a single patch.

Prefer loud wrongness over silent wrongness. If a workaround changes behavior in a risky way, make the compromise explicit in logs and code.

### 6. Pause when the only path forward becomes high-risk

Stop and realign with the developer if moving forward would require any of these:

- changing a public protocol or on-disk data format
- mutating real external systems or shared infrastructure
- broad architectural refactors
- speculative fixes that are larger than the blocker itself
- hiding a correctness problem in a way that would make later findings untrustworthy

Capture the issue, explain why it exceeds hack-through scope, and ask whether to stop, widen scope, or switch to a normal implementation workflow.

### 7. Synthesize before implementing real fixes

When the session ends, review the ledger, issue notes, and workaround commits together.

Write a final synthesis that answers:

- What distinct underlying issues were found?
- Which issues share a likely root cause?
- Which workarounds are throwaway-only and must not ship?
- What real fixes or design changes seem necessary?
- In what order should the real work happen?

Use the synthesis section from [references/log-template.md](./references/log-template.md), drawing from the per-issue notes under `<log-root>/issues/`.

### 8. Clean up only after the synthesis is captured

Do not discard `htt-branch` or its worktree until the synthesis is written. When cleanup is requested:

```bash
git worktree remove <worktree-path>
git branch -D <htt-branch>
```

The workaround commits are disposable. The logs stay in the main workspace.

## Guardrails

- Never lose or overwrite the developer's current uncommitted state.
- Never perform hack-through edits in the original checkout after the snapshot worktree exists.
- Never commit helper-managed log artifacts to `htt-branch`.
- Never present a temporary workaround as the final fix.
- Never keep going silently after a workaround invalidates trust in later observations; log the caveat.
- Never merge the throwaway branch into real work.

## Resources

- `./scripts/create_snapshot_worktree.sh`: Create a snapshot branch and separate worktree without touching the active checkout.
- [references/log-template.md](./references/log-template.md): Session log and synthesis template.
- [references/issue-template.md](./references/issue-template.md): Per-issue note template.
- [references/git-snapshot-plumbing.md](./references/git-snapshot-plumbing.md): Git plumbing reference for the non-invasive snapshot technique.

## Example Prompts

- `Use $hack-through-testing on this CLI and keep patching forward until the happy path finishes.`
- `Use $hack-through-testing on the demo under scripts/demo/foo, stop after 8 blockers, and give me a synthesis of the real fixes afterward.`
- `Use $hack-through-testing to exercise the full build-then-launch sequence — build a brain, start a session, send a prompt, stop — and patch through every failure.`
- `Use $hack-through-testing, but pause if the only workaround would change the protocol or persistent data format.`
- `Use $hack-through-testing with htt-home=/tmp/my-htt-home so the whole session state is easy to revisit later.`
