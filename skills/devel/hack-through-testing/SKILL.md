---
name: hack-through-testing
description: Manual invocation only; use only when the user explicitly requests `hack-through-testing` by exact name, or asks to drive a crashy, hanging, or half-broken program forward by applying temporary unblockers in a disposable snapshot so later failures can be discovered quickly. Snapshot the current dirty repo state without switching the active checkout, create a throwaway branch named like `hacktest/topic-slug` and its worktree, keep logs in the main workspace under `.agent-run-logs/hacktest/topic-slug/` so they are not committed with the throwaway branch, commit each workaround separately, and end with a cross-cutting review before implementing any real fix.
---

# Hack Through Testing

Manual invocation only: use this skill only when the developer explicitly wants this workflow.

Drive a fragile program to the end by patching forward in a disposable snapshot instead of solving each issue cleanly on first contact. Preserve the original checkout, keep each workaround reviewable, and finish with a synthesis that guides the real implementation.

If the developer wants testing without code changes, use `test-and-log` instead. If the developer wants a slow, stepwise session with approval before each action, use `do-interactive-test` instead.

## Defaults

Unless the developer says otherwise, use these defaults:

- Topic slug: derive from the target and normalize to hyphen-case
- Throwaway branch (`htt-branch`): `hacktest/<topic-slug>`
- Throwaway worktree: `<repo-root>/.shadow-repo/<htt-branch>`
- Log root in the main workspace: `<repo-root>/.agent-run-logs/hacktest/<topic-slug>/`
- Session log path: `<log-root>/<ts>.md`
- Issue IDs: `HT-01`, `HT-02`, ...
- Commit message format: `hack-through: <issue-id> <short workaround>`

If `.gitignore` exists and does not already ignore `.agent-run-logs/`, add `.agent-run-logs/` once so the logs stay out of commits.

## Workflow

### 1. Resolve the target and stopping rule

Identify before touching Git:

- the command, script, or entrypoint to run
- the topic slug that should name `htt-branch`
- what counts as "far enough" or "done"
- whether there is a time budget, issue budget, or both
- whether external services, network calls, or persistent state are in scope

If the developer does not set a stopping rule, use the first successful end-to-end run, 10 distinct blockers, or 90 minutes, whichever comes first.

Derive `<topic-slug>` from the target, make it concise, and keep it stable for the whole session. Example: `hacktest/login-cli-startup`.

### 2. Snapshot the current state without disturbing the active checkout

Never switch branches in the current worktree. Preserve the developer's exact dirty state, including untracked files.

Use the bundled helper:

```bash
bash scripts/create_snapshot_worktree.sh --topic TOPIC_SLUG
```

Optional arguments:

```bash
bash scripts/create_snapshot_worktree.sh --repo PATH --topic TOPIC_SLUG --branch hacktest/TOPIC_SLUG --path WORKTREE_PATH
```

The script creates:

- a snapshot commit from the current repository state, including untracked files
- `htt-branch`, which should normally be named `hacktest/<topic-slug>`
- a separate worktree checked out at `htt-branch`
- the log root in the main workspace at `.agent-run-logs/hacktest/<topic-slug>/`

Read `context/hints/howto-git-snapshot-branch-without-switching.md` only when you need the underlying Git plumbing or must adjust the helper script.

Run the workflow at the Git repo root that actually owns the files you expect to patch. If the interesting changes live inside nested Git repositories, snapshot those repositories separately; the helper can only preserve what the current repository is able to stage.

### 3. Start logs in the main workspace

Keep the logs outside `htt-branch` so they survive branch cleanup and are not part of throwaway commits.

Use [references/log-template.md](references/log-template.md) for the session record. Record:

- repository root, original `HEAD`, `htt-branch`, and throwaway worktree path
- the main-workspace log root at `.agent-run-logs/hacktest/<topic-slug>/`
- test target and stopping rule
- an issue ledger that maps issue IDs to commands, workaround commits, and status

Keep live notes concise but factual. Save interpretation for the synthesis section.

### 4. Loop: run, log, patch forward, commit, repeat

Work only inside the throwaway worktree after the snapshot is created.

For each blocker:

1. Run the target, preferably with a timeout when hangs are plausible.
2. Record the command, the failure mode, and the furthest point reached.
3. Assign the next issue ID.
4. Apply the smallest reversible workaround that unlocks the next stretch of execution.
5. Commit only that workaround with its issue ID in the message.
6. Re-run from the nearest meaningful checkpoint and continue.

Prefer quick unblockers such as:

- guards around bad inputs or absent state
- temporary stubs or shims
- force-fail-fast checks instead of hanging behavior
- narrow skips with loud comments and log entries
- temporary defaults that keep the path moving while making the compromise obvious

Avoid "real" solution work during the loop. The goal is discovery density, not code quality.

### 5. Keep temporary fixes obviously temporary

Use comments sparingly, but when a code marker helps, tag it clearly:

```text
HACK-THROUGH(HT-03): temporary unblocker to continue session
```

Keep each commit narrow. One blocker, one workaround, one commit. Do not refactor unrelated code, rename widely, or fold multiple issues into a single patch.

Prefer loud wrongness over silent wrongness. If a workaround changes behavior in a risky way, make that compromise explicit in logs and code.

### 6. Pause when the only path forward becomes high-risk

Stop and realign with the developer if moving forward would require any of these:

- changing a public protocol or on-disk data format
- mutating real external systems or shared infrastructure
- broad architectural refactors
- speculative fixes that are larger than the blocker itself
- hiding a correctness problem in a way that would make later findings untrustworthy

At that point, capture the blocker, explain why it exceeds hack-through scope, and ask whether to stop, widen scope, or switch to a normal implementation workflow.

### 7. Synthesize before implementing real fixes

When the session ends, review the ledger and workaround commits together.

Write a final synthesis that answers:

- What distinct blockers were found?
- Which blockers share a likely root cause?
- Which workarounds are throwaway-only and must not ship?
- What real fixes or design changes seem necessary?
- In what order should the real work happen?

Use the synthesis section from [references/log-template.md](references/log-template.md). The output should help the next implementation pass solve the set of issues coherently instead of replaying the temporary patches one by one.

### 8. Clean up only after the synthesis is captured

Do not discard `htt-branch` or its worktree until the synthesis is written and the developer has what they need.

When cleanup is requested:

```bash
git worktree remove <worktree-path>
git branch -D <htt-branch>
```

The workaround commits are disposable. The logs stay in the main workspace, so the branch and worktree can be deleted without losing the session record.

## Guardrails

- Never lose or overwrite the developer's current uncommitted state.
- Never perform hack-through edits in the original checkout after the snapshot worktree exists.
- Never commit `.agent-run-logs/` artifacts.
- Never present a temporary workaround as the final fix.
- Never keep going silently after a workaround invalidates trust in later observations; log the caveat.
- Never merge the throwaway branch into real work.

## Resources

- `scripts/create_snapshot_worktree.sh`: Create a snapshot branch and separate worktree without touching the active checkout.
- [references/log-template.md](references/log-template.md): Session log and synthesis template.
- `context/hints/howto-git-snapshot-branch-without-switching.md`: Git plumbing reference for the non-invasive snapshot technique.

## Example Prompts

- `Use $hack-through-testing on this CLI and keep patching forward until the happy path finishes.`
- `Use $hack-through-testing on the demo under scripts/demo/foo, stop after 8 blockers, and give me a synthesis of the real fixes afterward.`
- `Use $hack-through-testing, but pause if the only workaround would change the protocol or persistent data format.`
