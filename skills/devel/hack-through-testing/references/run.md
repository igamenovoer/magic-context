# Subskill: Run

Drive testing, patching forward through blockers. Uses autotest artifacts at `<htt-home>/autotest/` when they exist; otherwise drives testing directly from the target analysis performed during prepare. If the prepare subskill was not run separately, run it first.

## Mode Selection

Determine the isolation mode and driving mode before doing deeper work.

**Isolation modes:**

- **in-place** (default): stash uncommitted changes, operate directly in the current workspace on the current branch. Simpler setup, no worktree management.
- **worktree**: create a disposable snapshot worktree and throwaway branch. Full isolation from the developer's checkout. Use only when the developer explicitly asks for a worktree, shadow repo, temporary repo, or similar.

**Driving modes:**

- **automatic**: when autotest artifacts exist, execute automatic test scripts (`case-<id>.<ext>`) unattended via the harness. When no autotest artifacts exist, the agent drives the target commands directly — running the identified entrypoints, scripts, or workflow sequences and entering the patch-forward loop on failure.
- **interactive**: when autotest artifacts exist, follow interactive guides (`case-<id>.md`) step by step. When no autotest artifacts exist, the agent drives the target step by step, presenting each command and its result to the user, and waiting for the user to say "continue", "next", "retry", "investigate", or "skip" before proceeding.

**How to determine from context:**

- "run", "test", "execute", "drive", "patch forward" with existing autotest artifacts → run
- "interactive", "step by step", "I'll watch", "walk me through" → interactive driving
- "automatic", "unattended", "just run" → automatic driving
- "worktree", "shadow repo", "temporary repo", "disposable branch", "snapshot worktree" → worktree isolation
- If the mode cannot be determined → **in-place, automatic driving**

## Mode-Specific Defaults

**In-place mode:**

- **Initial stash message**: `hacktest snapshot <timestamp>`
- **Workaround stash message**: `hacktest <issue-id>: <short description>`
- Workarounds are captured as stash snapshots, not commits. The current branch stays clean.

**Worktree mode:**

- **Throwaway branch** (`htt-branch`): `hacktest/<topic-slug>`
- **Throwaway worktree**: `<htt-home>/repo` unless `--path` is provided
- Path references in logs use repo-relative paths on `htt-branch`.

## 1. Verify HTT Home Infrastructure

Confirm that `<htt-home>/` exists with the expected infrastructure dirs (logs, runs). If missing, run the prepare subskill first.

Check whether `<htt-home>/autotest/` exists. If it does, autotest artifacts are available for structured driving. If it does not, the run subskill drives testing directly from the target (commands, scripts, entrypoints identified during prepare).

## 2. Snapshot And Prepare

**In-place mode (default):**

Stash the developer's uncommitted changes, including untracked files, to preserve them:

```bash
git stash push --include-untracked -m "hacktest snapshot <timestamp>"
```

Record the stash ref in the session log. After stash, the workspace is at a clean HEAD state — testing proceeds from here on the current branch.

Each subsequent workaround is also captured as a stash snapshot (not a commit), keeping the current branch clean. Use `git stash create` + `git stash store` to snapshot without disturbing the working tree:

```bash
stash_sha=$(git stash create)
git stash store -m "hacktest <issue-id>: <short description>" "$stash_sha"
```

This creates a stash entry as a reviewable reference while leaving the workaround applied in the working tree. The session maintains a running list of stash refs in the session log.

Create the log and runs directories:

```bash
mkdir -p <htt-home>/logs/issues <htt-home>/runs
```

**Worktree mode:**

Never switch branches in the current worktree. Preserve the developer's exact dirty state, including untracked files.

Use the bundled helper:

```bash
bash ./scripts/create_snapshot_worktree.sh --topic TOPIC_SLUG
```

Optional arguments:

```bash
bash ./scripts/create_snapshot_worktree.sh --repo PATH --topic TOPIC_SLUG --branch hacktest/TOPIC_SLUG --htt-home HTT_HOME --path WORKTREE_PATH
```

Read [git-snapshot-plumbing.md](./git-snapshot-plumbing.md) only when you need the underlying Git plumbing or must adjust the helper script.

Run the workflow at the Git repo root that actually owns the files you expect to patch. If the interesting changes live inside nested Git repositories, snapshot those repositories separately; the helper can only preserve what the current repository is able to stage.

**Local resource bridging (worktree mode only).** Before the first test run, inspect the target and do a best-effort setup pass. If the target depends on untracked, ignored, external, or otherwise non-snapshotted resources, create the narrowest useful symlink into the worktree so the system under test can function. This bridging may also happen later when a missing resource is discovered during the run loop. Treat these symlinks as agent-managed test setup, not as workaround commits, and do not record setup gaps as product issues.

## 3. Start Logs

Use [log-template.md](./log-template.md) for the session record and [issue-template.md](./issue-template.md) for each issue note. Record:

- repository root, original `HEAD`, `htt-home`, isolation mode, and stash ref (in-place) or `htt-branch` and worktree path (worktree)
- the log root, runs root, session log path, and issue note directory
- test target and stopping rule
- an issue ledger mapping issue IDs to issue note files, latest verification state, workaround commits (worktree mode) or stash refs (in-place mode), and status

Keep live notes concise but factual. Save interpretation for the synthesis.

## 4. Drive Testing And Patch Forward

**Worktree mode:** `cd` into the throwaway worktree and do the rest of the session there.

**In-place mode:** remain in the current workspace directory.

**Automatic driving:** If autotest artifacts exist, execute the harness or individual `case-<id>.<ext>` scripts from `<htt-home>/autotest/`. If no autotest artifacts exist, drive the target commands directly — run the identified entrypoints, scripts, or workflow sequences. In both cases, the agent captures output and enters the patch-forward loop on failure.

**Interactive driving:** If autotest artifacts exist, open the relevant `case-<id>.md` guide from `<htt-home>/autotest/`. If no autotest artifacts exist, work through the target workflow step by step. In both cases, the agent executes each step on the user's behalf, presents results, and waits for the user to say "continue", "next", "retry", "investigate", or "skip" before proceeding. Enter the patch-forward loop when the user confirms a blocker needs a workaround.

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
10. **Worktree mode:** Commit only after verification succeeds. Use the issue ID in the commit message and record the commit SHA in the issue note. **In-place mode:** After verification succeeds, create a stash snapshot with `git stash create` + `git stash store -m "hacktest <issue-id>: <short description>"`. Record the stash ref in the issue note and session log stash ledger.
11. Continue from the newly reached point.

If a rerun shows the issue is not actually fixed, or later testing reveals a regression, keep iterating in the same issue note.

**Preferred unblockers:**

- guards around bad inputs or absent state
- temporary shims around non-essential or already-stubbed integrations
- force-fail-fast checks instead of hanging behavior
- narrow skips with loud comments and log entries
- temporary defaults that keep the path moving while making the compromise obvious

Avoid "real" solution work during the loop. The goal is discovery density, not code quality.

## 5. Keep Temporary Fixes Obviously Temporary

Tag workaround code clearly when a marker helps:

```text
HACK-THROUGH(HT-03): temporary unblocker to continue session
```

One verified workaround step, one commit (worktree mode) or one stash snapshot (in-place mode). Do not refactor unrelated code or fold multiple distinct issues into a single patch.

Prefer loud wrongness over silent wrongness. If a workaround changes behavior in a risky way, make the compromise explicit in logs and code.

## 6. Pause When The Only Path Forward Becomes High-Risk

Stop and realign with the developer if moving forward would require any of these:

- changing a public protocol or on-disk data format
- mutating real external systems or shared infrastructure
- broad architectural refactors
- speculative fixes that are larger than the blocker itself
- hiding a correctness problem in a way that would make later findings untrustworthy

Capture the issue, explain why it exceeds hack-through scope, and ask whether to stop, widen scope, or switch to a normal implementation workflow.

## 7. Synthesize Before Implementing Real Fixes

When the session ends, review the ledger, issue notes, and workaround commits together.

Write a final synthesis that answers:

- What distinct underlying issues were found?
- Which issues share a likely root cause?
- Which workarounds are throwaway-only and must not ship?
- What real fixes or design changes seem necessary?
- In what order should the real work happen?

Use the synthesis section from [log-template.md](./log-template.md), drawing from the per-issue notes under `<log-root>/issues/`.

## 8. Clean Up Only After The Synthesis Is Captured

Do not clean up until the synthesis is written.

**In-place mode:**

The workaround stash snapshots and the working tree contain the cumulative hack-through state. After synthesis, the developer can discard workarounds and restore their original work:

```bash
git checkout -- .            # discard workaround file changes in working tree
git clean -fd                # remove any untracked files created by workarounds
git stash drop <stash@{N}>   # drop each hacktest stash snapshot (repeat per entry)
git stash pop                # restore original uncommitted work (the initial snapshot)
```

The stash list serves as the reviewable record of each workaround step.

**Worktree mode:**

```bash
git worktree remove <worktree-path>
git branch -D <htt-branch>
```

The workaround commits (worktree mode) or stash snapshots (in-place mode) are disposable in both modes. The logs stay in `<htt-home>`.
