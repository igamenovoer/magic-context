---
name: hack-through-testing
description: Manual invocation only; use only when the user explicitly requests `hack-through-testing` by exact name, or asks to drive a crashy, hanging, or half-broken program forward by applying temporary unblockers in a disposable snapshot so later failures can be discovered quickly. Snapshot the current dirty repo state without switching the active checkout, create a throwaway branch named like `hacktest/topic-slug` and its worktree, keep logs and copied run artifacts in helper-created directories outside the throwaway branch so they are not committed with it, commit each verified workaround separately, and end with a cross-cutting review before implementing any real fix.
---

# Hack Through Testing

Manual invocation only: use this skill only when the developer explicitly wants this workflow.

Drive a fragile program to the end by patching forward in a disposable snapshot instead of solving each issue cleanly on first contact. Preserve the original checkout, keep each workaround reviewable, and finish with a synthesis that guides the real implementation.

If the developer wants testing without code changes, use `test-and-log` instead. If the developer wants a slow, stepwise session with approval before each action, use `do-interactive-test` instead.

## Defaults

Unless the developer says otherwise, use these defaults:

- Topic slug: derive from the target and normalize to hyphen-case
- Throwaway branch (`htt-branch`): `hacktest/<topic-slug>`
- Throwaway worktree: helper-generated default unless `--path` is provided
- Log root: helper-generated default under the target repository
- Runs root: helper-generated default under the target repository
- Session log path: `<log-root>/<ts>.md`
- Issue note path: `<log-root>/issues/<ts>-<what>.md`
- Run artifact path: `<runs-root>/<run-ts>/`
- Issue note rule: one issue note per underlying issue; append later fixes for the same issue to that same note
- Issue IDs: `HT-01`, `HT-02`, ...
- Commit message format: `hack-through: <issue-id> <short workaround>`

> Default note: the bundled helper currently resolves the default throwaway worktree to `<repo-root>/.agent-automation/hacktest/<topic-slug>/repo`, the default log root to `<repo-root>/.agent-automation/hacktest/<topic-slug>/logs`, and the default runs root to `<repo-root>/.agent-automation/hacktest/<topic-slug>/runs`.
>
> Issue note note: save each underlying issue as its own file under `<log-root>/issues/`, using a timestamp plus short hyphen-case description, for example `<log-root>/issues/20260317-154500-missing-config.md`. If the same issue later needs another fix or a prior fix is overturned, keep appending to the same issue note instead of creating a new one.
>
> Reference note: do not point issue logs at files under the throwaway worktree path. Refer to changed files as repo-relative paths on `<htt-branch>` and use commit SHAs so the logs stay useful even after the worktree is deleted.

If the repository ignore rules do not already exclude the helper-managed log directory, add a single ignore entry so the logs stay out of commits.

## Workflow

### 1. Resolve the target and stopping rule

Identify before touching Git:

- the command, script, or entrypoint to run
- the topic slug that should name `htt-branch`
- what counts as "far enough" or "done"
- whether there is a time budget, issue budget, or both
- whether external services, network calls, or persistent state are in scope

If the developer does not set a stopping rule, use the first successful end-to-end run, 10 distinct issues, or 90 minutes, whichever comes first.

Derive `<topic-slug>` from the target, make it concise, and keep it stable for the whole session. Example: `hacktest/login-cli-startup`.

### 2. Snapshot the current state without disturbing the active checkout

Never switch branches in the current worktree. Preserve the developer's exact dirty state, including untracked files.

Use the bundled helper:

```bash
bash ./scripts/create_snapshot_worktree.sh --topic TOPIC_SLUG
```

Optional arguments:

```bash
bash ./scripts/create_snapshot_worktree.sh --repo PATH --topic TOPIC_SLUG --branch hacktest/TOPIC_SLUG --path WORKTREE_PATH
```

The script creates:

- a snapshot commit from the current repository state, including untracked files
- `htt-branch`, which should normally be named `hacktest/<topic-slug>`
- a separate worktree checked out at `htt-branch`
- a dedicated log root outside the throwaway branch
- an `issues/` directory under the log root for per-issue notes
- a `runs/` directory under the topic root for copied run artifacts

Read [references/git-snapshot-plumbing.md](./references/git-snapshot-plumbing.md) only when you need the underlying Git plumbing or must adjust the helper script.

Run the workflow at the Git repo root that actually owns the files you expect to patch. If the interesting changes live inside nested Git repositories, snapshot those repositories separately; the helper can only preserve what the current repository is able to stage.

### 3. Start logs in the main workspace

Keep the logs outside `htt-branch` so they survive branch cleanup and are not part of throwaway commits.

Use [references/log-template.md](./references/log-template.md) for the session record and [references/issue-template.md](./references/issue-template.md) for each issue note. Record:

- repository root, original `HEAD`, `htt-branch`, and throwaway worktree path
- the helper-reported log root, runs root, session log path, and issue note directory
- test target and stopping rule
- an issue ledger that maps issue IDs to issue note files, latest verification state, workaround commits, and status

Keep live notes concise but factual. Save interpretation for the synthesis section.

When recording file references in session logs or issue notes, cite them as repo-relative paths on `htt-branch`, not as absolute paths under the throwaway worktree. Example: `src/houmao/runtime.py` on `hacktest/login-cli-startup`, not `/tmp/.../repo/src/houmao/runtime.py`.

Assume the throwaway worktree may disappear at any time. The logs should still stand on branch names, commit SHAs, repo-relative paths, and copied run artifacts alone.

### 4. Loop: run, log, patch forward, verify, commit, repeat

Work only inside the throwaway worktree after the snapshot is created.

For each issue encountered:

1. Choose a fresh run timestamp `run-ts` and create a run artifact directory at `<runs-root>/<run-ts>/`.
2. Run the target, preferably with a timeout when hangs are plausible.
3. Copy any generated artifacts worth keeping into that run directory. Examples: stdout and stderr captures, crash dumps, screenshots, exported files, and program outputs.
4. Record the command, the failure mode, the furthest point reached, and the run artifact directory.
5. Decide whether this is a new underlying issue or another manifestation of an existing one.
6. If it is new, assign the next issue ID and create a dedicated issue note at `<log-root>/issues/<ts>-<what>.md`.
7. If it is the same underlying issue as an earlier note, reopen that existing issue note and append a new fix attempt to it.
8. Apply the smallest reversible workaround that unlocks the next stretch of execution.
9. Re-run from the nearest meaningful checkpoint, or run the narrowest useful verification command, and confirm that the workaround actually fixes this manifestation of the issue. Use another fresh run timestamp and artifact directory for that rerun if it produces outputs worth keeping.
10. Commit only after that verification succeeds, using the issue ID in the commit message, and append the commit to the same issue note.
11. Update the verification record in the issue note so it names the commit SHA on `htt-branch` for which that verification result is known to hold.
12. Continue from the newly reached point.

Prefer quick unblockers such as:

- guards around bad inputs or absent state
- temporary stubs or shims
- force-fail-fast checks instead of hanging behavior
- narrow skips with loud comments and log entries
- temporary defaults that keep the path moving while making the compromise obvious

Avoid "real" solution work during the loop. The goal is discovery density, not code quality.

If the rerun shows the issue is not actually fixed, or later testing reveals that an earlier fix fails in another case, keep iterating in the same issue note and do not split that issue across multiple notes.

### 5. Keep temporary fixes obviously temporary

Use comments sparingly, but when a code marker helps, tag it clearly:

```text
HACK-THROUGH(HT-03): temporary unblocker to continue session
```

Keep each commit narrow. One verified workaround step, one commit. Do not refactor unrelated code, rename widely, or fold multiple distinct issues into a single patch.

An issue note should normally accumulate at least one verified workaround commit, and it may accumulate multiple commits if the same issue needs follow-up fixes or a previous fix is replaced.

Prefer loud wrongness over silent wrongness. If a workaround changes behavior in a risky way, make that compromise explicit in logs and code.

Treat the session log as the index and synthesis only. Put the per-issue command transcripts, failure details, fix attempts, verification reruns, and commit history in the dedicated issue file for that issue.
Keep generated artifacts outside the throwaway worktree by copying them to `<runs-root>/<run-ts>/`, then reference those copied artifacts from the logs.
For every verification entry, record the commit SHA on `htt-branch` where that verification was workable so later readers know exactly which tested state the result belongs to.

### 6. Pause when the only path forward becomes high-risk

Stop and realign with the developer if moving forward would require any of these:

- changing a public protocol or on-disk data format
- mutating real external systems or shared infrastructure
- broad architectural refactors
- speculative fixes that are larger than the blocker itself
- hiding a correctness problem in a way that would make later findings untrustworthy

At that point, capture the issue, explain why it exceeds hack-through scope, and ask whether to stop, widen scope, or switch to a normal implementation workflow.

### 7. Synthesize before implementing real fixes

When the session ends, review the ledger, issue notes, and workaround commits together.

Write a final synthesis that answers:

- What distinct underlying issues were found?
- Which issues share a likely root cause?
- Which workarounds are throwaway-only and must not ship?
- What real fixes or design changes seem necessary?
- In what order should the real work happen?

Use the synthesis section from [references/log-template.md](./references/log-template.md), drawing from the per-issue notes under `<log-root>/issues/`. The output should help the next implementation pass solve the set of issues coherently instead of replaying the temporary patches one by one.

### 8. Clean up only after the synthesis is captured

Do not discard `htt-branch` or its worktree until the synthesis is written and the developer has what they need if you can avoid it. If the developer deletes the worktree earlier, the logs should still remain valid because they point at the branch, commit SHAs, repo-relative paths, and copied artifacts rather than worktree-only paths.

When cleanup is requested:

```bash
git worktree remove <worktree-path>
git branch -D <htt-branch>
```

The workaround commits are disposable. The logs stay in the main workspace, so the branch and worktree can be deleted without losing the session record.

## Guardrails

- Never lose or overwrite the developer's current uncommitted state.
- Never perform hack-through edits in the original checkout after the snapshot worktree exists.
- Never commit helper-managed log artifacts.
- Never commit a workaround before verifying that it fixes or clearly unblocks the issue.
- Never present a temporary workaround as the final fix.
- Never keep going silently after a workaround invalidates trust in later observations; log the caveat.
- Never merge the throwaway branch into real work.
- Never split the same underlying issue across multiple issue notes.
- Never combine multiple distinct underlying issues into a single issue note.
- Never cite changed files in logs using absolute paths under the throwaway worktree.
- Never leave important generated artifacts only inside the throwaway worktree; copy them to `<runs-root>/<run-ts>/`.

## Resources

- `./scripts/create_snapshot_worktree.sh`: Create a snapshot branch and separate worktree without touching the active checkout.
- [references/log-template.md](./references/log-template.md): Session log and synthesis template.
- [references/issue-template.md](./references/issue-template.md): Per-issue note template.
- [references/git-snapshot-plumbing.md](./references/git-snapshot-plumbing.md): Git plumbing reference for the non-invasive snapshot technique.

## Example Prompts

- `Use $hack-through-testing on this CLI and keep patching forward until the happy path finishes.`
- `Use $hack-through-testing on the demo under scripts/demo/foo, stop after 8 blockers, and give me a synthesis of the real fixes afterward.`
- `Use $hack-through-testing, but pause if the only workaround would change the protocol or persistent data format.`
