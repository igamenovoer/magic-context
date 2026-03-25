---
name: hack-through-testing
description: "Manual invocation only. Drive a crashy, hanging, or half-broken system forward along a real production user path using real data. Two subskills: `prepare` to analyze the target and set up `<htt-home>/` with infrastructure dirs (logs, runs, issues); optionally creates `<htt-home>/autotest/` with automatic scripts and interactive guides only when the developer explicitly requests test-case generation. `run` drives testing — with or without autotest artifacts — patching forward through blockers. Run subskill operates in-place by default (stash + test on current branch) or in a disposable snapshot worktree when explicitly requested. Supports automatic and interactive driving. Default when ambiguous: both subskills, in-place, automatic. Not for CI-oriented unit, smoke, or mock-based integration tests."
---

# Hack Through Testing

Manual invocation only: use this skill only when the developer explicitly wants this workflow.

Drive a fragile system to the end by patching forward instead of solving each issue cleanly on first contact. Keep each workaround reviewable and finish with a synthesis that guides the real implementation.

If the developer wants testing without code changes, use `test-and-log` instead. If the developer wants a slow, stepwise session with approval before each action, use `do-interactive-test` instead.

## Testing Philosophy: Production-Level End-to-End, Not CI

Hack-through-testing targets **production-level end-to-end paths**: real data, real user workflows, real API calls, real output artifacts. It is not a CI smoke run, not a unit test harness, and not a mock-based integration check.

The distinction matters for choosing what to test:

- **Do target**: a full user workflow from input to final output; a real data pipeline with actual inputs; a multi-step interaction flow a real user would perform; an end-to-end scenario covering multiple system components in concert.
- **Do not target**: existing unit tests, existing smoke tests, isolated module tests, test suites that stub or mock external dependencies — these are already CI's job.

**If the only testable surface you can identify is CI-style** (unit tests, smoke scripts, mock integrations), **stop and ask the developer** what the real production user path or end-to-end scenario is before proceeding. For example:

> I can see unit/smoke/integration tests already covered by CI. What's the real production user path you want to exercise — the end-to-end scenario, the live data workflow, or a specific user journey?

Do not invent a CI-style test run and call it hack-through-testing.

## Subskills

This skill has exactly two subskills.

1. `prepare`
   Analyze the target and set up `<htt-home>/` with infrastructure dirs (logs, runs, issues). If the developer explicitly requests test-case generation (e.g., "prepare test cases", "prepare for auto test", "create autotest", "set up autotest"), also create `<htt-home>/autotest/` with automatic scripts and interactive guides.
   Primary guide: `references/prepare.md`

2. `run`
   Drive testing, patching forward through blockers. Uses autotest artifacts when they exist; otherwise drives testing directly from the target analysis. Operates in-place (default) or in a disposable snapshot worktree.
   Primary guide: `references/run.md`

## Subskill Selection

- "prepare", "bootstrap", "plan", "set up" → `prepare` (infrastructure only, no autotest)
- "prepare test cases", "prepare for auto test", "create autotest", "set up autotest", "create test cases" → `prepare` with autotest generation
- "run", "test", "execute", "drive", "patch forward" → `run` only
- If the developer asks for both, or does not specify → run `prepare` then `run`
- If autotest artifacts already exist and the developer wants to go straight to driving → `run` only

## Shared Defaults

Unless the developer says otherwise, use these defaults across both subskills:

- **Target**: if the developer does not point to a specific file, directory, command, or entrypoint — or says "repo root", "workspace root", "here", "current dir" — use the agent's current working directory as the target
- **Topic slug**: derive from the target; normalize to hyphen-case; keep stable for the whole session
- **HTT home** (`htt-home`): `<repo-root>/.agent-automation/hacktest/<topic-slug>` unless the developer explicitly sets `htt-home=<dir>`
- **Log root**: `<htt-home>/logs`
- **Runs root**: `<htt-home>/runs`
- **Session log**: `<log-root>/<ts>.md`
- **Issue notes**: `<log-root>/issues/<ts>-<what>.md` — one file per underlying issue; append later fixes to the same note rather than creating a new one
- **Run artifacts**: `<runs-root>/<run-ts>/` — always copy generated artifacts here so they survive session cleanup
- **Issue IDs**: `HT-01`, `HT-02`, ...
- **Commit message format (worktree mode)** / **Stash message format (in-place mode)**: `hack-through: <issue-id> <short workaround>`
- **Stopping rule**: first successful end-to-end run, 10 distinct issues, or 90 minutes — whichever comes first
- **Data realism**: use real data, real inputs, and live API calls wherever safe. Synthetic or stubbed inputs are a last resort. Never default to a CI-style smoke run unless the developer explicitly asks for one.
- **Path references in logs**: always use repo-relative paths plus commit SHAs. Logs must remain useful after the session ends.

If this skill creates `.agent-automation/hacktest/`, add it to `.gitignore`. If `.gitignore` already has commented `.agent-automation/hacktest` entries, do not auto-add the rule.

## Guardrails

- Never lose or overwrite the developer's current uncommitted state. In in-place mode, always stash before starting and record the stash ref. Never drop the initial stash until the developer explicitly requests cleanup.
- In worktree mode, never perform hack-through edits in the original checkout after the snapshot worktree exists.
- Never commit helper-managed log or autotest artifacts to `htt-branch` (worktree mode) or mix them into stash snapshots (in-place mode).
- Never present a temporary workaround as the final fix.
- Never keep going silently after a workaround invalidates trust in later observations; log the caveat.
- In worktree mode, never merge the throwaway branch into real work.
- Never reduce interactive guides (`case-<id>.md`) to wrappers that just say "run the automatic script"; they must be independent step-by-step procedures.
- Never skip the prepare subskill silently when the run subskill needs htt-home infrastructure and it is missing.

## Resources

- `./scripts/create_snapshot_worktree.sh`: Create a snapshot branch and separate worktree without touching the active checkout.
- [references/prepare.md](./references/prepare.md): Prepare subskill guide.
- [references/run.md](./references/run.md): Run subskill guide.
- [references/log-template.md](./references/log-template.md): Session log and synthesis template.
- [references/issue-template.md](./references/issue-template.md): Per-issue note template.
- [references/git-snapshot-plumbing.md](./references/git-snapshot-plumbing.md): Git plumbing reference for the non-invasive snapshot technique.

## Example Prompts

- `Use $hack-through-testing on this CLI and keep patching forward until the happy path finishes.`
- `Use $hack-through-testing to prepare for the demo under scripts/demo/foo.` (infrastructure only)
- `Use $hack-through-testing to prepare autotest cases for the demo under scripts/demo/foo.` (with autotest generation)
- `Use $hack-through-testing in run mode with interactive driving — I want to watch each step.`
- `Use $hack-through-testing with a worktree so my checkout stays clean.`
- `Use $hack-through-testing to exercise the full build-then-launch sequence — build a brain, start a session, send a prompt, stop — and patch through every failure.`
- `Use $hack-through-testing, but pause if the only workaround would change the protocol or persistent data format.`
- `Use $hack-through-testing with htt-home=/tmp/my-htt-home so the whole session state is easy to revisit later.`
- `Use $hack-through-testing to prepare test cases, then run them automatically in a shadow repo, stop after 8 blockers.`
