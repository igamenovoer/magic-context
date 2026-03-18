---
name: hack-through-testing
description: "Manual invocation only. Drive a crashy, hanging, or half-broken system forward along a real production user path using real data. Two stages: `prepare` to analyze the target and create `<htt-home>/autotest/` with automatic scripts and interactive guides, and `run` to drive testing in a disposable snapshot worktree using those artifacts, patching forward through blockers. Supports automatic driving (unattended scripts) and interactive driving (agent executes steps, user observes and decides). Default when ambiguous: both stages with automatic driving. Not for CI-oriented unit, smoke, or mock-based integration tests."
---

# Hack Through Testing

Manual invocation only: use this skill only when the developer explicitly wants this workflow.

Drive a fragile system to the end by patching forward in a disposable snapshot instead of solving each issue cleanly on first contact. Preserve the original checkout, keep each workaround reviewable, and finish with a synthesis that guides the real implementation.

It has two stages:

- `prepare`: analyze the target and create `<htt-home>/autotest/` with automatic scripts and interactive guides
- `run`: drive testing in a disposable snapshot worktree using the autotest artifacts, patching forward through blockers

The target can be anything from a single script to a multi-step test sequence that the agent drives by invoking multiple commands, inspecting intermediate state, and exercising different surfaces of the system under test. A "run" is not limited to launching one program — it is whatever sequence of actions is needed to reach the next blocker or confirm a workaround.

If the developer wants testing without code changes, use `test-and-log` instead. If the developer wants a slow, stepwise session with approval before each action, use `do-interactive-test` instead.

## Testing Philosophy: Production-Level End-to-End, Not CI

Hack-through-testing targets **production-level end-to-end paths**: real data, real user workflows, real API calls, real output artifacts. It is not a CI smoke run, not a unit test harness, and not a mock-based integration check.

The distinction matters for choosing what to test:

- **Do target**: a full user workflow from input to final output; a real data pipeline with actual inputs; a multi-step interaction flow a real user would perform; an end-to-end scenario covering multiple system components in concert.
- **Do not target**: existing unit tests, existing smoke tests, isolated module tests, test suites that stub or mock external dependencies — these are already CI's job.

**If the only testable surface you can identify is CI-style** (unit tests, smoke scripts, mock integrations), **stop and ask the developer** what the real production user path or end-to-end scenario is before proceeding. For example:

> I can see unit/smoke/integration tests already covered by CI. What's the real production user path you want to exercise — the end-to-end scenario, the live data workflow, or a specific user journey?

Do not invent a CI-style test run and call it hack-through-testing.

## Stage And Mode Selection

Determine the stage and driving mode before doing deeper work.

**Stages:**

- **prepare**: analyze the target, create `<htt-home>/autotest/` with automatic scripts and interactive guides. Does not snapshot or patch forward.
- **run**: use the autotest artifacts to drive testing in a disposable snapshot worktree, patching forward through blockers.

**Driving modes (applies to run stage):**

- **automatic**: execute automatic test scripts (`case-<id>.<ext>`) unattended via the harness. The agent runs scripts, captures output, and enters the patch-forward loop on failure.
- **interactive**: follow interactive guides (`case-<id>.md`) step by step. The agent executes each step, presents results to the user, and waits for the user to say "continue", "next", "retry", "investigate", or "skip" before proceeding.

**How to determine from context:**

- "prepare", "bootstrap", "plan", "create test cases", "set up autotest" → prepare only
- "run", "test", "execute", "drive", "patch forward" with existing autotest artifacts → run only
- "interactive", "step by step", "I'll watch", "walk me through" → interactive driving
- "automatic", "unattended", "just run" → automatic driving
- If the stage or mode cannot be determined → **both stages with automatic driving**

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
- **Data realism**: use real data, real inputs, and live API calls wherever safe. This is a production-level E2E run. Synthetic or stubbed inputs are a last resort, not the default. Never default to a CI-style smoke run unless the developer explicitly asks for one.
- **Path references in logs**: always use repo-relative paths on `htt-branch` plus commit SHAs — never absolute worktree paths. Logs must remain useful after the worktree is deleted.

If this skill creates `.agent-automation/hacktest/`, add it to `.gitignore`. If `.gitignore` already has commented `.agent-automation/hacktest` entries, do not auto-add the rule.

## Autotest Artifact Conventions

When the prepare stage creates test artifacts, use these conventions unless the developer asks for a different structure:

### Layout

```
<htt-home>/autotest/
├── case-<id>.<ext>              # automatic variant
├── case-<id>.md                 # interactive variant
├── helpers/                     # shared scripts and functions
│   └── <shared-helper>.<ext>
└── <harness-script>.<ext>       # standalone harness for case dispatch
```

### Automatic variant

- `case-<id>.<ext>` — an executable script that runs unattended and exits with a clear pass/fail signal.
- Choose the extension to match the target project, operating system, and execution model. It is not fixed to `.sh`.
- Each script should include preflight checks, the test sequence, and explicit exit codes.

### Interactive variant

- `case-<id>.md` — a step-by-step interactive test guide designed for agent-driven execution with user observation.
- Each guide must contain inline instructions that explain what to do at each step, what to observe, and what success or failure looks like.
- Do not reduce interactive guides to "run `case-<id>.<ext>`". They are independent test procedures where an agent executes steps on the user's behalf while the user watches results and decides how to proceed.
- Structure each guide as an ordered sequence of steps. Each step should include:
  - what the agent should do (command, action, or check)
  - what the expected outcome is
  - what to look for to confirm success or detect failure
  - decision points where the user may choose to continue, retry, or investigate

### Shared helpers

- Put reusable logic under `autotest/helpers/`.
- Case scripts should source or call helpers instead of duplicating common behavior.

### Standalone harness

- A harness script that owns case selection, shared preflight orchestration, and dispatch into the `case-*.<ext>` scripts.
- Choose the harness language and extension to match the target project. Examples: `.sh` for POSIX shell-first repos, `.py` for Python-oriented projects, `.ts` for TypeScript/Node projects.

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

### Stage: Prepare

Use this stage to analyze the target and create autotest artifacts. Skip this stage if autotest artifacts already exist at `<htt-home>/autotest/` and the developer wants to go straight to run.

#### 1. Resolve the target

Identify before creating anything:

- the test target: a **production-level end-to-end path** — a real user workflow, a live data pipeline, a multi-step scenario — not a CI test suite or smoke script (use directory-type guidance above if needed)
- the topic slug for naming
- whether the developer set `htt-home=...`
- individual test cases worth covering
- prerequisites, fixtures, environment assumptions
- success and failure signals for each case

**If the only candidate target is a CI-style test** (unit tests, smoke scripts, mock-based integration tests), do not proceed. Ask the developer what the real production user path or end-to-end scenario is before doing anything else.

#### 2. Create HTT home and autotest directory

```bash
mkdir -p <htt-home>/autotest/helpers
```

If this creates `.agent-automation/hacktest/`, ensure it is gitignored.

#### 3. Create autotest artifacts

For each identified case:

- Write an automatic script (`case-<id>.<ext>`) with preflight checks, the test sequence, and pass/fail exit codes.
- Write an interactive guide (`case-<id>.md`) with step-by-step instructions, expected outcomes, and decision points.
- Extract shared logic into `helpers/`.
- Write the standalone harness script.

Follow the autotest artifact conventions above.

### Stage: Run

Use this stage to drive testing using the autotest artifacts. If prepare was not run separately, run it first.

#### 1. Verify autotest artifacts

Confirm that `<htt-home>/autotest/` exists and contains the expected case scripts and guides. If missing, run the prepare stage first.

#### 2. Snapshot and prepare the worktree

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

#### 3. Start logs

Use [references/log-template.md](./references/log-template.md) for the session record and [references/issue-template.md](./references/issue-template.md) for each issue note. Record:

- repository root, original `HEAD`, `htt-home`, `htt-branch`, and throwaway worktree path
- the helper-reported log root, runs root, session log path, and issue note directory
- test target and stopping rule
- an issue ledger mapping issue IDs to issue note files, latest verification state, workaround commits, and status

Keep live notes concise but factual. Save interpretation for the synthesis.

#### 4. Drive testing and patch forward

`cd` into the throwaway worktree and do the rest of the session there.

**Automatic driving:** Execute the harness or individual `case-<id>.<ext>` scripts from `<htt-home>/autotest/`. The agent runs scripts, captures output, and enters the patch-forward loop on failure. Each iteration uses the autotest artifacts to determine what to run next.

**Interactive driving:** Open the relevant `case-<id>.md` guide from `<htt-home>/autotest/`. Execute each step on the user's behalf, present results, and wait for the user to say "continue", "next", "retry", "investigate", or "skip" before proceeding. Enter the patch-forward loop when the user confirms a blocker needs a workaround.

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

#### 5. Keep temporary fixes obviously temporary

Tag workaround code clearly when a marker helps:

```text
HACK-THROUGH(HT-03): temporary unblocker to continue session
```

One verified workaround step, one commit. Do not refactor unrelated code or fold multiple distinct issues into a single patch.

Prefer loud wrongness over silent wrongness. If a workaround changes behavior in a risky way, make the compromise explicit in logs and code.

#### 6. Pause when the only path forward becomes high-risk

Stop and realign with the developer if moving forward would require any of these:

- changing a public protocol or on-disk data format
- mutating real external systems or shared infrastructure
- broad architectural refactors
- speculative fixes that are larger than the blocker itself
- hiding a correctness problem in a way that would make later findings untrustworthy

Capture the issue, explain why it exceeds hack-through scope, and ask whether to stop, widen scope, or switch to a normal implementation workflow.

#### 7. Synthesize before implementing real fixes

When the session ends, review the ledger, issue notes, and workaround commits together.

Write a final synthesis that answers:

- What distinct underlying issues were found?
- Which issues share a likely root cause?
- Which workarounds are throwaway-only and must not ship?
- What real fixes or design changes seem necessary?
- In what order should the real work happen?

Use the synthesis section from [references/log-template.md](./references/log-template.md), drawing from the per-issue notes under `<log-root>/issues/`.

#### 8. Clean up only after the synthesis is captured

Do not discard `htt-branch` or its worktree until the synthesis is written. When cleanup is requested:

```bash
git worktree remove <worktree-path>
git branch -D <htt-branch>
```

The workaround commits are disposable. The logs stay in the main workspace.

## Guardrails

- Never lose or overwrite the developer's current uncommitted state.
- Never perform hack-through edits in the original checkout after the snapshot worktree exists.
- Never commit helper-managed log or autotest artifacts to `htt-branch`.
- Never present a temporary workaround as the final fix.
- Never keep going silently after a workaround invalidates trust in later observations; log the caveat.
- Never merge the throwaway branch into real work.
- Never reduce interactive guides (`case-<id>.md`) to wrappers that just say "run the automatic script"; they must be independent step-by-step procedures.
- Never skip the prepare stage silently when autotest artifacts are missing and the run stage needs them.

## Resources

- `./scripts/create_snapshot_worktree.sh`: Create a snapshot branch and separate worktree without touching the active checkout.
- [references/log-template.md](./references/log-template.md): Session log and synthesis template.
- [references/issue-template.md](./references/issue-template.md): Per-issue note template.
- [references/git-snapshot-plumbing.md](./references/git-snapshot-plumbing.md): Git plumbing reference for the non-invasive snapshot technique.

## Example Prompts

- `Use $hack-through-testing on this CLI and keep patching forward until the happy path finishes.`
- `Use $hack-through-testing to prepare autotest cases for the demo under scripts/demo/foo.`
- `Use $hack-through-testing in run mode with interactive driving — I want to watch each step.`
- `Use $hack-through-testing to exercise the full build-then-launch sequence — build a brain, start a session, send a prompt, stop — and patch through every failure.`
- `Use $hack-through-testing, but pause if the only workaround would change the protocol or persistent data format.`
- `Use $hack-through-testing with htt-home=/tmp/my-htt-home so the whole session state is easy to revisit later.`
- `Use $hack-through-testing to prepare test cases, then run them automatically, stop after 8 blockers.`
