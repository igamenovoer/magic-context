---
name: impl-via-git-worktree
description: Manual invocation only; use only when the developer explicitly requests `impl-via-git-worktree` or asks for implementation in a fresh local branch/worktree. Snapshot the current repository state into a new local `feature/topic-slug` or `fix/topic-slug` branch, create a separate git worktree, bridge the ignored local resources that worktree needs, implement and test there, and commit the result without pushing. If the target is an OpenSpec change or a path under `openspec/changes/`, derive the change from that target and, once inside the worktree, use OpenSpec tooling plus `openspec-apply-change` there instead of manually assuming artifact paths.
---

# Implement via Git Worktree

Manual invocation only: use this skill only when the developer explicitly wants this workflow.

Implement a change in an isolated worktree without disturbing the active checkout. Carry the current dirty repository state forward into a new local branch, bridge the local-only state the worktree needs, and do all edits, builds, and tests from inside that worktree.

## Defaults

Unless the developer says otherwise, use these defaults:

- **Topic slug**: derive from the target; normalize to hyphen-case; keep stable for the whole session
- **Branch kind**: use `fix` for broken behavior, regressions, failing tests, or bug repairs; otherwise use `feature`
- **Implementation branch**: `<branch-kind>/<topic-slug>`
- **Implementation home**: `<repo-root>/.agent-automation/impl-branches/<branch-kind>/<topic-slug>`
- **Worktree**: `<impl-home>/repo`
- **Extra link dirs**: none
- **Default linked dirs**: `.claude`, `.codex`, `.gemini`, `.github`, `.aider`, `.cursor`, `.continue`, `.windsurf`, `.kiro`; also `.pixi` when the repository appears Pixi-managed and `.pixi/` exists locally
- **Final commit**: one or more local commits on the new branch after relevant verification passes; do not push them

If this skill creates `.agent-automation/impl-branches/`, add it to `.gitignore`. If `.gitignore` already has commented `impl-branches` entries, do not auto-add the rule.

## Workflow

### 1. Resolve the target and branch naming

- Identify the exact change to implement and the verification commands you expect to use.
- Choose the branch kind: `fix` for repairs, `feature` for new behavior or refactors.
- Derive a stable topic slug from the target; normalize it to hyphen-case and keep it stable for the whole session.
- Decide whether the task obviously depends on extra local directories beyond the helper defaults.

#### Special case: OpenSpec target

When the target is an OpenSpec change, use this special-case path instead of assuming a normal repository task flow.

- Treat a path under `openspec/` that clearly points to a change, especially `openspec/changes/<change-name>/...`, as an explicit OpenSpec target signal.
- Prefer deriving the change name directly from that path. If the path is not enough to identify the change confidently, confirm it with `openspec list --json` before proceeding.
- Prefer the resolved change name as the topic slug unless the developer asks for a different slug.
- Plan to run the OpenSpec commands from inside the worktree and hand implementation flow to `$openspec-apply-change` there.

### 2. Create the isolated branch and worktree

Use the bundled helper instead of inventing the Git plumbing ad hoc:

```bash
bash <skill-dir>/scripts/create_impl_worktree.sh --topic TOPIC_SLUG --kind feature
```

For repair work, switch `--kind fix`:

```bash
bash <skill-dir>/scripts/create_impl_worktree.sh --topic TOPIC_SLUG --kind fix
```

Optional arguments:

```bash
bash <skill-dir>/scripts/create_impl_worktree.sh \
  --repo PATH \
  --topic TOPIC_SLUG \
  --kind feature \
  --branch feature/TOPIC_SLUG \
  --impl-home IMPL_HOME \
  --path WORKTREE_PATH \
  --link-dir RELATIVE_DIR
```

If the helper reports that the branch or path already exists, stop and choose whether to continue that existing isolated session or create a new topic slug. Do not silently reuse or overwrite an existing worktree.

### 3. Bridge local resources before testing

- Review the helper output for which directories were linked or skipped.
- Before the first test run, inspect the target and do a best-effort setup pass.
- If the task depends on ignored, untracked, external, or otherwise non-snapshotted resources, create the narrowest useful symlink into the worktree so the system can run.
- Treat these bridges as local setup, not as product changes. Do not commit them unless the repository intentionally tracks that path.

### 4. Move into the worktree and stay there

- `cd` into the `WORKTREE` path returned by the helper.
- After that point, read files, edit files, run builds, and run tests from the worktree only.
- Do not switch branches or patch files in the original checkout once the isolated worktree exists.
- For OpenSpec targets, this also becomes the only place where you run `openspec` commands or edit change artifacts.

### 5. Implement and verify in the isolated tree

- Make the requested code changes in the worktree.
- Run the most relevant tests, linters, or build steps there, using the repository's normal workflow.
- If a missing local resource blocks verification, bridge it into the worktree and rerun from there.
- Keep focus on the requested change; do not clean unrelated problems unless they directly block delivery.

#### Special case: OpenSpec target

When the target is an OpenSpec change, use this special-case path instead of assuming a normal repository implementation flow.

- Do not assume files such as `proposal.md`, `design.md`, or `tasks.md` exist just because the user pointed at an OpenSpec-looking path.
- First gather authoritative context in the worktree with OpenSpec tooling such as `openspec status --change "<change-name>" --json`.
- Then use `$openspec-apply-change` inside the worktree to drive the implementation loop.
- Let `$openspec-apply-change` own artifact discovery, context-file reading, task progression, and checkbox updates while this skill continues to own the isolated branch/worktree boundary.

### 6. Commit locally on the new branch

- Review `git status` in the worktree before committing.
- Commit the implementation on the new branch once relevant verification passes.
- Follow the repository's commit convention if it is obvious. Otherwise use a concise imperative message with `feat:` or `fix:` matching the branch kind.
- Never push unless the developer explicitly asks.

### 7. Report the isolated result

- Report the worktree path, branch name, final commit SHA, and the commands you used for verification.
- Call out any extra symlinked resources or manual setup assumptions that the developer should know about.
- State explicitly that the branch and worktree remain local.

## Guardrails

- Never switch branches in the developer's original checkout.
- Never keep implementing in the original checkout after the isolated worktree exists.
- For OpenSpec targets, never update the change artifacts from the original checkout; use OpenSpec tooling and `$openspec-apply-change` from inside the worktree.
- Never push, open a PR, or delete the branch/worktree unless the developer explicitly asks.
- Never copy the repository manually; use `git worktree`.
- Never treat a missing local resource as a product bug before checking whether it should simply be bridged into the worktree.
- For OpenSpec targets, never bypass OpenSpec CLI discovery by hard-coding the artifact layout when the tool output can tell you what artifacts actually exist.

## Resources

- `<skill-dir>/scripts/create_impl_worktree.sh`: Create the implementation branch/worktree from the current repository state and safely link reusable local-state directories.

## Example Prompts

- `Use $impl-via-git-worktree to implement this feature in a fresh local worktree and commit it there without pushing.`
- `Use $impl-via-git-worktree to fix the failing runtime bug on a local fix branch, run the relevant tests from inside that worktree, and leave the branch local for review.`
- `Use $impl-via-git-worktree for this refactor, keep my active checkout untouched, and tell me which extra local directories had to be linked into the worktree.`
- `Use $impl-via-git-worktree on openspec/changes/<change-name>, create a local worktree for it, and then use $openspec-apply-change inside that worktree to implement the remaining tasks.`
