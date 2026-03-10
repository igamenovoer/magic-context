---
name: create-git-worktree
description: Create a clean git worktree for the current repository and symlink reusable local tool-state directories into it. Use when the user asks to create a shadow worktree or shadow repo, spin up a clean checkout of the current branch or another branch without carrying over uncommitted changes, prepare an isolated checkout for an agent, or reuse untracked local homes such as `.claude`, `.codex`, `.gemini`, `.github`, or `.pixi` in the new worktree.
---

# Create Git Worktree

Create a fresh worktree from the current repository while leaving the active checkout untouched. Prefer the bundled script so branch handling, default paths, and safe symlink rules stay consistent.

## Quick Start

Run:

```bash
bash scripts/create_worktree.sh [--branch BRANCH] [--path TARGET_PATH] [--link-dir NAME]
```

Use these defaults unless the user says otherwise:

- Source ref: current branch from `git branch --show-current`
- Target path: `<repo-root>/.shadow-repo/worktree-<ts>`
- Timestamp format: `YYYYMMDD-HHMMSS` in UTC
- Extra symlink dirs: none

If the user specifies a path, use it instead of the default. If the user specifies extra directories to reuse, pass them with repeated `--link-dir` flags.

## Workflow

### 1. Resolve the source ref

- Use the user-selected branch or ref if provided.
- Otherwise use the current branch.
- If the current checkout is detached and the user did not provide a branch or ref, pause and ask for one.

### 2. Create the worktree with Git

- Use `git worktree add`; do not copy the repository manually.
- If the source ref is a local branch that is not checked out in another worktree, create a normal branch worktree.
- If that branch is already checked out elsewhere, create a detached worktree at the branch tip instead. This preserves the requested clean snapshot while avoiding Git's branch checkout conflict.

### 3. Symlink reusable local-state directories

Consider these default candidates:

- `.claude`
- `.codex`
- `.gemini`
- `.github`
- `.aider`
- `.cursor`
- `.continue`
- `.windsurf`
- `.kiro`

Apply these rules for each candidate and for any user-requested extra directory:

- Symlink only when the source path already exists in the repo root.
- Skip the path if Git tracks any files under it.
- Do not replace tracked content in the new worktree with a symlink.

### 4. Handle Pixi projects

- Detect Pixi via `pixi.toml`, `pixi.lock`, or a `[tool.pixi]` section in `pyproject.toml`.
- If the repo is Pixi-managed and `.pixi/` exists as an untracked directory, symlink it into the new worktree.

### 5. Verify and report

Report:

- The created worktree path
- The source ref used
- The resulting commit
- Whether the worktree is attached to a branch or detached
- Which directories were linked
- Which directories were skipped because they were tracked

Note that `git status` in the new worktree may show the linked local-state directories as untracked. That is expected.

## Resources

- `scripts/create_worktree.sh`: Create the worktree and add the safe symlinks.

## Example Prompts

- `Create a shadow worktree for this repo under .shadow-repo and link the agent home directories.`
- `Make a clean worktree of branch release/1.4 at tmp/release-wt and reuse .claude and .pixi.`
- `Create a worktree for this repo at /tmp/gig-agents-clean; keep my current uncommitted changes untouched.`
