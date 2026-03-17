# Git Snapshot Plumbing Notes

This note is the local reference for `openspec-ext-hack-through-test` in `run` mode when you need to inspect or adjust the snapshot helper.

## Goal

Create a disposable branch and worktree from the target repository's current state, including untracked files, without switching the developer's active checkout.

## Core Sequence

1. Resolve the target repository root and current `HEAD`.
2. Create a temporary index file so the real index is untouched.
3. Stage the repository state into the temporary index with `git add -A`.
4. Materialize a tree from that temporary index with `git write-tree`.
5. Create a snapshot commit with `git commit-tree`, parented to the original `HEAD`.
6. Create the throwaway branch at that snapshot commit.
7. Add a separate worktree for the throwaway branch.
8. Create the log directory and print the resolved paths for the session record.

## Why This Approach

- The active checkout stays on its current branch.
- Dirty tracked changes and untracked files are preserved in the snapshot.
- Each temporary workaround can be committed in isolation in the throwaway worktree.
- Cleanup stays simple because the snapshot branch and worktree are disposable.

## Caveats

- This only captures files stageable by the target repository. Nested Git repositories must be snapshotted separately.
- Ignored files remain excluded unless the helper is deliberately changed to include them.
- Existing throwaway branch names or worktree paths should be treated as hard-stop conflicts.
- If the helper changes its output locations, keep the log and issue templates in sync.

## Commands Used By The Helper

```bash
GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" add -A
tree_id="$(GIT_INDEX_FILE="$tmp_index" git -C "$repo_root" write-tree)"
snapshot_commit="$(
  git -C "$repo_root" commit-tree \
    "$tree_id" \
    -p "$parent_commit" \
    -m "hacktest snapshot $timestamp"
)"
git -C "$repo_root" branch "$snapshot_branch" "$snapshot_commit"
git -C "$repo_root" worktree add "$worktree_path" "$snapshot_branch"
```
