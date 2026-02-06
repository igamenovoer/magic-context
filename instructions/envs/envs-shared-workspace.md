# Shared Workspace Warning

**ATTENTION: SHARED WORKSPACE**

This workspace is simultaneously being accessed and edited by multiple agents and developers. 

## Guidelines for Safe Operation

1.  **Do Not Delete Unfamiliar Files:**
    - Strictly avoid deleting files or directories that you did not explicitly create or that are not clearly part of your immediate task scope.
    - Assume every file serves a purpose for another agent or developer.

2.  **Minimize Git Impact:**
    - Avoid wide-ranging git commands that affect the entire repository state (e.g., `git reset --hard`, `git clean -fd`, global reverts).
    - These actions can destroy uncommitted work or staged changes belonging to others.
    - Scope your git operations strictly to the specific files you are modifying.

3.  **Conflict Awareness:**
    - Be mindful that file states may change between your reads and writes.
    - If you encounter unexpected file contents, re-verify before overwriting.
