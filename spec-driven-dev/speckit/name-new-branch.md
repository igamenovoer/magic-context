# Prompt: Name a new branch (NNN-what)

When creating the next branch in this repo, **do not guess the next `NNN`**.

1) First list all branches (local + remote-tracking):
- `git branch -a --format='%(refname:short)' | sort`

2) Extract the numeric prefixes that match the pattern:
- `NNN-<what>` where `NNN` is a **3-digit, zero-padded** integer (e.g., `001-...`, `008-...`).
- Consider both `NNN-...` and `origin/NNN-...` entries.

3) Compute the next branch number:
- `next_nnn = max(existing_nnn) + 1`
- Format it as 3 digits with leading zeros (e.g., max `008` → next `009`).

4) Form the branch name:
- `<next_nnn>-<what>`
- `<what>` should be short, kebab-case, descriptive (no spaces).

5) Only then create the branch:
- `git switch -c <next_nnn>-<what>`

Example:
- Existing max is `008-mdx-manage-cli` → next branch should start with `009-...`.

