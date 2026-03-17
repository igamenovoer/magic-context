# Hack-Through Issue Template

Use this template for one underlying issue at a time. Save each issue as its own file at `<log-root>/issues/<ts>-<what>.md`, where `<what>` is a short hyphen-case description such as `missing-config` or `startup-timeout`. If the same issue later needs another fix, append another fix attempt section to the same file.

---

````markdown
# <issue-id>: <short title>

**Timestamp:** <YYYYMMDD-HHMMSS> UTC
**Issue file:** <absolute-path>
**Session log:** <absolute-path to session log>
**Issue ID:** <HT-01>
**Reference branch:** <hacktest/topic-slug>
**Repo path(s):** <repo-relative-path>, <repo-relative-path>
**Latest verified:** yes / no
**Status:** advanced / blocked / partial
**Commit(s):** <sha1>, <sha2>, or pending

## Issue Summary

<What is the underlying issue, and what symptoms or cases currently point to it?>

## Fix Attempt 1

### Run Artifacts

`<runs-root>/<run-ts>/`

### Command

```bash
<exact command run>
```

### Failure Observed

```text
<relevant stderr/stdout excerpt or hang description>
```

### Furthest Confirmed Progress

<What point became reachable before the blocker stopped progress?>

### Temporary Workaround

- <what changed>
- <why this was intentionally narrow>
- <what trust caveat remains>

### Verification

**Verification workable at commit:**
`<sha on htt-branch, or pending until the verified workaround is committed>`

**Verification command:**
```bash
<exact rerun or focused verification command>
```

**Verification result:**
```text
<evidence that the blocker is fixed or that a further iteration is still needed>
```

**Outcome after verification:** <What new path became reachable, or why progress is still blocked?>

### Commit
`<sha or pending>` `hack-through: <issue-id> <short workaround>`

### Follow-On Notes

- <anything this issue suggests about root cause>
- <links or references to related issue IDs if applicable>
- <copied artifact references under the matching run directory if relevant>

## Additional Fix Attempts

Repeat another `## Fix Attempt N` section here when the same underlying issue reappears, when a previous fix needs to be replaced, or when another verified commit belongs to the same issue.
````

---

## Usage Notes

- Keep exactly one underlying issue per issue file.
- Reuse the session log for shared context and synthesis, not for detailed blocker transcripts.
- Prefer exact command lines and short output excerpts over paraphrases.
- If the issue manifestation is a hang, record the timeout used and the signal that confirmed the hang.
- Leave the commit as `pending` until the verification step shows the workaround is working.
- After committing a verified workaround, replace the verification commit placeholder with the commit SHA for which that verification result is known to hold.
- Append later fix attempts and additional commits for the same issue to this file instead of opening a new one.
- Record changed files as repo-relative paths on the issue branch, not as absolute paths under the throwaway worktree.
- Copy generated outputs to `<runs-root>/<run-ts>/` before referencing them here.
- Keep this note valid even if the throwaway worktree is later deleted.
