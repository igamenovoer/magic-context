# OpenSpec Hack-Through Issue Template

Use this template for one underlying issue at a time. Save each issue as its own file at `<log-root>/issues/<ts>-<what>.md`, where `<what>` is a short hyphen-case description.

---

````markdown
# <issue-id>: <short title>

**Timestamp:** <YYYYMMDD-HHMMSS> UTC
**Change:** <change-name>
**Schema:** <schema-name>
**Issue file:** <absolute-path>
**Session log:** <absolute-path to session log>
**Issue ID:** <HT-01>
**HTT home:** <absolute-path>
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

### OpenSpec Follow-On

- <does this reveal a design/spec mismatch?>
- <should the change later be revised?>

## Additional Fix Attempts

Repeat another `## Fix Attempt N` section here when the same underlying issue reappears, when a previous fix needs to be replaced, or when another verified commit belongs to the same issue.
````
