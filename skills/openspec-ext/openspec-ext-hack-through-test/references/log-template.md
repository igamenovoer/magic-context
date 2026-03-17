# OpenSpec Hack-Through Testing Log Template

Use this template for the session record in the helper-created log directory, not inside `htt-branch`. Replace all `<...>` placeholders and delete sections that do not apply.

---

````markdown
# OpenSpec Hack-Through Session: <change-name>

**Date:** <YYYY-MM-DD>
**Timestamp:** <YYYYMMDD-HHMMSS> UTC
**Change:** <change-name>
**Schema:** <schema-name>
**Change dir:** <absolute-path>
**Artifacts consulted:** <path>, <path>
**Canonical path under test:** <short description>
**Repo root:** <absolute-path>
**Original HEAD:** <sha>
**HTT home:** <absolute-path>
**HTT branch:** <hacktest/topic-slug>
**Snapshot commit:** <sha>
**Throwaway worktree:** <absolute-path>
**Log root:** <absolute-path reported by helper>
**Runs root:** <absolute-path reported by helper>
**Session log:** <absolute-path>
**Issue directory:** <log-root>/issues/
**Stopping rule:** <success condition / time budget / issue cap>

---

## Scope Notes

- **Target command or sequence:** `<exact command or summarized sequence>`
- **Implementation entrypoints:** <repo-relative-path>, <repo-relative-path>
- **In scope:** <services, environments, data, or dependencies>
- **Out of scope:** <what was intentionally not touched>
- **Path reference rule:** use `<repo-relative-path>` on `<hacktest/topic-slug>`, not `<worktree>/...`

---

## Issue Ledger

| ID | Issue note | Symptom | Latest verified | Commit(s) | Status |
|---|---|---|---|---|---|
| HT-01 | `issues/<ts>-missing-config.md` | <crash / hang / wrong output> | yes | `<sha1>, <sha2>` | advanced / blocked / partial |

---

## Run Artifact Conventions

- Copy generated artifacts for each run to `<runs-root>/<run-ts>/`.
- Use a fresh `run-ts` for each meaningful run or verification rerun.
- Reference copied artifacts from that run directory, not files left inside the throwaway worktree.

## Synthesis

### Furthest Confirmed Progress

<What part of the intended change behavior actually ran by the end of the session?>

### Findings By Type

- <implementation bug>
- <design gap>
- <artifact mismatch>
- <environment/setup issue>

### Throwaway-Only Workarounds To Discard

- <commit / issue ID>: <why it should not ship>

### Likely Real Fixes

1. <real fix or design change>
2. <next real fix>

### Change Follow-Up

- <does the OpenSpec change need revise-mode updates?>
- <which artifact or requirement should be clarified?>

### Confidence And Caveats

- <what findings are solid>
- <what later observations were made under compromised assumptions>

---

## Cleanup

- **Throwaway branch kept or deleted:** <state>
- **Throwaway worktree kept or deleted:** <state>
- **Anything copied out before cleanup:** <notes>
````
