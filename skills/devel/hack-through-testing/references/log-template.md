# Hack-Through Testing Log Template

Use this template for the session record in the helper-created log directory, not inside `htt-branch`. Replace all `<...>` placeholders and delete sections that do not apply.

---

````markdown
# Hack-Through Session: <target>

**Date:** <YYYY-MM-DD>
**Timestamp:** <YYYYMMDD-HHMMSS> UTC
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

- **Target command:** `<exact command>`
- **In scope:** <services, environments, data, or dependencies>
- **Out of scope:** <what was intentionally not touched>
- **Path reference rule:** use `<repo-relative-path>` on `<hacktest/topic-slug>`, not `<worktree>/...`

---

## Issue Ledger

| ID | Issue note | Symptom | Latest verified | Commit(s) | Status |
|---|---|---|---|---|---|
| HT-01 | `issues/<ts>-missing-config.md` | <crash / hang / wrong output> | yes | `<sha1>, <sha2>` | advanced / blocked / partial |
| HT-02 | `issues/<ts>-startup-timeout.md` | <symptom> | no / yes | `<sha or pending>` | <status> |

---

## Issue Note Conventions

- Save each underlying issue as its own note at `<log-root>/issues/<ts>-<what>.md`.
- Keep `<what>` concise and hyphen-case.
- Link every ledger entry to exactly one issue note.
- Append multiple commits to the same ledger entry when they belong to the same underlying issue.
- Only add a commit SHA to the ledger after the issue note records a successful verification rerun for that fix attempt.

## Run Artifact Conventions

- Copy generated artifacts for each run to `<runs-root>/<run-ts>/`.
- Use a fresh `run-ts` for each meaningful run or verification rerun.
- Reference copied artifacts from that run directory, not files left inside the throwaway worktree.

## Synthesis

### Furthest Confirmed Progress

<What part of the program actually ran by the end of the session?>

### Likely Root-Cause Themes

- <theme 1>
- <theme 2>

### Throwaway-Only Workarounds To Discard

- <commit / issue ID>: <why it should not ship>
- <commit / issue ID>: <why it should not ship>

### Likely Real Fixes

1. <real fix or design change>
2. <next real fix>
3. <follow-on cleanup or protocol change>

### Recommended Implementation Order

1. <first durable change>
2. <second durable change>
3. <validation or migration work>

### Confidence And Caveats

- <what findings are solid>
- <what later observations were made under compromised assumptions>

---

## Cleanup

- **Throwaway branch kept or deleted:** <state>
- **Throwaway worktree kept or deleted:** <state>
- **Anything copied out before cleanup:** <notes>
````

---

## Usage Notes

- Keep one verified workaround step per commit.
- Keep one underlying issue per issue note file at `<log-root>/issues/<ts>-<what>.md`.
- Prefer exact output excerpts over paraphrases.
- Record hangs explicitly: include the command, timeout used, and what signal indicated the hang.
- Only commit after the issue note shows the fix was verified to work.
- If the same issue reappears later, append the new fix attempt and commit to the existing issue note instead of creating a new one.
- Cite changed files as repo-relative paths on `htt-branch`, not as absolute paths under the throwaway worktree.
- Copy generated outputs into `<runs-root>/<run-ts>/` before referring to them in logs.
- Keep the log valid even if the throwaway worktree is later deleted.
- If the developer supplied `htt-home=...`, record that exact `htt-home` path here so later readers can recover the session layout quickly.
- If the session stops early, record the furthest confirmed point reached and why progress stopped.
- Keep `htt-branch` and its worktree until the review is done, but keep the session log outside the throwaway branch.
- Ignore `.agent-automation/hacktest/` in Git so the helper-managed hacktest tree does not get committed accidentally, unless commented `.agent-automation/hacktest` entries in `.gitignore` indicate the developer wants to manage that path manually.
