# Hack-Through Testing Log Template

Use this template for the session record in the main workspace, not inside `htt-branch`. Replace all `<...>` placeholders and delete sections that do not apply.

---

```markdown
# Hack-Through Session: <target>

**Date:** <YYYY-MM-DD>
**Timestamp:** <YYYYMMDD-HHMMSS> UTC
**Repo root:** <absolute-path>
**Original HEAD:** <sha>
**HTT branch:** <hacktest/topic-slug>
**Snapshot commit:** <sha>
**Throwaway worktree:** <absolute-path>
**Log root:** <repo-root>/.agent-run-logs/hacktest/<topic-slug>/
**Session log:** <repo-root>/.agent-run-logs/hacktest/<topic-slug>/<ts>.md
**Stopping rule:** <success condition / time budget / issue cap>

---

## Scope Notes

- **Target command:** `<exact command>`
- **In scope:** <services, environments, data, or dependencies>
- **Out of scope:** <what was intentionally not touched>

---

## Issue Ledger

| ID | Step / Command | Symptom | Temporary workaround | Commit | Status |
|---|---|---|---|---|---|
| HT-01 | `<command>` | <crash / hang / wrong output> | <guard / stub / skip / default> | `<sha>` | advanced / blocked / partial |
| HT-02 | `<command>` | <symptom> | <workaround> | `<sha>` | <status> |

---

## Iterations

### HT-01: <short title>

**Command:**
```bash
<exact command run>
```

**Failure observed:**
```text
<relevant stderr/stdout excerpt or hang description>
```

**Temporary workaround:**

- <what changed>
- <why this was intentionally narrow>
- <what trust caveat remains>

**Commit:** `<sha>` `hack-through: HT-01 <short workaround>`

**Outcome after re-run:** <what new path became reachable>

---

### HT-02: <short title>

**Command:**
```bash
<exact command run>
```

**Failure observed:**
```text
<excerpt>
```

**Temporary workaround:**

- <what changed>
- <why it was acceptable for discovery>
- <remaining caveat>

**Commit:** `<sha>` `hack-through: HT-02 <short workaround>`

**Outcome after re-run:** <next blocker or success>

---

<!-- Repeat for each blocker -->

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
```

---

## Usage Notes

- Keep one blocker per workaround commit.
- Prefer exact output excerpts over paraphrases.
- Record hangs explicitly: include the command, timeout used, and what signal indicated the hang.
- If the session stops early, record the furthest confirmed point reached and why progress stopped.
- Keep `htt-branch` and its worktree until the review is done, but the logs themselves live in the main workspace.
- Ignore `.agent-run-logs/` in Git so the session notes do not get committed accidentally.
