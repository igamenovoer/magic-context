# Log Template

Use this template when writing the log file for a `test-and-log` run.
Replace all `<…>` placeholders. Remove sections that do not apply.

---

```markdown
# Test Run: <subject>

**Date:** <YYYY-MM-DD>
**Timestamp:** <YYYYMMDD-HHMMSS> UTC
**Log path:** context/logs/<ts>-<task-name>/<ts>.md

---

## Environment

| Item | Status | Notes |
|---|---|---|
| <prerequisite-1> | ✅ / ❌ / ⚠️ | <version or detail> |
| <prerequisite-2> | ✅ / ❌ / ⚠️ | <version or detail> |

---

## Steps

### 1. <Step name>

**Command:**
```
<exact command run>
```

**Result:** ✅ PASS / ❌ FAIL / ⚠️ ANOMALY

<One-line summary of outcome. Include exit code if non-zero.>

**Output excerpt:**
```
<relevant lines from stdout/stderr — truncate long output with […]>
```

---

### 2. <Step name>

**Command:**
```
<exact command run>
```

**Result:** ✅ PASS / ❌ FAIL / ⚠️ ANOMALY

<summary>

**Output excerpt:**
```
<output>
```

---

<!-- Repeat for each step -->

## Anomalies and Issues

### <Anomaly title>

- **Step:** <step number/name>
- **Severity:** Hard failure / Soft anomaly / Warning
- **What happened:** <concise factual description>
- **Relevant output:**
  ```
  <exact snippet>
  ```
- **Interpretation:** <if known; omit if unknown>

---

## Summary

**Overall outcome:** ✅ Full pass / ⚠️ Pass with anomalies / ❌ Failed

| Step | Result |
|---|---|
| 1. <step> | ✅ / ❌ / ⚠️ |
| 2. <step> | ✅ / ❌ / ⚠️ |

<Two to four sentences describing what was tested, what worked, and what issues were found. Do not include speculation about fixes.>

**No source code was modified during this test run.**

---

## Reproduction

```bash
<minimal command sequence to reproduce the test run>
```
```

---

## Usage Notes

### Log file naming

```
context/logs/<ts>-<task-name>/<ts>.md
```

- `<ts>` = UTC timestamp at run start, format `YYYYMMDD-HHMMSS`
- `<task-name>` = kebab-case slug derived from the subject being tested (e.g., `cao-interactive-full-pipeline-demo-test`)
- Both the directory name and the file name use the same timestamp

### Step result codes

| Symbol | Meaning |
|---|---|
| ✅ PASS | Step completed successfully, output was as expected |
| ❌ FAIL | Step produced a non-zero exit code or hard error |
| ⚠️ ANOMALY | Exit code 0 but output was unexpected, incorrect, or suspicious |

### What counts as an anomaly (⚠️)

Document as an anomaly even when the overall pipeline exits 0:

- Response/output content identical to a prior step when it shouldn't be
- Parser warnings or drift signals in structured output (e.g., `baseline_invalidated`, version floor fallback)
- Verify/check step passes structurally but the underlying data is semantically wrong
- Any warning emitted by the tool that could indicate silent failure

### Anomaly severity

| Level | Meaning |
|---|---|
| Hard failure | Non-zero exit, exception, timeout, crash |
| Soft anomaly | Exit 0 but wrong/stale result; parser drift; structural pass with wrong data |
| Warning | Noted but non-blocking; informational only |
