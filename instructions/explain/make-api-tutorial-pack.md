# Skill: Create Self-Contained API Tutorial Packs

This skill guides the creation of **"Self-Contained API Tutorial Packs"**. These are robust, reproducible, and git-tracked tutorials for CLI tools or library APIs.

## The Pattern

A "Tutorial Pack" is a single directory that contains **everything** needed to understand, run, and verify a specific usage scenario. It solves the common problem where documentation code blocks go stale or depend on untracked/missing local files.

### Key Characteristics

1.  **Self-Contained**: The tutorial directory contains its own inputs and expected outputs. It does *not* rely on transient workspace state.
2.  **Git-Tracked Verification**: It includes a "golden" snapshot of the expected output, allowing users (and CI) to verify that the current code produces the correct result.
3.  **One-Click Execution**: A single `run_demo.sh` script executes the entire pipeline end-to-end.
4.  **Non-Destructive**: The runner uses a temporary workspace (e.g., in `/tmp/` or `.gitignore`d path), ensuring it doesn't pollute the user's repo or modify the tutorial's tracked files.
5.  **Maintainable**: It includes tooling (scripts/flags) to easily regenerate the "golden" output when the code legitimately changes.

## Directory Structure

When creating a tutorial pack, create a directory (e.g., `docs/tutorial/howto/<topic>/`) with this layout:

```text
tut-<topic>/
├── inputs/                 # Minimal, tracked input files (e.g., tiny CSVs, JSONs).
├── expected_report/        # Tracked snapshot of the final output (sanitized).
├── scripts/
│   └── sanitize_report.py  # Script to strip timestamps/paths from outputs.
├── run_demo.sh             # The MAIN entry point (bash script).
└── README.md               # Explanation, prerequisites, and instructions.
```

## Implementation Recipe

Follow these steps to build a Tutorial Pack.

### 1. Scaffold the Directory
Create the folder structure. Use a descriptive name (e.g., `tut-sim-vs-real-basic`).

### 2. Create Minimal Inputs (`inputs/`)
Do not rely on large external datasets. Create tiny, representative inputs.
*   **Example**: Instead of a 1GB trace file, create a `inputs/trace_import.csv` with 5 rows.
*   **Why**: Keeps the repo light and the tutorial fast.

### 3. Write the Demo Script (`run_demo.sh`)
This is the core orchestrator. It must:
*   **Be Robust**: Use `set -euo pipefail`.
*   **Set Context**: define `REPO_ROOT` and `WORKSPACE_DIR` (temp dir).
*   **Check Prerequisites**: Verify environment, binaries, or assets exist before running.
*   **Copy Inputs**: Copy `inputs/*` from the tracked tutorial dir to the temp `WORKSPACE_DIR`.
*   **Run Pipeline**: Execute the series of commands (init, run, report).
*   **Support Snapshots**: specific flag (e.g., `--snapshot-report`) to overwrite the `expected_report/` with new results.

**Template (`run_demo.sh`):**
```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
WORKSPACE_DIR="$REPO_ROOT/tmp/tutorial_workspace_$(date +%s)" # gitignored location
mkdir -p "$WORKSPACE_DIR"

# 1. Setup Inputs
cp "$SCRIPT_DIR/inputs/data.csv" "$WORKSPACE_DIR/"

# 2. Run Commands (Example)
# Replace with project-specific commands
echo "Running pipeline..."
$REPO_ROOT/bin/my-cli run --input "$WORKSPACE_DIR/data.csv" --output "$WORKSPACE_DIR/out"

# 3. Snapshot (Maintenance Mode)
if [[ "${1:-}" == "--snapshot-report" ]]; then
  echo "Updating expected report..."
  # Run sanitizer here before copying back
  python "$SCRIPT_DIR/scripts/sanitize_report.py" "$WORKSPACE_DIR/out" "$SCRIPT_DIR/expected_report"
fi

echo "Done. Output at: $WORKSPACE_DIR/out"
```

### 4. Write the Sanitizer (`scripts/sanitize_report.py`)
Outputs often contain non-deterministic data (timestamps, absolute paths, run IDs). You MUST sanitize these before committing them to `expected_report/` to avoid noisy git diffs.

*   **Task**: Replace `$REPO_ROOT` with `<REPO_ROOT>`, timestamps with `<DATE>`, random IDs with `<ID>`.
*   **Tool**: Python is usually best for this.

### 5. Write the README (`README.md`)
The README should be a **step-by-step usage tutorial**, not just a description. Use the structure from `context/instructions/make-api-usage-tutorial.md` as the baseline, adapted to the Tutorial Pack pattern.

At minimum, the README should include:

1.  **Title + Question**: “How to `<task>` with `<API/SDK/service>`” + a clear problem statement.
2.  **Prerequisites (checklist)**: service running, env ready, required env vars, sample data assumptions.
3.  **Implementation Idea**: a high-level approach, as an ordered list of steps.
4.  **Critical Example Code**: copy/pasteable examples (Python and/or cURL) with **rich inline comments** explaining each step.
5.  **Input and Output**: show concrete example payloads and expected outputs (include sample JSON; keep it minimal and readable).
6.  **Run + Verify**:
    - How to run `run_demo.sh` end-to-end.
    - How to compare outputs with `expected_report/`.
    - How to refresh `expected_report/` using the snapshot flag (e.g., `--snapshot-report`) when behavior changes intentionally.

### 6. Verify
Run the script yourself. Ensure it passes. Check that `expected_report/` contains clean, readable files.

## Example Reference

For a concrete, production-grade example of this pattern, examine the `gpu-simulate-test` project (if available in context):

*   **Path**: `docs/tutorial/howto/tut-sim-vs-real-with-vidur-cli/`
*   **Key File**: `run_demo_static_from_pf_trace.sh` demonstrates robust environment checking, workspace management, and the snapshot flag pattern.
*   **Key File**: `scripts/sanitize_expected_report.py` shows how to clean JSON and Markdown artifacts recursively.

## When to Use This Skill
Activate this skill when the user asks to:
*   "Create a tutorial for X."
*   "Document how to use the API."
*   "Make a reproducible demo."
*   "Add a test case that serves as documentation."
