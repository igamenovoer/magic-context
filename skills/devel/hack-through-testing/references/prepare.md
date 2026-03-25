# Subskill: Prepare

Analyze the target, set up `<htt-home>/` with infrastructure dirs for logging and run artifacts. Optionally create `<htt-home>/autotest/` with automatic scripts and interactive guides — but only when the developer explicitly requests test-case generation.

Skip this subskill if `<htt-home>/` infrastructure already exists and the developer wants to go straight to run.

## 1. Resolve The Target

**If no explicit target is provided:** When the developer does not point to a specific file, directory, command, or entrypoint — or explicitly says "repo root", "workspace root", "here", "current dir", or similar — treat the agent's current working directory as the target directory. Then proceed with context gathering below to identify the testable surface from that directory.

Identify before creating anything:

- the test target: a **production-level end-to-end path** — a real user workflow, a live data pipeline, a multi-step scenario — not a CI test suite or smoke script
- the topic slug for naming
- whether the developer set `htt-home=...`
- individual test cases worth covering
- prerequisites, fixtures, environment assumptions
- success and failure signals for each case

**If the only candidate target is a CI-style test** (unit tests, smoke scripts, mock-based integration tests), do not proceed. Ask the developer what the real production user path or end-to-end scenario is before doing anything else.

## 2. Context Gathering By Directory Type

The developer may provide an explicit command, script, or entrypoint, or they may point at a directory and expect this skill to figure out what to test next. The developer may also provide no target at all, in which case the agent's current working directory is the target. The target may also be a multi-step workflow — a sequence of commands, checks, and interactions that the agent drives rather than a single invocation.

If the target is a directory rather than a concrete command or sequence, classify it before reading deeply and use that classification to identify the testable surface.

Choose the narrowest fitting category:

- **Generic directory**: a normal code or docs directory with no obvious runnable demo wrapper or OpenSpec identity
- **Demo/tutorial directory**: a directory centered on runnable instructions, demo scripts, inputs, expected outputs, or verification helpers
- **OpenSpec change directory**: a directory that appears to correspond to an OpenSpec change; do not assume its internal file layout, and use `openspec` commands first

### Generic Directory

- List files and subdirectories.
- Read the most relevant entrypoints such as `README`, runnable scripts, test files, or config files.
- Identify the best hack-through target — a single command, a test suite, or a multi-step interaction sequence.
- Avoid loading unrelated parts of the repository.

### Demo Or Tutorial Directory

- Read the local README or run instructions first.
- Read the runner script, verification helper, and expected output contract if present.
- Read the tests that exercise the demo flow if they exist.
- Identify prerequisites, environment variables, output directories, and cleanup behavior.
- Distinguish convenience wrappers from the underlying manual command flow, then choose the best hack-through target.

### OpenSpec Change Directory

Do not assume files such as `proposal.md` or `design.md` exist just because the target lives under an OpenSpec-looking path.

Use OpenSpec tooling first:

1. Derive the candidate change name from the target directory name.
2. Confirm it through `openspec list --json`.
3. Gather structured change context through:
   - `openspec show --type change --json --no-interactive <change-name>`
   - `openspec status --change <change-name> --json`
4. Use `openspec validate --type change --strict --json --no-interactive <change-name>` when validation state matters to the target under test.
5. Only after that, open specific files that are directly relevant to the testable surface you plan to hack through.

When reading files for an OpenSpec change, use the OpenSpec tool output to decide what to inspect next. Do not hard-code or assume the artifact layout inside the directory.

## 3. Create HTT Home Infrastructure

```bash
mkdir -p <htt-home>/logs/issues <htt-home>/runs
```

If this creates `.agent-automation/hacktest/`, ensure it is gitignored.

Report findings to the developer: summarize the target, testable surface, identified test cases, prerequisites, and success/failure signals.

**If the developer did not request autotest generation, stop here.** The infrastructure is ready for the run subskill to drive testing directly.

## 4. Create Autotest Artifacts (Only When Explicitly Requested)

Only create `<htt-home>/autotest/` when the developer explicitly asks for test-case generation — e.g., "prepare test cases", "prepare for auto test", "create autotest", "set up autotest". Do not create autotest artifacts by default.

```bash
mkdir -p <htt-home>/autotest/helpers
```

For each identified case:

- Write an automatic script (`case-<id>.<ext>`) with preflight checks, the test sequence, and pass/fail exit codes.
- Write an interactive guide (`case-<id>.md`) with step-by-step instructions, expected outcomes, and decision points.
- Extract shared logic into `helpers/`.
- Write the standalone harness script.

Follow the artifact conventions below.

## Autotest Artifact Conventions

### Layout

```
<htt-home>/autotest/
├── case-<id>.<ext>              # automatic variant
├── case-<id>.md                 # interactive variant
├── helpers/                     # shared scripts and functions
│   └── <shared-helper>.<ext>
└── <harness-script>.<ext>       # standalone harness for case dispatch
```

### Automatic Variant

- `case-<id>.<ext>` — an executable script that runs unattended and exits with a clear pass/fail signal.
- Choose the extension to match the target project, operating system, and execution model. It is not fixed to `.sh`.
- Each script should include preflight checks, the test sequence, and explicit exit codes.

### Interactive Variant

- `case-<id>.md` — a step-by-step interactive test guide designed for agent-driven execution with user observation.
- Each guide must contain inline instructions that explain what to do at each step, what to observe, and what success or failure looks like.
- Do not reduce interactive guides to "run `case-<id>.<ext>`". They are independent test procedures where an agent executes steps on the user's behalf while the user watches results and decides how to proceed.
- Structure each guide as an ordered sequence of steps. Each step should include:
  - what the agent should do (command, action, or check)
  - what the expected outcome is
  - what to look for to confirm success or detect failure
  - decision points where the user may choose to continue, retry, or investigate

### Shared Helpers

- Put reusable logic under `autotest/helpers/`.
- Case scripts should source or call helpers instead of duplicating common behavior.

### Standalone Harness

- A harness script that owns case selection, shared preflight orchestration, and dispatch into the `case-*.<ext>` scripts.
- Choose the harness language and extension to match the target project. Examples: `.sh` for POSIX shell-first repos, `.py` for Python-oriented projects, `.ts` for TypeScript/Node projects.
