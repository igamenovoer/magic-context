you are an AI assistant working on the **<PROJECT_NAME>** project.

your goal is to create a **task breakdown plan** for a concrete piece of work, and save it to a markdown file such as **<TASK_FILE_PATH>** (for example, under `context/tasks/working/` or a similar tasks directory).

this plan is intended for both humans and AI agents to execute over time. it should be:
- concrete enough to drive day-to-day work,
- structured enough to be scanned quickly,
- and generic enough to be adapted to new projects and tasks.

follow the guidelines below when generating such a plan.

---

## 1. Clarify the task scope and goals

before writing the plan, infer or restate the task scope:

- identify the **main objective** of the task in 1–3 sentences.  
  - example: “use <TOOL_NAME> to optimize <MODEL_NAME> in <PROJECT_NAME> and select a configuration that balances accuracy and latency.”
- identify the **context anchors**:
  - repo/project: `<PROJECT_NAME>`
  - task file path: `<TASK_FILE_PATH>`
  - main component(s): `<MAIN_COMPONENT>` (e.g., a model, service, CLI, or module)
  - critical constraints: `<CONSTRAINTS>` (e.g., hardware, datasets, deadlines)
- if the original task text is short or ambiguous, expand it into an explicit goal statement in the plan’s “What to do” section.

keep this section high-level; do not list implementation steps here.

---

## 2. Overall structure of the plan file

use a consistent markdown structure:

```markdown
# Task: <HIGH_LEVEL_TASK_TITLE>

## What to do

- short bullets summarizing the main objective(s) of this task, focused on outcomes.

## 1. <Milestone Title>

Short description: one or two sentences about this milestone.

- [ ] Job-001-001: first actionable item for this milestone
- [ ] Job-001-002: second actionable item for this milestone
...

## 2. <Milestone Title>
...
```

guidelines:
- use `## <number>. <Milestone Title>` for each major phase or milestone.
- each milestone gets:
  - a **short description** sentence or two,
  - a **TODO list** of concrete steps.
- each TODO entry must follow the naming scheme:
  - `- [ ] Job-<section-number>-<index>: <task description>`
  - **note that** `<section-number>` and `<index>` should both be 3 digits, zero-padded (e.g., `001`, `002`).
  - example: `- [ ] Job-003-002: wire calibration dataloader into quantization CLI`

---

## 3. Designing milestones

milestones are high-level phases that a human or AI can recognize and complete in sequence. design them so that:

- they map to **logical phases** of the work, such as:
  - understanding requirements / docs / constraints
  - preparing inputs, data, or environment
  - implementing the core functionality
  - integrating tools or external systems
  - benchmarking, validating, and iterating
  - documenting and packaging results
- they are **ordered**, with each milestone building on previous ones.
- each milestone can be explained in 1–3 sentences.

examples of milestone titles (adapt for <PROJECT_NAME> and the specific task):
- `Understand tooling, docs, and repo setup`
- `Prepare inputs and baseline`
- `Implement <FEATURE_OR_TECHNIQUE>`
- `Integrate with <EXTERNAL_TOOL_OR_SYSTEM>`
- `Evaluate, compare, and tune`
- `Document outcomes and next steps`

avoid overly small milestones; keep the number of milestones manageable (typically 3–7).

---

## 4. Writing good TODO items (Job-<n>-<m>)

each TODO item under a milestone should be:
- **actionable**: clearly describes an action, not just a vague idea.
- **scoped**: something that one person or one focused AI run could complete.
- **observable**: has a clear “done” condition (e.g., file exists, script runs, metrics recorded).

rules for naming and content:

- format: `- [ ] Job-<section-number>-<index>: <task description>`
  - `<section-number>` is the milestone number, formatted as a 3-digit code (e.g., `001`, `002`, `003`).
  - `<index>` is the item number within the milestone, also formatted as a 3-digit code (e.g., `001`, `002`, `003`).
- task descriptions:
  - start with a verb: “read”, “inspect”, “implement”, “wire”, “benchmark”, “document”, etc.
  - mention the relevant artifacts explicitly:
    - file paths: `<CODE_FILE_PATH>`, `<DOC_FILE_PATH>`
    - tools/commands: `<CLI_COMMAND>`, `<SCRIPT_NAME>`
    - datasets or resources: `<DATASET_NAME>`, `<RESOURCE_PATH>`
  - be specific about the outcome:
    - “verify that <SCRIPT> runs without errors”
    - “record baseline metrics in <RESULTS_FILE_PATH>”

examples:
- `- [ ] Job-002-001: run <CLI_COMMAND> to generate <ARTIFACT> at <OUTPUT_PATH>.`
- `- [ ] Job-002-003: assemble a small, representative dataset at <DATASET_DIR> for calibration and testing.`
- `- [ ] Job-004-002: build an optimized engine using <TOOL_NAME> and store it under <ENGINE_OUTPUT_DIR>.`
- `- [ ] Job-005-004: analyze collected metrics and select the recommended configuration based on predefined criteria.`

avoid:
- TODOs that are just “research X” with no deliverable.
- TODOs that combine multiple unrelated actions; split them.

---

## 5. Considerations to bake into the plan

when writing the plan, keep these dimensions in mind and reflect them in milestones/TODOs where relevant:

- **Reproducibility**
  - ensure there are tasks to:
    - codify commands (e.g., via `<PROJECT_COMMAND_RUNNER>` such as `pixi`, `make`, or `npm`).
    - store configuration in files rather than ad-hoc commands.
    - point to expected environment (GPU/CPU, OS, Python version).
- **Data and privacy**
  - avoid checking large or sensitive data into the repo.
  - include TODOs to document where data lives and how to obtain it.
- **Performance and quality**
  - include tasks for:
    - establishing baselines (e.g., latency, accuracy, throughput).
    - comparing new results vs baselines.
    - documenting acceptable regression thresholds.
- **Integration and ergonomics**
  - if the task involves external tools, add tasks to:
    - provide a single entrypoint command to run the workflow.
    - keep configs near code (e.g., `config/`, `scripts/`, or `models/<NAME>/helpers/`).
- **Documentation and handoff**
  - ensure there is at least one milestone or TODO focused on:
    - documenting how to rerun the workflow.
    - summarizing the outcomes and recommended settings.
    - linking to related plans, issues, or design docs.

---

## 6. Adapting to different projects and files

this template must be adaptable across projects and directories. always:

- replace placeholders like:
  - `<PROJECT_NAME>`
  - `<TASK_FILE_PATH>`
  - `<MAIN_COMPONENT>`
  - `<TOOL_NAME>`
  - `<MODEL_NAME>`
  - `<DATASET_NAME>`
  - `<RESULTS_FILE_PATH>`
  - `<CODE_FILE_PATH>`
  - `<DOC_FILE_PATH>`
  - `<CLI_COMMAND>`
  - `<ENGINE_OUTPUT_DIR>`
  - `<PROJECT_COMMAND_RUNNER>`
  - and similar
  with concrete values from the current repository.
- respect any project-specific instructions, such as:
  - where task files live (e.g., `context/tasks/working/`, `context/plans/`, `docs/`).
  - naming conventions for tasks and files.
  - existing patterns in other task/plan documents.

when in doubt, look for existing task or plan files in the same project and mirror their structure and naming while still using the `Job-<section-number>-<index>` pattern for TODOs.

---

## 7. Final checks before saving the plan

before finalizing the plan in **<TASK_FILE_PATH>**, ensure that:

- the file starts with a clear task title and a “What to do” section.
- milestones are numbered sequentially (`1.`, `2.`, `3.`, …).
- every milestone has:
  - a short description sentence; and
  - at least one TODO entry, with proper `Job-<section-number>-<index>` naming using 3-digit, zero-padded numbers (e.g., `Job-001-001`).
- all TODOs are actionable and scoped; none are placeholders like “do the task”.
- references to files, tools, datasets, and commands are consistent with the `<PROJECT_NAME>` repository layout.

once these checks pass, save the markdown and treat it as the canonical breakdown for this task in **<PROJECT_NAME>**.
