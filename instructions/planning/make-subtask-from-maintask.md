# Prompt Template: Breaking a Main Task into Subtasks with Numbered TODOs

This file is a reusable prompt template for breaking a **main task** into **subtasks**, each with its own TODO list and consistent job numbering.

You can adapt it to any project by replacing the placeholders with your own project name, task names, and paths.

---

## How to Use This Template

1. Decide which main task you want to break down.
2. Fill in the placeholders in the prompt below:
   - `<PROJECT_NAME>`
   - `<MAIN_TASK_NAME>`
   - `<MAIN_TASK_SECTION_NUMBER>` (e.g., `2`)
   - `<TASKS_DIR>` (where task/subtask Markdown files live, e.g., `context/tasks`)
3. Paste the filled-in prompt into your AI assistant and run it in the project repo.

The assistant should:
- Update the main task file to introduce **milestones** (subsections) under the chosen section, keeping high-level text in the main file and moving detailed TODOs into separate subtask files.
- Create one subtask file per milestone under `<TASKS_DIR>/subtask-<SECTION_3DIG>-<index>-<slug>.md`.
- Use the numbering scheme described below for all TODO items.

---

## Numbering Scheme (Jobs and Subtasks)

The numbering scheme is designed to keep jobs traceable from the main task down into subtasks, and **all numeric components are 3 digits with leading zeros**.

### 1. Main task section jobs

Under a main section with number `<MAIN_TASK_SECTION_NUMBER>` (e.g., section `2` → `002`), jobs in the main task file are written as:

- `Job-<SECTION_3DIG>-<SUBTASK_3DIG>`

where:
- `<SECTION_3DIG>` is the 3-digit main section number (e.g., `002` for section 2).
- `<SUBTASK_3DIG>` is a 3-digit code that groups jobs for a given subtask/milestone (e.g., `101`, `102`, `103`).

Examples for section `2`:
- `Job-002-101`
- `Job-002-102`
- `Job-002-103`

These jobs usually correspond to **milestones** or **subtasks** (e.g., “Complete subtask 2.1”).

### 2. Subtask files and IDs

For each milestone under section `<MAIN_TASK_SECTION_NUMBER>`, create:

- A subtask file:
  - `<TASKS_DIR>/subtask-<SECTION_3DIG>-<SUBTASK_INDEX>-<short-slug>.md`
  - Example: `subtask-002-3-load-scene.md`
- A human-readable heading in that file:
  - `# Subtask <MAIN_TASK_SECTION_NUMBER>.<SUBTASK_INDEX>: <Title>`

Inside each subtask file, TODO items use **three-level job IDs**:

- `Job-<SECTION_3DIG>-<SUBTASK_3DIG>-<LOCAL_3DIG>`

where:
- `<SECTION_3DIG>` is the 3-digit main section number (e.g., `002`).
- `<SUBTASK_3DIG>` is the same 3-digit subtask code used in the main task (e.g., `101` for subtask 2.1, `103` for subtask 2.3).
- `<LOCAL_3DIG>` is a per-subtask counter starting from `001` (`001`, `002`, `003`, ...).

Examples for subtask 2.1:
- `Job-002-101-001`
- `Job-002-101-002`

Examples for subtask 2.3:
- `Job-002-103-001`
- `Job-002-103-002`

This scheme ensures:
- Jobs in the main task file (e.g., `Job-002-103`) map directly to a subtask.
- Jobs inside each subtask (e.g., `Job-002-103-001`) are uniquely identifiable and grouped under their parent milestone.

---

## Prompt Template

You can copy-paste and customize the following prompt when working with your AI assistant:

```text
You are working in the <PROJECT_NAME> repository.

Goal:
- Break down the main task section `<MAIN_TASK_SECTION_NUMBER>. <MAIN_TASK_NAME>` in `<TASKS_DIR>/task-<MAIN_TASK_NAME>.md` into multiple subtasks (milestones), and create individual subtask files under `<TASKS_DIR>/` using a consistent numbering scheme.

Requirements:

1) Main task section restructuring
- In `<TASKS_DIR>/task-<MAIN_TASK_NAME>.md`, find section `<MAIN_TASK_SECTION_NUMBER>. <MAIN_TASK_NAME>` and:
  - Replace any generic, detailed TODO list under that section with:
    - A short "Scope" paragraph (kept in the main file).
    - A "Planned outputs" bullet list (kept in the main file).
    - A "Milestones (subtasks)" list, where each milestone is described as:
      - A short paragraph heading (e.g., `### <MAIN_TASK_SECTION_NUMBER>.<SUBTASK_INDEX> <Milestone title>`).
      - A one-paragraph "Goal" description (high-level only).
      - A line pointing to the subtask spec file:
        - `- Subtask spec: <TASKS_DIR>/subtask-<SECTION_3DIG>-<SUBTASK_INDEX>-<slug>.md`
    - The **detailed TODO items and investigation steps should live in the subtask files**, not in the main task section.
  - Add a new TODO list that uses job IDs at the milestone level:
    - `Job-<SECTION_3DIG>-<SUBTASK_3DIG>`
  - Each main TODO should say "Complete subtask <MAIN_TASK_SECTION_NUMBER>.<SUBTASK_INDEX> (...)" or similar.

2) Subtask files
- For each milestone created in step (1), create a new subtask file:
  - Path: `<TASKS_DIR>/subtask-<MAIN_TASK_SECTION_NUMBER>-<SUBTASK_INDEX>-<slug>.md`
  - Structure:
    - Title: `# Subtask <MAIN_TASK_SECTION_NUMBER>.<SUBTASK_INDEX>: <Title>`
    - "Scope" section describing the subtask boundaries.
    - "Planned outputs" section listing expected artifacts or outcomes.
    - "TODOs" section containing a Markdown checkbox list with job IDs using the three-level scheme:
      - `- [ ] Job-<SECTION_3DIG>-<SUBTASK_3DIG>-<LOCAL_3DIG> <description>`
    - Optional "Notes" section for open questions or constraints.

3) Numbering rules
- Main task TODOs:
  - Use `Job-<SECTION_3DIG>-<SUBTASK_3DIG>` per milestone (both numeric parts are 3 digits).
- Subtask TODOs:
  - Use `Job-<SECTION_3DIG>-<SUBTASK_3DIG>-<LOCAL_3DIG>`, where:
    - `<SUBTASK_3DIG>` is unique per subtask and matches the code used in the main task for that milestone.
    - `<LOCAL_3DIG>` is a 3-digit index (`001`, `002`, ...) within each subtask.

4) Content sources
- Use existing design/spec documents, if available, to shape subtasks.
- Respect existing naming, architecture, and domain concepts in <PROJECT_NAME>.

5) Output
- Apply changes directly to the repository files (do not just print the desired contents).
- Summarize:
  - Which milestones were added under the main task section.
  - Which subtask files were created and their purposes.
```

You can adapt the file and directory names to match your project conventions (for example, using `docs/tasks/` instead of `context/tasks/`).
