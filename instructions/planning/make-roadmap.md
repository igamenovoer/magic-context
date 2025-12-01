# Create a roadmap for a particular feature or module

you are tasked to create a detailed roadmap for implementing a new feature or module in a software project. The roadmap should include the following information.

by default, the output file is:
- placed in `context/plans/roadmaps/`, unless specified otherwise
- name it as `roadmap-for-(what-feature-or-module).md`, unless specified otherwise

```markdown
# Roadmap for Implementing [Feature/Module Name]

## Current Implementing Feature
(describe the currently implementing feature/module)
(for details, see the breakdown of features section below)

### Tasks
(just copy from the breakdown of features section below, excluding its own "task" header)

(note that, subtasks are only present in the implementing feature, because they are generated during the implementation of the task)

- [ ] `task-1`: [Description of task]
    - [ ] `subtask-1`: [current implementation status]
    - [ ] `subtask-2`: [current implementation status]
- [ ] `task-2`: [Description of task]

note that, `task-1` and `task-2` are placeholders for actual task names, they are named like these:
- for restful apis, they are named with routes, e.g., `GET /items`, `POST /items`
- for cli commands, they are named with command names, e.g., `(main-cmd) (sub-cmd) --(arg) (arg-value)`
- for classes or functions, they are named with class/function names, e.g., `ClassName.method_name()`, `function_name()`

note that `subtask-1` and `subtask-2` are placeholders for actual subtask names, they are named like these:
- they are self-contained, in that they are detailed enough to be understood just by reading the name.
- they are usually generated during the implementation of the task, so you do not have to plan them in advance.

---
## Planned Features

List the main features or functionalities that the module will include, like this:

- `feature-1`: Description of feature 1
- `feature-2`: Description of feature 2
...

note that, `feature-1`, `feature-2`, etc. are placeholders for actual feature names, they are named by major functionalities, e.g., `user-authentication`, `data-visualization`, `api-integration`, not necessarily corresponding to actual code names, classes or functions.

## Breakdown of Features

### feat: [Feature Name]

#### Overview
Provide a brief description of the feature/module, its purpose, and its importance to the overall project.

#### Requirements
- `requirement-1`: Description of requirement 1
- `requirement-2`: Description of requirement 2
...

note that, `requirement-1`, `requirement-2`, etc. are placeholders for actual requirement names, they will be named with summarized, high-level names of what the requirements are about, and followed by a brief description.

#### Tasks
- [ ] `task-1`: [Description of task]
- [ ] `task-2`: [Description of task]

note that, `task-1` and `task-2` are placeholders for actual task names, they are named like these:
- for restful apis, they are named with routes, e.g., `GET /items`, `POST /items`
- for cli commands, they are named with command names, e.g., `(main-cmd) (sub-cmd) --(arg) (arg-value)`
- for classes or functions, they are named with class/function names, e.g., `ClassName.method_name()`, `function_name()`

### feat: [Feature Name]
...

```