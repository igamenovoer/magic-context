# Prompt Template: Make an Implementation Plan for a Task

This file is a reusable prompt template for generating **implementation plans** (like `context/plans/plan-send-blender-plan-into-blender.md`) for any task or feature.

Users can request a plan either by:
- Pointing at an existing task/subtask file, or
- Just saying: `make a plan about <what>`

The assistant should then create a structured plan Markdown file under `context/plans/` by default.

---

## How to Use This Template

When you want an implementation plan:

- Simple request (no task file needed):
  - `make a plan about sending a Blender plan into Blender and binding BlenderRemoteContext.plan`
- With an explicit output file:
  - `make a plan about <what> and save it to context/plans/plan-<custom-name>.md`
- Based on an existing task/subtask doc:
  - `read context/tasks/task-execute-render-plan.md and make an implementation plan about subtask 2.2`

Unless the user specifies another location, the assistant should:
- Use `context/plans/plan-<slugified-what>.md` as the output path.
- Follow the conventions from `context/README.md` and `context/plans/README.md`.

---

## Assistant Instructions (Implementation Plan Shape)

When the user asks **“make a plan about `<what>`”**, follow these steps:

1. **Clarify scope (lightweight)**  
   - Infer the scope from `<what>` and any referenced files.  
   - If ambiguous, ask 1–2 short clarification questions or make a brief, reasonable assumption and note it in the plan.

2. **Choose output filename and location**  
   - Default: `context/plans/plan-<what>.md`, where `<what>` is converted to a short, kebab-case slug (e.g., `plan-send-blender-plan-into-blender.md`).  
   - If the user specifies a path, honor it instead of the default.

3. **Create a plan file with the following structure**  
   Use a structure similar to this:

   - Title:
     - `# Plan: <Short, specific title>`  
   - `## HEADER` block:
     - **Purpose**: One or two sentences on what this plan implements.  
     - **Status**: Start with `Draft`.  
     - **Date**: Current date (YYYY-MM-DD).  
     - **Dependencies**: Bullet list of relevant design docs, specs, and code files (paths).  
     - **Target**: Who this plan is for (e.g., “Rendering subsystem developers and AI assistants”).  
   - A horizontal rule (`---`).
   - `## 1. Purpose and Outcome`:
     - Explain what success looks like for this implementation.  
     - Mention key outputs or behaviors (e.g., “a helper function”, “a CLI command”, “an in-Blender script”).  
   - `## 2. Implementation Approach`:
     - Break into small subsections such as:
       - `### 2.1 High-level flow` – numbered steps for the overall strategy.  
       - `### 2.2 Sequence diagram (steady-state usage)` – include a **Mermaid sequence diagram** showing how the feature is used end-to-end.  
     - The sequence diagram should be inside section 2, not in a separate section.  
   - `## 3. Files to Modify or Add`:
     - Bullet list of code and doc paths with short comments on how they will be touched.  
     - Use bolded code paths (e.g., `**src/module/file.py**`) for scan-ability.  
   - `## 4. TODOs (Implementation Steps)`:
     - A checklist of concrete steps using GitHub-style checkboxes:
       - `- [ ] **Action title** short description...`
     - Each item should be actionable and testable (e.g., “Implement helper X”, “Add integration test Y”, “Update doc Z”).  

4. **Mermaid sequence diagram details**  
   - Place the diagram under `## 2. Implementation Approach`, in a subsection like `### 2.2 Sequence diagram (steady-state usage)`.  
   - Use simple, self-explanatory participant labels (e.g., `Dev`, `API`, `Runner`, `Client`, `Blender`).  
   - Focus on how the new feature is *used*, not on every internal detail.  

5. **Line-wrapping and formatting rules**  
   - Do **not** hard-wrap standard paragraphs; let the editor wrap lines.  
   - Use Markdown headings, bullet lists, and code fences where appropriate.  
   - Prefer short, descriptive section titles and bolded key phrases for scan-ability.

---

## Example User Prompts

- `make a plan about implementing a JSON-based RenderPlan export CLI`
- `make a plan about phase 4 controller runner, saving it to context/plans/plan-controller-runner-phase4.md`
- `make a plan about adding an API to resume RenderJobs from the RenderSink`
- `read context/tasks/subtask-002-3-load-scene.md and make a plan about loading the scene .blend into Blender via blender-remote`

For each of these, the assistant should:
- Inspect any referenced context/docs.  
- Create a new plan file under `context/plans/` (unless told otherwise).  
- Populate it with: HEADER, purpose, implementation approach (with sequence diagram), files to touch, and a TODO checklist.

