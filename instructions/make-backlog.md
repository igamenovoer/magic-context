# Backlog Task Creation Guide and Prompt Template

Create consistent backlog items that are easy to discover, reproduce, and implement. Use this guide and the prompt template below whenever you add a new backlog task.

## Where to put files
- Directory: `context/tasks/backlog/`
- File naming: `task-<short-slug>.md`
  - Example: `task-vposer-easy-interface.md`

## Required header metadata
Add a short metadata header at the very top for provenance:

```markdown
# Metadata

- Date: <YYYY-MM-DD>
- Branch: <git-branch>
- Commit: <git-commit-hash>
```

How to get Git metadata:
- Branch: `git rev-parse --abbrev-ref HEAD`
- Commit: `git rev-parse HEAD`

Use ISO date format `YYYY-MM-DD`.

## Recommended document structure
Use clear, actionable sections. Keep it concise and skimmable.

1. Title: `# <Task Title>`
2. Context: brief background and why it matters
3. Goals: bullets of what success looks like
4. Scope/Non-goals: define boundaries to avoid scope creep
5. Design/Approach: outline the approach at a high level
6. Minimal building blocks or API shape (if code-related)
7. Implementation plan: 3–7 concrete steps
8. Usage examples or UX notes (if applicable)
9. Pitfalls and tips: common mistakes to avoid
10. Acceptance criteria: checklist to verify completion
11. References: links to docs/specs/examples

## Code snippet guidelines
- Include short, focused snippets only; no need to be fully runnable
- Show only the relevant lines; avoid boilerplate
- Wrap code in fenced blocks with language hints
- When referencing files or symbols in this repo, wrap names in backticks

## Process checklist
- Place the file in `context/tasks/backlog/`
- Include the metadata header block at the top
- Update `context/tasks/backlog/README.md` with a bullet summary and a link/filename
- Keep external links minimal and authoritative (official docs, canonical repos)

## Backlog prompt template (copy-paste)
Use this template to quickly draft a new backlog task.

```markdown
# Metadata

- Date: {{ date_yyyy_mm_dd }}
- Branch: {{ git_branch }}
- Commit: {{ git_commit }}

# {{ task_title }}

## Context
- {{ short_background }}

## Goals
- {{ outcome_1 }}
- {{ outcome_2 }}
- {{ outcome_3_optional }}

## Scope / Non-goals
- In scope: {{ in_scope_examples }}
- Out of scope: {{ out_of_scope_examples }}

## Design / Approach
- {{ high_level_design_point_1 }}
- {{ high_level_design_point_2 }}

## Minimal building blocks / API shape
```python
# Show signatures or core helpers (concise)
# e.g.
# def easy_encode(self, x: Tensor | np.ndarray, *, precision: str = "auto", ...) -> Tensor:
#     ...
```

## Implementation plan
1. {{ step_1 }}
2. {{ step_2 }}
3. {{ step_3 }}

## Usage examples
```python
# Minimal example(s) showing how a user will call the new API or feature
```

## Pitfalls and tips
- {{ pitfall_1 }}
- {{ pitfall_2 }}

## Acceptance criteria
- [ ] {{ criterion_1 }}
- [ ] {{ criterion_2 }}
- [ ] {{ criterion_3 }}

## References
- {{ doc_or_spec_link_1 }}
- {{ doc_or_spec_link_2_optional }}
```

## Example (header only)
```markdown
# Metadata

- Date: 2025-09-15
- Branch: main
- Commit: <commit-hash-here>
```

## Notes
- Keep backlog files self-contained; assume readers may open them out of context.
- Prefer links to existing repo files over re-pasting large code blocks.
- If the task touches external libraries, cite official documentation.

## Related instructions
- `make-hint.md` — how to create a hint with examples and references
- `review-code-by-search.md` — techniques to find relevant code in the repo
- `check-online.md` / `tavily-guide.md` — when and how to find authoritative sources