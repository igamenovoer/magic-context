---
name: openspec-ext-revise-by-decision
description: Manual invocation only; use only when the user explicitly requests `openspec-ext-revise-by-decision` by exact name. Revise OpenSpec change artifacts from a review or decision document that contains questions plus `DECISION` blocks, applying chosen decisions from a review file such as `openspec/changes/<change>/review/review-*.md` back into proposal, design, specs, and tasks.
---

# OpenSpec Extension: Revise By Decision

Read a decision-bearing review document, determine which OpenSpec artifacts it affects, and update those artifacts so they reflect the final decisions.

**Goal:** turn chosen decisions from a review or decision document into coherent edits across the relevant OpenSpec artifacts.

## Inputs

Accept one of:

- A review or decision document path
- A change name plus a review or decision document path
- A request that clearly refers to the current change's latest review document

## Output

Revise the relevant OpenSpec artifacts in place and report which decisions were applied, which artifacts changed, and which questions remained unresolved.

## Workflow

### 1) Select The Decision Document

**Selection rules:**

- If the user provides a file path, use it.
- Otherwise, infer the current change from conversation context.
- If you know the change but not the document, inspect `openspec/changes/<change>/review/` and choose the most recent `review-*.md`.
- If selection is still ambiguous, list the candidate review files and ask the user to choose.

**Always:** state which review file you are using.

### 2) Resolve The Change And Artifact Context

**Prefer:** derive the change directly from the review path when it matches:

```text
openspec/changes/<change-name>/review/<file>.md
```

**Other valid signals:**

- an explicit change name from the user
- a clearly identified current change from conversation context
- a directory path that clearly contains the target OpenSpec change

**If the target change is still unclear, do not guess.**

Run:

```bash
openspec list --json
```

Then list the active OpenSpec changes as a **numbered list** and ask the user to choose by:

1. number
2. change name
3. directory path that contains the OpenSpec change in question

Only continue once the target change is unambiguous.

**Then:** gather the authoritative artifact context:

```bash
openspec status --change "<change-name>" --json
openspec instructions apply --change "<change-name>" --json
```

**Use these outputs to determine:**

- `changeDir`
- `schema`
- current artifact status
- `contextFiles`

**Treat `contextFiles` as a map** from artifact role to file path or glob.

**Read:** every existing file path referenced by the `contextFiles` values.
If a `contextFiles` value is a glob such as `specs/**/*.md`, expand it on disk and read the matching files.

**Also read:** any repository files that the decision document cites as concrete evidence when they materially affect the decision.

### 3) Parse Questions And Final Decisions

Treat the review document as the source of decision input.

**Preferred mapping rule:**

- Each question begins at a heading like `### Q1) ...`
- The final decision for that question is the nearest following blockquote that starts with `> **DECISION:`

If that pattern is not present, fall back to document-specific pattern discovery before giving up.

**Fallback procedure:**

- Inspect the whole document and identify the recurring structure it actually uses for questions, options, and final decisions
- Look for question-like anchors such as section headings, numbered prompts, `Question:`, `Open Question:`, or interrogative lead-ins
- Look for final-decision anchors such as `DECISION`, `Decision:`, `Final decision`, `Resolution`, `Chosen option`, `Outcome`, or other clearly authoritative resolution markers
- Pair each question with the nearest downstream final-decision marker that appears to close that question's discussion block
- If multiple candidate decisions exist for one question, prefer the most explicit and final-looking marker, usually the latest authoritative resolution inside that question's block
- If the document uses a consistent but different pattern, describe that discovered pattern briefly in your response and proceed using it

**Important:**

- The `DECISION` block is authoritative, even if it disagrees with the earlier `Proposal (Recommended)` subsection.
- If the exact `DECISION` marker is absent, treat the document's discovered final-resolution marker as authoritative instead of forcing the example format.
- If a question has no `DECISION` block, do not invent one.
- If a `DECISION` block is ambiguous, quote the exact ambiguity to the user instead of guessing.
- If no reliable question-to-decision pattern can be discovered, stop and ask the user rather than making speculative edits.

**For each resolved question, capture:**

- the question
- the final decision
- the rationale
- the artifact(s) affected

### 4) Map Decisions To Artifacts

Use both the review text and the current artifacts to decide where each decision belongs.

**Source of truth:** OpenSpec CLI output, especially `openspec status --change "<change-name>" --json` and `openspec instructions apply --change "<change-name>" --json`, determines the current artifact set, names, locations, and count.

Do not hardcode artifact names, paths, or expected artifact counts. OpenSpec schemas and tool versions may change these over time.

**Use conceptual roles first, then map them to the artifacts that actually exist in the current change.**

**Common spec-driven examples only:**

- **`proposal.md`**: update only when the decision changes scope, capability boundaries, motivation, or user-facing change summary
- **`design.md`**: update chosen approach, constraints, invariants, lifecycle rules, data shape, or trade-offs
- **`specs/**/*.md`**: update normative behavior, requirements, scenarios, command semantics, or validation rules
- **`tasks.md`**: update implementation order, add or remove tasks, or add test and verification work implied by the decision

**Prefer:** explicit file references from the review document first.
If the review names a code path as evidence, use it to understand the decision, but revise OpenSpec artifacts unless the user asks for code changes too.

### 5) Revise The Artifacts

Apply the decisions directly to the relevant OpenSpec artifacts.

**Editing rules:**

- Keep edits minimal and targeted
- Preserve existing structure and formatting conventions
- Keep scope and summary artifacts concise
- Keep design artifacts focused on chosen technical direction, not unresolved branches
- Keep requirements artifacts normative with `SHALL` or `MUST` language and at least one scenario per requirement when that artifact type uses requirement/scenario structure
- Keep task or checklist artifacts in the format required by the current schema and existing file conventions

**Coherence rules:**

- If a decision changes behavior, update specs first, then align design and tasks
- If a decision changes lifecycle or data shape, make design and spec language consistent
- Remove or rewrite stale text that now conflicts with the decision
- Preserve unresolved questions only when no final decision exists

### 6) Sanity-Check The Result

**After editing:**

- Re-read the modified artifacts for consistency
- Run:

  ```bash
  openspec status --change "<change-name>" --json
  ```

- Confirm that the edited artifacts still match the selected change and schema

If the project exposes an OpenSpec validation command and it is already part of the local workflow, run it. Otherwise do not invent one.

### 7) Report What Changed

**Summarize:**

- review file used
- decisions applied
- artifacts revised
- any questions skipped because they lacked a final decision
- any follow-up work the decisions imply

## Quick Mapping Heuristic

Use this shortcut when a decision clearly belongs to one layer, then map that layer to the current schema's actual artifacts from OpenSpec CLI output:

- **Behavioral contract or CLI outcome** -> spec
- **Chosen technical approach or state model** -> design
- **Work breakdown or tests** -> tasks
- **Capability or scope boundary** -> proposal

Most non-trivial decisions should update more than one artifact.

## Guardrails

- Do not revise artifacts before reading both the review document and the current OpenSpec artifacts.
- Do not treat `Proposal (Recommended)` as final when a later `DECISION` block exists.
- Do not assume the example question or decision formatting is present; inspect the document and discover its actual pattern when needed.
- Do not hardcode artifact names, locations, or counts; follow the current OpenSpec CLI output for the active change and schema.
- Do not guess the target change when the user, conversation, and decision file do not identify it clearly; list active changes and ask the user to choose.
- Do not fabricate decisions for unresolved questions.
- Do not silently skip conflicting text; rewrite it so the artifacts tell one coherent story.
- Do not change unrelated artifacts.
- Do not edit the review document itself unless the user explicitly asks.
- Prefer repository evidence and current OpenSpec CLI output over assumptions.
