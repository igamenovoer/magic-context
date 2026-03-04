---
name: openspec-ext-review-plan
description: Review an OpenSpec change (or a single OpenSpec change artifact file) for completeness, coherence, and alignment with existing system design; capture actionable feedback plus open questions; write a review report under the change directory (review/review-YYYYMMDD-HHMMSS.md).
---

# OpenSpec Extension: Review Plan

Review an OpenSpec change and produce a concrete, thread-friendly review report.

**Output:** Write to `CHANGE_DIR/review/review-YYYYMMDD-HHMMSS.md` (create `review/` if missing).

## Prerequisites

This skill depends on these companion skills for external reviewer invocation:

- `$claude-code-invoke-once`
- `$copilot-invoke-once`

Before running, verify both skills are discoverable/readable in the current environment.
If either dependency is missing, refuse to run this skill and return a clear error listing the missing skill(s).

## Workflow

### 1) Select The Change To Review

- If the user specifies a change name, use it.
- Otherwise, infer the “current” change from conversation context (the change the user is already discussing/implementing).
- If you still cannot determine it, list available changes and ask the user to pick one:

  ```bash
  openspec list --sort recent --json
  ```

  Show the most recent 3-6 changes with their task counts/status, mark the most recent as “(Recommended)”, then ask which to review.
  If your harness has an “AskUserQuestion” tool, use it; otherwise ask in plain text.

### 2) Load Change Artifacts (And Locate CHANGE_DIR)

1. Check schema + artifact status:

   ```bash
   openspec status --change "<change-name>" --json
   ```

2. Get change directory + context files:

   ```bash
   openspec instructions apply --change "<change-name>" --json
   ```

   Parse:
   - `changeDir` (this is `CHANGE_DIR`)
   - `contextFiles` (proposal/specs/design/tasks, depending on schema)

3. Read all available artifacts listed by `contextFiles`.
   If any entry is a glob (example: `specs/**/*.md`), expand it by listing files on disk.

### 3) Perform The Review

Cover at least:

- **Completeness:** Are required artifacts present for the schema? Are tasks concrete and ordered?
- **Internal coherence:** Do specs match design? Does design justify tasks? Any contradictions or missing invariants?
- **Design alignment with the existing system:**
  - Identify the impacted modules and existing architectural patterns in the repo.
  - Check whether the proposed design follows those patterns (naming, layering, config, error semantics, boundaries).
  - Call out mismatches with concrete file/path references and suggest a more “native” approach.
- **Risk and testability:** What could break? What needs unit vs integration tests?

When uncertain, be explicit about uncertainty and turn it into an open question or a suggested validation step.

### 4) Write The Review Report

Create `CHANGE_DIR/review/` if missing, then write `review-YYYYMMDD-HHMMSS.md` using a UTC timestamp.
Include an ISO-8601 timestamp in the header for machine readability.

## Review Report Template

Use this header (and keep it stable so reviews are easy to diff/scan):

```markdown
# OpenSpec Change Review: <change-name>

- Change: `<change-name>`
- Schema: `<schema-name>`
- Timestamp: `<YYYY-MM-DDTHH:MM:SSZ>`
- Artifacts reviewed:
  - <path>
  - <path>
  - ...

## How To Respond (Blockquote Protocol)

Use blockquotes so replies are visually distinct and easy to thread:

> AUTHOR-r1: <response to a specific bullet/question>
> REVIEWER-r1: <follow-up / acknowledgement>

Use `r2`, `r3`, ... for later rounds.

## Summary

- <2-6 bullets: what’s solid + what’s risky>

## Findings

### Must Fix / Blocking

- ...

### Should Fix / Important

- ...

### Nice To Have

- ...

## Design Alignment With Existing System

- Existing pattern(s) observed: <paths / modules>
- Proposed design alignment: <aligns/mismatches + concrete suggestions>

## Open Questions (With Proposed Defaults)

Follow the open-questions format from `magic-context/instructions/planning/discuss-list-open-questions.md`.

### Q1) <crisp question>

#### Why this matters

- ...

#### Options

**Option A: ...**

- Pros:
  - ...
- Cons:
  - ...

**Option B: ...**

- Pros:
  - ...
- Cons:
  - ...

#### Proposal (Recommended)

<recommended decision with concrete details>

#### Pros / Cons (Proposal)

- Pros:
  - ...
- Cons:
  - ...

## Suggested Next Steps

1) <concrete next step>
2) <concrete next step>
```

## External Review Agent Support

When the user asks to run this workflow via an external coding agent, support these commands:

- `claude-yunwu`
- `copilot`

Default reviewer selection:

- Default to `copilot` for external review.
- Use Claude Code (`claude-yunwu`) only when the user explicitly requests Claude.

Agent-specific requirements:

- For Claude Code, use the `claude-yunwu` wrapper with an Opus model.
- Invocation details for Claude Code are delegated to `$claude-code-invoke-once`; follow that skill for exact command/session/flag handling.
- For `copilot`, use the latest Opus model with high reasoning effort.
- Invocation details for `copilot` are delegated to `$copilot-invoke-once`; follow that skill for exact command/config composition.

## Guardrails

- Do not silently switch changes; always confirm the chosen change when selection was ambiguous.
- Prefer actionable feedback over vague opinions; cite concrete artifacts/paths.
- Keep “Open Questions” as a separate section and include a proposed default for each question.
- If the user asks to run this review via another coding agent, default to `copilot` unless the user explicitly requests Claude.
- If the user requests Claude Code, require the `claude-yunwu` wrapper + Opus model, and defer exact invocation mechanics to `$claude-code-invoke-once`.
- If the user requests `copilot`, require latest Opus model + high reasoning effort, and defer exact invocation mechanics to `$copilot-invoke-once`.
- If `$claude-code-invoke-once` or `$copilot-invoke-once` is unavailable, refuse to run and report missing dependencies instead of attempting ad-hoc fallback commands.
- Do NOT assume an external agent has this skill installed or can read files from this repo.
  - Construct a plain prompt that includes all instructions from this skill (paste the contents of this `SKILL.md`), plus the selected change name (if known), and ask the other agent to execute the workflow and produce the review report at the specified output path.
