---
name: openspec-ext-respond-to-review
description: Read an OpenSpec review report critically, evaluate the reviewer's proposals against the current change artifacts and repository context, and write developer-owned final decisions back into the review document. Use when the user explicitly mentions `openspec` or points to a path under `openspec/` while asking to examine a review report carefully, decide open questions, fill `DECISION` blocks, respond to an OpenSpec review file, or record final answers in an OpenSpec review document without yet revising the proposal, design, specs, or tasks.
---

# OpenSpec Extension: Respond To Review

Read an OpenSpec review report critically and turn its open questions into explicit developer decisions inside the review document itself.

**Goal:** capture final decisions in the review file while keeping later artifact revision as a separate follow-up step.

## Inputs

Accept one of:

- a review document path
- a change name plus a review document path
- a request that clearly refers to the current change's latest review report

## Output

Update the selected review document in place by filling the developer-owned final decision sections, usually `DECISION` blockquotes.

Report:

- which review file you used
- which questions were decided
- which questions remain unresolved
- whether a later artifact revision pass is still needed

## Example Prompts

- `take a look at openspec/changes/<change>/review/review-20260311-104348.md, examine the openspec review report carefully and critically, make your decisions and state that in the review report`
- `read the latest openspec review for add-agent-mailbox-protocol, decide each open question, and fill the DECISION blocks`
- `respond to this openspec review file, accept or reject the reviewer proposals with rationale, and write the final decisions back into the report`
- `review this openspec review document carefully, make the developer decisions, and leave artifact revision for a separate follow-up`
- `go through openspec/changes/<change>/review/review-*.md, choose the final answers for the blocking questions, and update the openspec review file in place`

Pointing to a file or directory under `openspec/` counts as the same trigger signal as explicitly saying `openspec`.

## Workflow

### 1) Select The Review Document

- If the user provides a file path, use it.
- Otherwise, infer the current change from conversation context.
- If you know the change but not the file, inspect `openspec/changes/<change>/review/` and choose the most recent `review-*.md`.
- If selection is still ambiguous, list the candidate review files and ask the user to choose.

**Always:** state which review file you are using.

### 2) Resolve The Change And Artifact Context

Prefer deriving the change directly from the review path when it matches:

```text
openspec/changes/<change-name>/review/<file>.md
```

Then gather the current OpenSpec context:

```bash
openspec status --change "<change-name>" --json
openspec instructions apply --change "<change-name>" --json
```

Use these outputs to determine:

- `changeDir`
- `schema`
- `contextFiles`
- current artifact status

Read every existing file referenced by `contextFiles`.
If a value is a glob such as `specs/**/*.md`, expand it on disk and read the matching files.

Also read:

- the selected review file
- any prior review files that the selected review explicitly relies on
- any repository files the review cites as concrete evidence when they materially affect the decision

### 3) Evaluate The Review Critically

Treat the review's `PROPOSED` blocks as reviewer recommendations, not as final answers.

For each open question:

- identify the actual decision being requested
- test the reviewer's recommendation against the current proposal, design, specs, tasks, and cited code paths
- check whether the recommendation fits existing repository patterns better than the alternatives
- decide whether to accept the proposal, reject it, or narrow or refine it

When useful, prefer decisions that:

- reduce undefined ownership or lifecycle edges
- align with existing repo patterns instead of introducing a second mechanism
- make v1 scope explicit instead of leaving behavior implied
- keep agent-facing contracts stable and discoverable
- separate "decide now" from "implement later"

If the evidence is insufficient to choose responsibly, do not force a decision. Leave the question unresolved and say why.

### 4) Update The Review Document

Edit the review document in place.

Default editing rule:

- fill only the developer-owned `DECISION` sections

If the review uses a different but clearly consistent final-decision pattern, follow that pattern rather than forcing the example format.

When updating a decision block:

- keep the decision summary short, direct, and answer-first
- keep the rationale concise and grounded in the current artifacts and codebase
- say explicitly when you are rejecting or modifying the reviewer's `PROPOSED` option
- preserve the reviewer's `PROPOSED` text unless the user explicitly asks for a broader rewrite

You may also update nearby review sections when they would otherwise become misleading after the decisions are filled in, for example:

- `Suggested Next Steps`
- a summary sentence that still describes a question as unresolved

Keep those cleanup edits minimal and local to the review document.

### 5) Sanity-Check The Result

After editing:

- re-read the updated review document
- confirm the filled decisions still match the selected change and current artifacts
- ensure any adjusted summary or next-step text is now consistent with the decisions

Run:

```bash
openspec status --change "<change-name>" --json
```

If helpful, use it as a final context sanity check.

### 6) Hand Off Cleanly

Summarize:

- the review file used
- the decisions made
- any unresolved questions that still need input
- whether the next step should be `$openspec-ext-revise-by-decision` to push those decisions into proposal, design, specs, or tasks

## Decision Heuristics

Use these quick checks when a review proposal looks plausible but not obviously correct:

- **Accept** when the proposal is well-supported by current artifacts and fits repo patterns cleanly.
- **Refine** when the proposal is directionally right but too broad for v1 or adds unnecessary mechanism.
- **Reject** when the proposal conflicts with existing architecture, introduces premature flexibility, or leaves ownership blurry.
- **Defer** only when the decision truly does not block coherent next steps.

## Guardrails

- Do not rubber-stamp the reviewer's `PROPOSED` blocks.
- Do not revise `proposal.md`, `design.md`, `specs/**/*.md`, or `tasks.md` unless the user explicitly asks; use `$openspec-ext-revise-by-decision` for that follow-up.
- Do not invent decisions for questions that remain materially ambiguous after reading the artifacts.
- Do not skip reading the current change artifacts and cited repository evidence before deciding.
- Do not silently change which review file or change you are responding to.
- Do not erase or overwrite reviewer-authored `PROPOSED` text unless the user asks for a broader cleanup.
- Do not rewrite the whole review when a targeted `DECISION` update is enough.
- If a `DECISION` block is already filled, treat it as authoritative unless the user explicitly asks you to reconsider or replace it.
