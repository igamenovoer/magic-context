---
name: openspec-ext-respond-to-review
description: Read an OpenSpec review report critically, evaluate the reviewer's proposals and findings against the current change artifacts and repository context, and write developer-owned final decisions/responses back into the review document. Use when the user explicitly mentions `openspec` or points to a path under `openspec/` while asking to examine a review report carefully, decide open questions, respond to findings, fill `DECISION` blocks, respond to an OpenSpec review file, or record final answers in an OpenSpec review document without yet revising the proposal, design, specs, or tasks.
---

# OpenSpec Extension: Respond To Review

Read an OpenSpec review report critically and turn its open questions and actionable findings into explicit developer-owned responses inside the review document itself.

**Goal:** capture final decisions and finding dispositions in the review file while keeping later artifact revision as a separate follow-up step.

## Inputs

Accept one of:

- a review document path
- a change name plus a review document path
- a request that clearly refers to the current change's latest review report

## Output

Update the selected review document in place by filling the developer-owned final decision sections for questions, usually `DECISION` blockquotes, and by adding or filling developer-owned responses for actionable findings when the review includes a `Findings` section.

Report:

- which review file you used
- which questions were decided
- which findings were accepted, refined, rejected, or deferred
- which questions remain unresolved
- which findings remain unresolved or need later artifact revision
- whether a later artifact revision pass is still needed

## Example Prompts

- `take a look at openspec/changes/<change>/review/review-20260311-104348.md, examine the openspec review report carefully and critically, make your decisions and state that in the review report`
- `read the latest openspec review for add-agent-mailbox-protocol, decide each open question, and fill the DECISION blocks`
- `respond to this openspec review file, accept or reject the reviewer proposals and findings with rationale, and write the final decisions back into the report`
- `review this openspec review document carefully, make the developer decisions, and leave artifact revision for a separate follow-up`
- `go through openspec/changes/<change>/review/review-*.md, choose the final answers for the blocking questions, respond to the findings, and update the openspec review file in place`

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
Treat the review's `Findings` section as reviewer-authored claims that also need explicit developer disposition when they are actionable.

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

For each actionable finding:

- identify the actual claim or requested correction
- test the finding against the current proposal, design, specs, tasks, and cited code paths
- decide whether to accept the finding, reject it, refine/narrow it, or defer it
- distinguish between "the reviewer correctly found a problem" and "the reviewer's proposed fix is the right next artifact change"

When reviewing findings, prefer responses that:

- preserve valid reviewer signal even when the proposed remediation is too broad
- separate "agree this is a real gap" from "agree this exact wording or task change is correct"
- avoid creating new scope just to satisfy a review comment mechanically

### 4) Update The Review Document

Edit the review document in place.

Default editing rule:

- fill developer-owned `DECISION` sections for questions
- respond to actionable findings with a short developer-owned response block placed directly after the finding when the review does not already provide one

If the review uses a different but clearly consistent final-decision pattern, follow that pattern rather than forcing the example format.

For findings that lack a developer-response placeholder, add one using a concise blockquote format such as:

```markdown
> **RESPONSE: Accept / Refine / Reject / Defer.**
> Rationale: <short developer-owned rationale grounded in the current artifacts/codebase>.
```

If the review already uses a different but consistent finding-response marker, reuse that marker instead of introducing a new one.

When updating a decision block:

- keep the decision summary short, direct, and answer-first
- keep the rationale concise and grounded in the current artifacts and codebase
- say explicitly when you are rejecting or modifying the reviewer's `PROPOSED` option
- preserve the reviewer's `PROPOSED` text unless the user explicitly asks for a broader rewrite

When updating or adding a finding response:

- keep the response answer-first and short
- say explicitly whether the finding is accepted, refined, rejected, or deferred
- make clear whether the finding changes the artifacts now, should be handled in a later revision pass, or is already satisfied
- preserve the reviewer-authored finding text unless the user explicitly asks for a broader cleanup

You may also update nearby review sections when they would otherwise become misleading after the decisions are filled in, for example:

- `Suggested Next Steps`
- a summary sentence that still describes a question as unresolved
- a summary sentence that still describes a finding as unaddressed

Keep those cleanup edits minimal and local to the review document.

### 5) Sanity-Check The Result

After editing:

- re-read the updated review document
- confirm the filled decisions still match the selected change and current artifacts
- confirm the finding responses are consistent with the selected change and current artifacts
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
- the finding responses made
- any unresolved questions that still need input
- any unresolved findings that still need input or later artifact work
- whether the next step should be `$openspec-ext-revise-by-decision` to push those decisions into proposal, design, specs, or tasks

## Decision Heuristics

Use these quick checks when a review proposal looks plausible but not obviously correct:

- **Accept** when the proposal is well-supported by current artifacts and fits repo patterns cleanly.
- **Refine** when the proposal is directionally right but too broad for v1 or adds unnecessary mechanism.
- **Reject** when the proposal conflicts with existing architecture, introduces premature flexibility, or leaves ownership blurry.
- **Defer** only when the decision truly does not block coherent next steps.

For findings:

- **Accept** when the finding identifies a real gap or inconsistency that should flow into artifact revision.
- **Refine** when the finding is directionally correct but overstates the problem, the severity, or the needed fix.
- **Reject** when the finding conflicts with the actual artifacts, misreads the codebase, or asks for the wrong change.
- **Defer** only when the finding is plausible but cannot be resolved responsibly from the current artifacts and evidence.

## Guardrails

- Do not rubber-stamp the reviewer's `PROPOSED` blocks.
- Do not rubber-stamp the reviewer's findings; each actionable finding needs an explicit developer disposition when the review format includes a `Findings` section.
- Do not revise `proposal.md`, `design.md`, `specs/**/*.md`, or `tasks.md` unless the user explicitly asks; use `$openspec-ext-revise-by-decision` for that follow-up.
- Do not invent decisions for questions that remain materially ambiguous after reading the artifacts.
- Do not invent finding responses for findings that are too ambiguous to evaluate from the review, artifacts, and cited evidence.
- Do not skip reading the current change artifacts and cited repository evidence before deciding.
- Do not silently change which review file or change you are responding to.
- Do not erase or overwrite reviewer-authored `PROPOSED` text unless the user asks for a broader cleanup.
- Do not erase or rewrite reviewer-authored finding text unless the user asks for a broader cleanup.
- Do not rewrite the whole review when a targeted `DECISION` update is enough.
- Do not ignore the `Findings` section when the user asks to respond to the review as a whole.
- If a `DECISION` block is already filled, treat it as authoritative unless the user explicitly asks you to reconsider or replace it.
- If a finding already has a developer-owned response block, treat it as authoritative unless the user explicitly asks you to reconsider or replace it.
