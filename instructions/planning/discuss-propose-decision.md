# Prompt: Review a Discussion Document and Record Decisions

You are given a **discussion document** containing open questions about a software design or implementation. Each question presents options, pros/cons, and a recommended proposal. Your job is to **analyze the relevant codebase**, **make a decision for each question**, and **record your decisions inline** in the document.

---

## What is a discussion document?

A discussion document captures open design questions that need resolution before implementation can proceed. Each question typically has:

- A **"Why this matters"** section explaining why the question exists.
- **Named options** (Option A, Option B, ...) with pros and cons.
- A **Proposal (Recommended)** section with the author's suggested choice and rationale.

The document may also include a summary of recommended defaults at the top and deferred/remaining questions at the end.

---

## Your task

For each question in the discussion document:

1. **Read the codebase** to gather evidence for your decision. Do not decide based on the proposal text alone.
2. **Make a concrete decision** -- accept the proposal, reject it, or modify it.
3. **Write a blockquote** immediately after the question's last section (after the final Pros/Cons), using this format:

```markdown
> **DECISION: <one-line summary of what you decided>.**
> Rationale: <concrete justification grounded in codebase evidence>.
```

### Reviewer stance (critical, not deferential)

Treat every "Proposal (Recommended)" as a hypothesis to test, not an answer to trust.

- **Do not naively trust recommended solutions.** The recommendation may be incomplete, outdated, or incorrect.
- **Do not assume the discussion doc author is highly experienced.** Evaluate the proposal on evidence, not perceived authority.
- **Actively look for failure modes and hidden costs.** Check edge cases, migration impact, operational risk, and consistency with existing architecture before accepting.

---

## How to investigate before deciding

Before writing each DECISION, gather evidence from the codebase:

- **Find existing patterns.** If the question is about how to handle auth, errors, config, or API shape, search for how similar concerns are already solved elsewhere in the repo. Decisions that follow existing patterns are cheaper to implement and review.
- **Read the actual code, not just docs.** Check function signatures, model definitions, error hierarchies, config loading logic, and persistence schemas. The code is the ground truth; docs and proposals may lag behind.
- **Trace the data flow.** If the question is about a request/response shape, trace how the underlying domain model is constructed, consumed, and persisted. This reveals whether a proposed shape matches or conflicts with internal contracts.
- **Check for missing pieces.** If the proposal assumes something exists (e.g. a specific error type, a config key, a domain model field), verify whether it actually does. If it doesn't, note what needs to be added.

---

## How to write a good DECISION blockquote

### Structure

```markdown
> **DECISION: <verb> <what you decided>.**
> Rationale: <why, citing specific codebase evidence>.
```

The one-line summary should be unambiguous and standalone -- someone scanning the document should understand the choice without reading the rationale.

### Rationale principles

1. **Ground every claim in code.** Reference specific files, functions, line numbers, class names, field names, or config keys. Example: "The gateway-http `settings.py` already pops `bearer_token` from TOML with a `RuntimeWarning` (lines 30-36)."

2. **Explain why, not just what.** Don't just restate the proposal. Explain why the codebase evidence supports this choice over alternatives. Example: "Returning full `ClusterRunSummary` would conflict with the `max_list_items` guardrail" is better than "Option B is recommended."

3. **Call out implementation implications.** If the decision requires adding a new error type, modifying a schema, or changing an existing function, say so explicitly. Example: "This requires adding `class RunConflictError(FaceClusteringError)` to `errors.py` and updating `persist_run` to raise it for duplicate `run_id`."

4. **Cross-reference related decisions.** If one decision depends on or interacts with another, reference it. Example: "`run_conflict` (409) requires a new `RunConflictError` subclass (see Q11 decision)."

5. **Keep it concise.** A rationale should be 2-5 sentences. If you need more, the decision is probably under-analyzed -- go read more code first.

### What NOT to do

- Don't write a rationale that only says "the proposal makes sense" or "this is reasonable." That adds no information.
- Don't copy-paste the proposal's pros/cons as your rationale. Your rationale should add new evidence from the codebase, not restate what the proposal already said.
- Don't hedge with "we could also consider..." -- make a clear decision. Uncertainty means you need to read more code, not write more caveats.
- Don't skip questions or defer without explanation. If you genuinely want to defer, write a DECISION that says "Defer to v2" with a rationale for why deferral is safe.

---

## Handling different question types

### "Accept or reject the proposal" questions

Most questions have a recommended proposal. You can:
- **Accept**: `> **DECISION: Accept <proposal summary>.**`
- **Accept with modification**: `> **DECISION: Accept <proposal summary>, with <modification>.**`
- **Reject**: `> **DECISION: Reject the proposal; use <alternative> instead.**`

### "Single proposal, no options" questions

Some questions present only a proposal without named alternatives. For these:
- **Accept**: `> **DECISION: Accept the proposed <thing>.**`
- **Modify**: `> **DECISION: Accept with modification: <what changes>.**`

### "Deferred / remaining" questions

For questions explicitly flagged as deferrable:
- Decide whether deferral is safe or whether a decision is needed now.
- If deferring: `> **DECISION: Defer to v2; <why deferral is safe>.**`
- If deciding now: treat like any other question.

---

## Workflow summary

1. Read the entire discussion document first to understand the full scope and inter-dependencies between questions.
2. Explore the codebase broadly: domain models, existing services, error hierarchies, config loading, persistence, and any similar features already implemented.
3. For each question, in order:
   a. Read the relevant code paths.
   b. Evaluate the options against codebase evidence.
   c. Write the DECISION blockquote inline, immediately after the question's last section.
4. After all questions are decided, review for cross-references and consistency.
