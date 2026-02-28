# Prompt Template: Create a “Discuss Doc” that Lists Open Questions (with Proposals)

This file is a reusable prompt template for creating a **discussion document** that captures **open questions** about a feature/design/implementation, along with **options**, **pros/cons**, and a **recommended proposal** for each question.

The goal is to make the unknowns **explicit**, reduce churn, and provide a concrete basis for decisions **before** writing code.

---

## When to write this doc

Create a “list open questions” discuss doc when:

- You’re about to implement a feature but requirements/design details still feel fuzzy.
- You’ve read a proposal/design/spec/task list and noticed decision points that aren’t resolved.
- The work touches multiple surfaces (API, persistence, auth, config, ops, tests) and needs coordination.
- You want to de-risk implementation by forcing crisp choices early.

This doc is especially valuable for changes involving:

- public or semi-public contracts (HTTP APIs, CLI contracts, file formats),
- configuration and orchestration (env/TOML flags, service wiring),
- error semantics and status mapping,
- performance/limits (payload size, pagination, timeouts),
- backwards compatibility and migration strategies.

---

## What the doc should cover

At minimum, cover questions in these buckets (use only what’s relevant):

1) **Surface / contract shape**
   - Request/response schema, naming, versioning, resource structure.
2) **Behavior and invariants**
   - Validation rules, idempotency, determinism, ordering guarantees, concurrency semantics.
3) **Error semantics**
   - Stable error envelope, status code mapping, machine-readable codes, “details” stability policy.
4) **Security / auth**
   - Auth mechanism, which endpoints are protected, how health checks behave, where secrets live.
5) **Limits and performance**
   - Request-size guardrails, pagination strategy, streaming vs batch, timeouts, memory risks.
6) **Transport / formats**
   - File/archive formats, content types, upload mechanisms, compatibility with common clients.
7) **Configuration and operations**
   - Config keys, defaults, overrides, orchestration integration, ports, logs/state locations.
8) **Testing and docs**
   - What to unit-test vs integration-test, tutorial/usage examples, operator docs updates.

The doc should explicitly identify:

- **Blocking decisions** (must be decided before implementation),
- **Deferrable decisions** (safe to postpone to a later version),
- **Assumptions** (what you’re assuming if a decision is deferred).

---

## Naming and placement (context-dependent)

Where you save the discuss doc should depend on what the user gave you.

### Priority order

1) **If the user explicitly specifies an output path, use it.**
   - Do not “improve” or relocate it.
   - If the user gives a directory, place the file in that directory as `discuss-<ts>.md` (unless they also specify a filename).

2) **If the user points you at a directory or file to review, derive the feature directory and write:**

   - `<feature-dir>/discuss/discuss-<ts>.md`

   “Feature directory” here means: the directory that the repo structure uses to group the materials you’re reviewing by feature/purpose (for example, a folder that contains multiple docs/specs/tasks for the same feature).

   **How to infer `<feature-dir>` (heuristics):**
   - Start from the user-provided path(s), interpreted relative to the workspace/repo root.
   - If a path is a **file**, start from its parent directory.
   - If a path is a **directory**, start from that directory.
   - If multiple paths are provided, first compute their **nearest common ancestor directory** and treat that as the starting candidate.
   - **Special-case toolkits (when detected): respect their directory structure.**
     - **OpenSpec** (spec-driven change management):
       - Detect: a top-level `openspec/` directory exists and the referenced materials live under it.
       - Grouping rule:
         - If the referenced material is under `openspec/changes/archive/<archived-change-id>/...`, then `<feature-dir>` is `openspec/changes/archive/<archived-change-id>/`.
         - Else if the referenced material is under `openspec/archive/<archived-change-id>/...`, then `<feature-dir>` is `openspec/archive/<archived-change-id>/`.
         - Else if the referenced material is under `openspec/changes/<change-id>/...`, then `<feature-dir>` is `openspec/changes/<change-id>/`.
         - If the referenced material is under `openspec/specs/<capability>/...`, then `<feature-dir>` is `openspec/specs/<capability>/`.
       - Create the discuss doc under `<feature-dir>/discuss/` (create the directory if needed).
     - **Spec Kit (Speckit)** (spec-driven development toolkit):
       - Detect: a `.specify/` directory exists and/or feature artifacts live under a `specs/` directory.
       - Grouping rule:
         - If the referenced material is under `specs/<feature-id>/...`, then `<feature-dir>` is `specs/<feature-id>/`.
         - Else if the referenced material is under `.specify/specs/<feature-id>/...`, then `<feature-dir>` is `.specify/specs/<feature-id>/` (but prefer `specs/<feature-id>/` if both exist).
       - Create the discuss doc under `<feature-dir>/discuss/` (create the directory if needed).
   - Look for an existing “grouping pattern” by scanning upward a small number of levels:
     - The directory contains multiple feature-scoped artifacts (for example: `proposal.*`, `design.*`, `tasks.*`, `spec.*`, `README.*`), **or**
     - The directory contains a substructure that looks feature-scoped (for example: `specs/`, `design/`, `tasks/`, `discuss/`), **or**
     - The directory name looks like a feature slug and has siblings that are other feature slugs (a “many siblings under one parent” pattern).
   - Create the `discuss/` directory under `<feature-dir>` if it does not exist.

3) **If you cannot confidently infer a feature directory, save the doc next to the referenced materials.**
   - If the user pointed to a **file**: save in the same directory as that file.
   - If the user pointed to a **directory**: save in that directory.
   - Use filename: `discuss-<ts>.md`.

### Timestamp conventions

Pick one timestamp format and use it consistently within a repo:

- `YYYYMMDD-HHMMSS` (UTC recommended) for the filename.
- Always include ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`) in the document header for machine readability.

---

## Formatting template (copy/paste)

Use this template as the default structure:

```markdown
# Discussion: <short title>

- Feature/Change: `<name>`
- Timestamp: `<YYYY-MM-DDTHH:MM:SSZ>`
- Context: <1–2 sentences on why this doc exists and what inputs were reviewed>

<Optional: one paragraph stating key constraints / goals / non-goals>

---

## Summary of recommended defaults

1) <decision-like recommendation #1>
2) <decision-like recommendation #2>
...

<Keep this list short; link to question numbers below.>

---

## Q1) <question phrased crisply>

### Why this matters

- <bullet: what will go wrong if undecided>
- <bullet: what it affects (API, storage, tests, ops, etc.)>

### Options

**Option A: <name>**

- Pros:
  - ...
- Cons:
  - ...

**Option B: <name>**

- Pros:
  - ...
- Cons:
  - ...

### Proposal (Recommended)

<One paragraph: the recommended option and concrete details.>

### Pros / Cons (Proposal)

- Pros:
  - ...
- Cons:
  - ...

<Optional: “Notes / Follow-ups” subsection for any deferred details.>

---

## Q2) ...
```

### Guidelines for writing good questions

- Make the question **binary** or “small-multiple” (A vs B vs C), not a vague topic.
- Prefer questions that can be answered with a **clear decision statement**.
- Keep options mutually exclusive; avoid “Option C: combine A and B” unless it’s truly distinct.
- Include **implementation implications** when relevant (new error type, schema change, migration need).
- If a question depends on another, cross-reference it (e.g., “depends on Q7 auth decision”).

### Guidelines for pros/cons

- Pros/cons must be **specific** and **actionable**, not generic (“simpler”, “better”).
- Mention concrete tradeoffs:
  - compatibility risks,
  - operational complexity,
  - test surface area,
  - performance risks,
  - future extensibility.

### Guidelines for proposals

Proposals should be concrete:

- Include exact names/keys where relevant (endpoint paths, config keys, error codes).
- Specify defaults numerically where needed (limits, timeouts, page sizes).
- Call out what is **stable contract** vs **best-effort** (especially for `details` fields).

---

## Quality checklist (before you finalize)

- The doc has a short **Summary of recommended defaults** at the top.
- Every question has:
  - “Why this matters”
  - Options (A/B/…)
  - Proposal (recommended)
  - Pros/cons for the proposal
- Blocking vs deferrable decisions are explicit.
- The doc is readable by someone who hasn’t been in your head:
  - minimal jargon
  - concrete examples where helpful
  - no missing “Option X” descriptions

---

## Optional next step: decision capture

If you want a structured follow-up, have a reviewer (human or agent) read this doc, investigate the codebase, and write inline decision blockquotes like:

```markdown
> **DECISION: Accept Option B (<short summary>).**
> Rationale: <codebase-grounded justification>.
```

This keeps the discuss doc as both:

- a record of open questions + reasoning, and
- a living decision log that implementation can follow.
```
