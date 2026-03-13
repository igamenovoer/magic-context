---
name: openspec-ext-explain
description: Create or update OpenSpec change explanation docs that capture developer-facing questions and answers under `openspec/changes/.../explain/`. Use when the user explicitly mentions `openspec` or points to a path under `openspec/` while asking to create, update, document, or maintain a Q&A, FAQ, explain note, or question-and-answer doc for an OpenSpec change based on user questions, implementation notes, review questions, or current chat context.
---

# OpenSpec Extension: Explain

Create or update a concise Q&A markdown document for an OpenSpec change and keep it under that change's `explain/` directory.

**Goal:** turn implementation questions into a durable, repo-grounded explain note for developers without relying on separate instruction files.

## Inputs

Accept any of:

- a change name
- a path under `openspec/changes/<change>/`
- a path to `openspec/changes/<change>/explain/`
- a request that clearly refers to the current OpenSpec change

The user may provide:

- one or more explicit questions to document
- a source to extract questions from, such as the current chat, an OpenSpec review, or implementation notes
- only a request to create the doc

## Output

Write or update one markdown file under the selected change's `explain/` directory.

Default filename:

- `qa-<what>.md`

For an OpenSpec change, prefer:

- `qa-<change-name>.md`

unless the user gives an explicit filename or there is already a more specific existing Q&A file that the request clearly targets.

Report:

- which file you created or updated
- whether the doc contains placeholder sections or concrete Q&A
- any questions that were intentionally left unanswered because the code/docs do not support them yet

## Required Document Shape

Use this structure:

```md
# Q&A: <what>

## Introduction

<1-2 sentences describing what the Q&A covers and who it is for>

**Related docs**
- `path/to/doc.md`

**Key entrypoints and modules**
- `path/to/code.py`

## <complete question sentence>
> Last revised at: `<UTC ISO-8601>` | Last revised base commit: `<git commit>`

- <concrete answer bullet>
```

Rules:

- Keep the title as `# Q&A: <what>`.
- Default audience is developers, including future maintainers.
- Include both `**Related docs**` and `**Key entrypoints and modules**`.
- Put per-question revision metadata immediately under each question heading.
- Do not add a global metadata block.
- Keep answers concrete: commands, defaults, artifacts, paths, behavior, constraints.
- Use 3-8 bullets or 1-3 short paragraphs per answer.

## Workflow

### 1. Resolve The Target Change With OpenSpec CLI

Prefer deriving the change from any explicit path under:

- `openspec/changes/<change>/...`

Other valid signals:

- an explicit change name from the user
- a request that clearly refers to the current OpenSpec change

If the target change is still ambiguous, inspect active changes first:

```bash
openspec list --sort recent --json
```

If needed, ask the user to choose the intended change.

Once the change is known, always gather authoritative context first:

```bash
openspec status --change "<change-name>" --json
openspec instructions apply --change "<change-name>" --json
```

Use these outputs to determine:

- `changeDir`
- `schema`
- `contextFiles`
- current artifact status

Treat the CLI output as the source of truth for where the change lives and which artifacts define it.

### 2. Resolve The Output Path

- If the user gives an explicit output file path, use it.
- If the user gives an `explain/` directory, create or update `qa-<change-name>.md` there unless the user specifies another filename.
- Otherwise, use `changeDir/explain/qa-<change-name>.md`.
- Create the `explain/` directory if it does not exist.

### 3. Gather Grounding Context

Read every existing file referenced by `contextFiles`.
If a `contextFiles` value is a glob such as `specs/**/*.md`, expand it on disk and read the matching files.

Also read:

- any existing Q&A doc that the request clearly targets
- relevant demo docs or README files
- the implementation entrypoints or modules directly implicated by the requested questions

Also collect:

- current base commit via `git rev-parse HEAD`
- current UTC timestamp via `date -u +%Y-%m-%dT%H:%M:%SZ`

Only read additional repository files when they are needed to answer the requested questions accurately.

### 4. Decide Which Questions To Record

Question capture policy:

- Do not automatically invent or extract questions unless the user explicitly asks for that.
- If the user provides explicit questions, document those questions.
- If the user explicitly asks to extract questions from the current chat, a review file, or notes, extract the relevant questions from that source.
- If the user asks to create the Q&A doc but provides no questions and does not ask for extraction, create placeholder question sections only.

Placeholder format:

- question heading: `## [question title]`
- answer body: `- [answer/code]`

When updating an existing Q&A doc:

- preserve unrelated questions and answers
- preserve each unchanged question's revision metadata
- only update the metadata line for questions whose content actually changed

### 5. Write Practical Answers

Each question must be:

- a complete sentence
- concise
- practical for maintainers or operators

Good examples:

- `How do I run the canonical persistence second step for this change?`
- `What data is persisted into canonical storage and what is skipped?`
- `How are demo-local ids translated into canonical identifiers?`

Answering rules:

- Ground answers in the repository's current docs and code, not guesses.
- Reference files with inline paths like `scripts/.../persist_canonical.py`.
- Prefer concrete artifacts, defaults, produced files, and failure modes.
- If a question cannot be answered from the code/docs, state:
  - what is missing
  - the minimal next step to confirm it
  - where that answer should be documented

### 6. Keep The Doc Self-Contained

- Include enough context in the introduction and answers that the reader does not need another instruction document to understand the Q&A file.
- Do not tell the reader to consult an external meta-instruction file for format rules.
- Summarize important repo-specific behavior directly in the Q&A answers when it matters.

### 7. Sanity-Check The Result

After writing:

- re-read the markdown file
- confirm the title, introduction, related-doc list, and entrypoint list are present
- confirm each question has exactly one metadata line directly beneath it
- confirm unchanged questions did not get fresh timestamps or commit hashes
- confirm placeholder sections were used when no explicit questions were provided

## Example Prompts

- `create an explain Q&A doc for openspec/changes/<change> and document these two questions`
- `under openspec/changes/<change>/explain, add Q&A entries for the questions we just discussed`
- `create a placeholder Q&A doc for this OpenSpec change`
- `extract the implementation questions from this review and turn them into an explain Q&A doc for the change`
- `update the existing explain Q&A with an answer to why frame_uri reuses the tmp keyframe path in this demo`

Pointing to a file or directory under `openspec/` counts as the same trigger signal as explicitly saying `openspec`.

## Guardrails

- Do not write the Q&A outside the change's `explain/` directory unless the user explicitly asks.
- Do not skip the OpenSpec CLI discovery step when the request is about an OpenSpec change; use `openspec status` and `openspec instructions apply` before deciding artifact context.
- Do not auto-extract questions from chat history, reviews, or notes unless the user explicitly asks for extraction.
- Do not fabricate unsupported answers when the code/docs are silent.
- Do not refresh all per-question metadata just because the file was touched.
- Do not turn the Q&A doc into a changelog or implementation diary.
- Do not add auxiliary files unless the user explicitly asks; this skill should usually need only `SKILL.md` and `agents/openai.yaml`.
