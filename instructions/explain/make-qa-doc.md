you are tasked to create a concise questions-and-answers (q&a) markdown doc for a development task/feature, based on questions that came up during implementation (e.g., chat log, pr review, notes).

determine the output path using the following rules:
- default filename is `qa-{{WHAT}}.md`
- if the user specifies an explicit output path (directory and/or filename), follow the user's command
  - if the user specifies only a directory, use that directory and the default filename
- otherwise, if the user indicates this q&a is about another doc (e.g., a design doc, plan doc), save it alongside that doc using the default filename
- otherwise, if the user indicates this q&a is about files in a specific directory (doc dir, source code dir, etc), save it in that directory using the default filename
- otherwise, save it in the workspace root using the default filename

`{{WHAT}}` should be a short, filesystem-safe slug (lowercase, hyphen-separated) derived from the referenced doc name/path basename, or otherwise summarized from the user prompt and relevant context info.

# q&a doc guidelines

## question capture policy

- do not automatically add questions to the q&a doc
- by default, only add a question/answer when the developer explicitly tells you to add it (typically by referencing a previous chat segment in the current dialog session)
- if the developer explicitly asks to record all questions, then include all relevant questions from the provided context
- if the user asks you to create the q&a doc but does not provide questions (and does not explicitly instruct you to extract them), create the doc with placeholder question sections only (use `TBD` for the question headings and answers)

## inputs (placeholders)

- project/repo: `{{PROJECT_NAME}}`
- optional output dir override: `{{OUTPUT_DIR}}` (if explicitly provided by user; may be empty)
- what slug: `{{WHAT}}` (derived)
  - if the user indicates this q&a is about a specific file/dir, prefer a slug based on that path basename first (unless it would conflict with an existing `qa-*.md` filename in the target dir)
  - otherwise, derive it from the user prompt and any relevant context info
- audience: `{{AUDIENCE}}` (defaults to "developers (including future maintainers)" unless the user specifies otherwise)
- per-question revision metadata:
  - last revised at (utc): `{{LAST_REVISED_AT}}` (iso-8601, e.g. `2025-01-31T12:34:56Z`)
  - last revised base commit: `{{LAST_REVISED_BASE_COMMIT}}` (the commit hash the answer was verified against)
  - each question has its own independent revision metadata
  - do not update a question’s revision metadata unless that specific question/answer content changes (do not auto-update due to unrelated commits or edits elsewhere in the doc)

## required structure

1. title
   - `# Q&A: {{WHAT}}`
   - if the user provides a preferred title, follow it

2. introduction
   - header: `## Introduction`
   - 1–2 sentences describing what this q&a covers and who it is for (`{{AUDIENCE}}`)
   - include two link lists (paths or urls):
     - `**Related docs**`
     - `**Key entrypoints and modules**`
   - do not add a global metadata block in the introduction

3. questions and answers
   - each question must be a complete sentence and concise
   - prefer practical questions (examples):
     - "How do I run …?"
     - "What does profile X change?"
     - "Is this from scratch or fine-tuning?"
     - "Where do the base artifacts come from?"
     - "Do I need PTQ first?"
   - for each question:
     - put the per-question metadata line immediately under the question header as a blockquote:
       - `> Last revised at: \`{{LAST_REVISED_AT}}\` | Last revised base commit: \`{{LAST_REVISED_BASE_COMMIT}}\``
     - only update that blockquote line when editing that question/answer
     - answer with 3–8 bullets or 1–3 short paragraphs
     - keep answers concrete (commands, defaults, and produced artifacts)
     - reference code/docs using inline paths like `path/to/file.py`

4. out-of-scope handling
   - if a question cannot be answered from existing code/docs:
     - state what is missing
     - state the minimal next step to confirm
     - state where it should be documented

## output skeleton

```md
# Q&A: {{WHAT}}

## Introduction

{{ONE_OR_TWO_SENTENCE_INTRO}}

**Related docs**
- `{{DOC_1}}`
- `{{DOC_2}}`

**Key entrypoints and modules**
- `{{CODE_FILE_1}}`
- `{{CODE_FILE_2}}`

## <question title>
> Last revised at: `{{LAST_REVISED_AT}}` | Last revised base commit: `{{LAST_REVISED_BASE_COMMIT}}`

- <answer/code>

...
```
