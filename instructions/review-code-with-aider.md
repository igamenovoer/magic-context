you are tasked to review the source with `aider`, which is an AI-powered coding tool that uses LLM to generate code suggestions and improvements.

# Guidelines

## Creating the report template

- by default, create a report in `context/logs/code-reivew`, name it as `<timestamp>-<what-to-review>.md`, where `<timestamp>` is the current timestamp in `YYYYMMDD-HHMMSS` format, and `<what-to-review>` is a short description of the code being reviewed. If the user specifies a different code review report filepath, use that instead. Denote this as `report-filepath`
- the code report should be created empty at first, and `aider` will fill in the details
- review instructions is in `.magic-context/instructions/review-code-by-mem.md`, denote this as `review-instructions`

## Calling `aider`

you call `aider` from the command line, with the following format:

```
aider <file1> <file2> <file3> ... --message "LLM prompts, can include references to other file paths will be read/write by LLM"
```

specifically, for reviewing code without a specific requirement, you should call `aider` with the following message:

```
aider `review-instructions` `report-filepath` --message "based on `review-instructions`, review the following code files: <file1>, <file2>, <file3>, ... , and write the review report to `report-filepath`. The report should include suggestions on how to improve the code, and references to any online resources used."
```

if the user gives a specific requirement (about what the code should do, user will say "requirement:<requirement description or a requirement file>"), you should include that in the message, for example:

- given requirement description:
```
aider `review-instructions` `report-filepath` --message "based on `review-instructions`, review the following code files: <file1>, <file2>, <file3>, ... , with respect to the following requirement: <requirement description>, and write the review report to `report-filepath`. The report should include suggestions on how to improve the code, and references to any online resources used."
```