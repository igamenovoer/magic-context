# Universal Project Rules

## Documentation
- When writing Markdown, do not hard-wrap normal paragraphs. Let Markdown viewers and editors handle line wrapping.

## Python
- Write Python in a strongly typed style. Tighter types are preferred over vague ones.
- Repo-owned Python should pass `mypy` after edits.
- Use NumPy-style docstrings for all public-facing Python functions, classes, and data models.

## C++
- Use Doxygen-style docstrings for all public-facing C++ functions, classes, and data models or structs.

## Feature Design
- This project is in active development and accepts breaking changes.
- When designing new features, do not spend effort on compatibility with previous iterations or external users unless explicitly requested.
- Favor a clear internal design over compatibility layers.
- If a change breaks another part of this repository, fix the dependent code in the same change so repo workflows continue to work together.
