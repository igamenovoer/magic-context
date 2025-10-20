# Project Principles and Constitution

## Targeting Python Users
- prioritize usability and readability for Python developers
- ensure seamless integration with popular Python libraries and frameworks
- code should be well-documented and easy to understand for new contributors, using python docstring and numpy doc style

## Coding Qualities
- prefer reusable, modular components, object-oriented design, and separation of concerns
- value clarity and maintainability over cleverness

## Python Style Guide

### General guidelines
- type annotate all parameters and return values for clarity and safety
- prefer absolute imports and group them by standard library, third-party, and local modules
- all Python code should pass `mypy` (type checking) and `ruff` (linting and formatting) checks before being considered complete

### Functional classes
- follow a strict object-oriented style with clear conventions for class design
- prefix member variables with `m_` and initialize them in `__init__` (default to `None` where appropriate)
- provide read-only access via `@property` and make changes through explicit `set_xxx()` methods (include validation in setters)
- keep constructors argument-free; use factory methods `cls.from_xxx()` for initialization

### Data model classes
- use framework-native field naming (no `m_` prefix) for compatibility with validators, serializers, and schema generation
 - recommended frameworks: `attrs` or `pydantic` for structured data models
 - default choice: use `attrs` for most data models; use `pydantic` for models intended for web request/response schemas and validation, unless otherwise specified
- define fields declaratively with types and defaults; avoid behavior-heavy methods in data models
- keep business logic outside data models; use separate service/helper classes for behavior
- document fields and validation rules using NumPy-style docstrings or framework-supported schema docs
- when using `attrs`: use `@define` and `field`; prefer `kw_only=True` to enforce keyword-only arguments unless otherwise specified
- for `attrs` fields, add helpful `metadata` (e.g., `{ "help": "..." }`) where appropriate to improve documentation clarity

## Python Runtime Environment
- avoid using system Python for development and testing; always prefer isolated environments
- determine the project's Python environment in this order:
  1. **Pixi environment** (preferred): check for `pixi.toml`, `pixi.lock`, or `pyproject.toml` with `[tool.pixi]` section; use `pixi info` to verify; run commands via `pixi run` or `pixi shell`
  2. **Virtual environment**: check for `.venv/`, `venv/`, or other virtualenv indicators; activate before running commands
  3. **System Python** (last resort): only use if no environment management is configured
- when writing documentation or scripts, specify which environment context is assumed

## Documentation Standards
- created markdowns should be readable and well-structured, using proper headings and formatting, prefer to have section numbers (x.y.z)
- coding examples should be included in the documentation where applicable
- in design docs, describe Python interfaces (classes, functions, APIs) using Python code blocks with docstrings to provide clear, concrete illustrations rather than prose-only descriptions

## Testing Requirements

We use three complementary test types: manual tests, unit tests, and integration tests.

**Determining test locations**: Before writing tests, establish the location conventions for the project:
1. First, check the project README.md or other project documentation for test location and organization instructions
2. If not documented, ask the user or use project-standard conventions if they are evident from existing test files
3. Common defaults (use as last resort): `tests/unit/`, `tests/integration/`, `tests/manual/` or similar; legacy projects may use `unittests/`

Use placeholders like `<unit_tests_root>`, `<integration_tests_root>`, `<manual_tests_root>` when referring to base test directories in the guidelines below.

### Manual tests (preferred for feature validation)
- provide manual test scripts for major functionality; designed for interactive use (Jupyter Notebook, console)
- do not use testing frameworks; avoid `if __name__ == "__main__":`; keep code mostly in global scope, using functions only for logical grouping
- prioritize visualization and inspectable outputs so humans can verify results clearly
- aim for reproducibility and simplicity; keep external resources configurable via environment variables if needed
- typical location pattern: `<manual_tests_root>/<feature_area>/test_<name>.py` or `<manual_tests_root>/test_<name>.py`

### Unit tests (targeted automation)
- use a framework like `pytest` (preferred) or `unittest`
- location pattern: `<unit_tests_root>/<subdir>/test_<name>.py` where `<subdir>` mirrors the module/feature being tested
- handle external resources via environment variables (e.g., `.env` with `TEST_RESOURCE_<NAME>`); log resolved endpoints/paths in test output
- when project uses Pixi, run in the appropriate environment (e.g., `pixi run -e dev`); otherwise follow the project's specified runner
- save generated artifacts under `<workspace>/tmp/unittests` unless otherwise specified

### Integration tests (system behavior over units)
- focus on end-to-end flows across modules/components; fewer but higher-value tests compared to unit tests
- may reuse patterns from unit tests for environment/resource handling and artifact output
- location pattern: `<integration_tests_root>/<subdir>/test_<name>.py` where `<subdir>` mirrors feature areas for discoverability
