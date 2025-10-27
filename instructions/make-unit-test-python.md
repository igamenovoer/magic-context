you are tasked to create a python unit test. Follow these guidelines:

- Use a testing framework like `pytest` or `unittest`.
- Location (discovery order and conventions):
  - Preferred: `tests/unit/(subdir)/test_(pick-a-name).py`
    - `(subdir)` mirrors the module or feature under test.
    - Check existing tests/subdirs to choose the right placement and naming.
  - Legacy (only if project still uses it): `unittests/(subdir)/test_(pick-a-name).py`
    - Prefer migrating to `tests/unit/`; treat `unittests/` as deprecated.
  - If the user specifies a path or naming, follow that instruction.
- External resources (file path, url, credentials, etc.) should be provided via environment variables:
  - Prefer `.env` entries named like `TEST_RESOURCE_<NAME>` with comments.
  - Reuse existing variables when possible.
  - If resources are used, print resolved endpoints/paths in test logs to avoid confusion.
- Environment execution:
  - If the user specifies an environment, use it.
  - If Pixi manages packages, prefer `pixi run -e dev` when available, otherwise `pixi run`.
  - Only if neither applies, fall back to system `python` (not recommended).
- Outputs:
  - Save generated artifacts under `<workspace>/tmp/unittests` unless the user specifies a different location.
