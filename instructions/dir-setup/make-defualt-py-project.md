# Default Python Project Setup Guide

This guide instructs AI agents on how to set up a new Python project with the standard directory structure and default set of libraries used in this environment.

## Project Structure Setup

Create the following directory structure for the new project (replacing `MyPythonLib` with the actual project name).

- **Root Directory**: The base folder for the project.
- **.github/workflows/**: Directory for CI/CD automation workflows.
- **src/<project_name>/**: The main package source code directory.
- **extern/**: Container for external dependencies.
    - **tracked/**: For git submodules that are tracked.
    - **orphan/**: For local-only clones that are git-ignored.
- **scripts/**: Directory for CLI tools and entry point scripts.
- **tests/**: The test suite.
    - **unit/**: Fast, deterministic unit tests.
    - **integration/**: Tests involving external services or I/O.
    - **manual/**: Manual test scripts not collected by CI.
- **docs/**: Documentation source files.
- **context/**: Directory for AI context and memory (refer to context setup guides for details).
- **pyproject.toml**: Project configuration file (managed by Pixi).

## Environment Initialization & Management

Use **Pixi** to manage the project's environment and dependencies.

### Key Operations

1.  **Initialization**: Initialize a new project using the `pyproject.toml` format.
    ```bash
    pixi init --format pyproject .
    ```
    This creates a `pyproject.toml` file (or updates an existing one) to manage dependencies, which is the preferred standard for Python projects.

2.  **Adding Dependencies**:
    *   **PyPI Packages**: When installing Python packages, prefer using the PyPI registry (often via a `--pypi` flag) to ensure you get the latest versions.
    *   **Standard Set**: Ensure the project is equipped with the standard set of tools for:
        *   Markdown management (e.g., `mdutils`)
        *   Type checking (e.g., `mypy`)
        *   Linting & Formatting (e.g., `ruff`)
        *   Scientific Computing (e.g., `scipy`)
        *   Documentation (e.g., `mkdocs-material`)
        *   Interactive Computing (e.g., `ipykernel`)
        *   Visualization (e.g., `matplotlib`)

3.  **Pixi Command Cheat Sheet**:
    *   `pixi add <package>`: Add a Conda package.
    *   `pixi add --pypi <package>`: Add a PyPI package (preferred for Python libs).
    *   `pixi run <command>`: Run a command (script, test, tool) within the project's environment.
    *   `pixi shell`: Activate a shell session inside the project's environment.
    *   `pixi task add <name> "<cmd>"`: Save a frequent command as a task alias.
    *   `pixi list`: Show installed packages.
    *   `pixi info`: Show environment details.

4.  **Verification**: Periodically check that the environment is consistent and that key libraries can be imported successfully.

