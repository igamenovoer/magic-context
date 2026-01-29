# Pixi Basic Usage Guide

## Finding Packages and Versions

To find available versions of a package (e.g., to determine the latest stable Python):

```bash
pixi search python
```
*Output will list available versions. Use this to identify the "one minor version behind" target.*

## Adding Packages

### Conda Packages (Default)
Preferred for system dependencies, data science libraries (numpy, pandas), and tools requiring binary compilation.

```bash
pixi add python                 # Add latest compatible
pixi add python=3.12            # Add specific version
pixi add "numpy>=1.20"          # Add with constraint
```

### PyPI Packages
Use for pure Python packages not available in conda channels or when specific PyPI versions are needed.

```bash
pixi add --pypi requests
pixi add --pypi "flask==3.0.0"
```

### Development Dependencies
For tools used only during development (testing, linting, docs).

```bash
pixi add --feature dev pytest ruff
# Or simply add to default if no features are used, but features are cleaner.
# Current skill workflow adds to default for simplicity unless specified.
```

## Running Commands

Run commands within the project's environment without activating a shell:

```bash
pixi run python script.py
pixi run pytest
```

Define tasks in `pixi.toml` / `pyproject.toml` to run complex commands:
```bash
pixi run test   # runs the 'test' task defined in config
```

## Environment Management

- **Shell**: `pixi shell` - Spawns a new shell with the environment activated.
- **Info**: `pixi info` - Shows project configuration, platforms, and paths.
- **Clean**: `pixi clean` - Removes the `.pixi` environment directory (useful for fresh installs).

## Customizing Project Name

`pixi init` defaults to using the current directory name as the project/package name (e.g., creating `src/<dir_name>/`). If a different name is required:

1.  **Initialize**: Run `pixi init` as usual.
2.  **Rename Source**: Rename the generated directory in `src/`:
    ```bash
    mv src/<dir_name> src/<new_name>
    ```
3.  **Update Config**: Edit `pyproject.toml`:
    - Update `name = "<new_name>"` under `[project]`.
    - Update the dependency key under `[tool.pixi.pypi-dependencies]`:
      ```toml
      [tool.pixi.pypi-dependencies]
      <new_name> = { path = ".", editable = true }
      ```

