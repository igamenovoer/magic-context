# How to Use pyproject.toml to Include Pixi for Python Development with PyPI Publishing

This guide demonstrates how to integrate pixi package management within `pyproject.toml` for Python projects intended for PyPI publishing, providing the best of both conda and PyPI ecosystems.

## Overview

Pixi is a modern package manager that unifies conda and PyPI ecosystems. By integrating pixi configuration within `pyproject.toml`, you can:

- Manage both conda and PyPI dependencies in a single file
- Use system dependencies and non-Python tools via conda
- Maintain PyPI compatibility for publishing
- Leverage pixi's cross-platform environment management
- Use conda packages as dependencies for PyPI packages

## Basic Structure

A `pyproject.toml` with pixi integration contains three main sections:

1. **Standard Python project metadata** (`[project]`, `[build-system]`)
2. **Pixi workspace configuration** (`[tool.pixi.*]`)
3. **Development tooling configuration** (`[tool.ruff]`, `[tool.mypy]`, etc.)

## Complete Example

```toml
[project]
name = "my-python-package"
version = "0.1.0"
description = "A Python package with pixi integration"
authors = [
    {name = "Your Name", email = "you@example.com"}
]
readme = "README.md"
license = {file = "LICENSE"}
requires-python = ">=3.9"
dependencies = [
    "requests>=2.28.0",
    "click>=8.0.0",
]
keywords = ["python", "package", "pixi"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.urls]
Homepage = "https://github.com/yourusername/my-python-package"
Repository = "https://github.com/yourusername/my-python-package"
Issues = "https://github.com/yourusername/my-python-package/issues"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Pixi configuration starts here
[tool.pixi.workspace]
name = "my-python-package"
authors = ["Your Name <you@example.com>"]
channels = ["conda-forge"]
platforms = ["win-64", "linux-64", "osx-64", "osx-arm64"]
version = "0.1.0"

[tool.pixi.dependencies]
python = ">=3.9"
# System dependencies via conda
cmake = "*"
compilers = "*"

[tool.pixi.pypi-dependencies]
# Install your package in editable mode
my-python-package = { path = ".", editable = true }

[tool.pixi.tasks]
# Development tasks
install = "pip install -e ."
test = "pytest tests/ -v"
test-cov = "pytest tests/ --cov=src/my_python_package --cov-report=term-missing"
lint = "ruff check src/ tests/"
format = "ruff format src/ tests/"
typecheck = "mypy src/"
build = "python -m build"
clean = "rm -rf dist/ build/ *.egg-info/"

# Combined workflows
quality = "pixi run lint && pixi run typecheck && pixi run test"
dev-setup = "pixi run install"

[tool.pixi.environments]
default = {features = [], solve-group = "default"}
dev = {features = ["dev"], solve-group = "default"}
docs = {features = ["docs"], solve-group = "default"}
```

## Key Concepts

### 1. Python Version Management

Pixi automatically handles Python installation based on `requires-python`:

```toml
[project]
requires-python = ">=3.9"

# Pixi will install Python >=3.9 from conda-forge
[tool.pixi.workspace]
channels = ["conda-forge"]
```

### 2. Dependency Resolution

Dependencies are resolved in this order of priority:
1. **Conda dependencies** (`[tool.pixi.dependencies]`) - highest priority
2. **PyPI dependencies** (`[tool.pixi.pypi-dependencies]`)
3. **Project dependencies** (`[project.dependencies]`) - treated as PyPI deps

```toml
[project]
dependencies = ["requests"]  # PyPI by default

[tool.pixi.dependencies]
requests = "*"  # This overrides the PyPI version

[tool.pixi.pypi-dependencies]
some-package = "^1.0.0"  # Explicit PyPI dependency
```

### 3. Environment Management

Create multiple environments for different purposes:

```toml
[tool.pixi.environments]
default = {features = [], solve-group = "default"}
dev = {features = ["dev"], solve-group = "default"}
test = {features = ["test"], solve-group = "default"}
prod = {features = [], solve-group = "production"}
```

### 4. Cross-Platform Support

Specify platforms explicitly for reproducible builds:

```toml
[tool.pixi.workspace]
platforms = ["win-64", "linux-64", "osx-64", "osx-arm64"]
```

### 5. System Requirements

Declare system-level requirements:

```toml
[tool.pixi.system-requirements]
cuda = ">=11.0"
libc = ">=2.17"
```

## Best Practices

### 1. Project Structure

Organize your project for both PyPI and pixi compatibility:

```
my-package/
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── module.py
├── tests/
├── docs/
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

### 2. Gitignore Configuration

Add pixi-specific entries to `.gitignore`:

```gitignore
# Pixi
.pixi/
pixi.lock  # Include this in version control for reproducibility
```

### 3. Development Workflow

1. **Initial setup**:
   ```bash
   pixi install
   ```

2. **Development**:
   ```bash
   pixi run dev-setup
   pixi shell  # Activate environment
   ```

3. **Testing**:
   ```bash
   pixi run test
   pixi run quality  # Run all checks
   ```

4. **Building for PyPI**:
   ```bash
   pixi run build
   twine upload dist/*
   ```

### 4. CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
    - uses: actions/checkout@v4
    - uses: prefix-dev/setup-pixi@v0.4.1
      with:
        pixi-version: v0.50.1
    
    - name: Run tests
      run: pixi run quality
      
    - name: Build package
      run: pixi run build
```

### 5. Optional Dependencies and Features

Map PyPI optional dependencies to pixi features:

```toml
[project.optional-dependencies]
gui = ["PyQt5"]
cli = ["rich", "click"]

# Pixi automatically creates features for these
[tool.pixi.environments]
default = {features = [], solve-group = "default"}
gui = {features = ["gui"], solve-group = "default"}
cli = {features = ["cli"], solve-group = "default"}
```

## Advanced Features

### 1. Custom Build Systems

For packages requiring compilation:

```toml
[tool.pixi.dependencies]
cmake = "*"
ninja = "*"
compilers = "*"  # Provides gcc, clang, etc.

[tool.pixi.tasks]
build-ext = "python setup.py build_ext --inplace"
```

### 2. GPU Support

Handle CUDA dependencies elegantly:

```toml
[tool.pixi.system-requirements]
cuda = ">=11.0"

[tool.pixi.dependencies]
pytorch = "*"  # Automatically selects CUDA version

[tool.pixi.tasks]
test-gpu = "python -c 'import torch; print(torch.cuda.is_available())'"
```

### 3. Documentation Generation

Integrate documentation tools:

```toml
[project.optional-dependencies]
docs = ["mkdocs", "mkdocs-material"]

[tool.pixi.tasks]
docs-serve = "mkdocs serve"
docs-build = "mkdocs build"
docs-deploy = "mkdocs gh-deploy"
```

## Migration from Other Tools

### From Poetry

1. Convert `pyproject.toml` project section
2. Move `tool.poetry.dependencies` to `project.dependencies`
3. Add `tool.pixi.*` sections
4. Migrate tasks from Makefile/scripts

### From requirements.txt

1. Move dependencies to `project.dependencies`
2. Convert dev dependencies to `project.optional-dependencies`
3. Add pixi configuration sections

### From conda environment.yml

1. Move conda dependencies to `tool.pixi.dependencies`
2. Convert pip dependencies to `project.dependencies`
3. Migrate channels and platform specifications

## Common Patterns

### 1. Web Applications

```toml
[tool.pixi.dependencies]
nginx = "*"  # System web server
postgresql = "*"  # Database

[project.dependencies]
fastapi = "^0.100.0"
uvicorn = "^0.23.0"
```

### 2. Data Science Projects

```toml
[tool.pixi.dependencies]
python = ">=3.9"
numpy = "*"  # Use conda for performance
scipy = "*"

[project.dependencies]
pandas = "^2.0.0"  # Can mix with PyPI
```

### 3. CLI Tools

```toml
[project.scripts]
my-tool = "my_package.cli:main"

[tool.pixi.tasks]
install-tool = "pip install -e ."
test-cli = "my-tool --help"
```

## Publishing to PyPI

The integration doesn't affect PyPI publishing. Use standard tools:

```bash
# Build package
pixi run build

# Upload to PyPI
twine upload dist/*

# Or use modern tools
pixi add --pypi-dependencies twine
pixi run twine upload dist/*
```

## Troubleshooting

### Common Issues

1. **Dependency conflicts**: Check conda vs PyPI resolution order
2. **Platform issues**: Ensure all target platforms are specified
3. **Build failures**: Verify system dependencies are available

### Debug Commands

```bash
pixi info                    # Show workspace information
pixi list                    # List installed packages
pixi tree                    # Show dependency tree
pixi environment list       # List environments
```

## Resources

- [Pixi Documentation](https://pixi.sh/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [PyProject.toml Specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [Pixi GitHub Repository](https://github.com/prefix-dev/pixi)

## Conclusion

Integrating pixi with `pyproject.toml` provides a powerful development environment while maintaining full PyPI compatibility. This approach allows you to leverage the rich conda ecosystem for system dependencies while keeping your Python package easily installable via pip.

The configuration supports both simple and complex projects, from basic Python libraries to data science applications requiring system dependencies and GPU support.
