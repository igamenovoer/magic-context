# How to Manage Pixi Environments with Submodules

This guide explains how to work with pixi environments when your workspace contains submodules that have their own `pixi.toml` configurations.

## Problem Statement

When you have a workspace with submodules that use pixi, you may want to work from the workspace root without constantly navigating into submodule directories. Currently, pixi doesn't support direct environment aliasing, but there are several effective workarounds.

**Example Scenario**: You have a workspace with a submodule at `path/to/submodule` that has its own pixi environment, and you want to access it from the workspace root as if it were a local environment alias.

## Current Limitations

- **No Environment Aliasing**: Pixi doesn't support creating aliases like `main-dev` that point to external pixi environments
- **No True Workspace Support**: Full workspace functionality (like Cargo or pnpm workspaces) is still under development
- **Environment Isolation**: Each `pixi.toml` creates its own isolated environment in `.pixi/envs/`

## Solution Approaches

### 1. Use `--manifest-path` (Recommended)

Work from workspace root while targeting submodule environments:

```bash
# Run commands in submodule environment from workspace root
pixi run --manifest-path ./path/to/submodule/pixi.toml <command>

# Shell into submodule environment
pixi shell --manifest-path ./path/to/submodule/pixi.toml

# Install packages in submodule environment
pixi add --manifest-path ./path/to/submodule/pixi.toml <package>

# Run specific tasks
pixi run --manifest-path ./path/to/submodule/pixi.toml <task-name>
```

### 2. Create Wrapper Tasks in Root `pixi.toml`

Add convenient tasks to your workspace root `pixi.toml`:

```toml
[tasks]
# Generic wrappers
submodule-shell = "pixi shell --manifest-path path/to/submodule/pixi.toml"
submodule-run = "pixi run --manifest-path path/to/submodule/pixi.toml"
submodule-install = "pixi add --manifest-path path/to/submodule/pixi.toml"

# Specific development tasks (replace with actual task names from submodule)
dev-task1 = "pixi run --manifest-path path/to/submodule/pixi.toml task1"
dev-task2 = "pixi run --manifest-path path/to/submodule/pixi.toml task2"
dev-task3 = "pixi run --manifest-path path/to/submodule/pixi.toml task3"

# Environment activation with specific commands
submodule-python = "pixi run --manifest-path path/to/submodule/pixi.toml python"
submodule-jupyter = "pixi run --manifest-path path/to/submodule/pixi.toml jupyter lab"
```

Usage:
```bash
pixi run submodule-shell    # Opens shell in submodule environment
pixi run dev-task1          # Runs task1 in submodule environment
pixi run submodule-jupyter  # Starts Jupyter in submodule environment
```

### 3. Duplicate Dependencies with Features

If submodule dependencies are compatible, use pixi features:

```toml
[project]
name = "your-workspace"
channels = ["conda-forge", "pytorch", "nvidia"]
platforms = ["win-64", "linux-64", "osx-64", "osx-arm64"]

[dependencies]
python = ">=3.8"

# Submodule dependencies as a feature
[feature.submodule.dependencies]
# Copy dependencies from submodule's pixi.toml
dependency1 = "*"
dependency2 = ">=1.0.0"
dependency3 = "*"

# Development tools feature
[feature.dev.dependencies]
jupyter = "*"
ipykernel = "*"
black = "*"
pytest = "*"

[environments]
default = { features = [] }
submodule = { features = ["submodule"] }
dev = { features = ["submodule", "dev"] }
```

Usage:
```bash
pixi shell --environment submodule
pixi run --environment dev jupyter lab
```

### 4. Shell Aliases and Functions

Add to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
# Generic pixi wrapper for submodule
function pixi-submodule() {
    local workspace_root="/path/to/your/workspace"
    pixi --manifest-path "$workspace_root/path/to/submodule/pixi.toml" "$@"
}

# Specific shortcuts
function submodule-shell() {
    local workspace_root="/path/to/your/workspace"
    pixi shell --manifest-path "$workspace_root/path/to/submodule/pixi.toml"
}

function submodule-jupyter() {
    local workspace_root="/path/to/your/workspace"
    pixi run --manifest-path "$workspace_root/path/to/submodule/pixi.toml" jupyter lab
}

# Windows PowerShell equivalent
function pixi-submodule {
    param([Parameter(ValueFromRemainingArguments)]$args)
    $workspaceRoot = "C:\path\to\your\workspace"
    pixi --manifest-path "$workspaceRoot\path\to\submodule\pixi.toml" @args
}
```

### 5. Development Workflow Example

For a typical development workflow with a submodule:

```toml
# In workspace root pixi.toml
[tasks]
# Setup tasks
setup-submodule = "pixi install --manifest-path path/to/submodule/pixi.toml"
update-submodule = "pixi update --manifest-path path/to/submodule/pixi.toml"

# Development tasks
submodule-dev = "pixi shell --manifest-path path/to/submodule/pixi.toml"
submodule-notebook = "pixi run --manifest-path path/to/submodule/pixi.toml jupyter lab --ip=0.0.0.0 --port=8888"

# Project-specific tasks (adapt to your submodule's actual tasks)
run-main = "pixi run --manifest-path path/to/submodule/pixi.toml python main.py"
run-tests = "pixi run --manifest-path path/to/submodule/pixi.toml pytest"
run-scripts = "pixi run --manifest-path path/to/submodule/pixi.toml python scripts/process_data.py"

# Testing
test-submodule = "pixi run --manifest-path path/to/submodule/pixi.toml pytest tests/"
```

## Best Practices

1. **Use Descriptive Task Names**: Prefix tasks with the submodule name (e.g., `submodule-*` or `myproject-*`)
2. **Group Related Tasks**: Organize tasks by functionality (setup, development, testing, etc.)
3. **Document Task Purpose**: Add comments in `pixi.toml` explaining what each task does
4. **Keep Environments Separate**: Don't merge incompatible dependencies into the root environment
5. **Version Pin Critical Dependencies**: Ensure reproducibility across different environments
6. **Use Consistent Naming**: Establish a naming convention for all submodule-related tasks

## Future Developments

The pixi team is developing proper workspace support that will provide:
- Automatic dependency resolution across workspace members
- Shared environments with isolated package sets
- Better integration between main projects and submodules

## Troubleshooting

### Common Issues

1. **Path Issues on Windows**: Use forward slashes or properly escape backslashes in task definitions
2. **Environment Not Found**: Ensure the submodule's `pixi.toml` exists and is valid
3. **Permission Issues**: Make sure you have write access to both workspace root and submodule directories

### Debugging Commands

```bash
# Check if submodule environment is properly configured
pixi info --manifest-path path/to/submodule/pixi.toml

# List available tasks in submodule
pixi task list --manifest-path path/to/submodule/pixi.toml

# Verify environment activation
pixi shell-hook --manifest-path path/to/submodule/pixi.toml
```

## References

- [Pixi Documentation - Environments](https://prefix-dev.github.io/pixi/dev/workspace/environment/)
- [Pixi GitHub Discussion - Monorepos and Workspaces](https://github.com/prefix-dev/pixi/discussions/387)
- [Pixi CLI Reference](https://prefix-dev.github.io/pixi/dev/reference/cli/)
- [Pixi Configuration Reference](https://prefix-dev.github.io/pixi/dev/reference/pixi_configuration/)
