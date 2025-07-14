# MCP Server Development with Pixi and VS Code

This guide explains how to set up any MCP (Model Context Protocol) server for development in VS Code using Pixi as the package manager.

## Prerequisites

- VS Code with MCP extension installed
- Pixi environment manager installed ([Installation Guide](https://pixi.sh/latest/))
- Your MCP server project with pixi configuration

## Project Setup

### 1. Initialize Pixi Project

If starting from scratch:

```bash
cd your-mcp-server
pixi init
```

### 2. Configure pyproject.toml

Add MCP server configuration to your `pyproject.toml`:

```toml
[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-64", "win-64"]

[tool.pixi.dependencies]
python = ">=3.11"
# Add your MCP server dependencies
fastmcp = ">=2.0.0"  # or your preferred MCP framework
click = "*"          # for CLI if needed
pydantic = "*"       # for data validation

[tool.pixi.pypi-dependencies]
your-mcp-server = { path = ".", editable = true }
# Add PyPI-only dependencies here

[tool.pixi.tasks]
# Define your MCP server startup task
mcp-server = "python -m your_mcp_server.main"
# Alternative approaches:
# mcp-server = "python -m your_mcp_server"
# mcp-server = "your-mcp-server-cli"  # if you have a console script
dev-server = "python -m your_mcp_server.main --debug"
```

## VS Code Configuration

### 1. Create MCP Configuration

Create `.vscode/mcp.json` in your project root:

```json
{
    "servers": {
        "your-mcp-server-dev": {
            "command": "pixi",
            "args": [
                "run",
                "mcp-server",
                "--port",
                "8080"
            ],
            "cwd": "/absolute/path/to/your-mcp-server"
        }
    },
    "inputs": []
}
```

### 2. Configuration Options

**Basic Configuration:**
```json
{
    "command": "pixi",
    "args": ["run", "mcp-server"],
    "cwd": "/path/to/project"
}
```

**With Arguments:**
```json
{
    "command": "pixi",
    "args": [
        "run", 
        "mcp-server",
        "--port", "8080",
        "--debug",
        "--config", "dev-config.json"
    ],
    "cwd": "/path/to/project"
}
```

**Multiple Environments:**
```json
{
    "servers": {
        "your-mcp-server-dev": {
            "command": "pixi",
            "args": ["run", "--environment", "dev", "mcp-server"],
            "cwd": "/path/to/project"
        },
        "your-mcp-server-test": {
            "command": "pixi",
            "args": ["run", "--environment", "test", "mcp-server"],
            "cwd": "/path/to/project"
        }
    }
}
```

## Key Configuration Elements

- **`"command": "pixi"`** - Uses pixi to manage the Python environment
- **`"args": ["run", "task-name", ...]`** - Runs predefined pixi task with arguments
- **`"cwd": "/absolute/path"`** - Sets working directory (must be absolute)
- **Task arguments** - Passed directly to your MCP server

## Common Pixi Task Patterns

### Simple Module Execution
```toml
[tool.pixi.tasks]
mcp-server = "python -m your_mcp_server"
```

### With Default Arguments
```toml
[tool.pixi.tasks]
mcp-server = "python -m your_mcp_server --port 8080"
dev-server = "python -m your_mcp_server --port 8080 --debug --reload"
```

### Using Console Scripts
```toml
[project.scripts]
your-mcp-server = "your_mcp_server.cli:main"

[tool.pixi.tasks]
mcp-server = "your-mcp-server"
```

### Environment-Specific Tasks
```toml
[tool.pixi.tasks]
mcp-server = "python -m your_mcp_server"

[tool.pixi.feature.dev.tasks]
dev-server = "python -m your_mcp_server --debug --reload"

[tool.pixi.feature.test.tasks]
test-server = "python -m your_mcp_server --test-mode"
```

## Development Workflow

### 1. Start Development Server

**Option A: VS Code MCP Extension**
1. Open VS Code in project directory
2. MCP extension detects configuration automatically
3. Start server through MCP extension interface

**Option B: Command Line Testing**
```bash
cd your-mcp-server
pixi run mcp-server
# or with arguments
pixi run mcp-server --debug --port 8080
```

### 2. Development Loop

1. **Make code changes** to your MCP server
2. **Restart server** (if not using auto-reload)
3. **Test in VS Code** with MCP client
4. **Iterate** and repeat

## Environment Management

### Multiple Environments
```toml
[tool.pixi.environments]
default = { solve-group = "default" }
dev = { features = ["dev"], solve-group = "default" }
test = { features = ["test"], solve-group = "default" }
prod = { solve-group = "prod" }
```

### Feature-Based Development
```toml
[tool.pixi.feature.dev.dependencies]
pytest = "*"
black = "*"
ruff = "*"
mypy = "*"

[tool.pixi.feature.dev.tasks]
test = "pytest tests/"
format = "black src/"
lint = "ruff check src/"
```

## Troubleshooting

### Common Issues

1. **"unexpected argument '--cwd' found"**
   ```json
   // ❌ Wrong - don't use --cwd in args
   "args": ["run", "--cwd", "/path", "mcp-server"]
   
   // ✅ Correct - use cwd property
   "args": ["run", "mcp-server"],
   "cwd": "/path/to/project"
   ```

2. **"Task not found"**
   - Check task name in `pyproject.toml` under `[tool.pixi.tasks]`
   - Ensure you're in the correct directory (`cwd`)
   - Verify pixi can find the task: `pixi task list`

3. **"Module not found"**
   - Ensure project is installed in editable mode
   - Check `[tool.pixi.pypi-dependencies]` includes your package
   - Verify with: `pixi run python -c "import your_mcp_server"`

4. **"Port already in use"**
   - Change port in VS Code configuration
   - Kill existing processes: `lsof -ti:8080 | xargs kill -9`

### Verification Commands

```bash
# Test pixi environment
pixi run python --version

# Test module import
pixi run python -c "import your_mcp_server; print('✅ Import successful')"

# List available tasks
pixi task list

# Test MCP server directly
pixi run mcp-server --help

# Check dependencies
pixi list
```

## Best Practices

### 1. Project Structure
```
your-mcp-server/
├── pyproject.toml          # Pixi and project configuration
├── pixi.lock              # Locked dependencies
├── .vscode/
│   └── mcp.json           # VS Code MCP configuration
├── src/
│   └── your_mcp_server/
│       ├── __init__.py
│       ├── main.py        # MCP server entry point
│       └── cli.py         # Optional CLI interface
└── tests/
```

### 2. Configuration Management
- Use environment variables for sensitive config
- Separate dev/test/prod configurations
- Document all available command-line arguments

### 3. Development Efficiency
- Use `--debug` and `--reload` flags during development
- Implement proper logging
- Add health check endpoints
- Use meaningful task names

### 4. Dependency Management
- Pin versions in production environments
- Use conda packages when available (faster installation)
- Keep PyPI dependencies minimal
- Regular `pixi update` for development

## Integration Examples

### FastMCP Framework
```toml
[tool.pixi.dependencies]
fastmcp = ">=2.0.0"

[tool.pixi.tasks]
mcp-server = "python -m your_mcp_server --host 0.0.0.0 --port 8080"
```

### MCP Python SDK
```toml
[tool.pixi.dependencies]
mcp = ">=1.0.0"

[tool.pixi.tasks]
mcp-server = "python -m your_mcp_server.server"
```

### Custom Framework
```toml
[tool.pixi.pypi-dependencies]
your-custom-mcp-lib = "*"

[tool.pixi.tasks]
mcp-server = "your-mcp-server start --config config.yaml"
```

## Advanced Configuration

### Auto-reload Development
```toml
[tool.pixi.feature.dev.dependencies]
watchdog = "*"

[tool.pixi.feature.dev.tasks]
dev-watch = "python -m your_mcp_server --reload --watch-dir src/"
```

### Docker Integration
```toml
[tool.pixi.tasks]
docker-build = "docker build -t your-mcp-server ."
docker-run = "docker run -p 8080:8080 your-mcp-server"
```

### Testing Integration
```toml
[tool.pixi.tasks]
test = "pytest tests/"
test-mcp = "python -m your_mcp_server.test_client"
integration-test = { depends-on = ["mcp-server"], cmd = "sleep 2 && pytest tests/integration/" }
```

This setup provides a robust, reproducible development environment for any MCP server using pixi and VS Code.
