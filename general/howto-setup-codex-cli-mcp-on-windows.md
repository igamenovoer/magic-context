# How to Configure MCP Servers for the Codex CLI

This guide provides the correct `config.toml` settings for running MCP (Model Context Protocol) servers with the Codex CLI, with specific workarounds required for Windows.

## TL;DR: The Solution

For immediate use, here are the working configurations for both Windows and Linux/WSL environments.

### Windows Configuration

On Windows, you must wrap the server command in `cmd.exe` and set environment variables inline.

```toml
# ~/.codex/config.toml

[mcp_servers.context7]
command = "bunx"
args = ["@upstash/context7-mcp@latest"]

[mcp_servers.tavily]
command = "bunx"
args = ["tavily-mcp@latest"]

[mcp_servers.tavily.env]
TAVILY_API_KEY = "<TAVILY-KEY>"
```

If npx/bunx does not work, you can also use Docker to run the MCP servers:

```toml

# note that mcp/context7 does not work in stdio mode, so we cannot use it with codex-cli
[mcp_servers.context7]
command = "docker"
args = ["run", "-i", "--rm", "--read-only", "acuvity/mcp-server-context-7:latest"]

[mcp_servers.tavily]
command = "docker"
args = ["run", "-i", "--rm", "-e", "TAVILY_API_KEY", "mcp/tavily"]

[mcp_servers.tavily.env]
TAVILY_API_KEY = "your_tavily_api_key_here"

```

### Linux/WSL Configuration

On Linux or the Windows Subsystem for Linux (WSL), the configuration is much simpler and works as expected.

```toml
# ~/.codex/config.toml

[mcp_servers.context7]
command = "npx"
args = ["-y", "@upstash/context7-mcp"]

[mcp_servers.tavily]
command = "npx"
args = ["-y", "tavily-mcp@latest"]
# The 'env' table works correctly on Linux/WSL
env = { "TAVILY_API_KEY" = "your_api_key_here" }
```

---

## Prerequisites & Explanations

If you are interested in the details, the following sections explain why the Windows configuration is more complex and provide necessary setup instructions.

### Installing Bun on Windows

The Windows examples use `bunx`, which is part of the Bun runtime. If you don't have Bun installed, open a PowerShell terminal and run the following command:

```powershell
powershell -c "irm bun.sh/install.ps1|iex"
```

After installation, close and reopen your terminal to ensure `bun` and `bunx` are in your system's PATH. For more details, refer to the [official Bun installation documentation](https://bun.sh/docs/installation).

### The Problem on Windows: Timeout Errors

The complex Windows configuration is necessary to work around several issues:
1.  **Process Spawning:** The Codex CLI struggles to spawn `bunx` or `npx` processes directly on Windows.
2.  **Environment Variables:** The CLI clears all inherited environment variables before starting the server, making the `[mcp_servers...env]` table ineffective on its own.
3.  **Argument Passing:** Arguments are not always passed correctly to the underlying shell.

### Why the Windows Solution Works

1.  **`command = "cmd.exe"`**: This forces the use of the Windows Command Prompt, providing a stable execution environment.
2.  **`args = ["/c", ...]`**: The `/c` flag tells `cmd.exe` to run the command that follows.
3.  **`set TAVILY_API_KEY=... && ...`**: This sets the environment variable *within the new shell's context* just before the server command runs, bypassing the environment clearing issue.

### Technical Root Cause in Codex Source Code

This behavior is caused by the server spawning logic in `mcp-client/src/mcp_client.rs`, which explicitly calls `.env_clear()` before starting the process.

```rust
// In codex-rs/mcp-client/src/mcp_client.rs

let mut child = Command::new(program)
    .args(args)
    .env_clear() // This line clears all inherited environment variables
    .envs(create_env_for_mcp_server(env)) // This only adds env vars from the config
    .spawn()?;
```


On Windows, this clean environment prevents tools like `bunx` from finding necessary system paths, leading to the timeout failures.
