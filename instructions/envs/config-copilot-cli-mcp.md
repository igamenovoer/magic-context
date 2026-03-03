# Configure MCP for Copilot CLI (Tavily + Context7)

This guide covers MCP configuration for GitHub Copilot CLI with two providers:

1. **Tavily MCP**
2. **Context7 MCP**

For each provider, it shows both direct-key and environment-variable setups.

---

## File Location

Copilot CLI reads MCP config from:

- `~/.copilot/mcp-config.json`

If you use a custom config dir, pass:

- `copilot --config-dir /path/to/config`

---

## Tavily MCP

### Option A: Embed Tavily Key Directly in JSON

Use this when you want a self-contained config (simple but less secure):

```json
{
  "mcpServers": {
    "tavily": {
      "type": "stdio",
      "command": "bunx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "tvly-REPLACE_WITH_YOUR_REAL_KEY"
      }
    }
  }
}
```

---

### Option B: Use Environment Variables (Recommended)

Use shell env vars and reference them from JSON. This avoids hard-coding secrets in files.

```json
{
  "mcpServers": {
    "tavily": {
      "type": "stdio",
      "command": "bunx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "${TAVILY_API_KEY}"
      }
    }
  }
}
```

Set your key in shell startup (for example `~/.bashrc`):

```bash
export TAVILY_API_KEY="tvly-REPLACE_WITH_YOUR_REAL_KEY"
```

Then reload shell/env before launching Copilot:

```bash
source ~/.bashrc
copilot
```

> Important: Copilot/MCP processes inherit env at startup. If you change env vars later, restart Copilot CLI.

---

### Proxy Problem (Important)

If `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` are set, Tavily MCP may route API calls through that proxy.

In some environments, proxies rewrite or block outbound HTTPS requests and Tavily calls can fail with errors like:

- `Tavily API error: Request failed with status code 400`

This can happen even when your API key is valid.

#### Fix: Disable Proxy Only for Tavily MCP

Configure Tavily MCP with explicit no-proxy env values:

```json
{
  "mcpServers": {
    "tavily": {
      "type": "stdio",
      "command": "bunx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "${TAVILY_API_KEY}",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
        "NO_PROXY": "api.tavily.com,tavily.com,localhost,127.0.0.1",
        "no_proxy": "api.tavily.com,tavily.com,localhost,127.0.0.1"
      }
    }
  }
}
```

This keeps proxy settings for your normal shell/tools, but forces Tavily MCP to bypass proxy.

---

### Verify Tavily in Copilot CLI

Start a fresh Copilot session and run:

```text
/mcp show tavily
```

Then test with a prompt that forces Tavily use, e.g.:

```text
Use tavily_search to search: What is today's date? Return one sentence.
```

If it still fails:

1. Ensure the running session uses the intended config dir.
2. Restart all Copilot CLI processes.
3. Re-check key validity.
4. Re-check that Tavily MCP env contains the no-proxy overrides.

---

## Context7 MCP

### Option A: Embed Context7 Key Directly in JSON

```json
{
  "mcpServers": {
    "context7-mcp": {
      "type": "stdio",
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "CONTEXT7_API_KEY": "ctx7sk-REPLACE_WITH_YOUR_REAL_KEY"
      }
    }
  }
}
```

### Option B: Use Environment Variables (Recommended)

```json
{
  "mcpServers": {
    "context7-mcp": {
      "type": "stdio",
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"
      }
    }
  }
}
```

Set key in shell startup (for example `~/.bashrc`):

```bash
export CONTEXT7_API_KEY="ctx7sk-REPLACE_WITH_YOUR_REAL_KEY"
```

Reload shell/env before launching Copilot:

```bash
source ~/.bashrc
copilot
```

> Important: Copilot/MCP processes inherit env at startup. If you change env vars later, restart Copilot CLI.

### Optional: Disable Proxy for Context7 MCP

Use this only if Context7 requests are failing in proxied environments:

```json
{
  "mcpServers": {
    "context7-mcp": {
      "type": "stdio",
      "command": "bunx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": ""
      }
    }
  }
}
```

### Verify Context7 in Copilot CLI

Start a fresh Copilot session and run:

```text
/mcp show context7-mcp
```

Then test with a prompt that forces Context7 usage, for example:

```text
Use context7-mcp to find docs for React useEffect cleanup.
```
