# How to Call MCP Servers Programmatically

A practical guide for calling Model Context Protocol (MCP) servers programmatically using Python.

## Overview

Three tested approaches for MCP server interaction:

1. **🏆 MCP CLI Tools (RECOMMENDED)** - Official MCP protocol using the `mcp[cli]` package and Python SDK
2. **Direct TCP Socket Connection** - Direct communication with MCP servers via socket connections  
3. **Direct Tool Import (FastMCP only)** - Importing and calling FastMCP server tools directly in Python

## Approach 1: MCP CLI Tools (RECOMMENDED) ✅

### Installation
```bash
# Using uv (recommended)
uv add "mcp[cli]"

# Using pip
pip install "mcp[cli]"

# Using conda/mamba
conda install -c conda-forge mcp
```

### Server Startup
```bash
# MCP Inspector (interactive testing)
uv run mcp dev path/to/your/server.py

# Direct execution
uv run mcp run path/to/your/server.py --transport stdio

# With dependencies
uv run mcp dev path/to/your/server.py --with dependency1 --with dependency2
```

### Python SDK Implementation
```python
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def interact_with_mcp_server(server_path):
    server_params = StdioServerParameters(
        command="python",
        args=[server_path],
        env=None,
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            # Call a tool
            result = await session.call_tool("tool_name", {"param": "value"})
            print(f"Result: {result}")
            
            return True

# Usage
asyncio.run(interact_with_mcp_server("path/to/your/server.py"))
```

### Key Benefits
- ✅ Official protocol compliance
- ✅ Type safety and validation  
- ✅ Excellent debugging tools (MCP Inspector)
- ✅ Claude Desktop integration
- ✅ Future-proof compatibility
- ✅ Built-in error handling

### Claude Desktop Integration
```bash
# Install in Claude Desktop
uv run mcp install path/to/your/server.py \
  --name "Your MCP Server" \
  -v ENV_VAR=value
```

## Approach 2: Direct TCP Socket Connection ✅

### Implementation
```python
import socket
import json

def call_mcp_command(host="127.0.0.1", port=8080, command_type="command_name", params=None):
    """Call MCP server command via direct TCP connection"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)  # 10 second timeout
    
    try:
        sock.connect((host, port))
        
        # Prepare command
        command = {"type": command_type, "params": params or {}}
        sock.sendall(json.dumps(command).encode('utf-8'))
        
        # Receive response
        response_data = sock.recv(8192)
        response = json.loads(response_data.decode('utf-8'))
        
        # Validate response
        if response.get("status") == "error":
            raise Exception(f"MCP Error: {response.get('message')}")
            
        return response.get("result")
        
    except socket.timeout:
        raise Exception("MCP server timeout")
    except ConnectionRefusedError:
        raise Exception("MCP server not running")
    finally:
        sock.close()

# Usage
try:
    result = call_mcp_command(port=8080, command_type="get_info")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
```

### Use Cases
- Low-level protocol control
- Non-MCP-compliant servers
- Simple automation scripts
- Legacy system integration

## Approach 3: Direct Tool Import (FastMCP only) ✅

**Note**: This approach only works with FastMCP-based implementations. It requires direct access to the server code and is primarily useful for development and testing.

### Implementation
```python
import sys
sys.path.insert(0, "path/to/fastmcp/server/src")

# Only works with FastMCP servers
try:
    from your_mcp_module.server import tool_function1, tool_function2
    from mcp.server.fastmcp import Context
    
    # Create mock context
    ctx = Context()
    
    # Call tools directly
    result1 = tool_function1(ctx)
    result2 = tool_function2(ctx, param="value")
    
    print(f"Tool 1 result: {result1}")
    print(f"Tool 2 result: {result2}")
    
except ImportError as e:
    print(f"FastMCP server import failed: {e}")
```

### Limitations
- Only compatible with FastMCP framework
- Requires server source code access
- Development/testing use only
- No protocol validation

## Testing & Debugging

### Server Health Check
```bash
# Check if server is running
netstat -tlnp | grep YOUR_PORT

# Test with curl (if HTTP endpoint available)
curl -X POST -H "Content-Type: application/json" \
  -d '{"type": "command_name", "params": {}}' \
  http://localhost:PORT/endpoint
```

### Test Script Template
```python
#!/usr/bin/env python3
"""MCP Server Test Script"""

import asyncio
from your_mcp_client import interact_with_mcp_server, call_mcp_command

async def test_mcp_server():
    print("=== MCP Server Testing ===")
    
    # Test 1: MCP CLI approach (recommended)
    try:
        success = await interact_with_mcp_server("path/to/server.py")
        print(f"✅ MCP CLI test: {'SUCCESS' if success else 'FAILED'}")
    except Exception as e:
        print(f"❌ MCP CLI failed: {e}")
    
    # Test 2: Direct TCP connection
    try:
        result = call_mcp_command(port=8080, command_type="health_check")
        print(f"✅ Direct TCP test: {result}")
    except Exception as e:
        print(f"❌ Direct TCP failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
```

## Common Patterns

### Error Handling
```python
async def safe_mcp_call(session, tool_name, params):
    try:
        result = await session.call_tool(tool_name, params)
        return result
    except Exception as e:
        print(f"Tool '{tool_name}' failed: {e}")
        return None
```

### Batch Operations
```python
async def batch_mcp_calls(session, operations):
    results = []
    for tool_name, params in operations:
        result = await safe_mcp_call(session, tool_name, params)
        results.append(result)
    return results
```

### Connection Management
```python
class MCPClient:
    def __init__(self, host="127.0.0.1", port=8080):
        self.host = host
        self.port = port
    
    def call_command(self, command_type, params=None):
        return call_mcp_command(self.host, self.port, command_type, params)
```

## Best Practices

### Development
- Use **MCP Inspector** for interactive testing
- Implement proper error handling and timeouts
- Validate all inputs and outputs
- Use type hints for better IDE support

### Production
- Use MCP CLI tools for standard compliance
- Implement connection pooling for high-frequency calls
- Add monitoring and health checks
- Follow MCP protocol specifications

### Testing
- Test both success and error scenarios
- Validate tool schemas and responses
- Use automated testing for CI/CD
- Mock external dependencies

## Troubleshooting

### Common Issues
- **Connection Refused**: Server not running or wrong port
- **Timeout**: Server overloaded or hung
- **Protocol Errors**: Version mismatch or invalid requests
- **Import Errors**: Missing dependencies or wrong paths

### Debug Commands
```bash
# Check server process
ps aux | grep your_server

# Check port availability
netstat -tlnp | grep PORT

# Test basic connectivity
telnet localhost PORT
```

## Conclusion

**Recommended**: Use **MCP CLI Tools (Approach 1)** for standard-compliant, robust MCP server interaction with excellent developer experience.

**Alternative**: Use **Direct TCP Socket Connection (Approach 2)** for low-level control or non-MCP-compliant servers.

**Development Only**: Use **Direct Tool Import (Approach 3)** exclusively for FastMCP development and testing scenarios.

The MCP CLI approach provides the best balance of functionality, reliability, and future compatibility for most use cases.