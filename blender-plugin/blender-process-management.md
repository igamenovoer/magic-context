# Blender Process Management for AI-Assisted Development

## HEADER
- **Purpose**: Guide for managing Blender processes in AI-assisted plugin development workflows
- **Status**: Active
- **Date**: 2025-07-10
- **Dependencies**: Blender 4.4+, Python services, asyncio patterns
- **Target**: AI assistants developing Blender plugins and automation tools

## Overview

Managing Blender processes properly is critical for AI-assisted development because assistants need to start Blender, read logs, test plugins, and manage background services without human intervention. This guide provides proven patterns for reliable Blender process management across different plugin types.

## Critical Process Management Rules

### Timeout Requirements (ESSENTIAL)
- **ALWAYS set timeout to 10 seconds maximum** when starting Blender with Bash commands
- **NEVER use long timeouts** (120+ seconds) - they waste time and provide no benefit
- **Background execution required**: Always start Blender with `&` to prevent blocking
- **Quick verification**: Wait exactly 10 seconds after start, then read console output

```bash
# CORRECT Pattern
Bash(command="blender &", timeout=10000)  # 10 seconds = 10000ms

# WRONG - Never do this  
Bash(command="blender &", timeout=120000)  # 2 minutes = 120000ms
```

### Rationale
Blender starts quickly (~5-10 seconds) but runs indefinitely in GUI mode. Long timeouts waste time since Blender doesn't exit on its own.

## Startup Patterns

### Standard Startup Sequence
```bash
# 1. Clean existing processes
pkill -f blender
sleep 2

# 2. Set environment variables (plugin-specific)
export PLUGIN_SERVICE_PORT=6688
export PLUGIN_AUTO_START=1
export BLENDER_ADDON_PATH="/path/to/addon"

# 3. Start Blender in background
/apps/blender-4.4.3-linux-x64/blender &

# 4. Wait for startup (critical timing)
sleep 10
```

### Startup Timing Breakdown
- **Blender loads**: ~3-5 seconds
- **Addons register**: ~1-2 seconds
- **Plugin initialization**: ~2-3 seconds
- **Total ready time**: ~10 seconds maximum
- **Same timing for both GUI and background modes**

## Background vs GUI Mode Differences

### Mode Detection
```python
import bpy
if bpy.app.background:
    print("Background mode (headless)")
else:
    print("GUI mode")
```

### Key Operational Differences

| Aspect | GUI Mode | Background Mode |
|--------|----------|-----------------|
| **Process Lifecycle** | Runs until killed | Exits immediately without blocking loop |
| **Context Access** | Full `bpy.context` available | `_RestrictContext` limitations |
| **Event Loop** | Modal operators work | Timer fallback required |
| **Plugin Startup** | Modal operator preferred | `bpy.app.timers` required |
| **Viewport Operations** | Screenshots available | Must use `bpy.ops.render.render()` |

### Background Mode Requirements
- **External process management** required to keep Blender alive
- **Timer-based asyncio processing** instead of modal operators
- **Context-safe API usage** (avoid `bpy.context` dependencies)
- **Manual event loop kicking** via `kick_async_loop()`

## Process Monitoring

### Check Process Status
```bash
# Check if Blender is running
ps aux | grep blender | grep -v grep

# Check if plugin services are listening (if using TCP/socket communication)
netstat -tlnp | grep <YOUR_PORT>

# Check multiple services
netstat -tlnp | grep -E "(port1|port2)"
```

### Runtime Health Monitoring
```python
# Plugin status via Python API (example)
def get_plugin_status():
    """Check if plugin is loaded and running"""
    return {
        'loaded': 'your_addon' in bpy.context.preferences.addons,
        'initialized': hasattr(bpy.types, 'YourAddonOperator'),
        'uptime': time.time() - start_time
    }

# Test connectivity (if plugin uses network communication)
def test_connectivity(port=6688):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0
    except:
        return False
```

## Common Failure Modes and Solutions

### 1. Modal Operator Context Restrictions
**Error**: `AttributeError: '_RestrictContext' object has no attribute 'view_layer'`

**Solution**: Implement timer fallback mechanism
```python
try:
    # Try modal operator first
    if hasattr(bpy.context, 'window_manager') and bpy.context.window_manager:
        result = bpy.ops.addon.modal_operator()
except (RuntimeError, AttributeError):
    # Fall back to app timer
    bpy.app.timers.register(kick_async_loop, first_interval=0.01)
```

### 2. Plugin Service Doesn't Respond
**Root Cause**: Asyncio tasks scheduled but event loop not processing (for async plugins)

**Solution**: Ensure event loop processing for async operations
```python
# In background mode (for async plugins)
while True:
    # Process async tasks if using asyncio
    stop_loop = process_async_tasks()
    if stop_loop:
        break
    time.sleep(0.01)
```

### 3. Port Binding Conflicts
**Detection**:
```python
def check_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        return True
    except OSError:
        return False
```

**Prevention**: Always clean up before starting new processes

### 4. Process Zombie/Orphan Issues
**Prevention**:
```bash
# Always clean up before starting
pkill -f blender
sleep 2
# Start fresh process
```

## Environment Configuration

### Plugin Configuration Pattern
```bash
# Port configuration (if plugin uses network communication)
export PLUGIN_PORT=6688                  # Default port
export PLUGIN_HOST="localhost"           # Default host

# Auto-start control
export PLUGIN_AUTO_START=1               # true/1 enables auto-start
export PLUGIN_DEBUG_MODE=1               # Enable debug logging

# Path configuration
export BLENDER_EXEC_PATH="/apps/blender-4.4.3-linux-x64/blender"
export PLUGIN_DATA_PATH="/path/to/plugin/data"
```

### Runtime Environment Detection
```python
import os
port = int(os.environ.get('PLUGIN_PORT', 6688))
auto_start = os.environ.get('PLUGIN_AUTO_START', '').lower() in ['1', 'true', 'yes']
debug_mode = os.environ.get('PLUGIN_DEBUG_MODE', '').lower() in ['1', 'true', 'yes']
```

## Testing Patterns

### Basic Plugin Testing
```python
# Test plugin loading
def test_plugin_loaded():
    addon_name = 'your_addon_name'
    loaded = addon_name in bpy.context.preferences.addons
    assert loaded, f"Plugin {addon_name} not loaded"

# Test plugin functionality
def test_plugin_operators():
    # Check if plugin operators are registered
    assert hasattr(bpy.ops, 'your_addon')
    assert hasattr(bpy.ops.your_addon, 'your_operator')

# Test network connectivity (if plugin uses networking)
def test_network_connectivity(port=6688):
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0
    except:
        return False
```

### Plugin Validation Workflow
```python
def validate_plugin_setup():
    tests = [
        ('Plugin Loading', test_plugin_loaded),
        ('Operators Registration', test_plugin_operators),
        ('Network Service', lambda: test_network_connectivity(6688))
    ]
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
```

## Process Lifecycle Management

### Startup Workflow
1. **Kill existing processes** (`pkill -f blender`)
2. **Set environment variables** (plugin-specific)
3. **Start Blender in background** (`blender &`)
4. **Wait exactly 10 seconds** (`sleep 10`)
5. **Verify plugin loading** (addon checks)
6. **Test plugin functionality**

### Shutdown Workflow
1. **Graceful plugin shutdown** (if supported)
2. **Kill Blender process** (`pkill -f blender`)
3. **Verify process termination**
4. **Clean up temporary files** (if needed)

### Hot Reload Development Pattern
```bash
# Update plugin files
cp -r addon_source/ ~/.config/blender/4.4/scripts/addons/plugin_name/

# Reload addon via Blender Python console or script
# No Blender restart required - saves ~15 seconds per iteration
import bpy
bpy.ops.preferences.addon_disable(module="plugin_name")
bpy.ops.preferences.addon_enable(module="plugin_name")
```

## Best Practices

### For AI Assistants
1. **Always use 10-second timeouts** when starting Blender
2. **Start Blender in background** with `&` to prevent blocking
3. **Check both GUI and background mode compatibility**
4. **Implement timer fallbacks** for background mode plugins
5. **Test plugin loading immediately** after startup
6. **Clean up processes** before starting new ones
7. **Monitor plugin health** throughout development sessions

### Development Environment
1. **Implement hot reload workflows** to speed iteration
2. **Test in both GUI and background modes**
3. **Monitor resource usage** (CPU, memory, open files)
4. **Log plugin events** for debugging
5. **Use version control** for plugin source code

### Error Recovery
1. **Implement automatic restarts** for failed plugins
2. **Provide clear error messages** with suggested solutions
3. **Fall back to alternative approaches** when primary methods fail
4. **Document known limitations** and workarounds

## Context-Safe Patterns

### Background Mode Plugin Initialization
```python
def ensure_plugin_startup():
    """Start plugin with fallback for background mode"""
    try:
        if hasattr(bpy.context, 'window_manager') and bpy.context.window_manager:
            # GUI mode - use modal operator
            result = bpy.ops.addon.start_plugin()
        else:
            raise RuntimeError("No valid window context")
    except (RuntimeError, AttributeError):
        # Background mode - fall back to timer
        bpy.app.timers.register(plugin_timer_callback, first_interval=0.01)
```

### API Usage Safety
```python
def safe_context_access():
    """Access bpy.context safely in both modes"""
    try:
        if hasattr(bpy.context, 'scene') and bpy.context.scene:
            return bpy.context.scene
        else:
            # Fall back to direct scene access
            return bpy.data.scenes[0]
    except (AttributeError, IndexError):
        return None
```

## Debugging Communication Channels

### Using MCP for Development Debugging

While your plugin may not be MCP-based, consider implementing a **temporary MCP debugging channel** during development. This allows AI assistants to investigate Blender's internal state directly without manually reading logs or guessing what's happening inside Blender.

#### Benefits of MCP Debugging Channel
- **Direct state inspection**: Query scene objects, materials, modifiers in real-time
- **Interactive debugging**: Execute test code and see immediate results
- **Error investigation**: Get detailed context when things go wrong
- **Development acceleration**: Skip the "restart Blender and check again" cycle

#### Simple MCP Debug Implementation
```python
# Add to your plugin for development only
def setup_debug_mcp():
    """Optional debugging channel for AI-assisted development"""
    if os.environ.get('PLUGIN_DEBUG_MCP', '').lower() == '1':
        # Start lightweight MCP server on debug port
        start_debug_mcp_server(port=7777)

def start_debug_mcp_server(port=7777):
    """Minimal MCP server for debugging"""
    # Implement basic handlers:
    # - execute_code: Run Python in Blender context
    # - get_scene_info: Return scene state
    # - get_object_info: Inspect specific objects
    pass
```

#### Usage Pattern
```bash
# Enable debug channel during development
export PLUGIN_DEBUG_MCP=1
blender &

# AI assistant can now:
# 1. Connect to debug port (7777)
# 2. Query scene state directly
# 3. Execute test code in Blender context
# 4. Investigate issues without restarting
```

#### When to Remove
Remove the debug MCP channel before releasing your plugin to users. It's purely a development tool for AI-assisted debugging and state inspection.

This guide provides battle-tested patterns for reliable Blender process management in AI-assisted development scenarios, based on real implementation experience and proven solutions.