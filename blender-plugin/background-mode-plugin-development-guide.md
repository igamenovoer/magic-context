# Background Mode Plugin Development Guide for Blender

## Overview

This guide provides proven architectural patterns and implementation strategies for developing Blender plugins that can run reliably in background mode (`blender --background`). Unlike GUI mode where Blender's UI event loop keeps the process alive, background mode requires explicit event loop management to prevent immediate process termination.

## Core Challenge

**Problem**: Blender background mode exits immediately after script execution unless you implement a proper keep-alive mechanism.

**Solution**: Drive the existing asyncio event loop from an external script while your plugin provides the event loop infrastructure.

## Plugin Architecture Requirements

### 1. Asyncio-Based Plugin Structure

Your Blender plugin must be built around asyncio to work properly in background mode:

```python
# your_plugin/__init__.py
import asyncio
import bpy
from . import async_loop, services

bl_info = {
    "name": "Your Background Plugin",
    "blender": (4, 0, 0),
    "category": "System",
}

def register():
    # Register your classes
    async_loop.ensure_async_loop()
    services.start_services()

def unregister():
    services.stop_services()
    async_loop.cleanup_async_loop()
```

### 2. Async Loop Management Module

Create a dedicated module for asyncio event loop management:

```python
# your_plugin/async_loop.py
import asyncio
import bpy
import logging

logger = logging.getLogger(__name__)

_loop = None
_modal_operator = None

def ensure_async_loop():
    """Ensure asyncio event loop is running"""
    global _loop, _modal_operator
    
    try:
        _loop = asyncio.get_event_loop()
    except RuntimeError:
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    
    # In GUI mode, use modal operator to drive the loop
    if bpy.context.window_manager:
        if not _modal_operator:
            bpy.ops.your_plugin.modal_operator('INVOKE_DEFAULT')
    else:
        # Background mode - external script will drive the loop
        logger.info("Background mode detected - external script should manage keep-alive loop")

def kick_async_loop():
    """Process pending asyncio tasks - called by external keep-alive script"""
    global _loop
    
    if not _loop:
        return True
    
    try:
        # Process all ready tasks
        if _loop.is_running():
            return False
            
        # Run until no more tasks are ready
        _loop._ready.clear()
        _loop._selector.select(0)  # Non-blocking select
        
        # Process ready callbacks
        while _loop._ready:
            handle = _loop._ready.popleft()
            if not handle._cancelled:
                handle._run()
        
        # Check if we have any pending tasks
        all_tasks = asyncio.all_tasks(_loop)
        if not all_tasks:
            return True  # No more tasks, can exit
            
        return False  # Still have tasks to process
        
    except Exception as e:
        logger.error(f"Error in kick_async_loop: {e}")
        return True  # Exit on error

def cleanup_async_loop():
    """Clean up the async loop"""
    global _loop, _modal_operator
    
    if _loop:
        # Cancel all pending tasks
        pending = asyncio.all_tasks(_loop)
        for task in pending:
            task.cancel()
        
        if pending:
            _loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        _loop.close()
        _loop = None
    
    _modal_operator = None

class YourPluginModalOperator(bpy.types.Operator):
    """Modal operator to drive asyncio loop in GUI mode"""
    bl_idname = "your_plugin.modal_operator"
    bl_label = "Your Plugin Modal Operator"
    
    _timer = None
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            kick_async_loop()
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        global _modal_operator
        _modal_operator = self
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
            self._timer = None
        
        global _modal_operator
        _modal_operator = None
```

### 3. Service Management Module

Organize your actual plugin functionality in a services module:

```python
# your_plugin/services.py
import asyncio
import logging

logger = logging.getLogger(__name__)

_services = {}

async def your_service_task():
    """Example service that runs continuously"""
    while True:
        try:
            # Your service logic here
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Service task cancelled")
            break
        except Exception as e:
            logger.error(f"Service error: {e}")
            await asyncio.sleep(5)  # Retry after error

def start_services():
    """Start all plugin services"""
    global _services
    
    loop = asyncio.get_event_loop()
    
    # Start your services as asyncio tasks
    _services['main_service'] = loop.create_task(your_service_task())
    
    logger.info("Plugin services started")

def stop_services():
    """Stop all plugin services"""
    global _services
    
    for name, task in _services.items():
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled service: {name}")
    
    _services.clear()
    logger.info("All plugin services stopped")

def is_service_running(service_name=None):
    """Check if services are running"""
    if service_name:
        task = _services.get(service_name)
        return task and not task.done()
    
    return any(not task.done() for task in _services.values())
```

## Background Mode Keep-Alive Script Pattern

### Generic Keep-Alive Script Template

```python
#!/usr/bin/env python3
"""
Generic Blender Background Mode Keep-Alive Script

This script keeps Blender running in background mode with your plugin active.
Customize the plugin import and cleanup sections for your specific plugin.

Usage:
    blender --background --python keep_alive.py

Environment Variables:
    YOUR_PLUGIN_CONFIG: Override plugin configuration
"""

import os
import time
import signal
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag to control the keep-alive loop
_keep_running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global _keep_running
    logger.info(f"Received signal {signum}, shutting down...")
    _keep_running = False
    
    # Try to gracefully shutdown your plugin services
    try:
        # Import your plugin and call cleanup
        import your_plugin
        your_plugin.services.stop_services()
        logger.info("Plugin services stopped")
    except Exception as e:
        logger.warning(f"Error stopping plugin services: {e}")
    
    # Allow time for cleanup
    time.sleep(0.5)
    logger.info("Shutdown complete")
    sys.exit(0)

# Install signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination

logger.info("Starting Blender background mode with your plugin")

try:
    # Start your plugin services
    logger.info("Initializing plugin services...")
    import your_plugin
    your_plugin.services.start_services()
    logger.info("Plugin services started successfully")
    
except Exception as e:
    logger.error(f"Failed to start plugin services: {e}")
    sys.exit(1)

logger.info("Blender running in background mode. Press Ctrl+C to exit.")
logger.info("Plugin is active and ready for operation.")

# Main keep-alive loop - drive the async event loop
try:
    # Give services time to start up
    logger.info("Waiting for services to fully initialize...")
    time.sleep(2)
    
    logger.info("Starting main background loop...")
    
    # Import the async loop module to drive the event loop
    from your_plugin import async_loop
    
    # Main keep-alive loop - drive the async event loop
    while _keep_running:
        # This is the heart of background mode operation
        # It drives the asyncio event loop, allowing services to run
        if async_loop.kick_async_loop():
            # The loop has no more tasks and wants to stop
            logger.info("Async loop completed, exiting...")
            break
        time.sleep(0.05)  # 50ms sleep to prevent high CPU usage
            
except KeyboardInterrupt:
    logger.info("Interrupted by user, shutting down...")
    _keep_running = False
except Exception as e:
    logger.error(f"Error in keep-alive loop: {e}")
    _keep_running = False

logger.info("Background mode keep-alive loop finished, Blender will exit.")
```

## Implementation Guidelines

### 1. Event Loop Management Best Practices

**✅ DO:**
- Use `kick_async_loop()` to drive the existing event loop
- Implement 50ms sleep intervals (responsive without high CPU usage)
- Handle asyncio task cancellation properly
- Use modal operators in GUI mode, external scripts in background mode

**❌ DON'T:**
- Create separate asyncio loops with `asyncio.new_event_loop()`
- Use `loop.run_forever()` in background mode
- Rely on status-based exit logic
- Use sleep intervals longer than 100ms

### 2. Service Architecture Patterns

**Modular Design:**
```python
# Separate concerns into different modules
your_plugin/
├── __init__.py           # Plugin registration
├── async_loop.py         # Event loop management
├── services.py           # Service lifecycle management
├── server.py             # Network server implementation
├── handlers.py           # Request/command handlers
└── utils.py              # Utility functions
```

**Service Lifecycle:**
```python
async def service_lifecycle():
    """Template for robust service implementation"""
    try:
        await service_startup()
        
        while not should_stop():
            try:
                await service_main_loop()
            except Exception as e:
                logger.error(f"Service error: {e}")
                await asyncio.sleep(1)  # Avoid rapid error loops
                
    except asyncio.CancelledError:
        logger.info("Service cancelled")
    finally:
        await service_cleanup()
```

### 3. Configuration Management

**Environment-Based Configuration:**
```python
# your_plugin/config.py
import os

class PluginConfig:
    def __init__(self):
        self.service_port = int(os.environ.get('YOUR_PLUGIN_PORT', '8888'))
        self.log_level = os.environ.get('YOUR_PLUGIN_LOG_LEVEL', 'INFO')
        self.auto_start = os.environ.get('YOUR_PLUGIN_AUTO_START', '0') == '1'
    
    @property
    def is_background_mode(self):
        """Detect if running in background mode"""
        import bpy
        return not bpy.context.window_manager
```

## Testing Strategies

### 1. Development Testing

```bash
# Test in GUI mode first
blender --python test_plugin.py

# Test background mode
blender --background --python keep_alive.py &

# Verify process is running
ps aux | grep blender | grep -v grep

# Test service functionality
curl http://localhost:8888/status  # or your service endpoint

# Test graceful shutdown
kill -TERM <blender_pid>
```

### 2. Automated Testing

```python
# test_background_mode.py
import subprocess
import time
import signal
import requests

def test_background_mode():
    """Test that plugin works in background mode"""
    
    # Start Blender in background
    process = subprocess.Popen([
        'blender', '--background', '--python', 'keep_alive.py'
    ])
    
    try:
        # Wait for startup
        time.sleep(5)
        
        # Test service is responding
        response = requests.get('http://localhost:8888/status', timeout=5)
        assert response.status_code == 200
        
        print("✅ Background mode test passed")
        
    finally:
        # Clean shutdown
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)

if __name__ == "__main__":
    test_background_mode()
```

## Common Patterns and Solutions

### 1. Network Services

```python
# For HTTP/TCP servers
async def start_server():
    server = await asyncio.start_server(
        handle_client, 
        host='127.0.0.1', 
        port=config.service_port
    )
    
    async with server:
        await server.serve_forever()
```

### 2. Periodic Tasks

```python
# For scheduled operations
async def periodic_task():
    while True:
        try:
            await do_periodic_work()
            await asyncio.sleep(60)  # Run every minute
        except asyncio.CancelledError:
            break
```

### 3. External Process Communication

```python
# For subprocess management
async def manage_external_process():
    process = await asyncio.create_subprocess_exec(
        'external_tool', '--option', 'value',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        stdout, stderr = await process.communicate()
        return process.returncode, stdout, stderr
    except asyncio.CancelledError:
        process.terminate()
        await process.wait()
        raise
```

## Troubleshooting Guide

### Process Exits Immediately
- **Check plugin registration**: Ensure `register()` is called
- **Verify async loop setup**: Check `ensure_async_loop()` is working
- **Look for exceptions**: Enable debug logging to see startup errors

### High CPU Usage
- **Check sleep intervals**: Ensure using `time.sleep(0.05)` or similar
- **Monitor task counts**: Excessive tasks may indicate runaway loops
- **Profile the keep-alive loop**: Use `cProfile` to identify bottlenecks

### Service Not Responding
- **Allow startup time**: Give services 2-5 seconds to initialize
- **Check port conflicts**: Ensure no other process uses the same port
- **Test in GUI mode first**: Validate functionality before background testing

### Memory Leaks
- **Cancel tasks properly**: Use `task.cancel()` in cleanup
- **Close resources**: Properly close files, sockets, etc.
- **Monitor with tools**: Use `memory_profiler` or similar tools

## Performance Optimization

### 1. Efficient Event Loop Driving
```python
def optimized_kick_async_loop():
    """Optimized version with minimal overhead"""
    if not _loop or _loop.is_closed():
        return True
    
    # Quick check for ready tasks
    if not _loop._ready and not _loop._scheduled:
        return len(asyncio.all_tasks(_loop)) == 0
    
    # Process ready tasks efficiently
    ready_count = 0
    while _loop._ready and ready_count < 100:  # Limit batch size
        handle = _loop._ready.popleft()
        if not handle._cancelled:
            handle._run()
        ready_count += 1
    
    return False
```

### 2. Resource Management
```python
class ResourceManager:
    """Manage plugin resources efficiently"""
    
    def __init__(self):
        self._resources = []
    
    def add_resource(self, resource):
        self._resources.append(resource)
        return resource
    
    async def cleanup_all(self):
        """Clean up all managed resources"""
        for resource in reversed(self._resources):
            try:
                if hasattr(resource, 'close'):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
            except Exception as e:
                logger.warning(f"Error closing resource: {e}")
        
        self._resources.clear()
```

## Advanced Patterns

### 1. Multi-Service Coordination
```python
class ServiceCoordinator:
    """Coordinate multiple services with dependencies"""
    
    def __init__(self):
        self.services = {}
        self.dependencies = {}
    
    async def start_service(self, name, service_func, depends_on=None):
        if depends_on:
            for dep in depends_on:
                if dep not in self.services:
                    raise ValueError(f"Dependency {dep} not found")
        
        task = asyncio.create_task(service_func())
        self.services[name] = task
        self.dependencies[name] = depends_on or []
        
        return task
    
    async def stop_all_services(self):
        """Stop services in reverse dependency order"""
        # Implementation left as exercise
        pass
```

### 2. Plugin Communication
```python
class PluginBridge:
    """Enable communication between multiple plugins"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.message_queues = {}
            self._initialized = True
    
    async def send_message(self, plugin_name, message):
        queue = self.message_queues.get(plugin_name)
        if queue:
            await queue.put(message)
    
    def register_plugin(self, plugin_name):
        self.message_queues[plugin_name] = asyncio.Queue()
        return self.message_queues[plugin_name]
```

## Summary

This guide provides a comprehensive framework for developing Blender plugins that work reliably in background mode. The key principles are:

1. **Event Loop Architecture**: Use asyncio and drive the event loop externally
2. **Modular Design**: Separate concerns into async_loop, services, and configuration modules
3. **Graceful Lifecycle**: Implement proper startup, operation, and shutdown phases
4. **Robust Error Handling**: Handle exceptions and cancellations appropriately
5. **Performance Optimization**: Use efficient event loop driving and resource management

By following these patterns, your Blender plugin will work seamlessly in both GUI and background modes, providing a reliable foundation for automation, server applications, and headless workflows.