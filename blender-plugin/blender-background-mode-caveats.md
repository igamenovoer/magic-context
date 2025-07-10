# Blender Background Mode Caveats and Pitfalls

## HEADER
- **Purpose**: Comprehensive guide to avoid crashes and bugs when developing Blender plugins for background mode
- **Status**: Active  
- **Date**: 2025-07-10
- **Dependencies**: Blender 4.0+, background mode (`blender --background`)
- **Target**: AI assistants and developers building background-compatible Blender plugins

## Overview

Running Blender in background mode (`blender --background` or `blender -b`) introduces severe restrictions that can cause crashes, errors, and unexpected behavior in plugins designed for GUI mode. Many standard Blender API operations simply don't work or behave differently in background mode.

This guide catalogs all known caveats, limitations, common error patterns, and proven solutions based on real-world implementation experience.

## Critical Context Restrictions

### The `_RestrictContext` Problem

**Core Issue**: `bpy.context` becomes a restricted object in background mode with many members unavailable.

**Unavailable Context Members**:
```python
# These are None or unavailable in background mode:
bpy.context.window          # No window in headless mode
bpy.context.screen          # No screen interface  
bpy.context.area            # No UI areas
bpy.context.region          # No UI regions
bpy.context.space_data      # No 3D viewport, timeline, etc.
bpy.context.region_data     # No region-specific data
bpy.context.window_manager  # Limited or None
```

**Common Error**: `AttributeError: '_RestrictContext' object has no attribute 'view_layer'`

**Safe Alternatives**:
```python
# AVOID - breaks in background mode:
active_object = bpy.context.active_object
scene = bpy.context.scene

# USE INSTEAD - works in both modes:
active_object = bpy.context.view_layer.objects.active
scene = bpy.data.scenes[0]  # or access by name
```

### Context-Safe Access Pattern
```python
def safe_context_access():
    """Access context members safely in both modes"""
    try:
        if hasattr(bpy.context, 'scene') and bpy.context.scene:
            return bpy.context.scene
        else:
            # Fall back to direct data access
            return bpy.data.scenes[0] if bpy.data.scenes else None
    except (AttributeError, IndexError):
        return None
```

## Modal Operator Failures

### Problem: Modal Operators Cannot Start

**Root Cause**: Modal operators require a window manager and GUI event loop that don't exist in background mode.

**Error Pattern**: `RuntimeError: Operator bpy.ops.xxx.poll() failed, context is incorrect`

**Critical Example**:
```python
# This FAILS during addon registration in background mode:
bpy.ops.addon.modal_operator('INVOKE_DEFAULT')
# Error: '_RestrictContext' object has no attribute 'view_layer'
```

**Robust Fallback Pattern**:
```python
def ensure_service_startup():
    """Start service with fallback for background mode"""
    try:
        # Try modal operator first (GUI mode)
        if hasattr(bpy.context, 'window_manager') and bpy.context.window_manager:
            result = bpy.ops.addon.modal_operator('INVOKE_DEFAULT')
            if result == {'RUNNING_MODAL'}:
                return True
        # If modal fails or unavailable, fall through to timer
        raise RuntimeError("Modal operator not available")
    except (RuntimeError, AttributeError):
        # Background mode - use timer fallback
        if not bpy.app.timers.is_registered(service_timer_callback):
            bpy.app.timers.register(service_timer_callback, first_interval=0.01)
            print("Background mode: Using timer fallback")
            return True
    return False
```

## Viewport and Rendering Limitations

### Completely Unavailable Operations

**OpenGL and Viewport Operations**:
```python
# ALL of these FAIL in background mode:
bpy.ops.screen.screenshot()                    # RuntimeError: context incorrect
bpy.ops.screen.screenshot_area()               # RuntimeError: context incorrect  
bpy.ops.view3d.view_camera()                   # RuntimeError: context incorrect
bpy.ops.view3d.view_selected()                 # RuntimeError: context incorrect
bpy.ops.view3d.zoom_border()                   # RuntimeError: context incorrect
bpy.ops.render.opengl()                        # RuntimeError: no OpenGL context
```

### Safe Alternative: Use Rendering Engine
```python
def capture_image_background_safe(output_path, width=1920, height=1080):
    """Background-compatible image generation"""
    scene = bpy.context.scene
    
    # Configure render settings
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    
    # This works in background mode
    bpy.ops.render.render(write_still=True)
    
    print(f"Rendered image to: {output_path}")
```

### Camera Control Without View3D
```python
def set_camera_position_background_safe(location, rotation):
    """Background-compatible camera control"""
    scene = bpy.context.scene
    if scene.camera:
        scene.camera.location = location
        scene.camera.rotation_euler = rotation
    else:
        print("Warning: No active camera in scene")
```

## UI Registration Problems

### Problem: UI Components Cannot Be Registered

**Unavailable in Background Mode**:
- Panel registration and drawing
- Menu operations  
- Popup dialogs
- Status bar updates
- Progress indicators (visual)

**Safe UI Registration Pattern**:
```python
def register_ui_components():
    """Only register UI in GUI mode"""
    if not bpy.app.background:
        try:
            bpy.utils.register_class(MyPanel)
            bpy.utils.register_class(MyMenu)
            bpy.utils.register_class(MyOperator)
            print("UI components registered")
        except Exception as e:
            print(f"UI registration failed: {e}")
    else:
        print("Background mode: Skipping UI registration")

def unregister_ui_components():
    """Safely unregister UI components"""
    if not bpy.app.background:
        try:
            bpy.utils.unregister_class(MyPanel)
            bpy.utils.unregister_class(MyMenu) 
            bpy.utils.unregister_class(MyOperator)
        except Exception:
            pass  # Already unregistered
```

## Event Loop and Asyncio Issues

### Problem: No GUI Event Loop for Asyncio

**Core Issue**: Background mode has no GUI event loop to drive asyncio operations. Blender wants to exit immediately after script execution.

**BROKEN Pattern**:
```python
# DON'T DO THIS - blocks before server starts:
def register():
    start_server()
    if bpy.app.background:
        asyncio.run(keep_alive())  # BLOCKS HERE - server never ready
```

**Working Pattern - External Script Management**:
```python
# External script keeps Blender alive:
def background_main():
    """External script event loop"""
    while True:
        # Process asyncio events manually
        stop_loop = async_loop.kick_async_loop()
        if stop_loop:
            print("Shutdown signal received")
            break
        time.sleep(0.01)  # Prevents CPU spinning - ~100fps
```

**Critical Timeline Difference**:
```
BROKEN Internal Blocking:
1. register() called
2. start_server() called  
3. asyncio.run(background_main()) called ← BLOCKS HERE
4. Server never finishes initialization
5. No connections possible

WORKING External Script:
1. External script starts Blender
2. Blender loads with addon
3. register() enables addon and starts server
4. start_server() completes and returns
5. External script: while True: kick_async_loop()
6. Server processes requests via manual event loop
```

## Process Lifecycle Management

### Problem: Blender Exits Immediately

**Issue**: `blender --background <script>` executes script and exits unless kept alive.

**Solution 1: External Script Pattern (PROVEN)**:
```python
# launcher.py - External script
import subprocess
import time

def launch_blender_background_service():
    blender_path = "/path/to/blender"
    script_path = "background_service.py"
    
    # Start Blender process
    process = subprocess.Popen([
        blender_path, '--background', '--python', script_path
    ])
    
    print(f"Blender background service started (PID: {process.pid})")
    return process

# background_service.py - Inside Blender
def background_service_main():
    # Initialize plugin
    enable_plugin()
    
    # Keep alive loop
    while True:
        stop_loop = kick_async_loop()  # Process events
        if stop_loop:
            break
        time.sleep(0.01)

if __name__ == "__main__":
    background_service_main()
```

**Solution 2: Timer-Based Keep-Alive**:
```python
def background_keep_alive():
    """Timer function to keep Blender alive"""
    try:
        # Process asyncio events
        kick_async_loop()
        # Return interval for next call (0.01 = 100fps)
        return 0.01
    except Exception as e:
        print(f"Keep-alive error: {e}")
        return None  # Stop timer

# Register timer during addon initialization
if bpy.app.background:
    bpy.app.timers.register(background_keep_alive, first_interval=0.01)
```

## Common Error Patterns

### Error 1: Context Polling Failures
```
RuntimeError: Operator bpy.ops.xxx.poll() failed, context is incorrect
```
**Cause**: Operator requires GUI context that doesn't exist  
**Solution**: Use direct data access or context overrides

```python
# Instead of:
bpy.ops.object.mode_set(mode='EDIT')  # May fail

# Use:
if bpy.context.active_object:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
```

### Error 2: RestrictContext Attribute Errors
```
AttributeError: '_RestrictContext' object has no attribute 'view_layer'
```
**Cause**: Accessing context during restricted periods  
**Solution**: Check context availability before access

```python
def safe_view_layer_access():
    """Safe access to view layer"""
    try:
        if hasattr(bpy.context, 'view_layer') and bpy.context.view_layer:
            return bpy.context.view_layer
        else:
            # Fallback to scene's view layer
            scene = bpy.data.scenes[0]
            return scene.view_layers[0]
    except (AttributeError, IndexError):
        return None
```

### Error 3: Server Starts But No Connections
```
Server reports "started" but connections fail
```
**Cause**: `asyncio.run()` blocks before server accepts connections  
**Solution**: Use external script with manual event loop kicking

### Error 4: Immediate Process Exit
```
Blender starts and exits immediately in background mode
```
**Cause**: No infinite loop to keep process alive  
**Solution**: Add `while True:` loop with event processing

## Object and Mode Operations

### Mode Changes and Selection
```python
def enter_edit_mode_background_safe(obj):
    """Background-compatible mode switching"""
    try:
        # Ensure object is active
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # Mode change should work in both modes
        bpy.ops.object.mode_set(mode='EDIT')
        return True
    except Exception as e:
        print(f"Mode change failed: {e}")
        return False

def safe_object_selection(obj):
    """Background-compatible object selection"""
    try:
        # Clear selection
        bpy.ops.object.select_all(action='DESELECT')
        
        # Select target object
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        return True
    except Exception as e:
        print(f"Selection failed: {e}")
        return False
```

## File Operations

### Safe Operations (Work in Background)
```python
# File operations that work in background:
bpy.ops.wm.open_mainfile(filepath="/path/to/file.blend")
bpy.ops.wm.save_as_mainfile(filepath="/path/to/output.blend")
bpy.ops.render.render(write_still=True)
bpy.ops.export_scene.obj(filepath="/path/to/export.obj")
```

### Unsafe Operations (GUI-Dependent)
```python
# AVOID - requires user interaction:
bpy.ops.wm.open_mainfile('INVOKE_DEFAULT')    # File browser
bpy.ops.render.view_show('INVOKE_DEFAULT')    # Render view
bpy.ops.wm.save_mainfile('INVOKE_DEFAULT')    # Save dialog
```

## Mode Detection and Adaptation

### Robust Mode Detection
```python
def detect_blender_mode():
    """Comprehensive mode detection"""
    mode_info = {
        'background': bpy.app.background,
        'has_window_manager': bool(getattr(bpy.context, 'window_manager', None)),
        'has_window': bool(getattr(bpy.context, 'window', None)),
        'has_screen': bool(getattr(bpy.context, 'screen', None)),
        'context_type': type(bpy.context).__name__
    }
    
    print(f"Blender mode detection: {mode_info}")
    return mode_info

def is_gui_available():
    """Check if GUI operations are available"""
    return (not bpy.app.background and 
            hasattr(bpy.context, 'window_manager') and
            bpy.context.window_manager is not None)
```

## Memory and Resource Management

### Background Mode Cleanup
```python
def cleanup_background_resources():
    """Robust cleanup for background mode"""
    global tcp_server, server_task, keep_alive_timer
    
    try:
        # Clean network resources
        if tcp_server:
            tcp_server.close()
            tcp_server = None
            
        # Clean asyncio tasks
        if server_task:
            server_task.cancel()
            server_task = None
            
        # Clean timers
        if keep_alive_timer and bpy.app.timers.is_registered(keep_alive_timer):
            bpy.app.timers.unregister(keep_alive_timer)
            
        # Clean scene properties safely
        for scene in bpy.data.scenes:
            if hasattr(scene, 'plugin_active'):
                scene.plugin_active = False
                
        print("Background resources cleaned up")
        
    except Exception as e:
        print(f"Cleanup error: {e}")
```

## Testing and Validation

### Test Script Template
```python
import bpy
import sys

def test_background_compatibility():
    """Test plugin functions in both modes"""
    print(f"Running in background mode: {bpy.app.background}")
    
    test_results = []
    
    # Test 1: Context access
    try:
        scene = safe_context_access()
        test_results.append(("Context Access", scene is not None))
    except Exception as e:
        test_results.append(("Context Access", False, str(e)))
    
    # Test 2: Mode detection
    try:
        mode_info = detect_blender_mode()
        test_results.append(("Mode Detection", True))
    except Exception as e:
        test_results.append(("Mode Detection", False, str(e)))
    
    # Test 3: Plugin initialization
    try:
        result = initialize_plugin_safe()
        test_results.append(("Plugin Init", result))
    except Exception as e:
        test_results.append(("Plugin Init", False, str(e)))
    
    # Report results
    all_passed = True
    for test in test_results:
        name, passed = test[0], test[1]
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
            if len(test) > 2:
                print(f"  Error: {test[2]}")
    
    return all_passed

if __name__ == "__main__":
    success = test_background_compatibility()
    sys.exit(0 if success else 1)
```

### Testing Commands
```bash
# Test GUI mode
blender --python test_plugin.py

# Test background mode  
blender --background --python test_plugin.py

# Test with specific blend file
blender --background test_scene.blend --python test_plugin.py
```

## Best Practices for Background Compatibility

### 1. Background-First Design
- Write code that works in background mode first
- Add GUI enhancements as optional features
- Always provide non-GUI alternatives

### 2. Defensive Context Access
```python
def safe_api_usage():
    """Template for safe Blender API usage"""
    # Check mode first
    if bpy.app.background:
        print("Background mode: using safe alternatives")
    
    # Use defensive context access
    try:
        if hasattr(bpy.context, 'scene') and bpy.context.scene:
            scene = bpy.context.scene
        else:
            scene = bpy.data.scenes[0]
    except (AttributeError, IndexError):
        print("Error: No scene available")
        return False
    
    # Use direct data access when possible
    objects = scene.objects  # Instead of bpy.context.selected_objects
    
    return True
```

### 3. Fallback Mechanisms
```python
def robust_operation_with_fallbacks():
    """Multi-level fallback pattern"""
    # Try GUI method first
    if is_gui_available():
        try:
            return gui_specific_operation()
        except Exception:
            pass
    
    # Try background-compatible method
    try:
        return background_compatible_operation()
    except Exception:
        pass
    
    # Last resort: basic functionality
    return basic_fallback_operation()
```

### 4. Error Recovery
```python
def error_resilient_function():
    """Handle background mode errors gracefully"""
    try:
        return primary_implementation()
    except RuntimeError as e:
        if "context is incorrect" in str(e):
            print("Context error - using background fallback")
            return background_fallback()
        else:
            raise
    except AttributeError as e:
        if "_RestrictContext" in str(e):
            print("RestrictContext error - using safe alternative")
            return safe_alternative()
        else:
            raise
```

## Proven Working Patterns

### External Script Management (MOST RELIABLE)
Based on proven implementations, this pattern works consistently:

```python
# launcher.py - External process manager
def start_blender_service():
    import subprocess
    process = subprocess.Popen([
        blender_path, '--background', '--python', 'service.py'
    ])
    return process

# service.py - Inside Blender  
def main():
    enable_addon()
    
    # Manual event loop - PROVEN to work
    while True:
        stop_loop = process_events()
        if stop_loop:
            break
        time.sleep(0.01)

if __name__ == "__main__":
    main()
```

**Why This Works**:
- ✅ External process management avoids blocking issues
- ✅ Manual event loop gives full control
- ✅ Clear separation between startup and runtime
- ✅ Reliable shutdown mechanism

This comprehensive guide should help developers avoid the most common background mode pitfalls and build robust Blender plugins that work reliably in both GUI and headless environments.