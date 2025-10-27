# Blender Plugin Automated Testing Guide

## HEADER
- **Purpose**: Comprehensive guide for setting up automated testing for Blender plugins
- **Status**: Active
- **Date**: 2025-07-10
- **Dependencies**: Blender 4.0+, pytest, process management tools
- **Target**: AI assistants and developers building testable Blender plugins

## Overview

Automated testing for Blender plugins presents unique challenges due to Blender's process management, GUI/background mode differences, and complex initialization sequences. This guide provides battle-tested patterns for building comprehensive test suites that ensure plugin reliability across different environments.

## Test Architecture and Organization

### Multi-Layered Test Strategy

Organize tests into multiple layers with different purposes and execution times:

#### 1. **Smoke Tests** (< 30 seconds)
- **Purpose**: Rapid verification during development iteration
- **Scope**: Basic plugin loading, service startup, minimal functionality
- **Usage**: Run after every code change for immediate feedback

```python
# tests/smoke_test.py
def test_plugin_basic_functionality():
    """Quick verification that plugin works at all"""
    # Start Blender with plugin
    start_blender_with_plugin()
    
    # Basic connectivity/loading test
    assert plugin_is_loaded()
    assert basic_functionality_works()
    
    # Cleanup
    cleanup_blender_process()
```

#### 2. **Unit Tests** (< 2 minutes)
- **Purpose**: Test individual plugin components in isolation
- **Scope**: API functions, data processing, error handling
- **Pattern**: Test plugin functions without full Blender integration

```python
# tests/test_plugin_in_development.py
class TestPluginCore:
    def test_plugin_initialization(self):
        """Test plugin setup and configuration"""
        
    def test_api_functionality(self):
        """Test plugin API methods"""
        
    def test_error_handling(self):
        """Test plugin error recovery"""
```

#### 3. **Integration Tests** (< 5 minutes)
- **Purpose**: Test plugin interaction with Blender and external systems
- **Scope**: Full workflow testing, cross-component interaction
- **Pattern**: End-to-end scenarios with real Blender processes

```python
# tests/integration/test_plugin_integration.py
class TestPluginIntegration:
    def test_full_workflow(self):
        """Test complete plugin workflow"""
        
    def test_blender_api_integration(self):
        """Test plugin interaction with Blender API"""
        
    def test_background_mode_compatibility(self):
        """Test plugin works in background mode"""
```

#### 4. **Performance Tests** (< 10 minutes)
- **Purpose**: Validate plugin performance and resource usage
- **Scope**: Stress testing, memory usage, response times
- **Pattern**: Quantitative validation with acceptable thresholds

```python
# tests/performance/test_plugin_performance.py
def test_plugin_performance():
    """Test plugin performance characteristics"""
    start_time = time.time()
    
    # Execute performance-critical operations
    for i in range(100):
        result = plugin.execute_operation()
    
    execution_time = time.time() - start_time
    assert execution_time < MAX_ACCEPTABLE_TIME
```

### Directory Structure

```
tests/
├── README.md                     # Test suite overview
├── conftest.py                   # pytest configuration and fixtures
├── smoke_test.py                 # Quick verification script
├── test_plugin_in_development.py # Main plugin tests
├── integration/                  # Integration test suite
│   ├── test_full_workflow.py
│   ├── test_blender_integration.py
│   └── test_background_mode.py
├── performance/                  # Performance test suite
│   ├── test_response_times.py
│   └── test_resource_usage.py
├── manual/                       # Manual testing aids
│   ├── debug_helper.py
│   └── interactive_test.py
└── utils/                        # Test utilities
    ├── blender_manager.py
    ├── plugin_helpers.py
    └── assertions.py
```

## Test Infrastructure Setup

### Core Test Configuration (`conftest.py`)

```python
import pytest
import subprocess
import time
import os
import socket

# Configuration
BLENDER_EXEC_PATH = os.environ.get('BLENDER_EXEC_PATH', '/path/to/blender')
PLUGIN_NAME = 'plugin-in-development'
PLUGIN_PORT = int(os.environ.get('PLUGIN_PORT', 6688))
TEST_TIMEOUT = 30

class BlenderManager:
    """Manage Blender process lifecycle for testing"""
    
    def __init__(self):
        self.process = None
        self.plugin_port = PLUGIN_PORT
    
    def start_with_plugin(self, background=False):
        """Start Blender with plugin enabled"""
        # Kill existing processes
        self.kill_existing_processes()
        
        # Set environment for plugin
        env = os.environ.copy()
        env['PLUGIN_AUTO_START'] = '1'
        env['PLUGIN_PORT'] = str(self.plugin_port)
        
        # Build command
        cmd = [BLENDER_EXEC_PATH]
        if background:
            cmd.append('--background')
        cmd.extend(['--python-expr', 
                   f"import bpy; bpy.ops.preferences.addon_enable(module='{PLUGIN_NAME}')"])
        
        # Start process
        self.process = subprocess.Popen(cmd, env=env)
        
        # Wait for plugin to be ready
        if not self.wait_for_plugin_ready():
            raise RuntimeError("Plugin failed to start")
    
    def wait_for_plugin_ready(self, timeout=TEST_TIMEOUT):
        """Wait for plugin to be fully operational"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_plugin_responding():
                return True
            time.sleep(0.5)
        return False
    
    def is_plugin_responding(self):
        """Check if plugin is responding to requests"""
        try:
            # Test basic connectivity (adapt to your plugin's interface)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', self.plugin_port))
            sock.close()
            return result == 0
        except:
            return False
    
    def kill_existing_processes(self):
        """Kill any existing Blender processes"""
        try:
            subprocess.run(['pkill', '-f', 'blender'], check=False, timeout=5)
            time.sleep(2)
        except subprocess.TimeoutExpired:
            pass
    
    def cleanup(self):
        """Clean up Blender process"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.kill_existing_processes()

@pytest.fixture(scope="session")
def clean_environment():
    """Ensure clean test environment"""
    manager = BlenderManager()
    manager.kill_existing_processes()
    
    # Verify port availability
    if not is_port_available(PLUGIN_PORT):
        pytest.skip(f"Port {PLUGIN_PORT} not available")
    
    yield
    
    # Cleanup after all tests
    manager.cleanup()

@pytest.fixture
def blender_with_plugin(clean_environment):
    """Provide Blender instance with plugin loaded"""
    manager = BlenderManager()
    manager.start_with_plugin()
    
    yield manager
    
    manager.cleanup()

@pytest.fixture
def blender_background(clean_environment):
    """Provide Blender instance in background mode"""
    manager = BlenderManager()
    manager.start_with_plugin(background=True)
    
    yield manager
    
    manager.cleanup()

def is_port_available(port):
    """Check if port is available for use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        return True
    except OSError:
        return False
```

### Plugin Test Client

```python
# tests/utils/plugin_client.py
import socket
import json
import time

class PluginTestClient:
    """Test client for plugin communication"""
    
    def __init__(self, host='127.0.0.1', port=6688):
        self.host = host
        self.port = port
    
    def execute_command(self, command, params=None):
        """Execute a command via plugin interface"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((self.host, self.port))
            
            # Send command (adapt to your plugin's protocol)
            message = {
                'command': command,
                'params': params or {}
            }
            sock.sendall(json.dumps(message).encode('utf-8'))
            
            # Receive response
            response = sock.recv(4096).decode('utf-8')
            sock.close()
            
            return json.loads(response)
        except Exception as e:
            raise RuntimeError(f"Plugin communication failed: {e}")
    
    def test_connectivity(self):
        """Test basic connectivity to plugin"""
        try:
            result = self.execute_command('ping')
            return result.get('status') == 'ok'
        except:
            return False
    
    def get_plugin_status(self):
        """Get plugin status information"""
        return self.execute_command('status')
```

## Automated Testing Patterns

### Dual-Channel Testing Strategy

For plugins that have reference implementations or external communication:

```python
class DualChannelTestStrategy:
    """Test plugin against reference implementation"""
    
    def __init__(self):
        self.plugin_client = PluginTestClient(port=6688)  # Your plugin
        self.reference_client = ReferenceClient(port=9876)  # Reference impl
    
    def compare_functionality(self, command, params):
        """Compare plugin response with reference"""
        plugin_result = self.plugin_client.execute_command(command, params)
        reference_result = self.reference_client.execute_command(command, params)
        
        # Compare results (implement comparison logic)
        return self.compare_responses(plugin_result, reference_result)
    
    def compare_responses(self, response1, response2, tolerance=0.001):
        """Compare two responses for functional equivalence"""
        if type(response1) != type(response2):
            return False, f"Type mismatch: {type(response1)} vs {type(response2)}"
        
        if isinstance(response1, (int, float)) and isinstance(response2, (int, float)):
            diff = abs(response1 - response2)
            return diff <= tolerance, f"Numeric difference {diff} exceeds tolerance"
        
        return response1 == response2, "Match" if response1 == response2 else "Differ"

# Usage in tests
def test_command_equivalence():
    """Test that plugin produces same results as reference"""
    tester = DualChannelTestStrategy()
    
    test_commands = [
        ('get_scene_info', {}),
        ('execute_python', {'code': 'print("hello")'}),
        ('get_object_count', {})
    ]
    
    for command, params in test_commands:
        matches, message = tester.compare_functionality(command, params)
        assert matches, f"Command {command} failed: {message}"
```

### Environment-Aware Testing

```python
class EnvironmentAwareTests:
    """Tests that adapt to different Blender execution modes"""
    
    @pytest.mark.parametrize("background_mode", [False, True])
    def test_plugin_in_both_modes(self, background_mode):
        """Test plugin in both GUI and background modes"""
        if background_mode:
            blender = self.start_background_blender()
            # Test background-specific functionality
            self.test_background_specific_features(blender)
        else:
            blender = self.start_gui_blender()
            # Test GUI-specific functionality
            self.test_gui_specific_features(blender)
        
        # Test common functionality
        self.test_common_functionality(blender)
    
    def test_background_specific_features(self, blender):
        """Test features that should work in background mode"""
        client = PluginTestClient()
        
        # Test basic API access
        result = client.execute_command('execute_python', {
            'code': 'import bpy; len(bpy.data.objects)'
        })
        assert 'result' in result
        
        # Test operations that should work without GUI
        assert client.execute_command('get_scene_info')
    
    def test_gui_specific_features(self, blender):
        """Test features that require GUI mode"""
        client = PluginTestClient()
        
        # Test viewport operations (if applicable)
        if hasattr(client, 'capture_viewport'):
            result = client.capture_viewport()
            assert result is not None
```

## Common Test Scenarios

### Plugin Loading and Initialization

```python
def test_plugin_loading():
    """Test plugin loads correctly and initializes"""
    with BlenderManager() as blender:
        blender.start_with_plugin()
        
        # Verify plugin is loaded
        result = blender.execute_python(f"""
import bpy
'{PLUGIN_NAME}' in bpy.context.preferences.addons
""")
        assert result is True
        
        # Verify plugin classes are registered
        result = blender.execute_python(f"""
import bpy
hasattr(bpy.types, 'PluginInDevelopmentOperator')
""")
        assert result is True

def test_plugin_startup_timing():
    """Test plugin starts within acceptable time"""
    start_time = time.time()
    
    with BlenderManager() as blender:
        blender.start_with_plugin()
        client = PluginTestClient()
        assert client.test_connectivity()
    
    startup_time = time.time() - start_time
    assert startup_time < MAX_STARTUP_TIME
```

### API Functionality Testing

```python
def test_plugin_api_basic():
    """Test basic plugin API functionality"""
    with BlenderManager() as blender:
        blender.start_with_plugin()
        client = PluginTestClient()
        
        # Test plugin status
        status = client.get_plugin_status()
        assert status['running'] is True
        
        # Test basic operations
        test_operations = [
            ('ping', {}),
            ('get_version', {}),
            ('get_capabilities', {})
        ]
        
        for operation, params in test_operations:
            result = client.execute_command(operation, params)
            assert 'error' not in result

def test_blender_api_integration():
    """Test plugin interaction with Blender API"""
    with BlenderManager() as blender:
        blender.start_with_plugin()
        client = PluginTestClient()
        
        # Test Blender API access through plugin
        test_code = """
import bpy
scene_name = bpy.context.scene.name
object_count = len(bpy.context.scene.objects)
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 2))
result = f"Scene: {scene_name}, Objects: {object_count + 1}"
"""
        
        result = client.execute_command('execute_python', {'code': test_code})
        assert 'Scene:' in result['output']
        assert 'Objects:' in result['output']
```

### Error Handling and Recovery

```python
def test_error_handling():
    """Test plugin handles errors gracefully"""
    with BlenderManager() as blender:
        blender.start_with_plugin()
        client = PluginTestClient()
        
        # Test invalid command
        result = client.execute_command('invalid_command', {})
        assert 'error' in result
        assert 'unknown command' in result['error'].lower()
        
        # Test invalid Python code execution
        result = client.execute_command('execute_python', {
            'code': 'undefined_variable + 1'
        })
        assert 'error' in result
        
        # Verify plugin still responds after errors
        status = client.get_plugin_status()
        assert status['running'] is True

def test_concurrent_operations():
    """Test plugin handles concurrent operations"""
    with BlenderManager() as blender:
        blender.start_with_plugin()
        
        clients = [PluginTestClient() for _ in range(3)]
        
        # Execute operations concurrently
        import threading
        
        def worker(client, operation_id):
            result = client.execute_command('execute_python', {
                'code': f'result = {operation_id} * 2'
            })
            return result
        
        threads = []
        for i, client in enumerate(clients):
            thread = threading.Thread(target=worker, args=(client, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all operations to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify plugin still responsive
        assert clients[0].test_connectivity()
```

### Performance and Resource Testing

```python
def test_plugin_performance():
    """Test plugin performance characteristics"""
    with BlenderManager() as blender:
        blender.start_with_plugin()
        client = PluginTestClient()
        
        # Measure response times
        response_times = []
        for i in range(10):
            start_time = time.time()
            client.execute_command('ping', {})
            response_time = time.time() - start_time
            response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < MAX_RESPONSE_TIME
        
        # Test memory usage (if measurable)
        status = client.get_plugin_status()
        if 'memory_usage' in status:
            assert status['memory_usage'] < MAX_MEMORY_USAGE

def test_stress_operations():
    """Test plugin under stress conditions"""
    with BlenderManager() as blender:
        blender.start_with_plugin()
        client = PluginTestClient()
        
        # Rapid sequential operations
        operations = [
            "bpy.ops.mesh.primitive_cube_add()",
            "bpy.ops.mesh.primitive_uv_sphere_add()",
            "bpy.ops.mesh.primitive_cylinder_add()"
        ] * 10
        
        start_time = time.time()
        for op in operations:
            result = client.execute_command('execute_python', {'code': op})
            assert 'error' not in result
        
        execution_time = time.time() - start_time
        assert execution_time < MAX_STRESS_TIME
```

## Test Execution Workflows

### Development Testing Script

```python
#!/usr/bin/env python3
# run_plugin_tests.py

import argparse
import subprocess
import sys
import time

def run_smoke_tests():
    """Run quick smoke tests (< 30 seconds)"""
    print("🔥 Running smoke tests...")
    start = time.time()
    result = subprocess.run(['python', 'tests/smoke_test.py'], capture_output=True)
    duration = time.time() - start
    
    if result.returncode == 0:
        print(f"✅ Smoke tests passed ({duration:.1f}s)")
        return True
    else:
        print(f"❌ Smoke tests failed ({duration:.1f}s)")
        print(result.stderr.decode())
        return False

def run_unit_tests():
    """Run unit tests"""
    print("🧪 Running unit tests...")
    result = subprocess.run(['pytest', 'tests/test_plugin_in_development.py', '-v'])
    return result.returncode == 0

def run_integration_tests():
    """Run integration tests"""
    print("🔗 Running integration tests...")
    result = subprocess.run(['pytest', 'tests/integration/', '-v'])
    return result.returncode == 0

def run_performance_tests():
    """Run performance tests"""
    print("⚡ Running performance tests...")
    result = subprocess.run(['pytest', 'tests/performance/', '-v'])
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Run plugin test suite')
    parser.add_argument('--quick', action='store_true', help='Run only quick tests')
    parser.add_argument('--smoke', action='store_true', help='Run only smoke tests')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--performance', action='store_true', help='Run only performance tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if not any([args.quick, args.smoke, args.unit, args.integration, args.performance, args.all]):
        args.quick = True  # Default to quick tests
    
    success = True
    
    if args.smoke or args.quick or args.all:
        if not run_smoke_tests():
            success = False
            if not args.all:
                sys.exit(1)
    
    if args.unit or args.all:
        if not run_unit_tests():
            success = False
            if not args.all:
                sys.exit(1)
    
    if args.integration or args.all:
        if not run_integration_tests():
            success = False
            if not args.all:
                sys.exit(1)
    
    if args.performance or args.all:
        if not run_performance_tests():
            success = False
    
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### CI/CD Integration

```yaml
# .github/workflows/test-plugin.yml
name: Test Plugin

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install Blender
      run: |
        wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
        tar -xf blender-4.0.0-linux-x64.tar.xz
        echo "BLENDER_EXEC_PATH=$PWD/blender-4.0.0-linux-x64/blender" >> $GITHUB_ENV
    
    - name: Install dependencies
      run: |
        pip install pytest
        pip install -r requirements-test.txt
    
    - name: Run smoke tests
      run: python run_plugin_tests.py --smoke
    
    - name: Run unit tests
      run: python run_plugin_tests.py --unit
    
    - name: Run integration tests
      run: python run_plugin_tests.py --integration
```

## Real-World Example: MCP Plugin Testing

### Example Implementation from blender-remote Project

The blender-remote project demonstrates sophisticated testing patterns:

#### **Dual-Service Testing Strategy**
```python
# Testing implementation plugin against reference
class MCPComparisonTest:
    def __init__(self):
        self.bld_remote_client = MCPClient(port=6688)    # Implementation
        self.blender_auto_client = MCPClient(port=9876)  # Reference
    
    def test_command_equivalence(self):
        """Verify implementation matches reference behavior"""
        test_commands = [
            {'type': 'get_scene_info', 'params': {}},
            {'type': 'execute_code', 'params': {'code': 'import bpy; len(bpy.data.objects)'}},
            {'type': 'get_object_info', 'params': {'object_name': 'Cube'}}
        ]
        
        for command in test_commands:
            impl_result = self.bld_remote_client.send_command(command)
            ref_result = self.blender_auto_client.send_command(command)
            
            matches, message = self.compare_responses(impl_result, ref_result)
            assert matches, f"Command {command['type']} mismatch: {message}"
```

#### **Performance Validation**
```python
def test_performance_comparison():
    """Ensure implementation performance is acceptable"""
    
    # Test execution time
    start_time = time.time()
    for i in range(10):
        result = client.execute_code(f"cube_{i} = bpy.ops.mesh.primitive_cube_add()")
    impl_time = time.time() - start_time
    
    # Performance should be within 2x of reference
    assert impl_time < reference_time * 2.0, f"Implementation too slow: {impl_time:.2f}s vs {reference_time:.2f}s"
```

#### **Background Mode Validation**
```python
def test_background_mode_compatibility():
    """Test plugin works in background mode"""
    with BlenderManager() as blender:
        blender.start_with_plugin(background=True)
        client = PluginTestClient()
        
        # Test operations that should work in background
        result = client.execute_command('get_scene_info')
        assert result['success']
        
        # Test Blender API access
        result = client.execute_command('execute_python', {
            'code': 'import bpy; bpy.app.background'
        })
        assert result['result'] is True
```

## Development Workflow Integration

### Hot Reload Testing

```python
def test_hot_reload_capability():
    """Test plugin hot reload without losing functionality"""
    with BlenderManager() as blender:
        blender.start_with_plugin()
        client = PluginTestClient()
        
        # Get initial state
        initial_status = client.get_plugin_status()
        
        # Perform hot reload
        reload_success = blender.reload_plugin(PLUGIN_NAME)
        assert reload_success
        
        # Verify functionality preserved
        final_status = client.get_plugin_status()
        assert final_status['running']
        assert client.test_connectivity()
```

### Debug-Friendly Testing

```python
# tests/manual/debug_helper.py
class DebugTestSession:
    """Interactive debugging session for manual testing"""
    
    def __init__(self):
        self.blender = BlenderManager()
        self.client = None
    
    def start_debug_session(self):
        """Start interactive debugging session"""
        print("🔧 Starting debug session...")
        self.blender.start_with_plugin()
        self.client = PluginTestClient()
        
        print(f"✅ Plugin loaded and running on port {PLUGIN_PORT}")
        print("Available commands:")
        print("  status() - Get plugin status")
        print("  execute(code) - Execute Python code")
        print("  test_api() - Test basic API functions")
        print("  quit() - Exit debug session")
        
        # Interactive loop
        while True:
            try:
                cmd = input("debug> ").strip()
                if cmd == 'quit()':
                    break
                elif cmd == 'status()':
                    print(self.client.get_plugin_status())
                elif cmd.startswith('execute('):
                    code = cmd[8:-1].strip('"\'')
                    result = self.client.execute_command('execute_python', {'code': code})
                    print(result)
                elif cmd == 'test_api()':
                    self.run_basic_api_tests()
                else:
                    print(f"Unknown command: {cmd}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        self.cleanup()
    
    def cleanup(self):
        if self.blender:
            self.blender.cleanup()

if __name__ == "__main__":
    debug = DebugTestSession()
    debug.start_debug_session()
```

## Best Practices

### 1. **Test Organization**
- Start with smoke tests for rapid iteration
- Build comprehensive unit tests for core functionality
- Add integration tests for full workflow validation
- Include performance tests for critical operations

### 2. **Process Management**
- Always clean up Blender processes between tests
- Use short timeouts (10s) for startup verification
- Handle both GUI and background mode testing
- Implement robust error recovery

### 3. **Test Data Management**
- Use temporary directories for test outputs
- Clean up test artifacts after each run
- Provide sample blend files for complex scenarios
- Version control test data and expected results

### 4. **Debugging Support**
- Provide interactive debugging tools
- Log detailed information for test failures
- Include manual testing helpers
- Support incremental testing during development

### 5. **CI/CD Integration**
- Ensure tests can run in headless environments
- Provide different test categories for different CI stages
- Cache Blender installation for faster CI runs
- Generate test reports and artifacts

This comprehensive testing framework provides the foundation for building reliable, well-tested Blender plugins that work across different environments and usage scenarios.