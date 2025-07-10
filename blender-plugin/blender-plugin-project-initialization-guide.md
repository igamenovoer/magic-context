# Blender Plugin Project Initialization Guide

## HEADER
- **Purpose**: Guide for initializing a professional Blender plugin project ready for PyPI publication
- **Status**: Active
- **Date**: 2025-07-10
- **Dependencies**: Python 3.11+, pixi, git, Blender 4.0+
- **Target**: Developers creating publishable Blender plugins with modern development practices

## Overview

This guide provides a proven project structure for professional Blender plugin development that supports both addon and external package distribution, ready for PyPI publication.

## Project Architecture

### Dual-Component Design

The recommended architecture separates internal Blender functionality from external control:

```
your-plugin-name/
├── blender_addon/           # Blender addon (runs inside Blender)
├── src/your_plugin_name/    # External Python package (PyPI distribution)
├── tests/                   # Multi-layered test suite
├── docs/                    # Documentation site
├── examples/                # Usage examples
├── context/                 # AI-assisted development workspace
└── Configuration files      # pyproject.toml, mkdocs.yml, etc.
```

**Why This Structure Works**:
- **Addon Component**: Direct Blender integration with UI panels and operators
- **External Package**: Remote control, automation, CI/CD integration  
- **Unified Distribution**: Single PyPI package contains both components
- **Professional Standards**: Modern Python tooling and practices

## Core Components

### 1. Blender Addon Directory (`blender_addon/`)

Contains the Blender addon that runs inside Blender:
- **Addon manifest** with bl_info metadata
- **UI panels and operators** for direct Blender integration
- **Service startup** for external communication
- **Background mode compatibility** for headless operation

The addon handles direct Blender integration and optionally starts a service for external control.

### 2. External Package (`src/your_plugin_name/`)

Python package for external control and automation:
- **Client API** for connecting to Blender from external scripts
- **Data models** for structured communication
- **CLI tools** for command-line automation
- **Exception handling** for robust error management

This allows external Python scripts to control Blender workflows.

### 3. Testing Infrastructure (`tests/`)

Multi-layered testing approach:
- **Smoke tests** (30s) for rapid development iteration
- **Unit tests** for isolated component testing
- **Integration tests** for full workflow validation
- **Performance tests** for ensuring acceptable response times
- **Manual test helpers** for interactive debugging

Includes both GUI and background mode testing patterns.

### 4. Documentation Site (`docs/`)

Professional documentation using MkDocs Material:
- **User guides** for installation and usage
- **API reference** with complete method documentation
- **Developer guides** for contributing and extending
- **Examples** showing common usage patterns

Automatically deployed to GitHub Pages.

### 5. Development Workspace (`context/`)

Structured information for AI-assisted development:
- **Design docs** for architecture decisions
- **Implementation hints** for common patterns
- **Development logs** tracking progress and solutions
- **Reference code** from related projects
- **Task tracking** for planned improvements

### 6. Configuration and Tooling

Essential configuration files:
- **pyproject.toml** with build system, dependencies, and tool configuration
- **mkdocs.yml** for documentation site generation
- **pixi environment** for reproducible development setup
- **GitHub Actions** for automated testing and deployment

## Development Workflow

### Environment Setup

Use **pixi** for reproducible environment management:
- Install dependencies automatically across platforms
- Manage conda and PyPI packages in single configuration
- Provide consistent development commands
- Handle environment isolation and reproducibility

### Development Commands

Standard development tasks automated through pixi:
- **Code quality**: formatting (black), linting (ruff), type checking (mypy)
- **Testing**: unit tests, integration tests, coverage reporting
- **Documentation**: local serving, static site generation
- **Build**: package building, distribution preparation

### Testing Strategy

Multi-layered approach for comprehensive validation:
- **Smoke tests** for rapid development feedback
- **Unit tests** for isolated component validation
- **Integration tests** for full workflow testing
- **Manual test helpers** for interactive debugging

### Documentation Generation

Professional documentation site using MkDocs Material:
- Automatically generated from source code
- Deployed to GitHub Pages on commits
- Includes API reference, user guides, examples
- Supports search, dark mode, mobile responsive

## Key Benefits

### Professional Standards
- Modern Python tooling and practices
- Comprehensive testing and documentation
- Automated quality checks and CI/CD
- Ready for PyPI publication

### Dual Distribution Model
- Single package supports both use cases
- Blender addon for direct integration
- External package for automation workflows
- Unified installation via pip

### AI-Assisted Development
- Structured context directory for AI assistants
- Documented patterns and troubleshooting guides
- Development history and decision tracking
- Reference implementations and examples

### Development Efficiency
- Hot reload capabilities for rapid iteration
- Automated testing across GUI and background modes
- Professional documentation and examples
- Consistent development environment across machines

## Customization

When adapting this structure for your plugin:

1. **Replace placeholders** throughout all files with your specific names
2. **Configure package metadata** in pyproject.toml
3. **Update addon manifest** with your plugin information
4. **Customize documentation** content and navigation
5. **Add specific dependencies** for your plugin functionality
6. **Configure repository** settings and deployment

This structure provides a solid foundation for professional Blender plugin development with modern practices and comprehensive tooling.