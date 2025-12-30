# Python Project Structure Guide

This guide outlines how to create a professional Python library project structure for `MyPythonLib` - designed for PyPI distribution with modern development practices.

## Project Philosophy

This structure emphasizes:
- **Separation of Concerns**: Clear organization with distinct purposes for each directory
- **Modern Standards**: Following current Python packaging best practices
- **AI Collaboration**: Structured approach for working with AI assistants
- **Professional Workflow**: Tools and processes for quality development
- **Future-Proof**: Designed to scale and evolve with changing requirements

## Directory Structure Overview

```
MyPythonLib/
├── .github/workflows/           # Automation workflows
├── src/mypythonlib/            # Main package code (src layout)
├── extern/                     # Third-party code (see extern structure below)
│   ├── tracked/                # Git submodules pinned via .gitmodules
│   │   └── <repo>/             # e.g. simulators, third-party libs
│   └── orphan/                 # Local clones/checkouts (git-ignored), keep README.md
│       └── <repo>/             # e.g. shallow clone for quick grepping/patching
├── scripts/                    # Command-line interface tools
├── tests/                      # Test suite (see details below)
│   ├── unit/                   # Fast, deterministic unit tests
│   │   └── <subdir>/test_*.py
│   ├── integration/            # I/O, service, or multi-component tests
│   │   └── <subdir>/test_*.py
│   └── manual/                 # Manually executed scripts (not CI-collected)
│       └── manual_*.py
├── docs/                       # Documentation source
├── context/                    # AI assistant workspace (see context-dir-guide.md)
├── tmp/                        # Temporary working files
└── [configuration files]       # Project setup and tooling
```

## Core Components

### Source Code Organization

**src/mypythonlib/** - Main Package
- Contains the core library functionality
- Uses "src layout" for better testing isolation
- Includes package initialization and module definitions
- Houses the primary API that users will import

**scripts/** - Command Line Tools
- Entry points for terminal commands
- Provides CLI interface to library functionality
- Designed for both user convenience and automation
- Configured as console scripts in packaging

### External Dependencies

**extern/** - Third-Party Code
- Prefer `extern/tracked/<repo>/` for dependencies you want reproducible and pinned (git submodules recorded in `.gitmodules`).
- Use `extern/orphan/<repo>/` for local-only clones/checkouts you do not want to commit (add `extern/.gitignore` rules to ignore all subdirectories under `extern/orphan/` while keeping `extern/orphan/README.md` tracked).
- Avoid editing code inside `extern/tracked/*` unless you intend to carry a fork or upstream a change; treat submodule updates as explicit, reviewed changes.

### Quality and Testing

**tests/** - Test Suite
- Comprehensive testing for all functionality
- Organized by purpose and discovery behavior
- Includes unit, integration, and manual tests
- Configured with modern testing frameworks and coverage tools

Testing layout conventions

- `tests/unit/(subdir)/test_*.py`
  - Pytest/unittest-based, fast and deterministic
  - Mirrors source structure by module/functionality under `(subdir)`
  - Discovered by CI (pytest) by default

- `tests/integration/(subdir)/test_*.py`
  - Exercises external services (e.g., DB, S3/MinIO) or filesystem I/O
  - May require local service endpoints configured via environment
  - Discovered by CI if enabled; can be skipped via markers

- `tests/manual/manual_*.py`
  - Manually executed scripts for heavy or environment-specific checks (e.g., Blender)
  - Should not be collected by CI (prefixed with `manual_` and placed under `tests/manual/`)
  - Follow manual-script guidance (see instructions/make-manual-python-script.md)

Notes
- Prefer `pixi run` for executing tests and scripts; avoid system Python.
- Keep unit tests hermetic; integration tests should skip gracefully when dependencies are unavailable.

### Documentation and Communication

**docs/** - Documentation Source
- User-facing documentation written in Markdown
- Built with MkDocs Material for professional presentation
- Automatically deployed to GitHub Pages
- Includes installation guides, API reference, and examples

### AI Collaboration Framework

**context/** - AI Assistant Workspace
This directory enables effective collaboration with AI coding assistants. For detailed structure and organization patterns, see [context-dir-guide.md](context-dir-guide.md) which provides comprehensive documentation on:
- Directory structure and purposes
- Naming conventions and best practices  
- Document format standards
- Implementation guidelines

The context directory serves as a centralized knowledge base containing project design documents, implementation plans, development history, and reference materials organized for effective human-AI collaboration.

### Project Configuration

**Root Level Files:**
- **pyproject.toml** - Python packaging configuration with dependencies and metadata
- **pixi.toml** - Environment management and development task definitions
- **mkdocs.yml** - Documentation build configuration
- **README.md** - Primary project introduction and usage guide
- **LICENSE** - Legal terms for code usage and distribution
- **.gitignore** - Version control exclusion patterns
- **.gitmodules** - Submodule pins (if using `extern/tracked/` or other submodules)

### Automation and Deployment

**.github/workflows/** - CI/CD Automation
- Documentation deployment to GitHub Pages
- Automated testing and quality checks
- Package building and release processes
- Integration with external services

## Development Workflow

### Environment Setup
Projects use Pixi for reproducible development environments, providing consistent dependency management across different machines and operating systems.

### Code Quality
Automated tools ensure consistent code style, type safety, and testing coverage. This includes linting, formatting, and static analysis integrated into the development workflow.

### Documentation Strategy
Documentation is treated as code - written in Markdown, version controlled, and automatically deployed. This ensures documentation stays current with code changes.

### AI Collaboration Model
The context directory structure facilitates effective human-AI collaboration by providing organized spaces for different types of project information and communication.

## Setup Process

### Initial Creation
1. Create project directory structure
2. Initialize git repository with main branch
3. Set up core configuration files
4. Create initial Python package structure
5. Configure development environment with Pixi

### GitHub Integration
1. Create remote repository on GitHub
2. Configure GitHub Pages for documentation
3. Set up automated workflows for deployment
4. Enable issue tracking and discussions

### Development Environment
1. Install Pixi environment manager
2. Initialize project dependencies
3. Configure development tools and workflows
4. Set up testing and quality assurance tools

## Key Principles

### Maintainability
- Clear separation between different types of content
- Consistent naming conventions and organization
- Documentation for all major components and decisions

### Scalability
- Structure supports growth from simple library to complex project
- Modular organization allows independent development of components
- Flexible configuration supports different deployment scenarios

### Collaboration
- AI-friendly organization with clear context and history
- Human-readable documentation and guides
- Version-controlled communication and decision tracking

### Professional Standards
- Follows modern Python packaging best practices
- Includes comprehensive testing and quality assurance
- Professional documentation and presentation
- Automated deployment and release processes

This structure provides a solid foundation for any Python library project, emphasizing clarity, maintainability, and effective collaboration between human developers and AI assistants.
