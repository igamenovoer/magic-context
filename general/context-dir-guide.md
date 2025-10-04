# Context Directory Guide

A standardized directory structure for AI-assisted development projects that helps organize project knowledge, track progress, and maintain development history.

## Purpose

The `context/` directory serves as a centralized knowledge base for AI assistants working on your project. It contains all the context, documentation, and reference materials needed for effective AI collaboration.

## Directory Structure

```
context/
├── design/          # Technical specifications and architecture
├── hints/           # How-to guides and troubleshooting tips
├── instructions/    # Reusable prompt snippets and commands
├── logs/            # Development session records and outcomes
├── plans/           # Implementation roadmaps and strategies
├── refcode/         # Reference implementations and examples
├── roles/           # Role-based system prompts and memory
│   ├── role1/       # Role-specific subdirectory
│   └── role2/       # Another role-specific subdirectory
├── summaries/       # Knowledge base and analysis documents
├── tasks/           # Current and planned work items
│   ├── features/    # Feature implementation tasks
│   ├── fixes/       # Bug fix tasks  
│   ├── refactor/    # Code refactoring tasks
│   └── tests/       # Testing-related tasks
└── tools/           # Custom development utilities
```

## Directory Purposes

**design/** - Store API specifications, architecture diagrams, and technical design documents. Include both high-level system design and detailed component specifications.

**hints/** - Create "howto-" guides for common development tasks, error solutions, and best practices specific to your project. These help AI assistants avoid known pitfalls.

**instructions/** - Store reusable prompt snippets, command templates, and standardized instruction patterns that can be referenced across the project. Useful for maintaining consistency in AI interactions.

**logs/** - Record development sessions with date prefixes and outcome status. Include both successful implementations and failed attempts with lessons learned.

**plans/** - Document implementation strategies, feature roadmaps, and multi-step development plans. Break down complex features into manageable tasks.

**refcode/** - Store reference implementations, code examples, and third-party integrations that serve as patterns for new development.

**roles/** - Contains role-based system prompts, memory, and context for different AI assistant personas. Each role has its own subdirectory with specialized prompts and accumulated knowledge for that specific role or domain expertise.

**summaries/** - Maintain analysis documents, project knowledge summaries, and consolidated findings from research or implementation work.

**tasks/** - Track current work items organized by type:
  - **features/** - New feature implementation tasks and requirements
  - **fixes/** - Bug reports, issue tracking, and fix documentation  
  - **refactor/** - Code refactoring plans and cleanup tasks
  - **tests/** - Testing strategies, test cases, and quality assurance tasks

**tools/** - House custom scripts, utilities, and development aids specific to your project workflow.

## Naming Patterns

**Logs** - Use date prefix with descriptive outcome:
- `YYYY-MM-DD_feature-name-implementation-success.md`
- `YYYY-MM-DD_bug-fix-attempt-failed.md`
- `YYYY-MM-DD_performance-optimization-complete.md`

**Hints** - Use action-oriented prefixes:
- `howto-setup-development-environment.md`
- `howto-debug-connection-issues.md`
- `why-this-error-occurs.md`
- `troubleshoot-build-failures.md`

**Instructions** - Use command or snippet type prefixes:
- `prompt-code-review-template.md`
- `command-git-workflow.md`
- `snippet-error-handling-pattern.md`
- `template-feature-documentation.md`

**Roles** - Use role-based directory and file structure:
- `roles/backend-developer/system-prompt.md`
- `roles/backend-developer/memory.md`
- `roles/frontend-specialist/context.md`
- `roles/devops-engineer/knowledge-base.md`

**Tasks** - Organized by type with clear action descriptions:
- `features/task-implement-user-authentication.md`
- `fixes/task-fix-memory-leak.md`
- `refactor/task-modernize-api-endpoints.md`
- `tests/task-add-integration-tests.md`
- `goal.md` (for main objective in subdirectories)

**Plans** - Use feature or system names:
- `api-redesign-roadmap.md`
- `database-migration-strategy.md`
- `v2-implementation-plan.md`

**Summaries** - Use descriptive analysis topics:
- `library-comparison-analysis.md`
- `architecture-decision-rationale.md`
- `implementation-lessons-learned.md`

**Design** - Use component or system focus:
- `api-specification.md`
- `database-schema-design.md`
- `authentication-flow.md`

## Document Format Rules

**HEADER Section** - Every context document should start with a header section containing:
- **Purpose**: What this document is for
- **Status**: Current state (active/completed/deprecated/failed)
- **Date**: When created or last updated
- **Dependencies**: What this relates to or requires
- **Target**: Intended audience (AI assistants, developers, etc.)

Example header format:
```markdown
# Document Title

## HEADER
- **Purpose**: Implement user authentication system
- **Status**: Completed
- **Date**: 2025-01-15
- **Dependencies**: Database schema, JWT library
- **Target**: AI assistants and backend developers

## Content
[Main document content follows...]
```

## Best Practices

1. **Include README.md in each directory** explaining its specific purpose and contents
2. **Use consistent prefixes** as shown in naming patterns above
3. **Include status indicators** in filenames when relevant (success/failed/complete)
4. **Start every document with HEADER section** providing essential metadata
5. **Keep content current** - update summaries and remove outdated information
6. **Reference from main documentation** - link to context files from your project's main README
7. **Make it discoverable** - AI assistants should be directed to use this context for better project understanding

## Benefits

- **Consistent knowledge base** for AI assistants across development sessions
- **Reduced context switching** - all project knowledge in one place
- **Historical tracking** of decisions, implementations, and lessons learned
- **Onboarding efficiency** for new team members or AI assistants
- **Pattern reuse** through documented examples and references

## Implementation

1. Create the directory structure in your project root
2. Add a main README.md explaining the context system
3. Start with the most critical directories for your project type
4. Document as you develop - don't wait until later

This structure scales from small projects to large codebases and helps maintain development momentum across multiple AI-assisted sessions.