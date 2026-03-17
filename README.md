# Magic Context

A reusable library of prompt templates, skill definitions, and reference knowledge for AI-assisted software engineering. Designed to be embedded as a submodule and consumed by AI coding agents (Claude Code, Codex, Copilot, etc.).

## Structure

### `instructions/`
Standalone prompt templates for common development tasks. Drop into a conversation or reference directly.

- **Code quality:** `write-code.md`, `check-code.md`, `review-code-*.md`, `strongly-typed.md`, `validate-types.md`
- **Debugging:** `debug/debug-code.md`, `debug/debug-deeply-py.md`
- **Documentation:** `document-python-code.md`, `explain/`, `revise-doc.md`
- **Testing:** `make-unit-test-python.md`, `gui-test-screenshots.md`, `headless-testing.md`, `nicegui-interactive-test.md`
- **Planning:** `planning/` — roadmaps, task breakdowns, refactor plans, open-question discussions
- **Project setup:** `dir-setup/` — Python project init, experiment dirs, context dirs
- **Research/reading:** `reading/` — paper conversion, survey, algorithm blocks, figure insertion
- **Environment:** `envs/` — Claude Code, Codex, Copilot setup guides; `linux-env.md`, `mac-env.md`, `win32-env.md`
- **Memory/context:** `make-memory.md`, `mem-as-*.md`
- **Web research:** `find-online.md`, `check-online.md`, `search-proactively.md`, `tavily-guide.md`

### `knowledge-base/`
Reference documentation for specific tools and technologies.

- `blender-plugin/` — Blender addon development (background mode, data persistence, testing)
- `general/` — Project profiles, Mermaid CLI, LiteLLM proxy, APT mirrors, DNN timing
- `mcp-dev/` — MCP server development and programmatic invocation
- `github/` — Secure CI secrets management
- `tensorrt/` — TensorRT setup with Pixi, ONNX QDQ → INT8 conversion
- `onnx/` — Parallel ONNX inference
- `ngc-dockers/` — NVIDIA NGC Docker environments, SSH setup

### `skills/`
Self-contained skill definitions consumed by AI agents. Each skill has `SKILL.md`, agent prompts, reference materials, and templates.

- **CLI agents:** Claude Code install/invoke, Codex CLI, Copilot
- **Air-gapped:** Offline Docker/Pixi projects, offline VSCode installer
- **Development:** Git worktrees, interactive testing, Pixi project init, MkDocs serving
- **Tools/Pixi:** NVIDIA CUDA setup, CUDA build env, offline package channels
- **Tools/Conan:** C++ package management basics
- **OpenSpec extensions:** explain, test-driven dev, respond-to-review, revise-by-decision
- **Research:** DNN model exploration (experiment report, stakeholder report templates)
- **Writing:** Program tutorial creation, Mermaid diagram generation
- **Libraries:** DeepFace basic usage

### `scripts/`
Utility scripts for document conversion and environment setup.

- `convert-html-to-markdown.py` — HTML → Markdown with image handling
- `convert-pdf-to-markdown.py` — PDF → Markdown with LLM image descriptions
- `download-html.py` — HTML downloading via Playwright
- `extract_references.py` — Reference extraction from papers
- `configure-submodule-https-ssh.sh` — Git submodule HTTPS/SSH switching
- `setup-envs.sh` / `setup-envs.ps1` — Environment initialization
- `setup-proxy.sh`, `setup-ssh-reverse-tunnel.sh` — Network setup

### `configs/`
Configuration file examples for AI agents and tools (Codex, tmux).

### `roles/`
AI persona definitions. Currently: `academic_survey_agent.md` — a structured persona for literature surveys with systematic search, synthesis, and academic integrity rules.

### `spec-driven-dev/`
OpenSpec integration tools and specification kits for Python and C++ projects.

## Usage

This repo is intended to be used as a **git submodule**. Reference files directly by path when constructing AI prompts, or copy templates into your project's `context/` directory.

```bash
# Add as submodule
git submodule add <repo-url> magic-context

# Update to latest
git submodule update --remote magic-context
```
