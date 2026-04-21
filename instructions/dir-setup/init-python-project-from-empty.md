# Initialize a Python Project From Empty

This guide describes a reusable sequence for bootstrapping a new Python project from an empty repository. It is intentionally generic: replace placeholders such as `<project-root>`, `<package-name>`, `<repo-url>`, and `<shared-skill-repo>` with values for your environment.

## Goal

Start from an empty directory and end with:

- a git repository with a clean Python project layout
- Pixi-based environment management
- `extern/tracked/` submodules for reusable external resources
- a populated `context/` workspace for AI-assisted development
- optional agent/skill wiring for local tooling
- contributor-facing docs such as `README.md` and `AGENTS.md`

## Recommended Order

1. Create the repository and initialize git.
2. Initialize Pixi and package metadata.
3. Scaffold the standard Python project directories.
4. Create the `context/` directory structure.
5. Add `magic-context` under `extern/tracked/`.
6. Add any additional tracked submodules under `extern/tracked/`.
7. Wire project skills or instructions via symlink when desired.
8. Update `README.md`, `.gitignore`, and `AGENTS.md`.
9. Validate the environment and commit the initial scaffold.

## 1. Create the Repository

```bash
mkdir -p <project-root>
cd <project-root>
git init -b main
git remote add origin <repo-url>
```

Use a normalized package name early. A common mapping is:

- project name: `<project-name>`
- Python package: `<package_name>`

## 2. Initialize Pixi

Prefer `pyproject.toml` format unless a project explicitly wants `pixi.toml`.

```bash
pixi init . --format pyproject -c conda-forge
pixi add python=<python-version>
```

Then add the editable local package entry and any baseline dependencies your project expects.

## 3. Scaffold the Project Layout

Create the standard layout described in `make-python-project-dir.md`:

```text
.github/workflows/
src/<package_name>/
extern/tracked/
extern/orphan/
scripts/
tests/{unit,integration,manual}/
docs/
context/
tmp/
```

Also create:

- `src/<package_name>/__init__.py`
- `extern/.gitignore` to ignore `extern/orphan/*` while keeping `extern/orphan/README.md`
- `README.md`, `.gitignore`, and any project-specific config files

## 4. Create the Context Workspace

Create the `context/` structure using `make-context-dir.md`. At minimum include:

```text
context/design/
context/hints/
context/instructions/
context/issues/{features,known,resolved}/
context/logs/
context/plans/
context/refcode/
context/roles/
context/rules/
context/skills/
context/summaries/
context/tasks/{working,done,backlog}/
context/tools/
```

Each top-level context directory should include a short `README.md`.

## 5. Add `magic-context`

If the project uses shared setup instructions, reusable skills, or common workflow snippets, add `magic-context` as a tracked submodule first:

```bash
git submodule add -b main https://github.com/igamenovoer/magic-context.git extern/tracked/magic-context
```

This gives the repository a stable in-tree location for:

- directory setup guides
- reusable skills
- project rules and snippets
- shared workflow scripts

After adding it, use its instructions to drive the rest of the bootstrap process rather than re-creating conventions from scratch.

## 6. Add Shared Resources as Submodules

Prefer `extern/tracked/<repo>/` for pinned external dependencies and reusable project toolkits.

```bash
git submodule add -b main <shared-skill-repo> extern/tracked/<repo-name>
```

Examples:

- shared workflow/tooling repo
- benchmark dataset repo
- reference starter kit repo

If the upstream repository uses Git LFS, document the required follow-up:

```bash
git submodule update --init --recursive
git -C extern/tracked/<repo-name> lfs pull
```

In many projects, `magic-context` is the first tracked submodule and other submodules are added afterward.

## 7. Optionally Expose Shared Skills

If local agent tooling expects project-local skill entrypoints, create relative symlinks into the tracked submodule instead of copying files:

```bash
ln -s ../../extern/tracked/magic-context/skills/<group>/<skill-name> .codex/skills/<skill-name>
ln -s ../../extern/tracked/magic-context/skills/<group>/<skill-name> .claude/skills/<skill-name>
ln -s ../../extern/tracked/magic-context/skills/<group>/<skill-name> .github/skills/<skill-name>
```

If `.codex/` or `.claude/` are git-ignored by default, narrow the ignore rules so only the intended tracked symlinks are unignored.

## 8. Document the Setup

Update:

- `README.md`: setup steps, required submodule initialization, dataset paths, and run commands
- `AGENTS.md`: repository-specific contributor guidance
- `.gitmodules`: ensure tracked submodules record the intended branch

Prefer repository-relative paths in docs:

- `extern/tracked/<repo-name>`
- `$PWD/extern/tracked/<dataset-repo>`

Avoid machine-specific absolute paths in committed documentation.

## 9. Validate Before First Commit

Run lightweight checks:

```bash
git submodule status --recursive
pixi run python -c "import <package_name>"
```

If the repo includes runnable scripts, validate at least one happy-path command such as:

```bash
pixi run python scripts/<entrypoint>.py
```

## 10. Commit the Scaffold

Use a short imperative commit message, for example:

- `scaffold project layout`
- `initialize pixi project`
- `add shared toolkit submodules`

If the repository is still empty of real feature work, keep the initial scaffold commit focused on setup only.

## Notes

- Prefer relative symlinks over absolute symlinks so the repository remains portable.
- Prefer `extern/tracked/magic-context` as the canonical location for shared setup instructions and reusable project skills.
- Treat changes to `extern/tracked/*` as reviewed dependency updates.
- Keep `extern/orphan/` for local-only clones that should not be committed.
- When a setup step changes another repo-owned workflow, fix the dependent docs or config in the same change.
