# External Reference Collection Directory Setup

This guide describes a standardized pattern for an "External Reference Collection Directory" used to hold third‑party code, large external checkouts, or symlinked folders that should not be committed directly to the main repository.

Use this pattern in any project where you need to bring external trees into the workspace in a controlled, reproducible way.

## Purpose

The external reference directory (e.g., `extern/`, `third_party/`, `vendor/`, or any other name appropriate for your project) is intended to:

- Hold **checked‑out** or **symlinked** external code and assets used for development, experiments, or builds.
- Keep external bits out of version control while still documenting exactly what is needed and how to obtain it.
- Provide a **single bootstrap entry point** so a new developer or CI job can populate the directory with one command.

The directory itself and its small meta‑files are committed; the downloaded or symlinked contents are not.

## Directory Structure

Pick a dedicated directory name for external references; in this guide we will refer to it as `<external-ref-dir>/`. Common choices:

- `extern/`
- `third_party/`
- `vendor/`

Inside `<external-ref-dir>/`, use the following core files:

```
<external-ref-dir>/
├── README.md          # Explains the purpose and lists managed externals
├── bootstrap.sh       # Populates / refreshes external checkouts/symlinks
└── .gitignore         # Ignores downloaded or symlinked entries
```

Concrete example (using `extern/` as the directory name):

```
extern/
├── README.md
├── bootstrap.sh
├── .gitignore
└── llama.cpp/        # Cloned or symlinked external repo (ignored by Git)
```

## README.md Content

`<external-ref-dir>/README.md` should:

- State that the directory holds external, non‑authored content.
- Enumerate each managed external entry with:
  - Local path under `<external-ref-dir>/`
  - Upstream URL and/or source (e.g., GitHub repo)
  - Purpose in the project (e.g., "C++ LLM runtime", "CUDA kernels for profiling")
- Point to `bootstrap.sh` as the canonical way to populate the directory.

Example outline:

```markdown
# External dependencies

This directory contains external, third‑party code and assets that are fetched or linked on demand.

Managed entries:

- `llama.cpp` – upstream C/C++ LLM runtime
  - Upstream: https://github.com/ggml-org/llama.cpp
  - Local path: <external-ref-dir>/llama.cpp
  - Usage: GPU‑enabled builds and experiments

To (re)populate this directory, run:

```bash
bash <external-ref-dir>/bootstrap.sh
```
```

Replace names and URLs with values appropriate for your project.

## bootstrap.sh Responsibilities

`<external-ref-dir>/bootstrap.sh` should be a small, idempotent script that:

- Validates required tools (at minimum: `git`, and optionally others like `ln`, `curl`, `wget`).
- For each external entry:
  - If it is meant to be a **git clone**:
    - Check whether `<external-ref-dir>/<entry>/.git` already exists.
    - If it exists, leave it alone or optionally pull/refresh.
    - If it does not exist, remove any stale directory and perform a **shallow clone** (e.g., `git clone --depth 1 <url> <target>`).
  - If it is meant to be a **symlink**:
    - Remove any non‑symlink path at the target.
    - Create a symlink using environment‑specific paths (e.g., `$EXTERNAL_ROOT/<entry>`).

Minimal pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERN_DIR="${SCRIPT_DIR}"

require_cmd() {
  for c in "$@"; do
    if ! command -v "$c" >/dev/null 2>&1; then
      echo "missing required command: $c" >&2
      exit 127
    fi
  done
}

require_cmd git

LLAMA_DIR="${EXTERN_DIR}/llama.cpp"

echo "Bootstrapping external dependencies in ${EXTERN_DIR} ..."

if [[ -d "${LLAMA_DIR}/.git" ]]; then
  echo "  - llama.cpp already present; leaving as‑is."
else
  echo "  - cloning llama.cpp (shallow) ..."
  rm -rf "${LLAMA_DIR}"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
fi

echo "Done."
```

Adapt the list of entries and URLs to your project; the core pattern (check‑or‑clone, check‑or‑symlink) should remain the same.

## .gitignore Policy

To keep the parent repository clean while still committing the metadata:

- Place a `.gitignore` inside `<external-ref-dir>/` with rules that ignore the **populated entries**, not the directory itself.
- Do **not** ignore `<external-ref-dir>/` at the workspace root; you want Git to track:
  - `<external-ref-dir>/README.md`
  - `<external-ref-dir>/bootstrap.sh`
  - `<external-ref-dir>/.gitignore`

Example `<external-ref-dir>/.gitignore`:

```gitignore
llama.cpp/
some-other-external/
*/source-data
*/source-data/**
```

This allows each project to declare exactly which entries are generated or environment‑specific (e.g., large clones, symlinks to host‑local data).

## Usage and Best Practices

- **Single entrypoint**: Document in your main `README.md` (or setup docs) that developers should run `<external-ref-dir>/bootstrap.sh` after cloning the repo.
- **No large binaries in Git**: Keep external code and large artifacts out of version control; only the metadata and bootstrap logic are tracked.
- **Environment‑specific paths**: For symlinks, read from environment variables (e.g., `$EXTERNAL_ROOT`, `$MODELS_ROOT`) rather than hard‑coding machine‑specific paths.
- **Idempotent operations**: Running `bootstrap.sh` multiple times should be safe and fast; it should only create or repair missing entries.
- **Clear attribution**: Always document upstream sources and licenses in `README.md` and/or per‑entry metadata so it is obvious what is third‑party.

By standardizing this pattern across projects, AI assistants and human contributors can quickly recognize how external dependencies are managed and how to reconstruct them on any machine.

