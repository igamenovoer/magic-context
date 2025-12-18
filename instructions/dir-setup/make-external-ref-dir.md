# External-Reference-Heavy Directory Setup

This guide describes a standardized pattern for an external-reference-heavy directory: a workspace directory that *mostly* contains third-party assets or machine-local data (models, datasets, large checkouts, etc.) plus a small amount of committed metadata and bootstrap logic.

This pattern is designed to work across many projects. In this guide, the primary directory is referred to as `<external-ref-heavy-dir>/` (for example: `models/`, `datasets/`, `extern/`, `third_party/`).

## Purpose

`<external-ref-heavy-dir>/` is intended to:

- Hold external assets used for development, experiments, evaluation, or builds (often too large or too volatile to commit).
- Keep external bits out of version control while still documenting exactly what is needed and how to obtain it.
- Provide one or more **bootstrap entry points** so a new developer (or CI) can populate the directory with one command.
- Keep each external reference **self-contained** (docs + bootstrap + derived artifacts live next to the reference).

The directory itself and its small meta-files are committed; the large or machine-local contents are not.

## Directory Structure

Pick a dedicated directory name for external references; in this guide we will refer to it as `<external-ref-heavy-dir>/`.

Preferred structure (per-reference subdirectories):

```
<external-ref-heavy-dir>/
├── README.md                 # High-level index of managed references
├── bootstrap.sh              # Optional: bootstraps all references (fan-out)
├── .gitignore                # Ignores machine-local / large contents
└── <ref-dir-name>/            # One external reference (dataset/model/checkout/etc.)
    ├── README.md              # What it is, upstream/source, how it's used
    ├── source-data -> ...     # Usually a symlink to machine-local storage (ignored)
    ├── bootstrap.sh           # Creates/repairs source-data and derived artifacts
    ├── metadata/              # Small, committed metadata (manifests, hashes, splits)
    ├── derived/               # Optional: derived outputs (often ignored if large)
    └── ...                    # Small, committed helper files/scripts
```

Key idea: `<external-ref-heavy-dir>/<ref-dir-name>/` is the unit of organization. Keep each reference's docs and automation next to it.

## README.md Content

`<external-ref-heavy-dir>/README.md` should:

- State that the directory holds external, non-authored, and/or machine-local content.
- Enumerate each managed reference with:
  - Local path under `<external-ref-heavy-dir>/` (usually `<ref-dir-name>/`)
  - Upstream URL and/or source (paper/dataset page/HF repo/Git repo/etc.)
  - What files are expected under `source-data` (and how big/where stored)
  - What derivatives exist (if any), and whether they are committed or ignored
- Point to the bootstrap entry point(s):
  - Top-level `<external-ref-heavy-dir>/bootstrap.sh` (if present)
  - Per-ref `<external-ref-heavy-dir>/<ref-dir-name>/bootstrap.sh`

Example outline:

````markdown
# External references

This directory contains external, third-party and/or machine-local assets. Only small metadata and bootstrap scripts are committed.

Managed references:

- `<ref-dir-name>` – short description
  - Source: <upstream-url-or-description>
  - Local path: <external-ref-heavy-dir>/<ref-dir-name>
  - `source-data`: expected contents and where it comes from
  - Derived: list of generated artifacts (if any)

To (re)populate everything, run:

```bash
bash <external-ref-heavy-dir>/bootstrap.sh
```
````

Replace names and URLs with values appropriate for your project.

## Per-Reference Layout

Each `<external-ref-heavy-dir>/<ref-dir-name>/` should be understandable in isolation.

Recommended contents:

- `README.md`: what the reference is, how to obtain it, licensing/attribution notes, and how the project uses it.
- `source-data`: **usually a symlink** to a machine-local location (external disk, shared cache, network mount). Treat it as non-committed.
  - Some projects may implement `source-data/` as a checkout or downloaded directory instead of a symlink; keep it ignored either way.
- `bootstrap.sh`: sets up `source-data` (symlink or download/clone) and optionally creates/refreshes derived artifacts.
- `metadata/`: small committed files such as:
  - subset file lists for a huge dataset
  - split definitions
  - checksums / manifests
  - small sample items for smoke testing
- Derived artifacts (examples; project-specific):
  - a quantized or converted form of a full-precision model
  - a filtered shard/index built from `source-data`
  - a precomputed embedding cache for a small validation subset

Where to put derived artifacts depends on size:

- If small and stable: commit them (often in `metadata/`).
- If large or machine-dependent: generate them into `derived/` and ignore them in Git.

## bootstrap.sh Responsibilities

You can have one top-level `bootstrap.sh` (fan-out) and/or a per-reference `bootstrap.sh`.

`bootstrap.sh` should be a small, idempotent script that:

- Validates required tools (commonly: `ln`, `readlink`, `git`, `curl`/`wget`).
- Uses environment variables for machine-local roots (avoid hard-coded absolute paths).
- Creates or repairs `source-data` (preferably as a symlink) and verifies it points somewhere sensible.
- Optionally creates/refreshes derived artifacts in a deterministic way.
- Emits clear status output and fails fast when required inputs are missing.

Minimal pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${SCRIPT_DIR}"

require_cmd() {
  for c in "$@"; do
    if ! command -v "$c" >/dev/null 2>&1; then
      echo "missing required command: $c" >&2
      exit 127
    fi
  done
}

require_cmd ln

SRC_LINK="${REF_DIR}/source-data"
: "${EXTERNAL_REF_ROOT:?set EXTERNAL_REF_ROOT to your machine-local storage root}"
TARGET="${EXTERNAL_REF_ROOT}/<ref-dir-name>"

echo "Bootstrapping external reference in ${REF_DIR} ..."

rm -rf "${SRC_LINK}"
ln -s "${TARGET}" "${SRC_LINK}"

echo "Done."
```

Adapt the env var names, target computation, and any derived-artifact steps to your project; keep the core pattern (validate → link/fetch → verify → derive) stable.

## .gitignore Policy

To keep the parent repository clean while still committing the metadata:

- Place a `.gitignore` inside `<external-ref-heavy-dir>/` with rules that ignore the **machine-local and derived contents**, not the directory itself.
- Do **not** ignore `<external-ref-heavy-dir>/` at the workspace root; you want Git to track the READMEs and bootstrap scripts.

Example `<external-ref-heavy-dir>/.gitignore`:

```gitignore
*/source-data
*/source-data/**
*/derived/
*/derived/**
```

This makes it hard to accidentally commit large directories if `source-data` stops being a symlink and becomes a real folder.

## Usage and Best Practices

- **One ref, one folder**: Prefer `<external-ref-heavy-dir>/<ref-dir-name>/` as a self-contained unit with its own README + bootstrap.
- **Shallow and deterministic**: If you must clone/fetch, pin versions (tag/commit) and record them in the per-ref README/metadata.
- **No large binaries in Git**: Keep large artifacts out of version control; track only small metadata and deterministic bootstrap logic.
- **Environment-specific paths**: For symlinks, read from environment variables (e.g., `$EXTERNAL_REF_ROOT`) rather than hard-coding machine-specific paths.
- **Idempotent operations**: Running bootstrap multiple times should be safe and fast; it should only create or repair missing pieces.
- **Clear attribution**: Document upstream sources and licenses in the per-ref README so it is obvious what is third-party.

By standardizing this pattern across projects, assistants and human contributors can quickly recognize how external references are managed and how to reconstruct them on any machine.
