---
name: make-program-tutorial
description: Create a clear, reproducible, step-by-step tutorial for using a specific API/SDK/library or an arbitrary set of functions/classes. Use when the user asks for a “how-to”, “tutorial”, or “guide” that includes runnable examples, expected outputs, and basic troubleshooting.
---

# Make Program Tutorial

## Inputs to Collect (ask if missing)

- Target topic:
  - Library/SDK name + version, or
  - The specific functions/classes/modules to cover
- Target audience (beginner vs experienced) and target language/runtime (Python/JS/etc.)
- Execution environment assumptions (OS, GPU/CPU, required services)
- Output location (default to the user’s chosen workspace area; if this tutorial is part of an experiment, place it under that experiment’s `tutorial/`)
- Any constraints (no network, no secrets, offline-only, time budget, etc.)

## Outputs (what to produce)

Create a tutorial folder with:

- `tutorial/step-by-step.md`
- `tutorial/inputs/` (small tutorial-specific inputs, if applicable)
- `tutorial/outputs/` (tutorial-specific outputs, if applicable)

Prefer referencing the real scripts/commands that actually run, rather than pasting large code blocks into the tutorial.

## Workflow

### 1) Identify the canonical sources

- Determine the authoritative docs/repo for the target API/library.
- Pin versions when possible (tag/commit/release).
- If the tutorial is for a set of functions/classes in a local repo, read the source and identify:
  - Import paths, initialization patterns, and required config
  - Any side effects (I/O, network calls, GPU usage)

### 2) Choose a minimal “happy path”

- Define the smallest end-to-end example that demonstrates value:
  - Inputs → processing → outputs
- Prefer real inputs if available; otherwise synthesize minimal inputs that satisfy the API contract.
- Record the exact commands used to run the example and the resulting artifacts.

### 3) Write `tutorial/step-by-step.md`

Follow the template in `templates/step-by-step.md`:
- Start with prerequisites and “what you’ll build/run”.
- Use numbered steps with exact commands.
- For each step, state:
  - What it does
  - What files it reads/writes
  - What success looks like (expected output paths, log lines, or shapes)
- Keep copy-pastable code blocks small; link to scripts for full implementations.

### 4) Include tutorial inputs/outputs

- Place tutorial-ready inputs under `tutorial/inputs/`.
- Place tutorial outputs under `tutorial/outputs/`.
- Keep them minimal and (when relevant) redistributable.

### 5) Add troubleshooting and verification

- Add a short “Troubleshooting” section with the top failure modes:
  - Missing deps, version mismatches, device selection issues, path errors
- Add a “Verification” section:
  - Exactly how to confirm outputs are correct (shapes, counts, sample prints, checksums)

## Guardrails

- Do not include secrets/tokens or require users to paste credentials into the tutorial.
- Avoid “works on my machine” ambiguity: always include the exact commands and expected outputs/paths.
- Prefer stable, minimal dependencies; if multiple installation methods exist, pick one and mention alternatives briefly.

