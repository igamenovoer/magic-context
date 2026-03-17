# OpenSpec Hack-Through-Testing Principles

Use these principles across all three subskills.

## Core Intent

Optimize for fast discovery of the next real blocker without losing track of safety, evidence, or the difference between a temporary unblocker and a durable design.

## Shared Principles

- Choose one canonical path worth driving first.
- Keep that path non-interactive whenever possible.
- Prefer realistic inputs, but keep unsafe side effects stubbed, redirected, or read-only.
- Fail fast when prerequisites are missing or the system would otherwise hang.
- Preserve logs, outputs, and generated artifacts so disposable runs leave durable evidence.

## How The Principles Translate By Subskill

### `propose`

Translate the principles into an implementation design:

- define the canonical path
- choose the runner surface
- define fixtures, samples, and stubs
- plan preflight checks and timeouts
- decide what artifacts should be saved

### `revise`

Translate the principles into OpenSpec artifacts:

- proposal explains why HTT-compatible execution matters
- design defines entrypoints, safety boundaries, and observable outcomes
- specs encode machine-detectable behavior
- tasks make test-enablement work happen early enough to support patch-forward discovery

### `run`

Apply the principles operationally:

- snapshot the repo without disturbing the active checkout
- patch forward in the throwaway worktree
- verify each workaround before committing it
- keep each issue and workaround reviewable
- end with a synthesis that identifies real fixes

## Stable Capabilities Worth Encoding In Design Or Specs

- one canonical runnable path
- deterministic or bounded inputs
- explicit preflight checks
- bounded failure behavior and timeouts
- machine-detectable success or failure signals
- stable output locations for logs or artifacts
- explicit handling of unsafe integrations during automated execution

## Details That Usually Stay Out Of Permanent Specs

- throwaway branch names
- per-issue workaround commit conventions
- temporary shims used only to advance a disposable session
- session-specific log file names

Let those details shape the operational workflow in `run`, but translate them into durable capabilities in `propose` and `revise`.

## Translation Rule

When a temporary tactic seems useful, ask:

`What permanent capability would make this safe and repeatable without the tactic?`

Examples:

- temporary timeout patch -> permanent fail-fast timeout behavior
- manual local resource linking -> explicit configurable fixture or preflight guidance
- copied run artifacts -> stable artifact output requirement
- narrow skip around unsafe integration -> documented test-mode stub or redirected execution
