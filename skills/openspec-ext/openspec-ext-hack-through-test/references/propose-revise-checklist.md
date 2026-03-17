# Propose And Revise Checklist

Use this checklist in `propose` and `revise` mode.

## Canonical Path Questions

- What is the first path worth driving end to end?
- What command, CLI surface, script, or interaction sequence should expose it?
- What counts as success for that path?
- What output or artifact proves it reached the intended point?

## Safety And Operability Questions

- What prerequisites must be checked before the path starts?
- What dependency is unsafe to hit directly and therefore needs a stub, redirect, or read-only mode?
- What part is likely to hang and therefore needs bounded failure behavior?
- What logs or generated artifacts should survive a failed run?

## `propose` Output Shape

When proposing a design, include these sections unless the user asks for a different shape:

- `# HTT-Ready Design Proposal: <topic>`
- `## Canonical Path`
- `## Runner Surface`
- `## Fixtures, Samples, And Stubs`
- `## Fail-Fast And Timeout Strategy`
- `## Logs, Outputs, And Artifacts`
- `## Implementation Order`
- `## Risks And Open Questions`

## `revise` Artifact Checklist

### Proposal

- explain why HTT-compatible execution matters
- name the canonical path
- make any scope cuts explicit

### Design

- define the non-interactive entrypoint or sequence
- define fixtures, sample data, and stub boundaries
- define preflight checks and bounded failure behavior
- define output and artifact locations
- define safe handling for external dependencies

### Specs

Prefer concrete requirements such as:

- `The system SHALL expose one canonical non-interactive invocation for <flow>.`
- `The automation flow SHALL terminate with a non-zero result when required prerequisites are missing.`
- `The automation flow SHALL emit machine-detectable success or failure outcomes.`
- `The automation flow SHALL persist logs or generated artifacts to a deterministic or caller-provided location.`
- `Unsafe external writes SHALL be disabled, stubbed, or redirected during automated runs.`

Avoid vague requirements such as:

- `The feature should be easy to test.`
- `The workflow should work in CI.`
- `Testing should be automated later.`

### Tasks

Prefer this order:

1. expose the canonical runner surface
2. add fixtures, samples, or stub boundaries
3. add preflight checks and bounded failure behavior
4. add logging and artifact capture
5. add automated smoke or integration coverage for the canonical path

## Smells

Revise the design or change artifacts if you find any of these:

- the only stated verification method is manual execution
- the design assumes mid-run operator intervention
- the specs never define a success signal
- the tasks postpone test-enablement work until after the main feature is done
- the change leaves unsafe external writes enabled with no safety story
