# Run-Mode OpenSpec Adaptation

Use this reference to adapt the generic hack-through-testing loop to an implemented OpenSpec change.

## Resolve The Target From The Change

Use OpenSpec CLI output as the source of truth for:

- change name
- schema
- change directory
- context files

Then inspect the change artifacts to identify:

- the canonical path the change expects to support
- the likely implementation entrypoints
- sample inputs, fixtures, or environment assumptions
- success and failure signals that the change intends

Read repository code only where the artifacts point you.

## Choosing What To Exercise

Prefer the narrowest high-signal path already implied by the change:

- one CLI command
- one integration test command
- one build-then-run sequence
- one multi-step agent-driven interaction sequence

Do not widen scope just because the implementation exposes additional surfaces.

## Deciding Whether The Change Is Ready For `run`

Stay in `run` mode only if:

- the target path appears to have at least partial implementation
- there is enough information to execute a meaningful command or sequence
- patch-forward testing can reveal new blockers instead of pure design gaps

Switch back toward `propose` or `revise` if:

- the canonical path is still undefined
- the implementation entrypoint does not exist yet
- the change artifacts contradict each other too much to pick a responsible target
- every next step would be speculative architecture work rather than a narrow unblocker

## Mapping Findings Back To OpenSpec

At the end of the run, classify each important result:

- implementation bug against an adequate change
- design gap that the change failed to specify
- artifact mismatch where the implementation and change disagree
- environment or setup gap that should not be treated as a product issue

Call these distinctions out in the synthesis so the next step is obvious.

## OpenSpec-Specific Synthesis Questions

- Did the implementation follow the canonical path described by the change?
- Which blockers reveal missing or ambiguous design requirements?
- Which blockers are purely implementation defects?
- Does the change need a `revise` follow-up before more implementation work is worthwhile?
