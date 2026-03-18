---
name: openspec-ext-hack-through-test
description: "Manual invocation only. OpenSpec-specific hack-through-testing workflow targeting production-level end-to-end paths using real data and real user workflows — not CI smoke/unit/integration tests. Three subskills: `propose` to create an OpenSpec change with HTT-ready test cases (automatic scripts and interactive guides) by invoking `openspec-propose` or `openspec-ff-change`, `revise` to update an existing OpenSpec change so its artifacts support hack-through-testing-driven implementation and testing, and `run` to exercise an implemented OpenSpec change in a disposable snapshot worktree using the full hack-through-testing loop. Use when the user explicitly asks for `openspec-ext-hack-through-test`, points to `openspec/changes/...` while asking to propose, revise, run, exercise, or prepare work under hack-through-testing principles, or wants OpenSpec work shaped for fast blocker discovery through patch-forward testing."
---

# OpenSpec Extension: Hack Through Test

Manual invocation only: use this skill only when the user explicitly wants this workflow.

Use this skill as the OpenSpec-specific version of hack-through-testing.

It has three subskills:

- `propose`: create an OpenSpec change with HTT-ready test cases (automatic scripts and interactive guides) by invoking `openspec-propose` or `openspec-ff-change`
- `revise`: revise an existing OpenSpec change so its artifacts support HTT-friendly implementation and testing
- `run`: drive an implemented OpenSpec change through the full hack-through-testing loop in a disposable snapshot worktree

This skill is self-contained. Use only the files bundled inside this skill directory for its workflow and references.

## Testing Philosophy: Production-Level End-to-End, Not CI

Hack-through-testing targets **production-level end-to-end paths**: real data, real user workflows, real API calls, real output artifacts. It is not a CI smoke run, not a unit test harness, and not a mock-based integration check.

In all three subskills, the canonical path to propose, revise for, or run must be a **real production user scenario** — the full flow a real user would perform, end to end. Do not treat existing CI test suites, smoke tests, or mock-based integration tests as the target.

**If the only testable surface you can identify from the change artifacts or repository context is CI-style**, stop and ask the user what the real production user path or end-to-end scenario is before proceeding. For example:

> I can see unit/smoke/integration tests already covered by CI. What's the real production user path you want to exercise — the end-to-end scenario, the live data workflow, or a specific user journey?

## Mode Selection

Choose the subskill before doing deeper work:

- Use `propose` when the user wants a design or implementation approach, there is no existing change to edit, or the request is grounded mainly in chat context.
- Use `revise` when the user points to an existing OpenSpec change and wants its proposal, design, specs, or tasks made compatible with hack-through-testing principles.
- Use `run` when the user says the OpenSpec change is already implemented, or clearly asks to run, test, exercise, or patch forward through the implementation.

If the request is ambiguous:

- prefer `revise` for an existing change path when the ask is about design or artifact readiness
- prefer `run` only when the user clearly wants execution against implementation
- ask one concise question only if choosing the wrong mode would waste substantial work

## Internal References

Read `references/hack-through-testing-principles.md` in every mode.

Then read the mode-specific references:

- `propose` or `revise`: `references/propose-revise-checklist.md`
- `run`: `references/run-mode-openspec-adaptation.md`

For `run`, also use these bundled resources:

- `scripts/create_snapshot_worktree.sh`
- `references/log-template.md`
- `references/issue-template.md`
- `references/git-snapshot-plumbing.md`

## Shared OpenSpec Context Rules

For `revise` and `run`, resolve the change through OpenSpec CLI before assuming artifact layout:

```bash
openspec status --change "<change-name>" --json
openspec instructions apply --change "<change-name>" --json
```

Use these outputs to determine:

- `changeDir`
- `schema`
- `contextFiles`
- current artifact status

Read every existing file referenced by `contextFiles`.
If a value is a glob such as `specs/**/*.md`, expand it on disk and read the matching files.

Use these commands when they add signal:

```bash
openspec show --type change --json --no-interactive "<change-name>"
openspec validate --type change --strict --json --no-interactive "<change-name>"
```

Do not assume files such as `proposal.md` or `design.md` exist without checking the OpenSpec output first.

## Shared Artifact Conventions For `propose` And `revise`

When the change includes automatic or agent-driven test execution as part of the intended delivery, use these artifact conventions unless the user explicitly asks for a different structure or the target repository already has a stronger local convention:

### Design-phase artifacts inside the OpenSpec change

- Add or revise `openspec/changes/<change>/testplans/`.
- Store one Markdown plan per automatic case as `testplans/case-<case-id>.md`.
- Treat these `testplans/` files as design-phase artifacts, not as final implementation docs.
- Each `case-*.md` should describe:
  - goal
  - intended implemented assets
  - intended runner surface
  - ordered steps
  - expected evidence
  - failure signals
- Each `case-*.md` should include at least one Mermaid `sequenceDiagram` that shows the canonical flow for that case.

### Implemented artifacts under the target implementation root

- Choose the implementation root based on the feature's owned directory structure. For example, for a tutorial pack or demo pack, prefer a pack-local directory such as `<implementation-root>/autotest/`.
- Put implemented test assets under `<implementation-root>/autotest/`.
- For each case, create two variants:

#### Automatic variant

- `autotest/case-<case-id>.<case-script-ext>` — an executable script that runs unattended and exits with a clear pass/fail signal.
- Choose `<case-script-ext>` to match the target project, operating system, and execution model. It is not fixed to `.sh`.

#### Interactive variant

- `autotest/case-<case-id>.md` — a step-by-step interactive test guide designed for agent-driven execution with user observation.
- Each interactive guide must contain inline instructions that explain what to do at each step, what to observe, and what success or failure looks like.
- Do not reduce interactive guides to "run `case-<id>.<ext>`". They are independent test procedures where an agent executes steps on the user's behalf while the user watches results and decides how to proceed.
- Structure each guide as an ordered sequence of steps. Each step should include:
  - what the agent should do (command, action, or check)
  - what the expected outcome is
  - what to look for to confirm success or detect failure
  - decision points where the user may choose to continue, retry, or investigate

- Do not require the implemented interactive guides to match the design-phase `openspec/.../testplans/` files line by line. They describe the shipped behavior for interactive testing; the change-owned `testplans/` remain the design source of truth.

### Shared implementation helpers

- Put shared automatic-test scripts, shell libraries, and reusable helper functions under `<implementation-root>/autotest/helpers/`.
- Case implementations should call or source shared logic from `autotest/helpers/` instead of duplicating common behavior.

### Standalone harness script

- Use a standalone harness script rather than bundling HTT case selection into an unrelated operator/demo wrapper.
- Place the harness under `<implementation-root>/autotest/<project-dependent-harness-script>`.
- Choose the harness language and extension to match the target project, operating system, and execution model. Use the same reasoning for each implemented case script. Examples:
  - `.sh` for POSIX shell-first repos
  - `.ps1` for PowerShell-first Windows automation
  - `.py` for Python-oriented projects
  - `.ts` for TypeScript/Node-oriented projects
- The harness should own case selection, shared preflight orchestration, and dispatch into the implemented `case-*.<case-script-ext>` files.

## Subskill: `propose`

Use `propose` to create an OpenSpec change with HTT-ready test cases — both automatic scripts and interactive guides.

### Goal

Create an OpenSpec change centered on one canonical **production-level end-to-end user path**: the full flow a real user would perform with real data, from input to output. The change must include test case designs (both automatic and interactive variants) that favor fail-fast behavior, explicit artifact capture, and an implementation order that supports patch-forward discovery.

If the feature intent from context does not make the real user path obvious, ask the user to describe it before designing. Do not default to designing a CI-style test harness.

### Output

An OpenSpec change created by invoking `openspec-propose` or `openspec-ff-change`, with test case artifacts included in the change design.

### Workflow

1. Gather the feature intent from chat context and only the repository files needed to ground the design.
2. Identify the first canonical path worth automating or driving forward.
3. Define:
   - runner surface or command shape
   - fixtures, samples, or stub boundaries
   - fail-fast and timeout behavior
   - external dependency safety boundaries
   - logs, outputs, and artifacts worth preserving
   - implementation order that enables incremental validation
   - design-phase `testplans/` layout with both automatic and interactive case plans
   - implemented `autotest/` layout: automatic scripts (`case-<id>.<ext>`) and interactive guides (`case-<id>.md`)
4. Invoke `openspec-propose` (for a single-step proposal) or `openspec-ff-change` (to fast-forward through all artifacts) to create the OpenSpec change. Pass the HTT design context so the change artifacts encode the canonical path, test case variants, and safety model.
5. After the change is created, verify that the resulting artifacts include test case plans for both automatic and interactive variants.
6. Call out open questions that would materially affect safe or useful hack-through-testing.

## Subskill: `revise`

Use `revise` to update an existing OpenSpec change so it supports hack-through-testing-friendly implementation and later execution.

### Goal

Revise the change artifacts in place so the implementation can expose one canonical non-interactive path, fail fast on missing prerequisites, preserve useful outputs, and isolate unsafe dependencies during automated or agent-driven testing.

### Output

Update the existing change artifacts in place.
Typical targets are:

- `proposal.md`
- `design.md`
- `specs/**/*.md`
- `tasks.md`

### Workflow

1. Resolve the change with the shared OpenSpec CLI rules.
2. Use the bundled checklist to identify the canonical path and the main HTT-compatibility gaps.
3. Revise the change artifacts so they encode stable capabilities rather than throwaway workaround details.
4. Make proposal, design, specs, and tasks point at the same canonical path and safety model.
5. When test cases are part of the design, make the artifacts agree on:
   - design-phase `openspec/changes/<change>/testplans/case-*.md` covering both automatic and interactive variants
   - Mermaid sequence diagrams in each case plan
   - implemented automatic scripts: `<implementation-root>/autotest/case-*.<ext>`
   - implemented interactive guides: `<implementation-root>/autotest/case-*.md`
   - shared helpers under `<implementation-root>/autotest/helpers/`
   - a standalone harness under `<implementation-root>/autotest/<project-dependent-harness-script>`
6. Validate the change when helpful:

```bash
openspec validate --type change --strict --json --no-interactive "<change-name>"
```

If important product decisions are still unresolved, make the smallest safe assumption only when the artifacts already lean that way; otherwise record the open question or tell the user a review or decision pass is still needed.

## Subskill: `run`

Use `run` when the OpenSpec change is already implemented and the user wants the full hack-through-testing workflow.

### Goal

Drive the implemented change forward in a disposable snapshot worktree along the **real production user path** — using real data and real service calls within safe limits — patching around blockers just enough to reach later failures quickly, while keeping every workaround reviewable and ending with a synthesis of the real fixes.

Do not target existing CI tests, unit tests, or smoke scripts as the canonical path. If the implementation's only runnable surface is CI-oriented, stop and ask the user what the real end-to-end user scenario looks like before starting the loop.

### Output

Produce:

- a helper-managed HTT log directory with a session log, issue notes, and saved run artifacts
- disposable workaround commits on a throwaway branch
- a final synthesis that separates throwaway unblockers from durable fixes

### Workflow

1. Resolve the change with the shared OpenSpec CLI rules.
2. Use `references/run-mode-openspec-adaptation.md` to determine:
   - the canonical path to exercise
   - the likely implementation entrypoints
   - whether the implementation is complete enough to run
3. Snapshot the current repository state into a throwaway worktree with the bundled helper:

```bash
bash ./scripts/create_snapshot_worktree.sh --topic TOPIC_SLUG
```

Optional arguments:

```bash
bash ./scripts/create_snapshot_worktree.sh --repo PATH --topic TOPIC_SLUG --branch hacktest/TOPIC_SLUG --htt-home HTT_HOME --path WORKTREE_PATH
```

4. Start logs using the bundled templates.
5. Run the standard hack-through-testing loop:
   - execute the next step in the canonical path
   - record failures and save artifacts
   - apply the smallest reversible workaround that unlocks progress
   - re-run to verify the workaround
   - commit only verified workaround steps
   - continue until success, stop-rule exhaustion, or a high-risk boundary
6. Finish with a synthesis that maps findings back to the OpenSpec change and identifies any needed follow-up in `revise` mode.

If the change is not implemented enough to exercise responsibly, stop and report that the correct next step is `propose` or `revise`, with the concrete evidence that led to that conclusion.

## Shared Heuristics

- Prefer one meaningful canonical path over broad but vague coverage.
- Prefer machine-detectable outcomes over human-only verification.
- Prefer explicit preflight failure over ambiguous hangs.
- Prefer realistic inputs with safe external boundaries.
- Prefer capturing logs and generated artifacts so disposable runs still produce durable evidence.
- Translate temporary workaround ideas into stable design capabilities when working in `propose` or `revise`.

## Example Prompts

- `Use $openspec-ext-hack-through-test in propose mode and create an OpenSpec change with HTT-ready test cases for this feature.`
- `Use $openspec-ext-hack-through-test in revise mode on openspec/changes/<change> and update the artifacts for HTT compatibility.`
- `Use $openspec-ext-hack-through-test in run mode on openspec/changes/<change> and patch forward through the implemented flow.`
- `Take this OpenSpec change and either revise or run it under hack-through-testing principles, depending on what the current state supports.`

Pointing to a file or directory under `openspec/` counts as the same trigger signal as explicitly saying `openspec`.

## Guardrails

- Do not revise change artifacts in `run` mode unless the user explicitly asks to switch modes.
- Do not assume a fixed OpenSpec artifact layout; use OpenSpec CLI output first.
- Do not reference workflow files outside this skill directory.
- Do not present temporary workarounds discovered in `run` mode as the final fix.
- Do not encode disposable-worktree mechanics as permanent product requirements when using `propose` or `revise`.
- Do not collapse design-phase OpenSpec `testplans/` and implemented `autotest/` artifacts into the same artifact role; keep the design-versus-implementation distinction explicit.
- Do not hide shared helper logic inside unrelated case scripts when the target implementation has multiple automatic cases; direct shared logic into `autotest/helpers/`.
- Do not reduce interactive guides (`autotest/case-*.md`) to wrappers that just say "run the automatic script"; they must be independent step-by-step procedures for agent-driven execution with user observation.
- Do not keep going silently once the remaining path forward would require high-risk product changes or unsafe external side effects.
- Do not target CI-style tests (unit, smoke, mock-based integration) as the canonical path. Ask the user for the real production user path if it is not clear from context.
