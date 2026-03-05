# Prompt: Allow Breaking Changes (Assume Unstable Targets)

You are working in a **known-unstable development environment**. The target(s) being developed (this repo, upstream codebases, specific source files, APIs, CLIs, data formats, etc.) are under active iteration, and users explicitly **expect breaking changes**.

## Default stance

- **Do not optimize for backward compatibility.**
- **Do not spend time on migration paths** (schema migrations, compatibility shims, adapters, dual-read/dual-write, deprecation periods, feature flags purely for rollout, etc.).
- **Prefer the clearest, simplest, most maintainable design**, even if it breaks existing behavior or interfaces.

## What to do

- Make the design/API you would build if you were starting fresh today.
- If an existing interface is awkward, remove or reshape it instead of layering compatibility.
- Update all in-repo call sites, tests, and docs to match the new design.
- When writing plans/specs, explicitly state that **breaking changes are allowed** and list the user-visible breaks you are introducing (briefly).

## What not to do (unless explicitly requested)

- Don't keep old parameters/fields/flags "just in case."
- Don't add deprecated aliases or "v1/v2" parallel APIs to preserve old behavior.
- Don't add migration scripts or backward-compatible parsers solely to support old artifacts.
- Don't contort the design to maintain wire-compat, file-compat, or API-compat with previous iterations.

## Still be disciplined

- Breaking changes must be **intentional and justified** by design clarity, correctness, or maintainability (avoid churn for its own sake).
- Preserve stability where it is *intrinsically valuable* (e.g., security invariants, data integrity, explicit contracts that other components in the same change rely on).
- If the user explicitly asks for compatibility/migration support, follow that request and treat it as a first-class requirement.

## When Breakage Comes From the Target Itself

Sometimes breaking behavior originates from **conflicts with the existing target** (e.g., the repo's current architecture, older design decisions, or implicit constraints you discover while reading the code).

In that case, do one of:

- **Revise the existing design** (preferred) so the overall system is coherent, even if that means changing pre-existing interfaces/structures.
- **Document the breakage** in the relevant design/spec/planning doc:
  - Clearly list what conflicts, what breaks, and why it is acceptable in this unstable context.
  - Record it as an explicit **next-step correction target** (e.g., a TODO checklist item, a follow-up task/plan, or a tracked design debt note).

If a specific module/function/script is **known to be broken** after the design change, make it **fail fast**:

- Raise an explicit error at its entrypoint (or exit non-zero for CLIs) with a clear message indicating it is intentionally broken due to the design change and what needs fixing.
- Prefer a loud, deterministic failure over silent no-ops or undefined behavior, so developers immediately see what to fix.
