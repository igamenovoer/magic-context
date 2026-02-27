---
name: mermaid-graphing
description: Create readable Mermaid diagrams inside Markdown files. Use for flowcharts and sequence diagrams that must render cleanly in common Markdown renderers (e.g., GitHub) without horizontal scrolling. Covers fenced mermaid blocks, init/theme styling, label wrapping with <br/>, and sequenceDiagram layout rules (short IDs, wrapped labels, don’t break identifiers).
compatibility: Requires a Markdown renderer that supports Mermaid fenced code blocks. Mermaid styling/config support varies by renderer/version; validate in Mermaid Live Editor when unsure.
metadata:
  author: agent-system-dissect
  version: "1.0"
---

# Mermaid Graphing

## When to Use This Skill

- You are writing `.md` docs/notes and want diagrams that render in-place via Mermaid.
- You need to document control flow, module interactions, or call flows (especially `sequenceDiagram`).
- Your diagrams are getting too wide or unreadable and need styling/wrapping rules.

## Core Rules

- Always use fenced code blocks with the `mermaid` info string (Mermaid does not render reliably as “inline” Markdown).
- Prefer clarity over completeness: split complex flows into multiple diagrams.
- Keep labels short and wrap with HTML breaks (`<br/>`) when needed. Mermaid does not support raw `\n` line breaks in labels.
- Never break important identifiers (function/class names) across lines; wrap around separators or move context to a new line.

## Markdown Embedding (Correct Pattern)

```markdown
```mermaid
sequenceDiagram
    participant A as Agent<br/>(Python)
    participant S as Service<br/>(C++)
    A->>S: do_work<br/>(arg1,arg2)
    S-->>A: result
```
```

Notes:
- `sequenceDiagram` keyword must be lowercase.
- Keep one diagram per code block; do not mix Mermaid and non-Mermaid text inside the fence.

## Styling and Layout (Avoid Wide Diagrams)

Most “bad” Mermaid diagrams are too wide because participant labels and message labels are long. Fix width first by:

- Using short participant IDs, and putting the long readable label in `as ...`.
- Wrapping long labels with `<br/>` (wrap the label, not the ID).
- Shortening repeated prefixes by declaring an alias once (participant label), then using the short ID everywhere.

For sequence diagrams, use the dedicated guide: `references/sequence-diagram-styling.md`.

## Diagram Init/Theme (Optional)

If your renderer supports Mermaid init blocks, you can apply consistent styling per diagram:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'fontFamily': 'ui-sans-serif', 'fontSize': '14px'}}}%%
sequenceDiagram
    participant A as Agent<br/>(Python)
    participant S as Service<br/>(C++)
    A->>S: do_work<br/>(arg1,arg2)
    S-->>A: result
```

If the init block does not render, remove it and fall back to label-wrapping and short IDs (these work everywhere).

## Quick Checklist (Before Shipping a Diagram)

- [ ] Diagram renders without horizontal scrolling in the target renderer.
- [ ] Labels use `<br/>` for wrapping; no raw `\n` escapes.
- [ ] Function/class names remain intact (no mid-identifier line breaks).
- [ ] Complex flows are split into multiple diagrams.
- [ ] Diagram validated in Mermaid Live Editor when styling/rendering is uncertain.

## Troubleshooting

See `troubleshoot.md` for common Mermaid parse errors and fixes (especially flowchart node labels that include `<br/>` and parentheses).
