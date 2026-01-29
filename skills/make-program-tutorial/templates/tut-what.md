# {{Tutorial Title}}

## What you will do

- Use: `{{library_or_api_name}}` (version: {{version_or_commit}})
- Goal: {{one-sentence goal}}
- Output: `tutorial/outputs/{{...}}`

## Prerequisites

- Runtime: {{python/node/etc}} {{version}}
- OS/hardware: {{cpu/gpu/etc}}
- Dependencies: {{how_to_install_or_where_it_is_declared}}
- Optional: {{services, model checkpoints, datasets}}

## Project layout (for this tutorial)

```
tutorial/
  tut-<what>.md
  inputs/
  outputs/
```

## Step-by-step

### 1) Verify environment

Run:

```bash
{{command_to_print_versions}}
```

Expected:
- {{expected output}}

### 2) Prepare inputs

- Source: {{workspace_path_or_synthesized}}
- Files:
  - `tutorial/inputs/{{file}}`

Run:

```bash
{{command_to_prepare_inputs}}
```

Expected:
- `tutorial/inputs/{{file}}` exists

### 3) Run the minimal “happy path”

Run:

```bash
{{command_to_run_example}}
```

Expected outputs:
- `tutorial/outputs/{{artifact}}`
- Console output contains: {{key log line / shape / count}}

### 4) Interpret results

- Output contract:
  - {{shape/type/meaning}}
- Quick sanity checks:
  - {{e.g., non-empty result, value ranges, image renders}}

## Troubleshooting

- `ModuleNotFoundError: {{pkg}}`: {{how to install / where to add deps}}
- `{{common error}}`: {{fix}}
- Device selection issues (GPU/CPU): {{how to force device}}

## Verification

- Re-run command: `{{command_to_run_example}}`
- Confirm:
  - Output file(s) exist under `tutorial/outputs/`
  - Output shapes/values match the contract

## References

- Docs: {{url}}
- Source: {{url}}
- Version pin: {{tag_or_commit}}

