# Experiment Report: {{model_name_or_repo}} ({{experiment_slug}})

Date: {{yyyy-mm-dd}}

## Summary

- Goal: Verify the model runs end-to-end (default: inference) in the current Python environment, document how to use it, and capture input/output + timing/benchmark results.
- Status: {{success|partial|failed}}
- What worked: {{one-liner}}
- What failed / missing: {{one-liner}}

## Artifacts

- Experiment directory: `{{experiment_dir}}`
- Plan: `{{experiment_dir}}/plan.md`
- Checkpoint: `{{checkpoint_path}}` (type: `{{pt|pth|onnx|engine|...}}`, size: {{bytes_or_human}}, sha256: {{optional}})
- Upstream source/docs:
  - Repo: {{url_or_path}}
  - Version pin: {{tag_or_commit_or_release}}
  - License notes: {{if_applicable}}

## Environment

- Environment manager: {{pixi|venv}}
- Python: {{version}}
- OS: {{os}}
- Device used: {{cuda|rocm|mps|cpu}} (details: {{gpu_model_or_cpu_model}})
- Key runtime libs:
  - {{torch|onnxruntime|tensorrt|opencv}}: {{version}}
  - numpy: {{version}}
  - opencv-python: {{version}}
  - pillow: {{version}}

## Dependency Resolution

### Required dependencies (derived)

- Runtime/framework: {{deps}}
- Model-specific libs: {{deps}}
- Utilities: {{deps}}
- Optional acceleration: {{deps}}

### Missing deps and chosen strategy

- Missing: {{deps}}
- User choice:
  - Pixi: {{modify_current_manifest|new_pixi_env}}
  - Venv: {{install_into_current_venv|new_venv}}
- Version strategy used:
  - First attempt: latest-resolved via {{pixi|pip|uv}}
  - Fallback: pinned to upstream constraints (if needed): {{constraints}}

### Notes / gotchas

- {{e.g., CUDA version mismatch, ABI, wheels unavailable, conflicts}}

## Model Contract (I/O)

### Inputs

- Modality: {{image|video|audio|text|tensor}}
- Accepted formats: {{jpg/png/mp4/wav/npy/...}}
- Expected shape/dtype: {{e.g., (1,3,640,640) float32}}
- Color space / channel order: {{RGB/BGR, CHW/HWC}}
- Preprocessing:
  - Resize/letterbox: {{details}}
  - Normalize: {{mean/std or scale}}
  - Other: {{tokenizer, sampling rate, etc.}}

### Outputs

- Raw outputs: {{tensor shapes / logits / boxes / masks / embeddings}}
- Postprocessing:
  - {{e.g., NMS thresholds, decode steps}}
- Output artifacts produced in this experiment:
  - `{{experiment_dir}}/outputs/{{...}}`

## How to Run (Repro Commands)

All commands are run from `{{experiment_dir}}`.

- Smoke test (imports + load model): `{{cmd}}`
- Inference run: `{{cmd}}`
- Visualization / decoding: `{{cmd}}`

## Inputs Used

### Inputs discovery

- Workspace search paths checked: {{e.g., datasets/, downloads/, ...}}
- Real inputs found: {{yes/no}}
- Synthetic inputs used: {{yes/no}} (why: {{reason}})

### Input inventory

| Input | Source | Notes |
|---|---|---|
| `inputs/{{file}}` | {{workspace_path|synthesized}} | {{resolution/duration/etc}} |

## Experiment Runs

### Run matrix

| Run ID | Script/Command | Device | Params | Inputs | Outputs |
|---|---|---|---|---|---|
| {{run_01}} | `scripts/{{script}}` | {{device}} | {{batch/precision/etc}} | `inputs/{{...}}` | `outputs/{{...}}` |

### Input → Output mapping (key examples)

| Input | Output(s) | Notes |
|---|---|---|
| `inputs/{{file}}` | `outputs/{{file_or_dir}}` | {{e.g., top-1 label, boxes count}} |

## Timing / Benchmark

### End-to-end timing (default)

- Method: {{cold run + warm runs}}
- Batch size: {{n}}
- Precision: {{fp32|fp16|int8|...}}

| Stage | Mean (ms) | Median (ms) | Notes |
|---|---:|---:|---|
| Total end-to-end | {{}} | {{}} | {{includes preprocess+model+postprocess}} |
| Preprocess | {{}} | {{}} | {{optional}} |
| Model | {{}} | {{}} | {{optional}} |
| Postprocess | {{}} | {{}} | {{optional}} |

### Throughput / memory (if measured)

- Throughput: {{items/s}}
- Peak RAM: {{MB/GB}}
- Peak VRAM: {{MB/GB}}
- Profiling notes: {{torch profiler / onnxruntime profiling / nsys / etc.}}

## User Metrics (Optional)

If the user provided metrics/targets, capture:

- Metric(s): {{mAP@0.5, IoU, F1, accuracy, latency budget, etc.}}
- Evaluation set: {{what inputs were used}}
- Result: {{value}}
- Target: {{value}}
- Notes: {{why met/not met, next steps}}

## Issues / Pitfalls

- {{issue_1}} (symptom: {{...}}, fix/workaround: {{...}})
- {{issue_2}}

## Conclusions

- Can the model run end-to-end here? {{yes/no/partially}}
- Recommended “known-good” invocation: {{script/command}}
- Next steps: {{e.g., add more real inputs, tune preprocessing, evaluate accuracy, quantize, etc.}}

## Appendix

- Logs: `{{experiment_dir}}/outputs/{{logs}}`
- Reports: `{{experiment_dir}}/reports/{{...}}`
- Tutorial: `{{experiment_dir}}/tutorial/step-by-step.md`
