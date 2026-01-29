# Experiment Report: {{model_name_or_repo}} ({{experiment_slug}})

Date: {{yyyy-mm-dd}}

> Generated programmatically from `{{experiment_dir}}/outputs/`. Do not edit manually.

Status: {{success|partial|failed}}

Verified: {{inference_end_to_end|training_sanity|both|neither}}

## Pointers

- Experiment dir: `{{experiment_dir}}`
- Checkpoint/model path (local): `{{checkpoint_path}}` (type: `{{pt|pth|onnx|engine|...}}`, sha256: {{optional}})
- Upstream repo/docs (used for canonical behavior): {{url_or_path}}
- Logs: `{{experiment_dir}}/logs/`
- Outputs: `{{experiment_dir}}/outputs/`
- Reports: `{{experiment_dir}}/reports/`
- Figures (for this report): `{{markdown_dir}}/figures/`

## Environment

- Env manager: {{pixi|venv}}
- Python: {{version}}
- OS: {{os}}
- Device: {{cuda|rocm|mps|cpu}} ({{gpu_or_cpu_model}})
- Key libs: {{torch/onnxruntime/opencv/etc}}={{version}}, numpy={{version}}

## Key parameters

List every non-default parameter you set for this experiment (preprocess, postprocess, runtime, thresholds, etc.).

| Name | Meaning | Value |
|---|---|---|
| {{param_name}} | {{what it controls}} | {{value}} |

## I/O Contract

### Input

- Modality/formats: {{image: jpg/png | video: mp4 | audio: wav | tensor: npy | ...}}
- Shape/dtype/color: {{e.g., 1x3x640x640 float32, RGB, CHW}}
- Preprocess: {{resize/letterbox + normalization + other}}
- Example input(s): `{{experiment_dir}}/inputs/{{...}}`

### Output

- Raw output(s): {{logits/boxes/masks/embeddings + shapes}}
- Postprocess: {{nms/thresholds/decode}}
- Example output(s): `{{experiment_dir}}/outputs/{{...}}`

## Results

- Timing method: {{cold + warm runs}}
- End-to-end latency (ms): mean={{mean_ms}}, median={{median_ms}}
- Throughput (optional): {{items_per_s}}
- Peak memory (optional): RAM={{ram_peak}}, VRAM={{vram_peak}}

### Figures (if inputs/outputs include images)

If the experiment involves images, copy the key input/output images into `{{markdown_dir}}/figures/` and embed them here using relative paths:

**Input images**

![{{caption}}](figures/{{input_image_filename}})

**Output images**

![{{caption}}](figures/{{output_image_filename}})

## User Metrics (optional)

- Metric(s): {{...}}
- Result vs target: {{...}}

## Issues / Notes

- {{issue_or_gotcha_1}}
- {{issue_or_gotcha_2}}
