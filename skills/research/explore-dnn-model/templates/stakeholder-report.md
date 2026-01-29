# Stakeholder Report: {{model_name_or_repo}} ({{experiment_slug}})

Date: {{yyyy-mm-dd}}

## Executive summary

- Status: {{success|partial|failed}}
- Verified: {{inference_end_to_end|training_sanity|both|neither}}
- Bottom line: {{one paragraph for non-technical readers}}

## What we tested

- Checkpoint/model path (local): `{{checkpoint_path}}`
- Environment: {{pixi|venv}}, Python {{version}}, {{os}}
- Device: {{cuda|rocm|mps|cpu}} ({{gpu_or_cpu_model}})
- Canonical behavior source: {{upstream_repo_or_docs}}

## Key findings

### Input → output contract

- Input expectations: {{file formats / shape / dtype / color / preprocess}}
- Output structure: {{logits/boxes/masks/embeddings + shapes + postprocess}}
- Key parameters: {{brief list; full table should live in experiment report}}

### Performance

- End-to-end latency: mean={{mean_ms}} ms, median={{median_ms}} ms ({{timing_method}})
- Throughput (if measured): {{items_per_s}}
- Peak memory (if measured): RAM={{ram_peak}}, VRAM={{vram_peak}}

### Quality / user metrics (optional)

- Metric(s): {{...}}
- Result vs target: {{...}}
- Notes: {{interpretation + confidence}}

## Analysis

- Does the result match expected behavior from upstream? {{yes/no/unknown}} (why)
- Main bottleneck(s): {{preprocess|model|postprocess|I/O}} (evidence)
- Sensitivities: {{thresholds, resize strategy, dtype, device, batch size}}
- Risks / assumptions:
  - {{risk_1}}
  - {{risk_2}}

## Recommendations / next steps

Prioritized actions:
1. {{next_step_1}}
2. {{next_step_2}}
3. {{next_step_3}}

If integration is planned, specify:
- Recommended invocation path (script/API entrypoint): {{...}}
- Required artifacts to persist: {{stats.json, outputs, logs}}
- Monitoring/alerts to add: {{latency budget, error rate, drift}}

## Pointers

- Experiment report (programmatic): `{{experiment_dir}}/reports/experiment-report.md`
- Key outputs: `{{experiment_dir}}/outputs/`
- Logs: `{{experiment_dir}}/logs/`
- Figures: `{{experiment_dir}}/reports/figures/`
