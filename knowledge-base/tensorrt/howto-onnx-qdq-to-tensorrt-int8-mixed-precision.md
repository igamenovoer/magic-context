# How to Convert Pre-Quantized QDQ ONNX Models to Mixed-Precision TensorRT Safely

## HEADER
- **Purpose**: Provide practical guidance for converting pre-quantized QDQ ONNX models (e.g., from QAT/PTQ pipelines) into high-accuracy TensorRT INT8 / mixed-precision engines, and avoid common pitfalls where TensorRT silently ignores or mishandles Q/DQ information.
- **Status**: Draft but intended as a reusable checklist for future projects.
- **Scope**: ONNX models that already contain `QuantizeLinear` / `DequantizeLinear` (Q/DQ) nodes, converted to TensorRT via `trtexec`, Polygraphy, or the TensorRT Python/C++ API; applies to classification, detection, and transformer-style networks.
- **Related topics**: ONNX QDQ quantization, QAT and PTQ toolkits, TensorRT INT8/QAT workflows, Polygraphy accuracy comparison.
- **External references**:
  - TensorRT “Working With Quantized Types” and QAT docs: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
  - TensorRT QAT/ONNX examples (PyTorch/TensorFlow quantization): https://github.com/NVIDIA/TensorRT/tree/main/tools
  - Polygraphy user guide (`polygraphy run`, INT8 flags, precision constraints): https://docs.nvidia.com/deeplearning/tensorrt/latest/polygraphy/polygraphy.html

## 1. Understand the two main INT8 paths in TensorRT

- **Calibrated INT8 (no Q/DQ in ONNX)**: The ONNX model is FP32/FP16, and TensorRT discovers INT8 ranges via a calibrator; you build with `--int8` (and optionally `--fp16`) plus a calibration cache or data loader; Q/DQ nodes do not appear in the ONNX graph.
- **Pre-quantized QDQ ONNX (QAT/PTQ)**: The ONNX model already contains `QuantizeLinear` / `DequantizeLinear` ops that encode scale/zero-point; TensorRT should treat them as quantization contracts and fuse them into INT8 layers where possible; you typically **do not** provide a calibrator.
- When working with pre-quantized QDQ ONNX, the goal is to let TensorRT respect the Q/DQ-defined precision scheme (weights and activations) rather than re-calibrating or heuristically choosing its own INT8 placement.

## 2. Exporting a QDQ ONNX model (brief recap)

The exact export path depends on your quantization toolkit, but the general pattern is:

- **PyTorch QAT / quantization**: Use toolkit-specific ONNX export wrappers to convert fake-quant modules (e.g., `QuantConv2d`, `FakeQuantize`) into ONNX `QuantizeLinear` / `DequantizeLinear` nodes before and after each quantized tensor.
- **TensorFlow QAT / `tensorflow-quantization`**: Quantize the model and then export to ONNX, ensuring that Q/DQ nodes appear in the exported graph:

```python
# Pseudocode only; adapt to your framework/toolkit
from tensorflow_quantization.quantize import quantize_model

model_fp32 = build_model()
q_model = quantize_model(model_fp32)

q_model.save("<OUTPUT_QAT_SAVEDMODEL_DIR>")
convert_saved_model_to_onnx(
    saved_model_dir="<OUTPUT_QAT_SAVEDMODEL_DIR>",
    onnx_model_path="<OUTPUT_QDQ_ONNX_PATH>"
)
```

- **Key requirement**: The resulting ONNX must contain `QuantizeLinear` / `DequantizeLinear` around tensors that you expect to run in INT8; TensorRT uses that structure to infer where INT8 should be applied.

## 3. First sanity check: ONNX Runtime evaluation (QDQ vs FP32)

Before involving TensorRT, verify that the QDQ ONNX model is numerically reasonable:

- Run both the FP32 ONNX and QDQ ONNX models with ONNX Runtime on a representative subset of your validation set using a consistent preprocessing/postprocessing pipeline.
- Compute accuracy metrics (e.g., top-1 / top-5 for classification, mAP for detection) and ensure the QDQ model is within an acceptable delta (for many QAT flows, within ~1–3% absolute is expected if calibration and quantization are configured correctly).

Example pattern (pseudocode):

```bash
python <EVAL_ONNX_SCRIPT> \
  --onnx_path <FP32_ONNX_PATH> \
  --data-root <DATA_ROOT> \
  --max-images 500 \
  --imgsz 640 \
  --out <OUT_FP32_JSON>

python <EVAL_ONNX_SCRIPT> \
  --onnx_path <QDQ_ONNX_PATH> \
  --data-root <DATA_ROOT> \
  --max-images 500 \
  --imgsz 640 \
  --out <OUT_QDQ_JSON>
```

- If QDQ ONNX already collapses (huge accuracy drop), fix quantization first (calibration set, op exclusions, quantization scheme) before touching TensorRT.

## 4. Building TensorRT engines from QDQ ONNX: critical flags

When converting a QDQ ONNX model to TensorRT, **just enabling `--int8` is not enough**. TensorRT needs to know that it should obey the compute precisions implied by Q/DQ nodes.

### 4.1 `trtexec` command-line pattern

Recommended pattern for a pre-quantized QDQ ONNX model:

```bash
trtexec \
  --onnx=<QDQ_ONNX_PATH> \
  --saveEngine=<ENGINE_PATH> \
  --int8 \
  --fp16 \  # optional but common for mixed precision
  --precisionConstraints=obey \
  --exportLayerInfo=<LAYER_INFO_JSON> \
  --profilingVerbosity=detailed
```

Key points:

- `--int8`: enables INT8 mode in the builder.
- `--fp16`: allows FP16 for non-INT8 parts of the graph (mixed precision).
- `--precisionConstraints=obey`:
  - Tells TensorRT to **follow the precisions implied by Q/DQ** (and any layer precision hints) rather than treating them as optional.
  - Without this, TensorRT may choose different precisions for some layers and effectively ignore parts of the QDQ scheme, which can cause significant numerical drift and accuracy loss.
- `--exportLayerInfo` + `--profilingVerbosity=detailed`:
  - Produces a JSON (and textual dump) describing each layer’s type, input/output datatypes, and whether Q/DQ was fused.
  - This is essential for verifying that the engine is actually using INT8 where you expect.

### 4.2 TensorRT Python API equivalent

If you build engines via the TensorRT Python API, the equivalent configuration looks like:

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, logger)

with open("<QDQ_ONNX_PATH>", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse QDQ ONNX")

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

profile = builder.create_optimization_profile()
profile.set_shape("input_tensor_name", min=(1, 3, 640, 640), opt=(1, 3, 640, 640), max=(1, 3, 640, 640))
config.add_optimization_profile(profile)

engine_bytes = builder.build_serialized_network(network, config)
with open("<ENGINE_PATH>", "wb") as f:
    f.write(engine_bytes)
```

- Do **not** attach an INT8 calibrator when using a properly pre-quantized QDQ ONNX model; the Q/DQ scales should already encode the quantization ranges.

## 5. Inspecting the generated engine for QDQ adherence

After building the engine, verify that TensorRT honored Q/DQ placement:

- **Layer info JSON** (`--exportLayerInfo`):
  - Inspect layers corresponding to Q/DQ blocks; they should either be:
    - Fused into adjacent Conv/MatMul layers running in INT8, or
    - Implemented as lightweight Reformat/Scale operations with INT8 inputs/outputs and `Origin: "QDQ"` in the metadata.
- **Check sensitive layers**:
  - For detection models, look at heads, logits, and critical projection layers; confirm they are INT8 or higher precision exactly as the QDQ scheme intended.

Example snippet to locate QDQ-related layers in `layer_info.json`:

```python
import json

with open("<LAYER_INFO_JSON>", "r") as f:
    info = json.load(f)

qdq_layers = [
    layer for layer in info["Layers"]
    if "QDQ" in layer.get("Origin", "") or "QuantizeLinear" in layer.get("Metadata", "")
]

for layer in qdq_layers[:10]:
    print(layer["Name"], layer["LayerType"], layer["Outputs"][0]["Format/Datatype"])
```

- If you see many standalone INT8 Reformat layers that are not fused or unexpected FP32 fallbacks in critical paths, revisit your QDQ scheme and builder configuration.

## 6. End-to-end accuracy check: ONNX QDQ vs TensorRT engine

Once the engine is built, check accuracy relative to ONNX QDQ:

- Use the same preprocessing and postprocessing as your ONNX evaluation, but swap ONNX Runtime inference for TensorRT.
- For small-scale, numeric comparison between ONNX Runtime and TensorRT, Polygraphy is convenient:

```bash
polygraphy run <QDQ_ONNX_PATH> \
  --onnxrt \
  --trt \
  --int8 \
  --fp16 \
  --precision-constraints=obey \
  --compare simple \
  --save-inputs <POLY_INPUTS_JSON> \
  --save-outputs <POLY_OUTPUTS_JSON>
```

- Expect some element-wise differences due to quantization and kernel-level differences, but:
  - Cosine similarity between full outputs should be close to 1.0 on representative inputs.
  - Downstream task metrics (accuracy, mAP, BLEU, etc.) should be close to those of the QDQ ONNX model.

## 7. Common pitfalls and failure modes

- **Pitfall 1: Building with `--int8` but no precision constraints**:
  - Symptom: The ONNX QDQ model evaluates well with ONNX Runtime, but the TensorRT INT8 or INT8+FP16 engine has drastically worse accuracy (e.g., mAP ~ 0).
  - Cause: TensorRT may treat Q/DQ nodes as hints rather than strict contracts, pick unexpected precisions, or require explicit `OBEY_PRECISION_CONSTRAINTS` flags to honor them.
  - Fix: Rebuild engines with `--precisionConstraints=obey` (or the Python API equivalent) and verify with layer info + accuracy comparison.

- **Pitfall 2: Mixing QDQ and calibrators**:
  - Symptom: Pre-quantized QDQ model plus a calibrator yields unstable or inconsistent accuracy.
  - Cause: Two overlapping sources of quantization information (Q/DQ vs calibration) can conflict.
  - Fix: For **fully pre-quantized QDQ** models, avoid supplying a calibrator; rely on Q/DQ scales only; use calibrators only when the model has no Q/DQ and you want TensorRT to discover INT8 ranges.

- **Pitfall 3: Pre/post-processing mismatches across stages**:
  - Symptom: QDQ ONNX and TensorRT INT8 both have similar raw outputs, but task-level metrics diverge from the FP32 baseline.
  - Cause: Different resizing, normalization, color ordering, or NMS logic between FP32 reference and inference code.
  - Fix: Centralize preprocessing and decoding; drive both ONNX and TensorRT engines from the same input tensors and postprocessing code.

- **Pitfall 4: Unsupported or partially supported ONNX ops**:
  - Symptom: TensorRT logs warnings about unsupported ops; accuracy drops even when QDQ looks correct.
  - Cause: TensorRT may fall back to FP32 plugins or different semantics; QDQ placement around unsupported ops may not be honored.
  - Fix: Inspect logs and layer info; if necessary, adjust the ONNX graph (e.g., fuse ops, replace unsupported patterns, or adjust QDQ placement).

## 8. Recommended debug workflow when accuracy drops

When a QDQ ONNX model performs well in ONNX Runtime but poorly in TensorRT:

1. **Confirm ONNX QDQ is healthy**:
   - Evaluate QDQ ONNX with ONNX Runtime on a representative validation slice.
   - If accuracy is already degraded, fix quantization first (calibration, op exclusions, QDQ scheme).
2. **Rebuild TensorRT engine with explicit precision constraints**:
   - Use `--int8 --fp16 --precisionConstraints=obey` and export layer info.
   - Ensure no calibrator is used for fully pre-quantized QDQ ONNX models.
3. **Check engine structure**:
   - Use `--exportLayerInfo` and `--profilingVerbosity=detailed` to confirm Q/DQ fusion and per-layer datatypes.
4. **Compare outputs ONNX QDQ vs TensorRT**:
   - Use Polygraphy or a custom script to run both backends on the same inputs and compare outputs numerically (MSE, cosine similarity).
5. **Iterate on QDQ scheme if needed**:
   - If certain layers show large discrepancies, consider adjusting QDQ placement, excluding especially sensitive layers from INT8, or revisiting the QAT/PTQ configuration.

Following this checklist significantly reduces the risk of silent accuracy collapse when deploying pre-quantized QDQ ONNX models to TensorRT and helps ensure that the engine faithfully implements the intended quantization scheme.

