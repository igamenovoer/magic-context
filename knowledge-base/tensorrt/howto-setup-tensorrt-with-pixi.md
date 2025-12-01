# How to Set Up TensorRT with Pixi

This repository cannot rely on Pixi’s normal dependency resolution to install NVIDIA TensorRT correctly (due to the way the `tensorrt` PyPI metapackage bootstraps sub-packages like bindings and libs). Instead, we:

- Keep TensorRT **out of** `[tool.pixi.pypi-dependencies]`.
- Make sure `pip` is available **inside** the Pixi environment.
- Add a Pixi **task** that runs `pip` *from within* the Pixi env to install TensorRT (and its CUDA / bindings subpackages) explicitly.

This pattern ensures:

- Pixi still fully manages the Python environment and most dependencies.
- TensorRT is installed using NVIDIA’s recommended `pip install tensorrt ...` flow, which can pull from `pypi.nvidia.com` and handle the internal `tensorrt-*` packages.

## 1. Pixi PyPI dependencies

In `pyproject.toml`, do **not** list `tensorrt` as a Pixi-managed dependency. Instead, keep only the normal project and tool dependencies, and ensure `pip` is present:

```toml
[tool.pixi.pypi-dependencies]
auto_quantize_model = { path = ".", editable = true }
ultralytics = "*"
torch = "*"
torchvision = "*"
onnxruntime-gpu = "*"
pip = "*"
supervision = "*"
scipy = "*"
opencv-python = "*"
imageio = "*"
attrs = "*"
omegaconf = "*"
hydra-core = "*"
mdutils = "*"
mkdocs-material = "*"
ruff = "*"
mypy = "*"
pytest = "*"
click = "*"
rich = "*"
pandas = "*"
huggingface-hub = "*"
py-cpuinfo = "*"
```

Notes:

- We **intentionally omit** any `tensorrt`, `tensorrt-bindings`, or `tensorrt-cu12*` entries here.
- `pip = "*"` ensures `python -m pip` is available inside the Pixi env.

## 2. Pixi task to install TensorRT via pip

Add a dedicated Pixi task that installs TensorRT from within the Pixi environment using NVIDIA’s PyPI index:

```toml
[tool.pixi.tasks]
setup-tensorrt = "env NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=0 python -m pip install tensorrt==10.7.0 --extra-index-url https://pypi.nvidia.com"
```

Key details:

- `NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=0` allows the `tensorrt` metapackage to perform its internal `pip` operations if needed.
- `tensorrt==10.7.0` should be chosen to match the CUDA / driver stack (e.g., CUDA 12.6 on this project).
- `--extra-index-url https://pypi.nvidia.com` lets pip pull the CUDA-specific wheels (`tensorrt-cu12*`) from NVIDIA’s index.

## 3. Usage flow

To prepare the environment from scratch:

```bash
pixi install
pixi run setup-tensorrt
```

Then, inside the Pixi env, TensorRT can be used normally:

```bash
pixi run python -c "import tensorrt as trt; print(trt.__version__)"
```

## 4. Why we avoid Pixi-managed `tensorrt`

Attempting to put `tensorrt` directly under `[tool.pixi.pypi-dependencies]` caused issues such as:

- `ModuleNotFoundError: No module named 'tensorrt_bindings'` when importing `tensorrt` inside the Pixi env.
- Inability to satisfy `tensorrt-bindings` for Python 3.12 with the available wheels.

Root cause:

- The `tensorrt` metapackage relies on a custom `setup.py` / nested-`pip` flow to install its own `tensorrt-*` submodules, which does not play well with Pixi’s resolver and PEP 668–style “externally managed” environments.

By:

- Letting Pixi manage **Python + tooling**, and
- Letting `pip` (inside Pixi) manage **TensorRT** via the official `pip install tensorrt ...` command,

we get a working TensorRT installation without fighting Pixi’s solver.

