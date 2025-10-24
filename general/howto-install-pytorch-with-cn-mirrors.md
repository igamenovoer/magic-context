# How to Install PyTorch via China Mirrors (pip, pixi, uv/pyproject)

## HEADER
- Purpose: Fast, reliable PyTorch installs from mainland China using verified mirrors
- Status: Active
- Date: 2025-10-24
- Dependencies: pip or uv; optional pixi; matching CUDA runtime
- Target: Developers and AI assistants

## Summary
This hint shows practical ways to install PyTorch from China using mirrors for GPU (CUDA) and CPU builds. It includes pip one‑liners, persistent pip config, uv/pyproject configuration, and pixi integration. The repo currently contains a working pixi example pointing to NJU’s mirror in `pyproject.toml`.

Validated mirror options (tested today)
- Official: `https://download.pytorch.org/whl/cu126` (GPU), `https://download.pytorch.org/whl/cpu` (CPU)
- Aliyun: `https://mirrors.aliyun.com/pytorch-wheels/cu126/` (GPU), `https://mirrors.aliyun.com/pytorch-wheels/cpu/` (CPU)
- NJU: `https://mirrors.nju.edu.cn/pytorch/whl/cu126/` (GPU), `https://mirrors.nju.edu.cn/pytorch/whl/cpu/` (CPU)
- SJTU: `https://mirror.sjtu.edu.cn/pytorch-wheels/cu126/` (redirects to official as of today)
- PyPI mirrors for general deps: `https://pypi.tuna.tsinghua.edu.cn/simple/`, `https://mirrors.aliyun.com/pypi/simple/`

Pick the CUDA track that matches your system/driver (e.g., `cu126`, `cu124`, or use CPU wheels). See: https://pytorch.org/get-started/locally/

## pip: quick commands
- GPU CUDA 12.6 (Aliyun mirror):
  - `pip install torch torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu126/ -i https://pypi.tuna.tsinghua.edu.cn/simple/`
- GPU CUDA 12.6 (NJU mirror):
  - `pip install torch torchvision torchaudio -f https://mirrors.nju.edu.cn/pytorch/whl/cu126/ -i https://pypi.tuna.tsinghua.edu.cn/simple/`
- CPU wheels (Aliyun mirror):
  - `pip install torch torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cpu/ -i https://pypi.tuna.tsinghua.edu.cn/simple/`

Notes
- `-f/--find-links` points pip at a flat index of wheels (this is the key for the mirrored wheels).
- Keep `-i` to a fast PyPI mirror for non-PyTorch deps.
- On Windows, consider PowerShell’s execution policy or corporate proxies if downloads hang.

## pip: persistent config (optional)
- Set PyPI mirror globally:
  - `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/`
- Add a CUDA 12.6 find-links mirror:
  - `pip config set global.find-links https://mirrors.aliyun.com/pytorch-wheels/cu126/`
- Example Windows file locations:
  - User: `%APPDATA%\pip\pip.ini`
  - Global: `%ProgramData%\pip\pip.ini`

Remove or adjust `find-links` if you switch CUDA tracks (e.g., `cu124` → `cu126`).

## uv/pyproject: pin to a mirror index
uv supports named indexes and can pin specific packages to a flat index in `pyproject.toml`.

Example (Aliyun CUDA 12.6 as a flat index)
```toml
# pyproject.toml
[project]
dependencies = ["torch", "torchvision", "torchaudio"]

# Define a named index (flat HTML listing of wheels)
[[tool.uv.index]]
name = "pytorch-cu126-aliyun"
url = "https://mirrors.aliyun.com/pytorch-wheels/cu126/"
format = "flat"
explicit = true

# Pin PyTorch packages to the mirror index
[tool.uv.sources]
torch = { index = "pytorch-cu126-aliyun" }
torchvision = { index = "pytorch-cu126-aliyun" }
torchaudio = { index = "pytorch-cu126-aliyun" }
```
- Install: `uv sync` or `uv pip install -r pyproject.toml`
- Switch CUDA track by changing the index `url` accordingly (e.g., `.../cu124/` or `.../cpu/`).

Docs
- uv indexes: https://docs.astral.sh/uv/concepts/indexes/

## pixi: use PyPI dependencies with mirrored index
This repo already configures NJU for PyTorch under `[tool.pixi.pypi-dependencies]` in `pyproject.toml`:

```toml
[tool.pixi.pypi-dependencies]
# Example existing config (works today)
torch = { version = "*", index = "https://mirrors.nju.edu.cn/pytorch/whl/cu126" }
torchvision = { version = "*", index = "https://mirrors.nju.edu.cn/pytorch/whl/cu126" }
```

Alternative (Aliyun mirror)
```toml
[tool.pixi.pypi-dependencies]
torch = { version = "2.9.*", index = "https://mirrors.aliyun.com/pytorch-wheels/cu126/" }
torchvision = { version = "0.24.*", index = "https://mirrors.aliyun.com/pytorch-wheels/cu126/" }
torchaudio = { version = "2.9.*", index = "https://mirrors.aliyun.com/pytorch-wheels/cu126/" }
```

Tips
- Keep your conda channels fast as well (e.g., TUNA mirrors) via pixi config mirrors if needed:
  - Pixi config docs: https://prefix-dev.github.io/pixi/dev/reference/pixi_configuration/
- Create/refresh the env: `pixi run` or `pixi install`.

## Troubleshooting: pixi install fails with Aliyun/NJU
- Symptom: `pixi install` fails to resolve/install `torch` even though the wheel exists on the mirror.
- Cause: The per‑package `index = ".../pytorch-wheels/cu126/"` points to a flat wheel listing, not a PEP 503 "simple" index. uv (used by pixi) won’t treat that as a normal index and resolution can fail.

Recommended fix (use find-links with a PyPI mirror)
```toml
# pyproject.toml
[tool.pixi.pypi-dependencies]
torch = "*"
torchvision = "*"
torchaudio = "*"

[tool.pixi.pypi-options]
# Fast PyPI mirror for non-PyTorch deps
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
# Flat wheel page for PyTorch (like pip --find-links)
find-links = [{ url = "https://mirrors.aliyun.com/pytorch-wheels/cu126/" }]
```
- Then run: `pixi install -v`
- This mirrors the pip pattern: `-f <ALIYUN CUDA INDEX>` + `-i <TUNA SIMPLE>`.

Alternative (explicit uv flat index)
```toml
[[tool.uv.index]]
name = "pytorch-cu126-aliyun"
url = "https://mirrors.aliyun.com/pytorch-wheels/cu126/"
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126-aliyun" }
torchvision = { index = "pytorch-cu126-aliyun" }
torchaudio = { index = "pytorch-cu126-aliyun" }
```

Check your Python/platform wheels
- Ensure the mirror has wheels for your exact Python and platform, e.g. `cp313` for Python 3.13 on `win_amd64`.
- If no matching wheel exists for your combo, either:
  - Pin `python` to a version with available wheels (e.g., `3.12.*`), or
  - Pin `torch`/`torchvision`/`torchaudio` to a version that ships the required wheels.

## Verify installs
- Python check:
  - `python -c "import torch; print(torch.__version__); print(torch.version.cuda)"`
- GPU availability:
  - `python -c "import torch; print(torch.cuda.is_available())"`

If CUDA is not available in `torch`, ensure your local NVIDIA drivers match the chosen CUDA track.

## Other viable options
- Official index with `--index-url`: `pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio`
- Additional mirrors sometimes used: SJTU (`https://mirror.sjtu.edu.cn/pytorch-wheels/`), but may redirect to official.
- For general dependencies, PyPI mirrors like TUNA or Aliyun are usually the fastest in CN.

## Use a locally downloaded wheel (no network)
If you already downloaded `torch-...whl` (and optionally `torchvision`/`torchaudio`) to local storage, you can install from disk.

Option A — pip from local file/folder
- Direct file:
  - `pip install C:\path\to\torch-2.9.0+cu126-cp313-cp313-win_amd64.whl`
- Folder of wheels (acts like a flat index):
  - `pip install torch torchvision torchaudio --no-index -f file:///C:/wheelhouse/`
- Note: Make sure wheels match your Python tag (e.g., `cp313`) and platform (`win_amd64`).

Option B — pixi with local cache (recommended)
- Use find-links to your wheel folder and keep a PyPI mirror as fallback for non-cached deps:
```toml
[tool.pixi.pypi-dependencies]
torch = "*"
torchvision = "*"
torchaudio = "*"

[tool.pixi.pypi-options]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple/"  # fallback for other deps
find-links = [
  { path = "./wheelhouse" },             # relative folder in repo
  # or absolute path / file URL
  # { path = "C:/wheelhouse" },
  # { url  = "file:///C:/wheelhouse" },
]
```
- Run: `pixi install -v`
- Tip: Put your local folder first in `find-links` so uv checks it before network mirrors.

Option C — uv flat index + pin
- Define a flat index for your wheel folder and pin torch-family packages to it:
```toml
[[tool.uv.index]]
name = "local-wheelhouse"
url = "file:///C:/wheelhouse"   # Linux: file:///home/user/wheelhouse
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "local-wheelhouse" }
torchvision = { index = "local-wheelhouse" }
torchaudio = { index = "local-wheelhouse" }
```

Common pitfalls
- Wheel tag mismatch: Ensure the wheel’s `cpXY` and platform match your interpreter and OS.
- Version spec: If uv doesn’t pick the local wheel, try pinning exactly, e.g., `torch = "==2.9.0+cu126"`.
- Missing companion wheels: If you only cached `torch`, keep PyPI mirror enabled so `torchvision`/`torchaudio` can resolve.

## References
- PyTorch Get Started: https://pytorch.org/get-started/locally/
- uv package indexes: https://docs.astral.sh/uv/concepts/indexes/
- Pixi configuration (mirrors, PyPI config): https://prefix-dev.github.io/pixi/dev/reference/pixi_configuration/
- Aliyun PyTorch wheels (verified): https://mirrors.aliyun.com/pytorch-wheels/cu126/
- NJU PyTorch wheels (verified): https://mirrors.nju.edu.cn/pytorch/whl/cu126/
- Official CUDA wheels: https://download.pytorch.org/whl/
- PyPI mirrors: https://pypi.tuna.tsinghua.edu.cn/simple/, https://mirrors.aliyun.com/pypi/simple/
