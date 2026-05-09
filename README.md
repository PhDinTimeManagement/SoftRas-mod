# Soft Rasterizer (SoftRas) — Modern CUDA/PyTorch Compatibility Fork

![SoftRas Fork: Modern CUDA Compatibility](https://img.shields.io/badge/SoftRas%20Fork-Modern%20CUDA%20Compatibility-brown.svg)

This repository is a compatibility-focused fork of the original **Soft Rasterizer (SoftRas)** PyTorch project. It preserves the original SoftRas algorithm, examples, license, and academic attribution while updating the codebase for modern CUDA/PyTorch extension builds and newer NVIDIA GPUs, including RTX 5090-class hardware when the local CUDA/PyTorch toolchain supports Blackwell / compute capability 12.0.

> This fork does **not** claim original ownership of SoftRas and does **not** introduce a new differentiable rendering method. The core renderer, examples, and research contribution belong to the original SoftRas authors.

## Original Project Credit

SoftRas was introduced in:

> **Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning**  
> Shichen Liu, Tianye Li, Weikai Chen, Hao Li  
> ICCV 2019 Oral

Original project: [ShichenLiu/SoftRas](https://github.com/ShichenLiu/SoftRas)  
Paper: [arXiv:1904.01786](https://arxiv.org/abs/1904.01786)

The original code was extended from [hiroharu-kato/neural_renderer](https://github.com/hiroharu-kato/neural_renderer?tab=readme-ov-file) and [daniilidis-group/neural_renderer](https://github.com/daniilidis-group/neural_renderer). <br>
The included `LICENSE` remains the MIT License and includes the original copyright notices.

## Fork Maintainer
[Chloe Xin DAI](https://github.com/PhDinTimeManagement) <br>

## Motivation

The upstream SoftRas README targets **PyTorch 1.6.0** and **CUDA 10.1**, which are older than modern Blackwell-generation GPUs. [NVIDIA lists](https://developer.nvidia.com/cuda/gpus) the GeForce RTX 5090 under **CUDA compute capability 12.0**, so an older CUDA/PyTorch stack cannot reliably build or run device code for this GPU.

In addition to the GPU architecture issue, the original source used several older PyTorch/CUDA idioms that are brittle in current extension builds, including hard-coded `.cuda()` calls, deprecated autograd patterns, implicit CUDA stream usage, limited extension input validation, and ambiguous `torch.cross` calls. This fork updates those areas while keeping the original CUDA kernel logic largely intact.

## Key Modifications

The exact changed files are recorded in `MODIFIED_FILES.txt`, and the most important changes are summarized below.

| Area | Main Files                                                                                          | Summary                                                                                                                                                                                                   | Purpose                                                                                                                |
| --- |-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| Build/package configuration | `setup.py`                                                                                          | Kept `torch.utils.cpp_extension.CUDAExtension`, switched package discovery to `find_packages(...)`, removed the unused `torchvision` install requirement, and set `zip_safe=False`.                       | Keeps the extension build path simple while avoiding an unnecessary dependency and improving package discovery.        |
| C++ extension wrappers | `soft_renderer/cuda/*_cuda.cpp`                                                                     | Replaced broad `torch/torch.h` includes with `torch/extension.h`, added explicit CUDA/contiguity/dtype/device checks, and added `c10::cuda::CUDAGuard`.                                                   | Fails early on invalid tensors and ensures kernels launch on the CUDA device associated with the input tensors.        |
| CUDA kernel launch path | `soft_renderer/cuda/*_kernel.cu`                                                                    | Added current-stream launches via `at::cuda::getCurrentCUDAStream()` and replaced manual `cudaGetLastError()` printing with `C10_CUDA_KERNEL_LAUNCH_CHECK()`.                                             | Makes custom kernels follow PyTorch's active CUDA stream and report launch errors as proper CUDA/PyTorch errors.       |
| Tensor dtype/device handling | `mesh.py`, `load_obj.py`, `save_obj.py`, `voxelization.py`, `soft_rasterize.py`, lighting utilities | Replaced hard-coded CUDA allocation patterns with device-aware `.to(device)` flows, dtype-aware tensor creation, explicit `int32` voxel buffers, and `.contiguous()` calls before extension entry points. | Prevents CPU/GPU mismatches, dtype mismatches, and non-contiguous tensor issues that can break native extension calls. |
| Deprecated PyTorch API usage | `examples/recon/*.py`, `examples/demo_deform.py`, `save_obj.py`                                     | Replaced `.data` with `.detach()` or `.item()`, removed `torch.autograd.Variable`, and added `map_location=device` when loading checkpoints.                                                              | Aligns example/runtime code with modern PyTorch autograd and checkpoint-loading behavior.                              |
| Math/API cleanup | `look.py`, `look_at.py`, `vertex_normals.py`, `mesh.py`, `transform.py`                             | Added explicit `dim=` arguments for `torch.cross` / `F.normalize`, fixed tensor conversion to preserve device and dtype, fixed `LookAt.eyes` / `Look.eyes`, and handled `look(..., up=None)`.             | Avoids modern PyTorch warnings/errors and fixes device/dtype inconsistencies in transformation code.                   |

## Compatibility Notes

- The renderer and voxelizer still rely on custom CUDA extensions. There is **no CPU implementation** of the main SoftRas rasterization kernels.
- For RTX 5090 / Blackwell GPUs, use a PyTorch CUDA build and CUDA toolkit/driver combination that supports compute capability **12.0**.
- This repository does not hard-code a complete version matrix. The exact working combination depends on your Python, PyTorch, CUDA toolkit, NVIDIA driver, compiler, and GPU architecture.
- The CUDA kernels are primarily compatibility-modernized, not algorithmically redesigned or performance-retuned.
- Rebuild the extension after changing PyTorch, Python, CUDA, compiler, or GPU architecture.

## Installation

`setup.py` imports PyTorch at build time, so install a CUDA-enabled PyTorch build before installing this package.

Suggested setup:

```bash
# 1. Create and activate an environment.

# 2. Install build helpers.
python -m pip install --upgrade pip setuptools wheel ninja

# 3. Install a CUDA-enabled PyTorch build that supports your CUDA/driver/GPU stack.
#    Use the official PyTorch selector for the command appropriate to your system:
#    https://pytorch.org/get-started/locally/

# 4. For RTX 5090 builds, make the target architecture explicit.
#    This is especially useful when building without the target GPU visible.
export TORCH_CUDA_ARCH_LIST="12.0"

# Optional: set CUDA_HOME if your CUDA toolkit is not auto-detected.
# export CUDA_HOME=/usr/local/cuda

# 5. Build and install SoftRas in editable mode.
python -m pip install -e .
```

Dependencies declared by `setup.py` are:

```text
numpy
torch
scikit-image
tqdm
imageio
```

Some example scripts also import packages that are not declared in `setup.py`, such as `matplotlib`; install those separately if your selected example requires them.

## Build Verification

After installation, verify that PyTorch sees your CUDA device and that the SoftRas package imports successfully:

```bash
python - <<'PY'
import torch
import soft_renderer as sr

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))

print("SoftRas:", sr.__version__)
PY
```

If you change CUDA/PyTorch versions or see stale binary errors, clean and rebuild:

```bash
rm -rf build *.egg-info soft_renderer/cuda/*.so
python -m pip install -e .
```
