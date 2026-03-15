# SoftRas modernization summary

This archive contains a source-level modernization of the original SoftRas codebase for current PyTorch/CUDA extension APIs.

## What was changed

### Extension / build-system updates
- Kept `torch.utils.cpp_extension.CUDAExtension`-based build wiring and simplified `setup.py`.
- Removed the unused `torchvision` install requirement.
- Preserved editable-install support with `pip install -e .`.

### C++ / CUDA API migration
- Replaced legacy-style tensor handling with modern `torch::Tensor` / `at::Tensor` conventions.
- Standardized tensor validation with `TORCH_CHECK`.
- Standardized dtype dispatch through `tensor.scalar_type()` and `AT_DISPATCH_FLOATING_TYPES(...)`.
- Standardized raw pointer access through `tensor.data_ptr<T>()`.
- Added CUDA device guards via `c10::cuda::CUDAGuard`.
- Switched kernel launches to PyTorch's current CUDA stream via `at::cuda::getCurrentCUDAStream()`.
- Replaced manual launch error handling with `C10_CUDA_KERNEL_LAUNCH_CHECK()`.

### Python runtime fixes
- Removed deprecated `.data` usage from the reconstruction examples.
- Removed deprecated `torch.autograd.Variable` usage.
- Replaced hardcoded `.cuda()` calls with device-aware `.to(device)` flows where appropriate.
- Fixed CPU/CUDA device checks such as `tensor.device.type == 'cpu'`.
- Fixed cross-product calls to pass an explicit `dim=` argument.
- Fixed `LookAt.eyes` / `Look.eyes` to return the correct internal eye state.
- Fixed `look.py` handling when `up=None`.
- Fixed OBJ export edge cases in `save_obj.py` / `save_voxel()`.

## Validation performed here
- Read and reviewed the full source tree.
- Searched the full project for deprecated patterns such as `.type()`, `.data`, `Variable`, hardcoded `torch.cuda.FloatTensor`, and hardcoded `.cuda()`.
- Re-ran Python syntax validation (`python -m py_compile`) across all package and example Python files.

## Important note
This environment did not provide a working CUDA/NVCC toolchain matching your target stack, so the extension build could not be executed end-to-end here. The changes were therefore validated statically at the source/API level rather than by a real `pip install -e .` compile on CUDA 12.8 + PyTorch 2.7.
