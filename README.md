# Compile your own GPU Kernel Playbook

A practical reference for writing, compiling, and running custom GPU kernels on AMD hardware using HIP and PyTorch.

---

## What is a GPU Kernel?

A GPU kernel is a function that runs **in parallel across thousands of GPU threads simultaneously**. Unlike a CPU function that executes once per call, a kernel is launched with a **grid** of **blocks**, each containing many **threads**, all executing the same code on different data.

```
Grid
└── Block [0]  Block [1]  Block [2]  ...
     └── Thread[0..255]  (each processes one data element)
```

### `__global__` — the kernel qualifier

In HIP/CUDA, `__global__` marks a function as a GPU kernel:

- It **runs on the GPU** (device)
- It is **launched from the CPU** (host)
- It **executes in parallel** across many GPU threads simultaneously

```c
__global__ void add_one(float* data, int n) { ... }
```

### Thread Indexing Model

When launching a kernel you specify two dimensions:

| Variable | Meaning |
|---|---|
| `gridDim` | Number of blocks in the grid |
| `blockDim` | Number of threads per block |

Each thread has access to three built-in read-only variables:

| Variable | Meaning |
|---|---|
| `blockIdx.x` | Which block this thread belongs to |
| `blockDim.x` | Number of threads in one block |
| `threadIdx.x` | Thread index within its block |

### Global Thread ID

These variables are combined to compute a globally unique thread index:

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

Total threads = `gridDim.x * blockDim.x`. Each thread processes one element independently — this is **data parallelism**. The same operation runs on many elements at once with no inter-thread dependency.

---

## AMD GPU Programming: HIP

AMD GPUs use **HIP** (Heterogeneous-Compute Interface for Portability), part of the **ROCm** (Radeon Open Compute) platform.

HIP is designed to be syntactically close to CUDA. Most CUDA code can be translated to HIP mechanically using the `hipify` tool (which is what generated the `.hip` files in this repo).

| CUDA | HIP equivalent |
|---|---|
| `cudaMalloc` | `hipMalloc` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `kernel<<<grid, block>>>` | `hipLaunchKernelGGL(kernel, ...)` |
| `nvcc` | `hipcc` |

**ROCm** is the full AMD open-source GPU compute stack: drivers, compilers, libraries, and runtime. HIP sits on top of ROCm.

---

## PyTorch + AMD/HIP

PyTorch ships a ROCm build where the CUDA API surface (`torch.cuda.*`) is transparently backed by HIP. This means:

- `torch.cuda.is_available()` works on AMD GPUs with ROCm
- `tensor.to("cuda")` allocates on the AMD GPU
- `torch.version.hip` exposes the HIP version

PyTorch also exposes `torch.cuda._compile_kernel()` — a high-level shortcut to JIT-compile a raw kernel string and get back a callable, without needing a separate build step.

---

## Setup

### 1. Uninstall Old Stack

```bash
pip uninstall torch torchvision torchaudio

# Remove old ROCm
sudo apt purge 'rocm*' 'hip*' 'hsa*' -y
sudo rm -rf /opt/rocm*
sudo rm -rf /etc/apt/sources.list.d/rocm*
sudo apt autoremove -y
sudo apt update
```

Verify it's gone: `rocminfo` shouldn't return anything.

### 2. Install ROCm 7.1.1

Check your Ubuntu version first:
- **24.04** → `noble`
- **22.04** → `jammy`

```bash
# Download the installer (noble shown; swap for jammy if needed)
wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb

# Install the installer package
sudo DEBIAN_FRONTEND=noninteractive apt install -y ./amdgpu-install_7.1.1.70101-1_all.deb

# Install ROCm + HIP
sudo amdgpu-install --usecase=rocm,hip -y
```

Verify:
```bash
hipcc --version
rocminfo
```

### 3. Install PyTorch for ROCm 7.1

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm7.1
```

Verify:
```bash
python3 -c "import torch; print(torch.version.hip)"
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Examples

### Example 1: Simple Vector Addition

#### Flow 1 — High-Level: `gpu_kernel.py`

The fast path. Kernel is written as a raw C++ string inside Python and compiled at runtime via PyTorch's built-in JIT.

**Files:** [gpu_kernel.py](gpu_kernel.py)

**How it works:**

```python
# 1. Kernel source as a string — standard __global__ CUDA/HIP syntax
KERNEL_SOURCE = """
extern "C"
__global__ void add_one(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000; i++)
            data[idx] += 1.0f;
    }
}
"""

# 2. Compile the kernel string — PyTorch calls hipcc under the hood on ROCm
add_one_kernel = torch.cuda._compile_kernel(KERNEL_SOURCE, "add_one")

# 3. Launch: specify grid/block dimensions and pass tensor args directly
add_one_kernel(
    grid=(grid_size, 1, 1),
    block=(block_size, 1, 1),
    args=[x, n],
)
```

The script also spawns a background thread that polls `rocm-smi` every 100ms to log peak and average GPU utilization during the kernel run.

**Run:**
```bash
python gpu_kernel.py
```


---

#### Flow 2 — Low-Level: HIP C++ Extension

The full manual path: write the kernel and Python binding in a single `.cu` file, compile it as a native extension using PyTorch's build system, then import and call it from Python.

**Files:**

| File | Role |
|---|---|
| [add_one_kernel.cu](add_one_kernel.cu) | Kernel + launcher + pybind11 binding — everything in one file |
| [setup.py](setup.py) | Build script — uses `CUDAExtension` to compile the `.cu` into a `.so` |

**How it works:**

**Step 1 — The kernel, launcher, and binding** ([add_one_kernel.cu](add_one_kernel.cu)):
```cpp
// GPU kernel — one thread per element
__global__ void add_one(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

// Launcher — bridges torch::Tensor to raw pointer, sets grid/block, runs kernel
void add_one_launcher(torch::Tensor tensor) {
    int n = tensor.numel();
    float* data = tensor.data_ptr<float>();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    add_one<<<grid_size, block_size>>>(data, n);
    hipDeviceSynchronize();
}

// Python binding — exposes add_one_launcher as add_one_ext.add_one
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one_launcher, "Add one kernel (HIP)");
}
```

**Step 2 — Build** ([setup.py](setup.py)):
```bash
python setup.py build_ext --inplace
```
`CUDAExtension` compiles `add_one_kernel.cu` and links against PyTorch/ROCm. On AMD, `hipcc` is invoked transparently. Produces `add_one_ext.cpython-*.so` in the same directory.

**Step 3 — Use from Python:**
```python
import add_one_ext
import torch

x = torch.ones(10, device="cuda")
add_one_ext.add_one(x)
```

**Expected output:**
```python
>>> x = torch.ones(10, device="cuda")
>>> x
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')
>>> add_one_ext.add_one(x)
>>> x
tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.], device='cuda:0')
```

---

## Flow Comparison

| | Flow 1 (`gpu_kernel.py`) | Flow 2 (C++ Extension) |
|---|---|---|
| Build step | None (JIT) | `python setup.py build_ext --inplace` |
| Kernel location | String in Python file | Separate `.cu` / `.hip` file |
| PyTorch integration | `torch.cuda._compile_kernel` | `CUDAExtension` + pybind11 |
| Use case | Prototyping, experimentation | Production, complex kernels |
| HIP translation | Handled internally by PyTorch | `hipcc` called by PyTorch build system |
