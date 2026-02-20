import torch

KERNEL_SOURCE = """
extern "C"
__global__ void add_one(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}
"""

print("HIP version:", torch.version.hip)
print("Device:", torch.cuda.get_device_name(0))

print("\nCompiling kernel at runtime via HIP RTC...")
add_one_kernel = torch.cuda._compile_kernel(KERNEL_SOURCE, "add_one")
print("Done.\n")

x = torch.arange(10, dtype=torch.float32, device="cuda")
print("Before:", x)

n = x.numel()
block_size = 256
grid_size = (n + block_size - 1) // block_size

add_one_kernel(
    grid=(grid_size, 1, 1),
    block=(block_size, 1, 1),
    args=[x, n],
)

print("After: ", x)
