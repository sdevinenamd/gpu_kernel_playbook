#include <hip/hip_runtime.h>

extern "C" {

__global__ void add_one(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

void add_one_cuda(float* data, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    add_one<<<grid_size, block_size>>>(data, n);
    hipDeviceSynchronize();
}

}
