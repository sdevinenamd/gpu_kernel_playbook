#include <torch/extension.h>

extern "C" void add_one_cuda(float* data, int n);

void add_one_launcher(torch::Tensor tensor) {
    int n = tensor.numel();
    float* data = tensor.data_ptr<float>();

    add_one_cuda(data, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one_launcher, "Add one kernel (HIP)");
}
