import torch
import torch.nn as nn

min_subtract_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_subtract_kernel(const float* x, const float* constant, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        val = min(val, constant[0]);
        val -= constant[0];
        out[idx] = val;
    }
}

torch::Tensor min_subtract_cuda(torch::Tensor x, torch::Tensor constant) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    min_subtract_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        constant.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    return out;
}
"""

min_subtract_cpp_source = "torch::Tensor min_subtract_cuda(torch::Tensor x, torch::Tensor constant);"

min_subtract = load_inline(
    name="min_subtract",
    cpp_sources=min_subtract_cpp_source,
    cuda_sources=min_subtract_source,
    functions=["min_subtract_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.min_subtract = min_subtract

    def forward(self, x):
        x = self.linear(x)
        x = self.min_subtract.min_subtract_cuda(x, self.constant.unsqueeze(0))
        return x

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, constant]