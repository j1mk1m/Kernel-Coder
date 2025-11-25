from torch.utils.cpp_extension import load_inline

# CUDA source code for mish
mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mish_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        data[i] = x * tanh(expf(x));
    }
}

torch::Tensor mish_cuda(torch::Tensor input) {
    int n = input.numel();
    auto output = torch::empty_like(input);
    mish_kernel<<<(n + 255) / 256, 256>>>(input.data_ptr<float>(), n);
    return output;
}
"""

mish_cpp_source = "torch::Tensor mish_cuda(torch::Tensor input);"

mish_module = load_inline(
    name="mish",
    cpp_sources=mish_cpp_source,
    cuda_sources=mish_source,
    functions=["mish_cuda"],
    verbose=True,
    extra_cflags=[],
    extra_ldflags=[],
)

# Use the mish module in your model
class ModelMish(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelMish, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mish = mish_module.mish_cuda

    def forward(self, x):
        x = self.linear(x)
        x = self.mish(x)
        return x