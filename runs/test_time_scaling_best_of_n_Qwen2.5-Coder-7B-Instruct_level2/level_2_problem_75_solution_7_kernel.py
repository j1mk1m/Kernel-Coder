import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Gemm
gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void cublas_gemm(const float* a, const float* b, float* c, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, 
                  (const void*)1.0f, (const void*)b, CUDA_R_32F, 
                  k, (const void*)a, CUDA_R_32F, n, 
                  (const void*)0.0f, (void*)c, CUDA_R_32F, n, CUDA_R_32F);
    cublasDestroy(handle);
}

torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b) {
    auto m = a.size(0);
    auto n = b.size(1);
    auto k = a.size(1);
    auto out = torch::zeros({m, n}, a.options());

    cublas_gemm(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), m, n, k);

    return out;
}
"""

gemm_cpp_source = (
    "torch::Tensor gemm_cuda(torch::Tensor a, torch::Tensor b);"
)

# Load the CUDA kernel
gemm = load_inline(
    name="gemm",
    cpp_sources=gemm_cpp_source,
    cuda_sources=gemm_source,
    functions=["gemm_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


# Define the custom CUDA kernel for Group Normalization
group_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void group_norm_forward_kernel(const float* input, float* output, float* mean, float* var, float eps, int channels, int groups, int height, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int group_idx = tid / (height * width);
    int channel_idx = (tid % (height * width)) / width;
    int hw_idx = tid % (width * height);

    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < groups; ++i) {
        if (group_idx == i) {
            sum += input[hw_idx * channels + channel_idx * groups + i];
            sum_sq += input[hw_idx * channels + channel_idx * groups + i] * input[hw_idx * channels + channel_idx * groups + i];
        }
    }

    __syncthreads();

    if (tid < groups * height * width) {
        mean[tid] = sum / (groups * height * width);
        var[tid] = sum_sq / (groups * height * width) - mean[tid] * mean[tid] + eps;
        output[hw_idx * channels + channel_idx * groups + group_idx] = (input[hw_idx * channels + channel_idx * groups + group_idx] - mean[tid]) / sqrt(var[tid]);
    }
}

torch::Tensor group_norm_forward_cuda(torch::Tensor input, int groups, float eps) {
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto batch_size = input.size(0);
    auto out = torch::zeros_like(input);
    auto mean = torch::zeros({batch_size, groups * height * width});
    auto var = torch::zeros({batch_size, groups * height * width});

    const int block_size = 256;
    const int num_blocks = (channels * height * width + block_size - 1) / block_size;

    group_norm_forward_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), out.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>(), eps, channels, groups, height, width);

    return out;
}
"""

group_norm_cpp_source = (
    "torch::Tensor group_norm_forward_cuda(torch::Tensor input, int groups, float eps);"
)

# Load the CUDA kernel
group_norm = load_inline(
    name="group_norm",
    cpp_sources=group_norm_cpp_source,
    cuda_sources=group_norm_source,
    functions=["group_norm_forward_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = gemm
        self.group_norm = group_norm
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm.gemm_cuda(x.view(-1, in_features), self.weight)
        x = x.view(-1, out_features, 1, 1)
        x = self.group_norm.group_norm_forward_cuda(x, num_groups, 1e-5)
        x = x.view(-1, out_features)
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = x + self.bias
        return x

# Example usage
if __name__ == "__main__":
    batch_size = 1024
    in_features = 8192
    out_features = 8192
    num_groups = 512
    bias_shape = (1, out_features, 1, 1)

    model_new = ModelNew(in_features, out_features, num_groups, bias_shape)
    inputs = get_inputs()
    outputs = model_new(inputs[0])
    print(outputs.shape)