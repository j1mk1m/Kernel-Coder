import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom fused kernel for InstanceNorm + element-wise add and multiply
instance_norm_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_instance_norm_add_mul_kernel(
    const torch::PackedTensorAccessor<scalar_t,2> x,
    const torch::PackedTensorAccessor<scalar_t,2> y,
    torch::PackedTensorAccessor<scalar_t,2> out,
    const scalar_t eps,
    const scalar_t* running_mean,
    const scalar_t* running_var,
    const scalar_t* weight,
    const scalar_t* bias,
    const int batch_size,
    const int channels) {

    extern __shared__ scalar_t shared[];

    // Each thread handles one element of x
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels) return;

    // Compute mean and variance for each channel (only one channel here since channels=out_features and H=W=1)
    __shared__ scalar_t mean_buf[channels];
    __shared__ scalar_t var_buf[channels];

    if (threadIdx.x < channels) {
        mean_buf[threadIdx.x] = 0;
        var_buf[threadIdx.x] = 0;
    }
    __syncthreads();

    // Accumulate sum and sum of squares
    scalar_t val = x[idx / channels][idx % channels];
    atomicAdd(&mean_buf[idx % channels], val);
    atomicAdd(&var_buf[idx % channels], val * val);

    __syncthreads();

    // Compute mean and variance
    scalar_t mean = mean_buf[threadIdx.x] / batch_size;
    scalar_t var = var_buf[threadIdx.x] / batch_size - mean * mean;
    scalar_t std = 1.0f / sqrt(var + eps);

    __syncthreads();

    // Normalize and apply scale/shift
    val = (val - mean) * std;
    if (weight != nullptr) val = val * weight[threadIdx.x];
    if (bias != nullptr) val = val + bias[threadIdx.x];

    // Perform add and multiply with y
    val = (val + y[blockIdx.x][threadIdx.x]) * y[blockIdx.x][threadIdx.x];

    out[blockIdx.x][threadIdx.x] = val;
}

torch::Tensor fused_instance_norm_add_mul_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {

    const int batch_size = x.size(0);
    const int channels = x.size(1);

    auto out = torch::empty({batch_size, channels}, x.options());

    dim3 blocks(batch_size);
    dim3 threads(channels);
    size_t shared_size = 2 * channels * sizeof(float); // For mean and var buffers

    AT_DISPATCH_FLOATING_TYPES(x.type(), "fused_instance_norm_add_mul_cuda", ([&] {
        fused_instance_norm_add_mul_kernel<scalar_t><<<blocks, threads, shared_size, at::cuda::getCurrentCUDAStream()>>>(
            x.packed_accessor<scalar_t,2>(),
            y.packed_accessor<scalar_t,2>(),
            out.packed_accessor<scalar_t,2>(),
            eps,
            running_mean.data_ptr<scalar_t>(),
            running_var.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            batch_size,
            channels);
    }));

    return out;
}
"""

instance_norm_fused_cpp_source = (
    "torch::Tensor fused_instance_norm_add_mul_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float eps);"
)

# Compile the fused kernel
fused_kernel = load_inline(
    name="fused_instance_norm_add_mul",
    cpp_sources=instance_norm_fused_cpp_source,
    cuda_sources=instance_norm_fused_source,
    functions=["fused_instance_norm_add_mul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        # Need to keep these parameters for the custom kernel
        self.weight = self.instance_norm.weight
        self.bias = self.instance_norm.bias
        self.running_mean = self.instance_norm.running_mean
        self.running_var = self.instance_norm.running_var
        self.eps = eps
        self.fused_kernel = fused_kernel

    def forward(self, x, y):
        x = self.bmm(x)
        # Directly use 2D tensors without unsqueezing/squeezing
        # Pass parameters to the fused kernel
        out = self.fused_kernel.fused_instance_norm_add_mul_cuda(
            x,
            y,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.eps
        )
        return out

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda(), torch.rand(batch_size, out_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]