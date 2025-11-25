import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom GELU kernel
gelu_source = """
#include <torch/extension.h>
#include <math.h>

__global__ void gelu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float x_cubed = x * x * x;
        float inside = x + 0.044715f * x_cubed;
        inside *= 0.7978845608f;  // sqrt(2/pi)
        float tanh_val = tanhf(inside);
        output[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

# Custom Softmax kernel
softmax_source = """
#include <torch/extension.h>

__global__ void softmax_kernel(float* input, float* output, int batch_size, int out_features) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    float* smem = shared;

    // Phase 1: Compute max
    float local_max = -FLT_MAX;
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = input[row * out_features + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem[tid + s] > smem[tid]) {
                smem[tid] = smem[tid + s];
            }
        }
        __syncthreads();
    }
    float global_max = smem[0];
    __syncthreads();

    // Phase 2: Compute exponentials and accumulate sum
    float local_sum = 0.0f;
    for (int i = tid; i < out_features; i += blockDim.x) {
        float val = input[row * out_features + i] - global_max;
        float exp_val = expf(val);
        output[row * out_features + i] = exp_val;
        local_sum += exp_val;
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float total_sum = smem[0];
    __syncthreads();

    // Phase 3: Divide by total_sum
    for (int i = tid; i < out_features; i += blockDim.x) {
        output[row * out_features + i] /= total_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int out_features = input.size(1);
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = batch_size; // one block per row
    int shared_size = block_size * sizeof(float); // shared memory per block

    softmax_kernel<<<num_blocks, block_size, shared_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, out_features);
    return output;
}
"""

# Compile the custom kernels
gelu = load_inline(
    name="gelu",
    cpp_sources="",
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
)

softmax = load_inline(
    name="softmax",
    cpp_sources="",
    cuda_sources=softmax_source,
    functions=["softmax_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gelu = gelu  # reference to the custom GELU function
        self.softmax = softmax  # reference to the custom Softmax function

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu.gelu_cuda(x)
        x = self.softmax.softmax_cuda(x)
        return x

# The get_inputs and get_init_inputs functions remain the same as in the original code
def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]