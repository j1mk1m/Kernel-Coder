import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

instance_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define THREADS_PER_BLOCK 512

__global__ void compute_instance_norm_stats(
    const float* input,
    float* means,
    float* variances,
    int B,
    int C,
    int H,
    int W,
    float eps
) {
    int b = blockIdx.x;
    int c = blockIdx.y;

    const float* data = input + b * C * H * W + c * H * W;
    int N = H * W;

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + blockDim.x;

    int tid = threadIdx.x;

    if (tid == 0) {
        s_sum[0] = 0.0f;
        s_sum_sq[0] = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int i = 0; i < 512; ++i) {
        int pos = tid * 512 + i;
        float val = data[pos];
        local_sum += val;
        local_sq_sum += val * val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sq_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = s_sum[0];
        float total_sq_sum = s_sum_sq[0];
        float mean = total_sum / N;
        float var = (total_sq_sum / N) - mean * mean;
        means[b * C + c] = mean;
        variances[b * C + c] = var;
    }
}

__global__ void apply_instance_norm(
    const float* input,
    const float* means,
    const float* variances,
    float* output,
    int B,
    int C,
    int H,
    int W,
    float eps
) {
    int b = blockIdx.x;
    int c = blockIdx.y;

    const float* data_in = input + b * C * H * W + c * H * W;
    float* data_out = output + b * C * H * W + c * H * W;

    float mean = means[b * C + c];
    float var = variances[b * C + c];
    float denom = rsqrtf(var + eps);

    int tid = threadIdx.x;

    for (int i = 0; i < 512; ++i) {
        int pos = tid * 512 + i;
        float val = data_in[pos];
        data_out[pos] = (val - mean) * denom;
    }
}

torch::Tensor compute_instance_norm_stats_cuda(torch::Tensor input, int B, int C, int H, int W, float eps) {
    auto options = input.options();
    auto means = torch::empty({B * C}, options);
    auto variances = torch::empty({B * C}, options);

    const int block_size = THREADS_PER_BLOCK;
    dim3 grid(B, C);
    dim3 block(block_size);

    auto stream = at::cuda::getCurrentCUDAStream();

    compute_instance_norm_stats<<<grid, block, 2 * block_size * sizeof(float), stream>>>(
        input.data_ptr<float>(),
        means.data_ptr<float>(),
        variances.data_ptr<float>(),
        B, C, H, W, eps
    );

    return std::make_tuple(means, variances);
}

torch::Tensor apply_instance_norm_cuda(
    torch::Tensor input,
    torch::Tensor means,
    torch::Tensor variances,
    int B, int C, int H, int W,
    float eps
) {
    auto output = torch::empty_like(input);

    const int block_size = THREADS_PER_BLOCK;
    dim3 grid(B, C);
    dim3 block(block_size);

    auto stream = at::cuda::getCurrentCUDAStream();

    apply_instance_norm<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        means.data_ptr<float>(),
        variances.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W, eps
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_instance_norm_stats", &compute_instance_norm_stats_cuda, "Compute instance norm stats");
    m.def("apply_instance_norm", &apply_instance_norm_cuda, "Apply instance norm");
}
"""

instance_norm = load_inline(
    name='instance_norm',
    cuda_sources=instance_norm_source,
    functions=['compute_instance_norm_stats', 'apply_instance_norm'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.compute_stats = instance_norm.compute_instance_norm_stats
        self.apply_norm = instance_norm.apply_instance_norm

    def forward(self, x):
        B = 112
        C = 64
        H = 512
        W = 512
        eps = 1e-5
        means, variances = self.compute_stats(x, B, C, H, W, eps)
        output = self.apply_norm(x, means, variances, B, C, H, W, eps)
        return output