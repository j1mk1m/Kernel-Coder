import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and wrapper
triplet_loss_cuda_src = """
#include <torch/extension.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

__global__ void triplet_loss_kernel(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* loss_out,
    float margin,
    int batch_size,
    int input_dim
) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum_ap = 0.0f;
    float sum_an = 0.0f;

    for (int k = tid; k < input_dim; k += stride) {
        int offset = sample_idx * input_dim + k;
        float a = anchor[offset];
        float p = positive[offset];
        float n = negative[offset];

        float diff_ap = a - p;
        float diff_an = a - n;

        sum_ap += diff_ap * diff_ap;
        sum_an += diff_an * diff_an;
    }

    __shared__ float shared_ap[THREADS_PER_BLOCK];
    __shared__ float shared_an[THREADS_PER_BLOCK];

    shared_ap[tid] = sum_ap;
    shared_an[tid] = sum_an;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_ap[tid] += shared_ap[tid + s];
            shared_an[tid] += shared_an[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float distance_ap = sqrtf(shared_ap[0]);
        float distance_an = sqrtf(shared_an[0]);
        float loss = distance_ap - distance_an + margin;
        loss_out[sample_idx] = (loss > 0.0f) ? loss : 0.0f;
    }
}

torch::Tensor triplet_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin,
    int batch_size,
    int input_dim
) {
    const int block_size = THREADS_PER_BLOCK;
    const int grid_size = batch_size;

    auto loss_out = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_loss_cuda", ([&] {
        triplet_loss_kernel<<<grid_size, block_size>>>(
            anchor.data_ptr<scalar_t>(),
            positive.data_ptr<scalar_t>(),
            negative.data_ptr<scalar_t>(),
            loss_out.data_ptr<float>(),
            margin,
            batch_size,
            input_dim
        );
    }));

    auto mean_loss = loss_out.mean();

    return mean_loss;
}
"""

# The C++ declarations needed for the wrapper function
triplet_loss_cuda_header = """
torch::Tensor triplet_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin,
    int batch_size,
    int input_dim
);
"""

# Compile the CUDA code
triplet_loss_cuda = load_inline(
    name="triplet_loss_cuda",
    cpp_sources=triplet_loss_header,
    cuda_sources=triplet_loss_cuda_src,
    functions=["triplet_loss_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        batch_size = anchor.size(0)
        input_dim = anchor.size(1)
        return triplet_loss_cuda(
            anchor,
            positive,
            negative,
            self.margin,
            batch_size,
            input_dim
        )

def get_inputs():
    batch_size = 32768
    input_shape = (8192,)
    scale = torch.rand(())
    a = torch.rand(batch_size, *input_shape) * scale
    p = torch.rand(batch_size, *input_shape)
    n = torch.rand(batch_size, *input_shape)
    return [a.cuda(), p.cuda(), n.cuda()]

def get_init_inputs():
    return [1.0]