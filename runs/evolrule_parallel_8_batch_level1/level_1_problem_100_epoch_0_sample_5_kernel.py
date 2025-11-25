import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

hinge_loss_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define THREADS_PER_BLOCK 256

template <typename scalar_t>
__global__ void hinge_loss_kernel(
    const scalar_t* predictions,
    const scalar_t* targets,
    scalar_t* sum,
    int batch_size,
    int input_size
) {
    extern __shared__ scalar_t sdata[];

    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int block_size = blockDim.x;

    scalar_t block_sum = 0.0;

    int total_elements = batch_size * input_size;

    for (int i = tid; i < total_elements; i += block_size * gridDim.x) {
        int row = i / input_size;
        int col = i % input_size;

        scalar_t pred = predictions[i];
        scalar_t target = targets[row];
        scalar_t product = pred * target;
        scalar_t value = fmax(1.0 - product, 0.0);

        block_sum += value;
    }

    __syncthreads();
    sdata[tid] = block_sum;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

torch::Tensor hinge_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    predictions = predictions.contiguous();
    targets = targets.contiguous();

    int batch_size = predictions.size(0);
    int input_size = predictions.size(1);
    int total_elements = batch_size * input_size;

    auto sum_result = torch::zeros({1}, predictions.options());

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((total_elements + block.x - 1) / block.x);

    size_t sm_size = block.x * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "hinge_loss_forward", [&] {
        auto predictions_data = predictions.data_ptr<scalar_t>();
        auto targets_data = targets.data_ptr<scalar_t>();
        auto sum_data = sum_result.data_ptr<scalar_t>();

        hinge_loss_kernel<scalar_t><<<grid, block, sm_size>>>(
            predictions_data,
            targets_data,
            sum_data,
            batch_size,
            input_size
        );
    });

    cudaDeviceSynchronize();

    auto mean = sum_result[0] / static_cast<float>(total_elements);

    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hinge_loss_forward", &hinge_loss_forward, "Hinge loss forward");
}
"""

hinge_loss_cuda = load_inline(
    name="hinge_loss_cuda",
    cuda_sources=hinge_loss_cuda_source,
    functions=["hinge_loss_forward"],
    verbose=True,
    extra_cflags=["-DWITH_CUDA"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hinge_loss_forward = hinge_loss_cuda.hinge_loss_forward

    def forward(self, predictions, targets):
        return self.hinge_loss_forward(predictions.cuda(), targets.cuda())