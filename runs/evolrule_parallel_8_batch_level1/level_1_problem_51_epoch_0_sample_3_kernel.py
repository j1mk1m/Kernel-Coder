import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

custom_argmax_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void custom_argmax_kernel(
    const float* input_data,
    int64_t* output_data,
    int axis,
    int n_dims,
    const int* input_shape,
    const int* input_strides,
    const int* output_shape,
    int L,
    int output_size
) {
    int out_idx = blockIdx.x;
    if (out_idx >= output_size) return;

    // Compute output coordinates
    int coords_out[n_dims - 1];
    int temp = out_idx;
    for (int i = n_dims - 2; i >= 0; --i) {
        coords_out[i] = temp % output_shape[i];
        temp /= output_shape[i];
    }

    // Convert output coordinates to input coordinates (excluding axis)
    int coords_in[n_dims];
    int idx = 0;
    for (int d = 0; d < n_dims; ++d) {
        if (d < axis) {
            coords_in[d] = coords_out[idx++];
        } else if (d == axis) {
            coords_in[d] = 0;  // Will iterate over this dimension
        } else {
            coords_in[d] = coords_out[idx++];
        }
    }

    // Compute base offset using element strides (divided by sizeof(float))
    int base_offset = 0;
    for (int d = 0; d < n_dims; ++d) {
        base_offset += coords_in[d] * (input_strides[d] / sizeof(float));
    }

    // Compute element stride along the axis
    int element_stride_axis = input_strides[axis] / sizeof(float);

    // Each thread processes a portion of the L elements along the axis
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    int start = (tid * L) / total_threads;
    int end = ((tid + 1) * L) / total_threads;

    float local_max = -INFINITY;
    int local_max_idx = -1;

    for (int i = start; i < end; ++i) {
        int offset = base_offset + i * element_stride_axis;
        float val = input_data[offset];
        if (val > local_max) {
            local_max = val;
            local_max_idx = i;
        } else if (val == local_max) {
            if (local_max_idx == -1 || i < local_max_idx) {
                local_max_idx = i;
            }
        }
    }

    // Reduction in shared memory
    __shared__ float s_max[256];
    __shared__ int s_idx[256];

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_max[threadIdx.x + s] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + s];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + s];
            } else if (s_max[threadIdx.x + s] == s_max[threadIdx.x]) {
                if (s_idx[threadIdx.x + s] < s_idx[threadIdx.x]) {
                    s_idx[threadIdx.x] = s_idx[threadIdx.x + s];
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output_data[out_idx] = s_idx[0];
    }
}

torch::Tensor custom_argmax_cuda(torch::Tensor input, int axis) {
    const int n_dims = input.dim();
    if (axis < 0) axis += n_dims;
    const int L = input.size(axis);

    // Compute output shape
    auto output_shape = input.sizes().vec();
    output_shape.erase(output_shape.begin() + axis);
    torch::IntArrayRef output_shape_ref(output_shape);
    torch::Tensor output = torch::empty(output_shape_ref, torch::dtype(torch::kInt64).device(input.device()));

    // Prepare input_shape, input_strides, output_shape as tensors on device
    auto input_shape = torch::from_blob(input.sizes().data(), {n_dims}, torch::kInt32).to(input.device());
    auto input_strides = torch::from_blob(input.strides().data(), {n_dims}, torch::kInt32).to(input.device());
    auto output_shape_tensor = torch::from_blob(output_shape.data(), {output_shape.size()}, torch::kInt32).to(input.device());

    // Launch the kernel
    int block_size = 256;
    dim3 grid(output.numel());
    dim3 block(block_size);
    custom_argmax_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        axis,
        n_dims,
        input_shape.data_ptr<int>(),
        input_strides.data_ptr<int>(),
        output_shape_tensor.data_ptr<int>(),
        L,
        output.numel()
    );

    return output;
}
"""

custom_argmax_cpp = """
torch::Tensor custom_argmax_cuda(torch::Tensor input, int axis);
"""

custom_argmax = load_inline(
    name="custom_argmax",
    cpp_sources=custom_argmax_cpp,
    cuda_sources=custom_argmax_source,
    functions=["custom_argmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.custom_argmax = custom_argmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_argmax.custom_argmax_cuda(x, self.dim)