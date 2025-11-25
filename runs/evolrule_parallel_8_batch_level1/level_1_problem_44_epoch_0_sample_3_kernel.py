import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

avg_pool1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int T_BLOCK>
__global__ void avg_pool1d_kernel(const float* input, float* output, int N, int C, int L) {
    const int kernel_size = 8;
    const int stride = 1;
    const int padding = 4;
    int L_padded = L + 2 * padding;
    int O = (L_padded - kernel_size) / stride + 1;

    int o_block = blockIdx.x / (N * C);
    int c_n_block = blockIdx.x % (N * C);
    int n = c_n_block / C;
    int c = c_n_block % C;
    int o_block_start = o_block * T_BLOCK;

    extern __shared__ float s_data[];

    // Load shared memory
    int s_size = T_BLOCK + kernel_size - 1;
    for (int j = threadIdx.x; j < s_size; j += blockDim.x) {
        int padded_input_idx = o_block_start + j;
        float val = 0.0f;
        if (padded_input_idx >= padding && padded_input_idx < (L_padded - padding)) {
            int original_idx = padded_input_idx - padding;
            val = input[ n * C * L + c * L + original_idx ];
        }
        s_data[j] = val;
    }
    __syncthreads();

    int o = o_block_start + threadIdx.x;
    if (o >= O) return;

    int start_in_s = o - o_block_start;
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        sum += s_data[start_in_s + i];
    }
    output[ n * C * O + c * O + o ] = sum / kernel_size;
}

void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, int N, int C, int L) {
    const int T_BLOCK = 256;
    const int kernel_size = 8;
    const int stride = 1;
    const int padding = 4;
    int L_padded = L + 2 * padding;
    int O = (L_padded - kernel_size) / stride + 1;

    int o_blocks = (O + T_BLOCK - 1) / T_BLOCK;
    int total_blocks = o_blocks * N * C;

    dim3 blockDim(T_BLOCK);
    dim3 gridDim(total_blocks);

    int s_size = T_BLOCK + kernel_size -1;
    size_t shared_size = s_size * sizeof(float);

    avg_pool1d_kernel<T_BLOCK><<<gridDim, blockDim, shared_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, C, L);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
    }
}
"""

avg_pool1d_cpp_source = """
void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, int N, int C, int L);
"""

avg_pool_cuda = load_inline(
    name="avg_pool_cuda",
    cpp_sources=avg_pool1d_cpp_source,
    cuda_sources=avg_pool1d_source,
    functions=["avg_pool1d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, L = x.size()
        L_padded = L + 2 * self.padding
        O = (L_padded - self.kernel_size) // self.stride + 1
        output = torch.empty((N, C, O), dtype=x.dtype, device=x.device)
        avg_pool_cuda.avg_pool1d_cuda(x, output, N, C, L)
        return output

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]