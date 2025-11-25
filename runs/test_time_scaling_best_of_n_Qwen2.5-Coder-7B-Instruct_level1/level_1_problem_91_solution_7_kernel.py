from torch.utils.cpp_extension import load_inline

reverse_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reverse_cumsum_kernel(const float* input, float* output, int batch_size, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len) {
        int batch = idx / seq_len;
        int seq = idx % seq_len;
        output[idx] = input[idx];
        for (int i = seq; i > 0; --i) {
            output[idx] += input[(batch * seq_len) + (i - 1)];
        }
    }
}

void reverse_cumsum_cuda(const float* input, float* output, int batch_size, int seq_len) {
    const int block_size = 256;
    const int num_blocks = (batch_size * seq_len + block_size - 1) / block_size;

    reverse_cumsum_kernel<<<num_blocks, block_size>>>(input, output, batch_size, seq_len);
}
"""

reverse_cumsum_cpp_source = (
    "void reverse_cumsum_cuda(const float* input, float* output, int batch_size, int seq_len);"
)

reverse_cumsum = load_inline(
    name="reverse_cumsum",
    cpp_sources=reverse_cumsum_cpp_source,
    cuda_sources=reverse_cumsum_source,
    functions=["reverse_cumsum_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ReverseCumSum(nn.Module):
    def __init__(self, dim):
        super(ReverseCumSum, self).__init__()
        self.dim = dim

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        output = torch.zeros_like(x)
        reverse_cumsum_cuda(x.view(-1).data_ptr(), output.view(-1).data_ptr(), batch_size, seq_len)
        return output