import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose1d_source = """
__global__ void conv_transpose1d_kernel(
    const float* input, const float* weight, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_length, int output_length,
    int kernel_size, int stride, int padding, int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * output_length) return;

    int n = idx / (out_channels * output_length);
    int remainder = idx % (out_channels * output_length);
    int oc = remainder / output_length;
    int ox = remainder % output_length;

    float result = 0.0f;

    for (int ic = 0; ic < in_channels; ic++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            // Flip the kernel for transposed convolution
            int k_idx = kernel_size - 1 - kx;
            int w_offset = oc * in_channels * kernel_size + ic * kernel_size + k_idx;
            float w = weight[w_offset];

            // Compute input position
            int numerator = ox + padding - kx * dilation;
            int input_pos = (numerator + (stride - 1)) / stride;
            if (input_pos < 0 || input_pos >= input_length) continue;

            int in_offset = n * in_channels * input_length + ic * input_length + input_pos;
            float in_val = input[in_offset];
            result += w * in_val;
        }
    }

    int out_offset = n * out_channels * output_length + oc * output_length + ox;
    output[out_offset] = result;
}

extern "C" {
    void conv_transpose1d_cuda(
        torch::Tensor input, torch::Tensor weight, torch::Tensor output,
        int batch_size, int in_channels, int out_channels,
        int input_length, int output_length,
        int kernel_size, int stride, int padding, int dilation
    ) {
        const int block_size = 256;
        const int total = batch_size * out_channels * output_length;
        const int grid_size = (total + block_size - 1) / block_size;
        
        conv_transpose1d_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_length, output_length,
            kernel_size, stride, padding, dilation
        );
        cudaDeviceSynchronize();
    }
}
"""

conv_transpose1d_cpp_header = """
void conv_transpose1d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor output,
    int batch_size, int in_channels, int out_channels,
    int input_length, int output_length,
    int kernel_size, int stride, int padding, int dilation
);
"""

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_header,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize weight parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using the same method as PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.size(2)
        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1

        output = torch.empty(
            x.size(0), self.out_channels, output_length,
            device=x.device, dtype=x.dtype
        )

        conv_transpose1d.conv_transpose1d_cuda(
            x.contiguous(), self.weight.contiguous(), output,
            x.size(0), self.in_channels, self.out_channels,
            input_length, output_length,
            self.kernel_size, self.stride, self.padding, self.dilation,
        )
        return output

def get_inputs():
    x = torch.rand(16, 32, 131072).cuda()
    return [x]

def get_init_inputs():
    return [32, 64, 3, 2, 1, 2]